#!/usr/bin/env bash
# Session 9 pod driver (docs/prompt_session9.md, STEP 3, with corrections 1 to 6): the only way the pod is driven. It runs on
# the Runpod pod from /root/eddm_s9/code (this repository's code, uploaded with src/tools/upload_to_pod.py --driver
# /root/eddm_s9/code/scripts/pod_session9.sh, which also starts these subcommands). Every stage detaches itself under nohup,
# so a closed connection does not stop it, and appends its log to /root/eddm_s9/logs/<stage>.log; a stage that a failure
# stopped resumes when it is run again (work already done is checked and skipped). A stage whose commands all succeeded
# writes /root/eddm_s9/stage_done/<stage>.txt (the time). pilot_long, tabicl and biogpt, when their NOT BUILT condition
# applies, write /root/eddm_s9/not_built/<stage>.txt with the reason and exit successfully, so that all continues; package
# writes results/fiorillo_v0_5/not_built.json from the markers. One process at a time runs a stage (a lock in
# /root/eddm_s9/locks/). Stages:
#   prepare        step 9: every uploaded file checked against /root/eddm_s9/uploads_manifest.json (upload_to_pod.py);
#                  pod_rate.txt's lines checked ("created_utc <YYYY-MM-DDTHH:MM:SSZ>", "hourly_usd <rate>", and optionally
#                  "maintenance_utc <time or none>"; Claude writes it in step 9); GPU check and installs (session 8's, with
#                  scikit-learn and pandas); Qwen3-4B-Base at its revision, its weights checked against fixed_versions.json;
#                  session 4's records rebuilt and checked against results/public_v0/checksums.json
#                  (records_manifest_s4_pod.json); EI's raw files downloaded at the fixed commit and checked, and session
#                  8's records rebuilt from article_selection.jsonl and checked (records_manifest_s8_pod.json); session 9's
#                  records rebuilt with ei_session9.py rebuild and checked against records_manifest.json
#                  (records_manifest_pod.json); s7_b3's and s8_ei's uploaded folders checked against records_manifest.json's
#                  models digests; the items (clean_train, swap_train and clean_monitor at 4,096 tokens, long_train and
#                  long_monitor at 6,144), and a check that the reading checks' inputs and the long validation and test
#                  records build with no cut or skip (/root/eddm_s9/items_check/); a cut or skipped record stops the stage
#   checks_s8      step 10: THE READING CHECKS on s8_ei. It refuses to run before /root/eddm_s9/osf_filed.txt exists (the
#                  lines "link <the OSF update's address>" and "filed_utc <time>"; exit 5). Raw test predictions of s8_ei's
#                  three runs on the none, title and swap inputs (system names s8_ei_<k>_<input>) into
#                  results/fiorillo_v0_5/checks_s8/, logits in checks_s8/logits/ (never in predictions/, logits/ or the
#                  staging folder); then reading_checks_session9.py score (checks_s8.json). Its stop (s8_ei's macro-F1 with
#                  no article above 0.85) exits 3 and stops all; once Johann has decided, Claude adds a "johann_decision"
#                  key to checks_s8.json, and the stage, run again, continues without scoring again
#   pilot_long     step 10: train_qwen_head.py pilot-s8 --headroom-gib 0 on the long items (pilot_long.json); if no setting
#                  passes, "NOT BUILT: long, <reason>" (not_built/pilot_long.txt), and long is dropped from the candidates
#   train          step 11: the cost rule first (recipe_session9.py cost into cost_rule.json, run only if that file does not
#                  exist; exit 4 is its stop, and Claude asks Johann; a cost_rule.json that stops, by the projection or an
#                  announced maintenance, and has no "johann_decision" key stops again with exit 4); then THE RECIPE
#                  RULE's loop: recipe_session9.py next says what to do (train a run from s7_b3, with session 8's fallback
#                  to gradient checkpointing when memory runs out, then predict the whole validation split; build the
#                  combination's records and items; build and predict the weight average; predict the exchanged
#                  validation records) until it says done; every output is copied into results, so the stage resumes from
#                  whatever exists; nothing in it reads a test file
#   trees          step 11, on the CPU beside train at nice 10: xgboost 3.2.0, catboost 1.2.10 and lightgbm 4.7.0 installed
#                  as session 8 installed its trees; trees_session9.py recode, refit (xgboost_codes and catboost_codes, 12
#                  threads) and lightgbm (lightgbm_tuned's refit and its check)
#   tabicl         step 12, after train, on the GPU: /root/eddm_s9/tabicl_env/ (uv venv --system-site-packages, then
#                  tabicl 2.2.0 and its non-torch requirements with --no-deps, so that the environment uses the image's
#                  torch; checked: the same torch as the training environment, CUDA, and tabicl's import);
#                  trees_session9.py tabicl; a failure writes "NOT BUILT: tabicl, <reason>"
#   prepare_large  step 12, only under GEMMA ON THE LONGER INPUT (recipe_selection.json's v05_ei.max_len is 6,144): session
#                  8's stage with its corrections 1, 5, 7, 8 and 10: vLLM's environment /root/eddm_s9/vllm_env/ and Gemma 4
#                  31B-it at its revision, checked
#   smoke_large    step 12, the same condition: session 8's smoke test (its corrections 9, 11 and 12) with session 9's paths,
#                  on the long EI validation prompts and session 8's PubMedQA prompts
#   large_validation  step 12, the same condition: large_model_session8.py on the long EI validation inputs, alone on the
#                  GPU (its prompts command for both questions, its answer and second commands for evidence_effect); the
#                  files it writes under the name gemma4_31b in results/fiorillo_v0_5/large_model/ copied to the name
#                  gemma4_31b_long (the originals kept)
#   biogpt         step 12, after train, on the GPU: sacremoses 0.2.0; microsoft/BioGPT-Large-PubMedQA at its revision;
#                  biogpt_session9.py validation (PubMedQA's 150 calibrate items); a failure writes "NOT BUILT: biogpt, ..."
#   calibrate      step 12: refuses before train, trees, tabicl, biogpt and (when they apply) the large stages have finished
#                  or written a NOT BUILT marker; calibrate_session9.py fit and ece; trees_session9.py pool on the GPU;
#                  handoff_curves_session9.py curves and fit-rule on validation for v0.5; power_session9.py validation
#   freeze         step 12: calibrate_session9.py freeze (release_candidate.json), before any test prediction of a v0.5
#                  system exists; refuses before osf_filed.txt exists (exit 5)
#   testpreds      step 13: refuses to start unless release_candidate.json exists; first writes testpreds_log.json (that
#                  file's SHA-256 and this stage's start time in UTC, later than the freeze's time; a retry keeps the first
#                  entry and adds its time); then one job at a time on the GPU: raw test predictions and logits for s9_b_1
#                  to s9_b_3, the winning recipe's runs and s9_ei_wavg at their recipe's length; THE READING CHECKS' inputs
#                  for v0.5's single-pass systems (results/fiorillo_v0_5/checks_v05/); gemma4_31b_long on test (under GEMMA
#                  ON THE LONGER INPUT, alone); biogpt test; tabicl test; then the tree members' test predictions (CPU); each
#                  checked for its rows; only raw files, logits, run summaries and the large model's files go into results;
#                  a retry runs only the jobs whose files are not all there
#   package        step 13: results/fiorillo_v0_5/not_built.json from the markers; fiorillo_v0_5_results.zip,
#                  fiorillo_v0_5_ei.zip (every s9_ run folder and s9_ei_wavg), fiorillo_v0_5_trees.zip and
#                  fiorillo_v0_5_logs.zip in /root/eddm_s9/outputs/, with SHA256SUMS.txt ("<SHA-256>  <bytes>  <name>");
#                  then the check of the package: the zips' sizes, SHA-256 and CRCs, and, once testpreds has finished, every
#                  file that steps 15 to 17 read in the zips with its rows, the digests release_candidate.json recorded,
#                  and testpreds_log.json naming its SHA-256 and a later start (before that the package is partial, as when
#                  a stop ends the run, and only the zips themselves are checked)
#   all            steps 10 to 13: prepare's successors in order (checks_s8, pilot_long, train with trees started beside
#                  it, tabicl, the large stages if they apply, biogpt, calibrate, freeze, testpreds, package), each started
#                  when the one before it has finished, in all's own process and not detached, each stage's output in its
#                  own log; a stage that finished before (stage_done) or wrote its NOT BUILT marker is not run again. all
#                  refuses before prepare has finished, and stops at the first failure (exit 1 or the stage's status), at
#                  the reading checks' stop (exit 3), at the cost rule's stop (exit 4) and while osf_filed.txt is missing
#                  (exit 5); the trees stage beside train is left to finish if all stops, and all, run again, waits for it
#   upload_list    on the linked shell, run from the repository: step 9's uploads, printed (nothing is written):
#                  "upload_list code" prints the files for the code root (the code as session 8 sent it, every file
#                  under scripts/ and src/ that git tracks, src/tests/ left out, and the results and records files step 9
#                  names); "upload_list" prints the upload commands, model folders included
# Usage on the pod: bash /root/eddm_s9/code/scripts/pod_session9.sh <stage>
BASE=${EDDM_POD_BASE:-/root/eddm_s9}  # the pod's folder; EDDM_POD_BASE only for a test run of this script off the pod
CODE=$BASE/code
MODELS=$BASE/models
LOGS=$BASE/logs
OUTS=$BASE/outputs
ITEMS=$BASE/items
ITEMS_CHECK=$BASE/items_check
LARGE_ITEMS=$BASE/large_items
STAGE_DIR=$BASE/predictions_staging
VENV=$BASE/vllm_env
TABICL_ENV=$BASE/tabicl_env
NOT_BUILT=$BASE/not_built
DONE=$BASE/stage_done
RES=$CODE/results/fiorillo_v0_5
RES8=$CODE/results/fiorillo_v0_4
REC8=$CODE/data/processed/session8
REC9=$CODE/data/processed/session9
QUESTIONS=src/questions/session8.yaml
QWEN_REV=906bfd4b4dc7f14ee4320094d8b41684abff8539
BIOGPT_REPO=microsoft/BioGPT-Large-PubMedQA
BIOGPT_REV=39c04fb31dfc8094b0f04c7bb7973cf612606c0c
GEMMA_REPO=google/gemma-4-31B-it
TREE_THREADS=12
EI=evidence_effect
PQA=pubmedqa_answer
RW=readmit_window
N_VAL=1237      # EI's validation prompts (every run predicts all of them)
N_TEST=1218     # EI's test prompts
N_PQA_VAL=150   # PubMedQA's calibrate validation items (BioGPT-Large)
N_PQA_TEST=500
N_RW_VAL=6997   # the readmission records
N_RW_TEST=6998
STOP_CHECKS=3   # the reading checks' stop (reading_checks_session9.py score's exit status)
STOP_COST=4     # the cost rule's stop (recipe_session9.py cost's exit status)
STOP_OSF=5      # osf_filed.txt is missing
STAGES="prepare checks_s8 pilot_long train trees tabicl prepare_large smoke_large large_validation biogpt calibrate freeze testpreds package all"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------- step 9's upload list (the linked shell; prints only)

upload_list_files() {  # the results and records files that step 9 names, with the two that the cost rule and the reading checks read
  cat <<'LIST'
results/fiorillo_v0_5/fixed_versions.json
results/fiorillo_v0_5/records_manifest.json
results/fiorillo_v0_5/license_recheck.csv
results/fiorillo_v0_5/validation_analyses/handoff_rule_s8.json
results/fiorillo_v0_4/records_manifest.json
results/fiorillo_v0_4/article_licenses.csv
results/fiorillo_v0_4/calibration_s8.json
results/fiorillo_v0_4/trees.json
results/fiorillo_v0_4/readmission_pool.json
results/fiorillo_v0_4/fixed_versions.json
results/fiorillo_v0_4/pilot.json
results/fiorillo_v0_4/comparison.json
results/fiorillo_v0_4/predictions/s8_ei-raw_evidence_effect_test.csv
results/fiorillo_v0_4/predictions/gemma4_31b_evidence_effect_validation.csv
results/fiorillo_v0_3/calibration_s7.json
results/fiorillo_v0_3/trees.json
results/fiorillo_v0_3/predictions/s7_b3_pubmedqa_answer_validation.csv
results/public_v0/checksums.json
data/processed/session8/article_selection.jsonl
data/processed/session8/bm25_stats.json
data/processed/session9/long/article_selection_long.jsonl
LIST
}

upload_list_code() {  # upload_list_code <repository root>: the paths for the code root, one a line (upload_to_pod.py --list)
  local root="$1" tracked untracked members paths p missing=0
  cd "$root" || return 1
  if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then echo "upload_list: $root is not a git repository" >&2; return 1; fi
  tracked=$(git ls-files scripts src | grep -v '^src/tests/') || return 1
  untracked=$(git ls-files --others --exclude-standard scripts src | grep -v '^src/tests/' || true)
  if [ -n "$untracked" ]; then
    echo "upload_list: not committed yet, listed anyway (COMMIT.txt will not name them): $(printf '%s ' $untracked)" >&2
  fi
  # the validation files of session 8's pool members (session 8's correction 3), as readmission_pool.json records them
  members=$(python3 - results/fiorillo_v0_4/readmission_pool.json <<'PY'
import json, sys
pool = json.load(open(sys.argv[1]))
for m in pool["members"]:
    print(pool["member_files"][m]["file"])
PY
) || return 1
  paths=$({ printf '%s\n' "$tracked"; if [ -n "$untracked" ]; then printf '%s\n' "$untracked"; fi; } | sort -u
          upload_list_files; printf '%s\n' "$members")
  paths=$(printf '%s\n' "$paths" | awk 'NF && !seen[$0]++')
  while IFS= read -r p; do
    if [ ! -f "$p" ]; then echo "upload_list: $p does not exist here" >&2; missing=1; fi
  done <<< "$paths"
  printf '%s\n' "$paths"
  return "$missing"
}

upload_list_commands() {  # upload_list_commands <repository root>: step 9's upload commands, model folders included
  local root="$1" folders tool name local_root files
  cd "$root" || return 1
  # each model folder's files are the ones prepare checks: records_manifest.json's models digests
  folders=$(python3 - results/fiorillo_v0_5/records_manifest.json <<'PY'
import json, sys
models = json.load(open(sys.argv[1]))["models"]
local = {"s7_b3": "models/fiorillo_v0_3/arm_b", "s8_ei_1": "models/fiorillo_v0_4/s8_ei_1", "s8_ei_2": "models/fiorillo_v0_4/s8_ei_2",
         "s8_ei_3": "models/fiorillo_v0_4/s8_ei_3"}
for name, folder in local.items():
    files = sorted((models.get(name) or {}).keys())
    if not files:
        sys.exit(f"records_manifest.json lists no files for {name}")
    print(name, folder, " ".join(files))
PY
) || return 1
  tool='EDDM_POD_TOKEN="$(cat models/fiorillo_v0_5/pod_token.txt)" python3 src/tools/upload_to_pod.py --url "$POD_URL" --driver /root/eddm_s9/code/scripts/pod_session9.sh'
  cat <<EOF
# Step 9's uploads (docs/prompt_session9.md, STEP 9), run from the repository root on the linked shell, each run again until
# it reports nothing left ("not_started_yet": 0, no "part_way"); every file's SHA-256 is checked on the pod and recorded in
# /root/eddm_s9/uploads_manifest.json, which prepare checks again. POD_URL is the pod's Jupyter address
# (https://<pod id>-8888.proxy.runpod.net); the token is read from models/fiorillo_v0_5/pod_token.txt and never printed.
mkdir -p /tmp/eddm_s9/upload && git rev-parse HEAD > /tmp/eddm_s9/upload/COMMIT.txt
bash scripts/pod_session9.sh upload_list code > /tmp/eddm_s9/upload/upload_list_code.txt
$tool upload --local-root . --remote-root /root/eddm_s9/code --list /tmp/eddm_s9/upload/upload_list_code.txt
$tool upload --local-root /tmp/eddm_s9/upload --remote-root /root/eddm_s9/code COMMIT.txt
EOF
  while read -r name local_root files; do
    echo "$tool upload --local-root $local_root --remote-root /root/eddm_s9/models/$name $files"
  done <<< "$folders"
  cat <<EOF
# Then Claude writes /root/eddm_s9/pod_rate.txt (the lines "created_utc <YYYY-MM-DDTHH:MM:SSZ>" and "hourly_usd <rate>", and
# "maintenance_utc <time>" if maintenance is announced for the pod's server, else "maintenance_utc none") and starts prepare:
$tool run prepare
EOF
}

STAGE="${1:-}"
if [ "$STAGE" = upload_list ]; then
  case "${2:-commands}" in
    code) upload_list_code "$(dirname "$SELF")/.." ;;
    commands) upload_list_commands "$(dirname "$SELF")/.." ;;
    *) echo "upload_list takes code or nothing (the commands)"; exit 1 ;;
  esac
  exit $?
fi
if [ -z "$STAGE" ] || [[ " $STAGES " != *" $STAGE "* ]]; then
  if [ -n "$STAGE" ]; then echo "unknown stage $STAGE"; fi
  awk 'NR > 1 && !/^#/ {exit} NR > 1 {print}' "$SELF"
  exit 1
fi
if [ -z "${EDDM_DETACHED:-}" ]; then
  mkdir -p "$LOGS"
  EDDM_DETACHED=1 nohup bash "$SELF" "$STAGE" >> "$LOGS/$STAGE.log" 2>&1 < /dev/null &
  echo "started $STAGE (process $!); log $LOGS/$STAGE.log"
  exit 0
fi
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
if ! command -v python > /dev/null 2>&1; then python() { python3 "$@"; }; fi
say() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
now() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
mkdir -p "$BASE/locks" "$LOGS"
exec 9> "$BASE/locks/$STAGE.lock"
if command -v flock > /dev/null 2>&1 && ! flock -n 9; then
  say "stage $STAGE is already running in another process (its lock $BASE/locks/$STAGE.lock is held); this start stops"
  exit 1
fi
cd "$CODE"
mkdir -p "$MODELS" "$OUTS" "$ITEMS" "$ITEMS_CHECK" "$LARGE_ITEMS" "$NOT_BUILT" "$DONE" \
  "$STAGE_DIR"/{predictions,logits,test/predictions,test/logits} "$RES"/{metrics,logs,predictions,logits,run_summaries}
say "stage $STAGE started"

# ---------------------------------------------------------------- helpers

json_get() {  # json_get <file> <python expression on the dict d>
  python -c "import json; d = json.load(open('$1')); print($2)"
}

json_true() {  # json_true <file> <python expression on d>: true when the file exists and the expression is true
  [ -f "$1" ] && [ "$(python -c "import json; d = json.load(open('$1')); print(bool($2))" 2> /dev/null || true)" = True ]
}

check_rows() {  # check_rows <file> <rows>
  local n
  n=$(($(wc -l < "$1") - 1))
  if [ "$n" != "$2" ]; then say "$1 has $n rows, expected $2; stopping"; exit 1; fi
}

rows_ok() {  # rows_ok <file> <rows>: true if the file exists with that many rows
  [ -f "$1" ] && [ "$(($(wc -l < "$1") - 1))" = "$2" ]
}

count_records() {  # count_records <records file>: its records (non-empty lines); called inside $(...), so it complains on stderr
  if [ ! -f "$1" ]; then say "$1 does not exist; stopping" >&2; exit 1; fi
  grep -c . "$1" || true
}

mark_done() {  # mark_done <stage>: every command of the stage succeeded
  now > "$DONE/$1.txt"
}

write_not_built() {  # write_not_built <stage> <reason>
  printf '%s (written %s)\n' "$2" "$(now)" > "$NOT_BUILT/$1.txt"
  say "$2; marker $NOT_BUILT/$1.txt written; the stage ends successfully so that all continues"
}

not_built_on_failure() {  # not_built_on_failure <stage> <what> <command...>: if the command fails, the stage's NOT BUILT marker
  local stage="$1" what="$2" rc=0
  shift 2
  "$@" || rc=$?
  if [ "$rc" != 0 ]; then
    write_not_built "$stage" "NOT BUILT: $stage, $what failed (exit $rc; the error is in logs/$stage.log)"
    exit 0
  fi
}

require_osf() {  # the OSF update's filing (Claude writes osf_filed.txt on the pod once docs/log.md records it)
  if [ ! -f "$BASE/osf_filed.txt" ]; then
    say "STOP: $BASE/osf_filed.txt is missing (the lines 'link <the OSF update's address>' and 'filed_utc <time>', written by Claude once docs/log.md records the filing); stage $STAGE does not run"
    exit "$STOP_OSF"
  fi
  python - "$BASE/osf_filed.txt" <<'PY'
import sys
lines = {}
for line in open(sys.argv[1], encoding="utf-8"):
    if " " in line.strip():
        key, value = line.strip().split(" ", 1)
        lines[key] = value.strip()
if not lines.get("link") or not lines.get("filed_utc"):
    sys.exit(f"{sys.argv[1]} needs the lines 'link <the OSF update's address>' and 'filed_utc <time>'")
print(f"the OSF update: {lines['link']}, filed {lines['filed_utc']}")
PY
}

gpu_used_mib() {  # the GPU's used memory in MiB (a container may not list other processes, so memory is what is checked)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

gpu_alone() {  # stops unless the GPU's used memory is under 1,000 MiB (waits up to 5 minutes for a leaving process)
  local used waited=0
  used=$(gpu_used_mib)
  while [ "${used:-0}" -ge 1000 ]; do
    if [ "$waited" -ge 300 ]; then say "the GPU holds $used MiB; this stage runs alone on the GPU; stopping"; nvidia-smi; exit 1; fi
    sleep 15
    waited=$((waited + 15))
    used=$(gpu_used_mib)
  done
  say "GPU free: $used MiB used"
}

qwen_items() {  # qwen_items <out> <max_len> <questions> <records...>
  local out="$1" max_len="$2" questions="$3"
  shift 3
  python src/train/qwen_items.py --records "$@" --questions "$questions" --tokenizer-dir "$MODELS/qwen3_4b_base" --out "$out" \
    --max-len "$max_len"
}

no_cuts() {  # no_cuts <items summary>: stops if any record was cut or skipped
  if [ "$(json_get "$1" "d['skipped'] == 0 and d['truncated'] == 0")" != "True" ]; then
    say "$1: records were cut or skipped ($(json_get "$1" "(d['skipped'], d['truncated'])")); stopping"; exit 1
  fi
}

build_items() {  # build_items <key> <max_len> <records> [<folder>]: <folder>/<key>.pt (default the items folder), unless it was
  # built before from the same records file at the same length; stops if a record is cut or skipped
  local key="$1" max_len="$2" records="$3" folder="${4:-$ITEMS}" n
  n=$(count_records "$records")
  if [ -f "$folder/$key.pt" ] && json_true "$folder/$key.summary.json" "d.get('max_len') == $max_len and d.get('records') == $n and d.get('files') == ['$records']"; then
    say "items $key: built before ($n records at $max_len tokens)"
  else
    qwen_items "$folder/$key.pt" "$max_len" "$QUESTIONS" "$records"
  fi
  no_cuts "$folder/$key.summary.json"
}

phase_done() {  # phase_done <system> <weights file>: the phase finished (its metrics JSON is written last)
  [ -f "$RES/metrics/$1_metrics.json" ] && [ -f "$2" ]
}

predict_split() {  # predict_split <system> <phase dir> <records dir> <split> <out dir> <logits dir> <run summary> <rows>
  local system="$1" phase="$2" records="$3" split="$4" out="$5" logits="$6" summary="$7" rows="$8"
  say "$system predicts $split from $records ($rows rows); GPU memory used now: $(gpu_used_mib) MiB"
  python src/train/train_qwen_head.py predict --base-dir "$MODELS/qwen3_4b_base" --phase-dir "$phase" --system "$system" \
    --records-dir "$records" --qids "$EI" --splits "$split" --questions "$QUESTIONS" --batch-size 4 --out "$out" \
    --logits-dir "$logits" --run-summary "$summary"
  if [ "$(json_get "$summary" "d['items_$split']['skipped'] == 0 and d['items_$split']['truncated'] == 0")" != "True" ]; then
    say "$system $split: records were cut or skipped in predict; stopping"; exit 1
  fi
  check_rows "$out/${system}-raw_${EI}_${split}.csv" "$rows"
  check_rows "$logits/${system}_${EI}_${split}_logits.csv" "$rows"
}

predict_validation() {  # predict_validation <system> <phase dir> <records dir> <rows>: the whole split, then copied into results
  local system="$1"
  predict_split "$system" "$2" "$3" validation "$STAGE_DIR/predictions" "$STAGE_DIR/logits" \
    "$RES/run_summaries/${system}_validation_run_summary.json" "$4"
  cp "$STAGE_DIR/predictions/${system}-raw_${EI}_validation.csv" "$RES/predictions/"
  cp "$STAGE_DIR/logits/${system}_${EI}_validation_logits.csv" "$RES/logits/"
  say "$system: validation predictions and logits in results ($4 rows)"
}

v05_max_len() {  # v0.5's EI recipe's length (recipe_selection.json's v05_ei.max_len), empty before the train stage has finished
  if json_true "$RES/recipe_selection.json" "'v05_ei' in d"; then json_get "$RES/recipe_selection.json" "d['v05_ei']['max_len']"; fi
}

long_only() {  # the large stages run only under GEMMA ON THE LONGER INPUT
  local len
  len=$(v05_max_len)
  if [ "$len" != 6144 ]; then
    say "GEMMA ON THE LONGER INPUT does not apply (v05_ei.max_len is ${len:-not decided: the train stage has not finished}); stage $STAGE does not run"
    exit 1
  fi
}

finished_or_not_built() {  # finished_or_not_built <stage>: true when the stage finished or wrote its NOT BUILT marker
  [ -f "$DONE/$1.txt" ] || [ -f "$NOT_BUILT/$1.txt" ]
}

copy_long() {  # copy_long <split>: large_model_session8.py's gemma4_31b files copied to the name gemma4_31b_long (originals kept)
  local d="$RES/large_model"
  cp "$d/gemma4_31b-raw_${EI}_$1.csv" "$d/gemma4_31b_long-raw_${EI}_$1.csv"
  cp "$d/gemma4_31b_logprobs_${EI}_$1.csv" "$d/gemma4_31b_long_logprobs_${EI}_$1.csv"
  cp "$d/gemma4_31b_letters.json" "$d/gemma4_31b_long_letters.json"
  cp "$d/gemma4_31b_timing.json" "$d/gemma4_31b_long_timing.json"
  say "large model ($1): gemma4_31b's files copied to the name gemma4_31b_long in results/fiorillo_v0_5/large_model/"
}

memory_setting() {  # memory_setting <base|long>: MB, GA and CKPT from the pilot's choice (session 8's pilot.json, or pilot_long.json)
  local file
  case "$1" in
    base) file="$RES8/pilot.json" ;;
    long) file="$RES/pilot_long.json" ;;
    *) say "unknown memory setting $1; stopping"; exit 1 ;;
  esac
  if [ ! -f "$file" ]; then say "$file does not exist; stopping"; exit 1; fi
  if json_true "$file" "d.get('stop_and_ask') or not d.get('chosen')"; then say "$file says no setting passed; stopping"; exit 1; fi
  MB=$(json_get "$file" "d['chosen']['micro_batch']")
  GA=$(json_get "$file" "d['chosen']['grad_accum']")
  CKPT=""
  if [ "$(json_get "$file" "d['chosen']['gradient_checkpointing']")" = "True" ]; then CKPT="--gradient-checkpointing"; fi
}

train_run() {  # train_run: the recipe's train action (EDDM_* from recipe_session9.py next), trained from s7_b3
  local system="$EDDM_NAME" rc=0 ckpt flag="$BASE/checkpointing_on_${EDDM_MEMORY_SETTING}.txt"
  if phase_done "$system" "$MODELS/$system/adapter.safetensors"; then say "train: $system already trained; not trained again"; return 0; fi
  memory_setting "$EDDM_MEMORY_SETTING"
  ckpt="$CKPT"
  if [ -z "$ckpt" ] && [ -f "$flag" ]; then ckpt="--gradient-checkpointing"; fi
  say "train: $system (recipe $EDDM_RECIPE, seed $EDDM_SEED, epochs $EDDM_EPOCHS, adapters' learning rate $EDDM_LR_ADAPTERS," \
      "max_len $EDDM_MAX_LEN, items $EDDM_TRAIN_ITEMS and $EDDM_MONITOR_ITEMS); memory setting $EDDM_MEMORY_SETTING: micro-batch $MB," \
      "accumulation $GA, ${ckpt:-no gradient checkpointing}"
  run_train() {
    # shellcheck disable=SC2086
    python src/train/train_qwen_head.py train --base-dir "$MODELS/qwen3_4b_base" --init "$MODELS/s7_b3" \
      --train "$ITEMS/$EDDM_TRAIN_ITEMS.pt" --val "$ITEMS/$EDDM_MONITOR_ITEMS.pt" --out "$MODELS/$system" --system-name "$system" \
      --seed "$EDDM_SEED" --epochs "$EDDM_EPOCHS" --lr-adapters "$EDDM_LR_ADAPTERS" --max-len "$EDDM_MAX_LEN" --keep last \
      --freeze-steps 0 --eval-every 80 --micro-batch "$MB" --grad-accum "$GA" \
      --metrics "$RES/metrics/${system}_metrics.json" --log "$RES/logs/${system}_train_log.csv" $1 2>&1 | tee "$LOGS/${system}.out"
  }
  run_train "$ckpt" || rc=$?
  if [ "$rc" != 0 ]; then
    if [ -z "$ckpt" ] && grep -qi "out of memory" "$LOGS/${system}.out"; then
      say "train: $system ran out of memory without gradient checkpointing; restarting it from its start with it on (kept on for later runs of the $EDDM_MEMORY_SETTING setting)"
      echo "$system ran out of memory at $(now); gradient checkpointing on from here for the $EDDM_MEMORY_SETTING setting" >> "$flag"
      run_train "--gradient-checkpointing"
    else
      say "train: $system failed (exit $rc)"
      exit 1
    fi
  fi
  say "train: $system trained"
}

ensure_items() {  # ensure_items <key> <records> <max_len>: the items a train action names (built here when missing, as for the
  # combination's training records); stops if items built earlier have another length
  local key="$1" records="$2" max_len="$3"
  if [ ! -f "$ITEMS/$key.pt" ]; then
    say "items $key are missing; built now from $records at $max_len tokens"
    build_items "$key" "$max_len" "$records"
  fi
  if [ "$(json_get "$ITEMS/$key.summary.json" "d['max_len']")" != "$max_len" ]; then
    say "items $key were built at $(json_get "$ITEMS/$key.summary.json" "d['max_len']") tokens, the recipe reads $max_len; stopping"; exit 1
  fi
}

cost_rule() {  # THE PILOT AND THE COST RULE: cost_rule.json before the first run; exit 4 is its stop (Claude asks Johann)
  local rc=0
  if [ -f "$RES/cost_rule.json" ]; then
    if json_true "$RES/cost_rule.json" "(d.get('stops') or d.get('maintenance_before_the_end')) and 'johann_decision' not in d"; then
      say "THE COST RULE'S STOP: cost_rule.json (projected total \$$(json_get "$RES/cost_rule.json" "round(d['projected_total_usd'], 2)")," \
          "stop line \$$(json_get "$RES/cost_rule.json" "d['stop_line_usd']"), maintenance before the end:" \
          "$(json_get "$RES/cost_rule.json" "d.get('maintenance_before_the_end')")) has no johann_decision yet; the train stage stops before" \
          "its first run, and Claude asks Johann (the options in cost_rule.json's options_in_order)"
      exit "$STOP_COST"
    fi
    say "the cost rule: cost_rule.json exists (projected total \$$(json_get "$RES/cost_rule.json" "round(d['projected_total_usd'], 2)")," \
        "johann_decision: $(json_get "$RES/cost_rule.json" "d.get('johann_decision', 'none needed')"), dropped:" \
        "$(json_get "$RES/cost_rule.json" "d.get('dropped') or []")); the train stage continues"
    return 0
  fi
  python src/eval/recipe_session9.py cost --pod-rate "$BASE/pod_rate.txt" --results "$RES" || rc=$?
  if [ "$rc" = "$STOP_COST" ]; then
    say "THE COST RULE'S STOP: the projection is above the stop line or the copy would end after an announced maintenance" \
        "(cost_rule.json); the train stage stops before its first run, and Claude asks Johann"
    exit "$STOP_COST"
  fi
  if [ "$rc" != 0 ]; then say "the cost rule failed (exit $rc); stopping"; exit 1; fi
}

RECIPE_KEYS="ACTION NAME RECIPE SEED PARTS LR_CHOICE TRAIN_RECORDS MONITOR_RECORDS TRAIN_ITEMS MONITOR_ITEMS EPOCHS LR_ADAPTERS
  MAX_LEN MEMORY_SETTING VALIDATION_RECORDS_DIR EXCHANGED_VALIDATION_DIR SOURCE_RUNS PHASE_SYSTEM RECORDS_DIR ROWS"

clear_recipe_vars() {  # the previous action's fields, cleared before the next action is read
  local k
  for k in $RECIPE_KEYS; do unset "EDDM_$k"; done
}

trees_running() {  # true while a trees stage holds its lock (all's own, or one started by hand)
  [ -f "$BASE/locks/trees.lock" ] && command -v flock > /dev/null 2>&1 && ! flock -n "$BASE/locks/trees.lock" true
}

# ---------------------------------------------------------------- all: the chain (steps 10 to 13)

all_stop() {  # all_stop <stage> <exit status>: the chain stops at a stage; says which stop applied and exits with its status
  local stage="$1" rc="$2" code=1 kind="the first failure"
  if [ "$stage" = checks_s8 ] && [ "$rc" = "$STOP_CHECKS" ] && json_true "$RES/checks_s8.json" "d.get('stop')"; then
    kind="THE READING CHECKS' STOP (s8_ei's macro-F1 with no article above 0.85; checks_s8.json): Claude tells Johann at once"
    code=$STOP_CHECKS
  elif [ "$stage" = train ] && [ "$rc" = "$STOP_COST" ] && \
       json_true "$RES/cost_rule.json" "(d.get('stops') or d.get('maintenance_before_the_end')) and 'johann_decision' not in d"; then
    kind="THE COST RULE'S STOP (cost_rule.json): Claude asks Johann"
    code=$STOP_COST
  elif [ "$rc" = "$STOP_OSF" ] && [ ! -f "$BASE/osf_filed.txt" ]; then
    kind="osf_filed.txt is missing"
    code=$STOP_OSF
  elif [ "$rc" != 0 ] && [ "$rc" != "$STOP_CHECKS" ] && [ "$rc" != "$STOP_COST" ] && [ "$rc" != "$STOP_OSF" ]; then
    code=$rc
  fi
  say "all: STOPPED at $stage: $kind (the stage's exit status $rc; its log $LOGS/$stage.log); the stages after it were not started"
  if [ -n "${TREES_PID:-}" ]; then
    say "all: the trees stage beside train (process $TREES_PID) is left to finish on its own; all, run again, waits for it"
  fi
  exit "$code"
}

run_stage() {  # run_stage <stage>: one stage of all, in all's own process tree (not detached), its output in the stage's log
  local stage="$1" rc=0
  if [ -f "$DONE/$stage.txt" ]; then say "all: $stage finished before ($(cat "$DONE/$stage.txt")); not run again"; return 0; fi
  if [ -f "$NOT_BUILT/$stage.txt" ]; then say "all: $stage has its NOT BUILT marker ($(cat "$NOT_BUILT/$stage.txt")); not run again"; return 0; fi
  say "all: $stage started (log $LOGS/$stage.log)"
  EDDM_DETACHED=1 bash "$SELF" "$stage" >> "$LOGS/$stage.log" 2>&1 < /dev/null 9>&- || rc=$?
  if [ "$rc" != 0 ]; then all_stop "$stage" "$rc"; fi
  if [ -f "$NOT_BUILT/$stage.txt" ]; then
    say "all: $stage wrote its NOT BUILT marker ($(cat "$NOT_BUILT/$stage.txt")); the chain continues"
  else
    say "all: $stage finished"
  fi
}

start_trees() {  # the trees stage beside train, on the CPU, as all's own child
  TREES_PID=""
  if [ -f "$DONE/trees.txt" ]; then say "all: trees finished before ($(cat "$DONE/trees.txt")); not run again"; return 0; fi
  if trees_running; then say "all: a trees stage is running already; all waits for it after train"; return 0; fi
  say "all: trees started beside train (log $LOGS/trees.log)"
  EDDM_DETACHED=1 bash "$SELF" trees >> "$LOGS/trees.log" 2>&1 < /dev/null 9>&- &
  TREES_PID=$!
}

wait_trees() {  # after train: the trees stage beside it must have finished
  local rc=0
  if [ -n "${TREES_PID:-}" ]; then
    say "all: waiting for the trees stage beside train (process $TREES_PID)"
    wait "$TREES_PID" || rc=$?
    TREES_PID=""
  fi
  if trees_running; then
    say "all: waiting for the trees stage that holds its lock"
    flock "$BASE/locks/trees.lock" true
  fi
  if [ -f "$DONE/trees.txt" ]; then say "all: trees finished"; return 0; fi
  if [ "$rc" = 0 ]; then rc=1; fi
  all_stop trees "$rc"
}

# ---------------------------------------------------------------- the stages

case "$STAGE" in
  prepare)
    # the uploads first, so that a missing or changed file stops the stage before the installs and downloads
    python - "$BASE/uploads_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad = []
for root, files in manifest["roots"].items():
    base = Path(root)
    for rel, digest in files.items():
        p = base / rel
        if not p.exists() or sha(p) != digest:
            bad.append(str(p))
if bad:
    sys.exit(f"{len(bad)} uploaded files missing or changed, for example {bad[:5]}")
print(f"uploads present and checked: {sum(len(v) for v in manifest['roots'].values())} files")
PY
    python - "$BASE/pod_rate.txt" <<'PY'
import datetime, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    sys.exit(f"{path} does not exist: Claude writes it in step 9 (the lines 'created_utc <YYYY-MM-DDTHH:MM:SSZ>' and 'hourly_usd <rate>')")
lines = {}
for line in path.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) >= 2:
        lines[parts[0]] = parts[1]
try:
    datetime.datetime.strptime(lines["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
    rate = float(lines["hourly_usd"])
    maintenance = lines.get("maintenance_utc", "none")
    if maintenance not in ("none", "None"):
        datetime.datetime.strptime(maintenance, "%Y-%m-%dT%H:%M:%SZ")
except (KeyError, ValueError) as error:
    sys.exit(f"{path} needs the lines 'created_utc <YYYY-MM-DDTHH:MM:SSZ>' and 'hourly_usd <rate>', and 'maintenance_utc' "
             f"<YYYY-MM-DDTHH:MM:SSZ or none> if any ({error!r})")
print(f"pod_rate.txt: pod created {lines['created_utc']}, ${rate} an hour, maintenance {maintenance}")
PY
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
    pip install -q "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "hf-xet==1.6.0" "safetensors==0.8.0" "pyyaml==6.0.3" \
      "pyarrow==25.0.1" "scipy==1.17.1" "numpy==2.4.6" "accelerate==1.15.0" "laya==0.3.20" "requests==2.34.2" \
      "scikit-learn==1.9.1" "pandas==3.0.6"
    python -c "import transformers, tokenizers, huggingface_hub, laya, numpy, scipy, sklearn, pandas, torch; print('transformers', transformers.__version__, '| tokenizers', tokenizers.__version__, '| huggingface_hub', huggingface_hub.__version__, '| laya', laya.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| scikit-learn', sklearn.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__)"
    python - "$QWEN_REV" "$MODELS/qwen3_4b_base" <<'PY'
import sys
from huggingface_hub import snapshot_download
rev, out = sys.argv[1:3]
snapshot_download("Qwen/Qwen3-4B-Base", revision=rev, local_dir=out)
print("Qwen3-4B-Base downloaded at", rev)
PY
    printf 'revision %s\ndownloaded with huggingface_hub snapshot_download (Qwen/Qwen3-4B-Base)\n' "$QWEN_REV" > "$MODELS/qwen3_4b_base/REVISION.txt"
    python - "$RES/fixed_versions.json" "$MODELS/qwen3_4b_base" "$QWEN_REV" <<'PY'
import hashlib, json, sys
from pathlib import Path
fixed, folder, rev = json.load(open(sys.argv[1]))["qwen3_4b_base"], Path(sys.argv[2]), sys.argv[3]
assert fixed["revision"] == rev, f"fixed_versions.json fixes revision {fixed['revision']}, the driver {rev}"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
for name, want in fixed["weights"].items():
    p = folder / name
    assert p.exists(), f"{p} is missing"
    assert p.stat().st_size == want["bytes"], f"{name}: {p.stat().st_size} bytes, fixed_versions.json says {want['bytes']}"
    got = sha(p)
    assert got == want["sha256"], f"{name}: SHA-256 {got}, fixed_versions.json says {want['sha256']}"
print(f"Qwen3-4B-Base at {rev}: its {len(fixed['weights'])} weight files have the sizes and SHA-256 that fixed_versions.json records")
PY
    if [ ! -f "$BASE/expected_checksums.json" ]; then cp results/public_v0/checksums.json "$BASE/expected_checksums.json"; fi
    python src/tasks/pubmedqa.py
    python src/tasks/uci_readmission.py
    python - "$BASE/expected_checksums.json" data/processed/public "$BASE/records_manifest_s4_pod.json" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
expected, folder, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
files = {}
for name, want in sorted(expected.items()):
    got = sha(folder / name) if (folder / name).exists() else None
    files[name] = {"sha256": got, "expected": want, "equal": got == want}
check = {"plan": "docs/prompt_session9.md, STEP 3 (prepare): session 4's records rebuilt on the pod",
         "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "expected_from": "results/public_v0/checksums.json", "files": files, "all_equal": all(v["equal"] for v in files.values())}
out.write_text(json.dumps(check, indent=1) + "\n", encoding="utf-8")
if not check["all_equal"]:
    sys.exit(f"record SHA-256 values differ from session 4's: {sorted(k for k, v in files.items() if not v['equal'])}")
print(f"session 4's records rebuilt: {len(files)} files, SHA-256 identical to results/public_v0/checksums.json")
PY
    cp "$BASE/records_manifest_s4_pod.json" "$RES/records_manifest_s4_pod.json"
    for attempt in 1 2 3 4 5; do  # the download resumes; each call runs until it is complete or its time is up
      status=$(python src/tasks/evidence_inference.py download --raw data/raw/evidence_inference --max-seconds 3000 2>&1 | tail -n 1 || true)
      echo "$status"
      if python -c "import json, sys; sys.exit(0 if json.loads(sys.argv[1])['complete'] else 1)" "$status" 2> /dev/null; then break; fi
      say "EI download: attempt $attempt incomplete or failed; resuming after 30 s"
      sleep 30
    done
    python - "$RES8/records_manifest.json" data/raw/evidence_inference <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, raw = json.load(open(sys.argv[1])), Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
expected = manifest["sha256"]["raw"]
bad = [k for k, v in expected.items() if not (raw / k).exists() or sha(raw / k) != v]
if bad:
    sys.exit(f"{len(bad)} of EI's raw files differ from results/fiorillo_v0_4/records_manifest.json, for example {bad[:3]}")
print(f"EI's raw files: {len(expected)} files, each identical to results/fiorillo_v0_4/records_manifest.json")
PY
    python src/tasks/evidence_inference.py from-selection --raw data/raw/evidence_inference --selection "$REC8/article_selection.jsonl" \
      --out-dir "$REC8" --manifest "$RES8/records_manifest.json" --check-out "$BASE/records_manifest_s8_pod.json"
    cp "$BASE/records_manifest_s8_pod.json" "$RES/records_manifest_s8_pod.json"
    # correction 7: rebuild writes long/article_selection_long.jsonl inside --out, so it reads a kept copy of the upload
    mkdir -p "$BASE/inputs"
    [ -f "$BASE/inputs/article_selection_long_uploaded.jsonl" ] || cp "$REC9/long/article_selection_long.jsonl" "$BASE/inputs/article_selection_long_uploaded.jsonl"
    python src/tasks/ei_session9.py rebuild --raw data/raw/evidence_inference --s8-dir "$REC8" --licenses "$RES/license_recheck.csv" \
      --selection-long "$BASE/inputs/article_selection_long_uploaded.jsonl" --out "$REC9" --manifest "$RES/records_manifest.json" \
      --check-out "$BASE/records_manifest_pod.json"
    cp "$BASE/records_manifest_pod.json" "$RES/records_manifest_pod.json"
    python - "$RES/records_manifest.json" "$MODELS" <<'PY'
import hashlib, json, sys
from pathlib import Path
models_manifest, models = json.load(open(sys.argv[1]))["models"], Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad, n = [], 0
for name in ("s7_b3", "s8_ei_1", "s8_ei_2", "s8_ei_3"):
    files = models_manifest.get(name) or {}
    if not files:
        bad.append(f"{name}: records_manifest.json lists no files")
    for rel, digest in files.items():
        p = models / name / rel
        n += 1
        if not p.exists() or sha(p) != digest:
            bad.append(str(p))
want = models_manifest.get("s7_b3_adapter_expected")
if want and (models_manifest.get("s7_b3") or {}).get("adapter.safetensors") != want:
    bad.append("s7_b3's adapter in records_manifest.json differs from session 8's record")
if bad:
    sys.exit(f"the uploaded model folders differ from results/fiorillo_v0_5/records_manifest.json's models digests: {bad[:5]}")
print(f"s7_b3's and s8_ei's folders: {n} files, each identical to results/fiorillo_v0_5/records_manifest.json's models digests")
PY
    build_items clean_train 4096 "$REC9/clean/${EI}_train.jsonl"
    build_items swap_train 4096 "$REC9/swap/${EI}_train.jsonl"
    build_items long_train 6144 "$REC9/long/${EI}_train.jsonl"
    build_items clean_monitor 4096 "$REC9/clean/${EI}_validation_monitor.jsonl"
    build_items long_monitor 6144 "$REC9/long/${EI}_validation_monitor.jsonl"
    # built once, only to check that the reading checks' inputs and the long validation and test records are not cut
    build_items check_none_test 4096 "$REC9/checks/none/${EI}_test.jsonl" "$ITEMS_CHECK"
    build_items check_title_test 4096 "$REC9/checks/title/${EI}_test.jsonl" "$ITEMS_CHECK"
    build_items check_swap_test 4096 "$REC9/checks/swap/${EI}_test.jsonl" "$ITEMS_CHECK"
    build_items check_swap_validation 4096 "$REC9/checks/swap/${EI}_validation.jsonl" "$ITEMS_CHECK"
    build_items check_swap_long_test 6144 "$REC9/checks/swap_long/${EI}_test.jsonl" "$ITEMS_CHECK"
    build_items check_swap_long_validation 6144 "$REC9/checks/swap_long/${EI}_validation.jsonl" "$ITEMS_CHECK"
    build_items check_long_validation 6144 "$REC9/long/${EI}_validation.jsonl" "$ITEMS_CHECK"
    build_items check_long_test 6144 "$REC9/long/${EI}_test.jsonl" "$ITEMS_CHECK"
    python - "$ITEMS" "$ITEMS_CHECK" <<'PY'
import json, sys
from pathlib import Path
for folder in sys.argv[1:3]:
    out = {p.name.replace(".summary.json", ""): json.load(open(p)) for p in sorted(Path(folder).glob("*.summary.json"))}
    print(f"items in {folder} (records, items, skipped, cut, max_len):")
    print(json.dumps({k: [v.get(x) for x in ("records", "items", "skipped", "truncated", "max_len")] for k, v in out.items()}))
PY
    df -h "$BASE" | tail -n 1
    ;;
  checks_s8)
    require_osf
    gpu_alone
    for k in 1 2 3; do
      for input in none title swap; do
        system="s8_ei_${k}_${input}"
        records="$REC9/checks/$input"
        rows=$(count_records "$records/${EI}_test.jsonl")
        if rows_ok "$RES/checks_s8/${system}-raw_${EI}_test.csv" "$rows" && rows_ok "$RES/checks_s8/logits/${system}_${EI}_test_logits.csv" "$rows"; then
          say "$system: its predictions exist ($rows rows); skipped"
          continue
        fi
        mkdir -p "$RES/checks_s8/logits"
        predict_split "$system" "$MODELS/s8_ei_$k" "$records" test "$RES/checks_s8" "$RES/checks_s8/logits" \
          "$RES/checks_s8/${system}_run_summary.json" "$rows"
      done
    done
    if json_true "$RES/checks_s8.json" "d.get('stop') and 'johann_decision' in d"; then
      say "checks_s8.json holds the stop and Johann's decision ($(json_get "$RES/checks_s8.json" "d['johann_decision']")); not scored again; the chain continues"
    else
      rc=0
      python src/eval/reading_checks_session9.py score --system s8_ei --results "$RES" || rc=$?
      if [ "$rc" = "$STOP_CHECKS" ]; then
        say "THE READING CHECKS' STOP: s8_ei's macro-F1 with no article is above 0.85 (checks_s8.json); the chain stops before the" \
            "train stage, and Claude tells Johann at once with the checks' numbers; after his decision, Claude adds a" \
            "\"johann_decision\" key to checks_s8.json and runs all again"
        exit "$STOP_CHECKS"
      fi
      if [ "$rc" != 0 ]; then say "reading_checks_session9.py score failed (exit $rc); stopping"; exit 1; fi
    fi
    ;;
  pilot_long)
    if [ -f "$RES/pilot_long.json" ]; then
      say "pilot_long.json exists (written $(json_get "$RES/pilot_long.json" "d.get('written_utc')")); the pilot is not run again"
    else
      gpu_alone
      python src/train/train_qwen_head.py pilot-s8 --base-dir "$MODELS/qwen3_4b_base" --train "$ITEMS/long_train.pt" \
        --val "$ITEMS/long_monitor.pt" --headroom-gib 0 --out-json "$RES/pilot_long.json"
    fi
    if json_true "$RES/pilot_long.json" "d.get('stop_and_ask') or not d.get('chosen')"; then
      reason=$(python - "$RES/pilot_long.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
parts = []
for s in d.get("settings", []):
    peak = s.get("peak_gpu_memory_gib")
    state = "out of memory" if s.get("out_of_memory") else (f"peak {peak:.1f} GiB" if peak is not None else "no peak measured")
    parts.append(f"setting {s.get('setting')} (micro-batch {s.get('micro_batch')}, accumulation {s.get('grad_accum')}, gradient "
                 f"checkpointing {'on' if s.get('gradient_checkpointing') else 'off'}): {state}")
print("no setting of pilot_long passed at 6,144 tokens (a setting passes if it did not run out of memory and its peak is at "
      "most 76 GiB): " + "; ".join(parts))
PY
)
      write_not_built pilot_long "NOT BUILT: long, $reason"
      exit 0
    fi
    say "pilot_long: the long runs use setting $(json_get "$RES/pilot_long.json" "d['chosen']['setting']") (micro-batch" \
        "$(json_get "$RES/pilot_long.json" "d['chosen']['micro_batch']"), accumulation $(json_get "$RES/pilot_long.json" "d['chosen']['grad_accum']")," \
        "gradient checkpointing $(json_get "$RES/pilot_long.json" "d['chosen']['gradient_checkpointing']"), training items a second" \
        "$(json_get "$RES/pilot_long.json" "d['chosen']['train_items_per_second']"), peak $(json_get "$RES/pilot_long.json" "d['chosen']['peak_gpu_memory_gib']") GiB)"
    ;;
  train)
    cost_rule
    gpu_alone
    last=""
    while true; do
      clear_recipe_vars
      out=$(python src/eval/recipe_session9.py next --format env --results "$RES" --not-built-dir "$NOT_BUILT")
      other=$(printf '%s\n' "$out" | grep -v '^EDDM_[A-Z0-9_]*=' || true)
      if [ -n "$other" ]; then say "recipe_session9.py next also printed (not read): $other"; fi
      # only its assignments (one a line, each value quoted by shlex.quote) are read
      eval "$(printf '%s\n' "$out" | grep '^EDDM_[A-Z0-9_]*=' || true)"
      step="${EDDM_ACTION:-} ${EDDM_NAME:-} ${EDDM_PARTS:-}"
      if [ "$step" = "$last" ]; then say "recipe_session9.py next asked for the same step twice ($step): the step did not take; stopping"; exit 1; fi
      last="$step"
      case "${EDDM_ACTION:-}" in
        train)
          ensure_items "$EDDM_TRAIN_ITEMS" "$EDDM_TRAIN_RECORDS" "$EDDM_MAX_LEN"
          ensure_items "$EDDM_MONITOR_ITEMS" "$EDDM_MONITOR_RECORDS" "$EDDM_MAX_LEN"
          train_run
          predict_validation "$EDDM_NAME" "$MODELS/$EDDM_NAME" "$EDDM_VALIDATION_RECORDS_DIR" "$N_VAL"
          say "train: $EDDM_NAME finished (its validation predictions, logits, metrics, log and run summary are in results)"
          ;;
        combo_records)
          len=4096
          if [[ " $EDDM_PARTS " == *" long "* ]]; then len=6144; fi
          say "the combination's records: parts $EDDM_PARTS, at $len tokens"
          # shellcheck disable=SC2086
          python src/tasks/ei_session9.py combo --parts $EDDM_PARTS --out "$REC9"
          qwen_items "$ITEMS/combo_train.pt" "$len" "$QUESTIONS" "$REC9/combo/${EI}_train.jsonl"
          no_cuts "$ITEMS/combo_train.summary.json"
          ;;
        wavg)
          read -r -a src_runs <<< "$EDDM_SOURCE_RUNS"
          if [ "${#src_runs[@]}" -lt 2 ] || [ "${#src_runs[@]}" -gt 3 ]; then say "the weight average takes the winner's two or three runs (correction 7), not '$EDDM_SOURCE_RUNS'; stopping"; exit 1; fi
          if [ -f "$MODELS/$EDDM_NAME/adapter.safetensors" ] && [ -f "$RES/weight_average.json" ]; then
            say "$EDDM_NAME was built before; not built again"
          else
            say "the weight average $EDDM_NAME from ${src_runs[*]}"
            wavg_runs=()
            for r in "${src_runs[@]}"; do wavg_runs+=("$MODELS/$r"); done
            python src/eval/weight_average_session9.py build --runs "${wavg_runs[@]}" \
              --out "$MODELS/$EDDM_NAME" --results "$RES" --device cuda
          fi
          predict_validation "$EDDM_NAME" "$MODELS/$EDDM_NAME" "$EDDM_VALIDATION_RECORDS_DIR" "$N_VAL"
          ;;
        exch)
          say "the exchanged validation records for $EDDM_PHASE_SYSTEM (correction 3): $EDDM_NAME"
          predict_validation "$EDDM_NAME" "$MODELS/$EDDM_PHASE_SYSTEM" "$EDDM_RECORDS_DIR" "$EDDM_ROWS"
          ;;
        done)
          say "THE RECIPE RULE is done: v0.5's EI specialist $(json_get "$RES/recipe_selection.json" "d['v05_ei']['system']")" \
              "(recipe $(json_get "$RES/recipe_selection.json" "d['v05_ei']['recipe']"), runs $(json_get "$RES/recipe_selection.json" "' '.join(d['v05_ei']['runs'])")," \
              "max_len $(json_get "$RES/recipe_selection.json" "d['v05_ei']['max_len']"), swap averaging" \
              "$(json_get "$RES/recipe_selection.json" "d['v05_ei']['swap_averaging']"))"
          break
          ;;
        *)
          say "recipe_session9.py next gave an unknown action '${EDDM_ACTION:-}'; stopping"; exit 1 ;;
      esac
    done
    ;;
  trees)
    pip install -q "xgboost==3.2.0" "catboost==1.2.10" "lightgbm==4.7.0" "pandas==3.0.6" "scikit-learn==1.9.1"
    python -c "import xgboost, catboost, lightgbm, pandas, sklearn; print('xgboost', xgboost.__version__, '| catboost', catboost.__version__, '| lightgbm', lightgbm.__version__, '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__)"
    nice -n 10 python src/eval/trees_session9.py recode --results "$RES" --models-dir "$MODELS/trees"
    nice -n 10 python src/eval/trees_session9.py refit --members xgboost_codes catboost_codes --threads "$TREE_THREADS" --results "$RES" \
      --models-dir "$MODELS/trees"
    nice -n 10 python src/eval/trees_session9.py lightgbm --results "$RES" --models-dir "$MODELS/trees"
    ;;
  tabicl)
    gpu_alone
    # the training environment keeps its packages: scikit-learn and pandas at the trees stage's versions (the new environment
    # sees them through --system-site-packages); tabicl and its non-torch requirements go into the new environment only
    not_built_on_failure tabicl "the training environment's scikit-learn and pandas" pip install -q "scikit-learn==1.9.1" "pandas==3.0.6"
    if ! command -v uv > /dev/null 2>&1; then not_built_on_failure tabicl "uv's install" pip install -q uv; fi
    py_bin=$(python -c "import sys; print(sys.executable)")
    if [ ! -x "$TABICL_ENV/bin/python" ]; then
      not_built_on_failure tabicl "the environment" uv venv --python "$py_bin" --system-site-packages "$TABICL_ENV"
    fi
    not_built_on_failure tabicl "tabicl's install" uv pip install --python "$TABICL_ENV/bin/python" --no-deps "tabicl==2.2.0" \
      "einops>=0.7" "psutil==7.2.2" "tqdm==4.70.1"
    train_torch=$(python -c "import torch; print(torch.__version__, torch.__file__)")
    not_built_on_failure tabicl "the environment's check" "$TABICL_ENV/bin/python" - "$train_torch" <<'PY'
import importlib.metadata as m
import sys
import torch
got = f"{torch.__version__} {torch.__file__}"
assert got == sys.argv[1], f"the environment's torch ({got}) is not the training environment's ({sys.argv[1]})"
assert torch.cuda.is_available(), "CUDA is not available in the environment"
from tabicl import TabICLClassifier  # noqa: F401
version = m.version("tabicl")
assert version == "2.2.0", f"tabicl {version}, expected 2.2.0"
print(f"tabicl_env: torch {torch.__version__} from the training environment ({torch.__file__}), CUDA on {torch.cuda.get_device_name(0)};",
      f"tabicl {version}, einops {m.version('einops')}, psutil {m.version('psutil')}, tqdm {m.version('tqdm')}")
PY
    not_built_on_failure tabicl "trees_session9.py tabicl" "$TABICL_ENV/bin/python" src/eval/trees_session9.py tabicl --device cuda \
      --results "$RES" --models-dir "$MODELS/trees"
    if ! rows_ok "$RES/predictions/tabicl_${RW}_validation.csv" "$N_RW_VAL"; then
      write_not_built tabicl "NOT BUILT: tabicl, its validation predictions do not have $N_RW_VAL rows"
      exit 0
    fi
    ;;
  prepare_large)
    long_only
    driver=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
    say "driver CUDA version $driver"
    pip install -q uv
    uv --version
    if [ ! -x "$VENV/bin/python" ]; then uv venv --python 3.12 "$VENV"; fi
    if python -c "import sys; v = tuple(int(x) for x in '$driver'.split('.')[:2]); sys.exit(0 if v >= (13, 0) else 1)"; then
      say "installing vLLM 0.30.0's default build (CUDA 13.0)"
      uv pip install --python "$VENV/bin/python" "vllm==0.30.0" "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "hf-xet==1.6.0"
    else
      say "installing vLLM 0.30.0's CUDA 12.9 build with torch 2.13.0 for CUDA 12.9 (session 8's correction 1)"
      uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
        "torch==2.13.0+cu129" "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "hf-xet==1.6.0"
      # torchcodec from PyPI is built for CUDA 13 (libnvrtc.so.13) and does not load here; vLLM imports it with its
      # multimodal code and uses it only to decode audio and video, so its CPU build is used (session 8's correction 10)
      uv pip install --python "$VENV/bin/python" --no-deps --index-url https://download.pytorch.org/whl/cpu "torchcodec==0.17.0+cpu"
    fi
    # written to a file and run from it, because Triton reads a kernel's source from its file (session 8's correction 8)
    cat > "$BASE/prepare_large_check.py" <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| CUDA build", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not available"
x = torch.ones(1, device="cuda")
print("CUDA starts:", torch.cuda.get_device_name(0))
import vllm._C_stable_libtorch  # noqa: F401  (vLLM 0.30.0's CUDA kernels, imported as vllm/platforms/cuda.py imports them; correction 7)
from vllm import _custom_ops as ops
a = torch.randn(4, 256, device="cuda", dtype=torch.float16)
w = torch.randn(256, device="cuda", dtype=torch.float16)
out = torch.empty_like(a)
ops.rms_norm(out, a, w, 1e-6)
torch.cuda.synchronize()
ref = a.float() * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2), "vLLM's rms_norm kernel gave a wrong result"
print("vLLM's compiled kernels loaded, and one of them (rms_norm) ran and matched torch")
import vllm.multimodal  # noqa: F401  (Gemma 4's engine imports it, and with it torchcodec; correction 10)
print("vllm.multimodal and torchcodec import")
import triton
import triton.language as tl
@triton.jit
def add_one(x_ptr, n, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    tl.store(x_ptr + idx, tl.load(x_ptr + idx, mask=mask) + 1.0, mask=mask)
y = torch.zeros(64, device="cuda", dtype=torch.float32)
add_one[(1,)](y, 64, BLOCK=64)
torch.cuda.synchronize()
assert float(y.sum()) == 64.0, "the Triton kernel gave a wrong result"
print("a Triton kernel compiled and ran")
PY
    if ! "$VENV/bin/python" "$BASE/prepare_large_check.py"; then
      say "vLLM does not import, CUDA does not start, or vLLM's or Triton's kernels do not run in $VENV; stopping, and Johann is asked (session 8's corrections 1 and 5)"
      exit 1
    fi
    GEMMA_REV=$(json_get "$RES/fixed_versions.json" "d['gemma_4_31b_it']['revision']")
    if [ ! -f "$MODELS/gemma4_31b/DOWNLOAD_DONE.txt" ]; then
      if ! python - "$GEMMA_REPO" "$GEMMA_REV" "$MODELS/gemma4_31b" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=out, allow_patterns=["*.json", "*.jinja", "*.safetensors"])
open(f"{out}/DOWNLOAD_DONE.txt", "w").write(f"{repo} at {rev}\n")
print("Gemma 4 31B-it downloaded at", rev)
PY
      then
        say "the download failed; if the model is gated, Johann runs this in the pod's terminal with his own token:"
        say "  hf auth login && python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$GEMMA_REPO', revision='$GEMMA_REV', local_dir='$MODELS/gemma4_31b', allow_patterns=['*.json', '*.jinja', '*.safetensors'])\" && echo done > $MODELS/gemma4_31b/DOWNLOAD_DONE.txt"
        while [ ! -f "$MODELS/gemma4_31b/DOWNLOAD_DONE.txt" ]; do sleep 60; done
      fi
    fi
    python - "$MODELS/gemma4_31b" "$RES/fixed_versions.json" <<'PY'
import json, sys
from pathlib import Path
folder, fixed = Path(sys.argv[1]), json.load(open(sys.argv[2]))
total = sum(p.stat().st_size for p in folder.glob("*.safetensors"))
want = fixed["gemma_4_31b_it"]["weights_bytes"]
assert total == want, f"Gemma's weights are {total} bytes, fixed_versions.json says {want}"
print(f"Gemma 4 31B-it: {total} bytes of weights, as fixed_versions.json says")
PY
    ;;
  smoke_large)
    long_only
    SMOKE_MODEL=$MODELS/gemma4_e2b_smoke
    SMOKE_OUT=$BASE/smoke_large
    E2B_REV=3e22461f65e89153144f8adb70e3b8c2cc9845a7  # google/gemma-4-E2B-it, not gated, the 31B's tokenizer.json
    mkdir -p "$SMOKE_OUT"
    say "smoke test: environment (no GPU work)"
    "$VENV/bin/python" - <<'PY'
import importlib.metadata as m, os, shutil
for p in ("vllm", "torch", "triton", "cuda-bindings", "cuda-python", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu12",
          "nvidia-cutlass-dsl-libs-cu13", "quack-kernels", "apache-tvm-ffi", "flashinfer-python", "transformers", "tokenizers"):
    try:
        print(f"{p:32s} {m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p:32s} not installed")
import cuda.bindings.driver as drv
print("cuda.bindings CUDA_VERSION", drv.CUDA_VERSION, "(below 13000 is needed on a 12.8 driver)")
nvcc = shutil.which("nvcc")
home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or (os.path.dirname(os.path.dirname(nvcc)) if nvcc else "/usr/local/cuda")
print("nvcc for FlashInfer's JIT:", os.path.join(home, "bin", "nvcc"), "exists:", os.path.exists(os.path.join(home, "bin", "nvcc")))
from vllm.vllm_flash_attn.flash_attn_interface import fa_version_unsupported_reason, is_fa_version_supported
print("FA3 supported:", is_fa_version_supported(3), "| FA4 supported:", is_fa_version_supported(4), fa_version_unsupported_reason(4) or "")
PY
    SMOKE_BASE=$(gpu_used_mib)
    say "smoke test: GPU used before (MiB) $SMOKE_BASE"
    if [ ! -f "$SMOKE_MODEL/DOWNLOAD_DONE.txt" ]; then
      "$VENV/bin/python" - "$SMOKE_MODEL" "$E2B_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
out, rev = sys.argv[1:3]
snapshot_download("google/gemma-4-E2B-it", revision=rev, local_dir=out, allow_patterns=["*.json", "*.jinja", "*.safetensors"])
open(f"{out}/DOWNLOAD_DONE.txt", "w").write(f"google/gemma-4-E2B-it at {rev}\n")
PY
    fi
    if ! cmp -s "$SMOKE_MODEL/tokenizer.json" "$MODELS/gemma4_31b/tokenizer.json"; then say "E2B's tokenizer.json differs from the 31B's; stopping"; exit 1; fi
    say "smoke test: E2B's tokenizer.json is identical to the 31B's"
    if [ ! -f "$LARGE_ITEMS/prompts_${EI}_validation.jsonl" ] || [ ! -f "$LARGE_ITEMS/prompts_${PQA}_validation.jsonl" ]; then
      python src/eval/large_model_session8.py prompts --splits validation --qids "$EI" "$PQA" --records-ei "$REC9/long" --results "$RES" \
        --items-dir "$LARGE_ITEMS"
    fi
    smoke_gpu_back() {  # session 8's correction 12: each smoke process must leave the GPU as it found it (waits up to a minute)
      local used waited=0
      used=$(gpu_used_mib)
      while [ "$used" -gt $((SMOKE_BASE + 3000)) ]; do
        if [ "$waited" -ge 60 ]; then
          say "smoke test: the GPU holds $used MiB against $SMOKE_BASE MiB before; a smoke process left its engine; stopping"; nvidia-smi; exit 1
        fi
        sleep 5
        waited=$((waited + 5))
        used=$(gpu_used_mib)
      done
      say "smoke test: GPU used $used MiB (before the smoke test: $SMOKE_BASE MiB)"
    }
    smoke_run() {  # smoke_run <mode> [options]: one smoke process under a 20-minute timeout, then the GPU check, also on a failure;
      # session 9's paths are passed explicitly, since smoke_large_session8.py's defaults name /root/eddm_s8
      local rc=0
      timeout -k 30 1200 "$VENV/bin/python" src/eval/smoke_large_session8.py "$@" --script "$CODE/src/eval/large_model_session8.py" \
        --tokenizer-dir "$MODELS/gemma4_31b" --model-dir "$SMOKE_MODEL" --items-dir "$LARGE_ITEMS" --out-dir "$SMOKE_OUT" || rc=$?
      if [ "$rc" -ne 0 ]; then
        say "smoke test: $1 stopped with exit status $rc (124 or 137: the 20-minute timeout ended it); stopping"
        nvidia-smi || true
        smoke_gpu_back
        exit 1
      fi
      smoke_gpu_back
    }
    say "smoke test: first process (max_logprobs 1,000; logprobs 20, then 1,000)"
    smoke_run first
    say "smoke test: second process (max_num_seqs 1, max_num_batched_tokens 1,024; next-token and prompt log-probabilities in one engine)"
    smoke_run second
    PIPE_MODEL=$SMOKE_OUT/e2b_with_31b_template  # E2B's files with the 31B's chat template linked in (session 8's correction 12)
    PIPE_DIR=$SMOKE_OUT/pipeline_$(date -u +%Y%m%dT%H%M%SZ)  # a new folder for each smoke run
    mkdir -p "$PIPE_MODEL"
    for f in "$SMOKE_MODEL"/*; do ln -sfn "$f" "$PIPE_MODEL/$(basename "$f")"; done
    if [ ! -f "$MODELS/gemma4_31b/chat_template.jinja" ]; then say "the 31B has no chat_template.jinja to link; stopping"; exit 1; fi
    ln -sfn "$MODELS/gemma4_31b/chat_template.jinja" "$PIPE_MODEL/chat_template.jinja"
    say "smoke test: the script's own answer() on the six items, then its second() and finish() in a new process, in $PIPE_DIR"
    smoke_run pipeline_answer --baseline-mib "$SMOKE_BASE" --pipeline-model-dir "$PIPE_MODEL" --pipeline-dir "$PIPE_DIR"
    smoke_run pipeline_second --baseline-mib "$SMOKE_BASE" --pipeline-model-dir "$PIPE_MODEL" --pipeline-dir "$PIPE_DIR"
    say "SMOKE TEST PASSED"
    ;;
  large_validation)
    long_only
    if rows_ok "$RES/large_model/gemma4_31b_long-raw_${EI}_validation.csv" "$N_VAL"; then
      say "gemma4_31b_long's validation predictions exist; not answered again"
    else
      if ! rows_ok "$RES/large_model/gemma4_31b-raw_${EI}_validation.csv" "$N_VAL"; then
        gpu_alone
        python src/eval/large_model_session8.py prompts --splits validation --qids "$EI" "$PQA" --records-ei "$REC9/long" --results "$RES" \
          --items-dir "$LARGE_ITEMS"
        "$VENV/bin/python" src/eval/large_model_session8.py answer --splits validation --qids "$EI" --model-dir "$MODELS/gemma4_31b" \
          --items-dir "$LARGE_ITEMS" --results "$RES"
        # always (session 8's correction 9): with no item pending, second loads no model and only writes the results again, so
        # a file left by an earlier run cannot stand in for this run's
        gpu_alone
        "$VENV/bin/python" src/eval/large_model_session8.py second --splits validation --qids "$EI" --model-dir "$MODELS/gemma4_31b" \
          --items-dir "$LARGE_ITEMS" --results "$RES"
      fi
      check_rows "$RES/large_model/gemma4_31b-raw_${EI}_validation.csv" "$N_VAL"
      copy_long validation
    fi
    python - "$RES/large_model/gemma4_31b_long_letters.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v = d["evidence_effect"]["validation"]
print("evidence_effect validation (6,144-token inputs): letters' summed probability", json.dumps(v["letters_summed_probability"]),
      "| answered the second way:", v["answered_second_way"])
if v["letters_summed_probability"]["median"] < 0.5:
    print("THE MEDIAN IS BELOW 0.5 ON VALIDATION: a correction is appended before any test prediction (session 8's THE LARGE MODEL)")
PY
    ;;
  biogpt)
    gpu_alone
    not_built_on_failure biogpt "sacremoses's install" pip install -q "sacremoses==0.2.0"
    if [ ! -f "$MODELS/biogpt_large_pqa/DOWNLOAD_DONE.txt" ]; then
      not_built_on_failure biogpt "the download of $BIOGPT_REPO at $BIOGPT_REV" python - "$BIOGPT_REPO" "$BIOGPT_REV" "$MODELS/biogpt_large_pqa" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=out, allow_patterns=["*.json", "*.txt", "*.bin"])
open(f"{out}/DOWNLOAD_DONE.txt", "w").write(f"{repo} at {rev}\n")
print(repo, "downloaded at", rev)
PY
    fi
    not_built_on_failure biogpt "biogpt_session9.py validation" python src/eval/biogpt_session9.py validation \
      --model-dir "$MODELS/biogpt_large_pqa" --results "$RES"
    if ! rows_ok "$RES/predictions/biogpt_large_pqa-raw_${PQA}_validation.csv" "$N_PQA_VAL"; then
      write_not_built biogpt "NOT BUILT: biogpt, its validation predictions do not have $N_PQA_VAL rows"
      exit 0
    fi
    ;;
  calibrate)
    waiting=""
    for s in train trees; do if [ ! -f "$DONE/$s.txt" ]; then waiting="$waiting $s"; fi; done
    for s in tabicl biogpt; do if ! finished_or_not_built "$s"; then waiting="$waiting $s"; fi; done
    if [ "$(v05_max_len)" = 6144 ] && [ ! -f "$DONE/large_validation.txt" ]; then waiting="$waiting large_validation"; fi
    if [ -n "$waiting" ]; then say "calibrate waits for$waiting to finish (or write a NOT BUILT marker); stopping"; exit 1; fi
    gpu_alone
    python src/eval/calibrate_session9.py fit --results "$RES"
    python src/eval/calibrate_session9.py ece --results "$RES"
    python src/eval/trees_session9.py pool --device cuda --not-built-dir "$NOT_BUILT" --results "$RES" --models-dir "$MODELS/trees"
    V05=$(json_get "$RES/recipe_selection.json" "d['v05_ei']['system']")
    SA=$(json_get "$RES/recipe_selection.json" "d['v05_ei']['swap_averaging']")
    runs_text=$(json_get "$RES/recipe_selection.json" "' '.join(d['v05_ei']['runs'])")
    read -r -a RUNS <<< "$runs_text"
    if [ "$(v05_max_len)" = 6144 ]; then
      LARGE=gemma4_31b_long
      LARGE_FILE=$RES/predictions/gemma4_31b_long_${EI}_validation.csv
    else
      LARGE=gemma4_31b
      LARGE_FILE=$RES8/predictions/gemma4_31b_${EI}_validation.csv
    fi
    V05_FILE=$RES/predictions/${V05}_${EI}_validation.csv
    run_files=()
    for r in "${RUNS[@]}"; do
      if [ "$SA" = True ] && [ -f "$RES/predictions/${r}_sa-raw_${EI}_validation.csv" ]; then
        run_files+=("$RES/predictions/${r}_sa-raw_${EI}_validation.csv")
        say "I-35's run spread: ${r}_sa-raw (the run swap-averaged)"
      else
        run_files+=("$RES/predictions/${r}-raw_${EI}_validation.csv")
        if [ "$SA" = True ]; then say "I-35's run spread: ${r}-raw (single pass; no swap-averaged file of this run exists)"
        else say "I-35's run spread: ${r}-raw"; fi
      fi
    done
    say "the hand-off analyses on validation: $V05 with $LARGE ($LARGE_FILE)"
    python src/eval/handoff_curves_session9.py curves --split validation \
      --pair "$EI" "$V05" "$V05_FILE" "$LARGE" "$LARGE_FILE" "$REC8/${EI}_validation.jsonl" --out "$RES/handoff_validation.json"
    python src/eval/handoff_curves_session9.py fit-rule --fiorillo-name "$V05" --fiorillo "$V05_FILE" --runs "${run_files[@]}" \
      --large-name "$LARGE" --large "$LARGE_FILE" --records "$REC8/${EI}_validation.jsonl" --out "$RES/handoff_rule_v05.json"
    python src/eval/power_session9.py validation --records "$REC8/${EI}_validation.jsonl" --system "$V05_FILE" --reference "$LARGE_FILE" \
      --out "$RES/power_validation.json"
    ;;
  freeze)
    require_osf
    python src/eval/calibrate_session9.py freeze --results "$RES" --base "$BASE" --staging "$STAGE_DIR"
    ;;
  testpreds)
    if [ ! -f "$RES/release_candidate.json" ]; then say "Stop: results/fiorillo_v0_5/release_candidate.json does not exist (the freeze comes first)"; exit 1; fi
    python - "$RES" <<'PY'
import datetime, hashlib, json, sys, time
from pathlib import Path
res = Path(sys.argv[1])
fmt = "%Y-%m-%dT%H:%M:%SZ"
rc_bytes = (res / "release_candidate.json").read_bytes()
digest = hashlib.sha256(rc_bytes).hexdigest()
frozen = datetime.datetime.strptime(json.loads(rc_bytes)["time_utc"], fmt).replace(tzinfo=datetime.timezone.utc)
while datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) <= frozen:  # the start is later than the freeze (step 14)
    time.sleep(0.2)
now = datetime.datetime.now(datetime.timezone.utc).strftime(fmt)
path = res / "testpreds_log.json"
if path.exists():  # a retry after a failure: the first entry stays; the frozen candidate must be the same file
    log = json.load(open(path))
    if log["release_candidate_sha256"] != digest:
        sys.exit("release_candidate.json changed after the test stage first started; stop")
    log.setdefault("retries_utc", []).append(now)
else:
    log = {"release_candidate_sha256": digest, "testpreds_start_utc": now}
tmp = path.with_name(path.name + ".tmp")
tmp.write_text(json.dumps(log, indent=2) + "\n")
tmp.replace(path)
print(json.dumps(log))
PY
    mkdir -p "$RES/checks_v05/logits"
    gpu_alone
    WINNER=$(json_get "$RES/recipe_selection.json" "d['v05_ei']['recipe']")
    TEST_DIR=$(json_get "$RES/recipe_selection.json" "d['v05_ei']['test_records_dir']")
    SWAP_DIR=$(json_get "$RES/recipe_selection.json" "d['v05_ei']['checks_swap_dir']")
    runs_text=$(json_get "$RES/recipe_selection.json" "' '.join(d['v05_ei']['runs'])")
    singles_text=$(json_get "$RES/recipe_selection.json" "' '.join(d['v05_ei']['single_pass_systems'])")
    read -r -a RUNS <<< "$runs_text"
    read -r -a SINGLES <<< "$singles_text"
    if [ "${#RUNS[@]}" -lt 2 ] || [ "${#RUNS[@]}" -gt 3 ] || [ "${#SINGLES[@]}" = 0 ]; then  # two runs: correction 8
      say "recipe_selection.json's v05_ei names ${#RUNS[@]} runs and ${#SINGLES[@]} single-pass systems; stopping"; exit 1
    fi
    jobs=("s9_b_1|$REC8" "s9_b_2|$REC8" "s9_b_3|$REC8")  # base's runs at 4,096 tokens (session 8's test records)
    if [ "$WINNER" != b ]; then for r in "${RUNS[@]}"; do jobs+=("$r|$TEST_DIR"); done; fi
    if [ -f "$MODELS/s9_ei_wavg/adapter.safetensors" ]; then jobs+=("s9_ei_wavg|$TEST_DIR"); fi
    for job in "${jobs[@]}"; do
      system="${job%%|*}"
      records="${job#*|}"
      if rows_ok "$RES/predictions/${system}-raw_${EI}_test.csv" "$N_TEST" && rows_ok "$RES/logits/${system}_${EI}_test_logits.csv" "$N_TEST"; then
        say "$system's test predictions exist"; continue
      fi
      predict_split "$system" "$MODELS/$system" "$records" test "$STAGE_DIR/test/predictions" "$STAGE_DIR/test/logits" \
        "$RES/run_summaries/${system}_test_run_summary.json" "$N_TEST"
      cp "$STAGE_DIR/test/predictions/${system}-raw_${EI}_test.csv" "$RES/predictions/"
      cp "$STAGE_DIR/test/logits/${system}_${EI}_test_logits.csv" "$RES/logits/"
    done
    # THE READING CHECKS on v0.5's EI specialist: each single-pass system on the none, title and swap inputs
    for single in "${SINGLES[@]}"; do
      for input in none title swap; do
        if [ "$input" = swap ]; then records="$SWAP_DIR"; else records="$REC9/checks/$input"; fi
        system="${single}_${input}"
        rows=$(count_records "$records/${EI}_test.jsonl")
        if rows_ok "$RES/checks_v05/${system}-raw_${EI}_test.csv" "$rows" && rows_ok "$RES/checks_v05/logits/${system}_${EI}_test_logits.csv" "$rows"; then
          say "$system: its reading-check predictions exist"; continue
        fi
        predict_split "$system" "$MODELS/$single" "$records" test "$RES/checks_v05" "$RES/checks_v05/logits" \
          "$RES/checks_v05/${system}_run_summary.json" "$rows"
      done
    done
    if [ "$(v05_max_len)" = 6144 ]; then  # GEMMA ON THE LONGER INPUT: gemma4_31b_long on test, alone on the GPU
      if rows_ok "$RES/large_model/gemma4_31b_long-raw_${EI}_test.csv" "$N_TEST"; then
        say "gemma4_31b_long's test predictions exist"
      else
        if ! rows_ok "$RES/large_model/gemma4_31b-raw_${EI}_test.csv" "$N_TEST"; then
          gpu_alone
          python src/eval/large_model_session8.py prompts --splits test --qids "$EI" --records-ei "$REC9/long" --results "$RES" \
            --items-dir "$LARGE_ITEMS"
          "$VENV/bin/python" src/eval/large_model_session8.py answer --splits test --qids "$EI" --model-dir "$MODELS/gemma4_31b" \
            --items-dir "$LARGE_ITEMS" --results "$RES"
          gpu_alone  # second always runs (session 8's correction 9)
          "$VENV/bin/python" src/eval/large_model_session8.py second --splits test --qids "$EI" --model-dir "$MODELS/gemma4_31b" \
            --items-dir "$LARGE_ITEMS" --results "$RES"
        fi
        check_rows "$RES/large_model/gemma4_31b-raw_${EI}_test.csv" "$N_TEST"
        copy_long test
      fi
    fi
    if [ -f "$NOT_BUILT/biogpt.txt" ]; then
      say "biogpt_large_pqa: NOT BUILT ($(cat "$NOT_BUILT/biogpt.txt")); no test predictions"
    elif rows_ok "$RES/predictions/biogpt_large_pqa-raw_${PQA}_test.csv" "$N_PQA_TEST"; then
      say "biogpt_large_pqa's test predictions exist"
    else
      gpu_alone
      python src/eval/biogpt_session9.py test --model-dir "$MODELS/biogpt_large_pqa" --results "$RES"
      check_rows "$RES/predictions/biogpt_large_pqa-raw_${PQA}_test.csv" "$N_PQA_TEST"
    fi
    if [ -f "$NOT_BUILT/tabicl.txt" ]; then
      say "tabicl: NOT BUILT ($(cat "$NOT_BUILT/tabicl.txt")); no test predictions"
    elif rows_ok "$RES/predictions/tabicl_${RW}_test.csv" "$N_RW_TEST"; then
      say "tabicl's test predictions exist"
    else
      gpu_alone
      "$TABICL_ENV/bin/python" src/eval/trees_session9.py tabicl --split test --device cuda --results "$RES" --models-dir "$MODELS/trees"
      check_rows "$RES/predictions/tabicl_${RW}_test.csv" "$N_RW_TEST"
    fi
    members=(xgboost_codes catboost_codes)
    lgb_passed=$(json_get "$RES/trees_v2.json" "bool(d['members']['lightgbm_tuned']['check']['passed'])")
    if [ "$lgb_passed" = True ]; then
      members+=(lightgbm_tuned)
    else
      say "lightgbm_tuned's refit failed its check (trees_v2.json): it is left out of the new pool and writes no test predictions"
    fi
    trees_missing=0
    for m in "${members[@]}"; do
      if ! rows_ok "$RES/predictions/${m}_${RW}_test.csv" "$N_RW_TEST"; then trees_missing=1; fi
    done
    if [ "$trees_missing" = 0 ]; then
      say "the tree members' test predictions exist"
    else
      python src/eval/trees_session9.py refit --split test --threads "$TREE_THREADS" --results "$RES" --models-dir "$MODELS/trees"
      python src/eval/trees_session9.py lightgbm --split test --results "$RES" --models-dir "$MODELS/trees"
      for m in "${members[@]}"; do check_rows "$RES/predictions/${m}_${RW}_test.csv" "$N_RW_TEST"; done
    fi
    say "test predictions written: ${jobs[*]%%|*}; the reading checks of ${SINGLES[*]}; the large model if it applies;" \
        "biogpt_large_pqa and tabicl unless NOT BUILT; ${members[*]}"
    ;;
  package)
    python - "$CODE" "$MODELS" "$LOGS" "$OUTS" "$BASE" <<'PY'
import datetime, hashlib, json, sys, zipfile
from pathlib import Path
code, models, logs, outs, base = (Path(a) for a in sys.argv[1:6])
RES_REL = "results/fiorillo_v0_5"
res = code / RES_REL
def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
markers = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted((base / "not_built").glob("*.txt"))}
(res / "not_built.json").write_text(json.dumps({"plan": "docs/prompt_session9.md, STEP 3: the pod stages' NOT BUILT markers",
                                                 "written_utc": now(), "stages": markers}, indent=1) + "\n", encoding="utf-8")
print("not_built.json:", json.dumps(markers))
def add_tree(z, folder, arc_root):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and "checkpoint_best" not in p.parts:
            z.write(p, str(Path(arc_root) / p.relative_to(folder)))
outs.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(outs / "fiorillo_v0_5_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, res, RES_REL)
runs = sorted(p.name for p in models.glob("s9_*") if p.is_dir())
if runs:
    with zipfile.ZipFile(outs / "fiorillo_v0_5_ei.zip", "w", zipfile.ZIP_STORED) as z:
        for s in runs:
            add_tree(z, models / s, s)
print("ei zip:", runs or "no s9_ folder, not written")
if (models / "trees").is_dir():
    with zipfile.ZipFile(outs / "fiorillo_v0_5_trees.zip", "w", zipfile.ZIP_STORED) as z:
        add_tree(z, models / "trees", "trees")
print("trees zip:", "written" if (models / "trees").is_dir() else "no trees folder, not written")
with zipfile.ZipFile(outs / "fiorillo_v0_5_logs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, logs, "logs/pod")
lines = [f"{sha(p)}  {p.stat().st_size}  {p.name}" for p in sorted(outs.glob("fiorillo_v0_5_*.zip"))]
(outs / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
    python - "$CODE" "$MODELS" "$OUTS" "$BASE" <<'PY'
import datetime, hashlib, json, sys, zipfile
from pathlib import Path
code, models, outs, base = (Path(a) for a in sys.argv[1:5])
RES_REL = "results/fiorillo_v0_5"
EI, PQA, RW = "evidence_effect", "pubmedqa_answer", "readmit_window"
ROWS = {(EI, "validation"): 1237, (EI, "test"): 1218, (PQA, "validation"): 150, (PQA, "test"): 500,
        (RW, "validation"): 6997, (RW, "test"): 6998}
problems = []
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
# the zips themselves: sizes and SHA-256 against SHA256SUMS.txt, and every member's CRC
zips = {}
for line in (outs / "SHA256SUMS.txt").read_text().splitlines():
    digest, size, name = line.split()
    p = outs / name
    if p.stat().st_size != int(size) or sha_file(p) != digest:
        problems.append(f"{name}: its size or SHA-256 differs from SHA256SUMS.txt")
    zips[name] = zipfile.ZipFile(p)
    bad = zips[name].testzip()
    if bad:
        problems.append(f"{name}: {bad} fails its CRC check")
print(f"zips checked: {sorted(zips)}")
if not (base / "stage_done" / "testpreds.txt").exists():
    print("PARTIAL PACKAGE: testpreds has not finished, so steps 15 to 17 cannot run on these zips; only the zips were checked")
    if problems:
        print("PACKAGE CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    sys.exit(0)
zr = zips.get("fiorillo_v0_5_results.zip")
names = set(zr.namelist()) if zr else set()
if zr is None:
    problems.append("fiorillo_v0_5_results.zip was not written")
def member(rel):
    return f"{RES_REL}/{rel}"
def need(rel, rows=None):
    m = member(rel)
    if m not in names:
        problems.append(f"missing from the results zip: {m}")
        return
    if rows is not None:
        with zr.open(m) as f:
            got = sum(1 for _ in f) - 1
        if got != rows:
            problems.append(f"{m}: {got} rows, expected {rows}")
def records(rel_dir, split):
    with open(code / rel_dir / f"{EI}_{split}.jsonl", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
def need_zip(zip_name, arc):
    z = zips.get(zip_name)
    if z is None or arc not in z.namelist():
        problems.append(f"missing from {zip_name}: {arc}")
for rel in ("release_candidate.json", "testpreds_log.json", "recipe_selection.json", "calibration_s9.json", "ece_limit_evidence_effect.json",
            "readmission_pool_v2.json", "trees_v2.json", "trees_v2_progress.json", "handoff_validation.json", "handoff_rule_v05.json",
            "power_validation.json", "checks_s8.json", "cost_rule.json", "pilot_long.json", "not_built.json", "records_manifest_pod.json",
            "records_manifest_s8_pod.json", "records_manifest_s4_pod.json"):
    need(rel)
markers = {p.stem for p in (base / "not_built").glob("*.txt")}
if all(member(r) in names for r in ("release_candidate.json", "testpreds_log.json", "recipe_selection.json", "trees_v2.json")):
    rc_bytes = zr.read(member("release_candidate.json"))
    rc, log = json.loads(rc_bytes), json.loads(zr.read(member("testpreds_log.json")))
    sel, trees = json.loads(zr.read(member("recipe_selection.json"))), json.loads(zr.read(member("trees_v2.json")))
    # step 14's guard: testpreds_log.json names release_candidate.json's SHA-256 and a start later than the freeze
    if log.get("release_candidate_sha256") != hashlib.sha256(rc_bytes).hexdigest():
        problems.append("testpreds_log.json does not name release_candidate.json's SHA-256 (step 14 would stop)")
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        if not datetime.datetime.strptime(log["testpreds_start_utc"], fmt) > datetime.datetime.strptime(rc["time_utc"], fmt):
            problems.append(f"testpreds_log.json's start {log['testpreds_start_utc']} is not later than the freeze's {rc['time_utc']} (step 14 would stop)")
    except (KeyError, ValueError) as error:
        problems.append(f"the times of testpreds_log.json and release_candidate.json cannot be compared ({error!r})")
    # the digests the freeze recorded: results/fiorillo_v0_5's files as the zip holds them, the others as the pod holds them
    for key, digest in (rc.get("sha256") or {}).items():
        if key.startswith(RES_REL + "/"):
            got = hashlib.sha256(zr.read(key)).hexdigest() if key in names else None
        else:
            got = sha_file(code / key) if (code / key).exists() else None
        if got != digest:
            problems.append(f"{key}: SHA-256 {got}, the freeze recorded {digest}")
    v = sel.get("v05_ei") or {}
    # the EI runs and the averages (validation), the exchanged validation records and the swap-averaged system
    runs = sorted(p.name for p in models.glob("s9_*") if p.is_dir() and p.name != "s9_ei_wavg")
    wavg = (models / "s9_ei_wavg").is_dir()
    for r in runs + (["s9_ei_wavg"] if wavg else []):
        need(f"predictions/{r}-raw_{EI}_validation.csv", ROWS[(EI, "validation")])
        need(f"logits/{r}_{EI}_validation_logits.csv", ROWS[(EI, "validation")])
        need(f"run_summaries/{r}_validation_run_summary.json")
        for f in ("adapter.safetensors", "config.json"):
            need_zip("fiorillo_v0_5_ei.zip", f"{r}/{f}")
    for r in runs:
        need(f"metrics/{r}_metrics.json")
        need(f"logs/{r}_train_log.csv")
    if wavg:
        need("weight_average.json")
    for name in ("s9_b", "s9_ei", f"{v.get('base_system')}_sa"):
        need(f"predictions/{name}-raw_{EI}_validation.csv", ROWS[(EI, "validation")])
    need(f"predictions/{v.get('system')}_{EI}_validation.csv", ROWS[(EI, "validation")])
    n_exch = records(v["checks_swap_dir"], "validation")
    for s in v.get("single_pass_systems", []):
        need(f"predictions/{s}_exch-raw_{EI}_validation.csv", n_exch)
        need(f"logits/{s}_exch_{EI}_validation_logits.csv", n_exch)
    # the EI test predictions (testpreds) and the reading checks of both systems
    tests = ["s9_b_1", "s9_b_2", "s9_b_3"] + (list(v.get("runs", [])) if v.get("recipe") != "b" else []) + (["s9_ei_wavg"] if wavg else [])
    for s in tests:
        need(f"predictions/{s}-raw_{EI}_test.csv", ROWS[(EI, "test")])
        need(f"logits/{s}_{EI}_test_logits.csv", ROWS[(EI, "test")])
        need(f"run_summaries/{s}_test_run_summary.json")
    checks9 = "data/processed/session9/checks"
    for k in (1, 2, 3):
        for inp in ("none", "title", "swap"):
            n = records(f"{checks9}/{inp}", "test")
            need(f"checks_s8/s8_ei_{k}_{inp}-raw_{EI}_test.csv", n)
            need(f"checks_s8/logits/s8_ei_{k}_{inp}_{EI}_test_logits.csv", n)
            need(f"checks_s8/s8_ei_{k}_{inp}_run_summary.json")
    for s in v.get("single_pass_systems", []):
        for inp in ("none", "title", "swap"):
            n = records(v["checks_swap_dir"] if inp == "swap" else f"{checks9}/{inp}", "test")
            need(f"checks_v05/{s}_{inp}-raw_{EI}_test.csv", n)
            need(f"checks_v05/logits/{s}_{inp}_{EI}_test_logits.csv", n)
            need(f"checks_v05/{s}_{inp}_run_summary.json")
    # the large model under GEMMA ON THE LONGER INPUT
    if v.get("max_len") == 6144:
        for split in ("validation", "test"):
            need(f"large_model/gemma4_31b_long-raw_{EI}_{split}.csv", ROWS[(EI, split)])
            need(f"large_model/gemma4_31b_long_logprobs_{EI}_{split}.csv", ROWS[(EI, split)])
        need("large_model/gemma4_31b_long_letters.json")
        need("large_model/gemma4_31b_long_timing.json")
        need(f"predictions/gemma4_31b_long_{EI}_validation.csv", ROWS[(EI, "validation")])
    # BioGPT-Large, TabICL and the tree members
    if "biogpt" not in markers:
        need("biogpt.json")
        for split in ("validation", "test"):
            need(f"predictions/biogpt_large_pqa-raw_{PQA}_{split}.csv", ROWS[(PQA, split)])
    if "tabicl" not in markers:
        need("tabicl.json")
        for split in ("validation", "test"):
            need(f"predictions/tabicl_{RW}_{split}.csv", ROWS[(RW, split)])
        for f in ("tabicl.pkl", "tabicl_state.json"):
            need_zip("fiorillo_v0_5_trees.zip", f"trees/{f}")
    lgb_passed = bool(((trees.get("members") or {}).get("lightgbm_tuned") or {}).get("check", {}).get("passed"))
    for m in ("xgboost_codes", "catboost_codes", "lightgbm_tuned"):
        need(f"predictions/{m}_{RW}_validation.csv", ROWS[(RW, "validation")])
        if m != "lightgbm_tuned" or lgb_passed:
            need(f"predictions/{m}_{RW}_test.csv", ROWS[(RW, "test")])
    need(f"predictions/readmission_pool_v2_{RW}_validation.csv", ROWS[(RW, "validation")])
    for f in ("recode_tables.json", "xgboost_codes.json", "catboost_codes.cbm", "lightgbm_tuned.txt"):
        need_zip("fiorillo_v0_5_trees.zip", f"trees/{f}")
if problems:
    print("PACKAGE CHECK FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print(f"PACKAGE CHECK PASSED: every file steps 15 to 17 read is in the zips with its rows; release_candidate.json's digests match; "
      f"testpreds_log.json names its SHA-256 and a later start ({len(names)} files in the results zip)")
PY
    ;;
  all)
    if [ ! -f "$DONE/prepare.txt" ]; then say "all: prepare has not finished (no $DONE/prepare.txt); run prepare first; all stops"; exit 1; fi
    if [ ! -f "$BASE/osf_filed.txt" ]; then
      say "all: STOPPED: $BASE/osf_filed.txt is missing (the OSF update's link and the time it was filed, written by Claude once" \
          "docs/log.md records the filing); no stage was started"
      exit "$STOP_OSF"
    fi
    say "all: the chain starts (checks_s8, pilot_long, train with trees beside it, tabicl, the large stages if they apply, biogpt," \
        "calibrate, freeze, testpreds, package)"
    TREES_PID=""
    run_stage checks_s8
    run_stage pilot_long
    start_trees
    run_stage train
    wait_trees
    run_stage tabicl
    if [ "$(v05_max_len)" = 6144 ]; then
      say "all: v0.5's EI recipe reads 6,144 tokens, so GEMMA ON THE LONGER INPUT applies"
      run_stage prepare_large
      run_stage smoke_large
      run_stage large_validation
    else
      say "all: v0.5's EI recipe reads 4,096 tokens, so GEMMA ON THE LONGER INPUT does not apply (gemma4_31b's session 8 files are used)"
    fi
    run_stage biogpt
    run_stage calibrate
    run_stage freeze
    run_stage testpreds
    run_stage package
    say "all: the chain finished; the zips and SHA256SUMS.txt are in $OUTS"
    ;;
  *)
    echo "unknown stage $STAGE"; exit 1 ;;
esac
if [ "$STAGE" != all ]; then mark_done "$STAGE"; fi
say "stage $STAGE finished"

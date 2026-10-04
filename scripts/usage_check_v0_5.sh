#!/bin/bash
# Session 10's usage check of the released v0.5 (docs/prompt_session10.md, THE USAGE CHECK; I-100).
# Johann runs it on the Mac, from the project folder:
#     caffeinate -i bash scripts/usage_check_v0_5.sh
# (1) It checks the release folder against its SHA256SUMS.txt (shasum -a 256 -c), after checking SHA256SUMS.txt's own
#     SHA-256 against the plan's FIXED INPUTS, and the three record files against ids.json, and stops on any mismatch.
# (2) It runs predict_v0_5.py, unchanged, once for each question, readmission first, then PubMedQA, then Evidence
#     Inference, each in its own process, with xgboost 3.2.0 and catboost 1.2.10 added for the run (the pod's versions;
#     every other package comes from uv.lock); each run's output and errors go to results/fiorillo_v0_5/usage_check/
#     run_<qid>.log with its start and end times (UTC) and its exit status.
# (3) When a question's run ends with an out-of-memory error on the Mac's GPU, it runs that question once more with
#     --device cpu and says so in its log.
# (4) It runs compare, which writes results/fiorillo_v0_5/usage_check/usage_check.json.
# The first run that reads the trials or the abstracts downloads Qwen3-4B-Base (about 8 GB) into the Hugging Face cache;
# no token is needed. Off a GPU the release runs the readers in float32, about 16 GB for the base model's weights.
set -u
cd "$(dirname "$0")/.." || exit 1

RELEASE=models/fiorillo_v0_5/release
SAMPLE=data/processed/session10
OUT=results/fiorillo_v0_5/usage_check
SUMS_SHA256=a04b52f0f3a83165201b7157c8d518f04ba93cc3cdc4f8a6040eb4f12ce80344
WITH=(--with xgboost==3.2.0 --with catboost==1.2.10)
QUESTIONS=(readmit_window pubmedqa_answer evidence_effect)

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "STOP: $*" >&2; exit 1; }

[ -f pyproject.toml ] && [ -d "$RELEASE" ] || stop "run this from the project folder (no $RELEASE here)"
command -v uv >/dev/null 2>&1 || stop "uv is not on PATH"
command -v shasum >/dev/null 2>&1 || stop "shasum is not on PATH"

# (1) The release folder and the record files, before anything runs.
echo "== (1) the release folder against SHA256SUMS.txt"
sums=$(shasum -a 256 "$RELEASE/SHA256SUMS.txt" | awk '{print $1}')
[ "$sums" = "$SUMS_SHA256" ] || stop "SHA256SUMS.txt has SHA-256 $sums, not $SUMS_SHA256 (FIXED INPUTS)"
(cd "$RELEASE" && shasum -a 256 -c SHA256SUMS.txt) || stop "the release folder does not match SHA256SUMS.txt"
echo "== (1) the record files against ids.json"
[ -f "$OUT/ids.json" ] || stop "$OUT/ids.json is missing (the sample is drawn on the linked shell in step 1)"
uv run python -c "import sys; sys.path.insert(0, 'src/eval'); import usage_check_v0_5 as U; sys.exit(U.check_records())" \
  || stop "the record files do not match ids.json"

# What --device auto resolves to on this machine (written into each run's log).
AUTO=$(uv run python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null)
[ -n "$AUTO" ] || AUTO=unknown
echo "   --device auto resolves to: $AUTO"
df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | tail -1 | awk '{print "   free space where the base model is cached: " $4}'

run_question() {
  local qid=$1 device=$2
  local log="$OUT/run_${qid}.log" records="$SAMPLE/usage_check_${qid}_records.jsonl"
  local n resolved status
  n=$(grep -c . "$records")
  resolved=$device
  [ "$device" = auto ] && resolved=$AUTO
  {
    echo "usage_check question: $qid"
    echo "usage_check records: $records ($n records)"
    echo "usage_check device_requested: $device"
    echo "usage_check device_resolved: $resolved"
    echo "usage_check start_utc: $(stamp)"
  } >> "$log"
  uv run "${WITH[@]}" python "$RELEASE/predict_v0_5.py" --records "$records" --release "$RELEASE" --device "$device" \
    --out "$OUT/answers_${qid}.jsonl" >> "$log" 2>&1
  status=$?
  { echo "usage_check end_utc: $(stamp)"; echo "usage_check exit_status: $status"; } >> "$log"
  return $status
}

out_of_memory() {
  # the text after the log's last start line holds an out-of-memory error
  awk '/^usage_check start_utc:/{buf=""} {buf = buf "\n" $0} END{print buf}' "$1" \
    | grep -E -i -q 'out of memory|OutOfMemoryError|MPS backend out of memory'
}

# (2) and (3) One run for each question.
for qid in "${QUESTIONS[@]}"; do
  log="$OUT/run_${qid}.log"
  : > "$log"
  : > "$OUT/answers_${qid}.jsonl"
  echo "== (2) $qid: predict_v0_5.py (log: $log)"
  if run_question "$qid" auto; then
    echo "   finished"
  elif out_of_memory "$log"; then
    echo "usage_check out_of_memory: the run above ended with an out-of-memory error on the GPU; it runs once more with --device cpu" >> "$log"
    echo "== (3) $qid: out of memory on the GPU; running it once more with --device cpu"
    if run_question "$qid" cpu; then echo "   finished on the CPU"; else echo "   the CPU run failed too (see $log)"; fi
  else
    echo "   failed (see $log); compare counts its records as failures"
  fi
done

# (4) The comparison.
echo "== (4) compare"
uv run "${WITH[@]}" python src/eval/usage_check_v0_5.py compare
status=$?
echo "== done: $OUT/usage_check.json (compare's exit status $status; 0 is PASS, 1 is FAIL). Tell Claude it has finished."
exit $status

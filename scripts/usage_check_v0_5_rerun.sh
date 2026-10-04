#!/bin/bash
# Session 10's second run of the usage check of the released v0.5 (docs/prompt_session10.md, corrections 5 and 6;
# I-100). Johann runs it on the Mac, from the project folder:
#     caffeinate -i bash scripts/usage_check_v0_5_rerun.sh
# The first run (scripts/usage_check_v0_5.sh) gave Evidence Inference session 8's 4,096-token inputs, while v0.5 was
# evaluated on session 9's 6,144-token inputs, and its readmission run ended with a segmentation fault. This run:
# (1) checks the release folder against its SHA256SUMS.txt (after checking SHA256SUMS.txt's own SHA-256 against the
#     plan's FIXED INPUTS) and the second run's record files against results/fiorillo_v0_5/usage_check/rerun/ids.json,
#     and stops on any mismatch;
# (2) runs readmission with predict_v0_5.py, unchanged, and Python's fault handler on (PYTHONFAULTHANDLER=1), so that a
#     segmentation fault prints where it happened; if that run fails, it runs readmission once more with xgboost and
#     catboost imported before predict_v0_5.py loads torch (predict_v0_5.py still unchanged), as a diagnosis;
# (3) runs Evidence Inference with predict_v0_5.py, unchanged, on the first run's 60 prompts with the inputs v0.5 was
#     evaluated on; on an out-of-memory error on the Mac's GPU, once more with --device cpu;
# (4) runs compare --rerun, which writes results/fiorillo_v0_5/usage_check/rerun/usage_check.json.
# PubMedQA is not run again: its first run passed, on the inputs its evaluated predictions read. The first run's files
# are not touched. xgboost 3.2.0 and catboost 1.2.10 are added for each run, as in the first run.
set -u
cd "$(dirname "$0")/.." || exit 1

RELEASE=models/fiorillo_v0_5/release
OUT=results/fiorillo_v0_5/usage_check/rerun
SUMS_SHA256=a04b52f0f3a83165201b7157c8d518f04ba93cc3cdc4f8a6040eb4f12ce80344
WITH=(--with xgboost==3.2.0 --with catboost==1.2.10)
RW_RECORDS=data/processed/session10/usage_check_readmit_window_records.jsonl
EI_RECORDS=data/processed/session10/usage_check_evidence_effect_long_records.jsonl

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
echo "== (1) the record files against $OUT/ids.json"
[ -f "$OUT/ids.json" ] || stop "$OUT/ids.json is missing (the second run's sample is drawn on the linked shell)"
uv run python -c "import sys; sys.path.insert(0, 'src/eval'); import usage_check_v0_5 as U; sys.exit(U.check_records(rerun=True))" \
  || stop "the record files do not match $OUT/ids.json"

# What --device auto resolves to on this machine (written into each run's log).
AUTO=$(uv run python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null)
[ -n "$AUTO" ] || AUTO=unknown
echo "   --device auto resolves to: $AUTO"

# One run of predict_v0_5.py, with the fault handler on: name (its log's and answers' name), qid, records, device, mode.
# Mode "plain" runs predict_v0_5.py as a user runs it; mode "trees_first" imports xgboost and catboost first.
run_one() {
  local name=$1 qid=$2 records=$3 device=$4 mode=$5
  local log="$OUT/run_${name}.log" out="$OUT/answers_${name}.jsonl"
  local n resolved status
  n=$(grep -c . "$records")
  resolved=$device
  [ "$device" = auto ] && resolved=$AUTO
  {
    echo "usage_check question: $qid"
    echo "usage_check records: $records ($n records)"
    echo "usage_check mode: $mode"
    echo "usage_check device_requested: $device"
    echo "usage_check device_resolved: $resolved"
    echo "usage_check start_utc: $(stamp)"
  } >> "$log"
  if [ "$mode" = trees_first ]; then
    PYTHONFAULTHANDLER=1 uv run "${WITH[@]}" python -c \
      "import sys, runpy; import xgboost, catboost; sys.argv = ['$RELEASE/predict_v0_5.py'] + sys.argv[1:]; runpy.run_path('$RELEASE/predict_v0_5.py', run_name='__main__')" \
      --records "$records" --release "$RELEASE" --device "$device" --out "$out" >> "$log" 2>&1
  else
    PYTHONFAULTHANDLER=1 uv run "${WITH[@]}" python "$RELEASE/predict_v0_5.py" --records "$records" --release "$RELEASE" \
      --device "$device" --out "$out" >> "$log" 2>&1
  fi
  status=$?
  { echo "usage_check end_utc: $(stamp)"; echo "usage_check exit_status: $status"; } >> "$log"
  return $status
}

out_of_memory() {
  # the text after the log's last start line holds an out-of-memory error
  awk '/^usage_check start_utc:/{buf=""} {buf = buf "\n" $0} END{print buf}' "$1" \
    | grep -E -i -q 'out of memory|OutOfMemoryError|MPS backend out of memory'
}

mkdir -p "$OUT"

# (2) Readmission, then its diagnosis if it fails.
: > "$OUT/run_readmit_window.log"
: > "$OUT/answers_readmit_window.jsonl"
echo "== (2) readmit_window: predict_v0_5.py with the fault handler on (log: $OUT/run_readmit_window.log)"
if run_one readmit_window readmit_window "$RW_RECORDS" auto plain; then
  echo "   finished"
else
  echo "   failed (see $OUT/run_readmit_window.log); running it once more with xgboost and catboost imported before torch"
  : > "$OUT/run_readmit_window_trees_first.log"
  : > "$OUT/answers_readmit_window_trees_first.jsonl"
  if run_one readmit_window_trees_first readmit_window "$RW_RECORDS" auto trees_first; then
    echo "   finished with the tree libraries imported first"
  else
    echo "   failed again (see $OUT/run_readmit_window_trees_first.log)"
  fi
fi

# (3) Evidence Inference, on the inputs v0.5 was evaluated on.
: > "$OUT/run_evidence_effect.log"
: > "$OUT/answers_evidence_effect.jsonl"
echo "== (3) evidence_effect: predict_v0_5.py on the 6,144-token inputs (log: $OUT/run_evidence_effect.log)"
if run_one evidence_effect evidence_effect "$EI_RECORDS" auto plain; then
  echo "   finished"
elif out_of_memory "$OUT/run_evidence_effect.log"; then
  echo "usage_check out_of_memory: the run above ended with an out-of-memory error on the GPU; it runs once more with --device cpu" >> "$OUT/run_evidence_effect.log"
  echo "== (3) evidence_effect: out of memory on the GPU; running it once more with --device cpu"
  if run_one evidence_effect evidence_effect "$EI_RECORDS" cpu plain; then echo "   finished on the CPU"; else echo "   the CPU run failed too (see $OUT/run_evidence_effect.log)"; fi
else
  echo "   failed (see $OUT/run_evidence_effect.log); compare counts its records as failures"
fi

# (4) The comparison.
echo "== (4) compare --rerun"
uv run "${WITH[@]}" python src/eval/usage_check_v0_5.py compare --rerun
status=$?
echo "== done: $OUT/usage_check.json (compare's exit status $status; 0 is PASS, 1 is FAIL). Tell Claude it has finished."
exit $status

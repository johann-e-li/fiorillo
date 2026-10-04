#!/usr/bin/env bash
# Session 8 pod driver (docs/prompt_session8.md, step 3, with corrections 1 to 5, 7 to 10 and 12): the only way the pod is driven. It runs
# on the Runpod pod from /root/eddm_s8/code (this repository's code, uploaded with src/tools/upload_to_pod.py --driver
# /root/eddm_s8/code/scripts/pod_session8.sh, which also starts these subcommands). Every stage detaches itself under
# nohup, so a closed connection does not stop it, and appends its log to /root/eddm_s8/logs/<stage>.log. Stages:
#   prepare        step 9: every uploaded file checked against /root/eddm_s8/uploads_manifest.json, which
#                  upload_to_pod.py writes as it places files (SHA-256; correction 5); GPU check
#                  and installs; Laya and Qwen3-4B-Base at their fixed revisions; session 4's records rebuilt and checked
#                  against a copy of results/public_v0/checksums.json; session 7's records built here and checked against
#                  session 7's records_manifest.json (session 7's correction 3); EI's raw files downloaded at the fixed
#                  commit (each checked against records_manifest.json) and EI's records rebuilt with evidence_inference.py
#                  from-selection, each file checked (records_manifest_pod.json); s7_b3's uploaded folder checked against
#                  its SHA-256; EI's items (training and monitoring subset, max_len 4,096), stopping if any is cut or skipped
#   prepare_a      step 9: reader v1 at its revision, checked and put into Laya's format (session 7's correction 4);
#                  label_pqa_u.py rebuild (phase 2's records, checked against labeler.json; if they differ, arm A is NOT
#                  BUILT and arm_a will not run); correction 5's phase 3 records; arm A's items
#   pilot          step 10: train_qwen_head.py pilot-s8 and results/fiorillo_v0_4/pilot.json
#   ei             step 11: s8_ei_1, s8_ei_2 and s8_ei_3 in that order, each trained from s7_b3 (the pilot's setting; if a
#                  run runs out of memory it restarts from its start with gradient checkpointing on, recorded) and then
#                  predicting the whole validation split (raw predictions, logits, run summary), copied into results; a
#                  seed whose folder and validation predictions exist is skipped, so the stage resumes (log ei.log)
#   arm_a          step 11: s8_a1, s8_a2 and s8_a3 (session 7's arm A with correction 5's phase 3, --keep last in every
#                  phase), finished phases skipped; then s8_a3's validation predictions and logits copied into results
#   trees          step 11 (correction 3), on the CPU beside ei at a lower priority (nice 10; correction 5): XGBoost
#                  3.2.0 and CatBoost 1.2.10 installed; trees_session8.py search, validation (models in
#                  /root/eddm_s8/models/trees/) and pool, 12 threads
#   prepare_large  step 11, beside ei: vLLM's environment (/root/eddm_s8/vllm_env/) at fixed_versions.json's version and
#                  build for the driver's CUDA version (correction 1); Gemma 4 31B-it at its revision (or, if it is gated,
#                  a wait for my download); a check that vLLM imports, that CUDA starts, that vLLM's compiled kernels load
#                  and that a small Triton kernel runs inside the environment (correction 5)
#   smoke_large    step 11 (correction 9), beside ei once arm_a has finished and outside any EI prediction pass: Gemma 4
#                  E2B-it (same class, head sizes, soft cap and tokenizer as the 31B) downloaded to
#                  /root/eddm_s8/models/gemma4_e2b_smoke/ and run at gpu_memory_utilization 0.25 through
#                  large_model_session8.py's own engine and parsing code (src/eval/smoke_large_session8.py first and
#                  second, then the script's own answer and second: correction 12), so FlashAttention 4 compiles on this
#                  driver before large_validation (FlashInfer's sampler is not used: correction 11); after each process
#                  the GPU must be back at its level from before the smoke test (correction 12); its outputs go only to
#                  /root/eddm_s8/smoke_large/ and are used for nothing else
#   large_validation  step 12: large_model_session8.py on the validation items, alone on the GPU (it stops if the GPU's
#                  used memory says another process holds it), then the second process if some item still needs it
#                  (correction 5)
#   calibrate      step 12: calibrate_session8.py fit; the ECE limit (ece_noise_floor.py, 5,000 draws, n 1,218, on
#                  s8_ei's calibrated validation predictions); handoff_session8.py on validation; power_session8.py
#                  validation
#   freeze         step 12: calibrate_session8.py freeze (release_candidate.json), before any test prediction exists; it
#                  refuses while the trees stage is still running or before readmission_pool.json exists (correction 5)
#   testpreds      step 13: refuses to start unless release_candidate.json exists; first writes testpreds_log.json (that
#                  file's SHA-256 and this stage's start time in UTC); then, one job at a time on the GPU, raw test
#                  predictions and logits for s8_ei_1, s8_ei_2, s8_ei_3 and s8_a3, s7_b3_timing on PubMedQA's test items
#                  (timing only; its predictions stay in /root/eddm_s8/timing/, and the largest difference from session
#                  7's s7_b3-raw test file goes into its run summary, correction 4), then the large model on the test
#                  items, alone; then the tree members' test predictions (CPU). Each is checked for its row count; only
#                  the raw files, the logits, the run summaries and the large model's files are copied into results.
#                  Run again after a failure, it keeps testpreds_log.json's first entry, adds the retry's time, and runs
#                  only the jobs whose files are not all there
#   package        step 13: fiorillo_v0_4_results.zip, fiorillo_v0_4_ei.zip, fiorillo_v0_4_arm_a.zip,
#                  fiorillo_v0_4_logs.zip and fiorillo_v0_4_trees.zip in /root/eddm_s8/outputs/, SHA-256 in SHA256SUMS.txt
# POD_RATE (dollars an hour) may be set for the hand-off's costs: bash pod_session8.sh calibrate with POD_RATE=3.52 in the
# environment, or written to /root/eddm_s8/pod_rate.txt. Usage on the pod:
#   bash /root/eddm_s8/code/scripts/pod_session8.sh <stage>
BASE=${EDDM_POD_BASE:-/root/eddm_s8}  # the pod's folder; EDDM_POD_BASE only for a test run of this script off the pod
CODE=$BASE/code
MODELS=$BASE/models
LOGS=$BASE/logs
OUTS=$BASE/outputs
ITEMS=$BASE/items
LARGE_ITEMS=$BASE/large_items
STAGE_DIR=$BASE/predictions_staging
TIMING=$BASE/timing
VENV=$BASE/vllm_env
RES=$CODE/results/fiorillo_v0_4
RES7=$CODE/results/fiorillo_v0_3
REC7=$CODE/data/processed/session7
REC8=$CODE/data/processed/session8
LAYA_REV=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
QWEN_REV=906bfd4b4dc7f14ee4320094d8b41684abff8539
READER_SHA=28e9a7b62be95615ae6ec80fbb3a92a4694644e27422c3c3048e01ec18c2d201
READER_REPO=johanneli/fiorillo-reader-v1
READER_REV=c92dc7a9f3be0e0887591ebd21874846eac2fb44
GEMMA_REPO=google/gemma-4-31B-it
TREE_THREADS=12
SEEDS=(20260927 20260928 20260929)

STAGE="${1:-}"
if [ -z "$STAGE" ]; then sed -n '2,52p' "$0"; exit 1; fi
if [ -z "${EDDM_DETACHED:-}" ]; then
  mkdir -p "$LOGS"
  EDDM_DETACHED=1 nohup bash "$0" "$STAGE" >> "$LOGS/$STAGE.log" 2>&1 < /dev/null &
  echo "started $STAGE (process $!); log $LOGS/$STAGE.log"
  exit 0
fi
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
if ! command -v python >/dev/null 2>&1; then python() { python3 "$@"; }; fi
cd "$CODE"
mkdir -p "$MODELS" "$OUTS" "$ITEMS" "$LARGE_ITEMS" "$TIMING" "$STAGE_DIR"/{predictions,logits,test/predictions,test/logits} \
  "$RES"/{metrics,logs,predictions,logits,run_summaries,large_model}
say() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
say "stage $STAGE started"

json_get() {  # json_get <file> <python expression on the dict d>
  python -c "import json; d = json.load(open('$1')); print($2)"
}

check_rows() {  # check_rows <file> <rows>
  local n
  n=$(($(wc -l < "$1") - 1))
  if [ "$n" != "$2" ]; then say "$1 has $n rows, expected $2; stopping"; exit 1; fi
}

rows_ok() {  # rows_ok <file> <rows>: true if the file exists with that many rows
  [ -f "$1" ] && [ "$(($(wc -l < "$1") - 1))" = "$2" ]
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

pod_rate() {
  if [ -n "${POD_RATE:-}" ]; then echo "$POD_RATE"; elif [ -f "$BASE/pod_rate.txt" ]; then cat "$BASE/pod_rate.txt"; fi
}

laya_items() {  # laya_items <out> <max_len> <questions> <records...>
  local out="$1" max_len="$2" questions="$3"
  shift 3
  python src/train/build_items.py --records "$@" --questions "$questions" --model-dir "$MODELS/reader_v1_laya" \
    --out "$out" --max-len "$max_len" --head-max-len 192
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

phase_done() {  # phase_done <system> <weights file>: the phase finished (its metrics JSON is written last)
  [ -f "$RES/metrics/$1_metrics.json" ] && [ -f "$2" ]
}

train_a() {  # train_a <system> <model dir> <train items> <val items> <extra flags...>
  local system="$1" model_dir="$2" train="$3" val="$4"
  shift 4
  if phase_done "$system" "$MODELS/$system/model.safetensors"; then say "arm A: $system already finished; not trained again"; return 0; fi
  say "arm A: $system from $model_dir"
  # shellcheck disable=SC2068
  python src/train/train_laya.py --model-dir "$model_dir" --train "$train" --val "$val" --test "$ITEMS/not_built.pt" --no-test \
    --out "$MODELS/$system" --system-name "$system" --device cuda --epochs 1 --micro-batch 8 --grad-accum 4 --max-len 768 \
    --batching by-question --calibration-mode separate --autocast float16 \
    --metrics "$RES/metrics/${system}_metrics.json" --log "$RES/logs/${system}_train_log.csv" $@
  say "arm A: $system done"
}

ei_setting() {  # the pilot's chosen micro-batch, accumulation and gradient checkpointing
  if [ ! -f "$RES/pilot.json" ]; then say "pilot.json is missing; run the pilot first"; exit 1; fi
  if [ "$(json_get "$RES/pilot.json" "d['stop_and_ask']")" = "True" ]; then say "pilot.json says to stop and ask (no setting passed)"; exit 1; fi
  MB=$(json_get "$RES/pilot.json" "d['chosen']['micro_batch']")
  GA=$(json_get "$RES/pilot.json" "d['chosen']['grad_accum']")
  CKPT=""
  if [ "$(json_get "$RES/pilot.json" "d['chosen']['gradient_checkpointing']")" = "True" ]; then CKPT="--gradient-checkpointing"; fi
}

train_ei() {  # train_ei <system> <seed>
  local system="$1" seed="$2" rc=0 ckpt="$CKPT"
  if phase_done "$system" "$MODELS/$system/adapter.safetensors"; then say "EI: $system already trained"; return 0; fi
  if [ -z "$ckpt" ] && [ -f "$BASE/ei_checkpointing_on.txt" ]; then ckpt="--gradient-checkpointing"; fi
  say "EI: $system, seed $seed, micro-batch $MB, accumulation $GA, ${ckpt:-no gradient checkpointing}"
  run_ei() {
    # shellcheck disable=SC2086
    python src/train/train_qwen_head.py train --base-dir "$MODELS/qwen3_4b_base" --init "$MODELS/s7_b3" --train "$ITEMS/ei_train.pt" \
      --val "$ITEMS/ei_monitor.pt" --out "$MODELS/$system" --system-name "$system" --seed "$seed" --epochs 2 --keep last \
      --freeze-steps 0 --eval-every 80 --max-len 4096 --micro-batch "$MB" --grad-accum "$GA" \
      --metrics "$RES/metrics/${system}_metrics.json" --log "$RES/logs/${system}_train_log.csv" $1 2>&1 | tee "$LOGS/${system}.out"
  }
  run_ei "$ckpt" || rc=$?
  if [ "$rc" != 0 ]; then
    if [ -z "$ckpt" ] && grep -qi "out of memory" "$LOGS/${system}.out"; then
      say "EI: $system ran out of memory without gradient checkpointing; restarting it from its start with it on (kept on for later seeds)"
      echo "$system ran out of memory at $(date -u '+%Y-%m-%dT%H:%M:%SZ'); gradient checkpointing on from here" >> "$BASE/ei_checkpointing_on.txt"
      run_ei "--gradient-checkpointing"
    else
      say "EI: $system failed (exit $rc)"
      exit "$rc"
    fi
  fi
  say "EI: $system trained"
}

predict_ei() {  # predict_ei <system> <split> <out dir> <logits dir>
  local system="$1" split="$2" out="$3" logits="$4"
  say "EI: $system predicts $split; GPU memory used now: $(gpu_used_mib) MiB (arm A may be training beside it)"
  python src/train/train_qwen_head.py predict --base-dir "$MODELS/qwen3_4b_base" --phase-dir "$MODELS/$system" --system "$system" \
    --records-dir "$REC8" --qids evidence_effect --splits "$split" --questions src/questions/session8.yaml --batch-size 4 \
    --out "$out" --logits-dir "$logits" --run-summary "$RES/run_summaries/${system}_${split}_run_summary.json"
  if [ "$(json_get "$RES/run_summaries/${system}_${split}_run_summary.json" "d['items_$split']['skipped'] == 0 and d['items_$split']['truncated'] == 0")" != "True" ]; then
    say "$system $split: records were cut or skipped in predict; stopping"; exit 1
  fi
}

case "$STAGE" in
  prepare)
    # the uploads first, so that a missing or changed file stops the stage before the installs and downloads
    python - "$BASE/uploads_manifest.json" "$CODE" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, code = json.load(open(sys.argv[1])), Path(sys.argv[2])
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
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
    pip install -q "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "hf-xet==1.6.0" "safetensors==0.8.0" "pyyaml==6.0.3" \
      "pyarrow==25.0.1" "scipy==1.17.1" "numpy==2.4.6" "accelerate==1.15.0" "laya==0.3.20" "requests==2.34.2"
    python -c "import transformers, tokenizers, huggingface_hub, laya, numpy, torch; print('transformers', transformers.__version__, '| tokenizers', tokenizers.__version__, '| huggingface_hub', huggingface_hub.__version__, '| laya', laya.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", revision="$LAYA_REV", local_dir="$MODELS/laya",
                  allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
snapshot_download("Qwen/Qwen3-4B-Base", revision="$QWEN_REV", local_dir="$MODELS/qwen3_4b_base")
print("downloads done")
PY
    printf 'revision %s\ndownloaded with huggingface_hub snapshot_download (Qwen/Qwen3-4B-Base)\n' "$QWEN_REV" > "$MODELS/qwen3_4b_base/REVISION.txt"
    if [ ! -f "$BASE/expected_checksums.json" ]; then cp results/public_v0/checksums.json "$BASE/expected_checksums.json"; fi
    python src/tasks/pubmedqa.py
    python src/tasks/uci_readmission.py
    python - "$BASE/expected_checksums.json" data/processed/public <<'PY'
import hashlib, json, sys
from pathlib import Path
expected, folder = json.load(open(sys.argv[1])), Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad = {k: v for k, v in expected.items() if sha(folder / k) != v}
if bad:
    sys.exit(f"record SHA-256 values differ from session 4's: {sorted(bad)}")
print(f"session 4's records rebuilt: {len(expected)} files, SHA-256 identical to the copy of results/public_v0/checksums.json")
PY
    python - <<'PY'
import sys
sys.path[:0] = ["src/tasks", "src/corpus"]
import records as R
from pubmedqa_exclusion import FILES, RAW, url
for config, (path_in_repo, sha) in FILES.items():
    R.fetch(url(path_in_repo), RAW / f"{config}.parquet", sha)
    print(f"{config}.parquet: downloaded and checked, SHA-256 {sha[:16]}...")
PY
    python src/tasks/pubmedqa_phases.py --processes 16 --max-seconds 100000 --manifest "$BASE/records_manifest_s7_pod.json"
    cp "$BASE/records_manifest_s7_pod.json" "$RES/records_manifest_s7_pod.json"
    python - "$RES7/records_manifest.json" "$REC7" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, folder = json.load(open(sys.argv[1])), Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
for name, entry in manifest["files"].items():
    got = sha(folder / name)
    assert got == entry["sha256"], f"{name}: SHA-256 {got} differs from the Mac's records_manifest.json (session 7)"
print(f"session 7's records built on the pod: {len(manifest['files'])} files, each identical to session 7's records_manifest.json")
PY
    for attempt in 1 2 3 4 5; do  # the download resumes; each call runs until it is complete or its time is up
      status=$(python src/tasks/evidence_inference.py download --raw data/raw/evidence_inference --max-seconds 3000 2>&1 | tail -n 1 || true)
      echo "$status"
      if python -c "import json, sys; sys.exit(0 if json.loads(sys.argv[1])['complete'] else 1)" "$status" 2>/dev/null; then break; fi
      say "EI download: attempt $attempt incomplete or failed; resuming after 30 s"
      sleep 30
    done
    python - "$RES/records_manifest.json" data/raw/evidence_inference <<'PY'
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
    sys.exit(f"{len(bad)} of EI's raw files differ from records_manifest.json, for example {bad[:3]}")
print(f"EI's raw files: {len(expected)} files, each identical to records_manifest.json")
PY
    python src/tasks/evidence_inference.py from-selection --raw data/raw/evidence_inference --selection "$REC8/article_selection.jsonl" \
      --out-dir "$REC8" --manifest "$RES/records_manifest.json" --check-out "$BASE/records_manifest_pod.json"
    cp "$BASE/records_manifest_pod.json" "$RES/records_manifest_pod.json"
    python - "$RES/records_manifest.json" "$MODELS/s7_b3/adapter.safetensors" <<'PY'
import hashlib, json, sys
manifest = json.load(open(sys.argv[1]))
h = hashlib.sha256()
with open(sys.argv[2], "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
assert h.hexdigest() == manifest["s7_b3_adapter_sha256"], f"s7_b3's adapter: SHA-256 {h.hexdigest()}, expected {manifest['s7_b3_adapter_sha256']}"
print("s7_b3's adapter.safetensors: SHA-256 checked")
PY
    qwen_items "$ITEMS/ei_train.pt" 4096 src/questions/session8.yaml "$REC8/evidence_effect_train.jsonl"
    no_cuts "$ITEMS/ei_train.summary.json"
    qwen_items "$ITEMS/ei_monitor.pt" 4096 src/questions/session8.yaml "$REC8/evidence_effect_validation_monitor.jsonl"
    no_cuts "$ITEMS/ei_monitor.summary.json"
    python - "$ITEMS" <<'PY'
import json, sys
from pathlib import Path
out = {p.name.replace(".summary.json", ""): json.load(open(p)) for p in sorted(Path(sys.argv[1]).glob("ei_*.summary.json"))}
print("EI's items built (records, items, skipped, cut):")
print(json.dumps({k: {x: v.get(x) for x in ("records", "items", "skipped", "truncated")} for k, v in out.items()}, indent=1))
PY
    ;;
  prepare_a)
    if [ ! -f "$MODELS/reader_v1/model.safetensors" ]; then
      python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$READER_REPO", revision="$READER_REV", local_dir="$MODELS/reader_v1",
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])
print("reader v1 downloaded from $READER_REPO at $READER_REV")
PY
    fi
    python - "$MODELS/reader_v1" "$READER_SHA" <<'PY'
import hashlib, sys
from pathlib import Path
reader, reader_sha = Path(sys.argv[1]), sys.argv[2]
missing = [n for n in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json") if not (reader / n).exists()]
if missing:
    sys.exit(f"reader v1 is not on the pod yet: {missing}")
h = hashlib.sha256()
with open(reader / "model.safetensors", "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
assert h.hexdigest() == reader_sha, f"reader v1: SHA-256 {h.hexdigest()}, expected {reader_sha}"
print("reader v1: SHA-256 checked")
PY
    if [ ! -f "$MODELS/reader_v1_laya/model.safetensors" ]; then
      python src/train/assemble_reader.py --laya-dir "$MODELS/laya" --reader-dir "$MODELS/reader_v1" --out "$MODELS/reader_v1_laya" \
        --laya-revision "convaiinnovations/laya@$LAYA_REV" --reader-revision "reader v1, model.safetensors $READER_SHA"
    fi
    if ! python src/eval/label_pqa_u.py rebuild --results "$RES7" --records-dir "$REC7"; then
      echo "NOT BUILT: arm A, phase 2's records differ from session 7's ($(date -u '+%Y-%m-%dT%H:%M:%SZ'))" > "$BASE/arm_a_not_built.txt"
      say "$(cat "$BASE/arm_a_not_built.txt"); arm_a will not run"
      exit 1
    fi
    python - "$REC7" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]) / "pubmedqa_pqa_l_train.jsonl"
if not out.exists():
    rows = [l for l in open("data/processed/public/pubmedqa_answer_train.jsonl", encoding="utf-8")
            if json.loads(l).get("group", {}).get("source") == "pqa_labeled"]
    out.write_text("".join(rows), encoding="utf-8")
print(out.name, sum(1 for _ in open(out, encoding="utf-8")), "records")
PY
    laya_items "$ITEMS/a_phase1.pt" 768 src/questions/public_v0.yaml "$REC7/pubmedqa_phase1_train.jsonl"
    laya_items "$ITEMS/a_val12.pt" 768 src/questions/public_v0.yaml data/processed/public/pubmedqa_answer_validation.jsonl
    laya_items "$ITEMS/a_phase2.pt" 768 src/questions/public_v0.yaml "$REC7/pubmedqa_phase2_train.jsonl"
    laya_items "$ITEMS/a_phase3.pt" 768 src/questions/public_v0.yaml data/processed/public/readmit_window_train.jsonl \
      "$REC7/pubmedqa_pqa_l_train.jsonl" "$REC7/pubmedqa_phase2_train.jsonl"
    laya_items "$ITEMS/a_val3.pt" 768 src/questions/public_v0.yaml data/processed/public/readmit_window_validation.jsonl \
      data/processed/public/pubmedqa_answer_validation.jsonl
    python - "$ITEMS" <<'PY'
import json, sys
from pathlib import Path
out = {p.name.replace(".summary.json", ""): json.load(open(p)) for p in sorted(Path(sys.argv[1]).glob("a_*.summary.json"))}
print("arm A's items built (records, items, skipped, cut):")
print(json.dumps({k: {x: v.get(x) for x in ("records", "items", "skipped", "truncated")} for k, v in out.items()}, indent=1))
if out["a_phase3"]["records"] != 87627:
    sys.exit(f"phase 3 has {out['a_phase3']['records']} records, expected 87,627 (correction 5 of session 7)")
PY
    ;;
  pilot)
    gpu_alone
    python src/train/train_qwen_head.py pilot-s8 --base-dir "$MODELS/qwen3_4b_base" --train "$ITEMS/ei_train.pt" --val "$ITEMS/ei_monitor.pt" \
      --out-json "$RES/pilot.json"
    ;;
  ei)
    ei_setting
    for k in 1 2 3; do
      system="s8_ei_$k"
      if [ -f "$MODELS/$system/adapter.safetensors" ] && rows_ok "$RES/predictions/${system}-raw_evidence_effect_validation.csv" 1237; then
        say "EI: $system and its validation predictions exist; skipped"
        continue
      fi
      train_ei "$system" "${SEEDS[$((k - 1))]}"
      predict_ei "$system" validation "$STAGE_DIR/predictions" "$STAGE_DIR/logits"
      check_rows "$STAGE_DIR/predictions/${system}-raw_evidence_effect_validation.csv" 1237
      check_rows "$STAGE_DIR/logits/${system}_evidence_effect_validation_logits.csv" 1237
      cp "$STAGE_DIR/predictions/${system}-raw_evidence_effect_validation.csv" "$RES/predictions/"
      cp "$STAGE_DIR/logits/${system}_evidence_effect_validation_logits.csv" "$RES/logits/"
      say "EI: $system finished (validation predictions, logits, metrics, log and run summary in results)"
    done
    ;;
  arm_a)
    if [ -f "$BASE/arm_a_not_built.txt" ]; then say "$(cat "$BASE/arm_a_not_built.txt")"; exit 1; fi
    train_a s8_a1 "$MODELS/reader_v1_laya" "$ITEMS/a_phase1.pt" "$ITEMS/a_val12.pt" --reinit-head --freeze-encoder-steps 250 --keep last --eval-every 0
    train_a s8_a2 "$MODELS/s8_a1" "$ITEMS/a_phase2.pt" "$ITEMS/a_val12.pt" --keep last --eval-every 0
    train_a s8_a3 "$MODELS/s8_a2" "$ITEMS/a_phase3.pt" "$ITEMS/a_val3.pt" --keep last --eval-every 250 \
      --predictions-dir "$STAGE_DIR/predictions" --logits-dir "$STAGE_DIR/logits"
    cp "$STAGE_DIR"/predictions/s8_a3-raw_*_validation.csv "$RES/predictions/"
    cp "$STAGE_DIR"/logits/s8_a3_*_validation_logits.csv "$RES/logits/"
    say "arm A: s8_a3's validation predictions and logits copied"
    ;;
  trees)
    pip install -q "xgboost==3.2.0" "catboost==1.2.10" "pandas==3.0.6" "scikit-learn==1.9.1"
    python -c "import xgboost, catboost, pandas, sklearn; print('xgboost', xgboost.__version__, '| catboost', catboost.__version__, '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__)"
    nice -n 10 python src/eval/trees_session8.py search --threads "$TREE_THREADS"
    nice -n 10 python src/eval/trees_session8.py validation --threads "$TREE_THREADS" --models-dir "$MODELS/trees"
    nice -n 10 python src/eval/trees_session8.py pool --processes "$TREE_THREADS"
    ;;
  prepare_large)
    driver=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
    say "driver CUDA version $driver"
    pip install -q uv
    uv --version
    if [ ! -x "$VENV/bin/python" ]; then uv venv --python 3.12 "$VENV"; fi
    if python -c "import sys; v = tuple(int(x) for x in '$driver'.split('.')[:2]); sys.exit(0 if v >= (13, 0) else 1)"; then
      say "installing vLLM 0.30.0's default build (CUDA 13.0)"
      uv pip install --python "$VENV/bin/python" "vllm==0.30.0" "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "hf-xet==1.6.0"
    else
      say "installing vLLM 0.30.0's CUDA 12.9 build with torch 2.13.0 for CUDA 12.9 (correction 1)"
      uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
        "torch==2.13.0+cu129" "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "hf-xet==1.6.0"
      # torchcodec from PyPI is built for CUDA 13 (libnvrtc.so.13) and does not load here; vLLM imports it with its
      # multimodal code and uses it only to decode audio and video, so its CPU build is used (correction 10)
      uv pip install --python "$VENV/bin/python" --no-deps --index-url https://download.pytorch.org/whl/cpu "torchcodec==0.17.0+cpu"
    fi
    # written to a file and run from it, because Triton reads a kernel's source from its file (correction 8)
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
      say "vLLM does not import, CUDA does not start, or vLLM's or Triton's kernels do not run in $VENV; stopping, and Johann is asked (corrections 1 and 5)"
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
    SMOKE_MODEL=$MODELS/gemma4_e2b_smoke
    E2B_REV=3e22461f65e89153144f8adb70e3b8c2cc9845a7  # google/gemma-4-E2B-it, not gated, the 31B's tokenizer.json
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
print("cuda.bindings CUDA_VERSION", drv.CUDA_VERSION, "(below 13000 is needed on this 12.8 driver)")
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
    if [ ! -f "$LARGE_ITEMS/prompts_evidence_effect_validation.jsonl" ]; then
      python src/eval/large_model_session8.py prompts --splits validation --items-dir "$LARGE_ITEMS"
    fi
    smoke_gpu_back() {  # correction 12: each smoke process must leave the GPU as it found it (waits up to a minute)
      local now waited=0
      now=$(gpu_used_mib)
      while [ "$now" -gt $((SMOKE_BASE + 3000)) ]; do
        if [ "$waited" -ge 60 ]; then
          say "smoke test: the GPU holds $now MiB against $SMOKE_BASE MiB before; a smoke process left its engine; stopping"; nvidia-smi; exit 1
        fi
        sleep 5
        waited=$((waited + 5))
        now=$(gpu_used_mib)
      done
      say "smoke test: GPU used $now MiB (before the smoke test: $SMOKE_BASE MiB)"
    }
    smoke_run() {  # smoke_run <mode> [options]: one smoke process under a 20-minute timeout, then the GPU check, also on a failure
      local rc=0
      timeout -k 30 1200 "$VENV/bin/python" src/eval/smoke_large_session8.py "$@" || rc=$?
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
    PIPE_MODEL=$BASE/smoke_large/e2b_with_31b_template  # E2B's files with the 31B's chat template linked in (correction 12)
    PIPE_DIR=$BASE/smoke_large/pipeline_$(date -u +%Y%m%dT%H%M%SZ)  # a new folder for each smoke run
    mkdir -p "$PIPE_MODEL"
    for f in "$SMOKE_MODEL"/*; do ln -sfn "$f" "$PIPE_MODEL/$(basename "$f")"; done
    if [ ! -f "$MODELS/gemma4_31b/chat_template.jinja" ]; then say "the 31B has no chat_template.jinja to link; stopping"; exit 1; fi
    ln -sfn "$MODELS/gemma4_31b/chat_template.jinja" "$PIPE_MODEL/chat_template.jinja"
    say "smoke test: the script's own answer() on the six items, then its second() and finish() in a new process (correction 12), in $PIPE_DIR"
    smoke_run pipeline_answer --baseline-mib "$SMOKE_BASE" --pipeline-model-dir "$PIPE_MODEL" --pipeline-dir "$PIPE_DIR"
    smoke_run pipeline_second --baseline-mib "$SMOKE_BASE" --pipeline-model-dir "$PIPE_MODEL" --pipeline-dir "$PIPE_DIR"
    say "SMOKE TEST PASSED"
    ;;
  large_validation)
    gpu_alone
    python src/eval/large_model_session8.py prompts --splits validation --items-dir "$LARGE_ITEMS"
    "$VENV/bin/python" src/eval/large_model_session8.py answer --splits validation --model-dir "$MODELS/gemma4_31b" --items-dir "$LARGE_ITEMS"
    # always (correction 9): with no item pending, second loads no model and only writes the results again, so a file
    # left by an earlier run cannot stand in for this run's
    gpu_alone
    "$VENV/bin/python" src/eval/large_model_session8.py second --splits validation --model-dir "$MODELS/gemma4_31b" --items-dir "$LARGE_ITEMS"
    check_rows "$RES/large_model/gemma4_31b-raw_evidence_effect_validation.csv" 1237
    check_rows "$RES/large_model/gemma4_31b-raw_pubmedqa_answer_validation.csv" 150
    python - "$RES/large_model/gemma4_31b_letters.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for q in ("evidence_effect", "pubmedqa_answer"):
    v = d[q]["validation"]
    print(q, "validation: letters' summed probability", json.dumps(v["letters_summed_probability"]), "| answered the second way:", v["answered_second_way"])
m = d["evidence_effect"]["validation"]["letters_summed_probability"]["median"]
if m < 0.5:
    print("THE MEDIAN IS BELOW 0.5 ON VALIDATION: a correction is appended before any test prediction (THE LARGE MODEL)")
PY
    ;;
  calibrate)
    if [ -f "$BASE/arm_a_not_built.txt" ]; then
      say "$(cat "$BASE/arm_a_not_built.txt"); s8_a3 is not calibrated"
      python src/eval/calibrate_session8.py fit --systems s8_ei s8_ei_1 s8_ei_2 s8_ei_3 gemma4_31b
    else
      python src/eval/calibrate_session8.py fit
    fi
    python src/eval/ece_noise_floor.py --predictions "$RES/predictions/s8_ei_evidence_effect_validation.csv" \
      --records "$REC8/evidence_effect_validation.jsonl" --n 1218 --draws 5000 --out "$RES/ece_limit_evidence_effect.json"
    rate=$(pod_rate)
    python src/eval/handoff_session8.py --split validation ${rate:+--hourly-rate "$rate"}
    python src/eval/power_session8.py validation --records "$REC8/evidence_effect_validation.jsonl" \
      --system "$RES/predictions/s8_ei_evidence_effect_validation.csv" --reference "$RES/predictions/gemma4_31b_evidence_effect_validation.csv"
    ;;
  freeze)
    if pgrep -f "trees_session8.py" > /dev/null; then say "the trees stage is still running; the freeze waits for readmission_pool.json; stopping"; exit 1; fi
    if [ ! -f "$RES/readmission_pool.json" ]; then say "readmission_pool.json does not exist; run trees first (or freeze with --pool-not-built REASON by hand); stopping"; exit 1; fi
    python src/eval/calibrate_session8.py freeze --staging "$STAGE_DIR"
    ;;
  testpreds)
    if [ ! -f "$RES/release_candidate.json" ]; then echo "Stop: results/fiorillo_v0_4/release_candidate.json does not exist"; exit 1; fi
    python - "$RES" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
res = Path(sys.argv[1])
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
digest = hashlib.sha256((res / "release_candidate.json").read_bytes()).hexdigest()
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
    gpu_alone
    for k in 1 2 3; do
      system="s8_ei_$k"
      if rows_ok "$RES/predictions/${system}-raw_evidence_effect_test.csv" 1218 && rows_ok "$RES/logits/${system}_evidence_effect_test_logits.csv" 1218; then
        say "$system's test predictions exist"; continue
      fi
      predict_ei "$system" test "$STAGE_DIR/test/predictions" "$STAGE_DIR/test/logits"
      check_rows "$STAGE_DIR/test/predictions/${system}-raw_evidence_effect_test.csv" 1218
      check_rows "$STAGE_DIR/test/logits/${system}_evidence_effect_test_logits.csv" 1218
      cp "$STAGE_DIR/test/predictions/${system}-raw_evidence_effect_test.csv" "$RES/predictions/"
      cp "$STAGE_DIR/test/logits/${system}_evidence_effect_test_logits.csv" "$RES/logits/"
    done
    if [ -f "$MODELS/s8_a3/model.safetensors" ] && ! { rows_ok "$RES/predictions/s8_a3-raw_pubmedqa_answer_test.csv" 500 \
         && rows_ok "$RES/predictions/s8_a3-raw_readmit_window_test.csv" 6998; }; then
      python src/eval/run_laya_model.py --model-dir "$MODELS/s8_a3" --system s8_a3 --records-dir data/processed/public \
        --qids pubmedqa_answer readmit_window --out "$STAGE_DIR/test/predictions" --logits-dir "$STAGE_DIR/test/logits" \
        --max-len 768 --head-max-len 192 --device cuda --calibration "$MODELS/s8_a3/calibration.json" --splits test
      check_rows "$STAGE_DIR/test/predictions/s8_a3-raw_pubmedqa_answer_test.csv" 500
      check_rows "$STAGE_DIR/test/predictions/s8_a3-raw_readmit_window_test.csv" 6998
      check_rows "$STAGE_DIR/test/logits/s8_a3_pubmedqa_answer_test_logits.csv" 500
      check_rows "$STAGE_DIR/test/logits/s8_a3_readmit_window_test_logits.csv" 6998
      cp "$STAGE_DIR"/test/predictions/s8_a3-raw_*_test.csv "$RES/predictions/"
      cp "$STAGE_DIR"/test/logits/s8_a3_*_test_logits.csv "$RES/logits/"
      if [ -f "$STAGE_DIR/test/s8_a3_run_summary.json" ]; then cp "$STAGE_DIR/test/s8_a3_run_summary.json" "$RES/run_summaries/s8_a3_test_run_summary.json"; fi
    fi
    if [ ! -f "$RES/run_summaries/s7_b3_timing_test_run_summary.json" ]; then
      python src/train/train_qwen_head.py predict --base-dir "$MODELS/qwen3_4b_base" --phase-dir "$MODELS/s7_b3" --system s7_b3_timing \
        --records-dir data/processed/public --qids pubmedqa_answer --splits test --out "$TIMING" \
        --run-summary "$TIMING/s7_b3_timing_test_run_summary.json"
      check_rows "$TIMING/s7_b3_timing-raw_pubmedqa_answer_test.csv" 500
      python - "$TIMING/s7_b3_timing-raw_pubmedqa_answer_test.csv" "$RES7/predictions/s7_b3-raw_pubmedqa_answer_test.csv" \
        "$TIMING/s7_b3_timing_test_run_summary.json" "$RES/run_summaries/s7_b3_timing_test_run_summary.json" <<'PY'
import csv, json, sys
from pathlib import Path
def read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return {r["id"]: (r["label"], [float(r[k]) for k in ("p0", "p1", "p2")]) for r in csv.DictReader(f)}
a, b = read(sys.argv[1]), read(sys.argv[2])
assert set(a) == set(b), "s7_b3_timing and session 7's s7_b3-raw test file hold different items"
assert all(a[i][0] == b[i][0] for i in a), "labels differ"
diff = max(abs(x - y) for i in a for x, y in zip(a[i][1], b[i][1]))
summary = json.load(open(sys.argv[3]))
summary["max_abs_difference_vs_session7"] = diff
summary["compared_with"] = "results/fiorillo_v0_3/predictions/s7_b3-raw_pubmedqa_answer_test.csv (correction 4)"
Path(sys.argv[4]).write_text(json.dumps(summary, indent=2) + "\n")
print(f"s7_b3_timing against session 7's s7_b3-raw test file: largest difference in a probability {diff:.3g}")
PY
    fi
    if ! { rows_ok "$RES/large_model/gemma4_31b-raw_evidence_effect_test.csv" 1218 && rows_ok "$RES/large_model/gemma4_31b-raw_pubmedqa_answer_test.csv" 500; }; then
      gpu_alone
      python src/eval/large_model_session8.py prompts --splits test --items-dir "$LARGE_ITEMS"
      "$VENV/bin/python" src/eval/large_model_session8.py answer --splits test --model-dir "$MODELS/gemma4_31b" --items-dir "$LARGE_ITEMS"
      gpu_alone  # second always runs (correction 9)
      "$VENV/bin/python" src/eval/large_model_session8.py second --splits test --model-dir "$MODELS/gemma4_31b" --items-dir "$LARGE_ITEMS"
      check_rows "$RES/large_model/gemma4_31b-raw_evidence_effect_test.csv" 1218
      check_rows "$RES/large_model/gemma4_31b-raw_pubmedqa_answer_test.csv" 500
    fi
    if ! { rows_ok "$RES/predictions/xgboost_tuned_readmit_window_test.csv" 6998 && rows_ok "$RES/predictions/catboost_tuned_readmit_window_test.csv" 6998; }; then
      python src/eval/trees_session8.py testpreds --threads "$TREE_THREADS" --models-dir "$MODELS/trees"
      check_rows "$RES/predictions/xgboost_tuned_readmit_window_test.csv" 6998
      check_rows "$RES/predictions/catboost_tuned_readmit_window_test.csv" 6998
    fi
    say "test predictions written: s8_ei_1, s8_ei_2, s8_ei_3, s8_a3, s7_b3_timing (summary only), gemma4_31b, xgboost_tuned, catboost_tuned"
    ;;
  package)
    python - "$CODE" "$MODELS" "$LOGS" "$OUTS" <<'PY'
import hashlib, sys, zipfile
from pathlib import Path
code, models, logs, outs = (Path(a) for a in sys.argv[1:5])
def add_tree(z, folder, arc_root):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and "checkpoint_best" not in p.parts:
            z.write(p, Path(arc_root) / p.relative_to(folder))
with zipfile.ZipFile(outs / "fiorillo_v0_4_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, code / "results" / "fiorillo_v0_4", "results/fiorillo_v0_4")
for name, systems in (("ei", ("s8_ei_1", "s8_ei_2", "s8_ei_3")), ("arm_a", ("s8_a1", "s8_a2", "s8_a3")), ("trees", ("trees",))):
    present = [s for s in systems if (models / s).exists()]
    if present:
        with zipfile.ZipFile(outs / f"fiorillo_v0_4_{name}.zip", "w", zipfile.ZIP_STORED) as z:
            for s in present:
                add_tree(z, models / s, s)
    print(name, "zip:", present, "missing:", [s for s in systems if s not in present])
with zipfile.ZipFile(outs / "fiorillo_v0_4_logs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, logs, "logs/pod")
lines = []
for p in sorted(outs.glob("*.zip")):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    lines.append(f"{h.hexdigest()}  {p.stat().st_size}  {p.name}")
(outs / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
    ;;
  *)
    echo "unknown stage $STAGE"; exit 1 ;;
esac
say "stage $STAGE finished"

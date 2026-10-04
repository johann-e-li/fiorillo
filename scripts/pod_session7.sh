#!/usr/bin/env bash
# Session 7 pod driver (docs/prompt_session7.md, step 3): the only way the pod is driven. It runs on the Runpod pod from
# /root/eddm_s7/code (this repository's code, uploaded with src/tools/upload_to_pod.py --driver
# /root/eddm_s7/code/scripts/pod_session7.sh, which also starts these subcommands). Every stage detaches itself under
# nohup, so a closed connection does not stop it, and writes its log to /root/eddm_s7/logs/<stage>.log
# (<stage>_<argument>.log for arm). Stages:
#   prepare    step 8: GPU check; installs; Laya and Qwen3-4B-Base downloaded at the fixed revisions; session 4's records
#              rebuilt with session 4's builders and checked against a copy of results/public_v0/checksums.json made
#              first; session 7's records built here with src/tasks/pubmedqa_phases.py from PubMedQA's three files at
#              the fixed revision (each checked against its SHA-256; correction 3) and every file checked against the
#              records_manifest.json uploaded from the Mac; arm B's items (qwen_items.py, max_len 768), with counts of
#              skipped and cut records
#   prepare_a  step 8 (corrections 3 and 4): reader v1 downloaded from johanneli/fiorillo-reader-v1 at its fixed revision
#              if it is not on the pod, checked against its SHA-256 and put into Laya's format; items for the labeler
#              (max_len 256) and arm A (build_items.py, max_len 768), with counts of skipped and cut records
#   pilot      step 9: arm B's pilot (train_qwen_head.py pilot) and results/fiorillo_v0_3/pilot.json
#   labeler    step 10: s7_lab_a, s7_lab_b, label_pqa_u.py, and phase 2's items for both arms; on any failure,
#              including phase 2's item builds, labeler.json records phase 2 as NOT BUILT, so that neither arm waits
#   arm a|b    step 10: that arm's three phases, one after another; phase 2 waits for labeler.json and then for its
#              items (at most 3 hours, then this stage stops with an error), and is skipped if labeler.json says, or
#              comes to say, that phase 2 was not built; arm B reads gradient checkpointing from pilot.json and, if a
#              phase runs out of memory, restarts that phase with gradient checkpointing on and keeps it on for the
#              phases after it (recorded in the stage log and in each phase's metrics JSON); a phase whose metrics JSON
#              and weights exist is not trained again when the stage is run again; at the end the arm's raw validation
#              predictions and logits are copied from the staging folder into results
#   correct_phase3  correction 5: keeps the first s7_a3 as s7_a3_orig; writes phase 3's corrected records (session 4's
#              readmission training records, its 1,050 PQA-L training records, and phase 2's records instead of the
#              20,000 PQA-A records, whose labels hold no maybe) and puts their items in place of a_phase3.pt and
#              b_phase3.pt (the first ones kept as *_phase3_orig.pt); arm a run again then trains s7_a3 from s7_a2 on them,
#              and the running arm b trains s7_b3 on them
#   select     step 11: select_session7.py select; calibrate_session7.py fit for s7_a3 and s7_b3; the ECE limits
#              (ece_noise_floor.py, 5,000 draws, n 500 and 6,998) on the candidate's calibrated validation predictions
#   freeze     step 11: release_candidate.json (select_session7.py freeze), before any test prediction exists
#   testpreds  step 12: refuses to start unless release_candidate.json exists; first writes testpreds_log.json (that
#              file's SHA-256 and this stage's start time in UTC); then raw test predictions and logits for s7_a3
#              (run_laya_model.py) and s7_b3 (train_qwen_head.py predict) into the staging folder, each checked for 500
#              and 6,998 rows; only the -raw files, the logits and the run summaries are copied into results. Run again
#              after a failure, it keeps testpreds_log.json's first entry, adds the retry's time (release_candidate.json
#              must be unchanged), and predicts only a system whose four test files are not all there
#   package    step 12: fiorillo_v0_3_results.zip, fiorillo_v0_3_arm_a.zip, fiorillo_v0_3_arm_b.zip and
#              fiorillo_v0_3_logs.zip in /root/eddm_s7/outputs/, with their SHA-256 in SHA256SUMS.txt
# Each stage's log is appended to, so a failed attempt's log is kept. Usage on the pod:
#   bash /root/eddm_s7/code/scripts/pod_session7.sh <stage> [argument]
BASE=${EDDM_POD_BASE:-/root/eddm_s7}  # the pod's folder; EDDM_POD_BASE only for a test run of this script off the pod
CODE=$BASE/code
MODELS=$BASE/models
LOGS=$BASE/logs
OUTS=$BASE/outputs
ITEMS=$BASE/items
STAGE_DIR=$BASE/predictions_staging
RES=$CODE/results/fiorillo_v0_3
REC7=$CODE/data/processed/session7
LAYA_REV=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
QWEN_REV=906bfd4b4dc7f14ee4320094d8b41684abff8539
READER_SHA=28e9a7b62be95615ae6ec80fbb3a92a4694644e27422c3c3048e01ec18c2d201
READER_REPO=johanneli/fiorillo-reader-v1
READER_REV=c92dc7a9f3be0e0887591ebd21874846eac2fb44
LABELER_WAIT_SECONDS=10800

STAGE="${1:-}"
ARG="${2:-}"
if [ -z "$STAGE" ]; then sed -n '2,43p' "$0"; exit 1; fi
if [ -z "${EDDM_DETACHED:-}" ]; then
  mkdir -p "$LOGS"
  NAME="$STAGE${ARG:+_$ARG}"
  EDDM_DETACHED=1 nohup bash "$0" "$STAGE" "$ARG" >> "$LOGS/$NAME.log" 2>&1 < /dev/null &
  echo "started $NAME (process $!); log $LOGS/$NAME.log"
  exit 0
fi
set -euo pipefail
# The image's system Python is marked externally managed (PEP 668), so pip needs this to install into it, as in
# sessions 4 and 5; and `python` is python3 where no `python` command exists.
export PIP_BREAK_SYSTEM_PACKAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
if ! command -v python >/dev/null 2>&1; then python() { python3 "$@"; }; fi
cd "$CODE"
mkdir -p "$MODELS" "$OUTS" "$ITEMS" "$STAGE_DIR"/{predictions,logits,test/predictions,test/logits} "$RES"/{metrics,logs,predictions,logits,run_summaries}
say() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
say "stage $STAGE ${ARG} started"

json_get() {  # json_get <file> <python expression on the dict d>
  python -c "import json; d = json.load(open('$1')); print($2)"
}

laya_items() {  # laya_items <out> <max_len> <questions> <records...>
  local out="$1" max_len="$2" questions="$3"
  shift 3
  python src/train/build_items.py --records "$@" --questions "$questions" --model-dir "$MODELS/reader_v1_laya" \
    --out "$out" --max-len "$max_len" --head-max-len 192
}

qwen_items() {  # qwen_items <out> <records...>
  local out="$1"
  shift
  python src/train/qwen_items.py --records "$@" --tokenizer-dir "$MODELS/qwen3_4b_base" --out "$out" --max-len 768
}

wait_for_labeler() {  # waits for labeler.json, at most LABELER_WAIT_SECONDS
  local waited=0
  while [ ! -f "$RES/labeler.json" ]; do
    if [ "$waited" -ge "$LABELER_WAIT_SECONDS" ]; then say "labeler.json did not appear within 3 hours; stopping"; exit 1; fi
    sleep 30
    waited=$((waited + 30))
  done
}

phase2_built() {
  json_get "$RES/labeler.json" "d.get('phase2_built') is True"
}

wait_for_phase2_items() {  # prints True when both arms' phase 2 items are there, False when labeler.json comes to say
  # that phase 2 was not built (a failure after labeler.json was first written). Both arms wait for arm B's summary
  # file, the last thing the labeler stage writes, so that neither arm trains phase 2 unless both item builds succeeded.
  local last="$ITEMS/b_phase2.summary.json" waited=0
  while [ ! -f "$last" ]; do
    if [ "$(phase2_built)" != "True" ]; then echo False; return 0; fi
    if [ "$waited" -ge "$LABELER_WAIT_SECONDS" ]; then say "$last did not appear within 3 hours; stopping" >&2; exit 1; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo True
}

phase_done() {  # phase_done <system> <weights file>: the phase finished (its metrics JSON is written last)
  [ -f "$RES/metrics/$1_metrics.json" ] && [ -f "$2" ]
}

copy_validation() {  # copy_validation <system>: raw validation predictions and logits from the staging folder
  cp "$STAGE_DIR"/predictions/"$1"-raw_*_validation.csv "$RES/predictions/"
  cp "$STAGE_DIR"/logits/"$1"_*_validation_logits.csv "$RES/logits/"
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

train_b() {  # train_b <system> <init or -> <train items> <val items> <extra flags...>
  local system="$1" init="$2" train="$3" val="$4"
  shift 4
  if phase_done "$system" "$MODELS/$system/adapter.safetensors"; then say "arm B: $system already finished; not trained again"; return 0; fi
  local ckpt=""
  if [ "$(json_get "$RES/pilot.json" "d.get('gradient_checkpointing_on') is True")" = "True" ]; then ckpt="--gradient-checkpointing"; fi
  for done_metrics in "$RES"/metrics/s7_b[12]_metrics.json; do  # a phase that ran out of memory keeps it on for the next
    if [ -f "$done_metrics" ] && [ "$(json_get "$done_metrics" "d.get('gradient_checkpointing') is True")" = "True" ]; then
      ckpt="--gradient-checkpointing"
    fi
  done
  local init_flag=""
  if [ "$init" != "-" ]; then init_flag="--init $MODELS/$init"; fi
  say "arm B: $system from ${init}; ${ckpt:-no gradient checkpointing}"
  local rc=0
  # shellcheck disable=SC2086,SC2068
  python src/train/train_qwen_head.py train --base-dir "$MODELS/qwen3_4b_base" $init_flag --train "$train" --val "$val" \
    --out "$MODELS/$system" --system-name "$system" --epochs 1 --micro-batch 8 --grad-accum 4 --max-len 768 \
    --metrics "$RES/metrics/${system}_metrics.json" --log "$RES/logs/${system}_train_log.csv" $ckpt $@ \
    2>&1 | tee "$LOGS/${system}.out" || rc=$?
  if [ "$rc" != 0 ]; then
    if [ -z "$ckpt" ] && grep -qi "out of memory" "$LOGS/${system}.out"; then
      say "arm B: $system ran out of memory without gradient checkpointing; restarting the phase from its start with it (kept on for later phases)"
      # shellcheck disable=SC2086,SC2068
      python src/train/train_qwen_head.py train --base-dir "$MODELS/qwen3_4b_base" $init_flag --train "$train" --val "$val" \
        --out "$MODELS/$system" --system-name "$system" --epochs 1 --micro-batch 8 --grad-accum 4 --max-len 768 \
        --metrics "$RES/metrics/${system}_metrics.json" --log "$RES/logs/${system}_train_log.csv" --gradient-checkpointing $@
    else
      say "arm B: $system failed (exit $rc)"
      exit "$rc"
    fi
  fi
  say "arm B: $system done"
}

check_rows() {  # check_rows <file> <rows>
  local n
  n=$(($(wc -l < "$1") - 1))
  if [ "$n" != "$2" ]; then say "$1 has $n rows, expected $2; stopping"; exit 1; fi
}

case "$STAGE" in
  prepare)
    # the uploads first, so that a missing file stops the stage before the installs and downloads
    python - "$RES/records_manifest.json" <<'PY'
import sys
from pathlib import Path
need = [Path(sys.argv[1]), Path("COMMIT.txt"), Path("results/public_v0/checksums.json"), Path("results/fiorillo_v0_2/selection.json"),
        Path("results/public_v0/predictions/arm_laya_pubmedqa_answer_validation.csv")]
missing = [str(p) for p in need if not p.exists()]
if missing:
    sys.exit(f"not uploaded yet: {missing}")
print(f"uploads present: {len(need)} files")
PY
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
    pip install -q "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "safetensors==0.8.0" "pyyaml==6.0.3" \
      "pyarrow==25.0.1" "scipy==1.17.1" "numpy==2.4.6" "accelerate==1.15.0" "laya==0.3.20"
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
    python src/tasks/pubmedqa_phases.py --processes 16 --max-seconds 100000 --manifest "$BASE/records_manifest_pod.json"
    python - "$RES/records_manifest.json" "$REC7" <<'PY'
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
    assert got == entry["sha256"], f"{name}: SHA-256 {got} differs from the Mac's records_manifest.json"
print(f"session 7's records built on the pod: {len(manifest['files'])} files, each identical to the Mac's (records_manifest.json)")
PY
    qwen_items "$ITEMS/b_phase1.pt" "$REC7/pubmedqa_phase1_train.jsonl"
    qwen_items "$ITEMS/b_val12.pt" data/processed/public/pubmedqa_answer_validation.jsonl
    qwen_items "$ITEMS/b_phase3.pt" data/processed/public/readmit_window_train.jsonl data/processed/public/pubmedqa_answer_train.jsonl
    qwen_items "$ITEMS/b_val3.pt" data/processed/public/readmit_window_validation.jsonl data/processed/public/pubmedqa_answer_validation.jsonl
    python - "$ITEMS" <<'PY'
import json, sys
from pathlib import Path
out = {p.name.replace(".summary.json", ""): json.load(open(p)) for p in sorted(Path(sys.argv[1]).glob("*.summary.json"))}
print("items built (records, items, skipped, cut):")
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
    laya_items "$ITEMS/lab_a_train.pt" 256 src/questions/session7.yaml "$REC7/labeler_pqa_a.jsonl"
    laya_items "$ITEMS/lab_b_train.pt" 256 src/questions/session7.yaml "$REC7/labeler_pqa_l_train.jsonl"
    laya_items "$ITEMS/lab_val.pt" 256 src/questions/session7.yaml "$REC7/labeler_validation.jsonl"
    laya_items "$ITEMS/a_phase1.pt" 768 src/questions/public_v0.yaml "$REC7/pubmedqa_phase1_train.jsonl"
    laya_items "$ITEMS/a_val12.pt" 768 src/questions/public_v0.yaml data/processed/public/pubmedqa_answer_validation.jsonl
    laya_items "$ITEMS/a_phase3.pt" 768 src/questions/public_v0.yaml data/processed/public/readmit_window_train.jsonl data/processed/public/pubmedqa_answer_train.jsonl
    laya_items "$ITEMS/a_val3.pt" 768 src/questions/public_v0.yaml data/processed/public/readmit_window_validation.jsonl data/processed/public/pubmedqa_answer_validation.jsonl
    python - "$ITEMS" <<'PY'
import json, sys
from pathlib import Path
out = {p.name.replace(".summary.json", ""): json.load(open(p)) for p in sorted(Path(sys.argv[1]).glob("*.summary.json"))}
print("items built (records, items, skipped, cut):")
print(json.dumps({k: {x: v.get(x) for x in ("records", "items", "skipped", "truncated")} for k, v in out.items()}, indent=1))
PY
    ;;
  correct_phase3)
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
    laya_items "$ITEMS/a_phase3c.pt" 768 src/questions/public_v0.yaml data/processed/public/readmit_window_train.jsonl \
      "$REC7/pubmedqa_pqa_l_train.jsonl" "$REC7/pubmedqa_phase2_train.jsonl"
    qwen_items "$ITEMS/b_phase3c.pt" data/processed/public/readmit_window_train.jsonl "$REC7/pubmedqa_pqa_l_train.jsonl" \
      "$REC7/pubmedqa_phase2_train.jsonl"
    for x in a b; do
      if [ ! -f "$ITEMS/${x}_phase3_orig.pt" ]; then
        mv "$ITEMS/${x}_phase3.pt" "$ITEMS/${x}_phase3_orig.pt"
        mv "$ITEMS/${x}_phase3.summary.json" "$ITEMS/${x}_phase3_orig.summary.json"
      fi
      mv "$ITEMS/${x}_phase3c.pt" "$ITEMS/${x}_phase3.pt"
      mv "$ITEMS/${x}_phase3c.summary.json" "$ITEMS/${x}_phase3.summary.json"
      say "${x}_phase3.pt now holds the corrected phase 3 items: $(json_get "$ITEMS/${x}_phase3.summary.json" "d['items']")"
    done
    if [ -d "$MODELS/s7_a3" ] && [ ! -d "$MODELS/s7_a3_orig" ]; then
      mv "$MODELS/s7_a3" "$MODELS/s7_a3_orig"
      mv "$RES/metrics/s7_a3_metrics.json" "$RES/metrics/s7_a3_orig_metrics.json"
      mv "$RES/logs/s7_a3_train_log.csv" "$RES/logs/s7_a3_orig_train_log.csv"
      for f in "$RES"/predictions/s7_a3-raw_*_validation.csv "$RES"/logits/s7_a3_*_validation_logits.csv \
               "$STAGE_DIR"/predictions/s7_a3*_validation.csv "$STAGE_DIR"/logits/s7_a3_*_validation_logits.csv; do
        if [ -f "$f" ]; then mv "$f" "$(dirname "$f")/$(basename "$f" | sed 's/^s7_a3/s7_a3_orig/')"; fi
      done
      say "the first s7_a3 is kept as s7_a3_orig (model folder, metrics, log, validation predictions and logits)"
    fi
    ;;
  pilot)
    python src/train/train_qwen_head.py pilot --base-dir "$MODELS/qwen3_4b_base" --train "$ITEMS/b_phase1.pt" --val "$ITEMS/b_val12.pt" \
      --max-steps 200 --micro-batch 8 --grad-accum 4 --max-len 768 --out-json "$RES/pilot.json"
    ;;
  labeler)
    not_built() {  # not_built <reason>: labeler.json records phase 2 as NOT BUILT, so that neither arm waits; then stop
      python src/eval/label_pqa_u.py --results "$RES" --not-built "$1 (logs/pod/labeler.log)"
      exit 1
    }
    python src/train/train_laya.py --model-dir "$MODELS/reader_v1_laya" --train "$ITEMS/lab_a_train.pt" --val "$ITEMS/lab_val.pt" \
      --test "$ITEMS/not_built.pt" --no-test --out "$MODELS/s7_lab_a" --system-name s7_lab_a --device cuda --epochs 1 --micro-batch 32 \
      --grad-accum 1 --max-len 256 --reinit-head --freeze-encoder-steps 250 --keep last --batching by-question --calibration-mode separate \
      --autocast float16 --metrics "$RES/metrics/s7_lab_a_metrics.json" --log "$RES/logs/s7_lab_a_train_log.csv" \
      || not_built "s7_lab_a stopped with an error"
    python src/train/train_laya.py --model-dir "$MODELS/s7_lab_a" --train "$ITEMS/lab_b_train.pt" --val "$ITEMS/lab_val.pt" \
      --test "$ITEMS/not_built.pt" --no-test --out "$MODELS/s7_lab_b" --system-name s7_lab_b --device cuda --epochs 3 --micro-batch 32 \
      --grad-accum 1 --max-len 256 --keep last --batching by-question --calibration-mode separate \
      --autocast float16 --metrics "$RES/metrics/s7_lab_b_metrics.json" --log "$RES/logs/s7_lab_b_train_log.csv" \
      || not_built "s7_lab_b stopped with an error"
    python src/eval/label_pqa_u.py --model-dir "$MODELS/s7_lab_b" --records-dir "$REC7" --results "$RES" \
      --v0-1-validation results/public_v0/predictions/arm_laya_pubmedqa_answer_validation.csv \
      || not_built "label_pqa_u.py stopped with an error"
    if [ "$(phase2_built)" = "True" ]; then
      laya_items "$ITEMS/a_phase2.pt" 768 src/questions/public_v0.yaml "$REC7/pubmedqa_phase2_train.jsonl" \
        || not_built "phase 2's items for arm A could not be built"
      qwen_items "$ITEMS/b_phase2.pt" "$REC7/pubmedqa_phase2_train.jsonl" \
        || not_built "phase 2's items for arm B could not be built"
      say "phase 2's items built for both arms"
    else
      say "$(json_get "$RES/labeler.json" "d.get('not_built')")"
    fi
    ;;
  arm)
    case "$ARG" in
      a)
        train_a s7_a1 "$MODELS/reader_v1_laya" "$ITEMS/a_phase1.pt" "$ITEMS/a_val12.pt" --reinit-head --freeze-encoder-steps 250 --keep last --eval-every 0
        wait_for_labeler
        prev=s7_a1
        built=False
        if [ "$(phase2_built)" = "True" ]; then built=$(wait_for_phase2_items); fi
        if [ "$built" = "True" ]; then
          train_a s7_a2 "$MODELS/s7_a1" "$ITEMS/a_phase2.pt" "$ITEMS/a_val12.pt" --keep last --eval-every 0
          prev=s7_a2
        else
          say "phase 2 not built; s7_a3 starts from s7_a1"
        fi
        train_a s7_a3 "$MODELS/$prev" "$ITEMS/a_phase3.pt" "$ITEMS/a_val3.pt" --keep best --eval-every 250 \
          --predictions-dir "$STAGE_DIR/predictions" --logits-dir "$STAGE_DIR/logits"
        copy_validation s7_a3
        ;;
      b)
        if [ ! -f "$RES/pilot.json" ]; then say "pilot.json is missing; run the pilot first"; exit 1; fi
        train_b s7_b1 - "$ITEMS/b_phase1.pt" "$ITEMS/b_val12.pt" --freeze-steps 250 --keep last --eval-every 0
        wait_for_labeler
        prev=s7_b1
        built=False
        if [ "$(phase2_built)" = "True" ]; then built=$(wait_for_phase2_items); fi
        if [ "$built" = "True" ]; then
          train_b s7_b2 s7_b1 "$ITEMS/b_phase2.pt" "$ITEMS/b_val12.pt" --keep last --eval-every 0
          prev=s7_b2
        else
          say "phase 2 not built; s7_b3 starts from s7_b1"
        fi
        train_b s7_b3 "$prev" "$ITEMS/b_phase3.pt" "$ITEMS/b_val3.pt" --keep best --eval-every 250 \
          --predictions-dir "$STAGE_DIR/predictions" --logits-dir "$STAGE_DIR/logits"
        copy_validation s7_b3
        ;;
      *) echo "arm takes a or b"; exit 1 ;;
    esac
    ;;
  select)
    python src/eval/select_session7.py select --results "$RES"
    for system in s7_a3 s7_b3; do
      if [ -f "$RES/logits/${system}_pubmedqa_answer_validation_logits.csv" ]; then
        python src/eval/calibrate_session7.py fit --system "$system" --logits-dir "$RES/logits" --out-json "$RES/calibration_s7.json" \
          --predictions-dir "$RES/predictions"
      fi
    done
    candidate=$(json_get "$RES/selection.json" "d['candidate_rule']['candidate'] or ''")
    if [ -n "$candidate" ]; then
      python src/eval/ece_noise_floor.py --predictions "$RES/predictions/${candidate}_pubmedqa_answer_validation.csv" \
        --records data/processed/public/pubmedqa_answer_validation.jsonl --n 500 --draws 5000 --out "$RES/ece_limit_pubmedqa_answer.json"
      python src/eval/ece_noise_floor.py --predictions "$RES/predictions/${candidate}_readmit_window_validation.csv" \
        --records data/processed/public/readmit_window_validation.jsonl --n 6998 --draws 5000 --out "$RES/ece_limit_readmit_window.json"
    else
      say "$(json_get "$RES/selection.json" "d['candidate_rule']['not_built']")"
    fi
    ;;
  freeze)
    python src/eval/select_session7.py freeze --results "$RES" --staging "$STAGE_DIR"
    ;;
  testpreds)
    if [ ! -f "$RES/release_candidate.json" ]; then echo "Stop: results/fiorillo_v0_3/release_candidate.json does not exist"; exit 1; fi
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
    complete() {  # complete <system>: its two raw test files and two logits files are there with every row
      local f
      for f in "$STAGE_DIR/test/predictions/$1-raw_pubmedqa_answer_test.csv:500" "$STAGE_DIR/test/predictions/$1-raw_readmit_window_test.csv:6998" \
               "$STAGE_DIR/test/logits/$1_pubmedqa_answer_test_logits.csv:500" "$STAGE_DIR/test/logits/$1_readmit_window_test_logits.csv:6998"; do
        [ -f "${f%:*}" ] || return 1
        [ "$(($(wc -l < "${f%:*}") - 1))" = "${f##*:}" ] || return 1
      done
    }
    if [ -f "$MODELS/s7_a3/model.safetensors" ] && ! complete s7_a3; then
      python src/eval/run_laya_model.py --model-dir "$MODELS/s7_a3" --system s7_a3 --records-dir data/processed/public \
        --qids pubmedqa_answer readmit_window --out "$STAGE_DIR/test/predictions" --logits-dir "$STAGE_DIR/test/logits" \
        --max-len 768 --head-max-len 192 --device cuda --calibration "$MODELS/s7_a3/calibration.json" --splits test
    fi
    if [ -f "$MODELS/s7_b3/adapter.safetensors" ] && ! complete s7_b3; then
      python src/train/train_qwen_head.py predict --base-dir "$MODELS/qwen3_4b_base" --phase-dir "$MODELS/s7_b3" --system s7_b3 \
        --records-dir data/processed/public --qids pubmedqa_answer readmit_window --splits test \
        --out "$STAGE_DIR/test/predictions" --logits-dir "$STAGE_DIR/test/logits"
    fi
    for system in s7_a3 s7_b3; do
      if [ -f "$STAGE_DIR/test/predictions/${system}-raw_pubmedqa_answer_test.csv" ]; then
        check_rows "$STAGE_DIR/test/predictions/${system}-raw_pubmedqa_answer_test.csv" 500
        check_rows "$STAGE_DIR/test/predictions/${system}-raw_readmit_window_test.csv" 6998
        check_rows "$STAGE_DIR/test/logits/${system}_pubmedqa_answer_test_logits.csv" 500
        check_rows "$STAGE_DIR/test/logits/${system}_readmit_window_test_logits.csv" 6998
        cp "$STAGE_DIR"/test/predictions/"$system"-raw_*_test.csv "$RES/predictions/"
        cp "$STAGE_DIR"/test/logits/"$system"_*_test_logits.csv "$RES/logits/"
        if [ -f "$STAGE_DIR/test/${system}_run_summary.json" ]; then cp "$STAGE_DIR/test/${system}_run_summary.json" "$RES/run_summaries/"; fi
      fi
    done
    say "test predictions written for s7_a3 and s7_b3"
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
with zipfile.ZipFile(outs / "fiorillo_v0_3_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, code / "results" / "fiorillo_v0_3", "results/fiorillo_v0_3")
for arm, system in (("a", "s7_a3"), ("b", "s7_b3")):
    if (models / system).exists():
        with zipfile.ZipFile(outs / f"fiorillo_v0_3_arm_{arm}.zip", "w", zipfile.ZIP_STORED) as z:
            add_tree(z, models / system, system)
with zipfile.ZipFile(outs / "fiorillo_v0_3_logs.zip", "w", zipfile.ZIP_DEFLATED) as z:
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
say "stage $STAGE ${ARG} finished"

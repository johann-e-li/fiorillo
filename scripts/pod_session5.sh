#!/usr/bin/env bash
# Session 5 pod driver (docs/prompt_session5.md, step 5): the only way the pod is driven. It runs on the Runpod pod
# from /root/eddm_s5/code (this repository's code, uploaded with src/tools/upload_to_pod.py, which also starts these
# subcommands). Every stage detaches itself under nohup, so a closed connection does not stop it, and writes its log to
# /root/eddm_s5/logs/<stage>.log (<stage>_<argument>.log for decision, mlm and select). Stages:
#   prepare            step 8: GPU check; installs; Laya, ModernBERT-large and BioClinical ModernBERT-large downloaded
#                      at the fixed revisions; session 4's records rebuilt and checked against their SHA-256 values;
#                      every uploaded part checked against its manifest; the decision-layer items; the MLM start
#                      (build_mlm_init.py); the abstracts tokenized first; then it starts stage text in the background
#   text               (started by prepare) the CPU text work: task text, PMC candidates, inventory and fetch for
#                      case_reports and pmc, and the tokenization of every other source; manifests copied back
#   timing             step 8: the timed run, 200 steps of mix A from the MLM start, nothing kept; on running out of
#                      memory, again with micro-batch 16 and accumulation 16 (recorded, and used by later MLM runs)
#   decision <system>  steps 9 to 12: one decision-layer run with session 4's method and --no-test; then the
#                      out-of-fold log losses (calibration_family_cv.py) for its validation predictions
#   select <rule>      a selection rule: maybe, reader, task or head (src/eval/select_session5.py)
#   mlm <A|B|C>        step 10: a 500-million-token comparison run from the MLM start
#   task               step 11: the task round from the winning mix's final checkpoint
#   long               step 12: the long run from the winner's resume.pt to 2,000 million tokens, then the task round
#                      if the task rule kept it; the result is reader v1
#   final              step 12: the final decision-layer runs, the head rule, and the candidate copied to
#                      /root/eddm_s5/models/fiorillo_v0_2_candidate/ with its calibration.json
#   freeze             step 13: the ECE limits (ece_noise_floor.py, n 500 and 6,998) and release_candidate.json,
#                      before any test prediction exists
#   testpreds          step 13: refuses to start unless release_candidate.json exists; first writes
#                      testpreds_log.json (that file's SHA-256 and this stage's start time in UTC); then test
#                      predictions for every decision-layer run with run_laya_model.py --calibration
#   package            step 13: fiorillo_v0_2_results.zip, fiorillo_v0_2_model.zip and reader_v1.zip (if reader v1
#                      exists) in /root/eddm_s5/outputs/, with their SHA-256 in SHA256SUMS.txt
# Usage on the pod: bash /root/eddm_s5/code/scripts/pod_session5.sh <stage> [argument]
BASE=/root/eddm_s5
CODE=$BASE/code
TEXT=$BASE/text
BLOCKS=$BASE/blocks
MODELS=$BASE/models
LOGS=$BASE/logs
OUTS=$BASE/outputs
ITEMS=$BASE/items
RES5=$CODE/results/fiorillo_v0_2
RESR=$CODE/results/reader_v1
TOK=$MODELS/laya/tokenizer/tokenizer.json
LAYA_REV=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
MB_REV=45bb4654a4d5aaff24dd11d4781fa46d39bf8c13
BIO_REV=c684f470c2ee60531d8c47c32187dba746c84201

STAGE="${1:-}"
ARG="${2:-}"
if [ -z "$STAGE" ]; then sed -n '2,32p' "$0"; exit 1; fi
if [ -z "${EDDM_DETACHED:-}" ]; then
  mkdir -p "$LOGS"
  NAME="$STAGE${ARG:+_$ARG}"
  EDDM_DETACHED=1 nohup bash "$0" "$STAGE" "$ARG" > "$LOGS/$NAME.log" 2>&1 < /dev/null &
  echo "started $NAME (process $!); log $LOGS/$NAME.log"
  exit 0
fi
set -euo pipefail
# The image's system Python is marked externally managed (PEP 668), so pip needs this to install into it, as the
# packages were installed into it in session 4; and `python` is python3 where no `python` command exists.
export PIP_BREAK_SYSTEM_PACKAGES=1
if ! command -v python >/dev/null 2>&1; then python() { python3 "$@"; }; fi
cd "$CODE"
mkdir -p "$TEXT" "$BLOCKS" "$MODELS" "$OUTS" "$ITEMS" "$RES5"/{metrics,predictions,oof,logs} "$RESR"/{logs,text_manifests,block_counts}
say() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
say "stage $STAGE ${ARG} started"

selection() {  # selection() <python expression on the selection dict s>
  python -c "import json; s = json.load(open('$RES5/selection.json')); print($1)"
}

tokenize() {  # tokenize <source>
  python src/corpus/tokenize_blocks.py --source "$1" --parts "$TEXT/$1" --out "$BLOCKS" --tokenizer "$TOK"
  python - "$BLOCKS/$1/counts.json" "$RESR/block_counts/$1.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.pop("heldout_document_indices", None)
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
}

copy_manifest() {  # copy_manifest <source>: the manifest without document text or lists, split under 1 MB
  python - "$TEXT/$1/manifest.json" "$RESR/text_manifests/$1_manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
docs = m.pop("document_list", None)
json.dump(m, open(sys.argv[2] + ".json", "w"), indent=1)
if docs:
    fields = m.get("document_fields", [])
    per, n = 12000, 0
    for start in range(0, len(docs), per):
        json.dump({"fields": fields, "documents": docs[start:start + per]}, open(f"{sys.argv[2]}_documents_{n:02d}.json", "w"))
        n += 1
PY
}

assemble() {  # assemble <reader folder> <name> <reader description>: a Laya-format folder with that reader
  local out="$MODELS/assembled/$2"
  if [ ! -f "$out/model.safetensors" ]; then
    python src/train/assemble_reader.py --laya-dir "$MODELS/laya" --reader-dir "$1" --out "$out" \
      --laya-revision "convaiinnovations/laya@$LAYA_REV" --reader-revision "$3" >&2
  fi
  echo "$out"
}

reader_v1() {  # the final MLM checkpoint that is reader v1
  cat "$MODELS/reader_v1_path.txt"
}

decision() {  # decision <system>
  local system="$1" head="--reinit-head --freeze-encoder-steps 250" maybe reader
  if [ "$system" = "s5_laya" ]; then maybe=none; elif [ "$system" = "s5_laya_w" ]; then maybe=auto
  else maybe=$(selection "s['maybe_rule']['recipe'].split()[-1]"); fi
  case "$system" in
    s5_laya|s5_laya_w) reader="$MODELS/laya" ;;
    s5_mix_a|s5_mix_b|s5_mix_c) local mix="${system#s5_mix_}"
      reader=$(assemble "$MODELS/reader_mix_$mix/final" "reader_mix_$mix" "reader_mix_$mix (session 5, 500 million tokens)") ;;
    s5_bioclinical) reader=$(assemble "$MODELS/bioclinical" bioclinical "thomas-sounack/BioClinical-ModernBERT-large@$BIO_REV") ;;
    s5_task) reader=$(assemble "$MODELS/reader_task/final" reader_task "the winning mix plus the task round") ;;
    s5_final_fresh_head) reader=$(assemble "$(reader_v1)" reader_v1 "reader v1") ;;
    s5_final_laya_head) head="--freeze-encoder-steps 250"
      if [ -f "$MODELS/reader_v1_path.txt" ]; then reader=$(assemble "$(reader_v1)" reader_v1 "reader v1"); else reader="$MODELS/laya"; fi ;;
    *) echo "unknown system $system"; exit 1 ;;
  esac
  say "decision $system: reader $reader; head flags: $head; maybe weight: $maybe"
  # shellcheck disable=SC2086
  python src/train/train_laya.py --train "$ITEMS/train.pt" --val "$ITEMS/validation.pt" --test "$ITEMS/not_built.pt" --no-test \
    --model-dir "$reader" --out "$MODELS/decision/$system" --device cuda --epochs 1 --micro-batch 8 --grad-accum 4 \
    --max-len 768 --eval-every 250 $head --maybe-weight "$maybe" --batching by-question --calibration-mode separate \
    --predictions-dir "$RES5/predictions" --system-name "$system" --metrics "$RES5/metrics/${system}_metrics.json" \
    --log "$RES5/logs/${system}_train_log.csv"
  for qid in pubmedqa_answer readmit_window; do
    python src/eval/calibration_family_cv.py --predictions "$RES5/predictions/${system}-raw_${qid}_validation.csv" \
      --records "$CODE/data/processed/public/${qid}_validation.jsonl" --folds 5 --repeats 20 \
      --out "$RES5/oof/${system}_${qid}_oof.json" > /dev/null
  done
  say "decision $system done"
}

mlm_batch() {  # the micro-batch and accumulation the timed run settled
  if [ -f "$BASE/mlm_batch.env" ]; then cat "$BASE/mlm_batch.env"; else echo "--micro-batch 32 --grad-accum 8"; fi
}

copy_mlm_results() {  # copy_mlm_results <model folder> <name>
  mkdir -p "$RESR/$2"
  for f in heldout_losses.json throughput.json run.json; do
    if [ -f "$1/$f" ]; then cp "$1/$f" "$RESR/$2/$f"; fi
  done
}

case "$STAGE" in
  prepare)
    nvidia-smi
    python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
    pip install -q "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "safetensors==0.8.0" "pyyaml==6.0.3" \
      "pyarrow==25.0.1" "scipy==1.17.1" "numpy==2.4.6" "accelerate==1.15.0" "laya==0.3.20"
    python -c "import transformers, tokenizers, huggingface_hub, laya, numpy, torch; print('transformers', transformers.__version__, '| tokenizers', tokenizers.__version__, '| huggingface_hub', huggingface_hub.__version__, '| laya', laya.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", revision="$LAYA_REV", local_dir="$MODELS/laya",
                  allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
snapshot_download("answerdotai/ModernBERT-large", revision="$MB_REV", local_dir="$MODELS/modernbert-large",
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
snapshot_download("thomas-sounack/BioClinical-ModernBERT-large", revision="$BIO_REV", local_dir="$MODELS/bioclinical",
                  allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
print("downloads done")
PY
    if [ ! -f "$BASE/expected_checksums.json" ]; then cp results/public_v0/checksums.json "$BASE/expected_checksums.json"; fi
    python src/tasks/pubmedqa.py
    python src/tasks/uci_readmission.py
    python - "$BASE/expected_checksums.json" data/processed/public/checksums.json <<'PY'
import json, sys
expected, got = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = {k: (got.get(k), v) for k, v in expected.items() if got.get(k) != v}
if bad:
    sys.exit(f"record checksums differ from session 4: {bad}")
print(f"session 4's records rebuilt: {len(expected)} files, SHA-256 identical")
PY
    python - "$TEXT" "$CODE/results/reader_v1/text_manifests/pubmedqa_exclusion.json" "$CODE/data/processed/corpus/session5" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
text, excl, s5 = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
for name, digest in excl["outputs_sha256"].items():
    assert sha(s5 / name) == digest, f"{name}: SHA-256 differs from pubmedqa_exclusion.json"
for source in ("abstracts", "web"):
    m = text / source / "manifest.json"
    if not m.exists():
        print(f"{source}: not uploaded")
        continue
    parts = json.load(open(m))["parts"]
    for p in parts:
        assert sha(text / source / p["file"]) == p["sha256"], f"{source}/{p['file']}: SHA-256 differs from its manifest"
    print(f"{source}: {len(parts)} parts checked against the manifest")
PY
    for split in train validation; do
      python src/train/build_items.py --records "data/processed/public/readmit_window_$split.jsonl" \
        "data/processed/public/pubmedqa_answer_$split.jsonl" --model-dir "$MODELS/laya" --out "$ITEMS/$split.pt" \
        --max-len 768 --head-max-len 192
    done
    python src/train/build_mlm_init.py --laya-dir "$MODELS/laya" --modernbert-dir "$MODELS/modernbert-large" \
      --out "$MODELS/mlm_init" --laya-revision "convaiinnovations/laya@$LAYA_REV" \
      --modernbert-revision "answerdotai/ModernBERT-large@$MB_REV"
    cp "$MODELS/mlm_init/init_report.json" "$RESR/mlm_init_report.json"
    copy_manifest abstracts
    tokenize abstracts
    say "abstracts tokenized; the timed run can start. Starting the CPU text work (stage text)."
    env -u EDDM_DETACHED bash "$0" text
    ;;
  text)
    python src/corpus/build_task_text.py --out "$TEXT/task" --tokenizer "$TOK"
    copy_manifest task
    tokenize task
    if [ -f "$TEXT/web/manifest.json" ]; then copy_manifest web; tokenize web; else say "web: not uploaded (mix C NOT BUILT)"; fi
    python src/corpus/fetch_pmc.py candidates
    python src/corpus/fetch_pmc.py inventory
    python src/corpus/fetch_pmc.py fetch case_reports --tokenizer "$TOK"
    python src/corpus/fetch_pmc.py fetch pmc --tokenizer "$TOK"
    cp "$TEXT/pmc_work/candidates.json" "$RESR/text_manifests/pmc_candidates.json"
    for source in case_reports pmc; do copy_manifest "$source"; tokenize "$source"; done
    say "text work complete: every source tokenized"
    ;;
  timing)
    rc=0
    python src/train/pretrain_reader.py --init "$MODELS/mlm_init" --mix A --out "$MODELS/timing" --log "$RESR/logs/timing.csv" \
      --blocks "$BLOCKS" --timing-steps 200 || rc=$?
    if [ "$rc" = 3 ]; then
      say "out of memory at micro-batch 32; again with micro-batch 16 and accumulation 16"
      cp "$MODELS/timing/timing.json" "$RESR/timing_oom_micro32.json"
      python src/train/pretrain_reader.py --init "$MODELS/mlm_init" --mix A --out "$MODELS/timing" --log "$RESR/logs/timing.csv" \
        --blocks "$BLOCKS" --timing-steps 200 --micro-batch 16 --grad-accum 16
      echo "--micro-batch 16 --grad-accum 16" > "$BASE/mlm_batch.env"
    elif [ "$rc" != 0 ]; then exit "$rc"
    else echo "--micro-batch 32 --grad-accum 8" > "$BASE/mlm_batch.env"; fi
    cp "$MODELS/timing/timing.json" "$RESR/timing.json"
    cat "$RESR/timing.json"
    ;;
  decision)
    decision "$ARG"
    ;;
  select)
    python src/eval/select_session5.py "$ARG" --metrics-dir "$RES5/metrics" --oof-dir "$RES5/oof" --selection "$RES5/selection.json"
    ;;
  mlm)
    mix=$(echo "$ARG" | tr '[:upper:]' '[:lower:]')
    # shellcheck disable=SC2046
    python src/train/pretrain_reader.py --init "$MODELS/mlm_init" --mix "$ARG" --tokens 500000000 --out "$MODELS/reader_mix_$mix" \
      --log "$RESR/logs/mlm_mix_$mix.csv" --blocks "$BLOCKS" $(mlm_batch)
    copy_mlm_results "$MODELS/reader_mix_$mix" "mlm_mix_$mix"
    ;;
  task)
    if [ "$(selection "s['reader_rule']['long_run']")" != "True" ]; then echo "Stop: the reader rule allowed no long run, so there is no task round"; exit 1; fi
    winner=$(selection "s['reader_rule']['winner']" | tr '[:upper:]' '[:lower:]')
    # shellcheck disable=SC2046
    python src/train/pretrain_reader.py --init "$MODELS/reader_mix_$winner/final" --mix T --out "$MODELS/reader_task" \
      --log "$RESR/logs/mlm_task.csv" --blocks "$BLOCKS" $(mlm_batch)
    copy_mlm_results "$MODELS/reader_task" mlm_task
    ;;
  long)
    if [ "$(selection "s['reader_rule']['long_run']")" != "True" ]; then echo "Stop: NOT BUILT: long run, no mix beat Laya's reader on validation"; exit 1; fi
    WINNER=$(selection "s['reader_rule']['winner']")
    winner=$(echo "$WINNER" | tr '[:upper:]' '[:lower:]')
    # shellcheck disable=SC2046
    python src/train/pretrain_reader.py --resume "$MODELS/reader_mix_$winner/decay_start/resume.pt" --mix "$WINNER" \
      --total-tokens 2000000000 --out "$MODELS/reader_long" --log "$RESR/logs/mlm_long.csv" --blocks "$BLOCKS" $(mlm_batch)
    copy_mlm_results "$MODELS/reader_long" mlm_long
    if [ "$(selection "s.get('task_rule', {}).get('task_round_kept')")" = "True" ]; then
      # shellcheck disable=SC2046
      python src/train/pretrain_reader.py --init "$MODELS/reader_long/final" --mix T --out "$MODELS/reader_long_task" \
        --log "$RESR/logs/mlm_long_task.csv" --blocks "$BLOCKS" $(mlm_batch)
      copy_mlm_results "$MODELS/reader_long_task" mlm_long_task
      echo "$MODELS/reader_long_task/final" > "$MODELS/reader_v1_path.txt"
    else
      echo "$MODELS/reader_long/final" > "$MODELS/reader_v1_path.txt"
    fi
    python -c "import json; json.dump({'reader_v1': open('$MODELS/reader_v1_path.txt').read().strip(), 'winner': '$WINNER'}, open('$RESR/reader_v1.json', 'w'), indent=1)"
    say "reader v1: $(reader_v1)"
    ;;
  final)
    if [ -f "$MODELS/reader_v1_path.txt" ]; then decision s5_final_fresh_head; fi
    decision s5_final_laya_head
    python src/eval/select_session5.py head --metrics-dir "$RES5/metrics" --oof-dir "$RES5/oof" --selection "$RES5/selection.json"
    candidate=$(selection "s['head_rule']['candidate']")
    if [ -e "$MODELS/fiorillo_v0_2_candidate" ]; then echo "Stop: $MODELS/fiorillo_v0_2_candidate already exists"; exit 1; fi
    cp -r "$MODELS/decision/$candidate" "$MODELS/fiorillo_v0_2_candidate"
    say "candidate: $candidate, copied to $MODELS/fiorillo_v0_2_candidate"
    ;;
  freeze)
    if ls "$RES5"/predictions/*_test.csv > /dev/null 2>&1; then echo "Stop: a test prediction already exists"; exit 1; fi
    candidate=$(selection "s['head_rule']['candidate']")
    python src/eval/ece_noise_floor.py --predictions "$RES5/predictions/${candidate}_pubmedqa_answer_validation.csv" \
      --records data/processed/public/pubmedqa_answer_validation.jsonl --n 500 --draws 5000 --out "$RES5/ece_limit_pubmedqa_answer.json"
    python src/eval/ece_noise_floor.py --predictions "$RES5/predictions/${candidate}_readmit_window_validation.csv" \
      --records data/processed/public/readmit_window_validation.jsonl --n 6998 --draws 5000 --out "$RES5/ece_limit_readmit_window.json"
    python - "$RES5" "$candidate" "$MODELS" <<'PY'
import datetime, json, sys
from pathlib import Path
res, candidate, models = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sel = json.load(open(res / "selection.json"))
limits = {}
for qid in ("pubmedqa_answer", "readmit_window"):
    p95 = json.load(open(res / f"ece_limit_{qid}.json"))["ece_if_calibrated"]["p95"]
    limits[qid] = {"p95_if_calibrated": p95, "limit": max(0.05, p95)}
reader = (models / "reader_v1_path.txt").read_text().strip() if (models / "reader_v1_path.txt").exists() else "Laya's reader (no long run)"
out = {"system": candidate, "reader": reader, "recipe": sel["maybe_rule"]["recipe"],
       "head": "fresh head (--reinit-head --freeze-encoder-steps 250)" if sel["head_rule"]["winner"] == "fresh" else "Laya's head (--freeze-encoder-steps 250)",
       "selection": sel, "ece_limits": limits, "time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(out, open(res / "release_candidate.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "selection"}, indent=2))
PY
    ;;
  testpreds)
    if [ ! -f "$RES5/release_candidate.json" ]; then echo "Stop: results/fiorillo_v0_2/release_candidate.json does not exist"; exit 1; fi
    python - "$RES5" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
res = Path(sys.argv[1])
log = {"release_candidate_sha256": hashlib.sha256((res / "release_candidate.json").read_bytes()).hexdigest(),
       "testpreds_start_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(log, open(res / "testpreds_log.json", "w"), indent=2)
print(json.dumps(log))
PY
    for dir in "$MODELS"/decision/*/; do
      system=$(basename "$dir")
      python src/eval/run_laya_model.py --model-dir "$dir" --system "$system" --records-dir data/processed/public \
        --qids pubmedqa_answer readmit_window --out "$RES5/predictions" --max-len 768 --head-max-len 192 --device cuda \
        --calibration "$dir/calibration.json" --splits test
    done
    say "test predictions written for every decision-layer run"
    ;;
  package)
    python - "$CODE" "$MODELS" "$TEXT" "$OUTS" <<'PY'
import hashlib, sys, zipfile
from pathlib import Path
code, models, text, outs = (Path(a) for a in sys.argv[1:5])
def add_tree(z, folder, arc_root):
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            z.write(p, Path(arc_root) / p.relative_to(folder))
with zipfile.ZipFile(outs / "fiorillo_v0_2_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, code / "results" / "fiorillo_v0_2", "results/fiorillo_v0_2")
    add_tree(z, code / "results" / "reader_v1", "results/reader_v1")
with zipfile.ZipFile(outs / "fiorillo_v0_2_model.zip", "w", zipfile.ZIP_STORED) as z:
    add_tree(z, models / "fiorillo_v0_2_candidate", "fiorillo_v0_2_candidate")
    z.write(text / "pmc_training_manifest.csv", "pmc_training_manifest.csv")
reader = models / "reader_v1_path.txt"
if reader.exists():
    with zipfile.ZipFile(outs / "reader_v1.zip", "w", zipfile.ZIP_STORED) as z:
        add_tree(z, Path(reader.read_text().strip()), "reader_v1")
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

#!/bin/bash
# Session 10's fix of the released Fiorillo v0.5 (docs/prompt_session10.md, corrections 7 and 9). Johann runs it on the
# Mac, from the project folder:
#     caffeinate -i bash scripts/fix_release_v0_5.sh            (the fix, the usage check's fourth run, then the upload)
#     bash scripts/fix_release_v0_5.sh --upload                  (the upload only, after a fourth run that PASSED)
# The fix (correction 9, the second fix): on macOS the tree libraries (xgboost, catboost) and torch cannot work in one
# process (whichever runs second ends it with a segmentation fault: readmission at the first revision, PubMedQA and
# Evidence Inference after correction 7's first fix, which loaded the trees first for every question). predict_v0_5.py
# now loads xgboost and catboost before torch only for a file of readmit_window records, and on macOS refuses, with a
# message, a file that mixes readmit_window with the other questions; the model card (README.md) notes the change;
# SHA256SUMS.txt lists the two new files. Nothing else in the release changes.
# (1) It checks the release folder against its SHA256SUMS.txt, which must be the released one or the first fix's. It
#     copies the three files it replaces into models/fiorillo_v0_5/replaced_session10/ (the released files) or
#     models/fiorillo_v0_5/replaced_session10_fix1/ (the first fix's), puts scripts/predict_v0_5.py and
#     docs/model_card_v0_5.md (as README.md) into the release folder and writes the new SHA256SUMS.txt, each checked
#     against the SHA-256 that correction 9 records, and checks the folder again. A folder that already holds the second
#     fix (an earlier run of this script) is only checked.
# (2) It checks the fourth run's record files against results/fiorillo_v0_5/usage_check/fixed2/ids.json.
# (3) It runs predict_v0_5.py, as fixed, once for each question (readmission, PubMedQA, Evidence Inference), each in its
#     own process with Python's fault handler on and xgboost 3.2.0 and catboost 1.2.10 added for the run, as the first
#     three runs did; on an out-of-memory error on the Mac's GPU, that question runs once more with --device cpu.
# (4) It runs compare --fixed2, which writes results/fiorillo_v0_5/usage_check/fixed2/usage_check.json.
# (5) Only if the fourth run PASSES, and only after Johann types yes at its prompt, it uploads the release folder to
#     johanneli/fiorillo-v0.5 with the hf command's own login (no token is read or passed), after checking that the
#     repository is public and holds exactly the release folder's files; the files that did not change are already there.
#     It then checks on Hugging Face that every file of the release folder is there with the same size and content, and
#     prints the new revision. It never changes the repository's visibility and deletes nothing.
# EDDM_PYTHON and EDDM_HF name the Python and the hf command (default: uv run python and uv run hf, the project's own);
# the tests set them, and a stand-in uv, so that nothing reaches Hugging Face.
set -u
cd "$(dirname "$0")/.." || exit 1
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

RELEASE=models/fiorillo_v0_5/release
KEEP=models/fiorillo_v0_5/replaced_session10
KEEP_FIX1=models/fiorillo_v0_5/replaced_session10_fix1
OUT=results/fiorillo_v0_5/usage_check/fixed2
REPO=johanneli/fiorillo-v0.5
OLD_SUMS=a04b52f0f3a83165201b7157c8d518f04ba93cc3cdc4f8a6040eb4f12ce80344
OLD_PREDICT=68c0166b811b56a3c2e392cc4ca1f7cac8df56fd7c0779e2f2fc7da802d0617a
OLD_README=e3ab40964f4b8dc4385355966767d9b0d67bd40712d98f13794ba5d8add4e804
FIX1_SUMS=057d4419eeaa7077272f289b2c2b096fc13ef037e7b0127e26c728ff241461b0
FIX1_PREDICT=387bf09d99dc54b0201e72418de6c7b5c21473aed6968eb9a5ecf74ec93124fc
FIX1_README=cc381ded29689b033cbf26dcfdfb08ae8d6187370a7795c9f0d40cc9a55c0265
NEW_SUMS=e4d5190ea789f8fdd70f0baaf4e05f2a9ab9a27a750eca5f39145b09ccddd44c
NEW_PREDICT=cbcc15277e5f468b66dc64e057423a8deaa68b5b23794b9481f1891b16b2027c
NEW_README=cca505eccfb33cadce5c868001c8525d1d5c15612e402447bccab10c986dcb98
WITH=(--with xgboost==3.2.0 --with catboost==1.2.10)
PY="${EDDM_PYTHON:-uv run python}"
HF="${EDDM_HF:-uv run hf}"
UPLOAD_ONLY=0
[ "${1:-}" = "--upload" ] && UPLOAD_ONLY=1

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "STOP: $*" >&2; exit 1; }
sha() { shasum -a 256 "$1" | awk '{print $1}'; }

[ -f pyproject.toml ] && [ -d "$RELEASE" ] || stop "run this from the project folder (no $RELEASE here)"
command -v shasum >/dev/null 2>&1 || stop "shasum is not on PATH"

# (1) The release folder: as released or with the first fix, then with the second fix; or already with the second fix.
echo "== (1) the release folder"
sums=$(sha "$RELEASE/SHA256SUMS.txt")
if [ "$sums" = "$OLD_SUMS" ] || [ "$sums" = "$FIX1_SUMS" ]; then
  [ "$UPLOAD_ONLY" = 0 ] || stop "the release folder does not hold the second fix yet; run this script without --upload first"
  (cd "$RELEASE" && shasum -a 256 -c SHA256SUMS.txt) || stop "the release folder does not match its SHA256SUMS.txt"
  [ "$(sha scripts/predict_v0_5.py)" = "$NEW_PREDICT" ] || stop "scripts/predict_v0_5.py is not the file correction 9 records"
  [ "$(sha docs/model_card_v0_5.md)" = "$NEW_README" ] || stop "docs/model_card_v0_5.md is not the card correction 9 records"
  if [ "$sums" = "$OLD_SUMS" ]; then
    dest=$KEEP; want_predict=$OLD_PREDICT; want_readme=$OLD_README; what="the released"
  else
    dest=$KEEP_FIX1; want_predict=$FIX1_PREDICT; want_readme=$FIX1_README; what="the first fix's"
  fi
  if [ -e "$dest" ]; then stop "$dest already exists; nothing is replaced"; fi
  mkdir -p "$dest" || stop "cannot make $dest"
  cp -p "$RELEASE/predict_v0_5.py" "$RELEASE/README.md" "$RELEASE/SHA256SUMS.txt" "$dest/" || stop "cannot copy the files to $dest"
  [ "$(sha "$dest/predict_v0_5.py")" = "$want_predict" ] && [ "$(sha "$dest/README.md")" = "$want_readme" ] \
    && [ "$(sha "$dest/SHA256SUMS.txt")" = "$sums" ] || stop "the copies in $dest do not match $what files"
  echo "   $what predict_v0_5.py, README.md and SHA256SUMS.txt are kept in $dest"
  cp scripts/predict_v0_5.py "$RELEASE/predict_v0_5.py" || stop "cannot copy the second fix's predict_v0_5.py"
  cp docs/model_card_v0_5.md "$RELEASE/README.md" || stop "cannot copy the card"
  $PY - "$RELEASE/SHA256SUMS.txt" "$NEW_PREDICT" "$NEW_README" <<'PYEOF' || stop "cannot write the new SHA256SUMS.txt"
import sys
path, predict, readme = sys.argv[1:4]
lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
out, changed = [], 0
for line in lines:
    digest, name = line.rstrip("\n").split("  ", 1)
    if name == "predict_v0_5.py":
        line, changed = f"{predict}  {name}\n", changed + 1
    elif name == "README.md":
        line, changed = f"{readme}  {name}\n", changed + 1
    out.append(line)
if changed != 2:
    sys.exit("SHA256SUMS.txt does not list predict_v0_5.py and README.md once each")
open(path, "w", encoding="utf-8").write("".join(out))
PYEOF
  [ "$(sha "$RELEASE/SHA256SUMS.txt")" = "$NEW_SUMS" ] || stop "the new SHA256SUMS.txt is not the one correction 9 records"
  echo "   the second fix's predict_v0_5.py and README.md and the new SHA256SUMS.txt are in the release folder"
elif [ "$sums" = "$NEW_SUMS" ]; then
  echo "   the release folder already holds the second fix"
else
  stop "SHA256SUMS.txt has SHA-256 $sums, neither the released one, the first fix's nor the second fix's"
fi
(cd "$RELEASE" && shasum -a 256 -c SHA256SUMS.txt) || stop "the release folder does not match its new SHA256SUMS.txt"

if [ "$UPLOAD_ONLY" = 0 ]; then
  command -v uv >/dev/null 2>&1 || [ -n "${EDDM_PYTHON:-}" ] || stop "uv is not on PATH"
  # (2) The record files.
  echo "== (2) the record files against $OUT/ids.json"
  [ -f "$OUT/ids.json" ] || stop "$OUT/ids.json is missing (the fourth run's sample is drawn on the linked shell)"
  for qid in readmit_window pubmedqa_answer evidence_effect; do
    if [ -s "$OUT/answers_${qid}.jsonl" ] || [ -s "$OUT/run_${qid}.log" ]; then
      stop "$OUT already holds a run's files; the fourth run is written once"
    fi
  done
  uv run python -c "import sys; sys.path.insert(0, 'src/eval'); import usage_check_v0_5 as U; sys.exit(U.check_records(fixed2=True))" \
    || stop "the record files do not match $OUT/ids.json"
  AUTO=$(uv run python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null)
  [ -n "$AUTO" ] || AUTO=unknown
  echo "   --device auto resolves to: $AUTO"

  run_one() {
    local qid=$1 device=$2
    local log="$OUT/run_${qid}.log" records
    case "$qid" in
      readmit_window) records=data/processed/session10/usage_check_readmit_window_records.jsonl ;;
      pubmedqa_answer) records=data/processed/session10/usage_check_pubmedqa_answer_records.jsonl ;;
      evidence_effect) records=data/processed/session10/usage_check_evidence_effect_long_records.jsonl ;;
    esac
    local n resolved status
    n=$(grep -c . "$records")
    resolved=$device
    [ "$device" = auto ] && resolved=$AUTO
    {
      echo "usage_check question: $qid"
      echo "usage_check records: $records ($n records)"
      echo "usage_check mode: fixed2"
      echo "usage_check device_requested: $device"
      echo "usage_check device_resolved: $resolved"
      echo "usage_check start_utc: $(stamp)"
    } >> "$log"
    PYTHONFAULTHANDLER=1 uv run "${WITH[@]}" python "$RELEASE/predict_v0_5.py" --records "$records" --release "$RELEASE" \
      --device "$device" --out "$OUT/answers_${qid}.jsonl" >> "$log" 2>&1
    status=$?
    { echo; echo "usage_check end_utc: $(stamp)"; echo "usage_check exit_status: $status"; } >> "$log"
    return $status
  }

  out_of_memory() {
    awk '/^usage_check start_utc:/{buf=""} {buf = buf "\n" $0} END{print buf}' "$1" \
      | grep -E -i -q 'out of memory|OutOfMemoryError|MPS backend out of memory'
  }

  # (3) One run for each question, each in its own process.
  for qid in readmit_window pubmedqa_answer evidence_effect; do
    log="$OUT/run_${qid}.log"
    : > "$log"
    : > "$OUT/answers_${qid}.jsonl"
    echo "== (3) $qid: the second fix's predict_v0_5.py (log: $log)"
    if run_one "$qid" auto; then
      echo "   finished"
    elif out_of_memory "$log"; then
      echo "usage_check out_of_memory: the run above ended with an out-of-memory error on the GPU; it runs once more with --device cpu" >> "$log"
      echo "== (3) $qid: out of memory on the GPU; running it once more with --device cpu"
      if run_one "$qid" cpu; then echo "   finished on the CPU"; else echo "   the CPU run failed too (see $log)"; fi
    else
      echo "   failed (see $log); compare counts its records as failures"
    fi
  done

  # (4) The comparison.
  echo "== (4) compare --fixed2"
  uv run "${WITH[@]}" python src/eval/usage_check_v0_5.py compare --fixed2
  status=$?
  [ "$status" = 0 ] || stop "the fourth run did not PASS ($OUT/usage_check.json); nothing is uploaded. Tell Claude."
fi

# (5) The upload, only after a fourth run that PASSED with this release folder.
$PY - "$OUT/usage_check.json" "$NEW_SUMS" "$NEW_PREDICT" <<'PYEOF' || stop "no PASSING fourth run with the second fix; nothing is uploaded"
import json, sys
path, sums, predict = sys.argv[1:4]
try:
    data = json.load(open(path, encoding="utf-8"))
except (OSError, ValueError):
    sys.exit(f"{path} is missing or unreadable")
rel = data.get("release") or {}
if data.get("verdict") != "PASS" or rel.get("SHA256SUMS.txt_sha256") != sums or rel.get("predict_v0_5.py_sha256") != predict:
    sys.exit(f"{path}: verdict {data.get('verdict')}, with SHA256SUMS.txt {rel.get('SHA256SUMS.txt_sha256')}")
print(f"the fourth run PASSED with the second fix ({path})")
PYEOF

echo "== (5) the upload to https://huggingface.co/$REPO"
WHO="$($HF auth whoami --format json)" || stop "the Hugging Face command line is not signed in; run 'uv run hf auth login' first. Nothing uploaded."
USER_NAME="$($PY -c 'import json, sys; print((json.loads(sys.argv[1] or "{}") or {}).get("user") or "")' "$WHO")"
[ "$USER_NAME" = "${REPO%%/*}" ] || stop "the Hugging Face command line is signed in as '$USER_NAME', not ${REPO%%/*}. Nothing uploaded."
INFO="$($HF models info "$REPO" --format json)" || stop "cannot read $REPO on Hugging Face. Nothing uploaded."
$PY - "$RELEASE" "$REPO" "$INFO" <<'PYEOF' || stop "nothing uploaded"
import json, sys
from pathlib import Path
rel, repo, info = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
if info.get("private") is not False:
    sys.exit(f"{repo} is not public as released")
local = {p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file()}
remote = {s["rfilename"] for s in info.get("siblings") or [] if s["rfilename"] != ".gitattributes"}
if remote != local:
    sys.exit(f"{repo} holds other files than the release folder: only there {sorted(remote - local)[:5]}, only here {sorted(local - remote)[:5]}")
print(f"{repo} is public, at revision {info.get('sha')}, and holds the release folder's {len(local)} files")
PYEOF
printf 'Upload the fixed predict_v0_5.py, README.md and SHA256SUMS.txt to https://huggingface.co/%s? Type yes to upload: ' "$REPO"
read -r ANSWER
[ "$ANSWER" = yes ] || stop "not uploaded (the answer was not yes); run 'bash scripts/fix_release_v0_5.sh --upload' to upload later"
$HF upload "$REPO" "$RELEASE" . --commit-message "predict_v0_5.py loads the tree libraries before torch only for readmission files (macOS); card note; SHA256SUMS.txt" \
  || stop "the upload failed; run 'bash scripts/fix_release_v0_5.sh --upload' to try again"

INFO="$($HF models info "$REPO" --format json)"
FILES="$($HF models ls "$REPO" -R --format json)"
REVISION="$($PY - "$RELEASE" "$REPO" "$INFO" "$FILES" <<'PYEOF'
import hashlib, json, re, sys
from pathlib import Path
rel, repo, info, files = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
sums = {}
for line in (rel / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line.rstrip("\r\n"))
    if m:
        sums[m.group(2)[2:] if m.group(2).startswith("./") else m.group(2)] = m.group(1).lower()
remote = {f["path"]: f for f in files if isinstance(f, dict) and "size" in f}
bad = [] if info.get("private") is False else ["the repository is not public"]
for p in sorted(q for q in rel.rglob("*") if q.is_file()):
    name = p.relative_to(rel).as_posix()
    r = remote.get(name)
    if r is None:
        bad.append(f"{name} is not on Hugging Face")
        continue
    size = p.stat().st_size
    if r.get("size") != size:
        bad.append(f"{name}: size {r.get('size')} on Hugging Face, {size} here")
    lfs = r.get("lfs") or {}
    if lfs.get("sha256"):
        want = sums.get(name) or hashlib.sha256(p.read_bytes()).hexdigest()
        if lfs["sha256"] != want:
            bad.append(f"{name}: SHA-256 {lfs['sha256']} on Hugging Face")
    elif r.get("blob_id"):
        blob = hashlib.sha1(b"blob %d\0" % size + p.read_bytes()).hexdigest()
        if r["blob_id"] != blob:
            bad.append(f"{name}: blob {r['blob_id']} on Hugging Face, {blob} here")
extra = sorted(n for n in remote if n != ".gitattributes" and not (rel / n).is_file())
bad += [f"on Hugging Face and not in the release folder: {n}" for n in extra]
if bad:
    print(f"Check failed on Hugging Face: {bad[:10]}", file=sys.stderr)
    sys.exit(1)
print(info.get("sha") or "unknown")
PYEOF
)" || stop "the check on Hugging Face failed after the upload. Tell Claude."
printf '== done: https://huggingface.co/%s is at revision %s, and every file of the release folder is there with the same size and content.\nPaste everything above into the chat so that it can be recorded.\n' "$REPO" "$REVISION"

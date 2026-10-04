#!/usr/bin/env bash
# Session 4, Mac part (docs/prompt_session4.md, step 9, with corrections 3 to 7): tests, the training outputs,
# Duan, plain Laya, the comparison, the release bar, commit and push. Run once, by hand, in Terminal on the project
# Mac, with five zip files: the release candidate's arm L results and model (run 3 by correction 7, or run 2 if run 3
# did not finish), then run 2's arm B, arm L-fresh and arm L results. caffeinate keeps the Mac awake while it runs
# (about an hour and a half):
#     cd ~/Documents/Research/ed-decision-model && Z2=models/public_v0/runpod_zips/run2 && Z3=models/public_v0/runpod_zips/run3
#     caffeinate -i bash scripts/finish_session4_macos.sh $Z3/arm_laya_results.zip $Z3/arm_laya_model.zip \
#         $Z2/arm_bioclinical_results.zip $Z2/arm_laya_fresh_results.zip $Z2/arm_laya_results.zip
# Stops at the first failure. Safe to run again: finished model runs are skipped. Nothing is deleted. Never reads
# the restricted folder. Uses your Hugging Face login only to download Duan (gated; your account has access).
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "Usage: bash $0 arm_laya_results.zip arm_laya_model.zip arm_bioclinical_results.zip arm_laya_fresh_results.zip run2_arm_laya_results.zip"
  exit 1
fi
for f in "$@"; do [ -s "$f" ] || { echo "Not found or empty: $f"; exit 1; }; done
LAYA_RESULTS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
LAYA_MODEL="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
BIO_RESULTS="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
FRESH_RESULTS="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
RUN2_LAYA_RESULTS="$(cd "$(dirname "$5")" && pwd)/$(basename "$5")"
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
export PYTHONDONTWRITEBYTECODE=1

printf '\n== 1/8 uv sync and tests\n'
uv sync
uv run pytest -q -p no:cacheprovider src/tests

printf '\n== 2/8 training outputs into results/public_v0/ and models/public_v0/arm_laya/\n'
uv run python - "$LAYA_RESULTS" "$LAYA_MODEL" "$BIO_RESULTS" "$FRESH_RESULTS" "$RUN2_LAYA_RESULTS" <<'PY'
import json, sys, zipfile
from pathlib import Path
laya_results, laya_model, bio_results, fresh_results, run2_laya_results = map(Path, sys.argv[1:6])
results = Path("results/public_v0")
builds = {"current records (run 3, correction 7)": json.loads((results / "checksums.json").read_text(encoding="utf-8")),
          "run 2 records": json.loads((results / "run2" / "checksums.json").read_text(encoding="utf-8"))}
current = builds["current records (run 3, correction 7)"]
for zpath, arm, system in ((laya_results, "arm_laya", "arm_laya"), (bio_results, "arm_bioclinical", "arm_bioclinical"),
                           (fresh_results, "arm_laya_fresh", "arm_laya_fresh"), (run2_laya_results, "arm_laya", "arm_laya_run2")):
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        got = json.loads(z.read("checksums.json"))
        build = next((b for b, c in builds.items() if got == c), None)
        if build is None:
            sys.exit(f"{zpath.name}: the records it was trained on match neither results/public_v0/checksums.json nor run2/checksums.json")
        if system in ("arm_bioclinical", "arm_laya_fresh", "arm_laya_run2") and build != "run 2 records":
            sys.exit(f"{zpath.name}: expected run 2 records for {system}")
        tests = [k for k in current if k.endswith("_test.jsonl")]
        if any(got.get(k) != current[k] for k in tests):
            sys.exit(f"{zpath.name}: its test records differ from the current build")
        need = [f"{arm}_metrics.json", f"{arm}_calibration.json", f"{arm}_train_log.csv"] + [
            f"predictions/{s}_{qid}_{split}.csv" for s in (arm, f"{arm}-raw")
            for qid in ("pubmedqa_answer", "readmit_window") for split in ("validation", "test")]
        missing = [n for n in need if n not in names]
        if missing:
            sys.exit(f"{zpath.name}: missing {missing}")
        for name in sorted(names - {"checksums.json", "counts.json"}):
            path = Path(name)
            if system != arm and not path.name.startswith(arm):
                sys.exit(f"{zpath.name}: unexpected file {name}")
            out_name = name if system == arm else str(path.with_name(system + path.name[len(arm):]))
            target = results / out_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
        print(f"{zpath.name} as {system}: {len(names) - 2} files into results/public_v0/ (trained on the {build})")
        metrics = json.loads(z.read(f"{arm}_metrics.json"))
        print(f"  temperatures {metrics.get('temperature')}; elapsed seconds {metrics.get('elapsed_seconds')}")
with zipfile.ZipFile(laya_model) as z:
    names = z.namelist()
    for n in ("arm_laya/model.safetensors", "arm_laya/rl_agent_config.json", "arm_laya/calibration.json",
              "arm_laya/encoder/config.json", "arm_laya/tokenizer/tokenizer_config.json"):
        if n not in names:
            sys.exit(f"{laya_model.name}: missing {n}")
    if any(n.startswith("/") or ".." in Path(n).parts for n in names):
        sys.exit(f"{laya_model.name}: unexpected paths")
    z.extractall("models/public_v0")
    print(f"{laya_model.name}: {len(names)} files into models/public_v0/arm_laya/")
PY
ls -l models/public_v0/arm_laya/model.safetensors

printf '\n== 2b/8 baselines again on the current records (the prior is the training split'"'"'s label frequencies,\n== and corrections 4 and 7 changed the PubMedQA training split; baselines.json still held run 1'"'"'s prior)\n'
uv run python src/eval/baselines.py

printf '\n== 3/8 Duan (wipen/Duan, pinned revision) in models/duan/, for evaluation only\n'
if [ ! -s models/duan/model.safetensors ]; then
  uv run hf download wipen/Duan --revision 50e0615b576e4829cbad5c193d42edd0cadf2784 --local-dir models/duan
fi
ls -l models/duan/model.safetensors

printf '\n== 4/8 our records in Duan'"'"'s question wording and state format\n'
uv run python src/eval/duan_adapter.py

run_model() {  # system, then run_laya_model.py options; skipped when the run already finished
  local system="$1"; shift
  if [ -s "results/public_v0/${system}_run_summary.json" ]; then
    echo "${system}: already done (results/public_v0/${system}_run_summary.json)"
  else
    uv run python src/eval/run_laya_model.py --system "$system" "$@"
  fi
}

printf '\n== 5/8 plain Laya, zero-shot (roughly 30 to 60 minutes)\n'
run_model laya_zeroshot --model-dir models/laya-base --qids pubmedqa_answer readmit_window --max-len 768 --head-max-len 192

printf '\n== 6/8 Duan on its own wording (primary), then on ours (sensitivity)\n'
run_model duan --model-dir models/duan --records-dir data/processed/public/duan --qids duan_conclusion_supported duan_readmit_window
run_model duan_ourwording --model-dir models/duan --qids pubmedqa_answer readmit_window

printf '\n== 7/8 comparison and release bar\n'
uv run python src/eval/compare.py

printf '\n== 8/8 commit and push\n'
git add -A
git status --short
if git diff --cached --quiet; then echo "Nothing new to commit."; else
  git commit -q -F - <<'MSGEOF'
session 4: public v0 results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MjS5gBCyYSnoiQUNer3VxS
MSGEOF
fi
git push
printf '\nSession 4 Mac part is finished. Copy everything from "== 7/8" down into the chat so it can be checked.\n'

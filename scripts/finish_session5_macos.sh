#!/usr/bin/env bash
# Session 5, Mac part (docs/prompt_session5.md, step 14): tests, the pod's outputs into place, the test-prediction check,
# the comparison with the release bar, commit and push. Run once, by hand, in Terminal on the project Mac, with the
# zip files copied from the pod in step 13 (reader_v1.zip only if there was a long run):
#     cd ~/Documents/Research/ed-decision-model && Z=models/fiorillo_v0_2/runpod_zips
#     caffeinate -i bash scripts/finish_session5_macos.sh $Z/fiorillo_v0_2_results.zip $Z/fiorillo_v0_2_model.zip $Z/reader_v1.zip
# Stops at the first failure; nothing is extracted unless the test predictions were made after the candidate was fixed.
# Safe to run again (the same model already in place is written again; a different one stops it). Nothing is deleted. Never reads the restricted folder. Uses no Hugging Face login.
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash $0 fiorillo_v0_2_results.zip fiorillo_v0_2_model.zip [reader_v1.zip]"
  exit 1
fi
for f in "$@"; do [ -s "$f" ] || { echo "Not found or empty: $f"; exit 1; }; done
RESULTS_ZIP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MODEL_ZIP="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
READER_ZIP=""
if [ "$#" -eq 3 ]; then READER_ZIP="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"; fi
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
export PYTHONDONTWRITEBYTECODE=1

printf '\n== 1/6 uv sync and tests\n'
uv sync
uv run pytest -q -p no:cacheprovider src/tests

printf '\n== 2/6 the test-prediction check on the zip, then the pod'"'"'s outputs into results/fiorillo_v0_2/, results/reader_v1/, models/fiorillo_v0_2/candidate/ and models/reader_v1/\n'
uv run python - "$RESULTS_ZIP" "$MODEL_ZIP" "$READER_ZIP" <<'PY'
import hashlib, sys, zipfile
from pathlib import Path
results_zip, model_zip = Path(sys.argv[1]), Path(sys.argv[2])
reader_zip = Path(sys.argv[3]) if sys.argv[3] else None

def safe(names, zname):
    bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
    if bad:
        sys.exit(f"{zname}: unexpected paths {bad[:5]}")

def extract(zpath, rules):
    """rules: (prefix in the zip, target folder); every file must match one rule."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        safe(names, zpath.name)
        counts = {}
        for n in names:
            rule = next(((p, t) for p, t in rules if n == p or n.startswith(p)), None)
            if rule is None:
                sys.exit(f"{zpath.name}: unexpected file {n}")
            prefix, target = rule
            out = Path(target) / n[len(prefix):] if prefix.endswith("/") else Path(target)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(n))
            counts[target] = counts.get(target, 0) + 1
        for target, k in counts.items():
            print(f"{zpath.name}: {k} files into {target}")

with zipfile.ZipFile(results_zip) as z:
    names = set(z.namelist())
    for need in ("results/fiorillo_v0_2/release_candidate.json", "results/fiorillo_v0_2/testpreds_log.json",
                 "results/fiorillo_v0_2/selection.json"):
        if need not in names:
            sys.exit(f"{results_zip.name}: missing {need}")
    # step 14's check, on the files inside the zip, before anything is extracted
    import datetime, json
    rc_bytes = z.read("results/fiorillo_v0_2/release_candidate.json")
    rc, log = json.loads(rc_bytes), json.loads(z.read("results/fiorillo_v0_2/testpreds_log.json"))
    sha = hashlib.sha256(rc_bytes).hexdigest()
    parse = lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    if log.get("release_candidate_sha256") != sha:
        sys.exit(f"Stop, nothing extracted: testpreds_log.json names {log.get('release_candidate_sha256')}, "
                 f"but release_candidate.json has SHA-256 {sha}")
    if not parse(log["testpreds_start_utc"]) > parse(rc["time_utc"]):
        sys.exit(f"Stop, nothing extracted: test predictions started at {log['testpreds_start_utc']}, "
                 f"not after the candidate was fixed at {rc['time_utc']}")
with zipfile.ZipFile(model_zip) as z:
    mnames = set(z.namelist())
for need in ("fiorillo_v0_2_candidate/model.safetensors", "fiorillo_v0_2_candidate/calibration.json",
             "fiorillo_v0_2_candidate/rl_agent_config.json", "pmc_training_manifest.csv"):
    if need not in mnames:
        sys.exit(f"{model_zip.name}: missing {need}")
def sha256_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()

placed = Path("models/fiorillo_v0_2/candidate/model.safetensors")
if placed.exists():  # a second run: the same model is written again in full; a different one stops the script
    with zipfile.ZipFile(model_zip) as z, z.open("fiorillo_v0_2_candidate/model.safetensors") as f:
        in_zip = sha256_stream(f)
    with open(placed, "rb") as f:
        on_disk = sha256_stream(f)
    if in_zip != on_disk:
        sys.exit("models/fiorillo_v0_2/candidate/ holds a different model; stop (nothing is overwritten there)")
    print("models/fiorillo_v0_2/candidate/: the same model is already in place; its files are written again")
extract(results_zip, [("results/fiorillo_v0_2/", "results/fiorillo_v0_2"), ("results/reader_v1/", "results/reader_v1")])
extract(model_zip, [("fiorillo_v0_2_candidate/", "models/fiorillo_v0_2/candidate"),
                    ("pmc_training_manifest.csv", "models/fiorillo_v0_2/pmc_training_manifest.csv")])
if reader_zip:
    extract(reader_zip, [("reader_v1/", "models/reader_v1")])
PY
ls -l models/fiorillo_v0_2/candidate/model.safetensors

printf '\n== 3/6 the same check on the extracted files\n'
uv run python - <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
res = Path("results/fiorillo_v0_2")
rc_bytes = (res / "release_candidate.json").read_bytes()
rc, log = json.loads(rc_bytes), json.loads((res / "testpreds_log.json").read_text(encoding="utf-8"))
sha = hashlib.sha256(rc_bytes).hexdigest()
if log.get("release_candidate_sha256") != sha:
    sys.exit(f"Stop: testpreds_log.json names {log.get('release_candidate_sha256')}, but release_candidate.json has SHA-256 {sha}")
parse = lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
if not parse(log["testpreds_start_utc"]) > parse(rc["time_utc"]):
    sys.exit(f"Stop: test predictions started at {log['testpreds_start_utc']}, not after the candidate was fixed at {rc['time_utc']}")
print(f"release_candidate.json SHA-256 {sha} matches; candidate {rc['system']} fixed at {rc['time_utc']}, "
      f"test predictions started at {log['testpreds_start_utc']}")
PY

printf '\n== 4/6 comparison (D1 to D5) and the release bar\n'
uv run python src/eval/compare.py --session5

printf '\n== 5/6 the release bar, from results/fiorillo_v0_2/comparison.json\n'
uv run python - <<'PY'
import json
bar = json.load(open("results/fiorillo_v0_2/comparison.json", encoding="utf-8"))["release_bar"]
for k, c in bar["criteria"].items():
    print(f"criterion {k}: {c['result']}" + (f" ({c['reason']})" if c["reason"] else ""))
print(f"overall: {bar['overall']} for {bar['system']}")
PY

printf '\n== 6/6 commit and push\n'
git add -A
git status --short
if git diff --cached --quiet; then echo "Nothing new to commit."; else
  git commit -q -F - <<'MSGEOF'
session 5: Fiorillo v0.2 results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MjS5gBCyYSnoiQUNer3VxS
MSGEOF
fi
git push
printf '\nSession 5 Mac part is finished. Copy everything from "== 4/6" down into the chat so it can be checked.\n'

#!/usr/bin/env bash
# Session 1, macOS part: the steps the session 1 sandbox could not do.
# Run once, by hand, in Terminal on the project Mac:
#     bash ~/Documents/Research/ed-decision-model/scripts/finish_session1_macos.sh
#
# In order:
#   1. Machine check (Apple Silicon, macOS 14 or later), appended to docs/environment.md
#   2. Homebrew and uv if missing (official installers), Python environment for Apple Silicon (uv sync),
#      then the data guard tests
#   3. MIMIC-IV-ED Demo and MIMIC-IV Demo fixtures: the wget command from each PhysioNet page, SHA-256
#      checksums verified, only the kept files copied into src/tests/fixtures/, download record appended
#   4. Row counts for every demo table (src/tools/inspect_fixtures.py)
#   5. Laya MLX smoke test (downloads about 850 MB of model weights the first time)
#   6. git repository, pre-commit data guard, first commit, private GitHub repository
# Stops at the first failure. Safe to run again: finished parts are skipped. Nothing is deleted.
# It never reads the restricted folder (EDDM_RESTRICTED_ROOT).
set -euo pipefail

PROJECT="$HOME/Documents/Research/ed-decision-model"
cd "$PROJECT"
DAY="$(date +%d)"; DAY="${DAY#0}"
TODAY_ISO="$(date +%Y-%m-%d)"
TODAY_LONG="$DAY $(date +%B) $(date +%Y)"
step() { printf '\n== %s\n' "$1"; }
commas() { uv run python -c 'import sys; print(f"{int(sys.argv[1]):,}")' "$1"; }

step "1/6 Machine check"
ARCH="$(uname -m)"
MACOS="$(sw_vers -productVersion)"
echo "uname -m: $ARCH    macOS: $MACOS"
if [ "$ARCH" != "arm64" ]; then echo "This is not an Apple Silicon Mac. Stopping."; exit 1; fi
if [ "${MACOS%%.*}" -lt 14 ]; then echo "macOS 14 or later is required. Stopping."; exit 1; fi

step "2/6 Python environment (uv sync) and data guard tests"
# Homebrew and uv are installed with their official installers, and only if missing.
if ! command -v brew >/dev/null 2>&1 && [ -x /opt/homebrew/bin/brew ]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# LightGBM wheels for macOS load the OpenMP runtime from Homebrew at import time.
if ! brew list libomp >/dev/null 2>&1; then brew install libomp; fi
uv sync
uv run python -c "import torch, transformers, datasets, accelerate, lightgbm, sklearn, pandas, pyarrow, duckdb, matplotlib, yaml, laya, laya_mlx; print('imports ok')"
PYVER="$(uv run python -c 'import platform; print(platform.python_version())')"
if ! grep -q "^## macOS check" docs/environment.md; then
  cat >> docs/environment.md <<ENVEOF

## macOS check (run on the Mac by scripts/finish_session1_macos.sh, $TODAY_LONG)

- \`uname -m\`: $ARCH
- \`sw_vers -productVersion\`: $MACOS
- Environment built with \`uv sync\` using $(uv --version); Python $PYVER in \`.venv/\`.
ENVEOF
fi
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -p no:cacheprovider src/tests

step "3/6 MIMIC demo fixtures (open access, PhysioNet)"
if ! command -v wget >/dev/null 2>&1; then brew install wget; fi
DL="$(mktemp -d "${TMPDIR:-/tmp}/eddm-physionet.XXXXXX")"
echo "Download folder, outside the project: $DL"
fetch_demo() {  # arguments: PhysioNet slug, fixture folder name, regular expression for the kept folders
  local slug="$1" name="$2" keep="$3"
  local dest="src/tests/fixtures/$name" src="$DL/physionet.org/files/$slug/2.2"
  local log="$DL/$name.wget.log" all="$DL/$name.all.sha256" kept="$DL/$name.kept.sha256" status=0
  if [ -f "$dest/SHA256SUMS.txt" ] && grep -q -- "- $name:" src/tests/fixtures/README.md; then
    echo "$name already in place; skipping."
    return
  fi
  echo "Running: wget -r -N -c -np https://physionet.org/files/$slug/2.2/  (full output saved to $log)"
  (cd "$DL" && wget -r -N -c -np "https://physionet.org/files/$slug/2.2/") > "$log" 2>&1 || status=$?
  if [ "$status" -eq 0 ]; then
    tail -n 3 "$log"
  else
    tail -n 20 "$log"
    echo "wget ended with status $status. The checksum check below decides whether the download is complete."
  fi
  # PhysioNet writes one space between checksum and file name; shasum -c reads only lines with two,
  # so the checks read a copy of the same lines rewritten with two spaces.
  awk '{ f = $2; sub(/^\*/, "", f); print $1 "  " f }' "$src/SHA256SUMS.txt" > "$all"
  echo "Checking every downloaded file against SHA256SUMS.txt:"
  (cd "$src" && shasum -a 256 -c "$all")
  awk -v keep="$keep" '$2 ~ keep' "$all" > "$kept"
  while read -r _hash file; do
    mkdir -p "$dest/$(dirname "$file")"
    cp "$src/$file" "$dest/$file"
  done < "$kept"
  cp "$src/SHA256SUMS.txt" "$dest/SHA256SUMS.txt"
  echo "Checking the kept files in $dest:"
  (cd "$dest" && shasum -a 256 -c "$kept")
  local files bytes
  files="$(find "$dest" -type f ! -name .DS_Store | wc -l | tr -d ' ')"
  bytes="$(find "$dest" -type f ! -name .DS_Store -exec stat -f %z {} + | awk '{ s += $1 } END { print s }')"
  echo "- $name: downloaded $TODAY_LONG with \`wget -r -N -c -np https://physionet.org/files/$slug/2.2/\`; $files files kept (including SHA256SUMS.txt), $(commas "$bytes") bytes on disk; SHA-256 checksums verified for every downloaded file and again for the kept files." >> src/tests/fixtures/README.md
}
fetch_demo mimic-iv-ed-demo mimic-iv-ed-demo-2.2 '^ed/'
fetch_demo mimic-iv-demo mimic-iv-demo-2.2 '^(hosp|icu)/'

step "4/6 Demo table row counts"
PYTHONDONTWRITEBYTECODE=1 uv run python src/tools/inspect_fixtures.py

step "5/6 Laya MLX smoke test"
PYTHONDONTWRITEBYTECODE=1 uv run python notebooks/00_laya_quickstart.py

step "6/6 git repository, data guard hook, first commit, private GitHub repository"
if [ ! -d .git ]; then git init -q -b main; fi
chmod +x src/tools/check_no_data.py
ln -sf ../../src/tools/check_no_data.py .git/hooks/pre-commit
if ! git config user.email >/dev/null 2>&1; then
  git config user.name "Johann Li"
  git config user.email "$(gh api user --jq .id)+johann-e-li@users.noreply.github.com"
fi
if ! grep -q "session 1, macOS part" docs/log.md; then
  echo "## $TODAY_ISO | Terminal, scripts/finish_session1_macos.sh | session 1, macOS part | machine check; uv sync; data guard tests; MIMIC-IV-ED Demo and MIMIC-IV Demo downloaded and verified; demo row counts; Laya MLX smoke test; git init; pre-commit hook; first commit; private GitHub repository | .venv/ (ignored), src/tests/fixtures/mimic-iv-ed-demo-2.2/, src/tests/fixtures/mimic-iv-demo-2.2/, data/scratch_demo.duckdb (ignored), results/practice/fixture_row_counts.csv (ignored by the *.csv rule), results/practice/quickstart_output.json, .git/ | NOT BUILT: none | open questions: none" >> docs/log.md
fi
git add -A
git status --short
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -q -F - <<'MSGEOF'
session 1: environment, repository, fixtures, smoke test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MjS5gBCyYSnoiQUNer3VxS
MSGEOF
fi
git log --oneline -1
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  gh repo create johann-e-li/ed-decision-model --private --source . --push
fi
printf '\nSession 1 is finished. Copy everything above this line into the chat so it can be checked.\n'

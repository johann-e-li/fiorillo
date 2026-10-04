#!/usr/bin/env bash
# Session 2, macOS part: rebuild the environment with lxml, run the tests under the project Python, push.
# Run once, by hand, in Terminal on the project Mac:
#     bash ~/Documents/Research/ed-decision-model/scripts/finish_session2_macos.sh
# Stops at the first failure. Safe to run again. Nothing is deleted. Never reads the restricted folder.
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
printf '\n== 1/3 uv sync (adds lxml)\n'
uv sync
uv run python -c "import lxml; print('lxml', lxml.__version__)"
printf '\n== 2/3 tests under the project Python\n'
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -p no:cacheprovider src/tests
printf '\n== 3/3 push\n'
git status --short
git log --oneline -3
git push
printf '\nSession 2 is finished. Copy everything above this line into the chat so it can be checked.\n'

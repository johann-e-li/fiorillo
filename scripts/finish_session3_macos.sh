#!/usr/bin/env bash
# Session 3, macOS part: tests under the project Python, the base model, the practice data, and a timed
# one-epoch run over 2,000 rows on the Apple GPU (MPS). Run once, by hand, in Terminal on the project Mac:
#     bash ~/Documents/Research/ed-decision-model/scripts/finish_session3_macos.sh
# Stops at the first failure. Safe to run again. Nothing is deleted. Never reads the restricted folder.
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
export PYTHONDONTWRITEBYTECODE=1

printf '\n== 1/5 uv sync and tests\n'
uv sync
uv run pytest -q -p no:cacheprovider src/tests

printf '\n== 2/5 base model (convaiinnovations/laya, pinned revision) in models/laya-base/\n'
if [ ! -s models/laya-base/model.safetensors ]; then
  uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851", local_dir="models/laya-base",
    allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
PY
fi
ls -l models/laya-base/model.safetensors

printf '\n== 3/5 practice data (DAIR Emotion)\n'
if [ ! -s data/processed/practice/emotion_test.pt ]; then uv run python src/train/practice_data.py; fi
cat data/processed/practice/emotion_counts.json | head -30

printf '\n== 4/5 timed run: one epoch over 2,000 rows on MPS\n'
uv run python src/train/train_laya.py --device mps --max-rows 2000 --epochs 1 \
  --out models/practice-laya-emotion-mps-timing --metrics results/practice/mps_timing.json --log results/practice/mps_timing_log.csv
uv run python - <<'PY'
import json
m = json.load(open("results/practice/mps_timing.json"))
sps = m["sequences_per_second"]; tokens_per_seq = 60  # DAIR Emotion items are about 60 tokens
print(f"sequences per second on MPS: {sps}")
print(f"one epoch over 16,000 rows: {16000 / sps / 3600:.2f} hours")
print(f"per billion tokens at this throughput (indication only): {1e9 / (sps * tokens_per_seq) / 3600:.0f} hours")
print("test accuracy after 2,000 rows, one epoch:", m["test_before_temperature"]["accuracy"], "| pass mark", m["pass_mark"])
PY

printf '\n== 5/5 commit and push\n'
git add -A
git status --short
if git diff --cached --quiet; then echo "Nothing new to commit."; else
  git commit -q -F - <<'MSGEOF'
session 3: timed Mac run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MjS5gBCyYSnoiQUNer3VxS
MSGEOF
fi
git push
printf '\nSession 3 Mac part is finished. Copy everything above this line into the chat so it can be checked.\n'

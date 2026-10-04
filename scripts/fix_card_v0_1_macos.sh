#!/usr/bin/env bash
# Correction 1 to docs/prompt_session5.md (27 September 2026): replace the priority sentence on Fiorillo v0.1's
# Hugging Face model card. Run by Johann only, by hand, in Terminal on the project Mac:
#     bash ~/Documents/Research/ed-decision-model/scripts/fix_card_v0_1_macos.sh
# It uploads one file, README.md, to johanneli/fiorillo-v0.1, and nothing else: the weights and every other file stay
# as released. It stops unless the repository's main branch is still the released commit 0a2e55e5cb8a, the README.md
# there is byte-identical to the released one (SHA-256 f3621abf...), and models/public_v0/card_correction_1/README.md is
# the corrected card as checked when it was written (SHA-256 e7a3b49a...). It prints the difference and uploads only if
# you type yes (or y). Afterwards it downloads README.md from the new commit, checks its SHA-256, and keeps both
# versions beside the corrected file (live_before.md, live_after.md). It can be run again until the upload is done. The DOI 10.57967/hf/10639 stays with revision 0a2e55e.
# Nothing is deleted. The upload command is the one in scripts/release_public_v0.sh (hf upload REPO_ID LOCAL_PATH
# PATH_IN_REPO --commit-message, huggingface_hub 1.33.0 as in uv.lock).
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

REPO=johanneli/fiorillo-v0.1
RELEASED_COMMIT=0a2e55e5cb8ac03f199ba9a6e1e86d8fe2267cd2
RELEASED_SHA=f3621abf4838378034e230bd4e05c04b69e4672da00a16b9db366db67c5564ff
DIR=models/public_v0/card_correction_1
NEW="$DIR/README.md"
NEW_SHA=e7a3b49ab0ddc8bc699d3d6dd47c7fdc7722cf1457113ceb8b17824477e80d88

if [ ! -s "$NEW" ]; then echo "Stop: $NEW is missing."; exit 1; fi
if [ "$(shasum -a 256 "$NEW" | cut -d' ' -f1)" != "$NEW_SHA" ]; then
  echo "Stop: $NEW is not the checked file (its SHA-256 differs); nothing uploaded."; exit 1
fi
if [ -e "$DIR/live_after.md" ]; then
  echo "Stop: $DIR/live_after.md exists, so the corrected card is already uploaded; nothing to do."; exit 1
fi

uv run python - "$REPO" "$RELEASED_COMMIT" "$RELEASED_SHA" "$DIR/live_before.md" <<'PY'
import hashlib, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
repo, commit, want, keep = sys.argv[1:5]
info = HfApi().model_info(repo)
if info.sha != commit:
    sys.exit(f"Stop: {repo} main is at {info.sha}, not the released commit {commit}; nothing uploaded.")
path = hf_hub_download(repo, "README.md", revision=info.sha)
got = hashlib.sha256(open(path, "rb").read()).hexdigest()
if got != want:
    sys.exit(f"Stop: the live README.md has SHA-256 {got}, not the released one; nothing uploaded.")
import os
if not os.path.exists(keep):
    shutil.copyfile(path, keep)
print(f"{repo} is at the released commit {commit[:10]} and its README.md is the released one.")
PY

echo
echo "The change (< released card, > corrected card):"
echo
diff "$DIR/live_before.md" "$NEW" || true
echo
read -r -p "Upload the corrected README.md to $REPO? Type yes to upload: " ANSWER
case "$ANSWER" in yes|YES|Yes|y|Y) ;; *) echo "Nothing uploaded (the answer was not yes)."; exit 0 ;; esac

uv run hf upload "$REPO" "$NEW" README.md --commit-message "Card: remove the priority sentence (impacte/laya-medical-escalation was published first)"

uv run python - "$REPO" "$NEW_SHA" "$DIR/live_after.md" <<'PY'
import datetime, hashlib, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
repo, want, keep = sys.argv[1:4]
info = HfApi().model_info(repo)
path = hf_hub_download(repo, "README.md", revision=info.sha)
got = hashlib.sha256(open(path, "rb").read()).hexdigest()
shutil.copyfile(path, keep)
now = datetime.datetime.now(datetime.timezone.utc)
print(f"New revision: {info.sha} (checked {now:%Y-%m-%d %H:%M} UTC)")
print("README.md on the page is the corrected card." if got == want else f"MISMATCH: README.md at {info.sha} has SHA-256 {got}.")
print(f"Files in the repository: {len(info.siblings)}")
PY
echo
echo "Done. Paste everything above into the chat so it can be recorded."

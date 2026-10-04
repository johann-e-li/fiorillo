#!/usr/bin/env bash
# Session 5, step 7 (docs/prompt_session5.md): download TheBlueScrubs-v1 files for the web text. Run by Johann only,
# by hand, in Terminal on the project Mac, after reading the dataset's access conditions on Hugging Face
# (https://huggingface.co/datasets/TheBlueScrubs/TheBlueScrubs-v1) and accepting them only if they allow commercial use:
#     bash ~/Documents/Research/ed-decision-model/scripts/fetch_web_text_macos.sh          # README.md and files 1 to 80
#     bash ~/Documents/Research/ed-decision-model/scripts/fetch_web_text_macos.sh --more   # the next 40 files on the list
# The list is src/configs/web_files_session5.txt: 200 of the dataset's 1,200 train files, in a seeded random order
# (Python's random.Random(20260927).shuffle of the sorted names) from its public file list at the fixed revision.
# Downloads go to data/raw/web/thebluescrubs/ (ignored by git) with
#     uv run hf download TheBlueScrubs/TheBlueScrubs-v1 --repo-type dataset --revision <rev> --include <file> ... --local-dir <dir>
# (options checked against `hf download --help` of huggingface_hub 1.33.0, the version in uv.lock: --include takes
# one glob per use and is repeated for each file). Your Hugging Face login supplies the access; the script never
# reads or prints a token. Afterwards it prints each listed file that is present, with its size and SHA-256, and
# writes the same list to data/raw/web/thebluescrubs/SHA256SUMS.txt. Nothing is deleted.
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

REPO=TheBlueScrubs/TheBlueScrubs-v1
REV=255bf6f5eb57e6d8ce4feaa94df0b27a5d13f508
DEST=data/raw/web/thebluescrubs
LIST=src/configs/web_files_session5.txt
mkdir -p "$DEST"

TOTAL=$(grep -c . "$LIST")
HAVE=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if [ -s "$DEST/$f" ]; then HAVE=$((HAVE + 1)); else break; fi
done < "$LIST"

if [ "${1:-}" = "--more" ]; then
  if [ "$HAVE" -lt 80 ]; then echo "Stop: files 1 to 80 are not all here yet ($HAVE are); run without --more first."; exit 1; fi
  FIRST=$((HAVE + 1)); COUNT=40
else
  FIRST=1; COUNT=80
fi
if [ "$FIRST" -gt "$TOTAL" ]; then echo "Every file on the list is already here ($TOTAL)."; exit 0; fi
LAST=$((FIRST + COUNT - 1))
if [ "$LAST" -gt "$TOTAL" ]; then LAST=$TOTAL; fi

ARGS=(--include README.md)
while IFS= read -r f; do
  ARGS+=(--include "$f")
done < <(grep . "$LIST" | sed -n "${FIRST},${LAST}p")
echo "Downloading README.md and files $FIRST to $LAST of the $TOTAL on the list into $DEST ..."
uv run hf download "$REPO" --repo-type dataset --revision "$REV" "${ARGS[@]}" --local-dir "$DEST"

echo
echo "Files on the list that are present (size in bytes, SHA-256):"
SUMS="$DEST/SHA256SUMS.txt"
: > "$SUMS.tmp"
for f in README.md $(grep . "$LIST"); do
  if [ -s "$DEST/$f" ]; then
    SIZE=$(wc -c < "$DEST/$f" | tr -d ' ')
    SHA=$(shasum -a 256 "$DEST/$f" | cut -d' ' -f1)
    printf '%s  %s  %s\n' "$SHA" "$SIZE" "$f" | tee -a "$SUMS.tmp"
  fi
done
mv "$SUMS.tmp" "$SUMS"
echo
echo "Done. $(($(wc -l < "$SUMS") - 1)) of the $TOTAL listed files are here. Tell Claude in the chat."

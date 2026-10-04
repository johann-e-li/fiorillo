#!/usr/bin/env bash
# PubMed annual baseline, the literature corpus for laya-med-base. Public data, no login.
# Run once on the Mac, plugged in with the lid open:
#     caffeinate -dims bash ~/Documents/Research/ed-decision-model/scripts/download_corpus.sh
# Resumable: if it stops or reports a problem, run the same command again. Nothing is deleted.
# Writes only under data/raw/corpus/ (ignored by git). Never reads the restricted folder.
#
# PMC full text is not downloaded here. NCBI removed its bulk full-text packages the week of
# 24 August 2026; full text is now one file per article on AWS and needs a license filter,
# so it gets its own script later.
set -uo pipefail

BASE="${PUBMED_BASE:-https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/}"
ROOT="${CORPUS_ROOT:-$HOME/Documents/Research/ed-decision-model/data/raw/corpus}"
DEST="$ROOT/pubmed-baseline"
LOG="$ROOT/download.log"
LIST="$ROOT/pubmed-baseline.files"
URLS="$ROOT/pubmed-baseline.urls"
NEED_GB="${NEED_GB:-80}"

mkdir -p "$DEST"
say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

if ! command -v wget >/dev/null 2>&1; then say "wget is missing. Run: brew install wget, then start again."; exit 1; fi
free_gb=$(( $(df -Pk "$ROOT" | awk 'NR==2 {print $4}') / 1048576 ))
if [ "$free_gb" -lt "$NEED_GB" ]; then say "Only $free_gb GB free; this needs about $NEED_GB GB. Stopping."; exit 1; fi

say "start: PubMed baseline from $BASE ($free_gb GB free)"
if ! listing="$(curl -fsSL --retry 5 --retry-delay 10 "$BASE")"; then
  say "Could not read $BASE. Check the internet connection and run again."; exit 1
fi
printf '%s\n' "$listing" | grep -oE '"pubmed[0-9]{2}n[0-9]{4}\.xml\.gz(\.md5)?"' | tr -d '"' | sort -u > "$LIST"
if printf '%s\n' "$listing" | grep -q '"README.txt"'; then echo "README.txt" >> "$LIST"; fi
n_xml=$(grep -c '\.xml\.gz$' "$LIST")
n_md5=$(grep -c '\.xml\.gz\.md5$' "$LIST")
say "listing: $n_xml data files and $n_md5 checksum files"
if [ "$n_xml" -lt "${MIN_FILES:-1000}" ]; then say "Expected more than 1,000 data files. Stopping."; exit 1; fi

sed "s#^#$BASE#" "$LIST" > "$URLS"
say "downloading (one file at a time; progress goes to this log)"
( cd "$DEST" && wget -c -nv --tries=10 --retry-connrefused --waitretry=10 -i "$URLS" ) >> "$LOG" 2>&1
say "download pass finished; checking every data file against its md5"

hash_of() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
ok=0; bad=0; missing=0
while read -r f; do
  case "$f" in *.xml.gz) ;; *) continue ;; esac
  if [ ! -s "$DEST/$f" ] || [ ! -s "$DEST/$f.md5" ]; then missing=$((missing + 1)); say "MISSING: $f"; continue; fi
  want="$(grep -oE '[0-9a-f]{32}' "$DEST/$f.md5" | head -1)"
  if [ "$want" = "$(hash_of "$DEST/$f")" ]; then
    ok=$((ok + 1))
  else
    bad=$((bad + 1)); mv -f "$DEST/$f" "$DEST/$f.corrupt"
    say "MD5 MISMATCH: $f (renamed to $f.corrupt so the next run downloads it again)"
  fi
done < "$LIST"

say "checked $n_xml data files: $ok match, $bad mismatched, $missing missing"
say "size on disk: $(du -sh "$DEST" | cut -f1)"
if [ "$bad" -eq 0 ] && [ "$missing" -eq 0 ]; then
  say "DONE: PubMed baseline complete and verified"
else
  say "NOT COMPLETE: run the same command again; it resumes and repairs"; exit 1
fi

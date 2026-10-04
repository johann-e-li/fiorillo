#!/usr/bin/env bash
# Reader v1 on Hugging Face as a PRIVATE repository (docs/prompt_session5.md, correction 6). Run by Johann only, by hand,
# in Terminal on the project Mac:
#     bash ~/Documents/Research/ed-decision-model/scripts/release_reader_v1.sh
# It stops unless: docs/prior_art.md has a release-day check for reader v1 dated today; docs/naming.md names the
# repository; docs/model_card_reader_v1.md holds the card after the line "<!-- model card begins -->"; docs/licenses/
# Apache-2.0.txt is the Apache License 2.0 text (SHA-256 below); the four model
# files in models/reader_v1/ have the SHA-256 values below; models/reader_v1/release/ is empty or absent; the signed-in
# Hugging Face account owns the repository name. Then it assembles models/reader_v1/release/ (config.json,
# model.safetensors, tokenizer.json, tokenizer_config.json, README.md from the card with {{NAME}} and {{REPO}} filled,
# LICENSE and NOTICE), creates the repository as private (an existing repository is used only if it is private and holds
# nothing but .gitattributes), uploads, checks on Hugging Face that it holds exactly these files with the same sizes, the
# weights' SHA-256 and that it is private, and prints the page to review. It never makes anything
# public: that is done by hand on the website. Nothing is deleted. The upload uses the same huggingface_hub commands as
# scripts/release_public_v0.sh.
#
# Dry run (no Hugging Face call; for checking the assembly): EDDM_DRY_RUN=<empty folder> EDDM_PYTHON=python3 bash ...
# assembles README.md and NOTICE in that folder with the repository name from docs/naming.md, and lists the model files
# it would copy.
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model" 2>/dev/null || cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
PY="${EDDM_PYTHON:-uv run python}"
DRY="${EDDM_DRY_RUN:-}"

TODAY="$(date '+%-d %B %Y')"
NAME="$(sed -n 's/^Chosen: \([A-Za-z][A-Za-z]*\).*/\1/p' docs/naming.md | head -1)"
REPO="$(sed -n 's/^Reader v1 release repository.*: \([A-Za-z0-9_.-]*\/[A-Za-z0-9_.-]*\)[[:space:]]*$/\1/p' docs/naming.md | head -1)"
if [ -z "$NAME" ] || [ -z "$REPO" ]; then echo "Stop: docs/naming.md has no chosen name or no reader v1 repository."; exit 1; fi
if [ -z "$DRY" ] && ! grep -q "^### Release-day check, $TODAY (reader v1)" docs/prior_art.md; then
  echo "Stop: docs/prior_art.md has no release-day check for reader v1 dated $TODAY (ask Claude to run it first)."; exit 1
fi
if ! grep -q '^<!-- model card begins -->$' docs/model_card_reader_v1.md 2>/dev/null; then
  echo "Stop: docs/model_card_reader_v1.md is missing or has no line <!-- model card begins -->."; exit 1
fi
SRC=models/reader_v1
$PY - "$SRC" <<'PY'
import hashlib, sys
from pathlib import Path
src = Path(sys.argv[1])
want = {  # correction 6 (a), and the other three files as copied from the pod on 28 September 2026
    "model.safetensors": "28e9a7b62be95615ae6ec80fbb3a92a4694644e27422c3c3048e01ec18c2d201",
    "config.json": "8529a5885302c2e2f10a2cddab87f781a99b199e73e0ca8fe508bd2072ab0105",
    "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12"}
for name, digest in want.items():
    p = src / name
    if not p.is_file():
        sys.exit(f"Stop: {p} is missing.")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != digest:
        sys.exit(f"Stop: {p} has SHA-256 {h.hexdigest()}, not {digest}.")
print("The four model files match their SHA-256 values.")
PY
LICENSE_SRC=docs/licenses/Apache-2.0.txt
LICENSE_SHA="$($PY -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$LICENSE_SRC" 2>/dev/null || true)"
if [ "$LICENSE_SHA" != "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" ]; then
  echo "Stop: $LICENSE_SRC is missing or is not the Apache License 2.0 text."; exit 1
fi
REL="${DRY:-models/reader_v1/release}"
if [ -d "$REL" ] && [ -n "$(ls -A "$REL")" ]; then echo "Stop: $REL is not empty; move it aside first."; exit 1; fi

if [ -z "$DRY" ]; then
  USER_NAME="$($PY -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
  if [ "${REPO%%/*}" != "$USER_NAME" ]; then echo "Stop: signed in to Hugging Face as $USER_NAME, but the repository is $REPO."; exit 1; fi
fi
printf 'Name %s; repository %s (private).\n' "$NAME" "$REPO"

mkdir -p "$REL"
if [ -z "$DRY" ]; then
  cp "$SRC/config.json" "$SRC/model.safetensors" "$SRC/tokenizer.json" "$SRC/tokenizer_config.json" "$REL/"
else
  echo "Dry run: would copy $SRC/config.json, model.safetensors, tokenizer.json and tokenizer_config.json into $REL."
fi
cp "$LICENSE_SRC" "$REL/LICENSE"
sed -n '/^<!-- model card begins -->$/,$p' docs/model_card_reader_v1.md | tail -n +2 > "$REL/README.md"
$PY - "$REL" "$NAME" "$REPO" <<'PY'
import sys
from pathlib import Path
rel, name, repo = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
readme = (rel / "README.md").read_text(encoding="utf-8").replace("{{NAME}}", name).replace("{{REPO}}", repo)
if "{{" in readme or "<<" in readme:
    sys.exit("Stop: README.md still has an unfilled field; nothing uploaded.")
(rel / "README.md").write_text(readme, encoding="utf-8")
notice = f"""{name} reader v1 (ed-decision-model, session 5)
Copyright 2026 Johann Emmanuel Li. Licensed under the Apache License, Version 2.0.

Base models: Laya (convaiinnovations/laya, revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851), Apache License 2.0,
Convai Innovations: its encoder. ModernBERT-large (answerdotai/ModernBERT-large, revision
45bb4654a4d5aaff24dd11d4781fa46d39bf8c13), Apache License 2.0, Answer.AI and LightOn: its masked-language-model head,
and the pretrained model Laya was fine-tuned from. The Apache License 2.0 text is in the file LICENSE.

Text the model was trained on further:
- PubMed citations and abstracts from the PubMed 2026 annual baseline, U.S. National Library of Medicine (NLM), used
  under NLM's terms for PubMed; the abstracts keep their own copyright. No abstract text is distributed with this model.

This model was trained in part on data from the U.S. National Library of Medicine (NLM): PubMed citations and abstracts.
NLM is not responsible for this model and has not endorsed it, and the data it was trained on may not reflect the most
current data available from NLM.
PubMed is a registered trademark of the U.S. Department of Health and Human Services (HHS). The name is used here only to
identify the source of the training data and implies no endorsement by NLM, the National Institutes of Health or HHS.

Excluded from the training text: every article in PubMedQA (MIT License. Jin Q, Dhingra B, Liu Z, Cohen W, Lu X.
PubMedQA: A Dataset for Biomedical Research Question Answering. Proceedings of EMNLP-IJCNLP 2019: 2567-2577).

No MIMIC, NRD, NIS or NHAMCS data, and no other clinical records, were used in the further training of this model.
"""
(rel / "NOTICE").write_text(notice, encoding="utf-8")
print(f"README.md and NOTICE written for {name} reader v1 ({repo}).")
PY
if [ -n "$DRY" ]; then echo "Dry run finished: $REL"; exit 0; fi

$PY - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api, repo = HfApi(), sys.argv[1]
api.create_repo(repo, private=True, exist_ok=True)
info = api.repo_info(repo)
if not info.private:
    sys.exit(f"Stop: {repo} already exists and is public; nothing uploaded.")
extra = sorted(s.rfilename for s in info.siblings if s.rfilename != ".gitattributes")
if extra:
    sys.exit(f"Stop: {repo} already holds files {extra[:5]}; nothing uploaded.")
print(f"{repo} exists, is private and holds no files yet.")
PY
if [ -n "$(find "$REL" -name '.*' -print -quit)" ]; then echo "Stop: $REL holds a hidden file; nothing uploaded."; exit 1; fi
uv run hf upload "$REPO" "$REL" . --commit-message "$NAME reader v1 (ed-decision-model, session 5)"
$PY - "$REPO" "$REL" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, rel = sys.argv[1], Path(sys.argv[2])
info = HfApi().model_info(repo, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
local = {p.name for p in rel.iterdir()}
bad = [f"not in the release folder: {name}" for name in sorted(remote) if name not in local and name != ".gitattributes"]
if not info.private:
    bad.append("the repository is not private")
for p in sorted(rel.iterdir()):
    s = remote.get(p.name)
    if s is None or s.size != p.stat().st_size:
        bad.append(p.name)
lfs = remote.get("model.safetensors").lfs if remote.get("model.safetensors") else None
sha = (lfs.sha256 if hasattr(lfs, "sha256") else (lfs or {}).get("sha256")) if lfs else None
if sha != "28e9a7b62be95615ae6ec80fbb3a92a4694644e27422c3c3048e01ec18c2d201":
    bad.append(f"model.safetensors SHA-256 on Hugging Face: {sha}")
if bad:
    sys.exit(f"Check failed on Hugging Face: {bad}")
print(f"On Hugging Face at revision {info.sha}: every file's size matches, and the weights' SHA-256 matches. Private: {info.private}.")
PY
printf '\nUploaded to the private repository https://huggingface.co/%s\nReview it there; make it public by hand only when it is right.\nPaste everything above into the chat so it can be recorded.\n' "$REPO"

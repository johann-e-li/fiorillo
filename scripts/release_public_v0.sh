#!/usr/bin/env bash
# Session 4, step 10 (docs/prompt_session4.md): put public v0 on Hugging Face as a PRIVATE repository. Run by Johann
# only, by hand, in Terminal on the project Mac, after the comparison exists and the model card is written:
#     bash ~/Documents/Research/ed-decision-model/scripts/release_public_v0.sh
# It stops unless: docs/naming.md has a chosen name; docs/prior_art.md has a check dated today; the release bar in
# results/public_v0/comparison.json has all three criteria PASS; docs/model_card_v0.md holds the model card after the
# line "<!-- model card begins -->"; models/public_v0/arm_laya/ holds the model; models/public_v0/release/ is empty or
# absent. Then it assembles models/public_v0/release/ (weights, configs, calibration.json, tokenizer, README.md from the
# card, with {{NAME}} and {{REPO}} filled in, NOTICE with the attributions), creates <username>/<name in lower case>-v0.1 as a private repository (an existing
# repository is used only if it is private), uploads, and prints the page to review. It never makes anything public:
# that is done by hand on the website. Nothing is deleted. The hf commands were checked against huggingface_hub 1.33.0,
# the version in uv.lock (hf upload REPO_ID LOCAL_PATH PATH_IN_REPO --commit-message; creation and the privacy check use
# the Python API of the same package).
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

TODAY="$(date '+%-d %B %Y')"
NAME="$(sed -n 's/^Chosen: \([A-Za-z][A-Za-z]*\).*/\1/p' docs/naming.md | head -1)"
if [ -z "$NAME" ] || [ "$NAME" = "awaiting" ]; then echo "Stop: docs/naming.md has no chosen name."; exit 1; fi
if ! grep -q "^###.*$TODAY" docs/prior_art.md; then echo "Stop: docs/prior_art.md has no check dated $TODAY."; exit 1; fi
uv run python - <<'PY'
import json, sys
bar = json.load(open("results/public_v0/comparison.json"))["release_bar"]
failed = [k for k, c in bar["criteria"].items() if c["result"] != "PASS"]
if failed or bar["overall"] != "PASS":
    sys.exit(f"Stop: release criteria not met: {failed}")
print("Release bar: all three criteria PASS.")
PY
if ! grep -q '^<!-- model card begins -->$' docs/model_card_v0.md 2>/dev/null; then
  echo "Stop: docs/model_card_v0.md is missing or has no line <!-- model card begins -->."; exit 1
fi
SRC=models/public_v0/arm_laya
for f in model.safetensors rl_agent_config.json calibration.json encoder/config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
  if [ ! -s "$SRC/$f" ]; then echo "Stop: $SRC/$f is missing."; exit 1; fi
done
REL=models/public_v0/release
if [ -d "$REL" ] && [ -n "$(ls -A "$REL")" ]; then echo "Stop: $REL is not empty; move it aside first."; exit 1; fi

USER_NAME="$(uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
REPO="$USER_NAME/$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]')-v0.1"
printf 'Name %s; account %s; repository %s (private).\n' "$NAME" "$USER_NAME" "$REPO"

mkdir -p "$REL/encoder" "$REL/tokenizer"
cp "$SRC/model.safetensors" "$SRC/rl_agent_config.json" "$SRC/calibration.json" "$REL/"
cp "$SRC"/encoder/* "$REL/encoder/"
cp "$SRC"/tokenizer/* "$REL/tokenizer/"
sed -n '/^<!-- model card begins -->$/,$p' docs/model_card_v0.md | tail -n +2 > "$REL/README.md"
uv run python - "$REL/README.md" "$NAME" "$REPO" <<'PY'
import sys
path, name, repo = sys.argv[1:4]
text = open(path, encoding="utf-8").read().replace("{{NAME}}", name).replace("{{REPO}}", repo)
if "{{" in text:
    sys.exit("Stop: README.md still has an unfilled {{...}} field; nothing uploaded.")
open(path, "w", encoding="utf-8").write(text)
print(f"README.md written from docs/model_card_v0.md for {name} ({repo}).")
PY
cat > "$REL/NOTICE" <<EOF
$NAME v0.1 (ed-decision-model, public v0)
Copyright 2026 Johann Emmanuel Li. Licensed under the Apache License, Version 2.0.

Base model: Laya (convaiinnovations/laya, revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851), Apache License 2.0,
ConvAI Innovations; built on ModernBERT-large (answerdotai/ModernBERT-large), Apache License 2.0, Answer.AI and LightOn.

Training data:
- PubMedQA, MIT License. Jin Q, Dhingra B, Liu Z, Cohen W, Lu X. PubMedQA: A Dataset for Biomedical Research
  Question Answering. Proceedings of EMNLP-IJCNLP 2019: 2567-2577.
- Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository, Creative Commons Attribution 4.0.
  Clore J, Cios K, DeShazo J, Strack B (2014). https://doi.org/10.24432/C5230J. Strack B, DeShazo JP, Gennings C,
  Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of
  70,000 Clinical Database Patient Records. BioMed Research International 2014: 781670.

No MIMIC, NRD, NIS or NHAMCS data were used to train this model.
EOF

uv run python - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api, repo = HfApi(), sys.argv[1]
api.create_repo(repo, private=True, exist_ok=True)
if not api.repo_info(repo).private:
    sys.exit(f"Stop: {repo} already exists and is public; nothing uploaded.")
print(f"{repo} exists and is private.")
PY
uv run hf upload "$REPO" "$REL" . --commit-message "public v0.1 (ed-decision-model session 4)"
printf '\nUploaded to the private repository https://huggingface.co/%s\nReview it there; make it public by hand only when it is right.\n' "$REPO"

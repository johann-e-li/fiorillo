#!/usr/bin/env bash
# Session 5, step 15 (docs/prompt_session5.md): put v0.2 on Hugging Face as a PRIVATE repository. Run by Johann only, by
# hand, in Terminal on the project Mac, after scripts/finish_session5_macos.sh, the model card and the release-day check:
#     bash ~/Documents/Research/ed-decision-model/scripts/release_fiorillo_v0_2.sh
# It stops unless: results/fiorillo_v0_2/comparison.json shows all four release criteria PASS; docs/prior_art.md has a
# release-day check dated today; models/fiorillo_v0_2/release/ is empty or absent; docs/naming.md has the chosen name;
# docs/model_card_v0_2.md holds the card after the line "<!-- model card begins -->"; models/fiorillo_v0_2/candidate/
# holds the model; models/fiorillo_v0_2/pmc_training_manifest.csv and the web manifest's part list exist; the card's
# paragraph on the other Laya-family models is dated today. Then it assembles models/fiorillo_v0_2/release/
# (weights and configs, calibration.json, tokenizer, README.md from the card with {{NAME}} and {{REPO}} filled, NOTICE with
# the attributions and terms of the text and data actually used, and training_manifest.csv: every PMC article fetched in
# session 5 with its PMCID, version and licence, a superset of any used, and the web part files with their document
# counts), creates <username>/<name in lower case>-v0.2 as a private repository (an existing repository is used
# only if it is private), uploads, and prints the page to review. It never makes anything public: that is done by hand
# on the website. Nothing is deleted. The upload uses the same huggingface_hub commands as scripts/release_public_v0.sh
# (huggingface_hub 1.33.0, as in uv.lock).
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model"
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

TODAY="$(date '+%-d %B %Y')"
NAME="$(sed -n 's/^Chosen: \([A-Za-z][A-Za-z]*\).*/\1/p' docs/naming.md | head -1)"
if [ -z "$NAME" ] || [ "$NAME" = "awaiting" ]; then echo "Stop: docs/naming.md has no chosen name."; exit 1; fi
if ! grep -q "^### Release-day check.*$TODAY" docs/prior_art.md; then echo "Stop: docs/prior_art.md has no release-day check dated $TODAY (run it first)."; exit 1; fi
if ! grep -q "Hugging Face check, $TODAY" docs/model_card_v0_2.md 2>/dev/null; then
  echo "Stop: the card's paragraph on the other Laya-family models is not from today's check; update it in"
  echo "scripts/build_model_card_v0_2.py (FAMILY_CHECK_DATE, FAMILY_MODELS) and build the card again."; exit 1
fi
uv run python - <<'PY'
import json, sys
bar = json.load(open("results/fiorillo_v0_2/comparison.json", encoding="utf-8"))["release_bar"]
failed = [k for k, c in bar["criteria"].items() if c["result"] != "PASS"]
if failed or bar["overall"] != "PASS" or sorted(bar["criteria"]) != ["1", "2", "3", "4"]:
    sys.exit(f"Stop: release criteria not met: {failed or bar['overall']}")
print(f"Release bar: all four criteria PASS for {bar['system']}.")
PY
if ! grep -q '^<!-- model card begins -->$' docs/model_card_v0_2.md 2>/dev/null; then
  echo "Stop: docs/model_card_v0_2.md is missing or has no line <!-- model card begins -->."; exit 1
fi
SRC=models/fiorillo_v0_2/candidate
for f in model.safetensors rl_agent_config.json calibration.json encoder/config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
  if [ ! -s "$SRC/$f" ]; then echo "Stop: $SRC/$f is missing."; exit 1; fi
done
if [ ! -s models/fiorillo_v0_2/pmc_training_manifest.csv ]; then echo "Stop: models/fiorillo_v0_2/pmc_training_manifest.csv is missing."; exit 1; fi
uv run python - <<'PY'
import json, sys
try:
    parts = json.load(open("results/reader_v1/text_manifests/web_manifest.json", encoding="utf-8"))["parts"]
except (OSError, KeyError, ValueError) as e:
    sys.exit(f"Stop: results/reader_v1/text_manifests/web_manifest.json has no list of part files ({e}); nothing assembled.")
if not parts:
    sys.exit("Stop: the web manifest lists no part files; nothing assembled.")
PY
REL=models/fiorillo_v0_2/release
if [ -d "$REL" ] && [ -n "$(ls -A "$REL")" ]; then echo "Stop: $REL is not empty; move it aside first."; exit 1; fi

USER_NAME="$(uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
REPO="$USER_NAME/$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]')-v0.2"
printf 'Name %s; account %s; repository %s (private).\n' "$NAME" "$USER_NAME" "$REPO"

mkdir -p "$REL/encoder" "$REL/tokenizer"
cp "$SRC/model.safetensors" "$SRC/rl_agent_config.json" "$SRC/calibration.json" "$REL/"
cp "$SRC"/encoder/* "$REL/encoder/"
cp "$SRC"/tokenizer/* "$REL/tokenizer/"
sed -n '/^<!-- model card begins -->$/,$p' docs/model_card_v0_2.md | tail -n +2 > "$REL/README.md"
uv run python - "$REL" "$NAME" "$REPO" <<'PY'
import csv, json, sys
from pathlib import Path
import yaml
rel, name, repo = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

readme = (rel / "README.md").read_text(encoding="utf-8").replace("{{NAME}}", name).replace("{{REPO}}", repo)
if "{{" in readme:
    sys.exit("Stop: README.md still has an unfilled {{...}} field; nothing uploaded.")
(rel / "README.md").write_text(readme, encoding="utf-8")
print(f"README.md written from docs/model_card_v0_2.md for {name} ({repo}).")

sel = json.load(open("results/fiorillo_v0_2/release_candidate.json", encoding="utf-8"))["selection"]
reader_rule, task_rule = sel["reader_rule"], sel.get("task_rule", {})
mixes = yaml.safe_load(open("src/configs/mixes_session5.yaml", encoding="utf-8"))["mixes"]
used = set(mixes[reader_rule["winner"]]["shares"]) if reader_rule.get("long_run") else set()
if task_rule.get("task_round_kept"):
    used.add("task")
pmc_used = bool(used & {"pmc", "case_reports"})
ack = ("This model was trained in part on data from the U.S. National Library of Medicine (NLM): PubMed citations and abstracts"
       + (", and full-text articles from PubMed Central (PMC)" if pmc_used else "")
       + ". NLM is not responsible for this model and has not endorsed it, and the data it was trained on may not reflect the "
       "most current data available from NLM.")
text_lines = []
if "abstracts" in used:
    text_lines.append("- PubMed citations and abstracts from the PubMed 2026 annual baseline, U.S. National Library of Medicine (NLM),\n"
                      "  used under NLM's terms for PubMed; the abstracts keep their own copyright.")
if pmc_used:
    text_lines.append("- Full-text articles from the PubMed Central (PMC) Open Access Subset licensed CC0 or CC BY, NLM, used under NLM's\n"
                      "  terms for PMC; each article's identifier, version and licence (its attribution) are in training_manifest.csv.")
if "web" in used:
    text_lines.append("- TheBlueScrubs-v1 (TheBlueScrubs/TheBlueScrubs-v1, revision 255bf6f5eb57e6d8ce4feaa94df0b27a5d13f508), Apache\n"
                      "  License 2.0 for the collection; its texts keep their sources' terms: Common Crawl's terms of use, and CC BY-SA\n"
                      "  for Wikipedia and StackExchange texts. The part files built from it are listed in training_manifest.csv.")
if "task" in used:
    text_lines.append("- The states of the decision-layer training records below (no validation or test record).")
if not text_lines:
    text_lines.append("- None: v0.2 keeps Laya's own reader.")
notice = f"""{name} v0.2 (ed-decision-model, session 5)
Copyright 2026 Johann Emmanuel Li. Licensed under the Apache License, Version 2.0.

Base models: Laya (convaiinnovations/laya, revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851), Apache License 2.0,
ConvAI Innovations; built on ModernBERT-large (answerdotai/ModernBERT-large, revision 45bb4654a4d5aaff24dd11d4781fa46d39bf8c13,
whose masked-language-model head was used to train the reader further), Apache License 2.0, Answer.AI and LightOn.

Text the reader was trained on further (reader v1):
{chr(10).join(text_lines)}

{ack}
The PubMed and PubMed Central names, wordmarks and logos are NLM's; their use here names the data sources only and implies
no endorsement by NLM, the National Institutes of Health or the U.S. Department of Health and Human Services.

Decision-layer training data:
- PubMedQA, MIT License. Jin Q, Dhingra B, Liu Z, Cohen W, Lu X. PubMedQA: A Dataset for Biomedical Research
  Question Answering. Proceedings of EMNLP-IJCNLP 2019: 2567-2577.
- Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository, Creative Commons Attribution 4.0.
  Clore J, Cios K, DeShazo J, Strack B (2014). https://doi.org/10.24432/C5230J. Strack B, DeShazo JP, Gennings C,
  Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of
  70,000 Clinical Database Patient Records. BioMed Research International 2014: 781670.

training_manifest.csv lists every PMC article fetched while the text was built (a superset of any used, with its PMCID,
version and licence) and the web part files built from TheBlueScrubs-v1, with their document counts.

No MIMIC, NRD, NIS or NHAMCS data were used to train this model.
"""
(rel / "NOTICE").write_text(notice, encoding="utf-8")

rows = 0
with open(rel / "training_manifest.csv", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["source", "identifier", "version", "license_code", "tokens", "documents"])
    with open("models/fiorillo_v0_2/pmc_training_manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r = {k.lower(): v for k, v in r.items()}
            w.writerow([r.get("source", "pmc"), r["pmcid"], r.get("version", ""), r.get("license_code", ""), r.get("tokens", ""), ""])
            rows += 1
    web = json.load(open("results/reader_v1/text_manifests/web_manifest.json", encoding="utf-8"))
    for part in web["parts"]:
        w.writerow(["web", f"web/{part['file']} (from TheBlueScrubs/TheBlueScrubs-v1)", "255bf6f5eb57e6d8ce4feaa94df0b27a5d13f508",
                    "", part.get("tokens", ""), part["documents"]])
        rows += 1
print(f"NOTICE written (reader text used: {sorted(used) or 'none'}); training_manifest.csv: {rows:,} rows.")
PY

uv run python - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api, repo = HfApi(), sys.argv[1]
api.create_repo(repo, private=True, exist_ok=True)
if not api.repo_info(repo).private:
    sys.exit(f"Stop: {repo} already exists and is public; nothing uploaded.")
print(f"{repo} exists and is private.")
PY
uv run hf upload "$REPO" "$REL" . --commit-message "v0.2 (ed-decision-model session 5)"
printf '\nUploaded to the private repository https://huggingface.co/%s\nReview it there; make it public by hand only when it is right.\n' "$REPO"

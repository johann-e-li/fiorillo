#!/usr/bin/env bash
# Session 7, step 5 (docs/prompt_session7.md): the Mac-training check (idea I-24). Johann runs this in Terminal on the Mac:
#     bash scripts/check_mac_training_macos.sh
# It (1) checks that models/laya-base/ holds Laya's fixed revision; (2) downloads Qwen3-4B-Base at its fixed revision into
# models/qwen3_4b_base/ with `uv run hf download` (the model is public, so no token is needed; options checked against
# `hf download --help` of huggingface_hub 1.33.0 on 29 September 2026); (3) puts reader v1 into Laya's format in
# models/fiorillo_v0_3/reader_v1_laya_format/ with src/train/assemble_reader.py; (4) runs src/train/mac_training_check.py
# under caffeinate -i, which writes results/fiorillo_v0_3/mac_training_check.json. Everything it writes under models/ is
# ignored by git. It reads only this project's public records, and it stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

QWEN_REV=906bfd4b4dc7f14ee4320094d8b41684abff8539
LAYA_REV=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
READER_SHA=28e9a7b62be95615ae6ec80fbb3a92a4694644e27422c3c3048e01ec18c2d201

echo "1. Laya's folder"
grep -q "revision $LAYA_REV" models/laya-base/REVISION.txt || { echo "models/laya-base/REVISION.txt does not name Laya's revision $LAYA_REV; stopping"; exit 1; }

echo "2. Qwen3-4B-Base at $QWEN_REV (about 8 GB)"
uv run hf download Qwen/Qwen3-4B-Base --revision "$QWEN_REV" --local-dir models/qwen3_4b_base
printf 'revision %s\ndownloaded %s with hf download (Qwen/Qwen3-4B-Base)\n' "$QWEN_REV" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/qwen3_4b_base/REVISION.txt

echo "3. reader v1 in Laya's format"
got=$(shasum -a 256 models/reader_v1/model.safetensors | cut -d' ' -f1)
[ "$got" = "$READER_SHA" ] || { echo "models/reader_v1/model.safetensors has SHA-256 $got, expected $READER_SHA; stopping"; exit 1; }
uv run python src/train/assemble_reader.py --laya-dir models/laya-base --reader-dir models/reader_v1 \
    --out models/fiorillo_v0_3/reader_v1_laya_format --laya-revision "$LAYA_REV" --reader-revision "reader v1, model.safetensors $READER_SHA"

echo "4. the check (about 15 to 40 minutes; the Mac stays awake)"
caffeinate -i uv run python src/train/mac_training_check.py
echo "done: results/fiorillo_v0_3/mac_training_check.json"

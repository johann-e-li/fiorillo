"""Practice fine-tune, data builder: the DAIR Emotion dataset as Laya training items.

Downloads dair-ai/emotion (16,000 train, 2,000 validation, 2,000 test short English texts, six labels) and
builds one Laya item per row with laya.common.build_sequence: the state is the text; the question is a choice
question, "Which emotion does this text express?", over the six labels in dataset order; the target is
one-hot on the gold label. Items are stored as the upstream notebook stores them (ids, markers, qtype, target,
label) in data/processed/practice/emotion_<split>.pt, with emotion_counts.json beside them.

Usage: python src/train/practice_data.py [--model-dir models/laya-base] [--out data/processed/practice] [--max-rows N]
"""
import argparse
import json
from pathlib import Path

import torch
from laya.common import QTYPES, build_sequence, render_options
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CRITERIA = {
    "sadness": "sad, hurt, lonely, disappointed or grieving",
    "joy": "happy, pleased, content, excited or proud",
    "love": "loving, affectionate, caring or tender toward someone",
    "anger": "angry, irritated, resentful or frustrated",
    "fear": "afraid, anxious, nervous or worried",
    "surprise": "surprised, amazed or startled",
}
QUESTION = {"t": "choice", "ins": "Which emotion does this text express?", "crit": CRITERIA}
SPLITS = ("train", "validation", "test")


def load_tokenizer_and_config(model_dir):
    """The tokenizer and the two sequence-length settings from a Laya model folder."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    config = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    return tokenizer, config


def build_item(tokenizer, text, label_index, max_len=512, head_max_len=192):
    """One training item, or None when the six options do not fit the head budget."""
    k = len(render_options(QUESTION))
    ids, markers = build_sequence(tokenizer, text, QUESTION, max_len, head_max_len)
    if len(markers) != k:
        return None
    target = [0.0] * k
    target[label_index] = 1.0
    return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target, "label": label_index}


def build_split(rows, tokenizer, max_len, head_max_len, max_rows=None):
    items, skipped = [], 0
    for i, row in enumerate(rows):
        if max_rows is not None and i >= max_rows:
            break
        item = build_item(tokenizer, row["text"], int(row["label"]), max_len, head_max_len)
        if item is None:
            skipped += 1
        else:
            items.append(item)
    return items, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "laya-base")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "practice")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    from datasets import load_dataset

    tokenizer, config = load_tokenizer_and_config(args.model_dir)
    max_len, head_max_len = config.get("max_len", 512), config.get("head_max_len", 192)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"labels": LABELS, "max_len": max_len, "head_max_len": head_max_len, "splits": {}}
    for split in SPLITS:
        rows = load_dataset("dair-ai/emotion", "split", split=split)
        assert rows.features["label"].names == LABELS, rows.features["label"].names
        items, skipped = build_split(rows, tokenizer, max_len, head_max_len, args.max_rows)
        torch.save(items, args.out / f"emotion_{split}.pt")
        counts["splits"][split] = {"rows": len(rows), "rows_used": min(len(rows), args.max_rows or len(rows)), "items": len(items), "skipped": skipped}
        print(f"{split}: {len(items):,} items, {skipped} skipped")
    (args.out / "emotion_counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

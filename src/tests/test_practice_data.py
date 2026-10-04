"""The practice data builder on three invented rows, with the pinned Laya tokenizer."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))
TOKENIZER_DIR = SRC.parent / "models" / "tokenizers" / "laya"

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="Laya tokenizer not downloaded")


def test_three_invented_rows_build_three_items():
    from transformers import AutoTokenizer

    from src.train.practice_data import LABELS, build_split

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    rows = [
        {"text": "i cannot stop smiling today everything went right", "label": 1},
        {"text": "i keep thinking something terrible is about to happen", "label": 4},
        {"text": "i miss them so much it hurts", "label": 0},
    ]
    items, skipped = build_split(rows, tokenizer, max_len=512, head_max_len=192)
    assert skipped == 0 and len(items) == 3
    for item, row in zip(items, rows):
        assert len(item["markers"]) == len(LABELS) == 6
        assert item["qtype"] == 0
        assert sum(item["target"]) == 1.0 and item["target"][row["label"]] == 1.0
        assert item["label"] == row["label"] == item["target"].index(max(item["target"]))
        assert len(item["ids"]) <= 512

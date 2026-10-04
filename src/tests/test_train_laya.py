"""Unit checks for the trainer's loss, temperature fit and collate, on synthetic tensors (no model download)."""
import math
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))

from src.train.train_laya import collate, fit_temperature, log_loss  # noqa: E402


def test_log_loss_matches_hand_computation():
    logits = torch.tensor([[2.0, 0.0, -1e4], [0.5, 0.5, 0.5]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    row1 = -math.log(math.exp(2.0) / (math.exp(2.0) + 1.0))
    row2 = -math.log(1.0 / 3.0)
    assert abs(float(log_loss(logits, target, mask)) - (row1 + row2) / 2) < 1e-5


def test_temperature_fit_softens_overconfident_logits():
    torch.manual_seed(0)
    logits, targets = [], []
    for i in range(200):
        z = [0.0, 0.0]
        gold = i % 2
        z[gold] = 6.0                      # very confident
        if i % 4 == 0:                     # but wrong a quarter of the time
            z = [z[1], z[0]]
        logits.append(z)
        t = [0.0, 0.0]
        t[gold] = 1.0
        targets.append(t)
    assert fit_temperature(logits, targets) > 1.0


def test_collate_pads_mixed_batch():
    items = [
        {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0, "target": [1.0, 0.0], "label": 0},
        {"ids": [1, 2, 3, 4, 5], "markers": [1, 2, 3], "qtype": 2, "target": [0.2, 0.3, 0.5], "label": 2},
    ]
    batch = collate(items, pad_id=0)
    assert batch["input_ids"].shape == (2, 5) and batch["input_ids"][0].tolist() == [1, 2, 3, 0, 0]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["marker_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["target"][0].tolist() == [1.0, 0.0, 0.0]
    assert batch["qtype"].tolist() == [0, 2] and batch["label"].tolist() == [0, 2]


def test_maybe_weight_on_made_up_records():
    from src.train.train_laya import item_weight, maybe_weight

    items = ([{"qid": "pubmedqa_answer", "label": 2, "source": "pqa_labeled"}] * 3
             + [{"qid": "pubmedqa_answer", "label": 0, "source": "pqa_labeled"}] * 5
             + [{"qid": "pubmedqa_answer", "label": 1, "source": "pqa_labeled"}] * 2
             + [{"qid": "pubmedqa_answer", "label": 0, "source": "pqa_artificial"}] * 40
             + [{"qid": "readmit_window", "label": 2, "source": None}] * 9)
    w, counts = maybe_weight(items)
    assert w == 47 / 7 and counts == {"pubmedqa_records": 50, "maybe_records": 3, "other_records": 47, "pqa_labeled_not_maybe": 7}
    labeled_share = 3 / 10
    assert abs(3 * w / (3 * w + 47) - labeled_share) < 1e-12  # maybe's weighted share equals its pqa_labeled share
    assert item_weight(items[0], w) == w and item_weight(items[3], w) == 1.0 and item_weight(items[-1], w) == 1.0
    assert item_weight(items[0], None) == 1.0
    assert abs(20924 / 924 - 22.645) < 0.001  # session 4's records: the prompt's 22.65


def test_weighted_loss_is_the_weighted_sum_over_the_row_count():
    from src.train.train_laya import weighted_log_loss

    logits = torch.tensor([[2.0, 0.0, -1e4], [0.5, 0.5, 0.5]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    row1 = -math.log(math.exp(2.0) / (math.exp(2.0) + 1.0))
    row2 = -math.log(1.0 / 3.0)
    ones = torch.ones(2)
    assert abs(float(weighted_log_loss(logits, target, mask, ones)) - float(log_loss(logits, target, mask))) < 1e-6
    assert abs(float(weighted_log_loss(logits, target, mask, torch.tensor([1.0, 22.65]))) - (row1 + 22.65 * row2) / 2) < 1e-4

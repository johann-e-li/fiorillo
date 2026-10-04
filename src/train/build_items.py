"""Tokenize public v0 records into Laya items for one model folder (docs/prompt_session4.md, step 6a).

Records (src/tasks/) hold a JSON-object state and a qid; the question bank (src/questions/public_v0.yaml) gives
each qid's type, instructions and criteria. Items are built with laya.common.build_sequence, which accepts a JSON
object as state and serializes it with json.dumps (ensure_ascii False), so the state rendering is laya's own.
An item: ids, markers, qtype, target, label, qid, id, calibrate, and source (the record's group.source, such as
pqa_labeled or pqa_artificial, or None; added in session 5 for train_laya.py --maybe-weight auto, and not read by the
model). Records whose options do not all fit the head budget are skipped and counted; records whose state was cut to
fit max_len are counted as truncated.

Usage: python src/train/build_items.py --records data/processed/public/readmit_window_train.jsonl \
    data/processed/public/pubmedqa_answer_train.jsonl --model-dir models/laya-base \
    --out data/processed/public/items/laya_train.pt [--max-len 768] [--head-max-len 192]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from laya.common import QTYPES, build_sequence, render_options, serialize_state
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "src" / "questions" / "public_v0.yaml"


def load_questions(path=QUESTIONS):
    """qid -> Laya's internal question form {t, ins, crit}."""
    bank = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]
    return {qid: {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]} for qid, q in bank.items()}


def read_records(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def build_items(records, questions, tokenizer, max_len, head_max_len):
    items, skipped, truncated = [], 0, 0
    empty = {}
    for r in records:
        q = questions[r["qid"]]
        n_options = len(render_options(q))
        if len(r["target"]) != n_options:
            raise ValueError(f"{r['id']}: target has {len(r['target'])} options, question {r['qid']} has {n_options}")
        state_ids = tokenizer(serialize_state(r["state"]).replace(tokenizer.mask_token, " "), add_special_tokens=False)["input_ids"]
        ids, markers = build_sequence(tokenizer, r["state"], q, max_len, head_max_len, state_ids=state_ids)
        if len(markers) != n_options:
            skipped += 1
            continue
        if r["qid"] not in empty:
            empty[r["qid"]] = len(build_sequence(tokenizer, "", q, max_len, head_max_len, state_ids=[])[0])
        if empty[r["qid"]] + len(state_ids) > max_len:
            truncated += 1
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": r["target"], "label": r["label"],
                      "qid": r["qid"], "id": r["id"], "calibrate": bool(r.get("calibrate", True)),
                      "source": (r.get("group") or {}).get("source")})
    return items, {"records": len(records), "items": len(items), "skipped": skipped, "truncated": truncated}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "laya-base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--head-max-len", type=int, default=None)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    max_len = args.max_len or config.get("max_len", 512)
    head_max_len = args.head_max_len or config.get("head_max_len", 192)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "tokenizer"))
    items, summary = build_items(read_records(args.records), load_questions(args.questions), tokenizer, max_len, head_max_len)
    summary.update({"max_len": max_len, "head_max_len": head_max_len, "files": [str(p) for p in args.records]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, args.out)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())

"""Items for arm B, Laya's decision head on Qwen3-4B-Base (docs/prompt_session7.md, ARM B, "Items").

The same records and question bank as arm A (src/train/build_items.py), rendered here rather than through
laya.common.build_sequence, because Qwen's tokenizer has no mask, class or separator token. Order and text:

    <state>  "\\n\\n"  "<type> question: <instructions>"  "\\n"  " <option 0>" "\\n"  " <option 1>" "\\n"  ...

- the state is serialized as laya.common.serialize_state does (json.dumps, ensure_ascii False), unchanged (no token is
  replaced, since Qwen has no mask token);
- "<type> question: <instructions>" are the words build_sequence uses;
- each option is laya.common.render_options' text with a leading space, cut to 48 tokens as build_sequence cuts it,
  and is followed by one "\\n" token (ID 198 in Qwen's tokenizer), whose position is that option's marker.
Each part is tokenized on its own, without special tokens, and the IDs are joined. A decoder reads left to right, so a
marker must come after the record and its own option to have read them; Laya's head, which reads every position in
both directions, then lets the markers read each other.

Length: the state is cut from its end so that the sequence fits max_len (Laya's rule); the question and the options
are never cut. A record whose question and options alone do not fit is skipped. Skipped records and records whose
state was cut are counted. An item has the fields build_items.py gives (ids, markers, qtype, target, label, qid, id,
calibrate, source), so train_laya.py's collate() pads it; padding goes on the right with the tokenizer's pad token.

Usage (on the pod, step 8's prepare):
    python src/train/qwen_items.py --records data/processed/public/readmit_window_train.jsonl ... \\
        --tokenizer-dir /root/eddm_s7/models/qwen3_4b_base --out /root/eddm_s7/items/b_phase3_train.pt [--max-len 768]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from laya.common import QTYPES, render_options, serialize_state

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
from build_items import QUESTIONS, load_questions, read_records  # noqa: E402

OPTION_CAP = 48
STATE_CHUNK = 512


def single_token(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens in this tokenizer, not 1")
    return ids[0]


class QwenFormat:
    """The fixed parts of one question's sequence, tokenized once."""

    def __init__(self, tokenizer, q):
        self.newline = single_token(tokenizer, "\n")
        self.gap = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        self.head = tokenizer("%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)["input_ids"]
        self.options = [tokenizer(" " + o, add_special_tokens=False, truncation=True, max_length=OPTION_CAP)["input_ids"]
                        for o in render_options(q)]
        tail = list(self.gap) + list(self.head) + [self.newline]
        self.offsets = []
        for o in self.options:
            tail.extend(o)
            self.offsets.append(len(tail))
            tail.append(self.newline)
        self.tail = tail

    def sequence(self, state_ids, max_len):
        """(ids, markers, cut) for one state's token IDs, or None if the question and options alone do not fit."""
        room = max_len - len(self.tail)
        if room < 0:
            return None
        st = list(state_ids[:room])
        ids = st + self.tail
        markers = [len(st) + off for off in self.offsets]
        return ids, markers, len(state_ids) > room


def build_qwen_sequence(tokenizer, state, q, max_len=768):
    """One sequence, for tests and single use: (ids, markers, cut), or None."""
    fmt = QwenFormat(tokenizer, q)
    state_ids = tokenizer(serialize_state(state), add_special_tokens=False)["input_ids"]
    return fmt.sequence(state_ids, max_len)


def build_qwen_items(records, questions, tokenizer, max_len=768):
    formats, items = {}, []
    summary = {"records": len(records), "items": 0, "skipped": 0, "truncated": 0}
    for start in range(0, len(records), STATE_CHUNK):
        chunk = records[start : start + STATE_CHUNK]
        states = tokenizer([serialize_state(r["state"]) for r in chunk], add_special_tokens=False)["input_ids"]
        for r, state_ids in zip(chunk, states):
            q = questions[r["qid"]]
            if r["qid"] not in formats:
                formats[r["qid"]] = QwenFormat(tokenizer, q)
            fmt = formats[r["qid"]]
            if len(r["target"]) != len(fmt.options):
                raise ValueError(f"{r['id']}: target has {len(r['target'])} options, question {r['qid']} has {len(fmt.options)}")
            built = fmt.sequence(state_ids, max_len)
            if built is None:
                summary["skipped"] += 1
                continue
            ids, markers, cut = built
            summary["truncated"] += int(cut)
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": r["target"], "label": r["label"],
                          "qid": r["qid"], "id": r["id"], "calibrate": bool(r.get("calibrate", True)),
                          "source": (r.get("group") or {}).get("source")})
    summary["items"] = len(items)
    return items, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=768)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    items, summary = build_qwen_items(read_records(args.records), load_questions(args.questions), tokenizer, args.max_len)
    summary.update({"max_len": args.max_len, "files": [str(p) for p in args.records], "pad_token_id": tokenizer.pad_token_id})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, args.out)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())

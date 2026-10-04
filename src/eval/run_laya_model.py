"""Predictions from any Laya-format model folder on public v0 records (docs/prompt_session4.md, step 9).

For each question named, reads <records-dir>/<qid>_validation.jsonl and <qid>_test.jsonl, builds items with the
model's own tokenizer (src/train/build_items.py), runs the model, fits one temperature per question on the
validation items marked calibrate, and writes validation and test predictions for <system> (with the temperature)
and <system>-raw (without) to results/public_v0/predictions/ in the step 5 format. Works for plain Laya, the
trained arms and Duan (whose folder follows the same layout). The model folder is loaded the way laya.Agent loads
it in laya 0.3.20: the tokenizer through laya's own loader, which first repairs a tokenizer_config.json written by
the mmBERT tokenizer (Duan's base) so that transformers can read it, then the weights with strict key and shape
matching, with the encoder's torch.compile path switched off. Any temperature stored in the model's
rl_agent_config.json is not applied: "raw" means no temperature at all.

--calibration FILE (session 5, docs/prompt_session5.md, step 4): use the temperatures in that calibration.json (the one
train_laya.py wrote for the model) instead of fitting them on validation, so the calibrated predictions are the
trained run's own.

--logits-dir DIR (session 7, docs/prompt_session7.md, ARM A): also writes each row's raw logits, before any
temperature, at full precision, to DIR/<system>_<qid>_<split>_logits.csv (train_laya.py's write_logits).

Usage: python src/eval/run_laya_model.py --model-dir models/laya-base --system laya_zeroshot \
    --records-dir data/processed/public --qids pubmedqa_answer readmit_window [--max-len 768 --head-max-len 192] \
    [--calibration models/x/calibration.json]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from laya.agent import _fix_tokenizer_config, _load_tokenizer
from laya.common import build_model
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
from build_items import QUESTIONS, build_items, load_questions, read_records  # noqa: E402
from train_laya import fit_temperatures, per_question, pick_device, probabilities, run, write_logits, write_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "processed" / "public")
    parser.add_argument("--qids", nargs="+", required=True)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "public_v0" / "predictions")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--head-max-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="a calibration.json whose temperatures are used instead of fitting them on validation")
    parser.add_argument("--logits-dir", type=Path, default=None,
                        help="also write each row's raw logits to DIR/<system>_<qid>_<split>_logits.csv")
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"],
                        help="splits to predict (with --calibration, test alone is enough; session 5's testpreds)")
    args = parser.parse_args()
    if not args.calibration and "validation" not in args.splits:
        parser.error("without --calibration the temperature is fitted on validation, so validation must be among --splits")

    device = pick_device(args.device)
    config = json.loads((args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    max_len = args.max_len or config.get("max_len", 512)
    head_max_len = args.head_max_len or config.get("head_max_len", 192)
    _fix_tokenizer_config(str(args.model_dir))
    tokenizer = _load_tokenizer(str(args.model_dir / "tokenizer"), config)
    model = build_model(config, encoder_dir=str(args.model_dir / "encoder"), pretrained=False)
    model.load_state_dict({k: v.to(torch.float32) for k, v in load_file(str(args.model_dir / "model.safetensors")).items()}, strict=True)
    model.encoder.config.reference_compile = False
    model.to(device)
    questions = load_questions(args.questions)
    summary = {"system": args.system, "model_dir": str(args.model_dir), "device": str(device), "max_len": max_len, "head_max_len": head_max_len}
    rows = {}
    for split in [s for s in ("validation", "test") if s in args.splits]:
        records = read_records([args.records_dir / f"{q}_{split}.jsonl" for q in args.qids])
        items, built = build_items(records, questions, tokenizer, max_len, head_max_len)
        summary[f"items_{split}"] = built
        rows[split] = run(model, items, tokenizer.pad_token_id, device, args.batch_size, None)
        print(f"{args.system} {split}: {len(rows[split])} rows")
    if args.calibration:
        temperature = {k: float(v) for k, v in json.loads(args.calibration.read_text(encoding="utf-8"))["temperature"].items()}
        summary["temperature_from"] = str(args.calibration)
    else:
        temperature = fit_temperatures(rows["validation"])
        summary["temperature_from"] = "fitted on the validation items marked calibrate"
    summary["temperature"] = {k: round(v, 4) for k, v in temperature.items()}
    for split in rows:
        write_predictions(args.out, args.system, split, rows[split], probabilities(rows[split], temperature))
        write_predictions(args.out, f"{args.system}-raw", split, rows[split], probabilities(rows[split]))
        if args.logits_dir:
            write_logits(args.logits_dir, args.system, split, rows[split])
    if "test" in rows:
        summary["test_per_question_raw"] = per_question(rows["test"], probabilities(rows["test"]))
        summary["test_per_question_calibrated"] = per_question(rows["test"], probabilities(rows["test"], temperature))
    path = args.out.parent / f"{args.system}_run_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())

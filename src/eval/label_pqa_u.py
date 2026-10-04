"""The labeler's predictions, its gate, the PQA-U subset and phase 2's records (docs/prompt_session7.md, THE LABELER).

1. Predictions. Loads the labeler (s7_lab_b, a Laya-format folder) the way src/eval/run_laya_model.py loads a model
   folder, builds items for labeler_validation.jsonl and labeler_pqa_u.jsonl with the session 7 question bank
   (src/questions/session7.yaml) and max_len 256, and writes the probabilities (softmax of the logits, no temperature)
   to results/fiorillo_v0_3/labeler_predictions/: labeler_validation.csv and labeler_pqa_u_part-<nnn>.csv (id, label,
   p0, p1, p2; each part under 1 MB).
2. Gate: the labeler's top-answer accuracy on the 150 calibrate validation items must be higher than v0.1's on the same
   items (v0.1's calibrated validation predictions, joined on record ID). If it is not: "NOT BUILT: phase 2, the
   labeler did not beat v0.1 on validation".
3. Subset: n_c is the number of PQA-U items whose top answer is c; with PQA-L's shares s (yes 0.552, no 0.338, maybe
   0.110, as the PubMedQA paper gives them), N = min(30,600, floor(min over c of n_c / s_c)); for each answer c, the
   floor(N x s_c) items with the highest probability of c among those whose top answer is c are labeled c, ties taken in
   PubMed ID order. If N is under 1,000: "NOT BUILT: phase 2, fewer than 1,000 items could be labeled in PQA-L's
   proportions".
4. phase2: those items as session 4's records (state: question and contexts, from pqa_u_unlabeled.jsonl; a one-hot
   target; group source pqa_unlabeled_labeler), in PubMed ID order, written to data/processed/session7/
   pubmedqa_phase2_train.jsonl; results/fiorillo_v0_3/phase2_labels.csv (id, label, probability of that label; in
   numbered parts if over 1 MB) lists them.
labeler.json records the gate, n_c, N, the counts taken, the lowest probability taken per answer and phase2's SHA-256,
or "NOT BUILT: phase 2" with the reason (also when an earlier part of the labeler stage failed: --not-built REASON), so
that neither arm waits for it.

Session 8 (docs/prompt_session8.md, ARM A and step 3): the rebuild subcommand writes phase 2's records again, on the pod,
from results/fiorillo_v0_3/phase2_labels.csv (the files labeler.json lists) and session 7's PQA-U records
(pqa_u_unlabeled.jsonl), in PubMed ID order, with the same writer as step 4 above, and checks the file's SHA-256 against
labeler.json's (or --expected-sha256); if they differ it stops with "NOT BUILT: arm A, phase 2's records differ from
session 7's". Nothing else is changed.

Usage (on the pod, the labeler stage):
    python src/eval/label_pqa_u.py --model-dir /root/eddm_s7/models/s7_lab_b --records-dir data/processed/session7 \\
        --v0-1-validation results/public_v0/predictions/arm_laya_pubmedqa_answer_validation.csv
    python src/eval/label_pqa_u.py --not-built "s7_lab_a stopped: <reason>"
Usage (session 8, the pod's prepare_a stage):
    python src/eval/label_pqa_u.py rebuild --results results/fiorillo_v0_3 --records-dir data/processed/session7
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "src" / "tasks"))
import metrics as M  # noqa: E402

SHARES = (("yes", Fraction("0.552")), ("no", Fraction("0.338")), ("maybe", Fraction("0.110")))
CAP = 30600
MIN_N = 1000
PART_BYTES = 1_000_000
RESULTS = ROOT / "results" / "fiorillo_v0_3"
RECORDS = ROOT / "data" / "processed" / "session7"
QUESTIONS7 = ROOT / "src" / "questions" / "session7.yaml"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pubid(record_id):
    return int(str(record_id).split("-", 1)[1])


def subset(ids, probs, shares=SHARES, cap=CAP):
    """The subset rule. ids: record IDs; probs: (n, 3) array. Returns (taken: list of (id, label, p)), details dict)."""
    probs = np.asarray(probs, dtype=float)
    top = probs.argmax(axis=1)
    n_c = [int((top == c).sum()) for c in range(len(shares))]
    bound = min(Fraction(n) / s for n, (_, s) in zip(n_c, shares))
    n_total = min(cap, math.floor(bound))
    take = [math.floor(Fraction(n_total) * s) for _, s in shares]
    taken, lowest = [], []
    for c, k in enumerate(take):
        idx = [i for i in range(len(ids)) if top[i] == c]
        idx.sort(key=lambda i: (-probs[i, c], pubid(ids[i])))
        chosen = idx[:k]
        taken.extend((ids[i], c, float(probs[i, c])) for i in chosen)
        lowest.append(float(probs[chosen[-1], c]) if chosen else None)
    taken.sort(key=lambda t: pubid(t[0]))
    details = {"n_c": dict(zip([name for name, _ in shares], n_c)), "shares": {name: float(s) for name, s in shares},
               "cap": cap, "N": n_total, "taken": dict(zip([name for name, _ in shares], take)), "taken_total": sum(take),
               "lowest_probability_taken": dict(zip([name for name, _ in shares], lowest))}
    return taken, details


def gate(labeler_rows, v01_path, calibrate_ids):
    """Top-answer accuracy of the labeler and of v0.1 on the calibrate validation items, joined on record ID."""
    ids, labels, probs = M.read_predictions(v01_path)
    v01 = {i: (int(y), int(np.argmax(p))) for i, y, p in zip(ids, labels, probs)}
    lab = {r["id"]: (int(r["label"]), int(np.argmax(r["probs"]))) for r in labeler_rows}
    keys = sorted(calibrate_ids)
    missing = [k for k in keys if k not in v01 or k not in lab]
    if missing:
        raise SystemExit(f"{len(missing)} calibrate items missing from the labeler's or v0.1's predictions, for example {missing[:3]}")
    for k in keys:
        if v01[k][0] != lab[k][0]:
            raise SystemExit(f"{k}: label {lab[k][0]} in the labeler records, {v01[k][0]} in v0.1's predictions")
    labeler_acc = float(np.mean([lab[k][1] == lab[k][0] for k in keys]))
    v01_acc = float(np.mean([v01[k][1] == v01[k][0] for k in keys]))
    return {"n": len(keys), "labeler_accuracy": labeler_acc, "v0_1_accuracy": v01_acc,
            "labeler_correct": int(sum(lab[k][1] == lab[k][0] for k in keys)), "v0_1_correct": int(sum(v01[k][1] == v01[k][0] for k in keys)),
            "passed": labeler_acc > v01_acc}


def write_csv_parts(directory, stem, header, rows, part_bytes=PART_BYTES, single_name=None):
    """rows as lists; one file if it fits under part_bytes (named single_name or <stem>.csv), else numbered parts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header) + "\n"] + [",".join(str(x) for x in r) + "\n" for r in rows]
    total = sum(len(s.encode("utf-8")) for s in lines)
    if total < part_bytes:
        path = directory / (single_name or f"{stem}.csv")
        path.write_text("".join(lines), encoding="utf-8")
        return [path]
    paths, current, size, k = [], [lines[0]], len(lines[0].encode("utf-8")), 0
    for line in lines[1:]:
        b = len(line.encode("utf-8"))
        if size + b >= part_bytes and len(current) > 1:
            path = directory / f"{stem}_part-{k:03d}.csv"
            path.write_text("".join(current), encoding="utf-8")
            paths.append(path)
            current, size, k = [lines[0]], len(lines[0].encode("utf-8")), k + 1
        current.append(line)
        size += b
    path = directory / f"{stem}_part-{k:03d}.csv"
    path.write_text("".join(current), encoding="utf-8")
    paths.append(path)
    return paths


def predict_rows(model_dir, records_path, questions_path, max_len=256, batch_size=64, device_name="auto"):
    """Probabilities (no temperature) for every record of one file, with the labeler loaded as run_laya_model.py loads a folder."""
    import torch
    from laya.agent import _fix_tokenizer_config, _load_tokenizer
    from laya.common import build_model
    from safetensors.torch import load_file
    from build_items import build_items, load_questions, read_records
    from train_laya import pick_device, run

    device = pick_device(device_name)
    config = json.loads((Path(model_dir) / "rl_agent_config.json").read_text(encoding="utf-8"))
    _fix_tokenizer_config(str(model_dir))
    tokenizer = _load_tokenizer(str(Path(model_dir) / "tokenizer"), config)
    model = build_model(config, encoder_dir=str(Path(model_dir) / "encoder"), pretrained=False)
    model.load_state_dict({k: v.to(torch.float32) for k, v in load_file(str(Path(model_dir) / "model.safetensors")).items()}, strict=True)
    model.encoder.config.reference_compile = False
    model.to(device)
    records = read_records([records_path])
    items, built = build_items(records, load_questions(questions_path), tokenizer, max_len, config.get("head_max_len", 192))
    rows = run(model, items, tokenizer.pad_token_id, device, batch_size, torch.float16 if device.type == "cuda" else None)
    out = []
    for r in rows:
        z = np.array(r["logits"], dtype=float)
        p = np.exp(z - z.max())
        out.append({"id": r["id"], "label": r["label"], "probs": (p / p.sum()).tolist()})
    return out, built


def phase2_decision(gate_result, details=None):
    """None when phase 2 is built; otherwise the NOT BUILT reason (the gate first, then N)."""
    if not gate_result["passed"]:
        return "the labeler did not beat v0.1 on validation"
    if details is not None and details["N"] < MIN_N:
        return "fewer than 1,000 items could be labeled in PQA-L's proportions"
    return None


def write_not_built(results, reason, extra=None):
    """labeler.json with phase 2 NOT BUILT. With --not-built after this script had already written labeler.json (a
    later part of the labeler stage failed), the numbers already there are kept under "before_the_failure"."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    path = results / "labeler.json"
    earlier = None
    if extra is None and path.exists():
        earlier = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k != "plan"}
    data = {"plan": "docs/prompt_session7.md, THE LABELER", "phase2_built": False, "not_built": f"NOT BUILT: phase 2, {reason}",
            "time_utc": now(), **(extra or {})}
    if earlier:
        data["before_the_failure"] = earlier
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(data["not_built"])
    return data


def write_phase2(records_dir, label_of, out_path):
    """Phase 2's records: the PQA-U items in label_of, as session 4's records, in PubMed ID order. Returns the count."""
    from pubmedqa import LABELS
    import records as R
    written = 0
    with open(Path(records_dir) / "pqa_u_unlabeled.jsonl", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as g_out:
        rows = [json.loads(line) for line in f if line.strip()]
        rows.sort(key=lambda r: pubid(r["id"]))
        for r in rows:
            if r["id"] in label_of:
                c = label_of[r["id"]]
                rec = {"id": r["id"], "qid": "pubmedqa_answer", "state": r["state"], "label": c,
                       "target": R.one_hot(c, len(LABELS)), "group": {"source": "pqa_unlabeled_labeler"}}
                g_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    return written


def read_phase2_labels(results, files):
    """{id: label} from phase2_labels.csv, or its numbered parts, as labeler.json lists them."""
    import csv
    label_of = {}
    for name in files:
        with open(Path(results) / name, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label_of[row["id"]] = int(row["label"])
    return label_of


def rebuild(results=RESULTS, records_dir=RECORDS, expected_sha256=None):
    """Session 8: phase 2's records from phase2_labels.csv and session 7's PQA-U records, checked against labeler.json."""
    labeler = json.loads((Path(results) / "labeler.json").read_text(encoding="utf-8"))
    if not labeler.get("phase2_built"):
        raise SystemExit("NOT BUILT: arm A, labeler.json records phase 2 as not built")
    info = labeler["phase2"]
    label_of = read_phase2_labels(results, info["labels_files"])
    out_path = Path(records_dir) / info["file"]
    written = write_phase2(records_dir, label_of, out_path)
    got = sha256_file(out_path)
    want = expected_sha256 or info["sha256"]
    if written != len(label_of) or written != info["records"] or got != want:
        raise SystemExit(f"NOT BUILT: arm A, phase 2's records differ from session 7's ({written} records, SHA-256 {got}; "
                         f"labeler.json: {info['records']} records, SHA-256 {want})")
    print(json.dumps({"rebuilt": str(out_path), "records": written, "sha256": got, "matches_labeler_json": True}))
    return {"records": written, "sha256": got}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["rebuild"]:
        rb = argparse.ArgumentParser(description="rebuild phase 2's records (session 8)")
        rb.add_argument("--results", type=Path, default=RESULTS)
        rb.add_argument("--records-dir", type=Path, default=RECORDS)
        rb.add_argument("--expected-sha256", default=None, help="tests only: the SHA-256 to check against")
        a = rb.parse_args(argv[1:])
        rebuild(a.results, a.records_dir, a.expected_sha256)
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--questions", type=Path, default=QUESTIONS7)
    parser.add_argument("--v0-1-validation", type=Path, default=ROOT / "results" / "public_v0" / "predictions" / "arm_laya_pubmedqa_answer_validation.csv")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--not-built", default=None, help="record phase 2 as NOT BUILT with this reason and stop")
    args = parser.parse_args(argv)
    if args.not_built:
        write_not_built(args.results, args.not_built)
        return 0
    if args.model_dir is None:
        parser.error("--model-dir is required unless --not-built is given")

    val_rows, val_built = predict_rows(args.model_dir, args.records_dir / "labeler_validation.jsonl", args.questions,
                                       args.max_len, args.batch_size, args.device)
    u_rows, u_built = predict_rows(args.model_dir, args.records_dir / "labeler_pqa_u.jsonl", args.questions,
                                   args.max_len, args.batch_size, args.device)
    pred_dir = args.results / "labeler_predictions"
    header = ["id", "label", "p0", "p1", "p2"]
    write_csv_parts(pred_dir, "labeler_validation", header,
                    [[r["id"], r["label"]] + [f"{x:.9g}" for x in r["probs"]] for r in val_rows])
    u_paths = write_csv_parts(pred_dir, "labeler_pqa_u", header,
                              [[r["id"], r["label"]] + [f"{x:.9g}" for x in r["probs"]] for r in u_rows])

    calibrate_ids = {json.loads(line)["id"] for line in open(args.records_dir / "labeler_validation.jsonl", encoding="utf-8")
                     if json.loads(line).get("calibrate", True)}
    g = gate(val_rows, args.v0_1_validation, calibrate_ids)
    base = {"items": {"labeler_validation": val_built, "labeler_pqa_u": u_built}, "gate": g,
            "predictions": [str(p.relative_to(args.results)) for p in [pred_dir / "labeler_validation.csv"] + u_paths]}
    # n_c and N are recorded in every case; nothing is taken unless the gate passes and N is at least 1,000
    taken, details = subset([r["id"] for r in u_rows], [r["probs"] for r in u_rows])
    base["subset"] = details
    reason = phase2_decision(g, details)
    if reason:
        base["subset"]["used"] = False
        write_not_built(args.results, reason, base)
        return 0

    label_of = {i: c for i, c, _ in taken}
    out_path = args.records_dir / "pubmedqa_phase2_train.jsonl"
    written = write_phase2(args.records_dir, label_of, out_path)
    if written != len(taken):
        raise SystemExit(f"{len(taken)} items taken but {written} found in pqa_u_unlabeled.jsonl")
    label_paths = write_csv_parts(args.results, "phase2_labels", ["id", "label", "probability"],
                                  [[i, c, f"{p:.9g}"] for i, c, p in taken], single_name="phase2_labels.csv")
    data = {"plan": "docs/prompt_session7.md, THE LABELER", "phase2_built": True, "time_utc": now(), **base,
            "phase2": {"file": str(out_path.name), "records": written, "sha256": sha256_file(out_path),
                       "labels_files": [str(p.relative_to(args.results)) for p in label_paths]}}
    tmp = args.results / "labeler.json.tmp"
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(args.results / "labeler.json")
    print(json.dumps({"gate": g, "N": details["N"], "taken": details["taken"], "records": written}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

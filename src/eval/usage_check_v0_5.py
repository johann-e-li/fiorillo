"""Session 10's usage check of the released v0.5 (docs/prompt_session10.md, THE USAGE CHECK; I-100): sample and compare.

The usage check runs the released files (models/fiorillo_v0_5/release/, the files of Hugging Face revision
56e5944aedcc9344d2f13850fb365686dfb25800) with predict_v0_5.py, unchanged, on Johann's Mac (scripts/usage_check_v0_5.sh),
and compares their answers with the evaluated test predictions of v0.5's specialists.

Subcommands:
  sample   (the linked shell, step 1) Evidence Inference: 60 test prompts drawn without replacement by
           random.Random(20261003).sample from the IDs of the evaluated file, kept in that file's order; PubMedQA: all 500
           test items; readmission: all 6,998 test rows. Each record is copied unchanged (its line, byte for byte) from its
           records file into data/processed/session10/usage_check_<qid>_records.jsonl, in the evaluated file's order.
           results/fiorillo_v0_5/usage_check/ids.json lists each question's IDs and the SHA-256 of every input file and
           of each record file written.
  compare  (the Mac, the script's last step) each answers file (answers_<qid>.jsonl) beside its evaluated file, record by
           record: the absolute differences of the three probabilities (readmission's levels 0, 1 and 2 against p0, p1 and
           p2) and the released and evaluated top answers; the verdicts below; usage_check.json, with each question's
           device, seconds per record and largest differences (from run_<qid>.log), the versions of Python, torch,
           transformers, xgboost and catboost, and every record that failed.

The verdicts, fixed in the plan: readmission PASSES when every row's largest absolute difference is at most 0.0001.
Evidence Inference and PubMedQA each PASS when (a) on every record whose evaluated top probability is at least 0.6 the
released top answer is the evaluated one, and (b) the mean absolute difference over all records and answers is at most
0.02. Records whose evaluated top probability is below 0.6 are listed with both answers and do not count against (a). A
record answered with an error, or not answered, is a failure, and a question with a failure does not pass. The check
PASSES when all three questions pass.

check_records() is the script's check of the three record files against ids.json (its step 1).

The second run (docs/prompt_session10.md, corrections 5 and 6; scripts/usage_check_v0_5_rerun.sh). The first run gave
Evidence Inference the records of data/processed/session8/evidence_effect_test.jsonl, session 8's 4,096-token selection,
while v0.5's evaluated predictions read data/processed/session9/long/evidence_effect_test.jsonl, the 6,144-token
selection; and its readmission run ended with a segmentation fault. With --rerun:
  sample   (the linked shell) Evidence Inference: the first run's 60 IDs (its ids.json), each record copied unchanged from
           the 6,144-token file, whose SHA-256 must be the one correction 5 gives, into
           data/processed/session10/usage_check_evidence_effect_long_records.jsonl; readmission: the first run's record
           file. results/fiorillo_v0_5/usage_check/rerun/ids.json lists them. The first run's files are not touched.
  compare  (the Mac) readmission and Evidence Inference from rerun/, by the same rules; readmission's answer is
           predict_v0_5.py's as a user runs it, and the diagnosis run with the tree libraries imported first, when it
           ran, is reported beside it without changing its verdict; PubMedQA, not run again, keeps its first run's
           verdict; writes rerun/usage_check.json with each run's fault report, if its log holds one.

The third run (correction 7; scripts/fix_release_v0_5.sh): all three questions again, with the fixed release (its
predict_v0_5.py loads xgboost and catboost before torch). sample --fixed writes results/fiorillo_v0_5/usage_check/fixed/
ids.json from the first run's readmission and PubMedQA record files and the second run's Evidence Inference file;
compare --fixed compares all three by the same rules and writes fixed/usage_check.json.

The fourth run (correction 9; scripts/fix_release_v0_5.sh as correction 9 changed it): all three questions again, with
the second fix (its predict_v0_5.py loads xgboost and catboost before torch only for a file of readmit_window records, and
on macOS refuses a file that mixes readmit_window with the other questions). sample --fixed2 and compare --fixed2 do for
results/fiorillo_v0_5/usage_check/fixed2/ what --fixed does for fixed/, from the same record files.

Usage: python3 src/eval/usage_check_v0_5.py sample [--rerun | --fixed | --fixed2]  (the linked shell, the project folder)
       uv run --with xgboost==3.2.0 --with catboost==1.2.10 python src/eval/usage_check_v0_5.py compare
       [--rerun | --fixed | --fixed2]   (the Mac)
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/prompt_session10.md, THE USAGE CHECK"
SEED = 20261003
EI, PQA, RW = "evidence_effect", "pubmedqa_answer", "readmit_window"
RUN_ORDER = (RW, PQA, EI)
N_EI_SAMPLE = 60
EXPECTED = {EI: 1218, PQA: 500, RW: 6998}  # the evaluated files' rows; Evidence Inference's sample is drawn from its 1,218
RW_LIMIT = 0.0001
MEAN_LIMIT = 0.02
CONFIDENT = 0.6
SOURCES = {
    EI: ("data/processed/session8/evidence_effect_test.jsonl",
         "results/fiorillo_v0_5/predictions/s9_ei_wavg_evidence_effect_test.csv"),
    PQA: ("data/processed/public/pubmedqa_answer_test.jsonl",
          "results/fiorillo_v0_3/predictions/s7_b3_pubmedqa_answer_test.csv"),
    RW: ("data/processed/public/readmit_window_test.jsonl",
         "results/fiorillo_v0_5/predictions/readmission_pool_v2_readmit_window_test.csv"),
}
SAMPLE_DIR = "data/processed/session10"
OUT_DIR = "results/fiorillo_v0_5/usage_check"
RELEASE = "models/fiorillo_v0_5/release"
TREES = ("xgboost_codes", "catboost_codes", "lightgbm_tuned", "xgboost_tuned", "catboost_tuned")
VERSIONS = ("torch", "transformers", "xgboost", "catboost")
LOG_KEY = re.compile(r"^usage_check (question|records|mode|device_requested|device_resolved|start_utc|end_utc|exit_status|out_of_memory): ?(.*)$")
# The same key after other text on its line: a run whose last output did not end its line (a fault handler's report, at
# the second run) leaves the script's next key on that line.
LOG_KEY_ANY = re.compile(r"usage_check (question|records|mode|device_requested|device_resolved|start_utc|end_utc|exit_status|out_of_memory): ?(.*)$")
# The second run (corrections 5 and 6).
RERUN_DIR = f"{OUT_DIR}/rerun"
PLAN_RERUN = "docs/prompt_session10.md, corrections 5 and 6 (the usage check's second run)"
LONG_SOURCE = "data/processed/session9/long/evidence_effect_test.jsonl"  # the inputs v0.5's evaluated predictions read
LONG_SHA256 = "eae8a2345b1800beea141cc6346457dc301d4017be78cf3e153c214ceaaf83aa"  # correction 5 (c)
LONG_RECORDS = f"{SAMPLE_DIR}/usage_check_{EI}_long_records.jsonl"
RERUN_ORDER = (RW, EI)
TREES_FIRST = f"{RW}_trees_first"  # readmission's diagnosis run: xgboost and catboost imported before torch
# The third run (correction 7): all three questions with the fixed release.
FIXED_DIR = f"{OUT_DIR}/fixed"
PLAN_FIXED = "docs/prompt_session10.md, correction 7 (the usage check's third run, on the fixed release)"
# The fourth run (correction 9): all three questions with the second fix.
FIXED2_DIR = f"{OUT_DIR}/fixed2"
PLAN_FIXED2 = "docs/prompt_session10.md, correction 9 (the usage check's fourth run, on the second fix)"
RULES = {
    RW: "PASS when every row's largest absolute difference is at most 0.0001",
    PQA: ("PASS when (a) on every record whose evaluated top probability is at least 0.6 the released top answer is the "
          "evaluated one and (b) the mean absolute difference over all records and answers is at most 0.02"),
    EI: ("PASS when (a) on every record whose evaluated top probability is at least 0.6 the released top answer is the "
         "evaluated one and (b) the mean absolute difference over all records and answers is at most 0.02"),
    "all": "a record answered with an error, or not answered, is a failure; the check PASSES when all three questions pass",
}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def records_name(qid):
    return f"{SAMPLE_DIR}/usage_check_{qid}_records.jsonl"


def evaluated_rows(path):
    """The IDs in the file's order and {id: (label, [p0, p1, p2])}."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    order, out = [], {}
    for r in rows:
        i = r["id"]
        if i in out:
            raise SystemExit(f"{path}: the ID {i} appears twice")
        out[i] = (int(r["label"]), [float(r["p0"]), float(r["p1"]), float(r["p2"])])
        order.append(i)
    return order, out


def record_lines(path):
    """{id: (qid, the record's line as bytes, ending in a newline)} of a JSON Lines records file."""
    out = {}
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            r = json.loads(raw)
            if r["id"] in out:
                raise SystemExit(f"{path}: the ID {r['id']} appears twice")
            out[r["id"]] = (r.get("qid"), raw if raw.endswith(b"\n") else raw + b"\n")
    return out


# ---------------------------------------------------------------- sample (step 1)

def sample(root=ROOT):
    root = Path(root)
    (root / SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    data = {"plan": PLAN, "seed": SEED, "written_utc": now(), "run_order": list(RUN_ORDER), "questions": {}}
    for qid in RUN_ORDER:
        src, evaluated = SOURCES[qid]
        order, _ = evaluated_rows(root / evaluated)
        if len(order) != EXPECTED[qid]:
            raise SystemExit(f"{evaluated} has {len(order):,} rows, not {EXPECTED[qid]:,}")
        lines = record_lines(root / src)
        if qid == EI:
            chosen = set(random.Random(SEED).sample(order, N_EI_SAMPLE))
            ids = [i for i in order if i in chosen]
            draw = (f"random.Random({SEED}).sample of {N_EI_SAMPLE} from the {len(order):,} IDs of the evaluated file, "
                    "without replacement, kept in that file's order")
        else:
            if set(lines) != set(order):
                raise SystemExit(f"{src} and {evaluated} hold different IDs")
            ids = list(order)
            draw = f"all {len(order):,} rows, in the evaluated file's order"
        missing = [i for i in ids if i not in lines]
        if missing:
            raise SystemExit(f"{src} lacks {len(missing)} of the sample's IDs, for example {missing[:3]}")
        wrong = [i for i in ids if lines[i][0] != qid]
        if wrong:
            raise SystemExit(f"{src}: {len(wrong)} of the sample's records are not {qid} records")
        out = root / records_name(qid)
        with open(out, "wb") as f:
            for i in ids:
                f.write(lines[i][1])
        data["questions"][qid] = {
            "n": len(ids), "draw": draw, "ids": ids,
            "records_source": src, "records_source_sha256": sha256_file(root / src),
            "evaluated": evaluated, "evaluated_sha256": sha256_file(root / evaluated),
            "records_file": records_name(qid), "records_file_sha256": sha256_file(out)}
        print(f"{qid}: {len(ids):,} records written to {records_name(qid)}", flush=True)
    path = root / OUT_DIR / "ids.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR}/ids.json", flush=True)
    return data


def sample_rerun(root=ROOT, long_sha256=LONG_SHA256):
    """The second run's sample (corrections 5 and 6): Evidence Inference's 60 IDs of the first run's ids.json, each record
    copied unchanged (its line, byte for byte) from LONG_SOURCE, in the same order; readmission's record file is the first
    run's, checked against the first run's ids.json. Writes RERUN_DIR/ids.json; stops when LONG_SOURCE's SHA-256 is not
    long_sha256 (the one correction 5 gives; tests pass their own)."""
    root = Path(root)
    first = json.loads((root / OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    src = root / LONG_SOURCE
    got = sha256_file(src)
    if long_sha256 is not None and got != long_sha256:
        raise SystemExit(f"{LONG_SOURCE} has SHA-256 {got}, not {long_sha256} (correction 5)")
    ei = first["questions"][EI]
    lines = record_lines(src)
    missing = [i for i in ei["ids"] if i not in lines]
    if missing:
        raise SystemExit(f"{LONG_SOURCE} lacks {len(missing)} of the sample's IDs, for example {missing[:3]}")
    wrong = [i for i in ei["ids"] if lines[i][0] != EI]
    if wrong:
        raise SystemExit(f"{LONG_SOURCE}: {len(wrong)} of the sample's records are not {EI} records")
    rw = first["questions"][RW]
    if sha256_file(root / rw["records_file"]) != rw["records_file_sha256"]:
        raise SystemExit(f"{rw['records_file']} no longer matches the first run's ids.json")
    out = root / LONG_RECORDS
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        for i in ei["ids"]:
            f.write(lines[i][1])
    data = {"plan": PLAN_RERUN, "written_utc": now(), "run_order": list(RERUN_ORDER), "first_run": f"{OUT_DIR}/ids.json",
            "questions": {
                RW: dict(rw),
                EI: {"n": len(ei["ids"]), "draw": "the first run's 60 IDs (its ids.json), in the same order",
                     "ids": list(ei["ids"]), "records_source": LONG_SOURCE, "records_source_sha256": got,
                     "evaluated": ei["evaluated"], "evaluated_sha256": sha256_file(root / ei["evaluated"]),
                     "records_file": LONG_RECORDS, "records_file_sha256": sha256_file(out)}}}
    (root / RERUN_DIR).mkdir(parents=True, exist_ok=True)
    (root / RERUN_DIR / "ids.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{EI}: {len(ei['ids'])} records written to {LONG_RECORDS}; {RW}: the first run's {rw['records_file']}", flush=True)
    print(f"wrote {RERUN_DIR}/ids.json", flush=True)
    return data


def sample_fixed(root=ROOT, folder=FIXED_DIR, plan=PLAN_FIXED):
    """The third run's sample (correction 7), or the fourth's with folder FIXED2_DIR (correction 9): readmission's and
    PubMedQA's record files as the first run wrote them (its ids.json), Evidence Inference's as the second run wrote them
    (its ids.json: the first run's 60 IDs with the inputs the evaluated predictions read), each checked against its
    SHA-256 there. Writes folder/ids.json."""
    root = Path(root)
    first = json.loads((root / OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    second = json.loads((root / RERUN_DIR / "ids.json").read_text(encoding="utf-8"))
    questions = {RW: dict(first["questions"][RW]), PQA: dict(first["questions"][PQA]), EI: dict(second["questions"][EI])}
    for qid, q in questions.items():
        got = sha256_file(root / q["records_file"])
        if got != q["records_file_sha256"]:
            raise SystemExit(f"{q['records_file']} has SHA-256 {got}, not the {q['records_file_sha256']} its ids.json gives")
    data = {"plan": plan, "written_utc": now(), "run_order": list(RUN_ORDER),
            "from": {RW: f"{OUT_DIR}/ids.json", PQA: f"{OUT_DIR}/ids.json", EI: f"{RERUN_DIR}/ids.json"},
            "questions": {qid: questions[qid] for qid in RUN_ORDER}}
    (root / folder).mkdir(parents=True, exist_ok=True)
    if (root / folder / "ids.json").exists():
        raise SystemExit(f"{folder}/ids.json exists; a run's sample is written once")
    (root / folder / "ids.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for qid in RUN_ORDER:
        print(f"{qid}: {len(questions[qid]['ids']):,} records, {questions[qid]['records_file']}", flush=True)
    print(f"wrote {folder}/ids.json", flush=True)
    return data


def check_records(root=ROOT, rerun=False, fixed=False, fixed2=False):
    """0 when the record files match ids.json (SHA-256 and IDs), 1 otherwise (each script's step 1); rerun reads the
    second run's ids.json and its two questions, fixed the third run's and its three, fixed2 the fourth run's."""
    root = Path(root)
    folder = FIXED2_DIR if fixed2 else (FIXED_DIR if fixed else (RERUN_DIR if rerun else OUT_DIR))
    path = root / folder / "ids.json"
    if not path.exists():
        print(f"{folder}/ids.json is missing", flush=True)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    bad = 0
    for qid in (RERUN_ORDER if rerun and not (fixed or fixed2) else RUN_ORDER):
        q = data["questions"].get(qid)
        if q is None:
            print(f"{qid}: not in ids.json", flush=True)
            bad += 1
            continue
        f = root / q["records_file"]
        if not f.exists():
            print(f"{q['records_file']}: missing", flush=True)
            bad += 1
            continue
        got = sha256_file(f)
        ids = list(record_lines(f))
        if got != q["records_file_sha256"] or ids != q["ids"]:
            print(f"{q['records_file']}: SHA-256 {got}, ids.json has {q['records_file_sha256']}"
                  + ("" if ids == q["ids"] else "; its IDs differ from ids.json's"), flush=True)
            bad += 1
        else:
            print(f"{q['records_file']}: OK ({len(ids):,} records)", flush=True)
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------- compare (step 4)

def read_answers(path):
    """{id: answer} from predict_v0_5.py's JSON Lines output; {} when the file is missing or empty."""
    out = {}
    path = Path(path)
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        if line.strip():
            a = json.loads(line)
            out[a.get("id")] = a
    return out


def read_log(path, anywhere=False):
    """The keys the script writes (lines "usage_check <key>: <value>"); each key's last value, so that a run made again on
    the CPU replaces the first run's device and times; None when the log is missing. With anywhere (the second run's
    logs), a key that follows other text on its line is read too (LOG_KEY_ANY)."""
    path = Path(path)
    if not path.exists():
        return None
    out = {"rerun_on_cpu": False}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LOG_KEY.match(line.rstrip("\n"))
        if not m and anywhere:
            m = LOG_KEY_ANY.search(line.rstrip("\n"))
        if not m:
            continue
        if m.group(1) == "out_of_memory":
            out["rerun_on_cpu"] = True
            out["out_of_memory"] = m.group(2).strip()
        else:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_utc(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def argmax(values):
    best = 0
    for k in range(1, len(values)):
        if values[k] > values[best]:
            best = k
    return best


def compare_question(qid, ids, evaluated, answers):
    """The record-by-record comparison of one question and its verdict (the plan's rules, fixed before the run)."""
    failures, row_max, diffs, below, changed_confident = [], [], [], [], []
    options = None
    for i in ids:
        label, ep = evaluated[i]
        a = answers.get(i)
        if a is None:
            failures.append({"id": i, "reason": "not answered"})
            continue
        if "error" in a:
            failures.append({"id": i, "reason": f"answered with an error: {a['error']}"})
            continue
        if a.get("qid") != qid:
            failures.append({"id": i, "reason": f"answered as {a.get('qid')}, not {qid}"})
            continue
        probs = a.get("probabilities")
        if not isinstance(probs, dict) or len(probs) != 3:
            failures.append({"id": i, "reason": "the answer does not hold three probabilities"})
            continue
        if options is None:
            options = list(probs)
        rp = [float(v) for v in probs.values()]
        if not all(math.isfinite(v) for v in rp):
            failures.append({"id": i, "reason": "a probability is not finite"})
            continue
        d = [abs(x - y) for x, y in zip(rp, ep)]
        diffs.extend(d)
        row_max.append(max(d))
        top_e, top_r = argmax(ep), argmax(rp)
        if ep[top_e] < CONFIDENT:
            below.append({"id": i, "evaluated_top": top_e, "released_top": top_r, "same": top_e == top_r,
                          "evaluated_top_probability": ep[top_e], "evaluated": ep, "released": rp})
        elif top_e != top_r:
            changed_confident.append({"id": i, "evaluated_top": top_e, "released_top": top_r,
                                      "evaluated_top_probability": ep[top_e], "evaluated": ep, "released": rp})
    answered = len(row_max)
    mean = sum(diffs) / len(diffs) if diffs else None
    largest = max(row_max) if row_max else None
    out = {"n": len(ids), "answered": answered, "failures": failures, "options": options,
           "largest_absolute_difference": largest, "mean_absolute_difference": mean, "rule": RULES[qid]}
    if qid == RW:
        above = sum(1 for m in row_max if m > RW_LIMIT)
        out["rows_above_0_0001"] = above
        out["verdict"] = "PASS" if (not failures and answered == len(ids) and above == 0) else "FAIL"
    else:
        crit_a = not changed_confident
        crit_b = mean is not None and mean <= MEAN_LIMIT
        out.update({"top_answer_changed_at_0_6_or_more": changed_confident,
                    "evaluated_top_probability_below_0_6": below,
                    "below_0_6_count": len(below), "below_0_6_top_changed": sum(1 for b in below if not b["same"]),
                    "criterion_a": "PASS" if crit_a else "FAIL", "criterion_b": "PASS" if crit_b else "FAIL"})
        out["verdict"] = "PASS" if (not failures and answered == len(ids) and crit_a and crit_b) else "FAIL"
    return out


def versions():
    out = {"python": platform.python_version()}
    for name in VERSIONS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def pool_device(root):
    """Readmission's device: the pool's members with weight above zero; trees run on the CPU whatever --device says."""
    path = Path(root) / RELEASE / "readmit_window" / "pool.json"
    if not path.exists():
        return None
    pool = json.loads(path.read_text(encoding="utf-8"))
    weighted = [m for m, w in pool.get("weights", {}).items() if w > 0]
    if weighted and all(m in TREES for m in weighted):
        return f"cpu (the pool's members with weight are trees: {', '.join(weighted)})"
    return None


def compare(root=ROOT):
    root = Path(root)
    out_dir = root / OUT_DIR
    ids_json = json.loads((out_dir / "ids.json").read_text(encoding="utf-8"))
    result = {"plan": PLAN, "written_utc": now(), "rules": RULES, "versions": versions(), "questions": {}}
    sums = root / RELEASE / "SHA256SUMS.txt"
    predict = root / RELEASE / "predict_v0_5.py"
    result["release"] = {"folder": RELEASE,
                         "SHA256SUMS.txt_sha256": sha256_file(sums) if sums.exists() else None,
                         "predict_v0_5.py_sha256": sha256_file(predict) if predict.exists() else None}
    for qid in RUN_ORDER:
        q = ids_json["questions"][qid]
        _, evaluated = evaluated_rows(root / q["evaluated"])
        missing = [i for i in q["ids"] if i not in evaluated]
        if missing:
            raise SystemExit(f"{q['evaluated']} lacks {len(missing)} of the sample's IDs")
        answers_path = out_dir / f"answers_{qid}.jsonl"
        block = compare_question(qid, q["ids"], evaluated, read_answers(answers_path))
        block["evaluated"] = q["evaluated"]
        block["answers_file"] = f"{OUT_DIR}/answers_{qid}.jsonl"
        block["answers_sha256"] = sha256_file(answers_path) if answers_path.exists() else None
        log = read_log(out_dir / f"run_{qid}.log")
        if log is None:
            block["run"] = None
        else:
            seconds = None
            if log.get("start_utc") and log.get("end_utc"):
                seconds = (parse_utc(log["end_utc"]) - parse_utc(log["start_utc"])).total_seconds()
            device = log.get("device_resolved")
            if qid == RW and pool_device(root):
                device = pool_device(root)
            block["run"] = {"log": f"{OUT_DIR}/run_{qid}.log", "device_requested": log.get("device_requested"),
                            "device": device, "rerun_on_cpu": log["rerun_on_cpu"], "start_utc": log.get("start_utc"),
                            "end_utc": log.get("end_utc"), "exit_status": log.get("exit_status"), "seconds": seconds,
                            "seconds_per_record": (seconds / len(q["ids"])) if seconds is not None else None}
            if qid == PQA:
                block["run"]["note"] = ("the first run that loads Qwen3-4B-Base; its time includes the model's download "
                                        "when the Hugging Face cache did not hold it")
        result["questions"][qid] = block
        print(f"{qid}: {block['verdict']} ({block['answered']:,} of {block['n']:,} answered; largest absolute difference "
              f"{block['largest_absolute_difference']}; mean {block['mean_absolute_difference']})", flush=True)
    result["verdict"] = "PASS" if all(result["questions"][q]["verdict"] == "PASS" for q in RUN_ORDER) else "FAIL"
    (out_dir / "usage_check.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"the usage check: {result['verdict']} ({OUT_DIR}/usage_check.json)", flush=True)
    return result


# ---------------------------------------------------------------- the second run's compare (corrections 5 and 6)

def fault_report(path):
    """The fault handler's report in a run's log: the lines from "Fatal Python error" to the script's next "usage_check"
    line (at most 80), or None when the log holds none."""
    path = Path(path)
    if not path.exists():
        return None
    out, on = [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.rstrip("\n")
        if not on:
            k = s.find("Fatal Python error")
            if k < 0:
                continue
            on, s = True, s[k:]
        cut = s.find("usage_check ")  # the script's next key ends the report, even on the report's own line
        if cut >= 0:
            if s[:cut].strip():
                out.append(s[:cut])
            break
        out.append(s)
    return "\n".join(out[:80]) if out else None


def rerun_facts(root, name, qid, n, folder=RERUN_DIR):
    """A second- or third-run log's facts (as compare's), with its mode and its fault report; None when the log is
    missing."""
    path = Path(root) / folder / f"run_{name}.log"
    log = read_log(path, anywhere=True)
    if log is None:
        return None
    seconds = None
    if log.get("start_utc") and log.get("end_utc"):
        seconds = (parse_utc(log["end_utc"]) - parse_utc(log["start_utc"])).total_seconds()
    device = log.get("device_resolved")
    if qid == RW and pool_device(root):
        device = pool_device(root)
    return {"log": f"{folder}/run_{name}.log", "mode": log.get("mode"), "device_requested": log.get("device_requested"),
            "device": device, "rerun_on_cpu": log["rerun_on_cpu"], "start_utc": log.get("start_utc"),
            "end_utc": log.get("end_utc"), "exit_status": log.get("exit_status"), "seconds": seconds,
            "seconds_per_record": (seconds / n) if seconds is not None else None, "fault": fault_report(path)}


def compare_rerun(root=ROOT):
    """The second run's comparison, by the same rules: readmission and Evidence Inference from RERUN_DIR; readmission's
    verdict is predict_v0_5.py's as a user runs it, and the diagnosis run (TREES_FIRST), when it ran, is reported beside
    it without changing that verdict; PubMedQA keeps its first run's verdict. Writes RERUN_DIR/usage_check.json."""
    root = Path(root)
    out_dir = root / RERUN_DIR
    ids_json = json.loads((out_dir / "ids.json").read_text(encoding="utf-8"))
    result = {"plan": PLAN_RERUN, "written_utc": now(), "rules": RULES, "versions": versions(), "questions": {}}
    sums = root / RELEASE / "SHA256SUMS.txt"
    predict = root / RELEASE / "predict_v0_5.py"
    result["release"] = {"folder": RELEASE,
                         "SHA256SUMS.txt_sha256": sha256_file(sums) if sums.exists() else None,
                         "predict_v0_5.py_sha256": sha256_file(predict) if predict.exists() else None}
    for qid in RERUN_ORDER:
        q = ids_json["questions"][qid]
        _, evaluated = evaluated_rows(root / q["evaluated"])
        missing = [i for i in q["ids"] if i not in evaluated]
        if missing:
            raise SystemExit(f"{q['evaluated']} lacks {len(missing)} of the sample's IDs")
        blocks = {}
        for name in [qid] + ([TREES_FIRST] if qid == RW else []):
            if name != qid:
                # the diagnosis counts only when readmission's own run failed and the diagnosis started after it ended
                # (a log left by an earlier invocation of the script is not read)
                direct = blocks[qid]["run"] or {}
                diag = read_log(out_dir / f"run_{name}.log", anywhere=True)
                if (diag is None or direct.get("exit_status") in (None, "0") or not diag.get("start_utc")
                        or not direct.get("end_utc") or parse_utc(diag["start_utc"]) < parse_utc(direct["end_utc"])):
                    continue
            answers_path = out_dir / f"answers_{name}.jsonl"
            block = compare_question(qid, q["ids"], evaluated, read_answers(answers_path))
            block["evaluated"] = q["evaluated"]
            block["records_file"] = q["records_file"]
            block["answers_file"] = f"{RERUN_DIR}/answers_{name}.jsonl"
            block["answers_sha256"] = sha256_file(answers_path) if answers_path.exists() else None
            block["run"] = rerun_facts(root, name, qid, len(q["ids"]))
            blocks[name] = block
        main_block = blocks[qid]
        if TREES_FIRST in blocks:
            main_block["trees_first"] = dict(blocks[TREES_FIRST], note=(
                "a diagnosis: predict_v0_5.py, unchanged, run with xgboost and catboost imported before it loads torch; "
                "its verdict does not change the question's, which is predict_v0_5.py's as a user runs it"))
        result["questions"][qid] = main_block
        print(f"{qid}: {main_block['verdict']} ({main_block['answered']:,} of {main_block['n']:,} answered; largest absolute "
              f"difference {main_block['largest_absolute_difference']}; mean {main_block['mean_absolute_difference']})"
              + (f"; with the tree libraries imported first: {blocks[TREES_FIRST]['verdict']}" if TREES_FIRST in blocks else ""),
              flush=True)
    first = json.loads((root / OUT_DIR / "usage_check.json").read_text(encoding="utf-8"))["questions"][PQA]
    result["questions"][PQA] = {
        "carried_over": f"{OUT_DIR}/usage_check.json",
        "note": "not run again: its first run passed, on the inputs its evaluated predictions read",
        "verdict": first["verdict"], "n": first["n"], "answered": first["answered"],
        "largest_absolute_difference": first["largest_absolute_difference"],
        "mean_absolute_difference": first["mean_absolute_difference"], "run": first.get("run")}
    result["verdict"] = "PASS" if all(result["questions"][q]["verdict"] == "PASS" for q in RUN_ORDER) else "FAIL"
    (out_dir / "usage_check.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"the usage check's second run: {result['verdict']} ({RERUN_DIR}/usage_check.json)", flush=True)
    return result


def compare_fixed(root=ROOT, folder=FIXED_DIR, plan=PLAN_FIXED, name="third"):
    """The third run's comparison (correction 7), or the fourth's with folder FIXED2_DIR (correction 9): all three
    questions from folder, run with the release as fixed, by the same rules; the release's SHA256SUMS.txt and
    predict_v0_5.py as they are when it runs. Writes folder/usage_check.json."""
    root = Path(root)
    out_dir = root / folder
    ids_json = json.loads((out_dir / "ids.json").read_text(encoding="utf-8"))
    result = {"plan": plan, "written_utc": now(), "rules": RULES, "versions": versions(), "questions": {}}
    sums = root / RELEASE / "SHA256SUMS.txt"
    predict = root / RELEASE / "predict_v0_5.py"
    result["release"] = {"folder": RELEASE,
                         "SHA256SUMS.txt_sha256": sha256_file(sums) if sums.exists() else None,
                         "predict_v0_5.py_sha256": sha256_file(predict) if predict.exists() else None}
    for qid in RUN_ORDER:
        q = ids_json["questions"][qid]
        _, evaluated = evaluated_rows(root / q["evaluated"])
        missing = [i for i in q["ids"] if i not in evaluated]
        if missing:
            raise SystemExit(f"{q['evaluated']} lacks {len(missing)} of the sample's IDs")
        answers_path = out_dir / f"answers_{qid}.jsonl"
        block = compare_question(qid, q["ids"], evaluated, read_answers(answers_path))
        block["evaluated"] = q["evaluated"]
        block["records_file"] = q["records_file"]
        block["answers_file"] = f"{folder}/answers_{qid}.jsonl"
        block["answers_sha256"] = sha256_file(answers_path) if answers_path.exists() else None
        block["run"] = rerun_facts(root, qid, qid, len(q["ids"]), folder=folder)
        result["questions"][qid] = block
        print(f"{qid}: {block['verdict']} ({block['answered']:,} of {block['n']:,} answered; largest absolute difference "
              f"{block['largest_absolute_difference']}; mean {block['mean_absolute_difference']})", flush=True)
    result["verdict"] = "PASS" if all(result["questions"][q]["verdict"] == "PASS" for q in RUN_ORDER) else "FAIL"
    (out_dir / "usage_check.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"the usage check's {name} run: {result['verdict']} ({folder}/usage_check.json)", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("sample", "compare"))
    parser.add_argument("--root", type=Path, default=ROOT, help="the project folder (default: this file's)")
    parser.add_argument("--rerun", action="store_true", help="the second run (docs/prompt_session10.md, corrections 5 and 6)")
    parser.add_argument("--fixed", action="store_true", help="the third run, on the fixed release (correction 7)")
    parser.add_argument("--fixed2", action="store_true", help="the fourth run, on the second fix (correction 9)")
    args = parser.parse_args(argv)
    if args.rerun + args.fixed + args.fixed2 > 1:
        parser.error("give at most one of --rerun, --fixed and --fixed2")
    if args.command == "sample":
        if args.fixed2:
            sample_fixed(args.root, FIXED2_DIR, PLAN_FIXED2)
        else:
            (sample_fixed if args.fixed else sample_rerun if args.rerun else sample)(args.root)
        return 0
    if args.fixed2:
        result = compare_fixed(args.root, FIXED2_DIR, PLAN_FIXED2, "fourth")
    else:
        result = (compare_fixed if args.fixed else compare_rerun if args.rerun else compare)(args.root)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

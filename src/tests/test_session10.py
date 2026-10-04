"""Session 10's tests (docs/prompt_session10.md, step 2), on made-up inputs written under tmp_path; no weights are
downloaded and no network is used. Two parts, each under a comment header:

1. src/eval/usage_check_v0_5.py and scripts/usage_check_v0_5.sh (THE USAGE CHECK): the sample is the same when drawn
   twice and holds the fixed counts; compare passes identical files, fails a changed top answer on a record whose
   evaluated top probability is at least 0.6, does not count a near tie against (a), fails a mean difference above 0.02
   and a readmission difference above 0.0001, and counts a record answered with an error as a failure; the script passes
   bash -n, checks SHA256SUMS.txt before it runs anything, and, with a stand-in uv, runs the questions in order, logs
   each run, runs a question again on the CPU after an out-of-memory error and ends with compare. Test names test_usage_*.
   The second run (corrections 5 and 6): sample --rerun copies the first run's 60 Evidence Inference IDs unchanged from
   the 6,144-token records, refuses a file whose SHA-256 is not the one given, and leaves the first run's files alone;
   compare --rerun passes identical files, keeps PubMedQA's first-run verdict, and reports readmission's diagnosis run
   (the tree libraries imported first) beside its own run without changing its verdict; the second script passes
   bash -n, checks SHA256SUMS.txt before it runs anything, and, with a stand-in uv whose direct readmission run faults,
   runs the diagnosis, then Evidence Inference, then compare --rerun. Test names test_usage_rerun_*.
   The fixed release, the third run (correction 7) and the fourth (correction 9): scripts/predict_v0_5.py, as the second
   fix changed it, reads the records' questions before torch is loaded, loads xgboost and catboost first only for a file
   of readmit_window records (python -X importtime), and on macOS refuses a mixed file and the two load orders that
   fail; sample and compare with --fixed and --fixed2, a sample written once; scripts/fix_release_v0_5.sh passes bash -n,
   reads no token, holds the second fix's SHA-256 values, stops before anything when the release differs from its sums,
   keeps the files it replaces (the released ones, or the first fix's), puts the second fix in place, runs the three
   questions and compare --fixed2, does not write over a run's files, and uploads (with a stand-in hf) only after a PASS
   and a typed yes. Test names test_fix_*, test_fix2_* and test_usage_fixed*.
2. src/eval/calibrate_s9_long_3_session10.py (S9_LONG_3): on made-up logits it calls session 8's rule
   (calibrate_session7.py's choose(), through calibrate_session9.py) on the 654 clean validation prompts, writes the files
   ROOT names with every validation row, scores the test split with article-resampled intervals, and changes no other
   file. Test names test_long3_*.

Run: python3 -m pytest src/tests/test_session10.py (with TMPDIR and --basetemp as HOW TO WORK says).
"""
import csv
import hashlib
import json
import os
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_SRC = Path(__file__).resolve().parents[1]
PROJECT = _PROJECT_SRC.parent
sys.path[:0] = [str(_PROJECT_SRC / "eval"), str(_PROJECT_SRC)]

import usage_check_v0_5 as U  # noqa: E402
import calibrate_s9_long_3_session10 as L  # noqa: E402

SCRIPT = PROJECT / "scripts" / "usage_check_v0_5.sh"
OPTIONS = {U.EI: ["significantly increased", "significantly decreased", "no significant difference"],
           U.PQA: ["yes", "no", "maybe"], U.RW: ["0", "1", "2"]}
PREFIX = {U.EI: "evidence", U.PQA: "pubmedqa", U.RW: "uci"}


# ---------------------------------------------------------------- 1. the usage check

def usage_probs(rng):
    p = rng.dirichlet([0.6, 0.6, 0.6])
    return [float(f"{x:.9g}") for x in p]


def usage_layout(root, seed=1):
    """A made-up project folder: the three records files and evaluated files with the plan's counts."""
    rng = np.random.default_rng(seed)
    for qid in U.RUN_ORDER:
        src, evaluated = U.SOURCES[qid]
        n = U.EXPECTED[qid]
        ids = [f"{PREFIX[qid]}-{k}" for k in range(n)]
        (root / src).parent.mkdir(parents=True, exist_ok=True)
        (root / evaluated).parent.mkdir(parents=True, exist_ok=True)
        order = list(ids)
        random.Random(seed).shuffle(order)  # the records file's order differs from the evaluated file's
        with open(root / src, "w", encoding="utf-8") as f:
            for i in order:
                f.write(json.dumps({"id": i, "qid": qid, "state": {"text": f"made-up {i}", "word": "naïve"}},
                                   ensure_ascii=False) + "\n")
        with open(root / evaluated, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "label", "p0", "p1", "p2"])
            for i in ids:
                w.writerow([i, int(rng.integers(0, 3))] + [f"{x:.9g}" for x in usage_probs(rng)])
    return root


def usage_evaluated(root, qid):
    return U.evaluated_rows(root / U.SOURCES[qid][1])[1]


def usage_write_evaluated(root, qid, rows):
    with open(root / U.SOURCES[qid][1], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p0", "p1", "p2"])
        for i, (label, p) in rows.items():
            w.writerow([i, label] + [f"{x:.9g}" for x in p])


def usage_write_answers(root, qid, ids, probs_of, errors=()):
    path = root / U.OUT_DIR / f"answers_{qid}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            if i in errors:
                a = {"id": i, "qid": qid, "error": "the prompt alone does not fit 6,144 tokens"}
            else:
                p = probs_of(i)
                a = {"id": i, "qid": qid, "system": "made-up", "probabilities": dict(zip(OPTIONS[qid], p)),
                     "answer": OPTIONS[qid][int(np.argmax(p))], "passes": 1, "notes": []}
            f.write(json.dumps(a) + "\n")


def usage_identical_answers(root, data):
    for qid in U.RUN_ORDER:
        ev = usage_evaluated(root, qid)
        usage_write_answers(root, qid, data["questions"][qid]["ids"], lambda i, ev=ev: ev[i][1])


def test_usage_sample_same_twice_and_fixed_counts(tmp_path):
    root = usage_layout(tmp_path)
    first = U.sample(root)
    files1 = {q: (root / U.records_name(q)).read_bytes() for q in U.RUN_ORDER}
    second = U.sample(root)
    files2 = {q: (root / U.records_name(q)).read_bytes() for q in U.RUN_ORDER}
    for q in U.RUN_ORDER:
        assert first["questions"][q]["ids"] == second["questions"][q]["ids"]
        assert files1[q] == files2[q]
    counts = {q: len(first["questions"][q]["ids"]) for q in U.RUN_ORDER}
    assert counts == {U.EI: 60, U.PQA: 500, U.RW: 6998}
    # Evidence Inference: random.Random(20261003).sample from the evaluated IDs, kept in the evaluated file's order
    order = U.evaluated_rows(root / U.SOURCES[U.EI][1])[0]
    expected = set(random.Random(20261003).sample(order, 60))
    assert first["questions"][U.EI]["ids"] == [i for i in order if i in expected]
    # each record copied unchanged, byte for byte, from its records file
    for q in U.RUN_ORDER:
        source = {json.loads(line)["id"]: line for line in (root / U.SOURCES[q][0]).read_bytes().splitlines(keepends=True)}
        written = (root / U.records_name(q)).read_bytes().splitlines(keepends=True)
        assert [json.loads(line)["id"] for line in written] == first["questions"][q]["ids"]
        assert all(line == source[json.loads(line)["id"]] for line in written)
        block = json.loads((root / U.OUT_DIR / "ids.json").read_text(encoding="utf-8"))["questions"][q]
        for key, rel in (("records_source_sha256", U.SOURCES[q][0]), ("evaluated_sha256", U.SOURCES[q][1]),
                         ("records_file_sha256", U.records_name(q))):
            assert block[key] == hashlib.sha256((root / rel).read_bytes()).hexdigest()
    assert U.check_records(root) == 0
    with open(root / U.records_name(U.PQA), "ab") as f:
        f.write(b" ")
    assert U.check_records(root) == 1


def test_usage_sample_refuses_wrong_counts(tmp_path):
    root = usage_layout(tmp_path)
    rows = usage_evaluated(root, U.PQA)
    rows.pop(next(iter(rows)))
    usage_write_evaluated(root, U.PQA, rows)
    with pytest.raises(SystemExit):
        U.sample(root)


def test_usage_compare_identical_files_pass(tmp_path):
    root = usage_layout(tmp_path)
    data = U.sample(root)
    usage_identical_answers(root, data)
    out = U.compare(root)
    assert out["verdict"] == "PASS"
    assert all(out["questions"][q]["verdict"] == "PASS" for q in U.RUN_ORDER)
    assert out["questions"][U.RW]["largest_absolute_difference"] < 1e-9
    saved = json.loads((root / U.OUT_DIR / "usage_check.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "PASS" and set(saved["versions"]) >= {"python", "torch", "transformers", "xgboost", "catboost"}


def test_usage_compare_changed_confident_top_answer_fails(tmp_path):
    root = usage_layout(tmp_path)
    data = U.sample(root)
    ids = data["questions"][U.EI]["ids"]
    ev = usage_evaluated(root, U.EI)
    ev[ids[0]] = (0, [0.9, 0.06, 0.04])
    usage_write_evaluated(root, U.EI, ev)
    usage_identical_answers(root, data)
    usage_write_answers(root, U.EI, ids, lambda i: [0.06, 0.9, 0.04] if i == ids[0] else ev[i][1])
    out = U.compare(root)["questions"][U.EI]
    assert out["mean_absolute_difference"] <= 0.02  # (b) holds, so the failure is (a)'s
    assert out["criterion_a"] == "FAIL" and out["criterion_b"] == "PASS" and out["verdict"] == "FAIL"
    assert [c["id"] for c in out["top_answer_changed_at_0_6_or_more"]] == [ids[0]]


def test_usage_compare_near_tie_not_counted(tmp_path):
    root = usage_layout(tmp_path)
    data = U.sample(root)
    ids = data["questions"][U.EI]["ids"]
    ev = usage_evaluated(root, U.EI)
    ev[ids[1]] = (0, [0.5, 0.45, 0.05])
    usage_write_evaluated(root, U.EI, ev)
    usage_identical_answers(root, data)
    usage_write_answers(root, U.EI, ids, lambda i: [0.45, 0.5, 0.05] if i == ids[1] else ev[i][1])
    out = U.compare(root)["questions"][U.EI]
    assert out["criterion_a"] == "PASS" and out["verdict"] == "PASS"
    listed = {b["id"]: b for b in out["evaluated_top_probability_below_0_6"]}
    assert ids[1] in listed and listed[ids[1]]["same"] is False
    assert listed[ids[1]]["evaluated_top"] == 0 and listed[ids[1]]["released_top"] == 1


def test_usage_compare_mean_difference_above_limit_fails(tmp_path):
    root = usage_layout(tmp_path)
    data = U.sample(root)
    ids = data["questions"][U.EI]["ids"]
    ev = {i: (0, [0.9, 0.05, 0.05]) for i in usage_evaluated(root, U.EI)}
    usage_write_evaluated(root, U.EI, ev)
    usage_identical_answers(root, data)
    usage_write_answers(root, U.EI, ids, lambda i: [0.86, 0.09, 0.05])  # every top answer kept; mean 0.0267
    out = U.compare(root)["questions"][U.EI]
    assert out["criterion_a"] == "PASS" and out["criterion_b"] == "FAIL" and out["verdict"] == "FAIL"
    assert out["mean_absolute_difference"] == pytest.approx(0.08 / 3)
    usage_write_answers(root, U.EI, ids, lambda i: [0.885, 0.065, 0.05])  # mean 0.01
    assert U.compare(root)["questions"][U.EI]["verdict"] == "PASS"


def test_usage_compare_readmission_limit(tmp_path):
    root = usage_layout(tmp_path)
    data = U.sample(root)
    ids = data["questions"][U.RW]["ids"]
    ev = usage_evaluated(root, U.RW)
    usage_identical_answers(root, data)

    def moved(i, d):
        p = list(ev[i][1])
        return [p[0] + d, p[1] - d, p[2]] if i == ids[5] else p

    usage_write_answers(root, U.RW, ids, lambda i: moved(i, 0.0002))
    out = U.compare(root)
    assert out["questions"][U.RW]["verdict"] == "FAIL" and out["questions"][U.RW]["rows_above_0_0001"] == 1
    assert out["verdict"] == "FAIL"
    usage_write_answers(root, U.RW, ids, lambda i: moved(i, 0.00005))
    assert U.compare(root)["questions"][U.RW]["verdict"] == "PASS"


def test_usage_compare_error_record_is_a_failure(tmp_path):
    root = usage_layout(tmp_path)
    data = U.sample(root)
    usage_identical_answers(root, data)
    ids = data["questions"][U.PQA]["ids"]
    ev = usage_evaluated(root, U.PQA)
    usage_write_answers(root, U.PQA, ids, lambda i: ev[i][1], errors={ids[3]})
    out = U.compare(root)["questions"][U.PQA]
    assert out["verdict"] == "FAIL" and out["answered"] == len(ids) - 1
    assert out["failures"][0]["id"] == ids[3] and "error" in out["failures"][0]["reason"]
    # a question with no answers at all (its run failed) fails too
    (root / U.OUT_DIR / f"answers_{U.EI}.jsonl").write_text("", encoding="utf-8")
    assert U.compare(root)["questions"][U.EI]["verdict"] == "FAIL"


def test_usage_script_passes_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    text = SCRIPT.read_text(encoding="utf-8")
    # the release and record checks come before the first run of predict_v0_5.py
    assert text.index("shasum -a 256 -c SHA256SUMS.txt") < text.index('"$RELEASE/predict_v0_5.py"')
    assert text.index("check_records") < text.index('"$RELEASE/predict_v0_5.py"')
    assert "xgboost==3.2.0" in text and "catboost==1.2.10" in text


FAKE_UV = r'''#!/usr/bin/env python3
"""A stand-in for uv: logs each call; runs python -c and the usage check's module for real; plays predict_v0_5.py."""
import csv, json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as f:
    f.write(" ".join(args) + "\n")
assert args[0] == "run"
rest = args[1:]
while rest and rest[0] == "--with":
    rest = rest[2:]
assert rest[0] == "python"
rest = rest[1:]
if rest[0] == "-c":
    code = rest[1]
    if "import torch" in code:
        print("mps")
        sys.exit(0)
    sys.exit(subprocess.run([sys.executable, "-c", code]).returncode)
if rest[0].endswith("predict_v0_5.py"):
    opts = dict(zip(rest[1::2], rest[2::2]))
    records, device, out = opts["--records"], opts["--device"], opts["--out"]
    rows = [json.loads(l) for l in open(records) if l.strip()]
    qid = rows[0]["qid"]
    if os.environ.get("FAKE_OOM") == qid and device != "cpu":
        print("RuntimeError: MPS backend out of memory (MPS allocated: 18.0 GB)", file=sys.stderr)
        sys.exit(1)
    ids = json.load(open("results/fiorillo_v0_5/usage_check/ids.json"))["questions"][qid]
    ev = {r["id"]: [float(r["p0"]), float(r["p1"]), float(r["p2"])] for r in csv.DictReader(open(ids["evaluated"]))}
    names = {"evidence_effect": ["significantly increased", "significantly decreased", "no significant difference"],
             "pubmedqa_answer": ["yes", "no", "maybe"], "readmit_window": ["0", "1", "2"]}[qid]
    with open(out, "w") as f:
        for r in rows:
            p = ev[r["id"]]
            f.write(json.dumps({"id": r["id"], "qid": qid, "system": "stand-in", "probabilities": dict(zip(names, p)),
                                "answer": names[p.index(max(p))], "passes": 1, "notes": []}) + "\n")
    print(f"answered {len(rows)} {qid} records on {device}")
    sys.exit(0)
sys.exit(subprocess.run([sys.executable] + rest).returncode)
'''


def usage_script_project(tmp_path, matching_release=True):
    """A made-up project folder for the script: the sample, a stand-in release whose SHA256SUMS.txt lists its files, the
    usage check's module, and a copy of the script whose fixed SHA-256 of SHA256SUMS.txt is the stand-in's."""
    root = usage_layout(tmp_path / "project")
    (root / "pyproject.toml").write_text("[project]\nname = 'made-up'\n", encoding="utf-8")
    U.sample(root)
    release = root / U.RELEASE
    (release / "readmit_window").mkdir(parents=True, exist_ok=True)
    (release / "predict_v0_5.py").write_text("# stand-in\n", encoding="utf-8")
    (release / "readmit_window" / "pool.json").write_text(json.dumps({"weights": {"xgboost_tuned": 0.4, "catboost_tuned": 0.6}}),
                                                          encoding="utf-8")
    lines = []
    for p in sorted(release.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(release).as_posix()}\n")
    (release / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")
    if not matching_release:
        (release / "predict_v0_5.py").write_text("# changed after the sums were taken\n", encoding="utf-8")
    sums = hashlib.sha256((release / "SHA256SUMS.txt").read_bytes()).hexdigest()
    (root / "scripts").mkdir(exist_ok=True)
    text = SCRIPT.read_text(encoding="utf-8")
    text = text.replace("SUMS_SHA256=a04b52f0f3a83165201b7157c8d518f04ba93cc3cdc4f8a6040eb4f12ce80344", f"SUMS_SHA256={sums}")
    (root / "scripts" / "usage_check_v0_5.sh").write_text(text, encoding="utf-8")
    (root / "src" / "eval").mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT / "src" / "eval" / "usage_check_v0_5.py", root / "src" / "eval" / "usage_check_v0_5.py")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(FAKE_UV, encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", FAKE_UV_LOG=str(tmp_path / "uv_calls.txt"),
               HF_HOME=str(tmp_path / "hf"))
    return root, env


def test_usage_script_stops_before_running_when_the_release_differs(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env = usage_script_project(tmp_path, matching_release=False)
    res = subprocess.run(["bash", "scripts/usage_check_v0_5.sh"], cwd=root, env=env, capture_output=True, text=True)
    assert res.returncode != 0 and "STOP" in res.stderr
    calls = (tmp_path / "uv_calls.txt").read_text() if (tmp_path / "uv_calls.txt").exists() else ""
    assert "predict_v0_5.py" not in calls and calls == ""  # nothing ran, not even the record check
    # SHA256SUMS.txt itself must have the fixed SHA-256
    root2, env2 = usage_script_project(tmp_path / "second")
    text = (root2 / "scripts" / "usage_check_v0_5.sh").read_text(encoding="utf-8")
    sums = hashlib.sha256((root2 / U.RELEASE / "SHA256SUMS.txt").read_bytes()).hexdigest()
    (root2 / "scripts" / "usage_check_v0_5.sh").write_text(text.replace(f"SUMS_SHA256={sums}", "SUMS_SHA256=" + "0" * 64),
                                                           encoding="utf-8")
    res2 = subprocess.run(["bash", "scripts/usage_check_v0_5.sh"], cwd=root2, env=env2, capture_output=True, text=True)
    assert res2.returncode != 0 and "FIXED INPUTS" in res2.stderr


def test_usage_script_end_to_end_with_a_stand_in_uv(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env = usage_script_project(tmp_path)
    env["FAKE_OOM"] = U.EI
    res = subprocess.run(["bash", "scripts/usage_check_v0_5.sh"], cwd=root, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = (tmp_path / "uv_calls.txt").read_text().splitlines()
    predicts = [c for c in calls if "predict_v0_5.py" in c]
    order = [next(q for q in U.RUN_ORDER if f"usage_check_{q}_records" in c) for c in predicts]
    assert order == [U.RW, U.PQA, U.EI, U.EI]  # Evidence Inference again, on the CPU, after its out-of-memory error
    assert all("--with xgboost==3.2.0 --with catboost==1.2.10" in c for c in predicts)
    assert "--device auto" in predicts[2] and "--device cpu" in predicts[3]
    assert calls.index(next(c for c in calls if "check_records" in c)) < calls.index(predicts[0])
    assert "usage_check_v0_5.py compare" in calls[-1]
    log = (root / U.OUT_DIR / f"run_{U.EI}.log").read_text(encoding="utf-8")
    assert "out of memory" in log and "usage_check out_of_memory:" in log and log.count("usage_check start_utc:") == 2
    parsed = U.read_log(root / U.OUT_DIR / f"run_{U.EI}.log")
    assert parsed["rerun_on_cpu"] and parsed["device_resolved"] == "cpu" and parsed["exit_status"] == "0"
    out = json.loads((root / U.OUT_DIR / "usage_check.json").read_text(encoding="utf-8"))
    assert out["verdict"] == "PASS"
    assert out["questions"][U.EI]["run"]["rerun_on_cpu"] is True and out["questions"][U.EI]["run"]["device"] == "cpu"
    assert out["questions"][U.PQA]["run"]["device"] == "mps"
    assert out["questions"][U.RW]["run"]["device"].startswith("cpu (the pool's members with weight are trees")
    assert out["questions"][U.PQA]["run"]["seconds_per_record"] is not None


# ---------------------------------------------------------------- 1b. the usage check's second run (corrections 5 and 6)

RERUN_SCRIPT = PROJECT / "scripts" / "usage_check_v0_5_rerun.sh"


def rerun_write_long_source(root):
    """A made-up LONG_SOURCE: every Evidence Inference record of the first run's source with a longer state."""
    src, _ = U.SOURCES[U.EI]
    rows = [json.loads(l) for l in open(root / src, encoding="utf-8") if l.strip()]
    (root / U.LONG_SOURCE).parent.mkdir(parents=True, exist_ok=True)
    with open(root / U.LONG_SOURCE, "w", encoding="utf-8") as f:
        for r in reversed(rows):  # a different order from the evaluated file's
            r = dict(r, state=dict(r["state"], text=r["state"]["text"] + ", with its 6,144-token selection"))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256((root / U.LONG_SOURCE).read_bytes()).hexdigest()


def rerun_write_answers(root, name, qid, ids, probs_of):
    path = root / U.RERUN_DIR / f"answers_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            p = probs_of(i)
            f.write(json.dumps({"id": i, "qid": qid, "system": "made-up", "probabilities": dict(zip(OPTIONS[qid], p)),
                                "answer": OPTIONS[qid][int(np.argmax(p))], "passes": 1, "notes": []}) + "\n")


def rerun_first_run(root):
    """The first run's sample and its usage_check.json (identical answers, so PubMedQA passes there)."""
    usage_layout(root)
    data = U.sample(root)
    usage_identical_answers(root, data)
    U.compare(root)
    return data


def test_usage_rerun_sample_copies_the_long_records(tmp_path):
    root = tmp_path
    first = rerun_first_run(root)
    long_sha = rerun_write_long_source(root)
    before = {p: p.read_bytes() for p in (root / U.OUT_DIR).iterdir() if p.is_file()}
    before.update({root / U.records_name(q): (root / U.records_name(q)).read_bytes() for q in U.RUN_ORDER})
    with pytest.raises(SystemExit):
        U.sample_rerun(root)  # the made-up file is not the one correction 5 names
    data = U.sample_rerun(root, long_sha256=long_sha)
    ids = first["questions"][U.EI]["ids"]
    assert data["run_order"] == [U.RW, U.EI] and data["questions"][U.EI]["ids"] == ids
    lines = U.record_lines(root / U.LONG_SOURCE)
    assert (root / U.LONG_RECORDS).read_bytes() == b"".join(lines[i][1] for i in ids)
    assert data["questions"][U.RW] == first["questions"][U.RW]
    assert data["questions"][U.EI]["records_source_sha256"] == long_sha
    assert U.check_records(root, rerun=True) == 0 and U.check_records(root) == 0
    assert all(p.read_bytes() == b for p, b in before.items())  # the first run's files are untouched
    raw = (root / U.LONG_RECORDS).read_bytes()
    (root / U.LONG_RECORDS).write_bytes(raw.replace(b"made-up", b"madE-up", 1))  # one changed byte, still JSON
    assert U.check_records(root, rerun=True) == 1


def test_usage_rerun_sample_refuses_missing_ids(tmp_path):
    root = tmp_path
    first = rerun_first_run(root)
    rerun_write_long_source(root)
    keep = [l for l in open(root / U.LONG_SOURCE, encoding="utf-8") if json.loads(l)["id"] != first["questions"][U.EI]["ids"][0]]
    (root / U.LONG_SOURCE).write_text("".join(keep), encoding="utf-8")
    with pytest.raises(SystemExit):
        U.sample_rerun(root, long_sha256=None)


def test_usage_rerun_compare_pass_and_pubmedqa_carried_over(tmp_path):
    root = tmp_path
    rerun_first_run(root)
    data = U.sample_rerun(root, long_sha256=rerun_write_long_source(root))
    for qid in U.RERUN_ORDER:
        ev = usage_evaluated(root, qid)
        rerun_write_answers(root, qid, qid, data["questions"][qid]["ids"], lambda i, ev=ev: ev[i][1])
    out = U.compare_rerun(root)
    assert out["verdict"] == "PASS"
    assert out["questions"][U.PQA]["verdict"] == "PASS" and out["questions"][U.PQA]["carried_over"].endswith("usage_check.json")
    assert "trees_first" not in out["questions"][U.RW]
    assert out["questions"][U.EI]["records_file"] == U.LONG_RECORDS
    assert (root / U.RERUN_DIR / "usage_check.json").exists()
    assert U.main(["compare", "--rerun", "--root", str(root)]) == 0


def test_usage_rerun_compare_diagnosis_does_not_change_the_verdict(tmp_path):
    root = tmp_path
    rerun_first_run(root)
    data = U.sample_rerun(root, long_sha256=rerun_write_long_source(root))
    ids_rw, ids_ei = data["questions"][U.RW]["ids"], data["questions"][U.EI]["ids"]
    (root / U.RERUN_DIR / f"answers_{U.RW}.jsonl").write_text("", encoding="utf-8")  # the direct run answered nothing
    (root / U.RERUN_DIR / f"run_{U.RW}.log").write_text(
        "usage_check question: readmit_window\nusage_check mode: plain\nusage_check start_utc: 2026-10-03T10:00:00Z\n"
        "Fatal Python error: Segmentation fault\n\nCurrent thread 0x1 (most recent call first):\n"
        '  File "models/fiorillo_v0_5/release/predict_v0_5.py", line 645 in tree_probs\n'
        "usage_check end_utc: 2026-10-03T10:00:02Z\nusage_check exit_status: 139\n", encoding="utf-8")
    (root / U.RERUN_DIR / f"run_{U.TREES_FIRST}.log").write_text(
        "usage_check question: readmit_window\nusage_check mode: trees_first\nusage_check start_utc: 2026-10-03T10:00:03Z\n"
        "usage_check end_utc: 2026-10-03T10:00:09Z\nusage_check exit_status: 0\n", encoding="utf-8")
    ev_rw, ev_ei = usage_evaluated(root, U.RW), usage_evaluated(root, U.EI)
    rerun_write_answers(root, U.TREES_FIRST, U.RW, ids_rw, lambda i: ev_rw[i][1])
    rerun_write_answers(root, U.EI, U.EI, ids_ei, lambda i: ev_ei[i][1])
    out = U.compare_rerun(root)
    rw = out["questions"][U.RW]
    assert rw["verdict"] == "FAIL" and rw["answered"] == 0 and rw["run"]["exit_status"] == "139"
    assert "line 645 in tree_probs" in rw["run"]["fault"] and rw["run"]["mode"] == "plain"
    assert rw["trees_first"]["verdict"] == "PASS" and rw["trees_first"]["run"]["mode"] == "trees_first"
    assert rw["trees_first"]["run"]["fault"] is None
    assert out["questions"][U.EI]["verdict"] == "PASS" and out["verdict"] == "FAIL"


def test_usage_rerun_compare_reads_a_key_after_the_fault_report(tmp_path):
    """The second run's real log: the fault handler's report did not end its line, so the script's end_utc key follows it
    on the same line; the second run's reading still finds it, and the first run's reading is unchanged."""
    root = tmp_path
    rerun_first_run(root)
    data = U.sample_rerun(root, long_sha256=rerun_write_long_source(root))
    ids_rw, ids_ei = data["questions"][U.RW]["ids"], data["questions"][U.EI]["ids"]
    (root / U.RERUN_DIR / f"answers_{U.RW}.jsonl").write_text("", encoding="utf-8")
    log = root / U.RERUN_DIR / f"run_{U.RW}.log"
    log.write_text("usage_check question: readmit_window\nusage_check mode: plain\nusage_check start_utc: 2026-10-03T09:26:03Z\n"
                   "Fatal Python error: Segmentation faultFatal Python error: usage_check end_utc: 2026-10-03T09:26:05Z\n"
                   "usage_check exit_status: 139\n", encoding="utf-8")
    (root / U.RERUN_DIR / f"run_{U.TREES_FIRST}.log").write_text(
        "usage_check question: readmit_window\nusage_check mode: trees_first\nusage_check start_utc: 2026-10-03T09:26:05Z\n"
        "usage_check end_utc: 2026-10-03T09:26:08Z\nusage_check exit_status: 0\n", encoding="utf-8")
    ev_rw, ev_ei = usage_evaluated(root, U.RW), usage_evaluated(root, U.EI)
    rerun_write_answers(root, U.TREES_FIRST, U.RW, ids_rw, lambda i: ev_rw[i][1])
    rerun_write_answers(root, U.EI, U.EI, ids_ei, lambda i: ev_ei[i][1])
    assert U.read_log(log).get("end_utc") is None  # the first run's reading, unchanged
    assert U.read_log(log, anywhere=True)["end_utc"] == "2026-10-03T09:26:05Z"
    assert U.fault_report(log) == "Fatal Python error: Segmentation faultFatal Python error: "
    out = U.compare_rerun(root)
    rw = out["questions"][U.RW]
    assert rw["run"]["seconds"] == 2.0 and rw["run"]["exit_status"] == "139"
    assert rw["verdict"] == "FAIL" and rw["trees_first"]["verdict"] == "PASS" and out["verdict"] == "FAIL"


def test_usage_rerun_script_passes_bash_n():
    assert subprocess.run(["bash", "-n", str(RERUN_SCRIPT)]).returncode == 0
    text = RERUN_SCRIPT.read_text(encoding="utf-8")
    first_predict = text.index('python "$RELEASE/predict_v0_5.py"') if 'python "$RELEASE/predict_v0_5.py"' in text else None
    first_run = text.index("run_one readmit_window ")
    assert text.index("shasum -a 256 -c SHA256SUMS.txt") < first_run and text.index("check_records(rerun=True)") < first_run
    assert first_predict is not None
    assert text.count("PYTHONFAULTHANDLER=1 uv run") == 2 and "import xgboost, catboost" in text
    assert "xgboost==3.2.0" in text and "catboost==1.2.10" in text and "compare --rerun" in text
    assert "usage_check_evidence_effect_long_records.jsonl" in text and "results/fiorillo_v0_5/usage_check/rerun" in text


FAKE_UV_RERUN = r'''#!/usr/bin/env python3
"""A stand-in for uv in the second run's tests: logs each call; runs python -c and the module for real; plays
predict_v0_5.py, run directly or through the trees-first launcher; FAKE_SEGV names a question whose direct run faults."""
import csv, json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as f:
    f.write(" ".join(args) + "\n")
assert args[0] == "run"
rest = args[1:]
while rest and rest[0] == "--with":
    rest = rest[2:]
assert rest[0] == "python"
rest = rest[1:]
launcher = False
if rest[0] == "-c":
    code = rest[1]
    if "runpy" in code:
        launcher = True
        rest = ["predict_v0_5.py"] + rest[2:]
    elif "import torch" in code:
        print("mps")
        sys.exit(0)
    else:
        sys.exit(subprocess.run([sys.executable, "-c", code]).returncode)
if rest[0].endswith("predict_v0_5.py"):
    assert os.environ.get("PYTHONFAULTHANDLER") == "1"
    opts = dict(zip(rest[1::2], rest[2::2]))
    records, device, out = opts["--records"], opts["--device"], opts["--out"]
    rows = [json.loads(l) for l in open(records) if l.strip()]
    qid = rows[0]["qid"]
    if os.environ.get("FAKE_SEGV") == qid and not launcher:
        print("Fatal Python error: Segmentation fault", file=sys.stderr)
        print('  File "models/fiorillo_v0_5/release/predict_v0_5.py", line 645 in tree_probs', file=sys.stderr)
        sys.exit(139)
    ids = json.load(open("results/fiorillo_v0_5/usage_check/rerun/ids.json"))["questions"][qid]
    ev = {r["id"]: [float(r["p0"]), float(r["p1"]), float(r["p2"])] for r in csv.DictReader(open(ids["evaluated"]))}
    names = {"evidence_effect": ["significantly increased", "significantly decreased", "no significant difference"],
             "readmit_window": ["0", "1", "2"]}[qid]
    with open(out, "w") as f:
        for r in rows:
            p = ev[r["id"]]
            f.write(json.dumps({"id": r["id"], "qid": qid, "system": "stand-in", "probabilities": dict(zip(names, p)),
                                "answer": names[p.index(max(p))], "passes": 1, "notes": []}) + "\n")
    print(f"answered {len(rows)} {qid} records on {device}" + (" (tree libraries first)" if launcher else ""))
    sys.exit(0)
sys.exit(subprocess.run([sys.executable] + rest).returncode)
'''


def rerun_script_project(tmp_path, matching_release=True):
    """The first run's made-up project (usage_script_project), its usage_check.json, the second run's sample, a copy of
    the second script with the stand-in release's SHA-256 of SHA256SUMS.txt, and the second run's stand-in uv."""
    root, env = usage_script_project(tmp_path, matching_release=True)
    data = json.loads((root / U.OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    usage_identical_answers(root, data)
    U.compare(root)
    U.sample_rerun(root, long_sha256=rerun_write_long_source(root))
    sums = hashlib.sha256((root / U.RELEASE / "SHA256SUMS.txt").read_bytes()).hexdigest()
    if not matching_release:
        (root / U.RELEASE / "predict_v0_5.py").write_text("# changed after the sums were taken\n", encoding="utf-8")
    text = RERUN_SCRIPT.read_text(encoding="utf-8")
    text = text.replace("SUMS_SHA256=a04b52f0f3a83165201b7157c8d518f04ba93cc3cdc4f8a6040eb4f12ce80344", f"SUMS_SHA256={sums}")
    (root / "scripts" / "usage_check_v0_5_rerun.sh").write_text(text, encoding="utf-8")
    uv = tmp_path / "bin" / "uv"
    uv.write_text(FAKE_UV_RERUN, encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return root, env


def test_usage_rerun_script_stops_before_running_when_the_release_differs(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env = rerun_script_project(tmp_path, matching_release=False)
    res = subprocess.run(["bash", "scripts/usage_check_v0_5_rerun.sh"], cwd=root, env=env, capture_output=True, text=True)
    assert res.returncode != 0 and "STOP" in res.stderr
    calls = (tmp_path / "uv_calls.txt").read_text() if (tmp_path / "uv_calls.txt").exists() else ""
    assert calls == ""  # nothing ran, not even the record check


def test_usage_rerun_script_end_to_end_with_a_stand_in_uv(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env = rerun_script_project(tmp_path)
    env["FAKE_SEGV"] = U.RW
    first_files = {p: p.read_bytes() for p in (root / U.OUT_DIR).iterdir() if p.is_file()}
    res = subprocess.run(["bash", "scripts/usage_check_v0_5_rerun.sh"], cwd=root, env=env, capture_output=True, text=True)
    assert res.returncode == 1, res.stdout + res.stderr  # readmission's direct run faulted, so the second run fails
    calls = (tmp_path / "uv_calls.txt").read_text().splitlines()
    predicts = [c for c in calls if "predict_v0_5.py" in c]
    assert len(predicts) == 3
    assert "usage_check_readmit_window_records.jsonl" in predicts[0] and "runpy" not in predicts[0]
    assert "usage_check_readmit_window_records.jsonl" in predicts[1] and "runpy" in predicts[1]
    assert "usage_check_evidence_effect_long_records.jsonl" in predicts[2] and "--device auto" in predicts[2]
    assert all("--with xgboost==3.2.0 --with catboost==1.2.10" in c for c in predicts)
    assert calls.index(next(c for c in calls if "check_records(rerun=True)" in c)) < calls.index(predicts[0])
    assert "usage_check_v0_5.py compare --rerun" in calls[-1]
    out = json.loads((root / U.RERUN_DIR / "usage_check.json").read_text(encoding="utf-8"))
    rw = out["questions"][U.RW]
    assert rw["verdict"] == "FAIL" and rw["run"]["exit_status"] == "139" and "tree_probs" in rw["run"]["fault"]
    assert rw["trees_first"]["verdict"] == "PASS" and rw["trees_first"]["run"]["mode"] == "trees_first"
    assert out["questions"][U.EI]["verdict"] == "PASS" and out["questions"][U.PQA]["verdict"] == "PASS"
    assert out["verdict"] == "FAIL"
    assert all(p.read_bytes() == b for p, b in first_files.items())  # the first run's files are untouched
    env.pop("FAKE_SEGV")
    res2 = subprocess.run(["bash", "scripts/usage_check_v0_5_rerun.sh"], cwd=root, env=env, capture_output=True, text=True)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    out2 = json.loads((root / U.RERUN_DIR / "usage_check.json").read_text(encoding="utf-8"))
    assert out2["verdict"] == "PASS" and "trees_first" not in out2["questions"][U.RW]


# ---------------------------------------------------------------- 1c. the fixed release: the third and fourth runs (corrections 7 and 9)

FIX_SCRIPT = PROJECT / "scripts" / "fix_release_v0_5.sh"
PREDICT = PROJECT / "scripts" / "predict_v0_5.py"
CARD = PROJECT / "docs" / "model_card_v0_5.md"
KEEP = Path("models") / "fiorillo_v0_5" / "replaced_session10"
KEEP_FIX1 = Path("models") / "fiorillo_v0_5" / "replaced_session10_fix1"


def predict_module():
    """scripts/predict_v0_5.py imported as a module (not run as __main__, so it loads no tree library first)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict_v0_5_second_fix", PREDICT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fix2_reads_the_questions_and_refuses_on_macos_what_would_fault(tmp_path):
    """The second fix (correction 9): asked_questions reads the --records file's questions before torch is loaded, and
    macos_check refuses, on macOS only, a file that mixes readmit_window with the other questions and the two load
    orders that fail there; the docstring and the card note the change."""
    P = predict_module()
    rw, mixed = tmp_path / "rw.jsonl", tmp_path / "mixed.jsonl"
    rw.write_text(json.dumps({"id": "a", "qid": "readmit_window", "state": {}}) + "\n\n", encoding="utf-8")
    mixed.write_text(json.dumps({"id": "a", "qid": "readmit_window", "state": {}}) + "\n"
                     + json.dumps({"id": "b", "qid": "pubmedqa_answer", "state": {}}) + "\n", encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text("not json\n", encoding="utf-8")
    assert P.asked_questions(["--records", str(rw), "--out", "x"]) == {"readmit_window"}
    assert P.asked_questions([f"--records={mixed}"]) == {"readmit_window", "pubmedqa_answer"}
    assert P.asked_questions(["--out", "x"]) is None
    assert P.asked_questions(["--records", str(tmp_path / "none.jsonl")]) is None
    assert P.asked_questions(["--records", str(tmp_path / "bad.jsonl")]) is None
    assert P.ASKED is None  # imported, not run

    def by(*qids):
        return {q: ([{"id": q, "qid": q}] if q in qids else []) for q in P.QIDS}

    for platform in ("linux", "win32"):
        for trees in (False, True):
            P.macos_check(platform, by(P.RW, P.PQA, P.EI), trees)  # nothing is refused outside macOS
    cases = [((P.RW, P.PQA), False, P.MACOS_MIXED), ((P.RW, P.EI), True, P.MACOS_MIXED),
             ((P.RW,), False, P.MACOS_TORCH_FIRST), ((P.PQA,), True, P.MACOS_TREES_FIRST),
             ((P.EI,), True, P.MACOS_TREES_FIRST)]
    for qids, trees, message in cases:
        with pytest.raises(SystemExit) as err:
            P.macos_check("darwin", by(*qids), trees)
        assert str(err.value) == message
    P.macos_check("darwin", by(P.RW), True)
    P.macos_check("darwin", by(P.PQA, P.EI), False)
    assert "Change since the release's first public revision (56e5944)" in P.__doc__ and "refuses" in P.__doc__
    card = CARD.read_text(encoding="utf-8")
    assert "Change since the first public revision (56e5944)" in card and "a file of their own" in card


def import_order(tmp_path, qid):
    """The modules that predict_v0_5.py, run as a script on a one-record file of qid, finishes importing, in order
    (python -X importtime). The run stops early, at the made-up release folder, which holds nothing."""
    records = tmp_path / f"{qid}.jsonl"
    records.write_text(json.dumps({"id": "a", "qid": qid, "state": {}}) + "\n", encoding="utf-8")
    (tmp_path / "release").mkdir(exist_ok=True)
    res = subprocess.run([sys.executable, "-X", "importtime", str(PREDICT), "--records", str(records),
                          "--release", str(tmp_path / "release"), "--device", "cpu",
                          "--out", str(tmp_path / f"answers_{qid}.jsonl")], capture_output=True, text=True, timeout=900)
    order = []
    for line in res.stderr.splitlines():
        if line.startswith("import time:") and line.count("|") == 2:
            name = line.rsplit("|", 1)[1].strip()
            if name and name not in order:
                order.append(name)
    assert "torch" in order, res.stderr[-2000:]
    assert res.returncode != 0 and not (tmp_path / f"answers_{qid}.jsonl").exists()
    return order


def test_fix2_loads_the_tree_libraries_first_only_for_a_readmission_file(tmp_path):
    rw = import_order(tmp_path, U.RW)
    assert rw.index("xgboost") < rw.index("torch") and rw.index("catboost") < rw.index("torch")
    pqa = import_order(tmp_path, U.PQA)
    assert not {"xgboost", "catboost"} & set(pqa[:pqa.index("torch")])


def fixed_first_and_second(root):
    rerun_first_run(root)
    U.sample_rerun(root, long_sha256=rerun_write_long_source(root))
    return U.sample_fixed(root)


def write_matching_answers(root, folder, data, mode):
    for qid in U.RUN_ORDER:
        ev = usage_evaluated(root, qid)
        path = root / folder / f"answers_{qid}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in data["questions"][qid]["ids"]:
                p = ev[i][1]
                f.write(json.dumps({"id": i, "qid": qid, "probabilities": dict(zip(OPTIONS[qid], p)),
                                    "answer": OPTIONS[qid][int(np.argmax(p))], "passes": 1, "notes": []}) + "\n")
        (root / folder / f"run_{qid}.log").write_text(
            f"usage_check question: {qid}\nusage_check mode: {mode}\nusage_check start_utc: 2026-10-03T11:00:00Z\n\n"
            "usage_check end_utc: 2026-10-03T11:00:10Z\nusage_check exit_status: 0\n", encoding="utf-8")


def test_usage_fixed_sample_and_compare(tmp_path):
    root = tmp_path
    data = fixed_first_and_second(root)
    assert data["run_order"] == list(U.RUN_ORDER)
    assert data["questions"][U.EI]["records_file"] == U.LONG_RECORDS
    assert data["questions"][U.RW]["records_file"] == U.records_name(U.RW)
    assert U.check_records(root, fixed=True) == 0
    write_matching_answers(root, U.FIXED_DIR, data, "fixed")
    out = U.compare_fixed(root)
    assert out["verdict"] == "PASS" and set(out["questions"]) == set(U.RUN_ORDER)
    assert out["questions"][U.EI]["run"]["seconds"] == 10.0 and out["questions"][U.EI]["run"]["mode"] == "fixed"
    (root / U.FIXED_DIR / f"answers_{U.RW}.jsonl").write_text("", encoding="utf-8")
    assert U.compare_fixed(root)["verdict"] == "FAIL"
    with pytest.raises(SystemExit):
        U.main(["sample", "--rerun", "--fixed", "--root", str(root)])


def test_usage_fixed2_sample_and_compare(tmp_path):
    """The fourth run (correction 9): the same record files as the third, in fixed2/; a sample is written once."""
    root = tmp_path
    third = fixed_first_and_second(root)
    assert U.main(["sample", "--fixed2", "--root", str(root)]) == 0
    data = json.loads((root / U.FIXED2_DIR / "ids.json").read_text(encoding="utf-8"))
    assert data["plan"] == U.PLAN_FIXED2 and data["questions"] == third["questions"]
    assert U.check_records(root, fixed2=True) == 0
    with pytest.raises(SystemExit):
        U.main(["sample", "--fixed2", "--root", str(root)])
    with pytest.raises(SystemExit):
        U.main(["sample", "--fixed", "--fixed2", "--root", str(root)])
    write_matching_answers(root, U.FIXED2_DIR, data, "fixed2")
    assert U.main(["compare", "--fixed2", "--root", str(root)]) == 0
    out = json.loads((root / U.FIXED2_DIR / "usage_check.json").read_text(encoding="utf-8"))
    assert out["plan"] == U.PLAN_FIXED2 and out["verdict"] == "PASS" and out["questions"][U.PQA]["run"]["mode"] == "fixed2"
    assert not (root / U.FIXED_DIR / "usage_check.json").exists()


FAKE_UV_FIXED = r'''#!/usr/bin/env python3
"""A stand-in for uv in the third and fourth runs' tests: logs each call; runs python -c and the module for real; plays
the fixed predict_v0_5.py from the evaluated files of the run folder's ids.json; FAKE_BAD names a question whose answers
it moves by 0.1."""
import csv, json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as f:
    f.write(" ".join(args) + "\n")
assert args[0] == "run"
rest = args[1:]
while rest and rest[0] == "--with":
    rest = rest[2:]
assert rest[0] == "python"
rest = rest[1:]
if rest[0] == "-c":
    code = rest[1]
    if "import torch" in code:
        print("mps")
        sys.exit(0)
    sys.exit(subprocess.run([sys.executable, "-c", code]).returncode)
if rest[0].endswith("predict_v0_5.py"):
    assert os.environ.get("PYTHONFAULTHANDLER") == "1"
    opts = dict(zip(rest[1::2], rest[2::2]))
    rows = [json.loads(l) for l in open(opts["--records"]) if l.strip()]
    qid = rows[0]["qid"]
    ids = json.load(open(os.path.join(os.path.dirname(opts["--out"]), "ids.json")))["questions"][qid]
    ev = {r["id"]: [float(r["p0"]), float(r["p1"]), float(r["p2"])] for r in csv.DictReader(open(ids["evaluated"]))}
    names = {"evidence_effect": ["significantly increased", "significantly decreased", "no significant difference"],
             "pubmedqa_answer": ["yes", "no", "maybe"], "readmit_window": ["0", "1", "2"]}[qid]
    with open(opts["--out"], "w") as f:
        for r in rows:
            p = ev[r["id"]]
            if os.environ.get("FAKE_BAD") == qid:
                p = [min(1.0, p[0] + 0.1), max(0.0, p[1] - 0.1), p[2]]
            f.write(json.dumps({"id": r["id"], "qid": qid, "probabilities": dict(zip(names, p)),
                                "answer": names[p.index(max(p))], "passes": 1, "notes": []}) + "\n")
    sys.exit(0)
sys.exit(subprocess.run([sys.executable] + rest).returncode)
'''

FAKE_HF_FIXED = r'''#!/usr/bin/env python3
"""A stand-in for hf: answers whoami, models info, models ls and upload from a state file; logs each call."""
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_HF_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
state_path = Path(os.environ["FAKE_HF_STATE"])
state = json.loads(state_path.read_text(encoding="utf-8"))
def entry(data):
    e = {"size": len(data), "blob_id": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
    if len(data) > 200:
        e["lfs"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pointer_size": 134}
    return e
if args[:2] == ["auth", "whoami"]:
    print(json.dumps({"user": state["user"]}))
elif args[:2] == ["models", "info"]:
    print(json.dumps({"id": args[2], "private": state["private"], "sha": state["sha"],
                      "siblings": [{"rfilename": n} for n in sorted(state["files"])]}))
elif args[:2] == ["models", "ls"]:
    print(json.dumps([{"path": n, **v} for n, v in sorted(state["files"].items())]))
elif args[:1] == ["upload"]:
    folder = Path(args[2])
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            state["files"][p.relative_to(folder).as_posix()] = entry(p.read_bytes())
    state["sha"] = hashlib.sha1(json.dumps(state["files"], sort_keys=True).encode()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    print("uploaded")
else:
    print(f"unexpected hf call: {args}", file=sys.stderr)
    sys.exit(2)
'''


def fix_project(tmp_path, state_of_release="released"):
    """A made-up project: the runs' samples (the fourth run's in fixed2/), a made-up release whose predict_v0_5.py and
    README.md stand for the released ones (or, with state_of_release "fix1", for the first fix's, the released ones kept
    in replaced_session10/ as the first fix left them), the second fix's predict_v0_5.py and card copied from this
    project, the module, and a copy of the script with the made-up files' SHA-256 values in place of the real ones; a
    stand-in uv and hf on PATH."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'made-up'\n", encoding="utf-8")
    fixed_first_and_second(root)
    U.sample_fixed(root, U.FIXED2_DIR, U.PLAN_FIXED2)
    rel = root / U.RELEASE
    (rel / "evidence_effect").mkdir(parents=True)
    files = {"predict_v0_5.py": "# the released script (made up)\n", "README.md": "# Fiorillo v0.5 (made up)\n",
             "LICENSE": "made-up license\n", "evidence_effect/adapter.safetensors": "w" * 500}
    for name, text in files.items():
        (rel / name).write_text(text, encoding="utf-8")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sums_text = lambda pred, card: "".join(
        f"{pred if n == 'predict_v0_5.py' else card if n == 'README.md' else sha(rel / n)}  {n}\n" for n in sorted(files))
    (rel / "SHA256SUMS.txt").write_text(sums_text(sha(rel / "predict_v0_5.py"), sha(rel / "README.md")), encoding="utf-8")
    old = {"OLD_SUMS": sha(rel / "SHA256SUMS.txt"), "OLD_PREDICT": sha(rel / "predict_v0_5.py"), "OLD_README": sha(rel / "README.md")}
    fix1_predict, fix1_card = "# the first fix's script (made up)\n", "# Fiorillo v0.5, first fix (made up)\n"
    fix1 = {"FIX1_PREDICT": hashlib.sha256(fix1_predict.encode()).hexdigest(),
            "FIX1_README": hashlib.sha256(fix1_card.encode()).hexdigest()}
    fix1["FIX1_SUMS"] = hashlib.sha256(sums_text(fix1["FIX1_PREDICT"], fix1["FIX1_README"]).encode()).hexdigest()
    if state_of_release == "fix1":
        keep = root / KEEP
        keep.mkdir(parents=True)
        for n in ("predict_v0_5.py", "README.md", "SHA256SUMS.txt"):
            shutil.copy(rel / n, keep / n)
        (rel / "predict_v0_5.py").write_text(fix1_predict, encoding="utf-8")
        (rel / "README.md").write_text(fix1_card, encoding="utf-8")
        (rel / "SHA256SUMS.txt").write_text(sums_text(fix1["FIX1_PREDICT"], fix1["FIX1_README"]), encoding="utf-8")
        assert sha(rel / "SHA256SUMS.txt") == fix1["FIX1_SUMS"]
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    shutil.copy(PREDICT, root / "scripts" / "predict_v0_5.py")
    shutil.copy(CARD, root / "docs" / "model_card_v0_5.md")
    (root / "src" / "eval").mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT / "src" / "eval" / "usage_check_v0_5.py", root / "src" / "eval" / "usage_check_v0_5.py")
    new_sums = hashlib.sha256(sums_text(sha(PREDICT), sha(CARD)).encode()).hexdigest()
    text = FIX_SCRIPT.read_text(encoding="utf-8")
    for name, value in list(old.items()) + list(fix1.items()) + [("NEW_SUMS", new_sums)]:
        text, k = __import__("re").subn(rf"^{name}=[0-9a-f]{{64}}$", f"{name}={value}", text, flags=__import__("re").M)
        assert k == 1, name
    (root / "scripts" / "fix_release_v0_5.sh").write_text(text, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("uv", FAKE_UV_FIXED), ("hf", FAKE_HF_FIXED)):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    remote = {}
    for p in sorted(rel.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            e = {"size": len(data), "blob_id": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
            if len(data) > 200:
                e["lfs"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pointer_size": 134}
            remote[p.relative_to(rel).as_posix()] = e
    state = tmp_path / "hf_state.json"
    state.write_text(json.dumps({"user": "johanneli", "private": False, "sha": "56e5944" + "0" * 33, "files": remote}),
                     encoding="utf-8")
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", FAKE_UV_LOG=str(tmp_path / "uv_calls.txt"),
               FAKE_HF_LOG=str(tmp_path / "hf_calls.jsonl"), FAKE_HF_STATE=str(state), EDDM_PYTHON=sys.executable,
               EDDM_HF="hf", HOME=str(tmp_path))
    return root, env, state


def fix_run(root, env, answer="yes\n", *args):
    return subprocess.run(["bash", "scripts/fix_release_v0_5.sh", *args], cwd=root, env=env, input=answer,
                          capture_output=True, text=True, timeout=300)


def test_fix_script_passes_bash_n_and_reads_no_token():
    assert subprocess.run(["bash", "-n", str(FIX_SCRIPT)]).returncode == 0
    code = [l for l in FIX_SCRIPT.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#")]
    for pattern in ("--token", "HF_TOKEN", "huggingface/token", "--private", "--public", "visibility", "rm ", "--delete"):
        assert not any(pattern in l for l in code), pattern
    text = "\n".join(code)
    assert text.index("shasum -a 256 -c SHA256SUMS.txt") < text.index('"$RELEASE/predict_v0_5.py" --records')
    assert text.index("compare --fixed2") < text.index('$HF upload') and text.index("read -r ANSWER") < text.index('$HF upload')
    for name in ("NEW_SUMS", "NEW_PREDICT", "NEW_README"):
        assert __import__("re").search(rf"^{name}=(?!0{{64}}$)[0-9a-f]{{64}}$", FIX_SCRIPT.read_text(encoding="utf-8"),
                                       flags=__import__("re").M), name
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert f"NEW_PREDICT={sha(PREDICT)}" in FIX_SCRIPT.read_text(encoding="utf-8")
    assert f"NEW_README={sha(CARD)}" in FIX_SCRIPT.read_text(encoding="utf-8")


def test_fix_script_stops_before_anything_when_the_release_differs(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env, _ = fix_project(tmp_path)
    (root / U.RELEASE / "LICENSE").write_text("changed\n", encoding="utf-8")
    res = fix_run(root, env)
    assert res.returncode != 0 and "STOP" in res.stderr
    assert not (tmp_path / "uv_calls.txt").exists() and not (tmp_path / "hf_calls.jsonl").exists()
    assert not (root / KEEP).exists() and not (root / KEEP_FIX1).exists()


def test_fix_script_end_to_end_with_stand_ins(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env, state = fix_project(tmp_path)
    rel = root / U.RELEASE
    old = {n: (rel / n).read_bytes() for n in ("predict_v0_5.py", "README.md", "SHA256SUMS.txt")}
    res = fix_run(root, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert {n: (root / KEEP / n).read_bytes() for n in old} == old  # the released files are kept
    assert not (root / KEEP_FIX1).exists()
    assert (rel / "predict_v0_5.py").read_bytes() == PREDICT.read_bytes()
    assert (rel / "README.md").read_bytes() == CARD.read_bytes()
    calls = (tmp_path / "uv_calls.txt").read_text().splitlines()
    predicts = [c for c in calls if "predict_v0_5.py --records" in c]
    assert [next(q for q in U.RUN_ORDER if f"usage_check_{q}" in c) for c in predicts] == list(U.RUN_ORDER)
    assert "usage_check_evidence_effect_long_records.jsonl" in predicts[2]
    assert all(f"--out {U.FIXED2_DIR}/answers_" in c for c in predicts)
    assert "usage_check_v0_5.py compare --fixed2" in calls[-1]
    out = json.loads((root / U.FIXED2_DIR / "usage_check.json").read_text(encoding="utf-8"))
    assert out["verdict"] == "PASS" and out["release"]["predict_v0_5.py_sha256"] == hashlib.sha256(PREDICT.read_bytes()).hexdigest()
    assert "usage_check mode: fixed2" in (root / U.FIXED2_DIR / f"run_{U.EI}.log").read_text(encoding="utf-8")
    assert not (root / U.FIXED_DIR / "usage_check.json").exists()  # the third run's folder is not written
    hf = [json.loads(l) for l in (tmp_path / "hf_calls.jsonl").read_text().splitlines()]
    assert [c[:2] for c in hf] == [["auth", "whoami"], ["models", "info"], ["upload", "johanneli/fiorillo-v0.5"],
                                   ["models", "info"], ["models", "ls"]]
    assert hf[2][2:4] == [str(Path(U.RELEASE)), "."] and "--private" not in hf[2]
    st = json.loads(state.read_text(encoding="utf-8"))
    assert st["files"]["predict_v0_5.py"]["size"] == PREDICT.stat().st_size and st["private"] is False
    assert f"is at revision {st['sha']}" in res.stdout
    # run again: the folder already holds the second fix, and the fourth run's files are not written over
    logs = {q: (root / U.FIXED2_DIR / f"run_{q}.log").read_bytes() for q in U.RUN_ORDER}
    res2 = fix_run(root, env)
    assert res2.returncode != 0 and "already holds the second fix" in res2.stdout and "written once" in res2.stderr
    assert {q: (root / U.FIXED2_DIR / f"run_{q}.log").read_bytes() for q in U.RUN_ORDER} == logs
    # --upload only: no predictions, the same upload
    n_calls = len((tmp_path / "uv_calls.txt").read_text().splitlines())
    res3 = fix_run(root, env, "yes\n", "--upload")
    assert res3.returncode == 0, res3.stdout + res3.stderr
    assert len((tmp_path / "uv_calls.txt").read_text().splitlines()) == n_calls


def test_fix_script_from_the_first_fix(tmp_path):
    """The Mac's state after the third run: the release folder holds the first fix, the released files are kept in
    replaced_session10/; the second fix keeps the first fix's files in replaced_session10_fix1/."""
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env, state = fix_project(tmp_path, "fix1")
    rel = root / U.RELEASE
    kept = {n: (root / KEEP / n).read_bytes() for n in ("predict_v0_5.py", "README.md", "SHA256SUMS.txt")}
    first = {n: (rel / n).read_bytes() for n in kept}
    res = fix_run(root, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "the first fix's predict_v0_5.py" in res.stdout
    assert {n: (root / KEEP_FIX1 / n).read_bytes() for n in first} == first
    assert {n: (root / KEEP / n).read_bytes() for n in kept} == kept
    assert (rel / "predict_v0_5.py").read_bytes() == PREDICT.read_bytes() and (rel / "README.md").read_bytes() == CARD.read_bytes()
    assert json.loads((root / U.FIXED2_DIR / "usage_check.json").read_text(encoding="utf-8"))["verdict"] == "PASS"


def test_fix_script_does_not_upload_after_a_failed_run_or_without_yes(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not on this machine")
    root, env, state = fix_project(tmp_path / "failed")
    env["FAKE_BAD"] = U.PQA
    res = fix_run(root, env)
    assert res.returncode != 0 and "did not PASS" in res.stderr
    assert not (tmp_path / "failed" / "hf_calls.jsonl").exists()
    res2 = fix_run(root, env, "yes\n", "--upload")  # --upload refuses without a passing fourth run
    assert res2.returncode != 0 and not (tmp_path / "failed" / "hf_calls.jsonl").exists()
    env.pop("FAKE_BAD")
    res3 = fix_run(root, env)  # the failed run's files stay; the run is not repeated over them
    assert res3.returncode != 0 and "written once" in res3.stderr and not (tmp_path / "failed" / "hf_calls.jsonl").exists()
    root, env, state = fix_project(tmp_path / "no")
    res4 = fix_run(root, env, "no\n")
    assert res4.returncode != 0 and "not uploaded" in res4.stderr
    hf = [json.loads(l) for l in (tmp_path / "no" / "hf_calls.jsonl").read_text().splitlines()]
    assert not any(c[:1] == ["upload"] for c in hf)
    assert json.loads(state.read_text(encoding="utf-8"))["sha"].startswith("56e5944")


# ---------------------------------------------------------------- 2. s9_long_3's calibration

def long3_write_logits(path, ids, labels, z):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "z0", "z1", "z2"])
        for i, y, row in zip(ids, labels, z):
            w.writerow([i, int(y)] + [repr(float(v)) for v in row])


def long3_layout(root, seed=3):
    """Made-up logits of s9_long_3 (validation: 700 prompts, 654 of them clean; test: 90 prompts from 30 articles),
    overconfident by a planted factor of 3, with the clean list and the test records."""
    rng = np.random.default_rng(seed)
    results, records9, records8 = root / "results", root / "records9", root / "records8"

    def made(n, prefix):
        y = rng.integers(0, 3, n)
        signal = np.eye(3)[y] * 1.2 + rng.normal(0, 1.0, (n, 3))
        return [f"{prefix}-{k}" for k in range(n)], y, 3.0 * signal

    v_ids, v_y, v_z = made(700, "evidence-v")
    t_ids, t_y, t_z = made(90, "evidence-t")
    long3_write_logits(results / "logits" / f"s9_long_3_evidence_effect_validation_logits.csv", v_ids, v_y, v_z)
    long3_write_logits(results / "logits" / f"s9_long_3_evidence_effect_test_logits.csv", t_ids, t_y, t_z)
    (records9 / "clean").mkdir(parents=True)
    clean = sorted(rng.choice(700, 654, replace=False))
    with open(records9 / "clean" / "evidence_effect_validation.jsonl", "w", encoding="utf-8") as f:
        for k in clean:
            f.write(json.dumps({"id": v_ids[k], "qid": "evidence_effect", "label": int(v_y[k])}) + "\n")
    records8.mkdir(parents=True)
    with open(records8 / "evidence_effect_test.jsonl", "w", encoding="utf-8") as f:
        for k, i in enumerate(t_ids):
            f.write(json.dumps({"id": i, "qid": "evidence_effect", "label": int(t_y[k]), "group": {"pmcid": f"PMC{1000 + k // 3}"}}) + "\n")
    (results / "calibration_s9.json").write_text('{"made-up": "session 9"}\n', encoding="utf-8")
    return results, records9, records8, {v_ids[k] for k in clean}


def test_long3_calls_session8_rule_and_writes_the_files(tmp_path, monkeypatch):
    results, records9, records8, clean = long3_layout(tmp_path)
    before = (results / "calibration_s9.json").read_bytes()
    calls = []
    real_choose = L.C9.choose

    def counting_choose(logp, y, *a, **k):
        calls.append(len(y))
        return real_choose(logp, y, *a, **k)

    monkeypatch.setattr(L.C9, "choose", counting_choose)
    data, test = L.run(results, records9, records8, resamples=60)
    assert calls == [654]  # session 8's rule, once, on the 654 clean prompts
    entry = data["s9_long_3"]["evidence_effect"]
    assert entry["chosen"] in ("temperature", "temperature_intercepts") and entry["n_calibrate"] == 654
    assert set(entry["families"]) == {"temperature", "temperature_intercepts"}
    for p in L.outputs(results):
        assert p.exists(), p
    pred = results / "predictions"
    v_rows = list(csv.DictReader(open(pred / "s9_long_3_evidence_effect_validation.csv", encoding="utf-8")))
    assert len(v_rows) == 700  # every validation row, not only the clean ones
    assert len(list(csv.DictReader(open(pred / "s9_long_3_temperature_evidence_effect_test.csv", encoding="utf-8")))) == 90
    # the planted overconfidence is corrected on the clean prompts
    assert entry["validation_calibrate_items"]["log_loss"] < entry["validation_calibrate_items"]["raw_log_loss"]
    assert set(test["metrics"]) == {"log_loss", "ece", "brier", "accuracy", "macro_f1"} and test["articles"] == 30
    for m, v in test["metrics"].items():
        assert v["low"] <= v["high"], m
    assert test["resamples"] == 60 and test["seed"] == 20260927
    assert (results / "calibration_s9.json").read_bytes() == before  # no file of session 9 changed


def test_long3_refuses_files_it_did_not_write(tmp_path):
    results, records9, records8, _ = long3_layout(tmp_path)
    (results / "predictions").mkdir(parents=True, exist_ok=True)
    (results / "predictions" / "s9_long_3_evidence_effect_test.csv").write_text("id,label,p0,p1,p2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        L.run(results, records9, records8, resamples=20)


def test_long3_runs_again_over_its_own_files(tmp_path):
    results, records9, records8, _ = long3_layout(tmp_path)
    first = L.run(results, records9, records8, resamples=20)[1]
    second = L.run(results, records9, records8, resamples=20)[1]
    assert first["metrics"] == second["metrics"]

"""Session 8 (docs/prompt_session8.md, step 16): the figure check of docs/fiorillo_v0_4.md.

It reads the document as written and checks three of its sections: "## How each stage was run", "## Results (test
split)" and "## Checks not pre-registered (corrections 14 and 15)", each from its heading to the next line that starts
with "## ".
It finds every number in them and compares each with its source at the printed precision: a number printed with four
decimals passes when it is the source value rounded to four decimals, a count when it equals the source, a time when it
equals the source's time of day. Table cells are matched to their sources by the table's heading ("### Table N.") and
its rows, which tables() below fixes for each table. Numbers in the prose are matched, in order, to the source keys of
the line's template (REGIONS below): a line is found by its shape, its text with every number replaced, so a line whose
numbers were changed still finds its keys. Some lines are checked whole against their source text (verbatim() below).
Each key is resolved here, from the results files, the pod's stage logs and docs/log.md, by this script's own code.
Every reading ("s8_ei better", "no difference was detected", PASS, FAIL and so on) is checked against its interval
and the direction of its metric, as compare_session8.py reads them, or against the rule's own result.

The other sections (the licenses, what was built, the method, the published results beside G3, the corrections)
hold the plan's fixed settings and figures cited to publications; they are not checked here.

Before the document is checked, a self-test plants errors in copies of it, one at a time (PLANTED below), and the script
stops unless each one is reported. Then the document itself is checked, and the script stops unless every number and
reading passes. It writes results/fiorillo_v0_4/figure_sources.json (every number: where it is, as written, its source
and the source's value) and docs/fiorillo_v0_4_figure_check.md (every number and reading, one row each, PASS or MISMATCH).

The same tables and templates also render the checked sections (the render subcommand), so each figure is written and
checked from one definition of its source; the check reads the document itself, not the rendering. Session 7's
check_fiorillo_v0_3_figures.py is the model for this script.

Usage: python3 src/eval/check_fiorillo_v0_4_figures.py            (the self-test, then the check)
       python3 src/eval/check_fiorillo_v0_4_figures.py render     (prints the checked sections with their figures)
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ("## How each stage was run", "## Results (test split)",
            "## Checks not pre-registered (corrections 14 and 15)")
SHORT = {SECTIONS[0]: "stages", SECTIONS[1]: "results", SECTIONS[2]: "corrections 14 and 15"}

EI = "evidence_effect"
LOWER = {"log_loss", "brier", "ece"}
UNRANKED = {"calibration_intercept_30d", "calibration_slope_30d"}
METRIC_LABEL = {
    "log_loss": "Log loss", "brier": "Brier score", "ece": "ECE", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
    "f1_increased": "F1, significantly increased", "f1_decreased": "F1, significantly decreased",
    "f1_no_difference": "F1, no significant difference", "macro_f1_tuned": "Macro-F1, decision weights",
    "quadratic_weighted_kappa": "Quadratic weighted kappa", "auroc_30d": "AUROC, 30 days",
    "calibration_intercept_30d": "Calibration intercept, 30 days", "calibration_slope_30d": "Calibration slope, 30 days"}
EV_LABEL = {EI: "Evidence Inference", "pubmedqa_answer": "PubMedQA", "readmit_window": "Readmission"}
BASE_METRICS = ("log_loss", "brier", "ece", "accuracy", "macro_f1")
EI_F1 = ("f1_increased", "f1_decreased", "f1_no_difference")
EI_METRICS = BASE_METRICS + EI_F1
READMIT_EXTRA = ("macro_f1_tuned", "quadratic_weighted_kappa", "auroc_30d", "calibration_intercept_30d", "calibration_slope_30d")
SEEDS = ("s8_ei_1", "s8_ei_2", "s8_ei_3")
EI_ORDER = ["prior", "s8_ei", "s8_ei_temperature", "s8_ei-raw", "s8_ei_1", "s8_ei_1_temperature", "s8_ei_1-raw", "s8_ei_2",
            "s8_ei_2_temperature", "s8_ei_2-raw", "s8_ei_3", "s8_ei_3_temperature", "s8_ei_3-raw", "gemma4_31b",
            "gemma4_31b_temperature", "gemma4_31b-raw"]
PQA_ORDER = ["prior", "arm_laya", "s7_a3", "s7_a3_temperature", "s7_b3", "s7_b3_temperature", "s7_b3-raw", "s8_a3",
             "s8_a3_temperature", "s8_a3-raw", "gemma4_31b", "gemma4_31b_temperature", "gemma4_31b-raw"]
READMIT_ORDER = ["prior", "arm_laya", "lightgbm", "lightgbm_tuned", "blend_v0_1", "xgboost_tuned", "catboost_tuned",
                 "readmission_pool", "s7_a3", "s7_b3", "s8_a3", "s8_a3_temperature", "s8_a3-raw"]
SYSTEM_NAME = {"s8_ei": "s8_ei", "prior": "the prior", "gemma4_31b": "Gemma 4 31B-it", "arm_laya": "v0.1", "s7_b3": "v0.4 (s7_b3)"}
HANDOFF_RATES = (0, 10, 20, 30, 50)
SUBSETS = ("all", "evidence inside the input", "evidence outside the input", "no usable evidence offsets",
           "evidence in the abstract", "evidence not in the abstract")
STEP13 = "session 8, step 13: test predictions"
STEP9 = "session 8, steps 8 and 9: the credit"
STEP10 = "session 8, step 10: the pilot and the cost rule"
STEP15 = "session 8, step 15: results"
OSF = "session 8: the OSF registration filed"

# identifiers and definitional constants that are not figures (listed in the check document)
N = r"\d+(?:,\d{3})*"
SKIP = [r"`[^`]*`", r"\S*[_/]\S*", r"\bv0\.[0-4]\b", r"\bQwen3(?:-4B(?:-Base)?)?\b", r"\bGemma 4(?: 31B-it| E2B-it)?\b",
        r"\bI-\d{2}\b", r"\bG[1-6]\b", r"\bH[1-4]\b", r"\bEvidence Inference 2\.0\b", r"\bEI 2\.0\b", r"\bvLLM 0\.30\.0\b",
        r"\bCUDA \d+(?:\.\d+)?\b", rf"\bTables? {N}(?:(?:, {N})*,? (?:and|to) {N})?",
        rf"\bcorrections? {N}(?:(?:, {N})*,? (?:and|to) {N})?",
        rf"\bcriteri(?:on|a) {N}(?:(?:, {N})*,? (?:and|to) {N})?", rf"\bsessions? {N}(?:'s)?(?: (?:and|to) {N})?",
        rf"\bsteps? {N}(?: (?:and|to) {N})?", rf"\bseeds? {N}(?:(?:, {N})*,? (?:and|to) {N})?",
        r"\b\d{1,2} (?:September|October)(?: 202[67])?\b", r"(?<![\d.,])\b20[12]\d\b(?![.,]\d)", r"\bSHA-256\b", r"\b95 percent\b", r"\b95th\b",
        r"\b30 days\b", r"\bprints? as 0\b", r"\b80 percent chance\b", r"\blevels? [0-2](?:, [0-2])*(?: and [0-2])?\b",
        r"\b4B\b", r"\b31B\b"]
SKIP_RE = re.compile("|".join(SKIP), re.I)
TOKEN_RE = re.compile(r"\b\d{2}:\d{2}(?::\d{2})?\b|(?<![\w.])\$?-?\d+(?:,\d{3})*(?:\.\d+)?")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f4(x):
    s = f"{x:.4f}"
    return "0.0000" if s == "-0.0000" else s


def as_time(value):
    s = str(value)
    return s[11:19] if len(s) >= 19 and s[10] == "T" else s


def fmt(value, how):
    """How a source value is printed in the document."""
    if value is None:
        return "not computed"
    if how == "int":
        return f"{int(value):,}"
    if how == "c0":
        return f"{round(float(value)):,}"
    if how == "time":
        return as_time(value)
    if how == "str":
        return str(value)
    if how in ("pct0", "pct1", "pct2"):
        return f"{100 * float(value):.{int(how[3])}f}"
    if how in ("usd0", "usd2"):
        return f"${float(value):.{int(how[3])}f}"
    if how[0] in "fg":
        d = int(how[1:])
        s = f"{float(value):,.{d}f}" if how[0] == "g" else f"{float(value):.{d}f}"
        return s.lstrip("-") if float(s.replace(",", "")) == 0 else s
    raise ValueError(how)


def reading(metric, d, system_better, reference_better):
    """The reading of a difference 'system minus reference' from its 95 percent interval, as compare_session8.py
    reads it: the direction comes from the interval, not from the estimate."""
    if metric in UNRANKED:
        return "not ranked"
    if d["low"] is None or d["high"] is None:
        return "not computed"
    if d["low"] <= 0 <= d["high"]:
        return f"no difference was detected; excludes below {f4(d['low'])} and above {f4(d['high'])}"
    lower = d["high"] < 0
    if metric in LOWER:
        return system_better if lower else reference_better
    return reference_better if lower else system_better


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Sources:
    """Every source the checked sections cite, read from the repository."""

    def __init__(self, root=ROOT):
        root = Path(root)
        self.root = root
        r = root / "results" / "fiorillo_v0_4"
        self.r = r
        self.cmp = read_json(r / "comparison.json")
        self.files = {
            "RC": "release_candidate.json", "TPL": "testpreds_log.json", "CAL": "calibration_s8.json",
            "LIM": "ece_limit_evidence_effect.json", "HV": "handoff_validation.json", "HT": "handoff_test.json",
            "POW": "power.json", "POWV": "power_validation.json", "PIL": "pilot.json", "TR": "trees.json",
            "POOL": "readmission_pool.json", "POOLT": "readmission_pool_test.json", "TRT": "trees_test.json",
            "Z": "zeros.json", "LET": "large_model/gemma4_31b_letters.json", "TIM": "large_model/gemma4_31b_timing.json",
            "MAN": "records_manifest.json", "ASEL": "article_selection.json", "CC": "contamination_checks.json"}
        self.data = {k: read_json(r / f) for k, f in self.files.items() if (r / f).exists()}
        self.met = {p.name[:-len("_metrics.json")]: read_json(p) for p in sorted((r / "metrics").glob("*_metrics.json"))}
        self.rs = {p.name[:-len("_run_summary.json")]: read_json(p) for p in sorted((r / "run_summaries").glob("*_run_summary.json"))}
        self.logs = {p.stem: p.read_text(encoding="utf-8").splitlines() for p in sorted((r / "logs" / "pod").glob("*.log"))}
        log_md = root / "docs" / "log.md"
        self.log_md = log_md.read_text(encoding="utf-8").splitlines() if log_md.exists() else []
        sums = root / "models" / "fiorillo_v0_4" / "runpod_zips" / "SHA256SUMS.txt"
        self.sums = {}
        if sums.exists():
            for line in sums.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sha, size, name = line.split()
                    self.sums[name] = (sha, int(size))
        self.der = self.derive(root)

    def derive(self, root):
        out = {}
        files = {EI: root / "data" / "processed" / "session8" / "evidence_effect_test.jsonl",
                 "pubmedqa_answer": root / "data" / "processed" / "public" / "pubmedqa_answer_test.jsonl",
                 "readmit_window": root / "data" / "processed" / "public" / "readmit_window_test.jsonl"}
        for q, path in files.items():
            if not path.exists():
                continue
            rel = path.relative_to(root)
            labels = [json.loads(line)["label"] for line in open(path, encoding="utf-8") if line.strip()]
            out[f"{q}_test_items"] = (len(labels), f"{rel}: records")
            for c in range(3):
                out[f"{q}_test_label_{c}"] = (sum(1 for x in labels if x == c), f"{rel}: records with label {c}")
        zeros = self.data.get("Z", {}).get("rows_whose_true_answer_prints_as_0", {})
        rows = [z for per_split in zeros.values() for per_q in per_split.values() for z in per_q.values()]
        out["zero_rows"] = (sum(z["rows"] for z in rows), "zeros.json: the sum of rows over every system, split and question")
        out["zero_printed"] = (sum(z["true_answer_printed_zero"] for z in rows),
                               "zeros.json: the sum of true_answer_printed_zero over every system, split and question")
        out["missing_predictions"] = (len(self.cmp.get("missing_predictions", [])), "comparison.json: missing_predictions, its length")
        cc = self.data.get("CC")
        if cc:
            matches = cc["pubmedqa"]["matches"]
            for source in ("pqa_labeled", "pqa_artificial", "pqa_unlabeled"):
                out[f"cc_matches_{source}"] = (sum(1 for m in matches if m["source"] == source),
                                               f"contamination_checks.json: pubmedqa.matches with source {source}")
            for split in ("validation", "test"):
                out[f"cc_matched_{split}"] = (len(cc["pubmedqa"]["matched_articles"][split]),
                                              f"contamination_checks.json: pubmedqa.matched_articles.{split}, its length")
                out[f"cc_matched_trained_{split}"] = (
                    len(cc["pubmedqa"]["matched_articles_whose_abstract_arm_b_trained_on"][split]),
                    f"contamination_checks.json: pubmedqa.matched_articles_whose_abstract_arm_b_trained_on.{split}, its length")
                out[f"cc_dups_{split}"] = (len(cc["ei_splits"]["flagged_articles"][split]),
                                           f"contamination_checks.json: ei_splits.flagged_articles.{split}, its length")
        return out

    def comparison(self, section, ev, system, reference, metric):
        subset = ""
        if "@" in section:
            section, subset = section.split("@", 1)
        hits = [(i, c) for i, c in enumerate(self.cmp["comparisons"])
                if (c["section"], c["evaluation"], c["system"], c["reference"], c["metric"], c.get("subset", "")) ==
                (section, ev, system, reference, metric, subset)]
        if len(hits) != 1:
            raise KeyError(f"no single comparison {section} {ev} {system} minus {reference} {metric} {subset} ({len(hits)} found)")
        return hits[0]

    @staticmethod
    def walk(obj, path, name):
        """(source description, value) at a path of keys and list positions separated by '/'."""
        parts = path.split("/") if path else []
        v, steps, i = obj, [], 0
        while i < len(parts):
            if isinstance(v, list):
                step = parts[i]
                v = v[int(step)]
                i += 1
            else:
                j = i + 1
                while "/".join(parts[i:j]) not in v and j < len(parts):
                    j += 1  # a key that holds a slash, such as a file name
                step = "/".join(parts[i:j])
                v = v[step]
                i = j
            steps.append(step)
        return (f"{name} {'.'.join(steps)}" if steps else name), v

    def plog(self, stage, which, text):
        lines = [x for x in self.logs.get(stage, []) if text in x]
        if not lines:
            raise KeyError(f"logs/pod/{stage}.log has no line holding {text!r}")
        line = lines[0] if which == "first" else lines[-1]
        return f"logs/pod/{stage}.log, the {which} line holding {text!r}", line.split()[0]

    def logmd(self, marker, pattern):
        entries = [x for x in self.log_md if x.startswith("## ") and marker in x]
        if len(entries) != 1:
            raise KeyError(f"docs/log.md: {len(entries)} entries hold {marker!r}")
        m = re.search(pattern, entries[0])
        if not m:
            raise KeyError(f"docs/log.md's entry {marker!r} does not match {pattern!r}")
        text = m.group(1)
        number = text.replace(",", "")
        try:
            value = float(number) if "." in number else int(number)
        except ValueError:
            value = text
        return f"docs/log.md, the entry \"{marker}\", {pattern!r}", value

    def resolve(self, key):
        """(source description, value) for a key 'kind|arguments'."""
        kind, *a = key.split("|")
        cmp = self.cmp
        if kind == "M":  # M|evaluation|system|metric|part
            return f"comparison.json metrics.{a[0]}.{a[1]}.{a[2]}.{a[3]}", cmp["metrics"][a[0]][a[1]][a[2]][a[3]]
        if kind == "C":  # C|section[@subset]|evaluation|system|reference|metric|part
            i, c = self.comparison(*a[:5])
            if a[5] in ("system_estimate", "reference_estimate", "n"):
                return f"comparison.json comparisons[{i}].{a[5]}", c[a[5]]
            where, v = self.walk(c["difference"], a[5], "difference")
            return f"comparison.json comparisons[{i}] ({a[0]}, {a[1]}, {c['system']} minus {c['reference']}, {a[4]}).{where}", v
        if kind == "BI":  # BI|reference|part
            return self.walk(cmp["bar_intervals"][f"s8_ei_minus_{a[0]}"], "/".join(a[1:]), f"comparison.json bar_intervals.s8_ei_minus_{a[0]}")
        tables = {"SET": "settings", "RB": "release_bar", "G2": "g2_spread", "G3": "g3", "G4L": "g4_load_check",
                  "SENS": "sensitivity", "EP": "ei_prior", "ECEL": "ece_limits", "G6": "g6"}
        if kind in tables:
            return self.walk(cmp[tables[kind]], "/".join(a), f"comparison.json {tables[kind]}")
        if kind in self.files:
            if kind not in self.data:
                return f"{self.files[kind]} (not written)", None
            obj = self.data[kind]["rows_whose_true_answer_prints_as_0"] if kind == "Z" else self.data[kind]
            return self.walk(obj, "/".join(a), self.files[kind] + (" rows_whose_true_answer_prints_as_0" if kind == "Z" else ""))
        if kind == "MET":
            return self.walk(self.met[a[0]], "/".join(a[1:]), f"metrics/{a[0]}_metrics.json")
        if kind == "RS":
            return self.walk(self.rs[a[0]], "/".join(a[1:]), f"run_summaries/{a[0]}_run_summary.json")
        if kind == "PLOG":  # PLOG|stage|first or last|text
            return self.plog(a[0], a[1], "|".join(a[2:]))
        if kind == "LOGMD":  # LOGMD|entry marker|regex with one group
            return self.logmd(a[0], "|".join(a[1:]))
        if kind == "SUMS":  # SUMS|zip name or total|size
            if a[0] == "total":
                return "models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt, the sum of the sizes", sum(s for _, s in self.sums.values())
            if a[0] == "count":
                return "models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt, its lines", len(self.sums)
            return f"models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt, {a[0]}'s size", self.sums[a[0]][1]
        if kind == "D":
            v, how = self.der[a[0]]
            return f"derived: {how}", v
        raise KeyError(key)


# ---------------------------------------------------------------- the tables
# A table is (number, title, header, rows); a row is a list of cells, each a label (a plain string, checked as text) or:
#   ("num", key, how)              one number
#   ("iv", key_without_part)       "estimate (low to high)", four decimals, from key|estimate, key|low and key|high
#   ("rng", key_without_part)      "low to high", four decimals, from key|low and key|high
#   ("read", C_key_without_part, metric, system_better, reference_better)   the reading of a difference's interval
#   ("str", key)                   a text value from a source (checked as text)
#   ("pf", key)                    PASS or FAIL from a source's true or false


def present(src, ev, systems):
    """The systems in order; a system the comparison lists as missing gets a NOT BUILT row instead of being left out."""
    missing = {m.get("system") if isinstance(m, dict) else str(m) for m in src.cmp.get("missing_predictions", [])}
    out = []
    for s in systems:
        if s in src.cmp["metrics"].get(ev, {}):
            out.append((s, True))
        elif s in missing:
            out.append((s, False))
    return out


def table_metrics(src, number, ev, systems, metrics, title):
    rows = []
    for s, built in present(src, ev, systems):
        if built:
            rows.append([f"`{s}`"] + [("iv", f"M|{ev}|{s}|{m}") for m in metrics])
        else:
            rows.append([f"`{s}`"] + ["NOT BUILT" for _ in metrics])
    return (number, title, ["System"] + [METRIC_LABEL[m] for m in metrics], rows)


def has_comparison(src, section, ev, system, reference, metric):
    try:
        src.comparison(section, ev, system, reference, metric)
        return True
    except KeyError:
        return False


def table_diffs(src, number, title, specs, with_estimates=False):
    """specs: (section, evaluation, system, reference, metrics, system_better, reference_better)."""
    rows = []
    for section, ev, system, reference, metrics, sb, rb in specs:
        for m in metrics:
            row = [f"`{system}` minus `{reference}`", METRIC_LABEL[m]]
            if not has_comparison(src, section, ev, system, reference, m):
                row += (["NOT BUILT", "NOT BUILT"] if with_estimates else []) + ["NOT BUILT", "NOT BUILT"]
                rows.append(row)
                continue
            base = f"C|{section}|{ev}|{system}|{reference}|{m}"
            if with_estimates:
                row += [("num", base + "|system_estimate", "f4"), ("num", base + "|reference_estimate", "f4")]
            row += [("iv", base), ("read", base, m, sb, rb)]
            rows.append(row)
    header = ["Comparison", "Metric"] + (["System", "Reference"] if with_estimates else []) + \
             ["Difference (95 percent interval)", "Reading"]
    return (number, title, header, rows)


def tables(src):
    out = []
    # 1. the stages
    rows = [
        ["prepare", ("num", "PLOG|prepare|first|stage prepare started", "time"),
         ("num", "PLOG|prepare|last|stage prepare finished", "time"),
         "installs; the fixed revisions; EI's records rebuilt on the pod from the selection file and checked against the Mac's"],
        ["prepare_a", ("num", "PLOG|prepare_a|first|stage prepare_a started", "time"),
         ("num", "PLOG|prepare_a|last|stage prepare_a finished", "time"),
         "reader v1 in Laya's format; phase 2's records rebuilt and checked against labeler.json"],
        ["pilot", ("num", "PLOG|pilot|first|stage pilot started", "time"),
         ("num", "PLOG|pilot|last|stage pilot finished", "time"), "alone on the GPU: the two settings and the cost rule"],
        ["trees, on the pod's processors", ("num", "PLOG|trees|first|stage trees started", "time"),
         ("num", "PLOG|trees|last|stage trees finished", "time"),
         "xgboost_tuned's and catboost_tuned's searches and refits; the readmission pool on validation"],
        ["ei, `s8_ei_1`", ("num", "PLOG|ei|first|EI: s8_ei_1, seed", "time"),
         ("num", "PLOG|ei|first|EI: s8_ei_1 finished", "time"), "trained beside arm A, then beside the first three smoke tests; validation predictions"],
        ["ei, `s8_ei_2`", ("num", "PLOG|ei|first|EI: s8_ei_2, seed", "time"),
         ("num", "PLOG|ei|first|EI: s8_ei_2 finished", "time"),
         "trained beside the fourth smoke test, and otherwise alone; validation predictions"],
        ["ei, `s8_ei_3`", ("num", "PLOG|ei|first|EI: s8_ei_3, seed", "time"),
         ("num", "PLOG|ei|first|EI: s8_ei_3 finished", "time"), "trained alone on the GPU; validation predictions"],
        ["arm_a, `s8_a1`", ("num", "PLOG|arm_a|first|arm A: s8_a1 from", "time"),
         ("num", "PLOG|arm_a|first|arm A: s8_a1 done", "time"), "beside `s8_ei_1`"],
        ["arm_a, `s8_a2`", ("num", "PLOG|arm_a|first|arm A: s8_a2 from", "time"),
         ("num", "PLOG|arm_a|first|arm A: s8_a2 done", "time"), "beside `s8_ei_1`"],
        ["arm_a, `s8_a3`", ("num", "PLOG|arm_a|first|arm A: s8_a3 from", "time"),
         ("num", "PLOG|arm_a|first|arm A: s8_a3 done", "time"), "beside `s8_ei_1`; validation predictions"],
        ["prepare_large", ("num", "PLOG|prepare_large|first|stage prepare_large started", "time"),
         ("num", "PLOG|prepare_large|last|stage prepare_large finished", "time"),
         "the large model's environment and weights, in four runs within these times (corrections 7, 8 and 10)"],
        ["smoke_large", ("num", "PLOG|smoke_large|first|stage smoke_large started", "time"),
         ("num", "PLOG|smoke_large|last|stage smoke_large finished", "time"),
         "the smoke test with Gemma 4 E2B-it, in four runs within these times (corrections 9 to 12)"],
        ["large_validation", ("num", "PLOG|large_validation|first|stage large_validation started", "time"),
         ("num", "PLOG|large_validation|last|stage large_validation finished", "time"),
         "Gemma 4 31B-it alone on the GPU, on the validation items"],
        ["calibrate", ("num", "PLOG|calibrate|first|stage calibrate started", "time"),
         ("num", "PLOG|calibrate|last|stage calibrate finished", "time"), "calibration_s8.json; the ECE limit"],
        ["freeze", ("num", "PLOG|freeze|first|stage freeze started", "time"),
         ("num", "PLOG|freeze|last|stage freeze finished", "time"), "release_candidate.json"],
        ["testpreds (1 October)", ("num", "PLOG|testpreds|first|stage testpreds started", "time"),
         ("num", "PLOG|testpreds|last|stage testpreds finished", "time"),
         "after the OSF registration (correction 13): testpreds_log.json first, then every test prediction"],
        ["package", ("num", "PLOG|package|first|stage package started", "time"),
         ("num", f"LOGMD|{STEP13}|package ran from \\S+ to (\\S+) UTC", "time"), "the five zips and SHA256SUMS.txt"],
        ["copy", ("num", f"LOGMD|{STEP13}|ran from (\\S+) to \\S+ UTC with a helper", "time"),
         ("num", f"LOGMD|{STEP13}|ran from \\S+ to (\\S+) UTC with a helper", "time"),
         "the zips copied to the Mac and checked"],
    ]
    out.append((1, "The stages on the pod (UTC; 30 September, then 1 October from testpreds on)",
                ["Stage", "Started", "Finished", "What it did"], rows))
    # 2. the three EI runs
    rows = []
    for s in SEEDS:
        if s in src.met:
            rs = f"{s}_validation"
            row = [f"`{s}`", ("num", f"MET|{s}|optimizer_steps", "int"), ("num", f"MET|{s}|train_seconds", "g1"),
                   ("num", f"MET|{s}|sequences_per_second", "f2"), ("num", f"MET|{s}|peak_gpu_memory_gib", "f1"),
                   ("num", f"MET|{s}|kept_validation_selection_log_loss_exact", "f4")]
            if rs in src.rs:
                row += [("num", f"RS|{rs}|validation_per_question_raw/{EI}/accuracy", "f4"),
                        ("num", f"RS|{rs}|validation_per_question_raw/{EI}/log_loss", "f4"),
                        ("num", f"RS|{rs}|validation_per_question_raw/{EI}/ece", "f4")]
            else:
                row += ["NOT RUN", "NOT RUN", "NOT RUN"]
            rows.append(row)
        else:
            rows.append([f"`{s}`"] + ["NOT BUILT"] * 8)
    out.append((2, "The three EI runs (validation figures before calibration)",
                ["Run", "Optimizer steps", "Training seconds", "Training items a second", "Peak GPU memory, GiB",
                 "Monitoring log loss at the last step", "Validation accuracy", "Validation log loss", "Validation ECE"], rows))
    # 3. calibration
    cal = src.data.get("CAL", {})
    rows = []
    for s, q in (("s8_ei", EI), ("s8_ei_1", EI), ("s8_ei_2", EI), ("s8_ei_3", EI), ("gemma4_31b", EI),
                 ("gemma4_31b", "pubmedqa_answer"), ("s8_a3", "pubmedqa_answer"), ("s8_a3", "readmit_window")):
        if s in cal and q in cal[s] and isinstance(cal[s][q], dict) and "chosen" in cal[s][q]:
            ch = cal[s][q]["chosen"]
            rows.append([f"`{s}`", EV_LABEL[q],
                         ("num", f"CAL|{s}|{q}|families/temperature/out_of_fold_log_loss/mean", "f4"),
                         ("num", f"CAL|{s}|{q}|families/temperature_intercepts/out_of_fold_log_loss/mean", "f4"),
                         ("str", f"CAL|{s}|{q}|chosen"),
                         ("num", f"CAL|{s}|{q}|families/{ch}/temperature", "f3"),
                         ("num", f"CAL|{s}|{q}|families/{ch}/intercepts/1", "f3"),
                         ("num", f"CAL|{s}|{q}|families/{ch}/intercepts/2", "f3")])
        else:
            rows.append([f"`{s}`", EV_LABEL[q]] + ["NOT BUILT"] * 6)
    out.append((3, "Calibration chosen on validation", ["System", "Question", "Out-of-fold log loss, temperature",
                "Out-of-fold log loss, temperature and intercepts", "Chosen", "Temperature", "Intercept, second answer",
                "Intercept, third answer"], rows))
    # 4. the large model's letters and time
    rows = []
    let, tim = src.data.get("LET", {}), src.data.get("TIM", {})
    for q in (EI, "pubmedqa_answer"):
        for split in ("validation", "test"):
            if q in let and split in let[q]:
                k = f"LET|{q}|{split}"
                row = [EV_LABEL[q], split, ("num", f"{k}|items", "int"),
                       ("num", f"{k}|letters_summed_probability/median", "f6"),
                       ("num", f"{k}|letters_summed_probability/p10", "f6"),
                       ("num", f"{k}|letters_summed_probability/share_below_0_5", "f4"),
                       ("num", f"{k}|by_method/top20", "int"), ("num", f"{k}|by_method/top1000", "int"),
                       ("num", f"{k}|by_method/prompt", "int")]
                row.append(("num", f"TIM|{q}|{split}|gpu_seconds_per_item", "f3") if q in tim and split in tim[q] else "not timed")
                rows.append(row)
    out.append((4, "Gemma 4 31B-it's letters", ["Question", "Split", "Items", "Letters' summed probability, median",
                "10th percentile", "Share below one half", "Answered from the top 20", "From the top 1,000",
                "From the prompt's log-probabilities", "GPU seconds an item"], rows))
    # 5 and 6. EI on the test split
    out.append(table_metrics(src, 5, EI, EI_ORDER, BASE_METRICS, "Evidence Inference, every system (test split)"))
    out.append(table_metrics(src, 6, EI, EI_ORDER, EI_F1, "Evidence Inference, each answer's F1 (test split)"))
    # 7. G1
    specs = [("G1", EI, "s8_ei", "prior", EI_METRICS, "s8_ei better", "the prior better"),
             ("G1", EI, "s8_ei", "gemma4_31b", EI_METRICS, "s8_ei better", "Gemma 4 31B-it better")]
    out.append(table_diffs(src, 7, "G1: s8_ei minus the prior and minus Gemma 4 31B-it", specs, with_estimates=True))
    # 8. the bar
    rows = [["1", "ECE", ("num", "RB|criteria/1/values/ece", "f4"), ("num", "RB|criteria/1/values/limit", "f4"),
             ("str", "RB|criteria/1/result")]]
    for k, ref in (("2", "prior"), ("3", "gemma4_31b")):
        if f"s8_ei_minus_{ref}" in src.cmp.get("bar_intervals", {}):
            rows.append([k, f"Log loss, s8_ei minus {SYSTEM_NAME[ref]}", ("num", f"BI|{ref}|estimate", "f4"),
                         ("rng", f"BI|{ref}"), ("str", f"RB|criteria/{k}/result")])
        else:
            rows.append([k, f"Log loss, s8_ei minus {SYSTEM_NAME[ref]}", "NOT RUN", "NOT RUN",
                         ("str", f"RB|criteria/{k}/result")])
    rows.append(["4", "Macro-F1, s8_ei against Gemma 4 31B-it", ("num", "RB|criteria/4/values/s8_ei", "f4"),
                 ("num", "RB|criteria/4/values/gemma4_31b", "f4"), ("str", "RB|criteria/4/result")])
    rows.append(["The bar", "all four", "", "", ("str", "RB|overall")])
    out.append((8, "The release bar (criteria 2 and 3 with the bar's article resamples)",
                ["Criterion", "Measure", "s8_ei, or the difference", "Limit, 95 percent interval, or Gemma 4 31B-it",
                 "Result"], rows))
    # 9. G2
    specs = [("G2", EI, "s8_ei", s, EI_METRICS, "s8_ei better", f"{s} better") for s in SEEDS]
    out.append(table_diffs(src, 9, "G2: s8_ei minus each run alone (descriptive)", specs))
    # 10. G3
    rows = []
    g3 = src.cmp.get("g3", {})
    for key, scoring in (("published", "strict"), ("published_lenient", "lenient")):
        for name, p in g3.get(key, {}).items():
            part = "macro_f1" if scoring == "strict" else "lenient_macro_f1"
            how = "f3" if name.startswith("deyoung") else "f4"
            rows.append([p["source"], scoring, ("num", f"G3|{key}/{name}/value", how), ("iv", f"G3|{part}"),
                         ("str", f"G3|{key}/{name}/s8_ei")])
    out.append((10, "G3: published results (descriptive)", ["Published system", "Scoring", "Its macro-F1",
                "s8_ei's macro-F1 (95 percent interval)", "The published value against s8_ei's interval"], rows))
    # 11 and 12. PubMedQA
    out.append(table_metrics(src, 11, "pubmedqa_answer", PQA_ORDER, BASE_METRICS, "PubMedQA, three answers (test split)"))
    specs = [("G4", "pubmedqa_answer", "s7_b3", "arm_laya", BASE_METRICS, "v0.4 better", "v0.1 better"),
             ("G4", "pubmedqa_answer", "s7_b3", "gemma4_31b", BASE_METRICS, "v0.4 better", "Gemma 4 31B-it better"),
             ("G4", "pubmedqa_answer", "s8_a3", "s7_a3", BASE_METRICS, "s8_a3 better", "s7_a3 better"),
             ("G4", "pubmedqa_answer", "s8_a3", "arm_laya", BASE_METRICS, "s8_a3 better", "v0.1 better")]
    out.append(table_diffs(src, 12, "G4: PubMedQA (descriptive)", specs))
    # 13, 14 and 15. readmission
    out.append(table_metrics(src, 13, "readmit_window", READMIT_ORDER, BASE_METRICS, "Readmission (test split), part one"))
    out.append(table_metrics(src, 14, "readmit_window", READMIT_ORDER, READMIT_EXTRA, "Readmission (test split), part two"))
    rm = BASE_METRICS + READMIT_EXTRA
    specs = [("G5", "readmit_window", "readmission_pool", "lightgbm_tuned", rm, "the pool better", "lightgbm_tuned better"),
             ("G5", "readmit_window", "readmission_pool", "blend_v0_1", rm, "the pool better", "blend_v0_1 better"),
             ("G5", "readmit_window", "readmission_pool", "arm_laya", rm, "the pool better", "v0.1 better"),
             ("G5", "readmit_window", "xgboost_tuned", "lightgbm_tuned", rm, "xgboost_tuned better", "lightgbm_tuned better"),
             ("G5", "readmit_window", "catboost_tuned", "lightgbm_tuned", rm, "catboost_tuned better", "lightgbm_tuned better"),
             ("G5", "readmit_window", "s8_a3", "s7_a3", rm, "s8_a3 better", "s7_a3 better"),
             ("G5", "readmit_window", "s8_a3", "arm_laya", rm, "s8_a3 better", "v0.1 better")]
    out.append(table_diffs(src, 15, "G5: readmission (descriptive)", specs))
    # 16 and 17. the hand-off, test then validation
    for number, kind, title in ((16, "HT", "G6: the hand-off on the test split (secondary, descriptive)"),
                                (17, "HV", "The hand-off on validation (secondary, descriptive)")):
        rows = []
        h_all = src.data.get(kind, {})
        for q in (EI, "pubmedqa_answer"):
            h = h_all.get("questions", {}).get(q)
            if not h:
                continue
            for rate in HANDOFF_RATES:
                k = f"{kind}|questions/{q}/rate_{rate}"
                row = [EV_LABEL[q], f"{rate}", ("num", f"{k}/handed_off", "int"), ("iv", f"{k}/accuracy"),
                       ("iv", f"{k}/macro_f1"), ("iv", f"{k}/log_loss")]
                if rate:
                    row += [("iv", f"{k}/handed_off_fiorillo_accuracy"), ("iv", f"{k}/handed_off_large_model_accuracy")]
                else:
                    row += ["", ""]
                row.append(("num", f"{kind}|questions/{q}/cost_per_1000_questions/rate_{rate}", "usd2"))
                rows.append(row)
            rows.append([EV_LABEL[q], "Gemma 4 31B-it alone", "", ("iv", f"{kind}|questions/{q}/large_model_alone/accuracy"),
                         ("iv", f"{kind}|questions/{q}/large_model_alone/macro_f1"),
                         ("iv", f"{kind}|questions/{q}/large_model_alone/log_loss"), "", "", ""])
        out.append((number, title, ["Question", "Percent handed off", "Items handed off", "Accuracy", "Macro-F1",
                    "Log loss", "Specialist's accuracy on the handed-off items", "Gemma 4 31B-it's accuracy on them",
                    "Cost per 1,000 questions"], rows))
    # 18. sensitivity
    specs = []
    for ref, rb in (("prior", "the prior better"), ("gemma4_31b", "Gemma 4 31B-it better")):
        specs.append(("G1_sensitivity@unflagged", EI, "s8_ei", ref, EI_METRICS, "s8_ei better", rb))
    for variant in ("s8_ei-raw", "s8_ei_temperature"):
        for ref, rb in (("prior", "the prior better"), ("gemma4_31b", "Gemma 4 31B-it better")):
            specs.append(("G1_sensitivity", EI, variant, ref, EI_METRICS, f"{variant} better", rb))
    t = table_diffs(src, 18, "Sensitivity: G1 without the flagged test prompts, and with s8_ei's raw and temperature-only "
                    "predictions (descriptive)", specs)
    rows = []
    for row, spec in zip(t[3], [s for s in specs for m in s[4]]):
        rows.append([("without the flagged prompts" if "@" in spec[0] else "all prompts")] + row)
    out.append((18, t[1], ["Prompts"] + t[2], rows))
    # 19. zeros (only rows that print a zero; the prose states the total)
    rows = []
    zeros = src.data.get("Z", {}).get("rows_whose_true_answer_prints_as_0", {})
    for s, per_split in zeros.items():
        for split, per_q in per_split.items():
            for q, z in per_q.items():
                if z.get("true_answer_printed_zero"):
                    rows.append([f"`{s}`", split, f"`{q}`", ("num", f"Z|{s}|{split}|{q}|rows", "int"),
                                 ("num", f"Z|{s}|{split}|{q}|true_answer_printed_zero", "int")])
    out.append((19, "Prediction rows whose probability of the true answer prints as 0", ["System", "Split", "Question", "Rows",
                "True answer printed as 0"], rows))
    # 20 and 21. correction 14's checks
    rows = []
    for split in ("test", "validation"):
        for system in ("s8_ei", "gemma4_31b"):
            for name in SUBSETS:
                k = f"CC|reading_the_evidence/by_subset/{split}/{system}/{name}"
                rows.append([split, SYSTEM_NAME.get(system, system), name, ("num", f"{k}/prompts", "int"),
                             ("num", f"{k}/articles", "int"), ("iv", f"{k}/accuracy"), ("iv", f"{k}/macro_f1")])
    out.append((20, "Accuracy where the evidence is inside the input and where it is not (correction 14; descriptive, "
                "not pre-registered)", ["Split", "System", "Prompts", "Number", "Articles", "Accuracy (95 percent interval)",
                                        "Macro-F1 (95 percent interval)"], rows))
    rows = []
    for split, what in (("test", "s8_ei_test_matched_or_not"), ("validation", "s8_ei_validation_matched_or_not")):
        for part, label in (("flagged", "articles that share sentences with a PubMedQA abstract"), ("the rest", "the rest")):
            k = f"CC|pubmedqa/{what}/{part}"
            rows.append([split, label, ("num", f"{k}/prompts", "int"), ("num", f"{k}/articles", "int"),
                         ("iv", f"{k}/accuracy")])
    for part, label in (("flagged", "articles that share a title or sentences with an EI training article"),
                        ("the rest", "the rest")):
        k = f"CC|ei_splits/s8_ei_test_flagged_or_not/{part}"
        rows.append(["test", label, ("num", f"{k}/prompts", "int"), ("num", f"{k}/articles", "int"),
                     ("iv", f"{k}/accuracy") if (src.data.get("CC") and src.resolve(f"{k}/prompts")[1]) else "none"])
    out.append((21, "s8_ei's accuracy on articles that overlap a training text, and on the rest (correction 14; descriptive, "
                "not pre-registered)", ["Split", "Which articles", "Prompts", "Articles", "Accuracy (95 percent interval)"], rows))
    rows = []
    for split in ("test", "validation"):
        for system in ("s8_ei", "gemma4_31b"):
            k = f"CC|reading_the_evidence/differences/{split}/{system}/outside minus inside"
            rows.append([split, SYSTEM_NAME.get(system, system), "where the evidence is outside the input, minus inside",
                         ("iv", k), ("read", k, "accuracy", "higher outside the input", "lower outside the input")])
    for split, what in (("test", "s8_ei_test_matched_or_not"), ("validation", "s8_ei_validation_matched_or_not")):
        k = f"CC|pubmedqa/{what}/flagged minus the rest"
        rows.append([split, "s8_ei", "articles that share sentences with a PubMedQA abstract, minus the rest",
                     ("iv", k), ("read", k, "accuracy", "higher on the shared articles", "lower on the shared articles")])
    out.append((22, "Differences in accuracy between the groups (correction 15; descriptive, not pre-registered)",
                ["Split", "System", "Difference in accuracy", "Estimate (95 percent interval)", "Reading"], rows))
    return out


def interval_text(src, base):
    parts = [src.resolve(f"{base}|{p}")[1] for p in ("estimate", "low", "high")]
    if parts[0] is None:
        return "not computed"
    if parts[1] is None:
        return f4(parts[0])
    return f"{f4(parts[0])} ({f4(parts[1])} to {f4(parts[2])})"


def range_text(src, base):
    parts = [src.resolve(f"{base}|{p}")[1] for p in ("low", "high")]
    if parts[0] is None:
        return "not computed"
    return f"{f4(parts[0])} to {f4(parts[1])}"


def cell_text(src, c):
    if isinstance(c, str):
        return c
    kind = c[0]
    if kind == "num":
        return fmt(src.resolve(c[1])[1], c[2])
    if kind == "iv":
        return interval_text(src, c[1])
    if kind == "rng":
        return range_text(src, c[1])
    if kind == "read":
        d = {p: src.resolve(f"{c[1]}|{p}")[1] for p in ("estimate", "low", "high")}
        return reading(c[2], d, c[3], c[4])
    if kind == "str":
        return str(src.resolve(c[1])[1])
    if kind == "pf":
        return "PASS" if src.resolve(c[1])[1] else "FAIL"
    raise ValueError(c)


def render_table(src, t):
    number, title, header, rows = t
    lines = [f"### Table {number}. {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(cell_text(src, c) for c in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- the checked sections, in document order
# Items: ("h", heading), ("x", a line without figures), ("p", id, template), ("t", table number), ("v", verbatim label).
# A template marks each figure as {{key:how}}.


def iv(base):
    return f"{{{{{base}|estimate:f4}}}} ({{{{{base}|low:f4}}}} to {{{{{base}|high:f4}}}})"


def L(pattern, marker=STEP13):
    return f"LOGMD|{marker}|{pattern}"


# docs/log.md keys, built here so that no template holds a backslash inside a placeholder
K_DISK = L(r"a (\d+) GB container disk", STEP9)
K_START = L(r"its container started at (\S+) UTC", STEP9)
K_STOP = L(r"stopped at about (\S+) UTC")
K_TERM = L(r"terminated at (\S+) UTC")
K_HOURS = L(r"(\d+) hours \d+ minutes, about")
K_MINUTES = L(r"\d+ hours (\d+) minutes, about")
K_RATE_COST = L(r"about \$(\S+) at that rate")
K_BEFORE = L(r"the balance was \$(\S+) before the pod")
K_AFTER = L(r"and \$(\S+) after the shutdown")
K_SPEND = L(r"spend is about \$(\S+), under")
K_CAP = L(r"under the \$(\d+) cap")
K_IDLE = L(r"idle from \S+ to \S+ UTC, (\d+) minutes")
K_IDLE_COST = L(r"minutes \(about \$(\S+)\)")
K_PROJ = L(r"projected total \$(\S+), under", STEP10)
K_LINE = L(r"under the \$(\d+) line", STEP10)
K_WAIT = L(r"was about \$(\S+)", OSF)
K_APPLY = L(r"apply on the test split at (\S+) UTC", STEP15)
K_CMP_END = L(r"compare_session8.py from \S+ to (\S+) UTC", STEP15)
K_ADDED = L(r"\(\$(\d+) added to \$", STEP9)
K_LEFT = L(r"added to \$(\S+), less", STEP9)


def P(key, how):
    return "{{" + key + ":" + how + "}}"


def regions(src):
    pool = "POOL"
    xg, cb = "TR|members/xgboost_tuned", "TR|members/catboost_tuned"
    ht = "HT|questions"
    cc = "CC|reading_the_evidence/by_subset"
    pc = "CC|pubmedqa"
    ins, out_ = "evidence inside the input", "evidence outside the input"
    return {
        SECTIONS[0]: [
            ("h", SECTIONS[0]),
            ("p", "pod",
             "Everything on the GPU ran on one Runpod pod (`0k22u0lbzm8r9t`: one H100 SXM GPU, Secure Cloud `CA-MTL-1`, a "
             f"{P(K_DISK, 'int')} GB container disk, {P('HT|hourly_rate', 'usd2')} an hour), driven by "
             f"scripts/pod_session8.sh from the linked shell. Its container started at {P(K_START, 'time')} UTC on 30 "
             f"September; it was stopped at about {P(K_STOP, 'time')} UTC on 1 October and terminated at "
             f"{P(K_TERM, 'time')} UTC, after {P(K_HOURS, 'int')} hours {P(K_MINUTES, 'int')} minutes, about "
             f"{P(K_RATE_COST, 'usd2')} at that rate (derived). The Runpod balance was {P(K_BEFORE, 'usd2')} before the pod "
             f"(derived: {P(K_LEFT, 'usd2')} left by session 7 and {P(K_ADDED, 'usd0')} added) and {P(K_AFTER, 'usd2')} "
             f"after it, so the session's Runpod spend was about {P(K_SPEND, 'usd2')} (derived), "
             f"under the {P(K_CAP, 'usd0')} cap. Times below are UTC, and every stage's log is in "
             "results/fiorillo_v0_4/logs/pod/."),
            ("t", 1),
            ("p", "records",
             "EI's records (step 2, built on the Mac and rebuilt on the pod from the selection file): "
             f"{P('MAN|drops/PMCID in no split file', 'int')} prompts whose PMCID is in no split file were dropped, with "
             f"{P('MAN|drops/train: no accepted row', 'int')} training, {P('MAN|drops/validation: no accepted row', 'int')} "
             f"validation and {P('MAN|drops/test: no accepted row', 'int')} test prompts without an accepted row, the "
             f"{P('MAN|drops/test: empty article text', 'int')} test prompts of the empty article, and the "
             f"{P('MAN|drops/train: listed as Incorrect in annotations/README.md', 'int')} training prompts listed as "
             f"Incorrect. Kept (records_manifest.json): training {P('MAN|counts/train/prompts', 'int')} prompts in "
             f"{P('MAN|counts/train/articles', 'int')} articles (significantly increased {P('MAN|counts/train/answers/0', 'int')}, "
             f"significantly decreased {P('MAN|counts/train/answers/1', 'int')}, no significant difference "
             f"{P('MAN|counts/train/answers/2', 'int')}); validation {P('MAN|counts/validation/prompts', 'int')} in "
             f"{P('MAN|counts/validation/articles', 'int')}; test {P('MAN|counts/test/prompts', 'int')} in "
             f"{P('MAN|counts/test/articles', 'int')}. The accepted labels of {P('MAN|accepted_labels_disagree/train', 'int')} "
             f"training, {P('MAN|accepted_labels_disagree/validation', 'int')} validation and "
             f"{P('MAN|accepted_labels_disagree/test', 'int')} test prompts disagree."),
            ("p", "tokens",
             f"Tokens per record: mean {P('MAN|tokens_per_record/train/mean', 'c0')} (training), "
             f"{P('MAN|tokens_per_record/validation/mean', 'c0')} (validation) and {P('MAN|tokens_per_record/test/mean', 'c0')} "
             f"(test); the largest is {P('MAN|tokens_per_record/train/largest', 'int')}, "
             f"{P('MAN|tokens_per_record/validation/largest', 'int')} and {P('MAN|tokens_per_record/test/largest', 'int')}, "
             f"under the limit of {P('ASEL|max_len', 'int')}, so no record was cut."),
            ("p", "coverage",
             f"An accepted evidence span lies wholly inside the kept text for {P('ASEL|coverage/train/covered', 'int')} of the "
             f"{P('ASEL|coverage/train/prompts_with_usable_offsets', 'int')} training prompts with usable offsets "
             f"({P('ASEL|coverage/train/coverage', 'pct2')} percent; the plan's check found "
             f"{P('ASEL|check_of_30_september_train', 'pct2')}) and for {P('ASEL|coverage/validation/covered', 'int')} of "
             f"{P('ASEL|coverage/validation/prompts_with_usable_offsets', 'int')} validation prompts "
             f"({P('ASEL|coverage/validation/coverage', 'pct2')} percent) (article_selection.json)."),
            ("p", "licenses",
             f"Europe PMC reported the licenses of {P('MAN|article_licenses/articles', 'int')} articles: CC BY "
             f"{P('MAN|article_licenses/tally/cc by', 'int')}, CC BY-NC {P('MAN|article_licenses/tally/cc by-nc', 'int')}, "
             f"CC BY-NC-SA {P('MAN|article_licenses/tally/cc by-nc-sa', 'int')}, CC BY-NC-ND "
             f"{P('MAN|article_licenses/tally/cc by-nc-nd', 'int')}, none recorded "
             f"{P('MAN|article_licenses/tally/none recorded', 'int')}, CC0 {P('MAN|article_licenses/tally/cc0', 'int')}, "
             f"CC BY-ND {P('MAN|article_licenses/tally/cc by-nd', 'int')} and CC BY-SA "
             f"{P('MAN|article_licenses/tally/cc by-sa', 'int')} (records_manifest.json)."),
            ("p", "pilot",
             f"The pilot ran alone on the GPU: setting a (micro-batch {P('PIL|settings/0/micro_batch', 'int')}, accumulation "
             f"{P('PIL|settings/0/grad_accum', 'int')}, no gradient checkpointing) trained "
             f"{P('PIL|settings/0/train_items_per_second', 'f4')} items a second at a peak of "
             f"{P('PIL|settings/0/peak_gpu_memory_gib', 'f2')} GiB, and setting b (micro-batch "
             f"{P('PIL|settings/1/micro_batch', 'int')}, accumulation {P('PIL|settings/1/grad_accum', 'int')}, gradient "
             f"checkpointing) {P('PIL|settings/1/train_items_per_second', 'f4')} items a second at "
             f"{P('PIL|settings/1/peak_gpu_memory_gib', 'f2')} GiB; both passed, and the seeds used setting a. The forward "
             f"pass over the {P('PIL|validation_items', 'int')} monitoring prompts ran at "
             f"{P('PIL|validation_items_per_second', 'f4')} items a second (pilot.json). The cost rule's projection was "
             f"{P(K_PROJ, 'usd2')} (derived), under the {P(K_LINE, 'usd0')} line."),
            ("p", "seeds",
             f"`s8_ei_1` trained beside arm A until {P('PLOG|arm_a|last|stage arm_a finished', 'time')} and then beside "
             f"the first three smoke tests, which started at {P('PLOG|smoke_large|first|stage smoke_large started', 'time')}, "
             f"until {P('PLOG|ei|first|EI: s8_ei_1 trained', 'time')}; `s8_ei_2` trained beside the fourth smoke test from "
             f"{P('PLOG|smoke_large|last|stage smoke_large started', 'time')} to "
             f"{P('PLOG|smoke_large|last|stage smoke_large finished', 'time')} and otherwise alone, until "
             f"{P('PLOG|ei|first|EI: s8_ei_2 trained', 'time')}; `s8_ei_3` trained alone, until "
             f"{P('PLOG|ei|first|EI: s8_ei_3 trained', 'time')} (Table 2). Arm A's runs kept their last weights: "
             f"`s8_a1` {P('MET|s8_a1|optimizer_steps', 'int')} optimizer steps, `s8_a2` {P('MET|s8_a2|optimizer_steps', 'int')} "
             f"and `s8_a3` {P('MET|s8_a3|optimizer_steps', 'int')} ({P('MET|s8_a3|sequences_per_second', 'f2')} items a second)."),
            ("t", 2),
            ("p", "trees",
             f"The trees, on the pod's processors (trees.json): `xgboost_tuned` is the best of "
             f"{P(xg + '/settings_drawn/drawn', 'int')} settings drawn from {P(xg + '/settings_drawn/from', 'int')} by "
             f"{P('TR|folds', 'int')}-fold cross-validated log loss on the {P(xg + '/refit/rows', 'int')} training records "
             f"(maximum depth {P(xg + '/chosen/max_depth', 'int')}, minimum child weight {P(xg + '/chosen/min_child_weight', 'int')}, "
             f"learning rate {P(xg + '/chosen/eta', 'f2')}, row and column samples {P(xg + '/chosen/subsample', 'f1')} and "
             f"{P(xg + '/chosen/colsample_bytree', 'f1')}, L2 penalty {P(xg + '/chosen/lambda', 'int')}), refitted with "
             f"{P(xg + '/refit/trees', 'int')} trees; `catboost_tuned` is the best of {P(cb + '/settings_drawn/drawn', 'int')} "
             f"drawn from {P(cb + '/settings_drawn/from', 'int')} (depth {P(cb + '/chosen/depth', 'int')}, learning rate "
             f"{P(cb + '/chosen/learning_rate', 'f2')}, L2 penalty {P(cb + '/chosen/l2_leaf_reg', 'int')}, bagging temperature "
             f"{P(cb + '/chosen/bagging_temperature', 'int')}, border count {P(cb + '/chosen/border_count', 'int')}), refitted "
             f"with {P(cb + '/refit/trees', 'int')} trees."),
            ("p", "pool",
             f"The readmission pool's weights, fitted on the {P(pool + '|rows', 'int')} validation records on a grid of tenths: "
             f"`xgboost_tuned` {P(pool + '|weights/xgboost_tuned', 'f1')}, `catboost_tuned` {P(pool + '|weights/catboost_tuned', 'f1')}, "
             f"`lightgbm_tuned` {P(pool + '|weights/lightgbm_tuned', 'f1')}, v0.1 {P(pool + '|weights/arm_laya', 'f1')} and "
             f"`s7_b3` {P(pool + '|weights/s7_b3', 'f1')}, with temperature {P(pool + '|T', 'f4')}; its validation log loss "
             f"is {P(pool + '|validation_log_loss', 'f5')}, and out of fold {P(pool + '|out_of_fold_log_loss/mean', 'f5')}, "
             f"which is optimistic because two members' calibrations were fitted on the same records (readmission_pool.json)."),
            ("p", "large",
             f"Gemma 4 31B-it read EI's {P('TIM|evidence_effect/validation/items', 'int')} validation prompts (mean "
             f"{P('TIM|evidence_effect/validation/prompt_tokens_per_item', 'c0')} prompt tokens) at "
             f"{P('TIM|evidence_effect/validation/gpu_seconds_per_item', 'f3')} GPU seconds an item, and its "
             f"{P('TIM|evidence_effect/test/items', 'int')} test prompts at {P('TIM|evidence_effect/test/gpu_seconds_per_item', 'f3')}; "
             f"PubMedQA's {P('TIM|pubmedqa_answer/validation/items', 'int')} calibrate items at "
             f"{P('TIM|pubmedqa_answer/validation/gpu_seconds_per_item', 'f3')} and {P('TIM|pubmedqa_answer/test/items', 'int')} "
             f"test items at {P('TIM|pubmedqa_answer/test/gpu_seconds_per_item', 'f3')} (gemma4_31b_timing.json; Table 4)."),
            ("p", "freeze",
             f"The ECE limit for EI is {P('ECEL|evidence_effect/limit', 'f2')}: the 95th percentile of a calibrated model's "
             f"ECE at {P('LIM|n', 'int')} prompts, from {P('LIM|draws', 'int')} draws, was {P('LIM|ece_if_calibrated/p95', 'f4')}, "
             f"below the floor of {P('ECEL|evidence_effect/floor', 'f2')} (ece_limit_evidence_effect.json)."),
            ("x", "So the floor set the limit, not the percentile. The noise floor draws prompts one by one, not by article, "
                  "which would make a limit set by the percentile a little strict."),
            ("p", "wait",
             f"testpreds waited for the OSF registration from the freeze to its start (correction 13), about "
             f"{P(K_WAIT, 'usd2')} of the pod's time (derived; docs/log.md)."),
            ("p", "testpreds",
             f"In testpreds the three EI runs took {P('RS|s8_ei_1_test|timing_test/forward_seconds', 'f1')}, "
             f"{P('RS|s8_ei_2_test|timing_test/forward_seconds', 'f1')} and {P('RS|s8_ei_3_test|timing_test/forward_seconds', 'f1')} "
             f"forward seconds for their {P('RS|s8_ei_1_test|timing_test/items', 'int')} prompts each, and `s7_b3_timing` "
             f"{P('RS|s7_b3_timing_test|timing_test/forward_seconds', 'f1')} seconds for PubMedQA's "
             f"{P('RS|s7_b3_timing_test|timing_test/items', 'int')} items; the two tree models reproduced their validation "
             f"predictions to within one billionth before predicting the test split (testpreds_log.json)."),
            ("p", "package",
             f"package wrote the {P('SUMS|count|size', 'int')} zips, {P('SUMS|total|size', 'int')} bytes in all "
             f"(SHA256SUMS.txt). The pod sat idle from the end of testpreds to the start of package, "
             f"{P(K_IDLE, 'int')} minutes (about {P(K_IDLE_COST, 'usd2')}, derived), while Claude ran background checks; Claude "
             f"should have kept those checks shorter or started them after the copy."),
            ("x", "Every zip's size and SHA-256 matched SHA256SUMS.txt, unzip -t found no errors in any of them, and "
                  "models/fiorillo_v0_4/check_package.py found nothing missing; the pod was then stopped and terminated."),
            ("x", "When testpreds loaded each EI run's tokenizer, transformers warned of \"an incorrect regex pattern\", a "
                  "check written for Mistral's tokenizers. The three runs' tokenizer files are byte-identical to s7_b3's, "
                  "which session 7 found identical to Qwen3-4B-Base's (docs/log.md, steps 12 and 14)."),
            ("x", "The logs zip was made while package ran, so results/fiorillo_v0_4/logs/pod/package.log holds only that "
                  "stage's first lines; its output, the zips' sizes and SHA-256, is in docs/log.md (step 13)."),
            ("p", "analyses",
             f"Step 14 (the unpack) and step 15 (the analyses, from {P(K_APPLY, 'time')} to {P(K_CMP_END, 'time')} UTC) ran "
             f"on the linked shell."),
        ],
        SECTIONS[1]: [
            ("h", SECTIONS[1]),
            ("p", "intro",
             f"EI's test split was read once, after the freeze: release_candidate.json was written at "
             f"{P('PLOG|freeze|last|stage freeze finished', 'time')} UTC on 30 September, and testpreds started at "
             f"{P('TPL|testpreds_start_utc', 'time')} UTC on 1 October and wrote that file's SHA-256 first. EI's test split "
             f"has {P('D|evidence_effect_test_items', 'int')} prompts in {P('MAN|counts/test/articles', 'int')} articles "
             f"(significantly increased {P('D|evidence_effect_test_label_0', 'int')}, significantly decreased "
             f"{P('D|evidence_effect_test_label_1', 'int')}, no significant difference {P('D|evidence_effect_test_label_2', 'int')}); "
             f"PubMedQA's has {P('D|pubmedqa_answer_test_items', 'int')} items and readmission's "
             f"{P('D|readmit_window_test_items', 'int')} encounters."),
            ("x", "PubMedQA's and readmission's test splits were read in sessions 4 to 7, and s7_b3 and the readmission "
                  "pool's members were chosen for v0.4 after those results, so Tables 11 to 15 are descriptive and favor the "
                  "chosen specialists."),
            ("p", "intervals",
             f"Every interval is a 95 percent percentile interval from {P('SET|resamples', 'int')} paired bootstrap resamples "
             f"(seed 20260927): EI's resample articles, PubMedQA's and readmission's resample items. Criteria 2 and 3 are "
             f"decided by {P('SET|resamples_bar', 'int')} article resamples (Table 8); the intervals of the same differences in "
             f"Table 7 are descriptive. A system's name alone is its chosen calibration; _temperature marks the temperature "
             f"family alone, and -raw the predictions before calibration."),
            ("h", "### Calibration and the large model on validation"),
            ("p", "cal",
             f"s8_ei and each of its three runs chose the temperature family on validation (s8_ei's temperature is "
             f"{P('CAL|s8_ei|evidence_effect|families/temperature/temperature', 'f3')}); Gemma 4 31B-it chose the temperature "
             f"with intercepts on both questions, and its PubMedQA temperature, "
             f"{P('CAL|gemma4_31b|pubmedqa_answer|families/temperature_intercepts/temperature', 'f3')}, is the family's upper "
             f"bound (Table 3). On EI's validation prompts Gemma 4 31B-it's log loss was "
             f"{P('CAL|gemma4_31b|evidence_effect|validation_calibrate_items/raw_log_loss', 'f4')} before calibration and "
             f"{P('CAL|gemma4_31b|evidence_effect|validation_calibrate_items/log_loss', 'f4')} after, in sample for the "
             f"calibration (out of fold "
             f"{P('CAL|gemma4_31b|evidence_effect|families/temperature_intercepts/out_of_fold_log_loss/mean', 'f4')})."),
            ("t", 3),
            ("x", "Gemma 4 31B-it's three letters held nearly all of its next-token probability on every split (Table 4), so "
                  "its answers were read as the plan fixed; the last columns count how each prompt's letters were found."),
            ("t", 4),
            ("h", "### Evidence Inference: every system"),
            ("p", "ei",
             f"s8_ei's macro-F1 is {iv('M|evidence_effect|s8_ei|macro_f1')} and its log loss "
             f"{iv('M|evidence_effect|s8_ei|log_loss')}; Gemma 4 31B-it's are {iv('M|evidence_effect|gemma4_31b|macro_f1')} "
             f"and {iv('M|evidence_effect|gemma4_31b|log_loss')} (Table 5)."),
            ("p", "ece_note",
             f"The lower end of s8_ei's ECE interval lies a little above its estimate (both are "
             f"{P('M|evidence_effect|s8_ei|ece|estimate', 'f4')} at four decimals): ECE computed on a resample is biased "
             f"upward, so its percentile interval can sit above the estimate. Criterion 1 reads the estimate, as fixed."),
            ("t", 5),
            ("t", 6),
            ("h", "### G1 and the release bar"),
            ("p", "bar",
             f"Every criterion passes (Table 8): s8_ei's ECE is {P('RB|criteria/1/values/ece', 'f4')} against the limit "
             f"{P('RB|criteria/1/values/limit', 'f2')}; its log loss minus the prior's is {P('BI|prior|estimate', 'f4')} "
             f"({P('BI|prior|low', 'f4')} to {P('BI|prior|high', 'f4')}) and minus Gemma 4 31B-it's "
             f"{P('BI|gemma4_31b|estimate', 'f4')} ({P('BI|gemma4_31b|low', 'f4')} to {P('BI|gemma4_31b|high', 'f4')}), each "
             f"over {P('BI|gemma4_31b|articles', 'int')} articles; and its macro-F1, {P('RB|criteria/4/values/s8_ei', 'f4')}, is "
             f"not below Gemma 4 31B-it's, {P('RB|criteria/4/values/gemma4_31b', 'f4')}."),
            ("v", "bar_outcome"),
            ("t", 7),
            ("t", 8),
            ("h", "### G2: the three runs"),
            ("p", "g2",
             f"Each run calibrated by its own rule had a test log loss of {P('G2|log_loss/s8_ei_1', 'f4')}, "
             f"{P('G2|log_loss/s8_ei_2', 'f4')} and {P('G2|log_loss/s8_ei_3', 'f4')}: a spread, largest minus smallest, of "
             f"{P('G2|max_minus_min', 'f4')} and a standard deviation of {P('G2|standard_deviation', 'f4')}, the run-to-run "
             f"noise of this recipe from data order and dropout (descriptive). s8_ei minus each run: Table 9."),
            ("v", "runs"),
            ("t", 9),
            ("h", "### G3: published results"),
            ("p", "g3",
             f"s8_ei's macro-F1, {iv('G3|macro_f1')}, is above each published value G3 names, and its interval excludes each "
             f"(Table 10; descriptive, since the published systems were scored and counted differently). Scored leniently, as Buchmann et al. scored (a prompt's gold label is replaced by the prediction "
             f"when the prediction is one of its accepted labels), it is {P('G3|lenient_macro_f1/estimate', 'f4')}: the rule "
             f"changed the gold label of {P('G3|lenient_changed_prompts', 'int')} prompt, since only "
             f"{P('MAN|accepted_labels_disagree/test', 'int')} test prompts have accepted labels that disagree."),
            ("v", "g3_note"),
            ("x", "What the published papers say about these figures, checked at their sources, is in the section after "
                  "this one."),
            ("t", 10),
            ("h", "### PubMedQA"),
            ("p", "pqa",
             f"v0.4 answers PubMedQA with s7_b3 and its session 7 calibration: log loss "
             f"{iv('M|pubmedqa_answer|s7_b3|log_loss')} and accuracy {iv('M|pubmedqa_answer|s7_b3|accuracy')} (Table 11)."),
            ("p", "load",
             f"The load check: s7_b3, loaded again on the pod as `s7_b3_timing`, gave raw test probabilities whose largest "
             f"difference from session 7's file is {P('G4L|max_abs_difference_vs_session7', 'f4')} in one probability, "
             f"more than the code's note expects from padding alone. Its raw test accuracy, log loss and ECE, "
             f"{P('RS|s7_b3_timing_test|test_per_question_raw/pubmedqa_answer/accuracy', 'f4')}, "
             f"{P('RS|s7_b3_timing_test|test_per_question_raw/pubmedqa_answer/log_loss', 'f4')} and "
             f"{P('RS|s7_b3_timing_test|test_per_question_raw/pubmedqa_answer/ece', 'f4')}, are session 7's `s7_b3-raw` "
             f"figures in Table 11, which is consistent with the model having loaded as it was."),
            ("t", 11),
            ("t", 12),
            ("h", "### Readmission"),
            ("p", "readmit",
             f"The pool's log loss is {iv('M|readmit_window|readmission_pool|log_loss')}, against "
             f"`lightgbm_tuned`'s {P('M|readmit_window|lightgbm_tuned|log_loss|estimate', 'f4')} and `blend_v0_1`'s "
             f"{P('M|readmit_window|blend_v0_1|log_loss|estimate', 'f4')} (Tables 13 to 15; descriptive). Macro-F1 with "
             f"decision weights multiplies each answer's probability by a weight fitted on validation to maximize macro-F1 "
             f"(session 4's rule)."),
            ("t", 13),
            ("t", 14),
            ("t", 15),
            ("h", "### G6: the hand-off"),
            ("p", "handoff_ei",
             f"On EI the combined answers' accuracy was {P(ht + '/evidence_effect/rate_0/accuracy/estimate', 'f4')} with no "
             f"hand-off and {P(ht + '/evidence_effect/rate_10/accuracy/estimate', 'f4')}, "
             f"{P(ht + '/evidence_effect/rate_20/accuracy/estimate', 'f4')}, {P(ht + '/evidence_effect/rate_30/accuracy/estimate', 'f4')} "
             f"and {P(ht + '/evidence_effect/rate_50/accuracy/estimate', 'f4')} with {P('HT|rates/1', 'pct0')}, "
             f"{P('HT|rates/2', 'pct0')}, {P('HT|rates/3', 'pct0')} and {P('HT|rates/4', 'pct0')} percent handed off; on the "
             f"handed-off items s8_ei's accuracy was {P(ht + '/evidence_effect/rate_10/handed_off_fiorillo_accuracy/estimate', 'f4')} "
             f"and Gemma 4 31B-it's {P(ht + '/evidence_effect/rate_10/handed_off_large_model_accuracy/estimate', 'f4')} at "
             f"{P('HT|rates/1', 'pct0')} percent (Table 16; descriptive, and the rates were not compared with each other)."),
            ("p", "handoff_pqa",
             f"On PubMedQA it was {P(ht + '/pubmedqa_answer/rate_0/accuracy/estimate', 'f4')} with no hand-off and "
             f"{P(ht + '/pubmedqa_answer/rate_50/accuracy/estimate', 'f4')} with {P('HT|rates/4', 'pct0')} percent handed off."),
            ("p", "cost",
             f"At {P('HT|hourly_rate', 'usd2')} an hour, a thousand EI questions cost "
             f"{P(ht + '/evidence_effect/cost_per_1000_questions/rate_0', 'usd2')} of GPU time with no hand-off and "
             f"{P(ht + '/evidence_effect/cost_per_1000_questions/rate_50', 'usd2')} with {P('HT|rates/4', 'pct0')} percent "
             f"handed off (derived), from the GPU seconds an item measured in testpreds, each job alone: s8_ei "
             f"{P(ht + '/evidence_effect/seconds_per_item/fiorillo', 'f4')} (its three runs' sum) and Gemma 4 31B-it "
             f"{P(ht + '/evidence_effect/seconds_per_item/large_model', 'f4')}. The specialist is counted on every question and "
             f"the large model on the handed-off share. A thousand PubMedQA questions cost "
             f"{P(ht + '/pubmedqa_answer/cost_per_1000_questions/rate_0', 'usd2')} and "
             f"{P(ht + '/pubmedqa_answer/cost_per_1000_questions/rate_50', 'usd2')} (`s7_b3_timing` "
             f"{P(ht + '/pubmedqa_answer/seconds_per_item/fiorillo', 'f4')} and Gemma 4 31B-it "
             f"{P(ht + '/pubmedqa_answer/seconds_per_item/large_model', 'f4')} seconds an item)."),
            ("x", "The large model's seconds an item come from its second process's time shared between the two questions "
                  "by item count, as the code does; the Method gives what a share by prompt tokens would make them (derived)."),
            ("x", "handoff_test.json is the file comparison.json names; the figure check compares their SHA-256."),
            ("t", 16),
            ("x", "On validation the hand-off ran on every EI validation prompt and on PubMedQA's calibrate items (Table 17). "
                  "Its predictions are calibrated on the items it scores, so its figures are in sample. PubMedQA's validation "
                  "cost is not computed, since no session 8 run of s7_b3 read the validation split (correction 4)."),
            ("p", "hv_note",
             f"handoff_validation.json's note allows that the seeds' validation passes shared the GPU with arm A; by the stage "
             f"logs none did, since arm A had finished at {P('PLOG|arm_a|last|stage arm_a finished', 'time')}. Its EI costs "
             f"differ from the test's mainly through the large model's seconds an item."),
            ("t", 17),
            ("h", "### Sensitivity, power and printed zeros"),
            ("p", "sens",
             f"Without the {P('SENS|flagged_test_prompts', 'int')} test prompts that EI's README flags, and with s8_ei's raw or "
             f"temperature-only predictions, the G1 differences are in Table 18 (descriptive)."),
            ("t", 18),
            ("p", "power",
             f"Power, derived before any prediction (power.json): at a per-prompt standard deviation of "
             f"{P('POW|cases/4/s', 'f1')} and a design effect of {P('POW|cases/4/D', 'f1')}, criterion 3's difference had a "
             f"standard error of {P('POW|cases/4/se', 'f4')}, a pass line of {P('POW|cases/4/pass_line', 'f4')}, and an 80 percent "
             f"chance of passing at a real gain of {P('POW|cases/4/real_gain_for_80_percent', 'f4')}. Measured on validation, for "
             f"the record only (power_validation.json): s {P('POWV|s', 'f4')}, design effect {P('POWV|D', 'f4')}, and s8_ei minus "
             f"Gemma 4 31B-it {P('POWV|validation_mean_difference', 'f4')} in log loss."),
            ("p", "zeros",
             f"No prediction row of any system, split or question has a true answer whose probability prints as 0: "
             f"{P('D|zero_printed', 'int')} of {P('D|zero_rows', 'int')} rows (zeros.json), and "
             f"{P('D|missing_predictions', 'int')} prediction files were missing."),
            ("t", 19),
            ("h", "### Known contamination"),
            ("v", "contamination"),
            ("x", "The checks after the Results (correction 14) measure part of this; they cannot reach the pretraining text."),
        ],
        SECTIONS[2]: [
            ("h", SECTIONS[2]),
            ("x", "Corrections 14 and 15 added these checks after the results, because s8_ei's macro-F1 is above each "
                  "published EI result found (descriptive). They were not registered, they have no pass or fail, and they "
                  "change no figure above."),
            ("p", "cc_recount",
             f"src/eval/contamination_checks_session8.py wrote them to results/fiorillo_v0_4/contamination_checks.json, "
             f"first recounting the builder's validation coverage, which matched article_selection.json "
             f"({P('CC|reading_the_evidence/coverage_recount_validation/covered', 'int')} of "
             f"{P('CC|reading_the_evidence/coverage_recount_validation/prompts_with_usable_offsets', 'int')})."),
            ("p", "cc_reading",
             f"Reading the evidence. On the test split the selection kept an accepted evidence span inside the input for "
             f"{P('CC|reading_the_evidence/coverage_test/covered', 'int')} of the "
             f"{P('CC|reading_the_evidence/coverage_test/prompts_with_usable_offsets', 'int')} prompts with usable offsets "
             f"({P('CC|reading_the_evidence/coverage_test/coverage', 'pct2')} percent). s8_ei's accuracy is "
             f"{iv(cc + '/test/s8_ei/' + ins + '/accuracy')} where an annotated evidence span lies inside the input and "
             f"{iv(cc + '/test/s8_ei/' + out_ + '/accuracy')} on the "
             f"{P(cc + '/test/s8_ei/' + out_ + '/prompts', 'int')} prompts in "
             f"{P(cc + '/test/s8_ei/' + out_ + '/articles', 'int')} articles where none does, though the result may still be "
             f"stated elsewhere in the input; Gemma 4 31B-it's "
             f"is {P(cc + '/test/gemma4_31b/' + ins + '/accuracy/estimate', 'f4')} and "
             f"{P(cc + '/test/gemma4_31b/' + out_ + '/accuracy/estimate', 'f4')}. On validation s8_ei's is "
             f"{P(cc + '/validation/s8_ei/' + ins + '/accuracy/estimate', 'f4')} and "
             f"{P(cc + '/validation/s8_ei/' + out_ + '/accuracy/estimate', 'f4')} "
             f"({P(cc + '/validation/s8_ei/' + out_ + '/prompts', 'int')} prompts outside)."),
            ("p", "cc_diff",
             f"Outside minus inside (correction 15), s8_ei's accuracy differs by "
             f"{iv('CC|reading_the_evidence/differences/test/s8_ei/outside minus inside')} on the test split and "
             f"{iv('CC|reading_the_evidence/differences/validation/s8_ei/outside minus inside')} on validation, and Gemma 4 "
             f"31B-it's by {iv('CC|reading_the_evidence/differences/test/gemma4_31b/outside minus inside')} and "
             f"{iv('CC|reading_the_evidence/differences/validation/gemma4_31b/outside minus inside')} (Table 22)."),
            ("x", "Each interval lies below zero: both models are less accurate where no annotated evidence span lies inside "
                  "the input. This is consistent with their reading the evidence; it is also consistent with those prompts "
                  "being harder for any reader, and it does not rule out that a model recalls some answers. The groups are "
                  "small, and their intervals are wide."),
            ("t", 20),
            ("p", "cc_pubmedqa",
             f"PubMedQA. {P('D|cc_matched_test', 'int')} test articles and {P('D|cc_matched_validation', 'int')} validation "
             f"articles share at least two normalized sentences of {P('CC|settings/sentence_min_chars', 'int')} characters or "
             f"more with one PubMedQA abstract (the matched abstracts: {P('D|cc_matches_pqa_artificial', 'int')} in PQA-A, "
             f"{P('D|cc_matches_pqa_unlabeled', 'int')} in PQA-U and {P('D|cc_matches_pqa_labeled', 'int')} in PQA-L), and arm B "
             f"read every one of those abstracts in training ({P('D|cc_matched_trained_test', 'int')} and "
             f"{P('D|cc_matched_trained_validation', 'int')} articles). s8_ei's test accuracy is "
             f"{iv(pc + '/s8_ei_test_matched_or_not/flagged/accuracy')} on their "
             f"{P(pc + '/s8_ei_test_matched_or_not/flagged/prompts', 'int')} prompts and "
             f"{P(pc + '/s8_ei_test_matched_or_not/the rest/accuracy/estimate', 'f4')} on the other "
             f"{P(pc + '/s8_ei_test_matched_or_not/the rest/prompts', 'int')} (Table 21); the difference, "
             f"{iv(pc + '/s8_ei_test_matched_or_not/flagged minus the rest')}, includes zero, so no difference was "
             f"detected, and the interval excludes an accuracy on the shared articles more than "
             f"{P(pc + '/s8_ei_test_matched_or_not/flagged minus the rest/high', 'f4')} above the rest's (Table 22)."),
            ("p", "cc_splits",
             f"EI's own splits. {P('D|cc_dups_test', 'int')} test and {P('D|cc_dups_validation', 'int')} validation articles "
             f"share a title, or two or more such sentences, with one of EI's {P('CC|ei_splits/train_articles', 'int')} "
             f"training articles."),
            ("t", 21),
            ("t", 22),
            ("x", "These checks cannot show whether Qwen3-4B-Base's or Gemma 4's pretraining text held EI's annotations. "
                  "EI's labels have been public since 2019, so that stays possible, and it would favor both models."),
        ],
    }


def verbatim(src):
    """Lines checked whole: (label, source description, the text the line must be)."""
    c = src.cmp
    return [
        ("bar_outcome", "comparison.json release_bar.outcome",
         f"The bar's outcome, as compare_session8.py wrote it: \"{c['release_bar']['outcome']}\"."),
        ("runs", "comparison.json runs", c["runs"]),
        ("g3_note", "comparison.json g3.note", c["g3"]["note"]),
        ("contamination", "comparison.json contamination", c["contamination"]),
    ]


# the self-test's planted errors: (text in the document, text put in its place), one at a time
PLANTED = [
    ("| `s8_ei-raw` | 0.2238 (0.1651 to 0.2915)", "| `s8_ei-raw` | 0.2283 (0.1651 to 0.2915)"),  # a table cell
    ("| `s8_ei` minus `prior` | ECE | 0.0119 | 0.0305 | -0.0186 (-0.0461 to 0.0187) | no difference was detected; excludes "
     "below -0.0461 and above 0.0187 |",
     "| `s8_ei` minus `prior` | ECE | 0.0119 | 0.0305 | -0.0186 (-0.0461 to 0.0187) | s8_ei better |"),  # a reading
    ("| -0.2084 | -0.2720 to -0.1505 | PASS |", "| -0.2084 | -0.2720 to -0.1505 | FAIL |"),  # a criterion's result
    ("each over 333 articles", "each over 334 articles"),  # a count in the prose
    ("| pilot | 13:56:42 | 14:15:54 |", "| pilot | 13:56:42 | 14:15:45 |"),  # a time from a pod log
    ("about $46.02 at that rate", "about $46.20 at that rate"),  # a figure from docs/log.md
    ("which would favor both;", "which would favor neither;"),  # a word in a line checked whole
    ("3,411,585,854 bytes", "3,411,585,845 bytes"),  # a figure from SHA256SUMS.txt
    ("### G2: the three runs", "### G2: the 3 runs"),  # a number in a heading
    ("| `s8_ei` minus `s8_ei_1` | Log loss | -0.0091 (-0.0151 to -0.0030) | s8_ei better |\n", ""),  # a row left out
    ("on the 59 prompts in 25 articles", "on the 69 prompts in 25 articles"),  # correction 14's section
    ("| -0.1063 (-0.2817 to 0.0051) | no difference was detected; excludes below -0.2817 and above 0.0051 |",
     "| -0.1063 (-0.2817 to 0.0051) | lower on the shared articles |"),  # correction 15's reading
    ("and $1.32 with 50 percent handed off", "and $1.23 with 50 percent handed off"),  # a cost
    ("(96.31 percent;", "(96.13 percent;"),  # a percentage
]


def render_prose(src, template):
    keys = []

    def sub(m):
        key, how = m.group(1).rsplit(":", 1)
        keys.append((key, how))
        return fmt(src.resolve(key)[1], how)

    return re.sub(r"\{\{(.+?)\}\}", sub, template), keys


def render_section(src, name):
    tabs = {t[0]: t for t in tables(src)}
    verb = {label: text for label, _, text in verbatim(src)}
    out = []
    for item in regions(src)[name]:
        kind = item[0]
        if kind == "h":
            out += [item[1], ""]
        elif kind == "x":
            out += [item[1], ""]
        elif kind == "p":
            out += [render_prose(src, item[2])[0], ""]
        elif kind == "v":
            out += [verb[item[1]], ""]
        elif kind == "t":
            t = tabs[item[1]]
            if t[3]:
                out += [render_table(src, t), ""]
    return "\n".join(out)


def tokens(text):
    return TOKEN_RE.findall(SKIP_RE.sub(" ", text))


def shape(text):
    """The line with every number replaced, so that a line whose numbers changed still finds its template."""
    return TOKEN_RE.sub("#", text)


def decimals(text):
    t = text.replace("$", "").replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


def as_number(text):
    return float(text.replace("$", "").replace(",", ""))


def section_lines(text, heading):
    """The lines of a section, from its heading to the next line that starts with '## '."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line == heading]
    if len(starts) != 1:
        sys.exit(f"the document has {len(starts)} lines {heading!r}")
    out = []
    for line in lines[starts[0]:]:
        if out and line.startswith("## "):
            break
        out.append(line)
    return out


class Check:
    """One pass over a document's checked sections."""

    def __init__(self, src, text):
        self.src, self.rows, self.readings, self.problems = src, [], [], []
        self.tables = {t[0]: t for t in tables(src)}
        self.seen_tables = set()
        self.prose = {}
        for name, items in regions(src).items():
            for item in items:
                if item[0] == "p":
                    line, keys = render_prose(src, item[2])
                    if shape(line) in self.prose:
                        sys.exit(f"two templates share a shape: {item[1]}")
                    self.prose[shape(line)] = (item[1], keys)
        self.verbatim = {text_: (label, source) for label, source, text_ in verbatim(src)}
        self.seen_prose, self.seen_verbatim = set(), set()
        self.heading, self.table = None, []
        g6 = src.cmp.get("g6", {}).get("sha256")
        ht = src.r / "handoff_test.json"
        if g6 is None or not ht.exists() or sha256_file(ht) != g6:
            self.problems.append("handoff_test.json's SHA-256 is not the one comparison.json's g6 names")
        for name in SECTIONS:
            self.run(name, section_lines(text, name))
        for number, t in self.tables.items():
            if number not in self.seen_tables and t[3]:
                self.problems.append(f"Table {number} ({t[1]}) is missing from the document")
        for name, items in regions(src).items():
            for item in items:
                if item[0] == "p" and item[1] not in self.seen_prose:
                    self.problems.append(f"the paragraph {item[1]!r} is missing from the document, or its words were changed")
        for label, _, _ in verbatim(self.src):
            if label not in self.seen_verbatim:
                self.problems.append(f"the line {label!r} is missing from the document, or differs from its source")

    def number(self, where, text, source, value, how):
        if how in ("str", "time"):
            want = fmt(value, how)
            self.rows.append((where, text, source, want, "PASS" if text == want else "MISMATCH"))
            return
        if value is None:
            self.rows.append((where, text, source, "null", "PASS" if text == "not computed" else "MISMATCH"))
            return
        d = decimals(text)
        try:
            written = as_number(text)
        except ValueError:
            self.rows.append((where, text, source, value, "MISMATCH"))
            return
        if how == "int":
            ok = d == 0 and written == value
            shown = f"{int(value):,}"
        elif how == "c0":
            ok = d == 0 and abs(written - float(value)) <= 0.5 + 1e-9
            shown = f"{float(value):.6f}"
        else:
            scale = 100.0 if how.startswith("pct") else 1.0
            want_d = int(how[3]) if how.startswith(("pct", "usd")) else int(how[1:])
            if how[0] == "g" and abs(float(value)) >= 1000 and "," not in text:
                ok_commas = False
            else:
                ok_commas = True
            ok = d == want_d and abs(written - scale * float(value)) <= 0.5 * 10 ** (-d) + 1e-12
            if text.startswith("-") and round(scale * float(value), d) == 0:
                ok = False  # a negative zero would be printed
            if how.startswith("usd") and not text.startswith("$"):
                ok = False
            if not ok_commas:
                ok = False
            shown = f"{scale * float(value):.6f}"
        self.rows.append((where, text, source, shown, "PASS" if ok else "MISMATCH"))

    def interval(self, where, text, base, parts_wanted=("estimate", "low", "high")):
        parts = {p: self.src.resolve(f"{base}|{p}") for p in ("estimate", "low", "high") if p in parts_wanted}
        first = parts_wanted[0]
        if parts[first][1] is None:
            self.rows.append((where, text, parts[first][0], "null", "PASS" if text == "not computed" else "MISMATCH"))
            return
        if parts_wanted == ("low", "high"):
            m = re.fullmatch(r"(-?\d+\.\d+) to (-?\d+\.\d+)", text)
        else:
            if parts["low"][1] is None:
                self.number(f"{where}, estimate", text, parts["estimate"][0], parts["estimate"][1], "f4")
                return
            m = re.fullmatch(r"(-?\d+\.\d+) \((-?\d+\.\d+) to (-?\d+\.\d+)\)", text)
        if not m:
            self.rows.append((where, text, parts[first][0], "(cell not parsed)", "MISMATCH"))
            return
        for p, t in zip(parts_wanted, m.groups()):
            self.number(f"{where}, {p}", t, parts[p][0], parts[p][1], "f4")

    def cell(self, where, text, spec):
        src = self.src
        if isinstance(spec, str):
            if text != spec:
                self.problems.append(f"{where}: label {text!r}, expected {spec!r}")
            return
        kind = spec[0]
        if kind == "num":
            source, value = src.resolve(spec[1])
            self.number(where, text, source, value, spec[2])
        elif kind == "iv":
            self.interval(where, text, spec[1])
        elif kind == "rng":
            self.interval(where, text, spec[1], ("low", "high"))
        elif kind == "read":
            want = cell_text(src, spec)
            self.readings.append((where, text, f"{want} (from the interval of {spec[1]})", "PASS" if text == want else "MISMATCH"))
        elif kind in ("str", "pf"):
            source = src.resolve(spec[1])[0]
            want = cell_text(src, spec)
            self.readings.append((where, text, f"{want} ({source})", "PASS" if text == want else "MISMATCH"))

    def flush(self):
        if not self.table:
            return
        lines, self.table = self.table, []
        m = re.match(r"### Table (\d+)\. (.*)", self.heading or "")
        if not m or int(m.group(1)) not in self.tables:
            self.problems.append(f"a table under {self.heading!r} that the check does not know")
            return
        number, title, header, rows = self.tables[int(m.group(1))]
        self.seen_tables.add(number)
        if m.group(2) != title:
            self.problems.append(f"Table {number}: title {m.group(2)!r}, expected {title!r}")
        got_header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
        if got_header != header:
            self.problems.append(f"Table {number}: header {got_header}, expected {header}")
            return
        body = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines[2:]]
        if len(body) != len(rows):
            self.problems.append(f"Table {number}: {len(body)} rows, expected {len(rows)}")
        for i, (got, want) in enumerate(zip(body, rows), start=1):
            if len(got) != len(want):
                self.problems.append(f"Table {number}, row {i}: {len(got)} cells, expected {len(want)}")
                continue
            labels = ", ".join(c for c in want if isinstance(c, str) and c)
            for text, spec, col in zip(got, want, header):
                self.cell(f"Table {number}, {labels}, {col}", text, spec)

    def run(self, name, section):
        self.heading, self.table = None, []
        for n, line in enumerate(section, start=1):
            if line.startswith("|"):
                self.table.append(line)
                continue
            self.flush()
            if line.startswith("#"):
                self.heading = line
                if tokens(line) and not re.match(r"### Table \d+\. ", line):
                    self.problems.append(f"heading with unexplained numbers: {line!r}")
                elif re.match(r"### Table \d+\. ", line) and tokens(line.split(". ", 1)[1]):
                    self.problems.append(f"table title with unexplained numbers: {line!r}")
                continue
            if not line.strip():
                continue
            where = f"{SHORT[name]}, line {n}"
            if line in self.verbatim:
                label, source = self.verbatim[line]
                self.seen_verbatim.add(label)
                self.rows.append((f"{where} ({label}, the whole line)", "verbatim", source, "the same text", "PASS"))
                continue
            found = tokens(line)
            entry = self.prose.get(shape(line))
            if entry is None:
                if found:
                    self.problems.append(f"{where}: numbers without a source key: {found} in {line[:90]!r}")
                continue
            pid, keys = entry
            self.seen_prose.add(pid)
            if len(found) != len(keys):
                self.problems.append(f"{where} ({pid}): {len(found)} numbers found, {len(keys)} source keys")
                continue
            for k, (t, (key, how)) in enumerate(zip(found, keys), start=1):
                source, value = self.src.resolve(key)
                self.number(f"{where} ({pid}), number {k}", t, source, value, how)
        self.flush()

    def failures(self):
        return [r for r in self.rows if r[4] != "PASS"] + [r for r in self.readings if r[3] != "PASS"] + self.problems


def self_test(src, text):
    for i, (old, new) in enumerate(PLANTED, start=1):
        if text.count(old) != 1:
            sys.exit(f"self-test: planted error {i} cannot be placed ({old!r} is not in the document exactly once)")
        if not Check(src, text.replace(old, new)).failures():
            sys.exit(f"self-test: planted error {i} ({new!r}) was not reported")
    return len(PLANTED)


def write_outputs(root, text, check, planted):
    out_doc = root / "docs" / "fiorillo_v0_4_figure_check.md"
    sources_out = root / "results" / "fiorillo_v0_4" / "figure_sources.json"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    now = dt.datetime.now(dt.timezone.utc)
    figures = [{"where": r[0], "as_written": r[1], "source": r[2], "source_value": r[3], "check": r[4]} for r in check.rows]
    readings = [{"where": r[0], "as_written": r[1], "expected": r[2], "check": r[3]} for r in check.readings]
    sources_out.write_text(json.dumps({"plan": "docs/prompt_session8.md, step 16", "document": "docs/fiorillo_v0_4.md",
                                       "document_sha256": sha, "time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       "figures": figures, "readings": readings}, indent=1) + "\n", encoding="utf-8")

    def esc(x):
        return str(x).replace("|", "/")

    out = [
        "# Fiorillo v0.4: figure check",
        f"Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: {now.day} {now.strftime('%B %Y')}",
        "Sources: docs/fiorillo_v0_4.md (its sections How each stage was run, Results (test split) and Checks not "
        "pre-registered); results/fiorillo_v0_4/ (comparison.json, release_candidate.json, testpreds_log.json, calibration_s8.json, "
        "ece_limit_evidence_effect.json, handoff_validation.json, handoff_test.json, power.json, power_validation.json, "
        "pilot.json, trees.json, readmission_pool.json, readmission_pool_test.json, trees_test.json, zeros.json, "
        "article_selection.json, records_manifest.json, contamination_checks.json, large_model/gemma4_31b_letters.json, "
        "large_model/gemma4_31b_timing.json, metrics/, run_summaries/ and logs/pod/); docs/log.md; "
        "models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt; data/processed/session8/ and data/processed/public/ (the test "
        "records' counts)",
        "Status: complete",
        "",
        "Abbreviations: EI is Evidence Inference 2.0; ECE is expected calibration error; AUROC is the area under the receiver "
        "operating characteristic curve; UTC is Coordinated Universal Time; SHA-256 is Secure Hash Algorithm 256; GPU is graphics "
        "processing unit; GiB is 2^30 bytes; PMCID is a PubMed Central identifier; PQA-A, PQA-U and PQA-L are PubMedQA's "
        "automatically labeled, unlabeled and expert-labeled parts; OSF is the Open Science Framework; G1 to G6 are the plan's "
        "comparisons (docs/prompt_session8.md, COMPARISONS).",
        "",
        "## How the check was made",
        "",
        "src/eval/check_fiorillo_v0_4_figures.py read three sections of docs/fiorillo_v0_4.md as written (SHA-256 of the file "
        f"checked: {sha}): How each stage was run, Results (test split), and Checks not pre-registered (corrections 14 and 15). It "
        "found every "
        "number in them and compared each with its source at the printed precision: a number printed with four decimals passes "
        "when it is the source value rounded to four decimals, a count when it equals the source, a time when it equals the "
        "source's time of day. Table cells were matched to their sources by the table's heading and its rows and columns, which "
        "the script fixes for each table. Numbers in the prose were matched, in order, to the source keys of the line's template "
        "in the script, and each key was resolved by the script's own code from the results files, the pod's stage logs, "
        "docs/log.md (the pod's times and cost) and SHA256SUMS.txt. Lines that state a source's text were checked whole against "
        "it. Every reading was checked against its interval and the direction of its metric, read as compare_session8.py reads "
        "it, and every PASS or FAIL against the rule's own result. The script also checked that handoff_test.json is the file "
        "comparison.json names, by its SHA-256. Before the document was checked, "
        f"{planted} errors were planted in copies of it, one at a time, and each was reported. The script checks numbers and "
        "readings, not the wording of claims. Every figure's source is also listed in results/fiorillo_v0_4/figure_sources.json.",
        "",
        "Not checked here: the document's other sections, which hold the plan's fixed settings (docs/prompt_session8.md) and the "
        "published figures, each cited to its paper and to docs/prior_art.md (the check of 1 October 2026).",
        "",
        "Not counted as numbers, wherever they appear: version names (v0.1 to v0.4, vLLM 0.30.0, CUDA versions), model names "
        "(Qwen3-4B-Base, Gemma 4 31B-it, Gemma 4 E2B-it, Evidence Inference 2.0), idea numbers (I-01 and so on), G1 to G6 and H1 "
        "to H4; table, correction, criterion, session, step and seed numbers; dates and the years 2010 to 2029; the phrases "
        "\"95 percent\", \"95th\", \"30 days\", \"80 percent chance\" and \"prints as 0\"; readmission's levels 0 to 2; and "
        "anything in code type or holding an underscore or a slash. Line numbers count each section's heading as line 1.",
        "",
        "## Result",
        "",
        f"{len(check.rows):,} numbers and whole lines: {len(check.rows):,} PASS, 0 MISMATCH. {len(check.readings):,} readings: "
        f"{len(check.readings):,} PASS, 0 MISMATCH.",
        "",
        "## Every number",
        "",
        "| # | Where | As written | Source | Source value | Check |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(check.rows, start=1):
        out.append(f"| {i} | {esc(r[0])} | {esc(r[1])} | {esc(r[2])} | {esc(r[3])} | {r[4]} |")
    out += ["", "## Every reading", "", "| # | Where | As written | From the source | Check |", "|---|---|---|---|---|"]
    for i, r in enumerate(check.readings, start=1):
        out.append(f"| {i} | {esc(r[0])} | {esc(r[1])} | {esc(r[2])} | {r[3]} |")
    out_doc.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out_doc, sources_out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", nargs="?", default="check", choices=("check", "render"))
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository (for a test on a copy)")
    args = parser.parse_args(argv)
    src = Sources(args.root)
    if args.mode == "render":
        for name in SECTIONS:
            print(render_section(src, name))
        return
    doc = args.root / "docs" / "fiorillo_v0_4.md"
    text = doc.read_text(encoding="utf-8")
    planted = self_test(src, text)
    check = Check(src, text)
    failures = check.failures()
    if failures:
        for f in failures:
            print("FAIL", f)
        sys.exit(f"{len(failures)} failures; the check document was not written")
    if not planted:
        sys.exit("no planted errors: the self-test did not run")
    out_doc, sources_out = write_outputs(args.root, text, check, planted)
    print(f"self-test: {planted} planted errors, each reported. {len(check.rows):,} numbers and whole lines and "
          f"{len(check.readings):,} readings, all PASS; wrote {sources_out.relative_to(args.root)} and {out_doc.relative_to(args.root)}")


if __name__ == "__main__":
    main()

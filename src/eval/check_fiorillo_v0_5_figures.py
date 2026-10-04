"""Session 9 (docs/prompt_session9.md, step 16): the figure check of docs/fiorillo_v0_5.md.

It does for v0.5 what src/eval/check_fiorillo_v0_4_figures.py did for v0.4. That script held the sentences and tables of
a document already written, with each figure's source key; this one was built before docs/fiorillo_v0_5.md's results were
written, so it holds no sentence and finds each number's source itself, among the files that produced it.

The sections checked: those whose level-2 headings start with "How each stage was run" or "Results" (--sections changes
the list), each from its heading to the next line that starts with "## ".

The sources: session 9's results files in results/fiorillo_v0_5/, listed in FILES with the module that writes each (its
code gave the file's layout). Step 16 added the run summaries and metrics, the large model's, BioGPT-Large's and the
reading checks' own files, pod_console.json (the Runpod console's figures as Claude read them) and the pod's stage logs
(logs/pod/*.log: each time-stamped line's time of day, and every number in a line, under the line's words; a FILES
entry may be a pattern). Every number in them is a source, under its file and key (a list's entries are named by
their place and by the fields that identify them, such as section, system, reference and metric); so is every list's
length, the time of day of every time stamp, every number written in a rule's text, and every interval (a part of a file
that holds estimate, low and high, or mean, min and max). A file that does not exist is listed and skipped.

The numbers: every number in the checked sections except identifiers and fixed phrases (SKIP: anything in code type or
holding an underscore or a slash, versions and model names, table, correction, criterion, session, step and seed numbers,
idea and comparison names, dates and years, "95 percent", "per 1,000" and the like). "estimate (low to high)" is read as
one interval and "low to high" as one range. A number followed by "percent", the numbers of a list that ends in
"percent", and the numbers of a table column whose heading says percent are percentages; "$" marks dollars; HH:MM or
HH:MM:SS is a time of day.

The rounding rule, as the documents print numbers. A number matches a source value when the value, rounded to the
number's printed decimals, is the number: the two differ by at most half a unit in the last printed place. A percentage
is compared with 100 times the value, or with the value where the source holds a percentage; a time with the time of day
of a time stamp, to the minute or to the second as printed. The documents print thousands separators (1,218) and never
-0.0000, so a number printed without them, or as a negative zero, does not match even where its value does.

Which source a number means. Every source's key is split into words (its file's name, its keys and the fields that name
its list entry), and so is the number's context: in a table, its column's heading, its row's labels, its cell's words
and the table's title (and, more weakly, the row's other whole numbers, such as a rate); in prose, the words before it
in its clause (from the number before it), the rest of its clause, the rest of its sentence, the identifiers of its
paragraph and its headings. Words are lowercased, plurals cut, common words dropped and the documents' names mapped to
the files' (Evidence Inference and EI to evidence_effect, readmission to readmit_window, Gemma 4 31B-it to gemma4_31b,
session 8 to session8 and so on: words()). A source fits a number by the words they share, each weighted by its place in
the context (W_NEAR, W_CLAUSE, W_SENTENCE, W_PARAGRAPH, W_HEADING, W_TITLE) and by how rare it is among the sources'
keys, log(1 + N / df) for N sources and df of them holding the word; a word of the source's own last key counts
LEAF_BONUS times. A sentence, row or table title that names results files of session 9 limits the search to those files.

The three outcomes, one for each number. A source is near a number when a word of the source's own last key (log_loss
for log_loss.estimate) stands in the number's clause, or in its table cell's row or column. The context names a source
clearly when the best fit is at least TAU (what one word next to the number gives when TAU_DF of the sources hold it)
and that source is near the number.
- PASS: a source prints as the number, a word of its context is in the source's key, and the source fits the context at
  least RHO times as well as the best fit, or is near the number. Failing that, when the context names no source
  clearly, a source that prints as the number and fits the context at least half TAU passes, and the note names the
  source that fits better. Its file and key are given. An interval passes when its three numbers are one source's
  estimate, low and high (also when no word of its context is in that source's key: three numbers seldom agree by
  chance); a range, when its two are one source's low and high and a word of its context is in that source's key.
- MISMATCH: no such source prints as it, and its context names a source clearly. The source its context names best is
  given, with its value as the documents would print it. An interval two of whose numbers are one source's is that
  source's, and its third number does not match. A number whose value matches but whose print form does not (no
  thousands separators, or -0.0000) is a MISMATCH.
- NO SOURCE: neither. Where a source holds its value but shares no word with its context, the row names it: a value
  alone is not a source, since thousands of the sources' values have four decimals. A number with no source is
  attributed when the document labels it derived or its sentence or table row names only files that are not sources
  (docs/log.md, an earlier session's results and the like): its source is outside the files checked. A number whose sentence calls it
  derived is marked so, and is NO SOURCE rather than MISMATCH when no source prints as it (the source its context names
  is given in the note): the check does not compute derived figures. So is a number whose sentence or table names .json
  files none of which is a source: its source is outside the files checked.

Readings. A cell of a table column headed "Reading", a reading in prose after an interval of the same sentence (no
difference was detected, lower (better), higher (worse) and the others compare_session8.py writes), PASS or FAIL, and a
quoted text are read as claims. A claim's source is the nearest reading (a key reading, difference_reading, result,
overall or outcome) above the sources that the numbers of its row or sentence matched, or else the reading its context
fits best. A comparison's reading is checked by its meaning (no difference, which side is better, or higher or lower;
"<name> better" names the system or the reference); any other reading, result or outcome by its words. A quoted text
that no reading of the sources fits is not a claim of the results and is left out.

The self-test. Before the document is checked, errors are planted in copies of it, one at a time: up to PLANTS numbers
that pass (intervals, table cells, prose numbers, counts, percentages, dollars and times, spread over the lines), each
with one digit changed to a value that no source holds, and up to READING_PLANTS readings that pass, each replaced by a
reading of another meaning. The script stops, writing nothing, unless each planted error is reported.

The check is complete when no number mismatches, no reading fails, and every number with no source is attributed.

Outputs: results/fiorillo_v0_5/figure_sources.json (every number and reading: where it is, as written, the outcome, its
source or the source its context names, and that source's value; the numbers and the readings are each one table, the
field names once and then one row a line, with a field whose values are dictionaries written as a list in the order
its "dict_fields" entry gives, so that the file stays under the data guard's 1 MB limit for results/; from_table()
reads a table back as dictionaries) and docs/fiorillo_v0_5_figure_check.md (every number,
one row each, with its file and key and PASS, MISMATCH or NO SOURCE; then the numbers that do not match; then the numbers
with no source; then every reading). Both are written whatever the outcome. The exit code is 0 when the check is
complete and 1 otherwise; it is 2, with nothing written, when the self-test fails or the document cannot be checked.

Usage: python3 src/eval/check_fiorillo_v0_5_figures.py [--root DIR] [--doc FILE] [--results DIR] [--out-doc FILE]
           [--out-sources FILE] [--sections HEADING [HEADING ...]]
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/prompt_session9.md, step 16"
STUDY = "ed-decision-model (open medical typed-decision encoder built on Laya)"
DOC = Path("docs") / "fiorillo_v0_5.md"
RESULTS = Path("results") / "fiorillo_v0_5"
OUT_DOC = Path("docs") / "fiorillo_v0_5_figure_check.md"
OUT_SOURCES = RESULTS / "figure_sources.json"
SECTIONS = ("How each stage was run", "Results")

# session 9's results files (in RESULTS), each with the module whose code gave its layout
FILES = (
    ("comparison.json", "src/eval/compare_session9.py"),
    ("recipe_selection.json", "src/eval/recipe_session9.py next"),
    ("cost_rule.json", "src/eval/recipe_session9.py cost"),
    ("calibration_s9.json", "src/eval/calibrate_session9.py fit"),
    ("release_candidate.json", "src/eval/calibrate_session9.py freeze"),
    ("reading_checks.json", "src/eval/reading_checks_session9.py combine"),
    ("handoff_curves.json", "src/eval/handoff_curves_session9.py curves, on test"),
    ("handoff_validation.json", "src/eval/handoff_curves_session9.py curves, v0.5 on validation"),
    ("handoff_rule_v05.json", "src/eval/handoff_curves_session9.py fit-rule, v0.5"),
    ("validation_analyses/handoff_curves_validation_s8.json", "src/eval/handoff_curves_session9.py curves, step 6"),
    ("validation_analyses/handoff_rule_s8.json", "src/eval/handoff_curves_session9.py fit-rule, step 6"),
    ("trees_v2.json", "src/eval/trees_session9.py recode, refit and lightgbm"),
    ("tabicl.json", "src/eval/trees_session9.py tabicl"),
    ("readmission_pool_v2.json", "src/eval/trees_session9.py pool"),
    ("trees_v2_test.json", "src/eval/trees_session9.py test"),
    ("readmission_pool_v2_test.json", "src/eval/trees_session9.py test"),
    ("weight_average.json", "src/eval/weight_average_session9.py build"),
    ("power.json", "src/eval/power_session9.py plan"),
    ("records_manifest.json", "src/tasks/ei_session9.py manifest"),
    ("license_recheck.json", "src/tools/ei_license_recheck.py classify"),
    # added in step 16, so that the How each stage was run section's figures have their sources too
    ("metrics/*_metrics.json", "src/train/train_qwen_head.py train (each run's monitoring, timing and memory)"),
    ("run_summaries/*_run_summary.json", "src/train/train_qwen_head.py predict (each prediction pass)"),
    ("checks_s8/*_run_summary.json", "src/train/train_qwen_head.py predict (THE READING CHECKS' passes, s8_ei)"),
    ("checks_v05/*_run_summary.json", "src/train/train_qwen_head.py predict (THE READING CHECKS' passes, v0.5)"),
    ("checks_s8.json", "src/eval/reading_checks_session9.py score, s8_ei"),
    ("checks_v05.json", "src/eval/reading_checks_session9.py score, v0.5"),
    ("large_model/gemma4_31b_long_timing.json", "src/eval/large_model_session8.py (copied to the name gemma4_31b_long)"),
    ("large_model/gemma4_31b_long_letters.json", "src/eval/large_model_session8.py (copied to the name gemma4_31b_long)"),
    ("biogpt.json", "src/eval/biogpt_session9.py"),
    ("pilot_long.json", "src/train/train_qwen_head.py pilot-s8"),
    ("fixed_versions.json", "step 0 (d)"),
    ("power_validation.json", "src/eval/power_session9.py validation"),
    ("ece_limit_evidence_effect.json", "src/eval/calibrate_session9.py fit"),
    ("testpreds_log.json", "scripts/pod_session9.sh testpreds"),
    ("not_built.json", "scripts/pod_session9.sh package"),
    ("zeros.json", "src/eval/compare_session9.py"),
    ("pod_console.json", "the Runpod console, read by Claude in steps 9 and 13 (docs/log.md)"),
    ("logs/pod/*.log", "scripts/pod_session9.sh (each stage's log: its time-stamped lines and the numbers in its lines)"),
)
LOG_NOISE = re.compile(r"it/s\]|s/it\]|Loading weights|INFO|WARNING|EngineCore|UserWarning|regex pattern|JIT kernel|Siglip|"
                       r"Fetching \d|warn\(|Warning:|^\s*$")
LOG_STAMP = re.compile(r"(\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)Z)\s+(.*)")

# the matching's settings (the module's docstring says what each does)
W_NEAR, W_CLAUSE, W_SENTENCE, W_PARAGRAPH, W_HEADING, W_TITLE = 3.0, 1.5, 0.75, 0.5, 0.5, 1.0
LEAF_BONUS = 1.5
RHO = 0.5
TAU_DF = 50
TAU_DF_READINGS = 5
PLANTS = 12
READING_PLANTS = 2

ROLE_CODES = {"estimate": 1, "low": 2, "high": 3, "mean": 1, "min": 2, "max": 3}
ROLE_SETS = (("estimate", "low", "high"), ("mean", "min", "max"))
PARTS = {"interval": ("estimate", "low", "high"), "range": ("low", "high")}
ID_FIELDS = ("section", "evaluation", "question", "split", "system", "reference", "metric", "subset", "group", "input",
             "statistic", "name", "fiorillo", "large_model", "candidate", "member", "family", "stage", "drop")
NUMERIC_ID_FIELDS = ("real_gain", "worse_than_s8_ei_by", "rate", "seed")
AXIS_SUFFIX = "_percent"
READING_KEYS = ("reading", "difference_reading", "result", "overall", "outcome")
DIRECTIONS = ("no difference was detected", "lower (better)", "higher (better)", "lower (worse)", "higher (worse)")
RESULT_WORDS = ("PASS", "FAIL")

# ---------------------------------------------------------------- words

STOP = frozenset("""a an the of to in on at by for from with and or is are was were be been being its it this that these
those as than then each every all which who whose has have had but so if into per about also both their there here when
where while after before between table tables s we our us one two three four five only more most less same other others
any such not whom what how can could would should may might will shall do does did done""".split())
SYNONYMS = {"training": "train", "readmission": "readmit", "criterion": "criteria", "dollar": "usd", "dollars": "usd",
            "percentage": "percent", "percentages": "percent", "coverage": "cover"}
PRE = tuple((re.compile(p), r) for p, r in (
    (r"gemma 4 31b-it", " gemma4_31b "), (r"gemma 4 e2b-it", " gemma4_e2b "), (r"\bgemma 4\b", " gemma4 "),
    (r"\bsessions? (\d+)", r" session\1 "), (r"\bv0\.(\d+)\b", r" v0\1 "), (r"(?<=\d),(?=\d{3}\b)", "")))
POST = tuple((re.compile(p), r) for p, r in (
    (r"\bhand[- ]?offs?\b|\bhanded off\b", " handoff "), (r"\bevidence inference\b", " evidence effect "),
    (r"\bei\b", " evidence effect "), (r"\bexpected calibration error\b", " ece "), (r"\bbrier scores?\b", " brier "),
    (r"\bstandard error\b", " se "), (r"\bdesign effect\b", " d "), (r"\b(\d+)(?:st|nd|rd|th) percentile\b", r" p\1 "),
    (r"\bweight[- ]averag\w*", " weight average wavg "), (r"\bswap[- ]averag\w*", " swap averaging "),
    (r"\blearning rates?\b", " learning rate lr ")))
DEC_RE = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
IDENT_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
WORD_RE = re.compile(r"[a-z0-9]+")


def canon(x):
    """A number as a word: 10 for 10.0, 0.05 for 0.050."""
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.8f}".rstrip("0").rstrip(".")


def stem(w):
    """A light stemmer, the same for the document's words and the keys' (covered and coverage, hour and hourly, score
    and scores): a suffix is cut, and then one more from what is left."""
    for _ in range(2):
        if len(w) < 5:
            return w
        for suffix in ("ing", "ed", "ly", "es", "s", "e"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                if suffix == "s" and w.endswith(("ss", "us", "is")):
                    continue
                if suffix == "es" and not w.endswith(("sses", "ches", "shes", "xes")):
                    continue
                w = w[:-len(suffix)]
                break
        else:
            return w
    return w


def words(text):
    """The words of a text or a key: identifiers (s9_ei) kept whole and split, numbers as canon() gives them, the
    documents' names mapped to the files' words, plurals cut and common words dropped."""
    t = str(text).lower().replace("\u2019", "'").replace("\u2212", "-")
    t = re.sub(r"'s\b", " ", t)
    for pat, rep in PRE:
        t = pat.sub(rep, t)
    out = {canon(m) for m in DEC_RE.findall(t)}
    t = DEC_RE.sub(" ", t)
    out.update(IDENT_RE.findall(t))
    t = t.replace("_", " ")
    for pat, rep in POST:
        t = pat.sub(rep, t)
    for w in WORD_RE.findall(t):
        if w.isdigit():
            out.add(canon(int(w)))
            continue
        w = stem(SYNONYMS.get(w, w))
        if w not in STOP:
            out.add(w)
    return out


def add(ctx, toks, weight):
    for t in toks:
        if ctx.get(t, 0.0) < weight:
            ctx[t] = weight


# ---------------------------------------------------------------- numbers in the document

N = r"\d+(?:,\d{3})*"
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"


def listed(word):
    """A word followed by one number or a list of them ("Tables 11 to 15", "corrections 2 and 3", "session 8's")."""
    return rf"\b{word} {N}(?:'s)?(?:(?:, {N})*,? (?:and|to|or) {N})?"


# identifiers and fixed phrases that are not figures (listed in the check document)
SKIP = [
    r"`[^`]*`", r"\S*[_/]\S*", r"\bv\d+(?:\.\d+)+\b", r"\b\d+(?:\.\d+){2,}\b",
    r"\b(?:Qwen|BioGPT|TabICL|Llama|Mistral)[\w.-]*", r"\bGemma \d+(?: \d+B-it| E\d+B-it)?",
    r"\bI-\d{2}\b", r"\b[GHK]\d{1,2}\b", r"\bEvidence Inference 2\.0\b", r"\bEI 2\.0\b", r"\bCUDA \d+(?:\.\d+)?\b",
    r"\bCC BY(?:-[A-Z]+)* \d\.\d\b", r"\bSHA-256\b", r"\bICD-\d+(?:-CM)?\b",
    listed("[Tt]ables?"), listed("[Ff]igures?"), listed("[Cc]orrections?"), listed("[Cc]riteri(?:on|a)"),
    listed("[Ss]essions?"), listed("[Ss]teps?"), listed("[Ss]eeds?"), listed("[Dd]ecisions?"), listed("[Rr]ounds?"),
    rf"\b\d{{1,2}} {MONTHS}(?: \d{{4}})?\b", rf"\b{MONTHS} \d{{1,2}}(?:, \d{{4}})?\b",
    r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?\b", r"(?<![\d.,])\b20[12]\d\b(?![.,]\d)", r"\b20\d{6}\b",
    r"\b95 percent\b", r"\b\d+(?:st|nd|rd|th)\b", r"\b30 days\b", r"\bprints? as 0\b", r"\b80 percent chance\b",
    r"\blevels? [0-2](?:, [0-2])*(?: and [0-2])?\b", r"\b\d+B\b", r"\bper 1,000\b", r"^\s*\d+\.\s",
]
SKIP_RE = re.compile("|".join(f"(?:{p})" for p in SKIP))
TIME = r"(?<![\d:])\d{2}:\d{2}(?::\d{2})?(?![\d:])"
SCI = r"(?<![\w.$])(?<![A-Za-z]-)[-\u2212]?\d+(?:\.\d+)?[eE][-+\u2212]?\d+(?![\w.])"
NUM = r"(?<![\w.$])(?<![A-Za-z]-)(?:[-\u2212]\$?|\$[-\u2212]?)?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z_])"
TOKEN_RE = re.compile(f"{TIME}|{SCI}|{NUM}")
TIME_RE = re.compile(r"\d{2}:\d{2}(?::\d{2})?")
PERCENT_AFTER = re.compile(r"\s?(?:percent\b|%)")
LIST_GAP = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|to)\s+")
OPEN_GAP = re.compile(r"\s*(?:percent\s*)?\(\s*")
TO_GAP = re.compile(r"\s+(?:percent\s+)?to\s+")
CLOSE_AFTER = re.compile(r"\s*(?:percent\s*)?\)")
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
JSON_RE = re.compile(r"[\w./-]+\.json\b")
LOG_RE = re.compile(r"[\w./-]+\.log\b")
OTHER_RE = re.compile(r"[\w./-]+\.(?:md|txt|csv)\b")  # documents and files that are never sources (docs/log.md and the like)
SENTENCE_GAP = re.compile(r"(?<! al\.)(?<!e\.g\.)(?<!i\.e\.)(?<! vs\.)(?<=[.!?])\s+(?=[A-Za-z(\"`*\u201c])")
CLAUSE_GAP = re.compile(r";\s*")
LABEL_HEADER = re.compile(r"\s*(?:#|no\.|criteri(?:on|a)|steps?|stages?|seeds?|sessions?|corrections?|"
                          r"tables?|rank|order)\s*", re.I)  # columns of labels, not figures
DIRECTION_RE = re.compile("|".join(re.escape(d) for d in DIRECTIONS), re.I)
RESULT_RE = re.compile(r"\b(?:PASS|FAIL)\b")
QUOTE_RE = re.compile(r"\"([^\"]{3,}?)\"|\u201c([^\u201d]{3,}?)\u201d")


class Fig:
    """One number as the document prints it: its text, its place in the line, its value and how it is printed."""

    def __init__(self, text, start, end):
        self.text, self.start, self.end, self.percent = text, start, end, False
        t = text.replace("\u2212", "-")
        self.negative = self.dollar = self.sci = False
        self.separators_ok = True
        if TIME_RE.fullmatch(t):
            self.kind, self.value, self.decimals = "time", None, None
            return
        self.kind = "number"
        self.dollar = "$" in t
        t = t.replace("$", "")
        self.negative = t.startswith("-")
        m = re.fullmatch(r"(\d+(?:,\d{3})*)(?:\.(\d+))?(?:[eE]([-+]?\d+))?", t.lstrip("-"))
        intpart, frac, exp = m.group(1), m.group(2) or "", m.group(3)
        self.sci = exp is not None
        self.decimals = len(frac) - (int(exp) if exp else 0)
        self.value = float(t.replace(",", ""))
        self.separators_ok = self.sci or bool(re.fullmatch(r"\d{1,3}(?:,\d{3})*", intpart))


def mask(line):
    """The line with every SKIP phrase blanked, its other characters where they were."""
    return SKIP_RE.sub(lambda m: " " * len(m.group(0)), line)


def find_figures(line):
    """The numbers of a line, in order, and the line as masked."""
    masked = mask(line)
    figs = [Fig(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(masked)]
    for i, f in enumerate(figs):
        if f.kind == "number" and PERCENT_AFTER.match(masked, f.end):
            f.percent = True
            j = i
            while j > 0 and figs[j - 1].kind == "number" and LIST_GAP.fullmatch(masked[figs[j - 1].end:figs[j].start]):
                figs[j - 1].percent = True
                j -= 1
    return figs, masked


def units_of(figs, masked):
    """The numbers grouped: [(kind, figures)], an interval "a (b to c)", a range "b to c", or a single number."""
    units, i = [], 0
    while i < len(figs):
        f = figs[i]
        nums = [g.kind == "number" for g in figs[i:i + 3]]
        if len(nums) == 3 and all(nums) and OPEN_GAP.fullmatch(masked[f.end:figs[i + 1].start]) \
                and TO_GAP.fullmatch(masked[figs[i + 1].end:figs[i + 2].start]) and CLOSE_AFTER.match(masked, figs[i + 2].end):
            units.append(("interval", figs[i:i + 3]))
            i += 3
            continue
        if len(nums) >= 2 and all(nums[:2]) and TO_GAP.fullmatch(masked[f.end:figs[i + 1].start]):
            units.append(("range", figs[i:i + 2]))
            i += 2
            continue
        units.append(("single", [f]))
        i += 1
    return units


def spans(text, pattern, offset=0):
    """(start, end) of the pieces of text between the matches of pattern."""
    out, s = [], 0
    for m in pattern.finditer(text):
        out.append((offset + s, offset + m.start()))
        s = m.end()
    out.append((offset + s, offset + len(text)))
    return out


def containing(pieces, pos):
    for a, b in pieces:
        if a <= pos < b:
            return a, b
    return pieces[-1]


def cells(line):
    """[(text, offset)] of a table row's cells; offset is where the cell's text starts in the line."""
    bars = [i for i, ch in enumerate(line) if ch == "|"]
    if not line.rstrip().endswith("|"):
        bars.append(len(line))
    out = []
    for a, b in zip(bars, bars[1:]):
        piece = line[a + 1:b]
        out.append((piece.strip(), a + 1 + len(piece) - len(piece.lstrip())))
    return out


def fig_kind(f):
    if f.kind == "time":
        return "time"
    return "dollars" if f.dollar else ("percentage" if f.percent else "number")


def printed_like(v, f):
    """A source's value as the documents would print it, at the number's precision and in its form."""
    if f.kind == "time":
        return v if len(f.text) == 8 else str(v)[:5]
    x = float(v)
    if f.percent and abs(100.0 * x - f.value) < abs(x - f.value):
        x *= 100.0
    if f.sci:
        mant = f.text.replace("$", "").lstrip("-\u2212")
        mant = re.split(r"[eE]", mant)[0]
        return f"{x:.{len(mant.split('.')[1]) if '.' in mant else 0}e}"
    d = max(f.decimals, 0)
    s = f"{abs(x):,.{d}f}"
    neg = x < 0 and float(s.replace(",", "")) != 0
    return ("-" if neg else "") + ("$" if f.dollar else "") + s


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def raw(v):
    """A source's value as the check lists it."""
    if isinstance(v, str):
        return v
    v = float(v)
    if v.is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    return f"{v:.8g}"


def id_label(named):
    d = dict(named)
    out = []
    for f, s in named:
        if f == "system" and "reference" in d:
            out.append(f"{s} minus {d['reference']}")
        elif not (f == "reference" and "system" in d):
            out.append(s)
    return ", ".join(out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stop(message):
    print(message, file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------- the sources

class Sources:
    """Every number, time of day, interval and reading in session 9's results files, with its file and key."""

    def __init__(self, results):
        self.results = Path(results)
        self.files, self.missing, self.sha, self.modules = [], [], {}, {}
        self._file, self._path, self._key, self._value, self._time = [], [], [], [], []
        self._tokens, self._leaf, self._parent = [], [], []
        self.parents, self.leaves, self.cache = [], [], {}
        self.expanded = []
        for pattern, module in FILES:
            paths = sorted(self.results.glob(pattern)) if any(c in pattern for c in "*?[") else [self.results / pattern]
            for path in paths:
                rel = path.relative_to(self.results).as_posix()
                if not path.exists():
                    self.missing.append(rel)
                    continue
                self.expanded.append(rel)
                self.files.append(rel)
                self.sha[rel], self.modules[rel] = sha256_file(path), module
                toks = frozenset(words(path.stem))
                if rel.endswith(".log"):
                    self._log(rel, path.read_text(encoding="utf-8", errors="replace"), toks)
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except ValueError as err:
                    stop(f"{rel} cannot be read as JSON: {err}")
                self._json(rel, data, toks)
        self._finish()

    def _json(self, rel, data, toks):
            if isinstance(data, dict):
                self._dict(rel, data, (), "", toks, frozenset())
            elif isinstance(data, list):
                self._list(rel, data, (), "", toks, frozenset(), None)

    def _log(self, rel, text, toks):
        """A stage log: each line's time stamp (a time of day) and each number in its text, under the line's words."""
        for i, line in enumerate(text.splitlines(), 1):
            if LOG_NOISE.search(line) or len(line) > 400:
                continue
            m = LOG_STAMP.match(line)
            msg = (m.group(3) if m else line).strip()
            if not msg:
                continue
            mw = frozenset(words(msg))
            key = f"line {i}: {msg[:80]}"
            if m:
                self._add(rel, (i,), key, toks | mw, mw, math.nan, m.group(2), -1)
            for mm in TOKEN_RE.finditer(mask(msg)):
                f = Fig(mm.group(0), mm.start(), mm.end())
                if f.kind == "number":
                    self._add(rel, (i, mm.start()), f"{key} (a number in it)", toks | mw, mw, f.value, None, -1)

    # ------------------------------------------------------------ reading the files

    def _add(self, file, path, key, toks, leaf, value, time, parent):
        self._file.append(file)
        self._path.append(path)
        self._key.append(key)
        self._value.append(value)
        self._time.append(time)
        self._tokens.append(frozenset(toks))
        self._leaf.append(frozenset(leaf))
        self._parent.append(parent)
        return len(self._value) - 1

    def _dict(self, file, obj, path, key, toks, leaf):
        pid = len(self.parents)
        self.parents.append({"file": file, "path": path, "key": key, "keys": {}, "roles": {}})
        axes = {str(k): v for k, v in obj.items() if str(k).endswith(AXIS_SUFFIX) and isinstance(v, list) and v
                and all(is_num(x) for x in v)}
        # an interval's parts are named by the interval's own key (log_loss for log_loss.estimate)
        parts = {k for names in ROLE_SETS if sum(is_num(obj.get(k)) for k in names) >= 2 for k in names}
        for k, v in obj.items():
            k = str(k)
            kw = frozenset(words(k))
            cp, ck, ct = path + (k,), f"{key}.{k}" if key else k, toks | kw
            if is_num(v):
                e = self._add(file, cp, ck, ct, (leaf | kw) if k in parts else kw, float(v), None, pid)
                if k in parts:
                    self.parents[pid]["keys"][k] = e
            elif isinstance(v, str):
                self._string(file, obj, k, v, cp, ck, ct, kw)
            elif isinstance(v, dict):
                self._dict(file, v, cp, ck, ct, kw)
            elif isinstance(v, list):
                axis = next(((ak, av) for ak, av in axes.items() if ak != k and len(av) == len(v)), None)
                self._list(file, v, cp, ck, ct, kw, axis)

    def _list(self, file, obj, path, key, toks, leaf, axis):
        size = frozenset({"length", "count", "number"})
        self._add(file, path + ("(its length)",), f"{key} (its length)", toks | size, size | leaf, float(len(obj)), None, -1)
        for i, v in enumerate(obj):
            extra, label = set(), f"[{i}]"
            if isinstance(v, dict):
                named = [(f, v[f]) for f in ID_FIELDS if isinstance(v.get(f), str) and v.get(f)]
                for _, s in named:
                    extra |= words(s)
                for f in NUMERIC_ID_FIELDS:
                    if is_num(v.get(f)):
                        extra.add(canon(v[f]))
                if named:
                    label = f"[{i}] ({id_label(named)})"
            if axis is not None:
                extra.add(canon(axis[1][i]))
                extra |= words(axis[0])
            cp, ck, ct = path + (i,), f"{key}{label}", toks | extra
            if is_num(v):
                self._add(file, cp, ck, ct, leaf, float(v), None, -1)
            elif isinstance(v, str):
                self._string(file, None, None, v, cp, ck, ct, leaf)
            elif isinstance(v, dict):
                self._dict(file, v, cp, ck, ct, leaf)
            elif isinstance(v, list):
                self._list(file, v, cp, ck, ct, leaf, None)

    def _string(self, file, owner, k, text, path, key, toks, leaf):
        m = ISO_RE.fullmatch(text.strip())
        if m:
            self._add(file, path, key, toks, leaf, math.nan, m.group(1), -1)
        k = k or ""
        if owner is not None and (k in READING_KEYS or k.endswith("_reading")):
            kind = "result" if text in RESULT_WORDS else ("direction" if text in DIRECTIONS or text == "not computed"
                                                         else "text")
            side = {s: owner.get(s) if isinstance(owner.get(s), str) else None for s in ("system", "reference")}
            ranked = owner.get("ranked") if isinstance(owner.get("ranked"), bool) else None
            self.leaves.append({"file": file, "path": path, "key": key, "text": text, "kind": kind,
                                "tokens": frozenset(toks), "ranked": ranked, **side})
        if k == "rule" or k.endswith("_rule"):
            for mm in TOKEN_RE.finditer(mask(text)):
                f = Fig(mm.group(0), mm.start(), mm.end())
                if f.kind == "number":
                    self._add(file, path, f"{key} (a number in its text)", toks | {"rule"}, {"rule"}, f.value, None, -1)

    def _finish(self):
        n = self.n = len(self._value)
        self.values = np.array(self._value, dtype=float)
        self.is_time = np.array([t is not None for t in self._time], dtype=bool)
        self.is_number = ~self.is_time
        index = {f: i for i, f in enumerate(self.files)}
        self.file_idx = np.array([index[f] for f in self._file], dtype=np.int64)
        self.parent = np.array(self._parent, dtype=np.int64)
        self.role = np.zeros(n, dtype=np.int64)
        for p in self.parents:
            keys = p["keys"]
            for names in ROLE_SETS:
                if all(k in keys for k in names):
                    p["roles"] = {ROLE_CODES[k]: keys[k] for k in names}
                    break
            else:
                for names in ROLE_SETS:
                    if any(k in keys for k in names):
                        p["roles"] = {ROLE_CODES[k]: keys[k] for k in names if k in keys}
                        break
            for code, e in p["roles"].items():
                self.role[e] = code
        df = collections.Counter(t for toks in self._tokens for t in toks)
        self.idf = {t: math.log(1.0 + n / c) for t, c in df.items()}
        post, leafpost = collections.defaultdict(list), collections.defaultdict(list)
        for i, toks in enumerate(self._tokens):
            for t in toks:
                post[t].append(i)
        for i, toks in enumerate(self._leaf):
            for t in toks:
                if t in self.idf:
                    leafpost[t].append(i)
        self.post = {t: np.array(v, dtype=np.int64) for t, v in post.items()}
        self.leafpost = {t: np.array(v, dtype=np.int64) for t, v in leafpost.items()}
        self.tau = W_NEAR * math.log(1.0 + n / TAU_DF) if n else math.inf
        self.times = collections.defaultdict(list)
        for i, t in enumerate(self._time):
            if t:
                self.times[t].append(i)
                self.times[t[:5]].append(i)
        self.groups = {}
        for codes in ((1, 2, 3), (2, 3)):
            pids = [i for i, p in enumerate(self.parents) if all(c in p["roles"] for c in codes)]
            self.groups[codes] = (np.array(pids, dtype=np.int64),
                                  np.array([[self.parents[i]["roles"][c] for c in codes] for i in pids], dtype=np.int64)
                                  .reshape(len(pids), len(codes)))
        nl = len(self.leaves)
        ldf = collections.Counter(t for leaf in self.leaves for t in leaf["tokens"])
        self.leaf_idf = {t: math.log(1.0 + nl / c) for t, c in ldf.items()}
        self.tau_leaf = W_NEAR * math.log(1.0 + nl / TAU_DF_READINGS) if nl else math.inf
        self.leaf_under = collections.defaultdict(list)
        for i, leaf in enumerate(self.leaves):
            for k in range(1, len(leaf["path"])):
                self.leaf_under[(leaf["file"], leaf["path"][:k])].append(i)

    # ------------------------------------------------------------ looking things up

    def value_of(self, e):
        return self._time[e] if self._time[e] is not None else self._value[e]

    def where(self, e):
        return f"{self._file[e]} {self._key[e]}"

    def info(self, e, f):
        v = self.value_of(e)
        return {"file": self._file[e], "key": self._key[e], "value": v if isinstance(v, str) else float(v),
                "printed": printed_like(v, f), "_entry": int(e)}

    def score(self, ctx):
        """How well every source fits a context (a dict of words and their weights)."""
        s = np.zeros(self.n)
        for t, w in ctx.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            s[self.post[t]] += w * idf
            lp = self.leafpost.get(t)
            if lp is not None:
                s[lp] += w * idf * (LEAF_BONUS - 1.0)
        return s

    def scope_mask(self, scope):
        if scope is None:
            return np.ones(self.n, dtype=bool)
        return np.isin(self.file_idx, [i for i, f in enumerate(self.files) if f in scope])

    def matches(self, f, mask_):
        """The sources (of the number's kind, within the mask) that print as the number."""
        if f.kind == "time":
            hit = np.zeros(self.n, dtype=bool)
            hit[self.times.get(f.text, [])] = True
            return np.flatnonzero(hit & mask_)
        tol = 0.5 * 10.0 ** (-f.decimals) * (1.0 + 1e-9) + 1e-12 * max(1.0, abs(f.value))
        with np.errstate(invalid="ignore"):
            hit = np.abs(self.values - f.value) <= tol
            if f.percent:
                hit |= np.abs(self.values * 100.0 - f.value) <= tol
        return np.flatnonzero(hit & self.is_number & mask_)

    def prints_as(self, f, e):
        if f.kind == "time":
            t = self._time[e]
            return t is not None and (t == f.text or t[:5] == f.text)
        v = self._value[e]
        if self._time[e] is not None or not math.isfinite(v):
            return False
        tol = 0.5 * 10.0 ** (-f.decimals) * (1.0 + 1e-9) + 1e-12 * max(1.0, abs(f.value))
        return abs(v - f.value) <= tol or (f.percent and abs(100.0 * v - f.value) <= tol)

    def cited(self, text):
        """(the session 9 results files a text names, or None if it names none; the other .json files it names)."""
        names = set(JSON_RE.findall(text)) | set(LOG_RE.findall(text)) | set(OTHER_RE.findall(text))
        hits, other = set(), []
        for nm in sorted(names):
            found = [rel for rel in self.expanded if nm == rel or nm.endswith("fiorillo_v0_5/" + rel)
                     or ("/" not in nm and nm == Path(rel).name)]
            if found:
                hits.update(found)
            else:
                other.append(nm)
        return (hits if hits else None), other


# ---------------------------------------------------------------- matching a number

def outcome(check, src=None, e=None, f=None, binding="", note=""):
    o = {"check": check, "binding": binding, "note": note, "source": None, "expected": None}
    if e is not None:
        o["source" if check == "PASS" else "expected"] = src.info(e, f)
    return o


def accept(src, f, e, binding):
    """PASS, unless the number's print form is not the documents' (no thousands separators, or -0.0000)."""
    if f.kind == "number" and not f.separators_ok:
        return outcome("MISMATCH", src, e, f, binding, "its value is this source's, but it is printed without thousands "
                       f"separators (the documents print {printed_like(src.value_of(e), f)})")
    if f.kind == "number" and f.negative and f.value == 0:
        return outcome("MISMATCH", src, e, f, binding, "its value is this source's, but it is printed as a negative zero "
                       f"(the documents print {printed_like(src.value_of(e), f)})")
    return outcome("PASS", src, e, f, binding)


def named(src, e, ctx):
    """Whether a word of a source's own last key is near the number (in its clause, or its table cell's row or column)."""
    return any(ctx.get(t, 0.0) >= W_CLAUSE for t in src._leaf[e])


def match_single(src, f, s, scope, ctx):
    """One number: PASS, MISMATCH or NO SOURCE, as the module's docstring sets out."""
    kmask = scope & (src.is_time if f.kind == "time" else src.is_number)
    pool = np.flatnonzero(kmask)
    if not len(pool):
        return outcome("NO SOURCE", note="no source of its kind is in the files searched")
    best = int(pool[np.argmax(s[pool])])
    top = float(s[best])
    clear = top >= src.tau and named(src, best, ctx)
    cand = src.matches(f, kmask)
    if len(cand):
        j = int(cand[np.argmax(s[cand])])
        if s[j] > 0 and (s[j] >= RHO * top or named(src, j, ctx)):
            return accept(src, f, j, "value and context")
        if clear:
            how = "which its context names less clearly" if s[j] > 0 else "whose key shares no word with its context"
            return outcome("MISMATCH", src, best, f, "context", f"it equals {src.where(j)}, {how}")
        if s[j] >= 0.5 * src.tau:
            o = accept(src, f, j, "value, with little context")
            if o["check"] == "PASS":
                o["note"] = f"its context names {src.where(best)} more closely"
            return o
        return outcome("NO SOURCE", note=f"{src.where(j)} holds its value, but its context hardly names that key")
    if clear:
        return outcome("MISMATCH", src, best, f, "context")
    return outcome("NO SOURCE")


def match_group(src, kind, figs, s, scope, ctx):
    """An interval or a range matched as one figure; None when it is not (a range is then matched number by number)."""
    codes = (1, 2, 3) if kind == "interval" else (2, 3)
    per = []
    for f, c in zip(figs, codes):
        ids = src.matches(f, scope & src.is_number)
        ids = ids[src.role[ids] == c]
        per.append({int(p) for p in src.parent[ids] if p >= 0})
    pids, ents = src.groups[codes]
    pos = {int(p): k for k, p in enumerate(pids)}

    def fit(p):
        return float(s[ents[pos[p]]].max()) if p in pos else 0.0

    full = {p for p in set.intersection(*per) if p in pos}
    if full:
        p = max(full, key=lambda q: (fit(q), -q))
        if fit(p) > 0 or kind == "interval":
            how = kind if fit(p) > 0 else f"{kind}, by its values alone"
            return [(f, accept(src, f, int(ents[pos[p]][k]), how)) for k, f in enumerate(figs)]
        return None
    if kind == "interval":
        counts = collections.Counter(q for st in per for q in st)
        two = [q for q, c in counts.items() if c == 2 and q in pos]
        if two:
            p = max(two, key=lambda q: (fit(q), -q))
            out = []
            for k, (f, st) in enumerate(zip(figs, per)):
                e = int(ents[pos[p]][k])
                out.append((f, accept(src, f, e, "interval") if p in st else
                            outcome("MISMATCH", src, e, f, "interval", "the interval's other two numbers are this source's")))
            return out
    keep = scope[ents[:, 0]] if len(pids) else np.zeros(0, dtype=bool)
    if keep.any():
        fits = s[ents].max(axis=1)
        fits[~keep] = -1.0
        k = int(np.argmax(fits))
        if fits[k] >= src.tau and named(src, int(ents[k][0]), ctx):
            out = []
            for j, f in enumerate(figs):
                e = int(ents[k][j])
                out.append((f, accept(src, f, e, kind) if src.prints_as(f, e) else outcome("MISMATCH", src, e, f, "context")))
            return out
    return None


def match(src, kind, figs, ctx, scope):
    """[(figure, outcome, part)] for one interval, range or single number."""
    s = src.score(ctx)
    smask = src.scope_mask(scope)
    if kind in PARTS:
        got = match_group(src, kind, figs, s, smask, ctx)
        if got is not None:
            return [(f, o, part) for (f, o), part in zip(got, PARTS[kind])]
        if kind == "interval":
            return [(f, outcome("NO SOURCE", note="no source holds this interval"), part) for f, part in zip(figs, PARTS[kind])]
    return [(f, match_single(src, f, s, smask, ctx), None) for f in figs]


# ---------------------------------------------------------------- readings

def same_words(a, b):
    def norm(t):
        return " ".join(str(t).lower().replace("\u201c", "").replace("\u201d", "").replace('"', "").split()).strip(" .;:,")
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def claim_direction(text, leaf):
    """What a reading of a comparison says: none, which side is better, or higher or lower (None if it says none of
    these)."""
    t = " ".join(text.lower().split())
    if "no difference was detected" in t:
        return "none"
    for phrase in ("not computed", "not ranked") + DIRECTIONS[1:]:
        if phrase in t:
            return phrase
    m = re.search(r"^(.*?)\b(better|worse)\b", t)
    if m:
        who, sysw, refw = words(m.group(1)), words(leaf.get("system") or ""), words(leaf.get("reference") or "")
        is_ref = bool(who & (refw - sysw))
        if m.group(2) == "better":
            return "reference better" if is_ref else "system better"
        return "system better" if is_ref else "reference better"
    if re.search(r"\blower\b", t):
        return "lower"
    if re.search(r"\bhigher\b", t):
        return "higher"
    return None


def reading_agrees(ckind, text, leaf):
    src_text = leaf["text"]
    if ckind == "result":
        return text.strip() == src_text
    if leaf["kind"] != "direction":
        return same_words(text, src_text)
    d = claim_direction(text, leaf)
    if d == "none":
        return src_text == "no difference was detected"
    if d == "not ranked":
        return leaf.get("ranked") is False
    if d in ("not computed",) + DIRECTIONS:
        return d == src_text
    if d == "system better":
        return src_text.endswith("(better)")
    if d == "reference better":
        return src_text.endswith("(worse)")
    if d in ("lower", "higher"):
        return src_text.startswith(d)
    return False


def bind_reading(src, ckind, entries, ctx):
    """(the reading a claim is checked against, how it was found), or (None, "")."""
    allowed = [i for i, leaf in enumerate(src.leaves) if (leaf["kind"] == "result") == (ckind == "result")]
    if not allowed:
        return None, ""
    aset = set(allowed)

    def fit(i):
        toks = src.leaves[i]["tokens"]
        return sum(w * src.leaf_idf.get(t, 0.0) for t, w in ctx.items() if t in toks)

    best, chosen = None, None
    for e in entries:
        file, path = src._file[e], src._path[e]
        for k in range(len(path) - 1, 0, -1):
            ids = [i for i in src.leaf_under.get((file, path[:k]), ()) if i in aset]
            if ids:
                below = set(path[k:])
                for i in ids:
                    key = (k, len(below & set(src.leaves[i]["path"][k:])), fit(i), -i)
                    if best is None or key > best:
                        best, chosen = key, i
                break
    if chosen is not None:
        return chosen, "the reading above the sources its numbers matched"
    i = max(allowed, key=lambda i: (fit(i), -i))
    if fit(i) >= src.tau_leaf:
        return i, "the reading its context names best"
    return None, ""


def claim_row(src, where, line_no, ckind, text, start, entries, ctx, quote=False):
    """A reading's row, or None for a quoted text that is not a reading of the sources."""
    i, how = bind_reading(src, ckind, entries, ctx)
    if i is not None and quote and how != "the reading above the sources its numbers matched" \
            and not reading_agrees(ckind, text, src.leaves[i]):
        i = None
    if i is None:
        same = [j for j, leaf in enumerate(src.leaves) if leaf["kind"] != "result" and same_words(text, leaf["text"])]
        if quote and not same:
            return None
        if quote:
            i, how = same[0], "a reading with the same words (its context names none)"
    row = {"where": where, "line": line_no, "start": start, "as_written": text, "kind": ckind, "binding": how,
           "source": None, "check": "NO SOURCE", "note": "" if i is not None else "no reading of the sources was found for it"}
    if i is not None:
        leaf = src.leaves[i]
        row["source"] = {"file": leaf["file"], "key": leaf["key"], "text": leaf["text"], "_leaf": i}
        row["check"] = "PASS" if reading_agrees(ckind, text, leaf) else "MISMATCH"
    return row


# ---------------------------------------------------------------- the document

def number_row(where, line_no, f, o, unit, context, derived, foreign, outside):
    """One number's row. outside: its sentence (or table) names .json files, none of them a source."""
    note = o["note"]
    if (derived or outside) and o["check"] == "MISMATCH" and o["binding"] == "context":
        e = o["expected"]
        o = dict(o, check="NO SOURCE", expected=None)
        note = "; ".join(x for x in (f"its context names {e['file']} {e['key']}, which prints as {e['printed']}", note) if x)
    if derived and o["check"] != "PASS":
        note = "; ".join(x for x in ("the document labels it derived, and the check does not compute derived figures",
                                     note) if x)
    if foreign and o["check"] == "NO SOURCE":
        names = f"{', '.join(foreign)}, which {'is' if len(foreign) == 1 else 'are'} not among the sources"
        note = "; ".join(x for x in (note, f"its sentence names {names}") if x)
    attributed = None  # a number with no source that the document itself attributes: labeled derived, or its sentence names
    if o["check"] == "NO SOURCE":  # only files that are not sources (docs/log.md, an earlier session's results and the like)
        attributed = "derived" if derived else ("outside" if foreign and outside else None)
    return {"where": where, "line": line_no, "start": f.start, "as_written": f.text, "kind": fig_kind(f), "unit": unit,
            "context": context, "check": o["check"], "source": o["source"], "expected": o["expected"],
            "binding": o["binding"], "note": note, "value": f.value, "decimals": f.decimals, "percent": f.percent,
            "dollar": f.dollar, "negative": f.negative, "attributed": attributed}


def entries_of(outs):
    return [o["source"]["_entry"] if o["source"] else o["expected"]["_entry"] for _, o, _ in outs
            if o["source"] or o["expected"]]


def analyze_prose(src, line, line_no, section, sub):
    """The numbers and readings of one line of prose (or a heading)."""
    figs, masked = find_figures(line)
    if not figs and not RESULT_RE.search(line) and not QUOTE_RE.search(line):
        return [], []
    units = units_of(figs, masked)
    order = {id(f): k for k, f in enumerate(figs, start=1)}
    sentences = spans(line, SENTENCE_GAP)
    para = {t for t in words(line) if "_" in t}
    head = words(section) | words(sub)
    label = section if len(section) <= 40 else section[:37] + "..."
    numbers, done = [], []
    for ui, (kind, fs) in enumerate(units):
        a, b = fs[0].start, fs[-1].end
        s0, s1 = containing(sentences, a)
        c0, c1 = containing(spans(line[s0:s1], CLAUSE_GAP, s0), a)
        prev = [u[1][-1].end for u in units[:ui] if c0 <= u[1][0].start < c1]
        nxt = [u[1][0].start for u in units[ui + 1:] if c0 <= u[1][0].start < c1]
        p_end, n_start = (prev[-1] if prev else c0), (nxt[0] if nxt else c1)
        ctx = {}
        add(ctx, words(line[p_end:a]), W_NEAR)
        add(ctx, words(line[c0:p_end]), W_CLAUSE)
        add(ctx, words(line[b:n_start]), W_CLAUSE)
        add(ctx, words(line[s0:c0] + " " + line[c1:s1]), W_SENTENCE)
        add(ctx, para, W_PARAGRAPH)
        add(ctx, head, W_HEADING)
        if any(f.dollar for f in fs):
            add(ctx, {"usd"}, W_NEAR)
        sentence = line[s0:s1]
        scope, foreign = src.cited(sentence)
        outs = match(src, kind, fs, ctx, scope)
        for f, o, part in outs:
            where = f"line {line_no} ({label}), number {order[id(f)]}" + (f", {part}" if part else "")
            numbers.append(number_row(where, line_no, f, o, kind, "prose", "derived" in sentence.lower(), foreign,
                                      scope is None and bool(foreign)))
        done.append((a, kind, entries_of(outs)))
    readings, k = [], 0
    for s0, s1 in sentences:
        sctx = {}
        add(sctx, words(line[s0:s1]), W_CLAUSE)
        add(sctx, head, W_HEADING)
        here = [u for u in done if s0 <= u[0] < s1]
        every = [e for u in here for e in u[2]]
        claims, taken = [], []
        for m in DIRECTION_RE.finditer(line, s0, s1):
            groups = [u for u in here if u[1] in PARTS and u[0] < m.start()]
            if groups:
                claims.append(("reading", m.group(0), m.start(), groups[-1][2], False))
                taken.append((m.start(), m.end()))
        for m in RESULT_RE.finditer(line, s0, s1):
            claims.append(("result", m.group(0), m.start(), every, False))
        for m in QUOTE_RE.finditer(line, s0, s1):
            g = 1 if m.group(1) is not None else 2
            if not any(a < m.end(g) and m.start(g) < b for a, b in taken):
                claims.append(("reading", m.group(g), m.start(g), every, True))
        for ckind, text, start, ents, quote in sorted(claims, key=lambda c: c[2]):
            row = claim_row(src, "", line_no, ckind, text, start, ents, sctx, quote)
            if row is not None:
                k += 1
                row["where"] = f"line {line_no} ({label}), reading {k}"
                readings.append(row)
    return numbers, readings


def analyze_row(src, line, line_no, header, title, section):
    """The numbers and readings of one row of a table."""
    cs = cells(line)
    figs, masked = find_figures(line)
    starts = [off for _, off in cs]

    def cell_of(pos):
        k = 0
        for j, off in enumerate(starts):
            if off <= pos:
                k = j
        return k

    def head_of(k):
        return header[k] if k < len(header) else ""

    figs = [f for f in figs if not LABEL_HEADER.fullmatch(head_of(cell_of(f.start)))]
    units = units_of(figs, masked)
    with_figs = {cell_of(f.start) for f in figs}
    claim_cells = {k for k, (c, _) in enumerate(cs) if c in RESULT_WORDS
                   or (re.search(r"\breading\b", head_of(k), re.I) and k not in with_figs and c)}
    labels = [c for k, (c, _) in enumerate(cs) if c and k not in with_figs and k not in claim_cells]
    m = re.match(r"Table (\d+)\.", title)
    tname = f"Table {m.group(1)}" if m else (title if len(title) <= 40 else title[:37] + "...")
    first = [canon(f.value) for f in figs if f.kind == "number" and cell_of(f.start) == 0]
    whole = {canon(f.value) for f in figs if f.kind == "number" and f.decimals == 0}  # the row's counts and rates
    scope, foreign = src.cited(title + " " + line)
    derived = "derived" in (title + " " + line).lower()
    numbers, every = [], []
    for kind, fs in units:
        k = cell_of(fs[0].start)
        text, off = cs[k] if k < len(cs) else ("", 0)
        own = list(text)
        for f in figs:
            if cell_of(f.start) == k:
                for p in range(f.start - off, min(f.end - off, len(own))):
                    if p >= 0:
                        own[p] = " "
        ctx = {}
        add(ctx, words(head_of(k)), W_NEAR)
        for lab in labels:
            add(ctx, words(lab), W_NEAR)
        add(ctx, words("".join(own)), W_NEAR)
        if k != 0:
            add(ctx, first, W_NEAR - 1.0)
        add(ctx, whole - {canon(f.value) for f in fs if f.kind == "number"}, W_SENTENCE)
        add(ctx, words(title), W_TITLE)
        add(ctx, words(section), W_HEADING)
        if any(f.dollar for f in fs):
            add(ctx, {"usd"}, W_NEAR)
        if re.search(r"percent|%", mask(head_of(k)), re.I):
            for f in fs:
                if f.kind == "number":
                    f.percent = True
        outs = match(src, kind, fs, ctx, scope)
        for f, o, part in outs:
            where = f"line {line_no}, {tname}, {', '.join(labels) or 'row'}, {head_of(k) or 'column ' + str(k + 1)}" + \
                    (f", {part}" if part else "")
            numbers.append(number_row(where, line_no, f, o, kind, "table", derived, foreign, scope is None and bool(foreign)))
        every += entries_of(outs)
    readings = []
    for k in sorted(claim_cells):
        text, off = cs[k]
        ctx = {}
        for lab in labels:
            add(ctx, words(lab), W_NEAR)
        add(ctx, words(head_of(k)), W_NEAR)
        add(ctx, words(title), W_TITLE)
        add(ctx, words(section), W_HEADING)
        ckind = "result" if text in RESULT_WORDS else "reading"
        where = f"line {line_no}, {tname}, {', '.join(labels) or 'row'}, {head_of(k) or 'column ' + str(k + 1)}"
        readings.append(claim_row(src, where, line_no, ckind, text, off, every, ctx))
    return numbers, readings


def section_ranges(lines, prefixes):
    heads = [(i, line[3:].strip()) for i, line in enumerate(lines) if line.startswith("## ")]
    out = []
    for k, (i, name) in enumerate(heads):
        if any(name.lower().startswith(p.lower()) for p in prefixes):
            out.append((name, i, heads[k + 1][0] if k + 1 < len(heads) else len(lines)))
    return out


class Check:
    """One pass over a document's checked sections: every number and every reading, one row each, in document order."""

    def __init__(self, src, text, sections=SECTIONS):
        self.src, self.numbers, self.readings = src, [], []
        lines = text.split("\n")
        found = section_ranges(lines, sections)
        if not found:
            stop(f"the document has no section whose heading starts with {' or '.join(repr(s) for s in sections)}")
        self.sections = [name for name, _, _ in found]
        for name, start, end in found:
            self.section(lines, name, start, end)

    def cached(self, key, fn, *args):
        hit = self.src.cache.get(key)
        if hit is None:
            hit = fn(self.src, *args)
            self.src.cache[key] = hit
        self.numbers += hit[0]
        self.readings += hit[1]

    def section(self, lines, name, start, end):
        sub, in_code, i = "", False, start
        while i < end:
            line = lines[i]
            if line.strip().startswith("```"):
                in_code = not in_code
                i += 1
                continue
            if in_code or not line.strip() or line.lstrip().startswith("<!--"):
                i += 1
                continue
            if line.lstrip().startswith("|"):
                j = i
                while j < end and lines[j].lstrip().startswith("|"):
                    j += 1
                block, title = lines[i:j], sub or name
                header = [c for c, _ in cells(block[0])]
                body = 2 if len(block) > 1 and all(re.fullmatch(r":?-{3,}:?", c) for c, _ in cells(block[1]) if c) else 1
                for r in range(body, len(block)):
                    self.cached(("row", i + r + 1, block[r], block[0], title, name), analyze_row, block[r], i + r + 1,
                                header, title, name)
                i = j
                continue
            if line.startswith("#") and i > start:
                sub = line.lstrip("#").strip()
            self.cached(("prose", i + 1, line, name, sub), analyze_prose, line, i + 1, name, sub)
            i += 1

    def counts(self):
        out = {}
        for what, rows in (("numbers", self.numbers), ("readings", self.readings)):
            c = collections.Counter(r["check"] for r in rows)
            out[what] = {k: c.get(k, 0) for k in ("PASS", "MISMATCH", "NO SOURCE")}
        a = collections.Counter(r.get("attributed") for r in self.numbers if r["check"] == "NO SOURCE")
        out["numbers_no_source_attributed"] = {"derived": a.get("derived", 0), "outside": a.get("outside", 0)}
        return out


def unexplained(counts):
    """What keeps the check from complete: mismatches, readings that fail, and numbers with no source that the document
    does not attribute (a number it labels derived, or whose sentence names only files that are not sources, is attributed)."""
    n, r, a = counts["numbers"], counts["readings"], counts["numbers_no_source_attributed"]
    return n["MISMATCH"] + (n["NO SOURCE"] - a["derived"] - a["outside"]) + r["MISMATCH"] + r["NO SOURCE"]


# ---------------------------------------------------------------- the self-test

def alterations(f):
    """The number with one digit changed, nearest first: its last digit up and down by 1, 2, 3 and so on, then the digit
    before it."""
    t = f.text
    body_end = len(t)
    if f.sci:
        body_end = re.search(r"[eE]", t).start()
    digits = [i for i, ch in enumerate(t[:body_end]) if ch.isdigit()]
    out = []
    for pos in reversed(digits):
        d = int(t[pos])
        for delta in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
            nd = d + delta
            if not 0 <= nd <= 9:
                continue
            new = t[:pos] + str(nd) + t[pos + 1:]
            if f.kind == "time":
                parts = [int(x) for x in new.split(":")]
                if parts[0] > 23 or any(x > 59 for x in parts[1:]):
                    continue
            elif nd == 0 and pos == digits[0] and len(re.match(r"[\d,]+", t[pos:body_end]).group(0).replace(",", "")) > 1:
                continue
            out.append(new)
    return out


def plant_numbers(src, check):
    """Numbers that pass, of every kind and spread over the lines, each with a changed text that no source holds."""
    cats = collections.OrderedDict((k, []) for k in ("table interval", "prose interval", "table", "prose", "count",
                                                     "percentage", "dollars", "time", "negative"))
    for r in check.numbers:
        if r["check"] != "PASS":
            continue
        if r["kind"] in ("time", "dollars", "percentage"):
            cats[r["kind"]].append(r)
        elif r["negative"]:
            cats["negative"].append(r)
        elif r["decimals"] == 0 and abs(r["value"]) >= 1000:
            cats["count"].append(r)
        elif r["unit"] in PARTS:
            cats[f"{r['context']} interval"].append(r)
        else:
            cats[r["context"]].append(r)
    everything = np.ones(src.n, dtype=bool)
    chosen, used = [], set()
    while len(chosen) < PLANTS and any(cats.values()):
        for name, rows in cats.items():
            if len(chosen) >= PLANTS or not rows:
                continue
            r = max(rows, key=lambda x: (min((abs(x["line"] - u) for u in used), default=10 ** 9), -x["line"]))
            rows.remove(r)
            f = Fig(r["as_written"], r["start"], r["start"] + len(r["as_written"]))
            f.percent = r["percent"]
            for new in alterations(f):
                g = Fig(new, f.start, f.start + len(new))
                g.percent = f.percent
                if not len(src.matches(g, everything)):
                    chosen.append({"what": "number", "category": name, "where": r["where"], "line": r["line"],
                                   "start": r["start"], "old": r["as_written"], "new": new})
                    used.add(r["line"])
                    break
    return chosen


def plant_readings(check):
    out = []
    for want in ("reading", "result"):
        for r in check.readings:
            if r["check"] != "PASS" or r["kind"] != want or len(out) >= READING_PLANTS:
                continue
            src_text = r["source"]["text"]
            if want == "result":
                new = "FAIL" if r["as_written"] == "PASS" else "PASS"
            elif src_text in DIRECTIONS or src_text == "not computed":
                new = "higher (better)" if src_text == "no difference was detected" else "no difference was detected"
            else:
                new = "not settled" if not same_words("not settled", src_text) else "partly"
            out.append({"what": "reading", "category": want, "where": r["where"], "line": r["line"], "start": r["start"],
                        "old": r["as_written"], "new": new})
            break
    return out


def self_test(src, text, sections=SECTIONS):
    """Plants each error in a copy of the document and stops unless the check reports it."""
    base = Check(src, text, sections)
    plants = plant_numbers(src, base) + plant_readings(base)
    if not plants:
        stop("self-test: no number or reading of the document passes, so no error can be planted; nothing was written")
    lines = text.split("\n")
    for p in plants:
        line = lines[p["line"] - 1]
        if line[p["start"]:p["start"] + len(p["old"])] != p["old"]:
            stop(f"self-test: the planted error at {p['where']} cannot be placed")
        copy = list(lines)
        copy[p["line"] - 1] = line[:p["start"]] + p["new"] + line[p["start"] + len(p["old"]):]
        c = Check(src, "\n".join(copy), sections)
        rows = c.numbers if p["what"] == "number" else c.readings
        hit = [r for r in rows if r["line"] == p["line"] and r["start"] == p["start"]]
        p["reported_as"] = hit[0]["check"] if hit else "not found"
        if not hit or hit[0]["check"] == "PASS":
            stop(f"self-test: the planted error at {p['where']} ({p['old']!r} made {p['new']!r}) was not reported; "
                 "nothing was written")
    return plants


# ---------------------------------------------------------------- the outputs

def rel(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def public(row):
    """A row without the check's own bookkeeping (keys starting with _)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        out[k] = v
    return out


def esc(x):
    return str(x).replace("|", "/").replace("\n", " ")


GUARD_LIMIT = 1024 * 1024  # src/tools/check_no_data.py: a file under results/ may be at most 1 MB


def as_table(rows):
    """Rows (dictionaries) as one table: the field names once, then one list a row. A field whose values are
    dictionaries (a number's source and expected value, a reading's source) is written as a list in the order its
    dict_fields entry gives, a key a dictionary lacks as null. Step 16: the first full check wrote figure_sources.json as
    a list of dictionaries, 1,857,407 bytes, over the data guard's limit for a file under results/."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    dict_fields = {}
    for k in fields:
        keys = list(dict.fromkeys(kk for r in rows if isinstance(r.get(k), dict) for kk in r[k]))
        if keys:
            dict_fields[k] = keys

    def cell(k, v):
        return [v.get(kk) for kk in dict_fields[k]] if k in dict_fields and isinstance(v, dict) else v
    return {"fields": fields, "dict_fields": dict_fields, "rows": [[cell(k, r.get(k)) for k in fields] for r in rows]}


def from_table(table):
    """A table of figure_sources.json (its numbers or its readings) as dictionaries again: as_table's inverse."""
    out = []
    for row in table["rows"]:
        r = dict(zip(table["fields"], row))
        for k, keys in table["dict_fields"].items():
            if isinstance(r.get(k), list):
                r[k] = dict(zip(keys, r[k]))
        out.append(r)
    return out


def dump_sources(data):
    """figure_sources.json's text: the fields before the tables indented as before, then the numbers' and the readings'
    tables with one row a line."""
    head = {k: v for k, v in data.items() if k not in ("numbers", "readings")}
    text = json.dumps(head, indent=1)[:-2]
    for key in ("numbers", "readings"):
        t = data[key]
        rows = ",\n".join(json.dumps(r, separators=(",", ":")) for r in t["rows"])
        text += (f',\n "{key}": {{"fields": {json.dumps(t["fields"])}, "dict_fields": {json.dumps(t["dict_fields"])}, '
                 f'"rows": [\n{rows}\n]}}')
    return text + "\n}\n"


NOT_COUNTED = ("anything in code type or holding an underscore or a slash (system names, keys, file names); version names "
               "and numbers (v0.1 to v0.5, 0.30.0); model names (Qwen3-4B-Base, Gemma 4 31B-it, BioGPT-Large, TabICLv2 and "
               "the like); idea numbers (I-01 and so on), G1 to G6, H1 to H4 and K1 to K10; table, figure, correction, "
               "criterion, session, step, seed, decision and round numbers; dates, time stamps, the years 2010 to 2029 and "
               "eight-digit seeds; ordinals; the phrases \"95 percent\", \"30 days\", \"80 percent chance\", \"prints as 0\" "
               "and \"per 1,000\"; readmission's levels 0 to 2; sizes such as 4B; and the numbers of numbered lists")


def write_outputs(root, doc, out_doc, out_sources, text, src, check, plants):
    now = dt.datetime.now(dt.timezone.utc)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    counts = check.counts()
    numbers = [public(r) for r in check.numbers]
    readings = [public(r) for r in check.readings]
    data = {"plan": PLAN, "source": "src/eval/check_fiorillo_v0_5_figures.py", "document": rel(doc, root),
            "document_sha256": sha, "time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "sections": check.sections,
            "sources": {"folder": rel(src.results, root),
                        "files": {f: {"sha256": src.sha[f], "module": src.modules[f]} for f in src.files},
                        "not_found": src.missing},
            "settings": {"weights": {"near": W_NEAR, "clause": W_CLAUSE, "sentence": W_SENTENCE, "paragraph": W_PARAGRAPH,
                                     "heading": W_HEADING, "title": W_TITLE}, "leaf_bonus": LEAF_BONUS, "rho": RHO,
                         "tau": src.tau, "tau_df": TAU_DF, "tau_readings": src.tau_leaf,
                         "tau_df_readings": TAU_DF_READINGS, "sources_indexed": src.n, "readings_indexed": len(src.leaves)},
            "self_test": plants, "counts": counts, "numbers": as_table(numbers), "readings": as_table(readings)}
    Path(out_sources).parent.mkdir(parents=True, exist_ok=True)
    sources_text = dump_sources(data)
    Path(out_sources).write_text(sources_text, encoding="utf-8")
    size = len(sources_text.encode("utf-8"))
    if size > GUARD_LIMIT:
        print(f"note: {out_sources} is {size:,} bytes, over the data guard's limit of {GUARD_LIMIT:,} bytes for a file "
              "under results/", file=sys.stderr)

    n, r, att = counts["numbers"], counts["readings"], counts["numbers_no_source_attributed"]
    bad = unexplained(counts)
    status = ("complete" + (f" (with {att['outside']:,} numbers whose sources the document names outside the files checked "
                            f"and {att['derived']:,} it labels derived, listed under Numbers with no source)"
                            if att["outside"] or att["derived"] else "")) if not bad else (
        f"draft ({n['MISMATCH']:,} numbers do not match their sources and {n['NO SOURCE']:,} have none, "
        f"{att['outside'] + att['derived']:,} of them attributed by the document; "
        f"{r['MISMATCH']:,} readings do not match and {r['NO SOURCE']:,} have none)")
    planted = "; ".join(f"{p['where']}: {p['old']} made {p['new']} ({p['reported_as']})" for p in plants)
    out = [
        "# Fiorillo v0.5: figure check",
        f"Study: {STUDY}   Prepared: {now.day} {now.strftime('%B %Y')}",
        f"Sources: {rel(doc, root)} (its sections {', '.join(check.sections)}); {rel(src.results, root)}/ "
        f"({', '.join(src.files)})" + (f"; not found there: {', '.join(src.missing)}" if src.missing else ""),
        f"Status: {status}",
        "",
        "Abbreviations: EI is Evidence Inference 2.0; ECE is expected calibration error; UTC is Coordinated Universal Time; "
        "SHA-256 is Secure Hash Algorithm 256; JSON is JavaScript Object Notation; IDF is inverse document frequency.",
        "",
        "## How the check was made",
        "",
        f"src/eval/check_fiorillo_v0_5_figures.py read the sections {', '.join(check.sections)} of {rel(doc, root)} as "
        f"written (SHA-256 of the file checked: {sha}) and found every number in them. It looked for each number's source "
        f"among session 9's results files ({len(src.files)} found, listed above): every number in them under its file and "
        "key, every list's length, the time of day of every time stamp, every number written in a rule's text, and every "
        f"interval ({src.n:,} sources in all). A number matches a source value when the value, rounded to the number's "
        "printed decimals, is the number (they differ by at most half a unit in the last printed place); a percentage is "
        "compared with 100 times the value, or with the value where the source holds a percentage, and a time with the time "
        "of day of a time stamp. A number printed without thousands separators, or as -0.0000, does not match, since the "
        "documents never print one so. An interval (estimate, low to high) or a range (low to high) is matched as one "
        "figure to a source that holds its parts.",
        "",
        "Which source a number means was found from the words that the source's key shares with the number's context: in a "
        "table, its column's heading, its row's labels and the table's title; in prose, the words before it in its clause, "
        "the rest of its clause and sentence, the identifiers of its paragraph and its headings. Each shared word is "
        f"weighted by its nearness and by its rarity among the keys (IDF). A number passes when a source that fits its "
        f"context at least {RHO} times as well as the best fit prints as it; it does not match when none does and its "
        f"context names a source clearly (a fit of at least {src.tau:.2f}, what one word next to the number gives when "
        f"{TAU_DF} keys hold it), and the source its context names is then given; otherwise it has no source. A sentence or "
        "table that names results files of session 9 was searched in those files only.",
        "",
        "Readings (the cells of Reading columns, the readings of comparisons in prose after an interval, PASS and FAIL, and "
        "quoted readings) were checked against the reading above the sources that their row's or sentence's numbers "
        "matched, or else the reading their context names best; a comparison's reading by its meaning (no difference, which "
        "side is better, or higher or lower), as compare_session8.py reads the interval, and any other by its words.",
        "",
        f"Before the document was checked, {len(plants)} errors were planted in copies of it, one at a time, and each was "
        f"reported: {planted}.",
        "",
        "Limits: a changed number that equals another source value which its context names as clearly would pass; figures "
        "that the document labels derived are not computed here, so they show as having no source unless a source holds "
        "them; "
        "numbers from files that are not among the sources (docs/log.md, the console's pages except as pod_console.json "
        "records them, and earlier sessions' results) have no source here; and the zipped stage logs stop where package "
        "wrote the logs zip, before its own last lines.",
        "",
        f"Not counted as numbers, wherever they appear: {NOT_COUNTED}. Line numbers are the document's.",
        "",
        "## Result",
        "",
        f"{len(numbers):,} numbers: {n['PASS']:,} PASS, {n['MISMATCH']:,} MISMATCH and {n['NO SOURCE']:,} NO SOURCE. "
        f"{len(readings):,} readings: {r['PASS']:,} PASS, {r['MISMATCH']:,} MISMATCH and {r['NO SOURCE']:,} NO SOURCE.",
        "",
        "## Every number",
        "",
        "| # | Where | As written | Source (file and key) | Source value | Check |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(numbers, start=1):
        s = row["source"] or row["expected"]
        where = f"{s['file']} {s['key']}" if s else "none found"
        if row["check"] == "MISMATCH" and s:
            where = "its context names " + where
        out.append(f"| {i} | {esc(row['where'])} | {esc(row['as_written'])} | {esc(where)} | "
                   f"{esc(raw(s['value'])) if s else ''} | {row['check']} |")
    out += ["", "## Numbers that do not match", ""]
    bad_rows = [(i, row) for i, row in enumerate(numbers, start=1) if row["check"] == "MISMATCH"]
    if bad_rows:
        out += ["| # | Where | As written | The source its context names | As the documents would print it | Note |",
                "|---|---|---|---|---|---|"]
        for i, row in bad_rows:
            s = row["expected"] or row["source"]
            out.append(f"| {i} | {esc(row['where'])} | {esc(row['as_written'])} | {esc(s['file'] + ' ' + s['key'])} | "
                       f"{esc(s['printed'])} | {esc(row['note'])} |")
    else:
        out.append("None.")
    out += ["", "## Numbers with no source", ""]
    none_rows = [(i, row) for i, row in enumerate(numbers, start=1) if row["check"] == "NO SOURCE"]
    if none_rows:
        out += ["| # | Where | As written | Note |", "|---|---|---|---|"]
        for i, row in none_rows:
            out.append(f"| {i} | {esc(row['where'])} | {esc(row['as_written'])} | {esc(row['note'])} |")
    else:
        out.append("None.")
    out += ["", "## Every reading", ""]
    if readings:
        out += ["| # | Where | As written | Source (file and key) | Source text | Check |", "|---|---|---|---|---|---|"]
        for i, row in enumerate(readings, start=1):
            s = row["source"]
            out.append(f"| {i} | {esc(row['where'])} | {esc(row['as_written'])} | "
                       f"{esc(s['file'] + ' ' + s['key']) if s else 'none found'} | {esc(s['text']) if s else ''} | "
                       f"{row['check']} |")
    else:
        out.append("None.")
    Path(out_doc).parent.mkdir(parents=True, exist_ok=True)
    Path(out_doc).write_text("\n".join(out) + "\n", encoding="utf-8")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository (relative paths below are inside it)")
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out-doc", type=Path, default=OUT_DOC)
    parser.add_argument("--out-sources", type=Path, default=OUT_SOURCES)
    parser.add_argument("--sections", nargs="+", default=list(SECTIONS))
    args = parser.parse_args(argv)
    root = args.root
    doc, results, out_doc, out_sources = (p if p.is_absolute() else root / p
                                          for p in (args.doc, args.results, args.out_doc, args.out_sources))
    if not doc.exists():
        stop(f"{doc} does not exist")
    src = Sources(results)
    if not src.files:
        stop(f"none of the sources exists in {results}")
    text = doc.read_text(encoding="utf-8")
    plants = self_test(src, text, args.sections)
    check = Check(src, text, args.sections)
    counts = write_outputs(root, doc, out_doc, out_sources, text, src, check, plants)
    n, r = counts["numbers"], counts["readings"]
    print(f"self-test: {len(plants)} planted errors, each reported. Numbers: {n['PASS']:,} PASS, {n['MISMATCH']:,} "
          f"MISMATCH, {n['NO SOURCE']:,} NO SOURCE. Readings: {r['PASS']:,} PASS, {r['MISMATCH']:,} MISMATCH, "
          f"{r['NO SOURCE']:,} NO SOURCE. Wrote {rel(out_sources, root)} and {rel(out_doc, root)}")
    return 0 if not unexplained(counts) else 1


if __name__ == "__main__":
    sys.exit(main())

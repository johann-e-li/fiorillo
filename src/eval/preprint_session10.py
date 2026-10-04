"""The preprint's tables and figures (docs/prompt_session10.md, THE PREPRINT, step 7), from the results files only.

Subcommands:
  tables   docs/preprint/tables/<name>.tex: LaTeX fragments (booktabs), one for each table the outline names
           (docs/preprint/outline.md), and docs/preprint/figure_sources.json with every number they print
  figures  docs/preprint/figures/<name>.pdf: matplotlib, monochrome, vector PDF, in TeX Gyre Termes and TeX Gyre Heros
           (the metric clones of Times and Helvetica that TeX Live ships), with the numbers their captions use added to
           docs/preprint/figure_sources.json

docs/preprint/figure_sources.json gives each number the preprint prints with its source: the file (relative to the
project), the key inside it (a dotted path; a list's element is chosen by its fields, as comparisons[section=K2,
system=s9_b,metric=log_loss]), the value read, the decimals and the scale it was printed with, and where it is printed
(a table, a figure or the text). The numbers the text uses are listed in TEXT below, under the same keys, so that each
sentence's number has a source. A number that comes from no results file (a published paper's figure, a count stated in
a cited document) is listed in ORIGINS with where it comes from. src/eval/check_preprint_session10.py reads the LaTeX
source and these fragments and checks every number against this file.

Rules: intervals print as "estimate (low to high)"; four decimals for scores and differences, three for dollars per
1,000 questions and temperatures, as the results reports print them; thousands separators for counts; never -0.0000.

Usage (the cloud workspace, matplotlib 3.10.9; from the project folder or with --root):
  python3 src/eval/preprint_session10.py tables
  python3 src/eval/preprint_session10.py figures
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/prompt_session10.md, THE PREPRINT (step 7)"
C5 = "results/fiorillo_v0_5/comparison.json"
C4 = "results/fiorillo_v0_4/comparison.json"
CAL = "results/fiorillo_v0_5/calibration_s9.json"
LONG3 = "results/fiorillo_v0_5/s9_long_3_test.json"
LIC = "results/fiorillo_v0_5/license_recheck.json"
PRED = "results/fiorillo_v0_5/predictions/s9_ei_wavg_evidence_effect_test.csv"
OUT_TABLES = "docs/preprint/tables"
OUT_FIGURES = "docs/preprint/figures"
SOURCES = "docs/preprint/figure_sources.json"
FONT_DIR = "/usr/share/texmf/fonts/opentype/public/tex-gyre"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value, digits, scale=1.0, thousands=False):
    """A number as the preprint prints it."""
    v = float(value) * scale
    if thousands and digits == 0:
        return f"{int(round(v)):,}"
    text = f"{v:,.{digits}f}" if thousands else f"{v:.{digits}f}"
    if re.fullmatch(r"-0(\.0+)?", text):
        text = text[1:]
    return text


# ---------------------------------------------------------------- the sources

def segments(key):
    """A dotted key's segments; a dot inside a selector's brackets does not split, nor one written as a backslash and
    a dot (a key that itself holds a dot, as "train: listed as Incorrect in annotations/README\\.md")."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(key):
        ch = key[i]
        if ch == "\\" and key[i + 1:i + 2] == ".":
            cur += "."
            i += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur)
    return out


def resolve(data, key):
    """The value at a dotted key; a segment name[f=v,g=w] picks the one list element whose fields match. A comma inside
    a selector's value is written as a backslash and a comma (input=swap\\, comparator a time point or baseline)."""
    cur = data
    for seg in segments(key):
        m = re.fullmatch(r"([^\[]+)(?:\[(.+)\])?", seg)
        name, sel = m.group(1), m.group(2)
        cur = cur[int(name)] if name.isdigit() and isinstance(cur, list) else cur[name]
        if sel:
            conds = [c.replace("\\,", ",").split("=", 1) for c in re.split(r"(?<!\\),", sel)]
            hits = [x for x in cur if all(str(x.get(f)) == v for f, v in conds)]
            if len(hits) != 1:
                raise KeyError(f"{key}: {len(hits)} elements match [{sel}]")
            cur = hits[0]
    return cur


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        self.entries = []
        self.dates = []

    def data(self, rel):
        if rel not in self.files:
            self.files[rel] = json.loads((self.root / rel).read_text(encoding="utf-8"))
        return self.files[rel]

    def value(self, rel, key):
        return resolve(self.data(rel), key)

    def num(self, rel, key, digits=4, where="text", scale=1.0, thousands=False, label=""):
        v = self.value(rel, key)
        printed = fmt(v, digits, scale, thousands)
        self.entries.append({"printed": printed, "file": rel, "key": key, "value": v, "digits": digits, "scale": scale,
                             "thousands": thousands, "where": where, "label": label})
        return printed

    def interval(self, rel, key, digits=4, where="text", scale=1.0, label=""):
        """"estimate (low to high)" from an object with estimate, low and high."""
        e = self.num(rel, f"{key}.estimate", digits, where, scale, label=label)
        lo = self.num(rel, f"{key}.low", digits, where, scale, label=label)
        hi = self.num(rel, f"{key}.high", digits, where, scale, label=label)
        return e, lo, hi

    def csv_rows(self, rel):
        """A CSV file's rows; the file's SHA-256 is recorded with the JSON files'."""
        if rel not in self.files:
            with open(self.root / rel, newline="", encoding="utf-8") as f:
                self.files[rel] = list(csv.DictReader(f))
        return self.files[rel]

    def derived(self, printed, value, how, where, digits, files=(), scale=1.0, thousands=False):
        """A number computed from results files (printed, labeled derived where the text uses it); files names them."""
        for rel in files:
            self.files.setdefault(rel, None)
        self.entries.append({"printed": printed, "derived": how, "files": list(files), "value": value, "digits": digits,
                             "scale": scale, "thousands": thousands, "where": where})
        return printed

    def write(self, extra=None):
        path = self.root / SOURCES
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        keep = [e for e in old.get("entries", []) if e.get("where", "").split(":")[0] not in self.wheres()]
        data = {"plan": PLAN, "source": "src/eval/preprint_session10.py", "time_utc": now(),
                "rules": "intervals as estimate (low to high); four decimals for scores and differences, three for dollars "
                         "and temperatures; thousands separators for counts; never -0.0000",
                "files": {rel: sha256_file(self.root / rel) for rel in sorted(set(self.files) | set(old.get("files", {})))
                          if (self.root / rel).exists()},
                "entries": keep + self.entries, "origins": ORIGINS,
                "dates": self.dates if self.dates else old.get("dates", [])}
        if extra:
            data.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return data

    def wheres(self):
        return {e["where"].split(":")[0] for e in self.entries}


def cell(e, lo, hi, mark=""):
    """A table cell: the estimate over its interval, in a smaller size; mark follows the estimate."""
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{e}{mark}\\[-1pt]{{\scriptsize ({lo} to {hi})}}\end{{tabular}}"


DAGGER = "\\textsuperscript{\\dag}"


def inline_cell(e, lo, hi):
    """A one-line table cell: the estimate, then its interval in a smaller size."""
    return rf"{e} {{\scriptsize ({lo} to {hi})}}"


def diff_cell(S, rel, key, w, digits=4):
    """A difference's cell, marked with a dagger when its interval includes zero (no difference was detected)."""
    printed = S.interval(rel, key, digits, w)
    lo, hi = S.value(rel, f"{key}.low"), S.value(rel, f"{key}.high")
    return cell(*printed, mark=DAGGER if lo <= 0 <= hi else "")


MINUS = re.compile(r"(?<![0-9A-Za-z\[])-(?=[0-9])")  # not a hyphen in a word, nor a length such as \\[-1pt]


def tex_minus(text):
    """A number's minus sign as LaTeX's math minus ($-$), not a hyphen."""
    return MINUS.sub("$-$", text)


def table(path, colspec, header, rows, gap="2pt"):
    """A booktabs tabular; "MID" is a rule between groups, and a row given as ("HEAD", text) spans every column.
    gap is the space after each row (one-line tables use less)."""
    ncol = len(header)
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MID":
            lines.append(r"\midrule")
        elif isinstance(r, tuple) and r[0] == "HEAD":
            lines.append(r"\multicolumn{" + str(ncol) + r"}{@{}l}{\textit{" + r[1] + r"}} \\[1pt]")
        else:
            lines.append(tex_minus(" & ".join(r)) + (r" \\[" + gap + "]" if gap else r" \\"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- tables

EI_NAMES = [("s9_ei_wavg", "Fiorillo v0.5 (\\texttt{s9\\_ei\\_wavg})"), ("s8_ei", "Fiorillo v0.4 (\\texttt{s8\\_ei})"),
            ("gemma4_31b_long", "Gemma 4 31B-it, same input"), ("prior", "Prior (training shares)")]
METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("log_loss", "Log loss"), ("brier", "Brier score"), ("ece", "ECE")]


def tab_main_ei(S, root):
    """Table 2: each system's metrics, then K1's paired differences (v0.5's specialist minus the large model and minus
    the prior, 1,000 article resamples) on the same metrics."""
    w = "table:main_ei"
    rows = []
    for sys_, label in EI_NAMES:
        r = [label]
        for m, _ in METRICS:
            r.append(cell(*S.interval(C5, f"metrics.evidence_effect.{sys_}.{m}", 4, w)))
        rows.append(r)
    rows.append("MID")
    for ref, label in [("gemma4_31b_long", "v0.5 minus Gemma 4 31B-it (K1)"), ("prior", "v0.5 minus the prior (K1)")]:
        r = [label]
        for m, _ in METRICS:
            key = f"comparisons[section=K1,system=s9_ei_wavg,reference={ref},metric={m}].difference"
            r.append(diff_cell(S, C5, key, w))
        rows.append(r)
    table(root / OUT_TABLES / "main_ei.tex", ">{\\raggedright\\arraybackslash}p{3.0cm}ccccc", ["System"] + [h for _, h in METRICS], rows)


def tab_k1_answers(S, root):
    """Appendix C: K1 on each answer's F1 (v0.5's specialist minus the large model and minus the prior)."""
    w = "table:k1_answers"
    rows = []
    for f1, label in [("f1_increased", "Significantly increased"), ("f1_decreased", "Significantly decreased"),
                      ("f1_no_difference", "No significant difference")]:
        r = [label, cell(*S.interval(C5, f"metrics.evidence_effect.s9_ei_wavg.{f1}", 4, w))]
        for ref in ("gemma4_31b_long", "prior"):
            key = f"comparisons[section=K1,system=s9_ei_wavg,reference={ref},metric={f1}].difference"
            r.append(diff_cell(S, C5, key, w))
        rows.append(r)
    table(root / OUT_TABLES / "k1_answers.tex", "lccc",
          ["Answer", "F1 of v0.5", "Minus Gemma 4 31B-it's", "Minus the prior's"], rows)


def tab_pqa_answers(S, root):
    """Appendix C: the PubMedQA specialist's calibration slope and intercept for each answer (against the rest)."""
    w = "table:pqa_answers"
    rows = []
    for ans, label in [("yes", "Yes"), ("no", "No"), ("maybe", "Maybe")]:
        k = f"calibration_slope_intercept.pubmedqa_answer.answers.{ans}"
        rows.append([label, cell(*S.interval(C5, f"{k}.slope", 4, w)), cell(*S.interval(C5, f"{k}.intercept", 4, w)),
                     S.num(C5, f"{k}.rows_with_this_answer", 0, w, thousands=True)])
    table(root / OUT_TABLES / "pqa_answers.tex", "lccr", ["Answer", "Calibration slope", "Calibration intercept", "Items"], rows)


def tab_handoff(S, root):
    """Section 9: K7 at the four registered rates, for v0.5's specialist with Gemma 4 31B-it on the same input."""
    w = "table:handoff"
    k = "k7.pairs[fiorillo=s9_ei_wavg]"
    rows = []
    for r in (10, 20, 30, 50):
        rr = f"{k}.registered_rates.rate_{r}"
        rows.append([f"{r} percent", S.num(C5, f"{rr}.handed_off", 0, w, thousands=True),
                     cell(*S.interval(C5, f"{rr}.confidence", 4, w)),
                     cell(*S.interval(C5, f"{k}.i35_rule.registered_rates.rate_{r}.rule", 4, w)),
                     cell(*S.interval(C5, f"{rr}.random_expected", 4, w)),
                     cell(*S.interval(C5, f"{rr}.large_model_confidence", 4, w)),
                     cell(*S.interval(C5, f"{rr}.oracle", 4, w))])
    table(root / OUT_TABLES / "handoff.tex", "lrccccc",
          ["Handed off", "Prompts", "Confidence rule", "Learned rule", "Random (expected)",
           "\\shortstack{Large model's\\\\ confidence}", "Oracle"], rows)


def tab_bar(S, root):
    """The four criteria with v0.4's values and result (H1 to H4) and v0.5's (the update's bar)."""
    w = "table:bar"
    k = "release_bar.criteria"
    rows = []

    def values(rel, n):
        if n == "1":
            return f"{S.num(rel, k + '.1.values.ece', 4, w)} (limit {S.num(rel, k + '.1.values.limit', 2, w)})"
        if n in ("2", "3"):
            return cell(S.num(rel, f"{k}.{n}.values.estimate", 4, w), S.num(rel, f"{k}.{n}.values.low", 4, w),
                        S.num(rel, f"{k}.{n}.values.high", 4, w))
        a, b = ("s8_ei", "gemma4_31b") if rel == C4 else ("specialist", "large_model")
        return f"{S.num(rel, f'{k}.4.values.{a}', 4, w)} against {S.num(rel, f'{k}.4.values.{b}', 4, w)}"

    for n, label in [("1", "ECE at most the limit"), ("2", "Log loss minus the prior's, interval below 0"),
                     ("3", "Log loss minus the large model's, interval below 0"), ("4", "Macro-F1 not below the large model's")]:
        rows.append([n, label, values(C4, n), S.value(C4, f"{k}.{n}.result"), values(C5, n), S.value(C5, f"{k}.{n}.result")])
    table(root / OUT_TABLES / "bar.tex", "c>{\\raggedright\\arraybackslash}p{3.4cm}cccc",
          ["", "Criterion", "v0.4 (\\texttt{s8\\_ei})", "Result", "v0.5 (\\texttt{s9\\_ei\\_wavg})", "Result"], rows)


def tab_published(S, root):
    w = "table:published"
    v5 = inline_cell(*S.interval(C5, "metrics.evidence_effect.s9_ei_wavg.macro_f1", 4, w))
    v4 = inline_cell(*S.interval(C4, "g3.macro_f1", 4, w))
    v4l = inline_cell(*S.interval(C4, "g3.lenient_macro_f1", 4, w))
    rows = []
    # The published values print with the precision the papers give them (DeYoung et al.: three decimals; Buchmann et
    # al.: two decimals of a percentage), as docs/fiorillo_v0_5.md, Table 25, prints them.
    # The labels name each setting as the paper does (docs/fiorillo_v0_5.md, the notes under Table 25): LED's lower
    # value is its run without structure infusion, its upper value its best setting; LongT5's is its run without
    # structure infusion, which is also its best.
    for key, label, scoring, mine, digits in [
            ("g3.published.deyoung_2020_pipeline.value", "\\citet{deyoung2020}, full-text pipeline", "strict", v4, 3),
            ("g3.published.deyoung_2020_oracle.value", "\\citet{deyoung2020}, evidence spans given", "strict", v4, 3),
            ("g3.published_lenient.buchmann_2024_led_low.value", "\\citet{buchmann2024structure}, LED, no structure infusion", "lenient", v4l, 4),
            ("g3.published_lenient.buchmann_2024_led_high.value", "\\citet{buchmann2024structure}, LED, best setting", "lenient", v4l, 4),
            ("g3.published_lenient.buchmann_2024_longt5.value", "\\citet{buchmann2024structure}, LongT5", "lenient", v4l, 4)]:
        rows.append([label, scoring, S.num(C4, key, digits, w), mine, v5])
    table(root / OUT_TABLES / "published.tex", ">{\\raggedright\\arraybackslash}p{6.0cm}lccc",
          ["Published system", "Scoring", "Its macro-F1", "v0.4, same scoring", "v0.5, strict"], rows, gap="1pt")


def tab_k2(S, root):
    w = "table:k2"
    rows = []
    for sys_, label in [("s9_b", "Same recipe, clean articles only (\\texttt{s9\\_b})"), ("s9_ei_wavg", "Fiorillo v0.5 (\\texttt{s9\\_ei\\_wavg})")]:
        r = [label]
        for m, _ in METRICS:
            key = f"comparisons[section=K2,system={sys_},reference=s8_ei,metric={m}]"
            r.append(diff_cell(S, C5, f"{key}.difference", w))
        rows.append(r)
    table(root / OUT_TABLES / "k2.tex", ">{\\raggedright\\arraybackslash}p{3.0cm}ccccc", ["Minus v0.4 (\\texttt{s8\\_ei})"] + [h for _, h in METRICS], rows)


def tab_reading(S, root):
    """K4's reading checks: one row a statistic, v0.4's and v0.5's side by side, each with its count of prompts."""
    w = "table:reading"
    spec = [
        ("No article", None),
        ("none", "accuracy", "Accuracy"), ("none", "macro_f1", "Macro-F1"),
        ("none", "agreement_with_full_input", "Agreement with the whole-article answer"),
        ("Title only", None),
        ("title", "accuracy", "Accuracy"), ("title", "macro_f1", "Macro-F1"),
        ("title", "agreement_with_full_input", "Agreement with the whole-article answer"),
        ("title minus none (prompts in both)", "none_macro_f1", "Macro-F1, no article, on the same prompts"),
        ("title minus none (prompts in both)", "macro_f1_difference", "Macro-F1, title only minus no article"),
        ("Intervention and comparator exchanged", None),
        ("swap", "accuracy", "Accuracy against the exchanged answers"),
        ("swap", "macro_f1", "Macro-F1 against the exchanged answers"),
        ("swap", "exchange_share", "Share of direction answers that exchange"),
        ("swap", "stay_share", "Share of no-difference answers that stay"),
        ("Exchanged, on the prompts whose comparator is a time point or a baseline", None),
        ("swap\\, comparator a time point or baseline", "exchange_share", "Share of direction answers that exchange"),
        ("swap\\, comparator a time point or baseline", "stay_share", "Share of no-difference answers that stay")]
    rows = []
    for item in spec:
        if item[1] is None:
            rows.append(("HEAD", item[0]))
            continue
        inp, stat, label = item
        r = [label]
        for sysk in ("s8_ei", "v05"):
            k = f"k4.table[system={sysk},input={inp},statistic={stat}]"
            r.append(S.num(C5, k + ".n", 0, w, thousands=True))
            r.append(inline_cell(S.num(C5, k + ".estimate", 4, w), S.num(C5, k + ".low", 4, w), S.num(C5, k + ".high", 4, w)))
        rows.append(r)
    table(root / OUT_TABLES / "reading.tex", ">{\\raggedright\\arraybackslash}p{5.4cm}rcrc",
          ["Statistic", "Prompts", "v0.4 (\\texttt{s8\\_ei})", "Prompts", "v0.5 (\\texttt{s9\\_ei\\_wavg})"], rows, gap="")


def tab_pubmedqa(S, root):
    w = "table:pubmedqa"
    rows = []
    for sys_, label in [("s7_b3", "Fiorillo v0.5 (\\texttt{s7\\_b3})"), ("gemma4_31b", "Gemma 4 31B-it"),
                        ("biogpt_large_pqa-raw", "BioGPT-Large-PubMedQA, uncalibrated"),
                        ("biogpt_large_pqa", "BioGPT-Large-PubMedQA, calibrated (secondary)"), ("prior", "Prior")]:
        r = [label]
        for m, _ in METRICS:
            r.append(cell(*S.interval(C5, f"metrics.pubmedqa_answer.{sys_}.{m}", 4, w)))
        rows.append(r)
    table(root / OUT_TABLES / "pubmedqa.tex", ">{\\raggedright\\arraybackslash}p{3.0cm}ccccc", ["System"] + [h for _, h in METRICS], rows)


def tab_readmission(S, root):
    w = "table:readmission"
    # The last three columns are for readmission within 30 days against the rest (the caption says so).
    cols = [("log_loss", "Log loss"), ("ece", "ECE"), ("accuracy", "Accuracy"), ("auroc_30d", "\\shortstack{AUROC,\\\\ 30 days}"),
            ("calibration_intercept_30d", "\\shortstack{Calibration\\\\ intercept, 30 days}"),
            ("calibration_slope_30d", "\\shortstack{Calibration\\\\ slope, 30 days}")]
    rows = []
    for sys_, label in [("readmission_pool_v2", "Fiorillo v0.5 (tree pool)"), ("prior", "Prior")]:
        r = [label]
        for m, _ in cols:
            obj = S.value(C5, f"metrics.readmit_window.{sys_}.{m}")
            if obj is None or obj.get("estimate") is None:
                r.append("not computed")
            else:
                r.append(cell(*S.interval(C5, f"metrics.readmit_window.{sys_}.{m}", 4, w)))
        rows.append(r)
    table(root / OUT_TABLES / "readmission.tex", ">{\\raggedright\\arraybackslash}p{2.0cm}cccccc", ["System"] + [h for _, h in cols], rows)


RECIPE = "results/fiorillo_v0_5/recipe_selection.json"
LONG3_CAL = "results/fiorillo_v0_5/calibration_s9_long_3.json"
CAL8 = "results/fiorillo_v0_4/calibration_s8.json"


def tab_runs(S, root):
    """Appendix A: every run and system of v0.5's Evidence Inference specialist, with s9_long_3's figures.

    The validation column is the recipe rule's score (recipe_selection.json, scores: session 8's calibration rule's mean
    out-of-fold log loss on the 654 clean validation prompts, the lower of its two families); s9_long_3's is session
    10's calibration of it by the same rule (calibration_s9_long_3.json), and s8_ei's is session 8's, on all 1,237
    validation prompts (results/fiorillo_v0_4/calibration_s8.json). Runs that were not run on test print NOT RUN."""
    w = "table:runs"
    rows = []
    groups = [
        ("Base recipe", ["s9_b_1", "s9_b_2", "s9_b_3", "s9_b"]),
        ("Swap augmentation", ["s9_swap_1", "s9_swap_2"]),
        ("First higher learning rate", ["s9_lr1_1", "s9_lr1_2"]),
        ("Second higher learning rate", ["s9_lr2_1", "s9_lr2_2"]),
        ("Longer inputs (chosen)", ["s9_long_1", "s9_long_2", "s9_long_3", "s9_ei", "s9_ei_wavg", "s9_ei_wavg_sa"]),
        ("Fiorillo v0.4", ["s8_ei"])]
    tested = set(S.data(C5)["metrics"]["evidence_effect"])
    for gi, (recipe, systems) in enumerate(groups):
        if gi:
            rows.append("MID")
        for s in systems:
            label = "\\texttt{" + s.replace("_", "\\_") + "}" + ("\\textsuperscript{b}" if s == "s9_long_3" else "")
            if s == "s9_long_3":
                lo = S.data(LONG3_CAL)["s9_long_3"]["evidence_effect"]
                val = S.num(LONG3_CAL, f"s9_long_3.evidence_effect.families.{lo['chosen']}.out_of_fold_log_loss.mean", 4, w)
                test = [cell(*S.interval(LONG3, f"metrics.{m}", 4, w)) for m in ("log_loss", "accuracy", "ece")]
            elif s == "s8_ei":
                c8 = S.data(CAL8)["s8_ei"]["evidence_effect"]
                fam = min(c8["families"], key=lambda f: c8["families"][f]["out_of_fold_log_loss"]["mean"])
                val = S.num(CAL8, f"s8_ei.evidence_effect.families.{fam}.out_of_fold_log_loss.mean", 4, w) + "\\textsuperscript{a}"
                test = [cell(*S.interval(C5, f"metrics.evidence_effect.{s}.{m}", 4, w)) for m in ("log_loss", "accuracy", "ece")]
            else:
                val = S.num(RECIPE, f"scores.{s}", 4, w)
                if s in tested:
                    test = [cell(*S.interval(C5, f"metrics.evidence_effect.{s}.{m}", 4, w)) for m in ("log_loss", "accuracy", "ece")]
                else:
                    test = ["NOT RUN"] * 3
            rows.append([recipe if s == systems[0] else "", label, val] + test)
    table(root / OUT_TABLES / "runs.tex", "llcccc",
          ["Recipe", "System", "Validation score", "Test log loss", "Test accuracy", "Test ECE"], rows)


def tab_answers(S, root):
    """Appendix C: each answer's F1 and its calibration slope and intercept (one answer against the rest)."""
    w = "table:answers"
    rows = []
    for ans, label, f1 in [("increased", "Significantly increased", "f1_increased"),
                           ("decreased", "Significantly decreased", "f1_decreased"),
                           ("no_difference", "No significant difference", "f1_no_difference")]:
        k = f"calibration_slope_intercept.evidence_effect.answers.{ans}"
        rows.append([label, cell(*S.interval(C5, f"metrics.evidence_effect.s9_ei_wavg.{f1}", 4, w)),
                     cell(*S.interval(C5, f"{k}.slope", 4, w)), cell(*S.interval(C5, f"{k}.intercept", 4, w)),
                     S.num(C5, f"{k}.rows_with_this_answer", 0, w, thousands=True)])
    table(root / OUT_TABLES / "answers.tex", "lcccr", ["Answer", "F1", "Calibration slope", "Calibration intercept", "Prompts"], rows)


def tab_cost(S, root):
    """Appendix D: the cost per 1,000 questions at the pod's rate."""
    w = "table:cost"
    k = "cost_per_1000_questions"
    rows = [["Evidence Inference", "Fiorillo v0.5", S.num(C5, f"{k}.specialists.evidence_effect.seconds_per_question", 4, w),
             "\\$" + S.num(C5, f"{k}.specialists.evidence_effect.usd", 3, w)],
            ["Evidence Inference", "Gemma 4 31B-it", S.num(C5, f"{k}.large_model.evidence_effect.seconds_per_question", 4, w),
             "\\$" + S.num(C5, f"{k}.large_model.evidence_effect.usd", 3, w)],
            ["PubMedQA", "Fiorillo v0.5", S.num(C5, f"{k}.specialists.pubmedqa_answer.seconds_per_question", 4, w),
             "\\$" + S.num(C5, f"{k}.specialists.pubmedqa_answer.usd", 3, w)],
            ["PubMedQA", "Gemma 4 31B-it", S.num(C5, f"{k}.large_model.pubmedqa_answer.seconds_per_question", 4, w),
             "\\$" + S.num(C5, f"{k}.large_model.pubmedqa_answer.usd", 3, w)]]
    table(root / OUT_TABLES / "cost.tex", "llrr", ["Question", "System", "\\shortstack{GPU seconds\\\\ per question}", "\\shortstack{Dollars per\\\\ 1,000 questions}"], rows)


# The numbers the text uses, section by section ("% SECTION: <name>" in the LaTeX source). Each item is
# (file, key, decimals, scale, thousands); interval items expand to the estimate, the low end and the high end. With a
# scale of -1 a difference prints as a positive amount ("lower by 0.0123"), and its low and high ends trade places.
# The figure check matches a number in a section only with the entries registered for that section.
POWER = "results/fiorillo_v0_5/power.json"
WA = "results/fiorillo_v0_5/weight_average.json"
RP2T = "results/fiorillo_v0_5/readmission_pool_v2_test.json"
SEL4 = "results/fiorillo_v0_4/article_selection.json"
RC5 = "results/fiorillo_v0_5/release_candidate.json"
LIC_CSV = "results/fiorillo_v0_5/license_recheck.csv"
# The usage check's fourth run, on the second fix (docs/prompt_session10.md, correction 9), which the text reports.
USAGE = "results/fiorillo_v0_5/usage_check/fixed2/usage_check.json"
MAN4 = "results/fiorillo_v0_4/records_manifest.json"
POWER4 = "results/fiorillo_v0_4/power.json"
CONT4 = "results/fiorillo_v0_4/contamination_checks.json"
ZEROS = "results/fiorillo_v0_5/zeros.json"


def N(rel, key, digits=4, scale=1.0, thousands=False):
    return [(rel, key, digits, scale, thousands)]


def C(rel, key):
    return [(rel, key, 0, 1.0, True)]


def I(rel, key, digits=4, scale=1.0):
    return [(rel, f"{key}.{p}", digits, scale, False) for p in ("estimate", "low", "high")]


def K(section, system, reference, metric):
    return f"comparisons[section={section},system={system},reference={reference},metric={metric}].difference"


def K4(system, inp, stat):
    return f"k4.table[system={system},input={inp},statistic={stat}]"


def K7(fiorillo):
    return f"k7.pairs[fiorillo={fiorillo}]"


BAR, BAR4 = "release_bar.criteria", "release_bar.criteria"
TMN = "title minus none (prompts in both)"
TPB = "swap\\, comparator a time point or baseline"
SP = "by_split"

TEXT = {
    "abstract": (N(RECIPE, "winner.settings.max_len", 0, 1.0, True)
                 + C(LIC, f"{SP}.train.clean_articles") + C(LIC, f"{SP}.train.articles") + C(C5, f"{BAR}.2.values.n")
                 + C(C5, f"{BAR}.2.values.articles") + N(C5, f"{BAR}.1.values.ece") + N(C5, f"{BAR}.1.values.limit", 2)
                 + N(C5, "metrics.evidence_effect.s9_ei_wavg.log_loss.estimate")
                 + I(C5, f"{BAR}.2.values", scale=-1.0) + I(C5, f"{BAR}.3.values", scale=-1.0)
                 + N(C5, f"{BAR}.4.values.specialist") + N(C5, f"{BAR}.4.values.large_model")
                 + I(C5, K("K2", "s9_b", "s8_ei", "accuracy"), scale=-1.0) + I(C5, K("K2", "s9_b", "s8_ei", "log_loss"))
                 + N(C5, K4("v05", "none", "macro_f1") + ".estimate") + I(C5, K4("v05", TMN, "macro_f1_difference"))
                 + N(C5, K4("v05", "swap", "exchange_share") + ".estimate")),
    "intro": [],
    "related": [],
    "data": (C(LIC, f"{SP}.train.articles") + C(LIC, f"{SP}.train.prompts") + C(LIC, f"{SP}.validation.articles")
             + C(LIC, f"{SP}.validation.prompts") + C(LIC, f"{SP}.test.articles") + C(LIC, f"{SP}.test.prompts")
             + C(LIC, f"{SP}.train.clean_articles") + C(LIC, f"{SP}.train.clean_prompts")
             + C(LIC, f"{SP}.validation.clean_articles") + C(LIC, f"{SP}.validation.clean_prompts")
             + C(LIC, f"{SP}.test.clean_articles") + C(LIC, f"{SP}.test.clean_prompts")
             + C(C5, "metrics.pubmedqa_answer.s7_b3.n") + C(C5, "metrics.readmit_window.readmission_pool_v2.n")
             + C(MAN4, "drops.train: no accepted row") + C(MAN4, "drops.validation: no accepted row")
             + C(MAN4, "drops.test: no accepted row") + C(MAN4, "drops.train: listed as Incorrect in annotations/README\\.md")
             + C(MAN4, "drops.test: empty article text") + C(MAN4, "drops.PMCID in no split file")),
    "model": (N(RECIPE, "winner.settings.max_len", 0, 1.0, True) + N(RECIPE, "winner.settings.lr_adapters", 5)
              + C(LIC, f"{SP}.train.clean_prompts") + N(WA, "rank", 0) + N(WA, "alpha", 0)
              + N(SEL4, "max_len", 0, 1.0, True) + C(RECIPE, "n_clean_validation") + N(RECIPE, "folds", 0)
              + N(RECIPE, "repeats", 0) + N(RECIPE, "margin.m")
              + N(CAL, "s9_ei_wavg_s8rule.evidence_effect.families.temperature.temperature", 3)
              + [(CAL, f"s9_ei_wavg.evidence_effect.families.{fam}.out_of_fold_log_loss.mean", 4, 1.0, False)
                 for fam in ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")]
              + N(RP2T, "weights.xgboost_tuned", 1) + N(RP2T, "weights.catboost_tuned", 1) + N(RP2T, "T")),
    "tests": (C(C5, f"{BAR}.2.values.n") + N(C5, "settings.ece_bins", 0) + N(C5, "ece_limits.evidence_effect.limit", 2)
              + C(C5, "ece_limits.evidence_effect.draws") + N(C5, "ece_limits.evidence_effect.p95")
              + N(RECIPE, "winner.settings.max_len", 0, 1.0, True) + C(C5, f"{BAR}.2.values.resamples")
              + C(C5, "settings.resamples") + N(POWER, "real_gain_for_80_percent")
              + N(POWER4, "cases[s=0.8,D=1.5].real_gain_for_80_percent")),
    "results": (N(C4, f"{BAR4}.1.values.ece") + I(C4, f"{BAR4}.2.values") + I(C4, f"{BAR4}.3.values")
                + C(USAGE, "questions.readmit_window.n") + C(USAGE, "questions.pubmedqa_answer.n")
                + C(USAGE, "questions.evidence_effect.n")
                + N(USAGE, "questions.pubmedqa_answer.mean_absolute_difference")
                + N(USAGE, "questions.pubmedqa_answer.largest_absolute_difference")
                + N(USAGE, "questions.evidence_effect.mean_absolute_difference")
                + N(USAGE, "questions.evidence_effect.largest_absolute_difference")
                + C(C5, "sensitivity.flagged_test_prompts") + C(C5, "sensitivity.unflagged_test_prompts")
                + I(C5, K("K1_sensitivity", "s9_ei_wavg", "prior", "log_loss"))
                + I(C5, K("K1_sensitivity", "s9_ei_wavg", "gemma4_31b_long", "log_loss"))
                + I(C5, "metrics.evidence_effect.s9_ei_wavg-raw.ece") + I(C5, "metrics.evidence_effect.s9_ei_wavg-raw.log_loss")
                + I(C5, "calibration_slope_intercept.pubmedqa_answer.answers.no.slope")
                + N(C5, "k6.choice.new_out_of_fold_log_loss") + N(C5, "k6.choice.session8_out_of_fold_log_loss")
                + N(C5, "k6.choice.margin", 3)
                + N(C4, f"{BAR4}.4.values.s8_ei") + N(C4, f"{BAR4}.4.values.gemma4_31b")
                + N(C5, f"{BAR}.1.values.ece") + N(C5, f"{BAR}.1.values.limit", 2)
                + I(C5, f"{BAR}.2.values") + I(C5, f"{BAR}.3.values")
                + N(C5, f"{BAR}.4.values.specialist") + N(C5, f"{BAR}.4.values.large_model")
                + C(C5, f"{BAR}.2.values.n") + C(C5, f"{BAR}.2.values.articles") + C(C5, f"{BAR}.2.values.resamples")
                + N(SEL4, "max_len", 0, 1.0, True) + N(RECIPE, "winner.settings.max_len", 0, 1.0, True)
                + I(C5, "metrics.evidence_effect.s9_ei_wavg.accuracy")
                + N(C5, "metrics.evidence_effect.s9_ei_wavg.macro_f1.estimate")
                + N(C5, "metrics.evidence_effect.s9_ei_wavg-raw.ece.estimate")
                + N(C5, "metrics.evidence_effect.s9_ei_wavg.ece.estimate")
                + N(C5, "metrics.evidence_effect.s9_ei_wavg-raw.log_loss.estimate")
                + N(C5, "metrics.evidence_effect.s9_ei_wavg.log_loss.estimate")
                + C(C5, "settings.resamples") + N(C5, "settings.ece_bins", 0)
                + C(C5, "metrics.pubmedqa_answer.s7_b3.n") + I(C5, "metrics.pubmedqa_answer.s7_b3.accuracy")
                + N(C5, "metrics.pubmedqa_answer.s7_b3.log_loss.estimate")
                + N(CAL, "biogpt_large_pqa.pubmedqa_answer.n_calibrate", 0)
                + I(C5, K("K5", "s7_b3", "biogpt_large_pqa-raw", "log_loss"), scale=-1.0)
                + I(C5, K("K5", "s7_b3", "biogpt_large_pqa-raw", "accuracy"))
                + I(C5, K("K5", "biogpt_large_pqa-raw", "gemma4_31b", "log_loss"))
                + I(C5, K("K5", "biogpt_large_pqa-raw", "gemma4_31b", "accuracy"))
                + N(C5, K("K5", "biogpt_large_pqa-raw", "gemma4_31b", "accuracy") + ".low", 4, -1.0)
                + C(C5, "metrics.readmit_window.readmission_pool_v2.n")
                + N(C5, "metrics.readmit_window.readmission_pool_v2.log_loss.estimate")
                + N(C5, "metrics.readmit_window.prior.log_loss.estimate")
                + N(C5, "metrics.readmit_window.readmission_pool_v2.auroc_30d.estimate")
                + I(C5, K("K3", "s9_long_1_2", "s9_b_1_2", "log_loss"), scale=-1.0)
                + I(C5, K("K3_test", "s9_ei_wavg", "s9_b", "log_loss"))
                + N(C5, K("K3_test", "s9_ei_wavg", "s9_b", "log_loss") + ".low", 4, -1.0)
                + I(C5, K("K8", "s9_ei_wavg", "s9_ei", "log_loss"))
                + I(C5, K("K10", "s9_ei_wavg_sa", "s9_ei_wavg", "log_loss"))
                + I(C5, K("K10", "s9_ei_wavg_sa", "s9_ei_wavg", "ece"))),
    "reading": (I(C5, K4("v05", "none", "macro_f1")) + I(C5, K4("s8_ei", "none", "macro_f1"))
                + I(C5, "k4.references.prior.macro_f1") + I(C5, K4("s8_ei", TMN, "macro_f1_difference"))
                + I(C5, K4("v05", "swap", "accuracy")) + I(C5, K4("s8_ei", "swap", "accuracy"))
                + C(C5, K4("v05", TMN, "macro_f1_difference") + ".n") + I(C5, K4("v05", TMN, "macro_f1_difference"))
                + N(C5, K4("s8_ei", TMN, "macro_f1_difference") + ".estimate")
                + N(C5, "k4.references.windisch_2026.accuracy_low", 2) + N(C5, "k4.references.windisch_2026.accuracy_high", 2)
                + N(C5, "k4.references.windisch_2026.trials", 0)
                + N(C5, "k4.references.windisch_2026.majority_class_accuracy", 3)
                + I(C5, K4("v05", "swap", "exchange_share")) + I(C5, K4("s8_ei", "swap", "exchange_share"))
                + N(C5, K4("v05", TPB, "exchange_share") + ".estimate") + C(C5, K4("v05", TPB, "exchange_share") + ".n")
                + N(C5, K4("v05", "swap", "accuracy") + ".estimate") + N(C5, K4("s8_ei", "swap", "accuracy") + ".estimate")
                + C(C5, "settings.resamples") + C(C5, K4("v05", "none", "accuracy") + ".n")
                + C(C5, K4("v05", "title", "accuracy") + ".n")),
    "license": (C(LIC, f"{SP}.train.clean_articles") + C(LIC, f"{SP}.train.articles")
                + I(C5, K("K2", "s9_b", "s8_ei", "log_loss")) + I(C5, K("K2", "s9_b", "s8_ei", "accuracy"), scale=-1.0)
                + I(C5, K("K2", "s9_ei_wavg", "s8_ei", "accuracy"), scale=-1.0)
                + I(C5, K("K2", "s9_ei_wavg", "s8_ei", "log_loss"))
                + N(C5, K("K2", "s9_ei_wavg", "s8_ei", "log_loss") + ".low", 4, -1.0)
                + C(C5, f"{BAR}.2.values.n") + C(C5, f"{BAR}.2.values.articles") + C(C5, "settings.resamples")
                + C(LIC, f"{SP}.validation.clean_prompts") + C(CAL8, "s8_ei.evidence_effect.n_calibrate")),
    "handoff": ([(C5, f"{K7('s9_ei_wavg')}.registered_rates.rate_{r}.confidence.estimate", 4, 1.0, False)
                 for r in (10, 20, 30, 50)]
                + N(C5, f"{K7('s9_ei_wavg')}.fiorillo_alone.accuracy") + N(C5, f"{K7('s9_ei_wavg')}.large_model_alone.accuracy")
                + N(C5, f"{K7('s9_ei_wavg')}.registered_rates.rate_10.oracle.estimate")
                + C(C5, f"{K7('s9_ei_wavg')}.n") + C(C5, f"{K7('s9_ei_wavg')}.room.fiorillo_wrong_large_model_right")
                + C(C5, f"{K7('s9_ei_wavg')}.room.fiorillo_right_large_model_wrong")
                + N(C5, f"{K7('s9_ei_wavg')}.room.share_fiorillo_wrong_large_model_right")
                + N(SEL4, "max_len", 0, 1.0, True)
                + N(C5, f"{K7('s8_ei')}.registered_rates.rate_10.confidence.estimate")
                + N(C5, f"{K7('s8_ei')}.fiorillo_alone.accuracy")
                + N(C5, f"{K7('s7_b3')}.registered_rates.rate_10.confidence.estimate")
                + N(C5, f"{K7('s7_b3')}.fiorillo_alone.accuracy")
                + N(C5, f"{K7('s9_ei_wavg')}.i35_rule.registered_rates.rate_10.rule_minus_confidence.high")
                + N(C5, f"{K7('s9_ei_wavg')}.i35_rule.registered_rates.rate_10.rule_minus_confidence.low", 4, -1.0)
                + I(C5, f"{K7('s9_ei_wavg')}.areas.aurc") + I(C5, f"{K7('s9_ei_wavg')}.areas.e_aurc")
                + I(C5, f"{K7('s9_ei_wavg')}.areas.augrc")
                + C(C5, "settings.resamples")
                + N(C5, "cost_per_1000_questions.hourly_usd", 2)
                + N(C5, "cost_per_1000_questions.specialists.evidence_effect.usd", 3)
                + N(C5, "cost_per_1000_questions.large_model.evidence_effect.usd", 3)),
    "limits": (N(RECIPE, "winner.settings.max_len", 0, 1.0, True)
               + C(CONT4, "pubmedqa.s8_ei_test_matched_or_not.flagged.articles")
               + C(CONT4, "pubmedqa.s8_ei_test_matched_or_not.flagged.prompts")
               + I(CONT4, "pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy")
               + I(CONT4, "pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest")
               + N(CONT4, "pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.low", 4, -1.0)),
    "discussion": N(C5, f"{BAR}.1.values.ece") + N(C5, f"{BAR}.1.values.limit", 2),
    "statements": [],
    "appA": (C(RECIPE, "n_clean_validation") + N(RECIPE, "margin.m") + N(RECIPE, "candidates.long.score")
             + N(RECIPE, "candidates.long.base_mean_12")
             + I(C5, K("K3", "s9_long_1_2", "s9_b_1_2", "log_loss")) + I(C5, K("K3", "s9_swap_1_2", "s9_b_1_2", "log_loss"))
             + I(C5, K("K3", "s9_lr1_1_2", "s9_b_1_2", "log_loss")) + I(C5, K("K3", "s9_lr2_1_2", "s9_b_1_2", "log_loss"))
             + N(C5, K("K3", "s9_lr2_1_2", "s9_b_1_2", "log_loss") + ".low", 4, -1.0)
             + N(LONG3_CAL, "s9_long_3.evidence_effect.families.temperature.out_of_fold_log_loss.mean")
             + I(LONG3, "metrics.log_loss") + I(LONG3, "metrics.accuracy")
             + C(C5, "settings.resamples") + C(C5, f"{BAR}.2.values.n") + C(CAL8, "s8_ei.evidence_effect.n_calibrate")
             # K6's recoded trees (moved here from Section 6 to keep the main text within its 12 pages)
             + I(C5, K("K6", "xgboost_codes", "xgboost_tuned", "log_loss"))
             + I(C5, K("K6", "catboost_codes", "catboost_tuned", "log_loss"))
             + I(C5, K("K6", "xgboost_codes", "xgboost_tuned", "brier"))
             + I(C5, K("K6", "xgboost_codes", "xgboost_tuned", "quadratic_weighted_kappa"), scale=-1.0)
             + I(C5, K("K6", "xgboost_codes", "xgboost_tuned", "calibration_intercept_30d"))
             + I(C5, K("K6", "catboost_codes", "catboost_tuned", "calibration_intercept_30d"))
             + C(C5, "metrics.readmit_window.readmission_pool_v2.n")),
    "appB": [],
    "appC": (N(C5, "calibration_slope_intercept.evidence_effect.answers.no_difference.slope.low")
             + N(C5, "calibration_slope_intercept.evidence_effect.answers.decreased.slope.high")
             + N(C5, "calibration_slope_intercept.evidence_effect.answers.no_difference.intercept.low")
             + N(C5, "calibration_slope_intercept.evidence_effect.answers.increased.intercept.high")
             + C(C5, "calibration_slope_intercept.resamples") + C(C5, f"{BAR}.2.values.n")
             + I(C5, "calibration_slope_intercept.pubmedqa_answer.answers.no.slope")
             + C(C5, "metrics.pubmedqa_answer.s7_b3.n")),
    "appD": N(C5, "cost_per_1000_questions.hourly_usd", 2),
}

# Derived numbers in the text, by section: (printed form's decimals, thousands, rule, files). The rule is written so
# that src/eval/check_preprint_session10.py can compute the number again from its files.
REL_BINS = 15


def derived_text(S):
    lic_b = S.data(LIC)["by_split"]
    out = []

    def add(section, value, digits, thousands, how, files):
        out.append((section, value, digits, thousands, how, files))

    add("data", lic_b["train"]["articles"] - lic_b["train"]["clean_articles"], 0, True,
        f"{LIC} by_split.train.articles minus by_split.train.clean_articles", [LIC])
    add("discussion", lic_b["train"]["clean_articles"] + lic_b["validation"]["clean_articles"], 0, True,
        f"{LIC} by_split.train.clean_articles plus by_split.validation.clean_articles", [LIC])
    k_none = K4("v05", "none", "accuracy") + ".n"
    k_title = K4("v05", "title", "accuracy") + ".n"
    add("reading", S.value(C5, k_none) - S.value(C5, k_title), 0, True, f"{C5} {k_none} minus {C5} {k_title}", [C5])
    z = S.data(ZEROS)["rows_whose_true_answer_prints_as_0"]

    def zsum(d, field):
        if isinstance(d, dict):
            if "rows" in d and "true_answer_printed_zero" in d:
                return d[field]
            return sum(zsum(v, field) for v in d.values())
        return 0
    add("results", zsum(z, "rows"), 0, True, f"the sum of rows over every system, split and question of {ZEROS}", [ZEROS])
    if zsum(z, "true_answer_printed_zero") != 0:
        raise SystemExit(f"{ZEROS} counts rows whose true answer prints as 0; the text says none")
    bins = reliability_table(S.root, REL_BINS)
    top = max(bins)
    for field, digits, thousands in (("n", 0, True), ("lower_edge", 4, False), ("share_right", 4, False),
                                     ("mean_probability", 4, False), ("wilson_low", 4, False), ("wilson_high", 4, False)):
        add("results", bins[top][field], digits, thousands,
            f"{field} of the reliability bin {top} of {PRED} ({REL_BINS} equal-width bins on the top answer's probability)",
            [PRED])
    for section, value, digits, thousands, how, files in out:
        S.derived(fmt(value, digits, thousands=thousands), value, how, f"text:{section}", digits, files=files,
                  thousands=thousands)


def reliability_table(root, bins):
    """The reliability bins by index, with the fields the text names (the figure's own bins, by the same rule)."""
    used, _ = reliability_bins(root, bins)
    out = {}
    for o in used:
        b = int(round(o["lo_edge"] * bins))
        out[b] = {"n": o["n"], "lower_edge": float(o["lo_edge"]), "share_right": o["accuracy"],
                  "mean_probability": o["mean_conf"], "wilson_low": o["wilson"][0], "wilson_high": o["wilson"][1]}
    return out


# Numbers that come from no results file, each with where it is printed and where it comes from (the figure check's
# NO SOURCE rows name it).
ALL_TEXT = ["abstract", "intro", "related", "data", "model", "tests", "results", "reading", "license", "handoff",
            "limits", "discussion", "statements", "appA", "appB", "appC", "appD"]
ORIGINS = [
    {"printed": "0.0001", "where": ["text:results"],
     "origin": "the usage check's limit for readmission, fixed before its first run (docs/prompt_session10.md, THE USAGE CHECK; the rules in results/fiorillo_v0_5/usage_check/fixed2/usage_check.json)"},
    {"printed": "0.6", "where": ["text:results"],
     "origin": "the usage check's confident top probability for EI and PubMedQA, fixed before its first run (docs/prompt_session10.md, THE USAGE CHECK; the rules in results/fiorillo_v0_5/usage_check/fixed2/usage_check.json)"},
    {"printed": "0.02", "where": ["text:results"],
     "origin": "the usage check's limit on the mean absolute difference for EI and PubMedQA, fixed before its first run (docs/prompt_session10.md, THE USAGE CHECK; the rules in results/fiorillo_v0_5/usage_check/fixed2/usage_check.json)"},
    {"printed": "95", "where": [f"text:{s}" for s in ALL_TEXT],
     "origin": "the intervals' level, 95 percent, fixed by the registration (docs/osf_prereg_fiorillo_v0_4.md, Statistical models)"},
    {"printed": "95th", "where": ["text:tests"],
     "origin": "the ECE limit's percentile, fixed by the registration (docs/osf_prereg_fiorillo_v0_4.md, Hypotheses, H1)"},
    {"printed": "4", "where": ["text:abstract", "text:discussion"],
     "origin": "the size of the base model, Qwen3-4B-Base: 4 billion parameters (its name and model card; docs/model_card_v0_5.md)"},
    {"printed": "4", "where": ["text:intro"],
     "origin": "Gorenshtein et al. 2026: the review's models of 4 billion parameters or fewer (its abstract, read at its source)"},
    {"printed": "11", "where": ["text:intro"],
     "origin": "Gorenshtein et al. 2026: the number of eligible studies (its abstract, read at its source; docs/prior_art.md)"},
    {"printed": "0.511", "where": ["text:related", "text:reading"],
     "origin": "DeYoung et al. 2020, Table 1: a classifier given the prompt alone (macro-F1); the readings fixed in the update cite it (docs/osf_update_session9.md)"},
    {"printed": "1", "where": ["text:related"], "origin": "the number of DeYoung et al. 2020's table that gives 0.511 (Table 1)"},
    {"printed": "0.86", "where": ["text:related"],
     "origin": "Buchmann, Liu and Gurevych 2024 (EMNLP), Table 3: macro-F1 86 on EI's test set, a fine-tuned model answering from the whole article (docs/fiorillo_v0_4.md and docs/fiorillo_v0_5.md, the published results beside G3; read at its source)"},
    {"printed": "0.87", "where": ["text:related"],
     "origin": "Buchmann, Liu and Gurevych 2024 (EMNLP), Table 3: macro-F1 87 from ten retrieved segments, and 87 for a commercial model on 100 test instances (the same notes)"},
    {"printed": "100", "where": ["text:related"],
     "origin": "Buchmann, Liu and Gurevych 2024 (EMNLP): the 100 test instances of their commercial model's run (the same notes)"},
    {"printed": "0", "where": ["text:results"], "origin": "the registered check's definition: a true answer whose probability prints as 0 (docs/osf_update_session9.md, Also reported)"},
    {"printed": "1", "where": ["text:results"], "origin": "the ideal calibration slope (comparison.json's calibration_slope_intercept.method)"},
    {"printed": "5", "where": ["text:handoff"], "origin": "the curves' step, 5 percent, fixed in the update (docs/osf_update_session9.md, The hand-off analyses)"},
    {"printed": "0.174", "where": ["text:appD"],
     "origin": "the large model's PubMedQA seconds an item if session 8's time were shared by prompt tokens (derived in docs/fiorillo_v0_4.md, Method, Hand-off, and docs/closeout_session8.md, step 12)"},
    {"printed": "1 October 2027", "where": ["text:appB"],
     "origin": "the embargo's end date with which the registration was filed (docs/osf_prereg_fiorillo_v0_4.md, As filed)"},
    {"printed": "30", "where": ["text:data", "text:results", "text:appA", "text:appC", "table:readmission"],
     "origin": "the readmission question's definition: readmission within 30 days, after 30 days, or none (UCI Diabetes 130-US hospitals data; docs/model_card_v0_5.md)"},
    {"printed": "16", "where": ["text:model"],
     "origin": "the LoRA rank of every run (docs/osf_prereg_fiorillo_v0_4.md, Study design; docs/osf_update_session9.md, The recipes)"},
    {"printed": "32", "where": ["text:model"],
     "origin": "the LoRA alpha of every run (docs/osf_prereg_fiorillo_v0_4.md, Study design; docs/osf_update_session9.md, The recipes)"},
    {"printed": "0.05", "where": ["text:model"],
     "origin": "the LoRA dropout of every run (docs/osf_prereg_fiorillo_v0_4.md, Study design; docs/osf_update_session9.md, The recipes)"},
    {"printed": "80", "where": ["text:tests"],
     "origin": "the power analysis's target, 80 percent (results/fiorillo_v0_5/power.json, real_gain_for_80_percent)"},
    {"printed": "0.60", "where": ["text:reading"],
     "origin": "the no-article reading's threshold, fixed in the update before the checks ran (docs/osf_update_session9.md, The readings)"},
    {"printed": "0.05", "where": ["text:reading"],
     "origin": "the title reading's threshold, fixed in the update before the checks ran (docs/osf_update_session9.md, The readings)"},
    {"printed": "2019", "where": ["text:reading", "text:limits"],
     "origin": "the year EI's repository became public (docs/osf_prereg_fiorillo_v0_4.md, Other: known contamination)"},
    {"printed": "10", "where": ["text:handoff", "table:handoff"], "origin": "a hand-off rate registered in the update (10, 20, 30 and 50 percent; docs/osf_update_session9.md)"},
    {"printed": "20", "where": ["text:handoff", "table:handoff"], "origin": "a hand-off rate registered in the update (docs/osf_update_session9.md)"},
    {"printed": "30", "where": ["text:handoff", "table:handoff"], "origin": "a hand-off rate registered in the update (docs/osf_update_session9.md)"},
    {"printed": "50", "where": ["text:handoff", "table:handoff"], "origin": "a hand-off rate registered in the update (docs/osf_update_session9.md)"},
    {"printed": "1,000", "where": ["text:handoff", "text:appB", "text:appD", "table:cost"],
     "origin": "the unit of the cost, per 1,000 questions, and its multiplier (docs/osf_update_session9.md, Also reported; comparison.json's cost_per_1000_questions.formula)"},
    {"printed": "3,600", "where": ["text:appD"], "origin": "the seconds in an hour (comparison.json's cost_per_1000_questions.formula)"},
    {"printed": "13", "where": ["text:appB"],
     "origin": "the corrections made before the registration was filed, 1 to 13 (docs/osf_prereg_fiorillo_v0_4.md, Other)"},
    {"printed": "1", "where": ["text:appC"], "origin": "the ideal calibration slope (comparison.json's calibration_slope_intercept.method)"},
    {"printed": "0", "where": ["text:appC"], "origin": "the ideal calibration intercept (comparison.json's calibration_slope_intercept.method)"},
    {"printed": "0", "where": ["table:bar"], "origin": "criteria 2 and 3's rule: the interval of the difference wholly below 0 (docs/osf_update_session9.md, Power and the bar)"},
    {"printed": "1", "where": ["table:bar"], "origin": "the criterion's number, 1 to 4 (H1 to H4 in the registration; criteria 1 to 4 in the update)"},
    {"printed": "2", "where": ["table:bar"], "origin": "the criterion's number, 1 to 4 (H1 to H4 in the registration; criteria 1 to 4 in the update)"},
    {"printed": "3", "where": ["table:bar"], "origin": "the criterion's number, 1 to 4 (H1 to H4 in the registration; criteria 1 to 4 in the update)"},
    {"printed": "4", "where": ["table:bar"], "origin": "the criterion's number, 1 to 4 (H1 to H4 in the registration; criteria 1 to 4 in the update)"},
    {"printed": "00:36:37 UTC", "where": ["text:tests"],
     "origin": "the registration's filing time as OSF recorded it (docs/osf_prereg_fiorillo_v0_4.md, As filed; OSF's record of https://osf.io/kaxmn/, read at its source, docs/preprint/build_log.md, row 29)"},
    {"printed": "2 October 2026", "where": ["text:appB"],
     "origin": "the day the release was made public (docs/log.md, 2 October 2026: Hugging Face's API showed johanneli/fiorillo-v0.5 public at 23:50:15 UTC)"},
    {"printed": "01:06:40 UTC", "where": ["text:appB"],
     "origin": "the time OSF showed the registration public, not embargoed (docs/log.md, the entry of 3 October 2026 written at 01:18 UTC)"},
    {"printed": "3 October 2026", "where": ["text:appB"],
     "origin": "the day OSF showed the registration public (docs/log.md, the entry of 3 October 2026 written at 01:18 UTC)"},
]

# Dates and times printed from results files: (where, file, key, format), format date, time_minutes or time_seconds.
DATES = [
    ("text:tests", RC5, "osf_update.filed_utc", "date"),
    ("text:tests", RC5, "osf_update.filed_utc", "time_minutes"),
    ("text:tests", RC5, "time_utc", "date"),
    ("text:tests", RC5, "time_utc", "time_seconds"),
]


def date_text(t, form):
    """A date or time as the preprint prints it: "day Month year" (no leading zero), "HH:MM UTC" or "HH:MM:SS UTC"."""
    if form == "date":
        return f"{t.day} {t.strftime('%B')} {t.year}"
    if form == "time_minutes":
        return t.strftime("%H:%M") + " UTC"
    if form == "time_seconds":
        return t.strftime("%H:%M:%S") + " UTC"
    raise ValueError(form)


def text_numbers(S):
    for section, items in TEXT.items():
        for rel, key, digits, scale, thousands in items:
            try:
                S.num(rel, key, digits, f"text:{section}", scale, thousands)
            except (KeyError, TypeError, IndexError) as e:
                raise SystemExit(f"TEXT[{section}]: {rel} has no {key} ({e})")
    derived_text(S)
    for where, rel, key, form in DATES:
        v = S.value(rel, key)
        t = dt.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
        S.dates.append({"printed": date_text(t, form), "file": rel, "key": key, "value": v, "format": form, "where": where})
    # The specialist's temperature: calibration_s9.json keeps it as s9_ei_wavg_s8rule's temperature, the exponential of
    # s9_ei_wavg's fitted parameter under the same family; the two must agree.
    t = S.value(CAL, "s9_ei_wavg_s8rule.evidence_effect.families.temperature.temperature")
    p = S.value(CAL, "s9_ei_wavg.evidence_effect.families.temperature.params.0")
    if abs(math.exp(p) - t) > 1e-9:
        raise SystemExit(f"the temperature {t} is not exp({p})")
    # license_recheck.csv and license_recheck.json must agree, since the text's counts come from the JSON file.
    rows = S.csv_rows(LIC_CSV)
    lic = S.data(LIC)
    if len(rows) != lic["articles"]:
        raise SystemExit(f"{LIC_CSV} has {len(rows)} rows, {LIC} counts {lic['articles']} articles")
    for split, b in lic["by_split"].items():
        mine = [r for r in rows if split in r["splits"].split()]
        if len(mine) != b["articles"] or sum(r["clean"] == "true" for r in mine) != b["clean_articles"]:
            raise SystemExit(f"{LIC_CSV} and {LIC} disagree on the {split} split")


def tables(root=ROOT):
    root = Path(root)
    S = Sources(root)
    for f in (tab_main_ei, tab_bar, tab_published, tab_k2, tab_reading, tab_pubmedqa, tab_readmission, tab_runs, tab_answers, tab_cost,
              tab_k1_answers, tab_handoff, tab_pqa_answers):
        f(S, root)
    text_numbers(S)
    data = S.write()
    n = sum(1 for e in data["entries"] if e["where"].startswith("table:"))
    t = sum(1 for e in data["entries"] if e["where"].startswith("text:"))
    print(f"wrote {OUT_TABLES}/ (13 fragments) and {SOURCES} ({n} table numbers, {t} text numbers, {len(data['dates'])} dates)",
          flush=True)
    return data


# ---------------------------------------------------------------- figures

def style():
    import matplotlib
    matplotlib.use("pdf")
    from matplotlib import font_manager
    for name in ("texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf",
                 "texgyreheros-regular.otf", "texgyreheros-bold.otf"):
        p = Path(FONT_DIR) / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros"], "font.serif": ["TeX Gyre Termes"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 3,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 0.9})
    return plt


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def reliability_bins(root, bins):
    """The reported ECE's bins, assigned as src/eval/metrics.py's ece() assigns them: equal-width bins on the top
    answer's probability, bin i holding edges[i] < p <= edges[i + 1] and bin 0 including 0 (numpy's searchsorted on the
    inner edges, side left), and the top answer by argmax (the first of tied answers)."""
    import numpy as np
    with open(Path(root) / PRED, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    edges = np.linspace(0, 1, bins + 1)
    out = [{"n": 0, "conf": 0.0, "correct": 0} for _ in range(bins)]
    for r in rows:
        p = np.array([float(r["p0"]), float(r["p1"]), float(r["p2"])])
        top = int(p.argmax())
        conf = float(p[top])
        b = int(np.searchsorted(edges[1:-1], conf, side="left"))
        out[b]["n"] += 1
        out[b]["conf"] += conf
        out[b]["correct"] += int(top == int(r["label"]))
    for b, o in enumerate(out):
        o["lo_edge"], o["hi_edge"] = edges[b], edges[b + 1]
        if o["n"]:
            o["mean_conf"] = o["conf"] / o["n"]
            o["accuracy"] = o["correct"] / o["n"]
            o["wilson"] = wilson(o["correct"], o["n"])
    return [o for o in out if o["n"]], len(rows)


def fig_reliability(S, root, plt):
    bins = int(S.value(C5, "settings.ece_bins"))
    used, n = reliability_bins(root, bins)
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--", zorder=1)
    ax.text(0.98, 0.04, "Dashed line: the probability\nequals the share that are right", fontsize=6.5, color="0.35",
            ha="right", va="bottom")
    for o in used:
        lo, hi = o["wilson"]
        ax.plot([o["mean_conf"], o["mean_conf"]], [lo, hi], color="0.55", lw=1.0, zorder=2)
        ax.plot(o["mean_conf"], o["accuracy"], "o", color="black", ms=3.2, zorder=3)
        ax.annotate(f"{o['n']:,}", (o["mean_conf"], o["accuracy"]), xytext=(-4, 5), textcoords="offset points", fontsize=6.5,
                    ha="right", color="0.25")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Probability of the top answer (mean in each bin)")
    ax.set_ylabel("Share of top answers that are right")
    ax.grid(False)
    fig.tight_layout()
    out = Path(root) / OUT_FIGURES / "reliability.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    for o in used:
        b = int(round(o["lo_edge"] * bins))
        S.derived(f"{o['n']:,}", o["n"], f"n of the reliability bin {b} of {PRED} ({bins} equal-width bins on the top answer's "
                  "probability)", "figure:reliability", 0, files=[PRED], thousands=True)
    return used, n


def fig_reading(S, root, plt):
    """Macro-F1 with the whole article, the title only and no article, for v0.4 and v0.5 (K4)."""
    w = "figure:reading"
    rows = []
    for sysk, sysm, label, marker in [("s8_ei", "s8_ei", "v0.4", "o"), ("v05", "s9_ei_wavg", "v0.5", "s")]:
        full = [float(S.num(C5, f"metrics.evidence_effect.{sysm}.macro_f1.{k}", 4, w)) for k in ("estimate", "low", "high")]
        title = [float(S.num(C5, f"k4.table[system={sysk},input=title,statistic=macro_f1].{k}", 4, w)) for k in ("estimate", "low", "high")]
        none = [float(S.num(C5, f"k4.table[system={sysk},input=none,statistic=macro_f1].{k}", 4, w)) for k in ("estimate", "low", "high")]
        rows.append((label, marker, [("Whole article", full), ("Title only", title), ("No article", none)]))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ys = {"Whole article": 2, "Title only": 1, "No article": 0}
    for j, (label, marker, vals) in enumerate(rows):
        off = 0.12 if j == 0 else -0.12
        for name, (e, lo, hi) in vals:
            y = ys[name] + off
            ax.plot([lo, hi], [y, y], color="0.55", lw=1.0)
            ax.plot(e, y, marker, ms=3.6, color="black", mfc="white" if j == 0 else "black")
        ax.annotate(label, (vals[0][1][0], ys["Whole article"] + off), xytext=(6, -2), textcoords="offset points", fontsize=6.5)
    ax.axvline(0.511, color="0.6", lw=0.7, ls="--")
    ax.annotate("prompt alone, published", (0.511, -0.45), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color="0.35")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["No article", "Title only", "Whole article"])
    ax.set_ylim(-0.6, 2.5)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("Macro-F1 on the test prompts (95 percent interval)")
    fig.tight_layout()
    out = Path(root) / OUT_FIGURES / "reading.pdf"
    fig.savefig(out)
    plt.close(fig)


def figures(root=ROOT):
    root = Path(root)
    plt = style()
    S = Sources(root)
    used, n = fig_reliability(S, root, plt)
    fig_reading(S, root, plt)
    import matplotlib
    data = S.write({"figures": {"matplotlib": matplotlib.__version__, "reliability_bins": [
        {k: v for k, v in o.items() if k in ("lo_edge", "hi_edge", "n", "mean_conf", "accuracy", "wilson")} for o in used],
        "reliability_prompts": n,
        "not_drawn": {"handoff": ("the risk-coverage curves (K7) are not drawn: the four registered rates in the text read "
                                  "in five seconds, and the curve between them carries no registered reading")}}})
    print(f"wrote {OUT_FIGURES}/reliability.pdf and reading.pdf; {SOURCES} updated", flush=True)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("tables", "figures"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    (tables if args.command == "tables" else figures)(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

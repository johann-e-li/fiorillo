"""Session 7 (docs/prompt_session7.md, step 15): the figure check of the Results section of docs/fiorillo_v0_3.md.

It reads the document as written and finds every number in its Results section (from the line "## Results (test split)"
to the next line that starts with "## "), and compares each with its source at the printed precision: a number printed
with four decimals passes when it is the source value rounded to four decimals, a count when it equals the source, a time
when it equals the source's time of day. Table cells are matched to their sources by the table's heading ("### Table N.")
and its rows, which TABLES below fixes for each table. Numbers in the prose are matched, in order, to the source keys of
the line's template (PROSE below): a line is found by its shape, its text with every number replaced, so a line whose
numbers were changed still finds its keys. Three lines are checked whole against their source text (VERBATIM below).
Each key is resolved here, from the results files, by this script's own code. Every reading ("candidate better", "no
difference was detected", PASS, FAIL and so on) is checked against its interval and the direction of its metric, or
against the rule's own result.

Before the document is checked, a self-test plants errors in copies of it, one at a time (PLANTED below), and the script
stops unless each one is reported. Then the document itself is checked, and the script stops unless every number and
reading passes. It writes results/fiorillo_v0_3/figure_sources.json (every number: where it is, as written, its source
and the source's value) and docs/fiorillo_v0_3_figure_check.md (every number and reading, one row each, PASS or MISMATCH).

The same tables and templates also render the section's figures (render_table and render_prose), so each figure is
written and checked from one definition of its source; the check reads the document itself, not the rendering.

Usage: python3 src/eval/check_fiorillo_v0_3_figures.py
"""
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "fiorillo_v0_3.md"
OUT = ROOT / "docs" / "fiorillo_v0_3_figure_check.md"
SOURCES_OUT = ROOT / "results" / "fiorillo_v0_3" / "figure_sources.json"
START = "## Results (test split)"

LOWER = {"log_loss", "brier", "ece"}
UNRANKED = {"calibration_intercept_30d", "calibration_slope_30d"}
METRIC_LABEL = {
    "log_loss": "Log loss", "brier": "Brier score", "ece": "ECE", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
    "macro_f1_tuned": "Macro-F1, decision weights", "quadratic_weighted_kappa": "Quadratic weighted kappa",
    "auroc_30d": "AUROC, 30 days", "calibration_intercept_30d": "Calibration intercept, 30 days",
    "calibration_slope_30d": "Calibration slope, 30 days"}
EV_LABEL = {"pubmedqa_answer": "PubMedQA, three answers", "pubmedqa_yes_no": "PubMedQA, yes or no",
            "readmit_window": "Readmission"}
BASE_METRICS = ("log_loss", "brier", "ece", "accuracy", "macro_f1")
READMIT_EXTRA = ("macro_f1_tuned", "quadratic_weighted_kappa", "auroc_30d", "calibration_intercept_30d", "calibration_slope_30d")
EVS = ("pubmedqa_answer", "pubmedqa_yes_no", "readmit_window")
SYSTEM_ORDER = ["prior", "arm_laya", "arm_laya-raw", "arm_laya_s7cal", "arm_laya_s7cal_temperature", "s5_final_fresh_head",
                "s5_final_fresh_head-raw", "s7_a3", "s7_a3_temperature", "s7_a3-raw", "s7_b3", "s7_b3_temperature", "s7_b3-raw",
                "duan", "duan-raw", "duan_ourwording", "duan_ourwording-raw", "lightgbm", "lightgbm_tuned", "blend_v0_1", "blend_v0_3"]

# identifiers and definitional constants that are not figures (listed in the check document)
N = r"\d+(?:,\d{3})*"
SKIP = [r"`[^`]*`", r"\S*[_/]\S*", r"\bv0\.[0-3]\b", r"\bQwen3-4B(?:-Base)?\b", r"\bQwen3\b", r"\bI-\d{2}\b", r"\bE[1-7]\b",
        rf"\bTables? {N}(?: (?:and|to) {N})?", rf"\bcorrections? {N}(?: (?:and|to) {N})?",
        rf"\bcriteri(?:on|a) {N}(?: (?:and|to) {N})?", rf"\bsessions? {N}(?:'s)?(?: (?:and|to) {N})?",
        rf"\bphases? {N}(?:'s)?(?: (?:and|to) {N})?", rf"\bsteps? {N}(?: (?:and|to) {N})?",
        r"\b\d{1,2} (?:September|October) 2026\b", r"\bSHA-256\b", r"\b95 percent\b", r"\b97\.5 percent\b", r"\b95th\b",
        r"\b30 days\b", r"\bprints? as 0\b", r"\b13-word\b", r"\blevels? [0-2](?:, [0-2])*(?: and [0-2])?\b"]
SKIP_RE = re.compile("|".join(SKIP), re.I)
TOKEN_RE = re.compile(r"\b\d{2}:\d{2}(?::\d{2})?\b|(?<![\w.])\$?-?\d+(?:,\d{3})*(?:\.\d+)?")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f4(x):
    s = f"{x:.4f}"
    return "0.0000" if s == "-0.0000" else s


def fmt(value, how):
    """How a source value is printed in the document."""
    if value is None:
        return "not computed"
    if how == "int":
        return f"{int(value):,}"
    if how == "time":
        return str(value)[11:19]
    if how == "str":
        return str(value)
    if how == "pct1":
        return f"{100 * float(value):.1f}"
    if how.startswith("f"):
        d = int(how[1:])
        s = f"{float(value):.{d}f}"
        return s.lstrip("-") if float(s) == 0 else s
    raise ValueError(how)


def reading(metric, d, system_better, reference_better):
    """The reading of a difference 'system minus reference' from its 95 percent interval."""
    if metric in UNRANKED:
        return "not ranked"
    if d["low"] is None or d["high"] is None:
        return "not computed"
    if d["low"] <= 0 <= d["high"]:
        return "no difference was detected"
    positive = d["estimate"] > 0
    if metric in LOWER:
        return reference_better if positive else system_better
    return system_better if positive else reference_better


class Sources:
    """Every source the Results section cites, read from the repository."""

    def __init__(self, root=ROOT):
        root = Path(root)
        r = root / "results" / "fiorillo_v0_3"
        self.cmp = read_json(r / "comparison.json")
        self.rc = read_json(r / "release_candidate.json")
        self.tpl = read_json(r / "testpreds_log.json")
        self.sel = read_json(r / "selection.json")
        self.cal = read_json(r / "calibration_s7.json")
        self.lab = read_json(r / "labeler.json")
        self.pow = read_json(r / "power.json")
        self.trees = read_json(r / "trees.json")
        self.zeros = read_json(r / "zeros.json")
        self.hand = read_json(r / "handoff_test.json")
        self.blend = {n: read_json(r / f"{n}.json") for n in ("blend_v0_1", "blend_v0_3") if (r / f"{n}.json").exists()}
        self.mac = read_json(r / "mac_training_check.json") if (r / "mac_training_check.json").exists() else None
        self.met = {p.name[:-len("_metrics.json")]: read_json(p) for p in sorted((r / "metrics").glob("*_metrics.json"))}
        self.cand = self.rc.get("system")
        self.der = self.derive(root)

    @staticmethod
    def derive(root):
        out = {}
        pub = root / "data" / "processed" / "public"
        for q in ("pubmedqa_answer", "readmit_window"):
            labels = [json.loads(line)["label"] for line in open(pub / f"{q}_test.jsonl", encoding="utf-8")]
            out[f"{q}_test_items"] = (len(labels), f"data/processed/public/{q}_test.jsonl: records")
            for c in range(3):
                out[f"{q}_test_label_{c}"] = (sum(1 for x in labels if x == c), f"data/processed/public/{q}_test.jsonl: records with label {c}")
        out["pubmedqa_yes_no_items"] = (out["pubmedqa_answer_test_label_0"][0] + out["pubmedqa_answer_test_label_1"][0],
                                        "data/processed/public/pubmedqa_answer_test.jsonl: records labeled yes or no")
        return out

    def comparison(self, section, ev, system, reference, metric):
        system = self.cand if system == "@cand" else system
        reference = self.cand if reference == "@cand" else reference
        hits = [(i, c) for i, c in enumerate(self.cmp["comparisons"])
                if (c["section"], c["evaluation"], c["system"], c["reference"], c["metric"]) == (section, ev, system, reference, metric)]
        if len(hits) != 1:
            raise KeyError(f"no single comparison {section} {ev} {system} minus {reference} {metric} ({len(hits)} found)")
        return hits[0]

    @staticmethod
    def walk(obj, path, name):
        """(source description, value) at a path of keys and list positions separated by '/'; a last step ending in
        '_abs' gives the absolute value of that step's value."""
        steps = path.split("/") if path else []
        absolute = bool(steps) and steps[-1].endswith("_abs")
        if absolute:
            steps[-1] = steps[-1][:-len("_abs")]
        v = obj
        for step in steps:
            v = v[int(step)] if isinstance(v, list) else v[step]
        where = f"{name} {'.'.join(steps)}" if steps else name
        if absolute:
            return where + ", absolute value", abs(v)
        return where, v

    def resolve(self, key):
        """(source description, value) for a key 'kind|arguments'."""
        kind, *a = key.split("|")
        a = [self.cand if x == "@cand" else x for x in a]
        cmp = self.cmp
        if kind == "M":  # M|evaluation|system|metric|part
            return f"comparison.json metrics.{a[0]}.{a[1]}.{a[2]}.{a[3]}", cmp["metrics"][a[0]][a[1]][a[2]][a[3]]
        if kind == "C":  # C|section|evaluation|system|reference|metric|part
            i, c = self.comparison(*a[:5])
            if a[5] in ("system_estimate", "reference_estimate"):
                return f"comparison.json comparisons[{i}].{a[5]}", c[a[5]]
            where, v = self.walk(c["difference"], a[5], "difference")
            return f"comparison.json comparisons[{i}] ({a[0]}, {a[1]}, {c['system']} minus {c['reference']}, {a[4]}).{where}", v
        tables = {"SET": (cmp["settings"], "comparison.json settings"),
                  "C4": (cmp["criterion_4_intervals"], "comparison.json criterion_4_intervals"),
                  "RB": (cmp["release_bar"], "comparison.json release_bar"),
                  "MAY": (cmp["maybe"], "comparison.json maybe"),
                  "H": (self.hand["systems"], "handoff_test.json systems"),
                  "LAB": (self.lab, "labeler.json"), "POW": (self.pow, "power.json"), "TR": (self.trees, "trees.json"),
                  "Z": (self.zeros["rows_whose_true_answer_prints_as_0"], "zeros.json rows_whose_true_answer_prints_as_0"),
                  "SEL": (self.sel, "selection.json"), "CAL": (self.cal, "calibration_s7.json"),
                  "RC": (self.rc, "release_candidate.json"), "TPL": (self.tpl, "testpreds_log.json")}
        if kind in tables:
            obj, name = tables[kind]
            return self.walk(obj, "/".join(a), name)
        if kind == "BL":
            return self.walk(self.blend[a[0]], "/".join(a[1:]), f"{a[0]}.json")
        if kind == "MET":
            return self.walk(self.met[a[0]], "/".join(a[1:]), f"metrics/{a[0]}_metrics.json")
        if kind == "MAC":
            if self.mac is None:
                return "mac_training_check.json (not written)", None
            return self.walk(self.mac, "/".join(a), "mac_training_check.json")
        if kind == "D":
            v, how = self.der[a[0]]
            return f"derived: {how}", v
        raise KeyError(key)


# ---------------------------------------------------------------- the tables
# A table is (number, title, header, rows); a row is a list of cells, each a label (a plain string, checked as text) or:
#   ("num", key, how)              one number
#   ("iv", key_without_part)       "estimate (low to high)", four decimals, from key|estimate, key|low and key|high
#   ("read", C_key_without_part, metric, system_better, reference_better)   the reading of a difference's interval
#   ("str", key)                   a text value from a source (checked as text)
#   ("pf", key)                    PASS or FAIL from a source's true or false
#   ("map", key, {value: text})    a source's value printed as the text the mapping gives it (checked as text)


def present(src, ev, systems):
    return [s for s in systems if s in src.cmp["metrics"].get(ev, {})]


def table_metrics(src, number, ev, metrics, title):
    rows = [[f"`{s}`"] + [("iv", f"M|{ev}|{s}|{m}") for m in metrics] for s in present(src, ev, SYSTEM_ORDER)]
    return (number, title, ["System"] + [METRIC_LABEL[m] for m in metrics], rows)


def table_diffs(src, number, title, specs):
    """specs: (section, evaluation, system, reference, metrics, system_better, reference_better)."""
    rows = []
    for section, ev, system, reference, metrics, sb, rb in specs:
        for m in metrics:
            try:
                src.comparison(section, ev, system, reference, m)
            except KeyError:
                continue
            base = f"C|{section}|{ev}|{system}|{reference}|{m}"
            rows.append([section, EV_LABEL[ev], METRIC_LABEL[m], ("iv", base), ("read", base, m, sb, rb)])
    return (number, title, ["Comparison", "Evaluation", "Metric", "Difference (95 percent interval)", "Reading"], rows)


def metrics_of(ev):
    return BASE_METRICS + (READMIT_EXTRA if ev == "readmit_window" else ())


def kept_index(m):
    hist = m["validation_history"]
    if m.get("keep") != "best":
        return len(hist) - 1
    return min(range(len(hist)), key=lambda i: (hist[i]["selection_log_loss"], i))


def tables(src):
    cand = src.cand
    out = []
    rows = []
    for s in ("s7_a3", "s7_b3", "s7_a3_orig"):
        if s in src.met:
            k = kept_index(src.met[s])
            rows.append([f"`{s}`", ("num", f"MET|{s}|validation_history/{k}/step", "int"),
                         ("num", f"MET|{s}|kept_validation_selection_log_loss_exact", "f5"),
                         ("num", f"MET|{s}|validation_history/{k}/per_question/pubmedqa_answer/log_loss", "f4"),
                         ("num", f"MET|{s}|validation_history/{k}/per_question/readmit_window/log_loss", "f4"),
                         ("num", f"MET|{s}|optimizer_steps", "int")])
    out.append((1, "Selection on validation (before calibration)",
                ["System", "Kept step", "Selection score", "PubMedQA log loss", "Readmission log loss", "Optimizer steps"], rows))
    rows = []
    for s in ("s7_a3", "s7_b3"):
        for q in ("pubmedqa_answer", "readmit_window"):
            if s in src.cal and q in src.cal[s]:
                ch = src.cal[s][q]["chosen"]
                rows.append([f"`{s}`", EV_LABEL[q],
                             ("num", f"CAL|{s}|{q}|families/temperature/out_of_fold_log_loss/mean", "f4"),
                             ("num", f"CAL|{s}|{q}|families/temperature_intercepts/out_of_fold_log_loss/mean", "f4"),
                             ("str", f"CAL|{s}|{q}|chosen"),
                             ("num", f"CAL|{s}|{q}|families/{ch}/temperature", "f3"),
                             ("num", f"CAL|{s}|{q}|families/{ch}/intercepts/1", "f3"),
                             ("num", f"CAL|{s}|{q}|families/{ch}/intercepts/2", "f3")])
    out.append((2, "Calibration chosen on validation", ["System", "Question", "Out-of-fold log loss, temperature",
                "Out-of-fold log loss, temperature and intercepts", "Chosen", "Temperature", "Intercept, no or level 1",
                "Intercept, maybe or level 2"], rows))
    out.append(table_metrics(src, 3, "pubmedqa_answer", BASE_METRICS, "PubMedQA, three answers (test split)"))
    out.append(table_metrics(src, 4, "pubmedqa_yes_no", BASE_METRICS, "PubMedQA, yes or no items (test split)"))
    out.append(table_metrics(src, 5, "readmit_window", BASE_METRICS, "Readmission (test split), part one"))
    out.append(table_metrics(src, 6, "readmit_window", READMIT_EXTRA, "Readmission (test split), part two"))
    if cand:
        specs = [("E1", ev, cand, "arm_laya", metrics_of(ev), "candidate better", "v0.1 better") for ev in EVS]
        out.append(table_diffs(src, 7, f"E1: the candidate ({cand}) minus v0.1 as released", specs))
    specs = []
    for ev in EVS:
        specs.append(("E2", ev, "s7_a3_temperature", "s5_final_fresh_head", metrics_of(ev), "phased better", "one phase better"))
        specs.append(("E3", ev, "s7_b3_temperature", "s7_a3_temperature", metrics_of(ev), "arm B better", "arm A better"))
    out.append(table_diffs(src, 12, "E2 and E3", specs))
    specs = [("E7", ev, "arm_laya_s7cal", "arm_laya", metrics_of(ev), "session 7 calibration better", "v0.1 as released better")
             for ev in EVS]
    out.append(table_diffs(src, 14, "E7: v0.1 with session 7's calibration minus v0.1 as released", specs))
    if cand:
        rows = []
        for c in src.cmp["comparisons"]:
            if c["section"] == "E4" and c["system"] == cand:
                base = f"C|E4|{c['evaluation']}|{cand}|{c['reference']}|{c['metric']}"
                rows.append([f"`{c['reference']}`", EV_LABEL[c["evaluation"]], METRIC_LABEL[c["metric"]],
                             ("num", base + "|system_estimate", "f4"), ("num", base + "|reference_estimate", "f4"),
                             ("iv", base), ("read", base, c["metric"], "candidate better", "Duan better")])
        out.append((10, "E4: the candidate minus Duan", ["Duan variant", "Evaluation", "Metric", "Candidate", "Duan",
                    "Difference (95 percent interval)", "Reading"], rows))
    rows = []
    for ev in EVS:
        for s in present(src, ev, [x for x in SYSTEM_ORDER if x != "prior"]):
            try:
                src.comparison("E5", ev, s, "prior", "log_loss")
            except KeyError:
                continue
            base = f"C|E5|{ev}|{s}|prior|log_loss"
            rows.append([f"`{s}`", EV_LABEL[ev], ("iv", base), ("read", base, "log_loss", "better than the prior", "the prior better")])
    out.append((11, "E5: every system minus the prior, log loss", ["System", "Evaluation", "Difference (95 percent interval)", "Reading"], rows))
    rows = []
    for c in src.cmp["comparisons"]:
        if c["section"] == "E6":
            base = f"C|E6|readmit_window|{c['system']}|{c['reference']}|{c['metric']}"
            rows.append([f"`{c['system']}` minus `{c['reference']}`", METRIC_LABEL[c["metric"]], ("iv", base),
                         ("read", base, c["metric"], f"{c['system']} better", f"{c['reference']} better")])
    out.append((13, "E6: readmission trees and blends", ["Comparison", "Metric", "Difference (95 percent interval)", "Reading"], rows))
    if cand and src.cmp.get("criterion_4_intervals"):
        rows = []
        for q in ("pubmedqa_answer", "readmit_window"):
            rows.append([EV_LABEL[q], ("num", f"C4|{q}|estimate", "f4"), ("num", f"C4|{q}|low_95", "f4"), ("num", f"C4|{q}|high_95", "f4"),
                         ("num", f"C4|{q}|low_97_5", "f4"), ("num", f"C4|{q}|high_97_5", "f4")])
        out.append((8, "Criterion 4: the candidate minus v0.1 in log loss, with criterion 4's resamples",
                    ["Question", "Difference", "95 percent interval, low", "95 percent interval, high", "97.5 percent interval, low",
                     "97.5 percent interval, high"], rows))
        rows = [[f"Criterion {k}", ("str", f"RB|criteria/{k}/result")] for k in ("1", "2", "3", "4")]
        rows.append(["Criterion 4, route (a)", ("pf", "RB|criteria/4/values/routes/a/passes")])
        rows.append(["Criterion 4, route (b)", ("pf", "RB|criteria/4/values/routes/b/passes")])
        rows.append(["The bar", ("str", "RB|overall")])
        out.append((9, "The release bar", ["Criterion", "Result"], rows))
    rows = [[f"`{s}`", ("num", f"MAY|{s}|top_answer_maybe", "int"), ("num", f"MAY|{s}|f1_maybe", "f4")]
            for s in SYSTEM_ORDER if s in src.cmp["maybe"]]
    out.append((15, "Maybe on the PubMedQA test items (descriptive)", ["System", "Items with maybe on top", "F1 of maybe"], rows))
    rows = []
    for s in SYSTEM_ORDER:
        for q, h in src.hand["systems"].get(s, {}).items():
            if "missing" in h:
                continue
            rows.append([f"`{s}`", f"`{q}`", ("iv", f"H|{s}|{q}|answered_accuracy_100"), ("iv", f"H|{s}|{q}|answered_accuracy_75"),
                         ("iv", f"H|{s}|{q}|answered_accuracy_50"), ("iv", f"H|{s}|{q}|handed_off_accuracy_50"), ("iv", f"H|{s}|{q}|aurc")])
    out.append((16, "Hand-off on the test split (secondary, descriptive)", ["System", "Question", "Accuracy, all answered",
                "Accuracy, most confident three quarters", "Accuracy, most confident half", "Accuracy, half handed off", "AURC"], rows))
    rows = []
    for s, per_split in src.zeros["rows_whose_true_answer_prints_as_0"].items():
        for split, per_q in per_split.items():
            for q, z in per_q.items():
                if z["true_answer_printed_zero"]:
                    rows.append([f"`{s}`", split, f"`{q}`", ("num", f"Z|{s}|{split}|{q}|rows", "int"),
                                 ("num", f"Z|{s}|{split}|{q}|true_answer_printed_zero", "int")])
    out.append((17, "Prediction rows whose probability of the true answer prints as 0", ["System", "Split", "Question", "Rows",
                "True answer printed as 0"], rows))
    if src.mac is not None:
        onoff = {True: "on", False: "off"}
        rows = []
        for arm in ("a", "b"):
            if arm in src.mac["arms"] and src.mac["arms"][arm].get("fit"):
                k = f"MAC|arms/{arm}"
                rows.append([f"Arm {arm.upper()}", ("num", f"{k}/micro_batch", "int"), ("num", f"{k}/accumulation", "int"),
                             ("map", f"{k}/gradient_checkpointing", onoff), ("str", f"{k}/precision"),
                             ("num", f"{k}/seconds_per_step", "f2"), ("num", f"{k}/peak_memory_gib", "f2"),
                             ("num", f"{k}/hours_per_100000_records", "f1"), ("num", f"{k}/time_utc", "time")])
        out.append((18, "The Mac-training check (step 5, with correction 6)", ["Arm", "Micro-batch", "Accumulation",
                    "Gradient checkpointing", "Precision", "Seconds per step", "Peak memory (GiB)", "Hours per 100,000 records",
                    "Measured (UTC)"], rows))
        rows = []
        for i, a in enumerate(src.mac["arms"].get("b", {}).get("attempts", [])):
            if "try_seconds" not in a:
                continue
            k = f"MAC|arms/b/attempts/{i}"
            rows.append([("num", f"{k}/micro_batch", "int"), ("map", f"{k}/gradient_checkpointing", onoff),
                         ("num", f"{k}/try_seconds", "f1"), ("map", f"{k}/fit", {True: "fit", False: "out of memory"}),
                         ("num", f"{k}/swap_used_gib_before", "f2"), ("num", f"{k}/swap_used_gib_after", "f2")])
        out.append((19, "Arm B's tries on the Mac, each in its own process under the memory cap (correction 6)", ["Micro-batch",
                    "Gradient checkpointing", "Try (seconds)", "Outcome", "Swap in use before (GiB)", "Swap in use after (GiB)"], rows))
    return out


def interval_text(src, base):
    parts = [src.resolve(f"{base}|{p}")[1] for p in ("estimate", "low", "high")]
    if parts[0] is None:
        return "not computed"
    if parts[1] is None:
        return f4(parts[0])
    return f"{f4(parts[0])} ({f4(parts[1])} to {f4(parts[2])})"


def cell_text(src, c):
    if isinstance(c, str):
        return c
    kind = c[0]
    if kind == "num":
        return fmt(src.resolve(c[1])[1], c[2])
    if kind == "iv":
        return interval_text(src, c[1])
    if kind == "read":
        d = {p: src.resolve(f"{c[1]}|{p}")[1] for p in ("estimate", "low", "high")}
        return reading(c[2], d, c[3], c[4])
    if kind == "str":
        return str(src.resolve(c[1])[1])
    if kind == "pf":
        return "PASS" if src.resolve(c[1])[1] else "FAIL"
    if kind == "map":
        return c[2][src.resolve(c[1])[1]]
    raise ValueError(c)


def render_table(src, t):
    number, title, header, rows = t
    lines = [f"### Table {number}. {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(cell_text(src, c) for c in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- the prose
# Each template is one line of the Results section, {{key:how}} marking a figure; every line of the section that holds a
# figure must match one template's shape (or be one of the VERBATIM lines).
C_E1_P = "C|E1|pubmedqa_answer|s7_a3|arm_laya|log_loss"
C_E1_R = "C|E1|readmit_window|s7_a3|arm_laya|log_loss"
C_E2_P = "C|E2|pubmedqa_answer|s7_a3_temperature|s5_final_fresh_head|log_loss"
C_E2_R = "C|E2|readmit_window|s7_a3_temperature|s5_final_fresh_head|log_loss"
C_E3_P = "C|E3|pubmedqa_answer|s7_b3_temperature|s7_a3_temperature|log_loss"
C_E3_R = "C|E3|readmit_window|s7_b3_temperature|s7_a3_temperature|log_loss"
C_E6_T = "C|E6|readmit_window|lightgbm_tuned|lightgbm|log_loss"
C_E6_B3T = "C|E6|readmit_window|blend_v0_3|lightgbm_tuned|log_loss"
C_E6_B3C = "C|E6|readmit_window|blend_v0_3|s7_a3|log_loss"
C_E6_B1 = "C|E6|readmit_window|blend_v0_1|arm_laya|log_loss"
C_E7_P = "C|E7|pubmedqa_answer|arm_laya_s7cal|arm_laya|log_loss"
C_E7_R = "C|E7|readmit_window|arm_laya_s7cal|arm_laya|log_loss"


def iv(base):
    return f"{{{{{base}|estimate:f4}}}} ({{{{{base}|low:f4}}}} to {{{{{base}|high:f4}}}})"


PROSE = [
    ("intro", "The test split was read once, after the freeze: release_candidate.json was written at {{RC|time_utc:time}} UTC, and the test "
     "stage started at {{TPL|testpreds_start_utc:time}} UTC and recorded that file's SHA-256 first. PubMedQA's test split has "
     "{{D|pubmedqa_answer_test_items:int}} items, {{D|pubmedqa_yes_no_items:int}} of them yes or no and {{D|pubmedqa_answer_test_label_2:int}} "
     "maybe; readmission's has {{D|readmit_window_test_items:int}} encounters, {{D|readmit_window_test_label_2:int}} of them readmitted within "
     "30 days. Every interval is a 95 percent interval from {{SET|resamples:int}} paired bootstrap resamples of test rows (the seed "
     "{{SET|seed:str}}), except criterion 4's, which use {{SET|resamples_criterion_4:int}}. A system's name alone is its chosen calibration, "
     "the suffix _temperature marks one temperature per question fitted on the same validation items, and -raw marks predictions before "
     "calibration."),
    ("selection", "On validation, s7_a3's selection score was {{SEL|candidate_rule/scores/s7_a3:f5}} and s7_b3's "
     "{{SEL|candidate_rule/scores/s7_b3:f4}}, so s7_a3 won the candidate rule, and its score is below the v0.2 candidate's "
     "{{SEL|candidate_rule/v0_2_candidate/score:f4}}: s7_a3 is the v0.3 candidate (Table 1). Its checkpoint at step 1,500 scored "
     "{{MET|s7_a3|validation_history/5/selection_log_loss:f4}}, close to the kept one at step 1,000, with readmission's log loss "
     "{{MET|s7_a3|validation_history/5/per_question/readmit_window/log_loss:f4}} against "
     "{{MET|s7_a3|validation_history/3/per_question/readmit_window/log_loss:f4}}. The first s7_a3, trained on the phase 3 that correction 5 "
     "replaced, had kept a score of {{MET|s7_a3_orig|kept_validation_selection_log_loss_exact:f5}}."),
    ("calibration", "The calibration rule chose temperature_intercepts for both arms and both questions (Table 2). The ECE limits fixed "
     "by the freeze are {{RC|ece_limits/pubmedqa_answer/limit:f4}} for PubMedQA (its 95th percentile, above the floor of "
     "{{RC|ece_limits/pubmedqa_answer/floor:f2}}) and {{RC|ece_limits/readmit_window/limit:f2}} for readmission (the floor; its 95th "
     "percentile is {{RC|ece_limits/readmit_window/p95:f4}})."),
    ("tables", "Tables 3 to 6 give every system in the comparisons on each evaluation. For ECE, the percentile interval does not always "
     "contain the estimate: v0.1's ECE on the three PubMedQA answers is {{M|pubmedqa_answer|arm_laya|ece|estimate:f4}}, with an interval "
     "of {{M|pubmedqa_answer|arm_laya|ece|low:f4}} to {{M|pubmedqa_answer|arm_laya|ece|high:f4}}; the intervals are reported as computed."),
    ("e1", "On PubMedQA's three answers, the candidate's log loss was {{M|pubmedqa_answer|s7_a3|log_loss|estimate:f4}} against v0.1's "
     "{{M|pubmedqa_answer|arm_laya|log_loss|estimate:f4}}, a difference of " + iv(C_E1_P) + ", and its accuracy "
     "{{M|pubmedqa_answer|s7_a3|accuracy|estimate:pct1}} percent against {{M|pubmedqa_answer|arm_laya|accuracy|estimate:pct1}} percent. "
     "On readmission its log loss was {{M|readmit_window|s7_a3|log_loss|estimate:f4}} against v0.1's "
     "{{M|readmit_window|arm_laya|log_loss|estimate:f4}}, a difference of " + iv(C_E1_R) + ", and its AUROC within 30 days "
     "{{M|readmit_window|s7_a3|auroc_30d|estimate:f4}} against {{M|readmit_window|arm_laya|auroc_30d|estimate:f4}}: v0.1 was better "
     "on readmission's log loss, Brier score and AUROC (Table 7)."),
    ("bar", "Criteria 1 to 3 passed (Table 9). Criterion 1: ECE {{RB|criteria/1/values/pubmedqa_answer/ece:f4}} against the limit "
     "{{RB|criteria/1/values/pubmedqa_answer/limit:f4}} on PubMedQA, and {{RB|criteria/1/values/readmit_window/ece:f4}} against "
     "{{RB|criteria/1/values/readmit_window/limit:f2}} on readmission. Criterion 2: log loss below the prior's on both questions, each "
     "interval of the difference excluding zero (Table 11). Criterion 3: macro-F1 on the yes or no items "
     "{{RB|criteria/3/values/pubmedqa_yes_no/s7_a3:f4}} against Duan's {{RB|criteria/3/values/pubmedqa_yes_no/duan:f4}}, and on "
     "readmission, with decision weights, {{RB|criteria/3/values/readmit_window/s7_a3:f4}} against "
     "{{RB|criteria/3/values/readmit_window/duan:f4}} (Table 10)."),
    ("c4", "Criterion 4 failed (Table 8). With {{SET|resamples_criterion_4:int}} resamples, PubMedQA's difference was "
     "{{C4|pubmedqa_answer|estimate:f4}}, with a 97.5 percent interval of {{C4|pubmedqa_answer|low_97_5:f4}} to "
     "{{C4|pubmedqa_answer|high_97_5:f4}}, wholly below zero, so route (a)'s own condition held. Readmission's difference was "
     "{{C4|readmit_window|estimate:f4}}, with a 95 percent interval of {{C4|readmit_window|low_95:f4}} to {{C4|readmit_window|high_95:f4}}: "
     "its lower end is above zero, so readmission was significantly worse and route (a) failed. Route (b) needed readmission's 97.5 percent "
     "interval wholly below zero; it was {{C4|readmit_window|low_97_5:f4}} to {{C4|readmit_window|high_97_5:f4}}. Before the runs, power.json "
     "put route (b)'s pass line at an estimate of about {{POW|routes/route_b/by_se_scale/1.0/pass_line:f4}} (derived)."),
    ("e2e3", "E2 (Table 12) compares the phased recipe with session 5's one mixed phase on reader v1, both with one temperature per question: the "
     "difference in log loss was " + iv(C_E2_P) + " on PubMedQA's three answers and " + iv(C_E2_R) + " on readmission, the phased "
     "recipe better on PubMedQA and worse on readmission. E3 compares the two bases under the same recipe and calibration: the difference "
     "was " + iv(C_E3_P) + " on PubMedQA's three answers, arm B better; on readmission no difference was detected ("
     "{{" + C_E3_R + "|estimate:f4}}, {{" + C_E3_R + "|low:f4}} to {{" + C_E3_R + "|high:f4}}): a gain for arm B of more than "
     "{{" + C_E3_R + "|low_abs:f4}} and a loss of more than {{" + C_E3_R + "|high:f4}} are excluded. On validation, arm B's selection "
     "score was the higher of the two (Table 1), so the fixed rule made arm A's s7_a3 the candidate; on the test split, s7_b3's log loss on "
     "PubMedQA's three answers was {{M|pubmedqa_answer|s7_b3|log_loss|estimate:f4}} and its accuracy "
     "{{M|pubmedqa_answer|s7_b3|accuracy|estimate:pct1}} percent (Table 3)."),
    ("e6", "E6 (Table 13): lightgbm_tuned, the best of {{TR|settings_drawn/drawn:int}} settings drawn from {{TR|settings_drawn/from:int}} "
     "({{TR|chosen/num_leaves:int}} leaves, learning rate {{TR|chosen/learning_rate:f2}}, {{TR|chosen/n_estimators:int}} trees), had a lower "
     "log loss than session 4's LightGBM (" + iv(C_E6_T).replace(" (", ", ", 1).rstrip(")") + "). blend_v0_3 pooled the candidate with "
     "lightgbm_tuned at w {{BL|blend_v0_3|w:f2}} and T {{BL|blend_v0_3|T:f3}}, fitted on validation (out-of-fold validation log loss "
     "{{BL|blend_v0_3|out_of_fold_log_loss/mean:f4}}, optimistic because the candidate's calibration was fitted on the same records); against "
     "lightgbm_tuned no difference was detected in log loss (" + iv(C_E6_B3T).replace(" (", ", ", 1).rstrip(")") + "), and it was better "
     "than the candidate (" + iv(C_E6_B3C).replace(" (", ", ", 1).rstrip(")") + "). blend_v0_1 (w {{BL|blend_v0_1|w:f2}}) was better than "
     "v0.1 (" + iv(C_E6_B1).replace(" (", ", ", 1).rstrip(")") + "). The blends are reported as their own systems and are not part of the "
     "release bar."),
    ("e7", "E7 (Table 14): v0.1 with session 7's calibration rule had a lower log loss than v0.1 as released on PubMedQA's three answers ("
     + iv(C_E7_P).replace(" (", ", ", 1).rstrip(")") + ") and on readmission (" + iv(C_E7_R).replace(" (", ", ", 1).rstrip(")") + ")."),
    ("maybe", "Maybe (Table 15; descriptive): the test split has {{D|pubmedqa_answer_test_label_2:int}} maybe items, and v0.1 and the v0.2 "
     "candidate never put maybe on top. With their chosen calibrations, s7_a3 put maybe on top for {{MAY|s7_a3|top_answer_maybe:int}} items "
     "and s7_b3 for {{MAY|s7_b3|top_answer_maybe:int}}; with one temperature, for {{MAY|s7_a3_temperature|top_answer_maybe:int}} and "
     "{{MAY|s7_b3_temperature|top_answer_maybe:int}}, with an F1 of maybe of {{MAY|s7_a3_temperature|f1_maybe:f4}} and "
     "{{MAY|s7_b3_temperature|f1_maybe:f4}}. The chosen calibrations fitted a negative intercept for maybe "
     "({{CAL|s7_a3|pubmedqa_answer|families/temperature_intercepts/intercepts/2:f3}} for s7_a3 and "
     "{{CAL|s7_b3|pubmedqa_answer|families/temperature_intercepts/intercepts/2:f3}} for s7_b3; Table 2)."),
    ("handoff", "Hand-off (Table 16; secondary, descriptive): answering only its most confident half, the candidate was right on "
     "{{H|s7_a3|pubmedqa_answer|answered_accuracy_50/estimate:pct1}} percent of those PubMedQA items and on "
     "{{H|s7_a3|pubmedqa_answer|handed_off_accuracy_50/estimate:pct1}} percent of the half it handed off, against "
     "{{H|arm_laya|pubmedqa_answer|answered_accuracy_50/estimate:pct1}} and {{H|arm_laya|pubmedqa_answer|handed_off_accuracy_50/estimate:pct1}} "
     "percent for v0.1; on readmission, {{H|s7_a3|readmit_window|answered_accuracy_50/estimate:pct1}} and "
     "{{H|s7_a3|readmit_window|handed_off_accuracy_50/estimate:pct1}} percent."),
    ("mac", "The check ran on my Mac (Apple M5, {{MAC|machine/memory_gib:f1}} GiB of memory). Arm A fit at "
     "micro-batch {{MAC|arms/a/micro_batch:int}} on its first try, at {{MAC|arms/a/seconds_per_step:f2}} seconds per optimizer step of "
     "{{MAC|records_per_step:int}} records, about {{MAC|arms/a/hours_per_100000_records:f1}} hours per hundred thousand records. Arm B's "
     "tries at micro-batch {{MAC|arms/b/attempts/0/micro_batch:int}}, {{MAC|arms/b/attempts/1/micro_batch:int}} and "
     "{{MAC|arms/b/attempts/2/micro_batch:int}} without gradient checkpointing ran out of memory under the cap of "
     "{{MAC|arms/b/cap_gib:f2}} GiB, and it fit at micro-batch {{MAC|arms/b/micro_batch:int}} with gradient checkpointing, at "
     "{{MAC|arms/b/seconds_per_step:f2}} seconds per step, about {{MAC|arms/b/hours_per_100000_records:f1}} hours per hundred thousand "
     "records, with a peak of {{MAC|arms/b/peak_memory_gib:f2}} GiB (Tables 18 and 19). Arm A was measured in the first run, before "
     "correction 6; its only try never ran out of memory, and its peak, {{MAC|arms/a/peak_memory_gib:f2}} GiB, is under that cap."),
    ("labeler", "The labeler (labeler.json): its top answer was right on {{LAB|gate/labeler_correct:int}} of the {{LAB|gate/n:int}} calibrate "
     "items against v0.1's {{LAB|gate/v0_1_correct:int}}, so the gate passed. Of PQA-U's {{LAB|items/labeler_pqa_u/records:int}} items, its "
     "top answer was yes for {{LAB|subset/n_c/yes:int}}, no for {{LAB|subset/n_c/no:int}} and maybe for {{LAB|subset/n_c/maybe:int}}; N was "
     "the cap, {{LAB|subset/N:int}}, and phase 2 took {{LAB|subset/taken/yes:int}} yes, {{LAB|subset/taken/no:int}} no and "
     "{{LAB|subset/taken/maybe:int}} maybe items ({{LAB|subset/taken_total:int}} in all), the lowest probabilities taken being "
     "{{LAB|subset/lowest_probability_taken/yes:f3}}, {{LAB|subset/lowest_probability_taken/no:f3}} and "
     "{{LAB|subset/lowest_probability_taken/maybe:f3}}."),
]


def verbatim(src):
    """Lines checked whole: (label, source description, the text the line must be)."""
    contamination = src.cmp["contamination"]
    marker = "Session 4's notes on Duan stand: "
    return [
        ("outcome", "comparison.json release_bar.outcome, then the plan's rule (POWER AND THE BAR)",
         f"{src.cmp['release_bar']['outcome']}. v0.1 stays the current version."),
        ("one run", "comparison.json one_run", src.cmp["one_run"]),
        ("Duan", "comparison.json contamination (session 4's notes on Duan)", contamination[contamination.index(marker):]),
    ]


# the self-test's planted errors: (text in the document, text put in its place), one at a time; filled in with the document
PLANTED = [
    ("| `s7_a3` | 0.6506 (0.5731 to 0.7302) |", "| `s7_a3` | 0.6507 (0.5731 to 0.7302) |"),  # a table cell's estimate
    ("| 0.1216 (0.0908 to 0.1563) |", "| 0.1216 (0.0909 to 0.1563) |"),  # a table cell's interval bound
    ("| E1 | Readmission | Log loss | 0.0057 (0.0019 to 0.0094) | v0.1 better |",
     "| E1 | Readmission | Log loss | 0.0057 (0.0019 to 0.0094) | candidate better |"),  # a reading
    ("| Criterion 4 | FAIL |", "| Criterion 4 | PASS |"),  # a rule's result
    ("| `s7_a3` | PubMedQA, three answers | 0.5892 | 0.5824 | temperature_intercepts |",
     "| `s7_a3` | PubMedQA, three answers | 0.5892 | 0.5824 | temperature |"),  # a chosen family
    ("| `s7_a3` | 4 | 0.0000 |", "| `s7_a3` | 5 | 0.0000 |"),  # a count in a table
    ("release_candidate.json was written at 06:21:31 UTC", "release_candidate.json was written at 06:21:32 UTC"),  # a time
    ("has 500 items, 445 of them yes or no", "has 500 items, 446 of them yes or no"),  # a count in the prose
    ("its accuracy 75.4 percent against 71.4 percent", "its accuracy 75.5 percent against 71.4 percent"),  # a percentage
    ("with a 97.5 percent interval of -0.1460 to -0.0391", "with a 97.5 percent interval of -0.1460 to -0.0381"),  # a prose bound
    ("right on 127 of the 150 calibrate items", "right on 128 of the 150 calibrate items"),  # a count from labeler.json
    ("v0.3 release bar not met: criterion 4. v0.1 stays the current version.",
     "v0.3 release bar not met: criterion 3. v0.1 stays the current version."),  # a whole-line statement
    ("at 54.75 seconds per step", "at 54.57 seconds per step"),  # a figure from mac_training_check.json
]


def render_prose(src, template):
    keys = []

    def sub(m):
        key, how = m.group(1).rsplit(":", 1)
        keys.append((key, how))
        return fmt(src.resolve(key)[1], how)

    return re.sub(r"\{\{(.+?)\}\}", sub, template), keys


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


class Check:
    """One pass over a document's Results section."""

    def __init__(self, src, text):
        self.src, self.rows, self.readings, self.problems = src, [], [], []
        if START not in text:
            sys.exit(f"the document has no line {START!r}")
        start = text.index(START)
        rest = text[start + len(START):]
        end = re.search(r"^## ", rest, re.M)
        self.section = (START + (rest[:end.start()] if end else rest)).splitlines()
        self.tables = {t[0]: t for t in tables(src)}
        self.seen_tables = set()
        self.prose = {}
        for pid, template in PROSE:
            line, keys = render_prose(src, template)
            if shape(line) in self.prose:
                sys.exit(f"two templates share a shape: {pid}")
            self.prose[shape(line)] = (pid, keys)
        self.verbatim = {text: (label, source) for label, source, text in verbatim(src)}
        self.seen_prose, self.seen_verbatim = set(), set()
        self.heading, self.table = None, []
        self.run()

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
        else:
            scale = 100.0 if how == "pct1" else 1.0
            want_d = 1 if how == "pct1" else int(how[1:])
            ok = d == want_d and abs(written - scale * float(value)) <= 0.5 * 10 ** (-d) + 1e-12
            if text.startswith("-") and round(scale * float(value), d) == 0:
                ok = False  # a negative zero would be printed
            shown = f"{scale * float(value):.6f}"
        self.rows.append((where, text, source, shown, "PASS" if ok else "MISMATCH"))

    def interval(self, where, text, base):
        parts = {p: self.src.resolve(f"{base}|{p}") for p in ("estimate", "low", "high")}
        if parts["estimate"][1] is None:
            self.rows.append((where, text, parts["estimate"][0], "null", "PASS" if text == "not computed" else "MISMATCH"))
            return
        if parts["low"][1] is None:
            self.number(f"{where}, estimate", text, parts["estimate"][0], parts["estimate"][1], "f4")
            return
        m = re.fullmatch(r"(-?\d+\.\d+) \((-?\d+\.\d+) to (-?\d+\.\d+)\)", text)
        if not m:
            self.rows.append((where, text, parts["estimate"][0], "(cell not parsed)", "MISMATCH"))
            return
        for p, t in zip(("estimate", "low", "high"), m.groups()):
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
        elif kind == "read":
            want = cell_text(src, spec)
            self.readings.append((where, text, f"{want} (from the interval of {spec[1]})", "PASS" if text == want else "MISMATCH"))
        elif kind in ("str", "pf", "map"):
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
            labels = ", ".join(c for c in want if isinstance(c, str))
            for text, spec, col in zip(got, want, header):
                self.cell(f"Table {number}, {labels}, {col}", text, spec)

    def run(self):
        for n, line in enumerate(self.section, start=1):
            if line.startswith("|"):
                self.table.append(line)
                continue
            self.flush()
            if line.startswith("#"):
                self.heading = line
                if tokens(line):
                    self.problems.append(f"heading with unexplained numbers: {line!r}")
                continue
            if not line.strip():
                continue
            if line in self.verbatim:
                label, source = self.verbatim[line]
                self.seen_verbatim.add(label)
                self.rows.append((f"line {n} ({label}, the whole line)", "verbatim", source, "the same text", "PASS"))
                continue
            found = tokens(line)
            entry = self.prose.get(shape(line))
            if entry is None:
                if found:
                    self.problems.append(f"line {n}: numbers without a source key: {found} in {line[:90]!r}")
                continue
            pid, keys = entry
            self.seen_prose.add(pid)
            if len(found) != len(keys):
                self.problems.append(f"line {n} ({pid}): {len(found)} numbers found, {len(keys)} source keys")
                continue
            for k, (t, (key, how)) in enumerate(zip(found, keys), start=1):
                source, value = self.src.resolve(key)
                self.number(f"line {n} ({pid}), number {k}", t, source, value, how)
        self.flush()
        for number, t in self.tables.items():
            if number not in self.seen_tables and t[3]:
                self.problems.append(f"Table {number} ({t[1]}) is missing from the document")
        for pid, _ in PROSE:
            if pid not in self.seen_prose:
                self.problems.append(f"the paragraph {pid!r} is missing from the document, or its words were changed")
        for label, _, _ in verbatim(self.src):
            if label not in self.seen_verbatim:
                self.problems.append(f"the line {label!r} is missing from the document, or differs from its source")

    def failures(self):
        return [r for r in self.rows if r[4] != "PASS"] + [r for r in self.readings if r[3] != "PASS"] + self.problems


def self_test(src, text):
    for i, (old, new) in enumerate(PLANTED, start=1):
        if text.count(old) != 1:
            sys.exit(f"self-test: planted error {i} cannot be placed ({old!r} is not in the document exactly once)")
        if not Check(src, text.replace(old, new)).failures():
            sys.exit(f"self-test: planted error {i} ({new!r}) was not reported")
    return len(PLANTED)


def write_outputs(text, check, planted):
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    now = dt.datetime.now(dt.timezone.utc)
    figures = [{"where": r[0], "as_written": r[1], "source": r[2], "source_value": r[3], "check": r[4]} for r in check.rows]
    readings = [{"where": r[0], "as_written": r[1], "expected": r[2], "check": r[3]} for r in check.readings]
    SOURCES_OUT.write_text(json.dumps({"plan": "docs/prompt_session7.md, step 15", "document": "docs/fiorillo_v0_3.md",
                                       "document_sha256": sha, "time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       "figures": figures, "readings": readings}, indent=1) + "\n", encoding="utf-8")

    def esc(x):
        return str(x).replace("|", "/")

    out = [
        "# Fiorillo v0.3: figure check",
        f"Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: {now.day} {now.strftime('%B %Y')}",
        "Sources: docs/fiorillo_v0_3.md (the Results section); results/fiorillo_v0_3/ (comparison.json, release_candidate.json, "
        "testpreds_log.json, selection.json, calibration_s7.json, labeler.json, power.json, trees.json, blend_v0_1.json, blend_v0_3.json, "
        "zeros.json, handoff_test.json and metrics/); data/processed/public/ (the test records' counts)",
        "Status: complete",
        "",
        "Abbreviations: ECE is expected calibration error; AUROC is the area under the receiver operating characteristic curve; "
        "AURC is the area under the risk-coverage curve; UTC is Coordinated Universal Time; SHA-256 is Secure Hash Algorithm 256; "
        "E1 to E7 are the plan's comparisons (docs/prompt_session7.md, COMPARISONS).",
        "",
        "## How the check was made",
        "",
        "src/eval/check_fiorillo_v0_3_figures.py read the Results section of docs/fiorillo_v0_3.md as written (SHA-256 of the file "
        f"checked: {sha}), found every number in it and compared each with its source at the printed precision: a number printed "
        "with four decimals passes when it is the source value rounded to four decimals, a count when it equals the source, a time "
        "when it equals the source's time of day. Table cells were matched to their sources by the table's heading and its rows and "
        "columns, which the script fixes for each table. Numbers in the prose were matched, in order, to the source keys of the "
        "line's template in the script, and each key was resolved from the files by the script's own code. Three lines were "
        "checked whole against their source text (the release bar's outcome, the one-run statement and session 4's notes on Duan). "
        "Every reading was checked against its interval and the direction of its metric, and every PASS or FAIL against the rule's "
        f"own result. Before the document was checked, {planted} errors were planted in copies of it, one at a time (in table cells, "
        "prose, a count, a time, a percentage, a reading and a whole-line statement), and each was reported. The script checks numbers "
        "and readings, not the wording of claims. Every figure's source is also listed in results/fiorillo_v0_3/figure_sources.json.",
        "",
        "Not counted as numbers: version names (v0.1 to v0.3), Qwen3-4B-Base, idea numbers (I-01 and so on), E1 to E7; table, "
        "correction, criterion, session, phase and step numbers; dates; \"95 percent\" and \"97.5 percent\" where they name an interval, "
        "and \"95th\"; \"30 days\" where it defines the readmission levels, and the levels 0 to 2; \"prints as 0\"; \"13-word\"; and "
        "system, question and file names in code type or with an underscore or slash. Line numbers count the heading "
        "\"## Results (test split)\" as line 1.",
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
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    src = Sources()
    text = DOC.read_text(encoding="utf-8")
    planted = self_test(src, text)
    check = Check(src, text)
    failures = check.failures()
    if failures:
        for f in failures:
            print("FAIL", f)
        sys.exit(f"{len(failures)} failures; {OUT.relative_to(ROOT)} was not written")
    if not planted:
        sys.exit("no planted errors: the self-test did not run")
    write_outputs(text, check, planted)
    print(f"self-test: {planted} planted errors, each reported. {len(check.rows):,} numbers and whole lines and "
          f"{len(check.readings):,} readings, all PASS; wrote {SOURCES_OUT.relative_to(ROOT)} and {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

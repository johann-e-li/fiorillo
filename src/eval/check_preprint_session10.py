"""The preprint's figure check (docs/prompt_session10.md, THE CHECKS: the figure check).

It reads docs/preprint/fiorillo_preprint.tex and the table fragments it inputs (docs/preprint/tables/), removes
commands, comments, labels, citation keys and the reference list, and checks every remaining number against
docs/preprint/figure_sources.json, which src/eval/preprint_session10.py writes. It writes docs/preprint/figure_check.md,
one row per number: PASS, MISMATCH, or NO SOURCE with where the number comes from.

How a number is checked.
- The text is split into the sections that the source marks with a line "% SECTION: <name>". A number in a section
  passes when figure_sources.json has an entry printed exactly as it is and registered for that section ("text:<name>"),
  and the entry itself checks: its value read again from its file and key (or, for a derived entry, computed again
  from its files by this script, independently of preprint_session10.py) prints as the entry says. A number in a
  table fragment passes the same way against the entries registered for that table ("table:<name>").
- MISMATCH: an entry whose file no longer gives the printed value, or a number whose entry does not check.
- NO SOURCE: a number that comes from no results file. It is allowed only when figure_sources.json's origins give it
  for that place, with where it comes from (a published paper's figure, a date, a count stated in a cited
  document, a setting fixed in the registration); a number with neither a source nor an origin fails the check.
- Dates and times (1 October 2026; 15:31 UTC) are checked as one token each, against the dates that
  figure_sources.json gives with their files or origins.
- Not numbers, and so not checked: names and identifiers that contain digits (Qwen3-4B-Base, Gemma 4 31B-it,
  Evidence Inference 2.0, v0.5, K1 to K10, H1 to H4, I-97, F1, the sessions' and corrections' numbers, software
  versions), DOIs, URLs, system names in typewriter type, cross-references and the reference list. The patterns are
  NAMES below. Placeholders (\\placeholder{...}) are listed apart: they are replaced before the draft is final.

The check passes when no row is MISMATCH and every NO SOURCE row names its origin.

Usage (from the project folder or with --root):
  python3 src/eval/check_preprint_session10.py            # writes docs/preprint/figure_check.md
  python3 src/eval/check_preprint_session10.py --list     # prints every number by section, writes nothing
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
TEX = "docs/preprint/fiorillo_preprint.tex"
TABLES = "docs/preprint/tables"
SOURCES = "docs/preprint/figure_sources.json"
OUT = "docs/preprint/figure_check.md"
PLAN = "docs/prompt_session10.md, THE CHECKS (the figure check)"

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
DATE = re.compile(r"\b\d{1,2} (?:" + MONTHS + r") \d{4}\b")
TIME = re.compile(r"\b\d{2}:\d{2}(?::\d{2})? UTC\b")
ORDINAL = re.compile(r"(?<![\w.])\d+(?:st|nd|rd|th)\b")
NUMBER = re.compile(r"(?<![\w.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.]?\w)")

# Names and identifiers that contain digits; they are removed before numbers are read.
NAMES = [
    r"Qwen3-4B-Base", r"Qwen3", r"Gemma 4 31B-it", r"Gemma 4", r"BioDecision-4B v2", r"BioGPT-Large-PubMedQA",
    r"LongT5", r"Evidence Inference 2\.0", r"Diabetes 130-US", r"TabICLv2", r"\bBM25\b", r"CC BY 4\.0",
    r"\bCC0\b", r"Apache License 2\.0", r"vLLM 0\.30\.0", r"\bCUDA \d+\.\d+\b", r"\bv\d+\.\d+\b", r"\bK\d+ to K\d+\b",
    r"\bK\d+\b", r"\bH\d to H\d\b", r"\bH\d\b", r"\bI-\d+\b", r"\bF1\b", r"\bsessions? \d+\b",
    r"\b[Cc]orrections? \d+(?: (?:to|and) \d+)?\b", r"\b[Cc]riteri(?:a|on) \d(?: and \d)?\b", r"\bSHA-256\b",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value, digits, scale=1.0, thousands=False):
    """A number as the preprint prints it (the same rule as preprint_session10.py's fmt, written here again)."""
    v = float(value) * scale
    if thousands and digits == 0:
        return f"{int(round(v)):,}"
    text = f"{v:,.{digits}f}" if thousands else f"{v:.{digits}f}"
    if re.fullmatch(r"-0(\.0+)?", text):
        text = text[1:]
    return text


def segments(key):
    """A dotted key's segments; a dot inside a selector's brackets does not split, nor one written as a backslash and
    a dot (a key that itself holds a dot)."""
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
    """The value at a dotted key; name[f=v,g=w] picks the one list element whose fields match (a comma inside a value
    is written as a backslash and a comma)."""
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


# ---------------------------------------------------------------- reading the LaTeX source

def strip_braced(text, command, nargs=1, keep=False, optional=True):
    """Remove \\command[opt]...{arg}... (nargs braced arguments), or keep the last argument's content if keep."""
    out, i = [], 0
    pat = re.compile(r"\\" + command + r"\*?(?![A-Za-z])")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        while optional and j < len(text) and text[j] == "[":
            depth, k = 0, j
            while k < len(text):
                if text[k] == "[":
                    depth += 1
                elif text[k] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            j = k + 1
        args = []
        for _ in range(nargs):
            while j < len(text) and text[j] in " \n":
                j += 1
            if j >= len(text) or text[j] != "{":
                break
            depth, k = 0, j
            while k < len(text):
                if text[k] == "{" and text[k - 1] != "\\":
                    depth += 1
                elif text[k] == "}" and text[k - 1] != "\\":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            args.append(text[j + 1:k])
            j = k + 1
        if keep and args:
            out.append(" " + args[-1] + " ")
        else:
            out.append(" ")
        i = j
    return "".join(out)


def balanced(text, j):
    """The index just past the balanced {...} group that starts at text[j] (which must be "{")."""
    if text[j] != "{":
        raise ValueError(f"no brace at {j}: {text[j:j + 20]!r}")
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{" and (k == 0 or text[k - 1] != "\\"):
            depth += 1
        elif text[k] == "}" and (k == 0 or text[k - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    raise ValueError("unbalanced braces")


def strip_tabular_specs(text):
    """Remove \\begin{tabular}[pos]{column spec} (and tabular*'s width), and a minipage's [pos] options and width, and
    keep what they hold: the column spec holds widths such as p{3.0cm} and a minipage's width such as 0.40\\textwidth,
    which are not the paper's numbers, while the cells, nested tabulars among them, and the minipage's caption hold
    the numbers the check reads."""
    out, i = [], 0
    pat = re.compile(r"\\begin\{(tabular\*?|minipage)\}")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        while j < len(text) and text[j] == "[":
            j = text.index("]", j) + 1
        for _ in range(2 if m.group(1) == "tabular*" else 1):
            while j < len(text) and text[j] in " \n":
                j += 1
            j = balanced(text, j)
        out.append(" ")
        i = j
    return "".join(out)


def clean(text, placeholders=None):
    """LaTeX to plain text in which only the prose's numbers remain."""
    text = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.split("\n"))
    if placeholders is not None:
        placeholders.extend(re.findall(r"\\placeholder\{([^}]*)\}", text))
    text = strip_braced(text, "placeholder")
    text = strip_braced(text, "multicolumn", 3, keep=True)
    for cmd, n in (("label", 1), ("ref", 1), ("eqref", 1), ("cite", 1), ("citet", 1), ("citep", 1), ("citealt", 1),
                   ("citeauthor", 1), ("input", 1), ("includegraphics", 1), ("url", 1), ("doi", 1), ("href", 2),
                   ("nolinkurl", 1), ("texttt", 1), ("sysname", 1), ("setlength", 2), ("captionsetup", 1),
                   ("bibitem", 1), ("newcommand", 2), ("renewcommand", 2), ("usepackage", 1), ("documentclass", 1),
                   ("textsuperscript", 1), ("vspace", 1), ("hspace", 1), ("arraybackslash", 0), ("idn", 1)):
        text = strip_braced(text, cmd, n)
    text = strip_tabular_specs(text)
    text = re.sub(r"\\(?:begin|end)\{[a-zA-Z*]+\}(?:\[[^\]]*\])?", " ", text)
    text = text.replace("$-$", "-").replace("\\$", "").replace("$", "")
    text = re.sub(r"\\[,;: !]", " ", text)
    text = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ").replace("\\&", " and ")
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"\[[-0-9.]+(?:pt|em|ex|cm|mm)\]", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"``|''|`|'", "'", text)
    return re.sub(r"[ \t]+", " ", text)


def tokens(text):
    """(kind, token, context) for every date, time, ordinal and number in plain text, names removed first."""
    out = []
    for pat in NAMES:
        text = re.sub(pat, " ", text)
    flat = re.sub(r"\s+", " ", text)
    taken = [False] * len(flat)
    for kind, pat in (("date", DATE), ("time", TIME), ("ordinal", ORDINAL), ("number", NUMBER)):
        for m in pat.finditer(flat):
            if any(taken[m.start():m.end()]):
                continue
            for k in range(m.start(), m.end()):
                taken[k] = True
            ctx = flat[max(0, m.start() - 50):m.end() + 30].strip()
            out.append((m.start(), kind, m.group(0), ctx))
    return [(k, t, c) for _, k, t, c in sorted(out)]


def sections(tex):
    """The body's sections by their "% SECTION:" markers, the bibliography removed."""
    body = tex.split("\\begin{document}", 1)[1]
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", body, flags=re.S)
    parts = re.split(r"^% SECTION: ([a-zA-Z0-9_]+)\s*$", body, flags=re.M)
    out = {}
    for k in range(1, len(parts), 2):
        out[parts[k]] = parts[k + 1]
    return out


# ---------------------------------------------------------------- checking the entries themselves

class Checker:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def json(self, rel):
        if rel not in self.cache:
            self.cache[rel] = json.loads((self.root / rel).read_text(encoding="utf-8"))
        return self.cache[rel]

    def csv(self, rel):
        key = ("csv", rel)
        if key not in self.cache:
            with open(self.root / rel, newline="", encoding="utf-8") as f:
                self.cache[key] = list(csv.DictReader(f))
        return self.cache[key]

    def entry(self, e):
        """(status, value found, note) for one entry of figure_sources.json."""
        if "file" in e and "key" in e and "format" in e:  # a date or time read from a file
            v = resolve(self.json(e["file"]), e["key"])
            t = dt.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
            printed = date_text(t, e["format"])
            return ("PASS" if printed == e["printed"] else "MISMATCH"), v, f"{e['file']} {e['key']}"
        if "file" in e and "key" in e:
            v = resolve(self.json(e["file"]), e["key"])
            printed = fmt(v, e["digits"], e.get("scale", 1.0), e.get("thousands", False))
            return ("PASS" if printed == e["printed"] else "MISMATCH"), v, f"{e['file']} {e['key']}"
        if "derived" in e:
            v = self.derive(e)
            if v is None:
                return "MISMATCH", None, "derived: this check cannot compute it again: " + e["derived"]
            printed = fmt(v, e["digits"], e.get("scale", 1.0), e.get("thousands", False))
            return ("PASS" if printed == e["printed"] else "MISMATCH"), v, "derived: " + e["derived"]
        return "MISMATCH", None, "an entry with neither a file and key nor a derivation"

    def derive(self, e):
        """Each derived number computed again here, from its files, by its stated rule."""
        how = e["derived"]
        lic_csv = "results/fiorillo_v0_5/license_recheck.csv"
        lic = "results/fiorillo_v0_5/license_recheck.json"
        if how.startswith(f"rows of {lic_csv} with clean true"):
            return sum(r["clean"] == "true" for r in self.csv(lic_csv))
        if how == f"rows of {lic_csv} in one of the three splits":
            return sum(bool(r["splits"]) for r in self.csv(lic_csv))
        if how == f"rows of {lic_csv} in one of the three splits with clean true":
            return sum(bool(r["splits"]) and r["clean"] == "true" for r in self.csv(lic_csv))
        m = re.fullmatch(rf"the sum of {re.escape(lic)} by_split\.<split>\.(\w+) over the three splits", how)
        if m:
            return sum(b[m.group(1)] for b in self.json(lic)["by_split"].values())
        m = re.fullmatch(rf"{re.escape(lic)} by_split\.(\w+)\.(\w+) minus by_split\.(\w+)\.(\w+)", how)
        if m:
            b = self.json(lic)["by_split"]
            return b[m.group(1)][m.group(2)] - b[m.group(3)][m.group(4)]
        m = re.fullmatch(rf"{re.escape(lic)} by_split\.(\w+)\.(\w+) plus by_split\.(\w+)\.(\w+)", how)
        if m:
            b = self.json(lic)["by_split"]
            return b[m.group(1)][m.group(2)] + b[m.group(3)][m.group(4)]
        m = re.fullmatch(r"(\S+) (\S+) minus (\S+) (\S+)", how)
        if m:
            return resolve(self.json(m.group(1)), m.group(2)) - resolve(self.json(m.group(3)), m.group(4))
        m = re.fullmatch(r"the sum of (rows|true_answer_printed_zero) over every system, split and question of (\S+)", how)
        if m:
            def walk(d):
                if isinstance(d, dict):
                    if "rows" in d and "true_answer_printed_zero" in d:
                        return d[m.group(1)]
                    return sum(walk(v) for v in d.values())
                return 0
            return walk(self.json(m.group(2))["rows_whose_true_answer_prints_as_0"])
        m = re.fullmatch(r"the exponential of (\S+) (\S+)", how)
        if m:
            return math.exp(resolve(self.json(m.group(1)), m.group(2)))
        m = re.fullmatch(r"(\w+) of the reliability bin (\d+) of (\S+) \((\d+) equal-width bins on the top answer's "
                         r"probability\)", how)
        if m:
            return self.reliability(m.group(3), int(m.group(4)))[int(m.group(2))][m.group(1)]
        m = re.fullmatch(r"the cost per 1,000 questions: 1,000 x (\S+) (\S+) / 3,600 x (\S+) (\S+)", how)
        if m:
            return 1000 * resolve(self.json(m.group(1)), m.group(2)) / 3600 * resolve(self.json(m.group(3)), m.group(4))
        return None

    def reliability(self, rel, bins):
        """The bins as src/eval/metrics.py's ece() assigns them (bin i holds edges[i] < p <= edges[i + 1], bin 0
        includes 0), computed here without numpy: the edges are i / bins, as numpy's linspace gives them to within a
        rounding error, and a probability within 1e-12 of an inner edge stops the check."""
        out = {}
        for r in self.csv(rel):
            p = [float(r["p0"]), float(r["p1"]), float(r["p2"])]
            top = max(range(3), key=lambda k: (p[k], -k))
            conf = p[top]
            b = 0
            while b < bins - 1 and conf > (b + 1) / bins:
                b += 1
            if any(abs(conf - k / bins) < 1e-12 for k in range(1, bins)):
                raise SystemExit(f"a probability on a bin edge in {rel}: {conf}")
            o = out.setdefault(b, {"n": 0, "conf": 0.0, "correct": 0})
            o["n"] += 1
            o["conf"] += conf
            o["correct"] += int(top == int(r["label"]))
        for b, o in out.items():
            o["mean_probability"] = o["conf"] / o["n"]
            o["share_right"] = o["correct"] / o["n"]
            lo, hi = wilson(o["correct"], o["n"])
            o["wilson_low"], o["wilson_high"] = lo, hi
            o["lower_edge"] = b / bins
        return out


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def date_text(t, form):
    """A date or time as the preprint prints it: "day Month year" (no leading zero), "HH:MM UTC" or "HH:MM:SS UTC"."""
    if form == "date":
        return f"{t.day} {t.strftime('%B')} {t.year}"
    if form == "time_minutes":
        return t.strftime("%H:%M") + " UTC"
    if form == "time_seconds":
        return t.strftime("%H:%M:%S") + " UTC"
    raise ValueError(form)


# ---------------------------------------------------------------- the check

def run(root=ROOT, write=True, listing=False):
    root = Path(root)
    tex = (root / TEX).read_text(encoding="utf-8")
    secs = sections(tex)
    tables = sorted(set(re.findall(r"\\input\{tables/([a-z_0-9]+)\.tex\}", tex)))
    places = []  # (where, kind, token, context)
    placeholders = []
    for name, body in secs.items():
        for kind, tok, ctx in tokens(clean(body, placeholders)):
            places.append((f"text:{name}", kind, tok, ctx))
    for t in tables:
        frag = (root / TABLES / f"{t}.tex").read_text(encoding="utf-8")
        for kind, tok, ctx in tokens(clean(frag)):
            places.append((f"table:{t}", kind, tok, ctx))
    if listing:
        for where, kind, tok, ctx in places:
            print(f"{where}\t{kind}\t{tok}\t{ctx}")
        print(f"{len(places)} numbers; placeholders: {placeholders}")
        return 0
    src = json.loads((root / SOURCES).read_text(encoding="utf-8"))
    C = Checker(root)
    checked = {}
    for i, e in enumerate(src["entries"] + src.get("dates", [])):
        try:
            checked[i] = C.entry(e)
        except (KeyError, TypeError, IndexError, ValueError) as err:
            checked[i] = ("MISMATCH", None, f"cannot read: {err}")
    entries = src["entries"] + src.get("dates", [])
    by_place = {}
    for i, e in enumerate(entries):
        by_place.setdefault((e["where"], e["printed"]), []).append(i)
    origins = {}
    for o in src.get("origins", []):
        for w in o["where"]:
            origins.setdefault((w, o["printed"]), []).append(o["origin"])
    rows = []
    for where, kind, tok, ctx in places:
        hits = by_place.get((where, tok), [])
        if hits:
            states = [checked[i] for i in hits]
            ok = [s for s in states if s[0] == "PASS"]
            status = "PASS" if ok else "MISMATCH"
            s = ok[0] if ok else states[0]
            rows.append((where, kind, tok, status, s[2], s[1], ctx))
        elif (where, tok) in origins:
            rows.append((where, kind, tok, "NO SOURCE", "origin: " + origins[(where, tok)][0], "", ctx))
        else:
            rows.append((where, kind, tok, "NO SOURCE", "", "", ctx))
    # Every number a table fragment prints is registered for that table, so every registered value must be among
    # the numbers read from the fragment: one that is not means the check failed to read it.
    unread = []
    for t in tables:
        read = {r[2] for r in rows if r[0] == f"table:{t}" and r[3] == "PASS"}
        missing = sorted({e["printed"] for e in src["entries"] if e["where"] == f"table:{t}"} - read)
        if missing:
            unread.append(f"table:{t}: registered but not read: {', '.join(missing[:8])}")
    files_now = {rel: sha256_file(root / rel) for rel in src.get("files", {}) if (root / rel).exists()}
    stale = [rel for rel, h in src.get("files", {}).items() if files_now.get(rel) != h]
    bad = [r for r in rows if r[3] == "MISMATCH" or (r[3] == "NO SOURCE" and not r[4])]
    for u in unread:
        print("   UNREAD", u, flush=True)
    unused = [i for i, e in enumerate(entries) if checked[i][0] != "PASS"]
    if write:
        write_report(root, rows, placeholders, stale, unused, entries, checked, src, unread)
    n = {s: sum(r[3] == s for r in rows) for s in ("PASS", "MISMATCH", "NO SOURCE")}
    print(f"{len(rows)} numbers: {n['PASS']} PASS, {n['MISMATCH']} MISMATCH, {n['NO SOURCE']} NO SOURCE "
          f"({sum(1 for r in rows if r[3] == 'NO SOURCE' and not r[4])} without an origin); "
          f"{len(placeholders)} placeholders; entries that do not check: {len(unused)}; files changed since "
          f"figure_sources.json was written: {len(stale)}", flush=True)
    for r in bad[:40]:
        print("  ", r[3], r[0], r[2], "|", r[6], flush=True)
    return 1 if bad or unused or stale or unread else 0


def write_report(root, rows, placeholders, stale, unused, entries, checked, src, unread=()):
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = {s: sum(r[3] == s for r in rows) for s in ("PASS", "MISMATCH", "NO SOURCE")}
    no_origin = sum(1 for r in rows if r[3] == "NO SOURCE" and not r[4])
    verdict = "PASSES" if not (n["MISMATCH"] or no_origin or unused or stale or unread) else "FAILS"
    lines = [
        "# The preprint's figure check",
        "Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: "
        + dt.datetime.now(dt.timezone.utc).strftime("%-d %B %Y").lstrip("0"),
        f"Sources: {TEX} and the table fragments it inputs ({TABLES}/); {SOURCES} (written {src.get('time_utc')}); "
        "the results files it names; src/eval/check_preprint_session10.py, which wrote this file at " + now,
        f"Status: complete (the check {verdict})",
        "",
        "Abbreviations: UTC is Coordinated Universal Time; EI is Evidence Inference 2.0; DOI is digital object "
        "identifier.",
        "",
        "## How the check works",
        "",
        "The check removes the LaTeX source's commands, comments, labels, citation keys, cross-references, URLs and "
        "DOIs, system names in typewriter type and the reference list, and reads every remaining number, date and "
        "time in each section (the source marks each with a line \"% SECTION: <name>\") and in each table fragment. "
        "Names and identifiers that contain digits (model names, versions, K1 to K10, H1 to H4, the register's row "
        "numbers, F1, the sessions' and corrections' numbers) are not numbers and are removed first; the patterns "
        "are NAMES in the script. A number PASSES when figure_sources.json registers a value printed exactly so for "
        "that section or table, and that value, read again from its file and key or computed again by this script "
        "from its files, still prints so. MISMATCH is a number whose registered value does not check. NO SOURCE is a "
        "number that comes from no results file; figure_sources.json's origins name where each comes from. Every "
        "value figure_sources.json registers for a table must also be among the numbers read from that table's "
        "fragment, so that a number the check failed to read cannot go unnoticed. The check passes with no MISMATCH, every NO SOURCE row "
        "with an origin, every registered value checking, every table read in full, and every results file unchanged "
        "since figure_sources.json was written.",
        "",
        "## Summary",
        "",
        f"- Numbers, dates and times checked: {len(rows):,}",
        f"- PASS: {n['PASS']:,}",
        f"- MISMATCH: {n['MISMATCH']:,}",
        f"- NO SOURCE: {n['NO SOURCE']:,}, of which without an origin: {no_origin:,}",
        f"- Registered values that do not check: {len(unused):,}",
        f"- Tables whose numbers were not all read: {len(unread):,}" + ("".join(f"\n  - {u}" for u in unread) if unread else ""),
        f"- Results files changed since figure_sources.json was written: {len(stale):,}"
        + (f" ({', '.join(stale)})" if stale else ""),
        f"- Placeholders in the draft, not checked: {len(placeholders):,}"
        + ("".join(f"\n  - {p}" for p in placeholders) if placeholders else ""),
        "",
        "## Every number",
        "",
        "| # | Where | Number | Status | Source (file and key, derivation or origin) | Value found | Context |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, (where, kind, tok, status, source, value, ctx) in enumerate(rows, 1):
        val = "" if value in ("", None) else (f"{value:.6f}" if isinstance(value, float) else str(value))
        ctx = ctx.replace("|", "/")
        lines.append(f"| {k} | {where} | {tok} | {status} | {source.replace('|', '/')} | {val} | {ctx} |")
    if unused:
        lines += ["", "## Registered values that do not check", ""]
        for i in unused:
            e = entries[i]
            lines.append(f"- {e.get('where')}: {e.get('printed')} ({checked[i][0]}; {checked[i][2]})")
    (root / OUT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--list", action="store_true", help="print every number by section and write nothing")
    args = parser.parse_args(argv)
    return run(args.root, write=not args.list, listing=args.list)


if __name__ == "__main__":
    sys.exit(main())

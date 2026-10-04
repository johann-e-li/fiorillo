"""Write Fiorillo v0.5's model card (docs/prompt_session9.md, THE RELEASE, with corrections 3 and 7 (f); idea I-40 of
docs/ideas_register.md, a question card for each question).

The card is written twice with the same text: the release folder's README.md (the Hugging Face model card) and
docs/model_card_v0_5.md. Its content is what THE RELEASE lists:
- for each question, a question card: its intended use and user; its inputs; its training data and their licenses; its
  validation and test splits and the dates they were read; log loss, ECE, the calibration slope and intercept (as
  COMPARISONS defines them), Brier score, accuracy and macro-F1 on test with intervals; the hand-off results, with v0.5
  handing nothing off as released; for EI, THE READING CHECKS' results and readings, and the passage selection with the
  exception for its word statistics (the plan's decision 3); known failure modes and contamination; and the update plan;
- why v0.4's EI specialist is not released; the licenses, and the clean articles credited (NOTICE lists each one);
- correction 7 (f)'s statement on PubMedQA's abstracts;
- the date of the release day's prior-art check, and no claim of a first (any first would be worded "to our knowledge");
- when swap averaging is kept (correction 3), that each question takes two passes of each reader, with the cost.
It carries no result from session 6: no file of session 6 is read.

Every number comes from a results file read here; none is typed into the prose. build() returns, with the card, each
number's file and key path (or, for a number derived from others, its inputs and the arithmetic), so that a test can trace
every number back. The only digits in the card that are not read from a file are in names, versions and citations
(LITERALS) and in the example records of the usage section's code block; the card's own date is the clock's.

A file that is missing, or that lacks what a part needs, gives that part the text "NOT BUILT: <part>, <reason>" in place
of its numbers, and the Status line counts those parts. The card is not written at all when comparison.json says v0.5's
bar was not met, since THE RELEASE runs only if it was met.

The release day's prior-art check is read from docs/prior_art.md as its last heading of the form
"### Release-day check, <D Month YYYY> (v0.5)" (step 17 appends it).

Inputs (each an option; the defaults are the plan's paths, relative to the repository root):
  --comparison         results/fiorillo_v0_5/comparison.json          (src/eval/compare_session9.py)
  --release-candidate  results/fiorillo_v0_5/release_candidate.json   (src/eval/calibrate_session9.py freeze)
  --calibration        results/fiorillo_v0_5/calibration_s9.json      (src/eval/calibrate_session9.py fit)
  --recipe             results/fiorillo_v0_5/recipe_selection.json    (src/eval/recipe_session9.py)
  --pool               results/fiorillo_v0_5/readmission_pool_v2.json (src/eval/trees_session9.py pool)
  --licenses           results/fiorillo_v0_5/license_recheck.json     (src/tools/ei_license_recheck.py)
  --fixed              results/fiorillo_v0_5/fixed_versions.json      (step 0)
  --coverage           results/fiorillo_v0_5/coverage_long.json       (src/tasks/ei_session9.py long)
  --selection8         results/fiorillo_v0_4/article_selection.json   (session 8's passage selection and its word statistics)
  --calibration7       results/fiorillo_v0_3/calibration_s7.json      (s7_b3's calibration)
  --comparison8        results/fiorillo_v0_4/comparison.json          (when v0.4's evaluation read the test splits)
  --comparison7        results/fiorillo_v0_3/comparison.json          (v0.3's)
  --comparison5        results/fiorillo_v0_2/comparison.json          (v0.2's)
  --comparison4        results/public_v0/comparison.json              (v0.1's)
  --prior-art          docs/prior_art.md
Outputs: --readme models/fiorillo_v0_5/release/README.md and --doc docs/model_card_v0_5.md (the same text).

Usage: python3 scripts/build_model_card_v0_5.py [--comparison FILE] [...] [--readme FILE] [--doc FILE]
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Fiorillo"
REPOSITORY = "johanneli/fiorillo-v0.5"
EI, PQA, RW = "evidence_effect", "pubmedqa_answer", "readmit_window"
INPUTS = (  # (name, option, default path relative to the repository root)
    ("comparison", "--comparison", "results/fiorillo_v0_5/comparison.json"),
    ("release_candidate", "--release-candidate", "results/fiorillo_v0_5/release_candidate.json"),
    ("calibration", "--calibration", "results/fiorillo_v0_5/calibration_s9.json"),
    ("recipe", "--recipe", "results/fiorillo_v0_5/recipe_selection.json"),
    ("pool", "--pool", "results/fiorillo_v0_5/readmission_pool_v2.json"),
    ("licenses", "--licenses", "results/fiorillo_v0_5/license_recheck.json"),
    ("fixed", "--fixed", "results/fiorillo_v0_5/fixed_versions.json"),
    ("coverage", "--coverage", "results/fiorillo_v0_5/coverage_long.json"),
    ("selection8", "--selection8", "results/fiorillo_v0_4/article_selection.json"),
    ("calibration7", "--calibration7", "results/fiorillo_v0_3/calibration_s7.json"),
    ("comparison8", "--comparison8", "results/fiorillo_v0_4/comparison.json"),
    ("comparison7", "--comparison7", "results/fiorillo_v0_3/comparison.json"),
    ("comparison5", "--comparison5", "results/fiorillo_v0_2/comparison.json"),
    ("comparison4", "--comparison4", "results/public_v0/comparison.json"),
    ("prior_art", "--prior-art", "docs/prior_art.md"),
)
EARLIER = (("comparison4", "v0.1"), ("comparison5", "v0.2"), ("comparison7", "v0.3"), ("comparison8", "v0.4"))
OUTPUTS = (("readme", "--readme", "models/fiorillo_v0_5/release/README.md"), ("doc", "--doc", "docs/model_card_v0_5.md"))
# Digits in the prose that are not results: names, versions and citations. A test checks that every other number in the
# card (outside the usage section's code block) was read from a results file.
LITERALS = ("Evidence Inference 2.0", "apache-2.0", "Apache-2.0", "SHA-256", "CC BY 4.0", "Gemma 4 31B-it", "laya==0.3.20", "30 days",
            "1999 to 2008", "130 US hospitals", "trials published in 2026", "DeYoung et al. 2020", "Lehman et al. 2019",
            "Strack et al. 2014", "Jin et al. 2019", "Traub et al. 2024", "Kull et al. 2019", "29 September 2026",
            "NVIDIA H100", "Fiorillo reader v1", "johanneli/fiorillo-reader-v1",
            "the levels 0 (no readmission), 1 (after more than 30 days) and 2 (within 30 days)")
PRIORITY = re.compile(r"\b(?:the|a|an)\s+first\s+(?:\w+\s+){0,3}?(?:model|release|system|version|specialist|encoder|kind)\b"
                      r"|\bfirst\s+(?:openly|open|publicly|public|commercially|medical|calibrated|typed)\b")
PRIOR_ART_HEADING = re.compile(r"^### Release-day check, (\d{1,2} [A-Z][a-z]+ \d{4}) \(v0\.5\)\s*$", re.M)
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
METRIC_LABELS = {"log_loss": "Log loss", "ece": "ECE", "brier": "Brier score", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}
CARD_METRICS = ("log_loss", "ece", "brier", "accuracy", "macro_f1")
EI_ANSWERS = (("increased", "significantly increased"), ("decreased", "significantly decreased"),
              ("no_difference", "no significant difference"))
PQA_ANSWERS = (("yes", "yes"), ("no", "no"), ("maybe", "maybe"))
SLOPE_LABELS = (("intercept", "Calibration intercept (0 ideal)"), ("slope", "Calibration slope (1 ideal)"),
                ("intercept_with_slope", "Intercept fitted with the slope (0 ideal)"))
CANDIDATES = ("swap", "long", "lr1", "lr2")


class Missing(Exception):
    """A part's input is missing: a file that does not exist, or a key or number the file does not hold."""


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def spell_date(text):
    """'2026-10-01T06:36:16Z' or '2026-10-01' as '1 October 2026'."""
    d = dt.date.fromisoformat(str(text)[:10])
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def spell_time(text):
    """'2026-10-01T06:36:16Z' as '1 October 2026, 06:36 UTC'; a date alone as the date."""
    s = str(text)
    if len(s) >= 16 and s[10] == "T":
        return f"{spell_date(s)}, {s[11:16]} UTC"
    return spell_date(s)


def sci(v):
    """A learning rate as the plan writes it: 5e-5, 1e-4, 2e-4."""
    mantissa, exponent = f"{float(v):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def count(v):
    v = float(v)
    if not v.is_integer():
        raise Missing(f"{v} is not a count")
    return f"{int(v):,}"


def word(v):
    v = float(v)
    if not v.is_integer() or not 0 <= v < len(WORDS):
        return count(v)
    return WORDS[int(v)]


CARD_WORDS = (  # step 17: wording of comparison.json's statements that the card gives otherwise (the report keeps it)
    ("; Laya's, ModernBERT's and reader v1's notes and session 4's notes on Duan stand.", "."),
    ("EI's test split was read in session 8, for v0.4's bar", "EI's test split was read before, for v0.4's bar"),
    ("(session 8's rule on those items)", "(v0.4's calibration rule on those items)"),
    (" (PUBMEDQA BESIDE BIOGPT-LARGE)", " (docs/fiorillo_v0_5.md, the project's report)"),
    ("(the project's site serves them to its leaderboard)", "(Evidence Inference's site serves them to its leaderboard)"),
    ("v0.5's recipe, calibration and pool were chosen on validation only and frozen in release_candidate.json before any "
     "test prediction of a v0.5 system existed", "v0.5's recipe and calibration were chosen on validation only and frozen "
     "before any test prediction of v0.5's Evidence Inference specialist existed"),
)


def card_words(v):
    """A statement of comparison.json as the card gives it: the clause on Laya's, ModernBERT's, reader v1's and Duan's
    notes left out, since no part of v0.5 uses those models' weights, and the project's session numbers and plan
    headings replaced by the version and the report they belong to."""
    text = str(v)
    for old, new in CARD_WORDS:
        text = text.replace(old, new)
    return text


FORMATS = {
    "f4": lambda v: f"{float(v):.4f}", "f3": lambda v: f"{float(v):.3f}", "f2": lambda v: f"{float(v):.2f}",
    "pct": lambda v: f"{100 * float(v):.2f}", "usd": lambda v: f"{float(v):.2f}", "g": lambda v: f"{float(v):g}",
    "sci": sci, "count": count, "word": word, "date": spell_date, "time": spell_time, "str": str, "card": card_words,
    "f5": lambda v: f"{float(v):.5f}",
}
NUMERIC_FORMATS = {"f4", "f3", "f2", "pct", "usd", "g", "sci", "count", "word", "f5"}


def path_text(keys):
    return " > ".join(str(k) for k in keys)


class Results:
    """The input files, and every number taken from them (sources: the text as printed, the file, the key path and the
    value), so that each number in the card can be traced to its file."""

    def __init__(self, paths, now=None):
        self.paths = {k: Path(v) for k, v in paths.items()}
        self.data, self.sources = {}, []
        self.now = now or dt.datetime.now(dt.timezone.utc)
        for name, path in self.paths.items():
            if path.exists():
                text = path.read_text(encoding="utf-8")
                self.data[name] = text if path.suffix == ".md" else json.loads(text)

    def label(self, name):
        return rel(self.paths[name])

    def exists(self, name):
        return name in self.data

    def get(self, name, *keys):
        if name not in self.data:
            raise Missing(f"{self.label(name)} does not exist")
        obj = self.data[name]
        for i, k in enumerate(keys):
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            elif isinstance(obj, list) and isinstance(k, int) and -len(obj) <= k < len(obj):
                obj = obj[k]
            else:
                raise Missing(f"{self.label(name)} has no {path_text(keys[:i + 1])}")
        return obj

    def has(self, name, *keys):
        try:
            self.get(name, *keys)
            return True
        except Missing:
            return False

    def text(self, name, *keys, fmt="f4"):
        """A value of a file as the card prints it, recorded with its file and key path."""
        v = self.get(name, *keys)
        if fmt in NUMERIC_FORMATS and (v is None or isinstance(v, bool) or not isinstance(v, (int, float))):
            raise Missing(f"{self.label(name)} gives no number at {path_text(keys)}")
        if fmt not in NUMERIC_FORMATS and (v is None or isinstance(v, (dict, list))):
            raise Missing(f"{self.label(name)} gives no value at {path_text(keys)}")
        out = FORMATS[fmt](v)
        self.sources.append({"text": out, "file": self.label(name), "keys": list(keys), "value": v, "format": fmt})
        return out

    def interval(self, name, *keys, fmt="f4"):
        """An entry {estimate, low, high} as 'estimate (low to high)'."""
        est = self.text(name, *keys, "estimate", fmt=fmt)
        entry = self.get(name, *keys)
        if entry.get("low") is None or entry.get("high") is None:
            return f"{est} (no interval)"
        return f"{est} ({self.text(name, *keys, 'low', fmt=fmt)} to {self.text(name, *keys, 'high', fmt=fmt)})"

    def derived(self, value, fmt, inputs, how):
        """A number computed from others: inputs are (file name, key, key, ...) tuples; how is 'difference' (the first
        minus the rest), 'sum', 'mean', 'complement' (1 minus the first), 'length' (the number of items of the first),
        'passes' (the first's length, doubled when the second is true) or 'readers' (the members of the first among
        READERS whose weight in the second is above zero)."""
        out = FORMATS[fmt](value)
        self.sources.append({"text": out, "file": "derived", "inputs": [[self.label(t[0]), list(t[1:])] for t in inputs],
                             "names": [list(t) for t in inputs], "how": how, "value": value, "format": fmt})
        return out

    def clock(self):
        out = spell_date(self.now.strftime("%Y-%m-%d"))
        self.sources.append({"text": out, "file": "clock", "keys": [], "value": self.now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "format": "date"})
        return out

    def year(self):
        out = str(self.now.year)
        self.sources.append({"text": out, "file": "clock", "keys": ["year"], "value": self.now.year, "format": "str"})
        return out


class Card:
    """The card's parts; a part whose input is missing becomes a NOT BUILT line, and the numbers it had already taken
    are dropped from the sources."""

    def __init__(self, res):
        self.res = res
        self.not_built = []

    def part(self, name, fn, *args):
        mark = len(self.res.sources)
        try:
            return fn(*args)
        except Missing as err:
            del self.res.sources[mark:]
            self.not_built.append(f"{name}: {err}")
            return f"NOT BUILT: {name}, {err}."



# ---------------------------------------------------------------- v0.5's specialists (release_candidate.json)

RC_EI = ("release_candidate", "specialists", EI)
RC_PQA = ("release_candidate", "specialists", PQA)
RC_RW = ("release_candidate", "specialists", RW)
LARGE_LABELS = {"gemma4_31b": "Gemma 4 31B-it, calibrated on validation",
                "gemma4_31b_long": "Gemma 4 31B-it reading the same longer inputs, calibrated on validation"}
LARGE_NAMES = {"gemma4_31b": "Gemma 4 31B-it", "gemma4_31b_long": "Gemma 4 31B-it on the same longer inputs"}
MEMBER_FILES = {"xgboost_codes": "xgboost_codes.json", "catboost_codes": "catboost_codes.cbm", "xgboost_tuned": "xgboost_tuned.json",
                "catboost_tuned": "catboost_tuned.cbm", "lightgbm_tuned": "lightgbm_tuned.txt", "tabicl": "tabicl.pkl and tabicl_state.json"}
MEMBER_WORDS = {
    "xgboost_codes": "XGBoost with v0.4's tuned setting, refitted with the recoded diagnosis fields",
    "catboost_codes": "CatBoost with v0.4's tuned setting, refitted with the recoded diagnosis fields",
    "xgboost_tuned": "XGBoost, tuned for v0.4",
    "catboost_tuned": "CatBoost, tuned for v0.4",
    "lightgbm_tuned": "LightGBM with v0.3's tuned setting, refitted for v0.5",
    "tabicl": "TabICLv2, a tabular foundation model, with the training split as its context",
    "arm_laya": f"{NAME} v0.1, Laya fine-tuned (johanneli/fiorillo-v0.1, not copied into this release)",
    "s7_b3": "s7_b3's own readmission answer, with its v0.3 calibration",
}
READERS = ("s7_b3", "arm_laya")


def ei_names(res):
    """v0.5's EI specialist, its form and its readers (release_candidate.json)."""
    return {"system": str(res.get(*RC_EI, "system")), "base": str(res.get(*RC_EI, "base_system")),
            "swap": bool(res.get(*RC_EI, "swap_averaging")), "runs": [str(r) for r in res.get(*RC_EI, "runs")],
            "singles": [str(s) for s in res.get(*RC_EI, "single_pass_systems")], "recipe": str(res.get(*RC_EI, "recipe")),
            "family": str(res.get(*RC_EI, "family"))}


def pool_members(res):
    """The readmission pool's members with a weight above zero, in its order (release_candidate.json)."""
    members = [str(m) for m in res.get(*RC_RW, "members")]
    weights = res.get(*RC_RW, "weights")
    return [m for m in members if float(weights[m]) > 0]


def ei_passes(res):
    n = ei_names(res)
    passes = len(n["singles"]) * (2 if n["swap"] else 1)
    return res.derived(passes, "word", [RC_EI + ("single_pass_systems",), RC_EI + ("swap_averaging",)], "passes")


def ei_form(res):
    n = ei_names(res)
    runs = res.derived(len(n["runs"]), "word", [RC_EI + ("runs",)], "length")
    if n["base"] == "s9_ei_wavg":
        form = f"one model whose adapters, projection and head average its recipe's {runs} runs exactly (s9_ei_wavg)"
    else:
        form = f"the probability average of its recipe's {runs} runs (s9_ei)"
    return form + (", with swap averaging" if n["swap"] else "")


def summary_table(res):
    n = ei_names(res)
    members = pool_members(res)
    rw_passes = res.derived(sum(1 for m in members if m in READERS), "word", [RC_RW + ("members",), RC_RW + ("weights",)],
                            "readers")
    weights = "; ".join(f"{m}, weight {res.text(*RC_RW, 'weights', m, fmt='g')}" for m in members)
    rows = ["| Question id | Specialist | Reader passes a question | Calibration |", "|---|---|---|---|",
            f"| {EI} | {n['system']}: {ei_form(res)} | {ei_passes(res)} | {res.text(*RC_EI, 'family', fmt='str')}, fitted on the "
            f"clean validation prompts |",
            f"| {PQA} | s7_b3, unchanged from v0.3 | one | {res.text(*RC_PQA, 'family', fmt='str')}, v0.3's fit |",
            f"| {RW} | the readmission pool: {weights} | {rw_passes} | the pool's temperature T, "
            f"{res.text(*RC_RW, 'T', fmt='f4')} |"]
    return "\n".join(rows)


def what_it_is(card):
    res = card.res
    return "\n\n".join([
        "## What v0.5 is",
        (f"{NAME} v0.5 routes each question to its own specialist and hands nothing off: as released, its hand-off rate is "
         "zero, and every answer is the specialist's own. The rate was fixed at zero before v0.5 was trained: in v0.4's "
         "evaluation, handing the least confident answers to Gemma 4 31B-it lowered Evidence Inference's accuracy at every "
         "rate tested and barely changed PubMedQA's. The hand-off analyses in the question cards are descriptive. A "
         "reader pass is one forward pass of a language-model reader; the tree members of the readmission pool need none."),
        card.part("the specialists", summary_table, res),
        card.part("the freeze and the registration", freeze_text, res)])


def freeze_text(res):
    return (f"Every choice for v0.5 (the recipe, the weight average, swap averaging, the calibrations, the readmission pool "
            f"and the limit of the release bar's first criterion) was made on validation data and frozen at "
            f"{res.text('release_candidate', 'time_utc', fmt='time')}, before any test prediction of v0.5's new Evidence "
            "Inference specialist existed. The PubMedQA specialist and the readmission pool that the pool rule kept are "
            "unchanged from v0.3 and v0.4, so their test results were known before the freeze; both were kept by decisions "
            "fixed before any v0.5 training. "
            f"The bar, the recipe rule, the calibration rules and the reading checks' readings had been filed on "
            f"{res.text('release_candidate', 'osf_update', 'filed_utc', fmt='time')} as an update to the project's "
            f"registration on the Open Science Framework (OSF): {res.text('release_candidate', 'osf_update', 'link', fmt='str')}.")


def not_released(res):
    tr = ("licenses", "by_split", "train")
    articles, clean = res.get(*tr, "articles"), res.get(*tr, "clean_articles")
    if not isinstance(articles, int) or not isinstance(clean, int):
        raise Missing(f"{res.label('licenses')} gives no article counts for the training split")
    return (f"v0.4's EI specialist, s8_ei, learned from all {res.text(*tr, 'articles', fmt='count')} articles "
            f"({res.text(*tr, 'prompts', fmt='count')} prompts) of Evidence Inference 2.0's training split. By their own "
            f"license statements, {res.derived(articles - clean, 'count', [tr + ('articles',), tr + ('clean_articles',)], 'difference')} "
            "of them (derived) are under non-commercial, no-derivatives, share-alike, verbatim-only or publishers' own terms, "
            "or state no license, and whether weights trained on them may be published is UNVERIFIABLE: no source says. s8_ei "
            "therefore stays "
            "a research model: its weights are not released, and its test results appear in the Evidence Inference card only "
            f"for reference. v0.5's EI specialist learned only from the {res.text(*tr, 'clean_articles', fmt='count')} training "
            f"articles ({res.text(*tr, 'clean_prompts', fmt='count')} prompts) whose own statement is CC BY, CC0 or the "
            "public domain.")


def licenses_part(res):
    tr, va = ("licenses", "by_split", "train"), ("licenses", "by_split", "validation")
    a, b = res.get(*tr, "clean_articles"), res.get(*va, "clean_articles")
    if not isinstance(a, int) or not isinstance(b, int):
        raise Missing(f"{res.label('licenses')} gives no clean article counts")
    total = res.derived(a + b, "count", [tr + ("clean_articles",), va + ("clean_articles",)], "sum")
    return "\n".join([
        (f"- Evidence Inference 2.0: its annotations are under the MIT License (DeYoung et al. 2020; Lehman et al. 2019). Every "
         "article that v0.5's EI specialist learned from, or was calibrated and chosen on, states CC BY (any version), CC0 or "
         "the public domain in its own license statement as PubMed Central serves it, and no part of the statement "
         "restricts commercial use, derivatives or the license of what is made from it: "
         f"{res.text(*tr, 'clean_articles', fmt='count')} training and {res.text(*va, 'clean_articles', fmt='count')} "
         f"validation articles, {total} in all (derived). The statements were read from PubMed Central between "
         f"{res.text('licenses', 'read_utc', 'first', fmt='time')} and {res.text('licenses', 'read_utc', 'last', fmt='time')} "
         f"and classified at {res.text('licenses', 'written_utc', fmt='time')}. NOTICE credits each of these articles with its "
         "PMCID, its license and its citation as Europe PMC's REST service gives it."),
        "- PubMedQA: MIT License (Jin et al. 2019); see the next section.",
        "- UCI Diabetes 130-US hospitals for years 1999 to 2008: CC BY 4.0 (Strack et al. 2014).",
        (f"- Qwen3-4B-Base: Apache-2.0 ({res.text('fixed', 'qwen3_4b_base', 'repo', fmt='str')} at revision "
         f"{res.text('fixed', 'qwen3_4b_base', 'revision', fmt='str')})."),
        "- The laya package, whose decision head the readers use: Apache-2.0.",
        ("- This release: Apache-2.0 (LICENSE). NOTICE carries the attributions, the clean articles and the line \"Courtesy "
         "of the U.S. National Library of Medicine\".")])


def pubmedqa_statement(res):
    members = pool_members(res)
    weights = res.get(*RC_RW, "weights")
    if "s7_b3" in members:
        pool_words = f"a member of the readmission pool, with weight {res.text(*RC_RW, 'weights', 's7_b3', fmt='g')}"
    elif "s7_b3" in weights:
        pool_words = "in the readmission pool with no weight, so it does not answer readmission"
    else:
        pool_words = "not a member of the readmission pool"
    return ("Every Evidence Inference article that v0.5's EI specialist learned from, or was calibrated and chosen on, is "
            "CC BY, CC0 or the public domain. The specialist's training started from s7_b3's weights, and s7_b3 learned "
            "from PubMedQA's abstracts (its automatically labeled items, PQA-A; its unlabeled items, PQA-U; and the training "
            "part of its expert-labeled items, PQA-L) under PubMedQA's MIT License. Unlike "
            "Evidence Inference's articles, those abstracts' own licenses were not checked one by one, and most PubMed "
            "abstracts are the publishers'; they rest on PubMedQA's license. s7_b3's labels for PQA-U came from this "
            "project's own labeler, not from an outside language model. The labeler is a model built on Fiorillo reader v1 "
            "(johanneli/fiorillo-reader-v1) that reads each abstract's conclusion. s7_b3 is v0.5's PubMedQA specialist and is "
            f"{pool_words}.")


def prior_art(res):
    text = res.get("prior_art")
    found = PRIOR_ART_HEADING.findall(text)
    if not found:
        raise Missing(f"{res.label('prior_art')} has no heading \"### Release-day check, <D Month YYYY> (v0.5)\"")
    date = found[-1]
    res.sources.append({"text": date, "file": res.label("prior_art"), "keys": ["heading", f"### Release-day check, {date} (v0.5)"],
                        "value": date, "format": "str"})
    return (f"The prior-art check of the release day, {date}, is in the project's docs/prior_art.md. This card makes no "
            "claim that v0.5 is the first of its kind; a claim of a first would be worded \"to our knowledge\". The check found "
            "nothing published since the previous check that makes v0.5 redundant.")


# ---------------------------------------------------------------- tables shared by the question cards

def cell(res, *keys, fmt="f4"):
    """An interval for a table cell; "not computed" when the file holds no number there (never a made-up one)."""
    mark = len(res.sources)
    try:
        return res.interval(*keys, fmt=fmt)
    except Missing:
        del res.sources[mark:]
        return "not computed"


def value_cell(res, *keys, fmt="f4"):
    mark = len(res.sources)
    try:
        return res.text(*keys, fmt=fmt)
    except Missing:
        del res.sources[mark:]
        return "not computed"


def intervals_note(res):
    s = ("comparison", "settings")
    return (f"Each test result has an interval ({res.text(*s, 'interval', fmt='str')}) from "
            f"{res.text(*s, 'resamples', fmt='count')} resamples: Evidence Inference resamples its articles, the other two "
            f"questions their items. ECE is the expected calibration error of the top answer in "
            f"{res.text(*s, 'ece_bins', fmt='count')} equal-width bins. A percentile interval of ECE can lie above its "
            "estimate: ECE adds up the gaps between confidence and accuracy over the bins, and the noise of a resample adds "
            "to those gaps, so resampled values tend to exceed the value on the whole split.")


def metrics_table(res, ev, rows):
    """The test metrics: the specialist's row first (required), then the references comparison.json holds."""
    m = ("comparison", "metrics", ev)
    res.get(*m)  # a missing comparison.json, or one without this question, makes the whole table NOT BUILT
    lines = ["| System | n | " + " | ".join(METRIC_LABELS[k] for k in CARD_METRICS) + " |", "|" + "---|" * (len(CARD_METRICS) + 2)]
    absent = []
    for i, (system, label) in enumerate(rows):
        if not res.has(*m, system):
            if i == 0:
                raise Missing(f"{res.label('comparison')} has no test metrics for {system}")
            absent.append(label)
            continue
        cells = [cell(res, *m, system, k) for k in CARD_METRICS]
        lines.append(f"| {label} | {value_cell(res, *m, system, 'n', fmt='count')} | " + " | ".join(cells) + " |")
    out = "\n".join(lines)
    if absent:
        out += "\n\nNot in comparison.json, so not shown: " + "; ".join(absent) + "."
    return out


def slope_table(res, ev, answers):
    s = ("comparison", "calibration_slope_intercept", ev)
    lines = [("Calibration slope and intercept on the test split, each answer against the rest: the slope is the coefficient "
              "of a logistic regression of the answer's indicator on the logit of its calibrated probability (ideal one); the "
              "intercept is that regression's intercept with the logit as an offset (ideal zero); the third column is the "
              f"intercept fitted together with the slope (ideal zero). {res.text(*s, 'n', fmt='count')} test items."), "",
             "| Answer | Items with this answer | Intercept | Slope | Intercept fitted with the slope |", "|---|---|---|---|---|"]
    for key, label in answers:
        a = s + ("answers", key)
        lines.append(f"| {label} | {value_cell(res, *a, 'rows_with_this_answer', fmt='count')} | {cell(res, *a, 'intercept')} | "
                     f"{cell(res, *a, 'slope')} | {cell(res, *a, 'intercept_with_slope')} |")
    return "\n".join(lines)


def rate_of(key):
    return int(str(key).split("_", 1)[1])


def handoff_part(res, ev, system):
    pairs = res.get("comparison", "k7", "pairs")
    found = [i for i, p in enumerate(pairs) if isinstance(p, dict) and p.get("question") == ev and p.get("fiorillo") == system]
    if not found:
        raise Missing(f"{res.label('comparison')} has no hand-off pair for {system} on {ev} (k7 > pairs)")
    p = ("comparison", "k7", "pairs", found[0])
    large = str(res.get(*p, "large_model"))
    rates = sorted(res.get(*p, "registered_rates"), key=rate_of)
    lines = [(f"As released, v0.5 hands nothing off: its accuracy on the {res.text(*p, 'n', fmt='count')} test items is "
              f"{res.text(*p, 'fiorillo_alone', 'accuracy')}, and {LARGE_NAMES.get(large, large)} alone scored "
              f"{res.text(*p, 'large_model_alone', 'accuracy')}. Descriptive, and not part of the release: had the specialist "
              "handed some answers to that model, the combined accuracy on the test split would have been as follows. The "
              "confidence rule hands off the least confident answers first; random hand-off is that rate's expected value; "
              "the oracle first hands off the items the specialist gets wrong and the large model right, which bounds what any "
              "rule could gain; the large model's own confidence is a reference, not a usable rule."), "",
             "| Hand-off rate, percent | Items handed off | Confidence rule | Random hand-off, expected | Oracle, a bound | "
             "Large model's own confidence |", "|---|---|---|---|---|---|"]
    for key in rates:
        r = p + ("registered_rates", key)
        res.sources.append({"text": str(rate_of(key)), "file": res.label("comparison"), "keys": list(r[1:]), "value": key,
                            "format": "rate"})
        lines.append(f"| {rate_of(key)} | {value_cell(res, *r, 'handed_off', fmt='count')} | {cell(res, *r, 'confidence')} | "
                     f"{cell(res, *r, 'random_expected')} | {cell(res, *r, 'oracle')} | {cell(res, *r, 'large_model_confidence')} |")
    lines += ["", (f"Selective answering on its own (no hand-off): the area under the risk-coverage curve (AURC) is "
                   f"{cell(res, *p, 'areas', 'aurc')}, its excess over a perfect ranking's (E-AURC) "
                   f"{cell(res, *p, 'areas', 'e_aurc')}, and the area under the generalized risk-coverage curve (AUGRC; Traub "
                   f"et al. 2024) {cell(res, *p, 'areas', 'augrc')}.")]
    return "\n".join(lines)


def split_reads(res, ev):
    """The project's evaluations whose comparisons scored this question's test split, with their dates."""
    out = []
    for name, version in EARLIER:
        if res.exists(name) and res.has(name, "metrics", ev):
            key = "time_utc" if res.has(name, "time_utc") else "created"
            out.append(f"{version} ({res.text(name, key, fmt='time')})")
    out.append(f"v0.5 ({res.text('comparison', 'written_utc', fmt='time')})")
    return out


# ---------------------------------------------------------------- the Evidence Inference card

EI_USE = ("For researchers who extract or check the results of randomized trials. Given a trial's article and one comparison "
          "in it (an intervention, a comparator and an outcome), it gives the probability that the article reports the outcome "
          "significantly higher with the intervention than with the comparator, significantly lower, or not significantly "
          "different. It reads the article as given; its only other source about a trial is what its base model may have "
          "learned in pretraining, which the reading checks below look for but cannot rule out. It is for research only, not "
          "for clinical decisions, and it has been tested only on Evidence Inference 2.0's prompts, in English.")
EI_UPDATE = ("This version is fixed as released. The project's next step for this question is a test on trials published "
             "after the base model was released, so that a good answer cannot come from recalling the trial; a later "
             "Evidence Inference specialist will be released only if it meets a bar fixed before it is trained, as this one did.")
RECIPE_WORDS = {
    "b": "v0.4's recipe, unchanged",
    "swap": ("swap augmentation, in which each clean training prompt is followed by a copy with the intervention and the "
             "comparator exchanged and its answer exchanged (significantly increased and significantly decreased swapped, no "
             "significant difference kept)"),
    "long": "longer inputs",
    "lr1": "a higher learning rate for the adapters",
    "lr2": "a higher learning rate for the adapters",
}
PART_WORDS = {"swap": "swap augmentation", "long": "longer inputs", "lr1": "a higher learning rate for the adapters",
              "lr2": "a higher learning rate for the adapters"}


def ei_inputs(res):
    max_len = res.text(*RC_EI, "max_len", fmt="count")
    q = ("A record whose state is {\"intervention\", \"comparator\", \"outcome\", \"article\"}: the comparison's three parts as "
         "short texts, and the article's full text in Evidence Inference's text format (the title and abstract, then the body, "
         "whose sections begin with lines that begin with \"BODY.\"). The passage selection below cuts the article so that the "
         f"whole input, article and question together, fits {max_len} tokens. The question, word for word: \"Based only on this "
         "article, how did the outcome differ with the intervention compared with the comparator?\", with the options "
         "significantly increased (the outcome was significantly higher with the intervention than with the comparator), "
         "significantly decreased (significantly lower) and no significant difference (the article reports no significant "
         "difference in the outcome between the intervention and the comparator).")
    return q


def recipe_words(res):
    w = ("recipe", "winner")
    recipe = str(res.get(*w, "recipe"))
    st = w + ("settings",)
    if recipe == "combo":
        parts = [str(x) for x in res.get(*st, "parts")]
        what = "the passing changes together: " + ", ".join(PART_WORDS.get(x, x) for x in parts)
    else:
        what = RECIPE_WORDS.get(recipe, recipe)
    return (f"is {what}. It trains for {res.text(*st, 'epochs', fmt='word')} epoch{'s' if res.get(*st, 'epochs') != 1 else ''} with "
            f"inputs of {res.text(*st, 'max_len', fmt='count')} tokens and the adapters' learning rate "
            f"{res.text(*st, 'lr_adapters', fmt='sci')}")


def ei_specialist(res):
    n = ei_names(res)
    runs = ", ".join(n["runs"])
    one_reader = len(n["singles"]) == 1
    text = [(f"v0.5's EI specialist is {n['system']}: {ei_form(res)}. {'Its reader is' if one_reader else 'Its readers are'} "
             "Qwen3-4B-Base "
             f"({res.text('fixed', 'qwen3_4b_base', 'repo', fmt='str')}, revision {res.text('fixed', 'qwen3_4b_base', 'revision', fmt='str')}) "
             "with low-rank adaptation (LoRA) adapters on every linear layer of the decoder, a projection and Laya's decision "
             "head, trained from s7_b3's weights; the base model's own weights stay fixed. Its recipe, chosen by the recipe rule "
             f"on the {res.text('recipe', 'n_clean_validation', fmt='count')} clean validation prompts, "
             f"{recipe_words(res)}. Its runs are {runs}.")]
    text += recipe_on_test(res, n)
    wa = ("recipe", "weight_average")
    if res.has(*wa, "score"):
        verdict = ("not above it by more than the margin, so v0.5 uses the weight average, one reader in place of the runs"
                   if res.get(*wa, "uses_wavg") else "above it by more than the margin, so v0.5 uses the runs' probability average")
        text.append(f"The weight average of the runs scored {res.text(*wa, 'score')} on the clean validation prompts against "
                    f"{res.text(*wa, 's9_ei_score')} for their probability average, with a margin of {res.text(*wa, 'm')}: "
                    f"{verdict}.")
    sa = ("recipe", "swap_averaging")
    if res.get(*sa, "kept"):
        text.append(f"Swap averaging is kept: with it the score was {res.text(*sa, 'swap_averaged_score')}, below the "
                    f"{res.text(*sa, 'plain_score')} without it by more than the margin, {res.text(*sa, 'm')}. So each question "
                    "takes two passes of each reader: one on the prompt as written and one on the prompt with the intervention "
                    "and the comparator exchanged (the same article), whose significantly increased and significantly decreased "
                    "are exchanged before the two answers are averaged; a prompt whose intervention and comparator are the same "
                    "text keeps its single answer.")
    else:
        text.append(f"Swap averaging is not kept: with it the score was {res.text(*sa, 'swap_averaged_score')}, against "
                    f"{res.text(*sa, 'plain_score')} without it, not below by more than the margin, {res.text(*sa, 'm')}, so each "
                    f"question takes one pass of {'its reader' if one_reader else 'each reader'}.")
    return " ".join(text)


def recipe_on_test(res, n):
    """K3 on the test split (descriptive): the specialist minus the base recipe's runs, beside the recipe's lead on the
    clean validation prompts (step 17: the card says when the lead was not seen on test)."""
    comps = res.get("comparison", "comparisons") if res.has("comparison", "comparisons") else []
    idx = [i for i, c in enumerate(comps) if isinstance(c, dict) and c.get("section") == "K3_test"
           and c.get("system") == n["system"] and c.get("reference") == "s9_b" and c.get("metric") == "log_loss"]
    win = ("recipe", "candidates", n["recipe"])
    if not idx or not res.has(*win, "base_mean_12") or not res.has(*win, "score"):
        return []
    d = ("comparison", "comparisons", idx[0])
    lead = res.derived(float(res.get(*win, "base_mean_12")) - float(res.get(*win, "score")), "f4",
                       [win + ("base_mean_12",), win + ("score",)], "difference")
    reading = res.text(*d, "reading", fmt="str")
    lo, hi = res.get(*d, "difference", "low"), res.get(*d, "difference", "high")
    seen = "was not seen on test: the interval excludes a lead that large" if -float(lead) < float(lo) or -float(lead) > float(hi) \
        else "is within the test interval"
    return [f"On the test split, its log loss minus that of the base recipe's runs (s9_b) was "
            f"{res.interval(*d, 'difference')}: {reading}. The recipe's lead on the clean validation prompts, {lead} "
            f"(derived), {seen}."]


def ei_recipe_table(res):
    mg = ("recipe", "margin")
    b1, b2 = res.get(*mg, "base_scores", "s9_b_1"), res.get(*mg, "base_scores", "s9_b_2")
    base = res.derived((float(b1) + float(b2)) / 2, "f4", [mg + ("base_scores", "s9_b_1"), mg + ("base_scores", "s9_b_2")], "mean")
    lines = [("The recipe rule's scores, each the out-of-fold log loss of v0.4's calibration rule (the lower of the two "
              "families temperature and temperature_intercepts) on the clean validation "
              "prompts (lower is better). A candidate's score is the mean of its runs' scores; it passes if that is below the "
              f"base recipe's, {base} (derived: the mean of s9_b_1's {res.text(*mg, 'base_scores', 's9_b_1')} and s9_b_2's "
              f"{res.text(*mg, 'base_scores', 's9_b_2')}), by more than the margin m, {res.text(*mg, 'm')} (the spread of the "
              f"three base runs' scores, {res.text(*mg, 'spread')}, with a floor of {res.text(*mg, 'floor')}), and each of its "
              "runs scores below the base run with the same seed."), "",
             "| Candidate | Runs | Score | Passes |", "|---|---|---|---|"]
    cand = res.get("recipe", "candidates")
    for c in CANDIDATES:
        if c not in cand:
            continue
        e = ("recipe", "candidates", c)
        if not res.has(*e, "score"):
            lines.append(f"| {c} | none | not run | {res.text(*e, 'status', fmt='str') if res.has(*e, 'status') else 'not run'} |")
            continue
        lines.append(f"| {c} | {', '.join(str(r) for r in res.get(*e, 'runs'))} | {res.text(*e, 'score')} | "
                     f"{'yes' if res.get(*e, 'passes') else 'no'} |")
    co = ("recipe", "combination")
    if res.has(*co, "score"):
        parts = ", ".join(str(x) for x in res.get(*co, "parts"))
        lines.append(f"| combination ({parts}) | {', '.join(str(r) for r in res.get(*co, 'runs'))} | {res.text(*co, 'score')} | "
                     f"{'yes' if res.get(*co, 'passes') else 'no'} |")
    lines += ["", f"The winning recipe: {res.text('recipe', 'winner', 'recipe', fmt='str')}."]
    return "\n".join(lines)


def ei_families(res):
    t = ("calibration", "families_table_v05")
    chosen = str(res.get(*t, "chosen"))
    lines = [("Calibration: the family with the lowest out-of-fold log loss on the clean validation prompts, among four, "
              f"fitted on all of them; the chosen family is {res.text(*t, 'chosen', fmt='str')}."), "",
             "| Family | Out-of-fold log loss: mean over the repeats (lowest and highest repeat) |", "|---|---|"]
    for fam in res.get(*t, "table"):
        f = t + ("table", fam)
        mark = " (chosen)" if fam == chosen else ""
        lines.append(f"| {fam}{mark} | {res.text(*f, 'mean')} ({res.text(*f, 'min')} to {res.text(*f, 'max')}) |")
    lines += ["", ("The families: temperature, one temperature for all the answers; temperature_intercepts, a temperature and "
                   "an intercept for each answer but the first; vector, a scale and an intercept for each answer; "
                   "dirichlet_odir, a full matrix on the log probabilities and an intercept for each answer, with the "
                   "matrix's off-diagonal entries regularized (Kull et al. 2019).")]
    return "\n".join(lines)


def ei_cost(res):
    c = ("comparison", "cost_per_1000_questions")
    e = c + ("specialists", EI)
    passes = res.derived(len(res.get(*e, "passes")), "word", [e + ("passes",)], "length")
    one = passes == "one"
    return (f"Cost: {passes} reader {'pass' if one else 'passes'} a question. In the test predictions, run alone on a "
            f"rented GPU (an NVIDIA H100), {'it' if one else 'they'} took "
            f"{res.text(*e, 'seconds_per_question', fmt='f3')} GPU seconds a question, which at its rate of "
            f"{res.text(*c, 'hourly_usd', fmt='usd')} dollars an hour is {res.text(*e, 'usd', fmt='f3')} dollars per thousand "
            "questions.")


def ei_training(res):
    tr, va = ("licenses", "by_split", "train"), ("licenses", "by_split", "validation")
    return (f"Evidence Inference 2.0 (DeYoung et al. 2020): its training split's prompts whose article is clean, "
            f"{res.text(*tr, 'clean_prompts', fmt='count')} of {res.text(*tr, 'prompts', fmt='count')} prompts in "
            f"{res.text(*tr, 'clean_articles', fmt='count')} of {res.text(*tr, 'articles', fmt='count')} articles. The "
            f"specialist was calibrated and chosen on the clean validation prompts, {res.text(*va, 'clean_prompts', fmt='count')} "
            f"of {res.text(*va, 'prompts', fmt='count')} in {res.text(*va, 'clean_articles', fmt='count')} of "
            f"{res.text(*va, 'articles', fmt='count')} articles. An article is clean when its own license statement, as PubMed "
            "Central serves it, is CC BY (any version), CC0 or the public domain and no part of it restricts commercial use, "
            "derivatives or the license of what is made from it (Licenses, above). The annotations are under the MIT License. "
            "Before Evidence Inference, the readers started from s7_b3, whose training data are in the PubMedQA card.")


def ei_splits(res):
    va = ("licenses", "by_split", "validation")
    v05 = str(res.get(*RC_EI, "system"))
    return "\n".join([
        (f"- Validation: Evidence Inference's validation split, {res.text(*va, 'prompts', fmt='count')} prompts in "
         f"{res.text(*va, 'articles', fmt='count')} articles. Every choice for v0.5 read only its "
         f"{res.text(*va, 'clean_prompts', fmt='count')} clean prompts: the recipe, the weight average, swap averaging and the "
         f"calibration, all frozen at {res.text('release_candidate', 'time_utc', fmt='time')}. The large model's calibration "
         "and the hand-off analyses read all of them."),
        (f"- Test: Evidence Inference's test split, {res.text('comparison', 'metrics', EI, v05, 'n', fmt='count')} prompts. It "
         "was first read for v0.4, whose comparisons are dated "
         f"{res.text('comparison8', 'time_utc' if res.has('comparison8', 'time_utc') else 'created', fmt='time')}; then, before "
         "v0.5's training, by the reading checks on v0.4's specialist, after the registration's update was filed "
         f"({res.text('release_candidate', 'osf_update', 'filed_utc', fmt='time')}); and for v0.5 after the freeze, its "
         f"comparisons written {res.text('comparison', 'written_utc', fmt='time')}. v0.5's bar is the second release bar judged "
         "on this split.")])


def ei_rows(res):
    v05 = str(res.get(*RC_EI, "system"))
    large = str(res.get("release_candidate", "reference", "criterion_3_large_model"))
    return [(v05, f"**v0.5's EI specialist ({v05})**"), (large, LARGE_LABELS.get(large, large)),
            ("s8_ei", "v0.4's EI specialist, s8_ei (not released)"), ("prior", "Prior: the training split's answer shares")]


def ei_metrics(res):
    return metrics_table(res, EI, ei_rows(res))


def bar_part(res):
    c = ("comparison", "release_bar", "criteria")
    lim = ("release_candidate", "ece_limits", EI)
    v05 = str(res.get(*RC_EI, "system"))
    large = str(res.get("release_candidate", "reference", "criterion_3_large_model"))
    m = ("comparison", "metrics", EI)
    nbar = res.text("comparison", "settings", "resamples_bar", fmt="count")
    rows = [("v0.5 was released because it met v0.4's four criteria on this test split, with v0.5's EI specialist in v0.4's "
             f"place (overall: {res.text('comparison', 'release_bar', 'overall', fmt='str')})."), "",
            "| Criterion | Result | Values |", "|---|---|---|",
            (f"| ECE at most the limit: the larger of {res.text(*lim, 'floor', fmt='f2')} and the 95th percentile of the ECE of a "
             "model calibrated by construction at the test split's size, fixed in the freeze | "
             f"{res.text(*c, '1', 'result', fmt='str')} | ECE {res.text(*c, '1', 'values', 'ece')} against a limit of "
             f"{res.text(*c, '1', 'values', 'limit')} (the percentile: {res.text(*lim, 'p95')}) |"),
            (f"| Log loss below the prior's, the interval of the difference ({nbar} article resamples) wholly below zero | "
             f"{res.text(*c, '2', 'result', fmt='str')} | {res.text(*m, v05, 'log_loss', 'estimate')} against "
             f"{res.text(*m, 'prior', 'log_loss', 'estimate')}; difference {res.interval(*c, '2', 'values')} |"),
            (f"| Log loss below that of {LARGE_NAMES.get(large, large)} (calibrated on validation), the interval of the difference ({nbar} article resamples) "
             f"wholly below zero | {res.text(*c, '3', 'result', fmt='str')} | {res.text(*m, v05, 'log_loss', 'estimate')} against "
             f"{res.text(*m, large, 'log_loss', 'estimate')}; difference {res.interval(*c, '3', 'values')} |"),
            (f"| Macro-F1 not below that large model's point estimate | {res.text(*c, '4', 'result', fmt='str')} | "
             f"{res.text(*c, '4', 'values', 'specialist')} against {res.text(*c, '4', 'values', 'large_model')} |")]
    return "\n".join(rows)


def reading_row(res, table, system, inp, stat):
    for i, r in enumerate(table):
        if isinstance(r, dict) and r.get("system") == system and r.get("input") == inp and r.get("statistic") == stat:
            return cell(res, "comparison", "k4", "table", i)
    return "not computed"


def reading_n(res, table, system, inp, stat):
    """The number of answers a reading check's statistic counts (the k4 table's n), or None."""
    for i, r in enumerate(table):
        if isinstance(r, dict) and r.get("system") == system and r.get("input") == inp and r.get("statistic") == stat \
                and r.get("n") is not None:
            return res.text("comparison", "k4", "table", i, "n", fmt="count")
    return None


SUBSET = "swap, comparator a time point or baseline"


def reading_subset(res, table):
    """Step 17: the swap check on the prompts whose comparator is a time point or baseline (I-37 reports them apart),
    where exchanging the intervention and the comparator need not reverse the answer."""
    parts = []
    for system, label in (("v05", "v0.5's specialist"), ("s8_ei", "v0.4's")):
        ne, ns = reading_n(res, table, system, SUBSET, "exchange_share"), reading_n(res, table, system, SUBSET, "stay_share")
        if ne is None or ns is None:
            continue
        parts.append(f"{label} exchanged {reading_row(res, table, system, SUBSET, 'exchange_share')} of its {ne} direction "
                     f"answers and kept {reading_row(res, table, system, SUBSET, 'stay_share')} of its {ns} "
                     "no-significant-difference answers")
    if not parts:
        return []
    return ["", ("On the exchanged prompts whose comparator is a time point or baseline, where exchanging the two need not "
                 "reverse the answer: " + "; ".join(parts) + ".")]


def reading_part(res):
    k4 = ("comparison", "k4")
    if res.has(*k4, "not_built"):
        raise Missing(f"{res.label('comparison')} k4: {res.get(*k4, 'not_built')}")
    table = res.get(*k4, "table")
    labels = {"none": "none: no article", "title": "title: the title alone",
              "swap": "swap: intervention and comparator exchanged"}
    lines = [("The reading checks ask how much of an EI specialist's answer comes from reading the article: with no article "
              "(none), with the article's title alone (title), and with the intervention and the comparator exchanged and the "
              "expected answer exchanged (swap). Accuracy and macro-F1 are against the original answers (none, title) or the "
              "exchanged ones (swap); agreement is the share of answers equal to the specialist's answer on its full input; for "
              "swap, the exchange share is the share of its increased and decreased answers that exchange, and the stay share "
              "the share of its no-significant-difference answers that stay. The readings were fixed before the checks ran and "
              "gate no release."), "",
             "| System | Input | Accuracy | Macro-F1 | Agreement, or exchange and stay shares | Reading |", "|---|---|---|---|---|---|"]
    for system, label in (("v05", "v0.5's EI specialist"), ("s8_ei", "v0.4's EI specialist, s8_ei")):
        for inp in ("none", "title", "swap"):
            if inp == "swap":
                third = (f"exchange {reading_row(res, table, system, inp, 'exchange_share')}; stay "
                         f"{reading_row(res, table, system, inp, 'stay_share')}")
            else:
                third = reading_row(res, table, system, inp, "agreement_with_full_input")
            lines.append(f"| {label} | {labels[inp]} | {reading_row(res, table, system, inp, 'accuracy')} | "
                         f"{reading_row(res, table, system, inp, 'macro_f1')} | {third} | "
                         f"{res.text(*k4, 'readings', system, inp, 'reading', fmt='str')} |")
    lines += reading_subset(res, table)
    lines += ["", ("The title minus none, macro-F1 on the prompts in both inputs: v0.5's specialist "
                   f"{reading_row(res, table, 'v05', 'title minus none (prompts in both)', 'macro_f1_difference')}; v0.4's "
                   f"{reading_row(res, table, 's8_ei', 'title minus none (prompts in both)', 'macro_f1_difference')}.")]
    caveats = []
    for inp in ("none", "title", "swap"):
        for j, _ in enumerate(res.get(*k4, "readings", "v05", inp, "caveats") or []):
            caveats.append(res.text(*k4, "readings", "v05", inp, "caveats", j, fmt="str"))
    if caveats:
        lines += ["", "Caveats for v0.5's specialist: " + "; ".join(dict.fromkeys(caveats)) + "."]
    refs = []
    if res.has(*k4, "references", "prior", "text"):
        refs.append(res.text(*k4, "references", "prior", "text", fmt="str"))
    if res.has(*k4, "references", "windisch_2026", "text"):
        refs.append(f"{res.text(*k4, 'references', 'windisch_2026', 'text', fmt='str')} "
                    f"({res.text(*k4, 'references', 'windisch_2026', 'source', fmt='str')})")
    if refs:
        lines += ["", "References: " + "; ".join(refs) + "."]
    return "\n".join(lines)


def coverage_keys(res):
    """Where the selection's coverage at the specialist's length is: coverage_long.json's coverage at 6,144 tokens, or
    session 8's at 4,096 that it repeats."""
    max_len = res.get(*RC_EI, "max_len")
    if res.get("coverage", "max_len") == max_len:
        return ("coverage", "coverage", "train")
    if res.has("selection8", "max_len") and res.get("selection8", "max_len") == max_len:
        return ("coverage", "at_4096_session8", "train")
    raise Missing(f"neither {res.label('coverage')} nor {res.label('selection8')} gives the coverage at {max_len} tokens")


def coverage_clean(res):
    """coverage_long.json's coverage on the clean training prompts, the ones the specialist learned from, if it gives it."""
    k = ("coverage", "coverage", "train_clean")
    if res.get(*RC_EI, "max_len") == res.get("coverage", "max_len") and res.has(*k, "coverage"):
        return k
    return None


def selection_part(res):
    b = ("selection8", "bm25")
    cov = coverage_keys(res)
    clean = coverage_clean(res)
    return "\n\n".join([
        (f"The article is cut by v0.4's selection rule at {res.text(*RC_EI, 'max_len', fmt='count')} tokens, the "
         "specialist's input length: the title and abstract first, cut from their end if they alone do not fit; then the "
         "passages, results sections first and then the others, each ranked by Okapi BM25 (a word-matching score; k1 "
         f"{res.text(*b, 'k1', fmt='g')}, b {res.text(*b, 'b', fmt='g')}) against the distinct words of the intervention, the "
         "comparator and the outcome, each passage kept if it still fits; the kept passages are joined again in article order. "
         + (f"At this length the selection kept an annotator-marked evidence span for "
            f"{res.text(*clean, 'coverage', fmt='pct')} percent of the clean training prompts that have one, the prompts the "
            f"specialist learned from ({res.text(*clean, 'covered', fmt='count')} of "
            f"{res.text(*clean, 'prompts_with_usable_offsets', fmt='count')}), and for "
            if clean else "At this length the selection kept an annotator-marked evidence span for ")
         + f"{res.text(*cov, 'coverage', fmt='pct')} percent of {'all ' if clean else ''}the training prompts that have one "
         f"({res.text(*cov, 'covered', fmt='count')} of {res.text(*cov, 'prompts_with_usable_offsets', fmt='count')})."),
        ("The selection's functions ship in evidence_effect/passage_selection/evidence_inference.py with its word statistics, "
         f"bm25_stats.json: for each of {res.text(*b, 'vocabulary', fmt='count')} words, the number of the "
         f"{res.text(*b, 'N', fmt='count')} passages of the {res.text(*b, 'articles', fmt='count')} training articles that "
         f"contain it, with the passages' mean length, {res.text(*b, 'avgL', fmt='f2')} words. These statistics are the one "
         "exception to the rule that nothing the EI specialist uses comes from an Evidence Inference article that is not "
         "clean: they were counted "
         "over every training article, clean or not. They hold single words and counts but no sentence of any article. They are "
         "kept so that every input, the specialist's and the large model's, is cut by the same rule, and they are shipped so "
         "that a new article can be cut the same way.")])


def ei_failures(res):
    cov = coverage_clean(res) or coverage_keys(res)
    covered = res.get(*cov, "coverage")
    which = "the clean training prompts" if cov[-1] == "train_clean" else "the training prompts"
    lines = [(f"- Evidence outside the input: for {res.derived(1 - float(covered), 'pct', [cov + ('coverage',)], 'complement')} "
              f"percent of {which} that have marked evidence (derived), the selection cut it away, and the "
              "specialist answers without it.")]
    k4 = ("comparison", "k4")
    for i, r in enumerate(res.get(*k4, "table") if res.has(*k4, "table") else []):
        if isinstance(r, dict) and r.get("system") == "v05" and r.get("input") == "none" and r.get("statistic") == "macro_f1":
            lines.append(f"- Answers without the article: with no article at all its macro-F1 was {cell(res, *k4, 'table', i)} "
                         "(the reading checks, above, say how far its answers depend on the article's text).")
            break
    k4t = res.get(*k4, "table") if res.has(*k4, "table") else []
    if reading_n(res, k4t, "v05", SUBSET, "exchange_share") is not None:
        lines.append("- Comparators that are a time point or baseline: when the intervention and such a comparator are "
                     "exchanged, the specialist seldom reverses its answer (the reading checks, above); whether its answers to "
                     "such prompts follow the order of the comparison is not known.")
    lines += ["- Only Evidence Inference's kind of prompt was tested: one comparison in a randomized trial's article, in English. "
              "Other questions, wordings and kinds of document are untested and uncalibrated.",
              contamination(res)]
    return "\n".join(lines)


def contamination(res):
    text = res.text("comparison", "contamination", fmt="card")
    return f"- {text}" if text.startswith("Known contamination") else f"- Known contamination: {text}"


def ei_card(card):
    res = card.res
    n = card.part
    return "\n\n".join([
        "## Question card: evidence_effect (Evidence Inference 2.0)",
        "### Intended use and user", EI_USE,
        "### Inputs", n("Evidence Inference inputs", ei_inputs, res),
        "### The specialist", n("Evidence Inference specialist", ei_specialist, res),
        n("the recipe rule", ei_recipe_table, res), n("the calibration families", ei_families, res),
        n("the cost of Evidence Inference answers", ei_cost, res),
        "### Training data and their licenses", n("Evidence Inference training data", ei_training, res),
        "### Validation and test splits, and when they were read", n("Evidence Inference splits", ei_splits, res),
        "### Test results", n("the intervals", intervals_note, res), n("Evidence Inference test metrics", ei_metrics, res),
        n("Evidence Inference calibration slope and intercept", slope_table, res, EI, EI_ANSWERS),
        n("the release bar", bar_part, res),
        "### Hand-off", n("Evidence Inference hand-off", ei_handoff, res),
        "### The reading checks", n("the reading checks", reading_part, res),
        "### The passage selection, and the exception for its word statistics", n("the passage selection", selection_part, res),
        "### Known failure modes and contamination", n("Evidence Inference failure modes", ei_failures, res),
        "### Update plan", EI_UPDATE])


def ei_handoff(res):
    return handoff_part(res, EI, str(res.get(*RC_EI, "system")))


# ---------------------------------------------------------------- the PubMedQA card

PQA_USE = ("For researchers who want a calibrated yes, no or maybe for a biomedical research question from its abstract, as "
           "PubMedQA frames it: whether the abstract's reported results support yes, support no, or are inconclusive. It is for "
           "research only, not for clinical decisions, and it has been tested only on PubMedQA's questions, in English.")
PQA_INPUTS = ("A record whose state is {\"question\", \"contexts\": [{\"section\", \"text\"}, ...]}: the research question and the "
              "abstract's sections in their order, without the conclusion (which states the answer). The input is cut from the "
              "end of the abstract if it does not fit the reader's input length (max_len in s7_b3's config.json). The "
              "question, word for word: \"Based only on "
              "this abstract, what is the answer to the research question?\", with the options yes (the reported results "
              "support yes), no (the reported results support no) and maybe (the reported results are inconclusive).")
PQA_UPDATE = ("s7_b3 is unchanged since v0.3 and no retraining of it is planned. The comparison with "
              "BioGPT-Large-PubMedQA is descriptive and changes nothing in this release.")


def pqa_specialist(res):
    c = ("calibration7", "s7_b3", PQA)
    return (f"s7_b3, unchanged from v0.3: Qwen3-4B-Base with LoRA adapters, a projection and Laya's decision head, trained "
            f"on PubMedQA (and, as one model, on the readmission question). Its calibration is "
            f"{res.text(*RC_PQA, 'family', fmt='str')}, chosen by v0.3's rule (the lower out-of-fold log loss of temperature "
            f"and temperature with intercepts) and fitted on PubMedQA's {res.text(*c, 'n_calibrate', fmt='count')} expert-labeled "
            f"validation items at {res.text(*c, 'time_utc', fmt='time')}. One reader pass a question.")


def pqa_training(res):
    return ("PubMedQA (Jin et al. 2019; MIT License): its automatically labeled items (PQA-A), its unlabeled items (PQA-U), "
            "labeled by this project's own labeler and not by an outside language model, and the training part of its "
            "expert-labeled items (PQA-L). Those abstracts' own licenses were not checked one by one: they rest on PubMedQA's "
            "license (PubMedQA's abstracts, above).")


def pqa_splits(res):
    c = ("calibration7", "s7_b3", PQA)
    return "\n".join([
        (f"- Validation: PubMedQA's {res.text(*c, 'n_calibrate', fmt='count')} expert-labeled validation items, read for s7_b3's "
         f"calibration at {res.text(*c, 'time_utc', fmt='time')}."),
        (f"- Test: PubMedQA's official test list, {res.text('comparison', 'metrics', PQA, 's7_b3', 'n', fmt='count')} items, read "
         "in the evaluation of each version: " + ", ".join(split_reads(res, PQA)) + ". s7_b3's test answers are v0.3's, "
         "reused. The split was also read on 29 September 2026, for a comparison that this card does not report.")])


def pqa_rows(res):
    return [("s7_b3", "**v0.5's PubMedQA specialist (s7_b3)**"),
            ("biogpt_large_pqa-raw", "BioGPT-Large-PubMedQA, uncalibrated (fine-tuned on PubMedQA)"),
            ("gemma4_31b", LARGE_LABELS["gemma4_31b"]), ("prior", "Prior: the training labels' shares")]


def pqa_metrics(res):
    return metrics_table(res, PQA, pqa_rows(res))


PQA_CONTAMINATION = ("- Known contamination: Qwen3's and Gemma 4's pretraining text is not published and may include PubMedQA "
                     "and the PubMed abstracts it is built from, which would favor both. PubMedQA's test split was read in the "
                     "evaluation of every version, and s7_b3 was chosen in v0.3 after earlier versions' test results were "
                     "known, so its comparisons here are descriptive.")
RW_CONTAMINATION = ("- Known contamination: the pool's members are tree models fitted only on the training part, with no "
                    "pretraining, and the UCI data are public. The readmission test split was read in the evaluation of every "
                    "version, and v0.4's pool was chosen after earlier versions' test results were known, so its comparisons "
                    "here are descriptive.")


def pqa_failures(res):
    lines = ["- Only PubMedQA's kind of question was tested, in English; other questions and wordings are untested.",
             (f"- About the BioGPT-Large-PubMedQA row: {res.text('comparison', 'k5', 'caveat', fmt='card')}."
              if res.has("comparison", "k5", "caveat") else ""),
             PQA_CONTAMINATION]
    return "\n".join(x for x in lines if x)


def pqa_card(card):
    res = card.res
    n = card.part
    return "\n\n".join([
        "## Question card: pubmedqa_answer (PubMedQA)",
        "### Intended use and user", PQA_USE,
        "### Inputs", PQA_INPUTS,
        "### The specialist", n("PubMedQA specialist", pqa_specialist, res),
        "### Training data and their licenses", n("PubMedQA training data", pqa_training, res),
        "### Validation and test splits, and when they were read", n("PubMedQA splits", pqa_splits, res),
        "### Test results", n("PubMedQA test metrics", pqa_metrics, res),
        n("PubMedQA calibration slope and intercept", slope_table, res, PQA, PQA_ANSWERS),
        "### Hand-off", n("PubMedQA hand-off", handoff_part, res, PQA, "s7_b3"),
        "### Known failure modes and contamination", n("PubMedQA failure modes", pqa_failures, res),
        "### Update plan", PQA_UPDATE])


# ---------------------------------------------------------------- the readmission card

RW_USE = ("For researchers who study readmission risk on the UCI Diabetes 130-US hospitals data: the probability that a "
          "patient with diabetes is not readmitted after a hospital discharge, readmitted after more than 30 days, or "
          "readmitted within 30 days. The data are from 130 US hospitals between 1999 and 2008, first encounters only, and "
          "care has changed since. It is for research only, not for clinical decisions.")
RW_INPUTS = ("A record whose state holds one encounter's fields: age band, sex, admission type, admission source and "
             "discharge disposition (this project's names for the UCI data's age, gender and three _id columns, with the codes "
             "written out), then, as the UCI data name them, time_in_hospital, num_lab_procedures, num_procedures, num_medications, "
             "number_outpatient, number_emergency, number_inpatient, number_diagnoses, diag_1, diag_2 and diag_3 (an "
             "ICD-9-CM code followed by its chapter's name in parentheses), max_glu_serum, A1Cresult, medications (each "
             "diabetes medication and its change), change, diabetesMed and medical_specialty; a field that is not known is left "
             "out. The question, word for word: \"How soon after this hospital discharge is the patient readmitted?\", with "
             "the levels no readmission after this discharge, readmitted more than 30 days after this discharge, and "
             "readmitted within 30 days of this discharge.")
RW_UPDATE = "No refit of the pool is planned; a change would be a new version, with its own bar."


def rw_specialist(res):
    members = pool_members(res)
    lines = [("The readmission pool combines its members' probabilities: softmax((sum over the members of w log p) / T), each "
              "member's probabilities clipped at a floor before the log, with the weights on a grid of tenths and T fitted on the "
              f"validation split. T is {res.text(*RC_RW, 'T', fmt='f4')}. The members with a weight above zero:"), "",
             "| Member | What it is | Weight |", "|---|---|---|"]
    for m in members:
        lines.append(f"| {m} | {MEMBER_WORDS.get(m, m)} | {res.text(*RC_RW, 'weights', m, fmt='g')} |")
    ch = ("pool", "choice")
    chosen = str(res.get(*ch, "chosen"))
    which = "the new pool" if chosen == "new" else "v0.4's pool"
    lines += ["", (f"v0.5 uses {which}: the new pool's out-of-fold log loss on validation was "
                   f"{res.text(*ch, 'new_out_of_fold_log_loss', fmt='f5')} against "
                   f"{res.text(*ch, 'session8_out_of_fold_log_loss', fmt='f5')} for v0.4's pool, a difference of "
                   f"{res.text(*ch, 'session8_minus_new', fmt='f5')}, and the rule needed at least "
                   f"{res.text(*ch, 'margin', fmt='g')} (the choice made at {res.text(*ch, 'time_utc', fmt='time')}).")]
    pools = ("pool", "pools")
    if res.has(*pools, "new", "members") and res.has(*pools, "session8", "members"):
        old = {str(m) for m in res.get(*pools, "session8", "members")}
        added = [str(m) for m in res.get(*pools, "new", "members") if str(m) not in old]
        if added:
            words = "; ".join(f"{m}, {MEMBER_WORDS.get(m, m)}" for m in added)
            lines[-1] += f" The new pool added to v0.4's members: {words}."
    nb = res.get("comparison", "not_built") if res.has("comparison", "not_built") else []
    if any(isinstance(x, dict) and x.get("system") == "tabicl" for x in nb):
        lines[-1] += (" TabICL, planned as a further member, was not built, because of an error in this project's check of "
                      "its version.")
    if "tabicl" in members:
        lines += ["", (f"TabICLv2's weights are not in this release: predict_v0_5.py downloads its checkpoint, "
                       f"{res.text('fixed', 'tabicl', 'checkpoint', 'file', fmt='str')} from "
                       f"{res.text('fixed', 'tabicl', 'checkpoint', 'repo', fmt='str')} at revision "
                       f"{res.text('fixed', 'tabicl', 'checkpoint', 'revision', fmt='str')}, and checks its SHA-256.")]
    if "arm_laya" in members:
        lines += ["", "v0.1's weights are not copied into this release: predict_v0_5.py reads them from johanneli/fiorillo-v0.1."]
    passes = res.derived(sum(1 for m in members if m in READERS), "word", [RC_RW + ("members",), RC_RW + ("weights",)], "readers")
    lines += ["", f"Reader passes a question: {passes}; the tree members{' and TabICL' if 'tabicl' in members else ''} need none."]
    return "\n".join(lines)


def rw_training(res):
    return ("The UCI Diabetes 130-US hospitals data (CC BY 4.0; Strack et al. 2014): the first encounter of each patient, "
            "encounters that ended in death or hospice left out, split by patient into training, validation and test parts. "
            f"The members were fitted on the training part; the pool's weights and T on the {res.text('pool', 'rows', fmt='count')} "
            "validation encounters.")


def rw_splits(res):
    return "\n".join([
        (f"- Validation: {res.text('pool', 'rows', fmt='count')} encounters, read in each version: v0.4's pool's weights and "
         "T were fitted on them before v0.4's evaluation ("
         f"{res.text('comparison8', 'time_utc' if res.has('comparison8', 'time_utc') else 'created', fmt='time')}), the new "
         "pool's in v0.5, and the choice between the two pools was made at "
         f"{res.text('pool', 'choice', 'time_utc', fmt='time')}."),
        (f"- Test: {res.text('comparison', 'metrics', RW, 'readmission_pool_v2', 'n', fmt='count')} encounters, read in the "
         "evaluation of each version: " + ", ".join(split_reads(res, RW)) + ". The split was also read on 29 September 2026, "
         "for a comparison that this card does not report.")])


def rw_rows(res):
    return [("readmission_pool_v2", "**v0.5's readmission pool**"), ("readmission_pool", "v0.4's readmission pool"),
            ("lightgbm_tuned", "LightGBM, tuned for v0.3"), ("prior", "Prior: the training labels' shares")]


def rw_metrics(res):
    return metrics_table(res, RW, rw_rows(res))


def rw_slope(res):
    s = ("comparison", "calibration_slope_intercept", RW)
    return ("Calibration for readmission within 30 days (the probability of the last level against the rest) on the test split: "
            f"the intercept is {res.interval(*s, 'intercept_30d')} (ideal zero) and the slope "
            f"{res.interval(*s, 'slope_30d')} (ideal one).")


def rw_failures(res):
    return "\n".join([
        "- Old data from one source: 130 US hospitals between 1999 and 2008, first encounters only; care has changed since, "
        "and the pool has not been tested on any other data.",
        "- Readmission within 30 days is rare in these data, so a calibrated model seldom makes it the top answer: use the "
        "probabilities, not the top answer.",
        RW_CONTAMINATION])


def rw_card(card):
    res = card.res
    n = card.part
    return "\n\n".join([
        "## Question card: readmit_window (UCI Diabetes 130-US hospitals)",
        "### Intended use and user", RW_USE,
        "### Inputs", RW_INPUTS,
        "### The specialist", n("the readmission pool", rw_specialist, res),
        "### Training data and their licenses", n("readmission training data", rw_training, res),
        "### Validation and test splits, and when they were read", n("readmission splits", rw_splits, res),
        "### Test results", n("readmission test metrics", rw_metrics, res),
        n("readmission calibration slope and intercept", rw_slope, res),
        "### Hand-off", ("The readmission pool hands nothing off, and no hand-off to a large model was analyzed for this "
                         "question."),
        "### Known failure modes and contamination", n("readmission failure modes", rw_failures, res),
        "### Update plan", RW_UPDATE])


# ---------------------------------------------------------------- how to use, license and citation

EXAMPLES = """```json
{"id": "my-trial-1", "qid": "evidence_effect", "state": {"intervention": "early mobilization", "comparator": "usual care", "outcome": "length of hospital stay", "article": "TITLE: ...\\nABSTRACT ...\\nBODY.METHODS:\\n..."}}
{"id": "my-question-1", "qid": "pubmedqa_answer", "state": {"question": "Does early mobilization shorten hospital stay after hip fracture surgery?", "contexts": [{"section": "BACKGROUND", "text": "..."}, {"section": "RESULTS", "text": "..."}]}}
{"id": "my-encounter-1", "qid": "readmit_window", "state": {"age band": "[70-80)", "sex": "Female", "admission type": "Emergency", "admission source": "Emergency Room", "discharge disposition": "Discharged to home", "time_in_hospital": 4, "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 9, "diag_1": "428 (Diseases of the circulatory system)", "max_glu_serum": "None", "A1Cresult": ">8", "medications": {"insulin": "Up"}, "change": "Ch", "diabetesMed": "Yes"}}
```"""


def usage(res):
    n = ei_names(res)
    members = pool_members(res)
    readers = (f"{n['singles'][0]}/ (the reader's folder" if len(n["singles"]) == 1
               else f"{', '.join(n['singles'])} (one folder for each reader")
    files = [f"- evidence_effect/: calibration.json (the specialist's form and calibration), {readers}: adapter.safetensors, "
             "config.json and the tokenizer files), and passage_selection/ (evidence_inference.py and bm25_stats.json)",
             "- pubmedqa_answer/: calibration.json and s7_b3/ (adapter.safetensors, config.json and the tokenizer files)",
             "- readmit_window/: pool.json (the members, weights and T)"
             + (", recode_tables.json" if any(m not in READERS for m in members) else "")
             + "".join(f", {MEMBER_FILES[m]}" for m in members if m in MEMBER_FILES)
             + (", s7_b3_calibration.json" if "s7_b3" in members else ""),
             "- config.json (a short description of the release's parts, which predict_v0_5.py does not read; the Hugging "
             "Face Hub counts a download each time it is requested), predict_v0_5.py, LICENSE, NOTICE and SHA256SUMS.txt"]
    return "\n\n".join([
        "## How to use",
        ("The release holds the readers' adapters and heads, not Qwen3-4B-Base itself, which predict_v0_5.py loads from "
         f"{res.text('fixed', 'qwen3_4b_base', 'repo', fmt='str')} at the revision in each reader's config.json (or from a local "
         "copy given with --base). Its files:"),
        "\n".join(files),
        ("Install torch, transformers, safetensors, numpy, pandas and laya==0.3.20"
         + (", and for readmission " + ", ".join(dict.fromkeys(MEMBER_LIB[m] for m in members if m in MEMBER_LIB))
            if any(m in MEMBER_LIB for m in members) else "")
         + ". Then write one record a line in a JSON Lines file and run predict_v0_5.py on it from the release folder:"),
        "```bash\npython predict_v0_5.py --records records.jsonl --out answers.jsonl\n```",
        "Example records (the state's fields are the inputs of each question card):",
        EXAMPLES,
        ("Each answer gives every option's calibrated probability, the most probable option and the reader passes it took. For "
         "Evidence Inference, predict_v0_5.py cuts the article with the shipped passage selection before the specialist reads "
         "it" + (" and answers both orders of the comparison (swap averaging)" if n["swap"] else "") + ". For readmit_window "
         "the options are the levels 0 (no readmission), 1 (after more than 30 days) and 2 (within 30 days).")])


MEMBER_LIB = {"xgboost_codes": "xgboost", "xgboost_tuned": "xgboost", "catboost_codes": "catboost", "catboost_tuned": "catboost",
              "lightgbm_tuned": "lightgbm", "tabicl": "tabicl and huggingface_hub", "arm_laya": "huggingface_hub"}


def license_citation(res):
    return "\n\n".join([
        "## License and citation",
        ("Apache-2.0 (LICENSE). The training data keep their own licenses (Licenses, above); NOTICE carries the attributions. The "
         "previous public version is Fiorillo v0.1 (johanneli/fiorillo-v0.1)."),
        (f"Li JE. {NAME} v0.5: calibrated answers to medical typed questions from routed specialists. Hugging Face, "
         f"{res.year()}. https://huggingface.co/{REPOSITORY}")])


# ---------------------------------------------------------------- the card

def front_matter(card):
    res = card.res
    lines = ["---", "license: apache-2.0"]
    try:
        lines.append(f"base_model: {res.text('fixed', 'qwen3_4b_base', 'repo', fmt='str')}")
    except Missing as err:
        card.not_built.append(f"the base_model line of the metadata: {err}")
    lines += ["language:", "- en", "tags:", "- medical", "- typed-decisions", "- calibrated-probabilities",
              "- evidence-inference", "- pubmedqa", "- hospital-readmission", "datasets:", "- qiaojin/PubMedQA", "---"]
    return "\n".join(lines)


def check_bar(res):
    """THE RELEASE runs only if v0.5's bar was met: stop when comparison.json says it was not."""
    if res.has("comparison", "release_bar", "overall") and res.get("comparison", "release_bar", "overall") != "PASS":
        outcome = res.get("comparison", "release_bar", "outcome") if res.has("comparison", "release_bar", "outcome") else ""
        raise SystemExit(f"{res.label('comparison')} says v0.5's bar was not met ({outcome}); THE RELEASE does not run and no "
                         "card is written")


def build(paths, now=None):
    """(the card's text, the sources of its numbers, the parts NOT BUILT)."""
    res = Results(paths, now)
    check_bar(res)
    card = Card(res)
    head = front_matter(card)
    prepared = res.clock()
    sources = "; ".join(rel(p) for name, p in res.paths.items() if res.exists(name))
    body = "\n\n".join([
        (f"{NAME} v0.5 answers three medical questions with calibrated probabilities, each routed to its own specialist: "
         "whether a trial's article reports an outcome significantly higher, significantly lower or not significantly "
         "different with an intervention than with a comparator (Evidence Inference 2.0); the answer to a biomedical research "
         "question from its abstract (PubMedQA: yes, no or maybe); and how soon a patient with diabetes is readmitted after a "
         "hospital discharge (UCI Diabetes 130-US hospitals: no readmission, after more than 30 days, or within 30 days). It is "
         "for research. It is not a medical device and must not be used to make clinical decisions."),
        ("Abbreviations: EI is Evidence Inference 2.0 (DeYoung et al. 2020); ECE is expected calibration error; PMC is PubMed "
         "Central; PMCID is a PubMed Central identifier; LoRA is low-rank adaptation; BM25 is Okapi BM25; UCI is the "
         "University of California, Irvine, Machine Learning Repository; ICD-9-CM is the International Classification of "
         "Diseases, Ninth Revision, Clinical Modification; CC BY is the Creative Commons Attribution license and CC0 the "
         "Creative Commons public-domain dedication; OSF is the Open Science Framework; (derived) marks a number this card "
         "computes from other numbers."),
        what_it_is(card),
        "## Why v0.4's EI specialist is not released", card.part("why v0.4's EI specialist is not released", not_released, res),
        "## Licenses, and the articles credited", card.part("the licenses", licenses_part, res),
        "## PubMedQA's abstracts", card.part("PubMedQA's abstracts", pubmedqa_statement, res),
        "## Prior art", card.part("the release day's prior-art check", prior_art, res),
        ei_card(card), pqa_card(card), rw_card(card),
        card.part("how to use", usage, res),
        license_citation(res)])
    status = ("complete" if not card.not_built else
              f"incomplete: {res.derived(len(card.not_built), 'word', [], 'not built parts')} parts are marked NOT BUILT")
    text = "\n".join([head, "", f"# {NAME} v0.5",
                      f"Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: {prepared}",
                      f"Sources: {sources}; written by scripts/build_model_card_v0_5.py",
                      f"Status: {status}", "", body, ""])
    check_text(text)
    return text, res.sources, card.not_built


def check_text(text):
    """The house style's checks: no em or en dash, and no claim of a first without "to our knowledge"."""
    if chr(0x2014) in text or chr(0x2013) in text:
        raise SystemExit("the card has an em or en dash")
    lower = text.lower()
    for sentence in re.split(r"(?<=[.!?])\s+", lower):
        if PRIORITY.search(sentence) and "to our knowledge" not in sentence:
            raise SystemExit(f"a claim of a first without \"to our knowledge\": {sentence[:120]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name, option, default in INPUTS + OUTPUTS:
        parser.add_argument(option, dest=name, type=Path, default=ROOT / default)
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name, _, _ in INPUTS}
    text, sources, not_built = build(paths)
    for name in ("readme", "doc"):
        out = getattr(args, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(json.dumps({"readme": rel(args.readme), "doc": rel(args.doc), "words": len(text.split()), "numbers": len(sources),
                      "not_built": not_built}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

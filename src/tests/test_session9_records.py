"""Tests for session 9's records (docs/prompt_session9.md, step 4), on made-up inputs only.

Checked, with made-up license statements, records and article texts in temporary folders and a stand-in tokenizer that
counts words: src/tools/ei_license_recheck.py's classes and its clean column (DECISIONS 3); and src/tasks/ei_session9.py's
clean records (by PMCID from license_recheck.csv, byte for byte, order kept), the clean monitoring subset (seeded, IDs
sorted as text), the swap records (THE RECIPES), the selection at 6,144 tokens against a hand calculation, with session
8's 4,096-token selection and records unchanged through the same code (THE LONGER INPUT), the reading checks' inputs
(none, the title by correction 6's rule, swap, and correction 3's exchanged validation records) in the folders that
src/train/train_qwen_head.py predict reads, the combination's records, a rebuild with no tokenizer giving byte-identical
files, and a stopped selection resuming to the same files. No real record, label or license statement is read; the only
project files read are the code and the question bank, src/questions/session8.yaml."""
import copy
import csv
import hashlib
import json
import random
import re
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC / "tasks"))
sys.path.insert(0, str(SRC / "train"))
sys.path.insert(0, str(SRC / "tools"))

import ei_license_recheck as LR  # noqa: E402
import ei_session9 as S9  # noqa: E402
import evidence_inference as EI  # noqa: E402

RECS9_CLEAN = {"cc by", "cc by (words)", "cc0", "public domain"}
RECS9_SWAP = {0: 1, 1: 0, 2: 2}


def recs9_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recs9_onehot(label):
    return [1.0 if i == label else 0.0 for i in range(3)]


def recs9_records(path):
    return [json.loads(line) for line in S9.read_lines(path)]


# ---------------------------------------------------------------- the license recheck (DECISIONS 3)

RECS9_BY = "https://creativecommons.org/licenses/by/4.0/"
RECS9_RESTRICTING = "open link or statement, restricting words in its text"


def recs9_lic(href=None, text="", kind="open-access", refs=()):
    """One <license> as fetch stores it: its license-type, its link, its ali:license_ref texts and its text."""
    return {"type": kind, "href": href, "refs": list(refs), "text": text}


def recs9_perm(pmcid, *licenses, copyright=""):
    """One line of permissions.jsonl as fetch writes it: parse's fields, then the PMCID and the time it was read."""
    return {"copyright": copyright, "licenses": list(licenses), "no_permissions_element": False,
            "journal": "Made-up Journal", "year": "2012", "pmcid": pmcid, "read_utc": "2026-10-01T06:30:00Z"}


RECS9_STATEMENTS = [  # (PMCID, the class expected, the statement)
    ("PMC101", "cc by", recs9_perm("PMC101", recs9_lic(  # a CC BY link with a CC0 waiver for the data only
        "http://creativecommons.org/licenses/by/2.0", "This is an Open Access article distributed under the terms of the "
        "Creative Commons Attribution License (http://creativecommons.org/licenses/by/2.0), which permits unrestricted "
        "use, distribution, and reproduction in any medium, provided the original work is properly cited. The Creative "
        "Commons Public Domain Dedication waiver (http://creativecommons.org/publicdomain/zero/1.0/) applies to the data "
        "made available in this article, unless otherwise stated."))),
    ("PMC102", RECS9_RESTRICTING, recs9_perm("PMC102", recs9_lic(  # a CC BY link beside non-commercial words
        RECS9_BY, "This article is available under the Creative Commons CC-BY license and permits non-commercial use of "
        "the work as published, without adaptation or alteration."))),
    ("PMC103", RECS9_RESTRICTING, recs9_perm("PMC103", recs9_lic(  # U+2010, a Unicode hyphen
        RECS9_BY, "Reuse is permitted for non\u2010commercial purposes only."))),
    ("PMC104", RECS9_RESTRICTING, recs9_perm("PMC104", recs9_lic(  # U+2011, a non-breaking hyphen
        RECS9_BY, "Reuse is permitted for non\u2011commercial purposes only."))),
    ("PMC105", RECS9_RESTRICTING, recs9_perm("PMC105", recs9_lic(
        RECS9_BY, "This is an open access article under the CC BY license, which does not permit commercial "
        "exploitation."))),
    ("PMC106", "conflicting license links", recs9_perm("PMC106", recs9_lic(RECS9_BY, "Open access."), recs9_lic(
        "https://creativecommons.org/licenses/by-nc/4.0/", "Open access."))),
    ("PMC107", RECS9_RESTRICTING, recs9_perm("PMC107", recs9_lic(  # a restricting license-type beside a CC BY link
        "http://creativecommons.org/licenses/by/3.0/", "This is an open access article.", kind="cc-by-nc"))),
    ("PMC108", "cc by (words)", recs9_perm("PMC108", recs9_lic(
        None, "This is an open-access article distributed under the terms of the Creative Commons Attribution License, "
        "which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and "
        "source are credited."))),
    ("PMC109", "cc by-nc, by-nd or by-sa (words)", recs9_perm("PMC109", recs9_lic(
        None, "This is an open-access article distributed under the terms of the Creative Commons "
        "Attribution-NonCommercial License, which permits use provided the original work is properly cited."))),
    ("PMC110", "publisher's own reuse terms", recs9_perm("PMC110", recs9_lic(
        None, "This article is distributed under the terms of iOpenAccess and the Creative Commons Attribution "
        "License."), copyright="Copyright 2011 Informa UK Ltd.")),
    ("PMC111", "other non-commercial or no-modification terms", recs9_perm("PMC111", recs9_lic(
        None, "Copies may be made provided that they are not for commercial gain.", kind=None),
        copyright="Copyright 2003 Smith.")),
    ("PMC112", "public domain", recs9_perm(
        "PMC112", copyright="This article is a US Government work and is in the public domain in the USA.")),
    ("PMC113", "general statement only", recs9_perm(  # the general rule alone, nothing about this article
        "PMC113", copyright="Contributions of employees of the federal government are not protected by copyright in the "
        "United States.")),
    ("PMC114", "verbatim copying only", recs9_perm(
        "PMC114", copyright="Verbatim copying and redistribution of this article are permitted in all media for any "
        "purpose, provided this notice is preserved along with the article's original URL.")),
    ("PMC115", "copyright notice only", recs9_perm(
        "PMC115", copyright="Copyright 2004 Smith et al.; licensee BioMed Central Ltd.")),
    ("PMC116", "no statement", {"missing": True, "pmcid": "PMC116", "read_utc": "2026-10-01T06:30:00Z"}),  # not returned
    ("PMC117", "no statement", recs9_perm("PMC117", recs9_lic(None, "", kind=None))),
    ("PMC118", "cc0", recs9_perm("PMC118", recs9_lic(  # the article itself dedicated by CC0
        "https://creativecommons.org/publicdomain/zero/1.0/", "This is an open access article, free of all copyright, and "
        "may be freely reproduced, distributed, transmitted, modified, built upon, or otherwise used by anyone for any "
        "lawful purpose. The work is made available under the Creative Commons CC0 public domain dedication."))),
    ("PMC119", "cc by-sa", recs9_perm(  # ShareAlike: left out by DECISIONS 3
        "PMC119", recs9_lic("https://creativecommons.org/licenses/by-sa/4.0/", "Open access."))),
    ("PMC120", "cc by", recs9_perm("PMC120", recs9_lic(  # the link in ali:license_ref, beside a permissive phrase
        None, "This article is distributed under the Creative Commons Attribution License. It may be reused for any "
        "purpose, even commercially.", kind=None, refs=[RECS9_BY]))),
    ("PMC121", RECS9_RESTRICTING, recs9_perm("PMC121", recs9_lic(  # a license-type that only the license-type check reads
        RECS9_BY, "This is an open access article.", kind="ccbyncnd"))),
    ("PMC122", "copyright notice only", recs9_perm("PMC122", recs9_lic(  # a CC0 waiver for the data only, no license
        None, "The Creative Commons Public Domain Dedication waiver applies to the data made available in this article "
        "(http://creativecommons.org/publicdomain/zero/1.0/).", kind=None), copyright="Copyright 2009 Smith.")),
]


def test_license_classes_and_the_clean_column(tmp_path, monkeypatch):
    assert LR.CLEAN == ("cc by", "cc by (words)", "cc0", "public domain")
    assert {want for _, want, _ in RECS9_STATEMENTS} <= set(LR.CLASSES)
    for pmcid, want, rec in RECS9_STATEMENTS:
        assert LR.classify_one(rec) == want, pmcid
    # classify reads fetch's permissions.jsonl, Europe PMC's CSV and session 8's records (their article IDs only)
    raw, records, out = tmp_path / "license_recheck", tmp_path / "session8", tmp_path / "out"
    raw.mkdir()
    records.mkdir()
    (raw / "permissions.jsonl").write_text("".join(json.dumps(r) + "\n" for _, _, r in RECS9_STATEMENTS), encoding="utf-8")
    with open(tmp_path / "article_licenses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmcid", "license", "read"])
        w.writerows([p, "cc by", "2026-09-30"] for p, _, _ in RECS9_STATEMENTS)
    for k, split in enumerate(EI.SPLITS):  # group.pmcid with and without the PMC prefix
        recs = [{"id": f"evidence-{i}", "group": {"pmcid": p if i % 2 else p[3:]}} for i, (p, _, _) in enumerate(RECS9_STATEMENTS[k::3])]
        (records / f"evidence_effect_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    for name, value in (("RAW_DIR", raw), ("LICENSES", tmp_path / "article_licenses.csv"), ("RECORDS", records), ("OUT_DIR", out)):
        monkeypatch.setattr(LR, name, value)
    assert LR.classify() == 0
    with open(out / "license_recheck.csv", newline="", encoding="utf-8") as f:
        rows = {r["pmcid"]: r for r in csv.DictReader(f)}
    for pmcid, want, _ in RECS9_STATEMENTS:
        assert (rows[pmcid]["own_statement"], rows[pmcid]["clean"]) == (want, "true" if want in RECS9_CLEAN else "false"), pmcid
        assert rows[pmcid]["prompts"] == "1" and len(rows[pmcid]["splits"].split()) == 1
    assert sorted(p for p, r in rows.items() if r["clean"] == "true") == ["PMC101", "PMC108", "PMC112", "PMC118", "PMC120"]
    summary = json.loads((out / "license_recheck.json").read_text(encoding="utf-8"))
    assert summary["clean_classes"] == list(LR.CLEAN) and summary["articles"] == len(RECS9_STATEMENTS)
    assert sum(summary["by_split"][s]["clean_prompts"] for s in EI.SPLITS) == 5


def test_fetch_parse_gives_the_shape_the_classifier_reads():
    xml = b"""<?xml version="1.0"?><pmc-articleset>
<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><journal-meta><journal-title-group><journal-title>Made-up
Journal</journal-title></journal-title-group></journal-meta><article-meta><article-id pub-id-type="pmc">PMC201</article-id>
<pub-date><year>2015</year></pub-date><permissions><copyright-statement>Copyright 2015 Smith.</copyright-statement>
<license license-type="open-access" xlink:href="http://creativecommons.org/licenses/by/4.0/"><license-p>This is an open
access article under the Creative Commons Attribution License.</license-p></license></permissions></article-meta></front>
</article><article><front><article-meta><article-id pub-id-type="pmcid">202</article-id>
<copyright-statement>Copyright 2001 Jones.</copyright-statement><license><license-p>Verbatim copying and redistribution of
this article are permitted in all media for any purpose.</license-p></license></article-meta></front></article>
<article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"><front><article-meta><article-id pub-id-type="pmc">203</article-id>
<permissions><license><ali:license_ref>https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref><license-p>Open
access.</license-p></license></permissions></article-meta></front></article></pmc-articleset>"""
    got = LR.parse(xml)
    assert sorted(got) == ["PMC201", "PMC202", "PMC203"]
    assert set(recs9_perm("x")) == set(got["PMC201"]) | {"pmcid", "read_utc"}  # the made-up statements' shape is fetch's
    assert all(set(lic) == set(recs9_lic()) for r in got.values() for lic in r["licenses"])
    assert got["PMC201"]["licenses"][0]["href"] == "http://creativecommons.org/licenses/by/4.0/" and got["PMC201"]["year"] == "2015"
    assert got["PMC202"]["no_permissions_element"] and got["PMC202"]["copyright"] == "Copyright 2001 Jones."  # an older record
    assert got["PMC203"]["licenses"][0]["refs"] == ["https://creativecommons.org/licenses/by-nc/4.0/"]
    assert [LR.classify_one(got[p]) for p in sorted(got)] == ["cc by", "verbatim copying only", "cc by-nc"]


# ---------------------------------------------------------------- clean records, the monitoring subset, swap

def recs9_licenses(path, clean):
    """license_recheck.csv with the tool's columns; clean maps each article's digits to its clean column."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmcid", "splits", "europe_pmc", "own_statement", "clean", "year", "prompts"])
        w.writerows([f"PMC{p}", "train", "cc by", "cc by" if c else "copyright notice only", "true" if c else "false", "2012", 1]
                    for p, c in clean.items())
    return Path(path)


def test_clean_filter_by_pmcid_byte_for_byte_in_order(tmp_path):
    path = recs9_licenses(tmp_path / "license_recheck.csv", {"1": True, "2": False, "30": True})
    assert S9.clean_pmcids(path) == {"PMC1", "PMC30"}
    # written by hand, with spacing and key order that json.dumps would not give, and a character outside ASCII
    lines = ['{"id": "evidence-%s",  "state": {"article": "caf\u00e9"}, "group": {"pmcid": "%s"}, "label": 2}\n' % (i, p)
             for i, p in (("9", "PMC30"), ("3", "2"), ("4", "1"), ("1", "PMC1"), ("2", "3"), ("5", "30"), ("6", "PMC2"))]
    got = S9.filter_clean(lines, S9.clean_pmcids(path))
    assert got == [lines[0], lines[2], lines[3], lines[5]]  # "1" and "30" without the prefix match; "3" is not "PMC30"


def test_clean_monitoring_subset_is_seeded_over_ids_sorted_as_text():
    assert (S9.MONITOR, S9.SEED) == (200, 20260927)
    ids = ["evidence-5", "evidence-12", "evidence-100", "evidence-7", "evidence-33", "evidence-1000"]
    lines = ['{"id": "%s", "group": {"pmcid": "PMC1"}}\n' % i for i in ids]
    got = S9.monitor_lines(lines, n=3)
    want = sorted(random.Random(20260927).sample(sorted(ids), 3))  # as text: evidence-100 before evidence-12
    assert [json.loads(x)["id"] for x in got] == want and set(got) <= set(lines)
    assert S9.monitor_lines(list(reversed(lines)), n=3) == got  # the order of the input does not matter


def test_swap_records():
    base = {"id": "evidence-4", "qid": "evidence_effect",
            "state": {"intervention": "drug C", "comparator": "placebo", "outcome": "fever", "article": "caf\u00e9 text"},
            "label": 0, "target": [1.0, 0.0, 0.0], "calibrate": True,
            "group": {"source": "evidence_inference_v2", "pmcid": "13", "flag": "none", "accepted_labels": [0, 2]}}
    kept = copy.deepcopy(base)
    got = S9.swapped(base)
    assert base == kept  # the record itself is not changed
    assert got["id"] == "evidence-4-swap"
    assert got["state"] == dict(base["state"], intervention="placebo", comparator="drug C")
    assert (got["label"], got["target"], got["group"]["accepted_labels"]) == (1, [0.0, 1.0, 0.0], [1, 2])
    assert set(got) == set(base) and (got["qid"], got["calibrate"]) == (base["qid"], base["calibrate"])  # nothing else changes
    assert all(got["group"][k] == base["group"][k] for k in ("source", "pmcid", "flag"))
    for label, want in ((0, 1), (1, 0), (2, 2)):  # increased and decreased exchanged, no significant difference kept
        rec = dict(base, label=label, target=recs9_onehot(label), group=dict(base["group"], accepted_labels=[label]))
        s = S9.swapped(rec)
        assert (s["label"], s["target"], s["group"]["accepted_labels"]) == (want, recs9_onehot(want), [want])
        assert S9.swapped(S9.swapped(rec, suffix=""), suffix="") == rec  # exchanging twice gives the record back
    for a, b in (("Drug  C", " drug c "), ("PLACEBO\tpill", "placebo\n pill")):  # the same text after whitespace and case
        assert S9.swapped(dict(base, state=dict(base["state"], intervention=a, comparator=b))) is None
    assert S9.swapped(dict(base, state=dict(base["state"], intervention="drug C", comparator="drug C2"))) is not None
    same = dict(base, id="evidence-5", state=dict(base["state"], comparator="DRUG c"))
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in (base, same)]
    train, n_same = S9.swap_training_lines(lines)
    assert n_same == 1 and train == [lines[0], json.dumps(got, ensure_ascii=False) + "\n", lines[1]]
    checks, n_same = S9.swap_check_lines(lines)
    assert n_same == 1 and checks == [json.dumps(dict(got, id="evidence-4"), ensure_ascii=False) + "\n"]  # the ID kept


# ---------------------------------------------------------------- a made-up copy of EI's files and a stand-in tokenizer

RECS9_TOKEN = re.compile(r"\\n|\n|[A-Za-z0-9]+")


def recs9_tokenize(text, add_special_tokens=False, truncation=False, max_length=None):
    """A stand-in for Qwen's tokenizer that counts words: one token for each run of letters and digits and for each
    newline, written or escaped as JSON escapes it (a backslash and n). Its counts add up over joined texts, so a
    selection can be worked out by hand."""
    def ids(t):
        got = [1] * len(RECS9_TOKEN.findall(t))
        return got[:max_length] if truncation and max_length else got
    return {"input_ids": [ids(t) for t in text] if isinstance(text, list) else ids(text)}


@pytest.fixture
def recs9_tok(monkeypatch):
    """transformers' AutoTokenizer.from_pretrained patched to give the stand-in; no tokenizer file is read."""
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: recs9_tokenize)


def recs9_words(n):
    return " ".join(["word"] * n)


# The long article (PMC11 in training, PMC21 in validation and PMC31 in test). With the stand-in, its title and abstract
# are 15 tokens (13 words and 2 newlines), and its passages are P0, a results section without the prompt's words (2,500
# tokens with its header line); P1, results, with them (2,000); P2, methods, without them (900); and P3, methods, with
# them (1,500).
RECS9_PREFIX = "TITLE: Drug A for pain\nABSTRACT.BACKGROUND:\nDrug A was compared with placebo."
RECS9_P = ["BODY.RESULTS:\n" + recs9_words(2497), "Pain fell with drug A " + recs9_words(1995),
           "BODY.METHODS:\n" + recs9_words(897), "Pain eased with drug A " + recs9_words(1495)]
RECS9_LONG = RECS9_PREFIX + "\n\n" + "\n\n".join(RECS9_P) + "\n"
RECS9_ARTICLES = {
    "11": RECS9_LONG,
    "12": "TITLE: Drug B for sleep\nABSTRACT.BACKGROUND:\nDrug B and sleep.\n\nBODY.RESULTS:\nSleep rose with drug B.\n",
    "13": "TITLE: Drug C\nABSTRACT.BACKGROUND:\nDrug C, nausea and fever.\n\nBODY.RESULTS:\nFever fell with drug C.\n",
    "21": RECS9_LONG,
    "22": "TITLE: Drug E for rash\nABSTRACT.BACKGROUND:\nDrug E and rash.\n\nBODY.RESULTS:\nRash rose with drug E.\n",
    "23": "TITLE: Drug F for itch\nABSTRACT.BACKGROUND:\nDrug F and itch.\n\nBODY.FINDINGS:\nItch fell with drug F.\n",
    "31": RECS9_LONG,
    "32": "TITLE: \nDrug H for mood\n\nMood rose with drug H.\n\nBODY.RESULTS:\nMood rose.\n",  # no ABSTRACT line
    "33": "ABSTRACT.BACKGROUND:\nDrug J and sleep.\n\nBODY.RESULTS:\nSleep was unchanged.\n",  # no title
}
RECS9_SPLITS = {"train": ["11", "12", "13"], "validation": ["21", "22", "23"], "test": ["31", "32", "33"]}
RECS9_PROMPTS = [  # PromptID, PMCID, outcome, intervention, comparator, Label Codes (1 increased, -1 decreased, 0 neither)
    ("1", "11", "pain", "drug A", "placebo", [-1]), ("2", "12", "sleep", "drug B", "usual care", [1]),
    ("3", "13", "nausea", "Drug C", " drug  c ", [0]),  # the same text after whitespace and case
    ("4", "13", "fever", "drug C", "placebo", [1, 1, 0]),  # increased; accepted answers 0 and 2
    ("5", "21", "cough", "drug D", "placebo", [1]), ("6", "22", "rash", "drug E", "placebo", [-1]),
    ("7", "23", "itch", "drug F", "DRUG F", [0]), ("8", "23", "itch", "drug F", "vehicle", [-1]),
    ("9", "31", "gout", "drug G", "placebo", [1]), ("10", "32", "mood", "drug H", "placebo", [-1]),
    ("11", "33", "sleep", "drug J", "drug J", [0]), ("12", "31", "flares", "drug G", "colchicine", [-1]),
]
RECS9_CLEAN_ARTICLES = {"11": True, "12": False, "13": True, "21": True, "22": False, "23": True, "31": True, "32": False,
                        "33": True}


def recs9_raw(folder):
    """EI's files at the fixed commit, made up: the split lists, prompts, annotations, the README's caveat lists, texts."""
    a = Path(folder) / "annotations"
    (a / "splits").mkdir(parents=True)
    (a / "txt_files").mkdir()
    for s, ids in RECS9_SPLITS.items():
        (a / "splits" / f"{s}_article_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    start = RECS9_LONG.index(RECS9_P[0]) + len("BODY.RESULTS:\n")  # prompt 1's evidence: P0's first word
    rows = []
    with open(a / "prompts_merged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PromptID", "PMCID", "Outcome", "Intervention", "Comparator"])
        for pid, pmcid, outcome, intervention, comparator, codes in RECS9_PROMPTS:
            w.writerow([pid, pmcid, outcome, intervention, comparator])
            span = (start, start + 4) if pid == "1" else (-1, -1)
            rows += [{"PromptID": pid, "PMCID": pmcid, "Valid Label": "True", "Valid Reasoning": "True", "Annotations": "x",
                      "Label Code": str(c), "Evidence Start": str(span[0]), "Evidence End": str(span[1])} for c in codes]
    with open(a / "annotations_merged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a / "README.md").write_text("### Incorrect: \n900\n\n### Questionable: \n10\n\n### Somewhat malformed: \n12\n",
                                 encoding="utf-8")
    for pmcid, text in RECS9_ARTICLES.items():
        (a / "txt_files" / f"PMC{pmcid}.txt").write_text(text, encoding="utf-8")
    files = {rel: {"sha256": "made up", "bytes": 1} for rel in EI.FILES}
    (Path(folder) / "download_progress.json").write_text(json.dumps({"commit": EI.COMMIT, "files": files}), encoding="utf-8")
    return Path(folder)


def recs9_world(tmp_path, monkeypatch):
    """The made-up raw folder; session 8's records built from it by session 8's builder with the stand-in tokenizer (a
    monitoring subset of 2); a made-up license_recheck.csv; and session 9's module pointed at these folders, with
    results/fiorillo_v0_4 beside results/fiorillo_v0_5 as rebuild expects. Needs the recs9_tok fixture."""
    w = types.SimpleNamespace(raw=recs9_raw(tmp_path / "raw"), s8=tmp_path / "s8", tok=tmp_path / "no_tokenizer_here",
                              r8=tmp_path / "results" / "fiorillo_v0_4", r9=tmp_path / "results" / "fiorillo_v0_5",
                              out=tmp_path / "s9")
    w.licenses = recs9_licenses(w.r9 / "license_recheck.csv", RECS9_CLEAN_ARTICLES)
    monkeypatch.setattr(EI, "MONITOR", 2)
    assert EI.build(w.raw, w.s8, w.r8, w.tok, 1e9, s7_b3=tmp_path / "no_adapter")["finished"]
    for name, value in (("ROOT", tmp_path), ("OUT", w.out), ("RESULTS", w.r9), ("S8_DIR", w.s8), ("LICENSES", w.licenses),
                        ("S8_MANIFEST", w.r8 / "records_manifest.json"), ("EXPECTED", {"train": (3, 2), "validation": (3, 2)})):
        monkeypatch.setattr(S9, name, value)
    monkeypatch.setattr(EI, "RAW", w.raw)
    # build_long calls select_long without s8_dir, and build_clean calls monitor_lines without n; both defaults were bound
    # when the module was imported (the real session 8 folder, and 200), so the made-up folder and 2 are given here.
    real_select, real_monitor = S9.select_long, S9.monitor_lines
    monkeypatch.setattr(S9, "select_long", lambda *a, **k: real_select(*a, **{"s8_dir": w.s8, **k}))
    monkeypatch.setattr(S9, "monitor_lines", lambda lines, n=2, seed=S9.SEED: real_monitor(lines, n, seed))
    return w


def recs9_build_all(w):
    """clean, swap, long, checks and manifest, in the linked shell's order, through the module's functions."""
    S9.build_clean(w.s8, w.out, w.licenses, w.r8 / "records_manifest.json")
    S9.build_swap(w.out)
    assert S9.build_long(w.raw, w.out, w.r9, w.tok, 1e9, w.licenses)["finished"]
    S9.build_checks(w.raw, w.out, w.s8, w.r8 / "records_manifest.json")
    return S9.build_manifest(w.out, w.r9, w.licenses, w.s8)


# ---------------------------------------------------------------- the longer input

def test_selection_against_a_hand_calculation():
    # The tail as src/train/qwen_items.py renders it: the blank line (2), "choice question: <instructions>" (19 words), a
    # newline (1), then each option's words and its newline (14 + 1, 14 + 1, 18 + 1): 71. The limit is the length less 16
    # to spare and the tail. The state with an empty article is 8 words; the title and abstract 15 tokens; a passage costs
    # its tokens and 2 for the blank line before it. The order: results sections first, then by BM25: P1, P0, P3, P2.
    # At 4,096 (limit 4,009): 8 + 15 = 23; P1 makes 2,025; P0 would make 4,527 (left out); P3 makes 3,527; P2 would make
    # 4,429 (left out). Kept P1 and P3, rejoined in article order: 3,527 + 71 = 3,598 tokens.
    # At 6,144 (limit 6,057): P1 makes 2,025; P0 4,527; P3 6,029; P2 would make 6,931 (left out): 6,029 + 71 = 6,100.
    assert S9.LONG_LEN == 6144
    prompt = {"pid": "1", "intervention": "drug A", "comparator": "placebo", "outcome": "pain", "spans": []}
    prefix, passages = EI.article_parts(RECS9_LONG)
    assert (prefix, [p[2] for p in passages], [p[3] for p in passages]) == (RECS9_PREFIX, RECS9_P, [True, True, False, False])
    pieces = [len(x) for x in recs9_tokenize([json.dumps(p[2])[1:-1] for p in passages])["input_ids"]]
    assert pieces == [2500, 2000, 900, 1500]
    stats = {"N": 40, "avgL": 300.0, "df": {"drug": 5, "pain": 3, "placebo": 8}}
    assert EI.order_passages(prompt, passages, stats) == [1, 0, 3, 2]
    for max_len, kept, total in ((4096, [1, 3], 3598), (S9.LONG_LEN, [0, 1, 3], 6100)):
        budget = EI.Budget(recs9_tokenize, max_len=max_len)
        assert len(budget.fmt.tail) == 2 + 19 + 1 + (14 + 1) + (14 + 1) + (18 + 1) and budget.limit == max_len - 16 - 71
        assert (budget.state_tokens(prompt, ""), budget.count_escaped(prefix), budget.sep) == (8, 15, 2)
        assert EI.select_one(prompt, prefix, passages, pieces, 15, stats, budget) == (len(prefix), kept, total)
    # Room for 5 tokens of article: the title and abstract cut from their end, before the first newline (the sixth
    # token), with no passage. With a limit of 7, below the 8 tokens of the state with an empty article, even the prompt
    # alone does not fit.
    small = EI.Budget(recs9_tokenize, max_len=16 + 71 + 8 + 5)
    assert EI.select_one(prompt, prefix, passages, pieces, 15, stats, small) == (len("TITLE: Drug A for pain"), [], 84)
    assert EI.select_one(prompt, prefix, passages, pieces, 15, stats, EI.Budget(recs9_tokenize, max_len=16 + 71 + 7)) is None


def test_long_selection_and_session8_selection_through_the_same_code(tmp_path, monkeypatch, recs9_tok):
    w = recs9_world(tmp_path, monkeypatch)
    s8_rows = EI.read_selection(w.s8 / "parts")  # session 8's builder, at 4,096 tokens
    assert s8_rows["evidence-1"] == {"id": "evidence-1", "pmcid": "11", "split": "train", "prefix_chars": len(RECS9_PREFIX),
                                     "passages": [1, 3], "tokens": 3598, "covered": False}
    assert S9.select_long(w.raw, w.out, w.tok, 1e9, s8_dir=w.s8)
    rows = S9.read_long_parts(w.out)
    assert rows["evidence-1"] == dict(s8_rows["evidence-1"], passages=[0, 1, 3], tokens=6100, covered=True)  # P0 holds the evidence
    others = {"evidence-5": ("21", "validation"), "evidence-9": ("31", "test"), "evidence-12": ("31", "test")}  # also 8 + 15
    for i, (pmcid, split) in others.items():
        assert rows[i] == dict(rows["evidence-1"], id=i, pmcid=pmcid, split=split, covered=None)
        assert s8_rows[i] == dict(s8_rows["evidence-1"], id=i, pmcid=pmcid, split=split, covered=None)
    assert sorted(rows) == sorted(s8_rows) and all(rows[i] == s8_rows[i] for i in rows if i not in others and i != "evidence-1")
    kept, _, _ = EI.select_prompts(w.raw)
    recs, skipped = EI.records_from(w.raw, kept, rows)
    assert skipped == {} and recs["train"][0]["state"]["article"] == RECS9_PREFIX + "".join("\n\n" + RECS9_P[i] for i in (0, 1, 3))
    # the same code at 4,096 tokens gives session 8's selection, its file and its records, byte for byte
    monkeypatch.setattr(S9, "LONG_LEN", 4096)
    assert S9.select_long(w.raw, tmp_path / "at4096", w.tok, 1e9, s8_dir=w.s8)
    for split in EI.SPLITS:
        assert ((tmp_path / "at4096" / "parts" / f"selection_long_{split}.jsonl").read_bytes()
                == (w.s8 / "parts" / f"selection_{split}.jsonl").read_bytes())
    rows4 = S9.read_long_parts(tmp_path / "at4096")
    assert "".join(S9.selection_lines(kept, rows4)) == (w.s8 / "article_selection.jsonl").read_text(encoding="utf-8")
    recs4, _ = EI.records_from(w.raw, kept, rows4)
    for split in EI.SPLITS:
        assert [S9.dump(r) for r in recs4[split]] == S9.read_lines(w.s8 / f"evidence_effect_{split}.jsonl")
    # a prompt that does not fit alone is a skip row, and no record
    monkeypatch.setattr(S9, "LONG_LEN", 16 + 71 + 7)
    assert S9.select_long(w.raw, tmp_path / "tiny", w.tok, 1e9, s8_dir=w.s8)
    tiny = S9.read_long_parts(tmp_path / "tiny")
    assert {r.get("skip") for r in tiny.values()} == {"the prompt alone does not fit"}
    assert EI.records_from(w.raw, kept, tiny) == ({s: [] for s in EI.SPLITS},
                                                  {f"{s}: the prompt alone does not fit": 4 for s in EI.SPLITS})


def test_stopped_selection_resumes_to_the_same_files(tmp_path, monkeypatch, recs9_tok):
    w = recs9_world(tmp_path, monkeypatch)
    whole, parted = tmp_path / "whole", tmp_path / "parted"
    for out in (whole, parted):
        S9.build_clean(w.s8, out, w.licenses, w.r8 / "records_manifest.json")  # build_long reads the monitoring subset
    assert S9.build_long(w.raw, whole, tmp_path / "r_whole", w.tok, 1e9, w.licenses)["finished"]
    stops = 0
    while not S9.build_long(w.raw, parted, tmp_path / "r_parted", w.tok, -1.0, w.licenses)["finished"]:  # select_long, -1.0
        stops += 1
        assert stops < 20
    assert stops == 8  # nine articles, one a run: each run stops before its second article
    names = sorted(str(p.relative_to(whole)) for p in whole.rglob("*") if p.is_file())
    assert names == sorted(str(p.relative_to(parted)) for p in parted.rglob("*") if p.is_file())
    assert {"parts/selection_long_test.jsonl", "long/article_selection_long.jsonl", "long/evidence_effect_test.jsonl"} <= set(names)
    for name in names:
        assert (parted / name).read_bytes() == (whole / name).read_bytes(), name
    cov = [json.loads((r / "coverage_long.json").read_text(encoding="utf-8")) for r in (tmp_path / "r_whole", tmp_path / "r_parted")]
    assert dict(cov[0], written_utc=None) == dict(cov[1], written_utc=None)
    assert cov[0]["coverage"]["train"]["coverage"] == 1.0 and cov[0]["coverage"]["train_clean"]["prompts"] == 3


# ---------------------------------------------------------------- the reading checks' inputs

def test_title_rule_of_correction_6():
    cases = [
        ("TITLE: Drug G for gout\nABSTRACT.BACKGROUND:\nGout fell.\n", "Drug G for gout"),  # on the label's line
        ("TITLE: \nDrug H for mood\nABSTRACT.BACKGROUND:\nMood rose.\n", "Drug H for mood"),  # "TITLE: " alone, then the title
        ("TITLE: Drug K   for pain\n  ABSTRACT.OBJECTIVE:\nPain fell.\n", "Drug K for pain"),  # an indented ABSTRACT line
        ("TITLE: Drug L for acne\n\nAcne fell with drug L.\n\nBODY.RESULTS:\nAcne fell.\n", "Drug L for acne"),  # a blank line
        ("ABSTRACT.BACKGROUND:\nNo title here.\n\nBODY.RESULTS:\nText.\n", ""),  # no title at all
        ("TITLE: Drug M for\n   migraine in adults\nABSTRACT.METHODS:\nText.\n", "Drug M for migraine in adults"),  # two lines
        ("TITLE: \nDrug N for\nasthma\n\nAsthma fell.\n", "Drug N for asthma"),  # on the next line, over two lines
        ("TITLE: Drug P\nBODY.RESULTS:\nText.\n", "Drug P"),  # up to a BODY line
        ("TITLE: \nABSTRACT.BACKGROUND:\nText.\n", ""),  # an empty title
    ]
    for text, want in cases:
        assert S9.title_of(text) == want, text


def test_builder_files_and_the_checks_inputs_in_the_folders_predict_reads(tmp_path, monkeypatch, recs9_tok, capsys):
    w = recs9_world(tmp_path, monkeypatch)
    manifest = recs9_build_all(w)
    assert "WARNING" not in capsys.readouterr().out  # the made-up counts are the expected ones
    s8 = {s: S9.read_lines(w.s8 / f"evidence_effect_{s}.jsonl") for s in EI.SPLITS}
    clean = {f"PMC{p}" for p, c in RECS9_CLEAN_ARTICLES.items() if c}
    for split in ("train", "validation"):  # session 8's lines of clean articles, byte for byte, in order
        want = [x for x in s8[split] if S9.pmc(json.loads(x)) in clean]
        assert S9.read_lines(w.out / "clean" / f"evidence_effect_{split}.jsonl") == want and len(want) == 3
    monitor = [r["id"] for r in recs9_records(w.out / "clean" / "evidence_effect_validation_monitor.jsonl")]
    assert monitor == sorted(random.Random(20260927).sample(["evidence-5", "evidence-7", "evidence-8"], 2))
    assert [r["id"] for r in recs9_records(w.out / "long" / "evidence_effect_validation_monitor.jsonl")] == monitor
    for name, ids in (("train", ["evidence-1", "evidence-3", "evidence-4"]), ("validation", [f"evidence-{i}" for i in (5, 6, 7, 8)]),
                      ("test", [f"evidence-{i}" for i in (9, 10, 11, 12)])):  # long: clean training records, all others
        assert [r["id"] for r in recs9_records(w.out / "long" / f"evidence_effect_{name}.jsonl")] == ids
    assert ([r["id"] for r in recs9_records(w.out / "swap" / "evidence_effect_train.jsonl")]
            == ["evidence-1", "evidence-1-swap", "evidence-3", "evidence-4", "evidence-4-swap"])
    assert manifest["swap"] == {"swapped_copies": 2, "clean_training_prompts_without_a_copy_same_text": 1}
    assert manifest["counts"]["clean/evidence_effect_train.jsonl"] == {"records": 3, "articles": 2, "answers": {"0": 1, "1": 1, "2": 1}}
    assert manifest["checks"] == {"title_left_out_no_title": 1, "swap_left_out_same_text": {"test": 1, "validation": 1},
                                  "swap_long_left_out_same_text": {"test": 1, "validation": 1}}
    assert manifest["sha256"]["files"] == {name: recs9_sha(w.out / name) for name in S9.ALL_NAMES}
    test8 = {r["id"]: r for r in map(json.loads, s8["test"])}
    none = recs9_records(w.out / "checks" / "none" / "evidence_effect_test.jsonl")
    assert [r["id"] for r in none] == list(test8)
    assert all(r == dict(test8[r["id"]], state=dict(test8[r["id"]]["state"], article="")) for r in none)
    title = recs9_records(w.out / "checks" / "title" / "evidence_effect_test.jsonl")  # PMC33 gives no title: left out
    assert {r["id"]: r["state"]["article"] for r in title} == {"evidence-9": "Drug A for pain", "evidence-10": "Drug H for mood",
                                                              "evidence-12": "Drug A for pain"}
    assert all(r == dict(test8[r["id"]], state=dict(test8[r["id"]]["state"], article=r["state"]["article"])) for r in title)
    sources = {"swap/evidence_effect_test.jsonl": s8["test"], "swap/evidence_effect_validation.jsonl": s8["validation"],
               "swap_long/evidence_effect_test.jsonl": S9.read_lines(w.out / "long" / "evidence_effect_test.jsonl"),
               "swap_long/evidence_effect_validation.jsonl": S9.read_lines(w.out / "long" / "evidence_effect_validation.jsonl")}
    swapped = {}
    for name, source in sources.items():
        orig = {r["id"]: r for r in map(json.loads, source)}
        got = swapped[name] = {r["id"]: r for r in recs9_records(w.out / "checks" / name)}
        assert list(got) == [i for i in orig if i not in ("evidence-7", "evidence-11")]  # the same text left out; IDs kept
        for i, r in got.items():
            o = orig[i]
            assert r["state"] == dict(o["state"], intervention=o["state"]["comparator"], comparator=o["state"]["intervention"])
            assert (r["label"], r["target"]) == (RECS9_SWAP[o["label"]], recs9_onehot(RECS9_SWAP[o["label"]]))
            assert r["group"] == dict(o["group"], accepted_labels=sorted(RECS9_SWAP[x] for x in o["group"]["accepted_labels"]))
            assert r.get("calibrate") == o.get("calibrate") == (True if "validation" in name else None)
    for split, i in (("validation", "evidence-5"), ("test", "evidence-9")):  # P0 is kept at 6,144 tokens, not at 4,096
        assert RECS9_P[0] not in swapped[f"swap/evidence_effect_{split}.jsonl"][i]["state"]["article"]
        assert RECS9_P[0] in swapped[f"swap_long/evidence_effect_{split}.jsonl"][i]["state"]["article"]
    # train_qwen_head.py predict reads <records-dir>/<qid>_<split>.jsonl and builds items at its run's length
    from build_items import load_questions, read_records
    from qwen_items import build_qwen_items
    questions = load_questions(EI.QUESTIONS8)
    for folder, split, max_len in (("none", "test", 4096), ("title", "test", 4096), ("swap", "test", 4096),
                                   ("swap", "validation", 4096), ("swap_long", "test", 6144), ("swap_long", "validation", 6144)):
        records = read_records([w.out / "checks" / folder / f"{EI.QID}_{split}.jsonl"])
        items, summary = build_qwen_items(records, questions, recs9_tokenize, max_len)
        assert (summary["skipped"], summary["truncated"], len(items)) == (0, 0, len(records))
        assert len(records) == (4 if folder == "none" else 3)  # none keeps every test prompt


def test_combination_records(tmp_path):
    out = tmp_path / "s9"
    for length, article in (("clean", "short"), ("long", "short and longer")):
        lines = [json.dumps({"id": f"evidence-{i}", "qid": "evidence_effect", "label": label, "target": recs9_onehot(label),
                             "state": {"intervention": a, "comparator": b, "outcome": "pain", "article": article},
                             "group": {"source": "evidence_inference_v2", "pmcid": str(i), "flag": "none",
                                       "accepted_labels": [label]}}) + "\n"
                 for i, a, b, label in ((1, "drug A", "placebo", 0), (2, "drug B", "Drug b", 2), (3, "drug C", "placebo", 1))]
        S9.write_lines(out / length / "evidence_effect_train.jsonl", lines)
        S9.write_lines(out / length / "evidence_effect_validation_monitor.jsonl", [lines[0].replace("evidence-1", "evidence-7")])
    combo = out / "combo"
    info = S9.build_combo(["swap", "long", "lr1"], out)  # swap and long: the 6,144-token records, each with its copy
    want, same = S9.swap_training_lines(S9.read_lines(out / "long" / "evidence_effect_train.jsonl"))
    assert S9.read_lines(combo / "evidence_effect_train.jsonl") == want and same == 1
    assert [json.loads(x)["id"] for x in want] == ["evidence-1", "evidence-1-swap", "evidence-2", "evidence-3", "evidence-3-swap"]
    assert ((combo / "evidence_effect_validation_monitor.jsonl").read_bytes()
            == (out / "long" / "evidence_effect_validation_monitor.jsonl").read_bytes())
    assert (info["parts"], info["records_from"], info["swapped"], info["same_text_without_a_copy"], info["counts"]) == (
        ["long", "lr1", "swap"], "long", True, 1, {"train": 5, "monitor": 1})
    assert info["sha256"] == {f"combo/{p.name}": recs9_sha(p) for p in sorted(combo.glob("*.jsonl"))}
    assert json.loads((combo / "combo.json").read_text(encoding="utf-8")) == info
    info = S9.build_combo(["long"], out)  # long alone: long's records unchanged
    assert (combo / "evidence_effect_train.jsonl").read_bytes() == (out / "long" / "evidence_effect_train.jsonl").read_bytes()
    assert (info["records_from"], info["swapped"], info["same_text_without_a_copy"], info["counts"]) == (
        "long", False, None, {"train": 3, "monitor": 1})
    info = S9.build_combo(["swap", "lr2"], out)  # without long: the 4,096-token records, swapped
    assert info["records_from"] == "clean" and S9.read_lines(combo / "evidence_effect_train.jsonl") == S9.swap_training_lines(
        S9.read_lines(out / "clean" / "evidence_effect_train.jsonl"))[0]


# ---------------------------------------------------------------- the pod's rebuild

def test_rebuild_with_no_tokenizer_gives_byte_identical_files(tmp_path, monkeypatch, recs9_tok):
    w = recs9_world(tmp_path, monkeypatch)
    manifest = recs9_build_all(w)
    monkeypatch.setitem(sys.modules, "transformers", None)  # from here, importing transformers fails
    pod, check_out = tmp_path / "pod", tmp_path / "records_manifest_pod.json"
    check = S9.rebuild(w.raw, w.s8, w.licenses, w.out / "long" / "article_selection_long.jsonl", pod,
                       w.r9 / "records_manifest.json", check_out)
    assert check["all_equal"] and json.loads(check_out.read_text(encoding="utf-8"))["all_equal"]
    written = sorted(str(p.relative_to(pod)) for p in pod.rglob("*") if p.is_file())
    assert written == sorted(S9.ALL_NAMES)
    for name in written:
        assert (pod / name).read_bytes() == (w.out / name).read_bytes(), name
        assert check["files"][name]["sha256"] == manifest["sha256"]["files"][name] == recs9_sha(pod / name)
    changed = recs9_licenses(tmp_path / "changed.csv", dict(RECS9_CLEAN_ARTICLES, **{"13": False}))  # a changed input
    with pytest.raises(SystemExit, match="differ"):
        S9.rebuild(w.raw, w.s8, changed, w.out / "long" / "article_selection_long.jsonl", tmp_path / "pod2",
                   w.r9 / "records_manifest.json", tmp_path / "check2.json")
    assert not json.loads((tmp_path / "check2.json").read_text(encoding="utf-8"))["all_equal"]
    wrong = tmp_path / "wrong_manifest.json"  # session 8's records are checked against session 8's manifest
    wrong.write_text(json.dumps({"sha256": {"records": {"evidence_effect_train.jsonl": "0" * 64}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not session 8's"):
        S9.s8_lines("train", w.s8, wrong)


def test_license_rules_of_correction_7():
    """Correction 7: the three rules a test agent found misreading statements on 1 October 2026."""
    bmc_waiver = ("The Creative Commons Public Domain Dedication waiver (http://creativecommons.org/publicdomain/zero/1.0/) "
                  "applies to the data made available in this article, unless otherwise stated.")
    # BioMed Central's waiver for data alone: the article itself is not dedicated by CC0
    assert LR.classify_one({"copyright": "Copyright 2010 Smith; licensee BioMed Central Ltd.",
                            "licenses": [recs9_lic(text=bmc_waiver, kind=None)]}) == "copyright notice only"
    # the same waiver beside a CC BY link stays CC BY
    assert LR.classify_one({"copyright": "", "licenses": [
        recs9_lic(href="http://creativecommons.org/licenses/by/2.0", text="This is an Open Access article distributed under the "
                  "terms of the Creative Commons Attribution License. " + bmc_waiver)]}) == "cc by"
    # "U.S." with dots, as Wiley prints it, is read as a US government work
    assert LR.classify_one({"copyright": "This article is a U.S. Government work and is in the public domain in the USA.",
                            "licenses": []}) == "public domain"
    assert LR.classify_one({"copyright": "This article is a U.S. Government work.", "licenses": []}) == "public domain"
    # the general rule about federal employees' contributions alone is not a statement about the article
    assert LR.classify_one({"copyright": "The contributions of employees of the Federal Government to this work are not "
                            "subject to copyright.", "licenses": []}) == "general statement only"
    # but a statement about the article beside that rule still counts
    assert LR.classify_one({"copyright": "The contributions of employees of the Federal Government to this work are not "
                            "subject to copyright. This article is a US Government work and is in the public domain.",
                            "licenses": []}) == "public domain"
    for cls in ("copyright notice only", "general statement only"):
        assert cls not in LR.CLEAN

"""Tests for EI's records (docs/prompt_session8.md, step 4: EI'S RECORDS and THE ARTICLE SELECTION), on made-up inputs.

A made-up copy of EI's files (split lists, prompts, annotations, the README's caveat lists and a few article texts) is
written to a temporary folder. Checked: the accepted-row filter, the label with its tie rule, the accepted labels, and the
drops and their counts; the "Incorrect" prompts dropped from training only; the title-and-abstract split, the passages,
their sections and the results-section test; BM25 against a hand calculation and the order (results sections first, ties
by position); coverage against a hand calculation; the monitoring subset (seeded, IDs sorted as text) and the PMCID in
every record. With Qwen's tokenizer (models/qwen3_4b_base/; skipped without it): every record fits 4,096 tokens with 16 to
spare as src/train/qwen_items.py renders it, passages rejoined in article order, the title and abstract cut from their
end when they alone do not fit; from-selection gives byte-identical records without a tokenizer; and a builder run
stopped part way resumes and gives the same files."""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC / "tasks"))
sys.path.insert(0, str(SRC / "train"))

import evidence_inference as EI  # noqa: E402

QWEN = ROOT / "models" / "qwen3_4b_base"
needs_qwen = pytest.mark.skipif(not (QWEN / "tokenizer.json").exists(), reason="Qwen's tokenizer not in models/qwen3_4b_base")

RESULTS_TEXT = "Pain decreased significantly with aspirin compared with placebo (p = 0.01)."


def article(title, abstract, sections):
    """An article text in EI's layout: TITLE, ABSTRACT sections, then BODY sections of paragraphs."""
    parts = [f"TITLE: \n{title}\n\n", f"ABSTRACT.BACKGROUND:\n{abstract}\n\n"]
    for header, paragraphs in sections:
        parts.append(f"BODY.{header}:\n" + "\n\n".join(paragraphs) + "\n\n")
    return "".join(parts)


ARTICLES = {
    "11": article("Aspirin for pain", "Aspirin was compared with placebo for pain.",
                  [("METHODS", ["We randomized adults to aspirin or placebo."]),
                   ("RESULTS", [RESULTS_TEXT, "Nausea was similar in both arms."]),
                   ("DISCUSSION", ["Aspirin helps."])]),
    "12": article("Drug B for sleep", "Drug B was compared with usual care.",
                  [("ADVERSE EVENTS", ["Headache was more common with drug B."]), ("CONCLUSIONS", ["Drug B is safe."])]),
    "21": article("Drug C for fever", "Drug C lowered fever.", [("PRIMARY END POINT", ["Fever fell with drug C."])]),
    "22": article("Drug D for cough", "Drug D and cough.", [("RESULTS", ["Cough was unchanged."])]),
    "23": article("Drug E for rash", "Drug E and rash.", [("FINDINGS", ["Rash cleared with drug E."])]),
    "31": article("Drug F for gout", "Drug F and gout.", [("OUTCOMES", ["Gout flares fell with drug F."])]),
    "32": "",
}
SPLIT_IDS = {"train": ["11", "12"], "validation": ["21", "22", "23"], "test": ["31", "32"]}
PROMPTS = [  # PromptID, PMCID, outcome, intervention, comparator
    ("1", "11", "pain", "aspirin", "placebo"),
    ("2", "11", "nausea", "aspirin", "placebo"),
    ("3", "12", "sleep", "drug B", "usual care"),
    ("4", "21", "fever", "drug C", "placebo"),
    ("5", "22", "cough", "drug D", "placebo"),
    ("6", "31", "gout flares", "drug F", "placebo"),
    ("7", "32", "anything", "drug G", "placebo"),
    ("8", "99", "anything", "drug H", "placebo"),
    ("9", "12", "headache", "drug B", "usual care"),
    ("10", "23", "rash", "drug E", "placebo"),
]


def ann(pid, pmcid, code, valid_label="True", valid_reasoning="True", text="evidence", start=-1, end=-1):
    return {"": "0", "UserID": "0", "PromptID": pid, "PMCID": pmcid, "Valid Label": valid_label, "Valid Reasoning": valid_reasoning,
            "Label": "x", "Annotations": text, "Label Code": str(code), "In Abstract": "False", "Evidence Start": str(start),
            "Evidence End": str(end)}


def make_raw(folder):
    folder = Path(folder)
    a = folder / "annotations"
    (a / "splits").mkdir(parents=True)
    (a / "txt_files").mkdir()
    for s, ids in SPLIT_IDS.items():
        (a / "splits" / f"{s}_article_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    with open(a / "prompts_merged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PromptID", "PMCID", "Outcome", "Intervention", "Comparator"])
        w.writerows(PROMPTS)
    start = ARTICLES["11"].index(RESULTS_TEXT)
    rows = [ann("1", "11", 1, start=start, end=start + len(RESULTS_TEXT)), ann("1", "11", -1),  # a tie: -1 wins
            ann("2", "11", 0),                                                              # Incorrect: dropped from training
            ann("3", "12", 0, valid_label="False"), ann("3", "12", 1, text="  "),         # no accepted row
            ann("4", "21", -1),                                                             # Incorrect, but validation: kept
            ann("5", "22", 0), ann("5", "22", 0), ann("5", "22", 1),                       # the mode 0; accepted 0 and 1
            ann("6", "31", 1), ann("7", "32", 0), ann("8", "99", 0),
            ann("9", "12", 0, valid_reasoning="False"), ann("9", "12", -1),
            ann("10", "23", 0)]
    fields = list(rows[0])
    with open(a / "annotations_merged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (a / "README.md").write_text("Caveats\n\n### Incorrect: \n2, 4\n\n### Questionable: \n6\n\n### Somewhat malformed: \n9, 10\n",
                                 encoding="utf-8")
    for pmcid, text in ARTICLES.items():
        (a / "txt_files" / f"PMC{pmcid}.txt").write_text(text, encoding="utf-8")
    files = {rel: {"sha256": "made up", "bytes": 1} for rel in EI.FILES}
    (folder / "download_progress.json").write_text(json.dumps({"commit": EI.COMMIT, "files": files}), encoding="utf-8")
    return folder


@pytest.fixture
def raw(tmp_path):
    return make_raw(tmp_path / "raw")


def by_pid(kept):
    return {p["pid"]: (s, p) for s, ps in kept.items() for p in ps}


def test_accepted_rows_label_tie_rule_and_drops(raw):
    assert [r["Label Code"] for r in EI.accepted([ann("x", "1", 0), ann("x", "1", 1, valid_label="False"),
                                                  ann("x", "1", -1, text=" ")])] == ["0"]
    assert EI.label_code([1, -1]) == -1 and EI.label_code([0, 0, 1]) == 0 and EI.label_code([1, 1, 0, 0, -1]) == 0
    kept, drops, disagree = EI.select_prompts(raw)
    assert drops == {"PMCID in no split file": 1, "train: no accepted row": 1, "test: empty article text": 1,
                     "train: listed as Incorrect in annotations/README.md": 1}
    assert disagree == {"train": 1, "validation": 1}
    got = by_pid(kept)
    assert sorted(got) == ["1", "10", "4", "5", "6", "9"]
    assert got["1"][1]["label"] == 1 and got["1"][1]["accepted_labels"] == [0, 1]  # Label Code 1 and -1 tie: -1, decreased
    assert got["5"][1]["label"] == 2 and got["5"][1]["accepted_labels"] == [0, 2]  # mode 0: no difference
    assert got["9"][1]["label"] == 1 and got["9"][1]["accepted_labels"] == [1]     # the row without valid reasoning is not accepted
    assert got["6"][1]["flag"] == "Questionable" and got["10"][1]["flag"] == "Somewhat malformed" and got["1"][1]["flag"] == "none"
    assert [p["pid"] for p in kept["train"]] == ["1", "9"]  # PromptID order, numeric


def test_incorrect_prompts_dropped_from_training_only(raw):
    kept, _, _ = EI.select_prompts(raw)
    got = by_pid(kept)
    assert "2" not in got  # training, listed as Incorrect
    assert got["4"][0] == "validation" and got["4"][1]["flag"] == "Incorrect"  # validation keeps it, flagged


def test_title_abstract_passages_and_sections():
    text = ARTICLES["11"]
    prefix, passages = EI.article_parts(text)
    assert prefix == "TITLE: \nAspirin for pain\n\nABSTRACT.BACKGROUND:\nAspirin was compared with placebo for pain."
    assert [p[2] for p in passages] == ["BODY.METHODS:\nWe randomized adults to aspirin or placebo.",
                                        "BODY.RESULTS:\n" + RESULTS_TEXT, "Nausea was similar in both arms.",
                                        "BODY.DISCUSSION:\nAspirin helps."]
    assert [p[3] for p in passages] == [False, True, True, False]  # the second results paragraph keeps its section
    assert all(text[s:e] == t for s, e, t, _ in passages)
    for header, want in (("ADVERSE EVENTS", True), ("PRIMARY END POINT", True), ("PRIMARY ENDPOINT", True), ("FINDINGS", True),
                         ("SAFETY", True), ("TABLE 2", True), ("CONCLUSIONS", False), ("METHODS", False)):
        _, ps = EI.article_parts(article("t", "a", [(header, ["x y"])]))
        assert ps[0][3] is want, header
    assert EI.article_parts("TITLE: \nonly a title\n")[1] == []  # no BODY line: the whole text is the title and abstract


def test_bm25_against_a_hand_calculation_and_the_order():
    stats = {"N": 4, "avgL": 5.0, "df": {"aspirin": 2, "pain": 1}}
    got = EI.bm25(["aspirin", "pain", "placebo"], ["aspirin", "pain", "pain", "reduced", "trial"], stats)
    idf_a = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    idf_p = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    hand = idf_a * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 5 / 5)) + idf_p * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 5 / 5))
    assert got == pytest.approx(hand, rel=1e-12)
    assert EI.words("The Aspirin-group vs. placebo, 2 patients; x A1") == ["aspirin", "placebo", "a1"]
    passages = [(0, 1, "unrelated words here", False), (2, 3, "aspirin pain", True), (4, 5, "unrelated", True),
                (6, 7, "aspirin pain pain", False), (8, 9, "aspirin pain", True)]
    prompt = {"intervention": "aspirin", "comparator": "placebo", "outcome": "pain"}
    order = EI.order_passages(prompt, passages, {"N": 10, "avgL": 2.0, "df": {"aspirin": 3, "pain": 3}})
    assert order == [1, 4, 2, 3, 0]  # results sections first, the higher score first, ties by position


def test_coverage_against_a_hand_calculation():
    passages = [(100, 150, "p0", False), (200, 260, "p1", True), (300, 320, "p2", False)]
    assert EI.coverage_of({"spans": [(210, 250)]}, 80, [1], passages)       # inside a kept passage
    assert not EI.coverage_of({"spans": [(210, 250)]}, 80, [0, 2], passages)  # its passage not kept
    assert EI.coverage_of({"spans": [(10, 70)]}, 80, [], passages)          # inside the kept title and abstract
    assert not EI.coverage_of({"spans": [(70, 90)]}, 80, [], passages)      # crosses the end of the kept prefix
    assert not EI.coverage_of({"spans": [(140, 210)]}, 80, [0, 1], passages)  # across two kept passages: not wholly inside one
    assert EI.coverage_of({"spans": [(10, 90), (300, 310)]}, 80, [2], passages)  # one of two spans is enough


def test_monitor_subset_is_seeded_and_sorted(monkeypatch):
    records = [{"id": f"evidence-{i}", "group": {"pmcid": str(i % 3)}} for i in (5, 12, 100, 7, 33)]
    monkeypatch.setattr(EI, "MONITOR", 3)
    got = EI.monitor_subset(records)
    ids = sorted(r["id"] for r in records)  # as text: evidence-100 before evidence-12
    want = sorted(random.Random(EI.SEED).sample(ids, 3))
    assert [r["id"] for r in got] == want and all("pmcid" in r["group"] for r in got)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_once(raw, out_dir, results, max_seconds=1e9, calls=40):
    for _ in range(calls):
        got = EI.build(raw, out_dir, results, QWEN, max_seconds, s7_b3=Path(out_dir) / "no_adapter")
        if got["finished"]:
            return got
    raise AssertionError("the build did not finish")


@needs_qwen
def test_budget_order_cut_from_selection_and_resume(raw, tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from build_items import load_questions
    from qwen_items import build_qwen_items

    monkeypatch.setattr(EI, "MONITOR", 2)
    long_abstract = " ".join(f"word{i} value{i}" for i in range(4000))  # the title and abstract alone do not fit
    ARTICLES_LONG = dict(ARTICLES, **{"23": article("Drug E for rash", long_abstract, [("FINDINGS", ["Rash cleared with drug E."])])})
    (raw / "annotations" / "txt_files" / "PMC23.txt").write_text(ARTICLES_LONG["23"], encoding="utf-8")
    out, results = tmp_path / "out", tmp_path / "results"
    got = build_once(raw, out, results)
    tok = AutoTokenizer.from_pretrained(str(QWEN))
    questions = load_questions(EI.QUESTIONS8)
    for split in EI.SPLITS:
        records = [json.loads(line) for line in open(out / f"evidence_effect_{split}.jsonl", encoding="utf-8")]
        items, summary = build_qwen_items(records, questions, tok, 4096)
        assert summary["skipped"] == 0 and summary["truncated"] == 0
        assert all(len(it["ids"]) <= 4096 - 16 for it in items)
        for r in records:
            assert r["group"]["pmcid"] and r["qid"] == "evidence_effect" and r["id"].startswith("evidence-")
            text = ARTICLES_LONG[r["group"]["pmcid"]]
            prefix, passages = EI.article_parts(text)
            art = r["state"]["article"]
            if r["group"]["pmcid"] == "23":
                assert len(art) < len(prefix) and prefix.startswith(art)  # cut from its end, no passage taken
            else:
                assert art.startswith(prefix)
                kept = [p[2] for p in passages if ("\n\n" + p[2]) in art]
                positions = [art.index(p) for p in kept]
                assert positions == sorted(positions)  # passages rejoined in article order
    assert (results / "records_manifest.json").exists() and got["manifest"]["counts"]["validation"]["prompts"] == 3
    sel = [json.loads(line) for line in open(out / "article_selection.jsonl", encoding="utf-8")]
    assert {r["id"] for r in sel} == {f"evidence-{p}" for p in ("1", "9", "4", "5", "10", "6")}
    # from-selection: byte-identical records with no tokenizer
    import builtins
    real_import = builtins.__import__

    def no_transformers(name, *a, **k):
        if name.startswith("transformers"):
            raise ImportError("no tokenizer in from-selection")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_transformers)
    check = EI.from_selection(raw, out / "article_selection.jsonl", tmp_path / "again", results / "records_manifest.json",
                              tmp_path / "check.json")
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert check["all_equal"]
    for name in check["files"]:
        assert sha(out / name) == sha(tmp_path / "again" / name)
    # a build stopped after every article resumes to the same files
    out2, results2 = tmp_path / "out2", tmp_path / "results2"
    calls = 0
    while True:
        calls += 1
        if EI.build(raw, out2, results2, QWEN, 0.0, s7_b3=out2 / "no_adapter")["finished"]:
            break
        assert calls < 50
    assert calls > 2
    for name in list(check["files"]) + ["article_selection.jsonl", "bm25_stats.json"]:
        assert sha(out / name) == sha(out2 / name), name

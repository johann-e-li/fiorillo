"""Tests for session 5's text builders (docs/prompt_session5.md, step 2), on made-up inputs only: the PMC cleaning,
routing and inventory, the date slicing of the case-report search, exclusion by PMID, the 13-word check, part files,
block packing, the mix sampler's shares, and the web rules with made-up field names given through the config."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "corpus"))

import build_abstract_parts as BA  # noqa: E402
import decontaminate as D  # noqa: E402
import fetch_pmc as F  # noqa: E402
import filter_web as W  # noqa: E402
import parts as P  # noqa: E402
import tokenize_blocks as T  # noqa: E402

REF_CONTEXT = ("Background: the aim of this study was to assess whether early mobilisation after hip fracture surgery "
               "reduces the length of stay in older adults")
REF_ANSWER = "Early mobilisation shortened the hospital stay without raising the rate of falls in this cohort of patients"


def write_reference(path):
    item = {"pubid": 1, "contexts": [REF_CONTEXT, "Methods: a cohort of 400 patients was followed."], "long_answer": REF_ANSWER}
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    return path


def test_words_lowercase_and_non_alphanumerics():
    assert D.words("Hip-fracture, (N=400); p<0.05 — émigré_x") == ["hip", "fracture", "n", "400", "p", "0", "05", "émigré", "x"]


def test_thirteen_word_check(tmp_path):
    checker = D.Checker(D.reference_texts(write_reference(tmp_path / "ref.jsonl")))
    run13 = "WHETHER early-mobilisation, after hip fracture surgery reduces the LENGTH of stay in"
    assert len(D.words(run13)) == 13 and checker.matches("Unrelated opening. " + run13 + " and more.")
    run12 = "whether early mobilisation after hip fracture surgery reduces the length of stay"
    assert len(D.words(run12)) == 12 and not checker.matches(run12)
    crossing = "cohort of 400 patients was followed. Early mobilisation shortened the hospital stay without"
    assert len(D.words(crossing)) == 13 and checker.matches(crossing)  # from the last context into the long answer
    assert not checker.matches("the the the the the the the the the the the the the the")


def test_parallel_checker_agrees_with_one_process(tmp_path):
    ref = write_reference(tmp_path / "ref.jsonl")
    texts = ["nothing here at all", "the aim of this study was to assess whether early mobilisation after hip fracture surgery",
             "a cohort of 400 patients was followed"] * 5
    serial = D.Checker(D.reference_texts(ref))
    with D.ParallelChecker(ref, processes=2) as pc:
        assert pc.check(texts, chunk=4) == [serial.matches(t) for t in texts]


def test_exclusion_list_and_abstract_text(tmp_path):
    path = tmp_path / "pmids.txt"
    path.write_text("12345\n  678\n\n", encoding="utf-8")
    assert BA.load_exclusion(path) == {12345, 678}
    assert BA.record_text(None, "Abstract.") == "Abstract." and BA.record_text("Title.", "Abstract.") == "Title. Abstract."
    names = [f"pubmed26n{i:04d}.parquet" for i in range(1, 30)]
    assert BA.shard_order(names) == BA.shard_order(list(reversed(names))) != sorted(names)


def test_part_files_are_deterministic(tmp_path):
    docs = [("1", "first text"), ("2", "second, with ünïcode")]
    a, b = P.write_part(tmp_path / "a" / P.part_name(0), docs), P.write_part(tmp_path / "b" / P.part_name(0), docs)
    assert a["sha256"] == b["sha256"] and a["documents"] == 2 and a["file"] == "part-0000.jsonl.gz"
    assert list(P.read_part(tmp_path / "a" / "part-0000.jsonl.gz")) == docs
    assert P.strip_special("a [MASK] b[SEP]", ["[MASK]", "[SEP]"]) == "a   b "


PMC_TXT = "\n".join([
    "", "JOURNAL INFORMATION", "==============================", "NLM Title Abbreviation: J Test", "",
    "ARTICLE INFORMATION", "==============================", "PMCID: PMC1", "Author One", "one@example.org",
    "Keywords: test", "", "\x9f==============================\x9f",
    "Introduction", "", "Body text   with  spaces.", "Variable\tValue\t", "Age\t54\t",
    "A References section is discussed here in the body.", "References", "", "1 Some cited paper",
    "Appendix", "Supplementary words.", "REFERENCES", "2 Another cited paper", ""])


def test_pmc_clean_text():
    text = F.clean_text(PMC_TXT)
    assert text == ("Introduction Body text with spaces. A References section is discussed here in the body. References "
                    "1 Some cited paper Appendix Supplementary words.")
    assert "one@example.org" not in text and "\t" not in text and "Age" not in text
    assert F.clean_text("\nJOURNAL INFORMATION\n==========\nno body separator\n") is None
    assert F.clean_text("\x9f==========\x9f\n\nReferences\n1 x\n") is None


def test_pmc_routing_rules():
    meta = {"license_code": "CC BY", "is_pmc_openaccess": True, "is_retracted": False, "is_historical_ocr": False, "pmid": 10}
    cases, exclusion = {10}, {99}
    assert F.route(meta, "case_reports", cases, exclusion) is None
    assert F.route(meta, "pmc", cases, exclusion) == "routing"
    assert F.route(dict(meta, pmid=11), "pmc", cases, exclusion) is None
    assert F.route(dict(meta, pmid=None), "pmc", cases, exclusion) is None
    assert F.route(dict(meta, pmid=None), "case_reports", cases, exclusion) == "routing"
    assert F.route(dict(meta, pmid=99), "pmc", cases, exclusion) == "pubmedqa_pmid"
    assert F.route(dict(meta, license_code="CC BY-SA"), "case_reports", cases, exclusion) == "licence"
    assert F.route(dict(meta, license_code="CC0"), "case_reports", cases, exclusion) is None
    assert F.route(dict(meta, is_pmc_openaccess=False), "case_reports", cases, exclusion) == "not_open_access"
    assert F.route(dict(meta, is_retracted=True), "case_reports", cases, exclusion) == "retracted"
    assert F.route(dict(meta, is_historical_ocr=True), "case_reports", cases, exclusion) == "historical_ocr"
    assert F.s3_https("s3://pmc-oa-opendata/PMC1.2/PMC1.2.txt?md5=abc") == (F.BUCKET + "PMC1.2/PMC1.2.txt", "abc")


def test_date_slices_split_until_under_the_cap():
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(366)]
    records = {d: (40 if d.month == 3 else 3) for d in days}

    def count(a, b):
        return sum(n for d, n in records.items() if a <= d <= b)

    slices = F.date_slices(count, dt.date(2020, 1, 1), dt.date(2020, 12, 31), cap=100)
    assert all(n <= 100 for _, _, n in slices) and sum(n for _, _, n in slices) == sum(records.values())
    assert all(slices[i][1] < slices[i + 1][0] for i in range(len(slices) - 1))
    with pytest.raises(RuntimeError):
        F.date_slices(lambda a, b: 500, dt.date(2020, 1, 1), dt.date(2020, 1, 1), cap=100)


def test_uid_slices_return_every_identifier_once():
    """Correction 3: identifier ranges split in halves until each slice holds at most the cap return every identifier
    exactly once, in order, however dense the range."""
    import bisect

    rng = np.random.default_rng(3)
    universe = sorted(set(rng.integers(1, 200_000, size=30_000).tolist()) | set(range(150_000, 170_000)))

    def search(a, b):
        hits = universe[bisect.bisect_left(universe, a): bisect.bisect_right(universe, b)]
        return len(hits), hits[:1000]

    slices = []
    for a in range(1, 200_001, 50_000):
        slices.extend(F.uid_slices(search, a, min(200_000, a + 49_999), cap=1000))
    assert [i for s in slices for i in s[3]] == universe
    assert all(s[2] <= 1000 and s[2] == len(s[3]) for s in slices)
    assert F.uid_slices(lambda a, b: (0, []), 1, 10) == []


def test_inventory_latest_versions():
    rows = [["pmc-oa-opendata", "metadata/PMC5.1.json", "t", "e"], ["pmc-oa-opendata", "metadata/PMC5.3.json", "t", "e"],
            ["pmc-oa-opendata", "metadata/PMC5.2.json", "t", "e"], ["pmc-oa-opendata", "metadata/PMC7.1.json", "t", "e"],
            ["pmc-oa-opendata", "PMC5.3/PMC5.3.txt", "t", "e"]]
    assert F.latest_versions(rows) == {5: 3, 7: 1}
    assert F.latest_versions(rows, wanted={7}) == {7: 1}
    xml = ("<Prefix>inventory-reports/pmc-oa-opendata/metadata/2026-09-26T01-00Z/</Prefix>"
           "<Prefix>inventory-reports/pmc-oa-opendata/metadata/2026-09-27T01-00Z/</Prefix>"
           "<Prefix>inventory-reports/pmc-oa-opendata/metadata/hive/</Prefix>")
    assert F.latest_inventory_prefix(xml).endswith("2026-09-27T01-00Z/")


def test_block_packing():
    docs = [np.arange(1, 1 + n, dtype=np.uint16) for n in (700, 500, 900, 30, 1000)]
    lengths = np.array([len(d) for d in docs])
    n_blocks = T.stream_length(lengths) // T.CONTENT
    dst = np.zeros((n_blocks, T.BLOCK), dtype=np.uint16)
    dropped = T.pack(iter(docs), n_blocks, dst, cls_id=50281, sep_id=50282)
    assert n_blocks == 3 and dropped == T.stream_length(lengths) - 3 * T.CONTENT
    assert (dst[:, 0] == 50281).all() and (dst[:, -1] == 50282).all()
    stream = np.concatenate([np.concatenate([d, [50282]]) for d in docs]).astype(np.uint16)
    assert np.array_equal(dst[:, 1:-1].reshape(-1), stream[: 3 * T.CONTENT])
    assert T.first_documents(lengths, n_blocks).tolist() == [0, 1, 2]
    chosen = T.choose_heldout(lengths, 1, seed=[1, 2, 3])
    assert sum(int(lengths[d]) + 1 for d in chosen) >= T.CONTENT and len(set(chosen)) == len(chosen)
    with pytest.raises(ValueError):
        T.choose_heldout(lengths, 10, seed=1)


def test_tokenize_blocks_keeps_heldout_documents_apart(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4}
    vocab.update({f"w{i}": 5 + i for i in range(300)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    rng = np.random.default_rng(0)
    docs = [(str(i), " ".join(f"w{j}" for j in rng.integers(0, 300, size=400))) for i in range(40)]
    docs[3] = ("3", docs[3][1] + " [MASK]")  # a special-token string in the text is replaced, not read as [MASK]
    parts_dir = tmp_path / "parts"
    info = [P.write_part(parts_dir / P.part_name(0), docs[:20]), P.write_part(parts_dir / P.part_name(1), docs[20:])]
    tokens = P.token_counts(P.load_tokenizer(tok_path), [t for _, t in docs], P.special_strings(tok_path))
    P.save_json(parts_dir / "manifest.json", {"parts": info, "tokens": int(sum(tokens))})
    counts = T.build("abstracts", parts_dir, tmp_path / "blocks", tok_path, n_heldout=2)
    assert counts["tokens_equal_manifest"] and counts["tokens"] == 40 * 400
    src = T.open_source(tmp_path / "blocks", "abstracts")
    assert src["heldout"].shape == (2, 1024) and src["train"].shape[0] == counts["train_blocks"]
    assert sorted(src["perm"].tolist()) == list(range(counts["train_blocks"]))
    assert not (np.asarray(src["train"]) == 4).any() and not (np.asarray(src["heldout"]) == 4).any()
    held = set(counts["heldout_document_indices"])
    assert len(held) == counts["heldout_documents"] and counts["heldout_documents"] + counts["train_documents"] == 40
    block_doc = np.load(tmp_path / "blocks" / "abstracts" / "block_doc.npy")
    assert block_doc[0] == min(set(range(40)) - held) and not held & set(block_doc.tolist())
    assert counts["train_blocks"] == (40 - len(held)) * 401 // 1022


def test_mix_shares_within_one_percent():
    perms = {s: np.random.default_rng(i).permutation(20000) for i, s in enumerate(("abstracts", "pmc", "case_reports"))}
    sampler = T.MixSampler({"abstracts": 50, "pmc": 40, "case_reports": 10}, perms)
    draws = sampler.draw_many(4096)
    for s, share in (("abstracts", 0.5), ("pmc", 0.4), ("case_reports", 0.1)):
        blocks = [b for src, b in draws if src == s]
        assert abs(len(blocks) / len(draws) - share) < 0.01
        assert len(blocks) == len(set(blocks)) and blocks == perms[s][: len(blocks)].tolist()
    resumed = T.MixSampler({"abstracts": 50, "pmc": 40, "case_reports": 10}, perms, state=sampler.state())
    fresh = T.MixSampler({"abstracts": 50, "pmc": 40, "case_reports": 10}, perms)
    fresh.draw_many(4096)
    assert resumed.draw_many(100) == fresh.draw_many(100)


def test_task_passes_start_a_fresh_permutation():
    perms = {"task": np.arange(10)}
    sampler = T.MixSampler({"task": 100}, perms, passes=3)
    draws = [b for _, b in sampler.draw_many(30)]
    assert draws[:10] == list(range(10)) and sorted(draws[10:20]) == list(range(10)) and draws[10:20] != draws[20:30]
    with pytest.raises(RuntimeError):
        sampler.draw()


def web_config(tmp_path):
    cfg = {"format": "jsonl", "text": "body", "source": "meta.origin", "cancer": "is_cancer", "cancer_true": [1, "yes"],
           "sources_kept": ["RedPajamaCommonCrawl", "RedPajamaC4", "RedPajamaWikipedia", "RedPajamaStackExchange"],
           "ratings": {"medical_relevance": "r1", "precision_and_factual_detail": "r2", "safety_and_ethical_standards": "r3"}}
    import yaml

    path = tmp_path / "fields.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return W.load_config(path)


def test_web_rules_with_made_up_fields(tmp_path):
    cfg = web_config(tmp_path)
    good = {"body": "text", "meta": {"origin": "RedPajamaC4"}, "is_cancer": 0, "r1": 5, "r2": "5", "r3": 5.0}
    assert W.judge(good, cfg) == ("keep", "text", False)
    assert W.judge(dict(good, is_cancer="yes"), cfg)[2] is True
    assert W.judge(dict(good, meta=json.dumps({"origin": "RedPajamaWikipedia"})), cfg)[0] == "keep"  # meta as a JSON string
    assert W.judge(dict(good, meta="{broken"), cfg)[0] == "unreadable_fields"
    assert W.judge(dict(good, r2=4), cfg)[0] == "rating_below_5"
    assert W.judge(dict(good, meta={"origin": "RedPajamaBook"}), cfg)[0] == "source_not_kept"
    assert W.judge({k: v for k, v in good.items() if k != "r3"}, cfg)[0] == "unreadable_fields"
    assert W.judge(dict(good, body="  "), cfg)[0] == "unreadable_fields"
    assert W.accept_cancer(0, 100, 50) and not W.accept_cancer(0, 100, 51) and not W.accept_cancer(40, 100, 11)
    (tmp_path / "empty.yaml").write_text("format: jsonl\ntext: null\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        W.load_config(tmp_path / "empty.yaml")


def test_filter_web_stopped_and_resumed_writes_the_same_documents(tmp_path):
    """filter_web.py stopped after every chunk and resumed (as on the linked shell) writes the same documents, in the
    same order, with the same counts, as one uninterrupted run; only the part boundaries differ."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    import yaml

    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4}
    vocab.update({f"w{i}": 5 + i for i in range(50)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    ref = write_reference(tmp_path / "ref.jsonl")
    cfg = {"format": "jsonl", "text": "body", "source": "meta.origin", "cancer": "is_cancer", "cancer_true": [1],
           "sources_kept": ["RedPajamaC4", "RedPajamaCommonCrawl"],
           "ratings": {"medical_relevance": "r1", "precision_and_factual_detail": "r2", "safety_and_ethical_standards": "r3"}}
    cfg_path = tmp_path / "fields.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rng = np.random.default_rng(1)
    raw = tmp_path / "raw"
    names = ["train/a.json", "train/b.json", "train/c.json"]  # c.json is on the list but not downloaded
    for name in names[:2]:
        lines = []
        for i in range(17):
            text = " ".join(f"w{j}" for j in rng.integers(0, 50, size=int(rng.integers(5, 40))))
            row = {"body": text, "meta": {"origin": "RedPajamaC4" if i % 5 else "RedPajamaBook"},
                   "is_cancer": int(i % 3 == 0), "r1": 5, "r2": 5.0 if i % 7 else 4.99, "r3": "5"}
            if i == 4:
                row["body"] = "intro " + REF_CONTEXT  # fails the 13-word check
            lines.append(json.dumps(row))
        lines.insert(8, "{not json")
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        (raw / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = tmp_path / "files.txt"
    files.write_text("\n".join(names) + "\n", encoding="utf-8")

    def run(out, *extra):
        W.main(["--raw", str(raw), "--files", str(files), "--config", str(cfg_path), "--out", str(out),
                "--reference", str(ref), "--tokenizer", str(tok_path), "--manifests", str(out / "manifests"),
                "--per-part", "4", "--chunk", "3", "--processes", "1", *extra])

    whole, pieces = tmp_path / "whole", tmp_path / "pieces"
    run(whole, "--max-seconds", "1000")
    run(whole, "--max-seconds", "1000", "--finish")

    def position():
        path = pieces / "progress.json"
        return json.loads(path.read_text(encoding="utf-8"))["file_pos"] if path.exists() else 0

    for _ in range(100):
        if position() >= 2:
            break
        run(pieces, "--max-seconds", "0")
    else:
        pytest.fail("the stopped runs never reached the end of the downloaded files")
    run(pieces, "--max-seconds", "0", "--finish")

    def documents(out):
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        return manifest, [d for part in manifest["parts"] for d in P.read_part(out / part["file"])]

    m1, d1 = documents(whole)
    m2, d2 = documents(pieces)
    assert d1 == d2 and len(d1) > 0
    for key in ("documents", "tokens", "cancer_tokens", "drops"):
        assert m1[key] == m2[key]
    assert m1["drops"]["thirteen_word"] == 2 and m1["drops"]["unparseable_row"] == 2
    assert [f["rows_read"] for f in m1["files_read"]] == [18, 18] == [f["rows_read"] for f in m2["files_read"]]
    assert len(m2["parts"]) > len(m1["parts"]) and 3 * m1["cancer_tokens"] <= m1["tokens"]


def test_web_config_in_the_repository_reads_the_dataset_rows():
    """The filled config (step 7) on a made-up row shaped like TheBlueScrubs-v1's rows."""
    cfg = W.load_config(SRC / "configs" / "web_fields_session5.yaml")
    meta = {"source": "RedPajamaC4", "probability": 0.93, "ID": "x", "Scope of Medical Relevance": 5.0,
            "Precision and Factual Detail": 5.0, "Safety and Ethical Standards": 5.0, "cancer": 1}
    row = {"text": "t", "meta": meta}
    assert W.judge(row, cfg) == ("keep", "t", True)
    assert W.judge({"text": "t", "meta": dict(meta, cancer=0)}, cfg) == ("keep", "t", False)
    assert W.judge({"text": "t", "meta": dict(meta, source="RedPajamaBook")}, cfg)[0] == "source_not_kept"
    assert W.judge({"text": "t", "meta": dict(meta, **{"Safety and Ethical Standards": 4.99})}, cfg)[0] == "rating_below_5"
    assert W.judge({"text": "t", "meta": "{damaged"}, cfg)[0] == "unreadable_fields"

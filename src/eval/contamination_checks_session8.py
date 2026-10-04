"""Correction 14 (docs/prompt_session8.md): three descriptive checks of s8_ei's EI result.

Not pre-registered. They change no registered number, criterion or reading, and they read only files that
already exist.

1. Reading the evidence. s8_ei's and gemma4_31b's accuracy and macro-F1 (their calibrated predictions) on EI's
   validation and test prompts, apart by whether the builder marked the prompt's evidence as inside its
   4,096-token input (data/processed/session8/parts/selection_<split>.jsonl, "covered"), outside it, or without
   usable offsets; and apart by whether an accepted annotation marks its evidence as in the abstract
   (annotations_merged.csv, "In Abstract"). 95 percent percentile intervals from 1,000 article resamples,
   seed 20260927, drawn as compare_session8.py draws them.
2. PubMedQA. Validation and test articles whose title and abstract share at least two normalized sentences of
   60 characters or more with one abstract of PQA-L, PQA-A or PQA-U (data/raw/public_s5/); whether each
   matched abstract is among the items arm B was trained on; s8_ei's accuracy on the matched articles' prompts
   and on the rest.
3. EI's own splits. Validation and test articles whose normalized title, or at least two normalized sentences
   of their title and abstract, also appear in an EI training article.

Correction 15 adds intervals for two differences: for s8_ei and gemma4_31b on each split, the accuracy where the
evidence is outside the input minus the accuracy where it is inside; for s8_ei on each split, the accuracy on the
articles that share sentences with a PubMedQA abstract minus the accuracy on the rest. Both groups' accuracies are
taken within each of 1,000 resamples of the split's articles (seed 20260927).

Writes results/fiorillo_v0_4/contamination_checks.json. PubMedQA's matches are kept per file in
models/fiorillo_v0_4/contamination_cache/ (ignored by git), so a run stopped by --max-seconds resumes.

Usage (the linked shell, from the project folder):
    python3 src/eval/contamination_checks_session8.py [--max-seconds 150]
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "tasks"))
import compare_session8 as S  # noqa: E402
import evidence_inference as E  # noqa: E402

RESULTS = ROOT / "results" / "fiorillo_v0_4"
RECORDS = ROOT / "data" / "processed" / "session8"
RAW = ROOT / "data" / "raw" / "evidence_inference"
PQA_DIR = ROOT / "data" / "raw" / "public_s5"
PQA_FILES = (("pqa_labeled", "pqa_labeled.parquet"), ("pqa_unlabeled", "pqa_unlabeled.parquet"),
             ("pqa_artificial", "pqa_artificial.parquet"))
TRAINED = (("data/processed/session7/pubmedqa_phase1_train.jsonl", "session 7, phase 1"),
           ("data/processed/session7/labeler_pqa_u.jsonl", "session 7, phase 2"),
           ("data/processed/public/pubmedqa_answer_train.jsonl", "session 4's PubMedQA training records, read by phase 3"))
CACHE = ROOT / "models" / "fiorillo_v0_4" / "contamination_cache"
OUT = RESULTS / "contamination_checks.json"
SPLITS = ("validation", "test")
SYSTEMS = ("s8_ei", "gemma4_31b")
NAMES = ("accuracy", "macro_f1")
MIN_CHARS = 60
MIN_SHARED = 2
MIN_TITLE = 20
BATCH = 5000
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
LABEL_RE = re.compile(r"(?m)^[ \t]*[A-Za-z][A-Za-z .&/()-]{0,40}:[ \t]*")
NONWORD_RE = re.compile(r"[^a-z0-9]+")
TITLE_RE = re.compile(r"\s*TITLE:\s*(.*?)\s*(?:\n\s*\n|ABSTRACT|$)", re.S)
ID_RE = re.compile(r'\{"id":\s*"pubmedqa-(\d+)"')


# ---------------------------------------------------------------- text

def norm(text):
    return NONWORD_RE.sub(" ", text.lower()).strip()


def sentence_hashes(text):
    """The hashes of the text's normalized sentences of MIN_CHARS characters or more, section labels removed."""
    out = set()
    for piece in SENT_RE.split(LABEL_RE.sub(" ", text)):
        n = norm(piece)
        if len(n) >= MIN_CHARS:
            out.add(int.from_bytes(hashlib.blake2b(n.encode("utf-8"), digest_size=8).digest(), "big"))
    return out


def title_of(prefix_text):
    m = TITLE_RE.match(prefix_text)
    return norm(m.group(1)) if m else ""


# ---------------------------------------------------------------- data

def read_records(split):
    ids, pmcids, labels = [], [], []
    with open(RECORDS / f"evidence_effect_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(r["id"])
                pmcids.append(r["group"]["pmcid"])
                labels.append(int(r["label"]))
    return ids, pmcids, np.array(labels, dtype=np.int64)


def read_covered(split, ids):
    rows = {}
    with open(RECORDS / "parts" / f"selection_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"selection_{split}.jsonl has no row for {len(missing)} records")
    return [rows[i].get("covered") for i in ids]


def read_in_abstract(ids):
    """Record ID -> True if an accepted annotation marks its evidence as in the abstract, False if none does,
    None if no accepted annotation has the flag."""
    _, _, ann, _ = E.read_inputs(RAW)
    out = {}
    for i in ids:
        acc = E.accepted(ann.get(i.split("-", 1)[1], []))
        flags = [r.get("In Abstract") for r in acc if r.get("In Abstract") in ("True", "False")]
        out[i] = None if not flags else ("True" in flags)
    return out


def article_parts_of(pmcid):
    prefix_text, _ = E.article_parts(E.article_text(RAW, pmcid))
    return prefix_text


# ---------------------------------------------------------------- 1. reading the evidence

def subset_scores(labels, probs, pmcids, mask):
    idx = np.flatnonzero(mask)
    out = {"prompts": int(len(idx)), "articles": int(len(set(pmcids[j] for j in idx))),
           "labels": {a: int(np.sum(labels[idx] == k)) for k, a in enumerate(S.ANSWERS)}}
    if len(idx) == 0:
        return out
    y, p = labels[idx], probs[idx]
    rows_of = S.article_rows([pmcids[j] for j in idx])
    est, low, high = S.cluster_bootstrap(lambda r: S.scores(NAMES, y[r], p[r]), rows_of, len(idx),
                                         resamples=S.RESAMPLES, seed=S.SEED)
    for k, m in enumerate(NAMES):
        out[m] = {"estimate": float(est[k]), "low": float(low[k]), "high": float(high[k])}
    return out


def difference(labels, probs, pmcids, mask_a, mask_b):
    """Accuracy on mask_a's prompts minus accuracy on mask_b's, with a 95 percent percentile interval from resamples
    of the split's articles (correction 15); a resample without a prompt in either group is left out."""
    correct = (probs.argmax(1) == labels).astype(float)
    rows_of = S.article_rows(pmcids)

    def stat(r):
        a, b = mask_a[r], mask_b[r]
        if not a.any() or not b.any():
            return np.array([np.nan])
        return np.array([correct[r][a].mean() - correct[r][b].mean()])

    est, low, high = S.cluster_bootstrap(stat, rows_of, len(labels), resamples=S.RESAMPLES, seed=S.SEED)
    return {"estimate": float(est[0]), "low": float(low[0]), "high": float(high[0]),
            "prompts": [int(mask_a.sum()), int(mask_b.sum())]}


def reading_checks(data):
    out = {}
    for split in SPLITS:
        d = data[split]
        cov = d["covered"]
        ab = d["in_abstract"]
        masks = {
            "all": np.ones(len(d["ids"]), dtype=bool),
            "evidence inside the input": np.array([c is True for c in cov]),
            "evidence outside the input": np.array([c is False for c in cov]),
            "no usable evidence offsets": np.array([c is None for c in cov]),
            "evidence in the abstract": np.array([ab[i] is True for i in d["ids"]]),
            "evidence not in the abstract": np.array([ab[i] is False for i in d["ids"]]),
            "outside the input and not in the abstract": np.array([(c is False) and (ab[i] is False)
                                                                  for c, i in zip(cov, d["ids"])]),
        }
        out[split] = {system: {name: subset_scores(d["labels"], d["probs"][system], d["pmcids"], m)
                               for name, m in masks.items()} for system in SYSTEMS}
    return out


def reading_differences(data):
    out = {}
    for split in SPLITS:
        d = data[split]
        outside = np.array([c is False for c in d["covered"]])
        inside = np.array([c is True for c in d["covered"]])
        out[split] = {system: {"outside minus inside": difference(d["labels"], d["probs"][system], d["pmcids"],
                                                                  outside, inside)} for system in SYSTEMS}
    return out


# ---------------------------------------------------------------- 2. PubMedQA

def pubmedqa_matches(articles, deadline):
    """[{pmcid, source, pubid, shared, ei_sentences}] for every article sharing MIN_SHARED or more sentences
    with one PubMedQA abstract; each PubMedQA file's result is kept in CACHE."""
    import pyarrow.parquet as pq
    index = collections.defaultdict(set)
    for pmcid, hashes in articles.items():
        for h in hashes:
            index[h].add(pmcid)
    key = hashlib.sha256(json.dumps(sorted((k, sorted(v)) for k, v in articles.items())).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    found, files = [], {}
    for source, name in PQA_FILES:
        cache = CACHE / f"pubmedqa_{source}_{key}.json"
        if cache.exists():
            got = json.loads(cache.read_text(encoding="utf-8"))
        else:
            if time.time() > deadline:
                return None, files
            pf = pq.ParquetFile(PQA_DIR / name)
            got = {"rows": pf.metadata.num_rows, "matches": []}
            for batch in pf.iter_batches(batch_size=BATCH, columns=["pubid", "context"]):
                cols = batch.to_pydict()
                for pubid, ctx in zip(cols["pubid"], cols["context"]):
                    counts = collections.Counter()
                    for h in sentence_hashes(" ".join(ctx["contexts"])):
                        for pmcid in index.get(h, ()):
                            counts[pmcid] += 1
                    for pmcid, n in counts.items():
                        if n >= MIN_SHARED:
                            got["matches"].append({"pmcid": pmcid, "pubid": int(pubid), "shared": n})
            tmp = cache.with_name(cache.name + ".tmp")
            tmp.write_text(json.dumps(got) + "\n", encoding="utf-8")
            tmp.replace(cache)
        files[source] = got["rows"]
        for m in got["matches"]:
            found.append(dict(m, source=source, ei_sentences=len(articles[m["pmcid"]])))
    return found, files


def trained_on(pubids):
    """{file: sorted pubids found in it} for the files arm B's PubMedQA training items came from."""
    out = {}
    for rel, what in TRAINED:
        hit = set()
        with open(ROOT / rel, encoding="utf-8") as f:
            for line in f:
                m = ID_RE.match(line, 0, 64)
                if m and int(m.group(1)) in pubids:
                    hit.add(int(m.group(1)))
        out[rel] = {"what": what, "pubids_found": sorted(hit)}
    return out


# ---------------------------------------------------------------- 3. EI's own splits

def split_duplicates(prefix, split_of):
    train = [a for a, s in split_of.items() if s == "train"]
    titles = collections.defaultdict(set)
    index = collections.defaultdict(set)
    for a in train:
        t = title_of(prefix[a])
        if len(t) >= MIN_TITLE:
            titles[t].add(a)
        for h in sentence_hashes(prefix[a]):
            index[h].add(a)
    out = []
    for a, s in sorted(split_of.items()):
        if s == "train":
            continue
        t = title_of(prefix[a])
        same_title = sorted(titles.get(t, ())) if len(t) >= MIN_TITLE else []
        counts = collections.Counter()
        for h in sentence_hashes(prefix[a]):
            for b in index.get(h, ()):
                counts[b] += 1
        shared = {b: n for b, n in counts.items() if n >= MIN_SHARED}
        if same_title or shared:
            out.append({"pmcid": a, "split": s, "same_title_in_train": same_title,
                        "train_articles_sharing_sentences": dict(sorted(shared.items()))})
    return out


# ---------------------------------------------------------------- main

def accuracy_apart(data, split, flagged, with_difference=False):
    d = data[split]
    mask = np.array([p in flagged for p in d["pmcids"]])
    out = {"flagged": subset_scores(d["labels"], d["probs"]["s8_ei"], d["pmcids"], mask),
           "the rest": subset_scores(d["labels"], d["probs"]["s8_ei"], d["pmcids"], ~mask)}
    if with_difference:
        out["flagged minus the rest"] = difference(d["labels"], d["probs"]["s8_ei"], d["pmcids"], mask, ~mask)
    return out


def run(max_seconds=None):
    t0 = time.time()
    deadline = t0 + max_seconds if max_seconds else float("inf")
    data = {}
    all_ids = []
    for split in SPLITS:
        ids, pmcids, labels = read_records(split)
        probs = {}
        for system in SYSTEMS:
            probs[system] = S.load_aligned(RESULTS / "predictions" / f"{system}_evidence_effect_{split}.csv", ids, labels)
        data[split] = {"ids": ids, "pmcids": pmcids, "labels": labels, "probs": probs,
                       "covered": read_covered(split, ids)}
        all_ids += ids
    in_abstract = read_in_abstract(all_ids)
    for split in SPLITS:
        data[split]["in_abstract"] = in_abstract

    # the builder's own coverage, recounted, must match article_selection.json
    selection = json.loads((RESULTS / "article_selection.json").read_text(encoding="utf-8"))
    cov = data["validation"]["covered"]
    usable = [c for c in cov if c is not None]
    recount = {"prompts_with_usable_offsets": len(usable), "covered": int(sum(usable))}
    want = selection["coverage"]["validation"]
    if (recount["prompts_with_usable_offsets"], recount["covered"]) != (want["prompts_with_usable_offsets"], want["covered"]):
        raise SystemExit(f"validation coverage recount {recount} differs from article_selection.json's {want}")
    test_usable = [c for c in data["test"]["covered"] if c is not None]
    coverage_test = {"prompts_with_usable_offsets": len(test_usable), "covered": int(sum(test_usable)),
                     "coverage": sum(test_usable) / len(test_usable) if test_usable else None}

    reading = reading_checks(data)
    differences = reading_differences(data)

    # the articles' titles and abstracts
    split_of = {}
    for line in (RAW / "annotations/splits/train_article_ids.txt").read_text(encoding="utf-8").split():
        split_of[line] = "train"
    for split in SPLITS:
        for a in set(data[split]["pmcids"]):
            split_of[a] = split
    train_used = set()
    with open(RECORDS / "evidence_effect_train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_used.add(json.loads(line)["group"]["pmcid"])
    split_of = {a: s for a, s in split_of.items() if s != "train" or a in train_used}
    prefix = {a: article_parts_of(a) for a in split_of}
    evaluated = {a: sentence_hashes(prefix[a]) for a, s in split_of.items() if s in SPLITS}

    matches, files = pubmedqa_matches(evaluated, deadline)
    if matches is None:
        print(json.dumps({"finished": False, "pubmedqa_files_done": sorted(files), "seconds": round(time.time() - t0, 1)}))
        return False
    pubids = {m["pubid"] for m in matches}
    trained = trained_on(pubids)
    trained_any = set().union(*[set(v["pubids_found"]) for v in trained.values()])
    matched = {s: sorted({m["pmcid"] for m in matches if split_of[m["pmcid"]] == s}) for s in SPLITS}
    matched_trained = {s: sorted({m["pmcid"] for m in matches if split_of[m["pmcid"]] == s and m["pubid"] in trained_any})
                       for s in SPLITS}

    dups = split_duplicates(prefix, split_of)
    dup_articles = {s: sorted({d["pmcid"] for d in dups if d["split"] == s}) for s in SPLITS}

    out = {
        "plan": "docs/prompt_session8.md, CORRECTIONS 14 and 15 (descriptive, not pre-registered)",
        "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {"resamples": S.RESAMPLES, "seed": S.SEED, "interval": "95 percent, percentile, articles resampled",
                     "sentence_min_chars": MIN_CHARS, "min_shared_sentences": MIN_SHARED, "title_min_chars": MIN_TITLE,
                     "predictions": "calibrated (s8_ei and gemma4_31b as compare_session8.py reads them)"},
        "reading_the_evidence": {
            "coverage_recount_validation": recount,
            "coverage_test": coverage_test,
            "by_subset": reading,
            "differences": differences,
        },
        "pubmedqa": {
            "files": {s: {"rows": n} for s, n in files.items()},
            "evaluated_articles": {s: len({a for a in evaluated if split_of[a] == s}) for s in SPLITS},
            "evaluated_articles_without_long_sentences": sum(1 for h in evaluated.values() if not h),
            "matches": sorted(matches, key=lambda m: (m["pmcid"], m["source"], m["pubid"])),
            "trained_files": trained,
            "matched_articles": matched,
            "matched_articles_whose_abstract_arm_b_trained_on": matched_trained,
            "s8_ei_test_matched_or_not": accuracy_apart(data, "test", set(matched["test"]), with_difference=True),
            "s8_ei_validation_matched_or_not": accuracy_apart(data, "validation", set(matched["validation"]),
                                                              with_difference=True),
        },
        "ei_splits": {
            "train_articles": sum(1 for s in split_of.values() if s == "train"),
            "flagged": dups,
            "flagged_articles": dup_articles,
            "s8_ei_test_flagged_or_not": accuracy_apart(data, "test", set(dup_articles["test"])),
        },
        "seconds": round(time.time() - t0, 1),
    }
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps({"status": "written", "path": str(OUT.relative_to(ROOT)), "seconds": out["seconds"]}))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args(argv)
    run(a.max_seconds)


if __name__ == "__main__":
    main()

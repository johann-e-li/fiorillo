"""Session 9's Evidence Inference 2.0 (EI) records (docs/prompt_session9.md: EI'S LICENSE-CLEAN RECORDS, THE RECIPES
(swap), THE LONGER INPUT, THE READING CHECKS, and corrections 2 and 3).

Inputs: session 8's records in data/processed/session8/ (evidence_effect_{train,validation,test}.jsonl, each checked
against its SHA-256 in results/fiorillo_v0_4/records_manifest.json before it is read); EI's raw files at the fixed commit
(data/raw/evidence_inference/); session 8's BM25 statistics (data/processed/session8/bm25_stats.json, unchanged); and
results/fiorillo_v0_5/license_recheck.csv (an article is clean when its clean column is true).

The files (data/processed/session9/):
- clean/: evidence_effect_train.jsonl and evidence_effect_validation.jsonl, session 8's records of clean articles, in
  the same order, byte for byte; evidence_effect_validation_monitor.jsonl, 200 of the clean validation records drawn
  with random.Random(20260927).sample over their IDs sorted as text, written in that order.
- swap/evidence_effect_train.jsonl: each clean training record followed by its swapped copy (ID <id>-swap): the
  intervention and the comparator exchanged in the state; the answers "significantly increased" (0) and "significantly
  decreased" (1) exchanged in the label, the target and the group's accepted_labels; "no significant difference" (2)
  kept. A prompt whose intervention and comparator are the same text, after whitespace is collapsed and case ignored,
  gets no swapped copy.
- long/: article_selection_long.jsonl (every prompt of every split), and the records at 6,144 tokens (session 8's
  selection rule with that budget and 16 to spare): evidence_effect_train.jsonl (clean), evidence_effect_validation.jsonl
  (all 1,237), evidence_effect_validation_monitor.jsonl (the clean monitoring subset's 200 IDs) and
  evidence_effect_test.jsonl (all 1,218); coverage_long.json in results/fiorillo_v0_5/.
- checks/: none/, title/, swap/ and swap_long/, each with evidence_effect_test.jsonl (the reading checks' inputs; IDs and
  groups kept): none, the article set to the empty text; title, the article set to the article's title (the text after
  the label "TITLE:", on its line or the next, up to the first blank line after it or the first line that begins, after
  its leading spaces, with "ABSTRACT" or "BODY", whitespace collapsed (correction 6); prompts whose article gives no
  title left out); swap and swap_long, the intervention and the comparator exchanged in the state of
  the 4,096-token and 6,144-token records, the answer exchanged as above, prompts whose intervention and comparator are
  the same text left out. For correction 3, checks/swap/ and checks/swap_long/ also hold evidence_effect_validation.jsonl,
  the same exchange of all 1,237 validation records (less those with the same text), calibrate kept.

- combo/ (the pod, only if THE RECIPE RULE calls for a combination): evidence_effect_train.jsonl, the clean training
  records at the combination's length (long's if long is one of its parts, else clean's), each followed by its swapped
  copy if swap is one of its parts; evidence_effect_validation_monitor.jsonl, the monitoring subset at that length; and
  combo.json (the parts, the counts and the files' SHA-256).

Subcommands (the linked shell, from the project folder, with python3):
  clean | swap | long [--max-seconds 150] | checks | manifest
  combo --parts swap long lr1   (the pod; the learning rate is not a records change and is ignored here)
  rebuild   (the pod) every file above from session 8's records, the raw files, license_recheck.csv and
            article_selection_long.jsonl alone, with no tokenizer; writes each file's SHA-256 against
            results/fiorillo_v0_5/records_manifest.json to --check-out, and fails if any differs.
Only the builder reads the test split's labels, to write and count them.
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "train"))
import evidence_inference as ei8  # noqa: E402
from records import one_hot, sha256_file  # noqa: E402

S8_DIR = ROOT / "data" / "processed" / "session8"
S8_MANIFEST = ROOT / "results" / "fiorillo_v0_4" / "records_manifest.json"
OUT = ROOT / "data" / "processed" / "session9"
RESULTS = ROOT / "results" / "fiorillo_v0_5"
LICENSES = RESULTS / "license_recheck.csv"
SPLITS = ("train", "validation", "test")
SEED = 20260927
MONITOR = 200
LONG_LEN = 6144
CHECK_LONG = 0.9891  # the check of 30 September at 6,144 tokens (docs/prompt_session8.md, THE ARTICLE SELECTION)
CHECK_LONG_MEAN_TOKENS = 5109
EXPECTED = {"train": (5279, 1431), "validation": (654, 179)}  # (prompts, articles), DECISIONS 3
SWAP_LABEL = {0: 1, 1: 0, 2: 2}
NAMES = {"clean": ["clean/evidence_effect_train.jsonl", "clean/evidence_effect_validation.jsonl",
                   "clean/evidence_effect_validation_monitor.jsonl"],
         "swap": ["swap/evidence_effect_train.jsonl"],
         "long": ["long/article_selection_long.jsonl", "long/evidence_effect_train.jsonl", "long/evidence_effect_validation.jsonl",
                  "long/evidence_effect_validation_monitor.jsonl", "long/evidence_effect_test.jsonl"],
         "checks": ["checks/none/evidence_effect_test.jsonl", "checks/title/evidence_effect_test.jsonl",
                    "checks/swap/evidence_effect_test.jsonl", "checks/swap/evidence_effect_validation.jsonl",
                    "checks/swap_long/evidence_effect_test.jsonl", "checks/swap_long/evidence_effect_validation.jsonl"]}
ALL_NAMES = [n for k in ("clean", "swap", "long", "checks") for n in NAMES[k]]


def now():
    return ei8.now()


# ---------------------------------------------------------------- inputs

def s8_lines(split, s8_dir=S8_DIR, manifest=S8_MANIFEST, check=True):
    """Session 8's record file for a split, as its lines (str, each ending in a newline), checked against its SHA-256."""
    path = Path(s8_dir) / f"evidence_effect_{split}.jsonl"
    if check:
        want = ei8.load_json(manifest, {}).get("sha256", {}).get("records", {}).get(path.name)
        got = sha256_file(path)
        if want != got:
            raise SystemExit(f"{path} has SHA-256 {got}, not session 8's {want}")
    with open(path, encoding="utf-8", newline="") as f:
        return [line for line in f if line.strip()]


def clean_pmcids(path=LICENSES):
    """The PMCIDs (as in the CSV, 'PMC<digits>') whose clean column is true."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["pmcid"] for r in csv.DictReader(f) if r["clean"] == "true"}


def pmc(rec):
    p = str(rec["group"]["pmcid"])
    return p if p.startswith("PMC") else "PMC" + p


def write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    tmp.replace(path)
    return path


def dump(rec):
    return json.dumps(rec, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- clean

def filter_clean(lines, clean):
    """The lines whose record's article is clean, in order, unchanged."""
    return [line for line in lines if pmc(json.loads(line)) in clean]


def monitor_lines(lines, n=MONITOR, seed=SEED):
    """n lines drawn with random.Random(seed).sample over the records' IDs sorted as text; written in ID order."""
    by_id = {json.loads(line)["id"]: line for line in lines}
    ids = sorted(by_id)
    pick = sorted(random.Random(seed).sample(ids, n))
    return [by_id[i] for i in pick]


def build_clean(s8_dir=S8_DIR, out=OUT, licenses=LICENSES, manifest=S8_MANIFEST):
    clean = clean_pmcids(licenses)
    tr = filter_clean(s8_lines("train", s8_dir, manifest), clean)
    va = filter_clean(s8_lines("validation", s8_dir, manifest), clean)
    for line in va:
        if json.loads(line).get("calibrate") is not True:
            raise SystemExit("a clean validation record lacks calibrate true")
    write_lines(Path(out) / "clean" / "evidence_effect_train.jsonl", tr)
    write_lines(Path(out) / "clean" / "evidence_effect_validation.jsonl", va)
    write_lines(Path(out) / "clean" / "evidence_effect_validation_monitor.jsonl", monitor_lines(va))
    return {"train": len(tr), "validation": len(va)}


# ---------------------------------------------------------------- swap

def same_text(a, b):
    return " ".join(a.split()).lower() == " ".join(b.split()).lower()


def swapped(rec, suffix="-swap"):
    """The record with the intervention and the comparator exchanged and the answer exchanged (0 and 1; 2 kept), or None
    when the two are the same text."""
    st = rec["state"]
    if same_text(st["intervention"], st["comparator"]):
        return None
    out = copy.deepcopy(rec)
    out["id"] = rec["id"] + suffix
    out["state"]["intervention"], out["state"]["comparator"] = st["comparator"], st["intervention"]
    out["label"] = SWAP_LABEL[rec["label"]]
    out["target"] = one_hot(out["label"], 3)
    if "group" in out and "accepted_labels" in out["group"]:
        out["group"]["accepted_labels"] = sorted(SWAP_LABEL[x] for x in rec["group"]["accepted_labels"])
    return out


def swap_training_lines(lines):
    """Each record, then its swapped copy when it has one; and the number left without one."""
    out, same = [], 0
    for line in lines:
        out.append(line)
        s = swapped(json.loads(line))
        if s is None:
            same += 1
        else:
            out.append(dump(s))
    return out, same


def swap_check_lines(lines):
    """The exchanged records with their IDs kept (the reading checks); prompts with the same text left out."""
    out, same = [], 0
    for line in lines:
        s = swapped(json.loads(line), suffix="")
        if s is None:
            same += 1
        else:
            out.append(dump(s))
    return out, same


def build_swap(out=OUT):
    lines = read_lines(Path(out) / "clean" / "evidence_effect_train.jsonl")
    got, same = swap_training_lines(lines)
    write_lines(Path(out) / "swap" / "evidence_effect_train.jsonl", got)
    return {"records": len(got), "clean": len(lines), "same_text": same}


def read_lines(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [line for line in f if line.strip()]


# ---------------------------------------------------------------- long (needs Qwen's tokenizer)

def long_parts(out):
    return Path(out) / "parts"


def select_long(raw=ei8.RAW, out=OUT, tokenizer_dir=ei8.QWEN_DIR, max_seconds=150.0, s8_dir=S8_DIR):
    """Session 8's selection at 6,144 tokens for every prompt of every split, resuming from parts/selection_long_<split>.jsonl.
    Returns True when every prompt is done."""
    from transformers import AutoTokenizer
    t0 = time.time()
    kept, _, _ = ei8.select_prompts(raw)
    stats = ei8.load_json(Path(s8_dir) / "bm25_stats.json", None)
    if stats is None:
        raise SystemExit("session 8's bm25_stats.json is missing")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    budget = ei8.Budget(tok, max_len=LONG_LEN)
    parts = long_parts(out)
    parts.mkdir(parents=True, exist_ok=True)
    articles_now = 0
    for split in SPLITS:
        part = parts / f"selection_long_{split}.jsonl"
        done = set()
        if part.exists():
            for line in open(part, encoding="utf-8"):
                if line.strip():
                    done.add(json.loads(line)["id"])
        by_article = collections.defaultdict(list)
        for p in kept[split]:
            if f"evidence-{p['pid']}" not in done:
                by_article[p["pmcid"]].append(p)
        with open(part, "a", encoding="utf-8") as f:
            for pmcid in sorted(by_article):
                if articles_now and time.time() - t0 > max_seconds:
                    print(json.dumps({"finished": False, "split": split, "seconds": round(time.time() - t0, 1)}))
                    return False
                articles_now += 1
                prefix_text, passages = ei8.article_parts(ei8.article_text(raw, pmcid))
                enc = tok([json.dumps(p[2], ensure_ascii=False)[1:-1] for p in passages] or [""], add_special_tokens=False)["input_ids"]
                piece_tokens = [len(x) for x in enc][: len(passages)]
                prefix_tokens = budget.count_escaped(prefix_text)
                rows = []
                for p in by_article[pmcid]:
                    got = ei8.select_one(p, prefix_text, passages, piece_tokens, prefix_tokens, stats, budget)
                    row = {"id": f"evidence-{p['pid']}", "pmcid": pmcid, "split": split}
                    if got is None:
                        row["skip"] = "the prompt alone does not fit"
                    else:
                        row.update({"prefix_chars": got[0], "passages": got[1], "tokens": got[2],
                                    "covered": ei8.coverage_of(p, got[0], got[1], passages) if p["spans"] else None})
                    rows.append(json.dumps(row) + "\n")
                f.writelines(rows)
                f.flush()
    print(json.dumps({"finished": True, "seconds": round(time.time() - t0, 1)}))
    return True


def read_long_parts(out=OUT):
    rows = {}
    for split in SPLITS:
        for line in open(long_parts(out) / f"selection_long_{split}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def selection_lines(kept, selection):
    """article_selection_long.jsonl's lines, in session 8's format and order."""
    out = []
    for split in SPLITS:
        for p in kept[split]:
            r = selection[f"evidence-{p['pid']}"]
            out.append(json.dumps({k: r[k] for k in ("id", "pmcid", "split", "prefix_chars", "passages") if k in r} |
                                  ({"skip": r["skip"]} if "skip" in r else {})) + "\n")
    return out


def long_records(raw, kept, selection, clean, monitor_ids):
    """The four long files' lines from the raw texts and the selection alone (no tokenizer)."""
    recs, skipped = ei8.records_from(raw, kept, selection)
    train = [dump(r) for r in recs["train"] if pmc(r) in clean]
    validation = [dump(r) for r in recs["validation"]]
    monitor = sorted((r for r in recs["validation"] if r["id"] in monitor_ids), key=lambda r: r["id"])
    if len(monitor) != len(monitor_ids):
        raise SystemExit("the long validation records lack some monitoring IDs")
    test = [dump(r) for r in recs["test"]]
    return {"long/evidence_effect_train.jsonl": train, "long/evidence_effect_validation.jsonl": validation,
            "long/evidence_effect_validation_monitor.jsonl": [dump(r) for r in monitor],
            "long/evidence_effect_test.jsonl": test}, skipped


def percentile(values, q):
    return ei8.percentile(values, q)


def coverage_table(kept, selection, clean):
    def cov(prompts):
        flags = [selection[f"evidence-{p['pid']}"].get("covered") for p in prompts]
        usable = [c for c in flags if c is not None]
        toks = [selection[f"evidence-{p['pid']}"]["tokens"] for p in prompts if "tokens" in selection[f"evidence-{p['pid']}"]]
        return {"prompts": len(prompts), "prompts_with_usable_offsets": len(usable), "covered": int(sum(usable)),
                "coverage": sum(usable) / len(usable) if usable else None,
                "mean_tokens": sum(toks) / len(toks) if toks else None}
    table = {"train": cov(kept["train"]), "validation": cov(kept["validation"]),
             "train_clean": cov([p for p in kept["train"] if "PMC" + str(p["pmcid"]) in clean])}
    diff = table["train"]["coverage"] - CHECK_LONG
    return {"plan": "docs/prompt_session9.md, THE LONGER INPUT", "written_utc": now(), "max_len": LONG_LEN, "spare": ei8.SPARE,
            "rule": "session 8's THE ARTICLE SELECTION (title and abstract, then results-section passages, then others, by BM25 "
                    "against the prompt's words) with a budget of 6,144 tokens; BM25's statistics are session 8's bm25_stats.json",
            "coverage": table,
            "check_of_30_september_train": {"coverage": CHECK_LONG, "mean_tokens": CHECK_LONG_MEAN_TOKENS},
            "difference_from_check_train": diff, "within_one_point": abs(diff) <= 0.01,
            "at_4096_session8": ei8.load_json(ROOT / "results" / "fiorillo_v0_4" / "article_selection.json", {}).get("coverage")}


def build_long(raw=ei8.RAW, out=OUT, results=RESULTS, tokenizer_dir=ei8.QWEN_DIR, max_seconds=150.0, licenses=LICENSES):
    if not select_long(raw, out, tokenizer_dir, max_seconds):
        return {"finished": False}
    kept, _, _ = ei8.select_prompts(raw)
    selection = read_long_parts(out)
    missing = [p for s in SPLITS for p in kept[s] if f"evidence-{p['pid']}" not in selection]
    if missing:
        raise SystemExit(f"{len(missing)} prompts have no long selection")
    write_lines(Path(out) / "long" / "article_selection_long.jsonl", selection_lines(kept, selection))
    clean = clean_pmcids(licenses)
    monitor_ids = {json.loads(line)["id"] for line in read_lines(Path(out) / "clean" / "evidence_effect_validation_monitor.jsonl")}
    files, skipped = long_records(raw, kept, selection, clean, monitor_ids)
    for name, lines in files.items():
        write_lines(Path(out) / name, lines)
    table = coverage_table(kept, selection, clean)
    table["skipped_by_the_selection"] = skipped
    ei8.save_json(Path(results) / "coverage_long.json", table)
    print(json.dumps({"finished": True, "coverage_train": table["coverage"]["train"]["coverage"],
                      "within_one_point": table["within_one_point"], "skipped": skipped}))
    return {"finished": True, "coverage": table}


# ---------------------------------------------------------------- the reading checks' inputs

def title_of(text):
    """The article's title (correction 6): the text after the label 'TITLE:', on its line or the next, up to the first
    blank line after it or the first line that begins, after its leading spaces, with 'ABSTRACT' or 'BODY'; whitespace
    collapsed. The empty text when no line begins with 'TITLE:' or the title is empty. (EI's files write 'TITLE: <title>'
    on one line in most articles and 'TITLE: ' alone in others, indent some 'ABSTRACT' lines, and in 106 of the 3,330
    articles have no 'ABSTRACT' line between the title and the abstract's text, which a blank line separates.)"""
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if line.lstrip().startswith("TITLE:"):
            got = [line.lstrip()[len("TITLE:"):]]
            started = bool(got[0].strip())
            for nxt in lines[k + 1:]:
                head = nxt.lstrip()
                if head.startswith("ABSTRACT") or head.startswith("BODY"):
                    break
                if not nxt.strip():
                    if started:
                        break
                    continue
                started = True
                got.append(nxt)
            return " ".join(" ".join(got).split())
    return ""


def with_article(line, article):
    rec = json.loads(line)
    rec["state"]["article"] = article
    return dump(rec)


def check_files(raw, test4096, validation4096, test6144, validation6144):
    """The checks' files from the record lines; returns ({name: lines}, counts)."""
    none = [with_article(line, "") for line in test4096]
    title, no_title, cache = [], 0, {}
    for line in test4096:
        rec = json.loads(line)
        p = str(rec["group"]["pmcid"]).removeprefix("PMC")
        if p not in cache:
            cache[p] = title_of(ei8.article_text(raw, p))
        if cache[p]:
            title.append(with_article(line, cache[p]))
        else:
            no_title += 1
    s_test, same_test = swap_check_lines(test4096)
    s_val, same_val = swap_check_lines(validation4096)
    l_test, same_ltest = swap_check_lines(test6144)
    l_val, same_lval = swap_check_lines(validation6144)
    files = {"checks/none/evidence_effect_test.jsonl": none, "checks/title/evidence_effect_test.jsonl": title,
             "checks/swap/evidence_effect_test.jsonl": s_test, "checks/swap/evidence_effect_validation.jsonl": s_val,
             "checks/swap_long/evidence_effect_test.jsonl": l_test, "checks/swap_long/evidence_effect_validation.jsonl": l_val}
    counts = {"title_left_out_no_title": no_title, "swap_left_out_same_text": {"test": same_test, "validation": same_val},
              "swap_long_left_out_same_text": {"test": same_ltest, "validation": same_lval}}
    return files, counts


def build_checks(raw=ei8.RAW, out=OUT, s8_dir=S8_DIR, manifest=S8_MANIFEST):
    files, counts = check_files(raw, s8_lines("test", s8_dir, manifest), s8_lines("validation", s8_dir, manifest),
                                read_lines(Path(out) / "long" / "evidence_effect_test.jsonl"),
                                read_lines(Path(out) / "long" / "evidence_effect_validation.jsonl"))
    for name, lines in files.items():
        write_lines(Path(out) / name, lines)
    ei8.save_json(Path(out) / "parts" / "checks_counts.json", counts)
    print(json.dumps(counts))
    return counts


# ---------------------------------------------------------------- the combination (the pod)

def build_combo(parts, out=OUT):
    """combo/'s files for a combination of passing changes (THE RECIPE RULE); the learning rate changes no record."""
    out = Path(out)
    parts = sorted(set(parts))
    base = "long" if "long" in parts else "clean"
    lines = read_lines(out / base / "evidence_effect_train.jsonl")
    same = None
    if "swap" in parts:
        lines, same = swap_training_lines(lines)
    train = write_lines(out / "combo" / "evidence_effect_train.jsonl", lines)
    monitor = write_lines(out / "combo" / "evidence_effect_validation_monitor.jsonl",
                          read_lines(out / base / "evidence_effect_validation_monitor.jsonl"))
    info = {"plan": "docs/prompt_session9.md, THE RECIPE RULE (the combination) and ROOT (combo/)", "written_utc": now(),
            "parts": parts, "records_from": base, "swapped": "swap" in parts, "same_text_without_a_copy": same,
            "counts": {"train": len(lines), "monitor": sum(1 for _ in open(monitor, encoding="utf-8"))},
            "sha256": {"combo/evidence_effect_train.jsonl": sha256_file(train),
                       "combo/evidence_effect_validation_monitor.jsonl": sha256_file(monitor)}}
    ei8.save_json(out / "combo" / "combo.json", info)
    print(json.dumps(info))
    return info


# ---------------------------------------------------------------- the manifest

def count_file(path):
    lines = read_lines(path)
    recs = [json.loads(line) for line in lines]
    if path.name == "article_selection_long.jsonl":
        return {"rows": len(recs), "skips": sum(1 for r in recs if "skip" in r)}
    return {"records": len(recs), "articles": len({pmc(r) for r in recs}),
            "answers": dict(sorted(collections.Counter(str(r["label"]) for r in recs).items()))}


def tokens_of(ids, selection_rows):
    t = [selection_rows[i]["tokens"] for i in ids if "tokens" in selection_rows.get(i, {})]
    return {"mean": sum(t) / len(t), "p90": percentile(t, 0.9), "largest": max(t), "n": len(t)} if t else None


def folder_digests(folder):
    folder = Path(folder)
    return {p.name: sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file()} if folder.exists() else None


def build_manifest(out=OUT, results=RESULTS, licenses=LICENSES, s8_dir=S8_DIR):
    out = Path(out)
    files = {name: out / name for name in ALL_NAMES}
    for name, p in files.items():
        if not p.exists():
            raise SystemExit(f"{p} is missing; run the subcommand that writes it")
    counts = {name: count_file(p) for name, p in files.items()}
    for split, (n, a) in EXPECTED.items():
        got = counts[f"clean/evidence_effect_{split}.jsonl"]
        if (got["records"], got["articles"]) != (n, a):
            print(json.dumps({"WARNING": f"clean {split}: {got['records']} records in {got['articles']} articles, the plan "
                                         f"expects {n} in {a}; append a correction before step 3"}))
    s8_parts = {}
    for split in SPLITS:
        part = Path(s8_dir) / "parts" / f"selection_{split}.jsonl"
        if part.exists():
            for line in open(part, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    s8_parts[r["id"]] = r
    long_rows = read_long_parts(out)
    ids = {name: [json.loads(line)["id"].removesuffix("-swap") for line in read_lines(p)] for name, p in files.items()
           if name != "long/article_selection_long.jsonl"}
    tokens = {}
    for name in ("clean/evidence_effect_train.jsonl", "clean/evidence_effect_validation.jsonl",
                 "clean/evidence_effect_validation_monitor.jsonl", "swap/evidence_effect_train.jsonl"):
        tokens[name] = tokens_of(ids[name], s8_parts)
    for name in ("long/evidence_effect_train.jsonl", "long/evidence_effect_validation.jsonl",
                 "long/evidence_effect_validation_monitor.jsonl", "long/evidence_effect_test.jsonl"):
        tokens[name] = tokens_of(ids[name], long_rows)
    checks_counts = ei8.load_json(out / "parts" / "checks_counts.json", {})
    swap = counts["swap/evidence_effect_train.jsonl"]["records"] - counts["clean/evidence_effect_train.jsonl"]["records"]
    cov = ei8.load_json(Path(results) / "coverage_long.json", {})
    s8m = ei8.load_json(S8_MANIFEST, {})
    manifest = {
        "plan": "docs/prompt_session9.md, EI'S LICENSE-CLEAN RECORDS, THE RECIPES, THE LONGER INPUT, THE READING CHECKS and correction 3",
        "written_utc": now(), "commit": ei8.COMMIT,
        "rule": "an article is clean when results/fiorillo_v0_5/license_recheck.csv's clean column is true",
        "expected": {"train": {"prompts": 5279, "articles": 1431}, "validation": {"prompts": 654, "articles": 179},
                     "test": {"prompts": 1218}},
        "counts": counts,
        "monitor": {"prompts": MONITOR, "seed": SEED, "drawn_from": "clean/evidence_effect_validation.jsonl, IDs sorted as text"},
        "swap": {"swapped_copies": swap,
                 "clean_training_prompts_without_a_copy_same_text": counts["clean/evidence_effect_train.jsonl"]["records"] - swap},
        "checks": checks_counts,
        "long": {"max_len": LONG_LEN, "coverage_train": cov.get("coverage", {}).get("train", {}).get("coverage"),
                 "within_one_point_of_the_check": cov.get("within_one_point"),
                 "skipped_by_the_selection": cov.get("skipped_by_the_selection")},
        "tokens_per_record": tokens,
        "sha256": {"files": {name: sha256_file(p) for name, p in files.items()},
                   "inputs": {"license_recheck.csv": sha256_file(licenses),
                              "session8_records": s8m.get("sha256", {}).get("records"),
                              "article_selection.jsonl": s8m.get("sha256", {}).get("article_selection.jsonl"),
                              "bm25_stats.json": s8m.get("sha256", {}).get("bm25_stats.json"),
                              "raw": s8m.get("sha256", {}).get("raw")}},
        "models": {"s7_b3": folder_digests(ROOT / "models" / "fiorillo_v0_3" / "arm_b"),
                   "s7_b3_adapter_expected": s8m.get("s7_b3_adapter_sha256"),
                   **{f"s8_ei_{k}": folder_digests(ROOT / "models" / "fiorillo_v0_4" / f"s8_ei_{k}") for k in (1, 2, 3)}},
    }
    if manifest["models"]["s7_b3"] and manifest["models"]["s7_b3"].get("adapter.safetensors") != manifest["models"]["s7_b3_adapter_expected"]:
        raise SystemExit("s7_b3's adapter differs from session 8's record")
    ei8.save_json(Path(results) / "records_manifest.json", manifest)
    print(json.dumps({"counts": {k: v for k, v in counts.items()}, "swap": manifest["swap"], "checks": checks_counts}, indent=1))
    return manifest


# ---------------------------------------------------------------- rebuild (the pod)

def rebuild(raw, s8_dir, licenses, selection_long, out, manifest_path, check_out):
    """Every session 9 record file from session 8's records, the raw files, license_recheck.csv and the long selection."""
    out = Path(out)
    s8_manifest = Path(manifest_path).parent.parent / "fiorillo_v0_4" / "records_manifest.json"
    build_clean(s8_dir, out, licenses, s8_manifest)
    build_swap(out)
    kept, _, _ = ei8.select_prompts(raw)
    selection = {}
    for line in open(selection_long, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            selection[r["id"]] = r
    write_lines(out / "long" / "article_selection_long.jsonl", selection_lines(kept, selection))
    clean = clean_pmcids(licenses)
    monitor_ids = {json.loads(line)["id"] for line in read_lines(out / "clean" / "evidence_effect_validation_monitor.jsonl")}
    files, _ = long_records(raw, kept, selection, clean, monitor_ids)
    for name, lines in files.items():
        write_lines(out / name, lines)
    files, _ = check_files(raw, s8_lines("test", s8_dir, s8_manifest), s8_lines("validation", s8_dir, s8_manifest),
                           read_lines(out / "long" / "evidence_effect_test.jsonl"),
                           read_lines(out / "long" / "evidence_effect_validation.jsonl"))
    for name, lines in files.items():
        write_lines(out / name, lines)
    expected = ei8.load_json(manifest_path, {}).get("sha256", {}).get("files", {})
    check = {"written_utc": now(), "files": {}}
    for name in ALL_NAMES:
        got = sha256_file(out / name)
        check["files"][name] = {"sha256": got, "expected": expected.get(name), "equal": got == expected.get(name)}
    check["all_equal"] = all(v["equal"] for v in check["files"].values())
    ei8.save_json(check_out, check)
    print(json.dumps({"all_equal": check["all_equal"]}))
    if not check["all_equal"]:
        raise SystemExit("the rebuilt session 9 records differ from records_manifest.json")
    return check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clean")
    sub.add_parser("swap")
    lg = sub.add_parser("long")
    lg.add_argument("--max-seconds", type=float, default=150.0)
    lg.add_argument("--tokenizer-dir", type=Path, default=ei8.QWEN_DIR)
    sub.add_parser("checks")
    sub.add_parser("manifest")
    cb = sub.add_parser("combo")
    cb.add_argument("--parts", nargs="+", required=True, choices=("swap", "long", "lr1", "lr2"))
    cb.add_argument("--out", type=Path, default=OUT)
    rb = sub.add_parser("rebuild")
    rb.add_argument("--raw", type=Path, required=True)
    rb.add_argument("--s8-dir", type=Path, required=True)
    rb.add_argument("--licenses", type=Path, required=True)
    rb.add_argument("--selection-long", type=Path, required=True)
    rb.add_argument("--out", type=Path, required=True)
    rb.add_argument("--manifest", type=Path, required=True)
    rb.add_argument("--check-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "clean":
        print(json.dumps(build_clean()))
    elif args.cmd == "swap":
        print(json.dumps(build_swap()))
    elif args.cmd == "long":
        build_long(tokenizer_dir=args.tokenizer_dir, max_seconds=args.max_seconds)
    elif args.cmd == "checks":
        build_checks()
    elif args.cmd == "manifest":
        build_manifest()
    elif args.cmd == "combo":
        build_combo(args.parts, args.out)
    else:
        rebuild(args.raw, args.s8_dir, args.licenses, args.selection_long, args.out, args.manifest, args.check_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

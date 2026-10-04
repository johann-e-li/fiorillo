"""Evidence Inference 2.0's records for session 8 (docs/prompt_session8.md: EI'S RECORDS, EI'S QUESTION, THE ARTICLE
SELECTION).

Source: github.com/jayded/evidence-inference at commit a661e8c14f973398380c8865cf2f27a535aaaf6d (MIT), texts and evidence
offsets from this one source only.

- Splits: a prompt belongs to the split whose article-ID file lists its PMCID.
- Accepted rows: Valid Label and Valid Reasoning both True, and a non-empty Annotations text. A prompt is kept when it has
  at least one. Its label is the most common Label Code among them, a tie going to the smallest code (scipy's mode, as
  EI's own code does). Drops, counted by reason: no split; no accepted row; an empty article text; in the training split
  only, the prompt IDs annotations/README.md lists as "Incorrect".
- The record: ID evidence-<PromptID>; qid evidence_effect; state {"intervention", "comparator", "outcome", "article"};
  label the answer's index (significantly increased 0, significantly decreased 1, no significant difference 2; Label Code
  1, -1 and 0); a one-hot target; "calibrate": true on validation records; group {"source": "evidence_inference_v2",
  "pmcid", "flag" (Incorrect, Questionable, Somewhat malformed or none), "accepted_labels" (the sorted distinct label
  indices of the accepted rows)}. Records are written in PromptID order (numeric).
- The article: the title and abstract (the text before the first line that begins with "BODY.") and the passages (the
  rest, split at blank lines, each stripped, empty ones dropped). A passage's section is the last line at or before its
  start that begins with "BODY" and ends with ":"; it is a results section when that line matches RESULTS_RE.
- Words: lowercase letter-and-digit words longer than one character and not in STOP. BM25 (k1 1.2, b 0.75; IDF
  ln(1 + (N - n + 0.5) / (n + 0.5))) scores each passage against the prompt's distinct words (intervention, comparator
  and outcome), with N, n and the mean passage length computed over every passage of the training split's articles that
  have a kept prompt (bm25_stats.json), for every split.
- The selection: the title and abstract first (cut from their end if they alone do not fit); then passages, results
  sections first, the highest score first, ties by position, each one that still fits; kept passages rejoined after the
  title and abstract in article order, separated by blank lines. The whole sequence (the serialized state, then the
  question and options as src/train/qwen_items.py renders them with Qwen's tokenizer) fits 4,096 tokens with 16 to spare.
- Coverage (article_selection.json): on the training split, the share of prompts with usable evidence offsets (an
  accepted row whose Evidence Start is at least 0 and whose Evidence End is at least its start) for which at least one
  such span [start, end) lies wholly inside a kept region.

Subcommands:
  download        the files at the fixed commit into data/raw/evidence_inference/ (resumes; download_progress.json)
  build           the records, article_selection.jsonl, bm25_stats.json, the monitoring subset, article_selection.json and
                  records_manifest.json (needs Qwen's tokenizer; resumes from data/processed/session8/parts/)
  from-selection  the records from the raw files and article_selection.jsonl alone, with no tokenizer; writes each
                  file's SHA-256 against records_manifest.json to --check-out, and fails if any differs
  licenses        results/fiorillo_v0_4/article_licenses.csv from Europe PMC's REST service, and its tally in
                  records_manifest.json

Usage (the linked shell, from the project folder):
    python3 src/tasks/evidence_inference.py download [--max-seconds 150]
    python3 src/tasks/evidence_inference.py build [--max-seconds 150]
    python3 src/tasks/evidence_inference.py licenses
On the pod (prepare):
    python3 src/tasks/evidence_inference.py download --raw /root/eddm_s8/raw/evidence_inference
    python3 src/tasks/evidence_inference.py from-selection --raw ... --selection ... --out-dir ... \\
        --manifest ... --check-out /root/eddm_s8/records_manifest_pod.json
"""
import argparse
import bisect
import collections
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "train"))
from records import one_hot, sha256_file, write_jsonl  # noqa: E402

COMMIT = "a661e8c14f973398380c8865cf2f27a535aaaf6d"
BASE_URL = f"https://raw.githubusercontent.com/jayded/evidence-inference/{COMMIT}/"
RAW = ROOT / "data" / "raw" / "evidence_inference"
OUT_DIR = ROOT / "data" / "processed" / "session8"
RESULTS = ROOT / "results" / "fiorillo_v0_4"
QWEN_DIR = ROOT / "models" / "qwen3_4b_base"
QUESTIONS8 = ROOT / "src" / "questions" / "session8.yaml"
S7_B3_ADAPTER = ROOT / "models" / "fiorillo_v0_3" / "arm_b" / "adapter.safetensors"
FILES = ["LICENSE", "README.md", "annotations/README.md", "annotations/prompts_merged.csv", "annotations/annotations_merged.csv",
         "annotations/splits/train_article_ids.txt", "annotations/splits/validation_article_ids.txt",
         "annotations/splits/test_article_ids.txt"]
SPLITS = ("train", "validation", "test")
QID = "evidence_effect"
SOURCE = "evidence_inference_v2"
LABEL_OF_CODE = {1: 0, -1: 1, 0: 2}
MAX_LEN = 4096
SPARE = 16
SEED = 20260927
MONITOR = 300
K1, B = 1.2, 0.75
STOP = set("the a an of and or in on for to with by at from vs versus than as is are was were be been this that these those "
           "group groups patients patient treatment treated receiving received".split())
RESULTS_RE = re.compile(r"RESULT|FINDING|OUTCOME|EFFICAC|SAFETY|ADVERSE|TABLE|RESPONSE|END ?POINT", re.I)
BODY_RE = re.compile(r"(^|\n)BODY\.")
HEADER_RE = re.compile(r"(?m)^(BODY[^\n]*):[ \t]*$")
WORD_RE = re.compile(r"[a-z0-9]+")
FLAG_NAMES = ("Incorrect", "Questionable", "Somewhat malformed")
CHECK_COVERAGE = 0.9576  # the check of 30 September (docs/prompt_session8.md, THE ARTICLE SELECTION)


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- download

def fetch_bytes(url, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ed-decision-model/session8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + 2 * k)
    raise RuntimeError(f"could not fetch {url}: {last}")


def needed_articles(raw):
    """PMCIDs in the three split files that have at least one prompt, sorted as text."""
    split_ids = set()
    for s in SPLITS:
        split_ids |= {line.strip() for line in open(raw / f"annotations/splits/{s}_article_ids.txt", encoding="utf-8") if line.strip()}
    with open(raw / "annotations/prompts_merged.csv", newline="", encoding="utf-8") as f:
        with_prompt = {r["PMCID"] for r in csv.DictReader(f)}
    return sorted(split_ids & with_prompt)


def download(raw=RAW, max_seconds=150.0, workers=16):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    progress_path = raw / "download_progress.json"
    progress = load_json(progress_path, {"commit": COMMIT, "files": {}})
    if progress.get("commit") != COMMIT:
        raise SystemExit(f"{progress_path} was written for another commit")
    t0 = time.time()

    def get(rel):
        data = fetch_bytes(BASE_URL + urllib.parse.quote(rel))
        dest = raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return rel, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    for rel in FILES:
        if rel not in progress["files"]:
            k, v = get(rel)
            progress["files"][k] = v
    save_json(progress_path, progress)
    rels = [f"annotations/txt_files/PMC{i}.txt" for i in needed_articles(raw)]
    todo = [r for r in rels if r not in progress["files"]]
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        pending = set()
        it = iter(todo)
        while True:
            while len(pending) < workers * 2 and time.time() - t0 < max_seconds:
                try:
                    pending.add(ex.submit(get, next(it)))
                except StopIteration:
                    break
            if not pending:
                break
            finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                k, v = fut.result()
                progress["files"][k] = v
                done += 1
            if done and done % 200 == 0:
                save_json(progress_path, progress)
    save_json(progress_path, progress)
    missing = [r for r in rels if r not in progress["files"]]
    status = {"articles": len(rels), "fetched_now": done, "missing": len(missing), "complete": not missing,
              "seconds": round(time.time() - t0, 1)}
    print(json.dumps(status))
    return status


# ---------------------------------------------------------------- reading EI

def read_flags(raw):
    """Prompt ID -> Incorrect, Questionable or Somewhat malformed, from annotations/README.md's Caveat lists."""
    text = (Path(raw) / "annotations/README.md").read_text(encoding="utf-8-sig")
    flags = {}
    for name in FLAG_NAMES:
        m = re.search(r"###\s*" + re.escape(name) + r"\s*:\s*\n+([0-9,\s]+)", text)
        if not m:
            raise SystemExit(f"annotations/README.md has no list headed {name!r}")
        for x in re.findall(r"\d+", m.group(1)):
            flags.setdefault(int(x), name)
    return flags


def read_inputs(raw):
    raw = Path(raw)
    splits = {s: [line.strip() for line in open(raw / f"annotations/splits/{s}_article_ids.txt", encoding="utf-8") if line.strip()]
              for s in SPLITS}
    art_split = {a: s for s in SPLITS for a in splits[s]}
    with open(raw / "annotations/prompts_merged.csv", newline="", encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))
    csv.field_size_limit(1 << 30)
    ann = collections.defaultdict(list)
    with open(raw / "annotations/annotations_merged.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ann[r["PromptID"]].append(r)
    return art_split, prompts, ann, read_flags(raw)


def accepted(rows):
    return [r for r in rows if r["Valid Label"] == "True" and r["Valid Reasoning"] == "True" and r["Annotations"].strip()]


def label_code(codes):
    counts = collections.Counter(codes)
    top = max(counts.values())
    return min(c for c in counts if counts[c] == top)


def article_text(raw, pmcid):
    return (Path(raw) / f"annotations/txt_files/PMC{pmcid}.txt").read_text(encoding="utf-8")


def select_prompts(raw):
    """{split: [prompt dict]} in PromptID order, and the drops by reason."""
    art_split, prompts, ann, flags = read_inputs(raw)
    kept = {s: [] for s in SPLITS}
    drops = collections.Counter()
    disagree = collections.Counter()
    empty = {}
    for row in prompts:
        pid, pmcid = row["PromptID"], row["PMCID"]
        split = art_split.get(pmcid)
        if split is None:
            drops["PMCID in no split file"] += 1
            continue
        acc = accepted(ann.get(pid, []))
        if not acc:
            drops[f"{split}: no accepted row"] += 1
            continue
        if pmcid not in empty:
            empty[pmcid] = (Path(raw) / f"annotations/txt_files/PMC{pmcid}.txt").stat().st_size == 0
        if empty[pmcid]:
            drops[f"{split}: empty article text"] += 1
            continue
        flag = flags.get(int(pid))
        if split == "train" and flag == "Incorrect":
            drops["train: listed as Incorrect in annotations/README.md"] += 1
            continue
        codes = [int(r["Label Code"]) for r in acc]
        if any(c not in LABEL_OF_CODE for c in codes):
            drops[f"{split}: a label outside the three answers"] += 1
            continue
        if len(set(codes)) > 1:
            disagree[split] += 1
        spans = [(int(r["Evidence Start"]), int(r["Evidence End"])) for r in acc
                 if int(r["Evidence Start"]) >= 0 and int(r["Evidence End"]) >= int(r["Evidence Start"])]
        kept[split].append({"pid": pid, "pmcid": pmcid, "intervention": row["Intervention"], "comparator": row["Comparator"],
                            "outcome": row["Outcome"], "label": LABEL_OF_CODE[label_code(codes)],
                            "accepted_labels": sorted({LABEL_OF_CODE[c] for c in codes}), "flag": flag or "none", "spans": spans})
    for s in SPLITS:
        kept[s].sort(key=lambda p: int(p["pid"]))
    return kept, dict(drops), dict(disagree)


# ---------------------------------------------------------------- articles, words, BM25

def article_parts(text):
    """(prefix_text, passages): the title and abstract, stripped at their end; passages as (start, end, text, results)."""
    m = BODY_RE.search(text)
    body_start = (m.start() + len(m.group(1))) if m else len(text)
    heads = [(h.start(), h.group(1)) for h in HEADER_RE.finditer(text)]
    starts = [p for p, _ in heads]
    passages, pos = [], body_start
    for chunk in text[body_start:].split("\n\n"):
        stripped = chunk.strip()
        if stripped:
            start = pos + (len(chunk) - len(chunk.lstrip()))
            k = bisect.bisect_right(starts, start) - 1
            results = k >= 0 and bool(RESULTS_RE.search(heads[k][1]))
            passages.append((start, start + len(stripped), stripped, results))
        pos += len(chunk) + 2
    return text[:body_start].rstrip(), passages


def words(text):
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 1]


def bm25_stats(raw, train_prompts):
    df, n_pass, total = collections.Counter(), 0, 0
    for pmcid in sorted({p["pmcid"] for p in train_prompts}):
        _, passages = article_parts(article_text(raw, pmcid))
        for _, _, t, _ in passages:
            w = words(t)
            df.update(set(w))
            n_pass += 1
            total += len(w)
    return {"N": n_pass, "avgL": total / n_pass, "df": dict(sorted(df.items())),
            "articles": len({p["pmcid"] for p in train_prompts}), "k1": K1, "b": B}


def bm25(query, passage_words, stats):
    tf = collections.Counter(passage_words)
    L, N, avgl, df = len(passage_words), stats["N"], stats["avgL"], stats["df"]
    score = 0.0
    for t in set(query):
        if tf[t]:
            n = df.get(t, 0)
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
            score += idf * tf[t] * (K1 + 1) / (tf[t] + K1 * (1 - B + B * L / avgl))
    return score


def order_passages(prompt, passages, stats):
    q = words(prompt["intervention"] + " " + prompt["comparator"] + " " + prompt["outcome"])
    scored = [(0 if p[3] else 1, -bm25(q, words(p[2]), stats), i) for i, p in enumerate(passages)]
    return [i for _, _, i in sorted(scored)]


def assemble(prefix_text, prefix_chars, passages, kept):
    return prefix_text[:prefix_chars] + "".join("\n\n" + passages[i][2] for i in sorted(kept))


def make_state(prompt, article):
    return {"intervention": prompt["intervention"], "comparator": prompt["comparator"], "outcome": prompt["outcome"],
            "article": article}


def make_record(split, prompt, article):
    rec = {"id": f"evidence-{prompt['pid']}", "qid": QID, "state": make_state(prompt, article), "label": prompt["label"],
           "target": one_hot(prompt["label"], 3)}
    if split == "validation":
        rec["calibrate"] = True
    rec["group"] = {"source": SOURCE, "pmcid": prompt["pmcid"], "flag": prompt["flag"], "accepted_labels": prompt["accepted_labels"]}
    return rec


# ---------------------------------------------------------------- the selection (needs Qwen's tokenizer)

class Budget:
    """Token counts as src/train/qwen_items.py renders a record: the serialized state, then the question's tail."""

    def __init__(self, tokenizer, questions_path=QUESTIONS8, max_len=MAX_LEN, spare=SPARE):
        from build_items import load_questions
        from qwen_items import QwenFormat
        from laya.common import serialize_state
        self.tok, self.serialize = tokenizer, serialize_state
        self.fmt = QwenFormat(tokenizer, load_questions(questions_path)[QID])
        self.limit = max_len - spare - len(self.fmt.tail)
        self.sep = self.count_escaped("\n\n")

    def count(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def count_escaped(self, text):
        return self.count(json.dumps(text, ensure_ascii=False)[1:-1])

    def state_tokens(self, prompt, article):
        return self.count(self.serialize(make_state(prompt, article)))

    def total(self, prompt, article):
        return self.state_tokens(prompt, article) + len(self.fmt.tail)


def select_one(prompt, prefix_text, passages, piece_tokens, prefix_tokens, stats, budget):
    """(prefix_chars, kept passage indices, exact total tokens, or None when even an empty article does not fit)."""
    base = budget.state_tokens(prompt, "")
    if base > budget.limit:
        return None
    if base + prefix_tokens > budget.limit:
        lo, hi = 0, len(prefix_text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if budget.state_tokens(prompt, prefix_text[:mid]) <= budget.limit:
                lo = mid
            else:
                hi = mid - 1
        return lo, [], budget.total(prompt, prefix_text[:lo])
    taken, approx = [], base + prefix_tokens
    for i in order_passages(prompt, passages, stats):
        cost = budget.sep + piece_tokens[i]
        if approx + cost <= budget.limit:
            taken.append(i)
            approx += cost
    while True:
        exact = budget.state_tokens(prompt, assemble(prefix_text, len(prefix_text), passages, taken))
        if exact <= budget.limit:
            return len(prefix_text), sorted(taken), exact + len(budget.fmt.tail)
        taken.pop()


def coverage_of(prompt, prefix_chars, kept, passages):
    regions = [(0, prefix_chars)] + [(passages[i][0], passages[i][1]) for i in kept]
    return any(any(s >= a and e <= b for a, b in regions) for s, e in prompt["spans"])


def build(raw=RAW, out_dir=OUT_DIR, results=RESULTS, tokenizer_dir=QWEN_DIR, max_seconds=150.0, s7_b3=S7_B3_ADAPTER):
    from transformers import AutoTokenizer
    raw, out_dir, results = Path(raw), Path(out_dir), Path(results)
    t0 = time.time()
    progress = load_json(raw / "download_progress.json", {})
    if not progress.get("files"):
        raise SystemExit("run the download first")
    kept, drops, disagree = select_prompts(raw)
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "bm25_stats.json"
    if not stats_path.exists():
        save_json(stats_path, bm25_stats(raw, kept["train"]))
    stats = load_json(stats_path, None)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    budget = Budget(tok)
    finished, articles_now = True, 0
    for split in SPLITS:
        part = parts / f"selection_{split}.jsonl"
        done = {}
        if part.exists():
            for line in open(part, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
        by_article = collections.defaultdict(list)
        for p in kept[split]:
            if f"evidence-{p['pid']}" not in done:
                by_article[p["pmcid"]].append(p)
        with open(part, "a", encoding="utf-8") as f:
            for pmcid in sorted(by_article):
                if articles_now and time.time() - t0 > max_seconds:  # every run selects at least one article, so each run makes progress
                    finished = False
                    break
                articles_now += 1
                prefix_text, passages = article_parts(article_text(raw, pmcid))
                enc = tok([json.dumps(p[2], ensure_ascii=False)[1:-1] for p in passages] or [""], add_special_tokens=False)["input_ids"]
                piece_tokens = [len(x) for x in enc][: len(passages)]
                prefix_tokens = budget.count_escaped(prefix_text)
                for p in by_article[pmcid]:
                    got = select_one(p, prefix_text, passages, piece_tokens, prefix_tokens, stats, budget)
                    row = {"id": f"evidence-{p['pid']}", "pmcid": pmcid, "split": split}
                    if got is None:
                        row.update({"skip": "the prompt alone does not fit"})
                    else:
                        row.update({"prefix_chars": got[0], "passages": got[1], "tokens": got[2],
                                    "covered": coverage_of(p, got[0], got[1], passages) if p["spans"] else None})
                    f.write(json.dumps(row) + "\n")
        if not finished:
            break
    if not finished:
        print(json.dumps({"finished": False, "seconds": round(time.time() - t0, 1)}))
        return {"finished": False}
    return write_outputs(raw, out_dir, results, kept, drops, disagree, stats_path, s7_b3)


def read_selection(parts):
    rows = {}
    for split in SPLITS:
        for line in open(Path(parts) / f"selection_{split}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def records_from(raw, kept, selection):
    """{split: records} and the skips, from the prompts and the selection alone (no tokenizer)."""
    out, skipped = {}, collections.Counter()
    for split in SPLITS:
        recs = []
        cache = {}
        for p in kept[split]:
            sel = selection[f"evidence-{p['pid']}"]
            if "skip" in sel:
                skipped[f"{split}: {sel['skip']}"] += 1
                continue
            if p["pmcid"] not in cache:
                cache = {p["pmcid"]: article_parts(article_text(raw, p["pmcid"]))}
            prefix_text, passages = cache[p["pmcid"]]
            recs.append(make_record(split, p, assemble(prefix_text, sel["prefix_chars"], passages, sel["passages"])))
        out[split] = recs
    return out, dict(skipped)


def monitor_subset(validation_records):
    ids = sorted(r["id"] for r in validation_records)
    pick = set(random.Random(SEED).sample(ids, MONITOR))
    return sorted((r for r in validation_records if r["id"] in pick), key=lambda r: r["id"])


def write_records(out_dir, recs):
    out_dir = Path(out_dir)
    paths = {}
    for split in SPLITS:
        paths[f"evidence_effect_{split}.jsonl"] = write_jsonl(out_dir / f"evidence_effect_{split}.jsonl", recs[split])
    paths["evidence_effect_validation_monitor.jsonl"] = write_jsonl(out_dir / "evidence_effect_validation_monitor.jsonl",
                                                                    monitor_subset(recs["validation"]))
    return paths


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def write_outputs(raw, out_dir, results, kept, drops, disagree, stats_path, s7_b3):
    selection = read_selection(out_dir / "parts")
    sel_path = out_dir / "article_selection.jsonl"
    with open(sel_path, "w", encoding="utf-8", newline="\n") as f:
        for split in SPLITS:
            for p in kept[split]:
                r = selection[f"evidence-{p['pid']}"]
                f.write(json.dumps({k: r[k] for k in ("id", "pmcid", "split", "prefix_chars", "passages") if k in r} |
                                   ({"skip": r["skip"]} if "skip" in r else {})) + "\n")
    recs, skipped = records_from(raw, kept, selection)
    paths = write_records(out_dir, recs)
    tokens = {s: [selection[r["id"]]["tokens"] for r in recs[s]] for s in SPLITS}
    cov = {}
    for split in ("train", "validation"):
        flags = [selection[f"evidence-{p['pid']}"].get("covered") for p in kept[split]]
        usable = [c for c in flags if c is not None]
        cov[split] = {"prompts_with_usable_offsets": len(usable), "covered": int(sum(usable)),
                      "coverage": sum(usable) / len(usable) if usable else None}
    stats = load_json(stats_path, {})
    progress = load_json(Path(raw) / "download_progress.json", {})
    selection_summary = {
        "plan": "docs/prompt_session8.md, THE ARTICLE SELECTION", "written_utc": now(), "max_len": MAX_LEN, "spare": SPARE,
        "rule": "title and abstract, then results-section passages, then others, by BM25 against the prompt's words, to 4,096 tokens",
        "coverage": cov, "check_of_30_september_train": CHECK_COVERAGE,
        "difference_from_check_train": cov["train"]["coverage"] - CHECK_COVERAGE,
        "within_one_point": abs(cov["train"]["coverage"] - CHECK_COVERAGE) <= 0.01,
        "mean_tokens_per_record": {s: sum(t) / len(t) for s, t in tokens.items() if t},
        "bm25": {"N": stats.get("N"), "avgL": stats.get("avgL"), "articles": stats.get("articles"), "vocabulary": len(stats.get("df", {})),
                 "k1": K1, "b": B, "file": "data/processed/session8/bm25_stats.json", "sha256": sha256_file(stats_path)}}
    save_json(Path(results) / "article_selection.json", selection_summary)
    manifest = {
        "plan": "docs/prompt_session8.md, EI'S RECORDS", "written_utc": now(), "commit": COMMIT,
        "counts": {s: {"prompts": len(recs[s]), "articles": len({r["group"]["pmcid"] for r in recs[s]}),
                       "answers": dict(sorted(collections.Counter(r["label"] for r in recs[s]).items()))} for s in SPLITS},
        "monitor": {"prompts": MONITOR, "seed": SEED},
        "drops": dict(sorted(drops.items())), "skipped_by_the_selection": skipped,
        "accepted_labels_disagree": disagree,
        "tokens_per_record": {s: {"mean": sum(t) / len(t), "p90": percentile(t, 0.9), "largest": max(t)} for s, t in tokens.items() if t},
        "sha256": {"records": {name: sha256_file(p) for name, p in paths.items()},
                   "article_selection.jsonl": sha256_file(sel_path), "bm25_stats.json": sha256_file(stats_path),
                   "raw": {k: v["sha256"] for k, v in sorted(progress.get("files", {}).items())}},
        "s7_b3_adapter_sha256": sha256_file(s7_b3) if Path(s7_b3).exists() else None,
    }
    old = load_json(Path(results) / "records_manifest.json", {})
    if "article_licenses" in old:
        manifest["article_licenses"] = old["article_licenses"]
    save_json(Path(results) / "records_manifest.json", manifest)
    print(json.dumps({"finished": True, "counts": manifest["counts"], "coverage_train": cov["train"]["coverage"],
                      "tokens": manifest["tokens_per_record"], "skipped": skipped}, indent=1))
    return {"finished": True, "manifest": manifest, "selection": selection_summary}


def from_selection(raw, selection_path, out_dir, manifest_path, check_out):
    kept, _, _ = select_prompts(raw)
    selection = {}
    for line in open(selection_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            selection[r["id"]] = r
    recs, _ = records_from(raw, kept, selection)
    paths = write_records(out_dir, recs)
    expected = load_json(manifest_path, {}).get("sha256", {}).get("records", {})
    check = {"written_utc": now(), "files": {}}
    for name, p in paths.items():
        got = sha256_file(p)
        check["files"][name] = {"sha256": got, "expected": expected.get(name), "equal": got == expected.get(name)}
    check["all_equal"] = all(v["equal"] for v in check["files"].values())
    save_json(check_out, check)
    print(json.dumps({"all_equal": check["all_equal"]}))
    if not check["all_equal"]:
        raise SystemExit("the rebuilt records differ from records_manifest.json")
    return check


def licenses(raw=RAW, results=RESULTS, out_dir=OUT_DIR, batch=10, workers=3, max_seconds=150.0):
    """Europe PMC's licence for every article, 10 PMCIDs a query (resultType core, about 11 seconds a query), 3 queries at
    once (more are refused with HTTP 503); progress saved in data/processed/session8/parts/licenses_progress.json, so a run
    the shell ends resumes, and a refused query is asked again on the next run."""
    ids = needed_articles(Path(raw))
    t0 = time.time()
    progress_path = Path(out_dir) / "parts" / "licenses_progress.json"
    progress = load_json(progress_path, {"batches": {}})
    chunks = {str(k): ids[k:k + batch] for k in range(0, len(ids), batch)}

    def query(chunk):
        q = " OR ".join(f"PMCID:PMC{i}" for i in chunk)
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" +
               urllib.parse.urlencode({"query": q, "format": "json", "resultType": "core", "pageSize": 100}))
        data = json.loads(fetch_bytes(url))
        return {r["pmcid"]: (r.get("license") or "none recorded") for r in data.get("resultList", {}).get("result", []) if r.get("pmcid")}

    todo = [k for k in chunks if k not in progress["batches"]]
    with cf.ThreadPoolExecutor(workers) as ex:
        pending, it = {}, iter(todo)
        while True:
            while len(pending) < workers and time.time() - t0 < max_seconds:
                try:
                    k = next(it)
                except StopIteration:
                    break
                pending[ex.submit(query, chunks[k])] = k
            if not pending:
                break
            finished, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                k = pending.pop(fut)
                try:
                    progress["batches"][k] = fut.result()
                except Exception:  # noqa: BLE001  (a refused query is asked again on the next run)
                    pass
            save_json(progress_path, progress)
    if len(progress["batches"]) < len(chunks):
        print(json.dumps({"finished": False, "batches_done": len(progress["batches"]), "batches": len(chunks)}))
        return None
    found = {}
    for got in progress["batches"].values():
        found.update(got)
    today = dt.date.today().isoformat()
    rows = [(f"PMC{i}", found.get(f"PMC{i}", "not found")) for i in ids]
    path = Path(results) / "article_licenses.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmcid", "license", "read"])
        for pmcid, lic in rows:
            w.writerow([pmcid, lic, today])
    tally = dict(sorted(collections.Counter(lic for _, lic in rows).items(), key=lambda kv: -kv[1]))
    manifest_path = Path(results) / "records_manifest.json"
    manifest = load_json(manifest_path, {})
    manifest["article_licenses"] = {"source": "Europe PMC REST search, resultType core", "read": today, "articles": len(rows), "tally": tally}
    save_json(manifest_path, manifest)
    print(json.dumps(manifest["article_licenses"], indent=1))
    return tally


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--raw", type=Path, default=RAW)
    d.add_argument("--max-seconds", type=float, default=150.0)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=RAW)
    b.add_argument("--out-dir", type=Path, default=OUT_DIR)
    b.add_argument("--results", type=Path, default=RESULTS)
    b.add_argument("--tokenizer-dir", type=Path, default=QWEN_DIR)
    b.add_argument("--max-seconds", type=float, default=150.0)
    f = sub.add_parser("from-selection")
    f.add_argument("--raw", type=Path, default=RAW)
    f.add_argument("--selection", type=Path, default=OUT_DIR / "article_selection.jsonl")
    f.add_argument("--out-dir", type=Path, default=OUT_DIR)
    f.add_argument("--manifest", type=Path, default=RESULTS / "records_manifest.json")
    f.add_argument("--check-out", type=Path, required=True)
    lc = sub.add_parser("licenses")
    lc.add_argument("--raw", type=Path, default=RAW)
    lc.add_argument("--results", type=Path, default=RESULTS)
    lc.add_argument("--out-dir", type=Path, default=OUT_DIR)
    lc.add_argument("--max-seconds", type=float, default=150.0)
    args = parser.parse_args(argv)
    if args.cmd == "download":
        download(args.raw, args.max_seconds)
    elif args.cmd == "build":
        build(args.raw, args.out_dir, args.results, args.tokenizer_dir, args.max_seconds)
    elif args.cmd == "from-selection":
        from_selection(args.raw, args.selection, args.out_dir, args.manifest, args.check_out)
    else:
        licenses(args.raw, args.results, args.out_dir, max_seconds=args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""PMC full text for session 5's pmc and case_reports sources (docs/prompt_session5.md, THE TEXT, step 2 and
corrections 2 and 3), run on the pod.

Stages, each resumable, working in --work (default /root/eddm_s5/text/pmc_work):

candidates: the PMC candidates, from E-utilities esearch on db=pmc for ("cc by license"[filter] OR "cc0 license"[filter])
  in slices by PMC identifier range (correction 3: E-utilities no longer returns this query's identifiers from the
  history server or past the first 9,999 of one search): the query AND <a>:<b>[uid], one esearch of up to 9,999
  identifiers per slice, ranges of 100,000 identifiers split in halves until no slice holds more than 9,999
  -> pmc_candidates.txt. The case reports
  (correction 2): esearch on db=pubmed for "case reports"[pt] AND "pubmed pmc"[sb], sliced by entry date until no
  slice holds more than 9,999 records, each slice's PMIDs fetched in one search -> case_report_pmids.txt; then the PMC
  ID Converter API, 200 PMIDs a request -> case_report_pmcids.tsv (pmid, pmcid). Every NCBI request carries
  tool=ed-decision-model and no email, runs at most 3 a second, and waits and retries on HTTP 429 and on errors. The
  queries' times, counts and slices go to candidates.json.
inventory: the latest daily S3 inventory under inventory-reports/pmc-oa-opendata/metadata/ (its manifest.json, then
  each CSV file, MD5-checked) -> the highest version listed for each candidate PMCID -> latest_versions.tsv.
fetch <pmc|case_reports>: the source's candidates in a seeded random order (seed 20260927 and the source's name). For
  each: the latest version's metadata JSON must have license_code CC0 or CC BY, is_pmc_openaccess true, is_retracted
  false and is_historical_ocr false; its pmid must agree with the routing (a case_reports article's pmid is a
  case-report PMID, a pmc article's is not) and must not be on the PubMedQA exclusion list; then the version's .txt,
  its MD5 checked against the metadata; 32 requests in parallel. The text is cleaned (clean_text), passed through the
  13-word check (src/corpus/decontaminate.py) with multiprocessing, and counted in Laya's tokens. Writes
  /root/eddm_s5/text/<source>/part-<nnnn>.jsonl.gz (about 5,000 documents per part; id PMCID.version) and
  manifest.json (parts; the documents with PMCID, version, license_code and tokens; drops by reason; the queries),
  until pmc has 900,000,000 tokens and case_reports 250,000,000. Every article whose text was fetched, used or not,
  is appended to /root/eddm_s5/text/pmc_training_manifest.csv (PMCID, version, license_code, source, tokens, status),
  the attribution list released with the model (step 15).

Usage on the pod:
    python src/corpus/fetch_pmc.py candidates
    python src/corpus/fetch_pmc.py inventory
    python src/corpus/fetch_pmc.py fetch pmc --tokenizer /root/eddm_s5/models/laya/tokenizer/tokenizer.json
    python src/corpus/fetch_pmc.py fetch case_reports --tokenizer ...
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "corpus"))
from decontaminate import ParallelChecker  # noqa: E402
from parts import load_json, load_tokenizer, part_name, save_json, special_strings, token_counts, write_part  # noqa: E402

SEED = 20260927
TOOL = "ed-decision-model"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
INVENTORY = "inventory-reports/pmc-oa-opendata/metadata/"
PMC_QUERY = '"cc by license"[filter] OR "cc0 license"[filter]'
CASE_QUERY = '"case reports"[pt] AND "pubmed pmc"[sb]'
LICENCES = ("CC0", "CC BY")
TARGETS = {"pmc": 900_000_000, "case_reports": 250_000_000}
SLICE_CAP = 9999


class NotFound(Exception):
    pass


def http_get(url, headers=None, timeout=90, attempts=8):
    """GET with retries on HTTP 429, 5xx and network errors (waits 1, 2, 4 ... 60 seconds); 404 raises NotFound."""
    h = {"User-Agent": TOOL}
    h.update(headers or {})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(url)
            if e.code != 429 and e.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed after {attempts} attempts: {url}")


class NcbiClient:
    """E-utilities and the ID converter, at most 3 requests a second (no API key)."""

    def __init__(self, interval=0.34):
        self.interval, self.lock, self.last = interval, threading.Lock(), 0.0

    def get(self, base, params):
        with self.lock:
            wait = self.last + self.interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
        return http_get(base + "?" + urllib.parse.urlencode(dict(params, tool=TOOL))).decode("utf-8")

    def esearch(self, db, term, attempts=5, **params):
        for attempt in range(attempts):
            x = self.get(EUTILS + "esearch.fcgi", dict(params, db=db, term=term))
            found = re.search(r"<Count>(\d+)</Count>", x)
            if found:
                break
            time.sleep(min(60, 2 ** (attempt + 1)))  # an <ERROR> answer without a count: wait and ask again
        else:
            answer = re.sub(r"\s+", " ", x)[:300]
            raise RuntimeError(f"esearch gave no count after {attempts} attempts: {answer}")
        errors = re.findall(r"<(?:FieldNotFound|PhraseNotFound|QuotedPhraseNotFound|OutputMessage)>(.*?)<", x)
        count = int(found.group(1))
        ids = [int(i) for i in re.findall(r"<Id>(\d+)</Id>", x)]
        webenv = re.search(r"<WebEnv>(.*?)</WebEnv>", x)
        key = re.search(r"<QueryKey>(\d+)</QueryKey>", x)
        translation = re.search(r"<QueryTranslation>(.*?)</QueryTranslation>", x)
        return {"count": count, "ids": ids, "webenv": webenv.group(1) if webenv else None,
                "query_key": key.group(1) if key else None, "errors": errors,
                "translation": translation.group(1) if translation else None}

    def uilist(self, db, webenv, query_key, retstart, retmax=10000):
        x = self.get(EUTILS + "efetch.fcgi", {"db": db, "WebEnv": webenv, "query_key": query_key, "retstart": retstart,
                                              "retmax": retmax, "rettype": "uilist", "retmode": "text"})
        return [int(line) for line in x.split() if line.strip().isdigit()]

    def idconv(self, pmids):
        data = json.loads(self.get(IDCONV, {"ids": ",".join(str(p) for p in pmids), "idtype": "pmid", "format": "json"}))
        return {int(r["pmid"]): int(r["pmcid"][3:]) for r in data.get("records", []) if r.get("pmcid")}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_slices(count, start, end, cap=SLICE_CAP):
    """Split the date range [start, end] in halves until each slice's count is at most cap; returns
    [(start, end, count)] for the slices with records. count(start, end) gives the records in a range."""
    n = count(start, end)
    if n == 0:
        return []
    if n <= cap:
        return [(start, end, n)]
    if start == end:
        raise RuntimeError(f"{start}: {n} records on one day, more than {cap}")
    mid = start + (end - start) // 2
    return date_slices(count, start, mid, cap) + date_slices(count, mid + dt.timedelta(days=1), end, cap)


def uid_slices(search, lo, hi, cap=SLICE_CAP):
    """Split the identifier range [lo, hi] in halves until each slice's count is at most cap (correction 3).
    search(lo, hi) returns (count, identifiers), the identifiers complete whenever count <= cap; returns
    [(lo, hi, count, identifiers)] for the slices with records."""
    n, ids = search(lo, hi)
    if n == 0:
        return []
    if n <= cap:
        return [(lo, hi, n, ids)]
    if lo == hi:
        raise RuntimeError(f"identifier {lo}: {n} records")
    mid = lo + (hi - lo) // 2
    return uid_slices(search, lo, mid, cap) + uid_slices(search, mid + 1, hi, cap)


UID_STEP = 100_000


def stage_candidates(work, client):
    work.mkdir(parents=True, exist_ok=True)
    info = load_json(work / "candidates.json", {})
    if not (work / "pmc_candidates.txt").exists():
        r = client.esearch("pmc", PMC_QUERY, retmax=SLICE_CAP)
        if r["errors"]:
            raise SystemExit(f"PMC query reported {r['errors']}; stop")
        top = (max(r["ids"]) // UID_STEP + 2) * UID_STEP  # 100,000 or more past the largest identifier on the first page

        def search(a, b):
            s = client.esearch("pmc", f"({PMC_QUERY}) AND {a}:{b}[uid]", retmax=SLICE_CAP)
            if s["count"] and s["errors"]:
                raise SystemExit(f"PMC slice {a}:{b} reported {s['errors']}; stop")
            return s["count"], s["ids"]

        slices = []
        for a in range(1, top + 1, UID_STEP):
            slices.extend(uid_slices(search, a, min(top, a + UID_STEP - 1)))
        ids = sorted({i for s in slices for i in s[3]})
        (work / "pmc_candidates.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        info["pmc"] = {"query": PMC_QUERY, "db": "pmc", "time": now(), "count": r["count"], "translation": r["translation"],
                       "method": "slices by PMC identifier range, the query AND <a>:<b>[uid], each at most 9,999 (correction 3)",
                       "identifier_ranges_up_to": top, "slices": [[a, b, n, len(s)] for a, b, n, s in slices],
                       "sum_of_slice_counts": sum(s[2] for s in slices), "identifiers_fetched": len(ids)}
        save_json(work / "candidates.json", info)
        print(f"PMC candidates: {len(ids):,} (count {r['count']:,})", flush=True)
    if not (work / "case_report_pmids.txt").exists():
        total = client.esearch("pubmed", CASE_QUERY, retmax=0)
        if total["errors"]:
            raise SystemExit(f"PubMed query reported {total['errors']}; stop")

        def count(a, b):
            return client.esearch("pubmed", CASE_QUERY, retmax=0, datetype="edat", mindate=a.strftime("%Y/%m/%d"),
                                  maxdate=b.strftime("%Y/%m/%d"))["count"]

        today = dt.date.today()
        ranges = [(dt.date(1800, 1, 1), dt.date(1944, 12, 31))] + [(dt.date(y, 1, 1), dt.date(y, 12, 31)) for y in range(1945, today.year + 1)]
        slices = []
        for a, b in ranges:
            slices.extend(date_slices(count, a, b))
        pmids, fetched = set(), []
        for a, b, n in slices:
            r = client.esearch("pubmed", CASE_QUERY, retmax=SLICE_CAP, datetype="edat", mindate=a.strftime("%Y/%m/%d"),
                               maxdate=b.strftime("%Y/%m/%d"))
            pmids.update(r["ids"])
            fetched.append([a.isoformat(), b.isoformat(), n, len(r["ids"])])
        (work / "case_report_pmids.txt").write_text("".join(f"{p}\n" for p in sorted(pmids)), encoding="utf-8")
        info["case_reports_pubmed"] = {"query": CASE_QUERY, "db": "pubmed", "time": now(), "count": total["count"],
                                       "translation": total["translation"], "slices": fetched,
                                       "sum_of_slice_counts": sum(s[2] for s in fetched), "pmids_fetched": len(pmids)}
        save_json(work / "candidates.json", info)
        print(f"case-report PMIDs: {len(pmids):,} (count {total['count']:,}, {len(fetched)} slices)", flush=True)
    if not (work / "case_report_pmcids.tsv").exists():
        pmids = [int(x) for x in (work / "case_report_pmids.txt").read_text().split()]
        mapping = {}
        for start in range(0, len(pmids), 200):
            mapping.update(client.idconv(pmids[start : start + 200]))
        with open(work / "case_report_pmcids.tsv", "w", encoding="utf-8") as f:
            for pmid in sorted(mapping):
                f.write(f"{pmid}\tPMC{mapping[pmid]}\n")
        candidates = {int(x) for x in (work / "pmc_candidates.txt").read_text().split()}
        converted = set(mapping.values())
        info["case_reports_converted"] = {"service": IDCONV, "time": now(), "pmids": len(pmids), "with_pmcid": len(mapping),
                                          "among_pmc_candidates": len(converted & candidates)}
        save_json(work / "candidates.json", info)
        print(f"converted: {len(mapping):,} with a PMCID, {len(converted & candidates):,} among the candidates", flush=True)


def latest_inventory_prefix(listing_xml):
    dates = re.findall(r"<Prefix>(" + re.escape(INVENTORY) + r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}Z/)</Prefix>", listing_xml)
    if not dates:
        raise SystemExit("no dated inventory found")
    return max(dates)


KEY = re.compile(r"^metadata/PMC(\d+)\.(\d+)\.json$")


def latest_versions(rows, wanted=None):
    """Highest version per PMCID number from inventory rows (bucket, key, ...); only numbers in wanted if given."""
    best = {}
    for row in rows:
        m = KEY.match(row[1]) if len(row) > 1 else None
        if not m:
            continue
        number, version = int(m.group(1)), int(m.group(2))
        if wanted is not None and number not in wanted:
            continue
        if version > best.get(number, 0):
            best[number] = version
    return best


def stage_inventory(work):
    candidates = {int(x) for x in (work / "pmc_candidates.txt").read_text().split()}
    prefix = latest_inventory_prefix(http_get(BUCKET + "?list-type=2&delimiter=/&prefix=" + INVENTORY).decode())
    manifest = json.loads(http_get(BUCKET + prefix + "manifest.json"))
    best, files = {}, []
    for f in manifest["files"]:
        blob = http_get(BUCKET + f["key"], timeout=600)
        if hashlib.md5(blob).hexdigest() != f["MD5checksum"]:
            raise SystemExit(f"{f['key']}: MD5 differs from the inventory manifest")
        rows = csv.reader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(blob)), encoding="utf-8"))
        for number, version in latest_versions(rows, candidates).items():
            if version > best.get(number, 0):
                best[number] = version
        files.append(f["key"])
    with open(work / "latest_versions.tsv", "w", encoding="utf-8") as out:
        for number in sorted(best):
            out.write(f"{number}\t{best[number]}\n")
    info = load_json(work / "candidates.json", {})
    info["inventory"] = {"prefix": prefix, "creation_timestamp_ms": manifest.get("creationTimestamp"), "files": len(files),
                         "candidates_with_a_version": len(best), "candidates": len(candidates), "time": now()}
    save_json(work / "candidates.json", info)
    print(f"inventory {prefix}: {len(best):,} of {len(candidates):,} candidates have a version", flush=True)


SEPARATOR = re.compile(r"^\W*={10,}\W*$")
REFERENCES = re.compile(r"^\s*(references|bibliography|literature cited)\s*$", re.IGNORECASE)


def clean_text(txt):
    """The article body of a PMC .txt file: what follows the separator line that opens the body (a line of = signs
    that does not underline a JOURNAL INFORMATION or ARTICLE INFORMATION heading), cut at the last line reading
    References, Bibliography or Literature Cited, without lines containing a tab, whitespace collapsed. None if there
    is no body separator or nothing is left."""
    lines = txt.split("\n")
    start = None
    for i, line in enumerate(lines):
        if SEPARATOR.match(line) and not (i > 0 and lines[i - 1].strip().upper().endswith("INFORMATION")):
            start = i + 1
            break
    if start is None:
        return None
    body = lines[start:]
    cuts = [i for i, line in enumerate(body) if REFERENCES.match(line)]
    if cuts:
        body = body[: cuts[-1]]
    text = " ".join(" ".join(line for line in body if "\t" not in line).split())
    return text or None


def route(meta, source, case_pmids, exclusion):
    """None if the version's metadata passes every check for the source, else the reason it is dropped."""
    if meta.get("license_code") not in LICENCES:
        return "licence"
    if meta.get("is_pmc_openaccess") is not True:
        return "not_open_access"
    if meta.get("is_retracted") is not False:
        return "retracted"
    if meta.get("is_historical_ocr") is not False:
        return "historical_ocr"
    pmid = meta.get("pmid")
    pmid = int(pmid) if pmid not in (None, "") else None
    is_case = pmid is not None and pmid in case_pmids
    if (source == "case_reports") != is_case:
        return "routing"
    if pmid is not None and pmid in exclusion:
        return "pubmedqa_pmid"
    return None


def s3_https(url):
    """s3://pmc-oa-opendata/<key>?md5=<md5> -> (https URL, md5)."""
    parsed = urllib.parse.urlparse(url)
    md5 = urllib.parse.parse_qs(parsed.query).get("md5", [None])[0]
    return BUCKET + parsed.path.lstrip("/"), md5


def fetch_one(number, version, source, case_pmids, exclusion):
    """(pmcid, version, license_code, status, text); status "ok" or a drop reason."""
    pmcid = f"PMC{number}"
    if not version:
        return pmcid, None, None, "no_version", None
    key = f"{pmcid}.{version}"
    try:
        meta = json.loads(http_get(BUCKET + f"metadata/{key}.json"))
    except Exception:
        return pmcid, version, None, "metadata_error", None
    licence = meta.get("license_code")
    reason = route(meta, source, case_pmids, exclusion)
    if reason:
        return pmcid, version, licence, reason, None
    if not meta.get("text_url"):
        return pmcid, version, licence, "no_text", None
    url, md5 = s3_https(meta["text_url"])
    try:
        blob = http_get(url)
    except Exception:
        return pmcid, version, licence, "text_error", None
    if md5 and hashlib.md5(blob).hexdigest() != md5:
        return pmcid, version, licence, "md5_mismatch", None
    text = clean_text(blob.decode("utf-8", "replace"))
    if text is None:
        return pmcid, version, licence, "no_body", None
    return pmcid, version, licence, "ok", text


def stage_fetch(source, work, text_root, exclusion_path, reference, tokenizer_json, per_part, threads, processes, batch):
    target = TARGETS[source]
    out = text_root / source
    progress_path = work / f"{source}_progress.json"
    state = load_json(progress_path) or {"walked": 0, "parts": [], "documents": 0, "tokens": 0, "drops": {},
                                         "complete": False, "target_tokens": target}
    if state["complete"]:
        print(f"{source}: complete (nothing to do)")
        return
    candidates = sorted(int(x) for x in (work / "pmc_candidates.txt").read_text().split())
    case_map = [line.split("\t") for line in (work / "case_report_pmcids.tsv").read_text().splitlines() if line]
    case_pmids = {int(p) for p, _ in case_map}
    case_numbers = {int(c[3:]) for _, c in case_map}
    pool_ids = [n for n in candidates if (n in case_numbers) == (source == "case_reports")]
    random.Random(f"{SEED}:{source}").shuffle(pool_ids)
    versions = {int(a): int(b) for a, b in (line.split("\t") for line in (work / "latest_versions.tsv").read_text().splitlines() if line)}
    exclusion = {int(x) for x in Path(exclusion_path).read_text().split()}
    tokenizer, specials = load_tokenizer(tokenizer_json), special_strings(tokenizer_json)
    documents_tsv = work / f"{source}_documents.tsv"
    training_csv = text_root / "pmc_training_manifest.csv"
    drops = dict(state["drops"])
    buffer, pending_docs, pending_rows = [], [], []
    tokens_total = state["tokens"]
    walked = state["walked"]
    t0 = time.time()
    print(f"{source}: {len(pool_ids):,} candidates; resuming at {walked:,} with {tokens_total:,} tokens", flush=True)
    with ThreadPoolExecutor(threads) as pool, ParallelChecker(reference, processes) as checker:
        while tokens_total < target:
            if walked >= len(pool_ids):
                print(f"{source}: candidates used up at {tokens_total:,} tokens", flush=True)
                break
            numbers = pool_ids[walked : walked + batch]
            results = list(pool.map(lambda n: fetch_one(n, versions.get(n), source, case_pmids, exclusion), numbers))
            ok = [k for k, r in enumerate(results) if r[3] == "ok"]
            flags = dict(zip(ok, checker.check([results[k][4] for k in ok])))
            clean = [k for k in ok if not flags[k]]
            counts = dict(zip(clean, token_counts(tokenizer, [results[k][4] for k in clean], specials)))
            for k, (pmcid, version, licence, status, text) in enumerate(results):
                if status == "ok" and flags[k]:
                    status = "thirteen_word"
                if status == "ok" and tokens_total >= target:
                    status = "over_target"
                if status == "ok":
                    n = counts[k]
                    tokens_total += n
                    buffer.append((f"{pmcid}.{version}", text))
                    pending_docs.append((pmcid, version, licence, n))
                    pending_rows.append((pmcid, version, licence, source, n, "used"))
                else:
                    drops[status] = drops.get(status, 0) + 1
                    if status in ("thirteen_word", "over_target", "md5_mismatch", "no_body"):
                        pending_rows.append((pmcid, version, licence, source, counts.get(k, 0), status))
            walked += len(numbers)
            done = tokens_total >= target
            if len(buffer) >= per_part or done or walked >= len(pool_ids):
                info = write_part(out / part_name(len(state["parts"])), buffer) if buffer else None
                if info:
                    info["tokens"] = int(sum(d[3] for d in pending_docs))
                    state["parts"].append(info)
                with open(documents_tsv, "a", encoding="utf-8") as f:
                    f.writelines(f"{p}\t{v}\t{lc}\t{n}\n" for p, v, lc, n in pending_docs)
                new_file = not training_csv.exists()
                with open(training_csv, "a", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(["pmcid", "version", "license_code", "source", "tokens", "status"])
                    w.writerows(pending_rows)
                state.update({"walked": walked, "documents": state["documents"] + len(pending_docs), "tokens": tokens_total,
                              "drops": dict(drops), "complete": done or walked >= len(pool_ids)})
                save_json(progress_path, state)
                buffer, pending_docs, pending_rows = [], [], []
                print(f"{source}: walked {walked:,}, {state['documents']:,} documents, {tokens_total:,} tokens "
                      f"({time.time() - t0:.0f} s)", flush=True)
    docs = [line.split("\t") for line in documents_tsv.read_text(encoding="utf-8").splitlines() if line] if documents_tsv.exists() else []
    manifest = {"source": source, "target_tokens": target, "documents": state["documents"], "tokens": state["tokens"],
                "candidates": len(pool_ids), "walked": state["walked"], "drops": state["drops"], "parts": state["parts"],
                "queries": load_json(work / "candidates.json", {}),
                "document_fields": ["pmcid", "version", "license_code", "tokens"],
                "document_list": [[p, int(v), lc, int(n)] for p, v, lc, n in docs]}
    save_json(out / "manifest.json", manifest)
    print(f"{source}: {state['documents']:,} documents, {state['tokens']:,} tokens; drops {state['drops']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("candidates", "inventory", "fetch"))
    parser.add_argument("source", nargs="?", choices=("pmc", "case_reports"))
    parser.add_argument("--work", type=Path, default=Path("/root/eddm_s5/text/pmc_work"))
    parser.add_argument("--text-root", type=Path, default=Path("/root/eddm_s5/text"))
    parser.add_argument("--exclusion", type=Path, default=ROOT / "data" / "processed" / "corpus" / "session5" / "pubmedqa_pmids.txt")
    parser.add_argument("--reference", type=Path, default=ROOT / "data" / "processed" / "corpus" / "session5" / "pubmedqa_labeled_text.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Laya's tokenizer.json (fetch only)")
    parser.add_argument("--per-part", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()
    if args.stage == "candidates":
        stage_candidates(args.work, NcbiClient())
    elif args.stage == "inventory":
        stage_inventory(args.work)
    else:
        if not args.source or not args.tokenizer:
            parser.error("fetch needs a source (pmc or case_reports) and --tokenizer")
        stage_fetch(args.source, args.work, args.text_root, args.exclusion, args.reference, args.tokenizer,
                    args.per_part, args.threads, args.processes, args.batch)


if __name__ == "__main__":
    sys.exit(main())

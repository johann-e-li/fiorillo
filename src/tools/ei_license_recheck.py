"""EI articles' own license statements, read from PubMed Central, beside Europe PMC's recorded license.

Written on 1 October 2026 at Johann's request ("look into the training articles? to see if they are licensed?"), before
docs/prompt_session9.md was committed, and revised the same night after an independent audit of its classes (the audit's
findings are in docs/log.md). Session 9's clean set (docs/prompt_session9.md, EI'S LICENSE-CLEAN RECORDS) rests on its
output: an article is clean when its own record states CC BY (any version), CC0 or the public domain for the article
itself, and no part of its statement (the license links, the license-type attributes, the license text and the copyright
statements) restricts commercial use, derivatives or the license of what is made from it. Where the parts disagree, the
more restrictive part decides: a CC BY link beside text that says non-commercial is not clean.

  fetch     the license statement of every article in results/fiorillo_v0_4/article_licenses.csv (EI's three splits),
            from NCBI's E-utilities (efetch, db=pmc, 10 PMCIDs a request, about one request a second, under NCBI's limit
            of three a second without a key): the <permissions> element of the article's front matter (every
            <license> with its license-type, link and text, and every <copyright-statement>), or, where an older record
            has no <permissions>, the <license> and <copyright-statement> elements directly under <article-meta>.
            Progress is saved after every request in data/processed/session9/license_recheck/permissions.jsonl (ignored
            by git), so a run stopped by the linked shell's time limit resumes when run again.
  classify  one class for each article's statement (CLASSES; the rules are tried in classify_one's order), the clean
            rule, and results/fiorillo_v0_5/license_recheck.csv (one row an article: PMCID, splits, Europe PMC's
            license, the class, clean, the year, and the article's number of prompts) and license_recheck.json (the
            counts, beside Europe PMC's). Only the records' article IDs are read; no label is read.

Usage, on the linked shell, from the project folder:
    TMPDIR=/tmp/eddm_tmp python3 src/tools/ei_license_recheck.py fetch       (run again until it prints "complete")
    TMPDIR=/tmp/eddm_tmp python3 src/tools/ei_license_recheck.py classify
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LICENSES = ROOT / "results" / "fiorillo_v0_4" / "article_licenses.csv"
RECORDS = ROOT / "data" / "processed" / "session8"
RAW_DIR = ROOT / "data" / "processed" / "session9" / "license_recheck"
OUT_DIR = ROOT / "results" / "fiorillo_v0_5"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={ids}&tool=eddm_license_recheck"
XLINK = "{http://www.w3.org/1999/xlink}href"
SPLITS = ("train", "validation", "test")

CLASSES = (
    "cc by",                      # one or more links to creativecommons.org/licenses/by/ and no other license link
    "cc by-nc", "cc by-nc-nd", "cc by-nc-sa", "cc by-nd", "cc by-sa",   # the only license link names this license
    "conflicting license links",  # links to more than one Creative Commons license
    "cc0",                        # the article itself dedicated by CC0 (a link or the words), not a waiver for data only
    "open link or statement, restricting words in its text",  # a CC BY or CC0 link, or a public-domain statement, beside
                                  # text or a license-type that restricts commercial use, derivatives or derived licenses
    "cc by (words)",              # "Creative Commons Attribution" or "CC BY" in words, no link, nothing restricting
    "cc by-nc, by-nd or by-sa (words)",  # Creative Commons named in words, with restricting words
    "public domain",              # the record says that this article is in the public domain or is a US government work
    "general statement only",     # a general rule (for example, that federal employees' contributions are not under
                                  # copyright) and nothing about this article as a whole
    "verbatim copying only",      # BioMed Central's early license
    "other non-commercial or no-modification terms",
    "publisher's own reuse terms",  # Frontiers' license agreements; Elsevier's "subject to certain conditions"; Informa's
                                  # and Taylor & Francis's iOpenAccess terms, even beside Creative Commons words, since
                                  # those terms are a separate document that this check does not read
    "copyright notice only",
    "no statement",
)
CLEAN = ("cc by", "cc by (words)", "cc0", "public domain")

DASHES = re.compile("[­‐‑‒–—―−]")
# Permissive phrases are removed before the restricting words are searched, so that "even commercially" or
# "commercial and non-commercial" does not read as a restriction.
PERMISSIVE = re.compile(r"even commercially|allows for commercial use|build upon this work,? for commercial use|"
                        r"commercial (?:and|or) non-?\s?commercial|for any purpose,? (?:even|including) commercial(?:ly)?")
NC_WORDS = (r"non-?\s?commerci|noncommerical|not for commercial|commercial gain|not for profit|exception of commercial|"
            r"except for commercial|other than commercial|commercial exploitation|not permit commercial|"
            r"not (?:be )?used for commercial|used commercially|personal use")
ND_WORDS = r"no-?\s?deriv|unmodified|unchanged|not altered|cannot be changed|no modifications|verbatim"
SA_WORDS = r"share-?\s?alike"
CODE_WORDS = r"\bcc-?\s?by-?\s?(?:nc|nd|sa)\b|\bby-(?:nc|nd|sa)\b"
NC = re.compile(NC_WORDS + r"|academic research|educational")
ND = re.compile(ND_WORDS)
SA = re.compile(SA_WORDS)
RESTRICT = re.compile("|".join((NC_WORDS, ND_WORDS, SA_WORDS, CODE_WORDS)))
CC_LINK = re.compile(r"creativecommons\.org/licen[sc]es/\s*(by(?:-nc)?(?:-nd)?(?:-sa)?)(?![a-z-])")
# Correction 7: a dot followed by a non-space (inside a link such as creativecommons.org/publicdomain/zero/1.0/) does not
# end the waiver's sentence, so BioMed Central's "waiver (http://...) applies to the data" is read as a waiver for data.
DATA_WAIVER = re.compile(r"(?:waiver|dedication)(?:[^.]|\.(?=\S)){0,200}?appl(?:y|ies) to (?:the )?data")
# Correction 7: "U.S." is read as "us" before the public-domain rule, so "a U.S. Government work" reads as "a US government work".
US_DOTS = re.compile(r"\bu\.\s?s\.")
PD_ARTICLE = re.compile(
    r"publicdomain/mark|lies in the public domain|free of all copyright|"
    r"(?:this|the) (?:article|work|publication|material|manuscript|paper|report)[^.]{0,120}?"
    r"\b(?:is|are|lies|was|has been(?: placed)?)\b[^.]{0,60}?"
    r"(?:public domain|work of the (?:u\.?s\.?|united states)(?: federal)? government|"
    r"(?:us|united states)(?: federal)? government work|not subject to copyright)")
# Correction 7: the general rule about federal employees' contributions says nothing about the article as a whole, so its
# sentence is removed before the public-domain rule is tried.
GENERAL_RULE_SENTENCE = re.compile(r"[^.]*contributions of employees of the federal government[^.]*\.?")
GENERAL_RULE = re.compile(r"contributions of employees of the federal government|title 17 of the united states code|"
                          r"work of the united states government")


def text(e):
    return " ".join("".join(e.itertext()).split()) if e is not None else ""


def norm(t):
    """Lowercase, NFKC and every dash or hyphen as "-", whitespace collapsed."""
    return " ".join(DASHES.sub("-", unicodedata.normalize("NFKC", t or "")).lower().split())


def parse(xml_bytes):
    """{PMCID: the article's license statement} from one efetch response."""
    out = {}
    for art in ET.fromstring(xml_bytes).iter("article"):
        meta = art.find("front/article-meta")
        if meta is None:
            continue
        pmc = None
        for a in meta.findall("article-id"):
            if a.get("pub-id-type") in ("pmc", "pmcid") and a.text:
                pmc = "PMC" + a.text.strip().replace("PMC", "")
        if pmc is None:
            continue
        perm = meta.find("permissions")
        holder = perm if perm is not None else meta   # older records put these elements directly under article-meta
        lics = []
        for lic in holder.iter("license") if perm is not None else holder.findall("license"):
            refs = [text(x) for x in lic.iter() if x.tag.endswith("license_ref")]
            lics.append({"type": lic.get("license-type"), "href": lic.get(XLINK), "refs": refs, "text": text(lic)[:4000]})
        statements = holder.iter("copyright-statement") if perm is not None else holder.findall("copyright-statement")
        year = meta.find("pub-date/year")
        journal = text(art.find("front/journal-meta/journal-title-group/journal-title")) or text(art.find("front/journal-meta/journal-title"))
        out[pmc] = {"copyright": " ".join(text(c) for c in statements), "licenses": lics,
                    "no_permissions_element": perm is None, "journal": journal, "year": year.text if year is not None else None}
    return out


def statement(rec):
    """Every part of an article's statement, normalized: links, license_ref text, license text, license-type, copyright."""
    parts = []
    for l in rec.get("licenses", []):
        parts += [l.get("href") or "", " ".join(l.get("refs") or []), l.get("text") or "", "license-type " + (l.get("type") or "")]
    parts.append(rec.get("copyright") or "")
    return US_DOTS.sub("us", norm(" ".join(parts)))


def pd_article(allt):
    """The public-domain rule on the statement without the general rule's sentence (correction 7)."""
    return PD_ARTICLE.search(GENERAL_RULE_SENTENCE.sub(" ", allt))


def classify_one(rec):
    """The class of one article's statement."""
    if rec.get("missing"):
        return "no statement"
    allt = statement(rec)
    scan = PERMISSIVE.sub(" ", allt)
    restrict = RESTRICT.search(scan)
    type_restrict = any(re.search(r"by-?\s?(?:nc|nd|sa)|non-?\s?commercial|no-?\s?deriv", norm(l.get("type")))
                        for l in rec.get("licenses", []))
    codes = set(CC_LINK.findall(allt))
    zero_link = "publicdomain/zero" in allt and not DATA_WAIVER.search(allt)
    if codes:
        if codes == {"by"}:
            if restrict or type_restrict:
                return "open link or statement, restricting words in its text"
            if "iopenaccess" in allt:
                return "publisher's own reuse terms"
            return "cc by"
        return "cc " + codes.pop() if len(codes) == 1 else "conflicting license links"
    if zero_link:
        return "open link or statement, restricting words in its text" if (restrict or type_restrict) else "cc0"
    nc, nd, sa = NC.search(scan), ND.search(scan), SA.search(scan)
    if ("creative commons" in allt and "attribution" in allt) or re.search(r"\bcc-?\s?by\b", allt):
        if nc or nd or sa or type_restrict or re.search(CODE_WORDS, allt):
            return "cc by-nc, by-nd or by-sa (words)"
        if "iopenaccess" in allt:
            return "publisher's own reuse terms"
        return "cc by (words)"
    if (re.search(r"\bcc0\b|public domain dedication", allt) and not DATA_WAIVER.search(allt)) or pd_article(allt):
        if restrict or type_restrict:
            return "open link or statement, restricting words in its text"
        return "public domain" if pd_article(allt) else "cc0"
    if GENERAL_RULE.search(allt):
        return "general statement only"
    if "verbatim copying and redistribution" in allt:
        return "verbatim copying only"
    if nc or nd or type_restrict:
        return "other non-commercial or no-modification terms"
    if ("frontiersin.org/licenseagreement" in allt or "frontiers media" in allt or "subject to certain conditions" in allt
            or "iopenaccess" in allt):
        return "publisher's own reuse terms"
    if not allt.replace("license-type", "").strip():
        return "no statement"
    return "copyright notice only"


def europe_pmc():
    with open(LICENSES, newline="", encoding="utf-8") as f:
        return {r["pmcid"]: r["license"] for r in csv.DictReader(f)}


def article_prompts():
    """{split: {PMCID: number of prompts}} from session 8's records (article IDs only)."""
    out = {}
    for split in SPLITS:
        counts = collections.Counter()
        with open(RECORDS / f"evidence_effect_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)["group"]
                counts["PMC" + str(g["pmcid"] if isinstance(g, dict) else g).replace("PMC", "")] += 1
        out[split] = counts
    return out


def fetch(max_seconds=150.0, batch=10, pause=0.4):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    progress = RAW_DIR / "permissions.jsonl"
    done = set()
    if progress.exists():
        with open(progress, encoding="utf-8") as f:
            done = {json.loads(line)["pmcid"] for line in f if line.strip()}
    lic = europe_pmc()
    todo = [p for p in sorted(lic) if p not in done]
    print(f"articles {len(lic)}, done {len(done)}, to do {len(todo)}", flush=True)
    t0 = time.time()
    with open(progress, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            if time.time() - t0 > max_seconds:
                print("time limit reached; run again to resume", flush=True)
                return 1
            ids = todo[i:i + batch]
            url = EFETCH.format(ids=",".join(p.replace("PMC", "") for p in ids))
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=90) as r:
                        got = parse(r.read())
                    break
                except Exception as e:  # network errors are retried, then the run stops and resumes later
                    print(f"retry {attempt}: {e}", flush=True)
                    time.sleep(3 + 3 * attempt)
            else:
                print(f"failed: {ids}; run again to resume", flush=True)
                return 2
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for p in ids:
                rec = got.get(p, {"missing": True})
                rec.update({"pmcid": p, "read_utc": now})
                f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(pause)
    print("complete", flush=True)
    return 0


def classify():
    progress = RAW_DIR / "permissions.jsonl"
    with open(progress, encoding="utf-8") as f:
        recs = {r["pmcid"]: r for r in (json.loads(line) for line in f if line.strip())}
    lic = europe_pmc()
    missing = sorted(set(lic) - set(recs))
    if missing:
        raise SystemExit(f"{len(missing)} articles not fetched yet, for example {missing[:3]}; run fetch until it prints complete")
    prompts = article_prompts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(lic):
        c = classify_one(recs[p])
        splits = [s for s in SPLITS if prompts[s].get(p)]
        rows.append({"pmcid": p, "splits": " ".join(splits), "europe_pmc": lic[p], "own_statement": c,
                     "clean": "true" if c in CLEAN else "false", "year": recs[p].get("year") or "",
                     "prompts": sum(prompts[s].get(p, 0) for s in SPLITS)})
    with open(OUT_DIR / "license_recheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "source": "NCBI E-utilities efetch (db=pmc), each article's license statement; Europe PMC's license from "
                         "results/fiorillo_v0_4/article_licenses.csv (read 30 September 2026)",
               "read_utc": {"first": min(r.get("read_utc", "") for r in recs.values()), "last": max(r.get("read_utc", "") for r in recs.values())},
               "rule": "clean when the article's own statement is CC BY (any version, by link or in words), CC0 or the public "
                       "domain for the article itself, and no part of it (links, license-type, license text, copyright "
                       "statements) restricts commercial use, derivatives or the license of what is made from it; where "
                       "the parts disagree, the more restrictive part decides; a statement that names a publisher's own "
                       "terms (iOpenAccess) is not clean even beside Creative Commons words; CC BY-SA is not clean "
                       "(docs/prompt_session9.md, DECISIONS 3)",
               "classes": list(CLASSES), "clean_classes": list(CLEAN), "articles": len(rows),
               "permissions_sha256": hashlib.sha256(progress.read_bytes()).hexdigest(), "by_split": {}}
    for s in SPLITS:
        in_split = [r for r in rows if s in r["splits"].split()]
        n = prompts[s]
        summary["by_split"][s] = {
            "articles": len(in_split), "prompts": sum(n.values()),
            "clean_articles": sum(r["clean"] == "true" for r in in_split),
            "clean_prompts": sum(n[r["pmcid"]] for r in in_split if r["clean"] == "true"),
            "europe_pmc_clean_articles": sum(r["europe_pmc"] in ("cc by", "cc0") for r in in_split),
            "europe_pmc_clean_prompts": sum(n[r["pmcid"]] for r in in_split if r["europe_pmc"] in ("cc by", "cc0")),
            "europe_pmc_by_own_statement": {f"{a} | {b}": c for (a, b), c in sorted(collections.Counter(
                (r["europe_pmc"], r["own_statement"]) for r in in_split).items())},
            "clean_now_not_by_europe_pmc": sorted(r["pmcid"] for r in in_split if r["clean"] == "true" and r["europe_pmc"] not in ("cc by", "cc0")),
            "clean_by_europe_pmc_not_now": sorted(r["pmcid"] for r in in_split if r["clean"] != "true" and r["europe_pmc"] in ("cc by", "cc0")),
        }
    (OUT_DIR / "license_recheck.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for s in SPLITS:
        b = summary["by_split"][s]
        print(s, {k: b[k] for k in ("articles", "prompts", "clean_articles", "clean_prompts", "europe_pmc_clean_articles", "europe_pmc_clean_prompts")})
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--max-seconds", type=float, default=150.0)
    sub.add_parser("classify")
    args = p.parse_args(argv)
    return fetch(args.max_seconds) if args.command == "fetch" else classify()


if __name__ == "__main__":
    sys.exit(main())

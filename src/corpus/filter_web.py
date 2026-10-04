"""The web source for session 5 (docs/prompt_session5.md, THE TEXT, steps 2 and 7).

Reads the TheBlueScrubs-v1 files downloaded into data/raw/web/thebluescrubs/ by scripts/fetch_web_text_macos.sh, in
the order of src/configs/web_files_session5.txt, stopping at the first file on the list that is not on disk (files are
downloaded in list order). Field names come from src/configs/web_fields_session5.yaml, filled in step 7 from the
dataset's README; this script stops while any of them is unfilled. A field name may be a dotted path into a nested
object, and a nested object stored as a JSON string is parsed.

A document is kept only if all three ratings (medical relevance, precision and factual detail, safety and ethical
standards) are 5, its SlimPajama source is one of the configured names for Common Crawl, C4, Wikipedia and
StackExchange, and it passes the 13-word check (src/corpus/decontaminate.py). A row whose ratings, source, cancer label
or text is missing or unreadable is dropped and counted. Cancer-labeled documents are capped at one third of the web
tokens: a cancer-labeled document is taken only if, with it, the cancer-labeled tokens stay at or below half of the
other tokens, which keeps the share at or below one third at every point. Parts of 10,000 documents,
data/processed/corpus/session5/web/part-<nnnn>.jsonl.gz, are written until 550,000,000 tokens (Laya's tokenizer) or
the files run out. manifest.json records the parts, tokens, the cancer share, drops by reason, the files read and the
field configuration; a copy goes to results/reader_v1/text_manifests/web_manifest.json.

Resumable like build_abstract_parts.py (progress.json). Each call stops near --max-seconds, because the linked shell
ends every command after about three minutes: it first writes the documents gathered since the last part as a part of
their own (or, if there are none, saves only its position), so part sizes depend on where calls stopped, while the
documents and their order do not. When the list's downloaded files are used up, the documents after the last part
wait for the next download; run with --finish once no more files will be fetched, to write them as the last part.
    python3 src/corpus/filter_web.py [--max-seconds 150] [--finish]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "corpus"))
from decontaminate import ParallelChecker  # noqa: E402
from parts import (load_json, load_tokenizer, part_name, save_json, sha256_file, special_strings,  # noqa: E402
                   token_counts, write_part)

SESSION5 = ROOT / "data" / "processed" / "corpus" / "session5"
RATINGS = ("medical_relevance", "precision_and_factual_detail", "safety_and_ethical_standards")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("format", "text", "source", "cancer") if not cfg.get(k)]
    missing += [f"ratings.{k}" for k in RATINGS if not (cfg.get("ratings") or {}).get(k)]
    if not cfg.get("sources_kept") or any(v in (None, "") for v in cfg["sources_kept"]):
        missing.append("sources_kept")
    if cfg.get("cancer_true") in (None, "", []):
        missing.append("cancer_true")
    if missing:
        raise SystemExit(f"{path} is not filled in yet ({', '.join(missing)}); fill it from the dataset's README (step 7)")
    if cfg["format"] not in ("jsonl", "json_array"):
        raise SystemExit("format must be jsonl or json_array")
    if not isinstance(cfg["cancer_true"], list):
        cfg["cancer_true"] = [cfg["cancer_true"]]
    return cfg


class Unreadable(Exception):
    pass


def field(record, dotted):
    value = record
    for key in dotted.split("."):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                raise Unreadable(dotted)
        if not isinstance(value, dict) or key not in value or value[key] is None:
            raise Unreadable(dotted)
        value = value[key]
    return value


def rating(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise Unreadable("rating")
    return number


def judge(record, cfg):
    """(status, text, is_cancer): status is "keep" or the reason the row is dropped."""
    try:
        text = field(record, cfg["text"])
        ratings = [rating(field(record, cfg["ratings"][k])) for k in RATINGS]
        source = field(record, cfg["source"])
        cancer = field(record, cfg["cancer"])
    except Unreadable:
        return "unreadable_fields", None, False
    if not isinstance(text, str) or not text.strip():
        return "unreadable_fields", None, False
    if any(r != 5 for r in ratings):
        return "rating_below_5", None, False
    if str(source) not in {str(s) for s in cfg["sources_kept"]}:
        return "source_not_kept", None, False
    is_cancer = cancer in cfg["cancer_true"] or str(cancer) in {str(v) for v in cfg["cancer_true"]}
    return "keep", text, is_cancer


def read_rows(path, fmt):
    if fmt == "jsonl":
        with open(path, encoding="utf-8") as f:
            return [line for line in f if line.strip()]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON array")
    return data


def parse_row(row, fmt):
    if fmt == "jsonl":
        try:
            return json.loads(row)
        except ValueError:
            return None
    return row if isinstance(row, dict) else None


def accept_cancer(cancer_tokens, other_tokens, tokens):
    """A cancer-labeled document is taken only if cancer tokens stay at or below half the other tokens."""
    return 2 * (cancer_tokens + tokens) <= other_tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "web" / "thebluescrubs")
    parser.add_argument("--files", type=Path, default=ROOT / "src" / "configs" / "web_files_session5.txt")
    parser.add_argument("--config", type=Path, default=ROOT / "src" / "configs" / "web_fields_session5.yaml")
    parser.add_argument("--out", type=Path, default=SESSION5 / "web")
    parser.add_argument("--reference", type=Path, default=SESSION5 / "pubmedqa_labeled_text.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "tokenizers" / "laya" / "tokenizer.json")
    parser.add_argument("--manifests", type=Path, default=ROOT / "results" / "reader_v1" / "text_manifests")
    parser.add_argument("--target-tokens", type=int, default=550_000_000)
    parser.add_argument("--per-part", type=int, default=10_000)
    parser.add_argument("--chunk", type=int, default=5_000)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--finish", action="store_true", help="no more files will be downloaded: write what is left")
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    names = [n.strip() for n in args.files.read_text(encoding="utf-8").splitlines() if n.strip()]
    progress_path = args.out / "progress.json"
    inputs = {"config_sha256": sha256_file(args.config), "files_list_sha256": sha256_file(args.files),
              "reference_sha256": sha256_file(args.reference), "tokenizer_sha256": sha256_file(args.tokenizer)}
    state = load_json(progress_path)
    if state is None:
        state = {"inputs": inputs, "target_tokens": args.target_tokens, "per_part": args.per_part, "file_pos": 0, "row": 0,
                 "parts": [], "documents": 0, "tokens": 0, "cancer_tokens": 0, "other_tokens": 0,
                 "drops": {k: 0 for k in ("unparseable_row", "unreadable_fields", "rating_below_5", "source_not_kept",
                                           "thirteen_word", "cancer_cap")},
                 "files_read": [], "complete": False}
    elif state["inputs"] != inputs or state["target_tokens"] != args.target_tokens or state["per_part"] != args.per_part:
        raise SystemExit("progress.json was started with other inputs or settings; stop (nothing changed)")
    if state["complete"]:
        print("complete (nothing to do)")
        return

    tokenizer, specials = load_tokenizer(args.tokenizer), special_strings(args.tokenizer)
    fmt = cfg["format"]
    pos, row = state["file_pos"], state["row"]
    drops, files_read = dict(state["drops"]), [dict(f) for f in state["files_read"]]
    totals = {k: state[k] for k in ("tokens", "cancer_tokens", "other_tokens")}
    buffer, buffer_tokens = [], []
    part_start = time.time()

    def commit(next_pos, next_row, final=False):
        """Write the buffer as the next part and save the position just after its last row."""
        nonlocal buffer, buffer_tokens, part_start
        info = write_part(args.out / part_name(len(state["parts"])), buffer)
        info["tokens"] = int(sum(buffer_tokens))
        state["parts"].append(info)
        state["documents"] += len(buffer)
        state.update(totals)
        state.update({"file_pos": next_pos, "row": next_row, "drops": dict(drops),
                      "files_read": [dict(f) for f in files_read], "complete": bool(final)})
        save_json(progress_path, state)
        print(f"{info['file']}: {info['documents']:,} documents, {info['tokens']:,} tokens; total {state['tokens']:,} "
              f"({time.time() - t0:.0f} s)", flush=True)
        seconds = time.time() - part_start
        buffer, buffer_tokens, part_start = [], [], time.time()
        return seconds

    def save_position(next_pos, next_row):
        """Save the position and the counts without writing a part (used when no document is waiting)."""
        state.update(totals)
        state.update({"file_pos": next_pos, "row": next_row, "drops": dict(drops),
                      "files_read": [dict(f) for f in files_read]})
        save_json(progress_path, state)

    with ParallelChecker(args.reference, args.processes) as checker:
        while not state["complete"]:
            if pos >= len(names) or not (args.raw / names[pos]).exists():
                break
            rows = read_rows(args.raw / names[pos], fmt)
            if not files_read or files_read[-1]["file"] != names[pos]:
                files_read.append({"file": names[pos], "rows": len(rows), "rows_read": 0, "documents_written": 0})
            while row < len(rows) and not state["complete"]:
                chunk = range(row, min(len(rows), row + args.chunk))
                judged = []
                for i in chunk:
                    record = parse_row(rows[i], fmt)
                    judged.append(("unparseable_row", None, False) if record is None else judge(record, cfg))
                keep = [k for k, (status, _, _) in enumerate(judged) if status == "keep"]
                flags = dict(zip(keep, checker.check([judged[k][1] for k in keep])))
                clean = [k for k in keep if not flags[k]]
                counts = dict(zip(clean, token_counts(tokenizer, [judged[k][1] for k in clean], specials)))
                for k, i in enumerate(chunk):
                    row = i + 1
                    files_read[-1]["rows_read"] += 1
                    status, text, is_cancer = judged[k]
                    if status != "keep":
                        drops[status] += 1
                        continue
                    if flags[k]:
                        drops["thirteen_word"] += 1
                        continue
                    n = counts[k]
                    if is_cancer and not accept_cancer(totals["cancer_tokens"], totals["other_tokens"], n):
                        drops["cancer_cap"] += 1
                        continue
                    totals["cancer_tokens" if is_cancer else "other_tokens"] += n
                    totals["tokens"] += n
                    buffer.append((f"{names[pos]}:{i}", text))
                    buffer_tokens.append(n)
                    files_read[-1]["documents_written"] += 1
                    done = totals["tokens"] >= args.target_tokens
                    if len(buffer) == args.per_part or done:
                        next_pos, next_row = (pos + 1, 0) if row >= len(rows) else (pos, row)
                        seconds = commit(next_pos, next_row, final=done)
                        if done:
                            break
                        if time.time() - t0 + 1.2 * seconds > args.max_seconds:
                            print("stopping here to stay inside the time limit; run again to continue")
                            return
                if not state["complete"] and time.time() - t0 > args.max_seconds:
                    next_pos, next_row = (pos + 1, 0) if row >= len(rows) else (pos, row)
                    if buffer:
                        commit(next_pos, next_row)
                    else:
                        save_position(next_pos, next_row)
                    print("stopping here to stay inside the time limit; run again to continue")
                    return
            pos, row = pos + 1, 0
    if not state["complete"]:
        if args.finish and buffer:
            commit(pos, row, final=True)
        elif args.finish:
            state["complete"] = True
            save_position(pos, row)
        else:
            if not buffer:
                save_position(pos, row)
            print(f"downloaded files used up at {totals['tokens']:,} tokens (the last {len(buffer):,} documents wait for "
                  "the next part); fetch more with --more, or run with --finish if no more files will be fetched")
            return
    manifest = {
        "source": "TheBlueScrubs/TheBlueScrubs-v1 at revision 255bf6f5eb57e6d8ce4feaa94df0b27a5d13f508",
        "rules": "all three ratings 5; SlimPajama source Common Crawl, C4, Wikipedia or StackExchange; cancer-labeled "
                 "tokens at most one third; 13-word check; until 550,000,000 tokens or the files run out",
        "documents": state["documents"], "tokens": state["tokens"], "cancer_tokens": state["cancer_tokens"],
        "cancer_share": round(state["cancer_tokens"] / max(1, state["tokens"]), 4), "drops": state["drops"],
        "parts": state["parts"], "files_read": state["files_read"], "fields": cfg, "inputs": state["inputs"],
    }
    save_json(args.out / "manifest.json", manifest)
    save_json(args.manifests / "web_manifest.json", manifest)
    print(f"complete: {state['documents']:,} documents, {state['tokens']:,} tokens, cancer share {manifest['cancer_share']}")


if __name__ == "__main__":
    sys.exit(main())

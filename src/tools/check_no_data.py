#!/usr/bin/env python3
"""Pre-commit data guard for ed-decision-model.

Installed as .git/hooks/pre-commit. Blocks the commit if any staged file:
  - sits under data/ or models/,
  - has a data or weight extension: .csv .csv.gz .parquet .duckdb .jsonl .safetensors .pt .bin,
  - or is larger than 1 MB.
Files under src/tests/fixtures/ are exempt from the path and extension rules and may be up to 25 MB.
Files under results/ (aggregate tables only) are exempt from the extension rule but keep the 1 MB limit.
Sizes are read from the staged content, not from the working copy.
Standard library only, so it runs under any python3.
"""
import subprocess
import sys

MB = 1024 * 1024
DEFAULT_LIMIT = 1 * MB
FIXTURE_LIMIT = 25 * MB
FIXTURE_PREFIX = "src/tests/fixtures/"
RESULTS_PREFIX = "results/"
BLOCKED_DIRS = ("data/", "models/")
BLOCKED_EXTENSIONS = (".csv", ".csv.gz", ".parquet", ".duckdb", ".jsonl", ".safetensors", ".pt", ".bin")


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def staged_files():
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def staged_size(path):
    return int(git("cat-file", "-s", ":" + path).decode().strip())


def problems_for(path, size):
    if path.startswith(FIXTURE_PREFIX):
        if size > FIXTURE_LIMIT:
            return [f"fixture larger than 25 MB ({size / MB:.1f} MB)"]
        return []
    problems = []
    if path.startswith(BLOCKED_DIRS):
        problems.append("under data/ or models/")
    if path.lower().endswith(BLOCKED_EXTENSIONS) and not path.startswith(RESULTS_PREFIX):
        problems.append("data or model-weight file extension")
    if size > DEFAULT_LIMIT:
        problems.append(f"larger than 1 MB ({size / MB:.1f} MB)")
    return problems


def main():
    blocked = []
    for path in staged_files():
        problems = problems_for(path, staged_size(path))
        if problems:
            blocked.append((path, problems))
    if blocked:
        print("Commit blocked by src/tools/check_no_data.py:", file=sys.stderr)
        for path, problems in blocked:
            print(f"  {path}: {'; '.join(problems)}", file=sys.stderr)
        print("Restricted data and model weights never go into this repository. See docs/data_rules.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

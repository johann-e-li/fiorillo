#!/usr/bin/env bash
# Fiorillo v0.5 on Hugging Face as a PRIVATE repository (docs/prompt_session9.md, THE RELEASE and step 17). Run by Johann
# only, by hand, in Terminal on the project Mac, after step 17 has built the release folder and its SHA256SUMS.txt:
#     bash ~/Documents/Research/ed-decision-model/scripts/release_fiorillo_v0_5.sh [RELEASE_FOLDER]
# RELEASE_FOLDER defaults to models/fiorillo_v0_5/release.
# It stops unless: the release folder holds SHA256SUMS.txt (lines "<SHA-256>  <file>", paths relative to the folder);
# every file listed there exists in the folder with that SHA-256; every other file in the folder is listed (only
# SHA256SUMS.txt itself is not), so no unchecked file such as .DS_Store is uploaded; the folder holds no hidden file and no
# symbolic link; and the Hugging Face command line (hf) is signed in as johanneli. Then it creates johanneli/fiorillo-v0.5
# as a private repository (an existing repository is used only if it is private and holds no file other than
# .gitattributes and the release folder's files), uploads the folder, checks on Hugging Face that the repository is still
# private and holds every file of the folder with the same size and content (the SHA-256 of a file stored with Git LFS,
# the git blob ID of any other), and prints the revision and the page to review, as scripts/release_reader_v1.sh does.
# It never makes anything public: that is done by hand on the website, after Claude has read the private page and checked
# its files against SHA256SUMS.txt. It never reads or writes a token: every Hugging Face call is the hf command
# (huggingface_hub 1.33.0, as in uv.lock), which uses its own login (hf auth login), and no call passes --token. It
# writes nothing and deletes nothing.
#
# EDDM_PYTHON and EDDM_HF name the Python and the hf command (default: uv run python and uv run hf, the project's own);
# the tests set them to the linked shell's python3 and a stand-in hf, so that nothing reaches Hugging Face.
set -euo pipefail
cd "$HOME/Documents/Research/ed-decision-model" 2>/dev/null || cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
PY="${EDDM_PYTHON:-uv run python}"
HF="${EDDM_HF:-uv run hf}"
REPO="johanneli/fiorillo-v0.5"
REL="${1:-models/fiorillo_v0_5/release}"
PAGE="https://huggingface.co/$REPO"

if [ ! -d "$REL" ]; then echo "Stop: $REL does not exist (step 17 builds it)."; exit 1; fi
if [ ! -f "$REL/SHA256SUMS.txt" ]; then echo "Stop: $REL/SHA256SUMS.txt does not exist; nothing uploaded."; exit 1; fi

$PY - "$REL" <<'PY'
import hashlib, os, re, sys
from pathlib import Path
rel = Path(sys.argv[1])
listed = {}
for n, line in enumerate((rel / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line.rstrip("\r\n"))
    if not m:
        sys.exit(f"Stop: SHA256SUMS.txt line {n} is not '<SHA-256>  <file>': {line[:100]!r}; nothing uploaded.")
    name = m.group(2)[2:] if m.group(2).startswith("./") else m.group(2)
    parts = Path(name).parts
    if Path(name).is_absolute() or ".." in parts or not parts:
        sys.exit(f"Stop: SHA256SUMS.txt names a file outside the release folder: {name}; nothing uploaded.")
    if name in listed:
        sys.exit(f"Stop: SHA256SUMS.txt lists {name} twice; nothing uploaded.")
    listed[name] = m.group(1).lower()
present, links, hidden = {}, [], []
for root, dirs, files in os.walk(rel):
    for d in dirs:
        if (Path(root) / d).is_symlink():
            links.append((Path(root) / d).relative_to(rel).as_posix())
    for f in files:
        p = Path(root) / f
        name = p.relative_to(rel).as_posix()
        if p.is_symlink():
            links.append(name)
        if any(part.startswith(".") for part in Path(name).parts):
            hidden.append(name)
        present[name] = p
if links:
    sys.exit(f"Stop: the release folder holds symbolic links {links[:5]}; nothing uploaded.")
if hidden:
    sys.exit(f"Stop: the release folder holds hidden files {hidden[:5]}; move them aside; nothing uploaded.")
present.pop("SHA256SUMS.txt", None)
missing = sorted(set(listed) - set(present))
extra = sorted(set(present) - set(listed))
if missing:
    sys.exit(f"Stop: SHA256SUMS.txt lists files the release folder does not hold: {missing[:5]}; nothing uploaded.")
if extra:
    sys.exit(f"Stop: the release folder holds files SHA256SUMS.txt does not list: {extra[:5]}; nothing uploaded.")
bad = []
for name in sorted(listed):
    h = hashlib.sha256()
    with open(present[name], "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != listed[name]:
        bad.append(name)
if bad:
    sys.exit(f"Stop: these files differ from SHA256SUMS.txt: {bad[:10]}{' and more' if len(bad) > 10 else ''}; nothing uploaded.")
print(f"The release folder matches SHA256SUMS.txt: {len(listed)} files.")
PY

if ! WHO="$($HF auth whoami --format json)"; then
  echo "Stop: the Hugging Face command line is not signed in; run 'uv run hf auth login' first. Nothing uploaded."; exit 1
fi
USER_NAME="$($PY -c 'import json, sys; print((json.loads(sys.argv[1] or "{}") or {}).get("user") or "")' "$WHO")"
if [ "$USER_NAME" != "${REPO%%/*}" ]; then
  echo "Stop: the Hugging Face command line is signed in as '$USER_NAME', not ${REPO%%/*}. Nothing uploaded."; exit 1
fi
printf 'Signed in as %s; repository %s (private).\n' "$USER_NAME" "$REPO"

$HF repos create "$REPO" --private --exist-ok >/dev/null
INFO="$($HF models info "$REPO" --format json)"
$PY - "$REL" "$REPO" "$INFO" <<'PY'
import json, sys
from pathlib import Path
rel, repo, info = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
if info.get("private") is not True:
    sys.exit(f"Stop: {repo} exists and is not private; nothing uploaded.")
local = {p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file()}
foreign = sorted(s["rfilename"] for s in info.get("siblings") or [] if s["rfilename"] != ".gitattributes" and s["rfilename"] not in local)
if foreign:
    sys.exit(f"Stop: {repo} already holds files that are not in the release folder ({foreign[:5]}); nothing uploaded.")
print(f"{repo} exists and is private.")
PY

$HF upload "$REPO" "$REL" . --private --commit-message "Fiorillo v0.5 (ed-decision-model, session 9)"

INFO="$($HF models info "$REPO" --format json)"
FILES="$($HF models ls "$REPO" -R --format json)"
REVISION="$($PY - "$REL" "$REPO" "$INFO" "$FILES" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
rel, repo, info, files = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
sums = {}
for line in (rel / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line.rstrip("\r\n"))
    if m:
        sums[m.group(2)[2:] if m.group(2).startswith("./") else m.group(2)] = m.group(1).lower()
remote = {f["path"]: f for f in files if isinstance(f, dict) and "size" in f}
bad = [] if info.get("private") is True else ["the repository is not private"]
for p in sorted(q for q in rel.rglob("*") if q.is_file()):
    name = p.relative_to(rel).as_posix()
    r = remote.get(name)
    if r is None:
        bad.append(f"{name} is not on Hugging Face")
        continue
    data_size = p.stat().st_size
    if r.get("size") != data_size:
        bad.append(f"{name}: size {r.get('size')} on Hugging Face, {data_size} here")
    lfs = r.get("lfs") or {}
    if lfs.get("sha256"):
        want = sums.get(name) or hashlib.sha256(p.read_bytes()).hexdigest()
        if lfs["sha256"] != want:
            bad.append(f"{name}: SHA-256 {lfs['sha256']} on Hugging Face")
    elif r.get("blob_id"):
        blob = hashlib.sha1(b"blob %d\0" % data_size + p.read_bytes()).hexdigest()
        if r["blob_id"] != blob:
            bad.append(f"{name}: blob {r['blob_id']} on Hugging Face, {blob} here")
extra = sorted(n for n in remote if n != ".gitattributes" and not (rel / n).is_file())
bad += [f"on Hugging Face and not in the release folder: {n}" for n in extra]
if bad:
    print(f"Check failed on Hugging Face: {bad[:10]}", file=sys.stderr)
    sys.exit(1)
print(info.get("sha") or "unknown")
PY
)"
printf 'On Hugging Face at revision %s: every file of the release folder is there with the same size and content, and the repository is private.\n' "$REVISION"
printf '\nUploaded to the private repository %s (revision %s).\nReview it there; make it public by hand only when it is right.\nPaste everything above into the chat so that it can be recorded.\n' "$PAGE" "$REVISION"

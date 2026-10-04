"""Upload files to the session 5 pod through its Jupyter server, and check each one there (docs/prompt_session5.md,
step 5). Run from the linked shell with its own python3; websocket-client is installed there with pip install --user
(docs/log.md).

The pod's Jupyter server is reached at its Runpod proxy address (https://<pod id>-8888.proxy.runpod.net) with the pod's
Jupyter token, read from the EDDM_POD_TOKEN environment variable (or --token); it is the pod's token, never a Hugging
Face token, and it is not printed. Files go up through the Jupyter Contents API in base64 chunks (the API's chunked
upload: chunk 1 creates the file, later chunks append, the last is numbered -1), into a staging folder, eddm_upload/,
under the server's root folder. A kernel on the pod then checks each file's SHA-256 against the local file's and moves
it into place under --remote-root; a mismatch leaves the staged copy where it is and is reported. A file already in
place with the same SHA-256 is skipped, so an interrupted upload resumes when the same command is run again. Each call
stops starting new files after --max-seconds, because the linked shell ends commands after about three minutes.

Session 7 (docs/prompt_session7.md, step 3): --driver takes the driver's absolute path on the pod (default session 5's,
/root/eddm_s5/code/scripts/pod_session5.sh, so session 5's use is unchanged), and the log folder follows from it (the
driver's base folder, then logs/); session 7's stages (pilot, labeler, arm) are accepted; and a file interrupted part way
through resumes: its staging name is fixed by its SHA-256, its staged size is read on the pod, and sending continues from
there; each call stops after --max-seconds, even in the middle of a file, but always starts at least one file and sends
at least one chunk; a staged copy whose SHA-256 does not match is renamed with ".mismatch" added, so the next call sends
that file again from its start.

Session 8 (docs/prompt_session8.md, step 3, and correction 5): session 8's stages (ei, arm_a, trees, prepare_large,
large_validation and calibrate, beside the names sessions 5 and 7 already use) are accepted, with --driver
/root/eddm_s8/code/scripts/pod_session8.sh; and after each upload call the pod's copy of a manifest, <driver's base
folder>/uploads_manifest.json ({"roots": {<remote root>: {<relative path>: <sha256>}}}), gains every file of the call
that is in place with the right SHA-256 (placed now or already there), so that the pod's prepare stage can check every
upload again before it starts.

Session 9 (docs/prompt_session9.md, step 3): session 9's stages are accepted (checks_s8, pilot_long, train, tabicl, biogpt
and all, beside the names sessions 5 to 8 already use, among them prepare, trees, prepare_large, smoke_large,
large_validation, calibrate, freeze, testpreds and package), with --driver /root/eddm_s9/code/scripts/pod_session9.sh; the
log folder (/root/eddm_s9/logs) and the uploads manifest (/root/eddm_s9/uploads_manifest.json) follow from it as before.
The upload commands of step 9 are printed by `bash scripts/pod_session9.sh upload_list` on the linked shell.

Two more commands keep the pod driven only through the driver script (scripts/pod_session5.sh, or the one --driver
names): `run` starts one of the driver's subcommands on the pod (the driver detaches itself with nohup and returns at
once), and `log` prints the end of a stage's log.

Usage, from the repository root on the linked shell:
  EDDM_POD_TOKEN=... python3 src/tools/upload_to_pod.py --url https://<pod>-8888.proxy.runpod.net upload \
      --local-root . --remote-root /root/eddm_s5/code --list <file of relative paths>
  EDDM_POD_TOKEN=... python3 src/tools/upload_to_pod.py --url ... upload --local-root data/processed/corpus/session5 \
      --remote-root /root/eddm_s5/text abstracts/manifest.json abstracts/part-0000.jsonl.gz
  EDDM_POD_TOKEN=... python3 src/tools/upload_to_pod.py --url ... run prepare
  EDDM_POD_TOKEN=... python3 src/tools/upload_to_pod.py --url ... log prepare [--lines 40]
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

DRIVER = "/root/eddm_s5/code/scripts/pod_session5.sh"
DRIVER_COMMANDS = ("prepare", "timing", "decision", "mlm", "task", "long", "final", "freeze", "testpreds", "package",
                   "select", "text", "pilot", "labeler", "arm", "prepare_a", "correct_phase3",
                   "ei", "arm_a", "trees", "prepare_large", "large_validation", "calibrate", "smoke_large",
                   "checks_s8", "pilot_long", "train", "tabicl", "biogpt", "all")
STAGING = "eddm_upload"


def log_dir(driver):
    """The driver's base folder, then logs/ (/root/eddm_s7/code/scripts/pod_session7.sh gives /root/eddm_s7/logs)."""
    return str(Path(driver).parents[2] / "logs")


def manifest_path(driver):
    """The pod's uploads manifest, beside the driver's base folder (/root/eddm_s8/uploads_manifest.json)."""
    return str(Path(driver).parents[2] / "uploads_manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Pod:
    def __init__(self, url, token, timeout=300):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"token {token}"
        self.kernel_id, self.ws = None, None

    def rest(self, method, path, **kw):
        r = self.session.request(method, f"{self.url}{path}", timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path}: HTTP {r.status_code} {r.text[:300]}")
        return r.json() if r.content else None

    # the kernel: run Python on the pod and collect what it prints
    def _connect(self):
        import websocket

        if self.ws is None:
            if self.kernel_id is None:
                self.kernel_id = self.rest("POST", "/api/kernels", json={"name": "python3", "path": ""})["id"]
            ws_url = self.url.replace("https://", "wss://").replace("http://", "ws://") + f"/api/kernels/{self.kernel_id}/channels"
            self.ws = websocket.create_connection(ws_url, header=[f"Authorization: token {self.token}"], timeout=self.timeout)
            self.session_id = uuid.uuid4().hex

    def python(self, code, timeout=600):
        """Runs code in the kernel and returns what it printed. The proxy closes a websocket left idle for about 100
        seconds (as during a long upload), so a closed connection is opened again to the same kernel, once."""
        import websocket

        try:
            return self._python_once(code, timeout)
        except TimeoutError:
            raise
        except (OSError, ValueError, EOFError, websocket.WebSocketException) as error:  # a closed websocket
            print(f"the connection to the pod's kernel was closed ({type(error).__name__}); connecting again", flush=True)
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
            return self._python_once(code, timeout)

    def _python_once(self, code, timeout):
        self._connect()
        msg_id = uuid.uuid4().hex
        self.ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "eddm", "session": self.session_id, "msg_type": "execute_request",
                       "version": "5.3", "date": dt.datetime.now(dt.timezone.utc).isoformat()},
            "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
            "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False,
                        "stop_on_error": True}}))
        out, deadline = [], time.time() + timeout
        while time.time() < deadline:
            frame = self.ws.recv()
            if not frame:
                raise EOFError("the websocket was closed")
            msg = json.loads(frame)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind = msg.get("msg_type") or msg["header"]["msg_type"]
            if kind == "stream":
                out.append(msg["content"]["text"])
            elif kind == "error":
                raise RuntimeError("error on the pod: " + "\n".join(msg["content"].get("traceback", []))[-2000:])
            elif kind == "status" and msg["content"]["execution_state"] == "idle":
                return "".join(out)
        raise TimeoutError("the pod did not finish in time")

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.rest("DELETE", f"/api/kernels/{self.kernel_id}")

    # the Contents API: staged, chunked uploads
    def mkdirs(self, api_dir):
        parts = [p for p in api_dir.split("/") if p]
        for i in range(1, len(parts) + 1):
            path = "/".join(parts[:i])
            r = self.session.get(f"{self.url}/api/contents/{path}", params={"content": 0}, timeout=self.timeout)
            if r.status_code == 404:
                self.rest("PUT", f"/api/contents/{path}", json={"type": "directory"})

    def put_file(self, local, api_path, chunk_bytes, staged_size=0, deadline=None):
        """Sends local to api_path in base64 chunks, continuing after staged_size bytes already on the pod (0 starts over).
        Returns True when the whole file is there, False if the deadline stopped it part way."""
        size = os.path.getsize(local)
        name = api_path.rsplit("/", 1)[-1]
        with open(local, "rb") as f:
            if staged_size <= 0 and size <= chunk_bytes:
                self.rest("PUT", f"/api/contents/{api_path}", json={"type": "file", "format": "base64", "name": name,
                                                                     "path": api_path, "content": base64.b64encode(f.read()).decode()})
                return True
            sent = max(0, staged_size)
            f.seek(sent)
            number = 1 if sent == 0 else 2
            while sent < size:
                if deadline is not None and time.time() > deadline and sent > staged_size:
                    return False
                block = f.read(chunk_bytes)
                sent += len(block)
                chunk = -1 if sent >= size else number
                self.rest("PUT", f"/api/contents/{api_path}", json={"type": "file", "format": "base64", "name": name, "path": api_path,
                                                                     "content": base64.b64encode(block).decode(), "chunk": chunk})
                number += 1
        return True


CHECK = r'''
import hashlib, json, os
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
wanted = json.loads(WANTED)
print(json.dumps({dest: (os.path.exists(dest) and sha(dest) == digest) for dest, digest in wanted.items()}))
'''

PLACE = r'''
import hashlib, json, os
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
staged, dest, digest = STAGED, DEST, DIGEST
got = sha(staged)
if got == digest:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(staged, dest)
else:
    os.replace(staged, staged + ".mismatch")  # kept for a look; the next run sends the file again from its start
print(json.dumps({"dest": dest, "sha256_ok": got == digest, "sha256": got}))
'''


MANIFEST = r'''
import json, os
path, root, entries = PATH, ROOT, json.loads(ENTRIES)
data = json.load(open(path)) if os.path.exists(path) else {"roots": {}}
data["roots"].setdefault(root, {}).update(entries)
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=1, sort_keys=True)
os.replace(tmp, path)
print(json.dumps({"manifest": path, "root": root, "files": len(data["roots"][root])}))
'''


def record_manifest(pod, driver, remote_root, entries):
    """Adds {rel: sha256} for the files in place under remote_root to the pod's uploads manifest."""
    if not entries:
        return None
    return pod.python(MANIFEST.replace("PATH", repr(manifest_path(driver))).replace("ROOT", repr(remote_root.rstrip("/")))
                      .replace("ENTRIES", repr(json.dumps(entries))))


def upload(pod, local_root, remote_root, rel_paths, chunk_mb, max_seconds, driver=None):
    t0 = time.time()
    pod.mkdirs(STAGING)
    probe = f"{STAGING}/probe-{uuid.uuid4().hex[:8]}"
    pod.rest("PUT", f"/api/contents/{probe}", json={"type": "file", "format": "text", "content": "probe"})
    root = pod.python(
        "import os\ncands = [os.getcwd(), '/', '/workspace', os.path.expanduser('~'), '/root']\n"
        f"hits = [c for c in cands if os.path.exists(os.path.join(c, {probe!r}))]\nprint(hits[0] if hits else '')").strip()
    if not root:
        raise SystemExit("the Jupyter root folder could not be found from the kernel; stop")
    files = [(rel, Path(local_root) / rel) for rel in rel_paths]
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"not found locally: {missing[:5]}")
    digests = {rel: sha256_file(p) for rel, p in files}
    wanted = {f"{remote_root.rstrip('/')}/{rel}": digests[rel] for rel, _ in files}
    present = json.loads(pod.python(CHECK.replace("WANTED", repr(json.dumps(wanted)))))
    done, skipped, failed, partial = [], [], [], []
    for rel, local in files:
        dest = f"{remote_root.rstrip('/')}/{rel}"
        if present.get(dest):
            skipped.append(rel)
            continue
        if (done or failed) and time.time() - t0 > max_seconds:  # every call starts at least one file, so each call makes progress
            break
        # the staging name is fixed by the file's SHA-256, so an interrupted file resumes where it stopped
        api_path = f"{STAGING}/{digests[rel][:16]}-{rel.replace('/', '__')}"
        staged = int(pod.python(f"import os\np = {os.path.join(root, api_path)!r}\nprint(os.path.getsize(p) if os.path.exists(p) else 0)").strip() or 0)
        if staged > local.stat().st_size:
            staged = 0
        started = time.time()
        complete = pod.put_file(local, api_path, int(chunk_mb * 2**20), staged, t0 + max_seconds)
        if not complete:
            now_staged = int(pod.python(f"import os\np = {os.path.join(root, api_path)!r}\nprint(os.path.getsize(p) if os.path.exists(p) else 0)").strip() or 0)
            print(f"{rel}: stopped part way, {now_staged / 2**20:.1f} of {local.stat().st_size / 2**20:.1f} MB on the pod; run again to resume",
                  flush=True)
            partial.append(rel)
            break
        result = json.loads(pod.python(PLACE.replace("STAGED", repr(os.path.join(root, api_path)))
                                       .replace("DEST", repr(dest)).replace("DIGEST", repr(digests[rel]))))
        mb = local.stat().st_size / 2**20
        rate = mb / max(1e-6, time.time() - started)
        print(f"{rel}: {mb:.1f} MB, {rate:.1f} MB/s, SHA-256 {'checked' if result['sha256_ok'] else 'MISMATCH'}", flush=True)
        (done if result["sha256_ok"] else failed).append(rel)
    left = len(files) - len(done) - len(skipped) - len(failed) - len(partial)
    if driver:
        in_place = {rel: digests[rel] for rel in done + skipped}
        if in_place:
            print(record_manifest(pod, driver, remote_root, in_place).strip(), flush=True)
    print(json.dumps({"uploaded_and_checked": len(done), "already_in_place": len(skipped), "sha256_mismatch": failed,
                      "part_way": partial, "not_started_yet": left, "seconds": round(time.time() - t0, 1)}))
    return left == 0 and not failed and not partial


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="the pod's Jupyter address, https://<pod id>-8888.proxy.runpod.net")
    parser.add_argument("--token", default=os.environ.get("EDDM_POD_TOKEN"), help="the pod's Jupyter token (or EDDM_POD_TOKEN)")
    parser.add_argument("--driver", default=DRIVER, help="the driver's absolute path on the pod (default session 5's)")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--local-root", type=Path, required=True)
    up.add_argument("--remote-root", required=True)
    up.add_argument("--list", type=Path, help="a file of paths relative to --local-root, one per line")
    up.add_argument("paths", nargs="*")
    up.add_argument("--chunk-mb", type=float, default=8.0)
    up.add_argument("--max-seconds", type=float, default=150.0)
    run = sub.add_parser("run")
    run.add_argument("driver_args", nargs="+")
    log = sub.add_parser("log")
    log.add_argument("stage")
    log.add_argument("--lines", type=int, default=40)
    args = parser.parse_args()
    if not args.token:
        parser.error("the pod's Jupyter token is needed (EDDM_POD_TOKEN)")
    pod = Pod(args.url, args.token)
    try:
        if args.command == "upload":
            rel = list(args.paths)
            if args.list:
                rel += [line.strip() for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
            ok = upload(pod, args.local_root, args.remote_root, rel, args.chunk_mb, args.max_seconds,
                        args.driver if args.driver != DRIVER else None)
            return 0 if ok else 2
        if args.command == "run":
            if args.driver_args[0] not in DRIVER_COMMANDS:
                parser.error(f"{args.driver_args[0]} is not a driver subcommand")
            code = ("import subprocess; r = subprocess.run(['bash', %r] + %r, capture_output=True, text=True); "
                    "print(r.stdout + r.stderr)") % (args.driver, list(args.driver_args))
            print(pod.python(code, timeout=120))
            return 0
        code = ("import os\np = '%s/%s.log'\n"
                "print(''.join(open(p, encoding='utf-8', errors='replace').readlines()[-%d:]) if os.path.exists(p) else 'no log ' + p)"
                % (log_dir(args.driver), args.stage, args.lines))
        print(pod.python(code, timeout=60))
        return 0
    finally:
        pod.close()


if __name__ == "__main__":
    sys.exit(main())

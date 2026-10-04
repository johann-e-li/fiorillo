"""BioGPT-Large-PubMedQA on PubMedQA (docs/prompt_session9.md, PUBMEDQA BESIDE BIOGPT-LARGE, with correction 5).

Descriptive, for comparison K5. BioGPT-Large-PubMedQA (microsoft/BioGPT-Large-PubMedQA at the revision fixed in
results/fiorillo_v0_5/fixed_versions.json, MIT on its card) answers PubMedQA as the system biogpt_large_pqa: PubMedQA's
150 calibrate validation items in the biogpt stage, and the 500 test items only after the freeze, in the testpreds
stage. It answers PubMedQA only. Its fine-tuning used PQA-L's fold 0 training and development sets, which hold every one
of this project's 150 validation items (src/tasks/pubmedqa.py), so its primary result is its uncalibrated probabilities
(renormalized over the three answers); calibrate_session9.py fits session 8's rule on the 150 items, reported beside
them with that caveat. Nothing here calibrates.

Correction 5, the checkpoint's own arguments. Before it scores any item, the validation command reads the fine-tuned
checkpoint's training arguments by streaming the start of the archive that microsoft/BioGPT's README links
(QA-PubMedQA-BioGPT-Large.tgz). The gzip is decompressed on the fly; the tar headers are walked (ustar name prefixes,
GNU long names, pax headers and GNU's base-256 sizes) to the first regular member whose name ends in ".pt" (AppleDouble
"._" files skipped); and the checkpoint's pickle is read from the start of that member. In the zip format of torch.save,
the zip local headers are walked to the entry whose name ends in "data.pkl", which is unpickled from the stream up to
its STOP opcode (torch writes that entry first, and torch 2.14, checked here, gives its size only in a data descriptor
after it; zip64 sizes and deflated entries are read too). In torch's legacy format, the pickles at the member's start
are read in turn (the magic number, the protocol version, the system information, then the checkpoint). The unpickler
imports nothing: each tensor storage becomes a placeholder whose bytes are never read, and every class or function the
pickle names (tensors, fairseq's meters, omegaconf's containers and nodes, argparse.Namespace) becomes an inert stand-in
that keeps its arguments and state, except OrderedDict and a few builtins. At most 1 GB (10^9 bytes as served) is read
and 10 minutes spent, network failures are retried within them, and nothing is saved; the archive's address is recorded
without its query string (the read-only signature), as fixed_versions.json records it. learned_prompt,
learned_prompt_pattern, prefix, manual_prompt, max_source_positions, max_target_positions and tokens_per_sample are
taken from the checkpoint's cfg (its task section, then its model section) or from its legacy args, and recorded in
biogpt.json with the time. Where they differ from fixed_versions.json's values, the checkpoint's are used (the learned
tokens' count and pattern, or a manual prompt; the order; the source's cut) and the difference is recorded; if they
cannot be read within the limits, fixed_versions.json's values are used and that is recorded. The test command uses the
settings that the validation command resolved (biogpt.json), so that both splits read the same input; it reads nothing
from the network.

The input for one item (microsoft/BioGPT at commit 648f7d65: examples/QA-PubMedQA/rebuild_data.py, preprocess_large.sh,
infer_large.sh and postprocess.py; inference.py; src/language_modeling_prompt.py and src/language_model_prompt_dataset.py):
- The source, for the reasoning-required setting (rebuild_data.py, loader and build_source_seq without the long answer):
  "question: {q} context: {c} ", q the question stripped, and c the contexts each stripped, joined with single spaces,
  newlines replaced by spaces, runs of whitespace collapsed to one space, and stripped. The records hold the question and
  the contexts as a list in source order (state.question; state.contexts[].text), so the source is rebuilt from them
  directly: on 1 October 2026 the 150 calibrate validation records gave the same 150 sources as the QUESTION and CONTEXTS
  of data/raw/public/pubmedqa/ori_pqal.json.
- The target's opening words: "the answer to the question given the context is" (build_target_seq writes "the answer to
  the question given the context is <answer>.").
- Tokens: fairseq's preprocessing (Moses tokenizer.perl -l en -a, fastBPE with BioGPT-Large's codes, fairseq-preprocess
  with its dictionary) is stood in for by the model's BioGptTokenizer (sacremoses' Moses tokenizer with aggressive hyphen
  splitting and escaping, then BPE), on the source and on the opening words separately, as fairseq preprocessed the
  source and target lines separately. A literal special-token string in a text ("</s>" and the like) is tokenized as
  text (split_special_tokens), as Moses escaping made it. A fairseq dictionary word w is BioGptTokenizer's "w</w>" (a
  word-final piece), and a continuation piece "w@@" is its "w".
- How fairseq built each training sequence, and the sequence fed here:
  1. fairseq-preprocess binarized each source line and each target line with an eos after it (</s>, id 2):
     the source s_1 ... s_N </s>, the target t_1 ... t_M </s>.
  2. LanguageModelPromptDataset.__getitem__ cuts a source longer than max_source_positions (P) tokens, its eos not
     counted: it keeps s_1 ... s_(P-1), then '...' (the dictionary's "...", BioGptTokenizer's "...</w>", id 11774), then
     </s>. (The target would be cut only past max_target_positions minus P minus the prompt's length, which a target of
     12 tokens never reaches; this is checked when max_target_positions is read.)
  3. collate's make_sentence drops the source's final eos and joins, with prefix False, source, prompt and target:
     s_1 ... s_n p_1 ... p_K t_1 ... t_M </s>; with prefix True it would be p_1 ... p_K s_1 ... s_n <seqsep> t_1 ... t_M </s>.
     The prompt p_1 ... p_K is learned1 to learnedK (setup_prompt: the pattern followed by 1 to K, each word added to the
     dictionary), or a manual prompt's words.
  4. The model reads that sentence merged with move_eos_to_beginning=True (fairseq's data_utils.collate_tokens: the
     first position becomes eos and the rest is the sentence without its last token), </s> s_1 ... s_n p_1 ... p_K
     t_1 ... t_M, and learns at each position the sentence's next token. The answer word is thus predicted at the
     position of the opening words' last token ("is"), from everything before it.
  5. BioGPT's inference (inference.py; build_dataset_for_inference with the target </s>) gives the generator the prefix
     </s> s_1 ... s_n p_1 ... p_K, after which it writes the target's words.
  The sequence fed to the Hugging Face model is therefore </s> s_1 ... s_n p_1 ... p_K o_1 ... o_9 (o the opening words'
  9 tokens), and the answer is read from the next-token distribution at o_9. Its leading </s> is fairseq's eos moved to
  the beginning: BioGptTokenizer puts </s> (its sep_token, id 2, fairseq's eos) before a text
  (build_inputs_with_special_tokens), so the source is tokenized with it and cut after it, and the source's own trailing
  eos is never in the sequence. Positions count from that </s> in both (fairseq's learned positions and the Hugging Face
  port's are both offset by 2).
- Each answer's probability: the next token's probability, after the opening words, of the token BioGptTokenizer gives
  for the word yes, no or maybe (yes</w> 22134, no</w> 126 and maybe</w> 28139 at the revision, each one token; if one
  is not one token, the first token of each is used and this is recorded), renormalized over the three. The three
  tokens' summed probability before renormalization (median, 10th percentile, smallest, share below 0.5) and the count of
  items whose most likely next token is one of them are recorded, as a check of the input. Rows are written in the
  project's answer order (src/tasks/pubmedqa.py: yes 0, no 1, maybe 2).
- Batches are padded on the right, with explicit positions 0, 1, 2, ..., so that every real token keeps its position and
  attends only to the tokens before it; each item's next-token logits are taken at its own last token.

The model: BioGptForCausalLM and BioGptTokenizer from --model-dir (downloaded on the pod at the fixed revision), after
every file that fixed_versions.json lists is checked (its size, and its SHA-256 where listed: pytorch_model.bin); a
missing or mismatched weight stops the run. The default dtype is float32: the checkpoint is stored in float32, the model
(about 1.6 billion parameters) takes about 6.3 GB of an H100's 80 GB, and the forward passes over 650 items of about 300
tokens take seconds, so float32 costs almost nothing and keeps bfloat16's rounding (8 significant bits) out of the
probabilities that the log loss reads; its matrix products run at full float32 precision
(torch.set_float32_matmul_precision("highest")). bfloat16 and float16 are accepted, and test always uses the dtype that
validation used.

Subcommands:
  args        correction 5's reader alone; writes its reading to --out (default <results>/biogpt.json, which it does not
              replace once that file holds the validation run's record).
  validation  the reader, then PubMedQA's 150 calibrate validation items (<records-public>/pubmedqa_answer_validation.jsonl,
              the rows not marked calibrate false): <results>/predictions/biogpt_large_pqa-raw_pubmedqa_answer_validation.csv
              (session 4's format: id,label,p0,p1,p2 with p in the order yes, no, maybe, 9 significant digits) and
              <results>/biogpt.json (the checkpoint's arguments, the input's settings, the model's files, the counts and
              the forward seconds).
  test        the 500 test items (<records-public>/pubmedqa_answer_test.jsonl), only when <results>/release_candidate.json
              exists, with the settings, dtype and model files that biogpt.json records:
              <results>/predictions/biogpt_large_pqa-raw_pubmedqa_answer_test.csv, its label column written from the
              records as large_model_session8.py writes it; biogpt.json gains its test section.
A failure (the model, the tokenizer, sacremoses or CUDA missing; a model file that differs from fixed_versions.json; an
input that cannot be built) stops the run with a message and a nonzero exit; the pod's driver then writes the NOT BUILT
marker.

Usage (pod): python src/eval/biogpt_session9.py validation --model-dir /root/eddm_s9/models/biogpt_large_pqa
             python src/eval/biogpt_session9.py test --model-dir /root/eddm_s9/models/biogpt_large_pqa
             python src/eval/biogpt_session9.py args [--out FILE]
       (options: --results, --records-public, --fixed-versions, --device cuda, --batch-size 16, --dtype float32, --url)
"""
import argparse
import codecs
import gzip
import http.client
import io
import json
import pickle
import re
import struct
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.calibrate_session7 import write_probs  # noqa: E402
from eval.calibrate_session8 import now, rel, save_json, sha256_file  # noqa: E402
from eval.large_model_session8 import records_for  # noqa: E402
from tasks.pubmedqa import LABELS, QID  # noqa: E402

RESULTS = ROOT / "results" / "fiorillo_v0_5"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
FIXED_VERSIONS = RESULTS / "fixed_versions.json"
FIXED_KEY = "biogpt_large_pubmedqa"
SYSTEM = "biogpt_large_pqa"
PLAN = "docs/prompt_session9.md, PUBMEDQA BESIDE BIOGPT-LARGE, with correction 5"
SEED = 20260927
EXPECTED_ITEMS = {"validation": 150, "test": 500}
ARCHIVE_BASE = "https://msralaphilly2.blob.core.windows.net/release/BioGPT/checkpoints/QA-PubMedQA-BioGPT-Large.tgz"
README_URL = "https://raw.githubusercontent.com/microsoft/BioGPT/{commit}/README.md"
ARCHIVE_URL = None  # resolved at run time from microsoft/BioGPT's README at the fixed commit (archive_url_from_readme)
READ_LIMIT_BYTES = 10 ** 9  # correction 5: at most 1 GB read
READ_SECONDS = 600  # correction 5: at most 10 minutes
RETRY_WAIT_SECONDS = 15
ARGUMENTS = ("learned_prompt", "learned_prompt_pattern", "prefix", "manual_prompt", "max_source_positions",
             "max_target_positions", "tokens_per_sample")
CHANGES_INPUT = ("learned_prompt", "learned_prompt_pattern", "prefix", "manual_prompt", "max_source_positions")
SEP_TOKEN = "<seqsep>"  # the task's default sep_token, used only when prefix is True
ELLIPSIS = "...</w>"
DTYPES = ("float32", "bfloat16", "float16")
DEFAULT_DTYPE = "float32"
LEGACY_MAGIC = 0x1950A86A20F9469CFC6C  # torch.serialization's MAGIC_NUMBER, at the start of a legacy-format file
NO_DATA = (b"1", b"2", b"3", b"4", b"5", b"6")  # tar member types without data blocks (links, devices, directories, fifos)


# ---------------------------------------------------------------- correction 5: the checkpoint's arguments

class ReadStop(Exception):
    """The archive could not be read as expected (its layout, the read limit or the deadline); not retried."""


class _Transient(Exception):
    """A network failure, worth another attempt within the time left."""


class _Budget(io.RawIOBase):
    """The archive's bytes as served, counted against the read limit, and stopped at the deadline."""

    def __init__(self, raw, counter, limit, deadline):
        self.raw, self.counter, self.limit, self.deadline = raw, counter, limit, deadline

    def readable(self):
        return True

    def readinto(self, b):
        if time.monotonic() > self.deadline:
            raise ReadStop("the time limit was reached")
        left = self.limit - self.counter["bytes"]
        if left <= 0:
            raise ReadStop(f"the read limit of {self.limit} bytes was reached")
        try:
            data = self.raw.read(min(len(b), left, 1 << 20))
        except (OSError, http.client.HTTPException) as e:
            raise _Transient(f"{type(e).__name__}: {e}") from e
        self.counter["bytes"] += len(data)
        b[:len(data)] = data
        return len(data)


class _Limited(io.RawIOBase):
    """At most n bytes of a stream (one tar member's data); each read fills its buffer unless the member ends."""

    def __init__(self, f, n):
        self.f, self.left = f, n

    def readable(self):
        return True

    def readinto(self, b):
        filled = 0
        while filled < len(b) and self.left > 0:
            data = self.f.read(min(len(b) - filled, self.left))
            if not data:
                break
            b[filled:filled + len(data)] = data
            filled += len(data)
            self.left -= len(data)
        return filled


class _Inflate(io.RawIOBase):
    """A zip entry stored with deflate (method 8), inflated on the fly."""

    def __init__(self, f):
        self.f, self.d, self.pending = f, zlib.decompressobj(-15), b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.pending and not self.d.eof:
            chunk = self.f.read(1 << 16)
            if not chunk:
                break
            self.pending = self.d.decompress(chunk)
        n = min(len(b), len(self.pending))
        b[:n] = self.pending[:n]
        self.pending = self.pending[n:]
        return n


def _read_exact(f, n):
    out = bytearray()
    while len(out) < n:
        chunk = f.read(n - len(out))
        if not chunk:
            raise ReadStop(f"the stream ended {n - len(out)} bytes before the {n} expected")
        out += chunk
    return bytes(out)


def _skip(f, n):
    while n > 0:
        chunk = f.read(min(n, 1 << 20))
        if not chunk:
            raise ReadStop("the stream ended inside a member being passed over")
        n -= len(chunk)


def _padded(n):
    return (n + 511) // 512 * 512


def _tar_number(field):
    """A tar header's number: octal text, or GNU's base-256 form (first byte 0x80) for sizes of 8 GiB and more."""
    if field[:1] == b"\x80":
        return int.from_bytes(field[1:], "big")
    if field[:1] == b"\xff":
        raise ReadStop("a tar header holds a negative number")
    try:
        text = field.split(b"\0", 1)[0].strip().decode("ascii")
        return int(text, 8) if text else 0
    except (UnicodeDecodeError, ValueError):
        raise ReadStop("a tar header's number is not octal; the stream is not a tar archive") from None


def _tar_checksum_ok(header):
    try:
        want = _tar_number(header[148:156])
    except ReadStop:
        return False
    unsigned = sum(header[:148]) + 8 * 32 + sum(header[156:])  # the checksum field counted as 8 spaces
    signed = sum(struct.unpack("148b", header[:148])) + 8 * 32 + sum(struct.unpack("356b", header[156:]))
    return want in (unsigned, signed)


def _pax_records(data):
    """A pax extended header's records ("<length> <key>=<value>\\n")."""
    out, pos = {}, 0
    while pos < len(data) and data[pos:pos + 1] not in (b"", b"\0"):
        space = data.index(b" ", pos)
        length = int(data[pos:space])
        key, _, value = data[space + 1:pos + length - 1].partition(b"=")
        out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
        pos += length
    return out


def find_pt_member(f, passed):
    """Walks the tar headers of the decompressed stream f to the first regular member whose name ends in ".pt" (not an
    AppleDouble "._" file) and returns (name, size), with f at the member's data. Members passed over are appended to
    passed."""
    long_name, pax = None, {}
    while True:
        header = _read_exact(f, 512)
        if header == b"\0" * 512:
            raise ReadStop("the archive ended before a member whose name ends in .pt")
        if not _tar_checksum_ok(header):
            raise ReadStop("a tar header's checksum does not match; the stream is not a tar archive")
        kind, size = header[156:157], _tar_number(header[124:136])
        if kind == b"L":  # GNU: the next member's long name
            long_name = _read_exact(f, _padded(size))[:size].split(b"\0", 1)[0].decode("utf-8", "replace")
            continue
        if kind == b"x":  # pax: the next member's extended header
            pax = _pax_records(_read_exact(f, _padded(size))[:size])
            continue
        name = header[:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        if header[257:263] == b"ustar\0":  # POSIX ustar: a name prefix
            prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
            if prefix:
                name = f"{prefix}/{name}"
        name = pax.get("path") or long_name or name
        size = int(pax["size"]) if "size" in pax else size
        long_name, pax = None, {}
        base = name.rstrip("/").rsplit("/", 1)[-1]
        if kind in (b"0", b"\0", b"7") and base.endswith(".pt") and not base.startswith("._"):
            return name, size
        passed.append({"name": name, "type": kind.decode("ascii", "replace"), "bytes": size})
        _skip(f, 0 if kind in NO_DATA else _padded(size))


class _Stub:
    """An inert stand-in for a class or function that the reader does not import; it keeps what the pickle gives it."""
    qualname = "?"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._args, obj._kwargs, obj._state, obj._items, obj._list = args, kwargs, None, {}, []
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def __setstate__(self, state):
        self._state = state

    def __setitem__(self, key, value):
        self._items[key] = value

    def append(self, value):
        self._list.append(value)

    def extend(self, values):
        self._list.extend(values)

    def __repr__(self):
        return f"<{self.qualname}>"


class _Storage:
    """A tensor storage's place in the checkpoint (its persistent ID); its bytes are never read."""

    def __init__(self, pid):
        self.pid = pid


def _reconstructor(cls, base, state):
    """copyreg._reconstructor, for the stand-ins only."""
    if isinstance(cls, type) and issubclass(cls, _Stub):
        obj = cls.__new__(cls)
        if state is not None:
            obj._args = (state,)
        return obj
    raise ReadStop(f"the checkpoint asks to rebuild {cls!r}, which the reader does not allow")


_ALLOWED = {("collections", "OrderedDict"): OrderedDict, ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
            ("builtins", "slice"): slice, ("builtins", "complex"): complex, ("builtins", "bytearray"): bytearray,
            ("builtins", "object"): object, ("__builtin__", "set"): set, ("__builtin__", "frozenset"): frozenset,
            ("__builtin__", "object"): object, ("_codecs", "encode"): codecs.encode,
            ("copyreg", "_reconstructor"): _reconstructor, ("copy_reg", "_reconstructor"): _reconstructor}
_STUBS = {}


class _Unpickler(pickle.Unpickler):
    """Imports nothing: storages become placeholders, and every name outside _ALLOWED becomes a stand-in class."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        key = f"{module}.{name}"
        if key not in _STUBS:
            _STUBS[key] = type(name, (_Stub,), {"qualname": key})
        return _STUBS[key]

    def persistent_load(self, pid):
        return _Storage(pid)


def _unpickle(f):
    return _Unpickler(f).load()


def _zip64_sizes(extra, usize, csize):
    pos = 0
    while pos + 4 <= len(extra):
        tag, length = struct.unpack("<HH", extra[pos:pos + 4])
        body = extra[pos + 4:pos + 4 + length]
        if tag == 0x0001:
            values = list(struct.unpack(f"<{len(body) // 8}Q", body[:len(body) // 8 * 8]))
            if usize == 0xFFFFFFFF and values:
                usize = values.pop(0)
            if csize == 0xFFFFFFFF and values:
                csize = values.pop(0)
            break
        pos += 4 + length
    return usize, csize


def _zip_data_pkl(member):
    """Walks a zip's local headers to the entry whose name ends in data.pkl and unpickles it from the stream."""
    while True:
        if _read_exact(member, 4) != b"PK\x03\x04":
            raise ReadStop("the zip's local headers ended before an entry whose name ends in data.pkl")
        _, flags, method, _, _, _, csize, usize, name_len, extra_len = struct.unpack("<HHHHHIIIHH", _read_exact(member, 26))
        name = _read_exact(member, name_len).decode("utf-8", "replace")
        extra = _read_exact(member, extra_len)
        if 0xFFFFFFFF in (csize, usize):
            usize, csize = _zip64_sizes(extra, usize, csize)
        if name.endswith("data.pkl"):
            if method == 0:
                return name, _unpickle(member)
            if method == 8:
                return name, _unpickle(io.BufferedReader(_Inflate(member)))
            raise ReadStop(f"{name} is stored with compression method {method}")
        if flags & 0x08:
            raise ReadStop(f"the entry {name}, before data.pkl, gives its size only after its data")
        _skip(member, csize)


def read_checkpoint(member):
    """(format, pickle entry, unpickled checkpoint) from the start of a .pt member's data (a buffered stream)."""
    head = member.peek(4)[:4]
    if head == b"PK\x03\x04":
        entry, obj = _zip_data_pkl(member)
        return "zip (torch.save)", entry, obj
    if head[:1] == b"\x80":
        if _unpickle(member) != LEGACY_MAGIC:
            raise ReadStop("the .pt member is a pickle without torch's legacy magic number")
        protocol = _unpickle(member)
        _unpickle(member)  # the system information
        return f"legacy (torch.save, protocol version {protocol})", None, _unpickle(member)
    raise ReadStop(f"the .pt member begins with {head!r}: neither a zip archive nor a pickle")


def plain(x, depth=0):
    """A JSON-ready copy of an unpickled value: omegaconf's containers and nodes, argparse.Namespace and the other
    stand-ins are replaced by their contents, and tensors by their class's name."""
    if depth > 60:
        return repr(x)
    if isinstance(x, _Stub):
        if x._items:
            return {str(k): plain(v, depth + 1) for k, v in x._items.items()}
        if x._list:
            return [plain(v, depth + 1) for v in x._list]
        state = x._state
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict) and not state[0]:
            state = state[1]  # (no dict, slots)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
            state = state[0]  # (dict, slots)
        if isinstance(state, dict):
            if "_content" in state:  # omegaconf's DictConfig and ListConfig
                return plain(state["_content"], depth + 1)
            if "_val" in state:  # omegaconf's value nodes
                return plain(state["_val"], depth + 1)
            return {str(k): plain(v, depth + 1) for k, v in state.items()}  # argparse.Namespace and other objects
        if state is not None:
            return plain(state, depth + 1)
        if len(x._args) == 1 and (x._args[0] is None or isinstance(x._args[0], (str, int, float, bool))):
            return x._args[0]  # an enum's value
        return f"<{x.qualname}>"
    if isinstance(x, _Storage):
        return "<storage>"
    if isinstance(x, dict):
        return {str(k): plain(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [plain(v, depth + 1) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (bytes, bytearray)):
        return f"<{len(x)} bytes>"
    return repr(x)


def checkpoint_arguments(obj):
    """The training arguments of an unpickled fairseq checkpoint: from cfg's task section, then its model section
    (fairseq 0.10 and later keep the configuration in cfg), then the legacy args (an argparse.Namespace)."""
    top = obj if isinstance(obj, dict) else {}
    cfg = plain(top.get("cfg")) if top.get("cfg") is not None else None
    args = plain(top.get("args")) if top.get("args") is not None else None
    sources = []
    if isinstance(cfg, dict):
        sources += [(f"cfg.{part}", cfg[part]) for part in ("task", "model") if isinstance(cfg.get(part), dict)]
    if isinstance(args, dict):
        sources.append(("args", args))
    values, found_in = {}, {}
    for key in ARGUMENTS:
        for where, d in sources:
            if key in d:
                values[key], found_in[key] = d[key], where
                break
    context = {}
    if isinstance(cfg, dict):
        for part, key in (("task", "_name"), ("model", "_name"), ("model", "arch"), ("common", "fp16"), ("common", "bf16"),
                          ("task", "sep_token")):
            if isinstance(cfg.get(part), dict) and key in cfg[part]:
                context[f"cfg.{part}.{key}"] = cfg[part][key]
    if isinstance(args, dict):
        for key in ("task", "arch", "fp16", "sep_token"):
            if key in args:
                context[f"args.{key}"] = args[key]
    return {"values": values, "found_in": found_in, "missing": [k for k in ARGUMENTS if k not in values],
            "checkpoint_keys": sorted(str(k) for k in top), "context": context,
            "context_note": "also read, for the record; not used"}


def url_for_record(url):
    """The archive's address as results record it: without its query string, which for the default address is the
    read-only signature in microsoft/BioGPT's README (as fixed_versions.json records it)."""
    base, _, query = url.partition("?")
    if not query:
        return base
    which = ("the read-only signature in microsoft/BioGPT's README at commit 648f7d65" if base == ARCHIVE_BASE
             else "a query string")
    return f"{base} (with {which})"


def _open_url(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": "ed-decision-model biogpt_session9 (correction 5)"})
    return urllib.request.urlopen(request, timeout=timeout)


def _read_once(url, deadline, limit, counter, opener, attempt):
    timeout = max(1.0, min(60.0, deadline - time.monotonic()))
    final = not url.lower().startswith(("http://", "https://"))  # a local file's failure does not change on a retry
    try:
        response = opener(url, timeout)
    except urllib.error.HTTPError as e:
        if e.code >= 500 and not final:
            raise _Transient(f"HTTP {e.code} {e.reason}") from e
        raise ReadStop(f"HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        if final:
            raise ReadStop(f"{type(e).__name__}: {e}") from e
        raise _Transient(f"{type(e).__name__}: {e}") from e
    try:
        headers = getattr(response, "headers", None)
        attempt["http"] = {"status": getattr(response, "status", None),
                           **{k.lower().replace("-", "_"): (headers.get(k) if headers is not None else None)
                              for k in ("Content-Length", "Last-Modified", "ETag", "Content-MD5")}}
        raw = io.BufferedReader(_Budget(response, counter, limit, deadline), 1 << 20)
        stream = gzip.GzipFile(fileobj=raw, mode="rb") if raw.peek(2)[:2] == b"\x1f\x8b" else raw
        passed = []
        name, size = find_pt_member(stream, passed)
        member = io.BufferedReader(_Limited(stream, size), 1 << 16)
        fmt, entry, obj = read_checkpoint(member)
        return {"member": name, "member_bytes": size, "members_passed": passed, "checkpoint_format": fmt,
                "pickle_entry": entry, **checkpoint_arguments(obj)}
    finally:
        close = getattr(response, "close", None)
        if close:
            close()


def archive_url_from_readme(fixed_versions=FIXED_VERSIONS, opener=None, timeout=60):
    """The archive's full address, with the read-only signature that microsoft/BioGPT's README gives at the fixed commit.
    The README is fetched at that commit and checked against the SHA-256 that fixed_versions.json records for it, so the
    signature is never stored in this repository (a public, read-only one, but a secret scanner would flag it)."""
    entry = fixed_entry(fixed_versions)
    scripts = entry["scripts"]
    want = scripts["files_sha256"]["README.md"]
    response = (opener or _open_url)(README_URL.format(commit=scripts["commit"]), timeout)
    try:
        data = response.read()
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    import hashlib
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ReadStop(f"microsoft/BioGPT's README has SHA-256 {got}, not fixed_versions.json's {want}")
    found = re.search(re.escape(ARCHIVE_BASE) + r"\?[^)\s\"'<>]+", data.decode("utf-8"))
    if not found:
        raise ReadStop("microsoft/BioGPT's README at the fixed commit has no link to the archive")
    return found.group(0)


def read_checkpoint_arguments(url=ARCHIVE_URL, seconds=READ_SECONDS, limit_bytes=READ_LIMIT_BYTES, opener=None):
    """Correction 5's reader. Never raises for a failed reading; its result's status says what happened ("read", "read;
    not found in the checkpoint: ...", or "not read: <reason>"). The reading runs in a thread that is given at most
    `seconds`; network failures are retried, RETRY_WAIT_SECONDS apart, while time is left."""
    opener = opener or _open_url
    started = time.monotonic()
    deadline = started + seconds
    counter, box, attempts = {"bytes": 0}, {}, []
    if url is None:
        try:
            url = archive_url_from_readme(opener=opener)
        except Exception as e:  # the README unreachable or changed: the fixed values are used (correction 5)
            return {"archive_url": ARCHIVE_BASE + " (address not resolved)", "limit_bytes": limit_bytes,
                    "limit_seconds": seconds, "started_utc": now(),
                    "status": f"not read: the archive's address could not be read from microsoft/BioGPT's README ({type(e).__name__}: {e})",
                    "bytes_read": 0, "seconds": round(time.monotonic() - started, 3), "finished_utc": now(), "attempts": [],
                    "saved": "nothing"}
    out = {"archive_url": url_for_record(url), "limit_bytes": limit_bytes, "limit_seconds": seconds, "started_utc": now()}

    def work():
        while True:
            attempt = {"started_utc": now()}
            attempts.append(attempt)
            try:
                box["result"] = _read_once(url, deadline, limit_bytes, counter, opener, attempt)
                attempt["outcome"] = "read"
                return
            except _Transient as e:
                attempt["outcome"] = f"network failure: {e}"
                if deadline - time.monotonic() <= RETRY_WAIT_SECONDS + 5:
                    box["error"] = f"{attempt['outcome']}; no time was left for another attempt"
                    return
                time.sleep(RETRY_WAIT_SECONDS)
            except Exception as e:  # the archive's layout, a limit, the deadline, an unreadable pickle
                attempt["outcome"] = f"{type(e).__name__}: {e}"
                box["error"] = attempt["outcome"]
                return

    thread = threading.Thread(target=work, name="biogpt-checkpoint-arguments", daemon=True)
    thread.start()
    thread.join(max(0.0, deadline - time.monotonic()))
    if "result" in box:
        result = box["result"]
        out.update(result)
        out["status"] = "read" if not result["missing"] else "read; not found in the checkpoint: " + ", ".join(result["missing"])
    else:
        out["status"] = "not read: " + (box.get("error") or f"the reading did not finish within {seconds} seconds")
    out.update({"bytes_read": counter["bytes"], "bytes_read_note": "bytes of the archive as served (gzip-compressed)",
                "seconds": round(time.monotonic() - started, 3), "finished_utc": now(), "attempts": list(attempts),
                "saved": "nothing; the stream was read in memory and closed"})
    return out


# ---------------------------------------------------------------- the settings: checkpoint or fixed_versions.json

def fixed_entry(fixed_versions=FIXED_VERSIONS):
    path = Path(fixed_versions)
    if not path.exists():
        raise SystemExit(f"{path} does not exist; the stage reads BioGPT-Large's fixed values from it")
    data = json.loads(path.read_text(encoding="utf-8"))
    if FIXED_KEY not in data:
        raise SystemExit(f"{path} has no {FIXED_KEY} entry")
    return data[FIXED_KEY]


def fixed_settings(entry):
    """fixed_versions.json's values for the input (step 0 (d)): the learned tokens, the order and the source's cut."""
    lp = entry["learned_prompt"]
    tokens, count = [str(t) for t in lp["tokens"]], int(lp["count"])
    pattern = re.sub(r"1$", "", tokens[0]) if tokens else "learned"
    if tokens != [f"{pattern}{i + 1}" for i in range(count)]:
        raise SystemExit("fixed_versions.json's learned tokens are not one pattern numbered from 1")
    if "prefix False" not in entry["order"]:
        raise SystemExit("fixed_versions.json's order is not fairseq's prefix False (source, prompt, target)")
    return {"learned_prompt": count, "learned_prompt_pattern": pattern, "prefix": False, "manual_prompt": None,
            "max_source_positions": int(entry["source_truncation"]["max_source_positions"]),
            "max_target_positions": None, "tokens_per_sample": None}


def _as_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    raise ValueError(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(value)


def normalized(key, value):
    if key == "prefix":
        return _as_bool(value)
    if key in ("learned_prompt", "max_source_positions", "max_target_positions", "tokens_per_sample"):
        return _as_int(value)
    if value is None or (isinstance(value, str) and value.strip() in ("", "None", "null")):
        return None
    if not isinstance(value, str):
        raise ValueError(value)
    return value


def resolve_settings(reading, fixed):
    """The settings the input uses: the checkpoint's arguments where read, fixed_versions.json's values otherwise, with
    each value's origin and every difference that changes the input."""
    used = dict(fixed)
    origin = {k: "fixed_versions.json" if fixed.get(k) is not None or k in CHANGES_INPUT else "not read; fixed_versions.json has no value"
              for k in ARGUMENTS}
    differences, unreadable = {}, {}
    if str(reading.get("status", "")).startswith("read"):
        for key in ARGUMENTS:
            if key not in reading.get("values", {}):
                continue
            try:
                value = normalized(key, reading["values"][key])
                if value is None and key in ("prefix", "learned_prompt_pattern", "max_source_positions"):
                    raise ValueError("fairseq cannot train without a value")
            except (TypeError, ValueError):
                unreadable[key] = repr(reading["values"][key])
                continue
            if key in CHANGES_INPUT and value != fixed[key]:
                differences[key] = {"checkpoint": value, "fixed_versions": fixed[key]}
            used[key] = value
            origin[key] = f"checkpoint ({reading['found_in'][key]})"
    if not isinstance(used["max_source_positions"], int) or used["max_source_positions"] < 2:
        raise SystemExit(f"max_source_positions {used['max_source_positions']!r} cannot cut a source")
    if used["learned_prompt"] is not None and used["learned_prompt"] < 0:
        raise SystemExit(f"learned_prompt {used['learned_prompt']!r} is not a count of tokens")
    return used, {"origin": origin, "differences_from_fixed_versions": differences, "unreadable_values": unreadable,
                  "used_from": ("fixed_versions.json, the checkpoint's arguments having not been read"
                                if not str(reading.get("status", "")).startswith("read") else
                                "the checkpoint's arguments where read, fixed_versions.json's values otherwise")}


# ---------------------------------------------------------------- the input

def source_text(state):
    """BioGPT's source for the reasoning-required setting (rebuild_data.py: loader and build_source_seq)."""
    context = " ".join(c["text"].strip() for c in state["contexts"])
    context = re.sub(r"\n", " ", context)
    context = re.sub(r"\s+", " ", context)
    return "question: {} context: {} ".format(state["question"].strip(), context.strip())


def _encode(tokenizer, text, special):
    return [int(i) for i in tokenizer(text, add_special_tokens=special, split_special_tokens=True)["input_ids"]]


def _token(tokenizer, i):
    out = tokenizer.convert_ids_to_tokens([int(i)])
    return out[0]


def cut_source(ids, max_source_positions, ellipsis_id):
    """LanguageModelPromptDataset.__getitem__'s cut: a source of more than max_source_positions tokens (its eos not
    counted) keeps its first max_source_positions - 1 tokens, then '...'. Returns (ids, whether it was cut)."""
    if len(ids) > max_source_positions:
        return list(ids[:max_source_positions - 1]) + [int(ellipsis_id)], True
    return list(ids), False


def answer_token_ids(tokenizer, opening_words, opening_ids, words=LABELS):
    """The token whose next-token probability is read for each answer, in the project's answer order: the token the
    tokenizer gives for the word; if a word is not one token, the first token of each word is used (recorded)."""
    info, multi = {}, []
    for word in words:
        ids = _encode(tokenizer, word, False)
        if not ids:
            raise SystemExit(f"the tokenizer gives no token for the answer {word!r}")
        line = _encode(tokenizer, f"{opening_words} {word}.", False)
        in_line = line[len(opening_ids):-1] if line[:len(opening_ids)] == list(opening_ids) else None
        info[word] = {"ids": ids, "tokens": [_token(tokenizer, i) for i in ids], "ids_in_the_target_line": in_line,
                      "same_in_the_target_line": in_line == ids}
        if len(ids) != 1:
            multi.append(word)
    used = [info[w]["ids"][0] for w in words]
    if len(set(used)) != len(used):
        raise SystemExit("two answers share their first token, so their probabilities cannot be told apart")
    return used, {"words": info, "ids_used": used, "order": list(words), "first_token_rule": bool(multi),
                  "multi_token_answers": multi,
                  "rule": ("each answer's token is the one the tokenizer gives for the word; if a word is not one token, "
                           "the first token of each word is used")}


def input_spec(tokenizer, settings, entry=None):
    """What every item's input shares: the leading </s>, the prompt's tokens, the order, '...', the opening words and the
    answers' tokens."""
    vocab = tokenizer.get_vocab()
    lead = tokenizer.sep_token_id
    if lead is None or _token(tokenizer, lead) != "</s>":
        raise SystemExit("the tokenizer's sep_token is not </s>; the leading eos cannot be placed")
    if settings["manual_prompt"]:
        kind = "manual"
        prompt_tokens = [w[:-2] if w.endswith("@@") else f"{w}</w>" for w in settings["manual_prompt"].split()]
    elif settings["learned_prompt"]:
        kind = "learned"
        prompt_tokens = [f"{settings['learned_prompt_pattern']}{i + 1}</w>" for i in range(settings["learned_prompt"])]
    else:
        kind, prompt_tokens = "none", []
    missing = [t for t in prompt_tokens if t not in vocab]
    if missing:
        raise SystemExit(f"the prompt's tokens {missing} are not in the model's vocabulary; the input cannot be built")
    sep_id = None
    if settings["prefix"]:
        found = [vocab[t] for t in (SEP_TOKEN, f"{SEP_TOKEN}</w>") if t in vocab]
        if not found:
            raise SystemExit(f"prefix is True but the vocabulary has no {SEP_TOKEN}; the input cannot be built")
        sep_id = int(found[0])
    ellipsis_id = int(vocab.get(ELLIPSIS, tokenizer.unk_token_id))
    opening = entry["target_opening_words"] if entry else "the answer to the question given the context is"
    opening_ids = _encode(tokenizer, opening, False)
    answer_ids, answers = answer_token_ids(tokenizer, opening, opening_ids)
    # the training target: the opening words, the answer, '.' and the target's eos
    target_tokens = len(opening_ids) + max(len(answers["words"][w]["ids"]) for w in LABELS) + 2
    if settings.get("max_target_positions") is not None:
        room = settings["max_target_positions"] - settings["max_source_positions"] - len(prompt_tokens) - (1 if settings["prefix"] else 0)
        if target_tokens > room:
            raise SystemExit(f"the training target ({target_tokens} tokens) would have been cut at {room}; the answer's "
                             "position would not be where this reads it")
    order = ("</s>, the prompt, the source (cut), <seqsep>, the opening words" if settings["prefix"] else
             "</s>, the source (cut), the prompt, the opening words")
    spec = {"lead_id": int(lead), "lead_token": "</s>", "prompt_kind": kind, "prompt_tokens": prompt_tokens,
            "prompt_ids": [int(vocab[t]) for t in prompt_tokens], "prefix": bool(settings["prefix"]), "sep_id": sep_id,
            "order": order, "max_source_positions": settings["max_source_positions"],
            "ellipsis_token": ELLIPSIS if ELLIPSIS in vocab else "<unk> (the vocabulary has no ...</w>)", "ellipsis_id": ellipsis_id,
            "opening_words": opening, "opening_ids": opening_ids,
            "opening_tokens": [_token(tokenizer, i) for i in opening_ids], "answer_ids": answer_ids, "answers": answers,
            "source_format": "question: {q} context: {c} (rebuild_data.py; the trailing space is dropped by the tokenizer)",
            "tokenizer_call": "add_special_tokens True for the source (</s> first), False for the opening words; split_special_tokens True"}
    if entry:
        fixed_answers = {w: entry["answer_words"][w]["id"] for w in LABELS if w in entry.get("answer_words", {})}
        spec["answer_ids_same_as_fixed_versions"] = fixed_answers == {w: answer_ids[k] for k, w in enumerate(LABELS)}
        if kind == "learned" and settings["learned_prompt"] == entry["learned_prompt"]["count"]:
            spec["prompt_ids_same_as_fixed_versions"] = spec["prompt_ids"] == [int(i) for i in entry["learned_prompt"]["ids"]]
    return spec


def item_sequence(state, tokenizer, spec):
    """One item's token IDs, (</s>, source cut, prompt, opening words) or, with prefix True, (</s>, prompt, source cut,
    <seqsep>, opening words); also the source's length before the cut and whether it was cut."""
    ids = _encode(tokenizer, source_text(state), True)
    if not ids or ids[0] != spec["lead_id"]:
        raise SystemExit("the tokenizer did not put </s> before the source; the input cannot be built")
    source, cut = cut_source(ids[1:], spec["max_source_positions"], spec["ellipsis_id"])
    if spec["prefix"]:
        seq = [spec["lead_id"]] + spec["prompt_ids"] + source + [spec["sep_id"]] + spec["opening_ids"]
    else:
        seq = [spec["lead_id"]] + source + spec["prompt_ids"] + spec["opening_ids"]
    return seq, len(ids) - 1, cut


# ---------------------------------------------------------------- probabilities

def answer_probabilities(next_logits, answer_ids):
    """From next-token logits (items by vocabulary): each item's probabilities of the answers' tokens renormalized over
    those tokens (the softmax of their logits), the tokens' summed probability over the whole vocabulary before
    renormalization, and the most likely next token. In double precision."""
    z = np.asarray(next_logits, dtype=np.float64)
    zmax = z.max(axis=1, keepdims=True)
    log_total = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
    a = z[:, list(answer_ids)]
    amax = a.max(axis=1, keepdims=True)
    e = np.exp(a - amax)
    probs = e / e.sum(axis=1, keepdims=True)
    mass = np.exp(amax[:, 0] + np.log(e.sum(axis=1)) - log_total)
    return probs, mass, z.argmax(axis=1)


def _sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def next_token_logits(model, sequences, pad_id, device, batch_size):
    """Each sequence's next-token logits after its last token: (items by vocabulary array, forward seconds). Batches are
    padded on the right with explicit positions 0, 1, 2, ..., so each real token keeps its position and attends only to
    the tokens before it; the final layer norm's output at each item's last token goes through the model's own output
    layer. The seconds count the model's passes only, with the GPU synchronized."""
    import torch
    head = model.get_output_embeddings()
    rows, seconds = [], 0.0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        width = max(len(s) for s in batch)
        ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, s in enumerate(batch):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        positions = torch.arange(width, dtype=torch.long).unsqueeze(0).expand(len(batch), width).contiguous()
        last = torch.tensor([len(s) - 1 for s in batch], dtype=torch.long, device=device)
        ids, mask, positions = ids.to(device), mask.to(device), positions.to(device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            hidden = model.base_model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).last_hidden_state
            logits = head(hidden[torch.arange(len(batch), device=device), last])
        _sync(device)
        seconds += time.perf_counter() - t0
        rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows, axis=0), seconds


# ---------------------------------------------------------------- the model

def downloaded_revision(model_dir):
    """The commit that huggingface_hub's download recorded in the folder, if it did (.cache/huggingface/download/)."""
    meta = Path(model_dir) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if meta.exists():
        lines = meta.read_text(encoding="utf-8").splitlines()
        return lines[0].strip() if lines else None
    return None


def check_model_files(model_dir, entry):
    """Every file fixed_versions.json lists for the model: present, of the listed size, and with the listed SHA-256
    where one is listed; returns each file's size and SHA-256."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} does not exist; download {entry['repo']} at revision {entry['revision']} there first")
    out = {}
    for name, want in entry["files"].items():
        path = model_dir / name
        if not path.exists():
            raise SystemExit(f"{path} does not exist; the model folder is not {entry['repo']} at revision {entry['revision']}")
        got = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if "bytes" in want and got["bytes"] != want["bytes"]:
            raise SystemExit(f"{path} has {got['bytes']} bytes; fixed_versions.json lists {want['bytes']}")
        if "sha256" in want and got["sha256"] != want["sha256"]:
            raise SystemExit(f"{path} has SHA-256 {got['sha256']}; fixed_versions.json lists {want['sha256']}")
        out[name] = got
    return out


def load_model(model_dir, dtype, device):
    """BioGptForCausalLM and BioGptTokenizer from model_dir, the model in eval mode on the device: (model, tokenizer,
    a record of the load). Stops if CUDA is asked for and missing, if sacremoses is missing, or if a weight is missing
    from the checkpoint or has the wrong shape."""
    try:
        import torch
        import transformers
        from transformers import BioGptForCausalLM, BioGptTokenizer
    except ImportError as e:
        raise SystemExit(f"torch and transformers are needed to run BioGPT-Large: {e}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; the biogpt stage runs on the GPU")
    try:
        tokenizer = BioGptTokenizer.from_pretrained(str(model_dir))
    except ImportError as e:
        raise SystemExit(f"BioGptTokenizer needs sacremoses (0.2.0 in fixed_versions.json): {e}")
    torch.manual_seed(SEED)
    if dtype == "float32":
        torch.set_float32_matmul_precision("highest")
    t0 = time.perf_counter()
    model, info = BioGptForCausalLM.from_pretrained(str(model_dir), dtype=getattr(torch, dtype), output_loading_info=True)
    missing = sorted(str(k) for k in (info.get("missing_keys") or []))
    mismatched = sorted(str(k) for k in (info.get("mismatched_keys") or []))
    if missing or mismatched:
        raise SystemExit(f"the checkpoint does not fill the model: missing {missing[:5]}, mismatched {mismatched[:5]}")
    model.to(device)
    model.eval()
    _sync(device)
    try:
        import sacremoses
        sacremoses_version = getattr(sacremoses, "__version__", None)
    except ImportError:
        sacremoses_version = None
    about = {"class": type(model).__name__, "tokenizer": type(tokenizer).__name__, "dtype": dtype, "device": str(device),
             "gpu": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
             "torch": torch.__version__, "transformers": transformers.__version__, "sacremoses": sacremoses_version,
             "attn_implementation": getattr(model.config, "_attn_implementation", None),
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "max_position_embeddings": int(model.config.max_position_embeddings), "vocab_size": int(model.config.vocab_size),
             "unexpected_keys": sorted(str(k) for k in (info.get("unexpected_keys") or [])),
             "load_seconds": time.perf_counter() - t0}
    return model, tokenizer, about


# ---------------------------------------------------------------- the commands

def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def score_split(rows, model, tokenizer, spec, about, device, batch_size):
    """Builds every item's input and scores it: (probabilities, the split's counts and timing)."""
    t0 = time.perf_counter()
    built = [item_sequence(r["state"], tokenizer, spec) for r in rows]
    tokenize_seconds = time.perf_counter() - t0
    sequences = [b[0] for b in built]
    longest = max(len(s) for s in sequences)
    if longest > about["max_position_embeddings"]:
        raise SystemExit(f"an input has {longest} tokens; the model has {about['max_position_embeddings']} positions")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    try:
        logits, seconds = next_token_logits(model, sequences, pad_id, device, batch_size)
    except Exception as e:
        traceback.print_exc()
        raise SystemExit(f"BioGPT-Large's forward passes failed: {type(e).__name__}: {e}")
    probs, mass, top = answer_probabilities(logits, spec["answer_ids"])
    if not np.all(np.isfinite(probs)):
        raise SystemExit("some answer probabilities are not finite")
    specials = ("<s>", "</s>", "<pad>", "<unk>")
    counts = {
        "items": len(rows), "batch_size": batch_size,
        "sources_cut": int(sum(1 for b in built if b[2])),
        "source_tokens_before_the_cut": _stats([b[1] for b in built]),
        "input_tokens": _stats([len(s) for s in sequences]),
        "sources_with_a_special_token_string": int(sum(1 for r in rows if any(s in source_text(r["state"]) for s in specials))),
        "answer_tokens_summed_probability": {"median": float(np.median(mass)), "p10": float(np.quantile(mass, 0.10)),
                                              "min": float(mass.min()), "share_below_0_5": float(np.mean(mass < 0.5)),
                                              "note": "before renormalization; a check of the input"},
        "most_likely_next_token_is_an_answer": int(sum(1 for t in top if int(t) in spec["answer_ids"])),
        "tokenize_seconds": tokenize_seconds, "forward_seconds": seconds,
        "seconds_per_item": seconds / max(1, len(rows)), "items_per_second": len(rows) / max(1e-9, seconds),
        "timing_note": "the model's passes over the built inputs, GPU synchronized; tokenizing and loading not counted",
    }
    first = sequences[0]
    counts["example_input"] = {"id": rows[0]["id"], "tokens": len(first),
                               "first_tokens": [_token(tokenizer, i) for i in first[:3]],
                               "last_tokens": [_token(tokenizer, i) for i in first[-25:]]}
    return probs, counts


def _write_predictions(results, split, rows, probs):
    path = Path(results) / "predictions" / f"{SYSTEM}-raw_{QID}_{split}.csv"
    tmp = path.with_name(path.name + ".tmp")
    write_probs(tmp, [r["id"] for r in rows], np.array([int(r["label"]) for r in rows]), probs)
    tmp.replace(path)
    return path


def _rows(split, records_public):
    rows = records_for(QID, split, records_public)
    expected = EXPECTED_ITEMS.get(split)
    if expected is not None and len(rows) != expected:
        raise SystemExit(f"{len(rows)} PubMedQA {split} items in {records_public}; the plan answers {expected}")
    if not rows:
        raise SystemExit(f"no PubMedQA {split} items in {records_public}")
    return rows


def _load_record(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_args(out, url=ARCHIVE_URL, fixed_versions=FIXED_VERSIONS, seconds=READ_SECONDS, limit_bytes=READ_LIMIT_BYTES):
    """Correction 5's reader alone, its reading written to out (merged into an existing file that holds no validation
    record)."""
    out = Path(out)
    existing = _load_record(out)
    if existing and "validation" in existing:
        raise SystemExit(f"{out} holds the validation run's record; pass --out to write the reading elsewhere")
    entry = fixed_entry(fixed_versions)
    reading = read_checkpoint_arguments(url, seconds, limit_bytes)
    settings, resolution = resolve_settings(reading, fixed_settings(entry))
    data = dict(existing or {})
    data.update({"plan": PLAN, "written_utc": now(), "checkpoint_arguments": reading,
                 "settings_if_scored_now": {"settings": settings, **resolution}})
    save_json(out, data)
    print(json.dumps({"status": reading["status"], "values": reading.get("values"), "settings": settings,
                      "differences": resolution["differences_from_fixed_versions"], "written": rel(out)}, indent=1), flush=True)
    return data


def run_validation(model_dir, results=RESULTS, records_public=RECORDS_PUBLIC, fixed_versions=FIXED_VERSIONS, device="cuda",
                   batch_size=16, dtype=DEFAULT_DTYPE, url=ARCHIVE_URL, seconds=READ_SECONDS, limit_bytes=READ_LIMIT_BYTES):
    """The biogpt stage: correction 5's reader, then the 150 calibrate validation items."""
    results = Path(results)
    entry = fixed_entry(fixed_versions)
    fixed = fixed_settings(entry)
    reading = read_checkpoint_arguments(url, seconds, limit_bytes)
    print(json.dumps({"checkpoint_arguments": reading["status"], "values": reading.get("values"),
                      "seconds": reading["seconds"], "bytes_read": reading["bytes_read"]}, indent=1), flush=True)
    settings, resolution = resolve_settings(reading, fixed)
    if resolution["differences_from_fixed_versions"]:
        print(f"the checkpoint's arguments differ from fixed_versions.json's; the checkpoint's are used: "
              f"{json.dumps(resolution['differences_from_fixed_versions'])}", flush=True)
    rows = _rows("validation", records_public)
    records_path = Path(records_public) / f"{QID}_validation.jsonl"
    files = check_model_files(model_dir, entry)
    model, tokenizer, about = load_model(model_dir, dtype, device)
    spec = input_spec(tokenizer, settings, entry)
    probs, counts = score_split(rows, model, tokenizer, spec, about, device, batch_size)
    path = _write_predictions(results, "validation", rows, probs)
    labels = [int(r["label"]) for r in rows]
    counts.update({"records": rel(records_path), "records_sha256": sha256_file(records_path),
                   "items_by_label": {w: labels.count(k) for k, w in enumerate(LABELS)},
                   "predictions": rel(path), "predictions_sha256": sha256_file(path), "written_utc": now()})
    record = {"plan": PLAN, "written_utc": now(), "system": SYSTEM, "seed": SEED,
              "model": {"repo": entry["repo"], "revision": entry["revision"], "dir": str(model_dir),
                        "downloaded_revision": downloaded_revision(model_dir), "files": files, **about},
              "checkpoint_arguments": reading,
              "input": {"settings": settings, **resolution, **spec},
              "validation": counts,
              "primary_result": ("the uncalibrated probabilities, renormalized over the three answers: BioGPT-Large was "
                                 "fine-tuned on PQA-L's fold 0 training and development sets, which hold all 150 validation "
                                 "items, so a calibration fitted on them is reported beside it with that caveat")}
    save_json(results / "biogpt.json", record)
    print(f"{SYSTEM} validation: {len(rows)} rows written to {rel(path)}; {counts['sources_cut']} sources cut; "
          f"{counts['forward_seconds']:.1f} forward seconds; answers' median summed probability "
          f"{counts['answer_tokens_summed_probability']['median']:.4f}", flush=True)
    return record


def run_test(model_dir, results=RESULTS, records_public=RECORDS_PUBLIC, fixed_versions=FIXED_VERSIONS, device="cuda",
             batch_size=16, dtype=None):
    """The 500 test items, after the freeze, with the input settings, dtype and model files of the validation run."""
    results = Path(results)
    if not (results / "release_candidate.json").exists():
        raise SystemExit(f"release_candidate.json does not exist in {results}; BioGPT-Large answers the test items only "
                         "after the freeze, in the testpreds stage")
    record = _load_record(results / "biogpt.json")
    if not record or "validation" not in record or "input" not in record:
        raise SystemExit(f"{results / 'biogpt.json'} holds no validation run; the test items are read with its settings")
    used_dtype = record["model"]["dtype"]
    if dtype is not None and dtype != used_dtype:
        raise SystemExit(f"--dtype {dtype} differs from validation's {used_dtype}; the test items are scored as the "
                         "validation items were")
    entry = fixed_entry(fixed_versions)
    files = check_model_files(model_dir, entry)
    changed = [n for n, f in files.items() if record["model"]["files"].get(n, {}).get("sha256") != f["sha256"]]
    if changed:
        raise SystemExit(f"the model's files {changed} differ from those the validation run scored with")
    settings = record["input"]["settings"]
    rows = _rows("test", records_public)
    records_path = Path(records_public) / f"{QID}_test.jsonl"
    model, tokenizer, about = load_model(model_dir, used_dtype, device)
    spec = input_spec(tokenizer, settings, entry)
    for key in ("lead_id", "prompt_ids", "sep_id", "ellipsis_id", "opening_ids", "answer_ids", "max_source_positions", "prefix"):
        if spec[key] != record["input"][key]:
            raise SystemExit(f"the test input's {key} ({spec[key]}) differs from validation's ({record['input'][key]})")
    probs, counts = score_split(rows, model, tokenizer, spec, about, device, batch_size)
    counts.pop("example_input", None)
    path = _write_predictions(results, "test", rows, probs)
    counts.update({"records": rel(records_path), "records_sha256": sha256_file(records_path),
                   "predictions": rel(path), "predictions_sha256": sha256_file(path),
                   "release_candidate_sha256": sha256_file(results / "release_candidate.json"),
                   "settings_from": "biogpt.json's input settings (the validation run's)",
                   "device": str(device), "gpu": about.get("gpu"), "dtype": used_dtype, "load_seconds": about["load_seconds"],
                   "written_utc": now()})
    record["test"] = counts
    record["written_utc"] = now()
    save_json(results / "biogpt.json", record)
    print(f"{SYSTEM} test: {len(rows)} rows written to {rel(path)}; {counts['forward_seconds']:.1f} forward seconds", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("args", "validation", "test"))
    parser.add_argument("--model-dir", type=Path, default=None, help="the model folder (microsoft/BioGPT-Large-PubMedQA at the fixed revision)")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--records-public", type=Path, default=RECORDS_PUBLIC)
    parser.add_argument("--fixed-versions", type=Path, default=FIXED_VERSIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=DTYPES, default=None,
                        help=f"validation: default {DEFAULT_DTYPE}; test: the dtype biogpt.json records (validation's)")
    parser.add_argument("--url", default=ARCHIVE_URL, help="the fine-tuned checkpoint's archive (correction 5; default: the "
                        "link in microsoft/BioGPT's README at the fixed commit, read at run time)")
    parser.add_argument("--out", type=Path, default=None, help="args: where to write the reading (default <results>/biogpt.json)")
    args = parser.parse_args(argv)
    if args.command in ("validation", "test") and args.model_dir is None:
        parser.error("--model-dir is needed")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.command == "args":
        run_args(args.out or args.results / "biogpt.json", args.url, args.fixed_versions)
    elif args.command == "validation":
        run_validation(args.model_dir, args.results, args.records_public, args.fixed_versions, args.device, args.batch_size,
                       args.dtype or DEFAULT_DTYPE, args.url)
    else:
        run_test(args.model_dir, args.results, args.records_public, args.fixed_versions, args.device, args.batch_size, args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())

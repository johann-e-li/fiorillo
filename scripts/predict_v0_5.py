"""Fiorillo v0.5's inference script (docs/prompt_session9.md, THE RELEASE, with correction 3): it loads the release
folder's parts and answers records of the three questions in session 4's record format.

A record is one JSON object: {"id", "qid", "state"} (label, target and group are not read). qid routes it:
- evidence_effect (Evidence Inference 2.0): state {"intervention", "comparator", "outcome", "article"}, the article's full
  text in Evidence Inference's text format (the title and abstract, then the body, whose sections begin with lines that
  begin with "BODY."). The article is cut by the shipped passage selection (evidence_effect/passage_selection/) to the
  input length the specialist was trained at; then each of the specialist's readers answers it (with swap averaging when
  it is kept, each also answers the prompt with the intervention and the comparator exchanged, the same article); the
  answers are averaged and the specialist's calibration applied.
- pubmedqa_answer (PubMedQA): state {"question", "contexts": [{"section", "text"}, ...]}, without the conclusion; s7_b3
  answers it and its calibration is applied.
- readmit_window (UCI Diabetes 130-US hospitals): state in session 4's fields (src/tasks/uci_readmission.py); each pool
  member with a weight above zero answers it, and the pool's weights and T combine them.
Output: one JSON line per record: id, qid, system, probabilities (each option's calibrated probability, in the
question's order; for readmit_window the options are levels 0, 1 and 2), answer (the most probable option), passes (the
readers' forward passes it took), and notes; a record that cannot be answered gets an error instead.

The release folder (this script's own folder unless --release names another):
  evidence_effect/calibration.json          release_candidate.json's specialists > evidence_effect, as the freeze wrote
                                            it: system, swap_averaging, single_pass_systems, max_len, family and params
  evidence_effect/<system>/                 one folder for each name in single_pass_systems (the winning recipe's runs,
                                            or s9_ei_wavg): adapter.safetensors, config.json and the tokenizer files, as
                                            src/train/train_qwen_head.py and src/eval/weight_average_session9.py write them
  evidence_effect/passage_selection/evidence_inference.py   src/tasks/evidence_inference.py, copied in (its selection
                                            functions: article_parts, select_one, assemble, make_state and the rest)
  evidence_effect/passage_selection/bm25_stats.json         data/processed/session8/bm25_stats.json
  pubmedqa_answer/calibration.json          release_candidate.json's specialists > pubmedqa_answer (s7_b3, family, params)
  pubmedqa_answer/s7_b3/                    models/fiorillo_v0_3/arm_b/ (s7_b3's adapter.safetensors, config.json and
                                            tokenizer files)
  readmit_window/pool.json                  release_candidate.json's specialists > readmit_window (members, weights, T)
  readmit_window/recode_tables.json         models/fiorillo_v0_5/trees/recode_tables.json (any tree member or TabICL)
  readmit_window/<member file>              each member with a weight above zero: xgboost_codes.json,
                                            catboost_codes.cbm, lightgbm_tuned.txt, xgboost_tuned.json,
                                            catboost_tuned.cbm, or tabicl.pkl with tabicl_state.json
  readmit_window/s7_b3_calibration.json     if s7_b3 has weight: results/fiorillo_v0_3/calibration_s7.json's
                                            "s7_b3" > "readmit_window" entry (its chosen family and parameters)
  arm_laya (v0.1) is not copied: if it has weight, --arm-laya names v0.1's repository or a local copy of it.

How each part answers (the project's code, reproduced here so that the release needs nothing else from the project):
- the readers (the EI specialist's and s7_b3): Qwen3-4B-Base (--base, at the revision in each folder's config.json)
  with the folder's LoRA adapters, projection and Laya's decision head (src/train/lora.py and train_qwen_head.py), on
  the item src/train/qwen_items.py renders (the state serialized as laya does, then the question and its options, the
  state cut from its end to fit); bfloat16 with autocast on a GPU, float32 otherwise, as train_qwen_head.py predict runs;
- the passage selection: the shipped functions with the token budget of evidence_inference.Budget (the whole sequence
  fits max_len with the module's SPARE tokens to spare), counted with the specialist's tokenizer;
- probabilities: the softmax of a reader's logits; an average is the mean of its readers' probabilities; swap averaging
  (correction 3) is the mean of the answer and the exchange ("significantly increased" and "significantly decreased"
  swapped) of the answer to the exchanged prompt, before calibration, and a prompt whose intervention and comparator are
  the same text (whitespace collapsed, case ignored) keeps its single answer;
- calibration: the family and parameters as src/eval/calibration_family_cv.py's apply_family applies them, to the
  logarithm of the averaged probabilities floored at 1e-300 (EI, as calibrate_session9.py) or to the log-softmax of the
  logits (s7_b3, as calibrate_session7.py);
- the readmission pool: each member's probabilities clipped at 1e-12 before the log, softmax of the weighted sum of the
  logs divided by T (src/eval/trees_session8.py's pool_probs); the trees and TabICL read session 8's frame built from
  recode_tables.json (the _codes members with the recoded diagnosis fields), as src/eval/trees_session9.py's
  frame_from_tables builds it; TabICL's weights are read from its checkpoint (tabicl_state.json names its repository,
  revision, file and SHA-256; --tabicl-checkpoint gives a local copy); arm_laya through laya's Agent with v0.1's own
  temperature, whose probabilities laya rounds to 4 decimals.

Requirements: torch, transformers, safetensors, laya 0.3.20, numpy; for readmission also pandas and the libraries of
the members with weight (xgboost, catboost, lightgbm, tabicl, huggingface_hub).

Usage: python predict_v0_5.py --records records.jsonl [--release DIR] [--base Qwen/Qwen3-4B-Base | FOLDER]
           [--device auto] [--arm-laya johanneli/fiorillo-v0.1 | FOLDER] [--tabicl-checkpoint FILE] [--threads N]
           [--out answers.jsonl]

Change since the release's first public revision (56e5944), made on 3 October 2026: the order in which the script loads
its libraries. On macOS, the readmission pool's tree libraries (xgboost, catboost) and torch, on which the readers run,
cannot work in one process: whichever runs second ends it with a segmentation fault, and at the first revision, which
loads torch first, the readmission question failed this way. The script now loads xgboost and catboost before torch when
every record of the file that --records names asks readmit_window, and does not load them before torch otherwise. On
macOS, predict() refuses with a message (MACOS_MIXED, MACOS_TORCH_FIRST, MACOS_TREES_FIRST) what would end in a
segmentation fault: a records file that mixes readmit_window with the other two questions (run each question's records as
a file of its own there) and the two load orders that fail. Nothing else changed: wherever the first revision answered,
the answers are the same.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import types
from pathlib import Path


def asked_questions(argv):
    """The questions (qid) of the records file that argv's --records names, read before torch is loaded; None when argv
    names no file or the file cannot be read here (main() then reports it)."""
    path = None
    for i, a in enumerate(argv):
        if a == "--records" and i + 1 < len(argv):
            path = argv[i + 1]
        elif a.startswith("--records="):
            path = a.split("=", 1)[1]
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return {json.loads(line).get("qid") for line in f if line.strip()}
    except (OSError, ValueError, AttributeError):
        return None


# The tree libraries before torch only for a file of readmit_window records (the change above). Any failure here is left
# for the readmission question to meet where it imports them.
ASKED = asked_questions(sys.argv[1:]) if __name__ == "__main__" else None
TORCH_ALREADY = "torch" in sys.modules
if ASKED == {"readmit_window"} and not TORCH_ALREADY:
    try:
        import xgboost  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        import catboost  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
TREES_BEFORE_TORCH = not TORCH_ALREADY and ("xgboost" in sys.modules or "catboost" in sys.modules)

import numpy as np
import torch
import torch.nn as nn
from laya.common import QTYPES, DecisionModel, render_options, serialize_state

EI, PQA, RW = "evidence_effect", "pubmedqa_answer", "readmit_window"
QIDS = (EI, PQA, RW)
# The three questions, word for word as the models were trained on them (src/questions/session8.yaml for
# evidence_effect, src/questions/public_v0.yaml for the other two), in Laya's internal form; the criteria's order is
# the label order.
QUESTIONS = {
    EI: {"t": "choice",
         "ins": "Based only on this article, how did the outcome differ with the intervention compared with the comparator?",
         "crit": {"significantly increased": "the outcome was significantly higher with the intervention than with the comparator",
                  "significantly decreased": "the outcome was significantly lower with the intervention than with the comparator",
                  "no significant difference": ("the article reports no significant difference in the outcome between the "
                                                "intervention and the comparator")}},
    PQA: {"t": "choice", "ins": "Based only on this abstract, what is the answer to the research question?",
          "crit": {"yes": "the reported results support yes", "no": "the reported results support no",
                   "maybe": "the reported results are inconclusive"}},
    RW: {"t": "score", "ins": "How soon after this hospital discharge is the patient readmitted?",
         "crit": ["no readmission after this discharge", "readmitted more than 30 days after this discharge",
                  "readmitted within 30 days of this discharge"]},
}
SWAP_COLUMNS = [1, 0, 2]  # "significantly increased" and "significantly decreased" exchanged; "no significant difference" kept
LOG_FLOOR = 1e-300  # the floor of an average before the log (calibrate_session9.py)
POOL_CLIP = 1e-12  # each pool member's probabilities before the log (trees_session8.py)
T_MIN, T_MAX = 0.1, 10.0
FAMILIES = ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROJECTION = 1024
HEAD_LAYERS = 2
N_ACT = 2
OPTION_CAP = 48
DEFAULT_BASE = "Qwen/Qwen3-4B-Base"
DEFAULT_ARM_LAYA = "johanneli/fiorillo-v0.1"
TREE_FILES = {"xgboost_codes": "xgboost_codes.json", "catboost_codes": "catboost_codes.cbm", "lightgbm_tuned": "lightgbm_tuned.txt",
              "xgboost_tuned": "xgboost_tuned.json", "catboost_tuned": "catboost_tuned.cbm"}
CODES_MEMBERS = ("xgboost_codes", "catboost_codes")
TABICL_STATE, TABICL_JSON = "tabicl.pkl", "tabicl_state.json"
TABLES_FILE = "recode_tables.json"
S7_B3_RW_CALIBRATION = "s7_b3_calibration.json"
FIELDS = ("diag_1", "diag_2", "diag_3")
MISSING = "missing"
CHAPTER_RANGES = (("001-139", 1, 139), ("140-239", 140, 239), ("240-279", 240, 279), ("280-289", 280, 289),
                  ("290-319", 290, 319), ("320-389", 320, 389), ("390-459", 390, 459), ("460-519", 460, 519),
                  ("520-579", 520, 579), ("580-629", 580, 629), ("630-679", 630, 679), ("680-709", 680, 709),
                  ("710-739", 710, 739), ("740-759", 740, 759), ("760-779", 760, 779), ("780-799", 780, 799),
                  ("800-999", 800, 999))
CHAPTERS = tuple(name for name, _, _ in CHAPTER_RANGES) + ("V", "E", MISSING)


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist; the release folder is incomplete")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def same_text(a, b):
    """Whether an intervention and a comparator are the same text, whitespace collapsed and case ignored."""
    return " ".join(str(a).split()).lower() == " ".join(str(b).split()).lower()


def options_of(qid):
    q = QUESTIONS[qid]
    return list(q["crit"].keys()) if q["t"] == "choice" else [str(i) for i in range(len(q["crit"]))]


# ---------------------------------------------------------------- calibration (src/eval/calibration_family_cv.py)

def calibrated(logp, params, k):
    """temperature and temperature_intercepts: softmax(log p / T + b), T = exp(params[0]) in [0.1, 10], b_0 = 0."""
    t = float(np.clip(np.exp(params[0]), T_MIN, T_MAX))
    b = np.zeros(k)
    if len(params) > 1:
        b[1:] = params[1:]
    return softmax(np.asarray(logp, dtype=float) / t + b)


def apply_family(name, params, logp):
    """Calibrated probabilities (items by answers): a family and its parameters, as calibration_s9.json and
    calibration_s7.json store them, applied to the log-probabilities."""
    logp = np.asarray(logp, dtype=float)
    k = logp.shape[1]
    if name in ("temperature", "temperature_intercepts"):
        if len(params) != (1 if name == "temperature" else k):
            raise SystemExit(f"{name} takes {1 if name == 'temperature' else k} parameters, not {len(params)}")
        return calibrated(logp, params, k)
    if not np.all(np.isfinite(logp)):
        raise SystemExit("the log-probabilities are not finite")
    if name == "vector":
        if len(params) != 2 * k - 1:
            raise SystemExit(f"vector takes {2 * k - 1} parameters, not {len(params)}")
        z = logp * np.asarray(params[:k], dtype=float)
        z[:, 1:] += np.asarray(params[k:], dtype=float)
        return softmax(z)
    if name == "dirichlet_odir":
        w, b = np.asarray(params["W"], dtype=float), np.asarray(params["b"], dtype=float)
        if w.shape != (k, k) or b.shape != (k,):
            raise SystemExit(f"dirichlet_odir takes W of {k} by {k} and b of {k}, not {w.shape} and {b.shape}")
        return softmax(logp @ w.T + b)
    raise SystemExit(f"unknown calibration family {name!r}; the families are {FAMILIES}")


def entry_family(entry):
    """(family, params) of a calibration entry: release_candidate.json's form (family, params) or calibration_s7.json's
    (chosen, families > family > params)."""
    if "family" in entry and "params" in entry:
        return entry["family"], entry["params"]
    fam = entry["chosen"]
    return fam, entry["families"][fam]["params"]


# ---------------------------------------------------------------- the readers (src/train/lora.py, train_qwen_head.py)

class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) B(A(dropout(x))), the base frozen; src/train/lora.py's layer, with its parameter names."""

    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = int(r), float(alpha), float(alpha) / float(r)
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False, dtype=torch.float32, device=base.weight.device)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False, dtype=torch.float32, device=base.weight.device)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        z = self.dropout(x)
        if not torch.is_autocast_enabled(x.device.type) and z.dtype != self.lora_A.weight.dtype:
            z = z.to(self.lora_A.weight.dtype)
        update = self.lora_B(self.lora_A(z)) * self.scale
        return out + update.to(out.dtype)


def apply_lora(model, targets=LORA_TARGETS, r=16, alpha=32, dropout=0.05):
    wrapped = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    if wrapped == 0:
        raise SystemExit(f"no torch.nn.Linear named {targets} was found in the base model")
    return wrapped


class QwenReader(nn.Module):
    """The adapted base and a projection to Laya's head width (train_qwen_head.py's QwenReader)."""

    def __init__(self, base, width=PROJECTION):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(base.config.hidden_size, width, dtype=torch.float32)
        self.config = types.SimpleNamespace(hidden_size=width)

    def forward(self, input_ids=None, attention_mask=None):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if not torch.is_autocast_enabled(h.device.type):
            h = h.to(self.proj.weight.dtype)
        return types.SimpleNamespace(last_hidden_state=self.proj(h))


def build_reader(base, seed, r, alpha, dropout):
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    apply_lora(base, LORA_TARGETS, r, alpha, dropout)
    model = DecisionModel(QwenReader(base), HEAD_LAYERS, N_ACT)
    model.head_checkpointing = False
    return model


def load_trainable(model, path):
    from safetensors.torch import load_file

    state = load_file(str(path))
    params = dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    if set(state) != want:
        missing, extra = sorted(want - set(state))[:5], sorted(set(state) - want)[:5]
        raise SystemExit(f"{path}: trainable parameters differ (missing {missing}, unexpected {extra})")
    with torch.no_grad():
        for n, t in state.items():
            if params[n].shape != t.shape:
                raise SystemExit(f"{path}: {n} has shape {tuple(t.shape)}, the model {tuple(params[n].shape)}")
            params[n].copy_(t.to(params[n].dtype))


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QwenFormat:
    """The fixed parts of one question's sequence, tokenized once (src/train/qwen_items.py)."""

    def __init__(self, tokenizer, q):
        newline = tokenizer("\n", add_special_tokens=False)["input_ids"]
        if len(newline) != 1:
            raise SystemExit(f"'\\n' is {len(newline)} tokens in this tokenizer, not 1")
        self.newline = newline[0]
        self.gap = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        self.head = tokenizer("%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)["input_ids"]
        self.options = [tokenizer(" " + o, add_special_tokens=False, truncation=True, max_length=OPTION_CAP)["input_ids"]
                        for o in render_options(q)]
        tail = list(self.gap) + list(self.head) + [self.newline]
        self.offsets = []
        for o in self.options:
            tail.extend(o)
            self.offsets.append(len(tail))
            tail.append(self.newline)
        self.tail = tail

    def sequence(self, state_ids, max_len):
        room = max_len - len(self.tail)
        if room < 0:
            return None
        st = list(state_ids[:room])
        return st + self.tail, [len(st) + off for off in self.offsets], len(state_ids) > room


class Readers:
    """Qwen3-4B-Base with one folder's adapters, projection and head at a time; the base is loaded again only when the
    adapters' rank or alpha changes (the weight average's rank differs from a run's)."""

    def __init__(self, base, device):
        self.base_src, self.device = str(base), pick_device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.autocast = torch.bfloat16 if self.device.type == "cuda" else None
        self.key, self.model, self.loaded = None, None, None
        self.tokenizers = {}

    def _base(self, revision):
        from transformers import AutoModel

        local = Path(self.base_src)
        if local.is_dir():
            rev_file = local / "REVISION.txt"
            if revision and rev_file.exists():
                words = rev_file.read_text(encoding="utf-8").split()
                if len(words) > 1 and words[1] != revision:
                    raise SystemExit(f"{local} holds Qwen3-4B-Base at revision {words[1]}, not {revision}")
            return AutoModel.from_pretrained(str(local), dtype=self.dtype, attn_implementation="sdpa")
        return AutoModel.from_pretrained(self.base_src, revision=revision, dtype=self.dtype, attn_implementation="sdpa")

    def load(self, folder):
        """(model, tokenizer, config) with folder's trainable weights in place."""
        from transformers import AutoTokenizer

        folder = Path(folder)
        cfg = load_json(folder / "config.json")
        a = cfg["adapters"]
        if list(a.get("targets", LORA_TARGETS)) != list(LORA_TARGETS):
            raise SystemExit(f"{folder}: adapters on {a.get('targets')}, not {list(LORA_TARGETS)}")
        if (cfg.get("projection", PROJECTION), cfg.get("head_layers", HEAD_LAYERS), cfg.get("n_act", N_ACT)) != (PROJECTION, HEAD_LAYERS, N_ACT):
            raise SystemExit(f"{folder}: a projection, head or action head this script does not build")
        key = (int(a["r"]), float(a["alpha"]), float(a["dropout"]), cfg.get("base_revision"))
        if key != self.key:
            self.model, self.loaded = None, None
            gc.collect()
            model = build_reader(self._base(cfg.get("base_revision")), int(cfg.get("seed", 20260927)), key[0], key[1], key[2])
            self.model, self.key = model.to(self.device), key
            self.model.eval()
        if self.loaded != str(folder):
            load_trainable(self.model, folder / "adapter.safetensors")
            self.model.eval()
            self.loaded = str(folder)
        if str(folder) not in self.tokenizers:
            self.tokenizers[str(folder)] = AutoTokenizer.from_pretrained(str(folder))
        return self.model, self.tokenizers[str(folder)], cfg

    def logits(self, model, tokenizer, fmt, state, max_len, qid):
        """The reader's raw logits for one state (one item, as train_laya.py's run() computes them), and whether the
        state was cut to fit."""
        state_ids = tokenizer(serialize_state(state), add_special_tokens=False)["input_ids"]
        built = fmt.sequence(state_ids, max_len)
        if built is None:
            raise SystemExit(f"{qid}'s question and options alone do not fit {max_len} tokens")
        ids, markers, cut = built
        k = len(markers)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)
        marker_pos = torch.tensor([markers], dtype=torch.long, device=self.device)
        marker_mask = torch.ones((1, k), dtype=torch.bool, device=self.device)
        qtype = torch.tensor([QTYPES[QUESTIONS[qid]["t"]]], device=self.device)
        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
                logits, _ = model(input_ids, mask, marker_pos, marker_mask, qtype)
        return np.asarray(logits.float().cpu()[0, :k].tolist(), dtype=np.float64), bool(cut)


# ---------------------------------------------------------------- the passage selection (shipped)

def load_selection(path):
    """The shipped copy of src/tasks/evidence_inference.py, imported under its own name. Its first lines import the
    project's record helpers (one_hot, sha256_file and write_jsonl, which the selection functions never call) and add
    two project folders to sys.path; both are given a stand-in and undone once it is loaded."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist; the passage selection is shipped with the release")
    spec = importlib.util.spec_from_file_location("fiorillo_v0_5_passage_selection", path)
    module = importlib.util.module_from_spec(spec)
    stub = types.ModuleType("records")
    stub.one_hot = lambda index, n: [1.0 if i == index else 0.0 for i in range(n)]
    stub.sha256_file = sha256_file
    stub.write_jsonl = lambda *a, **k: None
    saved_module, saved_path = sys.modules.get("records"), list(sys.path)
    sys.modules["records"] = stub
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        if saved_module is not None:
            sys.modules["records"] = saved_module
        else:
            sys.modules.pop("records", None)
    for name in ("article_parts", "select_one", "assemble", "make_state", "SPARE"):
        if not hasattr(module, name):
            raise SystemExit(f"{path} has no {name}; it is not src/tasks/evidence_inference.py")
    return module


class SelectionBudget:
    """evidence_inference.Budget's token counts (the serialized state, then the question's tail, with the module's SPARE
    tokens to spare), built on this script's item format instead of importing the project's."""

    def __init__(self, tokenizer, fmt, selection, max_len):
        self.tok, self.fmt, self.make_state = tokenizer, fmt, selection.make_state
        self.limit = int(max_len) - int(selection.SPARE) - len(fmt.tail)
        self.sep = self.count_escaped("\n\n")

    def count(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def count_escaped(self, text):
        return self.count(json.dumps(text, ensure_ascii=False)[1:-1])

    def state_tokens(self, prompt, article):
        return self.count(serialize_state(self.make_state(prompt, article)))

    def total(self, prompt, article):
        return self.state_tokens(prompt, article) + len(self.fmt.tail)


def cut_article(state, tokenizer, selection, stats, budget):
    """The record's article cut by the shipped selection (session 8's THE ARTICLE SELECTION at the specialist's length),
    as evidence_inference.py build and ei_session9.py long cut it; None when the prompt alone does not fit."""
    prompt = {k: state[k] for k in ("intervention", "comparator", "outcome")}
    prefix_text, passages = selection.article_parts(state.get("article") or "")
    enc = tokenizer([json.dumps(p[2], ensure_ascii=False)[1:-1] for p in passages] or [""], add_special_tokens=False)["input_ids"]
    piece_tokens = [len(x) for x in enc][: len(passages)]
    prefix_tokens = budget.count_escaped(prefix_text)
    got = selection.select_one(prompt, prefix_text, passages, piece_tokens, prefix_tokens, stats, budget)
    if got is None:
        return None
    return selection.assemble(prefix_text, got[0], passages, got[1])


# ---------------------------------------------------------------- the three specialists

def answer_ei(records, release, readers):
    folder = Path(release) / EI
    spec = load_json(folder / "calibration.json")
    singles = [str(s) for s in spec["single_pass_systems"]]
    swap, max_len = bool(spec["swap_averaging"]), int(spec["max_len"])
    family, params = entry_family(spec)
    selection = load_selection(folder / "passage_selection" / "evidence_inference.py")
    stats = load_json(folder / "passage_selection" / "bm25_stats.json")
    out, cut_states, probs = {}, {}, {}
    for s in singles:
        model, tok, cfg = readers.load(folder / s)
        if int(cfg["max_len"]) != max_len:
            raise SystemExit(f"{folder / s} was trained at {cfg['max_len']} tokens, but calibration.json says {max_len}")
        fmt = QwenFormat(tok, QUESTIONS[EI])
        if s == singles[0]:  # the article is cut once, with the first reader's tokenizer (the runs' tokenizer files are copies)
            budget = SelectionBudget(tok, fmt, selection, max_len)
            for r in records:
                article = cut_article(r["state"], tok, selection, stats, budget)
                if article is None:
                    out[r["id"]] = {"error": f"the prompt alone does not fit {max_len:,} tokens"}
                else:
                    cut_states[r["id"]] = dict(r["state"], article=article)
        for r in records:
            if r["id"] not in cut_states:
                continue
            st = cut_states[r["id"]]
            z, _ = readers.logits(model, tok, fmt, st, max_len, EI)
            p = softmax(z)
            if swap and not same_text(st["intervention"], st["comparator"]):
                exchanged = dict(st, intervention=st["comparator"], comparator=st["intervention"])
                ze, _ = readers.logits(model, tok, fmt, exchanged, max_len, EI)
                p = 0.5 * (p + softmax(ze)[SWAP_COLUMNS])
            probs.setdefault(r["id"], []).append(p)
    for r in records:
        if r["id"] in out:
            continue
        st = cut_states[r["id"]]
        mean = np.mean(probs[r["id"]], axis=0)
        cal = apply_family(family, params, np.log(np.maximum(mean, LOG_FLOOR))[None, :])[0]
        swapped = swap and not same_text(st["intervention"], st["comparator"])
        out[r["id"]] = {"probabilities": cal, "system": spec["system"], "passes": len(singles) * (2 if swapped else 1),
                        "notes": [f"article cut to fit {max_len:,} tokens by the shipped passage selection"]
                                 + (["swap averaging: each reader also answered the prompt with the intervention and the comparator "
                                     "exchanged"] if swapped else [])
                                 + (["the intervention and the comparator are the same text, so the single answer is kept"]
                                    if swap and not swapped else [])}
    return out


def answer_pqa(records, release, readers):
    folder = Path(release) / PQA
    spec = load_json(folder / "calibration.json")
    family, params = entry_family(spec)
    model, tok, cfg = readers.load(folder / spec.get("system", "s7_b3"))
    fmt = QwenFormat(tok, QUESTIONS[PQA])
    out = {}
    for r in records:
        z, cut = readers.logits(model, tok, fmt, r["state"], int(cfg["max_len"]), PQA)
        cal = apply_family(family, params, log_softmax(z)[None, :])[0]
        out[r["id"]] = {"probabilities": cal, "system": spec.get("system", "s7_b3"), "passes": 1,
                        "notes": [f"the abstract was cut from its end to fit {int(cfg['max_len']):,} tokens"] if cut else []}
    return out


# ---------------------------------------------------------------- readmission: frames, members and the pool

def diag_code(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text == "?":
        return None
    return text.split(" (", 1)[0].strip()


def three_char(code):
    if code is None:
        return MISSING
    c = str(code).strip().upper()
    if c.startswith("V"):
        return f"V{int(float(c[1:])):02d}"
    if c.startswith("E"):
        return f"E{int(float(c[1:])):03d}"
    return f"{int(float(c)):03d}"


def chapter(code3):
    if code3 == MISSING:
        return MISSING
    if code3[0] in "VE":
        return code3[0]
    number = int(code3)
    for name, low, high in CHAPTER_RANGES:
        if low <= number <= high:
            return name
    raise SystemExit(f"ICD-9-CM code outside every chapter: {code3}")


def uci_frame(rows, tables):
    """Session 8's frame (src/eval/baselines.py's uci_frame with the training split's categories), from the tables."""
    import pandas as pd

    cats, cols = tables["frame_categories"], tables["frame_columns"]
    data = {}
    for c in cols:
        if c in cats and c.startswith("med_"):
            data[c] = [(r["state"].get("medications") or {}).get(c[len("med_"):], "No") for r in rows]
        elif c in cats:
            data[c] = [r["state"].get(c) for r in rows]
        else:
            data[c] = [r["state"].get(c, np.nan) for r in rows]
    frame = pd.DataFrame(data)
    for c in cols:
        if c in cats:
            frame[c] = pd.Categorical(frame[c], categories=cats[c])
        else:
            frame[c] = frame[c].astype(float)
    return frame


def recoded_columns(rows, tables, index):
    """The 35 recoded diagnosis columns (src/eval/trees_session9.py's recode), encoded with the tables' whole-training
    shares; a code the tables do not hold gets the prior."""
    import pandas as pd

    codes = {f: [three_char(diag_code(r["state"].get(f))) for r in rows] for f in FIELDS}
    chapters = {f: [chapter(c) for c in codes[f]] for f in FIELDS}
    cols = {}
    for f in FIELDS:
        cols[f"{f}_code3"] = pd.Categorical(codes[f], categories=tables["code3_categories"][f])
        cols[f"{f}_chapter"] = pd.Categorical(chapters[f], categories=list(CHAPTERS))
    present = [{chapters[f][i] for f in FIELDS} for i in range(len(rows))]
    for ch in CHAPTERS:
        cols["diag_any_" + ch.replace("-", "_")] = np.array([int(ch in found) for found in present], dtype=np.int64)
    for f in FIELDS:
        table = tables["target_encoding"]["fields"][f]
        prior, shares = table["prior"], table["shares"]
        enc = np.array([shares.get(c, prior) for c in codes[f]], dtype=float).reshape(len(rows), len(prior))
        for a in range(enc.shape[1]):
            cols[f"{f}_code3_te{a}"] = np.asarray(enc[:, a], dtype=float)
    return pd.DataFrame(cols, index=index)


def frames(rows, tables):
    """(session 8's frame, the same with the recoded fields)."""
    import pandas as pd

    plain = uci_frame(rows, tables)
    if [str(c) for c in plain.columns] != list(tables["frame_columns"]):
        raise SystemExit("the frame's columns differ from recode_tables.json's")
    return plain, pd.concat([plain, recoded_columns(rows, tables, plain.index)], axis=1)


def cat_frame(x):
    """CatBoost's copy of a frame: the categorical columns as strings, a missing value as "missing"."""
    import pandas as pd

    x = x.copy()
    for c in [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]:
        x[c] = x[c].astype(object).where(x[c].notna(), "missing").astype(str)
    return x


def tree_probs(member, path, plain, coded, threads=None):
    frame = coded if member in CODES_MEMBERS else plain
    if member.startswith("xgboost"):
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(path))
        if threads:
            model.set_params(n_jobs=threads)
        return np.asarray(model.predict_proba(frame), dtype=float)
    if member.startswith("catboost"):
        import catboost as cb

        model = cb.CatBoostClassifier()
        model.load_model(str(path))
        return np.asarray(model.predict_proba(cat_frame(frame), thread_count=threads or -1), dtype=float)
    import lightgbm as lgb

    return np.asarray(lgb.Booster(model_file=str(path)).predict(frame), dtype=float)


def load_tabicl(state_path, checkpoint_path, device=None):
    """TabICL's saved state with its weights read from checkpoint_path, no download (trees_session9.py's load_tabicl)."""

    def relocating(cls):
        class Relocated(cls):
            def __setstate__(self, state):
                state = dict(state)
                state["model_path"] = str(checkpoint_path)
                state["allow_auto_download"] = False
                if device is not None:
                    state["device"] = device
                self.__class__ = cls
                setter = getattr(cls, "__setstate__", None)
                if setter is not None:
                    setter(self, state)
                else:
                    self.__dict__.update(state)
        Relocated.__name__ = Relocated.__qualname__ = cls.__name__
        return Relocated

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            cls = super().find_class(module, name)
            if name == "TabICLClassifier" and module.split(".")[0] == "tabicl":
                return relocating(cls)
            return cls

    with open(state_path, "rb") as f:
        return Unpickler(f).load()


def tabicl_probs(folder, plain, device, checkpoint=None):
    state = load_json(Path(folder) / TABICL_JSON)
    ck = state["checkpoint"]
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(repo_id=ck["repo"], filename=ck["file"], revision=ck["revision"])
    checkpoint = Path(checkpoint)
    got = sha256_file(checkpoint)
    if got != ck["sha256"]:
        raise SystemExit(f"TabICL's checkpoint {checkpoint} has SHA-256 {got}, not {ck['sha256']}")
    state_path = Path(folder) / TABICL_STATE
    if state.get("sha256") and sha256_file(state_path) != state["sha256"]:
        raise SystemExit(f"{state_path} differs from the state tabicl_state.json records")
    dev = str(pick_device(device))
    return np.asarray(load_tabicl(state_path, checkpoint, dev).predict_proba(plain), dtype=float)


def arm_laya_probs(rows, source):
    """v0.1 (arm_laya) through laya's Agent with its own temperature, as v0.1's card applies it."""
    from laya import Agent

    local = Path(source)
    if local.is_dir():
        calibration = load_json(local / "calibration.json")
    else:
        from huggingface_hub import hf_hub_download

        calibration = load_json(hf_hub_download(str(source), "calibration.json"))
    agent = Agent(str(source))
    agent.temperature_by_options = {"score:3-5": calibration["temperature"][RW]}
    q = {"type": "score", "instructions": QUESTIONS[RW]["ins"], "criteria": list(QUESTIONS[RW]["crit"])}
    out = []
    for r in rows:
        p = agent.system_one(r["state"], {RW: q})["answers"][RW]["probabilities"]
        out.append([float(p[str(j)]) for j in range(len(QUESTIONS[RW]["crit"]))])
    return np.asarray(out, dtype=float)


def s7_b3_rw_probs(rows, release, readers):
    folder = Path(release)
    entry = load_json(folder / RW / S7_B3_RW_CALIBRATION)
    family, params = entry_family(entry)
    model, tok, cfg = readers.load(folder / PQA / "s7_b3")
    fmt = QwenFormat(tok, QUESTIONS[RW])
    logp = np.array([log_softmax(readers.logits(model, tok, fmt, r["state"], int(cfg["max_len"]), RW)[0]) for r in rows])
    return apply_family(family, params, logp)


def pool_probs(logs, w, t):
    """softmax((sum over members of w_m log p_m) / T); logs has shape (members, rows, answers)."""
    z = np.tensordot(np.asarray(w, dtype=float), logs, axes=(0, 0)) / float(t)
    return softmax(z)


def answer_rw(records, release, readers, device="auto", arm_laya=DEFAULT_ARM_LAYA, tabicl_checkpoint=None, threads=None):
    folder = Path(release) / RW
    pool = load_json(folder / "pool.json")
    members = [m for m in pool["members"] if float(pool["weights"][m]) > 0]
    if not members:
        raise SystemExit("pool.json gives no member a weight above zero")
    plain = coded = None
    if any(m in TREE_FILES or m == "tabicl" for m in members):
        plain, coded = frames(records, load_json(folder / TABLES_FILE))
    probs = []
    for m in members:
        if m in TREE_FILES:
            p = tree_probs(m, folder / TREE_FILES[m], plain, coded, threads)
        elif m == "tabicl":
            p = tabicl_probs(folder, plain, device, tabicl_checkpoint)
        elif m == "s7_b3":
            p = s7_b3_rw_probs(records, release, readers)
        elif m == "arm_laya":
            p = arm_laya_probs(records, arm_laya)
        else:
            raise SystemExit(f"pool.json names a member this script does not know: {m}")
        if p.shape != (len(records), len(QUESTIONS[RW]["crit"])):
            raise SystemExit(f"{m} gave probabilities of shape {p.shape}")
        probs.append(p)
    logs = np.log(np.clip(np.stack(probs), POOL_CLIP, None))
    cal = pool_probs(logs, [float(pool["weights"][m]) for m in members], pool["T"])
    passes = sum(1 for m in members if m in ("s7_b3", "arm_laya"))  # the language-model readers' forward passes
    note = "the pool of " + ", ".join(f"{m} (weight {float(pool['weights'][m]):g})" for m in members) + f", T {float(pool['T']):.4f}"
    return {r["id"]: {"probabilities": cal[i], "system": pool.get("system", "readmission_pool_v2"), "passes": passes, "notes": [note]}
            for i, r in enumerate(records)}


# ---------------------------------------------------------------- the command

def read_records(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    seen = set()
    for r in rows:
        if not isinstance(r, dict) or "id" not in r or "qid" not in r or "state" not in r:
            raise SystemExit(f"{path}: every line must be a record with id, qid and state")
        if r["id"] in seen:
            raise SystemExit(f"{path}: the ID {r['id']} appears twice")
        seen.add(r["id"])
    return rows


MACOS_MIXED = ("on macOS, readmit_window cannot be answered in the same process as evidence_effect or pubmedqa_answer: the "
               "tree libraries (xgboost, catboost) and torch each bring their own OpenMP runtime, and whichever runs second "
               "ends the process with a segmentation fault. Put the readmit_window records in a file of their own and run "
               "this script once for each file.")
MACOS_TORCH_FIRST = ("on macOS, readmit_window needs xgboost and catboost loaded before torch, and they were not: run this "
                     "script from the command line with a records file that holds only readmit_window records (it then "
                     "loads them first), or import xgboost and catboost before importing this module.")
MACOS_TREES_FIRST = ("on macOS, evidence_effect and pubmedqa_answer cannot run in a process that loaded xgboost or catboost "
                     "before torch: run them in a new process.")


def macos_check(platform, by_qid, trees_before_torch):
    """On macOS, refuse what would end in a segmentation fault (the change in the module's docstring); elsewhere, nothing."""
    if platform != "darwin":
        return
    readers = bool(by_qid[EI] or by_qid[PQA])
    if by_qid[RW] and readers:
        raise SystemExit(MACOS_MIXED)
    if by_qid[RW] and not trees_before_torch:
        raise SystemExit(MACOS_TORCH_FIRST)
    if readers and trees_before_torch:
        raise SystemExit(MACOS_TREES_FIRST)


def predict(records, release, base=DEFAULT_BASE, device="auto", arm_laya=DEFAULT_ARM_LAYA, tabicl_checkpoint=None, threads=None):
    """One answer for each record, in the records' order."""
    release = Path(release)
    by_qid = {q: [r for r in records if r["qid"] == q] for q in QIDS}
    macos_check(sys.platform, by_qid, TREES_BEFORE_TORCH)
    readers = Readers(base, device)
    got = {}
    if by_qid[EI]:
        got.update(answer_ei(by_qid[EI], release, readers))
    if by_qid[PQA]:
        got.update(answer_pqa(by_qid[PQA], release, readers))
    if by_qid[RW]:
        got.update(answer_rw(by_qid[RW], release, readers, device, arm_laya, tabicl_checkpoint, threads))
    out = []
    for r in records:
        if r["qid"] not in QIDS:
            out.append({"id": r["id"], "qid": r["qid"], "error": f"unknown question {r['qid']}; this release answers {list(QIDS)}"})
            continue
        a = got[r["id"]]
        if "error" in a:
            out.append({"id": r["id"], "qid": r["qid"], "error": a["error"]})
            continue
        options = options_of(r["qid"])
        p = np.asarray(a["probabilities"], dtype=float)
        out.append({"id": r["id"], "qid": r["qid"], "system": a["system"],
                    "probabilities": {o: float(v) for o, v in zip(options, p)}, "answer": options[int(np.argmax(p))],
                    "passes": int(a["passes"]), "notes": a["notes"]})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", type=Path, required=True, help="JSON Lines, one record a line, in session 4's record format")
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parent,
                        help="the release folder (default: the folder this script is in)")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Qwen3-4B-Base: a Hugging Face repository (read at the revision in each config.json) or a local folder")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--arm-laya", default=DEFAULT_ARM_LAYA, help="v0.1's repository or a local copy (only if arm_laya has weight)")
    parser.add_argument("--tabicl-checkpoint", type=Path, default=None,
                        help="a local copy of TabICL's checkpoint (only if tabicl has weight); otherwise it is downloaded")
    parser.add_argument("--threads", type=int, default=None, help="threads for the tree members")
    parser.add_argument("--out", type=Path, default=None, help="JSON Lines of answers (default: printed)")
    args = parser.parse_args(argv)
    answers = predict(read_records(args.records), args.release, args.base, args.device, args.arm_laya, args.tabicl_checkpoint,
                      args.threads)
    text = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

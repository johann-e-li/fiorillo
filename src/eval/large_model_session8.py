"""The large model, Gemma 4 31B-it (docs/prompt_session8.md, THE LARGE MODEL; idea I-13 of docs/ideas_register.md).

It answers EI's validation and test prompts and PubMedQA's 150 calibrate validation items and 500 test items (test items
only in the testpreds stage); it does not answer readmission (I-15). Its outputs never train, tune or select a Fiorillo
model; they are used only in the hand-off and the comparisons, with its own calibration fitted on validation.

The prompt: Gemma's chat template with the generation prompt added and thinking off (apply_chat_template(messages,
add_generation_prompt=True, enable_thinking=False), no system message; checked on the tokenizer files at the fixed revision
on 30 September 2026: the generation prompt then ends with the empty thought block "<|channel>thought\\n<channel|>", so the
next token is the answer), and one user message in plain text, not JSON:
- EI: "Intervention: <intervention>", "Comparator: <comparator>" and "Outcome: <outcome>" on their own lines, a blank line,
  "Article:", and the same selected text Fiorillo reads (the record's state.article), with its real line breaks;
- PubMedQA: "Question: <question>", a blank line, "Abstract:", and each context as "<section>: <text>", separated by blank
  lines;
then a blank line, the question's instructions, a blank line, the options lettered "A.", "B." and "C." in the question's
option order with their rendered texts (laya.common.render_options), each on its own line, and on the next line "Answer
with the letter of one option only." The templated text is tokenized without adding special tokens (the template
already begins with <bos>).

Each answer's probability: one request per item, ending at the generation prompt, with at most 1 new token, temperature 0
and the top 20 log-probabilities of that token. The letters A, B and C, with and without a leading space, are each checked
to be one token in Gemma's tokenizer (a variant that is not is left out; if no variant of a letter is one token, the
script stops). An option's probability is the sum of its letter variants' probabilities in the top 20, and the three are
renormalized. An item for which some letter has no variant in the top 20 is answered again (correction 5): first, in the
same process, from the top 1,000 log-probabilities of the same next token (the engine allows up to 1,000); then, if a
letter is still missing, in a second process (gpu_memory_utilization 0.90, max_num_batched_tokens 1,024, max_num_seqs 1,
started once the first has left the GPU) from the prompt log-probability of that item's prompt followed by each letter
variant (vLLM computes these without its prefix cache); the variants' probabilities are summed per letter and
renormalized. Each item's method is recorded (top20, top1000 or prompt), and the items answered either later way are
counted as answered the second way. Each process shuts vLLM's engine down itself once its passes end, also when they stop
with an error, so that it exits and leaves the GPU (correction 12).

Subcommands (the pod script runs prompts in the training environment, which has laya, and answer and second in
/root/eddm_s8/vllm_env/):
  prompts  the user messages, from the records and the question banks: <items-dir>/prompts_<qid>_<split>.jsonl.
  answer   vLLM at the version in fixed_versions.json, bfloat16, text only (every multimodal input that vLLM registers
           for the model limited to 0: image and video, and audio if registered), without CUDA graphs (enforce_eager: one
           new token a request), maximum length 8,192 tokens; the template and length of every prompt checked before the
           model is loaded; the first pass over every item, then the top-1,000 pass over the items missing a letter;
           <items-dir>/first_<qid>_<split>.jsonl. If no item is left for the second process, it finishes (below).
  second   the second process, over the items still missing a letter; then it finishes.
  finish   (also run by answer and second) writes to <results>/large_model/: gemma4_31b-raw_<qid>_<split>.csv (session 4's
           format, 9 significant digits), gemma4_31b_logprobs_<qid>_<split>.csv (each item's method, its three letters'
           summed probability before renormalization, and each letter's log-probability), gemma4_31b_letters.json (for each
           question and split: the letters' summed probability, median, 10th percentile and share below 0.5, the count
           answered the second way and the count by method) and gemma4_31b_timing.json (each question's wall seconds, GPU
           seconds per item, since the process is alone on the GPU, and prompt tokens per item).

Usage (pod): python src/eval/large_model_session8.py prompts --splits validation --items-dir /root/eddm_s8/large_items
             /root/eddm_s8/vllm_env/bin/python src/eval/large_model_session8.py answer --splits validation \\
                 --model-dir /root/eddm_s8/models/gemma4_31b --items-dir /root/eddm_s8/large_items
             /root/eddm_s8/vllm_env/bin/python src/eval/large_model_session8.py second --splits validation ...
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Correction 11: vLLM's own sampler instead of FlashInfer's, whose sampling module vLLM would compile at engine start
# (the build tool it needs is not on the path of the vLLM environment's python). Every request here is greedy with one
# new token, and the log-probabilities are read from the raw logits, so the answers do not depend on the sampler. Set
# before vLLM is imported; the engine's own process inherits it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "fiorillo_v0_4"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
RECORDS_EI = ROOT / "data" / "processed" / "session8"
QUESTIONS = {"evidence_effect": ROOT / "src" / "questions" / "session8.yaml",
             "pubmedqa_answer": ROOT / "src" / "questions" / "public_v0.yaml"}
SYSTEM = "gemma4_31b"
LETTERS = ("A", "B", "C")
VARIANTS = {letter: (letter, " " + letter) for letter in LETTERS}
TOP = 20
TOP_AGAIN = 1000
MAX_LEN = 8192
FIRST_UTILIZATION = 0.90
SECOND_UTILIZATION = 0.90
SECOND_BATCHED_TOKENS = 1024
MODALITIES = ("image", "audio", "video")
ANSWER_LINE = "Answer with the letter of one option only."
PLAN = "docs/prompt_session8.md, THE LARGE MODEL"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp.replace(path)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- prompts

def options_block(options):
    return "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))


def ei_message(state, instructions, options):
    return (f"Intervention: {state['intervention']}\nComparator: {state['comparator']}\nOutcome: {state['outcome']}\n\n"
            f"Article:\n{state['article']}\n\n{instructions}\n\n{options_block(options)}\n{ANSWER_LINE}")


def pubmedqa_message(state, instructions, options):
    contexts = "\n\n".join(f"{c['section']}: {c['text']}" for c in state["contexts"])
    return (f"Question: {state['question']}\n\nAbstract:\n{contexts}\n\n{instructions}\n\n{options_block(options)}\n{ANSWER_LINE}")


def user_message(qid, state, question):
    from laya.common import render_options
    options = render_options(question)
    if len(options) != len(LETTERS):
        raise ValueError(f"{qid} has {len(options)} options; the prompt letters three")
    if qid == "evidence_effect":
        return ei_message(state, question["ins"], options)
    return pubmedqa_message(state, question["ins"], options)


def records_for(qid, split, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI):
    folder = Path(records_ei) if qid == "evidence_effect" else Path(records_public)
    rows = read_jsonl(folder / f"{qid}_{split}.jsonl")
    if qid == "pubmedqa_answer" and split == "validation":
        rows = [r for r in rows if r.get("calibrate", True)]  # the 150 calibrate items
    return rows


def prompts(qids, splits, items_dir, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI):
    sys.path.insert(0, str(ROOT / "src" / "train"))
    from build_items import load_questions
    out = {}
    for qid in qids:
        question = load_questions(QUESTIONS[qid])[qid]
        for split in splits:
            rows = [{"id": r["id"], "label": int(r["label"]), "qid": qid, "split": split, "text": user_message(qid, r["state"], question)}
                    for r in records_for(qid, split, records_public, records_ei)]
            write_jsonl(Path(items_dir) / f"prompts_{qid}_{split}.jsonl", rows)
            out[f"{qid}_{split}"] = len(rows)
    print(json.dumps({"prompts": out}))
    return out


# ---------------------------------------------------------------- letters and probabilities

def letter_ids(tokenizer):
    """{letter: [token IDs of its one-token variants]}; stops if a letter has none."""
    out, left_out = {}, []
    for letter, variants in VARIANTS.items():
        ids = []
        for v in variants:
            enc = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                ids.append(int(enc[0]))
            else:
                left_out.append(v)
        if not ids:
            raise SystemExit(f"no variant of the letter {letter} is one token in this tokenizer; stop")
        out[letter] = ids
    return out, left_out


def option_probabilities(top, ids_by_letter):
    """From one token's top log-probabilities ({token ID: log-probability}): (renormalized probabilities, the letters'
    summed probability, each letter's summed probability, the letters with no variant in the top). The probabilities are
    None when a letter has no variant there (the item goes the second way)."""
    per_letter, missing = [], []
    for letter in LETTERS:
        found = [math.exp(top[i]) for i in ids_by_letter[letter] if i in top]
        if not found:
            missing.append(letter)
        per_letter.append(sum(found))
    mass = float(sum(per_letter))
    if missing or mass <= 0:
        return None, mass, per_letter, missing
    return [p / mass for p in per_letter], mass, per_letter, []


def from_prompt_logprobs(logprob_by_variant, ids_by_letter):
    """From the prompt log-probability of each letter variant: (renormalized probabilities, summed mass, per letter)."""
    per_letter = [sum(math.exp(logprob_by_variant[i]) for i in ids_by_letter[letter]) for letter in LETTERS]
    mass = float(sum(per_letter))
    return [p / mass for p in per_letter], mass, per_letter


# ---------------------------------------------------------------- vLLM passes (pod, vllm_env)

def templated_ids(tokenizer, text):
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    if not rendered.endswith("<|channel>thought\n<channel|>"):
        raise SystemExit("the chat template with thinking off does not end with the empty thought block; stop")
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if ids[:1] != [tokenizer.bos_token_id] or ids[1:2] == [tokenizer.bos_token_id]:
        raise SystemExit("the templated prompt does not begin with exactly one <bos>; stop")
    return ids


def as_logprob(value):
    return float(getattr(value, "logprob", value))


def check_order(outputs, sent):
    """Stops unless vLLM returned one output per prompt, in the order sent (LLM.generate sorts by request ID; correction 9)."""
    if len(outputs) != len(sent) or any(list(o.prompt_token_ids) != list(s) for o, s in zip(outputs, sent)):
        raise SystemExit("vLLM's outputs do not match the prompts sent, one for one and in order; stop")


def make_llm(model_dir, utilization, max_logprobs=TOP, max_num_seqs=None, max_num_batched_tokens=None):
    """vLLM's engine, text only: every multimodal input limited to 0. vLLM 0.30.0 accepts a limit for any modality name
    and applies it to the modalities the model registers (image and video for Gemma 4; audio only for checkpoints with an
    audio tower, which the 31B does not have), so the full set is passed once; with image and video at 0 the vision
    tower is not built. The set passed is printed (correction 9)."""
    from vllm import LLM
    kwargs = {"model": str(model_dir), "tokenizer": str(model_dir), "dtype": "bfloat16", "max_model_len": MAX_LEN,
              "gpu_memory_utilization": utilization, "seed": 20260927, "max_logprobs": max_logprobs, "enforce_eager": True}
    if max_num_seqs:
        kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    modalities = list(MODALITIES)
    try:
        llm = LLM(limit_mm_per_prompt={m: 0 for m in modalities}, **kwargs)
    except BaseException:
        stop_children()  # an engine process started before the failure would hold the exit (correction 12)
        raise
    print(f"vLLM engine: multimodal inputs limited to 0 for {modalities}; utilization {utilization}; "
          f"max_logprobs {max_logprobs}; max_num_seqs {max_num_seqs}; max_num_batched_tokens {max_num_batched_tokens}", flush=True)
    return llm


def close_llm(llm, timeout=30):
    """Shuts vLLM's engine process down before this process ends (correction 12). vLLM 0.30.0 runs the model in its own
    engine process. On the pod on 30 September 2026, two smoke-test processes with the second process's settings
    (max_num_seqs 1, max_num_batched_tokens 1,024) left that to the finalizers at interpreter exit and did not exit: one
    after an error at about 17:07 UTC, one after finishing its work at 17:09:36 UTC; both were stopped by hand (at about
    17:08 and at 17:11:12) while their engine processes still held the GPU (about 21.8 GB). A process that called this
    shutdown had its engine stopped in 0.9 s and exited normally (17:12:47 UTC). MPClient.shutdown
    (vllm/v1/engine/core_client.py) sends the engine process SIGTERM, waits up to the timeout and then kills it. A failure
    is printed, and any child process still running is then stopped (stop_children), so that the process can exit; the
    next process's wait_for_gpu stops if the GPU has not been freed."""
    t0 = time.time()
    try:
        llm.llm_engine.engine_core.shutdown(timeout=timeout)
        print(f"vLLM engine shut down in {time.time() - t0:.1f} s", flush=True)
    except Exception as e:
        print(f"vLLM engine shutdown failed after {time.time() - t0:.1f} s: {type(e).__name__}: {e}", flush=True)
    stop_children(timeout)


def stop_children(timeout=30):
    """Stops every child process still running (correction 12, from the review of 17:50 UTC). vLLM's engine process is
    not a daemon, and at exit Python's multiprocessing joins such children, possibly before the finalizers that would stop
    them run, so a child left running holds the exit, and the GPU. After a normal shutdown there is none."""
    import multiprocessing
    left = multiprocessing.active_children()
    if not left:
        return
    print(f"child processes still running after vLLM's shutdown: {[p.name for p in left]}; stopping them", flush=True)
    for p in left:
        p.terminate()
    deadline = time.time() + timeout
    for p in left:
        p.join(max(0.1, deadline - time.time()))
        if p.is_alive():
            p.kill()
            p.join(10)


def gpu_used_mib():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else None


def wait_for_gpu(limit_mib=1000, seconds=300):
    """Waits until the GPU's used memory is under the limit (another process may still be leaving it)."""
    t0 = time.time()
    while True:
        used = gpu_used_mib()
        if used is None or used < limit_mib:
            return used
        if time.time() - t0 > seconds:
            raise SystemExit(f"the GPU still holds {used} MiB after {seconds} s; this process runs alone; stop")
        time.sleep(10)


def answer_from_top(top, ids_by_letter, method, tokenizer, r, n_tokens):
    probs, mass, per_letter, missing = option_probabilities(top, ids_by_letter)
    best = max(top, key=top.get)
    return {"id": r["id"], "label": r["label"], "prompt_tokens": n_tokens, "method": method if probs else "pending",
            "probs": probs, "letter_mass": mass, "per_letter": per_letter, "missing": missing,
            "top_token": tokenizer.convert_ids_to_tokens(best), "top_logprob": top[best]}


def answer(qids, splits, model_dir, items_dir, results=RESULTS):
    import vllm
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ids_by_letter, left_out = letter_ids(tokenizer)
    work = {}  # every prompt templated and checked before the model is loaded
    for qid in qids:
        for split in splits:
            rows = read_jsonl(Path(items_dir) / f"prompts_{qid}_{split}.jsonl")
            token_ids = [templated_ids(tokenizer, r["text"]) for r in rows]
            # the second process scores the prompt followed by one letter, and vLLM needs room for one new token after it
            too_long = [r["id"] for r, t in zip(rows, token_ids) if len(t) + 1 >= MAX_LEN]
            if too_long:
                raise SystemExit(f"{len(too_long)} {qid} {split} prompts do not fit {MAX_LEN} tokens with a letter appended, "
                                 f"for example {too_long[:3]}")
            work[(qid, split)] = (rows, token_ids)
    used = wait_for_gpu()
    t_load = time.time()
    llm = make_llm(model_dir, FIRST_UTILIZATION, max_logprobs=TOP_AGAIN)
    load_seconds = time.time() - t_load
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP)
    params_again = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_AGAIN)
    timing = {"vllm": vllm.__version__, "load_seconds_first": load_seconds, "gpu_used_mib_before_load": used, "passes": {}}
    pending = 0
    try:
        for (qid, split), (rows, token_ids) in work.items():
            t0 = time.time()
            outputs = llm.generate([{"prompt_token_ids": t} for t in token_ids], params, use_tqdm=False)
            seconds = time.time() - t0
            check_order(outputs, token_ids)
            first = []
            for r, t, out in zip(rows, token_ids, outputs):
                top = {int(k): as_logprob(v) for k, v in out.outputs[0].logprobs[0].items()}
                first.append(answer_from_top(top, ids_by_letter, "top20", tokenizer, r, len(t)))
            again = [k for k, x in enumerate(first) if x["method"] == "pending"]
            seconds_again = 0.0
            if again:
                t0 = time.time()
                outputs = llm.generate([{"prompt_token_ids": token_ids[k]} for k in again], params_again, use_tqdm=False)
                seconds_again = time.time() - t0
                check_order(outputs, [token_ids[k] for k in again])
                for k, out in zip(again, outputs):
                    top = {int(j): as_logprob(v) for j, v in out.outputs[0].logprobs[0].items()}
                    first[k] = answer_from_top(top, ids_by_letter, "top1000", tokenizer, rows[k], len(token_ids[k]))
            left = sum(1 for x in first if x["method"] == "pending")
            pending += left
            write_jsonl(Path(items_dir) / f"first_{qid}_{split}.jsonl", first)
            timing["passes"][f"{qid}_{split}"] = {"first_items": len(rows), "first_seconds": seconds, "top1000_items": len(again),
                                                   "top1000_seconds": seconds_again, "prompt_tokens_mean": float(np.mean([len(t) for t in token_ids]))}
            print(f"{qid} {split}: {len(rows)} items in {seconds:.1f} s; {len(again)} asked again with the top {TOP_AGAIN} "
                  f"({seconds_again:.1f} s); {left} left for the second process", flush=True)
    finally:  # on every way out, also an error (correction 12)
        close_llm(llm)
    timing.update({"letter_ids": ids_by_letter, "variants_left_out": left_out, "gpu_memory_utilization_first": FIRST_UTILIZATION,
                   "max_logprobs": TOP_AGAIN, "time_utc_first": now()})
    save_json(Path(items_dir) / "timing_first.json", timing)
    if pending == 0:
        finish(qids, splits, items_dir, results)
    else:
        print(f"{pending} items need the second process: run second", flush=True)
    return pending


def second(qids, splits, model_dir, items_dir, results=RESULTS):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ids_by_letter, _ = letter_ids(tokenizer)
    work = []
    for qid in qids:
        for split in splits:
            first = read_jsonl(Path(items_dir) / f"first_{qid}_{split}.jsonl")
            texts = {r["id"]: r["text"] for r in read_jsonl(Path(items_dir) / f"prompts_{qid}_{split}.jsonl")}
            work += [(qid, split, r, texts[r["id"]]) for r in first if r["method"] == "pending"]
    timing_path = Path(items_dir) / "timing_first.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    if work:
        variants = [i for letter in LETTERS for i in ids_by_letter[letter]]
        requests, owners = [], []
        for n, (_, _, r, text) in enumerate(work):
            base = templated_ids(tokenizer, text)
            for v in variants:
                requests.append({"prompt_token_ids": base + [v]})
                owners.append((n, v))
        timing["gpu_used_mib_before_second_load"] = wait_for_gpu()
        t_load = time.time()
        llm = make_llm(model_dir, SECOND_UTILIZATION, max_num_seqs=1, max_num_batched_tokens=SECOND_BATCHED_TOKENS)
        timing["load_seconds_second"] = time.time() - t_load
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        try:
            t0 = time.time()
            outputs = llm.generate(requests, params, use_tqdm=False)
            seconds = time.time() - t0
        finally:  # on every way out, also an error (correction 12)
            close_llm(llm)
        check_order(outputs, [q["prompt_token_ids"] for q in requests])
        found = {}
        for (n, v), out in zip(owners, outputs):
            found.setdefault(n, {})[v] = as_logprob(out.prompt_logprobs[-1][v])
        second_rows = {}
        for n, (qid, split, r, _) in enumerate(work):
            probs, mass, per_letter = from_prompt_logprobs(found[n], ids_by_letter)
            second_rows.setdefault((qid, split), []).append({**r, "method": "prompt", "probs": probs, "letter_mass": mass,
                                                              "per_letter": per_letter, "variant_logprobs": {str(k): v for k, v in found[n].items()}})
        items_per_pass = {k: len(v) for k, v in second_rows.items()}
        for (qid, split), rows in second_rows.items():
            write_jsonl(Path(items_dir) / f"second_{qid}_{split}.jsonl", rows)
            share = len(rows) / sum(items_per_pass.values())
            timing["passes"][f"{qid}_{split}"]["second_items"] = len(rows)
            timing["passes"][f"{qid}_{split}"]["second_seconds"] = seconds * share  # the pass's time shared by items
        timing["gpu_memory_utilization_second"] = SECOND_UTILIZATION
        timing["max_num_batched_tokens_second"] = SECOND_BATCHED_TOKENS
        timing["time_utc_second"] = now()
        save_json(timing_path, timing)
    finish(qids, splits, items_dir, results)
    return len(work)


# ---------------------------------------------------------------- results

def write_raw(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p0", "p1", "p2"])
        for r in rows:
            w.writerow([r["id"], r["label"]] + [f"{x:.9g}" for x in r["probs"]])


def write_logprobs(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "method", "letter_mass", "logp_A", "logp_B", "logp_C", "top_token"])
        for r in rows:
            w.writerow([r["id"], r["label"], r["method"], f"{r['letter_mass']:.9g}"]
                       + [repr(math.log(p)) if p > 0 else "-inf" for p in r["per_letter"]] + [r.get("top_token", "")])


def finish(qids, splits, items_dir, results=RESULTS):
    out_dir = Path(results) / "large_model"
    timing_first = json.loads((Path(items_dir) / "timing_first.json").read_text(encoding="utf-8"))
    letters_path, timing_path = out_dir / f"{SYSTEM}_letters.json", out_dir / f"{SYSTEM}_timing.json"
    letters = json.loads(letters_path.read_text(encoding="utf-8")) if letters_path.exists() else {"plan": PLAN}
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {"plan": PLAN}
    for qid in qids:
        for split in splits:
            first = read_jsonl(Path(items_dir) / f"first_{qid}_{split}.jsonl")
            second_path = Path(items_dir) / f"second_{qid}_{split}.jsonl"
            redo = {r["id"]: r for r in read_jsonl(second_path)} if second_path.exists() else {}
            # only items this run left pending take a second-process answer (a second file from an earlier run is ignored;
            # correction 9)
            rows = [redo.get(r["id"], r) if r["method"] == "pending" else r for r in first]
            left = [r["id"] for r in rows if r["method"] == "pending"]
            if left:
                raise SystemExit(f"{qid} {split}: {len(left)} items have no answer yet; run second")
            write_raw(out_dir / f"{SYSTEM}-raw_{qid}_{split}.csv", rows)
            write_logprobs(out_dir / f"{SYSTEM}_logprobs_{qid}_{split}.csv", rows)
            mass = np.array([r["letter_mass"] for r in rows])
            methods = {m: sum(1 for r in rows if r["method"] == m) for m in ("top20", "top1000", "prompt")}
            letters.setdefault(qid, {})[split] = {
                "items": len(rows), "letters_summed_probability": {"median": float(np.median(mass)), "p10": float(np.quantile(mass, 0.10)),
                                                                   "share_below_0_5": float(np.mean(mass < 0.5))},
                "answered_second_way": methods["top1000"] + methods["prompt"], "by_method": methods,
                "note": ("before renormalization; the prompt way's sum is exact, the top-20 and top-1,000 ways count the letter variants "
                         "among those log-probabilities (correction 5)")}
            p = timing_first["passes"][f"{qid}_{split}"]
            wall = p["first_seconds"] + p.get("top1000_seconds", 0.0) + p.get("second_seconds", 0.0)
            timing.setdefault(qid, {})[split] = {"items": len(rows), "wall_seconds": wall, "gpu_seconds_per_item": wall / max(1, len(rows)),
                                                 "prompt_tokens_per_item": p["prompt_tokens_mean"], "first_pass_seconds": p["first_seconds"],
                                                 "top1000_items": p.get("top1000_items", 0), "top1000_seconds": p.get("top1000_seconds", 0.0),
                                                 "second_pass_items": p.get("second_items", 0), "second_pass_seconds": p.get("second_seconds", 0.0),
                                                 "note": "the process runs alone on the GPU, so its wall time is GPU time; model loading not counted"}
            print(f"{qid} {split}: {len(rows)} rows written; letters' median summed probability {np.median(mass):.4f}", flush=True)
    letters.update({"letter_ids": timing_first.get("letter_ids"), "variants_left_out": timing_first.get("variants_left_out"),
                    "time_utc": now()})
    timing.update({"vllm": timing_first.get("vllm"), "load_seconds_first": timing_first.get("load_seconds_first"),
                   "load_seconds_second": timing_first.get("load_seconds_second"), "time_utc": now()})
    save_json(letters_path, letters)
    save_json(timing_path, timing)
    return letters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("prompts", "answer", "second", "finish"))
    parser.add_argument("--qids", nargs="+", default=["evidence_effect", "pubmedqa_answer"])
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), required=True)
    parser.add_argument("--items-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--records-public", type=Path, default=RECORDS_PUBLIC)
    parser.add_argument("--records-ei", type=Path, default=RECORDS_EI)
    args = parser.parse_args(argv)
    if args.command in ("answer", "second") and args.model_dir is None:
        parser.error("--model-dir is needed")
    if "test" in args.splits and args.command in ("prompts", "answer", "second"):
        for name in ("release_candidate.json", "testpreds_log.json"):
            if not (args.results / name).exists():
                raise SystemExit(f"{name} does not exist; the large model answers test items only in the testpreds stage")
    if args.command == "prompts":
        prompts(args.qids, args.splits, args.items_dir, args.records_public, args.records_ei)
    elif args.command == "answer":
        answer(args.qids, args.splits, args.model_dir, args.items_dir, args.results)
    elif args.command == "second":
        second(args.qids, args.splits, args.model_dir, args.items_dir, args.results)
    else:
        finish(args.qids, args.splits, args.items_dir, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

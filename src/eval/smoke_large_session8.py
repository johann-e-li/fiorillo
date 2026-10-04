"""Smoke test for src/eval/large_model_session8.py's vLLM calls (docs/prompt_session8.md, corrections 9 and 12), run by
the pod stage smoke_large in /root/eddm_s8/vllm_env/ beside another GPU job. It uses the script's own functions with
Gemma 4 E2B-it, which vLLM 0.30.0 runs through the same class as the 31B (Gemma4ForConditionalGeneration, gemma4_mm.py):
sliding layers of head size 256 (FlashAttention 3) and full layers of head size 512 (FlashAttention 4, compiled at engine
start by the CuTe DSL), final-logit soft-capping at 30, and the same tokenizer.json. The prompts are templated with the
31B's own chat template (E2B's chat_template.jinja differs), so the prompts and letter IDs are the real ones.

  first            the answer process's calls: make_llm(max_logprobs=1000), logprobs=20 then logprobs=1000 on the same
                   engine; checks the containers
  second           the second process's calls: make_llm(max_num_seqs=1, max_num_batched_tokens=1024), prompt_logprobs=0
                   on each prompt followed by each letter variant; checks the containers, and that in this same engine
                   each variant's prompt log-probability equals its next-token log-probability (correction 12; the
                   engine is built with max_logprobs 1,000 so that it can also return the top 1,000, a setting that only
                   bounds what a request may ask for); the differences from the first process are reported, not checked
  pipeline_answer  the script's own answer() on the six smoke items, as large_validation runs it (correction 12)
  pipeline_second  the script's own second() and finish() in a new process; checks the files finish writes, and that
                   second()'s log-probabilities equal the second mode's (both run one sequence at a time)
Each mode is its own process, as on the real run, and shuts its engine down with the script's close_llm (correction
12); the stage checks after each process that the GPU is back at its level from before the smoke test.

Why the first process's numbers are not held to a limit (correction 12): in the run of 17:05 UTC on 30 September 2026,
E2B's letters sat near -13 nats (its top token was "The") and the second process's values differed from the first
process's by up to 1.01 nats. In a hand check at 17:09 to 17:12 UTC, with one sequence at a time, the prompt and the
next-token log-probabilities agreed exactly within each engine, and chunked (1,024 tokens) and whole (8,192) prefill
gave the same numbers. The first engine schedules the six prompts together, so the difference comes from that batching:
vLLM's kernels are not batch-invariant. Each item's letters all come from one engine, so this is never mixed within an
item.

For the pipeline modes the GPU is not free (another job runs beside), so wait_for_gpu is replaced by a check that the
GPU holds no more than it did before the smoke test (--baseline-mib, plus 3,000 MiB); the utilization is the smoke
test's; and the top-1,000 pass is made a second top-20 pass (TOP_AGAIN = TOP), so that every item E2B leaves without a
letter in its top 20 goes on to the second process, which then runs with work. --pipeline-model-dir is E2B's folder with
the 31B's chat template linked in, and --pipeline-dir a new folder under --out-dir for each smoke run (the stage makes
both). The file names say gemma4_31b because they are the script's. Writes only under --out-dir.
"""
import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

QIDS = ("evidence_effect", "pubmedqa_answer")


def load_script(path):
    spec = importlib.util.spec_from_file_location("lm8", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    except FileNotFoundError:
        return "nvidia-smi not found"


def pick_prompts(items_dir, n_ei=4, n_pq=2):
    """The longest EI validation prompts and the first PubMedQA ones, from the prompts subcommand's files."""
    rows = []
    ei = Path(items_dir) / "prompts_evidence_effect_validation.jsonl"
    pq = Path(items_dir) / "prompts_pubmedqa_answer_validation.jsonl"
    if not ei.exists() or not pq.exists():
        sys.exit(f"run the prompts subcommand first (missing {ei} or {pq})")
    read = lambda p: [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    rows += sorted(read(ei), key=lambda r: -len(r["text"]))[:n_ei]
    rows += read(pq)[:n_pq]
    return rows


def first(args, lm8):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    ids_by_letter, left_out = lm8.letter_ids(tok)
    print("letter ids", ids_by_letter, "left out", left_out, flush=True)
    rows = pick_prompts(args.items_dir)
    token_ids = [lm8.templated_ids(tok, r["text"]) for r in rows]
    print("prompt tokens:", [len(t) for t in token_ids], flush=True)
    print("GPU used before load (MiB):", gpu_used(), flush=True)
    t0 = time.time()
    llm = lm8.make_llm(args.model_dir, args.util, max_logprobs=lm8.TOP_AGAIN)
    print(f"load {time.time() - t0:.1f} s; GPU used after load (MiB): {gpu_used()}", flush=True)
    result = {"ids_by_letter": ids_by_letter, "items": []}
    try:
        for k_top in (lm8.TOP, lm8.TOP_AGAIN):
            params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k_top)
            t0 = time.time()
            outs = llm.generate([{"prompt_token_ids": t} for t in token_ids], params, use_tqdm=False)
            print(f"logprobs={k_top}: {len(outs)} requests in {time.time() - t0:.2f} s", flush=True)
            lm8.check_order(outs, token_ids)
            for r, t, out in zip(rows, token_ids, outs):
                lp = out.outputs[0].logprobs
                assert isinstance(lp, list) and len(lp) == 1 and isinstance(lp[0], dict), f"logprobs container: {type(lp)}"
                top = {int(k): lm8.as_logprob(v) for k, v in lp[0].items()}
                assert k_top <= len(top) <= k_top + 1, f"{len(top)} entries for logprobs={k_top}"
                x = lm8.answer_from_top(top, ids_by_letter, f"top{k_top}", tok, r, len(t))
                print(f"  {r['id']}: method {x['method']}, letters' mass {x['letter_mass']:.4f}, top token {x['top_token']!r}, "
                      f"missing {x['missing']}", flush=True)
                if k_top == lm8.TOP_AGAIN:
                    variants = {str(i): top.get(i) for letter in lm8.LETTERS for i in ids_by_letter[letter]}
                    result["items"].append({"id": r["id"], "text": r["text"], "variants_top1000": variants})
    finally:
        lm8.close_llm(llm)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "smoke_first.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("FIRST PASS OK", flush=True)


def second(args, lm8):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    first_result = json.loads((Path(args.out_dir) / "smoke_first.json").read_text(encoding="utf-8"))
    ids_by_letter, _ = lm8.letter_ids(tok)
    variants = [i for letter in lm8.LETTERS for i in ids_by_letter[letter]]
    bases, requests, owners = [], [], []
    for n, item in enumerate(first_result["items"]):
        base = lm8.templated_ids(tok, item["text"])
        bases.append(base)
        for v in variants:
            requests.append({"prompt_token_ids": base + [v]})
            owners.append((n, v))
    print("GPU used before load (MiB):", gpu_used(), flush=True)
    t0 = time.time()
    llm = lm8.make_llm(args.model_dir, args.util, max_logprobs=lm8.TOP_AGAIN, max_num_seqs=1,
                       max_num_batched_tokens=lm8.SECOND_BATCHED_TOKENS)
    print(f"load {time.time() - t0:.1f} s", flush=True)
    try:
        # the next-token requests first, as in the hand check (the prompt-logprob requests do not read the prefix cache)
        t0 = time.time()
        nxt = llm.generate([{"prompt_token_ids": b} for b in bases],
                           SamplingParams(max_tokens=1, temperature=0.0, logprobs=lm8.TOP_AGAIN), use_tqdm=False)
        outs = llm.generate(requests, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
        print(f"{len(nxt)} next-token and {len(outs)} prompt-logprob requests in {time.time() - t0:.2f} s", flush=True)
    finally:
        lm8.close_llm(llm)
    lm8.check_order(nxt, bases)
    lm8.check_order(outs, [q["prompt_token_ids"] for q in requests])
    same = across = 0.0
    n_same = n_across = 0
    saved = {}
    for (n, v), req, out in zip(owners, requests, outs):
        pl = out.prompt_logprobs
        assert len(pl) == len(req["prompt_token_ids"]) and pl[0] is None, "one entry per prompt token, None first"
        assert set(pl[-1]) == {v}, f"last entry holds only the appended letter: {sorted(pl[-1])}"
        got = lm8.as_logprob(pl[-1][v])
        assert math.isfinite(got) and got <= 0.0
        top = {int(k): lm8.as_logprob(x) for k, x in nxt[n].outputs[0].logprobs[0].items()}
        here = top.get(v)
        ref = first_result["items"][n]["variants_top1000"].get(str(v))
        item_id = first_result["items"][n]["id"]
        saved[f"{item_id}|{v}"] = got
        if here is not None:
            same, n_same = max(same, abs(got - here)), n_same + 1
        if ref is not None:
            across, n_across = max(across, abs(got - ref)), n_across + 1
        print(f"  {item_id} variant {v}: prompt logprob {got:.4f}; next-token here "
              f"{'absent' if here is None else f'{here:.4f}'}; first process {'absent' if ref is None else f'{ref:.4f}'}", flush=True)
    (Path(args.out_dir) / "smoke_second.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print(f"same engine: {n_same} of {len(owners)} variants in its top 1,000; largest |prompt - next-token| {same:.6f}", flush=True)
    print(f"against the first process: {n_across} variants compared; largest difference {across:.4f} "
          "(reported, not checked: the first engine batches the prompts; correction 12)", flush=True)
    # the hand check of 17:09 UTC, with this engine's settings and order, found 25 of the 36 in the top 1,000
    assert n_same >= len(owners) // 3, "too few variants in this engine's top 1,000 to compare"
    assert same <= 1e-3, "the prompt log-probability differs from the next-token log-probability in the same engine"
    print("SECOND PASS OK", flush=True)


def pipeline(args, lm8):
    """The script's own answer() or second() (with finish()) on the six smoke items; see the module's docstring."""
    if args.baseline_mib is None or args.pipeline_model_dir is None or args.pipeline_dir is None:
        sys.exit("the pipeline modes need --baseline-mib, --pipeline-model-dir and --pipeline-dir")
    root = Path(args.pipeline_dir)  # a new folder for each smoke run, so no file from an earlier run is read
    if args.mode == "pipeline_answer" and root.exists():
        sys.exit(f"{root} exists already; each smoke run uses a new folder")
    items, results = root / "items", root / "results"
    lm8.FIRST_UTILIZATION = lm8.SECOND_UTILIZATION = args.util
    lm8.TOP_AGAIN = lm8.TOP

    def gpu_back(limit_mib=None, seconds=120):
        t0 = time.time()
        while True:
            used = lm8.gpu_used_mib()
            if used is not None and used <= args.baseline_mib + 3000:
                print(f"GPU used {used} MiB (before the smoke test: {args.baseline_mib} MiB)", flush=True)
                return used
            if time.time() - t0 > seconds:
                raise SystemExit(f"the GPU holds {used} MiB against {args.baseline_mib} MiB before the smoke test: "
                                 "an earlier smoke process left its engine; stop")
            time.sleep(5)

    lm8.wait_for_gpu = gpu_back
    if args.mode == "pipeline_answer":
        rows = pick_prompts(args.items_dir)
        for qid in QIDS:
            lm8.write_jsonl(items / f"prompts_{qid}_validation.jsonl", [r for r in rows if r["qid"] == qid])
        pending = lm8.answer(list(QIDS), ["validation"], args.pipeline_model_dir, items, results)
        print(f"answer(): {pending} of {len(rows)} items left for the second process", flush=True)
        assert pending >= 1, "no item was left for the second process, so second() would load no engine"
        print("PIPELINE ANSWER OK", flush=True)
        return
    n = lm8.second(list(QIDS), ["validation"], args.pipeline_model_dir, items, results)
    print(f"second(): {n} items answered from prompt log-probabilities", flush=True)
    out = results / "large_model"
    letters = json.loads((out / "gemma4_31b_letters.json").read_text(encoding="utf-8"))
    timing = json.loads((out / "gemma4_31b_timing.json").read_text(encoding="utf-8"))
    for qid, k in (("evidence_effect", 4), ("pubmedqa_answer", 2)):
        with open(out / f"gemma4_31b-raw_{qid}_validation.csv", encoding="utf-8") as f:
            raw = list(csv.reader(f))
        assert raw[0] == ["id", "label", "p0", "p1", "p2"] and len(raw) == k + 1, f"{qid}: {len(raw) - 1} rows"
        for row in raw[1:]:
            p = [float(x) for x in row[2:]]
            assert min(p) >= 0 and abs(sum(p) - 1) < 1e-6, row
        with open(out / f"gemma4_31b_logprobs_{qid}_validation.csv", encoding="utf-8") as f:
            assert len(list(csv.reader(f))) == k + 1
        v, t = letters[qid]["validation"], timing[qid]["validation"]
        assert v["items"] == k and sum(v["by_method"].values()) == k, v
        assert t["items"] == k and t["wall_seconds"] > 0 and t["second_pass_items"] == v["by_method"]["prompt"], t
        print(f"{qid}: {k} rows; methods {v['by_method']}; letters' median summed probability "
              f"{v['letters_summed_probability']['median']:.3g}; wall seconds {t['wall_seconds']:.2f}", flush=True)
    assert n == sum(letters[q]["validation"]["by_method"]["prompt"] for q in QIDS) and n >= 1
    ref = json.loads((Path(args.out_dir) / "smoke_second.json").read_text(encoding="utf-8"))
    worst, compared, total = 0.0, 0, 0
    for qid in QIDS:
        path = items / f"second_{qid}_validation.jsonl"
        for r in (lm8.read_jsonl(path) if path.exists() else []):
            for v, x in r["variant_logprobs"].items():
                total += 1
                if f"{r['id']}|{v}" in ref:
                    worst, compared = max(worst, abs(x - ref[f"{r['id']}|{v}"])), compared + 1
    print(f"second() against the second mode: {compared} of {total} variants compared; largest difference {worst:.6f}", flush=True)
    assert compared == total >= 1 and worst <= 1e-3, "second() and the second mode disagree"
    print("PIPELINE SECOND OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("first", "second", "pipeline_answer", "pipeline_second"))
    ap.add_argument("--script", type=Path, default=Path("/root/eddm_s8/code/src/eval/large_model_session8.py"))
    ap.add_argument("--tokenizer-dir", type=Path, default=Path("/root/eddm_s8/models/gemma4_31b"))
    ap.add_argument("--model-dir", type=Path, default=Path("/root/eddm_s8/models/gemma4_e2b_smoke"))
    ap.add_argument("--pipeline-model-dir", type=Path, default=None)
    ap.add_argument("--pipeline-dir", type=Path, default=None)
    ap.add_argument("--items-dir", type=Path, default=Path("/root/eddm_s8/large_items"))
    ap.add_argument("--out-dir", type=Path, default=Path("/root/eddm_s8/smoke_large"))
    ap.add_argument("--util", type=float, default=0.25)
    ap.add_argument("--baseline-mib", type=int, default=None)
    args = ap.parse_args()
    lm8 = load_script(args.script)
    {"first": first, "second": second}.get(args.mode, pipeline)(args, lm8)


if __name__ == "__main__":
    main()

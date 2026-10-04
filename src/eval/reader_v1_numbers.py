"""Reader v1's model card numbers (29 September 2026, at Johann's request to release reader v1; docs/prompt_session5.md,
correction 6). Reads only files already on disk and writes results/reader_v1/card_numbers.json, which
scripts/build_model_card_reader_v1.py turns into docs/model_card_reader_v1.md. Nothing is trained or selected here.

  - the text: results/reader_v1/text_manifests/abstracts_manifest.json, pubmedqa_exclusion.json and
    results/reader_v1/block_counts/abstracts.json;
  - the training: results/reader_v1/mlm_mix_a/ and mlm_long/ (run.json, throughput.json, heldout_losses.json) and
    mlm_init_report.json;
  - the comparisons fixed before training (D3 of docs/prompt_session5.md), read from results/fiorillo_v0_2/comparison.json;
  - one comparison that was not fixed before training, computed here with compare.py's metrics, resamples and seed
    (paired bootstrap of test rows): reader v1 against Laya's own reader, each under the same fresh decision head and
    recipe (s5_final_fresh_head minus s5_laya); the card labels it as computed afterwards;
  - release criterion 4 of Fiorillo v0.2, which the decision layer on reader v1 did not pass.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402
import metrics as M  # noqa: E402

R = ROOT / "results" / "reader_v1"
F = ROOT / "results" / "fiorillo_v0_2"
AFTER = ("s5_final_fresh_head", "s5_laya")  # reader v1 minus Laya's reader, same head and recipe
SYSTEMS = ("s5_laya", "s5_mix_a", "s5_mix_b", "s5_mix_c", "s5_bioclinical", "s5_final_fresh_head", "arm_laya")  # arm_laya is v0.1
D3_PAIRS = [("s5_mix_a", "s5_laya"), ("s5_mix_b", "s5_laya"), ("s5_mix_c", "s5_laya"), ("s5_bioclinical", "s5_laya"),
            ("s5_final_fresh_head", "s5_bioclinical")]
NAMES = {"pubmedqa_answer": ["log_loss", "accuracy", "ece"], "readmit_window": ["log_loss", "accuracy", "auroc_30d", "ece"]}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def after_the_fact():
    out = {}
    for ev, names in NAMES.items():
        rows = C.reference_rows(ROOT / "data" / "processed" / "public", ev)
        ids, labels = [r["id"] for r in rows], np.array([r["label"] for r in rows])
        a, b = (C.load(C.prediction_path(F / "predictions", s, ev), s, ev, ids, labels) for s in AFTER)
        est, low, high = M.bootstrap(lambda idx: C.scores(names, labels[idx], a[idx]) - C.scores(names, labels[idx], b[idx]),
                                     len(labels), C.RESAMPLES, C.SEED)
        ea = C.scores(names, labels, a)
        eb = C.scores(names, labels, b)
        out[ev] = {m: {AFTER[0]: float(ea[j]), AFTER[1]: float(eb[j]),
                       "difference": {"estimate": float(est[j]), "low": float(low[j]), "high": float(high[j])}}
                   for j, m in enumerate(names)}
        out[ev]["n"] = len(labels)
    return out


def planned_d3():
    comp = load_json(F / "comparison.json")
    keep = set(D3_PAIRS)
    out = {}
    for r in comp["comparisons"]:
        pair = (r["system"], r["reference"])
        if r["section"] == "D3" and pair in keep and r["evaluation"] in NAMES and r["metric"] in NAMES[r["evaluation"]]:
            out.setdefault(f"{pair[0]} minus {pair[1]}", {}).setdefault(r["evaluation"], {})[r["metric"]] = {
                pair[0]: r["system_estimate"], pair[1]: r["reference_estimate"], "difference": r["difference"], "n": r["n"]}
    systems = {ev: {name: {m: comp["metrics"][ev][name][m] for m in NAMES[ev]} for name in SYSTEMS} for ev in NAMES}
    return out, comp["release_bar"], systems


def run_to_run():
    """The largest move of test log loss between two trainings of the same reader with nearly the same recipe, per
    question, from results/fiorillo_v0_2/retry_check.json (the pairs it counts)."""
    rc = load_json(F / "retry_check.json")
    pairs = [r for r in rc["run_to_run_pairs"] if r.get("counted_in_spread")]
    return {q: {"largest_move": max(abs(r[q]["difference"]) for r in pairs), "pairs": [r["what"] for r in pairs],
                "sd_per_run": rc["run_to_run_sd_per_run"][q]} for q in ("pubmedqa_answer", "readmit_window")}


def main():
    import yaml
    mixes = yaml.safe_load((ROOT / "src" / "configs" / "mixes_session5.yaml").read_text(encoding="utf-8"))["mixes"]
    selection = load_json(F / "release_candidate.json")["selection"]
    heldout_counts = {src: {k: load_json(R / "block_counts" / f"{src}.json")[k] for k in ("heldout_blocks", "heldout_documents")}
                      for src in ("abstracts", "pmc", "case_reports", "web", "task")}
    manifest = load_json(R / "text_manifests" / "abstracts_manifest.json")
    exclusion = load_json(R / "text_manifests" / "pubmedqa_exclusion.json")
    blocks = load_json(R / "block_counts" / "abstracts.json")
    runs = {name: {"run": load_json(R / f"mlm_{name}" / "run.json"), "throughput": load_json(R / f"mlm_{name}" / "throughput.json"),
                   "heldout_last": load_json(R / f"mlm_{name}" / "heldout_losses.json")[-1]} for name in ("mix_a", "long")}
    d3, bar, systems = planned_d3()
    mix_a, long = runs["mix_a"], runs["long"]
    resumed = long["run"]["resumed_at"]
    derived_seconds = mix_a["throughput"]["seconds"] * resumed / mix_a["run"]["steps"] + long["throughput"]["seconds"]
    out = {
        "source": "src/eval/reader_v1_numbers.py",
        "text": {"documents": manifest["documents"], "tokens": manifest["tokens"], "drops": manifest["drops"],
                 "shards_used": len(manifest["shards_used"]),
                 "rules": manifest["rules"], "pubmedqa_pubids_excluded": exclusion["distinct_pubids_all_configs"],
                 "block_tokens": blocks["block_tokens"], "train_blocks": blocks["train_blocks"],
                 "heldout_blocks": blocks["heldout_blocks"], "heldout_documents": blocks["heldout_documents"]},
        "init": load_json(R / "mlm_init_report.json"),
        "training": {"steps": long["run"]["steps"], "resumed_at": resumed, "warmup": long["run"]["warmup"],
                     "decay_start": long["run"]["decay_start"], "peak_lr": long["run"]["peak_lr"],
                     "blocks_per_step": long["run"]["blocks_per_step"], "tokens_per_step": long["run"]["tokens_per_step"],
                     "seed": long["run"]["seed"], "tokens": long["run"]["tokens_per_step"] * long["run"]["steps"],
                     "blocks_drawn": long["throughput"]["blocks_drawn"]["abstracts"],
                     "mix_a_seconds": mix_a["throughput"]["seconds"], "long_seconds": long["throughput"]["seconds"],
                     "tokens_per_second_long": long["throughput"]["tokens_per_second"],
                     "peak_memory_gb": long["throughput"]["peak_memory_gb"],
                     "gpu_seconds_for_reader_v1_derived": round(derived_seconds, 1)},
        "heldout": {"mix_a_500m": mix_a["heldout_last"], "reader_v1": long["heldout_last"], "counts": heldout_counts},
        "mixes": {k: v["shares"] for k, v in mixes.items() if k in ("A", "B", "C")},
        "run_to_run": run_to_run(),
        "rules": {"reader_rule": {k: selection["reader_rule"][k] for k in ("scores", "margin", "winner", "long_run")},
                  "task_rule": selection["task_rule"]},
        "test_metrics": systems,
        "d3_planned": d3,
        "after_the_fact": {"pair": f"{AFTER[0]} minus {AFTER[1]}", "results": after_the_fact()},
        "v0_2_release_bar": {"overall": bar["overall"], "criterion_4": bar["criteria"]["4"]},
    }
    path = R / "card_numbers.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["after_the_fact"], indent=1))
    print(json.dumps({k: {e: {m: v["difference"] for m, v in d.items()} for e, d in x.items()} for k, x in d3.items()}, indent=1))
    print("derived GPU hours", round(derived_seconds / 3600, 2), "| tokens", out["training"]["tokens"], "| blocks drawn", out["training"]["blocks_drawn"])


if __name__ == "__main__":
    main()

"""THE RECIPE RULE, the weight average's and swap averaging's decisions, and the cost rule (docs/prompt_session9.md: THE
RECIPES, THE RECIPE RULE, THE PILOT AND THE COST RULE, and corrections 1 to 3).

Run on the pod by the train stage, which calls `next` after every step and does what it says, so the stage applies the
rule by itself, resumes from whatever exists, and never waits for Claude. Nothing here reads a test file.

Scores. A system's score is, on the 654 clean validation prompts, the out-of-fold log loss of session 8's calibration rule
(src/eval/calibrate_session7.py's choose(): the lower of the temperature and temperature_intercepts families' out-of-fold
log losses, 5 folds, 20 repeats, the seed), from its log-probabilities: a run's are the log-softmax of its logits
(<results>/logits/<run>_evidence_effect_validation_logits.csv); an average's are the logarithm of the mean of its runs'
softmaxed logits (double precision, as session 8 averaged s8_ei).

The rule, with correction 2 (two runs for each candidate):
- base: s9_b_1, s9_b_2 and s9_b_3 (seeds 20260927, 20260928 and 20260929); s9_b is their probability average.
- the margin m: the largest minus the smallest of the three base runs' scores, and at least 0.005.
- each candidate (swap, long, lr1, lr2) is trained twice, seeds 20260927 and 20260928 (s9_<candidate>_1 and _2), all
  first runs before any second run. Its score is the mean of its two runs' scores; it passes if that is below the mean of
  s9_b_1's and s9_b_2's scores by more than m and each of its runs scores below the base run with the same seed. A
  candidate left with one run (the cost rule's options) is judged by the rule as first written: its run below s9_b_1's
  score by more than m.
- lr1 and lr2 count as one change, the learning rate: if both pass, the one with the lower mean score.
- if two or more of the three changes (swap, long, the learning rate) pass, the combination (every passing change at
  once) is trained twice, seeds 20260927 and 20260928 (s9_combo_1 and _2); it passes if its mean score is below the best
  passing candidate's by more than m and each of its runs scores below that candidate's run with the same seed (with one
  run: below that candidate's first run by more than m).
- the winning recipe: the combination if it passed; otherwise the passing candidate with the lowest score; otherwise
  base. If it is not base, it is trained once more, seed 20260929 (s9_<recipe>_3). s9_ei is the probability average of
  its three runs (for base, s9_b).
- I-39: s9_ei_wavg is built from the same three runs (src/eval/weight_average_session9.py) and predicts the whole
  validation split; v0.5 uses it in place of s9_ei if its score is not above s9_ei's by more than m.
- correction 3: the system v0.5 would use (X, s9_ei or s9_ei_wavg) predicts the exchanged validation records at its
  recipe's length (checks/swap/ or checks/swap_long/); for s9_ei each of its runs does (X_exch is their average). Swap
  averaging: for each prompt, the mean of X's answer and the exchange of X_exch's answer ("significantly increased" and
  "significantly decreased" swapped), as raw probabilities, before calibration; a prompt with no exchanged record keeps its
  single answer. v0.5 uses swap averaging (X_sa) if its score is below X's by more than m.
Runs dropped by Johann's decision under the cost rule are listed in cost_rule.json's "dropped" (lr2, long_2, combo_2,
lr1_2, swap_2, long, combo, swap; correction 2's order), and long is dropped when pilot_long wrote its NOT BUILT marker.

Subcommands:
  next   the train stage's next action, printed as one JSON object (with --format env, as shell assignments): train a run,
         build the combination's records, build and predict the weight average, predict the exchanged validation records,
         or done; it also writes <results>/recipe_selection.json (every score, m, every pass or fail with its numbers, the
         combination's parts, the winning recipe, the weight average's and swap averaging's decisions, each decision's
         time) and the averaged systems' raw validation files in <results>/predictions/.
  cost   the cost rule (cost_rule.json): the spend so far (hours since the pod was created times the hourly rate, from
         /root/eddm_s9/pod_rate.txt) plus the longest path still possible at the measured rates; exits 4 if the projection
         is above the stop line (correction 1: $115) or the projected end of the copy is after an announced maintenance.

Usage (the pod, from /root/eddm_s9/code):
  python src/eval/recipe_session9.py next --format env [--results ...] [--not-built-dir /root/eddm_s9/not_built]
  python src/eval/recipe_session9.py cost --pod-rate /root/eddm_s9/pod_rate.txt
"""
import argparse
import datetime as dt
import json
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.calibrate_session7 import FOLDS, REPEATS, SEED, choose, log_softmax, read_logits, write_probs  # noqa: E402

RESULTS = ROOT / "results" / "fiorillo_v0_5"
RECORDS9 = ROOT / "data" / "processed" / "session9"
EI = "evidence_effect"
N_VALIDATION = 1237
N_CLEAN = 654
PLAN = "docs/prompt_session9.md, THE RECIPE RULE (with corrections 2 and 3)"
SEEDS = {1: 20260927, 2: 20260928, 3: 20260929}
CANDIDATES = ("swap", "long", "lr1", "lr2")
MARGIN_FLOOR = 0.005
SWAP_COLUMNS = [1, 0, 2]  # "significantly increased" and "significantly decreased" exchanged; "no significant difference" kept
LR = {"b": 5e-5, "swap": 5e-5, "long": 5e-5, "lr1": 1e-4, "lr2": 2e-4}
DROP_NAMES = ("lr2", "long_2", "combo_2", "lr1_2", "swap_2", "long", "combo", "swap")  # correction 2's order of what is dropped


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- recipes and runs

def recipe_settings(recipe, parts=None, lr_choice=None):
    """The training settings of a recipe (THE RECIPES): records, items, epochs, learning rate, length and memory setting.
    Paths are relative to the code root; item keys name the pod's items (<ITEMS>/<key>.pt)."""
    if recipe == "combo":
        parts = sorted(parts or [])
        long_ = "long" in parts
        swap = "swap" in parts
        lr = LR[lr_choice] if lr_choice else LR["b"]
        train_key = "combo_train" if swap else ("long_train" if long_ else "clean_train")
        return {"recipe": "combo", "parts": parts, "lr_choice": lr_choice,
                "train_records": ("data/processed/session9/combo/evidence_effect_train.jsonl" if swap else
                                  f"data/processed/session9/{'long' if long_ else 'clean'}/evidence_effect_train.jsonl"),
                "monitor_records": f"data/processed/session9/{'long' if long_ else 'clean'}/evidence_effect_validation_monitor.jsonl",
                "train_items": train_key, "monitor_items": "long_monitor" if long_ else "clean_monitor",
                "epochs": 1 if swap else 2, "lr_adapters": lr, "max_len": 6144 if long_ else 4096,
                "memory_setting": "long" if long_ else "base",
                "validation_records_dir": "data/processed/session9/long" if long_ else "data/processed/session8",
                "exchanged_validation_dir": "data/processed/session9/checks/swap_long" if long_ else "data/processed/session9/checks/swap"}
    base = {"recipe": recipe, "lr_adapters": LR[recipe], "epochs": 2, "max_len": 4096, "memory_setting": "base",
            "train_records": "data/processed/session9/clean/evidence_effect_train.jsonl",
            "monitor_records": "data/processed/session9/clean/evidence_effect_validation_monitor.jsonl",
            "train_items": "clean_train", "monitor_items": "clean_monitor",
            "validation_records_dir": "data/processed/session8",
            "exchanged_validation_dir": "data/processed/session9/checks/swap"}
    if recipe == "swap":
        base.update({"train_records": "data/processed/session9/swap/evidence_effect_train.jsonl", "train_items": "swap_train",
                     "epochs": 1})
    elif recipe == "long":
        base.update({"train_records": "data/processed/session9/long/evidence_effect_train.jsonl", "train_items": "long_train",
                     "monitor_records": "data/processed/session9/long/evidence_effect_validation_monitor.jsonl",
                     "monitor_items": "long_monitor", "max_len": 6144, "memory_setting": "long",
                     "validation_records_dir": "data/processed/session9/long",
                     "exchanged_validation_dir": "data/processed/session9/checks/swap_long"})
    elif recipe not in ("b", "lr1", "lr2"):
        raise ValueError(f"unknown recipe {recipe}")
    return base


def run_name(recipe, k):
    return f"s9_{recipe}_{k}"


# ---------------------------------------------------------------- probabilities and scores

def logits_path(results, system, split="validation"):
    return Path(results) / "logits" / f"{system}_{EI}_{split}_logits.csv"


def has_run(results, system, split="validation", n=N_VALIDATION):
    """A run counts as done when its logits (and raw predictions) are in results with every validation row."""
    lp = logits_path(results, system, split)
    rp = Path(results) / "predictions" / f"{system}-raw_{EI}_{split}.csv"
    if not (lp.exists() and rp.exists()):
        return False
    return sum(1 for _ in open(lp, encoding="utf-8")) - 1 == n and sum(1 for _ in open(rp, encoding="utf-8")) - 1 == n


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_probs(results, system, split="validation"):
    """(ids, labels, probabilities) of one run or of the weight average, from its logits, in double precision."""
    ids, labels, z = read_logits(logits_path(results, system, split))
    return list(ids), np.asarray(labels), softmax(np.asarray(z, dtype=np.float64))


def align(base_ids, ids, labels, probs, base_labels=None, what=""):
    pos = {i: k for k, i in enumerate(ids)}
    if set(pos) != set(base_ids) or len(pos) != len(base_ids):
        raise SystemExit(f"{what}: holds different prompts")
    order = np.array([pos[i] for i in base_ids])
    if base_labels is not None and not np.array_equal(np.asarray(labels)[order], base_labels):
        raise SystemExit(f"{what}: labels differ")
    return np.asarray(probs)[order]


def mean_probs(results, systems, split="validation"):
    """(ids, labels, mean probabilities) of several runs, rows in the first's order."""
    ids, labels, first = run_probs(results, systems[0], split)
    out = [first]
    for s in systems[1:]:
        i2, l2, p2 = run_probs(results, s, split)
        out.append(align(ids, i2, l2, p2, labels, s))
    return ids, labels, np.mean(out, axis=0)


def swap_average(p, p_exch):
    """Correction 3: the mean of an answer and the exchange of the answer to the exchanged prompt (raw probabilities)."""
    return 0.5 * (np.asarray(p) + np.asarray(p_exch)[:, SWAP_COLUMNS])


def sa_probs(ids, labels, p, exch_ids, exch_probs):
    """X_sa: rows with an exchanged answer are averaged with its exchange; a prompt with no exchanged record keeps its
    answer. The exchanged records hold the exchanged labels, which are not read."""
    pos = {i: k for k, i in enumerate(exch_ids)}
    out = np.array(p, dtype=np.float64, copy=True)
    for r, i in enumerate(ids):
        if i in pos:
            out[r] = 0.5 * (p[r] + np.asarray(exch_probs[pos[i]])[SWAP_COLUMNS])
    return out


def clean_ids(records9=RECORDS9):
    path = Path(records9) / "clean" / f"{EI}_validation.jsonl"
    ids = [json.loads(line)["id"] for line in open(path, encoding="utf-8") if line.strip()]
    if len(ids) != N_CLEAN:
        raise SystemExit(f"{path} has {len(ids)} records, expected {N_CLEAN}")
    return ids


def score(ids, labels, probs, keep):
    """Session 8's rule's out-of-fold log loss on the kept (clean) prompts: (score, chosen family, both families)."""
    pos = {i: k for k, i in enumerate(ids)}
    missing = [i for i in keep if i not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} clean prompts are missing, for example {missing[:3]}")
    rows = np.array([pos[i] for i in keep])
    logp = np.log(np.maximum(np.asarray(probs)[rows], 1e-300))
    chosen, fams = choose(logp, np.asarray(labels)[rows])
    return float(fams[chosen]["out_of_fold_log_loss"]["mean"]), chosen, {
        f: fams[f]["out_of_fold_log_loss"]["mean"] for f in fams}


def write_raw(results, system, ids, labels, probs, split="validation"):
    path = Path(results) / "predictions" / f"{system}-raw_{EI}_{split}.csv"
    write_probs(path, ids, labels, np.asarray(probs))
    return path


# ---------------------------------------------------------------- the decision state

def dropped_set(results, not_built_dir):
    cost = load_json(Path(results) / "cost_rule.json", {}) or {}
    dropped = set(cost.get("dropped") or [])
    unknown = dropped - set(DROP_NAMES)
    if unknown:
        raise SystemExit(f"cost_rule.json drops {sorted(unknown)}, which are not among {DROP_NAMES}")
    if not_built_dir and (Path(not_built_dir) / "pilot_long.txt").exists():
        dropped.add("long")
    return dropped


def run_dropped(dropped, recipe, k):
    if recipe in dropped:
        return True
    return k == 2 and f"{recipe}_2" in dropped


class Decider:
    def __init__(self, results=RESULTS, records9=RECORDS9, not_built_dir=None):
        self.results, self.records9 = Path(results), Path(records9)
        self.keep = clean_ids(records9)
        self.dropped = dropped_set(results, not_built_dir)
        self.old = load_json(self.results / "recipe_selection.json", {}) or {}
        self.decisions = dict(self.old.get("decisions") or {})
        self.scores, self.families = {}, {}

    def decide(self, key, value):
        """Records a decision with the time it was first made; a changed value gets a new time."""
        prev = self.decisions.get(key)
        if prev is None or prev.get("value") != value:
            self.decisions[key] = {"value": value, "time_utc": now()}
        return value

    def run_score(self, system):
        if system not in self.scores:
            ids, labels, p = run_probs(self.results, system)
            self.scores[system], _, self.families[system] = score(ids, labels, p, self.keep)
        return self.scores[system]

    def avg_score(self, name, runs, write=True):
        ids, labels, p = mean_probs(self.results, runs)
        if write:
            write_raw(self.results, name, ids, labels, p)
        self.scores[name], _, self.families[name] = score(ids, labels, p, self.keep)
        return self.scores[name], (ids, labels, p)

    def state(self):
        """(the next action or None, recipe_selection's content)."""
        sel = {"plan": PLAN, "written_utc": now(), "n_clean_validation": len(self.keep), "folds": FOLDS, "repeats": REPEATS,
               "seed": SEED, "dropped": sorted(self.dropped)}
        base_runs = [run_name("b", k) for k in (1, 2, 3)]
        for k, r in enumerate(base_runs, start=1):
            if not has_run(self.results, r):
                return self.train_action("b", k), self.finish(sel)
        s_b = {k: self.run_score(run_name("b", k)) for k in (1, 2, 3)}
        spread = max(s_b.values()) - min(s_b.values())
        m = max(spread, MARGIN_FLOOR)
        sel["margin"] = {"m": m, "spread": spread, "floor": MARGIN_FLOOR, "base_scores": {run_name("b", k): v for k, v in s_b.items()}}
        self.decide("margin", round(m, 12))
        s9_b_score, _ = self.avg_score("s9_b", base_runs)
        sel["s9_b"] = {"runs": base_runs, "score": s9_b_score}
        active = [c for c in CANDIDATES if c not in self.dropped]
        for k in (1, 2):
            for c in active:
                if run_dropped(self.dropped, c, k):
                    continue
                if not has_run(self.results, run_name(c, k)):
                    return self.train_action(c, k), self.finish(sel)
        base12 = (s_b[1] + s_b[2]) / 2
        cand = {}
        for c in CANDIDATES:
            if c not in active:
                cand[c] = {"status": "dropped" if c in self.dropped else "not run", "passes": False}
                continue
            runs = [run_name(c, k) for k in (1, 2) if has_run(self.results, run_name(c, k))]
            sc = {r: self.run_score(r) for r in runs}
            if len(runs) == 2:
                mean = (sc[runs[0]] + sc[runs[1]]) / 2
                below_mean = mean < base12 - m
                below_1, below_2 = sc[runs[0]] < s_b[1], sc[runs[1]] < s_b[2]
                passes = bool(below_mean and below_1 and below_2)
                cand[c] = {"runs": runs, "scores": sc, "score": mean, "base_mean_12": base12, "m": m,
                           "below_base_mean_by_more_than_m": bool(below_mean), "run_1_below_s9_b_1": bool(below_1),
                           "run_2_below_s9_b_2": bool(below_2), "passes": passes,
                           "rule": "two runs (correction 2)"}
            else:
                one = sc[runs[0]]
                passes = bool(one < s_b[1] - m)
                cand[c] = {"runs": runs, "scores": sc, "score": one, "passes": passes, "m": m,
                           "rule": "one run (its second was dropped): its run below s9_b_1's score by more than m, the rule as first written"}
            self.decide(f"candidate_{c}", passes)
        sel["candidates"] = cand
        lr_pass = [c for c in ("lr1", "lr2") if cand[c]["passes"]]
        lr_choice = min(lr_pass, key=lambda c: (cand[c]["score"], c)) if lr_pass else None
        sel["learning_rate"] = {"passing": lr_pass, "chosen": lr_choice,
                                "rule": "lr1 and lr2 count as one change; if both pass, the one with the lower mean score"}
        changes = [c for c in ("swap", "long") if cand[c]["passes"]] + ([lr_choice] if lr_choice else [])
        sel["passing_changes"] = changes
        passing_cands = [c for c in ("swap", "long", "lr1", "lr2") if cand[c]["passes"]]
        best = min(passing_cands, key=lambda c: (cand[c]["score"], CANDIDATES.index(c))) if passing_cands else None
        combo = {"called": False, "parts": changes}
        if len(changes) >= 2 and "combo" not in self.dropped:
            combo["called"] = True
            settings = recipe_settings("combo", [c for c in changes if c in ("swap", "long")] + ([lr_choice] if lr_choice else []),
                                       lr_choice)
            combo["settings"] = settings
            self.decide("combination_called", changes)
            if "swap" in changes and not (self.records9 / "combo" / "combo.json").exists():
                return {"action": "combo_records", "parts": [c for c in changes if c in ("swap", "long")]}, self.finish(sel | {"combination": combo})
            if "swap" in changes:
                info = load_json(self.records9 / "combo" / "combo.json", {})
                if sorted(info.get("parts", [])) != sorted(c for c in changes if c in ("swap", "long")):
                    raise SystemExit("combo/combo.json was built for other parts; stop")
                combo["records"] = info
            for k in (1, 2):
                if k == 2 and "combo_2" in self.dropped:
                    continue
                if not has_run(self.results, run_name("combo", k)):
                    return self.train_action("combo", k, combo=settings), self.finish(sel | {"combination": combo})
            runs = [run_name("combo", k) for k in (1, 2) if has_run(self.results, run_name("combo", k))]
            sc = {r: self.run_score(r) for r in runs}
            bscores = cand[best]["scores"]
            brun = {k: run_name(best, k) for k in (1, 2)}
            if len(runs) == 2 and all(brun[k] in bscores for k in (1, 2)):
                mean = (sc[runs[0]] + sc[runs[1]]) / 2
                below_mean = mean < cand[best]["score"] - m
                below = [sc[run_name("combo", k)] < bscores[brun[k]] for k in (1, 2)]
                passes = bool(below_mean and all(below))
                combo.update({"runs": runs, "scores": sc, "score": mean, "best_candidate": best, "best_score": cand[best]["score"],
                              "below_best_by_more_than_m": bool(below_mean), "each_run_below_its_seed": [bool(x) for x in below],
                              "passes": passes, "rule": "two runs (correction 2)"})
            else:
                one = sc[runs[0]]
                passes = bool(one < bscores[brun[1]] - m)
                combo.update({"runs": runs, "scores": sc, "score": one, "best_candidate": best, "passes": passes,
                              "rule": "one run: its run below the best candidate's first run by more than m"})
            self.decide("combination_passes", passes)
        elif len(changes) >= 2:
            combo["status"] = "dropped by Johann's decision under the cost rule"
        sel["combination"] = combo
        if combo.get("passes"):
            winner = "combo"
        elif best:
            winner = best
        else:
            winner = "b"
        self.decide("winner", winner)
        settings = combo["settings"] if winner == "combo" else recipe_settings(winner)
        sel["winner"] = {"recipe": winner, "settings": settings,
                         "rule": "the combination if it passed; otherwise the passing candidate with the lowest score; otherwise base"}
        if winner == "b":
            runs = base_runs
        else:
            if not has_run(self.results, run_name(winner, 3)):
                return self.train_action(winner, 3, combo=settings if winner == "combo" else None), self.finish(sel)
            runs = [run_name(winner, k) for k in (1, 2, 3) if has_run(self.results, run_name(winner, k))]
        sel["winner"]["runs"] = runs
        s9_ei_score, (ids, labels, p_ei) = self.avg_score("s9_ei", runs)
        sel["s9_ei"] = {"runs": runs, "score": s9_ei_score}
        if not has_run(self.results, "s9_ei_wavg"):
            return {"action": "wavg", "name": "s9_ei_wavg", "source_runs": runs, "max_len": settings["max_len"],
                    "validation_records_dir": settings["validation_records_dir"]}, self.finish(sel)
        w_score = self.run_score("s9_ei_wavg")
        uses_wavg = bool(w_score <= s9_ei_score + m)
        self.decide("uses_weight_average", uses_wavg)
        sel["weight_average"] = {"built": True, "score": w_score, "s9_ei_score": s9_ei_score, "m": m, "uses_wavg": uses_wavg,
                                 "rule": "v0.5 uses s9_ei_wavg if its score is not above s9_ei's by more than m"}
        x = "s9_ei_wavg" if uses_wavg else "s9_ei"
        exch_runs = ["s9_ei_wavg"] if uses_wavg else runs
        n_exch = sum(1 for _ in open(records_folder(self.results, settings["exchanged_validation_dir"]) / f"{EI}_validation.jsonl", encoding="utf-8"))
        for r in exch_runs:
            if not has_run(self.results, f"{r}_exch", n=n_exch):
                return {"action": "exch", "name": f"{r}_exch", "phase_system": r, "records_dir": settings["exchanged_validation_dir"],
                        "rows": n_exch}, self.finish(sel)
        if uses_wavg:
            x_ids, x_labels, x_p = run_probs(self.results, "s9_ei_wavg")
            e_ids, _, e_p = run_probs(self.results, "s9_ei_wavg_exch")
        else:
            x_ids, x_labels, x_p = ids, labels, p_ei
            e_ids, e_labels, e_p = mean_probs(self.results, [f"{r}_exch" for r in runs])
            write_raw(self.results, "s9_ei_exch", e_ids, e_labels, e_p)
        p_sa = sa_probs(x_ids, x_labels, x_p, e_ids, e_p)
        write_raw(self.results, f"{x}_sa", x_ids, x_labels, p_sa)
        x_score = w_score if uses_wavg else s9_ei_score
        sa_score, _, self.families[f"{x}_sa"] = score(x_ids, x_labels, p_sa, self.keep)
        self.scores[f"{x}_sa"] = sa_score
        keep_sa = bool(sa_score < x_score - m)
        self.decide("swap_averaging", keep_sa)
        sel["swap_averaging"] = {"system": x, "plain_score": x_score, "swap_averaged_score": sa_score, "m": m, "kept": keep_sa,
                                 "exchanged_records": settings["exchanged_validation_dir"], "exchanged_prompts": n_exch,
                                 "rule": "swap averaging is kept if the swap-averaged score is below the plain one by more than m (correction 3)"}
        sel["v05_ei"] = {"system": f"{x}_sa" if keep_sa else x, "base_system": x, "swap_averaging": keep_sa, "recipe": winner,
                         "runs": runs, "single_pass_systems": exch_runs, "max_len": settings["max_len"],
                         "validation_records_dir": settings["validation_records_dir"],
                         "test_records_dir": settings["validation_records_dir"],
                         "checks_swap_dir": settings["exchanged_validation_dir"],
                         "other_form": x if keep_sa else f"{x}_sa",
                         "swap_augmented": winner == "swap" or (winner == "combo" and "swap" in (combo.get("parts") or []))}
        return {"action": "done"}, self.finish(sel)

    def train_action(self, recipe, k, combo=None):
        settings = combo or recipe_settings(recipe)
        return {"action": "train", "name": run_name(recipe, k), "recipe": recipe, "seed": SEEDS[k], **settings}

    def finish(self, sel):
        sel["scores"] = dict(sorted(self.scores.items()))
        sel["score_families"] = {k: v for k, v in sorted(self.families.items())}
        sel["decisions"] = self.decisions
        save_json(self.results / "recipe_selection.json", sel)
        return sel


def records_folder(results, rel_path):
    """A records folder named relative to the code root, resolved from results' location (results/fiorillo_v0_5)."""
    p = Path(rel_path)
    return p if p.is_absolute() else Path(results).resolve().parents[1] / p


# ---------------------------------------------------------------- the cost rule

FIXED_HOURS = {  # THE PILOT AND THE COST RULE's fixed items still ahead when the cost rule runs (after pilot_long)
    "validation passes for the weight average and swap averaging (correction 3)": 25 / 60,
    "the weight average's build": 10 / 60,
    "BioGPT-Large and TabICL": 20 / 60,
    "calibration and the freeze": 15 / 60,
    "test predictions": 45 / 60,
    "the reading checks on v0.5's specialist (the plan's 30 minutes for the reading checks, counted in full; correction 8)": 30 / 60,
    "packaging and the copy": 45 / 60,
}
GEMMA_LONG_HOURS = 2.5
S8_RUN_TO_PILOT = 2.70 / 2.86  # session 8's runs trained at 2.70 items a second against its pilot's 2.86


def read_pod_rate(path):
    """created_utc, hourly_usd and an optional maintenance_utc, one 'key value' per line (Claude writes it in step 9)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    if "created_utc" not in out or "hourly_usd" not in out:
        raise SystemExit(f"{path} needs the lines 'created_utc <time>' and 'hourly_usd <rate>'")
    return out


def parse_utc(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def cost(pod_rate, pilot_s8, pilot_long, results=RESULTS, stop_usd=115.0, cap_usd=125.0, now_utc=None, counts=None):
    """The projection of THE PILOT AND THE COST RULE with correction 2's second runs, and its options in correction 2's order."""
    rate = read_pod_rate(pod_rate)
    hourly = float(rate["hourly_usd"])
    t_now = now_utc or dt.datetime.now(dt.timezone.utc)
    hours_so_far = (t_now - parse_utc(rate["created_utc"])).total_seconds() / 3600
    p8 = load_json(pilot_s8, {})
    base_rate = float(p8["chosen"]["train_items_per_second"]) * S8_RUN_TO_PILOT
    pl = load_json(pilot_long, None)
    long_ok = bool(pl and pl.get("chosen"))
    long_rate = float(pl["chosen"]["train_items_per_second"]) * S8_RUN_TO_PILOT if long_ok else None
    counts = counts or {"clean": 5279, "swap": 10557}
    items = {"b": 2 * counts["clean"], "swap": counts["swap"], "long": 2 * counts["clean"], "lr1": 2 * counts["clean"],
             "lr2": 2 * counts["clean"]}
    predict_h = {"base": 4 / 60, "long": 6 / 60}

    def run_h(recipe, n_items=None):
        if recipe == "long" or recipe == "combo_long":
            return (n_items or items["long"]) / long_rate / 3600 + predict_h["long"]
        return (n_items or items[recipe]) / base_rate / 3600 + predict_h["base"]

    plan = []
    for k in (1, 2, 3):
        plan.append(("s9_b_%d" % k, run_h("b")))
    for k in (1, 2):
        for c in CANDIDATES:
            if c == "long" and not long_ok:
                continue
            plan.append((f"s9_{c}_{k}", run_h(c)))
    longest = max(run_h(c) for c in CANDIDATES if c != "long" or long_ok)
    combo_h = run_h("combo_long", 2 * counts["clean"]) if long_ok else max(run_h("swap"), run_h("lr1"))
    if long_ok:  # a combination of swap and long reads the doubled set at 6,144 tokens for one epoch
        combo_h = max(combo_h, run_h("combo_long", counts["swap"]))
    plan += [("s9_combo_1", combo_h), ("s9_combo_2", combo_h), ("the winning recipe's third run", max(longest, combo_h))]
    fixed = dict(FIXED_HOURS)
    if long_ok:
        fixed["Gemma 4 31B-it on the longer input"] = GEMMA_LONG_HOURS
    remaining = sum(h for _, h in plan) + sum(fixed.values())
    total = (hours_so_far + remaining) * hourly
    drops = []
    current = dict(plan)
    for name in DROP_NAMES:
        removed = []
        for r in list(current):
            if (name in ("lr2", "long", "combo", "swap") and r.startswith(f"s9_{name}_")) or \
               (name.endswith("_2") and r == f"s9_{name}"):
                removed.append(r)
                current.pop(r)
        fixed_now = dict(fixed)
        if any(d in ("long",) for d in [x["drop"] for x in drops] + [name]):
            fixed_now.pop("Gemma 4 31B-it on the longer input", None)
        rem = sum(current.values()) + sum(fixed_now.values())
        drops.append({"drop": name, "runs_removed": removed, "projected_total_usd": (hours_so_far + rem) * hourly})
    end = t_now + dt.timedelta(hours=remaining)
    maint = rate.get("maintenance_utc")
    maint_conflict = bool(maint and maint not in ("none", "None") and parse_utc(maint) <= end)
    out = {"plan": "docs/prompt_session9.md, THE PILOT AND THE COST RULE (with corrections 1 and 2)", "written_utc": now(),
           "pod": {"created_utc": rate["created_utc"], "hourly_usd": hourly, "hours_so_far": hours_so_far,
                   "spend_so_far_usd": hours_so_far * hourly, "maintenance_utc": maint},
           "rates": {"base_items_per_second": base_rate, "long_items_per_second": long_rate,
                     "note": "the pilots' training rates times 2.70 / 2.86, the ratio of session 8's runs to its pilot",
                     "validation_prediction_hours": predict_h},
           "longest_path_hours": dict(plan), "fixed_hours": fixed, "remaining_hours": remaining,
           "projected_total_usd": total, "stop_line_usd": stop_usd, "cap_usd": cap_usd,
           "stops": bool(total > stop_usd), "projected_copy_checked_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "maintenance_before_the_end": maint_conflict,
           "options_in_order": drops,
           "options_note": ("if it stops, the options put to Johann, in correction 2's fixed order of what is dropped; nothing is "
                            "dropped without his decision, which Claude writes here as 'dropped' with the time"),
           "dropped": (load_json(Path(results) / "cost_rule.json", {}) or {}).get("dropped", [])}
    save_json(Path(results) / "cost_rule.json", out)
    print(json.dumps({k: out[k] for k in ("projected_total_usd", "stops", "remaining_hours", "projected_copy_checked_utc",
                                          "maintenance_before_the_end")}, indent=1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    nx = sub.add_parser("next")
    nx.add_argument("--results", type=Path, default=RESULTS)
    nx.add_argument("--records9", type=Path, default=RECORDS9)
    nx.add_argument("--not-built-dir", type=Path, default=None)
    nx.add_argument("--format", choices=("json", "env"), default="json")
    co = sub.add_parser("cost")
    co.add_argument("--pod-rate", type=Path, required=True)
    co.add_argument("--pilot-s8", type=Path, default=ROOT / "results" / "fiorillo_v0_4" / "pilot.json")
    co.add_argument("--pilot-long", type=Path, default=RESULTS / "pilot_long.json")
    co.add_argument("--results", type=Path, default=RESULTS)
    co.add_argument("--stop-usd", type=float, default=115.0)
    co.add_argument("--cap-usd", type=float, default=125.0)
    args = parser.parse_args(argv)
    if args.cmd == "cost":
        out = cost(args.pod_rate, args.pilot_s8, args.pilot_long, args.results, args.stop_usd, args.cap_usd)
        return 4 if (out["stops"] or out["maintenance_before_the_end"]) else 0
    action, _ = Decider(args.results, args.records9, args.not_built_dir).state()
    if args.format == "json":
        print(json.dumps(action))
    else:
        for k, v in action.items():
            key = "EDDM_" + k.upper()
            if isinstance(v, (list, tuple)):
                v = " ".join(str(x) for x in v)
            elif isinstance(v, dict):
                v = json.dumps(v)
            print(f"{key}={shlex.quote(str(v))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

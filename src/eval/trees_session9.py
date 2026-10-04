"""The readmission pool, version 2 (docs/prompt_session9.md, THE READMISSION POOL, VERSION 2; idea I-28 continued in
docs/ideas_register.md). Corrections 1 to 6 of the plan do not change this section. Run on the pod: recode, refit and
lightgbm on the CPU in the trees stage (beside train, at nice 10); tabicl on the GPU in the tabicl stage, in its own
virtual environment; pool on the GPU in the calibrate stage; the --split test runs of refit, lightgbm and tabicl in the
testpreds stage. test runs on the linked shell in step 15.

The frames are session 8's (src/eval/trees_session7.py's frames(): session 4's fields through src/eval/baselines.py's
uci_frame(), categories fitted on the training split; a category not seen in training is a missing value). Before the
freeze nothing here reads the test split: recode, refit, lightgbm, tabicl and pool read the training and validation
records and the validation prediction files only.

Subcommands:
  recode    the recoded diagnosis fields for diag_1, diag_2 and diag_3, added to session 8's frames (35 columns):
            - <field>_code3, the three-character code: a numeric code's integer part with leading zeros to three digits
              ("38" gives "038", "250.83" gives "250"), a V code as V and two digits, an E code as E and three digits, a
              missing value as "missing"; a pandas category whose categories are the training split's codes (sorted),
              so that a code not seen in training is a missing value, as session 8's frames make it;
            - <field>_chapter, its ICD-9-CM chapter by the code's number: 001-139, 140-239, 240-279, 280-289, 290-319,
              320-389, 390-459, 460-519, 520-579, 580-629, 630-679, 680-709, 710-739, 740-759, 760-779, 780-799 and
              800-999, with V, E and missing as their own (20 values, always these 20 categories);
            - diag_any_<chapter>, 20 indicator columns, 1 when the chapter appears in any of the three fields;
            - <field>_code3_te<answer>, the target encoding of each field's three-character code (9 columns): for each
              of the three answers, (rows with the code and the answer + 20 x the prior's share of the answer) / (rows
              with the code + 20). The training records are encoded out of fold: 5 folds (scikit-learn's
              StratifiedKFold, shuffled with the seed, the folds of sessions 7 and 8's searches), each fold's records
              encoded from the counts and the prior of the other four folds, so that no record's own label enters its
              own encoding. Validation, test and new records are encoded from the whole training split's counts and
              prior; a code not seen in training gets the prior.
            The tables are saved once in <models-dir>/recode_tables.json (session 8's frame categories and columns, the
            codes seen in training, the chapters, each field's whole-training shares and the added columns), so that
            frame_from_tables() encodes a new record with them alone (the release's path). trees_v2.json gets a summary.
  refit     xgboost_codes and catboost_codes: session 8's chosen settings for xgboost_tuned and catboost_tuned
            (results/fiorillo_v0_4/trees.json), refitted on all training records with the recoded fields added, with
            session 8's numbers of trees, by session 8's fit_member(); no new search. Saved in <models-dir> as
            xgboost_codes.json (XGBoost's JSON) and catboost_codes.cbm; their validation predictions. Before fitting,
            the tables rebuilt from the training records must equal recode_tables.json, and frame_from_tables() must
            give the validation frames exactly (the release's path); after fitting, the saved model must reproduce the
            validation predictions (largest difference at most 1e-8).
  lightgbm  lightgbm_tuned refitted from session 7's chosen setting (results/fiorillo_v0_3/trees.json) on session 8's
            frames by session 7's refit() (its parameters and its 4 threads), saved as lightgbm_tuned.txt (LightGBM's
            text model, which keeps the pandas categories); its validation predictions; and the check against session
            7's validation file (results/fiorillo_v0_3/predictions/lightgbm_tuned_readmit_window_validation.csv, which
            session 8's pool read; its SHA-256 must be the one results/fiorillo_v0_4/readmission_pool.json recorded):
            the largest difference in a probability at most 0.001. If the check passes, the refit's predictions stand
            for lightgbm_tuned in the new pool, on validation and on test; if it fails, lightgbm_tuned is left out of
            the new pool's members (recorded in trees_v2.json and readmission_pool_v2.json).
  tabicl    TabICLClassifier (tabicl at the version in results/fiorillo_v0_5/fixed_versions.json) with its defaults
            (checked against fixed_versions.json) and random_state 20260927. Its checkpoint is downloaded with
            huggingface_hub.hf_hub_download at the fixed revision (or read from --checkpoint), its SHA-256 checked, and
            passed as model_path with allow_auto_download false. Its input is session 8's frames as pandas DataFrames:
            the categorical columns as pandas categories with a missing value as NaN (TabICL ordinal-encodes them with
            a separate category for missing values), the numeric columns as floats (TabICL mean-imputes a NaN). Its
            context is the whole training split (its documentation sets no largest context below the 55,978 training
            records). predict_proba on validation. Its fitted state is saved by TabICLClassifier.save (model weights
            not saved, training data saved, KV cache saved if any) as tabicl.pkl, with tabicl_state.json beside it (the
            checkpoint's repository, revision, file and SHA-256, and the versions); load_tabicl() loads it with the
            weights read from a given checkpoint file. tabicl.json records the settings, the timings and the validation
            metrics. This path needs only numpy, pandas, scikit-learn, torch, huggingface_hub and tabicl: xgboost,
            catboost and lightgbm are imported only inside the functions that use them.
  pool      the new pool over session 8's five members (lightgbm_tuned, xgboost_tuned, catboost_tuned, arm_laya,
            s7_b3) and the new ones (xgboost_codes, catboost_codes, tabicl), lightgbm_tuned's refit standing for
            lightgbm_tuned if its check passed. A member that is NOT BUILT (its validation file does not exist; the
            reason is read from --not-built-dir when the stage wrote a marker) or left out is dropped, and the grid
            follows (19,448 points for eight members, 8,008 for seven, 3,003 for six). Session 8's method:
            softmax((sum over members of w_m log p_m) / T), log p clipped at 1e-12, the weights on the grid of steps of
            0.1 that sum to 1 (in itertools.product order), T in [0.1, 10] fitted by maximum likelihood at each grid
            point by session 8's damped Newton method on 1 / T (the log-likelihood is concave in 1 / T), and the point
            with the lowest validation log loss kept (the first in grid order among exact ties). The grid is evaluated
            in batches of grid points with torch in float64 (--device, --chunk). The out-of-fold log losses of the new
            pool and of session 8's pool (its five members and files as results/fiorillo_v0_4/readmission_pool.json
            records them) use the same folds: the rows in the order of the first file session 8's pool read, and 20
            repeats of 5 folds from calibration_family_cv.py's folds_for() with numpy's default_rng and the seed
            (session 8's folds), the weights and T refitted in each fold by the same torch fit; each fold's fit is
            saved in trees_v2_progress.json as it finishes. v0.5 uses the new pool if its out-of-fold log loss is lower
            than session 8's pool's by at least 0.002; otherwise it keeps session 8's pool, with the weights and T that
            session 8 kept. readmission_pool_v2.json records both pools (members, weights, T, files, losses), both
            out-of-fold losses, the choice and its time; its top-level members, weights, T and member_files are the
            chosen pool's. The chosen pool's validation predictions: readmission_pool_v2_readmit_window_validation.csv.
  test      step 15, on the linked shell, after the freeze: refuses to run unless release_candidate.json and
            testpreds_log.json exist; checks the SHA-256 that release_candidate.json recorded for
            readmission_pool_v2.json and for each of the chosen pool's members' validation files (it refuses on a
            mismatch or a missing record); applies the chosen pool's weights and T to its members' test files (the
            same folder and name, _test in place of _validation): readmission_pool_v2_readmit_window_test.csv,
            trees_v2_test.json (every member test file of both pools that exists, with its rows and SHA-256) and
            readmission_pool_v2_test.json.

--split test (refit, lightgbm and tabicl; the pod's testpreds stage) refuses to run unless release_candidate.json and
testpreds_log.json exist, checks the member's validation file against the SHA-256 that release_candidate.json recorded,
loads the saved model (TabICL's state with its checkpoint, fetched and checked again), checks that it reproduces the
member's validation predictions (largest difference at most 1e-8 for the trees, 0.001 for TabICL on the GPU), and writes
the member's test predictions; its record goes into testpreds_log.json under "trees_v2" or "tabicl". The trees' frames
for this are built from recode_tables.json by frame_from_tables(), as the release builds them. lightgbm_tuned writes no
test predictions when its check failed. frozen_inputs() gives the SHA-256 of every file these checks read, for the
freeze in calibrate_session9.py.

Prediction files are session 4's format (id, label, p0, p1, p2; 9 significant digits), named
<results>/predictions/<system>_readmit_window_<split>.csv for xgboost_codes, catboost_codes, lightgbm_tuned (the refit),
tabicl and readmission_pool_v2 (v0.5's pool, the chosen one). Progress is kept in <results>/trees_v2_progress.json
(recode, refit, lightgbm and the pool's fold fits), so that a stopped stage resumes; tabicl keeps its record in
tabicl.json only, so that the tabicl and trees stages never write the same file.

Usage (pod): python src/eval/trees_session9.py recode --models-dir /root/eddm_s9/models/trees
             python src/eval/trees_session9.py refit --threads 12 --models-dir /root/eddm_s9/models/trees
             python src/eval/trees_session9.py lightgbm --models-dir /root/eddm_s9/models/trees
             /root/eddm_s9/tabicl_env/bin/python src/eval/trees_session9.py tabicl --device cuda --models-dir /root/eddm_s9/models/trees
             python src/eval/trees_session9.py pool --device cuda --not-built-dir /root/eddm_s9/not_built
             (testpreds) refit --split test --threads 12; lightgbm --split test; tabicl --split test --device cuda
Usage (linked shell, step 15): python3 src/eval/trees_session9.py test
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from eval import metrics as M  # noqa: E402
from eval import trees_session7 as T7  # noqa: E402
from eval import trees_session8 as T8  # noqa: E402
from eval.calibration_family_cv import folds_for  # noqa: E402

SEED = 20260927
RESULTS = ROOT / "results" / "fiorillo_v0_5"
RECORDS = ROOT / "data" / "processed" / "public"
MODELS = ROOT / "models" / "fiorillo_v0_5" / "trees"
QID = "readmit_window"
PLAN = "docs/prompt_session9.md, THE READMISSION POOL, VERSION 2"
PLAN_TEST = "docs/prompt_session9.md, THE READMISSION POOL, VERSION 2 and STEP 15"
ANSWERS = 3
ANSWER_NAMES = ("NO", ">30", "<30")
FIELDS = ("diag_1", "diag_2", "diag_3")
MISSING = "missing"
CHAPTER_RANGES = (("001-139", 1, 139), ("140-239", 140, 239), ("240-279", 240, 279), ("280-289", 280, 289),
                  ("290-319", 290, 319), ("320-389", 320, 389), ("390-459", 390, 459), ("460-519", 460, 519),
                  ("520-579", 520, 579), ("580-629", 580, 629), ("630-679", 630, 679), ("680-709", 680, 709),
                  ("710-739", 710, 739), ("740-759", 740, 759), ("760-779", 760, 779), ("780-799", 780, 799),
                  ("800-999", 800, 999))
CHAPTERS = tuple(name for name, _, _ in CHAPTER_RANGES) + ("V", "E", MISSING)
TE_WEIGHT = 20
TE_FOLDS = 5
TABLES_FILE = "recode_tables.json"
CODES_BASE = {"xgboost_codes": "xgboost_tuned", "catboost_codes": "catboost_tuned"}
MODEL_FILES = {"xgboost_codes": "xgboost_codes.json", "catboost_codes": "catboost_codes.cbm",
               "lightgbm_tuned": "lightgbm_tuned.txt", "tabicl": "tabicl.pkl"}
TABICL_STATE_JSON = "tabicl_state.json"
REFIT_LIMIT = 0.001  # lightgbm_tuned's refit against session 7's validation file (the plan's limit)
REPRODUCE_LIMIT = 1e-8  # a saved tree model against its own validation predictions (session 8's limit)
TABICL_REPRODUCE_LIMIT = 0.001  # TabICL's saved state against its validation predictions, on the GPU (chosen here)
S8_MEMBERS = ("lightgbm_tuned", "xgboost_tuned", "catboost_tuned", "arm_laya", "s7_b3")
NEW_MEMBERS = ("xgboost_codes", "catboost_codes", "tabicl")
POOL_NAME = "readmission_pool_v2"
POOL_FOLDS, POOL_REPEATS = 5, 20
GRID_STEPS = 10  # tenths
MARGIN = 0.002
CHUNK = 1024  # grid points evaluated at once
NEWTON_ITERATIONS, NEWTON_TOLERANCE = 40, 1e-8  # session 8's _fit_chunk
now, sha256_file, rel = T8.now, T8.sha256_file, T8.rel
load_json, save_json, write_probs = T7.load_json, T7.save_json, T7.write_probs


# ---------------------------------------------------------------- paths and small helpers

def repo_of(results):
    """The repository that holds a results folder (<repo>/results/fiorillo_v0_5)."""
    return Path(results).resolve().parents[1]


def resolve(path, results):
    """A file named in a results file: a repository path (results/...) is read under the results folder's repository."""
    path = Path(path)
    return path if path.is_absolute() else repo_of(results) / path


def sibling(results, version):
    """Another session's results folder beside this one (fiorillo_v0_3, fiorillo_v0_4, public_v0)."""
    return Path(results).parent / version


def new_file(system, split, results=RESULTS):
    return Path(results) / "predictions" / f"{system}_{QID}_{split}.csv"


def twin_file(path):
    """A validation prediction file's test twin: the same folder and name, _test in place of _validation."""
    path = Path(path)
    tail = f"_{QID}_validation.csv"
    if not path.name.endswith(tail):
        raise SystemExit(f"{path.name} is not a {QID} validation prediction file")
    return path.with_name(path.name[: -len(tail)] + f"_{QID}_test.csv")


def versions(names):
    out = {}
    for name in names:
        try:
            out[name] = getattr(__import__(name), "__version__", None)
        except ImportError:
            out[name] = None
    return out


def progress_path(results):
    return Path(results) / "trees_v2_progress.json"


def load_progress(results):
    return load_json(progress_path(results), {"plan": PLAN})


def save_progress(results, progress):
    progress["plan"] = PLAN
    progress["written_utc"] = now()
    save_json(progress_path(results), progress)


def load_trees(results):
    return load_json(Path(results) / "trees_v2.json", {"plan": PLAN, "members": {}})


def save_trees(results, trees):
    trees["plan"] = PLAN
    trees["seed"] = SEED
    trees["written_utc"] = now()
    save_json(Path(results) / "trees_v2.json", trees)


def read_records(records_dir, split):
    return T7.M_read(Path(records_dir) / f"{QID}_{split}.jsonl")


def max_difference(path, ids, probs):
    """The largest difference in a probability between a prediction file and probabilities for the same records."""
    f_ids, _, f_probs = M.read_predictions(path)
    if len(f_ids) != len(ids) or set(f_ids) != set(ids):
        raise SystemExit(f"{rel(path)} holds different records from the predictions it is checked against")
    pos = {i: k for k, i in enumerate(f_ids)}
    order = np.array([pos[i] for i in ids])
    probs = np.asarray(probs, dtype=float)
    if f_probs.shape[1] != probs.shape[1]:
        raise SystemExit(f"{rel(path)} has {f_probs.shape[1]} answers, not {probs.shape[1]}")
    return float(np.max(np.abs(f_probs[order] - probs)))


def metrics_of(y, probs):
    return {"rows": int(len(y)), "log_loss": M.log_loss(y, probs), "auroc_30d": M.auroc(y == 2, probs[:, 2]), "ece": M.ece(y, probs)}


def read_marker(not_built_dir, stage):
    if not not_built_dir:
        return None
    path = Path(not_built_dir) / f"{stage}.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


# ---------------------------------------------------------------- recode: codes, chapters, indicators, target encoding

def diag_code(value):
    """The ICD-9-CM code of a record's diag field ("250.83 (Endocrine, ...)" gives "250.83"); None when it is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text == "?":
        return None
    return text.split(" (", 1)[0].strip()


def three_char(code):
    """The three-character code: a numeric code's integer part with leading zeros to three digits, V and two digits,
    E and three digits, or "missing"."""
    if code is None:
        return MISSING
    c = str(code).strip().upper()
    if c.startswith("V"):
        return f"V{int(float(c[1:])):02d}"
    if c.startswith("E"):
        return f"E{int(float(c[1:])):03d}"
    return f"{int(float(c)):03d}"


def chapter(code3):
    """A three-character code's ICD-9-CM chapter: its number's range, or V, E or missing."""
    if code3 == MISSING:
        return MISSING
    if code3[0] in "VE":
        return code3[0]
    number = int(code3)
    for name, low, high in CHAPTER_RANGES:
        if low <= number <= high:
            return name
    raise ValueError(f"ICD-9-CM code outside every chapter: {code3}")


def code3_column(field):
    return f"{field}_code3"


def chapter_column(field):
    return f"{field}_chapter"


def any_column(name):
    return "diag_any_" + name.replace("-", "_")


def te_column(field, answer):
    return f"{field}_code3_te{answer}"


def added_columns():
    cols = [c for f in FIELDS for c in (code3_column(f), chapter_column(f))]
    cols += [any_column(ch) for ch in CHAPTERS]
    cols += [te_column(f, a) for f in FIELDS for a in range(ANSWERS)]
    return cols


def row_codes(rows):
    """{field: each record's three-character code}."""
    return {f: [three_char(diag_code(r["state"].get(f))) for r in rows] for f in FIELDS}


def te_table(codes, y, weight=TE_WEIGHT, k=ANSWERS):
    """A field's target-encoding table from these records: the prior (each answer's share of the records) and, for
    each code, its share of each answer smoothed toward the prior with the weight:
    (records with the code and the answer + weight x prior) / (records with the code + weight)."""
    y = np.asarray(y, dtype=int)
    prior = np.bincount(y, minlength=k).astype(float) / len(y)
    counts = {}
    for c, label in zip(codes, y):
        if c not in counts:
            counts[c] = np.zeros(k)
        counts[c][label] += 1.0
    shares = {c: [float(v) for v in (n + weight * prior) / (n.sum() + weight)] for c, n in sorted(counts.items())}
    return {"prior": [float(p) for p in prior], "records": int(len(y)), "shares": shares,
            "records_with_code": {c: int(n.sum()) for c, n in sorted(counts.items())}}


def te_lookup(codes, table):
    """Each code's shares from a table; a code the table does not hold gets the prior."""
    prior = table["prior"]
    shares = table["shares"]
    return np.array([shares.get(c, prior) for c in codes], dtype=float).reshape(len(codes), len(prior))


def te_folds(y):
    """Fold labels for the out-of-fold encoding: StratifiedKFold(5), shuffled with the seed."""
    from sklearn.model_selection import StratifiedKFold

    fold = np.empty(len(y), dtype=int)
    for f, (_, held) in enumerate(StratifiedKFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        fold[held] = f
    return fold


def oof_encoding(codes, y, fold, weight=TE_WEIGHT):
    """Each record encoded from the table of the records in the other folds (their counts and their prior)."""
    codes, y = list(codes), np.asarray(y, dtype=int)
    out = np.empty((len(codes), ANSWERS))
    for f in np.unique(fold):
        fit = np.flatnonzero(fold != f)
        held = np.flatnonzero(fold == f)
        table = te_table([codes[i] for i in fit], y[fit], weight)
        out[held] = te_lookup([codes[i] for i in held], table)
    return out


def recoded_columns(rows, tables, encoding=None, index=None):
    """The 35 recoded columns for these records; encoding gives the training records' out-of-fold target encoding
    ({field: rows x 3}), and otherwise the tables' whole-training shares encode them."""
    import pandas as pd

    codes = row_codes(rows)
    chapters = {f: [chapter(c) for c in codes[f]] for f in FIELDS}
    cols = {}
    for f in FIELDS:
        cols[code3_column(f)] = pd.Categorical(codes[f], categories=tables["code3_categories"][f])
        cols[chapter_column(f)] = pd.Categorical(chapters[f], categories=list(CHAPTERS))
    present = [{chapters[f][i] for f in FIELDS} for i in range(len(rows))]
    for ch in CHAPTERS:
        cols[any_column(ch)] = np.array([int(ch in found) for found in present], dtype=np.int64)
    for f in FIELDS:
        enc = encoding[f] if encoding is not None else te_lookup(codes[f], tables["target_encoding"]["fields"][f])
        for a in range(ANSWERS):
            cols[te_column(f, a)] = np.asarray(enc[:, a], dtype=float)
    return pd.DataFrame(cols, index=index)


def make_tables(rows, frame, records_dir):
    """The recoding's tables from the training records and session 8's training frame."""
    import pandas as pd

    y = np.array([r["label"] for r in rows])
    codes = row_codes(rows)
    train_path = Path(records_dir) / f"{QID}_train.jsonl"
    cat_cols = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    return {
        "plan": PLAN,
        "what": ("the recoding's tables: session 8's frame columns and categories (fitted on the training split), the "
                 "three-character codes seen in training, the chapters, and each field's whole-training target encoding; "
                 "trees_session9.py's frame_from_tables() encodes a new record with them alone"),
        "training_records": {"file": train_path.name, "rows": len(rows), "sha256": sha256_file(train_path) if train_path.exists() else None},
        "frame_columns": [str(c) for c in frame.columns],
        "frame_categories": {str(c): [str(v) for v in frame[c].cat.categories] for c in cat_cols},
        "fields": list(FIELDS),
        "chapters": [{"chapter": name, "codes": f"{low:03d} to {high:03d}"} for name, low, high in CHAPTER_RANGES]
                    + [{"chapter": "V", "codes": "V codes"}, {"chapter": "E", "codes": "E codes"}, {"chapter": MISSING, "codes": "no code"}],
        "code3_categories": {f: sorted(set(codes[f])) for f in FIELDS},
        "target_encoding": {"weight": TE_WEIGHT, "answers": list(ANSWER_NAMES),
                            "training_records": ("encoded out of fold: StratifiedKFold(5) shuffled with seed 20260927, each fold "
                                                 "from the other four folds' counts and prior"),
                            "other_records": "the whole training split's shares below; a code not listed gets the prior",
                            "fields": {f: te_table(codes[f], y) for f in FIELDS}},
        "columns_added": added_columns(),
    }


def same_tables(a, b):
    def strip(t):
        return json.loads(json.dumps({k: v for k, v in (t or {}).items() if k != "written_utc"}, sort_keys=True))
    return strip(a) == strip(b)


def same_frame(a, b):
    import pandas as pd

    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError as error:
        return False, (str(error).strip().splitlines() or ["the frames differ"])[0]
    return True, None


def frame_from_tables(rows, tables, codes=True):
    """Records' frame from the saved tables alone (the release's path): session 8's frame, with the recoded fields
    added when codes is true."""
    import pandas as pd
    from baselines import uci_frame

    frame, _ = uci_frame(rows, tables["frame_categories"])
    if [str(c) for c in frame.columns] != tables["frame_columns"]:
        raise SystemExit("session 8's frame columns differ from those in the recoding's tables")
    if not codes:
        return frame
    return pd.concat([frame, recoded_columns(rows, tables, None, frame.index)], axis=1)


def build_frames(records_dir, splits=("train", "validation")):
    """Session 8's frames with the recoded fields added (training records encoded out of fold), the tables, and
    session 8's frames as they are."""
    import pandas as pd

    plain = T7.frames(records_dir, tuple(splits))
    x, y, rows = plain["train"]
    tables = make_tables(rows, x, records_dir)
    codes = row_codes(rows)
    fold = te_folds(y)
    encoding = {f: oof_encoding(codes[f], y, fold) for f in FIELDS}
    out = {"train": (pd.concat([x, recoded_columns(rows, tables, encoding, x.index)], axis=1), y, rows)}
    for split, (frame, ys, rs) in plain.items():
        if split != "train":
            out[split] = (pd.concat([frame, recoded_columns(rs, tables, None, frame.index)], axis=1), ys, rs)
    return out, tables, plain


def recode_summary(data, tables):
    x, y, rows = data["train"]
    codes = row_codes(rows)
    seen = {f: set(tables["code3_categories"][f]) for f in FIELDS}
    out = {"fields": {}}
    val = data.get("validation")
    val_codes = row_codes(val[2]) if val else None
    for f in FIELDS:
        chapters = [chapter(c) for c in codes[f]]
        out["fields"][f] = {
            "codes_in_training": len(tables["code3_categories"][f]),
            "training_records_missing": int(sum(c == MISSING for c in codes[f])),
            "training_records_by_chapter": {ch: int(sum(c == ch for c in chapters)) for ch in CHAPTERS},
            "validation_records_with_a_code_not_seen_in_training": (int(sum(c not in seen[f] for c in val_codes[f])) if val else None)}
    out["indicators_share_of_training_records"] = {any_column(ch): float(x[any_column(ch)].mean()) for ch in CHAPTERS}
    out["target_encoding_prior"] = tables["target_encoding"]["fields"][FIELDS[0]]["prior"]
    return out


def recode(results=RESULTS, records_dir=RECORDS, models_dir=MODELS):
    t0 = time.time()
    data, tables, _ = build_frames(records_dir, ("train", "validation"))
    path = Path(models_dir) / TABLES_FILE
    if path.exists():
        if not same_tables(load_json(path, None), tables):
            raise SystemExit(f"{path} exists and differs from the tables rebuilt from the training records; it is written once")
        status = "existed; the tables rebuilt from the training records equal it"
    else:
        save_json(path, dict(tables, written_utc=now()))
        status = "written"
    digest = sha256_file(path)
    x = data["train"][0]
    summary = {"status": status, "tables": path.name, "tables_sha256": digest, "training_records": tables["training_records"],
               "columns_added": tables["columns_added"], "columns": {"session8": len(tables["frame_columns"]),
                                                                    "added": len(tables["columns_added"]), "total": int(x.shape[1])},
               "target_encoding": {"weight": TE_WEIGHT, "folds": TE_FOLDS, "fold_method": "StratifiedKFold, shuffled with the seed",
                                   "answers": list(ANSWER_NAMES)},
               **recode_summary(data, tables), "seconds": round(time.time() - t0, 1), "time_utc": now()}
    trees = load_trees(results)
    trees["recode"] = summary
    save_trees(results, trees)
    progress = load_progress(results)
    progress["recode"] = {"done": True, "tables": path.name, "tables_sha256": digest, "time_utc": now()}
    save_progress(results, progress)
    print(json.dumps({"recode": status, "tables_sha256": digest, "columns": summary["columns"], "seconds": summary["seconds"]}, indent=1), flush=True)
    return summary


# ---------------------------------------------------------------- the tree members

def load_tree(member, models_dir=MODELS):
    """A saved tree member: xgboost_codes (XGBClassifier), catboost_codes (CatBoostClassifier) or lightgbm_tuned
    (LightGBM's Booster from its text model)."""
    path = Path(models_dir) / MODEL_FILES[member]
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run {'lightgbm' if member == 'lightgbm_tuned' else 'refit'} first")
    if member == "xgboost_codes":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model
    if member == "catboost_codes":
        import catboost as cb
        model = cb.CatBoostClassifier()
        model.load_model(str(path))
        return model
    if member == "lightgbm_tuned":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    raise ValueError(f"{member} is not a tree member")


def predict_tree(member, model, frame, threads=None):
    """A tree member's probabilities; frame is session 8's frame, with the recoded fields for the _codes members."""
    if member == "lightgbm_tuned":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(frame), dtype=float)
        return np.asarray(model.predict(frame), dtype=float)
    return T8.predict_member(CODES_BASE[member], model, frame, threads)


def check_tables(models_dir, tables):
    path = Path(models_dir) / TABLES_FILE
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run recode first")
    saved = load_json(path, None)
    if not same_tables(saved, tables):
        raise SystemExit(f"{path} differs from the tables rebuilt from the training records; stop")
    return saved, sha256_file(path)


def done_before(record, *pairs):
    """Whether a member's recorded files all exist with the recorded SHA-256 (pairs of path and record key)."""
    if not record:
        return False
    return all(Path(p).exists() and sha256_file(p) == record.get(key) for p, key in pairs)


def refit(results=RESULTS, records_dir=RECORDS, models_dir=MODELS, members=tuple(CODES_BASE), threads=4):
    t0 = time.time()
    s8_path = sibling(results, "fiorillo_v0_4") / "trees.json"
    s8 = load_json(s8_path, None)
    if not s8 or any(CODES_BASE[m] not in (s8.get("members") or {}) for m in members):
        raise SystemExit(f"{rel(s8_path)} must hold session 8's chosen settings for {[CODES_BASE[m] for m in members]}")
    data, tables, plain = build_frames(records_dir, ("train", "validation"))
    saved, tables_sha = check_tables(models_dir, tables)
    rows_v = data["validation"][2]
    for codes, built in ((True, data["validation"][0]), (False, plain["validation"][0])):
        same, why = same_frame(frame_from_tables(rows_v, saved, codes=codes), built)
        if not same:
            raise SystemExit(f"frame_from_tables() does not give the validation frame (codes {codes}): {why}; stop")
    x, y, _ = data["train"]
    xv, yv, rows = data["validation"]
    trees, progress = load_trees(results), load_progress(results)
    trees.setdefault("members", {})
    for member in members:
        base = CODES_BASE[member]
        model_path, pred_path = Path(models_dir) / MODEL_FILES[member], new_file(member, "validation", results)
        if done_before(progress.get("refit", {}).get(member), (model_path, "model_sha256"), (pred_path, "predictions_sha256")):
            print(f"{member}: refitted before (its model and validation predictions are unchanged); skipped", flush=True)
            continue
        setting = s8["members"][base]["chosen"]
        t_fit = time.time()
        model, n_trees, _ = T8.fit_member(base, setting, x, y, threads=threads, n_trees=setting["n_estimators"])
        fit_seconds = round(time.time() - t_fit, 2)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        t_pred = time.time()
        probs = T8.predict_member(base, model, xv, threads)
        predict_seconds = round(time.time() - t_pred, 2)
        again = predict_tree(member, load_tree(member, models_dir), xv, threads)
        reproduced = float(np.max(np.abs(again - probs)))
        if reproduced > REPRODUCE_LIMIT:
            raise SystemExit(f"{member}: the saved model does not reproduce its validation predictions (largest difference {reproduced})")
        write_probs(pred_path, rows, probs)
        entry = {"base": base, "setting": setting, "setting_source": rel(s8_path), "trees": int(n_trees),
                 "columns": {"total": int(x.shape[1]), "added": len(tables["columns_added"])},
                 "refit": {"rows": int(len(y)), "seconds": fit_seconds, "threads": threads},
                 "model_file": model_path.name, "model_sha256": sha256_file(model_path),
                 "saved_model_reproduces_validation_max_abs_difference": reproduced,
                 "tables": TABLES_FILE, "tables_sha256": tables_sha,
                 "validation": {**metrics_of(yv, probs), "predict_seconds": predict_seconds},
                 "predictions": rel(pred_path), "predictions_sha256": sha256_file(pred_path), "time_utc": now()}
        trees["members"][member] = entry
        trees["versions"] = versions(("xgboost", "catboost", "lightgbm", "sklearn", "pandas", "numpy"))
        save_trees(results, trees)
        progress.setdefault("refit", {})[member] = {"done": True, "model_sha256": entry["model_sha256"],
                                                    "predictions_sha256": entry["predictions_sha256"], "time_utc": now()}
        save_progress(results, progress)
        print(f"{member}: {n_trees} trees on {len(y)} records in {fit_seconds} s; validation log loss "
              f"{entry['validation']['log_loss']:.5f}", flush=True)
    print(json.dumps({"refit": list(members), "seconds": round(time.time() - t0, 1)}), flush=True)
    return trees


def refit_check(ids, labels, probs, path, limit=REFIT_LIMIT):
    """lightgbm_tuned's refit against session 7's validation file: the largest difference in a probability, and
    whether it is at most the limit. The two must hold the same records with the same labels."""
    path = Path(path)
    s_ids, s_labels, _ = M.read_predictions(path)
    if len(s_ids) != len(ids) or set(s_ids) != set(ids):
        raise SystemExit(f"{path.name} holds different records from the refit's predictions")
    pos = {i: k for k, i in enumerate(s_ids)}
    if not np.array_equal(s_labels[np.array([pos[i] for i in ids])], np.asarray(labels)):
        raise SystemExit(f"{path.name}: the labels differ from the refit's records")
    diff = max_difference(path, list(ids), probs)
    return {"against": rel(path), "against_sha256": sha256_file(path), "rows": int(len(ids)), "max_abs_difference": diff,
            "limit": limit, "passed": bool(diff <= limit),
            "consequence": ("the refit's predictions stand for lightgbm_tuned in the new pool, on validation and on test" if diff <= limit
                            else "lightgbm_tuned is left out of the new pool's members")}


def lightgbm_refit(results=RESULTS, records_dir=RECORDS, models_dir=MODELS):
    import lightgbm as lgb

    t0 = time.time()
    s7_path = sibling(results, "fiorillo_v0_3") / "trees.json"
    s7 = load_json(s7_path, None)
    if not s7 or "chosen" not in s7:
        raise SystemExit(f"{rel(s7_path)} must hold session 7's chosen setting for lightgbm_tuned")
    s7_file = sibling(results, "fiorillo_v0_3") / "predictions" / f"lightgbm_tuned_{QID}_validation.csv"
    if not s7_file.exists():
        raise SystemExit(f"{rel(s7_file)} does not exist; the check needs session 7's validation file")
    s8_pool = load_json(sibling(results, "fiorillo_v0_4") / "readmission_pool.json", {})
    recorded = ((s8_pool.get("member_files") or {}).get("lightgbm_tuned") or {}).get("sha256")
    if recorded and recorded != sha256_file(s7_file):
        raise SystemExit(f"{rel(s7_file)} is not the file session 8's pool read (its SHA-256 differs); stop")
    model_path, pred_path = Path(models_dir) / MODEL_FILES["lightgbm_tuned"], new_file("lightgbm_tuned", "validation", results)
    trees, progress = load_trees(results), load_progress(results)
    trees.setdefault("members", {})
    if done_before(progress.get("lightgbm"), (model_path, "model_sha256"), (pred_path, "predictions_sha256")) and "lightgbm_tuned" in trees["members"]:
        print("lightgbm_tuned: refitted before (its model and validation predictions are unchanged); skipped", flush=True)
        return trees["members"]["lightgbm_tuned"]
    data = T7.frames(records_dir, ("train", "validation"))
    x, y, _ = data["train"]
    xv, yv, rows = data["validation"]
    t_fit = time.time()
    model = T7.refit(s7["chosen"], data)
    fit_seconds = round(time.time() - t_fit, 2)
    probs = np.asarray(model.predict_proba(xv), dtype=float)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    reproduced = float(np.max(np.abs(predict_tree("lightgbm_tuned", load_tree("lightgbm_tuned", models_dir), xv) - probs)))
    if reproduced > REPRODUCE_LIMIT:
        raise SystemExit(f"lightgbm_tuned: the saved model does not reproduce its validation predictions (largest difference {reproduced})")
    write_probs(pred_path, rows, probs)
    check = refit_check([r["id"] for r in rows], yv, probs, s7_file)
    entry = {"setting": s7["chosen"], "setting_source": rel(s7_path), "trees": int(s7["chosen"]["n_estimators"]),
             "parameters": "trees_session7.py's lgb_params (4 threads, deterministic) with the chosen number of trees",
             "refit": {"rows": int(len(y)), "seconds": fit_seconds, "lightgbm": lgb.__version__},
             "model_file": model_path.name, "model_sha256": sha256_file(model_path),
             "saved_model_reproduces_validation_max_abs_difference": reproduced,
             "validation": metrics_of(yv, probs), "predictions": rel(pred_path), "predictions_sha256": sha256_file(pred_path),
             "check": {**check, "session8_pool_record_sha256": recorded}, "time_utc": now()}
    trees["members"]["lightgbm_tuned"] = entry
    trees["versions"] = versions(("xgboost", "catboost", "lightgbm", "sklearn", "pandas", "numpy"))
    save_trees(results, trees)
    progress["lightgbm"] = {"done": True, "model_sha256": entry["model_sha256"], "predictions_sha256": entry["predictions_sha256"],
                            "passed": check["passed"], "time_utc": now()}
    save_progress(results, progress)
    print(json.dumps({"lightgbm_tuned": {"check": check, "seconds": round(time.time() - t0, 1)}}, indent=1), flush=True)
    return entry


def add_to_testpreds_log(results, key, entries):
    path = Path(results) / "testpreds_log.json"
    log = json.loads(path.read_text(encoding="utf-8"))
    log.setdefault(key, {}).update(entries)
    save_json(path, log)


def trees_test(results=RESULTS, records_dir=RECORDS, models_dir=MODELS, members=tuple(CODES_BASE), threads=4):
    """--split test of refit and lightgbm: each member's test predictions from its saved model, after the freeze."""
    rc = T8.guard(results)
    trees = load_trees(results)
    tables_path = Path(models_dir) / TABLES_FILE
    if not tables_path.exists():
        raise SystemExit(f"{tables_path} does not exist; the test frames are built from it")
    recorded = (trees.get("recode") or {}).get("tables_sha256")
    if recorded and recorded != sha256_file(tables_path):
        raise SystemExit(f"{tables_path} changed after recode wrote it; stop")
    tables = load_json(tables_path, None)
    rows_v, rows_t = read_records(records_dir, "validation"), read_records(records_dir, "test")
    log, previous = {}, json.loads((Path(results) / "testpreds_log.json").read_text(encoding="utf-8")).get("trees_v2", {})
    for member in members:
        entry = (trees.get("members") or {}).get(member)
        if member == "lightgbm_tuned" and entry and not entry["check"]["passed"]:
            log[member] = {"written": False, "reason": "the refit's check failed; lightgbm_tuned is not a member of the new pool",
                           "time_utc": now()}
            add_to_testpreds_log(results, "trees_v2", {member: log[member]})
            print(f"{member}: no test predictions (its check failed)", flush=True)
            continue
        if not entry:
            raise SystemExit(f"trees_v2.json has no record of {member}; it was not built")
        val_path, test_path = new_file(member, "validation", results), new_file(member, "test", results)
        T8.check_frozen(rc, val_path)
        if member in previous and test_path.exists() and sha256_file(test_path) == previous[member].get("test_predictions_sha256"):
            print(f"{member}: test predictions written before; skipped", flush=True)
            continue
        model = load_tree(member, models_dir)
        codes = member in CODES_BASE
        xv = frame_from_tables(rows_v, tables, codes=codes)
        diff = max_difference(val_path, [r["id"] for r in rows_v], predict_tree(member, model, xv, threads))
        if diff > REPRODUCE_LIMIT:
            raise SystemExit(f"{member}: the saved model does not reproduce its frozen validation predictions (largest difference {diff})")
        t_pred = time.time()
        probs = predict_tree(member, model, frame_from_tables(rows_t, tables, codes=codes), threads)
        write_probs(test_path, rows_t, probs)
        log[member] = {"model": MODEL_FILES[member], "model_sha256": sha256_file(Path(models_dir) / MODEL_FILES[member]),
                       "validation_reproduced_max_abs_difference": diff, "test_rows": len(rows_t), "seconds": round(time.time() - t_pred, 2),
                       "test_predictions": rel(test_path), "test_predictions_sha256": sha256_file(test_path), "time_utc": now()}
        add_to_testpreds_log(results, "trees_v2", {member: log[member]})
        print(f"{member}: validation reproduced (largest difference {diff:.2e}); {len(rows_t)} test rows written", flush=True)
    return log


# ---------------------------------------------------------------- TabICL

def tabicl_spec(fixed_path):
    fixed = load_json(fixed_path, None)
    if not fixed or "tabicl" not in fixed:
        raise SystemExit(f"{fixed_path} has no tabicl entry; it is fixed in step 0")
    spec = fixed["tabicl"]
    missing = [k for k in ("version", "defaults", "checkpoint") if k not in spec]
    missing += [f"checkpoint.{k}" for k in ("repo", "revision", "file", "sha256") if k not in spec.get("checkpoint", {})]
    if missing:
        raise SystemExit(f"{fixed_path}'s tabicl entry lacks {missing}")
    return spec


def fetch_checkpoint(spec, checkpoint=None, cache_dir=None):
    """TabICL's checkpoint at the fixed revision (downloaded with hf_hub_download, or the file given), its SHA-256 checked."""
    ck = spec["checkpoint"]
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=ck["repo"], filename=ck["file"], revision=ck["revision"],
                                    cache_dir=str(cache_dir) if cache_dir else None))
    else:
        path = Path(checkpoint)
    if not path.exists():
        raise SystemExit(f"TabICL's checkpoint {path} does not exist")
    got = sha256_file(path)
    if got != ck["sha256"]:
        raise SystemExit(f"TabICL's checkpoint {path} has SHA-256 {got}, not the fixed {ck['sha256']}; stop")
    return path


def resolve_device(device):
    if device in (None, "auto"):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def gpu_name(device):
    if not str(device).startswith("cuda"):
        return None
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def tabicl_estimator(spec, checkpoint_path, device):
    """TabICLClassifier with its defaults, random_state 20260927, the checked checkpoint and no other download."""
    import tabicl

    installed = getattr(tabicl, "__version__", None)
    if installed != spec["version"]:
        raise SystemExit(f"tabicl {installed} is installed; the fixed version is {spec['version']}")
    from tabicl import TabICLClassifier

    est = TabICLClassifier(random_state=SEED, model_path=str(checkpoint_path), allow_auto_download=False, device=device)
    params = est.get_params()
    differ = {k: {"installed": params.get(k), "fixed": v} for k, v in spec["defaults"].items() if k != "random_state" and params.get(k) != v}
    if differ:
        raise SystemExit(f"TabICLClassifier's defaults differ from fixed_versions.json's: {differ}")
    if params.get("checkpoint_version") != spec["checkpoint"]["file"]:
        raise SystemExit(f"TabICLClassifier's checkpoint_version {params.get('checkpoint_version')} is not {spec['checkpoint']['file']}")
    return est, params


def save_tabicl(est, path):
    """TabICL's fitted state: TabICLClassifier.save without the model's weights (they are read from the checkpoint on
    load), with the training data (the context) and any KV cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(est, "save"):
        est.save(str(path), save_model_weights=False, save_training_data=True, save_kv_cache=True)
        return "TabICLClassifier.save(path, save_model_weights=False, save_training_data=True, save_kv_cache=True)"
    with open(path, "wb") as f:
        pickle.dump(est, f, protocol=5)
    return "pickle.dump(estimator, protocol=5)"


def _relocating(cls, checkpoint_path, device):
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


def load_tabicl(state_path, checkpoint_path, device=None):
    """TabICL's saved state (save_tabicl), with the model's weights read from checkpoint_path (the fixed checkpoint,
    checked by the caller) in place of the path it was fitted with, no download, and the device given. TabICL rebuilds
    its model in __setstate__ from model_path, so the loader sets model_path, allow_auto_download and device before
    that runs; the object loaded is a plain TabICLClassifier."""

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            cls = super().find_class(module, name)
            if name == "TabICLClassifier" and module.split(".")[0] == "tabicl":
                return _relocating(cls, checkpoint_path, device)
            return cls

    with open(state_path, "rb") as f:
        return Unpickler(f).load()


def tabicl_run(results=RESULTS, records_dir=RECORDS, models_dir=MODELS, split="validation", device="auto", fixed_path=None,
               checkpoint=None, checkpoint_dir=None):
    fixed_path = Path(fixed_path) if fixed_path else Path(results) / "fixed_versions.json"
    if split == "test":
        return tabicl_test(results, records_dir, models_dir, device, fixed_path, checkpoint, checkpoint_dir)
    t0 = time.time()
    spec = tabicl_spec(fixed_path)
    out_path = Path(results) / "tabicl.json"
    state_path, pred_path = Path(models_dir) / MODEL_FILES["tabicl"], new_file("tabicl", "validation", results)
    record = load_json(out_path, None)
    if record and done_before(record.get("state"), (state_path, "sha256")) and done_before(record, (pred_path, "predictions_sha256")):
        print("tabicl: built before (its state and validation predictions are unchanged); skipped", flush=True)
        return record
    device = resolve_device(device)
    ck_path = fetch_checkpoint(spec, checkpoint, checkpoint_dir)
    est, params = tabicl_estimator(spec, ck_path, device)
    data = T7.frames(records_dir, ("train", "validation"))
    x, y, _ = data["train"]
    xv, yv, rows = data["validation"]
    t_fit = time.time()
    est.fit(x, y)
    fit_seconds = round(time.time() - t_fit, 2)
    t_pred = time.time()
    probs = np.asarray(est.predict_proba(xv), dtype=float)
    predict_seconds = round(time.time() - t_pred, 2)
    classes = [int(c) for c in getattr(est, "classes_", [])]
    if classes != list(range(ANSWERS)) or probs.shape != (len(yv), ANSWERS):
        raise SystemExit(f"tabicl gave classes {classes} and probabilities of shape {probs.shape}")
    how = save_tabicl(est, state_path)
    import pandas as pd
    cat_cols = [str(c) for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
    ck = spec["checkpoint"]
    state = {"plan": PLAN, "written_utc": now(), "file": state_path.name, "sha256": sha256_file(state_path), "how": how,
             "checkpoint": {"repo": ck["repo"], "revision": ck["revision"], "file": ck["file"], "sha256": ck["sha256"],
                            "download": "huggingface_hub.hf_hub_download(repo_id, filename, revision); check the SHA-256"},
             "load": "trees_session9.py's load_tabicl(state, checkpoint path, device): the weights are read from the checkpoint",
             "estimator": "tabicl.TabICLClassifier", "random_state": SEED,
             "versions": versions(("tabicl", "torch", "sklearn", "pandas", "numpy", "huggingface_hub"))}
    save_json(Path(models_dir) / TABICL_STATE_JSON, state)
    write_probs(pred_path, rows, probs)
    record = {"plan": PLAN, "written_utc": now(), "estimator": "tabicl.TabICLClassifier", "fixed_version": spec["version"],
              "settings": {"defaults": {k: params.get(k) for k in spec["defaults"] if k != "random_state"}, "random_state": SEED,
                           "checkpoint_version": params.get("checkpoint_version"), "allow_auto_download": False,
                           "model_path": "the checked checkpoint below", "device": device, "gpu": gpu_name(device)},
              "checkpoint": {**{k: ck[k] for k in ("repo", "revision", "file", "sha256")}, "sha256_checked": True},
              "context": {"records": int(len(y)), "what": "the whole training split",
                          "why": "TabICL's documentation sets no largest context below the 55,978 training records (fixed_versions.json)"},
              "inputs": {"columns": [str(c) for c in x.columns], "categorical": cat_cols,
                         "numeric": [str(c) for c in x.columns if str(c) not in cat_cols],
                         "missing": ("categorical columns are pandas categories with a missing value (or a category not seen in "
                                     "training) as NaN, which TabICL ordinal-encodes as its own category; numeric NaN is "
                                     "mean-imputed by TabICL")},
              "fit_seconds": fit_seconds, "validation": {**metrics_of(yv, probs), "predict_seconds": predict_seconds},
              "predictions": rel(pred_path), "predictions_sha256": sha256_file(pred_path),
              "state": {"file": state_path.name, "sha256": state["sha256"], "how": how, "beside": TABICL_STATE_JSON},
              "versions": state["versions"], "seconds": round(time.time() - t0, 1), "time_utc": now()}
    save_json(out_path, record)
    print(json.dumps({"tabicl": {"validation": record["validation"], "fit_seconds": fit_seconds, "device": device}}, indent=1), flush=True)
    return record


def tabicl_test(results, records_dir, models_dir, device, fixed_path, checkpoint, checkpoint_dir):
    rc = T8.guard(results)
    val_path, test_path = new_file("tabicl", "validation", results), new_file("tabicl", "test", results)
    T8.check_frozen(rc, val_path)
    record = load_json(Path(results) / "tabicl.json", None)
    state_path = Path(models_dir) / MODEL_FILES["tabicl"]
    if not record or not done_before(record.get("state"), (state_path, "sha256")):
        raise SystemExit(f"{state_path} is missing or differs from the state tabicl.json recorded; stop")
    previous = json.loads((Path(results) / "testpreds_log.json").read_text(encoding="utf-8")).get("tabicl", {})
    if test_path.exists() and sha256_file(test_path) == previous.get("test_predictions_sha256"):
        print("tabicl: test predictions written before; skipped", flush=True)
        return previous
    spec = tabicl_spec(fixed_path)
    device = resolve_device(device)
    est = load_tabicl(state_path, fetch_checkpoint(spec, checkpoint, checkpoint_dir), device)
    data = T7.frames(records_dir, ("train", "validation", "test"))
    xv, _, rows_v = data["validation"]
    xt, yt, rows_t = data["test"]
    diff = max_difference(val_path, [r["id"] for r in rows_v], est.predict_proba(xv))
    if diff > TABICL_REPRODUCE_LIMIT:
        raise SystemExit(f"tabicl: the saved state does not reproduce its frozen validation predictions (largest difference {diff})")
    t_pred = time.time()
    probs = np.asarray(est.predict_proba(xt), dtype=float)
    write_probs(test_path, rows_t, probs)
    entry = {"state": state_path.name, "state_sha256": sha256_file(state_path), "device": device, "gpu": gpu_name(device),
             "validation_reproduced_max_abs_difference": diff, "limit": TABICL_REPRODUCE_LIMIT, "test_rows": len(rows_t),
             "seconds": round(time.time() - t_pred, 2), "test_predictions": rel(test_path),
             "test_predictions_sha256": sha256_file(test_path), "time_utc": now()}
    log_path = Path(results) / "testpreds_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log["tabicl"] = entry
    save_json(log_path, log)
    print(f"tabicl: validation reproduced (largest difference {diff:.2e}); {len(rows_t)} test rows written", flush=True)
    return entry


# ---------------------------------------------------------------- the pool

def weight_grid(m, steps=GRID_STEPS):
    """Every weight vector of m members in steps of 1 / steps that sums to 1, in itertools.product order (the order of
    session 8's weight_grid, built directly, since 11^8 products are too many to filter)."""
    out = []

    def extend(prefix, left, slots):
        if slots == 1:
            out.append(prefix + (left,))
            return
        for k in range(left + 1):
            extend(prefix + (k,), left - k, slots - 1)

    extend((), steps, m)
    return np.array(out, dtype=float) / steps


def grid_size(m, steps=GRID_STEPS):
    return math.comb(steps + m - 1, m - 1)


def _newton_torch(z, zy, iterations=NEWTON_ITERATIONS, tolerance=NEWTON_TOLERANCE):
    """Session 8's damped Newton method on beta = 1 / T for a batch of grid points (z: points x rows x answers, the
    weighted log-probabilities; zy: the true answer's), beta kept in [1 / T_MAX, 1 / T_MIN]."""
    import torch

    def stats(beta):
        s = z * beta[:, None, None]
        smax = s.amax(dim=2)
        e = torch.exp(s - smax[:, :, None])
        total = e.sum(dim=2)
        p = e / total[:, :, None]
        loss = (torch.log(total) + smax - beta[:, None] * zy).mean(dim=1)
        ez = (p * z).sum(dim=2)
        grad = (ez - zy).mean(dim=1)
        hess = ((p * z * z).sum(dim=2) - ez ** 2).mean(dim=1)
        return loss, grad, hess

    lo, hi = 1.0 / T8.T_MAX, 1.0 / T8.T_MIN
    beta = torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
    loss, grad, hess = stats(beta)
    for _ in range(iterations):
        proposal = torch.clamp(beta - grad / torch.clamp(hess, min=1e-12), lo, hi)
        new_loss, new_grad, new_hess = stats(proposal)
        worse = new_loss > loss + 1e-13  # a rise within rounding is not a rise
        if bool(worse.any()):
            proposal = torch.where(worse, (beta + proposal) / 2, proposal)
            new_loss, new_grad, new_hess = stats(proposal)
        moved = float((proposal - beta).abs().max())
        beta, loss, grad, hess = proposal, new_loss, new_grad, new_hess
        if moved < tolerance:
            break
    return loss, beta


def grid_losses_torch(logs, y, grid, device="cpu", chunk=CHUNK):
    """For each weight vector of the grid, evaluated with torch in batches of chunk points: the log loss at the
    maximum-likelihood T and 1 / T (two arrays in grid order)."""
    import torch

    logs = np.asarray(logs, dtype=float)
    grid = np.asarray(grid, dtype=float)
    dev = torch.device(device)
    m, n, k = logs.shape
    flat = torch.as_tensor(np.ascontiguousarray(logs.reshape(m, n * k)), dtype=torch.float64, device=dev)
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64), device=dev)
    losses, betas = [], []
    with torch.no_grad():
        for start in range(0, len(grid), chunk):
            w = torch.as_tensor(grid[start: start + chunk], dtype=torch.float64, device=dev)
            z = (w @ flat).reshape(-1, n, k)
            zy = z.gather(2, yt.view(1, n, 1).expand(z.shape[0], n, 1)).squeeze(2)
            loss, beta = _newton_torch(z, zy)
            losses.append(loss.cpu().numpy())
            betas.append(beta.cpu().numpy())
    return np.concatenate(losses), np.concatenate(betas)


def fit_pool_torch(logs, y, grid=None, device="cpu", chunk=CHUNK):
    """Session 8's fit_pool evaluated with torch in batches of grid points: for each weight vector, T by maximum
    likelihood and the log loss; the point with the lowest log loss (the first in grid order among exact ties)."""
    grid = weight_grid(np.asarray(logs).shape[0]) if grid is None else np.asarray(grid, dtype=float)
    losses, betas = grid_losses_torch(logs, y, grid, device, chunk)
    i = int(np.argmin(losses))
    return {"grid_index": i, "weights": [float(v) for v in grid[i]], "T": float(1.0 / betas[i]), "validation_log_loss": float(losses[i])}


def choose(new_oof, s8_oof, margin=MARGIN):
    """v0.5 uses the new pool if its out-of-fold log loss is lower than session 8's pool's by at least the margin."""
    difference = float(s8_oof) - float(new_oof)
    return ("new" if difference >= margin else "session8"), difference


def read_aligned(entries, ids=None, labels=None):
    """(ids, labels, log-probabilities (members, rows, answers)) for [(member, file)]: the rows follow ids if given,
    and otherwise the first file; every file must hold the same records with the same labels."""
    logs = []
    for member, path in entries:
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"{rel(path)} does not exist; {member}'s file must be there")
        m_ids, m_labels, m_probs = M.read_predictions(path)
        if m_probs.shape[1] != ANSWERS:
            raise SystemExit(f"{path.name} has {m_probs.shape[1]} answers")
        if ids is None:
            ids, labels = list(m_ids), np.asarray(m_labels)
        if len(m_ids) != len(ids) or set(m_ids) != set(ids):
            raise SystemExit(f"{path.name} holds different records from the first member's file")
        pos = {i: k for k, i in enumerate(m_ids)}
        order = np.array([pos[i] for i in ids])
        if not np.array_equal(np.asarray(m_labels)[order], labels):
            raise SystemExit(f"{path.name}: the labels differ from the first member's file")
        logs.append(np.log(np.clip(np.asarray(m_probs)[order], 1e-12, None)))
    return ids, labels, np.stack(logs)


def pool_entries(results, not_built_dir=None):
    """Session 8's pool (its record and its members' files, checked against their recorded SHA-256), the new pool's
    members and files, and the members dropped from the new pool with the reasons."""
    s8_path = sibling(results, "fiorillo_v0_4") / "readmission_pool.json"
    s8 = load_json(s8_path, None)
    if not s8:
        raise SystemExit(f"{rel(s8_path)} does not exist; the new pool is compared with session 8's")
    if tuple(s8["members"]) != S8_MEMBERS:
        raise SystemExit(f"session 8's pool has members {s8['members']}, not {list(S8_MEMBERS)}")
    s8_entries = []
    for m in s8["members"]:
        path = resolve(s8["member_files"][m]["file"], results)
        if not path.exists():
            raise SystemExit(f"{rel(path)} does not exist; session 8's member {m} must be there")
        if sha256_file(path) != s8["member_files"][m]["sha256"]:
            raise SystemExit(f"{rel(path)} differs from the file session 8's pool read; stop")
        s8_entries.append((m, path))
    s8_files = dict(s8_entries)
    trees = load_trees(results)
    tabicl_record = load_json(Path(results) / "tabicl.json", {})
    new_entries, dropped = [], []
    lgb_entry = (trees.get("members") or {}).get("lightgbm_tuned")
    refit_path = new_file("lightgbm_tuned", "validation", results)
    if lgb_entry and lgb_entry["check"]["passed"] and refit_path.exists():
        if sha256_file(refit_path) != lgb_entry["predictions_sha256"]:
            raise SystemExit(f"{rel(refit_path)} changed after the lightgbm stage wrote it; stop")
        new_entries.append(("lightgbm_tuned", refit_path))
    elif lgb_entry:
        dropped.append({"member": "lightgbm_tuned", "status": "left out",
                        "reason": (f"the refit's check failed (largest difference {lgb_entry['check']['max_abs_difference']:.6g}, "
                                   f"limit {REFIT_LIMIT}); the release needs every member's saved model")})
    else:
        dropped.append({"member": "lightgbm_tuned", "status": "NOT BUILT",
                        "reason": read_marker(not_built_dir, "trees") or "trees_v2.json holds no record of lightgbm_tuned's refit"})
    new_entries += [(m, s8_files[m]) for m in ("xgboost_tuned", "catboost_tuned", "arm_laya", "s7_b3")]
    for m in NEW_MEMBERS:
        path = new_file(m, "validation", results)
        if not path.exists():
            dropped.append({"member": m, "status": "NOT BUILT",
                            "reason": read_marker(not_built_dir, "tabicl" if m == "tabicl" else "trees") or f"{rel(path)} does not exist"})
            continue
        recorded = (tabicl_record if m == "tabicl" else (trees.get("members") or {}).get(m) or {}).get("predictions_sha256")
        if recorded is None:
            raise SystemExit(f"{rel(path)} exists but no record of its stage does; stop")
        if sha256_file(path) != recorded:
            raise SystemExit(f"{rel(path)} changed after its stage wrote it; stop")
        new_entries.append((m, path))
    return s8, s8_path, s8_entries, new_entries, dropped


def pool_key(name, members, files):
    blob = {"pool": name, "members": members, "sha256": [sha256_file(f) for f in files], "folds": POOL_FOLDS, "seed": SEED,
            "grid_points": grid_size(len(members))}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def out_of_fold(name, members, files, logs, y, fold_sets, results, progress, device, chunk):
    """Weights and T refitted on each fold's other rows (resumable), and the mean of the repeats' out-of-fold log losses."""
    grid = weight_grid(len(members))
    key = pool_key(name, members, files)
    store = progress.setdefault("pool", {})
    if (store.get(name) or {}).get("key") != key:
        store[name] = {"key": key, "members": list(members), "fits": {}}
    fits = store[name]["fits"]
    for r, fold in enumerate(fold_sets):
        for f in range(POOL_FOLDS):
            if f"{r}/{f}" in fits:
                continue
            fitted = fit_pool_torch(logs[:, fold != f], y[fold != f], grid, device, chunk)
            fits[f"{r}/{f}"] = {"grid_index": fitted["grid_index"], "weights": fitted["weights"], "T": fitted["T"]}
            save_progress(results, progress)
    losses = []
    for r, fold in enumerate(fold_sets):
        oof = np.empty((len(y), logs.shape[2]))
        for f in range(POOL_FOLDS):
            held = fold == f
            oof[held] = T8.pool_probs(logs[:, held], fits[f"{r}/{f}"]["weights"], fits[f"{r}/{f}"]["T"])
        losses.append(M.log_loss(y, oof))
    return {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses)), "folds": POOL_FOLDS,
            "repeats": len(fold_sets), "seed": SEED, "grid_points": len(grid)}


def release_files(member, models_dir_name="models/fiorillo_v0_5/trees"):
    """What the release needs for a pool member with nonzero weight (THE RELEASE); recode_tables.json holds the frame
    categories that every tree member and TabICL need to encode a new record."""
    return {"xgboost_codes": [f"{models_dir_name}/{MODEL_FILES['xgboost_codes']}", f"{models_dir_name}/{TABLES_FILE}"],
            "catboost_codes": [f"{models_dir_name}/{MODEL_FILES['catboost_codes']}", f"{models_dir_name}/{TABLES_FILE}"],
            "lightgbm_tuned": [f"{models_dir_name}/{MODEL_FILES['lightgbm_tuned']}", f"{models_dir_name}/{TABLES_FILE}"],
            "tabicl": [f"{models_dir_name}/{MODEL_FILES['tabicl']}", f"{models_dir_name}/{TABICL_STATE_JSON}", f"{models_dir_name}/{TABLES_FILE}"],
            "xgboost_tuned": ["models/fiorillo_v0_4/trees/xgboost_tuned.json", f"{models_dir_name}/{TABLES_FILE}"],
            "catboost_tuned": ["models/fiorillo_v0_4/trees/catboost_tuned.cbm", f"{models_dir_name}/{TABLES_FILE}"],
            "arm_laya": ["v0.1 as released (session 4): its repository is named in the release"],
            "s7_b3": ["models/fiorillo_v0_3/arm_b/ with s7_b3's readmit_window calibration in results/fiorillo_v0_3/calibration_s7.json"],
            }[member]


def pool(results=RESULTS, device="auto", chunk=CHUNK, threads=None, repeats=POOL_REPEATS, not_built_dir=None):
    import torch

    t0 = time.time()
    if threads:
        torch.set_num_threads(int(threads))
    device = resolve_device(device)
    s8, s8_path, s8_entries, new_entries, dropped = pool_entries(results, not_built_dir)
    ids, y, logs8 = read_aligned(s8_entries)
    _, _, logs_new = read_aligned(new_entries, ids, y)
    members8, members_new = [m for m, _ in s8_entries], [m for m, _ in new_entries]
    files8, files_new = [p for _, p in s8_entries], [p for _, p in new_entries]
    for members in (members8, members_new):
        if len(weight_grid(len(members))) != grid_size(len(members)):
            raise SystemExit("the weight grid has the wrong size")
    rng = np.random.default_rng(SEED)
    fold_sets = [folds_for(y, POOL_FOLDS, rng) for _ in range(repeats)]
    progress = load_progress(results)
    oof8 = out_of_fold("session8", members8, files8, logs8, y, fold_sets, results, progress, device, chunk)
    oof_new = out_of_fold("new", members_new, files_new, logs_new, y, fold_sets, results, progress, device, chunk)
    new_fit = fit_pool_torch(logs_new, y, weight_grid(len(members_new)), device, chunk)
    s8_refit = fit_pool_torch(logs8, y, weight_grid(len(members8)), device, chunk)
    s8_weights = [float(s8["weights"][m]) for m in members8]
    probs8 = T8.pool_probs(logs8, s8_weights, float(s8["T"]))
    probs_new = T8.pool_probs(logs_new, new_fit["weights"], new_fit["T"])
    chosen, difference = choose(oof_new["mean"], oof8["mean"])
    choice_time = now()

    def files_record(entries):
        return {m: {"file": rel(p), "sha256": sha256_file(p)} for m, p in entries}

    pools = {
        "session8": {"members": members8, "weights": dict(zip(members8, s8_weights)), "T": float(s8["T"]),
                     "grid_points": grid_size(len(members8)), "source": rel(s8_path), "source_sha256": sha256_file(s8_path),
                     "member_files": files_record(s8_entries),
                     "validation_log_loss": M.log_loss(y, probs8), "validation_log_loss_recorded": s8.get("validation_log_loss"),
                     "validation": {"auroc_30d": M.auroc(y == 2, probs8[:, 2]), "ece": M.ece(y, probs8)},
                     "out_of_fold_log_loss": oof8,
                     "out_of_fold_log_loss_recorded": (s8.get("out_of_fold_log_loss") or {}).get("mean"),
                     "torch_fit_on_validation": {**s8_refit, "matches_session8": bool(s8_refit["weights"] == s8_weights
                                                                                       and abs(s8_refit["T"] - float(s8["T"])) < 1e-6)},
                     "members_alone_validation_log_loss": {m: M.log_loss(y, np.exp(logs8[k])) for k, m in enumerate(members8)}},
        "new": {"members": members_new, "weights": dict(zip(members_new, new_fit["weights"])), "T": new_fit["T"],
                "grid_index": new_fit["grid_index"], "grid_points": grid_size(len(members_new)),
                "member_files": files_record(new_entries), "dropped": dropped,
                "validation_log_loss": M.log_loss(y, probs_new), "validation_log_loss_fit": new_fit["validation_log_loss"],
                "validation": {"auroc_30d": M.auroc(y == 2, probs_new[:, 2]), "ece": M.ece(y, probs_new)},
                "out_of_fold_log_loss": oof_new,
                "members_alone_validation_log_loss": {m: M.log_loss(y, np.exp(logs_new[k])) for k, m in enumerate(members_new)}},
    }
    kept = pools[chosen]
    probs = probs_new if chosen == "new" else probs8
    out_path = new_file(POOL_NAME, "validation", results)
    write_probs(out_path, [{"id": i, "label": int(v)} for i, v in zip(ids, y)], probs)
    nonzero = [m for m in kept["members"] if kept["weights"][m] > 0]
    out = {"plan": PLAN, "written_utc": now(), "system": POOL_NAME, "rows": int(len(y)),
           "rule": ("v0.5 uses the new pool if its out-of-fold log loss is lower than session 8's pool's by at least 0.002; "
                    "otherwise it keeps session 8's pool"),
           "choice": {"chosen": chosen, "new_out_of_fold_log_loss": oof_new["mean"], "session8_out_of_fold_log_loss": oof8["mean"],
                      "session8_minus_new": difference, "margin": MARGIN, "time_utc": choice_time},
           "chosen": chosen, "members": kept["members"], "weights": kept["weights"], "T": kept["T"],
           "member_files": kept["member_files"], "validation_log_loss": kept["validation_log_loss"],
           "validation": kept["validation"], "nonzero_weight_members": nonzero,
           "release_files": {m: release_files(m) for m in nonzero},
           "pools": pools,
           "folds": {"folds": POOL_FOLDS, "repeats": repeats, "seed": SEED, "rows_in_the_order_of": rel(files8[0]),
                     "method": "calibration_family_cv.py's folds_for with numpy's default_rng(seed), as session 8's pool drew them",
                     "note": "the weights and T are refitted in each fold, for both pools, by the same torch fit"},
           "fit": {"method": ("softmax((sum of w_m log p_m) / T); log p clipped at 1e-12; weights in steps of 0.1 summing to 1 "
                              "(itertools.product order); T in [0.1, 10] by session 8's damped Newton method on 1 / T; the lowest "
                              "validation log loss kept, the first in grid order among exact ties"),
                   "device": device, "gpu": gpu_name(device), "torch": torch.__version__, "dtype": "float64", "chunk": chunk,
                   "newton_iterations": NEWTON_ITERATIONS, "newton_tolerance": NEWTON_TOLERANCE},
           "predictions": rel(out_path), "predictions_sha256": sha256_file(out_path), "seconds": round(time.time() - t0, 1)}
    save_json(Path(results) / f"{POOL_NAME}.json", out)
    print(json.dumps({"chosen": chosen, "difference_session8_minus_new": difference,
                      "new": {k: pools["new"][k] for k in ("members", "weights", "T", "validation_log_loss")},
                      "out_of_fold": {"new": oof_new["mean"], "session8": oof8["mean"]}, "dropped": dropped,
                      "seconds": out["seconds"]}, indent=1), flush=True)
    return out


def frozen_inputs(results=RESULTS):
    """{repository path: SHA-256} of every file the test stages check against release_candidate.json:
    readmission_pool_v2.json and the validation file of every member of both pools. The freeze adds them to
    release_candidate.json's sha256 table; it stops if a member's file changed after the pool was fitted."""
    path = Path(results) / f"{POOL_NAME}.json"
    kept = load_json(path, None)
    if not kept:
        raise SystemExit(f"{rel(path)} does not exist; run pool first")
    out = {rel(path): sha256_file(path)}
    for p in kept["pools"].values():
        for m, v in p["member_files"].items():
            f = resolve(v["file"], results)
            got = sha256_file(f)
            if got != v["sha256"]:
                raise SystemExit(f"{v['file']} changed after the pool was fitted; stop")
            out[rel(f)] = got
    return out


def pool_test(results=RESULTS):
    rc = T8.guard(results)
    pool_path = Path(results) / f"{POOL_NAME}.json"
    kept = load_json(pool_path, None)
    if not kept:
        raise SystemExit(f"{rel(pool_path)} does not exist; run pool first")
    T8.check_frozen(rc, pool_path)
    members = kept["members"]
    val_paths = {m: resolve(kept["member_files"][m]["file"], results) for m in members}
    for m in members:
        T8.check_frozen(rc, val_paths[m])
    entries = [(m, twin_file(val_paths[m])) for m in members]
    log = json.loads((Path(results) / "testpreds_log.json").read_text(encoding="utf-8"))
    written = {**{m: v for m, v in (log.get("trees_v2") or {}).items() if isinstance(v, dict)}, "tabicl": log.get("tabicl") or {}}
    for m, path in entries:
        want = (written.get(m) or {}).get("test_predictions_sha256")
        if want and Path(path).exists() and sha256_file(path) != want and rel(path) == (written.get(m) or {}).get("test_predictions"):
            raise SystemExit(f"{rel(path)} differs from the test predictions that testpreds_log.json recorded; stop")
    ids, y, logs = read_aligned(entries)
    probs = T8.pool_probs(logs, [kept["weights"][m] for m in members], kept["T"])
    out_path = new_file(POOL_NAME, "test", results)
    write_probs(out_path, [{"id": i, "label": int(v)} for i, v in zip(ids, y)], probs)
    listed = {}
    for name, p in kept["pools"].items():
        for m, v in p["member_files"].items():
            path = twin_file(resolve(v["file"], results))
            key = rel(path)
            if key in listed:
                listed[key]["pools"].append(name)
            elif path.exists():
                listed[key] = {"member": m, "pools": [name], "rows": len(M.read_predictions(path)[0]), "sha256": sha256_file(path)}
            else:
                listed[key] = {"member": m, "pools": [name], "exists": False}
    save_json(Path(results) / "trees_v2_test.json",
              {"plan": PLAN_TEST, "written_utc": now(), "rows": int(len(y)), "chosen": kept["chosen"], "files": listed})
    out = {"plan": PLAN_TEST, "written_utc": now(), "system": POOL_NAME, "chosen": kept["chosen"], "members": members,
           "weights": kept["weights"], "T": kept["T"], "rows": int(len(y)),
           "member_test_files": {m: rel(p) for m, p in entries}, "predictions": rel(out_path), "predictions_sha256": sha256_file(out_path),
           f"{POOL_NAME}_json_sha256": T8.frozen_sha(rc, pool_path)}
    save_json(Path(results) / f"{POOL_NAME}_test.json", out)
    print(json.dumps({"rows": out["rows"], "chosen": out["chosen"], "members": members, "predictions": out["predictions"]}, indent=1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("recode", "refit", "lightgbm", "tabicl", "pool", "test"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="refit, lightgbm and tabicl: test after the freeze (needs release_candidate.json and testpreds_log.json)")
    parser.add_argument("--members", nargs="+", default=None, help="refit: xgboost_codes, catboost_codes (default both)")
    parser.add_argument("--threads", type=int, default=4, help="the trees' threads; pool: torch's CPU threads")
    parser.add_argument("--device", default="auto", help="tabicl and pool: cuda, cpu or auto")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="pool: grid points evaluated at once")
    parser.add_argument("--repeats", type=int, default=POOL_REPEATS, help="tests only: fewer out-of-fold repeats")
    parser.add_argument("--fixed-versions", type=Path, default=None, help="tabicl: default <results>/fixed_versions.json")
    parser.add_argument("--checkpoint", type=Path, default=None, help="tabicl: a local copy of the fixed checkpoint (SHA-256 checked)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="tabicl: hf_hub_download's cache_dir")
    parser.add_argument("--not-built-dir", type=Path, default=None, help="pool: the folder of the stages' NOT BUILT markers")
    args = parser.parse_args(argv)
    members = tuple(args.members) if args.members else tuple(CODES_BASE)
    if any(m not in CODES_BASE for m in members):
        parser.error(f"--members takes {sorted(CODES_BASE)}")
    if args.command == "recode":
        recode(args.results, args.records_dir, args.models_dir)
    elif args.command == "refit":
        if args.split == "test":
            trees_test(args.results, args.records_dir, args.models_dir, members, args.threads)
        else:
            refit(args.results, args.records_dir, args.models_dir, members, args.threads)
    elif args.command == "lightgbm":
        if args.split == "test":
            trees_test(args.results, args.records_dir, args.models_dir, ("lightgbm_tuned",), args.threads)
        else:
            lightgbm_refit(args.results, args.records_dir, args.models_dir)
    elif args.command == "tabicl":
        tabicl_run(args.results, args.records_dir, args.models_dir, args.split, args.device, args.fixed_versions, args.checkpoint,
                   args.checkpoint_dir)
    elif args.command == "pool":
        pool(args.results, args.device, args.chunk, args.threads, args.repeats, args.not_built_dir)
    else:
        pool_test(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

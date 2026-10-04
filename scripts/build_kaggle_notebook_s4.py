"""Write the session 4 notebooks (docs/prompt_session4.md, step 7, with corrections 4 and 5):
notebooks/02a_public_v0_arm_laya_kaggle.ipynb, notebooks/02b_public_v0_arm_bioclinical_kaggle.ipynb and
notebooks/02c_public_v0_arm_bioclinical_diagnosis.ipynb.

Each notebook is self-contained: the scripts under src/ and the question bank are embedded as %%writefile cells
(the files in this repository stay the single source of truth; rebuild, never edit the notebooks by hand). Each
rebuilds the public records from their sources and stops unless their SHA-256 values equal
results/public_v0/checksums.json (embedded). The notebooks run on Kaggle (one T4) or on a rented GPU (Runpod, one
H100); run 1 showed that two notebooks run side by side in one container shared /tmp/eddm and
/kaggle/working/outputs, so (correction 4) each notebook now works in its own folder, /tmp/eddm_<name>, and writes
only to /kaggle/working/outputs/<name>/, which is what is kept.

Correction 5a adds notebooks/02d_public_v0_arm_laya_fresh_kaggle.ipynb (arm L-fresh).

Usage: python scripts/build_kaggle_notebook_s4.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
BIOCLINICAL_REVISION = "c684f470c2ee60531d8c47c32187dba746c84201"
EMBEDDED = [
    "src/tasks/records.py",
    "src/tasks/pubmedqa.py",
    "src/tasks/uci_readmission.py",
    "src/questions/public_v0.yaml",
    "src/train/build_items.py",
    "src/train/train_laya.py",
    "src/train/assemble_reader.py",
]
EVAL_EVERY = 250  # correction 4 (run 1: 1,000)
ARMS = {
    "laya": {"file": "02a_public_v0_arm_laya_kaggle.ipynb", "title": "arm L (Laya, the release candidate)",
             "base": "/tmp/models/laya-base", "zip_model": True, "extra_train_flags": ""},
    # Correction 5a: arm B's reader gets a freshly initialized decision head (Laya's head erased the information that
    # BioClinical's reader carries to the answer slots), trained alone for the first 250 steps; arm L-fresh is Laya's
    # reader treated the same way, so that C3 compares the two readers like with like.
    "bioclinical": {"file": "02b_public_v0_arm_bioclinical_kaggle.ipynb",
                    "title": "arm B (BioClinical ModernBERT-large reader with a freshly initialized decision head; paper only, never released)",
                    "base": "/tmp/models/arm_bioclinical_init", "zip_model": False, "extra_train_flags": "--reinit-head --freeze-encoder-steps 250"},
    "laya_fresh": {"file": "02d_public_v0_arm_laya_fresh_kaggle.ipynb",
                   "title": "arm L-fresh (Laya's reader with a freshly initialized decision head; the like-with-like partner of arm B; paper only)",
                   "base": "/tmp/models/laya-base", "zip_model": False, "extra_train_flags": "--reinit-head --freeze-encoder-steps 250"},
}
DIAGNOSIS = {"file": "02c_public_v0_arm_bioclinical_diagnosis.ipynb", "name": "diagnosis",
             "assembled": "/tmp/models/arm_bioclinical_init_diagnosis"}


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def setup_cells(work, out, extra_packages=""):
    cells = [
        cell("code", "!nvidia-smi\n"),
        cell("code", f'!pip install -q "laya==0.3.20" datasets safetensors huggingface_hub lightgbm pyyaml{extra_packages}\n'
                     'import laya, torch, transformers\nprint("laya", laya.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__)\n'),
        cell("code", f'''import os, subprocess
for d in ["{work}/src/tasks", "{work}/src/train", "{work}/src/eval", "{work}/src/questions", "{work}/items", "/tmp/models", "{out}/predictions"]:
    os.makedirs(d, exist_ok=True)


def sh(command):
    """Run a shell command, stream its output here, and stop the notebook if it fails."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    if process.wait() != 0:
        raise RuntimeError(f"failed with exit code {{process.returncode}}: {{command}}")
'''),
    ]
    return cells


def embed(work, sources):
    return [cell("code", f"%%writefile {work}/{s}\n" + (ROOT / s).read_text(encoding="utf-8")) for s in sources]


def download_cell(with_bioclinical):
    text = f'''from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", revision="{LAYA_REVISION}", local_dir="/tmp/models/laya-base",
                  allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
'''
    if with_bioclinical:
        text += f'''snapshot_download("thomas-sounack/BioClinical-ModernBERT-large", revision="{BIOCLINICAL_REVISION}", local_dir="/tmp/models/bioclinical",
                  allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
'''
    return cell("code", text)


def assemble_cell(work, target):
    return cell("code", f'sh("python {work}/src/train/assemble_reader.py --laya-dir /tmp/models/laya-base --reader-dir /tmp/models/bioclinical '
                        f'--out {target} --laya-revision {LAYA_REVISION} --reader-revision {BIOCLINICAL_REVISION}")\n')


def records_cells(work, expected):
    return [
        cell("code", f'sh("cd {work} && python src/tasks/pubmedqa.py && python src/tasks/uci_readmission.py")\n'),
        cell("code", f'''import json
EXPECTED = {json.dumps(expected, indent=1, sort_keys=True)}
got = json.load(open("{work}/data/processed/public/checksums.json"))
if got != EXPECTED:
    raise RuntimeError("record checksums differ from the project build: " + json.dumps({{k: (got.get(k), v) for k, v in EXPECTED.items() if got.get(k) != v}}))
print("record checksums match the project build")
'''),
    ]


def items_cell(work, model_dir, splits):
    return cell("code", f"""for s in {splits!r}:
    sh(f"cd {work} && python src/train/build_items.py --records data/processed/public/readmit_window_{{s}}.jsonl data/processed/public/pubmedqa_answer_{{s}}.jsonl --model-dir {model_dir} --out {work}/items/{{s}}.pt --max-len 768 --head-max-len 192")
""")


def notebook(arm, spec, expected):
    name = f"arm_{arm}"
    work, out = f"/tmp/eddm_{name}", f"/kaggle/working/outputs/{name}"
    model_out = f"{work}/models/{name}"
    extra = f" {spec['extra_train_flags']}" if spec["extra_train_flags"] else ""
    cells = [cell("markdown", f"""# Public v0, {spec['title']}

Session 4 of ed-decision-model (docs/prompt_session4.md, with corrections 1 to 5). Rebuilds the public records
(PubMedQA and UCI Diabetes readmission), checks them against the project's checksums, trains Laya's typed decision
head and encoder with the log loss for one epoch, checks validation every {EVAL_EVERY} optimizer steps, fits one
temperature per question on the validation split, and writes predictions for validation and test. Runs on Kaggle
(Accelerator GPU T4 x2, one used, Internet on; about 4 hours) or on a rented GPU (one H100; about 30 minutes).
Working folder {work}; outputs in {out}/: {name}_results.zip{" and arm_laya_model.zip" if spec["zip_model"] else ""}.

Built by scripts/build_kaggle_notebook_s4.py from the files under src/; do not edit the embedded files here.""")]
    cells += setup_cells(work, out)
    cells += embed(work, EMBEDDED)
    cells.append(download_cell(arm == "bioclinical"))
    if arm == "bioclinical":
        cells.append(assemble_cell(work, spec["base"]))
    cells += records_cells(work, expected)
    cells.append(items_cell(work, spec["base"], ("train", "validation", "test")))
    cells.append(cell("code", f"""sh("python {work}/src/train/train_laya.py"
   " --train {work}/items/train.pt --val {work}/items/validation.pt --test {work}/items/test.pt"
   " --model-dir {spec['base']} --out {model_out}"
   " --device cuda --epochs 1 --micro-batch 8 --grad-accum 4 --max-len 768 --eval-every {EVAL_EVERY}{extra}"
   " --batching by-question --calibration-mode separate"
   " --predictions-dir {out}/predictions --system-name {name}"
   " --metrics {out}/{name}_metrics.json --log {out}/{name}_train_log.csv")
"""))
    package = f'''import json, shutil, zipfile
from pathlib import Path
out, model = Path("{out}"), Path("{model_out}")
shutil.copy(model / "calibration.json", out / "{name}_calibration.json")
for split in ("train", "validation", "test"):
    shutil.copy(f"{work}/items/{{split}}.summary.json", out / f"{name}_items_{{split}}.summary.json")
for f in ("counts.json", "checksums.json"):
    shutil.copy(f"{work}/data/processed/public/{{f}}", out / f)
small = [p for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".csv")] + sorted((out / "predictions").glob("{name}*.csv"))
with zipfile.ZipFile(out / "{name}_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in small:
        z.write(p, p.relative_to(out))
'''
    if spec["zip_model"]:
        package += f'''with zipfile.ZipFile(out / "arm_laya_model.zip", "w", zipfile.ZIP_STORED) as z:
    for p in sorted(model.rglob("*")):
        if p.is_file():
            z.write(p, Path("arm_laya") / p.relative_to(model))
'''
    package += f'''m = json.load(open(out / "{name}_metrics.json"))
print("temperatures:", m["temperature"])
print("test, after temperature:", json.dumps(m["test_per_question_after_temperature"], indent=1))
print("sequences per second:", m["sequences_per_second"], "| elapsed seconds:", m["elapsed_seconds"])
for z in sorted(out.glob("*.zip")):
    print(z.name, round(z.stat().st_size / 1e6, 1), "MB")
print("Done. Download the zip files from {out}/ (on Kaggle, from the Output tab).")
'''
    cells.append(cell("code", package))
    return wrap(cells)


def diagnosis_notebook(expected):
    work, out = "/tmp/eddm_diagnosis", "/kaggle/working/outputs/diagnosis"
    cells = [cell("markdown", f"""# Public v0, arm B diagnosis (correction 5)

Why did run 1's arm B give nearly the same answer for every example? Builds arm B's starting model (BioClinical
ModernBERT-large's reader under Laya's decision head) exactly as notebook 02b does, and compares it with Laya as
published, before any training, with src/eval/diagnose_reader.py on 1,500 training and 400 validation items per
question. Trains nothing. Runs on a rented GPU (one H100; a few minutes). Output: {out}/arm_b_diagnosis.json.

Built by scripts/build_kaggle_notebook_s4.py from the files under src/; do not edit the embedded files here.""")]
    cells += setup_cells(work, out, extra_packages=" scikit-learn")
    cells += embed(work, EMBEDDED + ["src/eval/diagnose_reader.py"])
    cells.append(download_cell(True))
    cells.append(assemble_cell(work, DIAGNOSIS["assembled"]))
    cells += records_cells(work, expected)
    cells.append(items_cell(work, "/tmp/models/laya-base", ("train", "validation")))
    cells.append(cell("code", f"""sh("python {work}/src/eval/diagnose_reader.py --model laya=/tmp/models/laya-base --model bioclinical={DIAGNOSIS['assembled']}"
   " --train-items {work}/items/train.pt --val-items {work}/items/validation.pt --out {out}/arm_b_diagnosis.json --n-train 1500 --n-val 400")
print(open("{out}/arm_b_diagnosis.json").read())
"""))
    return wrap(cells)


def wrap(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    expected = json.loads((ROOT / "results" / "public_v0" / "checksums.json").read_text(encoding="utf-8"))
    books = [(spec["file"], notebook(arm, spec, expected)) for arm, spec in ARMS.items()]
    books.append((DIAGNOSIS["file"], diagnosis_notebook(expected)))
    for file, nb in books:
        path = ROOT / "notebooks" / file
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()

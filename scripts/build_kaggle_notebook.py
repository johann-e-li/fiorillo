"""Write notebooks/01_practice_finetune_kaggle.ipynb from src/train/practice_data.py and src/train/train_laya.py.

The notebook is self-contained: the two scripts are embedded as %%writefile cells, so the files under src/train/
stay the single source of truth and the notebook is rebuilt, never edited by hand.

Usage: python scripts/build_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_practice_finetune_kaggle.ipynb"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main():
    practice_data = (ROOT / "src" / "train" / "practice_data.py").read_text(encoding="utf-8")
    train_laya = (ROOT / "src" / "train" / "train_laya.py").read_text(encoding="utf-8")
    cells = [
        cell("markdown", """# Practice fine-tune: Laya on DAIR Emotion (Kaggle, one T4)

Session 3 of ed-decision-model. Trains Laya's typed decision head and encoder on the public DAIR Emotion
dataset with the log loss, then evaluates on the test split. Pass mark: test accuracy above 0.573 (Laya's
reported zero-shot accuracy). Settings for this notebook: Accelerator GPU T4 x2, Internet on. Runs in about
20 to 40 minutes. Outputs land in /kaggle/working/outputs/.

Built by scripts/build_kaggle_notebook.py from src/train/practice_data.py and src/train/train_laya.py; do not
edit the embedded scripts here."""),
        cell("code", "!nvidia-smi\n"),
        cell("code", '!pip install -q "laya==0.3.20" "datasets>=3.0.0" safetensors huggingface_hub\nimport laya, torch, transformers, datasets\nprint("laya", laya.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)\n'),
        cell("code", f'''from huggingface_hub import snapshot_download
MODEL_DIR = snapshot_download("convaiinnovations/laya", revision="{REVISION}", local_dir="/kaggle/working/laya-base",
    allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
print(MODEL_DIR)
'''),
        cell("markdown", "## Data builder (src/train/practice_data.py)"),
        cell("code", "%%writefile /kaggle/working/practice_data.py\n" + practice_data),
        cell("markdown", "## Trainer (src/train/train_laya.py)"),
        cell("code", "%%writefile /kaggle/working/train_laya.py\n" + train_laya),
        cell("code", "!python /kaggle/working/practice_data.py --model-dir /kaggle/working/laya-base --out /kaggle/working/practice\n"),
        cell("code", """!python /kaggle/working/train_laya.py \\
    --train /kaggle/working/practice/emotion_train.pt \\
    --val /kaggle/working/practice/emotion_validation.pt \\
    --test /kaggle/working/practice/emotion_test.pt \\
    --model-dir /kaggle/working/laya-base \\
    --out /kaggle/working/outputs/practice-laya-emotion-v0 \\
    --device cuda --epochs 3 --micro-batch 16 --grad-accum 2 --calibration-mode config \\
    --metrics /kaggle/working/outputs/emotion_metrics.json \\
    --log /kaggle/working/outputs/emotion_train_log.csv
"""),
        cell("code", """import json, shutil
m = json.load(open("/kaggle/working/outputs/emotion_metrics.json"))
print("test accuracy:", m["test_before_temperature"]["accuracy"], "| pass mark:", m["pass_mark"], "| passed:", m["passed"])
print("sequences per second:", m["sequences_per_second"], "| elapsed seconds:", m["elapsed_seconds"])
shutil.copy("/kaggle/working/practice/emotion_counts.json", "/kaggle/working/outputs/emotion_counts.json")
print("Download /kaggle/working/outputs/ (metrics, log, counts, and the model folder) from the Output tab.")
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()

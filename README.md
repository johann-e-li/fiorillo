# Fiorillo

Fiorillo answers typed medical questions with calibrated probabilities. Version 0.5 answers three, each with its own specialist: whether a randomized trial's article reports an outcome significantly higher, significantly lower or not significantly different with an intervention than with a comparator (Evidence Inference 2.0); the answer to a biomedical research question from its abstract (PubMedQA: yes, no or maybe); and how soon a patient with diabetes is readmitted after a hospital discharge (the Diabetes 130-US Hospitals data of the University of California, Irvine (UCI) Machine Learning Repository). It is for research. It is not a medical device and must not be used to make clinical decisions.

This repository holds the code, the results files and the sources of the paper "Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release" (Johann Emmanuel Li, 2026). The paper's arXiv link will be added here when it is posted.

- The model: https://huggingface.co/johanneli/fiorillo-v0.5 (digital object identifier 10.57967/hf/10722)
- The registration and its update, on the Open Science Framework: https://osf.io/kaxmn/
- ORCID iD: https://orcid.org/0009-0009-8726-460X

## What is here

- `src/`: the code that builds the question records, trains the readers and decision heads, calibrates and evaluates them, and checks reported numbers against the results files. Its tests are in `src/tests/`.
- `scripts/`: the scripts I ran on my Mac and on rented cloud servers with graphics processing units, including the release script and the release's usage check.
- `results/`: the results files of v0.4 and v0.5, the two versions the paper reports, and the earlier results files that the documents below cite.
- `docs/`: the registration (`osf_prereg_fiorillo_v0_4.md`) and its update (`osf_update_session9.md`), the reports on v0.4 and v0.5 with their figure checks, the model card of v0.5, the paper's sources (`docs/preprint/`) with the checks I ran on it, and an independent review of its draft (`docs/research/`).

This is a copy of the parts of my working repository that the paper relies on. Some files cite my working plans, log and notes, which are not part of it.

## What is not here

- The weights and calibration files. They are on Hugging Face, with `predict_v0_5.py` to run them.
- The datasets. The code in `src/tasks/` downloads them from their sources into a local `data/` folder: PubMedQA from Hugging Face, Evidence Inference 2.0 from its GitHub repository, and the UCI data from the UCI Machine Learning Repository.
- Data under a data use agreement. None is in this repository. The only test fixtures are two small invented files in PubMed's XML format.

## Running the code

`pyproject.toml` and `uv.lock` fix the environment for an Apple Silicon Mac with Python 3.11: run `uv sync`, then `uv run pytest src/tests`. On other machines, install the packages that `pyproject.toml` lists. Tests that need a base model's tokenizer are skipped when it is not on disk, and two tests that read my working repository's history and local data are skipped here (`src/tests/conftest.py` says which).

## License

The code and the results files are under the Apache License 2.0 (`LICENSE`). The datasets, base models and articles keep their own licenses, which the model card (`docs/model_card_v0_5.md`) lists.

# 🛰️ EarthLens AI — AI-Based Satellite Image Classification

Deep-learning system (TensorFlow/Keras) that classifies **EuroSAT RGB** Sentinel-2
tiles (64×64 px, 27,000 images) into **10 land-use classes**: AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake.

Built for the **AIE231 Neural Networks course project** (Weeks 9–14), following
the full engineering lifecycle: problem framing → EDA → leak-free data pipeline →
baselines → 3 neural models → tuning/ablations → test evaluation + error analysis →
inference demo.

## Problem formulation

An environmental or land-use analyst needs to label small RGB satellite tiles to prioritize manual map review. Incorrect labels can misdirect review resources and obscure minority land-use types. Input is a 64×64×3 RGB tile; output is a ten-class softmax vector. The network uses categorical cross-entropy on one-hot labels. Primary metric is validation macro-F1, which gives each class equal importance despite uneven frequencies; accuracy and per-class recall supplement it. This is a decision-support prototype, not a production GIS system. Geographic overlap and near-duplicates cannot be excluded without reliable scene-group metadata; exact RGB duplicates are removed.

## Quick test first

Upload `START_COLAB.ipynb` and the ZIP to Google Colab and follow the cells. Smoke runs train on a capped dataset and are never eligible for final selection. The full-run default is 12 epochs with early stopping and a 27-minute training-loop guard; actual total runtime depends on GPU and overhead. The report and slides must be completed with actual measured results; see `report/README.md`. AI-assisted changes are documented in `docs/CHANGES.md`; each team member must verify, understand and commit their own work.

## Team & file ownership

| Member | Role | Files they own | Milestone deliverables |
|---|---|---|---|
| Member 1 | Data Engineer | `data/download_data.py`, `src/data.py`, `notebooks/01_eda.ipynb` | dataset audit, leak-free split, augmentation + pipelines |
| Member 2 | Baselines & Model A | `src/baselines.py`, `src/train.py` | 2 measured baselines, Model A (custom CNN), tuning comparisons |
| Mariam (Member 3) | Tech Lead & Model Architect | `configs/config.py`, `src/models.py` | problem/metric plan, Model B (frozen ResNet50), Model C (fine-tuned) |
| Member 4 | Experimentation & Evaluation | `src/run_experiments.py`, `src/evaluate.py`, `experiments/experiment_log.csv` | LR/batch/optimizer studies, final test metrics, 20-error analysis |
| Member 5 | Deployment & Integration | `src/predict.py`, `app/app.py`, `README.md`, `notebooks/02_results.ipynb`, `docs/` | repo setup, modular integration, inference + demo, report/slides |

Branch → pull-request workflow and merge order: see **`CONTRIBUTING.md`** and
**`docs/GIT_WORKFLOW.md`**. Every member commits from their own account on a
feature branch, merged through a PR reviewed by a different member.

## Results table (fill at Milestone 4, 10 Dec)

| Model | Params | val accuracy | val macro-F1 | **test accuracy** | **test macro-F1** |
|---|---|---|---|---|---|
| Majority-class baseline | 0 | — | — | — | — |
| MLP 1-hidden-layer | ~3.1M | — | — | — | — |
| Model A — custom CNN | — | — | — | — | — |
| Model B — ResNet50 frozen | 23.6M | — | — | — | — |
| Model C — ResNet50 fine-tuned | — | — | — | — | — |

Numbers come from `experiments/experiment_log.csv` and `experiments/final_evaluation/comparison.json`.
A deep model that does not beat the baselines is a finding to explain, not to hide.

## Setup

```bash
# 1) environment (Python 3.11–3.12; free Colab GPU works as-is)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) dataset (~95 MB, one time — NOT stored in git)
python data/download_data.py
python -m src.data --audit                            # Member 1's audit report

# 3) baselines (Member 2)
python -m src.baselines --epochs 30

# 4) train the three models
python -m src.train --model model_a --author "Member2"
python -m src.train --model model_b --author "Mariam"
python -m src.train --model model_c --author "Mariam" --lr 1e-4

# 5) tuning + ablations (Member 4)
python -m src.run_experiments --study lr --author "Member4"
python -m src.run_experiments --study batch  --author "Member4"
python -m src.run_experiments --study optimizer --author "Member4"
python -m src.run_experiments --study architecture --author "Member4"
python -m src.run_experiments --study ablation_aug --author "Member1"
python -m src.run_experiments --study ablation_dropout --author "Member3"
python -m src.run_experiments --study ablation_weights --author "Member1"
python -m src.run_experiments --study ablation_bn --author "Member3"
python -m src.run_experiments --study seeds --author "Member4"

# 6) final test evaluation (once!) + error analysis (Member 4)
python -m src.evaluate --latest --all-models

# 7) inference + demo (Member 5)
python -m src.predict --latest --image data/EuroSAT/River/River_10.jpg
streamlit run app/app.py
```

## Repository structure

```
earthlens-ai/
├── README.md                 this file: problem, results, setup, commands
├── requirements.txt          pinned package versions
├── configs/
│   └── config.py             ONE file with every hyper-parameter (Mariam)
├── data/
│   └── download_data.py      download script + README info (no raw data in git!)
├── notebooks/
│   ├── 01_eda.ipynb          EDA — imports from src/ (Member 1)
│   └── 02_results.ipynb      results presentation (Member 5)
├── src/
│   ├── data.py               loading, cleaning, splitting, pipelines (Member 1)
│   ├── baselines.py          both baselines (Member 2)
│   ├── models.py             Models A/B/C, hand param counts, GradientTape (Mariam)
│   ├── train.py              training loop, callbacks, logging (Member 2)
│   ├── run_experiments.py    tuning studies, ablations (Member 4)
│   ├── evaluate.py           metrics, confusion matrix, ROC, errors (Member 4)
│   └── predict.py            inference on raw input (Member 5)
├── experiments/
│   ├── experiment_log.csv    every run: hyper-params, seed, time, metrics
│   └── runs/                 per-run training curves
├── models/                   saved .keras models + manifest.json (not in git)
├── app/
│   └── app.py                Streamlit live demo (Member 5)
├── docs/
│   └── GIT_WORKFLOW.md       step-by-step GitHub workflow for the team
└── report/                   report PDF + slides (Week 14)
```

## Dataset

EuroSAT RGB — 27,000 tiles, 10 classes, 64×64, Sentinel-2 MSI (Helber et al., 2019).
Citation: `Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). EuroSAT: A Novel
Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,
IEEE JSTARS. https://zenodo.org/records/7711810 — licence CC BY 4.0 (academic use
permitted). Download with `python data/download_data.py`.

## Reproducibility

- fixed split seed (`SEED = 42`); training `--seed` changes initialization and augmentation only
- every run logged with its hyper-parameters, seed, wall-clock time and metrics
- one command reproduces the final model's test metrics:
  `python -m src.evaluate --latest --all-models` (after all full runs and `--study seeds`; final evaluation writes a frozen report)
- pinned dependency versions in `requirements.txt`

## Rules we must not break (from the AIE231 spec)

1. **Test discipline** — the test split is used once, only by `src/evaluate.py`.
2. **No raw data in git** — `data/` is in `.gitignore`.
3. **Modular code** — notebooks import from `src/`; no monolithic notebook.
4. **Meaningful commits from every member's own account**, on feature branches,
   merged through PRs reviewed by a different member.
5. Pretrained weights: only ResNet50/VGG16 ImageNet (image track) — we use ResNet50.

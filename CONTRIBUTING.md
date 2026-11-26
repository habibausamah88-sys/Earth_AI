# Contributing — EarthLens AI (AIE231)

كل عضو يشتغل على **branch خاصة بيه** وترفع بـ **Pull Request** تتراجع بواحد
تاني قبل الميرج. الملفات دي مسؤوليتك (زي ما متفقين في الشيت):

| Member | Branch | الملفات اللي بتعدلها |
|---|---|---|
| Member 1 | `feat/member1-data` | `data/download_data.py`, `src/data.py`, `notebooks/01_eda.ipynb` |
| Member 2 | `feat/member2-baselines` | `src/baselines.py`, `src/train.py` |
| Mariam  | `feat/mariam-models` | `configs/config.py`, `src/models.py` |
| Member 4 | `feat/member4-experiments` | `src/run_experiments.py`, `src/evaluate.py`, `experiments/` |
| Member 5 | `feat/member5-deploy` | `src/predict.py`, `app/app.py`, `README.md`, `docs/`, `notebooks/02_results.ipynb` |

## Branch naming

```
feat/<member>-<topic>      # new work        e.g. feat/member1-data
fix/<member>-<topic>       # bug fix         e.g. fix/member2-loader-crash
docs/<member>-<topic>      # documentation   e.g. docs/member5-readme
```

## Commit messages

```
<type>: <what changed in <=50 chars>

data: add stratified 70/15/15 split with leakage check
models: add hand parameter count report for Model A
```

Types: `data`, `baselines`, `models`, `train`, `experiments`, `evaluate`,
`predict`, `app`, `docs`, `report`. **Commit from your own GitHub account**
(the spec gives code-quality marks for commit history from all members —
one-commit uploads at the deadline lose those marks).

## Pull Request rules

1. Title: `[Milestone 3] Member 1: augmentation + data pipeline`.
2. Description: what changed, how you tested it (paste the command + output).
3. Reviewer = any member who does NOT own the files you touched.
4. The reviewer must be able to run your code on their machine.
5. Merge after **1 approval**; squash-merge is fine; delete the branch after.
6. Never push to `main` directly — even one-line fixes go through a PR.

## Definition of done (per milestone sheet)

- [ ] Code runs on a clean machine / free Colab GPU (`pip install -r requirements.txt`)
- [ ] `experiments/experiment_log.csv` has a row for every run you made
- [ ] No `data/` or `*.keras` files in the diff (they are gitignored)
- [ ] Docstrings on every function you wrote; no hard-coded paths
- [ ] Your report section (for your work package) is drafted in `report/`

## Milestone merge order (merge left → right)

| Milestone | Due | Branches to merge (in this order) |
|---|---|---|
| M1+2 | 26 Nov | member5 (repo skeleton) → mariam (config) → member1 (data) → member2 (baselines+train) → member4 (evaluate scaffolding) |
| M3 | 3 Dec | member1 (augmentation) → member2 (Model A) → mariam (Model B) → member4 (experiment log) → member5 (integration) |
| M4 | 10 Dec | member1 (ablations) → member2 (CNN comparisons) → mariam (Model C) → member4 (studies) → member5 (predict+demo) |
| Final | 17 Dec `v1.0` | all report sections → member5 (report/slides/tag) |

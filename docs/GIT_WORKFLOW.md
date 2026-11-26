# GitHub workflow — step by step for the EarthLens AI team

Goal: 5 members, one repository, no conflicts, and a commit history that shows
everybody worked (the spec grades this). Total time for the setup below: ~15 min.

## 0) One-time setup (Member 5 does this BEFORE anyone else)

```bash
# Member 5: create the repo on github.com → New repository → name: earthlens-ai
#           (Private is fine — the TA needs access; add teammates as collaborators)
mkdir earthlens-ai && cd earthlens-ai
git init
# copy the project skeleton files here (from the team package)
git add .
git commit -m "docs: project skeleton, README and git workflow"
git branch -M main
git remote add origin https://github.com/<team-account>/earthlens-ai.git
git push -u origin main
# GitHub → Settings → Collaborators → invite Member 1, 2, 3 (Mariam), 4
```

## 1) Every member: clone + identity (once per machine)

```bash
git clone https://github.com/<team-account>/earthlens-ai.git
cd earthlens-ai
git config user.name  "Your Name"            # YOUR name, not the team name
git config user.email "your-github@email.com"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py                 # dataset stays local, never in git
```

## 2) Daily loop: branch → work → push → PR → merge

```bash
# start from the latest main
git switch main && git pull origin main

# create YOUR branch (use the names from CONTRIBUTING.md)
git switch -c feat/member1-data

# ... edit only the files YOU own (see the ownership table) ...

git add <your-files>                         # add files explicitly, never 'git add .'
git commit -m "data: add stratified split with leakage check"
git push -u origin feat/member1-data

# open a Pull Request on github.com → Compare & pull request
# reviewer (a different member) reviews → Approve → Merge pull request
git switch main && git pull origin main      # refresh your local main
git branch -d feat/member1-data              # clean up
```

## 3) Getting your teammate's merged work

```bash
git switch main && git pull origin main
```

## 4) When a PR has conflicts (usually configs or README)

```bash
git switch feat/member1-data
git fetch origin
git merge origin/main
# git marks conflicts:  <<<<<<< HEAD | ======= | >>>>>>> origin/main
# open the file, keep the correct lines, delete the markers
git add <fixed-file>
git commit -m "merge: resolve conflict with main"
git push
# the PR updates automatically → reviewer merges
```

## 5) Rules that keep us out of trouble

- Never `git push --force`, never commit to `main` directly.
- Never commit `data/`, `models/*.keras`, `.venv/` (all gitignored — check with
  `git status` before every commit).
- Small PRs (≤ ~300 lines) merge fast; giant PRs stall the team.
- If two members need the same file, the earlier milestone owner merges first,
  the second rebases: `git fetch origin && git rebase origin/main`.
- Final delivery (17 Dec): everything merged, Member 5 tags the repo —
  `git tag v1.0 && git push origin v1.0` (spec 6.4 requires the `v1.0` tag).

## 6) Where each deliverable lands (per the team sheet)

| Milestone | Date | Member → what to submit |
|---|---|---|
| M1+2 | 26 Nov | M1: EDA plots + leak-free split · M2: two baselines + comparison · Mariam: problem statement + metric · M4: leakage checks + metric script · M5: repo + README + dataset citation |
| M3 | 3 Dec | M1: augmentation + loader · M2: Model A + curves + summary · Mariam: Model B frozen + architecture notes · M4: experiment CSV + curves · M5: modular src/ + reproducible commands |
| M4 | 10 Dec | M1: augmentation/imbalance ablations · M2: CNN comparisons + param counts · Mariam: Model C fine-tuned + frozen comparison · M4: LR/batch/optimizer studies · M5: predict.py + demo |
| Final | 17 Dec (tag v1.0) | M1: data decisions section · M2: CNN results + failure explanations · Mariam: final model review · M4: test metrics + confusion matrix + ROC + 20 errors · M5: report PDF + slides + tagged repo |

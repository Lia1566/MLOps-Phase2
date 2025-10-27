
DVC (Data Version Control) Setup Instructions
==============================================

Objectives:
- Track ONLY RAW data with DVC (e.g., 'data/raw/')
- Generate processed data/models/figures via pipeline ('dvc.yaml' outs)
- Never 'dvc add' processed files

Project root: /Users/lia/Desktop/Fase1

Step 0: One-time Git init (if needed)
--------------------------------------
git init
git add .
git commit -m "init repo"

Step 1: Initialize DVC
----------------------
Use the helper script you created (portable & repeatable):

./scripts/init_dvc.sh

(Equivalent to `dvc init`, configure, and commit .dvc files.)

Step 2: Track ONLY raw data with DVC
-----------------------------------
Place the original dataset(s) in `data/raw/` and run:

./scripts/track_data_dvc.sh

This executes:
- dvc add data/raw
- git add data/raw.dvc
- git commit -m "track raw data with DVC"

Step 3: Define and Run the pipeline
--------------------------------
The `dvc.yaml` stage runs preprocessing and declares outs:

dvc repro

On each run, it (re)creates:
- data/processed/*.csv
- models/ (if produced)
- reports/figures/ (plots)

Step 4: Check status & push artifacts
----------------------------
dvc status
git add dvc.yaml dvc.lock
git commit -m "pipeline run"
dvc push

(If you haven't set a remote, the init script added a local one: ../dvcstore)

Common Commands
---------------
# Check what changed
dvc status

# Pull data from remote
dvc pull

# Push data to remote
dvc push

# Reproduce pipeline
dvc repro

# Show pipeline DAG
dvc dag

Benefits of DVC
===============
✓ Version control for large datasets
✓ Reproducibility of data pipeline
✓ Efficient storage (only metadata in Git)
✓ Easy collaboration with team members
✓ Data integrity verification with MD5 hashes

#!/bin/bash
# DVC Initialization Script
# This script sets up DVC for data version control

set -euo pipefail

echo "=================================================="
echo "INITIALIZING DVC FOR DATA VERSION CONTROL"
echo "=================================================="

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "ERROR: DVC is not installed"
    echo "Install with: pip install dvc"
    exit 1
fi

# Check if git repository exists
if [ ! -d .git ]; then
    echo "ERROR: Not a git repository"
    echo "Initialize git first: git init && git add . && git commit -m 'init repo'"
    exit 1
fi

# Initialize DVC
echo ""
echo "Step 1: Initializing DVC..."
dvc init

# Check if initialization was successful
if [ $? -eq 0 ]; then
    echo "✓ DVC initialized successfully"
else
    echo "DVC initialization failed"
    exit 1
fi

echo ""
echo "Step 2: Configuring DVC..."
# Set autostage to true (automatically stage DVC files)
dvc config core.autostage true
echo "✓ DVC configuration complete"

echo ""
echo "Step 3: Adding DVC files to git..."
git add .dvc .dvcignore || true
git commit -m "chore: init dvc" || true

echo ""
echo "Step 4: Add a local remote for storage"
mkdir -p ../dvcstore
dvc remote add -d localstore ../dvcstore || true
git add .dvc/config || true
git commit -m "chore: add local dvc remote" || true

echo ""
echo "=================================================="
echo " DVC INITIALIZATION COMPLETE"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Put raw data in: data/raw/"
echo "2. Track raw dir: dvc add data/raw"
echo "3. Commit pointer: git add data/raw.dvc && git commit -m 'track raw data'"
echo "4. Push data: dvc push"
echo "5. Define pipeline outs in dvc.yaml for data/processed, models, reports/figures"

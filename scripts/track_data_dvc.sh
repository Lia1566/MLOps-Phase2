#!/bin/bash
set -euo pipefail
# DVC Data Tracking Script
# Tracks all data files with DVC

echo "=================================================="
echo "ADDING RAW DATA FILES TO DVC TRACKING"
echo "=================================================="

RAW_DIR="data/raw"

if [ -d "$RAW_DIR" ]; then
  echo "Tracking $RAW_DIR ..."
  dvc add "$RAW_DIR"
  echo "✓ Added $RAW_DIR to DVC"
else
  echo "ERROR: $RAW_DIR not found. Create it and place your raw files there."
  exit 1
fi

echo ""
echo "Adding .dvc pointers to git..."
git add data/raw.dvc || true
git commit -m "chore: track raw data with DVC" || true

echo ""
echo "Pushing raw data to DVC remote..."
dvc push || true

echo ""
echo "=================================================="
echo "✓ DATA TRACKING COMPLETE"
echo "=================================================="
echo ""
echo "NOTE: Processed data, models, and figures should be pipeline outs in dvc.yaml."
echo "Run the pipeline with: dvc repro"

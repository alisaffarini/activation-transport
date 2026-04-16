#!/bin/bash
# Reproducibility script for research run: run_047_interp
# Generated: 2026-03-11T09:16:23.051563
#
# This script recreates the experimental results from the paper.
# Requires Python 3.10+ and pip.

set -e

echo "=== Reproducibility Script for Run run_047_interp ==="
echo "Setting up environment..."

# Create virtual environment
python3 -m venv .reproduce_env
source .reproduce_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy scipy
pip install torch numpy torchvision scipy

echo "Environment ready. Running experiment..."

# Run the experiment
python experiment.py 2>&1 | tee experiment_log.txt

echo ""
echo "=== Experiment complete ==="
echo "Results saved to experiment_log.txt"
echo "Look for RESULTS: line in output for structured results."

# Deactivate
deactivate

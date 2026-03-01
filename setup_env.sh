#!/bin/bash
#
# Creates a Python venv on top of the mlmi2 conda environment.
# Inherits torch, torchvision, numpy, pillow etc. from mlmi2,
# and adds scipy (missing from mlmi2).
#
# Run once on a login node before submitting any SLURM jobs:
#   bash setup_env.sh
#
# The venv is created at ~/ddpm_venv and is reused by all SLURM scripts.

set -e

MLMI2_PYTHON=/rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/envs/mlmi2/bin/python
VENV_DIR="$HOME/ddpm_venv"

echo "Using base Python: $MLMI2_PYTHON"
echo "Creating venv at:  $VENV_DIR"

# --system-site-packages: inherit torch, torchvision, numpy, pillow, etc. from mlmi2
$MLMI2_PYTHON -m venv --system-site-packages "$VENV_DIR"

echo "Installing additional packages (scipy)..."
"$VENV_DIR/bin/pip" install --quiet scipy

echo ""
echo "Done. Venv ready at $VENV_DIR"
echo "Python: $VENV_DIR/bin/python"
echo ""
"$VENV_DIR/bin/python" -c "import torch, torchvision, numpy, scipy; print(f'torch {torch.__version__}, scipy {scipy.__version__} — all OK')"

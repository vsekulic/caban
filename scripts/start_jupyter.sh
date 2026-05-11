#!/usr/bin/env bash
# Launch JupyterLab inside the caban conda env, listening on loopback only.
# Intended to be run inside a tmux or screen session so the server survives
# SSH disconnects. See docs/JUPYTER_SETUP.md.
set -euo pipefail

# Activate conda env (works with both miniforge/mamba and stock conda)
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate caban

cd /Users/vsekulic/code/sstca2

exec jupyter lab

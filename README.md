# caban

A Python toolkit for analysis of hippocampal calcium imaging data, with a focus on population dynamics, representational change, and spatial coding in longitudinal in vivo recordings.

## Features

- Miniscope calcium imaging analysis
- Longitudinal neuron registration support
- Population vector analyses
- Bayesian decoding
- Spatial tuning analysis
- Trace fear conditioning analysis workflows
- Visualization and plotting utilities

## Running on a remote server (JupyterLab + VS Code)

The recommended way to run the pipeline on a Linux server is via a persistent
JupyterLab kernel, with code editing through VS Code's Remote-SSH extension.
This setup keeps the kernel alive across browser closes and SSH disconnects,
and lets you step through the analyses cell-by-cell instead of running the
whole `caban.main.py` script.

See [docs/JUPYTER_SETUP.md](docs/JUPYTER_SETUP.md) for end-to-end instructions
covering the conda env, JupyterLab config, `tmux`/`screen` kernel persistence,
SSH tunnelling, and VS Code Remote-SSH wiring.

## Author

Vladislav Sekulic <vlad.sekulic@gmail.com>

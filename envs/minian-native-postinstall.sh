#!/bin/bash
# Finish minian-native after `mamba env create -f envs/minian-native.yml`.
#
# Reproduces what the RIKEN CBP server's working `minian` env had but conda cannot give
# on osx-arm64 (see plans/local_minian_pipeline_plan.md §4.2):
#   - pure-Python packages with no arm64 conda build at the server's version, from PyPI;
#   - the server's one-line jinja2 patch to bokeh 1.4.0 and panel 0.8.0, which otherwise
#     fail to import against jinja2 3.1 (`Markup` moved to markupsafe).
#
# Re-run after ANY conda operation on the env: datashader makes conda reinstate
# bokeh 2.4.3 underneath the pip-installed 1.4.0.
#
# Usage:  bash envs/minian-native-postinstall.sh [env-name]   (default: minian-native)

set -euo pipefail
ENV=${1:-minian-native}
PREFIX="$(conda run -n "$ENV" python -c 'import sys; print(sys.prefix)')"
PIP="$PREFIX/bin/pip"
SP="$PREFIX/lib/python3.8/site-packages"

"$PIP" install --no-deps --force-reinstall \
  bokeh==1.4.0 panel==0.8.0 selenium==3.141.0 medpy==0.4.0 SimpleITK==2.1.1

patch_markup() {   # $1 file, $2 md5 of the server's patched copy
  local f="$SP/$1"
  sed -i '' 's/^from jinja2 import Environment, Markup, FileSystemLoader$/from jinja2 import Environment, FileSystemLoader\
from markupsafe import Markup/' "$f"
  local got; got=$(md5 -q "$f")
  [ "$got" = "$2" ] || { echo "ERROR: $1 md5 $got != server's $2" >&2; exit 1; }
  echo "patched $1 (identical to the server's copy)"
}
patch_markup bokeh/core/templates.py 025af6b4fbd4e28640b7e076964f770d
patch_markup panel/io/resources.py   5ac7867d1edf5e4fc15d5904041a15d9

"$PREFIX/bin/python" -c "import bokeh, panel, holoviews; print('bokeh', bokeh.__version__, '| panel', panel.__version__, '| holoviews', holoviews.__version__)"

#!/bin/bash
# Finish minian-native on Linux (the Razer, WSL2) after
#   mamba env create -f envs/minian-native-linux.yml
# The Linux twin of envs/minian-native-postinstall.sh (which uses macOS sed/md5): the same PyPI
# packages at the Mac's versions, and the RIKEN server's one-line jinja2 patch to bokeh/panel,
# checked against the server's md5. datashape 0.5.4 is not on PyPI; conda-forge already supplies
# it as a datashader dependency. Re-run after any conda operation on the env.
set -euo pipefail
ENV=${1:-minian-native}
PREFIX="$(conda run -n "$ENV" python -c 'import sys; print(sys.prefix)')"
SP="$PREFIX/lib/python3.8/site-packages"
"$PREFIX/bin/pip" install --no-deps --force-reinstall \
  bokeh==1.4.0 panel==0.8.0 selenium==3.141.0 medpy==0.4.0 SimpleITK==2.1.1
patch_markup() {   # $1 file, $2 md5 of the server's patched copy
  local f="$SP/$1"
  sed -i 's/^from jinja2 import Environment, Markup, FileSystemLoader$/from jinja2 import Environment, FileSystemLoader\nfrom markupsafe import Markup/' "$f"
  local got; got=$(md5sum "$f" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "ERROR: $1 md5 $got != server's $2" >&2; exit 1; }
  echo "patched $1 (identical to the server's copy)"
}
patch_markup bokeh/core/templates.py 025af6b4fbd4e28640b7e076964f770d
patch_markup panel/io/resources.py   5ac7867d1edf5e4fc15d5904041a15d9
"$PREFIX/bin/python" -c "import bokeh, panel, holoviews; print('bokeh', bokeh.__version__, '| panel', panel.__version__, '| holoviews', holoviews.__version__)"

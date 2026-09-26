"""Set a mouse's cross-registration output aside before re-running the crossreg notebook.

The cross-registration notebook saves ``mappings_<name>.pkl``/``.csv``, ``cents_<name>.pkl``
and ``shiftds_<name>.nc`` into the mouse folder, where ``<name>`` comes from its
``f_pattern_prefix`` and ``f_pattern`` (e.g. ``crossreg_7``). This renames the existing
ones to ``-ORIG`` names (``mappings_crossreg_7-ORIG.csv``, ...), which ``caban`` keeps
reading. Refuses if anything is already set aside.
See ``caban.session_queue.set_aside_crossreg_output``.

    cd ~/code/caban && ~/miniforge3/envs/caban/bin/python -m scripts.set_aside_crossreg_output \\
        /Volumes/MINISCOPE/SSTCa2/G10-ST703_hM3D crossreg_7 --reason "crossreg re-run"
"""

import argparse
import json

from caban.session_queue import set_aside_crossreg_output

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("mouse_dir", help="the mouse folder holding mappings_<name>.* files")
parser.add_argument("name", help="the crossreg output name, e.g. crossreg_7")
parser.add_argument("--reason", required=True, help="why it is being re-run; recorded")
args = parser.parse_args()
print(json.dumps(set_aside_crossreg_output(args.mouse_dir, args.name, args.reason), indent=2))

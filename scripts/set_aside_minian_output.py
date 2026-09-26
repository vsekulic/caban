"""Prepare a session's Miniscope/ folder for a run of the Minian pipeline notebook.

Renames existing notebook output (``minian/``, ``minian_intermediate/``, ``minian.mp4``,
``minian_mc.mp4``) to ``-ORIG`` names, so the notebook, pointed at that same folder,
writes fresh output without touching the original. Refuses if anything is already set
aside. With ``--scratch``, also makes ``minian_intermediate`` a symlink to a fresh folder
on a fast drive -- the notebook is unchanged, the heavy intermediates land there.
See ``caban.session_queue.set_aside_minian_output``.

    cd ~/code/caban && ~/miniforge3/envs/caban/bin/python -m scripts.set_aside_minian_output \\
        <.../Miniscope> --reason "gate re-run" [--scratch /Volumes/FUTROLA/minian_scratch]
"""

import argparse
import json

from caban.session_queue import set_aside_minian_output

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("session_dir", help="the session's Miniscope/ folder")
parser.add_argument("--reason", required=True, help="why it is being re-run; recorded")
parser.add_argument("--scratch", help="root folder on a fast drive for minian_intermediate")
args = parser.parse_args()
print(json.dumps(set_aside_minian_output(args.session_dir, args.reason, args.scratch), indent=2))

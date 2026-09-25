"""Set a session's existing Minian output aside before re-running the pipeline notebook.

Renames ``minian/`` and ``minian_intermediate/`` in a session's ``Miniscope/`` folder to
``minian-ORIG/`` and ``minian_intermediate-ORIG/``, so the notebook, pointed at that
same folder, writes fresh output without touching the original. Refuses if anything is
already set aside. See ``caban.session_queue.set_aside_minian_output``.

    cd ~/code/caban && python -m scripts.set_aside_minian_output <.../Miniscope> --reason "gate re-run"
"""

import argparse
import json

from caban.session_queue import set_aside_minian_output

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("session_dir", help="the session's Miniscope/ folder")
parser.add_argument("--reason", required=True, help="why it is being re-run; recorded")
args = parser.parse_args()
print(json.dumps(set_aside_minian_output(args.session_dir, args.reason), indent=2))

"""Run the homecage pre/post-CNO analysis (caban.homecage_cno) without the dataset, e.g. on the Razer.

    python scripts/run_homecage_cno.py <export_dir> <plots_dir>

<export_dir> holds the output of scripts/export_tfc_day_homecage_s.py; results go to
<plots_dir>/homecage_cno/.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban.loader import MOUSE_GROUPS, mice_per_group_from
from caban.homecage_cno import run_homecage_cno

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run_homecage_cno(sys.argv[2], mice_per_group_from(MOUSE_GROUPS), export_dir=sys.argv[1])

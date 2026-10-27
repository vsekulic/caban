"""One full Minian run of a mouse's parameter-check session (HC2) with the decided SST set, on the
Razer (plan §6.1): sst_dio_validation.DECIDED_OVERRIDES (the pyramidal set), the session's lens
mask (session_queue.SEED_MASKS, else sst_dio_common.SEED_MASKS_PROPOSED) and its line-artefact
frames (yra_recompute.NOTEBOOK_DEL_FRAMES). Written outside the session folder, as the grids.

    python scripts/sst_dio_final_set_check.py <mouse> <data_root> <out_root> [<variant> <overrides_json>]

Output: <out_root>/<mouse>_HC2/decided_set/ (the run, its variant.json and the runner's summary_*.png).
With <variant> and <overrides_json> (e.g. decided_set_diff_thres_2 '{"param_seeds_init": {"diff_thres": 2}}'),
the decided set with those values changed, into <out_root>/<mouse>_HC2/<variant>/.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import sst_dio_validation as sdv

if __name__ == '__main__':
    if len(sys.argv) not in (4, 6):
        raise SystemExit(__doc__)
    mouse, data_root, out_root = sys.argv[1:4]
    variant, extra = (sys.argv[4], json.loads(sys.argv[5])) if len(sys.argv) == 6 else ('decided_set', {})
    overrides = sdv.merged_overrides(sdv.DECIDED_OVERRIDES, extra)
    session_dir = os.path.join(sdv.parameter_check_session_dir(data_root, mouse), 'Miniscope')
    tail = sq.session_tail(session_dir)
    seed_mask = sq.SEED_MASKS.get(tail, sdv.SEED_MASKS_PROPOSED.get(tail))
    if seed_mask is None:
        raise SystemExit('no seed mask for {}'.format(tail))
    root = os.path.join(out_root, '{}_HC2'.format(mouse))
    os.makedirs(root, exist_ok=True)
    print('{}: {} {}, seed mask {}'.format(tail, variant, overrides, seed_mask), flush=True)
    record = sdv.run_parameter_variant(session_dir, variant, overrides, root, seed_mask)
    print(record['report'])

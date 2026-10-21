"""Raw-video projections of the four SST DIO gate sessions (HC2 of the conditioning day), on the Razer.

    python scripts/sst_dio_gate_projections.py <data_root> <out_dir>

Writes <out_dir>/gate_projections.png (rows G22, G23, G26, G27; mean | max - mean | s.d.) and one
.npz per mouse with the arrays. plans/sst_dio_dreadd_validation_plan.md §6.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban.sst_dio_validation import (SST_DIO_DISPLAY_ORDER, gate_session_dir, raw_projections,
                                      plot_projection_grid)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data_root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    projections = {}
    for mouse in SST_DIO_DISPLAY_ORDER:
        session_dir = gate_session_dir(data_root, mouse)
        print('{}: {}'.format(mouse, session_dir), flush=True)
        projections[mouse] = raw_projections(session_dir)
        np.savez_compressed(os.path.join(out_dir, '{}_HC2_projections.npz'.format(mouse)),
                            **{k: v for k, v in projections[mouse].items()})
    print(plot_projection_grid(projections, os.path.join(out_dir, 'gate_projections.png')))

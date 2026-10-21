"""One stage of the staged Minian parameter walk on an SST DIO session, on the Razer (plan §6).

    python scripts/sst_dio_parameter_stage.py <mouse> <stage> <data_root> <out_root> [<n_parallel>]

Runs sst_dio_validation.STAGE_GRIDS[<stage>] on the mouse's parameter-check session (HC2),
each variant with DECIDED_OVERRIDES (the earlier stages' decisions) and stopping after the stage;
then the stage's measurements on the reference cells of the one-at-a-time grid
(<out_root>/../parameter_grid/<mouse>_HC2/<REFERENCE_UNITS_VARIANT>) and its figures. Output:
<out_root>/<mouse>_HC2/<stage>/.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import sst_dio_validation as sdv

if __name__ == '__main__':
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    mouse, stage, data_root, out_root = sys.argv[1:5]
    n_parallel = int(sys.argv[5]) if len(sys.argv) == 6 else 4
    if stage not in sdv.STAGE_GRIDS:
        raise SystemExit('no grid for stage {!r}; built: {}'.format(stage, list(sdv.STAGE_GRIDS)))
    session_dir = os.path.join(sdv.parameter_check_session_dir(data_root, mouse), 'Miniscope')
    tail = sq.session_tail(session_dir)
    seed_mask = sq.SEED_MASKS.get(tail, sdv.SEED_MASKS_PROPOSED.get(tail))
    if seed_mask is None:
        raise SystemExit('no seed mask for {}'.format(tail))
    stage_root = os.path.join(out_root, '{}_HC2'.format(mouse), stage)
    os.makedirs(stage_root, exist_ok=True)
    grid = {name: sdv.merged_overrides(sdv.DECIDED_OVERRIDES, overrides)
            for name, overrides in sdv.STAGE_GRIDS[stage].items()}
    # The seed mask acts on seeds, so it is applied from the seeds stage on (the preprocessing
    # runs stop before seeds_init); the measurements use it to define background pixels.
    run_mask = None if stage == 'preprocessing' else seed_mask
    print('{} {}: {} variants, decided so far {}'.format(tail, stage, len(grid), sdv.DECIDED_OVERRIDES),
          flush=True)
    sdv.run_parameter_grid(session_dir, grid, stage_root, run_mask, n_parallel,
                           cut_after=sdv.STAGE_CUT_AFTER[stage])

    cells = sdv.reference_cells(os.path.join(os.path.dirname(os.path.normpath(out_root)),
                                             'parameter_grid', '{}_HC2'.format(mouse)))
    rows = []
    for name in grid:
        row = {'variant': name}
        if stage == 'preprocessing':
            row.update(ksize=grid[name]['param_denoise']['ksize'],
                       wnd=grid[name]['param_background_removal']['wnd'])
            row.update(sdv.preprocessing_metrics(os.path.join(stage_root, name), cells, seed_mask))
        rows.append(row)
        print(row, flush=True)
    metrics = pd.DataFrame(rows).set_index('variant')
    metrics.to_csv(os.path.join(stage_root, 'stage_metrics.csv'))
    if stage == 'preprocessing':
        sdv.plot_preprocessing_grid(stage_root, os.path.join(stage_root, 'max_proj_grid.png'))
        sdv.plot_preprocessing_metrics(metrics, os.path.join(stage_root, 'stage_metrics.png'))
    print(metrics.to_string())

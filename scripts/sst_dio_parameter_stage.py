"""One stage of the staged Minian parameter walk on an SST DIO session, on the Razer (plan §6).

    python scripts/sst_dio_parameter_stage.py <mouse> <stage> <data_root> <out_root> [<n_parallel> [<variant> ...]]

Runs sst_dio_validation.STAGE_GRIDS[<stage>] (or only the named variants, e.g. re-running a failed
one after moving its folder into failed_attempts/) on the mouse's parameter-check session (HC2),
each variant with DECIDED_OVERRIDES (the earlier stages' decisions) and stopping after the stage.
Then the stage's measurements on every variant whose run finished ('done' in its variant.json),
on the reference cells of the one-at-a-time grid
(<out_root>/../parameter_grid/<mouse>_HC2/<REFERENCE_UNITS_VARIANT>), and its figures. A failed
variant is not measured: it is listed, by name, in the output and in stage_metrics.csv.
Output: <out_root>/<mouse>_HC2/<stage>/.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import sst_dio_validation as sdv

if __name__ == '__main__':
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    mouse, stage, data_root, out_root = sys.argv[1:5]
    n_parallel = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    if stage not in sdv.STAGE_GRIDS:
        raise SystemExit('no grid for stage {!r}; built: {}'.format(stage, list(sdv.STAGE_GRIDS)))
    full_grid = {name: sdv.merged_overrides(sdv.DECIDED_OVERRIDES, overrides)
                 for name, overrides in sdv.STAGE_GRIDS[stage].items()}
    names = sys.argv[6:] or list(full_grid)
    unknown = [n for n in names if n not in full_grid]
    if unknown:
        raise SystemExit('not in the {} grid: {}'.format(stage, unknown))
    session_dir = os.path.join(sdv.parameter_check_session_dir(data_root, mouse), 'Miniscope')
    tail = sq.session_tail(session_dir)
    seed_mask = sq.SEED_MASKS.get(tail, sdv.SEED_MASKS_PROPOSED.get(tail))
    if seed_mask is None:
        raise SystemExit('no seed mask for {}'.format(tail))
    stage_root = os.path.join(out_root, '{}_HC2'.format(mouse), stage)
    os.makedirs(stage_root, exist_ok=True)
    # The seed mask acts on seeds, so it is applied from the seeds stage on (the preprocessing
    # runs stop before seeds_init); the measurements use it to define background pixels.
    run_mask = None if stage == 'preprocessing' else seed_mask
    print('{} {}: running {} of {} variants, decided so far {}'.format(
        tail, stage, len(names), len(full_grid), sdv.DECIDED_OVERRIDES), flush=True)
    sdv.run_parameter_grid(session_dir, {n: full_grid[n] for n in names}, stage_root, run_mask,
                           n_parallel, cut_after=sdv.STAGE_CUT_AFTER[stage])

    cells = sdv.reference_cells(os.path.join(os.path.dirname(os.path.normpath(out_root)),
                                             'parameter_grid', '{}_HC2'.format(mouse)))
    rows = []
    for name, overrides in full_grid.items():
        record_path = os.path.join(stage_root, name, 'variant.json')
        if not os.path.isfile(record_path):
            raise FileNotFoundError('{} has not been run: {}'.format(name, record_path))
        with open(record_path) as fh:
            status = json.load(fh)['status']
        row = {'variant': name, 'status': status}
        if stage == 'preprocessing':
            row.update(ksize=overrides['param_denoise']['ksize'],
                       wnd=overrides['param_background_removal']['wnd'])
        elif stage == 'cnmf':
            row.update(first_merge=overrides['param_first_merge']['thres_corr'],
                       dl_wnd=overrides['param_first_spatial']['dl_wnd'])
        if status == 'done':
            if stage == 'preprocessing':
                row.update(sdv.preprocessing_metrics(os.path.join(stage_root, name), cells, seed_mask))
            elif stage == 'cnmf':
                row.update(sdv.cnmf_metrics(os.path.join(stage_root, name), cells))
        else:
            print('NOT MEASURED (run {}): {}'.format(status, name), flush=True)
        rows.append(row)
    metrics = pd.DataFrame(rows).set_index('variant')
    metrics.to_csv(os.path.join(stage_root, 'stage_metrics.csv'))
    done = metrics[metrics['status'] == 'done']
    if stage == 'preprocessing':
        sdv.plot_preprocessing_grid(stage_root, set(done.index), os.path.join(stage_root, 'max_proj_grid.png'))
        sdv.plot_preprocessing_metrics(done, os.path.join(stage_root, 'stage_metrics.png'))
    elif stage == 'cnmf':
        sdv.plot_cnmf_metrics(done, os.path.join(stage_root, 'stage_metrics.png'))
    print(metrics.to_string())

"""Stage 3 of the staged parameter walk: initialisation and its merge (plan §6.1).

Runs in Minian's own env. From stage 1's decided movies and stage 2's decided seeds, Minian's
initialisation as the pipeline notebook calls it (cells 149-159): ``initA(Y_hw_chk, seeds,
**param_initialize)`` -> ``initC(Y_fm_chk, A_init)`` -> ``unit_merge(A_init, C_init,
**param_init_merge)``. ``initA``/``initC`` run once per (initialise thres_corr, wnd); the merge per
init-merge threshold. Each variant's merged A and C are saved as ``<variant>/minian/{A,C}.zarr``
with the max projection and a ``variant.json``, so the review module's ``compare`` draws them as it
draws the grid's.

Measured on the reference cells: recall (cells with a unit centroid within MATCH_PX), cells holding
two or more units (split), units whose footprint covers two or more reference centroids (fused),
unmatched units, footprint size.
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from minian.cnmf import unit_merge
from minian.initialization import initA, initC
from minian.utilities import save_minian

from caban.sst_dio_common import MATCH_PX, footprint_centroids, nearest_distance

INIT_THRES_CORRS = (0.7, 0.8, 0.9)
INIT_WNDS = (10, 15)
INIT_MERGE_THRES_CORRS = (0.6, 0.7, 0.8, 0.9)
FUSED_LEVEL = 0.3     # a reference centroid inside a unit's footprint above this fraction of its peak


def variant_name(init_corr, init_wnd, merge_corr):
    return 'init_corr_{}_wnd_{}_merge_{}'.format(str(init_corr).replace('.', '_'), init_wnd,
                                                 str(merge_corr).replace('.', '_'))


def decided_seeds(seed_table_path):
    """Stage 2's final seeds (the merged step of the decided seed variant)."""
    table = pd.read_csv(seed_table_path)
    seeds = table[table['step'] == 'merged'].reset_index(drop=True)
    if len(seeds) == 0:
        raise ValueError('{}: no merged seeds'.format(seed_table_path))
    return seeds


def run_init_grid(arrays, params, seeds, out_dir, intpath, init_corrs=INIT_THRES_CORRS, init_wnds=INIT_WNDS,
                  merge_corrs=INIT_MERGE_THRES_CORRS):
    """Every (initialise thres_corr, wnd, init-merge thres_corr); saves each variant's merged A, C.
    ``intpath``: where A_init and C_init are saved between the steps (overwritten per pair)."""
    Y_fm_chk, Y_hw_chk, max_proj = arrays['Y_fm_chk'], arrays['Y_hw_chk'], arrays['max_proj']
    names = []
    for init_corr in init_corrs:
        for init_wnd in init_wnds:
            init_params = dict(params['param_initialize'], thres_corr=init_corr, wnd=init_wnd)
            # As the notebook (cells 149, 153): saved and reopened, so unit_merge gets dask arrays.
            A_init = save_minian(initA(Y_hw_chk, seeds, **init_params).rename('A_init'), intpath,
                                 overwrite=True)
            C_init = save_minian(initC(Y_fm_chk, A_init).rename('C_init'), intpath, overwrite=True,
                                 chunks={'unit_id': 1, 'frame': -1})
            print('initialise corr {} wnd {}: {} units'.format(init_corr, init_wnd, A_init.sizes['unit_id']),
                  flush=True)
            for merge_corr in merge_corrs:
                name = variant_name(init_corr, init_wnd, merge_corr)
                merge_params = dict(params['param_init_merge'], thres_corr=merge_corr)
                A, C = unit_merge(A_init, C_init, **merge_params)
                folder = os.path.join(out_dir, name, 'minian')
                os.makedirs(folder)
                A.compute().rename('A').to_dataset().to_zarr(os.path.join(folder, 'A.zarr'))
                C.compute().rename('C').to_dataset().to_zarr(os.path.join(folder, 'C.zarr'))
                max_proj.rename('max_proj').to_dataset().to_zarr(os.path.join(folder, 'max_proj.zarr'))
                with open(os.path.join(out_dir, name, 'variant.json'), 'w') as fh:
                    json.dump({'variant': name, 'status': 'done', 'timings': {'total_s': 0},
                               'overrides': {'param_initialize': {'thres_corr': init_corr, 'wnd': init_wnd},
                                             'param_init_merge': {'thres_corr': merge_corr}},
                               'n_seeds': len(seeds), 'n_init_units': int(A_init.sizes['unit_id'])}, fh, indent=1)
                names.append(name)
                print('  {}: {} units after merging'.format(name, A.sizes['unit_id']), flush=True)
    return names


def init_metrics(out_dir, name, cells):
    """Recall, split cells, fused units, unmatched units and footprint size of one variant."""
    A = xr.open_zarr(os.path.join(out_dir, name, 'minian', 'A.zarr'))['A'].compute()
    height, width = footprint_centroids(A)
    dist, which = nearest_distance(height, width, cells[0], cells[1])
    matched = dist <= MATCH_PX
    per_cell = np.bincount(which[matched], minlength=len(cells[0]))
    values = A.values.reshape(A.sizes['unit_id'], -1)
    scaled = values / values.max(axis=1, keepdims=True)
    cell_pixels = (np.round(cells[0]).astype(int) * A.sizes['width'] + np.round(cells[1]).astype(int))
    covered = (scaled[:, cell_pixels] > FUSED_LEVEL).sum(axis=1)
    return {'n_units': int(A.sizes['unit_id']), 'recall': float((per_cell > 0).mean()),
            'cells_with_two_or_more': int((per_cell >= 2).sum()),
            'fused_units': int((covered >= 2).sum()), 'unmatched_units': int((~matched).sum()),
            'footprint_px_median': float(np.median((values > 0).sum(axis=1)))}


def plot_init_metrics(metrics, out_path):
    """Each measurement against the init-merge threshold, one line per (initialise corr, wnd)."""
    columns = ('n_units', 'recall', 'cells_with_two_or_more', 'fused_units', 'unmatched_units')
    fig, axes = plt.subplots(1, len(columns), figsize=(4.0 * len(columns), 3.4))
    for ax, column in zip(axes, columns):
        for (init_corr, init_wnd), part in metrics.groupby(['init_corr', 'init_wnd']):
            part = part.sort_values('merge_corr')
            ax.plot(part['merge_corr'], part[column], marker='o',
                    label='init corr {}, wnd {}'.format(init_corr, init_wnd))
        ax.set_xlabel('init-merge thres_corr')
        ax.set_title(column, fontsize=10)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path

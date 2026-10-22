"""Stage 2 of the staged parameter walk: seeds (plans/sst_dio_dreadd_validation_plan.md §6.1).

Runs in Minian's own env (``minian-native``). Starts from the decided stage-1 run's saved movies
(``Y_fm_chk``, ``Y_hw_chk``, ``max_proj``) and calls Minian's seed functions exactly as the pipeline
notebook does (cells 109-142: ``seeds_init`` -> lens mask -> ``pnr_refine`` -> ``ks_refine`` ->
``seeds_merge``), on a LocalCluster set up as the notebook's (cell 23), with every other parameter
taken from that run's ``parameters.json``. Production then runs the chosen values through the
notebook itself; this only explores.

Shared work: ``seeds_init`` once per ``max_wnd``; ``pnr_refine`` computes each seed's PNR and, for a
numeric ``thres``, sets ``mask_pnr = pnr > thres`` (minian/initialization.py), so the PNR is computed
once per ``max_wnd`` and every threshold is applied to the same values; ``ks_refine`` is per seed and
independent of the threshold; ``seeds_merge`` runs per (``max_wnd``, threshold).

Measured against the reference cells (the one-at-a-time grid's merge 0.7 run; positions only):
recall (reference cells with a seed within MATCH_PX), seeds per matched cell (> 1: a cell may be
split), and unmatched seeds (new candidate cells or noise -- VS judges them on the images).
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from distributed import Client, LocalCluster
from minian.initialization import ks_refine, pnr_refine, seeds_init, seeds_merge
from minian.utilities import TaskAnnotation

from caban.sst_dio_common import MATCH_PX, footprint_centroids, inside_ellipse, nearest_distance

SEED_MAX_WNDS = (10, 15, 20)
SEED_PNR_THRESHOLDS = (0.75, 1.0, 1.25, 1.5)
SEED_STEPS = ('initial', 'inside_mask', 'pnr_and_ks', 'merged')
STEP_COLOURS = {'initial': '#ffd92f', 'inside_mask': '#1f77b4', 'pnr_and_ks': '#ff7f0e', 'merged': '#d62728'}
CROP_HALF_PX = 20
FRAMERATE_HZ = 19.76     # sessions.MINISCOPE_FPS, for the trace time axis only


def start_cluster(n_workers):
    """The pipeline notebook's cluster (cell 23), dashboard on a free port."""
    cluster = LocalCluster(n_workers=n_workers, memory_limit='32GB', resources={'MEM': 1},
                           threads_per_worker=1, dashboard_address=':0')
    cluster.scheduler.add_plugin(TaskAnnotation())
    return Client(cluster)


def open_stage1(stage1_dir):
    """The decided stage-1 run's movies and max projection, and its kernel-recorded parameters."""
    arrays = {name: xr.open_zarr(os.path.join(stage1_dir, 'minian_intermediate', name + '.zarr'))[name]
              for name in ('Y_fm_chk', 'Y_hw_chk')}
    arrays['max_proj'] = xr.open_zarr(os.path.join(stage1_dir, 'minian', 'max_proj.zarr'))['max_proj'].compute()
    with open(os.path.join(stage1_dir, 'parameters.json')) as fh:
        params = json.load(fh)['param']
    return arrays, params


def reference_cells(reference_dir):
    """(height, width) centroids of the reference run's units."""
    A = xr.open_zarr(os.path.join(reference_dir, 'minian', 'A.zarr'))['A'].compute()
    return footprint_centroids(A)


def run_seed_grid(stage1_dir, out_dir, seed_mask, max_wnds=SEED_MAX_WNDS, thresholds=SEED_PNR_THRESHOLDS):
    """Every (max_wnd, PNR threshold); writes ``seeds_<variant>.csv`` (all seeds with each step's
    flag) per variant, and returns them by variant name."""
    arrays, params = open_stage1(stage1_dir)
    Y_fm_chk, Y_hw_chk, max_proj = arrays['Y_fm_chk'], arrays['Y_hw_chk'], arrays['max_proj']
    out = {}
    for max_wnd in max_wnds:
        seeds = seeds_init(Y_fm_chk, **dict(params['param_seeds_init'], max_wnd=max_wnd))
        inside = inside_ellipse(seeds['height'], seeds['width'], seed_mask)
        masked = seeds[inside].reset_index(drop=True)
        print('max_wnd {}: {} seeds, {} inside the lens mask'.format(max_wnd, len(seeds), len(masked)), flush=True)
        refined, pnr, _ = pnr_refine(Y_hw_chk, masked.copy(), **params['param_pnr_refine'])
        refined = ks_refine(Y_hw_chk, refined, **params['param_ks_refine'])
        refined['pnr'] = pnr.to_pandas()
        for thres in thresholds:
            name = 'max_wnd_{}_pnr_{}'.format(max_wnd, str(thres).replace('.', '_'))
            seeds_t = refined.copy()
            seeds_t['mask_pnr'] = (pnr > thres).to_pandas()
            kept = seeds_t[seeds_t['mask_ks'] & seeds_t['mask_pnr']].reset_index(drop=True)
            if len(kept) == 0:
                raise ValueError('{}: no seed passes PNR and KS'.format(name))
            merged = seeds_merge(Y_hw_chk, max_proj, kept, **params['param_seeds_merge'])
            table = pd.concat([
                seeds.assign(step='initial'),
                masked.assign(step='inside_mask'),
                kept.assign(step='pnr_and_ks'),
                merged[merged['mask_mrg']].assign(step='merged'),
            ], ignore_index=True)
            table.to_csv(os.path.join(out_dir, 'seeds_{}.csv'.format(name)), index=False)
            out[name] = table
            print('  {}: {} after PNR and KS, {} after merging'.format(
                name, len(kept), int(merged['mask_mrg'].sum())), flush=True)
    return out, max_proj


def seed_metrics(table, cells):
    """Per step: seed count; for the final seeds, recall of the reference cells, seeds per matched
    cell, and unmatched seeds."""
    row = {'n_{}'.format(step): int((table['step'] == step).sum()) for step in SEED_STEPS}
    final = table[table['step'] == 'merged']
    dist, which = nearest_distance(final['height'], final['width'], cells[0], cells[1])
    matched = dist <= MATCH_PX
    per_cell = np.bincount(which[matched], minlength=len(cells[0]))
    row.update(recall=float((per_cell > 0).mean()),
               seeds_per_matched_cell=float(per_cell[per_cell > 0].mean()),
               cells_with_two_or_more=int((per_cell >= 2).sum()),
               unmatched_seeds=int((~matched).sum()))
    return row


def plot_seed_steps(table, max_proj, cells, title, out_path):
    """Seeds after each step over the max projection; reference cells circled in the last panel."""
    fig, axes = plt.subplots(1, len(SEED_STEPS), figsize=(5.2 * len(SEED_STEPS), 5.6))
    low, high = np.percentile(max_proj, (0.5, 99.5))
    for ax, step in zip(axes, SEED_STEPS):
        part = table[table['step'] == step]
        ax.imshow(max_proj, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        ax.scatter(part['width'], part['height'], s=7, color=STEP_COLOURS[step], lw=0)
        ax.set_title('{}: {} seeds'.format(step.replace('_', ' '), len(part)), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].scatter(cells[1], cells[0], s=60, facecolor='none', edgecolor='#2ca02c', lw=0.8)
    fig.suptitle('{} (green circles: reference cells)'.format(title), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_seed_metrics(metrics, out_path):
    """Recall, seeds per matched cell, cells with >= 2 seeds and unmatched seeds against the PNR
    threshold, one line per max_wnd."""
    columns = ('recall', 'seeds_per_matched_cell', 'cells_with_two_or_more', 'unmatched_seeds', 'n_merged')
    fig, axes = plt.subplots(1, len(columns), figsize=(4.0 * len(columns), 3.4))
    for ax, column in zip(axes, columns):
        for max_wnd in sorted(metrics['max_wnd'].unique()):
            part = metrics[metrics['max_wnd'] == max_wnd].sort_values('pnr_thres')
            ax.plot(part['pnr_thres'], part[column], marker='o', label='max_wnd {}'.format(max_wnd))
        ax.set_xlabel('PNR threshold')
        ax.set_title(column, fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_unmatched_seeds(table, Y_hw_chk, max_proj, cells, title, out_path):
    """Final seeds over the max projection, matched to a reference cell (green) or not (red,
    numbered); then, per unmatched seed, a crop of the max projection and its Y_hw_chk trace with
    its PNR -- to judge whether it is a faint cell the reference lacks, or noise."""
    final = table[table['step'] == 'merged'].reset_index(drop=True)
    dist, _ = nearest_distance(final['height'], final['width'], cells[0], cells[1])
    unmatched = final[dist > MATCH_PX].reset_index(drop=True)
    n = len(unmatched)
    fig = plt.figure(figsize=(16, 7 + 1.7 * n))
    grid = fig.add_gridspec(1 + n, 4, height_ratios=[7] + [1.7] * n)
    ax = fig.add_subplot(grid[0, :2])
    low, high = np.percentile(max_proj, (0.5, 99.5))
    ax.imshow(max_proj, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
    matched = final[dist <= MATCH_PX]
    ax.scatter(matched['width'], matched['height'], s=14, color='#2ca02c', lw=0)
    ax.scatter(unmatched['width'], unmatched['height'], s=14, color='#d62728', lw=0)
    for i, row in unmatched.iterrows():
        ax.annotate(str(i + 1), (row['width'], row['height']), color='#d62728', fontsize=9,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_title('{}: {} final seeds, {} on reference cells (green), {} not (red)'.format(
        title, len(final), len(matched), n), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for i, row in unmatched.iterrows():
        h, w = int(row['height']), int(row['width'])
        crop = max_proj[max(h - CROP_HALF_PX, 0):h + CROP_HALF_PX, max(w - CROP_HALF_PX, 0):w + CROP_HALF_PX]
        ax_crop = fig.add_subplot(grid[1 + i, 0])
        ax_crop.imshow(crop, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        ax_crop.plot(min(w, CROP_HALF_PX), min(h, CROP_HALF_PX), '+', color='#d62728', ms=8)
        ax_crop.set_xticks([])
        ax_crop.set_yticks([])
        ax_crop.set_ylabel(str(i + 1), rotation=0, fontsize=10, labelpad=12)
        trace = Y_hw_chk.sel(height=row['height'], width=row['width']).values
        ax_trace = fig.add_subplot(grid[1 + i, 1:])
        ax_trace.plot(np.arange(len(trace)) / FRAMERATE_HZ, trace, lw=0.4, color='0.2')
        ax_trace.set_title('seed {}: PNR {:.2f}'.format(i + 1, row['pnr']), fontsize=9, loc='left')
        ax_trace.set_xlim(0, len(trace) / FRAMERATE_HZ)
    if n:
        ax_trace.set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path

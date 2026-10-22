"""Review of the SST DIO Minian parameter grids, for VS in Jupyter on the Razer (plan §6).

Runs in the ``minian-native`` env (Notebook 6, ipywidgets, Minian's own viewers), so it imports
only what that env has: no caban module that needs papermill or the caban env. Notebook:
``notebooks/sst_dio_parameter_review.ipynb``. Each comparison drawn is also saved as a PNG under
``<grid_root>/review_snapshots/``, so Claude sees what VS sees; decisions go to
``<grid_root>/parameter_log.csv`` through :func:`log_decision`.
"""

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

FOOTPRINT_LEVEL = 0.3          # a unit's outline: where its footprint exceeds this fraction of its peak
SNAPSHOT_DIR = 'review_snapshots'
LOG_NAME = 'parameter_log.csv'
LOG_FIELDS = ('time', 'stage', 'values_tried', 'value_chosen', 'why')
# Outline colours (RGBA): the reference run, the variant, and pixels on both outlines (VS, 2026-10-22).
ORANGE = (1.0, 0.6, 0.0, 1.0)
BLUE = (0.2, 0.6, 1.0, 1.0)
PURPLE = (0.75, 0.2, 0.95, 1.0)


def variants(grid_root):
    """``variant.json`` of every variant, by name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(grid_root, '*', 'variant.json'))):
        with open(path) as fh:
            record = json.load(fh)
        out[record['variant']] = record
    return out


def summary_table(grid_root):
    """One row per variant: what changed, status, units, footprint size, close pairs, minutes."""
    rows = []
    for name, r in variants(grid_root).items():
        report = r.get('report', {})
        rows.append({'variant': name,
                     'changed': '; '.join('{} {}'.format(p, v) for p, v in r['overrides'].items()) or '—',
                     'status': r['status'],
                     'units': report.get('n_units'),
                     'footprint_px_median': report.get('footprint_px', {}).get('median'),
                     'close_pairs': report.get('close_pairs'),
                     'minutes': round(r['timings']['total_s'] / 60, 1),
                     'error': r.get('error', '')[:80]})
    return pd.DataFrame(rows).set_index('variant')


def open_array(grid_root, variant, name, intermediate=False):
    """One saved array of a variant: final output (``minian/``) or an intermediate."""
    folder = 'minian_intermediate' if intermediate else 'minian'
    path = os.path.join(grid_root, variant, folder, name + '.zarr')
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return xr.open_zarr(path)[name]


def unit_outlines(A, level=FOOTPRINT_LEVEL):
    """Boolean image of all units' outlines: each pixel labelled by the unit whose footprint,
    scaled to its own peak, is largest there (if above ``level``); boundaries between labels."""
    values = A.values.reshape(A.sizes['unit_id'], -1)
    scaled = values / values.max(axis=1, keepdims=True)
    label = np.where(scaled.max(axis=0) > level, scaled.argmax(axis=0) + 1, 0)
    return find_boundaries(label.reshape(A.sizes['height'], A.sizes['width']), mode='inner')


def _show(ax, max_proj, outlines, colour, title):
    low, high = np.percentile(max_proj, (0.5, 99.5))
    ax.imshow(max_proj, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
    overlay = np.zeros(outlines.shape + (4,))
    overlay[outlines] = colour
    ax.imshow(overlay, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def compare(grid_root, variant, reference='frozen', save=True):
    """``reference`` and ``variant`` side by side (outlines over each one's max_proj), and both
    outlines over the variant's max_proj; saved to ``review_snapshots/<variant>.png``."""
    records = variants(grid_root)
    panels = []
    for name in (reference, variant):
        if records[name]['status'] != 'done':
            raise ValueError('{} did not finish: {}'.format(name, records[name].get('error')))
        A = open_array(grid_root, name, 'A').compute()
        panels.append((name, open_array(grid_root, name, 'max_proj').values, unit_outlines(A),
                       A.sizes['unit_id']))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    _show(axes[0], panels[0][1], panels[0][2], ORANGE, '{}: {} units'.format(panels[0][0], panels[0][3]))
    _show(axes[1], panels[1][1], panels[1][2], BLUE, '{}: {} units'.format(panels[1][0], panels[1][3]))
    reference_only = panels[0][2] & ~panels[1][2]
    variant_only = panels[1][2] & ~panels[0][2]
    in_both = panels[0][2] & panels[1][2]
    _show(axes[2], panels[1][1], reference_only, ORANGE,
          'orange: {} only · blue: {} only · purple: both'.format(reference, variant))
    overlay = np.zeros(in_both.shape + (4,))
    overlay[variant_only] = BLUE
    overlay[in_both] = PURPLE
    axes[2].imshow(overlay, interpolation='nearest')
    fig.tight_layout()
    if save:
        os.makedirs(os.path.join(grid_root, SNAPSHOT_DIR), exist_ok=True)
        fig.savefig(os.path.join(grid_root, SNAPSHOT_DIR, '{}_vs_{}.png'.format(variant, reference)),
                    dpi=110)
    return fig


def open_stage_array(stage_root, variant, name, intermediate=False):
    """One saved array of a staged-walk variant (same layout as the grid's)."""
    return open_array(stage_root, variant, name, intermediate=intermediate)


def compare_preprocessing(stage_root, variant, reference='ksize_5_wnd_15', crop=((180, 480), (150, 450)),
                          save=True):
    """Max projections (after background removal and motion correction) of ``reference`` and
    ``variant``: full field and a crop, same display range per pair; saved as a snapshot."""
    images = {name: open_stage_array(stage_root, name, 'max_proj').values for name in (reference, variant)}
    low = min(np.percentile(im, 0.5) for im in images.values())
    high = max(np.percentile(im, 99.5) for im in images.values())
    (h0, h1), (w0, w1) = crop
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for col, name in enumerate((reference, variant)):
        for row, image in enumerate((images[name], images[name][h0:h1, w0:w1])):
            ax = axes[row, col]
            ax.imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{}{}'.format(name, '' if row == 0 else ' (crop)'), fontsize=10)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.join(stage_root, SNAPSHOT_DIR), exist_ok=True)
        fig.savefig(os.path.join(stage_root, SNAPSHOT_DIR, '{}_vs_{}.png'.format(variant, reference)), dpi=90)
    return fig


def log_decision(grid_root, stage, values_tried, value_chosen, why):
    """Append one decision to ``parameter_log.csv`` (the parameter log of plan §6)."""
    path = os.path.join(grid_root, LOG_NAME)
    new = not os.path.isfile(path)
    with open(path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({'time': pd.Timestamp.now().isoformat(timespec='seconds'), 'stage': stage,
                         'values_tried': values_tried, 'value_chosen': value_chosen, 'why': why})
    return pd.read_csv(path)

"""Shared, dependency-light pieces of the SST DIO validation (numpy only), importable from both the
caban env (caban.sst_dio_validation) and Minian's own env (caban.sst_dio_seed_grid,
caban.sst_dio_parameter_review). plans/sst_dio_dreadd_validation_plan.md §6.
"""

import numpy as np

# Lens-interior ellipses proposed for the parameter check, before VS approves them into
# session_queue.SEED_MASKS (which production runs read). Drawn over the baseline max_proj.
SEED_MASKS_PROPOSED = {
    'G27-ST895-hM4D-DIOGC/2023_05_22-TFC_cond/15_21_41-HC2': {
        'center_height': 335.0, 'center_width': 300.0, 'radius_height': 265.0, 'radius_width': 285.0},
}
MATCH_PX = 6.0     # a seed within this distance of a reference cell's centroid belongs to that cell


def inside_ellipse(height, width, mask):
    """True where (height, width) lies inside the ellipse ``mask`` (minian_runner.SEED_MASK_KEYS)."""
    return (((np.asarray(height) - mask['center_height']) / mask['radius_height']) ** 2
            + ((np.asarray(width) - mask['center_width']) / mask['radius_width']) ** 2) <= 1


def footprint_centroids(A):
    """(height, width) centroid of each unit's footprint, weighted by footprint value."""
    values = A.values.reshape(A.sizes['unit_id'], -1)
    hh, ww = np.meshgrid(A.coords['height'].values, A.coords['width'].values, indexing='ij')
    weights = values / values.sum(axis=1, keepdims=True)
    return weights @ hh.ravel(), weights @ ww.ravel()


def nearest_distance(points_h, points_w, cells_h, cells_w):
    """For each point, the distance to the nearest cell centroid, and that cell's index."""
    dist = np.sqrt((np.asarray(points_h)[:, None] - np.asarray(cells_h)[None]) ** 2
                   + (np.asarray(points_w)[:, None] - np.asarray(cells_w)[None]) ** 2)
    return dist.min(axis=1), dist.argmin(axis=1)

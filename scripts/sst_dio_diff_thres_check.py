"""seeds_init's diff_thres (3 = the pyramidal set; Minian's default 2; 1) on SST DIO sessions,
Minian's env on the Razer (plan §6.1). G22's clearest missed cells had no seed at all: their
transients are 2-4 raw counts, short of diff_thres 3.

    python scripts/sst_dio_diff_thres_check.py <out_root> <n_workers> <mouse>=<run_dir> ...

<run_dir> holds a run's saved movies, max projection and parameters (a decided-set check run, or
stage 1's decided preprocessing run). Seed steps at max_wnd 15, PNR 1.0, diff_thres 3/2/1. Per
mouse: the cell-like spots of the max projection (seed_grid.candidate_spots), how many get a
merged seed, and how many merged seeds sit on no spot; <out_root>/<mouse>_HC2/diff_thres.csv and
diff_thres.png.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser('~/code/minian_vsekulic'))
from caban.sst_dio_common import MATCH_PX, SEED_MASKS_PROPOSED, nearest_distance
from caban import sst_dio_seed_grid as sg

DIFF_THRES_VALUES = (3, 2, 1)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out_root, n_workers, pairs = sys.argv[1], int(sys.argv[2]), [a.split('=', 1) for a in sys.argv[3:]]
    client = sg.start_cluster(n_workers)
    for mouse, run_dir in pairs:
        tail = [k for k in SEED_MASKS_PROPOSED if k.startswith(mouse + '-') and k.endswith('-HC2')]
        if len(tail) != 1:
            raise ValueError('expected one HC2 seed mask for {}, found {}'.format(mouse, tail))
        seed_mask = SEED_MASKS_PROPOSED[tail[0]]
        out_dir = os.path.join(out_root, '{}_HC2'.format(mouse))
        os.makedirs(out_dir)
        os.environ['MINIAN_INTERMEDIATE'] = os.path.join(out_dir, 'minian_intermediate')
        os.makedirs(os.environ['MINIAN_INTERMEDIATE'])
        tables, max_proj = sg.run_seed_grid(run_dir, out_dir, seed_mask, max_wnds=(15,), thresholds=(1.0,),
                                            diff_thres_values=DIFF_THRES_VALUES)
        spots = sg.candidate_spots(max_proj.values, seed_mask)
        rows = []
        fig, axes = plt.subplots(1, len(DIFF_THRES_VALUES), figsize=(6.2 * len(DIFF_THRES_VALUES), 6.6))
        low, high = np.percentile(max_proj.values, (0.5, 99.5))
        for ax, diff in zip(axes, DIFF_THRES_VALUES):
            table = tables['diff_{}_max_wnd_15_pnr_1_0'.format(diff)]
            final = table[table['step'] == 'merged']
            dist, _ = nearest_distance(final['height'], final['width'], spots[0], spots[1])
            on_spot = dist <= MATCH_PX
            spot_dist, _ = nearest_distance(spots[0], spots[1], final['height'], final['width'])
            seeded = spot_dist <= MATCH_PX
            rows.append({'diff_thres': diff, 'n_initial': int((table['step'] == 'initial').sum()),
                         'n_merged_seeds': len(final), 'spots': len(spots[0]), 'spots_seeded': int(seeded.sum()),
                         'seeds_on_no_spot': int((~on_spot).sum())})
            ax.imshow(max_proj.values, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            ax.scatter(spots[1][seeded], spots[0][seeded], s=60, facecolor='none', edgecolor='#2ca02c', lw=1)
            ax.scatter(spots[1][~seeded], spots[0][~seeded], s=60, facecolor='none', edgecolor='#d62728', lw=1)
            ax.scatter(final['width'][~on_spot], final['height'][~on_spot], s=10, marker='x', color='#ffd92f', lw=0.8)
            ax.set_title('diff_thres {}: {}/{} spots seeded (green; red: not), {} seeds on no spot (x)'.format(
                diff, int(seeded.sum()), len(spots[0]), int((~on_spot).sum())), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('{} HC2: merged seeds at max_wnd 15, PNR 1.0, by diff_thres'.format(mouse), fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'diff_thres.png'), dpi=90)
        plt.close(fig)
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(out_dir, 'diff_thres.csv'), index=False)
        print(mouse, table.to_string(), flush=True)
    client.close()

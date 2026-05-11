"""
caban.TFC.py — Python conversion of TFC.m / TFC_load_data.m
Tone Fear Conditioning analysis: freeze score data loading, assembly, plotting, and statistics.
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from itertools import combinations

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
saveit = True
for_paper = False
use_test = 'anova'
post_hoc = 'bonferroni'

if for_paper:
    my_linewidth = 0.2
    my_markersize = 1
    font_size = 5
    my_sz = 12
else:
    my_linewidth = 1
    my_markersize = 6
    font_size = 18
    my_sz = 42

# Colors (from my_colours.m)
my_r = np.array([1, 0, 0])
my_b = np.array([0, 0, 1])
my_k = np.array([0, 0, 0])
my_h = np.array([0.5, 0.5, 0.5])
my_light_grey = np.array([0.8, 0.8, 0.8])

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = font_size

file_path = r"C:\Users\vlads\Dropbox\1-McHugh postdoc\3-PAPER\paper_plots\fig1\plots\TFC"

# ─────────────────────────────────────────────
# Time bin arrays
# ─────────────────────────────────────────────
onset = np.arange(0, 820, 20)           # 41 bins
onset_full = np.arange(0, 900, 20)      # 45 bins
onset_1wk = np.arange(0, 900, 20)       # 45 bins
onset_3min = np.arange(0, 180, 20)      # 9 bins
onset_1min = np.arange(0, 80, 20)       # 4 bins
onset_2min = np.arange(0, 140, 20)      # 7 bins
onset_2ndh = np.arange(80, 180, 20)     # 5 bins
onset_tfc = np.arange(0, 1300, 20)      # 65 bins
onset_testA = np.arange(0, 360, 20)     # 18 bins
onset_testA_1wk = np.arange(0, 360, 20) # 18 bins

# Sound/shock event times
sound_onsets = np.array([180, 420, 660, 900, 1140])
sound_offsets = sound_onsets + 20
shock_onsets = sound_offsets + 20
shock_offsets = shock_onsets + 2
sound_onsets_test = sound_onsets[:3]
sound_offsets_test = sound_offsets[:3]

# ─────────────────────────────────────────────
# Period index calculations (0-based)
# ─────────────────────────────────────────────
tone = []
tone_post_tone = []
post_tone_only = []
first_3min = []

for i in range(len(sound_onsets)):
    idx = np.where(onset == sound_onsets[i])[0]
    if len(idx) > 0:
        tone.append(int(idx[0]))

    start_idx = np.where(onset >= sound_onsets[i])[0]
    if i + 1 < len(sound_onsets):
        end_idx = np.where(onset < sound_onsets[i + 1])[0]
        valid = np.intersect1d(start_idx, end_idx)
    else:
        valid = start_idx
    tone_post_tone.extend(valid.tolist())

    # post_tone_only: bins after sound_onsets[0] and after sound_onsets[i], before next sound
    mask = (onset > sound_onsets[0]) & (onset > sound_onsets[i])
    if i + 1 < len(sound_onsets):
        mask &= (onset < sound_onsets[i + 1])
    post_tone_only.extend(np.where(mask)[0].tolist())

tone = np.array(tone)
tone_post_tone = np.array(sorted(set(tone_post_tone)))
post_tone_only = np.array(sorted(set(post_tone_only)))
post_tone_20s = tone + 1

for val in onset_3min:
    idx = np.where(onset == val)[0]
    if len(idx) > 0:
        first_3min.append(idx[0])
first_3min = np.array(first_3min)

first_min = ' 3min'

# ─────────────────────────────────────────────
# Load JSON data
# ─────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'freeze_data')

with open(os.path.join(data_dir, 'FSTE.json'), 'r') as f:
    fste_data = json.load(f)
with open(os.path.join(data_dir, 'FSTI.json'), 'r') as f:
    fsti_data = json.load(f)
with open(os.path.join(data_dir, 'FSTC.json'), 'r') as f:
    fstc_data = json.load(f)
with open(os.path.join(data_dir, 'FSTH.json'), 'r') as f:
    fsth_data = json.load(f)


def _a(key, d):
    """Convert a JSON key to numpy array."""
    return np.array(d[key])

# ─────────────────────────────────────────────
# Assemble _tot arrays
# ─────────────────────────────────────────────

# --- FSTE ---
# FSTE_tot = FSTE1 rows [0,2,3]; FSTE2 row [1]; FSTE3 cols 0:41; FSTE4 cols 0:41
FSTE_tot = np.vstack([
    _a('FSTE1', fste_data)[[0, 2, 3], :],
    _a('FSTE2', fste_data)[[1], :],
    _a('FSTE3', fste_data)[:, :41],
    _a('FSTE4', fste_data)[:, :41],
])

FSTE_1wk_tot = np.vstack([
    _a('FSTE1_1wk', fste_data)[[0, 2, 3], :],
    _a('FSTE2_1wk', fste_data)[[1], :],
    _a('FSTE3_1wk', fste_data),
    _a('FSTE4_1wk', fste_data),
])

FSTE_TFC_tot = np.vstack([
    _a('FSTE1_TFC', fste_data)[[0, 2, 3], :],
    _a('FSTE2_TFC', fste_data)[[1], :],
    _a('FSTE3_TFC', fste_data),
    _a('FSTE4_TFC', fste_data),
])

FSTE_TestA_tot = np.vstack([
    _a('FSTE1_TestA', fste_data)[[0, 2, 3], :],
    _a('FSTE2_TestA', fste_data)[[1], :],
    _a('FSTE3_TestA', fste_data),
    _a('FSTE4_TestA', fste_data),
])

FSTE_TestA_1wk_tot = np.vstack([
    _a('FSTE1_TestA_1wk', fste_data)[[0, 2, 3], :],
    _a('FSTE2_TestA_1wk', fste_data)[[1], :],
    _a('FSTE3_TestA_1wk', fste_data),
    _a('FSTE4_TestA_1wk', fste_data),
])

# --- FSTI ---
# FSTI_tot = FSTI1 all; FSTI2 row [1]; FSTI5 all; FSTI6 cols 0:41
FSTI_tot = np.vstack([
    _a('FSTI1', fsti_data),
    _a('FSTI2', fsti_data)[[1], :],
    _a('FSTI5', fsti_data),
    _a('FSTI6', fsti_data)[:, :41],
])

FSTI_1wk_tot = np.vstack([
    _a('FSTI1_1wk', fsti_data),
    _a('FSTI2_1wk', fsti_data)[[1], :],
    _a('FSTI6_1wk', fsti_data),
])

FSTI_TFC_tot = np.vstack([
    _a('FSTI1_TFC', fsti_data),
    _a('FSTI2_TFC', fsti_data)[[1], :],
    _a('FSTI6_TFC', fsti_data),
])

FSTI_TestA_tot = np.vstack([
    _a('FSTI1_TestA', fsti_data),
    _a('FSTI2_TestA', fsti_data)[[1], :],
    _a('FSTI5_TestA', fsti_data),
    _a('FSTI6_TestA', fsti_data),
])

FSTI_TestA_1wk_tot = np.vstack([
    _a('FSTI1_TestA_1wk', fsti_data),
    _a('FSTI2_TestA_1wk', fsti_data)[[1], :],
    _a('FSTI6_TestA_1wk', fsti_data),
])

# --- FSTC ---
# FSTC_tot = FSTC1 cols 0:41; FSTC2 cols 0:41; FSTC3 cols 0:41; FSTC4 cols 0:41
FSTC_tot = np.vstack([
    _a('FSTC1', fstc_data)[:, :41],
    _a('FSTC2', fstc_data)[:, :41],
    _a('FSTC3', fstc_data)[:, :41],
    _a('FSTC4', fstc_data)[:, :41],
])

FSTC_1wk_tot = np.vstack([
    _a('FSTC1_1wk', fstc_data),
    _a('FSTC2_1wk', fstc_data),
    _a('FSTC3_1wk', fstc_data),
    _a('FSTC4_1wk', fstc_data),
])

FSTC_TFC_tot = np.vstack([
    _a('FSTC1_TFC', fstc_data),
    _a('FSTC2_TFC', fstc_data),
    _a('FSTC3_TFC', fstc_data),
    _a('FSTC4_TFC', fstc_data),
])

FSTC_TestA_tot = np.vstack([
    _a('FSTC1_TestA', fstc_data),
    _a('FSTC2_TestA', fstc_data),
    _a('FSTC3_TestA', fstc_data),
    _a('FSTC4_TestA', fstc_data),
])

FSTC_TestA_1wk_tot = np.vstack([
    _a('FSTC1_TestA_1wk', fstc_data),
    _a('FSTC2_TestA_1wk', fstc_data),
    _a('FSTC3_TestA_1wk', fstc_data),
    _a('FSTC4_TestA_1wk', fstc_data),
])

# --- FSTH (HC) ---
FSTE_HC_TFC_tot = np.array(fsth_data['FSTH1_TFC']) if fsth_data['FSTH1_TFC'] else np.empty((0, len(onset_tfc)))


# ─────────────────────────────────────────────
# Statistical helper functions
# ─────────────────────────────────────────────

def do_stats_tfc(data_E, data_I, data_C, test, post_hoc_method, offset):
    """
    Between-groups statistical test for 3 groups (E, I, C).
    Returns (groups, pvalues) for pairwise comparisons.
    Groups are bar position pairs (with offset applied).
    """
    pairs = [(data_I, data_C), (data_E, data_I), (data_E, data_C)]
    group_pairs = [
        [2 + offset, 3 + offset],
        [1 + offset, 2 + offset],
        [1 + offset, 3 + offset],
    ]
    pvals = []

    if test == 'anova':
        F_stat, p_omnibus = sp_stats.f_oneway(data_E, data_I, data_C)
        for a, b in pairs:
            _, p = sp_stats.ttest_ind(a, b)
            # Bonferroni correction
            if post_hoc_method == 'bonferroni':
                p = min(p * 3, 1.0)
            pvals.append(p)
    elif test == 'kw':
        H, p_omnibus = sp_stats.kruskal(data_E, data_I, data_C)
        for a, b in pairs:
            _, p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
            if post_hoc_method == 'bonferroni':
                p = min(p * 3, 1.0)
            pvals.append(p)
    elif test == 'ranksum':
        for a, b in pairs:
            _, p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
            pvals.append(p)
    elif test == 'ttest2':
        for a, b in pairs:
            _, p = sp_stats.ttest_ind(a, b)
            pvals.append(p)
    else:
        raise ValueError(f"Unknown test: {test}")

    return group_pairs, pvals


def _ttest_paired_or_ind(a, b):
    """Paired t-test if same length, otherwise unpaired."""
    if len(a) == len(b):
        _, p = sp_stats.ttest_rel(a, b)
    else:
        _, p = sp_stats.ttest_ind(a, b)
    return p


def stats_ranova(data_list, post_hoc_method, offset):
    """
    Repeated-measures within-group comparison (3 conditions).
    Uses paired t-tests with optional Bonferroni correction.
    Returns (groups, pvalues).
    """
    n_cond = len(data_list)
    n_pairs = n_cond * (n_cond - 1) // 2
    group_pairs = []
    pvals = []
    for i, j in combinations(range(n_cond), 2):
        _, p = sp_stats.ttest_rel(data_list[i], data_list[j])
        if post_hoc_method == 'bonferroni':
            p = min(p * n_pairs, 1.0)
        pvals.append(p)
        group_pairs.append([i + 1 + offset, j + 1 + offset])
    return group_pairs, pvals


def sigstar_text(ax, groups, pvals, y_start=None, sep=2.5):
    """
    Add significance stars/text above bars.
    """
    if y_start is None:
        y_start = ax.get_ylim()[1] * 0.85
    y = y_start
    for (g1, g2), p in zip(groups, pvals):
        if p < 0.001:
            txt = '***'
        elif p < 0.01:
            txt = '**'
        elif p < 0.05:
            txt = '*'
        else:
            txt = 'n.s.'
        ax.plot([g1, g1, g2, g2], [y, y + sep * 0.3, y + sep * 0.3, y], 'k-', linewidth=0.8)
        ax.text((g1 + g2) / 2, y + sep * 0.35, txt, ha='center', va='bottom', fontsize=8)
        y += sep


def _save(fig, name):
    if saveit:
        os.makedirs(file_path, exist_ok=True)
        fig.savefig(os.path.join(file_path, f'{name}.svg'), format='svg')
        fig.savefig(os.path.join(file_path, f'{name}.png'), format='png', dpi=300)
        print(f"Saved {name}")


def _setup_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(my_linewidth)


def _shade_tones(ax, ymax, use_tfc=False):
    """Draw shaded tone regions on a binned freezing plot."""
    onsets_to_use = sound_onsets if use_tfc else sound_onsets_test
    for s_on, s_off in zip(onsets_to_use, sound_offsets[:len(onsets_to_use)]):
        ax.fill_between([s_on, s_off], 0, ymax, color=my_light_grey, alpha=0.1, linewidth=0)
    if use_tfc:
        for sh_on, sh_off in zip(shock_onsets, shock_offsets):
            ax.fill_between([sh_on, sh_off], 0, ymax, color=my_r, alpha=0.5, linewidth=0)


def _time_ticks(ax, onset_arr):
    """Set x-ticks in minute increments (every 2 min)."""
    ticks = np.arange(60, np.floor(onset_arr.max()) + 1, 120)
    labels = [str(int(t)) for t in np.arange(1, np.floor(onset_arr.max() / 60) + 1, 2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels[:len(ticks)])


def _mean_sem(data, axis=0):
    m = np.mean(data, axis=axis)
    s = np.std(data, axis=axis, ddof=0) / np.sqrt(data.shape[axis])
    return m, s


def _period_stats(data, indices):
    """Per-animal mean across given column indices, then group mean/std/sem."""
    per_animal = np.mean(data[:, indices], axis=1)
    m = np.mean(per_animal)
    sd = np.std(per_animal, ddof=0)
    se = sd / np.sqrt(len(per_animal))
    return per_animal, m, sd, se


# ─────────────────────────────────────────────
# Binned freezing line plot helper
# ─────────────────────────────────────────────

def plot_binned_freezing(onset_arr, data_dict, title_str, file_str, ylim_max,
                          use_tfc=False, paper_size=None):
    """
    Generic binned freezing time-series plot.
    data_dict: {'label': (data_2d, color), ...}
    """
    if paper_size is None:
        paper_size = (1.25, 0.75) if for_paper else (6, 4)

    fig, ax = plt.subplots(figsize=paper_size)
    _shade_tones(ax, ylim_max, use_tfc=use_tfc)

    for label, (data, color) in data_dict.items():
        m, s = _mean_sem(data)
        ax.errorbar(onset_arr, m, yerr=s, fmt='o-', color=color,
                     markerfacecolor=color, linewidth=my_linewidth,
                     markersize=my_markersize, capsize=0, label=label)

    _time_ticks(ax, onset_arr)
    ax.set_xlim([0, onset_arr.max()])
    ax.set_ylim([0, ylim_max])
    _setup_axes(ax)

    if not for_paper:
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Freezing (%)')
        ax.set_title(title_str)

    fig.tight_layout()
    _save(fig, file_str)
    return fig


# ═════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# Plot 1a: Binned freezing TFC conditioning
# ─────────────────────────────────────────────
plot_binned_freezing(
    onset_tfc,
    {'SST Inh': (FSTI_TFC_tot, my_b),
     'SST Exc': (FSTE_TFC_tot, my_r),
     'SST Ctl': (FSTC_TFC_tot, my_k)},
    'Conditioning', '1a_TFC_binned_freezing', 85, use_tfc=True,
)

# ─────────────────────────────────────────────
# Plot 1a_HC: Binned freezing TFC-HC (empty / placeholder)
# ─────────────────────────────────────────────
if FSTE_HC_TFC_tot.shape[0] > 0:
    plot_binned_freezing(
        onset_tfc,
        {'SST Inh': (FSTI_TFC_tot, my_b),
         'SST Exc': (FSTE_TFC_tot, my_r),
         'SST Ctl': (FSTC_TFC_tot, my_k)},
        'Conditioning', '1a_TFC_binned_freezing_HC', 85, use_tfc=True,
    )

# ─────────────────────────────────────────────
# Plot 2a: Binned freezing 48hr test
# ─────────────────────────────────────────────
plot_binned_freezing(
    onset,
    {'SST Inh': (FSTI_tot, my_b),
     'SST Exc': (FSTE_tot, my_r),
     'SST Ctl': (FSTC_tot, my_k)},
    'Test 48hr', '2a_48hr_binned_freezing', 70,
    paper_size=(3, 2) if for_paper else (6, 4),
)

# ─────────────────────────────────────────────
# Plot 3a: Binned freezing 1wk test
# ─────────────────────────────────────────────
plot_binned_freezing(
    onset_1wk,
    {'SST Inh': (FSTI_1wk_tot, my_b),
     'SST Exc': (FSTE_1wk_tot, my_r),
     'SST Ctl': (FSTC_1wk_tot, my_k)},
    'Test 1wk', '3a_1wk_binned_freezing', 70,
    paper_size=(3, 2) if for_paper else (6, 4),
)


# ─────────────────────────────────────────────
# Calculate period-averaged freeze scores (48hr)
# ─────────────────────────────────────────────

tone_FSTI, mean_tone_FSTI, std_tone_FSTI, sem_tone_FSTI = _period_stats(FSTI_tot, tone)
tone_FSTE, mean_tone_FSTE, std_tone_FSTE, sem_tone_FSTE = _period_stats(FSTE_tot, tone)
tone_FSTC, mean_tone_FSTC, std_tone_FSTC, sem_tone_FSTC = _period_stats(FSTC_tot, tone)

_, mean_tpt_FSTI, _, sem_tpt_FSTI = _period_stats(FSTI_tot, tone_post_tone)
_, mean_tpt_FSTE, _, sem_tpt_FSTE = _period_stats(FSTE_tot, tone_post_tone)
_, mean_tpt_FSTC, _, sem_tpt_FSTC = _period_stats(FSTC_tot, tone_post_tone)

post_tone_only_FSTI, mean_pto_FSTI, std_pto_FSTI, sem_pto_FSTI = _period_stats(FSTI_tot, post_tone_only)
post_tone_only_FSTE, mean_pto_FSTE, std_pto_FSTE, sem_pto_FSTE = _period_stats(FSTE_tot, post_tone_only)
post_tone_only_FSTC, mean_pto_FSTC, std_pto_FSTC, sem_pto_FSTC = _period_stats(FSTC_tot, post_tone_only)

first_3min_FSTI, mean_f3_FSTI, std_f3_FSTI, sem_f3_FSTI = _period_stats(FSTI_tot, first_3min)
first_3min_FSTE, mean_f3_FSTE, std_f3_FSTE, sem_f3_FSTE = _period_stats(FSTE_tot, first_3min)
first_3min_FSTC, mean_f3_FSTC, std_f3_FSTC, sem_f3_FSTC = _period_stats(FSTC_tot, first_3min)

_, mean_pt20_FSTI, _, sem_pt20_FSTI = _period_stats(FSTI_tot, post_tone_20s)
_, mean_pt20_FSTE, _, sem_pt20_FSTE = _period_stats(FSTE_tot, post_tone_20s)
_, mean_pt20_FSTC, _, sem_pt20_FSTC = _period_stats(FSTC_tot, post_tone_20s)


# ─────────────────────────────────────────────
# Plot 4a: 48hr period-averaged freezing (tone + post-tone only)
# ─────────────────────────────────────────────

def plot_period_bars(means_left, sems_left, means_right, sems_right,
                     data_left, data_right,
                     title_str, file_str,
                     xlabels_left=None, xlabels_right=None,
                     paper_size=None, offset_right=4):
    if paper_size is None:
        paper_size = (2, 2) if for_paper else (4, 4)

    fig, ax = plt.subplots(figsize=paper_size)

    positions_l = [1, 2, 3]
    positions_r = [1 + offset_right, 2 + offset_right, 3 + offset_right]
    colors = [my_r, my_b, my_k]

    # Left group
    for pos, m, se, c in zip(positions_l, means_left, sems_left, colors):
        ax.bar(pos, m, width=1.0, color=c)
        ax.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

    G_l, P_l = do_stats_tfc(data_left[0], data_left[1], data_left[2], use_test, post_hoc, 0)
    max_sem_l = max(sems_left)
    sigstar_text(ax, G_l, P_l, y_start=max(means_left) + max_sem_l + 3)

    # Right group
    for pos, m, se, c in zip(positions_r, means_right, sems_right, colors):
        ax.bar(pos, m, width=1.0, color=c)
        ax.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

    G_r, P_r = do_stats_tfc(data_right[0], data_right[1], data_right[2], use_test, post_hoc, offset_right)
    max_sem_r = max(sems_right)
    sigstar_text(ax, G_r, P_r, y_start=max(means_right) + max_sem_r + 3)

    all_pos = positions_l + positions_r
    if xlabels_left is None:
        xlabels_left = ['Exc', 'Inh', 'Ctl']
    if xlabels_right is None:
        xlabels_right = ['Exc', 'Inh', 'Ctl']
    ax.set_xticks(all_pos)
    ax.set_xticklabels(xlabels_left + xlabels_right, rotation=-45, ha='left')
    ax.set_ylabel('Freezing (%)')
    ax.set_title(title_str)
    _setup_axes(ax)
    fig.tight_layout()
    _save(fig, file_str)
    return fig


fig_4a = plot_period_bars(
    [mean_tone_FSTE, mean_tone_FSTI, mean_tone_FSTC],
    [sem_tone_FSTE, sem_tone_FSTI, sem_tone_FSTC],
    [mean_pto_FSTE, mean_pto_FSTI, mean_pto_FSTC],
    [sem_pto_FSTE, sem_pto_FSTI, sem_pto_FSTC],
    data_left=[
        np.mean(FSTE_tot[:, tone], axis=1),
        np.mean(FSTI_tot[:, tone], axis=1),
        np.mean(FSTC_tot[:, tone], axis=1),
    ],
    data_right=[
        np.mean(FSTE_tot[:, post_tone_only], axis=1),
        np.mean(FSTI_tot[:, post_tone_only], axis=1),
        np.mean(FSTC_tot[:, post_tone_only], axis=1),
    ],
    title_str='Test 48hr',
    file_str='4a_48hr_period_freezing',
)

# ─────────────────────────────────────────────
# Plot 4b: 48hr period freezing with first_3min
# ─────────────────────────────────────────────

fig_4b, ax_4b = plt.subplots(figsize=(3, 2) if for_paper else (6, 4))

# Tone bars
for pos, m, se, c in zip([1, 2, 3],
                          [mean_tone_FSTE, mean_tone_FSTI, mean_tone_FSTC],
                          [sem_tone_FSTE, sem_tone_FSTI, sem_tone_FSTC],
                          [my_r, my_b, my_k]):
    ax_4b.bar(pos, m, width=1.0, color=c)
    ax_4b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

G, P = do_stats_tfc(np.mean(FSTE_tot[:, tone], axis=1),
                     np.mean(FSTI_tot[:, tone], axis=1),
                     np.mean(FSTC_tot[:, tone], axis=1), use_test, post_hoc, 0)
sigstar_text(ax_4b, G, P, y_start=max(mean_tone_FSTE, mean_tone_FSTI, mean_tone_FSTC) + max(sem_tone_FSTE, sem_tone_FSTI, sem_tone_FSTC) + 3)

# Post-tone-only bars
for pos, m, se, c in zip([5, 6, 7],
                          [mean_pto_FSTE, mean_pto_FSTI, mean_pto_FSTC],
                          [sem_pto_FSTE, sem_pto_FSTI, sem_pto_FSTC],
                          [my_r, my_b, my_k]):
    ax_4b.bar(pos, m, width=1.0, color=c)
    ax_4b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

G, P = do_stats_tfc(np.mean(FSTE_tot[:, post_tone_only], axis=1),
                     np.mean(FSTI_tot[:, post_tone_only], axis=1),
                     np.mean(FSTC_tot[:, post_tone_only], axis=1), use_test, post_hoc, 4)
sigstar_text(ax_4b, G, P, y_start=max(mean_pto_FSTE, mean_pto_FSTI, mean_pto_FSTC) + max(sem_pto_FSTE, sem_pto_FSTI, sem_pto_FSTC) + 3)

# First 3min bars
for pos, m, se, c in zip([9, 10, 11],
                          [mean_f3_FSTE, mean_f3_FSTI, mean_f3_FSTC],
                          [sem_f3_FSTE, sem_f3_FSTI, sem_f3_FSTC],
                          [my_r, my_b, my_k]):
    ax_4b.bar(pos, m, width=1.0, color=c)
    ax_4b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

G, P = do_stats_tfc(np.mean(FSTE_tot[:, first_3min], axis=1),
                     np.mean(FSTI_tot[:, first_3min], axis=1),
                     np.mean(FSTC_tot[:, first_3min], axis=1), use_test, post_hoc, 8)
sigstar_text(ax_4b, G, P, y_start=max(mean_f3_FSTE, mean_f3_FSTI, mean_f3_FSTC) + max(sem_f3_FSTE, sem_f3_FSTI, sem_f3_FSTC) + 3)

ax_4b.set_xticks(range(1, 12))
ax_4b.set_xticklabels(['Tone', 'Tone', 'Tone', '', 'Post tone', 'Post tone', 'Post tone', '',
                        f'First{first_min}', f'First{first_min}', f'First{first_min}'], rotation=-45, ha='left')
ax_4b.set_title('Test 48hr')
_setup_axes(ax_4b)
fig_4b.tight_layout()
_save(fig_4b, '4b_48hr_period_freezing_first_3min')

# ─────────────────────────────────────────────
# Plot 4c: 48hr within-groups comparison
# ─────────────────────────────────────────────

fig_4c, ax_4c = plt.subplots(figsize=(3, 2) if for_paper else (6, 4))

for grp_offset, color, scatter_c, tot_data, f3, tn, pto, label in [
    (0, my_r, my_k, FSTE_tot, first_3min_FSTE, tone_FSTE, post_tone_only_FSTE, 'Exc'),
    (4, my_b, my_k, FSTI_tot, first_3min_FSTI, tone_FSTI, post_tone_only_FSTI, 'Inh'),
    (8, my_k, my_h, FSTC_tot, first_3min_FSTC, tone_FSTC, post_tone_only_FSTC, 'Ctl'),
]:
    means = [np.mean(f3), np.mean(tn), np.mean(pto)]
    sems = [np.std(f3, ddof=0) / np.sqrt(len(f3)),
            np.std(tn, ddof=0) / np.sqrt(len(tn)),
            np.std(pto, ddof=0) / np.sqrt(len(pto))]
    positions = [1 + grp_offset, 2 + grp_offset, 3 + grp_offset]

    for pos, m, se in zip(positions, means, sems):
        ax_4c.bar(pos, m, width=1.0, color=color)
        ax_4c.errorbar(pos, m, yerr=se, fmt='none', ecolor=color, capsize=0, linewidth=my_linewidth)

    # Individual data points + paired lines
    for pos, vals in zip(positions, [f3, tn, pto]):
        ax_4c.scatter(np.full(len(vals), pos), vals, s=my_sz, color=scatter_c, zorder=3)
    for i in range(len(f3)):
        ax_4c.plot([positions[0], positions[1]], [f3[i], tn[i]], color=scatter_c, linewidth=0.5)
        ax_4c.plot([positions[1], positions[2]], [tn[i], pto[i]], color=scatter_c, linewidth=0.5)

    # Repeated-measures stats
    data_rm = [
        np.mean(tot_data[:, first_3min], axis=1),
        np.mean(tot_data[:, tone], axis=1),
        np.mean(tot_data[:, post_tone_only], axis=1),
    ]
    G, P = stats_ranova(data_rm, post_hoc, grp_offset)
    max_sem = max(sems)
    sigstar_text(ax_4c, G, P, y_start=max(means) + max_sem + 3)

ax_4c.set_xticks(range(1, 12))
ax_4c.set_xticklabels([f'First{first_min}', 'Tone', 'Post tone', '',
                        f'First{first_min}', 'Tone', 'Post tone', '',
                        f'First{first_min}', 'Tone', 'Post tone'], rotation=-45, ha='left')
ax_4c.set_ylabel('Freezing (%)')
ax_4c.set_title('Test 48hr - within groups')
_setup_axes(ax_4c)
fig_4c.tight_layout()
_save(fig_4c, '4c_48hr_period_freezing_first_3min_comparison')


# ─────────────────────────────────────────────
# Calculate period-averaged freeze scores (1wk)
# ─────────────────────────────────────────────

tone_FSTI_1wk, mean_tone_FSTI_1wk, std_tone_FSTI_1wk, sem_tone_FSTI_1wk = _period_stats(FSTI_1wk_tot, tone)
tone_FSTE_1wk, mean_tone_FSTE_1wk, std_tone_FSTE_1wk, sem_tone_FSTE_1wk = _period_stats(FSTE_1wk_tot, tone)
tone_FSTC_1wk, mean_tone_FSTC_1wk, std_tone_FSTC_1wk, sem_tone_FSTC_1wk = _period_stats(FSTC_1wk_tot, tone)

_, mean_tpt_FSTI_1wk, _, sem_tpt_FSTI_1wk = _period_stats(FSTI_1wk_tot, tone_post_tone)
_, mean_tpt_FSTE_1wk, _, sem_tpt_FSTE_1wk = _period_stats(FSTE_1wk_tot, tone_post_tone)
_, mean_tpt_FSTC_1wk, _, sem_tpt_FSTC_1wk = _period_stats(FSTC_1wk_tot, tone_post_tone)

post_tone_only_FSTI_1wk, mean_pto_FSTI_1wk, std_pto_FSTI_1wk, sem_pto_FSTI_1wk = _period_stats(FSTI_1wk_tot, post_tone_only)
post_tone_only_FSTE_1wk, mean_pto_FSTE_1wk, std_pto_FSTE_1wk, sem_pto_FSTE_1wk = _period_stats(FSTE_1wk_tot, post_tone_only)
post_tone_only_FSTC_1wk, mean_pto_FSTC_1wk, std_pto_FSTC_1wk, sem_pto_FSTC_1wk = _period_stats(FSTC_1wk_tot, post_tone_only)

_, mean_pt20_FSTI_1wk, _, sem_pt20_FSTI_1wk = _period_stats(FSTI_1wk_tot, post_tone_20s)
_, mean_pt20_FSTE_1wk, _, sem_pt20_FSTE_1wk = _period_stats(FSTE_1wk_tot, post_tone_20s)
_, mean_pt20_FSTC_1wk, _, sem_pt20_FSTC_1wk = _period_stats(FSTC_1wk_tot, post_tone_20s)

first_3min_FSTI_1wk, mean_f3_FSTI_1wk, std_f3_FSTI_1wk, sem_f3_FSTI_1wk = _period_stats(FSTI_1wk_tot, first_3min)
first_3min_FSTE_1wk, mean_f3_FSTE_1wk, std_f3_FSTE_1wk, sem_f3_FSTE_1wk = _period_stats(FSTE_1wk_tot, first_3min)
first_3min_FSTC_1wk, mean_f3_FSTC_1wk, std_f3_FSTC_1wk, sem_f3_FSTC_1wk = _period_stats(FSTC_1wk_tot, first_3min)


# ─────────────────────────────────────────────
# Plot 5a: 1wk period-averaged freezing (tone + post-tone only)
# ─────────────────────────────────────────────

fig_5a = plot_period_bars(
    [mean_tone_FSTE_1wk, mean_tone_FSTI_1wk, mean_tone_FSTC_1wk],
    [sem_tone_FSTE_1wk, sem_tone_FSTI_1wk, sem_tone_FSTC_1wk],
    [mean_pto_FSTE_1wk, mean_pto_FSTI_1wk, mean_pto_FSTC_1wk],
    [sem_pto_FSTE_1wk, sem_pto_FSTI_1wk, sem_pto_FSTC_1wk],
    data_left=[
        np.mean(FSTE_1wk_tot[:, tone], axis=1),
        np.mean(FSTI_1wk_tot[:, tone], axis=1),
        np.mean(FSTC_1wk_tot[:, tone], axis=1),
    ],
    data_right=[
        np.mean(FSTE_1wk_tot[:, post_tone_only], axis=1),
        np.mean(FSTI_1wk_tot[:, post_tone_only], axis=1),
        np.mean(FSTC_1wk_tot[:, post_tone_only], axis=1),
    ],
    title_str='Test 1wk',
    file_str='5a_1wk_period_freezing',
)

# ─────────────────────────────────────────────
# Plot 5b: 1wk period freezing with first_3min
# ─────────────────────────────────────────────

fig_5b, ax_5b = plt.subplots(figsize=(3, 2) if for_paper else (6, 4))

# Tone bars
for pos, m, se, c in zip([1, 2, 3],
                          [mean_tone_FSTE_1wk, mean_tone_FSTI_1wk, mean_tone_FSTC_1wk],
                          [sem_tone_FSTE_1wk, sem_tone_FSTI_1wk, sem_tone_FSTC_1wk],
                          [my_r, my_b, my_k]):
    ax_5b.bar(pos, m, width=1.0, color=c)
    ax_5b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

G, P = do_stats_tfc(np.mean(FSTE_1wk_tot[:, tone], axis=1),
                     np.mean(FSTI_1wk_tot[:, tone], axis=1),
                     np.mean(FSTC_1wk_tot[:, tone], axis=1), use_test, post_hoc, 0)
sigstar_text(ax_5b, G, P, y_start=max(mean_tone_FSTE_1wk, mean_tone_FSTI_1wk, mean_tone_FSTC_1wk) +
             max(sem_tone_FSTE_1wk, sem_tone_FSTI_1wk, sem_tone_FSTC_1wk) + 3)

# Post-tone-only bars
for pos, m, se, c in zip([5, 6, 7],
                          [mean_pto_FSTE_1wk, mean_pto_FSTI_1wk, mean_pto_FSTC_1wk],
                          [sem_pto_FSTE_1wk, sem_pto_FSTI_1wk, sem_pto_FSTC_1wk],
                          [my_r, my_b, my_k]):
    ax_5b.bar(pos, m, width=1.0, color=c)
    ax_5b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

G, P = do_stats_tfc(np.mean(FSTE_1wk_tot[:, post_tone_only], axis=1),
                     np.mean(FSTI_1wk_tot[:, post_tone_only], axis=1),
                     np.mean(FSTC_1wk_tot[:, post_tone_only], axis=1), use_test, post_hoc, 4)
sigstar_text(ax_5b, G, P, y_start=max(mean_pto_FSTE_1wk, mean_pto_FSTI_1wk, mean_pto_FSTC_1wk) +
             max(sem_pto_FSTE_1wk, sem_pto_FSTI_1wk, sem_pto_FSTC_1wk) + 3)

# First 3min bars
for pos, m, se, c in zip([9, 10, 11],
                          [mean_f3_FSTE_1wk, mean_f3_FSTI_1wk, mean_f3_FSTC_1wk],
                          [sem_f3_FSTE_1wk, sem_f3_FSTI_1wk, sem_f3_FSTC_1wk],
                          [my_r, my_b, my_k]):
    ax_5b.bar(pos, m, width=1.0, color=c)
    ax_5b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

G, P = do_stats_tfc(np.mean(FSTE_1wk_tot[:, first_3min], axis=1),
                     np.mean(FSTI_1wk_tot[:, first_3min], axis=1),
                     np.mean(FSTC_1wk_tot[:, first_3min], axis=1), use_test, post_hoc, 8)
sigstar_text(ax_5b, G, P, y_start=max(mean_f3_FSTE_1wk, mean_f3_FSTI_1wk, mean_f3_FSTC_1wk) +
             max(sem_f3_FSTE_1wk, sem_f3_FSTI_1wk, sem_f3_FSTC_1wk) + 3)

ax_5b.set_xticks(range(1, 12))
ax_5b.set_xticklabels(['Tone', 'Tone', 'Tone', '', 'Post tone', 'Post tone', 'Post tone', '',
                        f'First{first_min}', f'First{first_min}', f'First{first_min}'], rotation=-45, ha='left')
ax_5b.set_title('Test 1wk')
_setup_axes(ax_5b)
fig_5b.tight_layout()
_save(fig_5b, '5b_1wk_period_freezing_first_3min')


# ─────────────────────────────────────────────
# Plot 5c: 1wk within-groups comparison
# ─────────────────────────────────────────────

fig_5c, ax_5c = plt.subplots(figsize=(3, 2) if for_paper else (6, 4))

for grp_offset, color, scatter_c, tot_data, f3, tn, pto, label in [
    (0, my_r, my_k, FSTE_1wk_tot, first_3min_FSTE_1wk, tone_FSTE_1wk, post_tone_only_FSTE_1wk, 'Exc'),
    (4, my_b, my_k, FSTI_1wk_tot, first_3min_FSTI_1wk, tone_FSTI_1wk, post_tone_only_FSTI_1wk, 'Inh'),
    (8, my_k, my_h, FSTC_1wk_tot, first_3min_FSTC_1wk, tone_FSTC_1wk, post_tone_only_FSTC_1wk, 'Ctl'),
]:
    means = [np.mean(f3), np.mean(tn), np.mean(pto)]
    sems = [np.std(f3, ddof=0) / np.sqrt(len(f3)),
            np.std(tn, ddof=0) / np.sqrt(len(tn)),
            np.std(pto, ddof=0) / np.sqrt(len(pto))]
    positions = [1 + grp_offset, 2 + grp_offset, 3 + grp_offset]

    for pos, m, se in zip(positions, means, sems):
        ax_5c.bar(pos, m, width=1.0, color=color)
        ax_5c.errorbar(pos, m, yerr=se, fmt='none', ecolor=color, capsize=0, linewidth=my_linewidth)

    for pos, vals in zip(positions, [f3, tn, pto]):
        ax_5c.scatter(np.full(len(vals), pos), vals, s=my_sz, color=scatter_c, zorder=3)
    for i in range(len(f3)):
        ax_5c.plot([positions[0], positions[1]], [f3[i], tn[i]], color=scatter_c, linewidth=0.5)
        ax_5c.plot([positions[1], positions[2]], [tn[i], pto[i]], color=scatter_c, linewidth=0.5)

    data_rm = [
        np.mean(tot_data[:, first_3min], axis=1),
        np.mean(tot_data[:, tone], axis=1),
        np.mean(tot_data[:, post_tone_only], axis=1),
    ]
    G, P = stats_ranova(data_rm, post_hoc, grp_offset)
    max_sem = max(sems)
    sigstar_text(ax_5c, G, P, y_start=max(means) + max_sem + 3, sep=3.5)

ax_5c.set_xticks(range(1, 12))
ax_5c.set_xticklabels([f'First{first_min}', 'Tone', 'Post tone', '',
                        f'First{first_min}', 'Tone', 'Post tone', '',
                        f'First{first_min}', 'Tone', 'Post tone'], rotation=-45, ha='left')
ax_5c.set_ylabel('Freezing (%)')
ax_5c.set_title('Test B 1wk - within groups')
_setup_axes(ax_5c)
fig_5c.tight_layout()
_save(fig_5c, '5c_1wk_period_freezing_first_3min_comparison')


# ─────────────────────────────────────────────
# Plot 5d: Compare 48hr vs 1wk
# ─────────────────────────────────────────────

fig_5d, ax_5d = plt.subplots(figsize=(4, 2) if for_paper else (8, 4))

# Tone: positions 1-6
tone_48hr = [mean_tone_FSTE, mean_tone_FSTE_1wk, mean_tone_FSTI, mean_tone_FSTI_1wk, mean_tone_FSTC, mean_tone_FSTC_1wk]
tone_sems = [sem_tone_FSTE, sem_tone_FSTE_1wk, sem_tone_FSTI, sem_tone_FSTI_1wk, sem_tone_FSTC, sem_tone_FSTC_1wk]
tone_colors = [my_r, my_r, my_b, my_b, my_k, my_k]
for pos, m, se, c in zip(range(1, 7), tone_48hr, tone_sems, tone_colors):
    ax_5d.bar(pos, m, width=1.0, color=c)
    ax_5d.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

# Paired t-tests for tone (falls back to unpaired if different N)
p_dwE = _ttest_paired_or_ind(np.mean(FSTE_tot[:, tone], axis=1), np.mean(FSTE_1wk_tot[:, tone], axis=1))
p_dwI = _ttest_paired_or_ind(np.mean(FSTI_tot[:, tone], axis=1), np.mean(FSTI_1wk_tot[:, tone], axis=1))
p_dwC = _ttest_paired_or_ind(np.mean(FSTC_tot[:, tone], axis=1), np.mean(FSTC_1wk_tot[:, tone], axis=1))
sigstar_text(ax_5d, [[1, 2], [3, 4], [5, 6]], [p_dwE, p_dwI, p_dwC],
             y_start=max(tone_48hr) + max(tone_sems) + 3)

# Post-tone-only: positions 8-13
pto_vals = [mean_pto_FSTE, mean_pto_FSTE_1wk, mean_pto_FSTI, mean_pto_FSTI_1wk, mean_pto_FSTC, mean_pto_FSTC_1wk]
pto_sems_cmp = [sem_pto_FSTE, sem_pto_FSTE_1wk, sem_pto_FSTI, sem_pto_FSTI_1wk, sem_pto_FSTC, sem_pto_FSTC_1wk]
for pos, m, se, c in zip(range(8, 14), pto_vals, pto_sems_cmp, tone_colors):
    ax_5d.bar(pos, m, width=1.0, color=c)
    ax_5d.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

p_dwE = _ttest_paired_or_ind(np.mean(FSTE_tot[:, post_tone_only], axis=1), np.mean(FSTE_1wk_tot[:, post_tone_only], axis=1))
p_dwI = _ttest_paired_or_ind(np.mean(FSTI_tot[:, post_tone_only], axis=1), np.mean(FSTI_1wk_tot[:, post_tone_only], axis=1))
p_dwC = _ttest_paired_or_ind(np.mean(FSTC_tot[:, post_tone_only], axis=1), np.mean(FSTC_1wk_tot[:, post_tone_only], axis=1))
sigstar_text(ax_5d, [[8, 9], [10, 11], [12, 13]], [p_dwE, p_dwI, p_dwC],
             y_start=max(pto_vals) + max(pto_sems_cmp) + 3)

# First 3min: positions 15-20
f3_vals = [mean_f3_FSTE, mean_f3_FSTE_1wk, mean_f3_FSTI, mean_f3_FSTI_1wk, mean_f3_FSTC, mean_f3_FSTC_1wk]
f3_sems_cmp = [sem_f3_FSTE, sem_f3_FSTE_1wk, sem_f3_FSTI, sem_f3_FSTI_1wk, sem_f3_FSTC, sem_f3_FSTC_1wk]
for pos, m, se, c in zip(range(15, 21), f3_vals, f3_sems_cmp, tone_colors):
    ax_5d.bar(pos, m, width=1.0, color=c)
    ax_5d.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

p_dwE = _ttest_paired_or_ind(np.mean(FSTE_tot[:, first_3min], axis=1), np.mean(FSTE_1wk_tot[:, first_3min], axis=1))
p_dwI = _ttest_paired_or_ind(np.mean(FSTI_tot[:, first_3min], axis=1), np.mean(FSTI_1wk_tot[:, first_3min], axis=1))
p_dwC = _ttest_paired_or_ind(np.mean(FSTC_tot[:, first_3min], axis=1), np.mean(FSTC_1wk_tot[:, first_3min], axis=1))
sigstar_text(ax_5d, [[15, 16], [17, 18], [19, 20]], [p_dwE, p_dwI, p_dwC],
             y_start=max(f3_vals) + max(f3_sems_cmp) + 3)

ax_5d.set_xticks(list(range(1, 7)) + list(range(8, 14)) + list(range(15, 21)))
ax_5d.set_xticklabels(
    ['Exc', 'Exc+', 'Inh', 'Inh+', 'Ctl', 'Ctl+'] * 3,
    rotation=-45, ha='left')
ax_5d.set_ylabel('Freezing (%)')
ax_5d.set_title('Test 48hr vs 1wk')
_setup_axes(ax_5d)
fig_5d.tight_layout()
_save(fig_5d, '5d_compare_48hr_1wk_period_freezing')


# ─────────────────────────────────────────────
# Plot 5e: Compare first 3min 48hr and 1wk
# ─────────────────────────────────────────────

fig_5e, ax_5e = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))

f3_bar_vals = [mean_f3_FSTE, mean_f3_FSTE_1wk, mean_f3_FSTI, mean_f3_FSTI_1wk, mean_f3_FSTC, mean_f3_FSTC_1wk]
f3_bar_sems = [sem_f3_FSTE, sem_f3_FSTE_1wk, sem_f3_FSTI, sem_f3_FSTI_1wk, sem_f3_FSTC, sem_f3_FSTC_1wk]
positions_5e = [1, 2, 4, 5, 7, 8]
colors_5e = [my_r, my_r, my_b, my_b, my_k, my_k]
for pos, m, se, c in zip(positions_5e, f3_bar_vals, f3_bar_sems, colors_5e):
    ax_5e.bar(pos, m, width=1.0, color=c)
    ax_5e.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

p_3min_E = _ttest_paired_or_ind(np.mean(FSTE_tot[:, first_3min], axis=1), np.mean(FSTE_1wk_tot[:, first_3min], axis=1))
p_3min_I = _ttest_paired_or_ind(np.mean(FSTI_tot[:, first_3min], axis=1), np.mean(FSTI_1wk_tot[:, first_3min], axis=1))
p_3min_C = _ttest_paired_or_ind(np.mean(FSTC_tot[:, first_3min], axis=1), np.mean(FSTC_1wk_tot[:, first_3min], axis=1))
sigstar_text(ax_5e, [[1, 2], [4, 5], [7, 8]], [p_3min_E, p_3min_I, p_3min_C],
             y_start=max(f3_bar_vals) + max(f3_bar_sems) / 2 + 3)

ax_5e.set_xticks(positions_5e + [3, 6])
ax_5e.set_xticklabels(['48hr Exc', '1wk Exc', '', '48hr Inh', '1wk Inh', '', '48hr Ctl', '1wk Ctl'],
                       rotation=-45, ha='left')
# re-order ticks
ax_5e.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax_5e.set_xticklabels(['48hr Exc', '1wk Exc', '', '48hr Inh', '1wk Inh', '', '48hr Ctl', '1wk Ctl'],
                       rotation=-45, ha='left')
ax_5e.set_title('Test B first 3 minutes')
_setup_axes(ax_5e)
fig_5e.tight_layout()
_save(fig_5e, '5e_compare_48hr_1wk_first_3min')


# ─────────────────────────────────────────────
# Test A calculations
# ─────────────────────────────────────────────

FSTI_TestA_mean = np.mean(np.mean(FSTI_TestA_tot, axis=1))
FSTE_TestA_mean = np.mean(np.mean(FSTE_TestA_tot, axis=1))
FSTC_TestA_mean = np.mean(np.mean(FSTC_TestA_tot, axis=1))

FSTI_TestA_std = np.std(np.mean(FSTI_TestA_tot, axis=1), ddof=0)
FSTE_TestA_std = np.std(np.mean(FSTE_TestA_tot, axis=1), ddof=0)
FSTC_TestA_std = np.std(np.mean(FSTC_TestA_tot, axis=1), ddof=0)

FSTI_TestA_sem = FSTI_TestA_std / np.sqrt(FSTI_TestA_tot.shape[0])
FSTE_TestA_sem = FSTE_TestA_std / np.sqrt(FSTE_TestA_tot.shape[0])
FSTC_TestA_sem = FSTC_TestA_std / np.sqrt(FSTC_TestA_tot.shape[0])

FSTI_TestA_1wk_mean = np.mean(np.mean(FSTI_TestA_1wk_tot, axis=1))
FSTE_TestA_1wk_mean = np.mean(np.mean(FSTE_TestA_1wk_tot, axis=1))
FSTC_TestA_1wk_mean = np.mean(np.mean(FSTC_TestA_1wk_tot, axis=1))

FSTI_TestA_1wk_std = np.std(np.mean(FSTI_TestA_1wk_tot, axis=1), ddof=0)
FSTE_TestA_1wk_std = np.std(np.mean(FSTE_TestA_1wk_tot, axis=1), ddof=0)
FSTC_TestA_1wk_std = np.std(np.mean(FSTC_TestA_1wk_tot, axis=1), ddof=0)

FSTI_TestA_1wk_sem = FSTI_TestA_1wk_std / np.sqrt(FSTI_TestA_1wk_tot.shape[0])
FSTE_TestA_1wk_sem = FSTE_TestA_1wk_std / np.sqrt(FSTE_TestA_1wk_tot.shape[0])
FSTC_TestA_1wk_sem = FSTC_TestA_1wk_std / np.sqrt(FSTC_TestA_1wk_tot.shape[0])


# ─────────────────────────────────────────────
# Plot 6a: Test A - 48hr and 1wk
# ─────────────────────────────────────────────

fig_6a, ax_6a = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))

# 48hr Test A bars
for pos, m, se, c in zip([1, 2, 3],
                          [FSTE_TestA_mean, FSTI_TestA_mean, FSTC_TestA_mean],
                          [FSTE_TestA_sem, FSTI_TestA_sem, FSTC_TestA_sem],
                          [my_r, my_b, my_k]):
    ax_6a.bar(pos, m, width=1.0, color=c)
    ax_6a.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

# Ranksum tests for 48hr Test A (as in original MATLAB)
_, p_EI = sp_stats.mannwhitneyu(np.mean(FSTE_TestA_tot, axis=1), np.mean(FSTI_TestA_tot, axis=1), alternative='two-sided')
_, p_EC = sp_stats.mannwhitneyu(np.mean(FSTE_TestA_tot, axis=1), np.mean(FSTC_TestA_tot, axis=1), alternative='two-sided')
_, p_IC = sp_stats.mannwhitneyu(np.mean(FSTI_TestA_tot, axis=1), np.mean(FSTC_TestA_tot, axis=1), alternative='two-sided')
max_sem = max(FSTE_TestA_sem, FSTI_TestA_sem, FSTC_TestA_sem)
sigstar_text(ax_6a, [[2, 3], [1, 2], [1, 3]], [p_IC, p_EI, p_EC],
             y_start=max(FSTE_TestA_mean, FSTI_TestA_mean, FSTC_TestA_mean) + max_sem + 3)

# 1wk Test A bars
for pos, m, se, c in zip([5, 6, 7],
                          [FSTE_TestA_1wk_mean, FSTI_TestA_1wk_mean, FSTC_TestA_1wk_mean],
                          [FSTE_TestA_1wk_sem, FSTI_TestA_1wk_sem, FSTC_TestA_1wk_sem],
                          [my_r, my_b, my_k]):
    ax_6a.bar(pos, m, width=1.0, color=c)
    ax_6a.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

_, p_EI = sp_stats.mannwhitneyu(np.mean(FSTE_TestA_1wk_tot, axis=1), np.mean(FSTI_TestA_1wk_tot, axis=1), alternative='two-sided')
_, p_EC = sp_stats.mannwhitneyu(np.mean(FSTE_TestA_1wk_tot, axis=1), np.mean(FSTC_TestA_1wk_tot, axis=1), alternative='two-sided')
_, p_IC = sp_stats.mannwhitneyu(np.mean(FSTI_TestA_1wk_tot, axis=1), np.mean(FSTC_TestA_1wk_tot, axis=1), alternative='two-sided')
max_sem = max(FSTE_TestA_1wk_sem, FSTI_TestA_1wk_sem, FSTC_TestA_1wk_sem)
sigstar_text(ax_6a, [[6, 7], [5, 6], [5, 7]], [p_IC, p_EI, p_EC],
             y_start=max(FSTE_TestA_1wk_mean, FSTI_TestA_1wk_mean, FSTC_TestA_1wk_mean) + max_sem + 3)

ax_6a.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax_6a.set_xticklabels(['Exc 48hr', 'Inh 48hr', 'Ctl 48hr', '', 'Exc 1wk', 'Inh 1wk', 'Ctl 1wk'],
                       rotation=-45, ha='left')
ax_6a.set_ylim([0, 60])
ax_6a.set_ylabel('Freezing (%)')
ax_6a.set_title('Test A')
_setup_axes(ax_6a)
fig_6a.tight_layout()
_save(fig_6a, '6a_testA_48hr_1wk_freezing')


# ─────────────────────────────────────────────
# Plot 6b: Test A 48hr vs 1wk (paired comparison)
# ─────────────────────────────────────────────

fig_6b, ax_6b = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))

testA_cmp_vals = [FSTE_TestA_mean, FSTE_TestA_1wk_mean, FSTI_TestA_mean, FSTI_TestA_1wk_mean,
                  FSTC_TestA_mean, FSTC_TestA_1wk_mean]
testA_cmp_sems = [FSTE_TestA_sem, FSTE_TestA_1wk_sem, FSTI_TestA_sem, FSTI_TestA_1wk_sem,
                  FSTC_TestA_sem, FSTC_TestA_1wk_sem]
positions_6b = [1, 2, 4, 5, 7, 8]
colors_6b = [my_r, my_r, my_b, my_b, my_k, my_k]

for pos, m, se, c in zip(positions_6b, testA_cmp_vals, testA_cmp_sems, colors_6b):
    ax_6b.bar(pos, m, width=1.0, color=c)
    ax_6b.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

# Paired t-tests
p_TestA_E = _ttest_paired_or_ind(np.mean(FSTE_TestA_tot, axis=1), np.mean(FSTE_TestA_1wk_tot, axis=1))
p_TestA_I = _ttest_paired_or_ind(np.mean(FSTI_TestA_tot, axis=1), np.mean(FSTI_TestA_1wk_tot, axis=1))
p_TestA_C = _ttest_paired_or_ind(np.mean(FSTC_TestA_tot, axis=1), np.mean(FSTC_TestA_1wk_tot, axis=1))
max_sem = max(testA_cmp_sems)
sigstar_text(ax_6b, [[1, 2], [4, 5], [7, 8]], [p_TestA_E, p_TestA_I, p_TestA_C],
             y_start=max(testA_cmp_vals) + max_sem + 3)

ax_6b.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax_6b.set_xticklabels(['48hr Exc', '1wk Exc', '', '48hr Inh', '1wk Inh', '', '48hr Ctl', '1wk Ctl'],
                       rotation=-45, ha='left')
ax_6b.set_ylim([0, 50])
ax_6b.set_ylabel('Freezing (%)')
ax_6b.set_title('Test A 48hr vs 1wk')
_setup_axes(ax_6b)
fig_6b.tight_layout()
_save(fig_6b, '6b_compare_48hr_1wk_TestA')


# ─────────────────────────────────────────────
# Plot 7a: Generalization - First 3min vs Test A (48hr)
# ─────────────────────────────────────────────

fig_7a, ax_7a = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))

gen_vals = [mean_f3_FSTE, FSTE_TestA_mean, mean_f3_FSTI, FSTI_TestA_mean, mean_f3_FSTC, FSTC_TestA_mean]
gen_sems = [sem_f3_FSTE, FSTE_TestA_sem, sem_f3_FSTI, FSTI_TestA_sem, sem_f3_FSTC, FSTC_TestA_sem]
# Note: MATLAB uses sem_first_3min_FSTE for Ctl errorbar position 7 (looks like a bug) — we use correct sem
positions_7a = [1, 2, 4, 5, 7, 8]
colors_7a = [my_r, my_r, my_b, my_b, my_k, my_k]

for pos, m, se, c in zip(positions_7a, gen_vals, gen_sems, colors_7a):
    ax_7a.bar(pos, m, width=1.0, color=c)
    ax_7a.errorbar(pos, m, yerr=se, fmt='none', ecolor=c, capsize=0, linewidth=my_linewidth)

# Paired t-tests
p_E = _ttest_paired_or_ind(np.mean(FSTE_tot[:, first_3min], axis=1), np.mean(FSTE_TestA_tot, axis=1))
p_I = _ttest_paired_or_ind(np.mean(FSTI_tot[:, first_3min], axis=1), np.mean(FSTI_TestA_tot, axis=1))
p_C = _ttest_paired_or_ind(np.mean(FSTC_tot[:, first_3min], axis=1), np.mean(FSTC_TestA_tot, axis=1))
max_sem = max(gen_sems)
sigstar_text(ax_7a, [[1, 2], [4, 5], [7, 8]], [p_E, p_I, p_C],
             y_start=max(gen_vals) + max_sem + 3)

ax_7a.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax_7a.set_xticklabels([f'Exc First{first_min}', 'Exc Test A', '',
                        f'Inh First{first_min}', 'Inh Test A', '',
                        f'Ctl First{first_min}', 'Ctl Test A'],
                       rotation=-45, ha='left')
ax_7a.set_ylim([0, 50])
ax_7a.set_ylabel('Freezing (%)')
ax_7a.set_title(f'Test B First{first_min}\nvs Test A (48hr)')
_setup_axes(ax_7a)
fig_7a.tight_layout()
_save(fig_7a, '7a_compare_First_3min_vs_TestA_48hr')


plt.show()

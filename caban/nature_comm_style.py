from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

from caban.utilities import FONT_SANS_SERIF


def set_rc_params(
    fontsize=8,
    titlesize=None,
    rcParams=None
):
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_SANS_SERIF,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.labelsize': fontsize,
        'lines.markersize': 0.5,
        'lines.linewidth': 1,
        'legend.fontsize': fontsize,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.linewidth': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
    }
    if titlesize:
        params['axes.titlesize'] = titlesize
    if rcParams:
        params.update(rcParams)
    for k, v in params.items():
        plt.rcParams[k] = v
        print(f'nature_comm_style setting {k} to {v}')


@contextmanager
def nature_comm_style(
    column_width='single',
    width_factor=1,
    height_factor=0.618,
    titlesize=None,
    fontsize=8,
    rcParams=None,
    set_figsize=True,
):
    saved_rcParams = plt.rcParams.copy()

    single_column_width = 88 / 25.4
    double_column_width = 180 / 25.4
    widths = {
        'single': single_column_width,
        'double': double_column_width,
        'one-third': double_column_width * (1 / 3),
        'two-thirds': double_column_width * (2 / 3),
    }

    if column_width not in widths:
        raise ValueError("column_width not among valid options: 'single', 'double', 'one-third', 'two-thirds'")

    width = widths[column_width] * width_factor
    height = width * height_factor
    if set_figsize:
        print(f'nature_comm_style setting figsize to width {width} height {height}')
        plt.rcParams['figure.figsize'] = (width, height)

    set_rc_params(fontsize=fontsize, titlesize=titlesize, rcParams=rcParams)
    try:
        yield
    finally:
        plt.rcParams.update(saved_rcParams)

# --- Nature / Nature Neuroscience style for paper figures (VS, 2026-10-14) ----------------------
# Arial; 7 pt for axis labels, tick labels and panel tags, 6 pt for legends; 0.5 pt axes and ticks,
# outward ticks, no top/right spines, no grid; ~0.75 pt data lines; sizes in mm; SVG with editable
# text plus a 600-dpi PNG. Figure 2's panels are drawn at the size they occupy in the composite
# (caban.figure2_composite), so 7 pt on a panel is 7 pt in the figure.

MM = 1 / 25.4
FONT_PT = 7
LEGEND_PT = 6
LETTER_PT = 8
LINE_PT = 0.75
AXIS_PT = 0.5
PNG_DPI = 600

NATURE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': FONT_PT,
    'axes.labelsize': FONT_PT,
    'axes.titlesize': FONT_PT,
    'axes.titleweight': 'normal',
    'axes.titlepad': 3,
    'axes.labelpad': 2,
    'xtick.labelsize': FONT_PT,
    'ytick.labelsize': FONT_PT,
    'legend.fontsize': LEGEND_PT,
    'legend.frameon': False,
    'legend.handlelength': 1.2,
    'legend.borderaxespad': 0.2,
    'figure.titlesize': FONT_PT,
    'axes.linewidth': AXIS_PT,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': AXIS_PT,
    'ytick.major.width': AXIS_PT,
    'xtick.minor.width': AXIS_PT * 0.8,
    'ytick.minor.width': AXIS_PT * 0.8,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'xtick.minor.size': 1.2,
    'ytick.minor.size': 1.2,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'lines.linewidth': LINE_PT,
    'lines.markersize': 2.0,
    'patch.linewidth': AXIS_PT,
    'svg.fonttype': 'none',          # editable text in the SVG
    'pdf.fonttype': 42,
    'savefig.dpi': PNG_DPI,
    'figure.dpi': 150,
}

def figsize_mm(width_mm, height_mm):
    return (width_mm * MM, height_mm * MM)


@contextmanager
def nature_style():
    """Matplotlib settings for a paper figure; restores the previous ones on exit."""
    with plt.rc_context(NATURE_RC):
        yield


def pastel(colour, amount=0.6):
    """``colour`` mixed towards white by ``amount`` (the box fill of the box-and-strip style)."""
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def box_strip(ax, values_per_group, group_order, colours, labels=None, point_size=6, width=0.55,
              seed=0):
    """The project's categorical comparison (CLAUDE.md): pastel box, darker matching points with
    black edges (alpha 0.85, edge 0.3 pt), black median line; one point per value (e.g. per animal).

    values_per_group : {group: 1-D array}. ``labels`` maps groups to tick labels (e.g.
    single_unit_common.GROUP_LABELS); default the group names. Returns the x positions.
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(len(group_order))
    data = [np.asarray(values_per_group[g], dtype=float) for g in group_order]
    bp = ax.boxplot(data, positions=xs, widths=width, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=LINE_PT),
                    whiskerprops=dict(linewidth=AXIS_PT), capprops=dict(linewidth=AXIS_PT),
                    boxprops=dict(linewidth=AXIS_PT))
    for patch, g in zip(bp['boxes'], group_order):
        patch.set_facecolor(pastel(colours[g]))
        patch.set_edgecolor(colours[g])
    for x, v, g in zip(xs, data, group_order):
        jitter = rng.uniform(-width * 0.28, width * 0.28, size=v.size)
        ax.scatter(x + jitter, v, s=point_size, color=colours[g], alpha=0.85, edgecolors='black',
                   linewidths=0.3, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([(labels or {}).get(g, g) for g in group_order])
    ax.set_xlim(-0.6, len(group_order) - 0.4)
    return xs


def save_paper_panel(fig, path_root):
    """Save ``path_root``.svg (editable text) and ``path_root``.png (600 dpi), at the figure's own
    size -- no tight bbox, so a panel keeps exactly the size the composite allots it."""
    fig.savefig(path_root + '.svg', format='svg')
    fig.savefig(path_root + '.png', format='png', dpi=PNG_DPI)

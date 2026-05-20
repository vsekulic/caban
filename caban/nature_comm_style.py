from contextlib import contextmanager

from matplotlib import pyplot as plt


def set_rc_params(
    fontsize=8,
    titlesize=None,
    rcParams=None
):
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
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
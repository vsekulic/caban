from matplotlib import pyplot as plt
from contextlib import contextmanager
import os

def set_rc_params(
    fontsize=8,
    titlesize=None,
    rcParams=None
):
    # Build default rcParams
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.labelsize': fontsize,
        'lines.markersize': 0.5,
        'lines.linewidth': 1,
        'legend.fontsize': fontsize,  # Set legend fontsize
    }
    if titlesize:
        params['axes.titlesize'] = titlesize
    if rcParams:
        params.update(rcParams)
    for k, v in params.items():
        plt.rcParams[k] = v
        print(f'set_rc_params setting {k} to {v}')

@contextmanager
def nature_comm_style(
    column_width='single',
    width_factor=1,
    height_factor=0.618,
    titlesize=None,
    fontsize=8,
    rcParams=None,
    set_figsize=True
):
    saved_rcParams = plt.rcParams.copy()
    plt.style.reload_library()  # Reload all styles        
    # Apply base style
    plt.style.use('nature_comm')

    single_column_width = 88 / 25.4  # 88mm width
    double_column_width = 180 / 25.4  # 180mm width
    widths = {
        'single': single_column_width,
        'double': double_column_width,
        'one-third': double_column_width * (1/3),
        'two-thirds': double_column_width * (2/3)
    }

    if column_width not in widths:
        raise ValueError("column_width not among valid options: 'single', 'double', 'one-third', 'two-thirds'")
    width = widths[column_width] * width_factor
    height = width * height_factor
    if set_figsize:
        print('nature_comm_style setting figsize to width {} height {}'.format(width, height))
        plt.rcParams['figure.figsize'] = (width, height)

    set_rc_params(fontsize=fontsize, titlesize=titlesize, rcParams=rcParams)
    try:
        yield
    finally:
        plt.rcParams.update(saved_rcParams)

def savefig(fig, PLOTS_DIR, subdir, outfile, paper_plot=True, dpi=300):
    save_path = os.path.join(PLOTS_DIR, subdir)
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, outfile)
    fig.savefig(plot_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    if paper_plot:
        fig.savefig(f'{plot_path}.pdf', format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    print('SAVING {}'.format(outfile))

from matplotlib import pyplot as plt
import os
from caban.nature_comm_style import nature_comm_style, set_rc_params

def savefig(fig, PLOTS_DIR, subdir, outfile, paper_plot=True, dpi=300):
    save_path = os.path.join(PLOTS_DIR, subdir)
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, outfile)
    fig.savefig(plot_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    if paper_plot:
        fig.savefig(f'{plot_path}.pdf', format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    print('SAVING {}'.format(outfile))

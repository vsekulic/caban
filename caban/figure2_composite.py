"""Figure 2 composite: the panels the pipeline writes, assembled in one Nature double-column page.

Layout follows VS's PowerPoint draft (~/Documents/DB/1-McHugh postdoc/3-PAPER/Figure2_ppt_draft.pptx,
2026-08-24), panels a-j, plus the start of the per-cell block that replaces its "TODO" box: k, the
held-out tone-aligned heatmaps (epoch_modulation panel K, sorted on other trials, §O.8), on YrA, the
primary signal. Further per-cell panels (modulation index, split-half stability) wait until they are
drawn at final size and the YrA check is done (VS, 2026-10-14).

Every panel is placed from its CURRENT output file, so rerunning the pipeline and then this rebuilds
the figure. Panels a (injection schematic) and b (mean field of view, Minian footprints) are cropped
from the earlier Illustrator figure; the pipeline does not produce them. A missing file is an error.

Placement only: each panel keeps the type sizes it was drawn with, scaled to its box. Nature
sizing inside each panel (7 pt text at final size) needs each plotting function to draw at its
final size -- a separate step.

    from caban.figure2_composite import build_figure2_composite
    build_figure2_composite(cfg.PLOTS_DIR, out_dir)          # fig2.png (600 dpi) + fig2.svg
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Panels not produced by the pipeline (VS's earlier Illustrator figure and its Minian viewer exports).
STATIC_DIR = os.path.expanduser('~/Documents/DB/1-McHugh postdoc/3-PAPER/paper_plots/fig2')
SCHEMATIC_TIFF = 'fig2.tif'
# Boxes (left, top, right, bottom) in fig2.tif (300 dpi, 2078 x 2702 px), found from the whitespace
# around each panel, without the old capital letters (2026-10-14): a, the virus list and injection
# schematic; b, the mean field of view and the Minian footprints.
STATIC_CROPS = {'a': (33, 5, 463, 237), 'b': (524, 5, 1021, 261)}

# Panel letter -> file under PLOTS_DIR.
PANEL_FILES = {
    'c': 'plot_sample_traces/plot_sample_traces.png',
    'd': 'sp_rates_lmm/TFC_cond/manipulation_check.png',
    'e': 'sp_rates_lmm/TFC_cond/decomposition.png',
    'f': 'sp_rates_lmm/TFC_cond/width_height_matched_examples_vertical.png',
    'g': 'sp_rates_lmm/paper/tfc_amplitude_rate/tfc_amplitude_rate_by_epoch.png',
    'h': 'sp_rates_lmm/TFC_cond/amplitude_ecdf.png',
    'i': 'sp_rates_lmm/TFC_cond/epoch_profile.png',
    'j': 'sp_rates_lmm/paper/recall/Test_B/testb_amplitude_rate_modulation.png',
    'k': 'epoch_modulation/YrA/panel_K_tone_aligned_heatmaps.png',
}

FIG_WIDTH_MM = 183.0          # Nature double column
GAP_MM = 3.0
# Panel boxes (x, y from the top, width, height) in mm.
LAYOUT_MM = {
    'a': (0, 0, 48, 30),
    'b': (50, 0, 54, 30),
    'c': (106, 0, 77, 30),
    'd': (0, 33, 62, 36),
    'e': (64, 33, 119, 36),
    'f': (0, 72, 22, 76),
    'g': (24, 72, 74, 76),
    'h': (100, 72, 36, 36),
    'i': (138, 72, 45, 36),
    'j': (100, 111, 83, 37),
    'k': (0, 151, 183, 66),
}
LETTER_PT = 8
DPI_PNG = 600

# Written beside the figure on every build. No numbers: those live in docs/paper/paper_figure2.md and
# the stats files, which a rerun updates.
CAPTION = """**Fig. 2 | Chemogenetic modulation of SST interneurons dissociates the magnitude and the frequency of
CA1 pyramidal calcium events during trace fear conditioning.**
**a**, Viral strategy: SomaGCaMP6f with Cre-dependent hM3D(Gq), hM4D(Gi) or mCherry in dorsal CA1 of
SST-Cre mice, imaged through a GRIN lens. **b**, Mean field of view (left) and Minian spatial footprints
(right) from one animal. **c**, Example traces: calcium signal with deconvolved activity and detected
events (circles), control (black), hM4D (blue) and hM3D (red) cells. **d**, Manipulation check across the
drug-free (LT1) and CNO (LT2) linear-track sessions: within-cell change in log per-event amplitude
(left) and the fraction of LT1 cells not re-registered in LT2 (right); points are animals. **e**,
Decomposition of trace-interval activity: fraction of active cells, event rate among active cells,
event rate over all cells, per-event amplitude and their product; small points are cells, large
points animals. **f**, Peak-height-matched example events, one per group; shading marks the run.
**g**, Per-event amplitude (top) and population event rate (bottom) in the three 20-s epochs
(pre-tone, trace, post-shock); small points are cells, large points animals; brackets show
Holm-corrected treatment-versus-control contrasts of the group x epoch mixed model (animal as
random intercept). **h**, Cumulative distribution of trace-interval per-cell amplitude, per group
(thick) and per animal (thin). **i**, Per-animal amplitude profile across five conditioning epochs
(thin lines) with group means and 95% intervals (thick lines, bands); descriptive. **j**, 48-h recall
(Test B): per-animal change from pre-tone to post-tone in per-event amplitude (left) and population
event rate (right). **k**, Tone-aligned activity of every cell (YrA, standardized once per cell over
the session), rows sorted by the trace-interval modulation index on trials 1, 3 and 5 and showing the
mean of trials 2 and 4, so the order is independent of the data displayed; dotted lines mark tone
offset, shock and the end of the post-shock window, grey band the shock. n = 5 hM3D, 6 hM4D and 6
mCherry animals; the animal is the unit of inference throughout.
"""


def _read_png(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'figure 2 composite: panel source {path} not found; run its '
                                f'pipeline section first')
    return np.asarray(Image.open(path).convert('RGB'))


def _static_panels(static_dir):
    path = os.path.join(static_dir, SCHEMATIC_TIFF)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'figure 2 composite: static panel source {path} not found')
    tif = Image.open(path).convert('RGB')
    return {letter: np.asarray(tif.crop(box)) for letter, box in STATIC_CROPS.items()}


def build_figure2_composite(plots_dir, out_dir, static_dir=STATIC_DIR):
    """Write ``fig2.png`` (600 dpi) and ``fig2.svg`` to ``out_dir``; returns their paths."""
    images = _static_panels(static_dir)
    images.update({k: _read_png(os.path.join(plots_dir, v)) for k, v in PANEL_FILES.items()})

    height_mm = max(y + h for _, y, _, h in LAYOUT_MM.values())
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
                         'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, height_mm / 25.4))
        for letter, (x, y, w, h) in LAYOUT_MM.items():
            ax = fig.add_axes([x / FIG_WIDTH_MM, 1 - (y + h) / height_mm, w / FIG_WIDTH_MM, h / height_mm])
            ax.imshow(images[letter], interpolation='lanczos')
            ax.set_anchor('NW')
            ax.axis('off')
            fig.text(x / FIG_WIDTH_MM, 1 - y / height_mm, letter, fontsize=LETTER_PT,
                     fontweight='bold', ha='left', va='top')
        os.makedirs(out_dir, exist_ok=True)
        png, svg = os.path.join(out_dir, 'fig2.png'), os.path.join(out_dir, 'fig2.svg')
        fig.savefig(png, dpi=DPI_PNG, facecolor='white')
        fig.savefig(svg, dpi=DPI_PNG, facecolor='white')  # dpi sets the embedded images' resolution
        plt.close(fig)
    with open(os.path.join(out_dir, 'fig2_caption.md'), 'w') as fh:
        fh.write(CAPTION)
    print(f'figure 2 composite: {png} ({DPI_PNG} dpi), {svg}')
    return png, svg

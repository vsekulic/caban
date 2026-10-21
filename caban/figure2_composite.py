"""Figure 2 composite: the panels the pipeline writes, assembled in one Nature double-column page.

Layout follows VS's PowerPoint draft (~/Documents/DB/1-McHugh postdoc/3-PAPER/Figure2_ppt_draft.pptx,
2026-08-24), panels a-j, plus the start of the per-cell block that replaces its "TODO" box: k, the
event-aligned activity around tone, trace, shock and post-shock onset (epoch_modulation panel M,
event-proximal lane), on YrA, the primary signal (VS, 2026-10-15; it replaced the held-out heatmaps). Further per-cell panels (modulation index, split-half stability) wait until they are
drawn at final size and the YrA check is done (VS, 2026-10-14).

Every panel is placed from its CURRENT output file, so rerunning the pipeline and then this rebuilds
the figure. Panels a (injection schematic) and b (mean field of view, Minian footprints) are cropped
from the earlier Illustrator figure; the pipeline does not produce them. A missing file is an error.

The pipeline draws panels c-k in Nature style at exactly their box sizes (panel_figsize), so the
composite places them 1:1, nothing rescaled: the SVG nests each panel's SVG (editable text) and the
PNG pastes each panel's 600-dpi PNG. A panel file of the wrong size -- i.e. drawn before the
2026-10-14 restyle -- is an error: rerun its section.

    from caban.figure2_composite import build_figure2_composite
    build_figure2_composite(cfg.PLOTS_DIR, out_dir)          # fig2.png (600 dpi) + fig2.svg
"""

import base64
import io
import os
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

# Panels not produced by the pipeline (VS's earlier Illustrator figure and its Minian viewer exports).
STATIC_DIR = os.path.expanduser('~/Documents/DB/1-McHugh postdoc/3-PAPER/paper_plots/fig2')
SCHEMATIC_TIFF = 'fig2.tif'
# Boxes (left, top, right, bottom) in fig2.tif (300 dpi, 2078 x 2702 px), found from the whitespace
# around each panel, without the old capital letters (2026-10-14): a, the virus list and injection
# schematic; b, the mean field of view and the Minian footprints.
STATIC_CROPS = {'a': (33, 5, 463, 237), 'b': (524, 5, 1021, 261)}

# Panel letter -> file under PLOTS_DIR (the .png and the .svg beside it).
PANEL_FILES = {
    'c': 'plot_sample_traces/plot_sample_traces.png',
    'd': 'sp_rates_lmm/TFC_cond/manipulation_check.png',
    'e': 'sp_rates_lmm/TFC_cond/decomposition.png',
    'f': 'sp_rates_lmm/TFC_cond/width_height_matched_examples_vertical.png',
    'g': 'sp_rates_lmm/paper/tfc_amplitude_rate/tfc_amplitude_rate_by_epoch.png',
    'h': 'sp_rates_lmm/TFC_cond/amplitude_ecdf.png',
    'i': 'sp_rates_lmm/TFC_cond/epoch_profile.png',
    'j': 'sp_rates_lmm/paper/recall/Test_B/testb_amplitude_rate_modulation.png',
    'k': 'epoch_modulation/YrA/event_proximal/panel_M_event_aligned_traces.png',
}

FIG_WIDTH_MM = 183.0          # Nature double column
# Panel boxes (x, y from the top, width, height) in mm. The pipeline draws panels c-k at exactly
# these sizes (panel_figsize), so their 7-pt text is 7 pt in the composite.
LAYOUT_MM = {
    'a': (0, 0, 46, 32),
    'b': (48, 0, 52, 32),
    'c': (102, 0, 81, 32),
    'd': (0, 35, 58, 38),
    'e': (60, 35, 123, 38),
    'f': (0, 76, 24, 74),
    'g': (26, 76, 78, 74),
    'h': (106, 76, 36, 36),
    'i': (144, 76, 39, 36),
    'j': (106, 115, 77, 35),
    'k': (0, 153, 183, 46),
}


# Each box's top LETTER_BAND_MM holds the panel letter; the panel itself fills the rest.
LETTER_BAND_MM = 3.2


def content_box(letter, layout=None):
    """(x, y, width, height) in mm of the panel itself, below its letter band."""
    x, y, w, h = (LAYOUT_MM if layout is None else layout)[letter]
    return (x, y + LETTER_BAND_MM, w, h - LETTER_BAND_MM)


def panel_figsize(letter, layout=None):
    """(width, height) in inches of Figure 2 panel ``letter``, for the function that draws it."""
    _, _, w, h = content_box(letter, layout)
    return (w / 25.4, h / 25.4)


LETTER_PT = 8
DPI_PNG = 600
LETTER_FONT = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
SIZE_TOLERANCE_PX = 3

# Written beside the figure on every build. No numbers: those live in docs/paper/paper_figure2.md and
# the stats files, which a rerun updates.
CAPTION = """**Fig. 2 | Chemogenetic modulation of SST interneurons dissociates the magnitude and the frequency of
CA1 pyramidal calcium events during trace fear conditioning.**
**a**, Viral strategy: SomaGCaMP6f with Cre-dependent hM3D(Gq) (Exc), hM4D(Gi) (Inh) or mCherry (Ctl) in
dorsal CA1 of SST-Cre mice, imaged through a GRIN lens. **b**, Mean field of view (left) and Minian
spatial footprints (right) from one animal. **c**, Example traces from three cells per group: calcium
signal (grey), deconvolved activity (colour) and detected events (circles); each trace is scaled to
its own maximum. Scale bar, 5 s. **d**, Manipulation check across the drug-free (LT1) and CNO (LT2)
linear-track sessions: within-cell change in log per-event amplitude (left) and the fraction of LT1
cells not re-registered in LT2 (right). Boxes show median and quartiles; points are animals. **e**,
Decomposition of trace-interval activity into the fraction of active cells, the event rate among
active cells, the event rate over all cells (population event rate), per-event amplitude (log) and
amplitude rate; small points are cells, large points animals. **f**, Peak-height-matched example
events, one per group; shading marks the detected run, the dotted line the detection threshold.
**g**, Per-event amplitude (top) and population event rate (bottom) in the three 20-s epochs; small
points are cells, large points animals. **h**, Cumulative distribution of trace-interval per-cell log
amplitude, per group (thick) and per animal (thin). **i**, Log amplitude across five conditioning
epochs: animals (thin lines), group means (thick lines, markers) and s.e.m. (bands); descriptive.
**j**, 48-h recall (Test B): per-animal change from pre-tone to post-tone in log per-event amplitude
(left) and log population event rate (right); dashed line, no change; brackets show unadjusted
model-derived contrasts. **k**, Event-aligned activity (YrA, standardized once per cell over the session, minus each trial's
pre-tone mean) around tone onset, trace onset (tone offset) and shock onset, all retained trials:
mean (lines) and s.e.m. (bands) across animals; grey, the windows analysed: 3 s from tone and
trace onset; on Shock, the 2-s shock (darker) and the 3 s from shock offset (lighter; dashed black
line, shock offset); dashed grey line, zero.
Brackets in d and e: Welch tests on animal values, Holm-corrected over the two
treatment-versus-control comparisons (Exc versus Inh uncorrected); in g: contrasts of the group x
epoch mixed model with animal as random intercept, Holm-corrected over the two treatment-versus-
control comparisons within each epoch. * P < 0.05, ** P < 0.01, *** P < 0.001. n = 5 Exc, 6 Inh and
6 Ctl animals; the animal is the unit of inference throughout.
"""


# Empty boxes for panels a layout places before they are built (drawn and printed, never silent).
PENDING_FILL = '#f2f2f2'
PENDING_EDGE = '#999999'
FONT_PT_PENDING = 6


def _pending_label(letter, spec, w_mm):
    """``"<what>\n(not built yet)"``, wrapped to the box width."""
    words, lines, line = spec['pending'][letter].split(), [], ''
    max_chars = max(12, int(w_mm / (FONT_PT_PENDING * 25.4 / 72 * 0.55)))
    for word in words:
        if line and len(line) + 1 + len(word) > max_chars:
            lines.append(line)
            line = word
        else:
            line = f'{line} {word}'.strip()
    lines.append(line)
    return '\n'.join(lines + ['(not built yet)'])


# The current figure (VS's PowerPoint draft, 2026-08-24, with k from 2026-10-15).
FIGURE2_SPEC = {'name': 'fig2', 'layout': LAYOUT_MM, 'panel_files': PANEL_FILES,
                'caption': CAPTION, 'pending': {}}


# The redesign (plans/figure2_redesign_plan.md, agreed 2026-10-21): SST cells, pyramidal cells before
# and after CNO, the double dissociation during conditioning, redone example events, and the shock.
# Panels reused from the current figure keep their box sizes, so their files place 1:1 unchanged.
LAYOUT_V2_MM = {
    'a': (0, 0, 46, 32),
    'b': (48, 0, 52, 32),
    'c': (102, 0, 81, 32),
    'd': (0, 35, 123, 40),
    'e': (125, 35, 58, 38),         # = the current figure's d box: the LT1 -> LT2 check places 1:1
    'f': (0, 78, 78, 74),
    'g': (80, 78, 103, 74),
    'h': (0, 155, 40, 46),
    'i': (42, 155, 141, 46),
}
PANEL_FILES_V2 = {
    'c': PANEL_FILES['c'],
    'e': PANEL_FILES['d'],          # LT1 -> LT2 within-cell amplitude change and dropout (VS, 2026-10-21)
    'f': PANEL_FILES['g'],          # the primary model, per-event amplitude and population rate by epoch
    'h': 'sp_rates_lmm/TFC_cond/group_median_examples_vertical.png',   # each group's own median event
    'i': 'shock_events/shock_locked_events.png',   # caban.shock_events
}
# Move a letter from here to PANEL_FILES_V2 when its section draws it at its box size.
PENDING_V2 = {
    'd': 'SST cells (DIO-GCaMP6f): event rate and amplitude before vs after CNO',
    'g': 'Double dissociation: per-animal amplitude vs rate change; 2 x 2 effects; other sessions',
}
CAPTION_V2 = """**Fig. 2 (redesign, in progress) | Chemogenetic excitation and inhibition of SST interneurons
dissociate the magnitude and the frequency of CA1 pyramidal calcium events.**
**a**, Viral strategy. **b**, Mean field of view and Minian spatial footprints from one animal.
**c**, Example traces from three cells per group. **d**, SST cells: activity before and after CNO
(not built). **e**, Manipulation check across the drug-free (LT1) and CNO (LT2) linear-track sessions:
within-cell change in log per-event amplitude (left) and the fraction of LT1 cells not re-registered
in LT2 (right).
**f**,
Per-event amplitude (top) and population event rate (bottom) in the three 20-s conditioning epochs;
small points are cells, large points animals; contrasts of the group x epoch mixed model with animal
as random intercept, Holm-corrected over the two treatment-versus-control comparisons within each
epoch. **g**, Double dissociation (not built). **h**, Example trace-interval events, one per group: the event whose per-event amplitude is
closest to that group's own median, on a common scale; shading marks the detected run, the dotted
line the detection threshold. **i**, Pyramidal events around
shock onset, 1-s bins: event rate per cell (left) and log per-event amplitude (right); mean (lines)
and s.e.m. (bands) across animals; darker shading, the 2-s shock; dashed line, shock offset. n = 5 Exc, 6 Inh and 6 Ctl animals; the animal is the unit of inference
throughout. Plan: plans/figure2_redesign_plan.md.
"""
FIGURE2_V2_SPEC = {'name': 'fig2_v2', 'layout': LAYOUT_V2_MM,
                             'panel_files': PANEL_FILES_V2,
                             'caption': CAPTION_V2, 'pending': PENDING_V2}
for _spec in (FIGURE2_SPEC, FIGURE2_V2_SPEC):
    _placed = set(_spec['panel_files']) | set(_spec['pending']) | set(STATIC_CROPS)
    if _placed != set(_spec['layout']) or set(_spec['panel_files']) & set(_spec['pending']):
        raise RuntimeError(f"{_spec['name']}: every layout panel must be exactly one of static, a "
                           f"file or pending")


SVG_NS = 'http://www.w3.org/2000/svg'
XLINK_NS = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', XLINK_NS)


def _px(mm):
    return int(round(mm / 25.4 * DPI_PNG))


def _static_panels(static_dir):
    path = os.path.join(static_dir, SCHEMATIC_TIFF)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'figure 2 composite: static panel source {path} not found')
    tif = Image.open(path).convert('RGB')
    return {letter: tif.crop(box) for letter, box in STATIC_CROPS.items()}


def _panel_paths(plots_dir, letter, panel_files):
    root = os.path.splitext(os.path.join(plots_dir, panel_files[letter]))[0]
    paths = (root + '.png', root + '.svg')
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f'figure 2 composite: panel {letter} source {p} not found; run '
                                    f'its pipeline section first')
    return paths


def _check_size(letter, image, layout):
    _, _, w, h = content_box(letter, layout)
    want = (_px(w), _px(h))
    if abs(image.size[0] - want[0]) > SIZE_TOLERANCE_PX or abs(image.size[1] - want[1]) > SIZE_TOLERANCE_PX:
        raise ValueError(f'figure 2 composite: panel {letter} is {image.size} px, its box is {want} px '
                         f'at {DPI_PNG} dpi: the file predates the Nature-style restyle (2026-10-14); '
                         f'rerun its section')


def _fit(image, w_mm, h_mm):
    """A static (raster) panel scaled to fit its box, top-left aligned."""
    bw, bh = _px(w_mm), _px(h_mm)
    scale = min(bw / image.size[0], bh / image.size[1])
    return image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
                        Image.LANCZOS)


def _build_png(plots_dir, static, out_path, height_mm, spec):
    layout = spec['layout']
    page = Image.new('RGB', (_px(FIG_WIDTH_MM), _px(height_mm)), 'white')
    font = ImageFont.truetype(LETTER_FONT, size=round(LETTER_PT / 72 * DPI_PNG))
    pending_font = ImageFont.truetype(LETTER_FONT.replace(' Bold', ''), size=round(FONT_PT_PENDING / 72 * DPI_PNG))
    draw = ImageDraw.Draw(page)
    for letter in layout:
        x, y, w, h = content_box(letter, layout)
        if letter in spec['pending']:
            draw.rectangle((_px(x), _px(y), _px(x + w), _px(y + h)), fill=PENDING_FILL, outline=PENDING_EDGE,
                           width=_px(0.25))
            draw.multiline_text((_px(x + w / 2), _px(y + h / 2)), _pending_label(letter, spec, w),
                                fill='black', font=pending_font, anchor='mm', align='center')
            continue
        if letter in static:
            img = _fit(static[letter], w, h)
        else:
            img = Image.open(_panel_paths(plots_dir, letter, spec['panel_files'])[0]).convert('RGB')
            _check_size(letter, img, layout)
        page.paste(img, (_px(x), _px(y)))
    for letter, (x, y, _, _) in layout.items():
        draw.text((_px(x), _px(y)), letter, fill='black', font=font)
    page.save(out_path, dpi=(DPI_PNG, DPI_PNG))


def _prefixed_panel_svg(path, prefix):
    """A panel's SVG root, every id prefixed so panels nested in one document cannot collide."""
    root = ET.parse(path).getroot()
    ids = {el.get('id') for el in root.iter() if el.get('id')}
    href = f'{{{XLINK_NS}}}href'
    for el in root.iter():
        if el.get('id'):
            el.set('id', prefix + el.get('id'))
        for attr, value in list(el.attrib.items()):
            if attr in (href, 'href') and value.startswith('#') and value[1:] in ids:
                el.set(attr, '#' + prefix + value[1:])
            elif 'url(#' in value:
                for i in ids:
                    value = value.replace(f'url(#{i})', f'url(#{prefix}{i})')
                el.set(attr, value)
    return root


def _image_element(image, x, y, w, h):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    data = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    el = ET.Element(f'{{{SVG_NS}}}image', {'x': f'{x}', 'y': f'{y}', 'width': f'{w}', 'height': f'{h}',
                                           'preserveAspectRatio': 'xMinYMin meet'})
    el.set(f'{{{XLINK_NS}}}href', data)
    return el


def _build_svg(plots_dir, static, out_path, height_mm, spec):
    layout = spec['layout']
    # User units are millimetres; each nested panel keeps its own viewBox (points).
    doc = ET.Element(f'{{{SVG_NS}}}svg', {'version': '1.1', 'width': f'{FIG_WIDTH_MM}mm',
                                          'height': f'{height_mm}mm',
                                          'viewBox': f'0 0 {FIG_WIDTH_MM} {height_mm}'})
    ET.SubElement(doc, f'{{{SVG_NS}}}rect', {'width': '100%', 'height': '100%', 'fill': 'white'})
    pending_size = FONT_PT_PENDING * 25.4 / 72
    for letter in layout:
        x, y, w, h = content_box(letter, layout)
        if letter in spec['pending']:
            ET.SubElement(doc, f'{{{SVG_NS}}}rect', {'x': f'{x}', 'y': f'{y}', 'width': f'{w}',
                                                       'height': f'{h}', 'fill': PENDING_FILL,
                                                       'stroke': PENDING_EDGE, 'stroke-width': '0.25'})
            lines = _pending_label(letter, spec, w).split('\n')
            for i, line in enumerate(lines):
                t = ET.SubElement(doc, f'{{{SVG_NS}}}text', {
                    'x': f'{x + w / 2}', 'y': f'{y + h / 2 + (i - (len(lines) - 1) / 2) * pending_size * 1.3}',
                    'font-family': 'Arial', 'font-size': f'{pending_size:.3f}', 'text-anchor': 'middle',
                    'dominant-baseline': 'middle', 'fill': 'black'})
                t.text = line
            continue
        if letter in static:
            doc.append(_image_element(static[letter], x, y, w, h))
            continue
        panel = _prefixed_panel_svg(_panel_paths(plots_dir, letter, spec['panel_files'])[1], f'p{letter}_')
        for attr, value in (('x', x), ('y', y), ('width', w), ('height', h)):
            panel.set(attr, f'{value}')
        doc.append(panel)
    letter_size = LETTER_PT * 25.4 / 72          # points to mm (user units)
    for letter, (x, y, _, _) in layout.items():
        t = ET.SubElement(doc, f'{{{SVG_NS}}}text', {
            'x': f'{x}', 'y': f'{y + letter_size * 0.8}', 'font-family': 'Arial',
            'font-weight': 'bold', 'font-size': f'{letter_size:.3f}', 'fill': 'black'})
        t.text = letter
    ET.ElementTree(doc).write(out_path, encoding='utf-8', xml_declaration=True)


def build_figure2_composite(plots_dir, out_dir, static_dir=STATIC_DIR, spec=None):
    """Write ``<name>.png`` (600 dpi), ``<name>.svg`` and ``<name>_caption.md`` to ``out_dir``.

    ``spec`` is a layout (FIGURE2_SPEC, the default, or FIGURE2_V2_SPEC)."""
    spec = FIGURE2_SPEC if spec is None else spec
    static = _static_panels(static_dir)
    height_mm = max(y + h for _, y, _, h in spec['layout'].values())
    os.makedirs(out_dir, exist_ok=True)
    name = spec['name']
    png, svg = os.path.join(out_dir, f'{name}.png'), os.path.join(out_dir, f'{name}.svg')
    _build_png(plots_dir, static, png, height_mm, spec)
    _build_svg(plots_dir, static, svg, height_mm, spec)
    with open(os.path.join(out_dir, f'{name}_caption.md'), 'w') as fh:
        fh.write(spec['caption'])
    print(f'{name} composite: {png} ({DPI_PNG} dpi), {svg}')
    for letter, what in spec['pending'].items():
        print(f'  panel {letter}: NOT BUILT YET (empty box) -- {what}')
    return png, svg

"""Build notebooks/crossreg_BASELINE.ipynb, the protected template of the cross-registration runner,
from the fork's ``cross-registration-WORKING.ipynb`` (plans/crossreg_batch_runner_plan.md §5).

    python scripts/build_crossreg_template.py

Which notebook: of the fork's four, ``cross-registration-WORKING.ipynb`` is the one that wrote the
published ``mappings_crossreg_<N>`` files (``-WORKING2`` writes ``mappings_minian__crossreg1``,
``-WORKING-RED`` is a G25 variant, ``cross-registration.ipynb`` is Minian's stock notebook).

Every edit is listed here and nothing else changes; all computing and saving cells are kept verbatim:

1. outputs and execution counts stripped; kernel set to ``minian-native``;
2. the parameter cell's one active ``dpath =`` line commented out (the runner inserts the scratch view);
3. the session-renaming cell: the hard-coded G17 fix (``new_session[2] = '918_26_30'``, which renames
   whatever session sorts third -- TFC_cond in grouping 1, a different session in larger groupings) is
   replaced by an explicit map ``SESSION_NAMES`` {stored session name: folder start time}, which the
   runner sets from the registry (VS, 2026-10-11);
4. the interactive AlignViewer cell and the June 2026 figure-export cell (needs a browser driver,
   writes SVG/PNG into the working directory), with their two markdown cells, and the trailing empty
   cell, removed;
5. the sessions opened without their ``unit_labels`` coordinate (``open_minian``'s ``post_process``
   hook), and the display-only cell that printed it removed: only some outputs carry it, and xarray 0.17
   (the pinned ``minian-native`` env) refuses to concatenate such datasets. The published runs used a
   newer xarray (the ``minian`` env, since deleted), which fills the missing coordinate. Cross-
   registration uses only ``A`` and ``max_proj``;
6. a markdown cell at the top saying what this file is.

The result is made read-only; the runner checks the md5 of its code (crossreg_runner.TEMPLATE_SOURCE_MD5).
"""
import os
import stat
import sys

import nbformat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import minian_runner as mr

SOURCE = os.path.join(mr.MINIAN_FORK_DIR, "cross-registration-WORKING.ipynb")
TARGET = os.path.join(mr.REPO_DIR, "notebooks", "crossreg_BASELINE.ipynb")

ACTIVE_DPATH = 'dpath = "/Users/vsekulic/data/vsekulic/OF_test/G10-ST703_hM3D"'
G17_FIX = """# Fix out of order sessions in G17
if 'G17' in dpath and TFC_cond:
    new_session[2] = np.array('918_26_30')
"""
RENAMING_LOOP = """new_session = []
for i in range(len(minian_ds['session'])):
    new_session.append(np.array(str(minian_ds['session'][i].data).split('-')[0]))
"""
RENAMING_BY_REGISTRY = """# Session names from the registry (caban.crossreg_runner sets SESSION_NAMES in the parameter cell):
# each session is named by its folder's start time. Replaces the split on '-' and the hard-coded
# G17 fix, whose output folder stores '918_26_30-TFC_cond' (the server's misnamed folder).
new_session = [np.array(SESSION_NAMES[str(s)]) for s in minian_ds['session'].values]
"""
OPEN_CALL = """minian_ds = open_minian_mf(
    dpath, id_dims, pattern=f_pattern_prefix+f_pattern)"""
OPEN_WITHOUT_UNIT_LABELS = """# caban: unit_labels dropped -- only some outputs carry it, and xarray 0.17 cannot concatenate those
minian_ds = open_minian_mf(
    dpath, id_dims, pattern=f_pattern_prefix+f_pattern,
    post_process=lambda ds, path: ds.drop_vars("unit_labels", errors="ignore"))"""
REMOVED_CELL_STARTS = (
    "len(minian_ds['A'][1].unit_labels.unit_id)",
    "## visualize mappings",
    "We visualize the matching of cells",
    "hv.output(size=int(output_size * 0.7))\nalnviewer = AlignViewer(",
    "import importlib\nimport bokeh",
)
HEADER = """# Cross-registration: protected template (caban)

Built by `scripts/build_crossreg_template.py` from the fork's `cross-registration-WORKING.ipynb`; the
edits are listed in that script. Run only through `caban.crossreg_runner`, which sets `dpath`,
`f_pattern` and `SESSION_NAMES` (plans/crossreg_batch_runner_plan.md §5). Do not edit: the runner
checks the md5 of this notebook's code.
"""


def replace_once(source: str, old: str, new: str, what: str) -> str:
    if source.count(old) != 1:
        raise ValueError("{}: expected exactly one occurrence, found {}".format(what, source.count(old)))
    return source.replace(old, new)


def main() -> int:
    nb = nbformat.read(SOURCE, as_version=4)
    cells = []
    removed = 0
    for cell in nb.cells:
        if cell.source.startswith(REMOVED_CELL_STARTS) or (cell.cell_type == "code" and not cell.source.strip()):
            removed += 1
            continue
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            if ACTIVE_DPATH in cell.source:
                cell.source = replace_once(cell.source, "\n" + ACTIVE_DPATH, "\n#" + ACTIVE_DPATH, "active dpath")
            if G17_FIX in cell.source:
                cell.source = replace_once(cell.source, G17_FIX, "", "G17 fix")
                cell.source = replace_once(cell.source, RENAMING_LOOP, RENAMING_BY_REGISTRY, "renaming loop")
            if OPEN_CALL in cell.source:
                cell.source = replace_once(cell.source, OPEN_CALL, OPEN_WITHOUT_UNIT_LABELS, "open call")
        cells.append(cell)
    if removed != len(REMOVED_CELL_STARTS) + 1:
        raise ValueError("removed {} cells, expected {}".format(removed, len(REMOVED_CELL_STARTS) + 1))
    if not any("SESSION_NAMES" in c.source for c in cells) or not any("post_process" in c.source for c in cells):
        raise ValueError("the session-renaming or the open cell was not found")
    if any(c.cell_type == "code" and "\ndpath =" in "\n" + c.source for c in cells):
        raise ValueError("an active dpath line is left")
    nb.cells = [nbformat.v4.new_markdown_cell(HEADER)] + cells
    nb.metadata["kernelspec"] = {"display_name": "minian-native", "language": "python", "name": mr.KERNEL_NAME}
    if os.path.exists(TARGET):
        os.chmod(TARGET, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    nbformat.write(nb, TARGET)
    os.chmod(TARGET, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("wrote {} ({} cells, read-only); code md5 {}".format(TARGET, len(nb.cells), mr.template_source_md5(nb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

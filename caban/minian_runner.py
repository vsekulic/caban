"""Run the Minian pipeline notebook itself over many sessions.

`plans/minian_batch_runner_plan.md`. The runner replaces the human loop -- uncomment
the next ``dpath``, run all, mark it ``DONE`` -- and nothing else: each session gets an
edited copy of the protected template (:data:`TEMPLATE_PATH`) executed top to bottom
by papermill in the ``minian-native`` kernel. Nothing here re-implements a pipeline
step, because later cells override the parameter cell (§2) and a re-implementation is
one missed line from silently different output.

What the copy changes is only what VS changed by hand, all in the parameter cell
(§4): the ``dpath`` line and four flags. One read-only cell is appended that writes
the effective parameters from the live kernel to ``minian_run/parameters.json``.

One session (§5): set existing output aside and link ``minian_intermediate`` onto
scratch; execute, sampling memory; report from the saved ``minian/``; re-encode the
notebook's two videos in place; delete the scratch; write ``minian_run/run.json``,
which is also the queue's sidecar (`caban.session_queue`).
"""

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional

import dask
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import papermill
import psutil
from IPython.display import Image, Markdown, display
from nbconvert import HTMLExporter

from caban import session_queue as sq
from caban import yra_recompute as yr

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The protected template (§3): an output-free copy of pipeline-WORKING-TFC_cond-4,
# read-only on disk, tracked in git, and checked by the md5 of its joined cell sources.
TEMPLATE_PATH = os.path.join(REPO_DIR, "notebooks", "minian_pipeline_BASELINE.ipynb")
TEMPLATE_SOURCE_MD5 = "5eb502c4a0252de26fc6170626d8ad00"
# The first cell imports `minian` before `sys.path.append`, so the fork is the cwd.
MINIAN_FORK_DIR = os.path.expanduser("~/code/minian_vsekulic")
KERNEL_NAME = "minian-native"
DEFAULT_SCRATCH_ROOT = "/Volumes/FUTROLA/minian_scratch"

RUN_DIR_NAME = sq.RUN_DIR_NAME
SIDECAR_NAME = os.path.join(RUN_DIR_NAME, "run.json")
OUTPUT_NAME = "minian"
EXECUTED_NOTEBOOK = "pipeline.ipynb"
EXECUTED_HTML = "pipeline.html"
PAPERMILL_LOG = "papermill.log"
PARAMETERS_JSON = "parameters.json"
MEMORY_CSV = "memory.csv"
README_NAME = "README.md"

# §4. The parameter cell is the one that derives `minian_ds_path` from `dpath`.
PARAMETER_CELL_ANCHOR = 'minian_ds_path = os.path.join(dpath, "minian")'
NOTEBOOK_FLAGS = {
    "interactive": False,        # gates viewers, previews and the parameter grid searches
    "interactive_CNMF": False,   # gates CNMFViewer and the unit_labels assignment (§4.1)
    "want_mc_video": True,
    "want_final_video": True,
}
RECORDED_FLAGS = tuple(NOTEBOOK_FLAGS) + ("interactive_noparam",)
# Versions recorded from inside the kernel; opencv and minian carry no dist metadata
# there, so they are read from the modules themselves.
KERNEL_PACKAGES = (
    "numpy", "scipy", "xarray", "dask", "distributed", "zarr", "pandas", "scikit-image",
    "scikit-learn", "cvxpy", "ecos", "scs", "osqp", "networkx", "pymetis", "pyfftw",
    "numba", "SimpleITK", "holoviews", "bokeh",
)

# Scope (parent plan §2): G05-G21. Stubs -- bare-timestamp session names and these
# types -- are recording mistakes to eyeball before queueing, never queued by default.
MOUSE_RANGE = ("G05", "G21")
STUB_SESSION_TYPES = ("iso", "HC1a", "HC2b", "LT1a")
# Parent plan §6.1: the four sessions that kept all 27 intermediate arrays.
GATE_LABELS = (
    "G06/2021_10_18-TFC_cond/09_52_24-HC1",
    "G06/2021_10_18-TFC_cond/10_34_09-CNO1",
    "G06/2021_10_18-TFC_cond/10_49_52-CNO2",
    "G06/2021_10_18-TFC_cond/11_26_52-HC2",
)

# §7: same pixel dimensions, compressed only.
NOTEBOOK_VIDEOS = ("minian.mp4", "minian_mc.mp4")
VIDEO_CRF = 23
VIDEO_PRESET = "slow"

MEMORY_SAMPLE_INTERVAL_S = 5
PROGRESS_PRINT_INTERVAL_S = 600

# Figure colours: categorical slot 1 for the fitted trace, muted ink for its context.
COLOR_C = "#2a78d6"
COLOR_CONTEXT = "#8f8e88"

VIDEO_DESCRIPTIONS = {
    "minian.mp4": (
        "2x2, written by the notebook's `generate_videos` (baseline cell 287), "
        "re-encoded in place at crf {crf}.\n"
        "- **top-left** raw movie (`varr`)\n"
        "- **top-right** CNMF input (`Y_fm_chk`: denoised, background-removed, "
        "motion-corrected; x255/max, x1.5 gain)\n"
        "- **bottom-left** residual `Y - A.C`\n"
        "- **bottom-right** reconstruction `A.C` (footprints x denoised traces), scaled "
        "to `Y` by a least-squares factor fitted on 200 random frames\n"
        "No `YrA` panel."
    ),
    "minian_mc.mp4": (
        "1x2, written by the notebook's `write_video` (baseline cell 99), re-encoded in "
        "place at crf {crf}.\n"
        "- **left** `varr_ref` (denoised + background-removed, *before* motion "
        "correction)\n"
        "- **right** `Y_fm_chk` (the same, *after* motion correction)"
    ),
}


# --- the template and the copy that runs ------------------------------------


def template_source_md5(nb) -> str:
    return hashlib.md5("".join(c.source for c in nb.cells).encode()).hexdigest()


def load_template(path: str = TEMPLATE_PATH):
    """Read the protected template, refusing if it is writable or its code has changed."""
    if os.access(path, os.W_OK):
        raise PermissionError(
            "{} is writable; the template must stay read-only (chmod a-w)".format(path))
    nb = nbformat.read(path, as_version=4)
    md5 = template_source_md5(nb)
    if md5 != TEMPLATE_SOURCE_MD5:
        raise ValueError(
            "{} cell sources have md5 {}, expected {}: the template was edited. Inspect "
            "`git diff` before trusting any run from it.".format(path, md5, TEMPLATE_SOURCE_MD5))
    kernel = nb.metadata.get("kernelspec", {}).get("name")
    if kernel != KERNEL_NAME:
        raise ValueError("{} has kernel {!r}, expected {!r}".format(path, kernel, KERNEL_NAME))
    return nb


def parameters_cell_source(parameters_path: str) -> str:
    """The appended cell: effective parameters as the kernel holds them at the end (§4)."""
    return '''\
# Appended by caban.minian_runner (plans/minian_batch_runner_plan.md §4). Records the
# effective parameters as the kernel holds them after every cell has run; later cells
# override the parameter cell, so the notebook text is not a reliable record. Read-only.
import json as _json
import sys as _sys
from importlib import metadata as _metadata

import cv2 as _cv2
import minian as _minian


def _jsonable(value):
    if isinstance(value, dict):
        return {{str(k): _jsonable(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


_packages = {{p: _metadata.version(p) for p in {packages!r}}}
_packages["opencv"] = _cv2.__version__
_record = {{
    "param": {{k: _jsonable(v) for k, v in sorted(globals().items()) if k.startswith("param_")}},
    "flags": {{k: globals()[k] for k in {flags!r}}},
    "FRAMERATE": FRAMERATE,
    "subset": _jsonable(subset),
    "subset_mc": _jsonable(subset_mc),
    "n_workers": n_workers,
    "dpath": dpath,
    "intpath": intpath,
    "python": _sys.version,
    "packages": _packages,
    "minian_file": _minian.__file__,
}}
with open({path!r}, "w") as _fh:
    _json.dump(_record, _fh, indent=2, sort_keys=True)
print("parameters written to", {path!r})
'''.format(packages=list(KERNEL_PACKAGES), flags=list(RECORDED_FLAGS), path=parameters_path)


def build_run_notebook(template, session_dir: str, parameters_path: str):
    """The copy that runs: ``dpath`` set, the §4 flags set, the parameters cell appended.

    Every edit must land exactly once, and the flags must be assigned nowhere but the
    parameter cell -- otherwise a later cell would silently undo the edit. Returns the
    notebook and the list of edits, which goes into the run record.
    """
    nb = copy.deepcopy(template)
    code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    anchored = [i for i in code if PARAMETER_CELL_ANCHOR in nb.cells[i].source]
    if len(anchored) != 1:
        raise ValueError("expected one parameter cell containing {!r}, found cells {}".format(
            PARAMETER_CELL_ANCHOR, anchored))
    param_idx = anchored[0]
    flag_assignment = re.compile(r"^\s*({})\s*=".format("|".join(NOTEBOOK_FLAGS)), re.M)
    elsewhere = [i for i in code if i != param_idx and flag_assignment.search(nb.cells[i].source)]
    if elsewhere:
        raise ValueError("cells {} also assign one of {}; the parameter-cell edit would be "
                         "overridden".format(elsewhere, list(NOTEBOOK_FLAGS)))

    lines = nb.cells[param_idx].source.split("\n")
    active_dpath = [ln for ln in lines if re.match(r"\s*dpath\s*=", ln)]
    if active_dpath:
        raise ValueError("the template's parameter cell has an active dpath line: {}".format(
            active_dpath))
    edits = []
    for name, value in NOTEBOOK_FLAGS.items():
        hits = [k for k, ln in enumerate(lines) if re.match(r"{}\s*=".format(name), ln)]
        if len(hits) != 1:
            raise ValueError("expected one `{} =` line in the parameter cell, found {}".format(
                name, len(hits)))
        new = "{} = {!r}".format(name, value)
        edits.append({"cell": param_idx, "was": lines[hits[0]], "now": new})
        lines[hits[0]] = new
    anchor_line = [k for k, ln in enumerate(lines) if ln.startswith(PARAMETER_CELL_ANCHOR)]
    if len(anchor_line) != 1:
        raise ValueError("anchor line not found exactly once in the parameter cell")
    dpath_line = "dpath = {!r}  # set by caban.minian_runner".format(os.path.abspath(session_dir))
    lines.insert(anchor_line[0], dpath_line)
    edits.append({"cell": param_idx, "was": None, "now": dpath_line})
    nb.cells[param_idx].source = "\n".join(lines)

    appended = nbformat.v4.new_code_cell(parameters_cell_source(parameters_path))
    appended.metadata["tags"] = ["caban-minian-runner"]
    nb.cells.append(appended)
    edits.append({"cell": len(nb.cells) - 1, "was": None, "now": "appended parameters cell"})
    return nb, edits


def _git_state(path: str) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", path] + list(args), capture_output=True,
                              text=True, check=True).stdout.strip()
    return {
        "path": path,
        "commit": git("rev-parse", "HEAD"),
        "modified_tracked_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
    }


# --- which sessions --------------------------------------------------------------


def session_type(session: str) -> str:
    """``09_52_24-HC1`` -> ``HC1``; a bare timestamp -> ``""``."""
    return session.split("-", 1)[1] if "-" in session else ""


def is_stub(item: sq.SessionWork) -> bool:
    kind = session_type(item.session)
    return kind == "" or kind in STUB_SESSION_TYPES


def is_never_processed(item: sq.SessionWork) -> bool:
    """No Minian output from before this runner: nothing, or only this runner's own run.

    A folder holding partial output (e.g. ``minian_intermediate`` without a complete
    ``minian/``) is not never-processed -- it needs a look, not a run.
    """
    c = item.candidate
    if c.complete_minian_dirs or c.set_aside_dirs:
        return False
    if os.path.isfile(os.path.join(item.session_dir, SIDECAR_NAME)):
        return True
    return not any(os.path.lexists(os.path.join(item.session_dir, n))
                   for n in sq.NOTEBOOK_OUTPUT_DIRS + sq.NOTEBOOK_OUTPUT_FILES)


def discover_sessions(roots=None, verbose: bool = True) -> List[sq.SessionWork]:
    """Every in-scope session with video. Selects no work -- see :func:`select_sessions`."""
    candidates = sq.scan_sessions(roots=roots, verbose=verbose)
    return [
        sq.SessionWork(candidate=c, minian_dir=None, output_dir=c.session_dir)
        for c in candidates
        if c.n_avi and MOUSE_RANGE[0] <= c.mouse <= MOUSE_RANGE[1]
    ]


def select_sessions(
    items: List[sq.SessionWork],
    labels=None,
    mice=None,
    session_types=None,
    path=None,
) -> List[sq.SessionWork]:
    """Choose the queue (§8).

    ``path`` names exact sessions -- ``<mouse>/<day>/<session>`` or the session's folder --
    in the order given, processed or not (a production session re-run is a gate run). It
    stands alone: with any other filter set it is an error.

    Otherwise the queue comes from the never-processed sessions; with nothing set, all of
    them. Each filter that is set must hold (AND); the values
    within one are alternatives (OR):

    - ``labels``: regex patterns searched in ``<mouse>/<day>/<session>`` --
      ``"track_day0"``, ``"2021_08_20"``, a full label, ``"HC[12]$"``;
    - ``mice``: e.g. ``["G05", "G08"]``;
    - ``session_types``: type prefixes, e.g. ``["HC"]`` (HC1-HC4) or ``["LT", "CNO"]``.

    Sessions that match but already have Minian output (production, or partial) are
    listed as not queued (parent plan §3.1: the processed ones are never re-run). Stubs
    are listed and left out. A single string stands for a one-item list.
    """
    labels, mice, session_types, path = ([v] if isinstance(v, str) else v
                                         for v in (labels, mice, session_types, path))
    if path:
        if labels or mice or session_types:
            raise ValueError("PATH stands alone; set LABELS, MICE and SESSION_TYPES to None")
        by_key = {i.label: i for i in items}
        by_key.update({os.path.normpath(i.session_dir): i for i in items})
        by_key.update({os.path.dirname(os.path.normpath(i.session_dir)): i for i in items})
        missing = [p for p in path if p.rstrip("/") not in by_key and os.path.normpath(p) not in by_key]
        if missing:
            raise KeyError("not found among {} scanned sessions: {}".format(len(items), missing))
        return [by_key.get(p.rstrip("/"), by_key.get(os.path.normpath(p))) for p in path]
    matching = items
    if labels:
        matching = [i for i in matching if any(re.search(p, i.label) for p in labels)]
    if mice:
        matching = [i for i in matching if i.mouse in mice]
    if session_types:
        matching = [i for i in matching
                    if any(re.match(p, session_type(i.session)) for p in session_types)]
    fresh = [i for i in matching if is_never_processed(i)]
    if labels or mice or session_types:
        have_output = [i for i in matching if not is_never_processed(i)]
        print("{} sessions match; {} already have Minian output, not queued{}".format(
            len(matching), len(have_output), ":" if have_output else "."))
        for item in have_output:
            print("  " + item.label)
    stubs = [i for i in fresh if is_stub(i)]
    print("{} never-processed sessions in this selection; {} of them are stubs, left out{}".format(
        len(fresh), len(stubs), ":" if stubs else "."))
    for stub in stubs:
        print("  " + stub.label)
    return [i for i in fresh if not is_stub(i)]


# --- status: run.json is the sidecar ------------------------------------------


def run_record(item: sq.SessionWork) -> dict:
    return sq.read_sidecar(item.output_dir, SIDECAR_NAME)


def run_status(item: sq.SessionWork) -> str:
    """``pending``, ``done``, ``failed``, ``interrupted``, or ``running``.

    ``running`` with nothing running means the driving kernel died mid-run; treat it
    as failed (:func:`clear_failed_run` with ``even_if_running=True``).
    """
    record = run_record(item)
    if not record:
        return "pending"
    if record["status"] == "done" and not sq.output_present(item, SIDECAR_NAME, OUTPUT_NAME):
        raise ValueError("{}: run.json says done but {}/ is missing".format(item.label, OUTPUT_NAME))
    return record["status"]


# How the queue table shows a status: nothing for a session never run, DONE as in the
# `# ... - DONE` marks of the hand-run notebooks; failures keep their names so they stand out.
STATUS_LABELS = {"pending": "", "done": "DONE"}


def unit_counts(minian_dir: str) -> str:
    """``"88 (S), 88 (C), 88 (YrA)"``, read from the saved arrays, for sanity checking.

    Equal counts can hide different unit sets (G10 TFC_cond: 799 each, but YrA holds
    343/346 where C holds 344/347), so then the shared count is added:
    ``"799 (S), 799 (C), 799 (YrA, 797 shared with C)"``.
    """
    ids = {n: yr.open_minian_array(minian_dir, n).coords["unit_id"].values
           for n in ("S", "C", "YrA")}
    text = "{} (S), {} (C), {} (YrA)".format(*(len(ids[n]) for n in ("S", "C", "YrA")))
    shared = len(set(ids["YrA"].tolist()) & set(ids["C"].tolist()))
    if shared != len(ids["C"]) or len(ids["YrA"]) != len(ids["C"]):
        text = text[:-1] + ", {} shared with C)".format(shared)
    return text


def _status_columns(item: sq.SessionWork, record: dict) -> dict:
    done = record.get("status") == "done"
    return {
        "type": session_type(item.session),
        "status": STATUS_LABELS.get(record.get("status", "pending"), record.get("status")),
        "wall_h": round(record["timings"]["total_s"] / 3600, 2) if "total_s" in record.get("timings", {}) else None,
        "peak_mem_gb": record.get("memory", {}).get("peak_rss_gb"),
        "units": unit_counts(os.path.join(item.session_dir, OUTPUT_NAME)) if done else "",
        "YrA_recomputed": "DONE" if done and yra_recomputed(item) else "",
    }


def yra_recomputed(item: sq.SessionWork) -> bool:
    """A complete recompute made from *this* run's ``minian/`` -- not production's."""
    sidecar = sq.read_sidecar(yra_output_dir(item), yr.SIDECAR_NAME)
    return bool(sidecar.get("complete")) and sidecar.get("minian_dir") == os.path.join(
        item.session_dir, OUTPUT_NAME)


def queue_status(items: List[sq.SessionWork]) -> pd.DataFrame:
    """One row per session, indexed mouse -> day -> session so each prints once.

    ``done`` from the shared queue frame is dropped: ``status`` says the same and more.
    """
    frame = sq.queue_frame(items, SIDECAR_NAME, OUTPUT_NAME, extra_columns=_status_columns)
    return frame.drop(columns="done").set_index(["mouse", "day", "session"]).sort_index()


def show_queue(status: pd.DataFrame) -> None:
    """The queue table, one table per mouse, so every mouse carries its own header."""
    for mouse in status.index.get_level_values("mouse").unique():
        display(Markdown("**{}**".format(mouse)))
        display(status.xs(mouse, level="mouse").fillna(""))


def pending_items(items: List[sq.SessionWork]) -> List[sq.SessionWork]:
    return [i for i in items if run_status(i) == "pending"]


# --- executing the notebook ------------------------------------------------------


def _process_tree(pid: int) -> List[psutil.Process]:
    root = psutil.Process(pid)
    return [root] + root.children(recursive=True)


def _tree_rss(pid: int) -> tuple:
    """(total RSS in bytes, process count) over papermill, its kernel and dask workers."""
    total, count = 0, 0
    for proc in _process_tree(pid):
        # A worker can exit between listing and reading; that is a race, not an error.
        try:
            total += proc.memory_info().rss
        except psutil.NoSuchProcess:
            continue
        count += 1
    return total, count


def _terminate_tree(pid: int) -> None:
    procs = _process_tree(pid)
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(procs, timeout=30)
    for proc in alive:
        proc.kill()


def _progress(log_path: str) -> str:
    """Papermill's last ``n/total`` cell count in its log.

    The whole log, not its tail: the notebook's ffmpeg writes its own progress into the
    same stream while a video encodes, burying papermill's last line.
    """
    with open(log_path, "rb") as fh:
        text = fh.read().decode(errors="replace")
    hits = re.findall(r"(\d+)/(\d+) \[", text)
    return "{}/{} cells".format(*hits[-1]) if hits else "starting"


def execute_notebook(nb, run_dir: str, n_workers: Optional[int] = None) -> dict:
    """Run ``nb`` with papermill in the ``minian-native`` kernel, sampling memory.

    Papermill runs as a child process so that its kernel and the dask workers form one
    process tree to measure, and so an interrupt can take the whole tree down. The
    executed notebook is saved after every cell, so progress and a failure's traceback
    are on disk as it runs. ``n_workers`` sets ``MINIAN_NWORKERS`` (§9); ``None`` keeps
    the notebook's own default.
    """
    out_path = os.path.join(run_dir, EXECUTED_NOTEBOOK)
    log_path = os.path.join(run_dir, PAPERMILL_LOG)
    env = dict(os.environ)
    if n_workers is not None:
        env["MINIAN_NWORKERS"] = str(n_workers)
    cmd = [sys.executable, "-m", "papermill", "-", out_path, "--kernel", KERNEL_NAME,
           "--cwd", MINIAN_FORK_DIR, "--start-timeout", "120"]
    samples = []
    started = time.time()
    last_print = started
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        try:
            proc.stdin.write(nbformat.writes(nb).encode())
            proc.stdin.close()
            while proc.poll() is None:
                rss, count = _tree_rss(proc.pid)
                samples.append((time.time() - started, rss, count,
                                psutil.virtual_memory().available))
                if time.time() - last_print >= PROGRESS_PRINT_INTERVAL_S:
                    last_print = time.time()
                    print("  {:6.1f} min  {}  rss {:.1f} GB over {} processes".format(
                        (last_print - started) / 60, _progress(log_path), rss / 1e9, count))
                time.sleep(MEMORY_SAMPLE_INTERVAL_S)
        except BaseException:
            _terminate_tree(proc.pid)
            raise
    trace = pd.DataFrame(samples, columns=["t_s", "rss_bytes", "n_processes", "available_bytes"])
    trace.to_csv(os.path.join(run_dir, MEMORY_CSV), index=False)
    result = {"returncode": proc.returncode, "execute_s": time.time() - started, "memory": {}}
    if trace.empty:
        # Only an immediate failure takes no sample; the caller reports it from the log.
        return result
    peak = trace.loc[trace["rss_bytes"].idxmax()]
    result["memory"] = {
            "peak_rss_gb": round(peak["rss_bytes"] / 1e9, 2),
            "peak_at_min": round(peak["t_s"] / 60, 1),
            "peak_n_processes": int(peak["n_processes"]),
            "min_available_gb": round(trace["available_bytes"].min() / 1e9, 2),
        "sample_interval_s": MEMORY_SAMPLE_INTERVAL_S,
        "n_samples": len(trace),
    }
    return result


def notebook_error(path: str) -> Optional[dict]:
    """The first error output in an executed notebook, with the cell that raised it."""
    nb = nbformat.read(path, as_version=4)
    for index, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                return {
                    "executed_cell_index": index,
                    "execution_count": cell.get("execution_count"),
                    "cell_head": "\n".join(cell.source.split("\n")[:3]),
                    "ename": output.ename,
                    "evalue": output.evalue,
                }
    return None


def export_html(nb_path: str, html_path: str) -> None:
    body, _ = HTMLExporter().from_filename(nb_path)
    with open(html_path, "w") as fh:
        fh.write(body)


# --- report (in this env, from the saved minian/) ---------------------------------


def report_session(minian_dir: str, run_dir: str, framerate: float) -> dict:
    """Cell count, footprint sizes, corr(C, YrA) and the summary figures (§5 step 4).

    Hard-fails unless ``A`` and ``S`` carry ``C``'s units in ``C``'s order, ``YrA``
    shares at least 90 % of them, and ``C``/``YrA`` share frames and are finite.
    ``YrA`` units missing from ``C``'s set, or extra, are recorded (see below).

    ``YrA`` may hold them in another order: the notebook reorders ``A`` to ``C``
    (cell 284) but saves ``YrA`` as ``compute_trace`` left it -- the misalignment
    ``sessions.Session._align_YrA_to_S_units`` repairs at load time. It is aligned by
    ``unit_id`` here, for the correlation only, and whether it matched is recorded.
    The saved arrays are never modified.
    """
    arrays = {n: yr.open_minian_array(minian_dir, n) for n in ("A", "C", "S", "YrA", "max_proj")}
    units = arrays["C"].coords["unit_id"].values
    for name in ("A", "S"):
        if not np.array_equal(arrays[name].coords["unit_id"].values, units):
            raise ValueError("{}: unit_id of {} differs from C's ({} vs {} units)".format(
                minian_dir, name, arrays[name].sizes["unit_id"], len(units)))
    yra_units = arrays["YrA"].coords["unit_id"].values
    # The notebook saves `YrA.sel(unit_id=mask)` with YrA sorted and `mask` in C's order,
    # which can select a few wrong units: G10 TFC_cond's production output and its re-run
    # both hold YrA units 343/346 in place of C's 344/347. Recorded, and the correlation
    # uses the shared units; caban's loader repairs this (_align_YrA_to_S_units).
    c_only = sorted(set(units.tolist()) - set(yra_units.tolist()))
    yra_only = sorted(set(yra_units.tolist()) - set(units.tolist()))
    shared = np.array([u for u in units if u not in set(c_only)])
    if len(shared) < 0.9 * len(units):
        raise ValueError("{}: YrA shares only {} of C's {} units".format(
            minian_dir, len(shared), len(units)))
    if not np.array_equal(arrays["C"].coords["frame"].values, arrays["YrA"].coords["frame"].values):
        raise ValueError("{}: C and YrA have different frames".format(minian_dir))
    C = arrays["C"].values
    YrA = arrays["YrA"].sel(unit_id=shared).values
    C_shared = arrays["C"].sel(unit_id=shared).values
    for name, values in (("C", C), ("YrA", YrA)):
        if not np.isfinite(values).all():
            raise ValueError("{}: {} holds non-finite values".format(minian_dir, name))
    A = arrays["A"]
    footprint_px = (A > 0).sum(["height", "width"]).compute().values
    corr = yr._per_cell_correlation(C_shared, YrA)

    figures = {
        "summary_footprints.png": _plot_footprints(arrays["max_proj"].values, A),
        "summary_traces.png": _plot_traces(C_shared, YrA, shared, framerate),
        "summary_distributions.png": _plot_distributions(footprint_px, corr),
    }
    for name, fig in figures.items():
        fig.savefig(os.path.join(run_dir, name), dpi=120, bbox_inches="tight")
        plt.close(fig)
    return {
        "n_units": int(len(units)),
        "n_frames": int(C.shape[1]),
        "duration_min": round(C.shape[1] / framerate / 60, 2),
        "height": int(A.sizes["height"]),
        "width": int(A.sizes["width"]),
        "n_empty_footprints": int((footprint_px == 0).sum()),
        "n_all_zero_C": int((C == 0).all(axis=1).sum()),
        "yra_unit_order_matches_C": bool(np.array_equal(yra_units, units)),
        "yra_missing_C_units": c_only,
        "yra_extra_units": yra_only,
        "footprint_px": yr._summary(footprint_px.astype(float)),
        "corr_C_YrA": yr._summary(corr),
        "figures": list(figures),
    }


def _plot_footprints(max_proj: np.ndarray, A) -> plt.Figure:
    peak = A.max(["height", "width"])
    overlay = (A / peak).max("unit_id").compute().values
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax in axes:
        ax.imshow(max_proj, cmap="gray")
        ax.set_axis_off()
    axes[0].set_title("max projection of Y_fm_chk", fontsize=10)
    axes[1].imshow(np.ma.masked_less(overlay, 0.2), cmap="Blues", vmin=0, vmax=1, alpha=0.8)
    axes[1].set_title("footprints, each scaled to its peak ({} units)".format(A.sizes["unit_id"]),
                      fontsize=10)
    return fig


def _plot_traces(C: np.ndarray, YrA: np.ndarray, units: np.ndarray, framerate: float,
                 n_shown: int = 8) -> plt.Figure:
    rng = np.random.default_rng(0)
    shown = np.sort(rng.choice(len(units), size=min(n_shown, len(units)), replace=False))
    minutes = np.arange(C.shape[1]) / framerate / 60
    fig, ax = plt.subplots(figsize=(12, 1 + 0.9 * len(shown)))
    for row, k in enumerate(shown):
        scale = max(np.abs(YrA[k]).max(), np.abs(C[k]).max())
        offset = -row * 1.2
        ax.plot(minutes, YrA[k] / scale + offset, color=COLOR_CONTEXT, lw=0.5,
                label="YrA (before temporal denoising)" if row == 0 else None)
        ax.plot(minutes, C[k] / scale + offset, color=COLOR_C, lw=1.0,
                label="C (denoised)" if row == 0 else None)
    ax.set_yticks([-row * 1.2 for row in range(len(shown))])
    ax.set_yticklabels(["unit {}".format(units[k]) for k in shown], fontsize=8)
    ax.set_xlabel("time (min)")
    ax.set_xlim(minutes[0], minutes[-1])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(1, 1.08))
    ax.set_title("{} random units, each scaled to its own max".format(len(shown)), fontsize=10,
                 loc="left")
    return fig


def _plot_distributions(footprint_px: np.ndarray, corr: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    panels = (
        (footprint_px, "footprint area (pixels > 0)", 40),
        (corr[np.isfinite(corr)], "per-unit corr(C, YrA)", np.linspace(-0.2, 1, 49)),
    )
    for ax, (values, label, bins) in zip(axes, panels):
        ax.hist(values, bins=bins, color=COLOR_C, edgecolor="white", linewidth=0.5)
        median = np.median(values)
        ax.axvline(median, color="black", lw=1, ls="--")
        ax.text(median, ax.get_ylim()[1] * 0.95, " median {:.3g}".format(median),
                fontsize=8, va="top")
        ax.set_xlabel(label)
        ax.set_ylabel("units")
        ax.spines[["top", "right"]].set_visible(False)
    return fig


# --- videos ------------------------------------------------------------------


def _decoded_frame_count(path: str) -> int:
    """Frames counted by decoding the whole stream -- proof the file is readable."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", path],
        capture_output=True, check=True)
    return int(json.loads(proc.stdout)["streams"][0]["nb_read_frames"])


def reencode_video(path: str, crf: int = VIDEO_CRF, preset: str = VIDEO_PRESET) -> dict:
    """Re-encode ``path`` in place at the same pixel dimensions (§7).

    The original is replaced only after the new file decodes in full with the same
    frame count and dimensions; on a mismatch the new file is left beside it.
    """
    before = yr.probe_video(path)
    tmp = path[:-len(".mp4")] + ".reencode.mp4"
    if os.path.lexists(tmp):
        raise FileExistsError("{} exists -- a leftover from an interrupted re-encode".format(tmp))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0",
                    "-c:v", "libx264", "-preset", preset, "-crf", str(crf), tmp], check=True)
    after = yr.probe_video(tmp)
    decoded = _decoded_frame_count(tmp)
    if (after["width"], after["height"]) != (before["width"], before["height"]) \
            or decoded != before["n_frames"]:
        raise ValueError("re-encode of {} does not match: before {}, after {} with {} decoded "
                         "frames; left at {}".format(path, before, after, decoded, tmp))
    bytes_before = os.path.getsize(path)
    os.replace(tmp, path)
    return {"width": before["width"], "height": before["height"], "n_frames": decoded,
            "crf": crf, "preset": preset, "mb_before": round(bytes_before / 1e6, 1),
            "mb_after": round(os.path.getsize(path) / 1e6, 1)}


# --- README -----------------------------------------------------------------------


def write_readme(item: sq.SessionWork, record: dict, parameters: dict) -> str:
    """``minian_run/README.md``: the run, each video panel by panel, every parameter."""
    run_dir = os.path.join(item.session_dir, RUN_DIR_NAME)
    report = record["report"]
    lines = [
        "# {}".format(item.label), "",
        "Minian CNMF-E output produced by `caban.minian_runner` "
        "(`plans/minian_batch_runner_plan.md`) on {}.".format(record["finished"]), "",
        "## Run", "",
        "| | |", "|---|---|",
        "| wall time | {:.2f} h |".format(record["timings"]["total_s"] / 3600),
        "| peak memory (papermill + kernel + dask workers, RSS) | {} GB |".format(
            record["memory"]["peak_rss_gb"]),
        "| dask workers | {} |".format(parameters["n_workers"]),
        "| units | {} |".format(report["n_units"]),
        "| frames | {} ({} min at {} fps) |".format(report["n_frames"], report["duration_min"],
                                                    parameters["FRAMERATE"]),
        "| median footprint | {:.0f} px |".format(report["footprint_px"]["median"]),
        "| median corr(C, YrA) | {:.3f} |".format(report["corr_C_YrA"]["median"]),
        "| `YrA.zarr` unit order matches `C`/`S` | {} (if not, align by `unit_id`) |".format(
            report["yra_unit_order_matches_C"]),
        "| `YrA.zarr` units missing / extra vs `C` | {} / {} (the notebook's "
        "`YrA.sel(unit_id=mask)` can pick wrong units) |".format(
            report["yra_missing_C_units"] or "none", report["yra_extra_units"] or "none"),
        "| template | `{}` (source md5 `{}`) |".format(
            os.path.relpath(record["template"]["path"], REPO_DIR), record["template"]["md5"]),
        "| minian fork | `{}` |".format(record["minian_fork"]["commit"]),
        "", "## Videos", "",
    ]
    for name in NOTEBOOK_VIDEOS:
        lines += ["### `{}`".format(name), "",
                  VIDEO_DESCRIPTIONS[name].format(crf=record["videos"][name]["crf"]), ""]
    lines += [
        "## Other files", "",
        "- `pipeline.ipynb` / `pipeline.html` -- the executed notebook, every figure",
        "- `summary_*.png` -- max projection and footprints, sample traces, distributions",
        "- `parameters.json` -- every effective parameter, read from the kernel at the end",
        "- `run.json` -- the full run record; `memory.csv` -- the memory trace; "
        "`papermill.log` -- papermill's own log",
        "",
        "`A`, `C`, `S`, `c0`, `b0` carry no `unit_labels` coordinate: the notebook assigns "
        "it only under `interactive_CNMF`, which is off (plan §4.1). Without manual "
        "curation it would equal `unit_id`.",
        "", "## Effective parameters", "",
        "| step | parameter | value |", "|---|---|---|",
    ]
    for step, values in sorted(parameters["param"].items()):
        for key, value in sorted(values.items()):
            lines.append("| `{}` | `{}` | `{}` |".format(step, key, json.dumps(value)))
    lines += ["", "Notebook edits made by the runner:", ""]
    lines += ["- cell {}: `{}` -> `{}`".format(e["cell"], e["was"], e["now"])
              for e in record["notebook_edits"]]
    path = os.path.join(run_dir, README_NAME)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --- one session ------------------------------------------------------------------


def _remove_scratch(session_dir: str, scratch: str) -> None:
    link = os.path.join(session_dir, sq.SCRATCH_DIR_NAME)
    if not os.path.islink(link) or os.readlink(link) != scratch:
        raise ValueError("{} is not the symlink to {} that preparation made".format(link, scratch))
    shutil.rmtree(scratch)
    os.unlink(link)


def yra_output_dir(item: sq.SessionWork) -> str:
    """Where :func:`recompute_yra` writes: the run's own ``minian/``, beside ``A``/``C``/``S``."""
    return os.path.join(item.session_dir, OUTPUT_NAME)


def recompute_yra(item: sq.SessionWork, n_workers: Optional[int] = None) -> dict:
    """Recompute ``YrA`` for this run's ``minian/`` with ``caban.yra_recompute`` (step 2).

    The notebook's own ``YrA.zarr`` can hold a few wrong units (``YrA.sel(unit_id=mask)``;
    see :func:`unit_counts`), so the same recompute the production sessions get is run
    here, unchanged: the movie is replayed from the ``.avi`` files and ``YrA`` computed
    against the saved ``A``/``C``/``b``/``f``, which puts ``S``'s ``unit_id`` on it by
    construction. Output: ``YrA_recomputed.zarr`` and its sidecar in ``minian/`` beside
    ``A``/``C``/``S`` (:func:`yra_output_dir`); ``minian/YrA.zarr`` is untouched and serves
    as the comparison.
    While the run's scratch still exists, the replayed movie is also checked against
    its saved ``Y_fm_chk``.
    """
    session_dir = item.session_dir
    minian_dir = os.path.join(session_dir, OUTPUT_NAME)
    saved_movie = os.path.join(session_dir, sq.SCRATCH_DIR_NAME, "Y_fm_chk.zarr")
    with dask.config.set(scheduler="threads", num_workers=n_workers or 6):
        done = yr.recompute_session_yra(
            session_dir,
            minian_dir,
            yra_output_dir(item),
            yr.notebook_del_frames(session_dir),
            existing_yra_path=os.path.join(minian_dir, "YrA.zarr"),
            saved_movie_path=saved_movie if os.path.isdir(saved_movie) else None,
        )
    return {"output": os.path.join(yra_output_dir(item), yr.OUTPUT_ARRAY_NAME),
            "written_utc": done["written_utc"], "report": done["report"]}


def run_session(
    item: sq.SessionWork,
    scratch_root: str = DEFAULT_SCRATCH_ROOT,
    keep_scratch: bool = False,
    n_workers: Optional[int] = None,
    reason: str = "caban.minian_runner",
    recompute_yra: bool = True,
) -> dict:
    """Prepare, execute, report, recompute YrA, re-encode, clean up, record (§5).

    ``recompute_yra`` runs :func:`recompute_yra` on the new output (see there). It writes
    into the run's own ``minian/``, so a gate re-run of a production session is safe too:
    production's recompute sits in its ``minian_crossreg*`` folder.

    ``keep_scratch`` keeps the intermediates -- for a gate run, which is compared
    against ``minian_intermediate-ORIG`` stage by stage. Any failure is recorded in
    ``run.json`` and re-raised; the session then shows as ``failed`` until
    :func:`clear_failed_run`.
    """
    status = run_status(item)
    if status != "pending":
        raise ValueError("{} is {}, not pending".format(item.label, status))
    template = load_template()
    if not os.path.isdir(MINIAN_FORK_DIR):
        raise FileNotFoundError("minian fork not found at {}".format(MINIAN_FORK_DIR))

    started = time.time()
    session_dir = item.session_dir
    run_dir = os.path.join(session_dir, RUN_DIR_NAME)
    set_aside = sq.set_aside_minian_output(session_dir, reason, scratch_root=scratch_root)
    os.makedirs(run_dir)
    parameters_path = os.path.join(run_dir, PARAMETERS_JSON)
    nb, edits = build_run_notebook(template, session_dir, parameters_path)
    record = {
        "label": item.label,
        "session_dir": session_dir,
        "status": "running",
        "complete": False,
        "started": pd.Timestamp.now().isoformat(timespec="seconds"),
        "template": {"path": TEMPLATE_PATH, "md5": TEMPLATE_SOURCE_MD5},
        "minian_fork": _git_state(MINIAN_FORK_DIR),
        "caban": _git_state(REPO_DIR),
        "kernel": KERNEL_NAME,
        "papermill": papermill.__version__,
        "minian_nworkers_env": n_workers,
        "notebook_edits": edits,
        "set_aside": set_aside,
        "scratch_kept": keep_scratch,
        "recompute_yra": recompute_yra,
        "n_workers": n_workers,
        "timings": {"prepare_s": round(time.time() - started, 1)},
    }
    sq.write_sidecar(session_dir, SIDECAR_NAME, record)
    try:
        executed = execute_notebook(nb, run_dir, n_workers=n_workers)
        record["memory"] = executed["memory"]
        record["timings"]["execute_s"] = round(executed["execute_s"], 1)
        nb_path = os.path.join(run_dir, EXECUTED_NOTEBOOK)
        log_path = os.path.join(run_dir, PAPERMILL_LOG)
        if not os.path.isfile(nb_path):
            with open(log_path) as fh:
                tail = fh.read()[-2000:]
            raise RuntimeError("papermill exited {} without writing a notebook:\n{}".format(
                executed["returncode"], tail))
        export_html(nb_path, os.path.join(run_dir, EXECUTED_HTML))
        if executed["returncode"] != 0:
            record["notebook_error"] = notebook_error(nb_path)
            raise RuntimeError("papermill exited {}: {}; see {}".format(
                executed["returncode"], record["notebook_error"],
                os.path.join(run_dir, PAPERMILL_LOG)))

        _finish_after_notebook(item, record, started)
    except BaseException as error:
        _record_failure(record, error, started)
        raise
    finally:
        sq.write_sidecar(session_dir, SIDECAR_NAME, record)
    return record


def _record_failure(record: dict, error: BaseException, started: float) -> None:
    record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    record["error"] = "{}: {}".format(type(error).__name__, error)
    record["timings"]["total_s"] = round(time.time() - started, 1)


def _finish_after_notebook(item: sq.SessionWork, record: dict, started: float) -> None:
    """Everything after a notebook that ran to the end: check, report, videos, clean up.

    ``started`` is the wall-clock origin of the run, for ``total_s``.
    """
    session_dir = item.session_dir
    run_dir = os.path.join(session_dir, RUN_DIR_NAME)
    with open(os.path.join(run_dir, PARAMETERS_JSON)) as fh:
        parameters = json.load(fh)
    wrong = {k: parameters["flags"][k] for k, v in NOTEBOOK_FLAGS.items()
             if parameters["flags"][k] != v}
    if wrong or parameters["dpath"] != os.path.abspath(session_dir):
        raise ValueError("kernel ended with flags {} / dpath {} -- the edits did not "
                         "hold".format(wrong, parameters["dpath"]))

    mark = time.time()
    record["report"] = report_session(os.path.join(session_dir, OUTPUT_NAME), run_dir,
                                      parameters["FRAMERATE"])
    record["timings"]["report_s"] = round(time.time() - mark, 1)

    # Records written before this step existed carry no flag; they did not run it.
    if record.get("recompute_yra", False):
        mark = time.time()
        record["yra_recompute"] = recompute_yra(item, n_workers=record.get("n_workers"))
        record["timings"]["yra_s"] = round(time.time() - mark, 1)

    mark = time.time()
    record["videos"] = {name: reencode_video(os.path.join(session_dir, name))
                        for name in NOTEBOOK_VIDEOS}
    record["timings"]["videos_s"] = round(time.time() - mark, 1)

    if not record["scratch_kept"]:
        _remove_scratch(session_dir, record["set_aside"]["scratch"])
    record["timings"]["total_s"] = round(time.time() - started, 1)
    record["finished"] = pd.Timestamp.now().isoformat(timespec="seconds")
    write_readme(item, record, parameters)
    record["status"] = "done"
    record["complete"] = True


def resume_after_notebook(item: sq.SessionWork) -> dict:
    """Finish a ``failed`` run whose notebook itself completed, without re-running it.

    For a failure in the steps after the notebook (report, videos, clean-up): refuses
    unless the executed notebook holds no error and ``parameters.json`` was written, so
    the CNMF-E output on disk is the notebook's complete output. The earlier error is
    kept in the record as ``resumed_from``.
    """
    record = run_record(item)
    if record.get("status") != "failed":
        raise ValueError("{} is {}, not failed".format(item.label, record.get("status", "pending")))
    run_dir = os.path.join(item.session_dir, RUN_DIR_NAME)
    nb_path = os.path.join(run_dir, EXECUTED_NOTEBOOK)
    if "notebook_error" in record or "execute_s" not in record["timings"] \
            or notebook_error(nb_path) is not None \
            or not os.path.isfile(os.path.join(run_dir, PARAMETERS_JSON)):
        raise ValueError("{}: the notebook did not complete; clear_failed_run and re-run "
                         "instead".format(item.label))
    started = time.time() - record["timings"]["prepare_s"] - record["timings"]["execute_s"]
    record["resumed_from"] = {"error": record.pop("error"),
                              "when": pd.Timestamp.now().isoformat(timespec="seconds")}
    record["status"] = "running"
    for key in ("report_s", "yra_s", "videos_s", "total_s"):
        record["timings"].pop(key, None)
    sq.write_sidecar(item.session_dir, SIDECAR_NAME, record)
    try:
        _finish_after_notebook(item, record, started)
    except BaseException as error:
        _record_failure(record, error, started)
        raise
    finally:
        sq.write_sidecar(item.session_dir, SIDECAR_NAME, record)
    return record


def clear_failed_run(item: sq.SessionWork, even_if_running: bool = False) -> str:
    """Return a failed session to ``pending``; the failed run's folder is kept.

    Removes what the run wrote into the session -- ``minian/``, the notebook videos,
    the scratch link and its folder -- and renames ``minian_run/`` to
    ``minian_run-failed-<timestamp>/`` so its notebook and traceback survive. Set-aside
    originals (``*-ORIG``) are not touched. Preparation guaranteed none of these existed
    before the run, so everything removed was the run's own.
    """
    record = run_record(item)
    allowed = ("failed", "interrupted") + (("running",) if even_if_running else ())
    if record.get("status") not in allowed:
        raise ValueError("{} is {}; only {} can be cleared".format(
            item.label, record.get("status", "pending"), allowed))
    session_dir = item.session_dir
    minian = os.path.join(session_dir, OUTPUT_NAME)
    if os.path.isdir(minian):
        shutil.rmtree(minian)
    for name in NOTEBOOK_VIDEOS:
        for path in (os.path.join(session_dir, name),
                     os.path.join(session_dir, name[:-len(".mp4")] + ".reencode.mp4")):
            if os.path.isfile(path):
                os.remove(path)
    scratch = record["set_aside"].get("scratch")
    if scratch and os.path.islink(os.path.join(session_dir, sq.SCRATCH_DIR_NAME)):
        _remove_scratch(session_dir, scratch)
    kept = os.path.join(session_dir, "{}-failed-{}".format(
        RUN_DIR_NAME, pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")))
    os.rename(os.path.join(session_dir, RUN_DIR_NAME), kept)
    return kept


# --- the queue ----------------------------------------------------------------


def format_report(record: dict) -> str:
    lines = ["{}: {}".format(record["label"], record["status"])]
    if "total_s" in record.get("timings", {}):
        lines.append("  wall {:.2f} h ({})".format(
            record["timings"]["total_s"] / 3600,
            ", ".join("{} {:.0f} s".format(k[:-2], v) for k, v in record["timings"].items()
                      if k != "total_s")))
    if "memory" in record:
        m = record["memory"]
        lines.append("  peak RSS {} GB at {} min over {} processes; min system available {} GB"
                     .format(m["peak_rss_gb"], m["peak_at_min"], m["peak_n_processes"],
                             m["min_available_gb"]))
    if "report" in record:
        r = record["report"]
        lines.append("  {} units x {} frames ({} min); footprint median {:.0f} px; "
                     "corr(C, YrA) median {:.3f}; empty footprints {}, all-zero C {}; "
                     "YrA unit order matches C: {}".format(
                         r["n_units"], r["n_frames"], r["duration_min"],
                         r["footprint_px"]["median"], r["corr_C_YrA"]["median"],
                         r["n_empty_footprints"], r["n_all_zero_C"],
                         r["yra_unit_order_matches_C"]))
        if r["yra_missing_C_units"] or r["yra_extra_units"]:
            lines.append("  YrA.zarr unit set differs from C's: missing {}, extra {} "
                         "(the notebook's YrA.sel(unit_id=mask))".format(
                             r["yra_missing_C_units"], r["yra_extra_units"]))
    for name, v in record.get("videos", {}).items():
        lines.append("  {}: {} -> {} MB (crf {})".format(name, v["mb_before"], v["mb_after"], v["crf"]))
    if "error" in record:
        lines.append("  error: {}".format(record["error"]))
    return "\n".join(lines)


def show_session(item: sq.SessionWork) -> None:
    """Attended mode: the report and the summary figures, inline."""
    record = run_record(item)
    print(format_report(record))
    for name in record.get("report", {}).get("figures", []):
        display(Image(os.path.join(item.session_dir, RUN_DIR_NAME, name)))


def run_all(items: List[sq.SessionWork], attended: bool = True,
            stop_on_error: Optional[bool] = None, **kwargs) -> List[dict]:
    """Every pending session in ``items``, serially (§8, §9).

    Attended: each session's report and figures inline, and a failure stops the queue.
    Unattended: a failure is recorded and the queue moves on -- read the returned
    records and :func:`queue_status` afterwards.
    """
    def one(item, **kw):
        record = run_session(item, **kw)
        if attended:
            show_session(item)
        else:
            print(format_report(record))
        return record

    return sq.run_queue(pending_items(items), one,
                        stop_on_error=attended if stop_on_error is None else stop_on_error,
                        **kwargs)

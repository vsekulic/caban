"""Cross-registration by registry: the protected cross-registration notebook, run per mouse and grouping.

plans/crossreg_batch_runner_plan.md §5; the groupings: reports/crossreg_groupings_report.md and
caban/crossreg_groupings.json (compiled from caban/crossreg_groupings_spec.yaml).

One run is one (mouse, grouping) ``n``:

1. **Members** from the registry, each resolved to its output folder. The session name stored in each
   output must be its folder's start time (one listed exception, :data:`STORED_SESSION_NAMES`); the
   notebook is told every name explicitly (``SESSION_NAMES``), so G17's conditioning session is named
   ``18_26_30`` as its folder, not ``918_26_30`` as stored (VS, 2026-10-11).
2. **Scratch view**: ``<scratch>/<mouse folder>/crossreg_<n>/<day>/<session>/Miniscope/`` per member,
   holding one link ``minian_crossreg<n>`` to the real output, so the unchanged notebook
   (``open_minian_mf`` matches on names) opens exactly the registry's sessions. Nothing is renamed.
3. **The notebook** (``notebooks/crossreg_BASELINE.ipynb``, md5-checked) runs with papermill in the
   ``minian-native`` kernel (``minian_runner.execute_and_check``), ``dpath`` = the view.
4. **Checks** (hard failures): the mapping's session columns are exactly the members' start times; the
   members' ``A``/``C``/``S`` files are unchanged (sizes and modification times, before and after).
5. **Published groupings 1-7**: the new mapping is compared by value with the published one
   (``comparison.json``); a difference is a result to report, not a failure.
6. **Install**: the published files the run supplants are moved into ``<mouse>/crossreg_legacy/``
   (``session_queue.move_to_crossreg_legacy``; readers going through ``original_output_path`` keep
   reading them), then the outputs are copied into the mouse folder as
   ``{mappings,cents,shiftds}_crossreg_<n>``.

Records of each run: ``<mouse>/crossreg_runs/crossreg_<n>/`` (executed notebook and HTML, papermill
log, memory, ``parameters.json`` as the kernel held them, ``run.json``, ``comparison.json``).
"""

import ast
import copy
import json
import os
import shutil
import sys
from typing import List, Optional

import nbformat
import numpy as np
import pandas as pd
import xarray as xr

from caban import minian_runner as mr
from caban import session_queue as sq

REGISTRY_PATH = os.path.join(mr.REPO_DIR, "caban", "crossreg_groupings.json")
TEMPLATE_PATH = os.path.join(mr.REPO_DIR, "notebooks", "crossreg_BASELINE.ipynb")
# scripts/build_crossreg_template.py prints it; a changed template must be re-checked by hand.
TEMPLATE_SOURCE_MD5 = "1b884e21f8b1ff300783c9d3d9883553"
PARAMETER_CELL_ANCHOR = 'f_pattern_prefix = r"crossreg"'
TEMPLATE_F_PATTERN_LINE = 'f_pattern = r"4"'
SET_NAMES = ("dpath", "f_pattern", "SESSION_NAMES")

SCRATCH_ROOTS = {"darwin": "/Volumes/FUTROLA/crossreg_scratch",
                 "linux": os.path.expanduser("~/crossreg_scratch")}
RUNS_DIR_NAME = "crossreg_runs"
RUN_RECORD = "run.json"
COMPARISON_JSON = "comparison.json"
PARAMETERS_JSON = "parameters.json"
OUTPUT_FILES = ("mappings_crossreg_{}.csv", "mappings_crossreg_{}.pkl", "cents_crossreg_{}.pkl",
                "shiftds_crossreg_{}.nc")
# The published files of a grouping, in the folder of its mapping file: 2-7 in the mouse folder as
# ``*_crossreg_<n>``; 1 in the conditioning-day folder as ``*_crossreg1`` (G15: ``*_crossreg_1``).
PUBLISHED_FILE_STEMS = ("mappings_crossreg{}{}.csv", "mappings_crossreg{}{}.pkl", "cents_crossreg{}{}.pkl",
                        "shiftds_crossreg{}{}.nc")
CHECKED_ARRAYS = ("A.zarr", "C.zarr", "S.zarr")

# Outputs whose stored session name is not their folder's (yra_recompute_plan.md §6.5): G17's
# conditioning session was processed on the server, where its folder was misnamed 918_26_30.
STORED_SESSION_NAMES = {
    "G17-ST741-mCherry/2022_01_24-TFC_cond/18_26_30-TFC_cond": "918_26_30-TFC_cond",
}
# Published mappings known not to be reproducible (crossreg_groupings_report.md §4).
EXPECTED_DIFFERENCES = {
    ("G15", "2"): "stale: the published mapping refers to an earlier run (prev/minian)",
}
DISTANCE_TOLERANCE = 1e-6


# --- registry and members ------------------------------------------------------------


def load_registry(path: str = REGISTRY_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def data_root() -> str:
    roots = sq.default_data_roots()
    if len(roots) != 1:
        raise ValueError("cross-registration needs one data root holding every session, got {}".format(roots))
    return roots[0]


def start_time(session_label: str) -> str:
    """``G17-.../2022_01_24-TFC_cond/18_26_30-TFC_cond`` -> ``18_26_30``."""
    return os.path.basename(session_label).split("-")[0]


def stored_session_name(output_dir: str) -> str:
    """The ``session`` coordinate Minian stored in the output (what ``open_minian_mf`` keys on)."""
    with xr.open_zarr(os.path.join(output_dir, "A.zarr"), consolidated=False) as ds:
        return str(ds["session"].values.item())


def _current_output(registry: dict, session_label: str) -> str:
    """For a member whose published output is stale: the output the session has in other groupings."""
    outputs = {s["output"] for g in registry["groupings"].values() for m in g["mice"].values()
               for s in m["sessions"]
               if s["session"] == session_label and "stale" not in s and s["output"] != "pending"}
    if len(outputs) != 1:
        raise ValueError("{}: cannot pick its current output from {}".format(session_label, sorted(outputs)))
    return outputs.pop()


def members(registry: dict, n: str, mouse: str, root: str) -> List[dict]:
    """The sessions of grouping ``n`` for ``mouse``, each with its output folder and names checked."""
    entry = registry["groupings"][n]["mice"][mouse]
    out = []
    for s in entry["sessions"]:
        output = s["output"]
        if output == "pending":
            raise ValueError("{} has no Minian output yet (registry: pending); recompile the registry "
                             "after the Minian batch".format(s["session"]))
        if "stale" in s:
            output = _current_output(registry, s["session"])
        output_dir = os.path.join(root, s["session"], "Miniscope", output)
        missing = [a for a in CHECKED_ARRAYS if not os.path.isdir(os.path.join(output_dir, a))]
        if missing:
            raise FileNotFoundError("{} lacks {}".format(output_dir, missing))
        stored = stored_session_name(output_dir)
        start = start_time(s["session"])
        expected = STORED_SESSION_NAMES.get(s["session"])
        if expected is not None and stored != expected:
            raise ValueError("{}: stored session name {!r}, expected {!r}".format(s["session"], stored, expected))
        if expected is None and stored.split("-")[0] != start:
            raise ValueError("{}: stored session name {!r} does not start with the folder's start time {!r}; "
                             "add it to STORED_SESSION_NAMES only after checking why".format(
                                 s["session"], stored, start))
        out.append({"session": s["session"], "output": output, "output_dir": output_dir,
                    "stored_name": stored, "name": start})
    for key in ("stored_name", "name"):
        names = [m[key] for m in out]
        if len(set(names)) != len(names):
            raise ValueError("{} grouping {}: {} not unique: {}".format(mouse, n, key, names))
    return out


def mouse_dir(registry: dict, n: str, mouse: str, root: str) -> str:
    folders = {s["session"].split("/")[0] for s in registry["groupings"][n]["mice"][mouse]["sessions"]}
    if len(folders) != 1:
        raise ValueError("{} grouping {}: sessions span folders {}".format(mouse, n, folders))
    return os.path.join(root, folders.pop())


def planned_runs(registry: dict, groupings, mice=None) -> List[tuple]:
    """(mouse, grouping) pairs, in the order given, for every mouse the grouping has."""
    runs = []
    for n in groupings:
        if n not in registry["groupings"]:
            raise KeyError("no grouping {!r} in the registry; it has {}".format(n, list(registry["groupings"])))
        for mouse in sorted(registry["groupings"][n]["mice"]):
            if mice is None or mouse in mice:
                runs.append((mouse, n))
    return runs


# --- scratch view --------------------------------------------------------------------


def link_name(n: str) -> str:
    return "minian_crossreg{}".format(n)


def remove_view(view_dir: str) -> None:
    """Remove a scratch view: only folders, links and the notebook's outputs may be in it."""
    allowed = set(output_names(os.path.basename(view_dir).split("_", 1)[1]))  # view folder: crossreg_<n>
    for top, dirs, files in os.walk(view_dir):
        for name in files + dirs:
            path = os.path.join(top, name)
            if os.path.islink(path) or os.path.isdir(path) or (top == view_dir and name in allowed):
                continue
            raise ValueError("{} is not part of a scratch view; not removing {}".format(path, view_dir))
    shutil.rmtree(view_dir)


def build_view(view_dir: str, n: str, root: str, mems: List[dict]) -> None:
    """One link ``minian_crossreg<n>`` per member, in a copy of its ``<day>/<session>/Miniscope/`` path."""
    if os.path.lexists(view_dir):
        remove_view(view_dir)
    for m in mems:
        day_session = m["session"].split("/", 1)[1]
        folder = os.path.join(view_dir, day_session, "Miniscope")
        os.makedirs(folder)
        os.symlink(os.path.realpath(m["output_dir"]), os.path.join(folder, link_name(n)))
    found = sorted(os.path.relpath(os.path.join(top, d), view_dir)
                   for top, dirs, _ in os.walk(view_dir) for d in dirs if d == link_name(n))
    expected = sorted(os.path.join(m["session"].split("/", 1)[1], "Miniscope", link_name(n)) for m in mems)
    if found != expected:
        raise ValueError("scratch view holds {}, expected {}".format(found, expected))


# --- the notebook --------------------------------------------------------------------


def parameters_cell_source(parameters_path: str) -> str:
    """The appended cell: what the kernel holds at the end of the run."""
    return '''\
# Appended by caban.crossreg_runner: the effective parameters as the kernel holds them. Read-only.
import json as _json
import sys as _sys
from importlib import metadata as _metadata

import minian as _minian

_record = {{
    "param_dist": param_dist,
    "f_pattern_prefix": f_pattern_prefix,
    "f_pattern": f_pattern,
    "dpath": dpath,
    "SESSION_NAMES": SESSION_NAMES,
    "sessions": [str(s) for s in minian_ds["session"].values],
    "n_mapping_rows": int(len(mappings_meta_fill)),
    "python": _sys.version,
    "packages": {{p: _metadata.version(p) for p in {packages!r}}},
    "minian_file": _minian.__file__,
}}
with open({path!r}, "w") as _fh:
    _json.dump(_record, _fh, indent=2, sort_keys=True)
print("parameters written to", {path!r})
'''.format(packages=list(mr.KERNEL_PACKAGES), path=parameters_path)


def load_template():
    return mr.load_template(TEMPLATE_PATH, TEMPLATE_SOURCE_MD5)


def build_run_notebook(template, view_dir: str, n: str, session_names: dict, parameters_path: str):
    """The copy that runs: ``dpath``, ``f_pattern``, ``SESSION_NAMES`` set; the parameters cell appended."""
    nb = copy.deepcopy(template)
    code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    anchored = [i for i in code if PARAMETER_CELL_ANCHOR in nb.cells[i].source]
    if len(anchored) != 1:
        raise ValueError("expected one parameter cell containing {!r}, found {}".format(PARAMETER_CELL_ANCHOR, anchored))
    idx = anchored[0]
    for i in code:
        hits = [name for name in SET_NAMES for ln in nb.cells[i].source.split("\n") if ln.startswith(name + " =")]
        allowed = ["f_pattern"] if i == idx else []
        if sorted(hits) != allowed:
            raise ValueError("cell {} assigns {}; only the parameter cell may, and only f_pattern".format(i, hits))
    lines = nb.cells[idx].source.split("\n")
    hits = [k for k, ln in enumerate(lines) if ln.startswith(TEMPLATE_F_PATTERN_LINE)]
    if len(hits) != 1:
        raise ValueError("expected one {!r} line in the parameter cell".format(TEMPLATE_F_PATTERN_LINE))
    edits = [{"cell": idx, "was": lines[hits[0]], "now": 'f_pattern = r"{}"'.format(n)}]
    lines[hits[0]] = 'f_pattern = r"{}"  # set by caban.crossreg_runner'.format(n)
    inserted = ["dpath = {!r}  # set by caban.crossreg_runner".format(view_dir),
                "SESSION_NAMES = {!r}  # set by caban.crossreg_runner".format(session_names)]
    edits += [{"cell": idx, "was": None, "now": ln} for ln in inserted]
    nb.cells[idx].source = "\n".join(inserted + lines)
    appended = nbformat.v4.new_code_cell(parameters_cell_source(parameters_path))
    appended.metadata["tags"] = ["caban-crossreg-runner"]
    nb.cells.append(appended)
    edits.append({"cell": len(nb.cells) - 1, "was": None, "now": "appended parameters cell"})
    return nb, edits


def preflight(template) -> dict:
    return mr.preflight_kernel(template, parameters_cell_source(os.devnull))


# --- checks --------------------------------------------------------------------------


def array_snapshot(mems: List[dict]) -> dict:
    """``{file: [bytes, mtime_ns]}`` of every file of every member's ``A``/``C``/``S``."""
    out = {}
    for m in mems:
        for name in CHECKED_ARRAYS:
            top_dir = os.path.join(os.path.realpath(m["output_dir"]), name)
            for top, _, files in os.walk(top_dir):
                for f in files:
                    st = os.stat(os.path.join(top, f))
                    out[os.path.join(top, f)] = [st.st_size, st.st_mtime_ns]
    return out


def read_mapping(path: str, renames: Optional[dict] = None) -> pd.DataFrame:
    """A mapping CSV as columns ``group`` (sorted tuple of session names), one per session, ``distance``.

    ``renames`` maps old session names to current ones, in the header and the group tuples.
    """
    renames = renames or {}
    raw = pd.read_csv(path, header=[0, 1], index_col=0)
    tops = list(raw.columns.get_level_values(0))
    if tops[0] != "group" or tops[-1] != "variable" or set(tops[1:-1]) != {"session"}:
        raise ValueError("{}: unexpected header {}".format(path, list(raw.columns)))
    sessions = [renames.get(s, s) for s in raw.columns.get_level_values(1)[1:-1]]
    df = pd.DataFrame(raw.iloc[:, 1:-1].to_numpy(dtype=float), columns=sessions)
    df.insert(0, "group", [tuple(sorted(renames.get(s, s) for s in ast.literal_eval(g)))
                           for g in raw.iloc[:, 0]])
    df["distance"] = raw.iloc[:, -1].to_numpy(dtype=float)
    return df


def mapping_sessions(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("group", "distance")]


def _row_keys(df: pd.DataFrame, sessions: List[str]) -> pd.Series:
    ids = df[sessions].fillna(-1).astype(int)
    return pd.Series([tuple(r) for r in ids.itertuples(index=False)], index=df.index)


def compare_mappings(published: pd.DataFrame, new: pd.DataFrame) -> dict:
    """By value: the same sessions, the same matched-unit rows and groups, distances within tolerance."""
    pub_s, new_s = mapping_sessions(published), mapping_sessions(new)
    if sorted(pub_s) != sorted(new_s):
        return {"identical": False, "sessions_published": pub_s, "sessions_new": new_s}
    sessions = sorted(pub_s)
    a = published.assign(key=_row_keys(published, sessions)).set_index("key")
    b = new.assign(key=_row_keys(new, sessions)).set_index("key")
    if not (a.index.is_unique and b.index.is_unique):
        raise ValueError("a mapping repeats a row of unit ids")
    only_pub, only_new = a.index.difference(b.index), b.index.difference(a.index)
    common = a.index.intersection(b.index)
    group_diff = int((a.loc[common, "group"] != b.loc[common, "group"]).sum())
    da, db = a.loc[common, "distance"].to_numpy(), b.loc[common, "distance"].to_numpy()
    nan_diff = int((np.isnan(da) != np.isnan(db)).sum())
    both = ~np.isnan(da) & ~np.isnan(db)
    max_dd = float(np.max(np.abs(da[both] - db[both]))) if both.any() else 0.0
    return {
        "identical": not len(only_pub) and not len(only_new) and not group_diff and not nan_diff
                     and max_dd <= DISTANCE_TOLERANCE,
        "sessions": sessions, "rows_published": len(a), "rows_new": len(b),
        "rows_only_published": len(only_pub), "rows_only_new": len(only_new),
        "rows_group_differs": group_diff, "rows_distance_nan_differs": nan_diff,
        "max_distance_difference": max_dd,
    }


def published_renames(mems: List[dict]) -> dict:
    """Old session names a published mapping may carry: the stored names that differ from the folder's."""
    return {m["stored_name"].split("-")[0]: m["name"] for m in mems
            if m["stored_name"].split("-")[0] != m["name"]}


# --- install -------------------------------------------------------------------------


def output_names(n: str) -> List[str]:
    return [f.format(n) for f in OUTPUT_FILES]


def supplanted_files(registry: dict, n: str, mouse: str, root: str, mdir: str) -> List[str]:
    """Published files this run supplants: the grouping's published set beside its mapping file, and
    anything already at the output names in the mouse folder."""
    entry = registry["groupings"][n]["mice"][mouse]
    found = [os.path.join(mdir, f) for f in output_names(n) if os.path.lexists(os.path.join(mdir, f))]
    if "mapping" in entry:
        folder = os.path.dirname(os.path.join(root, entry["mapping"]))
        for sep in ("_", ""):
            found += [os.path.join(folder, f.format(sep, n)) for f in PUBLISHED_FILE_STEMS
                      if os.path.lexists(os.path.join(folder, f.format(sep, n)))]
        if os.path.join(root, entry["mapping"]) not in found:
            raise FileNotFoundError("published mapping {} not found".format(entry["mapping"]))
    return sorted(set(found))


def install_outputs(view_dir: str, mdir: str, n: str, supplanted: List[str], reason: str) -> dict:
    sources = [os.path.join(view_dir, f) for f in output_names(n)]
    missing = [s for s in sources if not os.path.isfile(s)]
    if missing:
        raise FileNotFoundError("the notebook did not write {}".format(missing))
    moved = sq.move_to_crossreg_legacy(supplanted, reason) if supplanted else []
    installed = []
    for src in sources:
        dst = os.path.join(mdir, os.path.basename(src))
        if os.path.lexists(dst):
            raise FileExistsError("{} exists after the published files were moved".format(dst))
        shutil.copy2(src, dst)
        if os.path.getsize(dst) != os.path.getsize(src):
            raise IOError("copy of {} is incomplete".format(src))
        installed.append(os.path.basename(dst))
    return {"moved_to_legacy": moved, "installed": installed}


# --- one run -------------------------------------------------------------------------


def run_dir_of(mdir: str, n: str) -> str:
    return os.path.join(mdir, RUNS_DIR_NAME, "crossreg_{}".format(n))


def read_record(mdir: str, n: str) -> Optional[dict]:
    path = os.path.join(run_dir_of(mdir, n), RUN_RECORD)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _write_record(run_dir: str, record: dict) -> None:
    with open(os.path.join(run_dir, RUN_RECORD), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)


def clear_failed_run(mdir: str, n: str) -> str:
    """Keep a failed run's records as ``crossreg_<n>-failed-<time>``, so the grouping can be rerun."""
    record = read_record(mdir, n)
    if record is not None and record.get("status") == "done":
        raise ValueError("crossreg_{} of {} is done; not clearing it".format(n, mdir))
    src = run_dir_of(mdir, n)
    dst = "{}-failed-{}".format(src, pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"))
    os.rename(src, dst)
    return dst


def run_one(registry: dict, mouse: str, n: str, template, root: Optional[str] = None,
            scratch_root: Optional[str] = None) -> dict:
    """Cross-register grouping ``n`` of ``mouse`` (module docstring); returns the run record."""
    root = root or data_root()
    scratch_root = scratch_root or SCRATCH_ROOTS[sys.platform]
    mdir = mouse_dir(registry, n, mouse, root)
    run_dir = run_dir_of(mdir, n)
    if os.path.lexists(run_dir):
        raise FileExistsError("{} exists (status {!r}); a failed run is cleared with clear_failed_run".format(
            run_dir, (read_record(mdir, n) or {}).get("status")))
    mems = members(registry, n, mouse, root)
    grouping = registry["groupings"][n]
    published = grouping["status"] == "published"
    supplanted = supplanted_files(registry, n, mouse, root, mdir)
    if not published and supplanted:
        raise FileExistsError("grouping {} is new, yet {} exist".format(n, supplanted))
    os.makedirs(run_dir)
    record = {
        "mouse": mouse, "grouping": n, "name": grouping["name"], "status": "running",
        "started": pd.Timestamp.now().isoformat(timespec="seconds"), "host": os.uname().nodename,
        "members": mems, "published": published,
        "template": {"path": TEMPLATE_PATH, "md5": TEMPLATE_SOURCE_MD5},
        "caban_git": mr._git_state(mr.REPO_DIR), "minian_git": mr._git_state(mr.MINIAN_FORK_DIR),
        "timings": {},
    }
    _write_record(run_dir, record)
    try:
        view_dir = os.path.join(scratch_root, os.path.basename(mdir), "crossreg_{}".format(n))
        build_view(view_dir, n, root, mems)
        record["view_dir"] = view_dir
        nb, record["edits"] = build_run_notebook(
            template, view_dir, n, {m["stored_name"]: m["name"] for m in mems},
            os.path.join(run_dir, PARAMETERS_JSON))
        before = array_snapshot(mems)
        mr.execute_and_check(nb, run_dir, record)
        if array_snapshot(mems) != before:
            raise RuntimeError("A/C/S of a member changed during the run")
        record["arrays_unchanged"] = len(before)

        new = read_mapping(os.path.join(view_dir, output_names(n)[0]))
        if sorted(mapping_sessions(new)) != sorted(m["name"] for m in mems):
            raise ValueError("mapping sessions {} != members {}".format(
                mapping_sessions(new), [m["name"] for m in mems]))
        record["n_mapping_rows"] = len(new)
        record["n_matched_in_all"] = int(new[mapping_sessions(new)].notna().all(axis=1).sum())

        if published:
            entry = grouping["mice"][mouse]
            pub = read_mapping(sq.original_output_path(os.path.join(root, entry["mapping"])),
                               published_renames(mems))
            comparison = compare_mappings(pub, new)
            comparison["published_file"] = entry["mapping"]
            comparison["expected_difference"] = EXPECTED_DIFFERENCES.get((mouse, n))
            with open(os.path.join(run_dir, COMPARISON_JSON), "w") as fh:
                json.dump(comparison, fh, indent=2, sort_keys=True)
            record["comparison"] = {"identical": comparison["identical"],
                                    "expected_difference": comparison["expected_difference"]}

        record["install"] = install_outputs(
            view_dir, mdir, n, supplanted,
            "supplanted by the re-derived crossreg_{} (caban.crossreg_runner, {})".format(n, RUNS_DIR_NAME))
        remove_view(view_dir)
        record["status"] = "done"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        record["finished"] = pd.Timestamp.now().isoformat(timespec="seconds")
        _write_record(run_dir, record)
    return record

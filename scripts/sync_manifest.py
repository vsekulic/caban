"""The Razer side of the sync MINIRAZER -> MINISCOPE (`scripts/sync_to_miniscope.py` runs on the Mac).

    python scripts/sync_manifest.py manifest <out.json>   every done, not yet synced session: what to carry
                                                          and the md5 + size of every file in it
    python scripts/sync_manifest.py mark <labels.json>    write the sync record into each listed session

What a session carries (handover §4): the run's output and records -- ``minian/`` (with
``YrA_recomputed.zarr``), ``minian_run/``, the two videos, ``minian_set_aside.json``, any
``minian_run-failed-*`` -- never the raw data. The set-aside record is included in the manifest so
the Mac can replay its renames (``minian`` -> ``minian-ORIG``, ...) on MINISCOPE first; rsync
cannot express a rename.
"""

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import minian_runner as mr
from caban import session_queue as sq
from caban import session_staging as ss

SYNC_RECORD = "synced_to_miniscope.json"
CARRIED = (mr.OUTPUT_NAME, mr.RUN_DIR_NAME) + mr.NOTEBOOK_VIDEOS + (sq.SET_ASIDE_RECORD,)


def carried_items(session_dir: str) -> list:
    names = [n for n in CARRIED if os.path.lexists(os.path.join(session_dir, n))]
    names += sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(session_dir, mr.RUN_DIR_NAME + "-failed-*")))
    return names


def file_manifest(session_dir: str, name: str) -> dict:
    """``{relative path: [bytes, md5]}`` for a file or every file under a folder (the sync record
    itself excluded: it is written after the sync)."""
    root = os.path.join(session_dir, name)
    out = {}
    for rel in ss._tree_files(root):
        path = os.path.join(root, rel) if rel else root
        if os.path.basename(path) == SYNC_RECORD:
            continue
        out[rel] = [os.path.getsize(path), ss.md5_file(path)]
    return out


def manifest(out_path: str) -> None:
    sessions = []
    for item in mr.discover_sessions(verbose=False):
        if not os.path.isfile(os.path.join(item.session_dir, mr.SIDECAR_NAME)):
            continue
        if mr.run_status(item) != "done":
            continue
        if os.path.isfile(os.path.join(item.session_dir, mr.RUN_DIR_NAME, SYNC_RECORD)):
            continue
        names = carried_items(item.session_dir)
        sessions.append({
            "label": item.label,
            "tail": os.path.join(*os.path.normpath(item.session_dir).split(os.sep)[-4:]),
            "session_dir": item.session_dir,
            "set_aside": sq.read_sidecar(item.session_dir, sq.SET_ASIDE_RECORD),
            "items": {name: file_manifest(item.session_dir, name) for name in names},
        })
        print("{}: {} items, {} files".format(item.label, len(names),
                                              sum(len(v) for v in sessions[-1]["items"].values())))
    with open(out_path, "w") as fh:
        json.dump({"written": pd.Timestamp.now().isoformat(timespec="seconds"),
                   "sessions": sessions}, fh)
    print("manifest: {} sessions -> {}".format(len(sessions), out_path))


def mark(labels_path: str) -> None:
    with open(labels_path) as fh:
        synced = json.load(fh)
    items = {i.label: i for i in mr.discover_sessions(verbose=False)}
    for entry in synced:
        item = items[entry["label"]]
        record = dict(entry, when=pd.Timestamp.now().isoformat(timespec="seconds"))
        sq.write_sidecar(os.path.join(item.session_dir, mr.RUN_DIR_NAME), SYNC_RECORD, record)
        print("marked synced: {}".format(item.label))


if __name__ == "__main__":
    {"manifest": manifest, "mark": mark}[sys.argv[1]](sys.argv[2])

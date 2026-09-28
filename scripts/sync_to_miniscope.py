"""Sync MINIRAZER -> MINISCOPE: carry the Razer's finished runs onto MINISCOPE, the Mac's APFS copy
(handover §4; the Razer side is `scripts/sync_manifest.py`).

    python scripts/sync_to_miniscope.py <manifest.json> [--dry-run]

Per session, from the manifest the Razer wrote (what each session carries and the md5 + size of every
file in it):

1. Replay the set-aside renames of its ``minian_set_aside.json`` on MINISCOPE (``minian`` ->
   ``minian-ORIG``, the old videos, ...), so the old output is kept there exactly as on MINIRAZER --
   rsync cannot express a rename. A rename already made (by the Mac's own run, or an earlier sync) is
   recognised by its target existing.
2. For each carried item: if MINISCOPE already has it, it must equal the manifest file for file, md5
   for md5 (a run made on the Mac is already there) -- anything else stops the session. Otherwise it
   is pulled from the Razer with rsync into ``<name>.sync-partial``.
3. Every file of every partial copy is checked against the manifest (same files, same sizes, same
   md5s); only then are the partial copies renamed to their real names.

Nothing on MINISCOPE is deleted or overwritten -- except a leftover ``.sync-partial`` of an
interrupted sync, which only this script writes. The raw data are never copied. A record of the run
goes to ``MINISCOPE/_provenance/``; the Razer marks each synced session afterwards (its report's
``on_MINISCOPE`` column).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_staging as ss

MINISCOPE_ROOT = "/Volumes/MINISCOPE"
DATA_ROOT = os.path.join(MINISCOPE_ROOT, "SSTCa2")
PROVENANCE_DIR = os.path.join(MINISCOPE_ROOT, "_provenance")
RAZER = "minastirith"
PARTIAL_SUFFIX = ".sync-partial"
SYNC_RECORD = "synced_to_miniscope.json"
# Finder's window-state files: created on MINISCOPE by browsing it on the Mac, left out of the copy to
# MINIRAZER (robocopy exclusions, razer plan Phase 2), never data.
IGNORED_NAMES = (".DS_Store",)


def local_manifest(root: str) -> dict:
    """``{relative path: [bytes, md5]}`` of a file or folder, as `sync_manifest.file_manifest`."""
    out = {}
    for rel in ss._tree_files(root):
        path = os.path.join(root, rel) if rel else root
        if os.path.basename(path) in (SYNC_RECORD,) + IGNORED_NAMES:
            continue
        out[rel] = [os.path.getsize(path), ss.md5_file(path)]
    return out


def plan_session(session: dict) -> dict:
    """What syncing this session would do; raises on anything that is not clean."""
    dst = os.path.join(DATA_ROOT, session["tail"])
    if not os.path.isdir(dst):
        raise FileNotFoundError("{}: {} is not on MINISCOPE".format(session["label"], dst))
    renames = []
    for name, orig in session["set_aside"].get("renamed", {}).items():
        if os.path.lexists(os.path.join(dst, orig)):
            continue
        if not os.path.lexists(os.path.join(dst, name)):
            raise FileNotFoundError("{}: the set-aside record renamed {} -> {}, but MINISCOPE has "
                                    "neither".format(session["label"], name, orig))
        renames.append((name, orig))
    renamed_away = {name for name, _ in renames}
    copy, present = [], []
    for name, files in session["items"].items():
        target = os.path.join(dst, name)
        if os.path.lexists(target) and name not in renamed_away:
            if local_manifest(target) != files:
                raise ValueError("{}: MINISCOPE already has {} and it differs from MINIRAZER's; "
                                 "resolve by hand".format(session["label"], target))
            present.append(name)
        else:
            copy.append(name)
    return {"dst": dst, "renames": renames, "copy": copy, "present": present}


def pull(src: str, dst: str, is_dir: bool) -> None:
    """rsync from the Razer: a folder's contents into ``dst/``, or one file to ``dst``."""
    subprocess.run(["rsync", "-rt", "-e", "ssh",
                    "{}:{}{}".format(RAZER, src, "/" if is_dir else ""),
                    dst + ("/" if is_dir else "")], check=True)


def sync_session(session: dict, plan: dict) -> dict:
    dst = plan["dst"]
    for name, orig in plan["renames"]:
        os.rename(os.path.join(dst, name), os.path.join(dst, orig))
        print("  renamed {} -> {}".format(name, orig))
    partials = []
    for name in plan["copy"]:
        files = session["items"][name]
        partial = os.path.join(dst, name + PARTIAL_SUFFIX)
        if os.path.lexists(partial):
            print("  removing {} (left by an interrupted sync)".format(partial))
            shutil.rmtree(partial) if os.path.isdir(partial) else os.remove(partial)
        is_dir = "" not in files
        pull(os.path.join(session["session_dir"], name), partial, is_dir)
        got = local_manifest(partial)
        if got != files:
            missing = sorted(set(files) - set(got))[:5]
            extra = sorted(set(got) - set(files))[:5]
            wrong = sorted(k for k in set(files) & set(got) if files[k] != got[k])[:5]
            raise IOError("{}: the copy of {} does not verify: missing {}, extra {}, differing {}; "
                          "left at {}".format(session["label"], name, missing, extra, wrong, partial))
        partials.append((partial, os.path.join(dst, name)))
        print("  copied and verified {} ({} files)".format(name, len(files)))
    for partial, target in partials:
        os.rename(partial, target)
    return {"label": session["label"], "renamed": plan["renames"], "copied": plan["copy"],
            "already_there": plan["present"],
            "files": sum(len(session["items"][n]) for n in plan["copy"]),
            "bytes": sum(v[0] for n in plan["copy"] for v in session["items"][n].values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest")
    parser.add_argument("--dry-run", action="store_true", help="print what would be done; change nothing")
    args = parser.parse_args()
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit("MINISCOPE is not mounted ({} missing)".format(DATA_ROOT))
    with open(args.manifest) as fh:
        manifest = json.load(fh)
    plans = []
    for session in manifest["sessions"]:
        plan = plan_session(session)
        plans.append((session, plan))
        print("{}: rename {}; copy {}; already there {}".format(
            session["label"], plan["renames"] or "-", plan["copy"] or "-", plan["present"] or "-"))
    if args.dry_run:
        return 0
    done = []
    for session, plan in plans:
        print(session["label"])
        done.append(sync_session(session, plan))
    stamp = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    record_path = os.path.join(PROVENANCE_DIR, "sync_minirazer_{}.json".format(stamp))
    with open(record_path, "w") as fh:
        json.dump({"when": stamp, "manifest_written": manifest["written"], "sessions": done}, fh, indent=2)
    with open(os.path.splitext(args.manifest)[0] + "_synced.json", "w") as fh:
        json.dump([{"label": d["label"], "provenance": record_path, "copied": d["copied"],
                    "already_there": d["already_there"]} for d in done], fh, indent=2)
    print("synced {} sessions ({} files, {:.2f} GB); record {}".format(
        len(done), sum(d["files"] for d in done), sum(d["bytes"] for d in done) / 1e9, record_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

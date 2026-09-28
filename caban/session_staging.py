"""The copier: a session's videos onto local disk before a run, its results back after.

`plans/session_staging_copier_plan.md`. On the Razer the data drive (MINIRAZER) is a USB
hard disk read through WSL's ``/mnt/e``: 80 MB/s for one reader, 58 MB/s in total for three
(seek contention), and it has logged a controller error under load. So a run does not work
on it. The session's videos are copied to a *work folder* on the local NVMe, the notebook
and every later step run there, and the results are copied back -- each copy verified
before anything relies on it. All of it is sequential, one file after another, so the disk
streams instead of seeking.

The work folder mirrors the last four components of the session path,
``<stage_root>/<mouse folder>/<day>/<session>/Miniscope``: the notebook names the saved
arrays' ``animal`` and ``session`` coordinates from those components
(``param_save_minian["meta_dict"]``), so a run in the work folder writes exactly what a run
in place would.

Mechanism only. Which sessions, when, and the run record are `caban.minian_runner`'s.
"""

import hashlib
import os
import re
import shutil
import time
from typing import Dict, List, Sequence

from natsort import natsorted

# Suffix of a copy that is not yet verified: a folder or file on either side that carries
# it is never taken for the real thing, and is the copier's own to delete.
PARTIAL_SUFFIX = ".stage-partial"
HASH_BLOCK_BYTES = 8 * 1024 * 1024
# Free space the local disk must have before a stage-in: the videos (<= ~7 GB), the
# notebook's intermediates on the same disk (up to ~76 GB for a 26-file session) and the
# outputs, with room for the next session's prefetched videos.
MIN_FREE_GB = 150


def work_dir_for(session_dir: str, stage_root: str) -> str:
    """``<stage_root>/<mouse folder>/<day>/<session>/Miniscope`` for a session folder."""
    parts = os.path.normpath(session_dir).split(os.sep)
    if len(parts) < 4 or parts[-1] != "Miniscope":
        raise ValueError("{} is not a <mouse>/<day>/<session>/Miniscope folder".format(session_dir))
    return os.path.join(os.path.abspath(stage_root), *parts[-4:])


def input_files(session_dir: str, pattern: str) -> List[str]:
    """The files a run reads: those in ``session_dir`` whose name matches ``pattern`` (searched,
    as Minian's ``load_videos`` does), natsorted. Hard-fails on none."""
    names = natsorted(n for n in os.listdir(session_dir)
                      if re.search(pattern, n) and os.path.isfile(os.path.join(session_dir, n)))
    if not names:
        raise FileNotFoundError("no file matching {!r} in {}".format(pattern, session_dir))
    return names


def md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_hashing(src: str, dst: str) -> str:
    """Copy ``src`` to ``dst`` in one sequential pass; the md5 of the bytes as read."""
    digest = hashlib.md5()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for block in iter(lambda: fin.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
            fout.write(block)
    shutil.copystat(src, dst)
    return digest.hexdigest()


def copy_verified(src: str, dst: str, reread_source: bool) -> Dict:
    """Copy one file and prove the copy: the md5 of the bytes read while copying must equal
    the md5 of the copy read back and, with ``reread_source``, of the source read again.

    The second source read is what catches a *read* that went wrong (G14, 2026-09-28): a
    copy of wrong bytes is otherwise self-consistent. Its limit: Windows may serve the
    re-read from its file cache, so a fault below the cache can repeat identically -- see
    the plan, "What the verification proves".
    """
    copied = copy_hashing(src, dst)
    readback = md5_file(dst)
    source_again = md5_file(src) if reread_source else copied
    size_src, size_dst = os.path.getsize(src), os.path.getsize(dst)
    if not (copied == readback == source_again) or size_src != size_dst:
        raise IOError("copy of {} to {} does not verify: md5 while copying {}, copy read back {}, "
                      "source read again {}; {} vs {} bytes".format(
                          src, dst, copied, readback, source_again if reread_source else "-",
                          size_src, size_dst))
    return {"bytes": size_dst, "md5": copied}


def _require_free_space(root: str, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < min_free_gb:
        raise OSError("{} has {:.0f} GB free; a stage-in needs at least {} GB".format(
            root, free_gb, min_free_gb))


def _remove_partial(path: str) -> None:
    """Delete a leftover unverified copy -- only ever something this module wrote."""
    if not path.endswith(PARTIAL_SUFFIX):
        raise ValueError("refusing to delete {}: not a {} copy".format(path, PARTIAL_SUFFIX))
    print("removing {} (an unverified copy left by an interrupted copier)".format(path))
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def stage_in(session_dir: str, stage_root: str, pattern: str,
             min_free_gb: float = MIN_FREE_GB) -> Dict:
    """Copy a session's input files into its work folder, verified; return the manifest.

    Copies into ``<work folder>.stage-partial`` and renames it to the work folder only once
    every file verifies, so a work folder always holds a complete, verified copy. Refuses if
    the work folder exists (the caller decides whether an old one is disposable); a leftover
    ``.stage-partial`` is an interrupted copy and is replaced.
    """
    if not os.path.isdir(stage_root):
        raise FileNotFoundError("stage root {} does not exist".format(stage_root))
    _require_free_space(stage_root, min_free_gb)
    work_dir = work_dir_for(session_dir, stage_root)
    if os.path.lexists(work_dir):
        raise FileExistsError("work folder {} already exists".format(work_dir))
    partial = work_dir + PARTIAL_SUFFIX
    if os.path.lexists(partial):
        _remove_partial(partial)
    started = time.time()
    names = input_files(session_dir, pattern)
    os.makedirs(partial)
    files = {name: copy_verified(os.path.join(session_dir, name), os.path.join(partial, name),
                                 reread_source=True)
             for name in names}
    os.rename(partial, work_dir)
    total = sum(f["bytes"] for f in files.values())
    seconds = time.time() - started
    print("staged in {} files, {:.2f} GB in {:.0f} s ({:.0f} MB/s incl. verification) -> {}".format(
        len(files), total / 1e9, seconds, total / 1e6 / max(seconds, 1e-9), work_dir))
    return {"session_dir": session_dir, "work_dir": work_dir, "files": files,
            "bytes": total, "seconds": round(seconds, 1)}


def _tree_files(root: str) -> List[str]:
    """Every file under ``root``, relative to it; ``root`` may be a single file."""
    if os.path.isfile(root):
        return [""]
    out = []
    for dirpath, _, filenames in os.walk(root):
        out += [os.path.relpath(os.path.join(dirpath, f), root) for f in filenames]
    return sorted(out)


def _copy_tree_verified(src_root: str, dst_root: str) -> Dict:
    """Copy a file or a folder tree, every file verified by md5 of the copy read back."""
    n, total = 0, 0
    for rel in _tree_files(src_root):
        src = os.path.join(src_root, rel) if rel else src_root
        dst = os.path.join(dst_root, rel) if rel else dst_root
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        total += copy_verified(src, dst, reread_source=False)["bytes"]
        n += 1
    return {"files": n, "bytes": total}


def stage_out(work_dir: str, session_dir: str, outputs: Sequence[str], record_dir: str,
              record_keep: str) -> Dict:
    """Copy a finished run's results from the work folder back into the session folder.

    ``outputs`` (``minian`` and the two videos) must not exist in the session folder: each is
    copied to ``<name>.stage-partial``, every file verified by reading the copy back, and
    only when all have verified are they renamed to their real names -- so the session
    folder never shows a partial result under a real name. A leftover ``.stage-partial``
    from an interrupted stage-out is replaced.

    ``record_dir`` (``minian_run``) already exists in the session folder, holding the run
    record ``record_keep`` (``run.json``), which the runner writes there directly: the
    work folder's ``record_dir`` files are copied into it (verified), and the work folder
    must not carry its own ``record_keep``.
    """
    started = time.time()
    for name in outputs:
        if os.path.lexists(os.path.join(session_dir, name)):
            raise FileExistsError("{} already exists in {}; refusing to overwrite".format(
                name, session_dir))
        if not os.path.lexists(os.path.join(work_dir, name)):
            raise FileNotFoundError("{} missing from the work folder {}".format(name, work_dir))
    local_records = os.path.join(work_dir, record_dir)
    if os.path.lexists(os.path.join(local_records, record_keep)):
        raise ValueError("{} holds a {}; the run record lives only in the session folder".format(
            local_records, record_keep))
    if not os.path.isdir(os.path.join(session_dir, record_dir)):
        raise FileNotFoundError("{} has no {}/ for the run record".format(session_dir, record_dir))

    copied = {}
    for name in outputs:
        partial = os.path.join(session_dir, name + PARTIAL_SUFFIX)
        if os.path.lexists(partial):
            _remove_partial(partial)
        copied[name] = _copy_tree_verified(os.path.join(work_dir, name), partial)
    copied[record_dir] = _copy_tree_verified(local_records, os.path.join(session_dir, record_dir))
    for name in outputs:
        os.rename(os.path.join(session_dir, name + PARTIAL_SUFFIX), os.path.join(session_dir, name))
    total = sum(c["bytes"] for c in copied.values())
    seconds = time.time() - started
    print("staged out {} files, {:.2f} GB in {:.0f} s -> {}".format(
        sum(c["files"] for c in copied.values()), total / 1e9, seconds, session_dir))
    return {"copied": copied, "bytes": total, "seconds": round(seconds, 1)}

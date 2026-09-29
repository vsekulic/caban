"""A restartable, disk-backed work queue over Miniscope session directories.

Both pipelines in this repository work session by session over the same recording
tree, one session at a time, resuming after a crash:

- `caban.yra_recompute` -- recompute `YrA` for sessions that already have Minian
  output (`plans/yra_recompute_plan.md`);
- the local CNMF-E pipeline -- generate `A`/`C`/`S` for sessions that have none
  (`plans/local_minian_pipeline_plan.md`).

They differ entirely in what counts as work: one wants sessions *with* a complete
`minian_crossreg*`, the other wants sessions *without* one. What they share is the
scan, the cursor, the idempotency and the reporting, and that is what lives here.

The split is mechanism/policy. This module answers "what is on disk and what has
already been done"; each pipeline answers "which of these is my work, and how do I
do one". Concretely: :func:`scan_sessions` reports facts about every session it
finds and selects nothing, and :func:`next_pending` decides *position* but never
*acceptance* -- a session that merely looks finished is still re-checked by the
pipeline's own authoritative test.

Position is re-derived from the sidecars on every call rather than held in a
generator, so a queue survives a kernel restart and advances by itself once a
session finishes.
"""

import dataclasses
import glob
import json
import os
import re
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

# Where the raw sessions live. The backup drives are read-only; `MINISCOPE` is the
# consolidated APFS volume (`plans/local_minian_pipeline_plan.md` §5).
DATA_ROOTS_BACKUP = (
    "/Volumes/1a-MINISCOPE-BAK/data/vsekulic/OF_test",
    "/Volumes/1b-MINISCOPE-BAK/data/vsekulic/OF_test",
)
DATA_ROOTS_CONSOLIDATED = ("/Volumes/MINISCOPE/SSTCa2",)
# The Razer's working copy of MINISCOPE: drive MINIRAZER (E:) seen from WSL
# (`plans/razer_runner_plan.md`).
DATA_ROOTS_MINIRAZER = ("/mnt/e/SSTCa2",)
# Overrides the search below: data roots separated by os.pathsep.
DATA_ROOTS_ENV = "CABAN_DATA_ROOTS"

MOUSE_DIR_PATTERN = r"^G\d\d"
VIDEO_PATTERN = r"^[0-9]+\.avi$"
# A Minian output directory counts as complete only with all of these present.
REQUIRED_MINIAN_ARRAYS = ("A", "C", "S", "b", "f", "motion")

# --- Timestamp rows without imaging (`plans/yra_recompute_plan.md` §15) -------------
#
# The Miniscope software writes a `timeStamps.csv` row per frame and `framesPerFile`
# (1000, per every session's metaData.json) frames per numbered .avi. Production's `C`
# holds exactly the frames of the files Minian could read, so row i of the timestamps is
# frame i of `C` only where no row before it lacks imaging. Both the YrA replay (which
# files to read) and `caban.sessions` (which timestamp rows to drop) use these tables;
# the analysis side reads no raw data, so they are explicit, not derived from the files.
FRAMES_PER_FILE = 1000

# Videos production never read: their header was never finalised (ffprobe nb_frames
# N/A), which Minian's load_avi_lazy needs. Established 2026-09-26 (plan §15):
# production's C frame count equals exactly the sum over the remaining files, and the
# replay's frame-count check re-proves that every run. Listed explicitly rather than
# skipped by rule, so an unreadable file anywhere else is still a hard failure.
UNUSED_VIDEOS = {
    # 19.avi: 865 decodable frames, no header count; at the end of the recording.
    "G15-ST721-hM4D/2022_01_13-TFC_test_B/16_19_13-TFC_test_B": ["19.avi"],
    # 11.avi: 593 decodable frames; 12.avi, 13.avi: 14 KB, no frames. In the MIDDLE of
    # the recording: no imaging from 553.9 s to 705.9 s of the experiment, and C frame
    # 11,000 is timestamp row 14,000 (§15.1).
    "G21-ST762-hM4D/2022_03_24-TFC_test_B/14_35_02-TFC_test_B": ["11.avi", "12.avi", "13.avi"],
}

# Sessions whose timestamps run on past the last video frame, from this row on: every
# file was read, but the recording kept logging timestamps it wrote no frames for.
TIMESTAMPS_WITHOUT_VIDEO_FROM = {
    # 19 full files = 19,000 frames = C; 19,586 timestamp rows. The experiment ends at
    # row 19,022, 1.1 s after the last imaged frame (found 2026-09-27).
    "G09-ST702_mCherry/2021_11_08-TFC_cond/18_54_05-TFC_cond": 19000,
}


def session_tail(session_dir: str) -> str:
    """``<mouse>/<day>/<session>`` of a session path, ``Miniscope`` stripped.

    Splits on both separators: the analysis side carries Windows-style paths.
    """
    parts = [p for p in re.split(r"[\\/]+", session_dir) if p]
    if parts and parts[-1] == "Miniscope":
        parts = parts[:-1]
    return "/".join(parts[-3:])


def unused_videos(session_dir: str) -> List[str]:
    """The ``.avi`` files production did not read for this session (:data:`UNUSED_VIDEOS`)."""
    return list(UNUSED_VIDEOS.get(session_tail(session_dir), []))


def unimaged_timestamp_rows(session_dir: str) -> List[tuple]:
    """``[(first_row, stop_row_or_None), ...]``: timestamp rows with no frame in ``C``."""
    ranges = []
    for name in unused_videos(session_dir):
        match = re.match(r"^([0-9]+)\.avi$", name)
        if match is None:
            raise ValueError("UNUSED_VIDEOS entry {!r} for {} is not a numbered .avi".format(
                name, session_tail(session_dir)))
        k = int(match.group(1))
        ranges.append((k * FRAMES_PER_FILE, (k + 1) * FRAMES_PER_FILE))
    start = TIMESTAMPS_WITHOUT_VIDEO_FROM.get(session_tail(session_dir))
    if start is not None:
        ranges.append((start, None))
    return sorted(ranges)


def imaged_timestamp_rows(session_dir: str, n_rows: int) -> np.ndarray:
    """Boolean mask over a session's ``n_rows`` timestamp rows: True where ``C`` has the frame.

    A range reaching past ``n_rows`` is simply cut there: the last file of a recording
    is usually partial.
    """
    imaged = np.ones(n_rows, dtype=bool)
    for first, stop in unimaged_timestamp_rows(session_dir):
        if first >= n_rows:
            raise ValueError("{}: unimaged rows from {} but the session has only {} timestamp "
                             "rows".format(session_tail(session_dir), first, n_rows))
        imaged[first:stop] = False
    return imaged

# Existing output is renamed with this suffix before a notebook re-run writes in its
# place (`plans/local_minian_pipeline_plan.md` §5.2). What each notebook writes:
#   pipeline notebook      -> <session>/Miniscope/: these dirs and videos
#   cross-registration     -> <mouse>/: mappings_<name>.{pkl,csv}, cents_<name>.pkl,
#                             shiftds_<name>.nc  (cell 55 of cross-registration-WORKING)
#   caban.minian_runner    -> <session>/Miniscope/minian_run/ (executed notebook, record)
SET_ASIDE_SUFFIX = "-ORIG"
RUN_DIR_NAME = "minian_run"
NOTEBOOK_OUTPUT_DIRS = ("minian", "minian_intermediate", RUN_DIR_NAME)
NOTEBOOK_OUTPUT_FILES = ("minian.mp4", "minian_mc.mp4")
SCRATCH_DIR_NAME = "minian_intermediate"
SET_ASIDE_RECORD = "minian_set_aside.json"


def default_data_roots() -> tuple:
    """``$CABAN_DATA_ROOTS`` if set; else MINISCOPE (the Mac), MINIRAZER (the Razer), or
    the two backup drives -- the first that is mounted.

    Hard-fails rather than returning an empty tuple: a queue built from no roots
    would silently report "nothing to do", which is the one answer that must never
    be produced by accident. So does an override that names a missing folder.
    """
    override = os.environ.get(DATA_ROOTS_ENV)
    if override:
        roots = tuple(r for r in override.split(os.pathsep) if r)
        missing = [r for r in roots if not os.path.isdir(r)]
        if missing:
            raise FileNotFoundError("{} names missing folders: {}".format(DATA_ROOTS_ENV, missing))
        return roots
    candidates = (DATA_ROOTS_CONSOLIDATED, DATA_ROOTS_MINIRAZER, DATA_ROOTS_BACKUP)
    for roots in candidates:
        if all(os.path.isdir(r) for r in roots):
            return roots
    raise FileNotFoundError(
        "no data root is mounted; tried {}".format([list(c) for c in candidates]))


@dataclasses.dataclass
class SessionCandidate:
    """What is on disk for one session. Facts only -- no judgement about work.

    ``plain_minian_dir``, ``intermediate_dir`` and ``saved_movie_path`` always
    describe the session's *original* output. Once that has been set aside for a
    notebook re-run (:func:`set_aside_minian_output`), they point into the
    ``*-ORIG`` folders, and the re-run's fresh ``minian/`` is described by nothing
    here -- so no consumer can take a re-run for the original by accident.
    """

    mouse: str
    day: str
    session: str
    session_dir: str
    n_avi: int
    complete_minian_dirs: List[str]          # minian_crossreg* holding every required array
    plain_minian_dir: Optional[str]          # minian/ -- processed, never cross-registered
    intermediate_dir: Optional[str]          # minian_intermediate/ -- rare, precious
    existing_yra_path: Optional[str]
    saved_movie_path: Optional[str]          # minian_intermediate/Y_fm_chk.zarr
    set_aside_dirs: List[str]                # minian-ORIG/ etc.: see set_aside_minian_output

    @property
    def label(self) -> str:
        return "{}/{}/{}".format(self.mouse, self.day, self.session)


@dataclasses.dataclass
class SessionWork:
    """One queued item: a candidate, plus where its output goes."""

    candidate: SessionCandidate
    minian_dir: Optional[str]
    output_dir: str

    @property
    def label(self) -> str:
        return self.candidate.label

    # Read-through to the candidate's facts, so callers need not reach inside.
    @property
    def mouse(self) -> str:
        return self.candidate.mouse

    @property
    def day(self) -> str:
        return self.candidate.day

    @property
    def session(self) -> str:
        return self.candidate.session

    @property
    def session_dir(self) -> str:
        return self.candidate.session_dir

    @property
    def n_avi(self) -> int:
        return self.candidate.n_avi

    @property
    def existing_yra_path(self) -> Optional[str]:
        return self.candidate.existing_yra_path

    @property
    def saved_movie_path(self) -> Optional[str]:
        return self.candidate.saved_movie_path


def _complete_minian_dirs(session_dir: str) -> List[str]:
    return [
        d
        for d in sorted(glob.glob(os.path.join(session_dir, "minian_crossreg*")))
        if all(os.path.isdir(os.path.join(d, n + ".zarr")) for n in REQUIRED_MINIAN_ARRAYS)
    ]


def _find_existing_yra(session_dir: str, minian_dirs: List[str]) -> Optional[str]:
    """Locate an exported ``YrA.zarr``, if the session has one.

    The backup drives file it at the ``Miniscope/`` level, the CBP server files it
    inside ``minian_crossreg*``; `plans/yra_recompute_plan.md` §2.1 established the
    two are the same export, so either is fine.
    """
    for candidate in [os.path.join(d, "YrA.zarr") for d in minian_dirs] + [
        os.path.join(session_dir, "YrA.zarr")
    ]:
        if os.path.isdir(candidate):
            return candidate
    return None


def scan_sessions(
    roots=None, mouse_pattern: str = MOUSE_DIR_PATTERN, verbose: bool = True
) -> List[SessionCandidate]:
    """Report every session directory under ``roots``, with what each one holds.

    Selects nothing: a session with no video and no Minian output is still
    returned, so a pipeline that filters it out does so visibly rather than by
    omission. The printed summary accounts for everything scanned.
    """
    roots = roots or default_data_roots()
    candidates: List[SessionCandidate] = []
    for root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError("data root {} is not mounted".format(root))
        for mouse_dir in sorted(os.listdir(root)):
            mouse_path = os.path.join(root, mouse_dir)
            if not os.path.isdir(mouse_path) or not re.match(mouse_pattern, mouse_dir):
                continue
            for day in sorted(os.listdir(mouse_path)):
                day_path = os.path.join(mouse_path, day)
                if not os.path.isdir(day_path):
                    continue
                for session in sorted(os.listdir(day_path)):
                    session_dir = os.path.join(day_path, session, "Miniscope")
                    if not os.path.isdir(session_dir):
                        continue
                    entries = os.listdir(session_dir)
                    complete = _complete_minian_dirs(session_dir)
                    set_aside = _set_aside_dirs(session_dir)
                    orig = set_aside_name if set_aside else (lambda n: n)
                    plain = os.path.join(session_dir, orig("minian"))
                    intermediate = os.path.join(session_dir, orig("minian_intermediate"))
                    movie = os.path.join(intermediate, "Y_fm_chk.zarr")
                    candidates.append(SessionCandidate(
                        mouse=mouse_dir[:3],
                        day=day,
                        session=session,
                        session_dir=session_dir,
                        n_avi=len([v for v in entries if re.search(VIDEO_PATTERN, v)]),
                        complete_minian_dirs=complete,
                        plain_minian_dir=plain if all(
                            os.path.isdir(os.path.join(plain, n + ".zarr"))
                            for n in REQUIRED_MINIAN_ARRAYS
                        ) else None,
                        intermediate_dir=intermediate if os.path.isdir(intermediate) else None,
                        existing_yra_path=_find_existing_yra(session_dir, complete),
                        saved_movie_path=movie if os.path.isdir(movie) else None,
                        set_aside_dirs=set_aside,
                    ))
    if verbose:
        with_video = [c for c in candidates if c.n_avi]
        print("scanned {} Miniscope folders under {} root(s)".format(
            len(candidates), len(roots)))
        print("  {} have numbered .avi files".format(len(with_video)))
        print("  {} have a complete minian_crossreg*".format(
            sum(1 for c in with_video if c.complete_minian_dirs)))
        print("  {} have only a plain minian/ (never cross-registered)".format(
            sum(1 for c in with_video if not c.complete_minian_dirs and c.plain_minian_dir)))
        print("  {} have no Minian output at all".format(
            sum(1 for c in with_video
                if not c.complete_minian_dirs and not c.plain_minian_dir)))
        print("  {} carry an exported YrA.zarr; {} kept a minian_intermediate/".format(
            sum(1 for c in with_video if c.existing_yra_path),
            sum(1 for c in with_video if c.intermediate_dir)))
        set_aside = [c for c in candidates if c.set_aside_dirs]
        if set_aside:
            print("  {} have their original Minian output set aside as *{} (reported "
                  "above from there; any minian/ beside it is a re-run):".format(
                      len(set_aside), SET_ASIDE_SUFFIX))
            for c in set_aside:
                print("    {}".format(c.label))
    return candidates


# --- setting existing output aside before a notebook re-run ------------------


def set_aside_name(name: str) -> str:
    """``minian`` -> ``minian-ORIG``; ``mappings_crossreg_7.csv`` -> ``mappings_crossreg_7-ORIG.csv``.

    The suffix goes before the extension, so a set-aside file still opens as what it is.
    """
    stem, ext = os.path.splitext(name)
    return stem + SET_ASIDE_SUFFIX + ext


def original_output_path(path: str) -> str:
    """The original of an output that a re-run may have replaced.

    Returns the ``-ORIG`` sibling when one exists, else ``path`` itself -- so a reader
    routed through here keeps reading the original after a re-run writes beside it.
    """
    orig = os.path.join(os.path.dirname(path), set_aside_name(os.path.basename(path)))
    return orig if os.path.exists(orig) else path


def _set_aside_dirs(session_dir: str) -> List[str]:
    return [
        os.path.join(session_dir, set_aside_name(name))
        for name in NOTEBOOK_OUTPUT_DIRS
        if os.path.isdir(os.path.join(session_dir, set_aside_name(name)))
    ]


def _set_aside(base_dir: str, names, record_name: str, reason: str) -> dict:
    """Rename whichever of ``names`` exist in ``base_dir`` to their ``-ORIG`` names.

    Refuses, touching nothing, if any ``-ORIG`` target or the record already exists --
    renaming again would move a re-run's output into ``-ORIG`` and lose the original
    -- and if none of ``names`` is present, so a call never implies an original was
    kept when there was none. Records what it did in ``record_name``.
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError("{} is not a directory".format(base_dir))
    taken = [set_aside_name(n) for n in names
             if os.path.lexists(os.path.join(base_dir, set_aside_name(n)))]
    record_path = os.path.join(base_dir, record_name)
    if taken or os.path.lexists(record_path):
        raise FileExistsError(
            "{} already has set-aside output {} (record: {}); renaming again would "
            "overwrite the original with a re-run. Resolve by hand.".format(
                base_dir, taken, os.path.exists(record_path)))
    present = [n for n in names if os.path.lexists(os.path.join(base_dir, n))]
    if not present:
        raise FileNotFoundError("{} has none of {} -- nothing to set aside".format(
            base_dir, list(names)))
    renamed = {}
    for name in present:
        os.rename(os.path.join(base_dir, name), os.path.join(base_dir, set_aside_name(name)))
        renamed[name] = set_aside_name(name)
    record = {"renamed": renamed, "reason": reason,
              "when": pd.Timestamp.now().isoformat(timespec="seconds")}
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    return record


def scratch_target(session_dir: str, scratch_root: str) -> str:
    """``<scratch_root>/<mouse>/<day>/<session>/minian_intermediate``, checked unused."""
    if not os.path.isdir(scratch_root):
        raise FileNotFoundError("scratch root {} is not mounted".format(scratch_root))
    session, day, mouse = (os.path.basename(p) for p in (
        os.path.dirname(session_dir),
        os.path.dirname(os.path.dirname(session_dir)),
        os.path.dirname(os.path.dirname(os.path.dirname(session_dir)))))
    target = os.path.join(os.path.abspath(scratch_root), mouse, day, session, SCRATCH_DIR_NAME)
    if os.path.lexists(target):
        raise FileExistsError(
            "scratch target {} already exists -- a leftover from an earlier run? Delete "
            "it by hand if it is disposable.".format(target))
    return target


def set_aside_minian_output(session_dir: str, reason: str, scratch_root: Optional[str] = None,
                            link_scratch: bool = True) -> dict:
    """Prepare a session's ``Miniscope/`` folder for a pipeline-notebook run.

    Renames existing notebook output -- ``minian/``, ``minian_intermediate/``,
    ``minian_run/``, ``minian.mp4``, ``minian_mc.mp4`` -- to its ``-ORIG`` names (instant on APFS,
    nothing copied), with the refusals of :func:`_set_aside`.

    With ``scratch_root``, then makes ``minian_intermediate`` a symlink to a fresh
    ``<scratch_root>/<mouse>/<day>/<session>/minian_intermediate``: the notebook
    writes to ``dpath/minian_intermediate`` unchanged, and the heavy intermediates land
    on the fast drive. A session with no output yet needs only this step, so that is
    not an error. Every check runs before anything is renamed or created.

    ``link_scratch=False`` creates the scratch folder but no symlink: the caller points the
    notebook's ``intpath`` at it instead (``caban.minian_runner`` does, 2026-09-27 -- NTFS seen
    from WSL cannot hold the link reliably, and the session folder then holds only output).

    A ``minian_intermediate`` that is already a symlink is an earlier run's scratch,
    not an original, and is refused: remove the link (and its scratch folder, if
    disposable) first. After a run, delete both when the intermediates are no longer
    needed -- except for a gate re-run, which is compared against them.
    """
    session_dir = os.path.abspath(session_dir)
    link = os.path.join(session_dir, SCRATCH_DIR_NAME)
    if os.path.islink(link):
        raise FileExistsError(
            "{} is a symlink to {} -- an earlier run's scratch, not an original. Remove "
            "the link first.".format(link, os.readlink(link)))
    target = scratch_target(session_dir, scratch_root) if scratch_root is not None else None
    names = NOTEBOOK_OUTPUT_DIRS + NOTEBOOK_OUTPUT_FILES
    has_output = any(os.path.lexists(os.path.join(session_dir, n)) for n in names)
    record = {}
    if has_output or target is None:
        record = _set_aside(session_dir, names, SET_ASIDE_RECORD, reason)
    if target is not None:
        os.makedirs(target)
        if link_scratch:
            os.symlink(target, link)
        record["scratch"] = target
        record["scratch_linked"] = link_scratch
    return record


def set_aside_crossreg_output(mouse_dir: str, name: str, reason: str) -> dict:
    """Rename a mouse's cross-registration output ``<name>`` to its ``-ORIG`` names.

    ``name`` is what the notebook builds from ``f_pattern_prefix`` and ``f_pattern``,
    e.g. ``crossreg_7`` for ``mappings_crossreg_7.csv``. Readers in ``caban`` go
    through :func:`original_output_path`, so they keep loading the original.
    """
    files = ("mappings_{}.pkl", "mappings_{}.csv", "cents_{}.pkl", "shiftds_{}.nc")
    return _set_aside(mouse_dir, tuple(f.format(name) for f in files),
                      "crossreg_set_aside_{}.json".format(name), reason)


def resolve_minian_dir(candidate: SessionCandidate, overrides: Optional[dict] = None) -> str:
    """Pick the one Minian output directory meant for this session.

    Two complete ``minian_crossreg*`` folders is a hard failure, not a coin flip:
    G16's ``2022_01_26-TFC_test_B / 14_23_45-LT1`` has two that disagree about both
    unit and frame count, and only one of them matches the session's own videos.
    Pin it through ``overrides``, keyed by the ``<mouse>/<day>/<session>`` label,
    whose value is the chosen folder's *name* (e.g. ``"minian_crossreg3"``) so the
    pin holds whichever drive the session is read from.
    """
    overrides = overrides or {}
    complete = candidate.complete_minian_dirs
    if candidate.label in overrides:
        name = overrides[candidate.label]
        if os.sep in name:
            raise ValueError(
                "override for {} is a path ({}); give the folder name only, so it does "
                "not depend on which drive is mounted".format(candidate.label, name))
        chosen = os.path.join(candidate.session_dir, name)
        if chosen not in complete:
            raise ValueError(
                "override for {} names {}, which is not one of the complete Minian "
                "outputs {}".format(candidate.label, chosen, complete)
            )
        return chosen
    if not complete:
        raise ValueError("{} has no complete Minian output".format(candidate.label))
    if len(complete) > 1:
        raise ValueError(
            "{} has {} complete Minian outputs and no override:\n  {}\n"
            "Pick one explicitly via the overrides map -- they can disagree about both "
            "unit and frame count, so there is no safe default.".format(
                candidate.label, len(complete), "\n  ".join(complete))
        )
    return complete[0]


def session_output_dir(candidate: SessionCandidate, output_root: str) -> str:
    """`<output_root>/<mouse>/<day>/<session>` -- unique across all 657 sessions."""
    return os.path.join(
        os.path.expanduser(output_root), candidate.mouse, candidate.day, candidate.session
    )


# --- sidecars: the record of what has been done ----------------------------


def sidecar_path(output_dir: str, sidecar_name: str) -> str:
    return os.path.join(output_dir, sidecar_name)


def read_sidecar(output_dir: str, sidecar_name: str) -> dict:
    path = sidecar_path(output_dir, sidecar_name)
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def write_sidecar(output_dir: str, sidecar_name: str, record: dict) -> str:
    path = sidecar_path(output_dir, sidecar_name)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    return path


def output_present(item: SessionWork, sidecar_name: str, output_name: str) -> bool:
    """Cheap "looks finished" test: a complete sidecar with its output beside it.

    Deliberately cheap, because the queue view calls it for every session while the
    authoritative test re-hashes the inputs. A session that only *looks* finished is
    still caught and redone by the pipeline -- this is the index, not the verdict.
    """
    record = read_sidecar(item.output_dir, sidecar_name)
    return bool(record.get("complete")) and os.path.isdir(
        os.path.join(item.output_dir, output_name)
    )


def next_pending(
    items: List[SessionWork], sidecar_name: str, output_name: str
) -> Optional[SessionWork]:
    """The first session not already done, or ``None`` when all are."""
    for item in items:
        if not output_present(item, sidecar_name, output_name):
            return item
    return None


def queue_frame(
    items: List[SessionWork],
    sidecar_name: str,
    output_name: str,
    extra_columns: Optional[Callable[[SessionWork, dict], dict]] = None,
) -> pd.DataFrame:
    """One row per session. ``extra_columns`` adds the pipeline's own reporting.

    It receives the item and its sidecar record (``{}`` when absent) and returns a
    dict of columns, so the shared view carries task-specific numbers without this
    module knowing anything about them.
    """
    rows = []
    for item in items:
        record = read_sidecar(item.output_dir, sidecar_name)
        row = {
            "mouse": item.mouse,
            "day": item.day,
            "session": item.session,
            "n_avi": item.n_avi,
            "done": output_present(item, sidecar_name, output_name),
        }
        if extra_columns:
            row.update(extra_columns(item, record))
        rows.append(row)
    return pd.DataFrame(rows)


def run_queue(
    items: List[SessionWork],
    runner: Callable[..., dict],
    stop_on_error: bool = True,
    **kwargs,
) -> List[dict]:
    """Work through a queue serially, printing each report before the next session.

    Serial by design (`plans/yra_recompute_plan.md` §11.5): each session is already
    memory- and I/O-bound through its own dask graph. ``stop_on_error`` left at
    ``True`` means a failed hard check halts the batch, which is the point of having
    hard checks -- set it to ``False`` only for an unattended sweep whose failures
    you intend to read afterwards in the returned list.
    """
    records = []
    for n, item in enumerate(items, start=1):
        print("\n{}\n[{}/{}] {}\n{}".format("=" * 78, n, len(items), item.label, "=" * 78))
        try:
            records.append(runner(item, **kwargs))
        except Exception as error:
            if stop_on_error:
                raise
            print("FAILED: {}: {}".format(type(error).__name__, error))
            records.append({
                "label": item.label,
                "failed": "{}: {}".format(type(error).__name__, error),
            })
    return records

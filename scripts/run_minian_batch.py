"""Run the Minian batch headless -- the Choose and Run cells of
`notebooks/run_minian_pipeline.ipynb` as a command, for `screen` sessions on the Razer
(`plans/razer_runner_plan.md` Phase 4). Unattended: a failed session is recorded and the
queue moves on; its record says why.

Selection is the notebook's: nothing = every session that needs Minian; --labels (regex
patterns in <mouse>/<day>/<session>), --mice and --session-types narrow it (AND); --path
names exact sessions, processed or not, and stands alone.

With a stage root (the Razer's default) the copier moves every session's videos onto local
disk before its run and its results back after, in a background thread
(plans/session_staging_copier_plan.md).

Run with the caban env activated -- ffmpeg/ffprobe come from it:
    conda activate caban
    python scripts/run_minian_batch.py --mice G05 G06 --n-workers 4
"""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import minian_runner as mr
from caban import session_queue as sq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", nargs="+", default=None,
                        help="folders holding <mouse>/<day>/<session> (default: the mounted one, "
                             "see session_queue.default_data_roots)")
    parser.add_argument("--scratch-root", default=mr.DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--stage-root", default=mr.DEFAULT_STAGE_ROOT,
                        help="work folders on local disk: each session's videos are copied there, "
                             "the run works there, results are copied back -- by the copier "
                             "(plans/session_staging_copier_plan.md); default ~/minian_stage on Linux. "
                             "Without it (the Mac's default) runs work in the session folder.")
    parser.add_argument("--n-workers", type=int, default=6, help="dask workers (MINIAN_NWORKERS)")
    parser.add_argument("--keep-scratch", action="store_true", help="keep intermediates (gate runs)")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--mice", nargs="+")
    parser.add_argument("--session-types", nargs="+")
    parser.add_argument("--path", nargs="+")
    parser.add_argument("--by-day", action="store_true",
                        help="the whole batch in the order of minian_runner.BATCH_DAY_GROUPS (whole "
                             "experiment days first); stops between groups and sessions at the stop "
                             "switch ~/minian_stop. Needs a stage root; no other selection.")
    parser.add_argument("--dry-run", action="store_true", help="show the queue, run nothing")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit("{} not on PATH -- activate the caban env first".format(tool))
    # Once up front, so a missing package stops the batch rather than failing every session;
    # run_session repeats it per session.
    print("pre-flight: {}".format(mr.preflight_kernel(mr.load_template())))

    pd.set_option("display.max_rows", 1000, "display.width", 220)
    items = mr.discover_sessions(roots=args.data_root)
    if args.by_day:
        return run_by_day(items, args)
    queue = mr.select_sessions(items, labels=args.labels, mice=args.mice,
                               session_types=args.session_types, path=args.path)
    status = mr.queue_status(queue)
    print(status["status"].replace("", "not run").value_counts().to_string())
    print(status.fillna("").to_string())
    if args.dry_run:
        return 0

    run_options = dict(scratch_root=args.scratch_root, keep_scratch=args.keep_scratch,
                       n_workers=args.n_workers)
    if args.stage_root:
        print("staged: work folders under {}".format(args.stage_root))
        records = mr.run_all_staged(queue, args.stage_root, **run_options)
    else:
        print("not staged: runs work in the session folders")
        records = mr.run_all(queue, attended=False, **run_options)
    failed = [r for r in records if "failed" in r]
    print("\n{} sessions run; {} failed".format(len(records), len(failed)))
    for r in failed:
        print("  {}: {}".format(r["label"], r["failed"]))
    return 1 if failed else 0


def run_by_day(items, args) -> int:
    """The batch in BATCH_DAY_GROUPS order, one group after another, each staged."""
    if args.labels or args.mice or args.session_types or args.path:
        raise SystemExit("--by-day takes no other selection")
    if not args.stage_root:
        raise SystemExit("--by-day needs a stage root")
    if os.path.exists(mr.STOP_FILE) and not args.dry_run:
        raise SystemExit("{} exists; delete it to start the batch".format(mr.STOP_FILE))
    failed = []
    for name, patterns in mr.BATCH_DAY_GROUPS:
        if os.path.exists(mr.STOP_FILE):
            print("=== {} exists: stopping before the group '{}' ===".format(mr.STOP_FILE, name))
            break
        print("\n=== batch: {} {}, {} ===".format(name, list(patterns) or "(all)",
                                               pd.Timestamp.now().strftime("%F %T")))
        queue = mr.select_sessions(items, labels=list(patterns) or None)
        pending = mr.pending_items(queue)
        print("{}: {} sessions, {} pending".format(name, len(queue), len(pending)))
        if args.dry_run or not pending:
            continue
        records = mr.run_all_staged(pending, args.stage_root, scratch_root=args.scratch_root,
                                    keep_scratch=args.keep_scratch, n_workers=args.n_workers)
        failed += [r for r in records if "failed" in r]
        print("=== batch: {} ended: {} run, {} failed ===".format(
            name, len(records), sum("failed" in r for r in records)))
    print("\n{} sessions failed in this batch".format(len(failed)))
    for r in failed:
        print("  {}: {}".format(r["label"], r["failed"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

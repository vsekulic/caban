"""Run the Minian batch headless -- the Choose and Run cells of
`notebooks/run_minian_pipeline.ipynb` as a command, for `screen` sessions on the Razer
(`plans/razer_runner_plan.md` Phase 4). Unattended: a failed session is recorded and the
queue moves on; its record says why.

Selection is the notebook's: nothing = every session that needs Minian; --labels (regex
patterns in <mouse>/<day>/<session>), --mice and --session-types narrow it (AND); --path
names exact sessions, processed or not, and stands alone.

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
    parser.add_argument("--n-workers", type=int, default=6, help="dask workers (MINIAN_NWORKERS)")
    parser.add_argument("--keep-scratch", action="store_true", help="keep intermediates (gate runs)")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--mice", nargs="+")
    parser.add_argument("--session-types", nargs="+")
    parser.add_argument("--path", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="show the queue, run nothing")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit("{} not on PATH -- activate the caban env first".format(tool))

    pd.set_option("display.max_rows", 1000, "display.width", 220)
    items = mr.discover_sessions(roots=args.data_root)
    queue = mr.select_sessions(items, labels=args.labels, mice=args.mice,
                               session_types=args.session_types, path=args.path)
    status = mr.queue_status(queue)
    print(status["status"].replace("", "not run").value_counts().to_string())
    print(status.fillna("").to_string())
    if args.dry_run:
        return 0

    records = mr.run_all(queue, attended=False, scratch_root=args.scratch_root,
                         keep_scratch=args.keep_scratch, n_workers=args.n_workers)
    failed = [r for r in records if "failed" in r]
    print("\n{} sessions run; {} failed".format(len(records), len(failed)))
    for r in failed:
        print("  {}: {}".format(r["label"], r["failed"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

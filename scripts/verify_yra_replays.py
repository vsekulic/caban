"""Re-check production YrA recomputes against production's max_proj (caban.yra_recompute.verify_all;
plans/yra_recompute_plan.md §14.2) -- the gate the analysis loader requires before it loads a
recomputed YrA (plans/figure2_per_cell_completion_plan.md §2.2).

    python scripts/verify_yra_replays.py --tfc-cond only|exclude|all [--data-root ROOT]

Same sessions and G16 override as notebooks/recompute_yra.ipynb. Records the result in each session's
YrA_recompute.json on the drive it runs against; YrA_recomputed.zarr is never rewritten. Sessions
already passing are skipped. `--tfc-cond` splits the work between machines (2026-09-29: the Razer
the 17 TFC_cond sessions, the Mac the rest on MINISCOPE); the updated sidecars are then copied across.
"""

import argparse
import os
import sys

import dask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import yra_recompute as yr

MINIAN_DIR_OVERRIDES = {"G16/2022_01_26-TFC_test_B/14_23_45-LT1": "minian_crossreg3"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tfc-cond", choices=("only", "exclude", "all"), required=True)
    parser.add_argument("--data-root", nargs="+", default=None)
    parser.add_argument("--n-workers", type=int, default=6)
    args = parser.parse_args()
    items = yr.discover_sessions(roots=args.data_root, minian_dir_overrides=MINIAN_DIR_OVERRIDES)
    is_cond = [i.session.endswith("-TFC_cond") for i in items]
    if args.tfc_cond == "only":
        items = [i for i, c in zip(items, is_cond) if c]
    elif args.tfc_cond == "exclude":
        items = [i for i, c in zip(items, is_cond) if not c]
    print("{} recomputes to check ({})".format(len(items), args.tfc_cond))
    with dask.config.set(scheduler="threads", num_workers=args.n_workers):
        records = yr.verify_all(items, stop_on_error=False)
    failed = [r for r in records if "failed" in r]
    print("\n{} of {} replays match max_proj; {} failed".format(
        len(records) - len(failed), len(records), len(failed)))
    for r in failed:
        print("  {}: {}".format(r["label"], r["failed"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

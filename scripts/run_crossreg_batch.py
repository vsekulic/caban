"""Run cross-registration headless over the registry (caban.crossreg_runner), for `screen` on the Razer.
Unattended: a failed run is recorded (its run.json and the log say why) and the batch moves on.

    conda activate caban
    python scripts/run_crossreg_batch.py --gate                  # G10 grouping 4, G17 groupings 1 and 6
    python scripts/run_crossreg_batch.py --published             # groupings 1-7, every mouse
    python scripts/run_crossreg_batch.py --groupings 8 9 --mice G05
    python scripts/run_crossreg_batch.py --published --dry-run   # members of each run, nothing run

Runs already done (their run.json says so) are passed over. Stops between runs at ~/crossreg_stop.
A status table is rewritten after every run (~/crossreg_batch_status.md).
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import crossreg_runner as cr

GATE_RUNS = [("G10", "4"), ("G17", "1"), ("G17", "6")]
PUBLISHED = ["1", "2", "3", "4", "5", "6", "7"]
STOP_FILE = os.path.expanduser("~/crossreg_stop")
STATUS_PATH = os.path.expanduser("~/crossreg_batch_status.md")


def status_table(registry: dict, runs, root: str) -> pd.DataFrame:
    rows = []
    for mouse, n in runs:
        record = cr.read_record(cr.mouse_dir(registry, n, mouse, root), n) or {}
        comparison = record.get("comparison") or {}
        rows.append({
            "mouse": mouse, "grouping": n, "status": record.get("status", "pending"),
            "sessions": len(registry["groupings"][n]["mice"][mouse]["sessions"]),
            "rows": record.get("n_mapping_rows", ""), "in_all": record.get("n_matched_in_all", ""),
            "vs_published": {True: "identical", False: "DIFFERS"}.get(comparison.get("identical"), ""),
            "min": round(record.get("timings", {}).get("execute_s", 0) / 60, 1) or "",
            "peak_GB": record.get("memory", {}).get("peak_rss_gb", ""),
            "note": record.get("error") or comparison.get("expected_difference") or "",
        })
    return pd.DataFrame(rows)


def write_status(table: pd.DataFrame) -> None:
    with open(STATUS_PATH, "w") as fh:
        fh.write("# Cross-registration batch ({})\n\n".format(pd.Timestamp.now().isoformat(timespec="minutes")))
        fh.write(table.to_markdown(index=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--gate", action="store_true", help="the gate runs: {}".format(GATE_RUNS))
    which.add_argument("--published", action="store_true", help="groupings 1-7")
    which.add_argument("--groupings", nargs="+")
    parser.add_argument("--mice", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = cr.load_registry()
    root = cr.data_root()
    if args.gate:
        runs = GATE_RUNS
    else:
        runs = cr.planned_runs(registry, PUBLISHED if args.published else args.groupings, args.mice)

    if args.dry_run:
        for mouse, n in runs:
            mems = cr.members(registry, n, mouse, root)
            print("{} grouping {}: {} sessions".format(mouse, n, len(mems)))
            for m in mems:
                print("    {}  {}  (stored {!r})".format(m["name"], os.path.join(m["session"], m["output"]),
                                                         m["stored_name"]))
        print(status_table(registry, runs, root).to_string(index=False))
        return 0

    if os.path.exists(STOP_FILE):
        raise SystemExit("{} exists; delete it to start the batch".format(STOP_FILE))
    template = cr.load_template()
    print("pre-flight: {}".format(cr.preflight(template)))
    failed = []
    for k, (mouse, n) in enumerate(runs, 1):
        if os.path.exists(STOP_FILE):
            print("=== {} exists: stopping before {} grouping {} ===".format(STOP_FILE, mouse, n))
            break
        record = cr.read_record(cr.mouse_dir(registry, n, mouse, root), n)
        if record is not None and record.get("status") == "done":
            continue
        print("=" * 78)
        print("[{}/{}] {} grouping {} ({})".format(k, len(runs), mouse, n, registry["groupings"][n]["name"]))
        print("=" * 78, flush=True)
        started = time.time()
        try:
            record = cr.run_one(registry, mouse, n, template, root=root)
            comparison = record.get("comparison")
            print("{} grouping {}: done in {:.1f} min, {} rows, {} in all sessions{}".format(
                mouse, n, (time.time() - started) / 60, record["n_mapping_rows"], record["n_matched_in_all"],
                "" if comparison is None else "; vs published: {}".format(
                    "identical" if comparison["identical"] else "DIFFERS")), flush=True)
        except Exception as exc:
            failed.append((mouse, n))
            print("{} grouping {}: FAILED: {}: {}".format(mouse, n, type(exc).__name__, exc), flush=True)
        write_status(status_table(registry, runs, root))
    table = status_table(registry, runs, root)
    write_status(table)
    print(table.to_string(index=False))
    print("{} failed: {}".format(len(failed), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Write the Minian batch's status report: every session with video in G05-G21, what is done, what
is missing, and whether each result has reached MINISCOPE. Generated from the runs' own records
(`minian_run/run.json`, `YrA_recompute.json`, the sync record), never filled in by hand.

    python scripts/minian_status_report.py reports/minian_batch_status.md

Run on the Razer, where MINIRAZER is mounted (the data root the scan finds); the report is then
committed in the repo as `reports/minian_batch_status.md`.
"""

import collections
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import minian_runner as mr
from caban import session_queue as sq
from caban import yra_recompute as yr

# Written into minian_run/ by the sync MINIRAZER -> MINISCOPE after a verified copy (the sync
# script, handover §4); read here only.
SYNC_RECORD = "synced_to_miniscope.json"
MAC_DATA_ROOT = "/Volumes/MINISCOPE"

# What the records cannot say (VS and the plans; each entry cites where it was decided).
NOTES = {
    "G05/2021_09_07-TFC_test_B_1wk-redux/16_22_40-TFC_test_B_1wk":
        "test day (bedding change?); processed, not for analysis by default (VS, open items)",
    "G07/2021_10_20-TFC_test_B/15_49_59-TFC_test_B":
        "unfinished recordings (no frame count in headers; 0.47 GB for 18 files); VS to decide",
    "G14/2022_01_18-TFC_test_B_1wk/14_52_35-TFC_test_B_1wk-borked":
        "unfinished recordings (no frame count in headers); VS to decide",
    "G09/2021_11_08-TFC_cond/19_28_33-HC3": "partial old output, all-NaN motion; VS to decide (open items 10)",
    "G05/2021_08_30-TFC_cond/18_22_57-TFC_cond":
        "production; also the Razer production check's re-run in minian/ (razer plan Phase 3.2)",
    "G10/2021_11_23-TFC_cond/16_32_14-TFC_cond":
        "production; also the Mac gate re-run in minian/ (batch runner plan §13)",
}

STATUS_ORDER = ("done", "computed", "running", "failed", "interrupted", "pending")


def category(item) -> str:
    """production / a runner status / an exclusion -- one word for the table."""
    c = item.candidate
    if c.complete_minian_dirs:
        return "production"
    if mr.needs_minian(item):
        return "stub (excluded)" if mr.is_stub(item) else mr.run_status(item)
    if c.set_aside_dirs:
        return "set aside by hand (excluded)"
    return "partial output (excluded)"


def yra_cell(item, cat: str) -> str:
    if cat == "production":
        record = {}
        for d in item.candidate.complete_minian_dirs:
            record = sq.read_sidecar(d, yr.SIDECAR_NAME) or record
        return "Y" if record.get("complete") else "N"
    if cat in ("done", "computed"):
        return "Y" if mr.yra_recomputed(item) else "N"
    return ""


def frames_checked(item) -> str:
    """The full-frame check: how many frames of the replayed movie were compared, and the result."""
    sidecar = sq.read_sidecar(mr.yra_output_dir(item), yr.SIDECAR_NAME)
    check = sidecar.get("report", {}).get("replayed_Y_vs_saved_Y_fm_chk")
    if not check:
        return "-"
    whole = check["n_frames_compared"] == sidecar["report"]["n_frames"]
    return "{} {} {}".format("all" if whole else "sample of", check["n_frames_compared"],
                             "equal" if check["exactly_equal"] else "differ: max {:g} at {} px".format(
                                 check["max_abs_diff"], check["n_pixels_differing"]))


def miniscope_cell(item, cat: str, record: dict) -> str:
    if cat == "production":
        return "Y (source)"
    if cat not in ("done",):
        return ""
    sync = sq.read_sidecar(os.path.join(item.session_dir, mr.RUN_DIR_NAME), SYNC_RECORD)
    if sync:
        return "Y ({})".format(sync.get("when", "")[:10])
    if record.get("session_dir", "").startswith(MAC_DATA_ROOT):
        yra = sq.read_sidecar(mr.yra_output_dir(item), yr.SIDECAR_NAME)
        return "partial (YrA not)" if yra.get("session_dir", "").startswith("/mnt/") else "Y (Mac run)"
    return "N"


def row(item) -> dict:
    cat = category(item)
    record = mr.run_record(item) if cat in STATUS_ORDER and cat != "pending" else {}
    report = record.get("report", {})
    note = NOTES.get(item.label, "")
    if cat in ("failed", "interrupted") and not note:
        note = record.get("error", "")[:110]
    if record.get("stage_out_error"):
        note = "stage-out failed: " + record["stage_out_error"][:90]
    return {
        "mouse": item.mouse, "day": item.day, "session": item.session,
        "type": mr.session_type(item.session), "videos": item.n_avi,
        "group": mr.day_group(item.label), "status": cat,
        "finished": (record.get("staged_out") or record.get("finished") or "")[:16].replace("T", " "),
        "wall_h": "{:.2f}".format(record["timings"]["total_s"] / 3600) if "total_s" in record.get("timings", {}) else "",
        "units": report.get("n_units", ""),
        "YrA": yra_cell(item, cat),
        "frames_checked": frames_checked(item) if cat == "done" else "",
        "on_MINISCOPE": miniscope_cell(item, cat, record),
        "note": note,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def main(out_path: str) -> None:
    items = mr.discover_sessions(verbose=False)
    rows = pd.DataFrame([row(i) for i in items])
    group_names = [g for g, _ in mr.BATCH_DAY_GROUPS]
    runner = rows[rows["status"].isin(STATUS_ORDER)]

    summary = []
    for g in group_names:
        sub = runner[runner["group"] == g]
        counts = collections.Counter(sub["status"])
        summary.append({"#": group_names.index(g) + 1, "group": g, "sessions": len(sub),
                        **{s: counts.get(s, 0) for s in STATUS_ORDER},
                        "on MINISCOPE": int(sub["on_MINISCOPE"].str.startswith("Y").sum())})
    summary = pd.DataFrame(summary)
    totals = {"#": "", "group": "**all**", "sessions": int(summary["sessions"].sum()),
              **{s: int(summary[s].sum()) for s in STATUS_ORDER},
              "on MINISCOPE": int(summary["on MINISCOPE"].sum())}
    summary = pd.concat([summary, pd.DataFrame([totals])], ignore_index=True)

    running = runner[runner["status"].isin(("running", "computed"))]
    pending = runner[runner["status"] == "pending"].copy()
    pending["order"] = pending["group"].map(group_names.index)
    upcoming = pending.sort_values(["order", "mouse", "day", "session"]).head(5)
    other = collections.Counter(rows.loc[~rows["status"].isin(STATUS_ORDER), "status"])

    lines = [
        "# Minian batch: status of every session",
        "",
        "Generated {} on the Razer from MINIRAZER by `scripts/minian_status_report.py` -- from the runs' "
        "own records, not by hand. How the batch runs: "
        "[plans/razer_runner_plan.md](../plans/razer_runner_plan.md) (Phase 4), "
        "[plans/session_staging_copier_plan.md](../plans/session_staging_copier_plan.md).".format(
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
        "",
        "## Summary",
        "",
        "Sessions the batch processes, by priority group (the batch runs the groups in this order; "
        "`--by-day`). **{}** sessions in all with video in G05-G21: {} in the batch; {}.".format(
            len(rows), len(runner), ", ".join("{} {}".format(n, s) for s, n in sorted(other.items()))),
        "",
        markdown_table(summary),
        "",
        "**Now**: " + ("; ".join("{}/{}/{} ({})".format(r.mouse, r.day, r.session, r.status)
                                 for r in running.itertuples()) or "nothing running"),
        "",
        "**Next**: " + ("; ".join("{}/{}/{}".format(r.mouse, r.day, r.session)
                                  for r in upcoming.itertuples()) or "nothing pending"),
        "",
        "## Columns",
        "",
        "- **status**: `done` (results in the session folder on MINIRAZER), `computed` (finished, being "
        "copied back from the work folder), `running`, `failed` (see note), `pending`; `production` = "
        "the 130 published, cross-registered sessions (never re-run); `stub` / `partial output` / "
        "`set aside by hand` = left out of the batch (open items 10).",
        "- **group**: the batch priority group (whole TFC/test days first, then track days 1-3, then the rest).",
        "- **finished**: when the results reached the session folder. **wall_h**: the run's own time.",
        "- **YrA**: `YrA_recomputed.zarr` present (the corrected residual traces).",
        "- **frames_checked**: the full-frame check -- the replayed movie against the notebook's own, "
        "`all N equal` since 2026-09-28; `sample of 50` before. On runs made on the Mac a difference of "
        "max 1 grey level is expected (the Mac's cv2 5.0 against the notebook's 4.5; batch runner plan §13).",
        "- **on_MINISCOPE**: `Y (date)` = copied back and verified by the sync; `Y (Mac run)` = run on "
        "the Mac, so already there; `partial (YrA not)` = run on the Mac, YrA recomputed later on the "
        "Razer; `N` = only on MINIRAZER so far; `Y (source)` = production, which came from MINISCOPE.",
        "",
    ]
    for mouse in sorted(rows["mouse"].unique()):
        sub = rows[rows["mouse"] == mouse].drop(columns="mouse")
        n_done = int((sub["status"] == "done").sum())
        n_batch = int(sub["status"].isin(STATUS_ORDER).sum())
        lines += ["## {} -- {} of {} batch sessions done".format(mouse, n_done, n_batch), "",
                  markdown_table(sub), ""]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    counts = collections.Counter(runner["status"])
    print("status {}: {} of {} batch sessions done, {} failed, {} pending; now: {} -> {}".format(
        pd.Timestamp.now().strftime("%H:%M"), counts.get("done", 0), len(runner),
        counts.get("failed", 0) + counts.get("interrupted", 0), counts.get("pending", 0),
        "; ".join("{}/{}/{}".format(r.mouse, r.day, r.session) for r in running.itertuples()) or "nothing",
        out_path))


if __name__ == "__main__":
    main(sys.argv[1])

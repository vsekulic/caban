# reports/ index

Status and results reports for caban, in the order the work was taken up. See also: [docs index](../docs/0_INDEX_docs.md), [plans index](../plans/0_INDEX_plans.md), [notebooks index](../notebooks/0_INDEX_notebooks.md).

| # | Report | Date | Description | Plan |
|---|---|---|---|---|
| 1 | [minian_batch_status.md](minian_batch_status.md) | 2026-09-28 (upd. 10-07) | Status of every session in the Minian batch, generated on the Razer by `scripts/minian_status_report.py` from the runs' own records, not written by hand. Snapshot of 2026-10-07 00:13, after the final sync: the batch is finished — 484 of 492 sessions done and all 484 on MINISCOPE, 8 failed, nothing running or pending; TFC_cond is 86 of 88 done. Of the failures, five are a papermill error at the `varr = load_…` cell, two are unfinished recordings left for VS to decide, and one (G18 `CNO2-borked`) is a `SparseArray` ValueError. Regenerated in place, so it is a moving snapshot. | [razer_runner_plan.md](../plans/razer_runner_plan.md) (Phase 4), [session_staging_copier_plan.md](../plans/session_staging_copier_plan.md) |
| 2 | [crossreg_groupings_report.md](crossreg_groupings_report.md) | 2026-10-11 | Everything about cross-registration in one place: how it works, the 18 groupings and why, the conditioning-day roles across the two cohorts, what the checks found in the published mappings (G15 stale grouping 2, G16 duplicate, G17 naming), VS's session notes, and every unused session with its reason. | — |

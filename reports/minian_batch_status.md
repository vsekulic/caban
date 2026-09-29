# Minian batch: status of every session

Generated 2026-09-29 12:29 on the Razer from MINIRAZER by `scripts/minian_status_report.py` -- from the runs' own records, not by hand. How the batch runs: [plans/razer_runner_plan.md](../plans/razer_runner_plan.md) (Phase 4), [plans/session_staging_copier_plan.md](../plans/session_staging_copier_plan.md).

## Summary

Sessions the batch processes, by priority group (the batch runs the groups in this order; `--by-day`). **657** sessions in all with video in G05-G21: 492 in the batch; 1 partial output (excluded), 130 production, 34 stub (excluded).

| # | group | sessions | done | computed | running | failed | interrupted | pending | on MINISCOPE |
|---|---|---|---|---|---|---|---|---|---|
| 1 | TFC_cond days | 88 | 60 | 0 | 1 | 0 | 0 | 27 | 16 |
| 2 | TFC_test_B days | 54 | 1 | 0 | 0 | 1 | 0 | 52 | 1 |
| 3 | TFC_test_B_1wk days | 63 | 2 | 0 | 0 | 1 | 0 | 60 | 2 |
| 4 | TFC_test_A days | 66 | 1 | 0 | 0 | 0 | 0 | 65 | 1 |
| 5 | TFC_test_A_1wk days | 62 | 0 | 0 | 0 | 0 | 0 | 62 | 0 |
| 6 | track day 1 | 44 | 2 | 0 | 0 | 0 | 0 | 42 | 2 |
| 7 | track day 2 | 48 | 1 | 0 | 0 | 0 | 0 | 47 | 1 |
| 8 | track day 3 | 48 | 1 | 0 | 0 | 0 | 0 | 47 | 1 |
| 9 | everything else | 19 | 3 | 0 | 0 | 0 | 0 | 16 | 3 |
|  | **all** | 492 | 71 | 0 | 1 | 2 | 0 | 418 | 27 |

**Now**: G17/2022_01_24-TFC_cond/17_49_36-HC2 (running)

**Next**: G17/2022_01_24-TFC_cond/17_57_20-CNO1; G17/2022_01_24-TFC_cond/18_18_19-CNO2; G17/2022_01_24-TFC_cond/18_52_59-HC3; G17/2022_01_24-TFC_cond/19_13_33-HC4; G18/2022_02_07-TFC_cond/14_03_44-HC1

## Columns

- **status**: `done` (results in the session folder on MINIRAZER), `computed` (finished, being copied back from the work folder), `running`, `failed` (see note), `pending`; `production` = the 130 published, cross-registered sessions (never re-run); `stub` / `partial output` / `set aside by hand` = left out of the batch (open items 10).
- **group**: the batch priority group (whole TFC/test days first, then track days 1-3, then the rest).
- **finished**: when the results reached the session folder. **wall_h**: the run's own time.
- **YrA**: `YrA_recomputed.zarr` present (the corrected residual traces).
- **frames_checked**: the full-frame check -- the replayed movie against the notebook's own, `all N equal` since 2026-09-28; `sample of 50` before. On runs made on the Mac a difference of max 1 grey level is expected (the Mac's cv2 5.0 against the notebook's 4.5; batch runner plan §13).
- **on_MINISCOPE**: `Y (date)` = copied back and verified by the sync; `Y (Mac run)` = run on the Mac, so already there; `partial (YrA not)` = run on the Mac, YrA recomputed later on the Razer; `N` = only on MINIRAZER so far; `Y (source)` = production, which came from MINISCOPE.

## G05 -- 12 of 37 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_08_20-track_day0 | 14_49_26-HC1 | HC1 | 13 | everything else | pending |  |  |  |  |  |  |  |
| 2021_08_20-track_day0 | 15_10_26-LT1 | LT1 | 13 | everything else | done | 2026-09-27 09:30 | 0.59 | 462 | Y | sample of 50 equal | Y (2026-09-28) |  |
| 2021_08_20-track_day0 | 15_44_41-HC2 | HC2 | 6 | everything else | done | 2026-09-28 11:14 | 0.32 | 167 | Y | sample of 50 equal | Y (2026-09-28) |  |
| 2021_08_23-track_day1-tests | 13_52_23-HC1 | HC1 | 6 | track day 1 | done | 2026-09-28 15:48 | 0.25 | 82 | Y | all 5962 equal | Y (2026-09-28) |  |
| 2021_08_23-track_day1-tests | 13_59_41-HC1a | HC1a | 6 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_08_23-track_day1-tests | 14_16_58-LT1 | LT1 | 9 | track day 1 | done | 2026-09-27 09:49 | 0.32 | 180 | Y | sample of 50 differ: max 1 at 1163 px | Y (2026-09-28) |  |
| 2021_08_23-track_day1-tests | 14_23_55-LT1a | LT1a | 9 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_08_23-track_day1-tests | 14_42_54-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_08_25-track_day2 | 14_25_21-HC1 | HC1 | 8 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_08_25-track_day2 | 14_41_34-LT1 | LT1 | 19 | track day 2 | done | 2026-09-27 10:24 | 0.59 | 202 | Y | sample of 50 equal | Y (2026-09-28) |  |
| 2021_08_25-track_day2 | 15_07_25-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_08_27-track_day3 | 15_44_42-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_08_27-track_day3 | 16_08_43-LT1 | LT1 | 13 | track day 3 | done | 2026-09-28 17:43 | 0.64 | 479 | Y | all 12766 equal | Y (2026-09-28) |  |
| 2021_08_27-track_day3 | 16_43_50-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_08_30-TFC_cond | 16_29_01-HC1 | HC1 | 7 | TFC_cond days | done | 2026-09-28 18:03 | 0.27 | 170 | Y | all 6145 equal | Y (2026-09-28) |  |
| 2021_08_30-TFC_cond | 16_47_02-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_08_30-TFC_cond | 17_12_02-CNO1 | CNO1 | 6 | TFC_cond days | done | 2026-09-28 16:02 | 0.25 | 135 | Y | all 5990 equal | Y (2026-09-28) |  |
| 2021_08_30-TFC_cond | 17_30_19-CNO2 | CNO2 | 7 | TFC_cond days | done | 2026-09-28 18:17 | 0.25 | 67 | Y | all 6238 equal | Y (2026-09-28) |  |
| 2021_08_30-TFC_cond | 17_44_51-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_08_30-TFC_cond | 18_07_14-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-28 18:37 | 0.30 | 201 | Y | all 6443 equal | Y (2026-09-28) |  |
| 2021_08_30-TFC_cond | 18_22_57-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) | production; also the Razer production check's re-run in minian/ (razer plan Phase 3.2) |
| 2021_08_30-TFC_cond | 18_59_38-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-28 18:51 | 0.25 | 103 | Y | all 6219 equal | Y (2026-09-28) |  |
| 2021_09_01-TFC_test_B | 15_11_43-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_09_01-TFC_test_B | 15_20_05-iso | iso | 5 | TFC_test_B days | stub (excluded) |  |  |  |  |  |  |  |
| 2021_09_01-TFC_test_B | 15_37_05-LT1 | LT1 | 13 | TFC_test_B days | production |  |  |  | N |  | Y (source) |  |
| 2021_09_01-TFC_test_B | 16_00_30-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_09_01-TFC_test_B | 16_20_07-TFC_test_B | TFC_test_B | 25 | TFC_test_B days | production |  |  |  | N |  | Y (source) |  |
| 2021_09_01-TFC_test_B | 16_54_22-HC3 | HC3 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_09_03-TFC_test_A | 15_09_20-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_09_03-TFC_test_A | 15_29_57-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_09_03-TFC_test_A | 15_58_41-HC2 | HC2 | 16 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_09_03-TFC_test_A | 16_34_42-TFC_test_A | TFC_test_A | 8 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_09_03-TFC_test_A | 16_55_42-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_09_06-TFC_test_B_1wk | 15_10_04-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_06-TFC_test_B_1wk | 15_58_07-LT1 | LT1 | 13 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_06-TFC_test_B_1wk | 16_26_39-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_06-TFC_test_B_1wk | 16_51_41-TFC_test_B_1wk | TFC_test_B_1wk | 19 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_09_06-TFC_test_B_1wk | 17_25_35-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_07-TFC_test_B_1wk-redux | 15_12_23-HC1 | HC1 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2021_09_07-TFC_test_B_1wk-redux | 15_30_14-LT1 | LT1 | 12 | everything else | pending |  |  |  |  |  |  |  |
| 2021_09_07-TFC_test_B_1wk-redux | 15_54_52-HC2 | HC2 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2021_09_07-TFC_test_B_1wk-redux | 16_22_40-TFC_test_B_1wk | TFC_test_B_1wk | 18 | everything else | done | 2026-09-28 17:00 | 0.83 | 488 | Y | all 17878 equal | Y (2026-09-28) | test day (bedding change?); processed, not for analysis by default (VS, open items) |
| 2021_09_07-TFC_test_B_1wk-redux | 16_55_30-HC3 | HC3 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2021_09_09-TFC_test_A_1wk | 15_41_49-HC1 | HC1 | 6 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_09-TFC_test_A_1wk | 16_11_07-LT1 | LT1 | 13 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_09-TFC_test_A_1wk | 16_36_18-HC2 | HC2 | 6 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_09_09-TFC_test_A_1wk | 17_32_00-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_09_09-TFC_test_A_1wk | 17_55_33-HC3 | HC3 | 10 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G06 -- 5 of 29 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_10_07-check1 | 15_00_46 |  | 5 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_11-track_day1 | 11_41_23-HC1 | HC1 | 6 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_10_11-track_day1 | 11_58_00-LT1 | LT1 | 16 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_10_11-track_day1 | 12_22_01-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_10_13-track_day2 | 11_14_18-HC1 | HC1 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_10_13-track_day2 | 11_36_34-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_10_13-track_day2 | 11_59_01-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_10_15-track_day3 | 14_55_21-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_10_15-track_day3 | 15_28_28-LT1 | LT1 | 13 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_10_15-track_day3 | 15_50_04-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_10_18-TFC_cond | 09_52_24-HC1 | HC1 | 6 | TFC_cond days | done | 2026-09-26 14:05 | 0.25 | 334 | Y | sample of 50 differ: max 1 at 5 px | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 10_14_41-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_18-TFC_cond | 10_34_09-CNO1 | CNO1 | 6 | TFC_cond days | done | 2026-09-28 19:09 | 0.28 | 263 | Y | all 5964 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 10_49_52-CNO2 | CNO2 | 6 | TFC_cond days | done | 2026-09-28 19:25 | 0.27 | 217 | Y | all 5965 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 11_05_51-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_18-TFC_cond | 11_26_52-HC2 | HC2 | 6 | TFC_cond days | done | 2026-09-28 19:39 | 0.24 | 162 | Y | all 5971 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 11_42_00-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_18-TFC_cond | 12_17_29-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-28 19:55 | 0.25 | 153 | Y | all 6372 equal | Y (2026-09-28) |  |
| 2021_10_20-TFC_test_B | 12_27_23-HC1 | HC1 | 6 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_20-TFC_test_B | 12_45_49-LT1 | LT1 | 15 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_20-TFC_test_B | 13_08_37-HC2 | HC2 | 6 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_20-TFC_test_B | 13_32_28-TFC_test_B | TFC_test_B | 19 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_20-TFC_test_B | 14_10_06-HC3 | HC3 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 11_57_08-HC1 | HC1 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 12_18_46-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 12_41_02-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 12_58_00-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_22-TFC_test_A | 13_14_21-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 11_11_01-HC1 | HC1 | 6 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 11_28_09-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 11_51_27-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 12_23_30-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_25-TFC_test_B_1wk | 12_52_19-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 13_42_40-HC1 | HC1 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 14_01_05-LT1 | LT1 | 13 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 14_23_16-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 14_45_19-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_27-TFC_test_A_1wk | 15_06_23-HC3 | HC3 | 8 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G07 -- 5 of 22 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_10_11-track_day1 | 13_29_38 |  | 7 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_11-track_day1 | 13_46_01 |  | 14 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_11-track_day1 | 14_09_38 |  | 7 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_13-track_day2 | 13_02_40 |  | 7 | track day 2 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_13-track_day2 | 13_19_46 |  | 17 | track day 2 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_13-track_day2 | 13_44_59 |  | 7 | track day 2 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_15-track_day3 | 16_26_00 |  | 12 | track day 3 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_15-track_day3 | 16_52_58 |  | 25 | track day 3 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_15-track_day3 | 17_43_54 |  | 10 | track day 3 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_18-TFC_cond | 12_53_17-HC1 | HC1 | 6 | TFC_cond days | done | 2026-09-28 20:17 | 0.34 | 422 | Y | all 5979 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 13_13_24-LT1 | LT1 | 15 | TFC_cond days | done | 2026-09-28 21:05 | 0.82 | 647 | Y | all 14533 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 13_25_42-LT1b | LT1b | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_18-TFC_cond | 14_03_02-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-28 21:33 | 0.33 | 421 | Y | all 6271 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 14_19_09-CNO2 | CNO2 | 7 | TFC_cond days | done | 2026-09-28 21:49 | 0.30 | 162 | Y | all 6828 equal | Y (2026-09-28) |  |
| 2021_10_18-TFC_cond | 14_35_40-LT2 | LT2 | 16 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_18-TFC_cond | 15_14_32-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_18-TFC_cond | 16_06_52-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-28 22:09 | 0.31 | 362 | Y | all 6625 equal | Y (2026-09-28) |  |
| 2021_10_20-TFC_test_B | 14_39_00-HC1 | HC1 | 6 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_20-TFC_test_B | 15_03_23-LT1 | LT1 | 15 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_20-TFC_test_B | 15_27_47-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_20-TFC_test_B | 15_49_59-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | failed |  | 0.00 |  |  |  |  | unfinished recordings (no frame count in headers; 0.47 GB for 18 files); VS to decide |
| 2021_10_20-TFC_test_B | 16_25_43-HC3 | HC3 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 14_19_25-HC1 | HC1 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 14_41_53-LT1 | LT1 | 15 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 15_05_33-HC2 | HC2 | 6 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_22-TFC_test_A | 15_26_13-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_22-TFC_test_A | 15_46_55-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 14_08_09-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 14_31_21-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 14_55_20-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_25-TFC_test_B_1wk | 15_19_51-TFC_test_B_1wk | TFC_test_B_1wk | 19 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_25-TFC_test_B_1wk | 15_50_38-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 15_48_22-HC1 | HC1 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 16_06_09-LT1 | LT1 | 14 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 16_30_12-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_10_27-TFC_test_A_1wk | 16_49_38-TFC_test_A_1wk | TFC_test_A_1wk | 8 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_10_27-TFC_test_A_1wk | 17_09_53-HC3 | HC3 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G08 -- 4 of 19 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_10_26-track_day1 | 12_14_54 |  | 8 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_26-track_day1 | 12_33_46 |  | 17 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_26-track_day1 | 12_59_02 |  | 7 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_28-track_day2 | 10_54_33 |  | 7 | track day 2 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_28-track_day2 | 11_17_27 |  | 14 | track day 2 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_10_28-track_day2 | 11_42_43 |  | 7 | track day 2 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_01-track_day3 | 11_41_06 |  | 7 | track day 3 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_01-track_day3 | 12_18_19 |  | 13 | track day 3 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_01-track_day3 | 12_40_16 |  | 6 | track day 3 | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_03-track_day4 | 13_52_36 |  | 7 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_03-track_day4 | 14_16_11 |  | 13 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_03-track_day4 | 14_41_06 |  | 7 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_05-track_day5 | 12_52_19 |  | 7 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_05-track_day5 | 13_13_59 |  | 15 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_05-track_day5 | 13_43_26 |  | 7 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_11_08-TFC_cond | 13_05_47-HC1 | HC1 | 6 | TFC_cond days | done | 2026-09-28 22:23 | 0.26 | 179 | Y | all 5953 equal | Y (2026-09-28) |  |
| 2021_11_08-TFC_cond | 13_22_42-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_08-TFC_cond | 14_07_31-CNO1 | CNO1 | 6 | TFC_cond days | done | 2026-09-28 22:43 | 0.31 | 197 | Y | all 5947 equal | N |  |
| 2021_11_08-TFC_cond | 14_23_22-CNO2 | CNO2 | 7 | TFC_cond days | done | 2026-09-28 23:04 | 0.35 | 240 | Y | all 6453 equal | N |  |
| 2021_11_08-TFC_cond | 14_43_03-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_08-TFC_cond | 15_13_23-TFC_cond | TFC_cond | 27 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_08-TFC_cond | 15_50_16-HC3 | HC3 | 8 | TFC_cond days | done | 2026-09-28 23:22 | 0.30 | 158 | Y | all 7777 equal | N |  |
| 2021_11_10-TFC_test_B | 13_27_20-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_10-TFC_test_B | 13_54_11-LT1 | LT1 | 15 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_10-TFC_test_B | 14_18_51-HC2 | HC2 | 6 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_10-TFC_test_B | 14_45_42-TFC_test_B | TFC_test_B | 19 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_10-TFC_test_B | 15_23_24-HC3 | HC3 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 13_05_51-HC1 | HC1 | 6 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 13_35_08-LT1 | LT1 | 16 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 14_00_29-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 14_38_53-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_12-TFC_test_A | 15_04_25-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 13_40_03-HC1 | HC1 | 6 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 14_08_34-LT1 | LT1 | 16 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 14_32_30-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 14_58_20-TFC_test_B_1wk | TFC_test_B_1wk | 19 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_15-TFC_test_B_1wk | 15_28_34-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 14_15_05-HC1 | HC1 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 14_39_00-LT1 | LT1 | 16 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 15_04_04-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 15_24_18-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_17-TFC_test_A_1wk | 16_02_26-HC3 | HC3 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G09 -- 4 of 34 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_10_26-track_day1 | 13_45_47-HC1 | HC1 | 9 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_10_26-track_day1 | 14_09_57-LT1 | LT1 | 13 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_10_26-track_day1 | 14_30_24-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_10_28-track_day2 | 12_13_39-HC1 | HC1 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_10_28-track_day2 | 12_28_07-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_10_28-track_day2 | 12_50_55-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_01-track_day3 | 13_30_46-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_01-track_day3 | 13_49_47-LT1 | LT1 | 14 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_01-track_day3 | 14_14_12-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_03-track_day4 | 15_18_23-HC1 | HC1 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2021_11_03-track_day4 | 15_37_14-LT1 | LT1 | 15 | everything else | pending |  |  |  |  |  |  |  |
| 2021_11_03-track_day4 | 16_05_26-HC2 | HC2 | 8 | everything else | pending |  |  |  |  |  |  |  |
| 2021_11_05-track_day5 | 14_11_23-HC1 | HC1 | 6 | everything else | pending |  |  |  |  |  |  |  |
| 2021_11_05-track_day5 | 14_28_11-LT1 | LT1 | 13 | everything else | pending |  |  |  |  |  |  |  |
| 2021_11_05-track_day5 | 14_50_19-HC2 | HC2 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2021_11_08-TFC_cond | 16_33_10-HC1 | HC1 | 7 | TFC_cond days | done | 2026-09-28 23:40 | 0.30 | 224 | Y | all 6644 equal | N |  |
| 2021_11_08-TFC_cond | 16_53_07-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_08-TFC_cond | 17_24_22-CNO1 | CNO1 | 6 | TFC_cond days | done | 2026-09-28 23:54 | 0.24 | 70 | Y | all 5966 equal | N |  |
| 2021_11_08-TFC_cond | 17_49_48-CNO2 | CNO2 | 7 | TFC_cond days | done | 2026-09-29 00:10 | 0.26 | 77 | Y | all 6667 equal | N |  |
| 2021_11_08-TFC_cond | 18_08_37-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_08-TFC_cond | 18_29_51-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-29 00:25 | 0.25 | 46 | Y | all 6313 equal | N |  |
| 2021_11_08-TFC_cond | 18_54_05-TFC_cond | TFC_cond | 19 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_08-TFC_cond | 19_28_33-HC3 | HC3 | 7 | TFC_cond days | partial output (excluded) |  |  |  |  |  |  | partial old output, all-NaN motion; VS to decide (open items 10) |
| 2021_11_10-TFC_test_B | 15_58_30-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_10-TFC_test_B | 16_21_56-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_10-TFC_test_B | 16_45_04-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_10-TFC_test_B | 17_12_18-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_10-TFC_test_B | 17_44_41-HC3 | HC3 | 9 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 15_33_00-HC1 | HC1 | 6 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 15_51_11-LT1 | LT1 | 14 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 16_13_16-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_12-TFC_test_A | 16_33_31-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_12-TFC_test_A | 16_54_57-HC3 | HC3 | 6 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 16_11_16-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 16_29_03-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 16_51_09-HC2 | HC2 | 15 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_15-TFC_test_B_1wk | 17_15_18-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_15-TFC_test_B_1wk | 17_42_46-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 16_33_51-HC1 | HC1 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 16_55_57-LT1 | LT1 | 15 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 17_19_09-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_17-TFC_test_A_1wk | 17_47_11-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_17-TFC_test_A_1wk | 18_03_11-HC3 | HC3 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G10 -- 5 of 28 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_11_16-track_day1 | 14_42_35-HC1 | HC1 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_11_16-track_day1 | 15_00_51-LT1 | LT1 | 15 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_11_16-track_day1 | 15_23_30-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_11_18-track_day2 | 13_41_54-HC1 | HC1 | 8 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_18-track_day2 | 14_03_01-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_18-track_day2 | 14_25_50-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 14_00_19-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 14_05_55-HC1_high_led | HC1_high_led | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 14_22_22-LT1 | LT1 | 15 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 14_45_34-HC2 | HC2 | 6 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_23-TFC_cond | 14_34_39-HC1 | HC1 | 6 | TFC_cond days | done | 2026-09-29 00:53 | 0.38 | 737 | Y | all 5951 equal | N |  |
| 2021_11_23-TFC_cond | 14_51_37-LT1 | LT1 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_23-TFC_cond | 15_22_29-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-29 01:13 | 0.36 | 495 | Y | all 6475 equal | N |  |
| 2021_11_23-TFC_cond | 15_38_21-CNO2 | CNO2 | 6 | TFC_cond days | done | 2026-09-29 01:30 | 0.30 | 393 | Y | all 5951 equal | N |  |
| 2021_11_23-TFC_cond | 15_53_29-LT2 | LT2 | 18 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_23-TFC_cond | 16_18_55-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-29 01:55 | 0.38 | 682 | Y | all 6251 equal | N |  |
| 2021_11_23-TFC_cond | 16_32_14-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) | production; also the Mac gate re-run in minian/ (batch runner plan §13) |
| 2021_11_23-TFC_cond | 17_07_00-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-29 02:14 | 0.33 | 516 | Y | all 6106 equal | N |  |
| 2021_11_25-TFC_test_B | 14_18_38-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_25-TFC_test_B | 14_39_23-LT1 | LT1 | 17 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_25-TFC_test_B | 15_16_15-HC2 | HC2 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_25-TFC_test_B | 15_28_34-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_25-TFC_test_B | 15_57_42-HC3 | HC3 | 9 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_27-TFC_test_A | 14_52_16-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_27-TFC_test_A | 15_08_43-LT1 | LT1 | 15 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_27-TFC_test_A | 15_34_25-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_27-TFC_test_A | 15_52_21-HC2 | HC2 | 6 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_30-TFC_test_B_1wk | 15_08_20-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_30-TFC_test_B_1wk | 15_36_58-LT1 | LT1 | 13 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_30-TFC_test_B_1wk | 16_07_16-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_30-TFC_test_B_1wk | 16_35_34-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_12_02-TFC_test_A_1wk | 13_17_21-HC1 | HC1 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_12_02-TFC_test_A_1wk | 13_34_50-LT1 | LT1 | 16 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_12_02-TFC_test_A_1wk | 13_59_52-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_12_02-TFC_test_A_1wk | 14_17_43-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G11 -- 7 of 26 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_11_16-track_day1 | 16_25_49-HC1 | HC1 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_11_16-track_day1 | 16_57_16-LT1 | LT1 | 15 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_11_16-track_day1 | 17_43_41-HC2 | HC2 | 9 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_11_18-track_day2 | 14_53_56-HC1 | HC1 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_18-track_day2 | 15_20_53-LT1 | LT1 | 15 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_18-track_day2 | 15_51_46-HC2 | HC2 | 8 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 15_13_38-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 15_30_53-LT1 | LT1 | 17 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_20-track_day3 | 15_55_01-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_11_23-TFC_cond | 17_33_23-HC1 | HC1 | 6 | TFC_cond days | done | 2026-09-29 02:33 | 0.33 | 515 | Y | all 5965 equal | N |  |
| 2021_11_23-TFC_cond | 17_51_25-LT1 | LT1 | 18 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_23-TFC_cond | 18_25_41-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-29 02:51 | 0.30 | 380 | Y | all 6116 equal | N |  |
| 2021_11_23-TFC_cond | 18_41_08-CNO2 | CNO2 | 7 | TFC_cond days | done | 2026-09-29 03:10 | 0.32 | 368 | Y | all 6643 equal | N |  |
| 2021_11_23-TFC_cond | 18_55_26-LT2 | LT2 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_23-TFC_cond | 19_15_48-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-29 03:29 | 0.30 | 379 | Y | all 6128 equal | N |  |
| 2021_11_23-TFC_cond | 19_24_52-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_23-TFC_cond | 19_59_33-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-29 03:49 | 0.32 | 453 | Y | all 6122 equal | N |  |
| 2021_11_25-TFC_test_B | 16_29_57-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_25-TFC_test_B | 16_47_19-LT1 | LT1 | 19 | TFC_test_B days | done | 2026-09-28 14:03 | 2.77 | 748 | Y | sample of 50 equal | Y (2026-09-28) |  |
| 2021_11_25-TFC_test_B | 17_17_33-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_25-TFC_test_B | 17_45_04-HC2 | HC2 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2021_11_27-TFC_test_A | 16_16_47-HC1 | HC1 | 6 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_27-TFC_test_A | 16_32_25-LT1 | LT1 | 16 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_27-TFC_test_A | 17_00_04-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_27-TFC_test_A | 17_11_20-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2021_11_30-TFC_test_B_1wk | 17_08_44-HC1 | HC1 | 6 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_11_30-TFC_test_B_1wk | 17_25_43-LT1 | LT1 | 18 | TFC_test_B_1wk days | done | 2026-09-28 14:06 | 2.82 | 911 | Y | sample of 50 equal | Y (2026-09-28) |  |
| 2021_11_30-TFC_test_B_1wk | 17_57_20-TFC_test_B_1wk | TFC_test_B_1wk | 19 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_11_30-TFC_test_B_1wk | 18_30_29-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2021_12_02-TFC_test_A_1wk | 14_43_48-HC1 | HC1 | 6 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_12_02-TFC_test_A_1wk | 14_57_08-LT1 | LT1 | 13 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2021_12_02-TFC_test_A_1wk | 15_20_57-TFC_test_A_1wk | TFC_test_A_1wk | 8 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2021_12_02-TFC_test_A_1wk | 15_32_02-HC2 | HC2 | 6 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G12 -- 5 of 32 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_12_13 | 14_43_31 |  | 6 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 13_14_25-HC1 | HC1 | 6 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 13_27_10-LT1 | LT1 | 13 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 13_38_18-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 13_43_30-HC3 | HC3 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 14_04_00-HC1 | HC1 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 14_15_40-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 14_27_41-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 14_33_15-HC3 | HC3 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 15_04_46-HC1 | HC1 | 6 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 15_18_18-LT1 | LT1 | 13 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 15_29_21-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 15_34_36-HC3 | HC3 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_03-TFC_cond | 13_59_56-HC1 | HC1 | 6 | TFC_cond days | done | 2026-09-29 04:02 | 0.26 | 129 | Y | all 5994 equal | N |  |
| 2022_01_03-TFC_cond | 14_11_51-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_03-TFC_cond | 14_23_02-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-29 04:23 | 0.32 | 283 | Y | all 6660 equal | N |  |
| 2022_01_03-TFC_cond | 14_32_19-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-29 04:42 | 0.32 | 258 | Y | all 6269 equal | N |  |
| 2022_01_03-TFC_cond | 14_52_29-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_03-TFC_cond | 15_03_39-HC3 | HC3 | 6 | TFC_cond days | done | 2026-09-29 05:00 | 0.29 | 266 | Y | all 5990 equal | N |  |
| 2022_01_03-TFC_cond | 15_12_38-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_03-TFC_cond | 15_37_11-HC4 | HC4 | 7 | TFC_cond days | done | 2026-09-29 05:16 | 0.28 | 237 | Y | all 6062 equal | N |  |
| 2022_01_05-TFC_test_B | 13_57_17-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_05-TFC_test_B | 14_03_29-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_05-TFC_test_B | 14_15_33-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_05-TFC_test_B | 14_28_46-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_05-TFC_test_B | 14_46_41-HC3 | HC3 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 15_37_50-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 15_43_53-LT1 | LT1 | 16 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 15_57_32-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 16_08_09-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_07-TFC_test_A | 16_16_42-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 15_01_34-HC1 | HC1 | 6 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 15_06_38-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 15_18_57-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 15_31_10-TFC_test_B_1wk | TFC_test_B_1wk | 19 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_10-TFC_test_B_1wk | 15_49_09-HC3 | HC3 | 6 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 13_36_37-HC1 | HC1 | 8 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 13_43_14-LT1 | LT1 | 14 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 13_55_18-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 14_03_29-TFC_test_A_1wk | TFC_test_A_1wk | 8 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_12-TFC_test_A_1wk | 14_12_53-HC3 | HC3 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G13 -- 7 of 33 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_12_13 | 15_53_03 |  | 7 | everything else | stub (excluded) |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 14_19_13-HC1 | HC1 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 14_30_13-LT1 | LT1 | 13 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 14_41_10-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_27-track_day1 | 14_50_49-HC3 | HC3 | 6 | track day 1 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 15_01_17-HC1 | HC1 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 15_12_32-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 15_24_29-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_29-track_day2 | 15_29_54-HC3 | HC3 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 16_22_07-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 16_39_19-LT1 | LT1 | 13 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 16_50_18-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2021_12_31-track_day3 | 16_55_33-HC3 | HC3 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_03-TFC_cond | 16_05_10-HC1 | HC1 | 7 | TFC_cond days | done | 2026-09-29 05:32 | 0.27 | 105 | Y | all 6132 equal | N |  |
| 2022_01_03-TFC_cond | 16_10_53-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_03-TFC_cond | 16_21_45-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-29 05:47 | 0.26 | 100 | Y | all 6083 equal | N |  |
| 2022_01_03-TFC_cond | 16_28_35-CNO1 | CNO1 | 9 | TFC_cond days | done | 2026-09-29 06:09 | 0.36 | 85 | Y | all 8434 equal | N |  |
| 2022_01_03-TFC_cond | 16_48_37-CNO2 | CNO2 | 7 | TFC_cond days | done | 2026-09-29 06:25 | 0.27 | 65 | Y | all 6181 equal | N |  |
| 2022_01_03-TFC_cond | 16_53_52-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_03-TFC_cond | 17_04_37-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-29 06:40 | 0.26 | 50 | Y | all 6104 equal | N |  |
| 2022_01_03-TFC_cond | 17_13_06-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_03-TFC_cond | 17_37_35-HC4 | HC4 | 8 | TFC_cond days | done | 2026-09-29 06:58 | 0.29 | 36 | Y | all 7048 equal | N |  |
| 2022_01_05-TFC_test_B | 15_24_03-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_05-TFC_test_B | 15_29_44-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_05-TFC_test_B | 15_41_23-HC2 | HC2 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_05-TFC_test_B | 15_53_23-TFC_test_B | TFC_test_B | 19 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_05-TFC_test_B | 16_11_43-HC3 | HC3 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 16_43_27-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 16_49_21-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 17_00_54-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_07-TFC_test_A | 17_09_27-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_07-TFC_test_A | 17_17_44-HC3 | HC3 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 16_18_02-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 16_23_48-LT1 | LT1 | 13 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 16_35_20-HC2 | HC2 | 4 | TFC_test_B_1wk days | done | 2026-09-26 13:32 | 0.20 | 88 | Y | - | Y (2026-09-28) |  |
| 2022_01_10-TFC_test_B_1wk | 16_38_56-HC2b | HC2b | 7 | TFC_test_B_1wk days | stub (excluded) |  |  |  |  |  |  |  |
| 2022_01_10-TFC_test_B_1wk | 16_50_10-TFC_test_B_1wk | TFC_test_B_1wk | 20 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_10-TFC_test_B_1wk | 17_09_11-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 14_43_45-HC1 | HC1 | 6 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 14_48_49-LT1 | LT1 | 13 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 15_00_05-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_12-TFC_test_A_1wk | 15_07_47-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_12-TFC_test_A_1wk | 15_16_01-HC3 | HC3 | 6 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G14 -- 6 of 31 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_01_04-track_day1 | 14_28_03-HC1 | HC1 | 6 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_04-track_day1 | 14_33_47-LT1 | LT1 | 14 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_04-track_day1 | 14_45_35-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 15_42_26-HC1 | HC1 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 15_49_30-LT1 | LT1 | 12 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 16_00_14-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 16_05_37-HC3 | HC3 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_08-track_day3 | 12_48_36-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_08-track_day3 | 12_54_19-LT1 | LT1 | 13 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_08-track_day3 | 13_06_00-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_11-TFC_cond | 14_47_56-HC1 | HC1 | 7 | TFC_cond days | done | 2026-09-29 07:19 | 0.31 | 341 | Y | all 6209 equal | N |  |
| 2022_01_11-TFC_cond | 14_53_32-LT1 | LT1 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_11-TFC_cond | 15_05_21-HC2 | HC2 | 9 | TFC_cond days | done | 2026-09-29 07:42 | 0.38 | 325 | Y | all 8157 equal | N |  |
| 2022_01_11-TFC_cond | 15_33_45-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-29 07:58 | 0.29 | 231 | Y | all 6647 equal | N |  |
| 2022_01_11-TFC_cond | 15_39_23-LT2 | LT2 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_11-TFC_cond | 15_51_35-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-29 08:17 | 0.30 | 310 | Y | all 6259 equal | N |  |
| 2022_01_11-TFC_cond | 16_00_53-TFC_cond | TFC_cond | 29 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_11-TFC_cond | 16_28_19-HC4 | HC4 | 7 | TFC_cond days | done | 2026-09-29 08:34 | 0.29 | 244 | Y | all 6239 equal | N |  |
| 2022_01_13-TFC_test_B | 14_32_17-HC1 | HC1 | 9 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_13-TFC_test_B | 14_39_28-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_13-TFC_test_B | 14_51_39-HC2 | HC2 | 10 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_13-TFC_test_B | 15_05_26-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_13-TFC_test_B | 15_24_05-HC3 | HC3 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test-A | 17_03_52-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test-A | 17_14_14-LT1 | LT1 | 18 | TFC_test_A days | done | 2026-09-28 15:29 | 2.32 | 403 | Y | all 17683 equal | Y (2026-09-28) |  |
| 2022_01_15-TFC_test-A | 17_29_31-HC2 | HC2 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test-A | 17_39_32-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_15-TFC_test-A | 17_47_59-HC3 | HC3 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_18-TFC_test_B_1wk | 14_21_15-HC1 | HC1 | 9 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_18-TFC_test_B_1wk | 14_28_04-LT1 | LT1 | 15 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_18-TFC_test_B_1wk | 14_41_02-HC2 | HC2 | 8 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_18-TFC_test_B_1wk | 14_52_35-TFC_test_B_1wk-borked | TFC_test_B_1wk-borked | 18 | TFC_test_B_1wk days | failed |  | 0.00 |  |  |  |  | unfinished recordings (no frame count in headers); VS to decide |
| 2022_01_18-TFC_test_B_1wk | 15_10_50-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_18-TFC_test_B_1wk | 15_28_43-HC3 | HC3 | 9 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_21-TFC_test_A_1wk | 15_41_59-HC1 | HC1 | 8 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_21-TFC_test_A_1wk | 15_48_10-LT1 | LT1 | 16 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_21-TFC_test_A_1wk | 16_02_05-HC2 | HC2 | 10 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_21-TFC_test_A_1wk | 16_13_09-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_21-TFC_test_A_1wk | 16_21_48-HC3 | HC3 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G15 -- 5 of 21 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_01_04-track_day1 | 16_21_46-HC1 | HC1 | 6 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_04-track_day1 | 16_27_01-LT1 | LT1 | 13 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_04-track_day1 | 16_38_24-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 16_42_20-HC1 | HC1 | 10 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 16_50_04-LT1 | LT1 | 13 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_06-track_day2 | 17_00_45-HC2 | HC2 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_08-track_day3 | 13_34_11-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_08-track_day3 | 13_39_54-LT1 | LT1 | 14 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_08-track_day3 | 13_51_27-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_11-TFC_cond | 16_57_42-HC1 | HC1 | 10 | TFC_cond days | done | 2026-09-29 08:53 | 0.33 | 92 | Y | all 9104 equal | N |  |
| 2022_01_11-TFC_cond | 17_05_25-LT1 | LT1 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_11-TFC_cond | 17_16_49-HC2 | HC2 | 9 | TFC_cond days | done | 2026-09-29 09:15 | 0.35 | 163 | Y | all 8178 equal | N |  |
| 2022_01_11-TFC_cond | 17_45_31-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-29 09:31 | 0.28 | 81 | Y | all 6788 equal | N |  |
| 2022_01_11-TFC_cond | 17_51_17-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_11-TFC_cond | 18_02_26-HC3 | HC3 | 7 | TFC_cond days | done | 2026-09-29 09:49 | 0.29 | 194 | Y | all 6137 equal | N |  |
| 2022_01_11-TFC_cond | 18_10_56-TFC_cond | TFC_cond | 30 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_11-TFC_cond | 18_39_20-HC4 | HC4 | 9 | TFC_cond days | done | 2026-09-29 10:10 | 0.34 | 142 | Y | all 8173 equal | N |  |
| 2022_01_13-TFC_test_B | 15_51_23-HC1 | HC1 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_13-TFC_test_B | 15_57_25-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_13-TFC_test_B | 16_09_14-HC2 | HC2 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_13-TFC_test_B | 16_19_13-TFC_test_B | TFC_test_B | 20 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_13-TFC_test_B | 16_38_14-HC3 | HC3 | 11 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test_A | 18_16_31-HC1 | HC1 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test_A | 18_22_53-LT1 | LT1 | 14 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test_A | 18_35_23-HC2 | HC2 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_15-TFC_test_A | 18_45_26-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_15-TFC_test_A | 18_53_47-HC3 | HC3 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |

## G16 -- 5 of 29 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_01_17-track_day1 | 17_47_45-HC1 | HC1 | 8 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_17-track_day1 | 17_54_47-LT1 | LT1 | 15 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_17-track_day1 | 18_07_17-HC2 | HC2 | 8 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_19-track_day2 | 14_17_00-HC1 | HC1 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_19-track_day2 | 14_22_35-LT1 | LT1 | 16 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_19-track_day2 | 14_36_30-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_21-track_day3 | 13_56_35-HC1 | HC1 | 8 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_21-track_day3 | 14_02_33-LT1 | LT1 | 17 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_21-track_day3 | 14_16_49-HC2 | HC2 | 9 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_24-TFC_cond | 14_53_32-HC1 | HC1 | 7 | TFC_cond days | done | 2026-09-29 10:35 | 0.37 | 509 | Y | all 6743 equal | N |  |
| 2022_01_24-TFC_cond | 14_59_15-LT1 | LT1 | 15 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_24-TFC_cond | 15_12_16-HC2 | HC2 | 7 | TFC_cond days | done | 2026-09-29 10:56 | 0.36 | 469 | Y | all 6668 equal | N |  |
| 2022_01_24-TFC_cond | 15_41_33-CNO1 | CNO1 | 7 | TFC_cond days | done | 2026-09-29 11:14 | 0.31 | 356 | Y | all 6040 equal | N |  |
| 2022_01_24-TFC_cond | 15_47_20-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_24-TFC_cond | 15_58_23-HC3 | HC3 | 6 | TFC_cond days | done | 2026-09-29 11:36 | 0.33 | 461 | Y | all 5968 equal | N |  |
| 2022_01_24-TFC_cond | 16_06_44-TFC_cond | TFC_cond | 28 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_24-TFC_cond | 16_33_17-HC4 | HC4 | 9 | TFC_cond days | done | 2026-09-29 12:05 | 0.47 | 529 | Y | all 8862 equal | N |  |
| 2022_01_26-TFC_test_B | 14_17_03-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_26-TFC_test_B | 14_23_45-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_26-TFC_test_B | 14_36_08-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_26-TFC_test_B | 14_47_09-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_26-TFC_test_B | 15_04_50-HC3 | HC3 | 10 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 13_32_58-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 13_38_27-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 13_51_04-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 14_01_26-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_28-TFC_test_A | 14_09_25-HC3 | HC3 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 13_34_47-HC1 | HC1 | 10 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 13_43_03-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 13_54_52-HC2 | HC2 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 14_08_02-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_31-TFC_test_B_1wk | 14_28_46-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 14_16_41-HC1 | HC1 | 9 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 14_24_38-LT1 | LT1 | 14 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 14_36_32-HC2 | HC2 | 10 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 14_49_27-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_02-TFC_test_A_1wk | 14_58_24-HC3 | HC3 | 10 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G17 -- 1 of 30 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_01_17-track_day1 | 18_39_13-HC1 | HC1 | 9 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_17-track_day1 | 18_46_21-LT1 | LT1 | 16 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_17-track_day1 | 19_00_34-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_01_19-track_day2 | 15_12_27-HC1 | HC1 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_19-track_day2 | 15_18_08-LT1 | LT1 | 15 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_19-track_day2 | 15_31_17-HC2 | HC2 | 12 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_01_21-track_day3 | 14_48_20-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_21-track_day3 | 14_53_55-LT1 | LT1 | 14 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_21-track_day3 | 15_06_06-HC2 | HC2 | 8 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_01_24-TFC_cond | 17_31_07-HC1 | HC1 | 9 | TFC_cond days | done | 2026-09-29 12:23 | 0.35 | 269 | Y | all 8027 equal | N |  |
| 2022_01_24-TFC_cond | 17_38_16-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_24-TFC_cond | 17_49_36-HC2 | HC2 | 7 | TFC_cond days | running |  |  |  |  |  |  |  |
| 2022_01_24-TFC_cond | 17_57_20-CNO1 | CNO1 | 6 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_01_24-TFC_cond | 18_18_19-CNO2 | CNO2 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_01_24-TFC_cond | 18_26_30-TFC_cond | TFC_cond | 27 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_24-TFC_cond | 18_52_59-HC3 | HC3 | 8 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_01_24-TFC_cond | 19_02_10-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_24-TFC_cond | 19_13_33-HC4 | HC4 | 6 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_01_26-TFC_test_B | 15_36_39-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_26-TFC_test_B | 15_54_39-LT1 | LT1 | 11 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_26-TFC_test_B | 16_04_58-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_26-TFC_test_B | 16_14_47-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_26-TFC_test_B | 16_32_17-HC3 | HC3 | 14 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 14_37_42-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 14_42_59-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 14_54_20-HC2 | HC2 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_28-TFC_test_A | 15_02_16-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_28-TFC_test_A | 15_10_20-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 15_31_56-HC1 | HC1 | 8 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 15_38_40-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 15_51_55-HC2 | HC2 | 12 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_01_31-TFC_test_B_1wk | 16_06_22-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_01_31-TFC_test_B_1wk | 16_24_36-HC3 | HC3 | 8 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 15_28_06-HC1 | HC1 | 9 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 15_35_03-LT1 | LT1 | 16 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 15_48_29-HC2 | HC2 | 11 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_02-TFC_test_A_1wk | 16_00_11-TFC_test_A_1wk | TFC_test_A_1wk | 8 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_02-TFC_test_A_1wk | 16_09_29-HC3 | HC3 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G18 -- 0 of 28 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_02_01-track_day1 | 13_36_06 |  | 7 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2022_02_01-track_day1 | 13_41_59 |  | 14 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2022_02_01-track_day1 | 13_54_06 |  | 7 | track day 1 | stub (excluded) |  |  |  |  |  |  |  |
| 2022_02_03-track_day2 | 15_35_07-HC1 | HC1 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_02_03-track_day2 | 15_41_30-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_02_03-track_day2 | 15_53_20-HC2 | HC2 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_02_05-track_day3 | 14_54_56-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_02_05-track_day3 | 15_00_17-LT1 | LT1 | 14 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_02_05-track_day3 | 15_11_58-HC2 | HC2 | 10 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 14_03_44-HC1 | HC1 | 8 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 14_19_24-LT1 | LT1 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_07-TFC_cond | 14_30_50-HC2 | HC2 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 14_38_13-CNO1 | CNO1 | 6 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 14_57_23-CNO2-borked | CNO2-borked | 9 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 15_03_49-CNO2 | CNO2 | 8 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 15_10_08-LT2 | LT2 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_07-TFC_cond | 15_21_43-HC3 | HC3 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 15_29_25-TFC_cond | TFC_cond | 27 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_07-TFC_cond | 15_55_09-HC4 | HC4 | 6 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_09-TFC_test_B | 14_55_34-HC1 | HC1 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_02_09-TFC_test_B | 15_01_36-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_09-TFC_test_B | 15_13_39-HC2 | HC2 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_02_09-TFC_test_B | 15_25_42-TFC_test_B | TFC_test_B | 19 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_09-TFC_test_B | 15_43_46-HC3 | HC3 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 11_51_11-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 11_56_41-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 12_07_42-HC2 | HC2 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 12_19_08-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_12-TFC_test_A | 12_27_12-HC3 | HC3 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 13_17_37-HC1 | HC1 | 8 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 13_24_41-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 13_36_18-HC2 | HC2 | 9 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 13_49_23-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_15-TFC_test_B_1wk | 14_17_02-HC3 | HC3 | 11 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 14_19_15-HC1 | HC1 | 9 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 14_26_40-LT1 | LT1 | 14 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 14_38_21-HC2 | HC2 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 14_47_39-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_16-TFC_test_A_1wk | 14_57_31-HC3 | HC3 | 10 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G19 -- 0 of 30 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_02_01-track_day1 | 14_17_11-HC1 | HC1 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_02_01-track_day1 | 14_23_04-LT1 | LT1 | 14 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_02_01-track_day1 | 14_34_23-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_02_03-track_day2 | 16_28_00-HC1 | HC1 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_02_03-track_day2 | 16_34_41-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_02_03-track_day2 | 16_46_52-HC2 | HC2 | 12 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_02_05-track_day3 | 15_45_03-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_02_05-track_day3 | 15_50_35-LT1 | LT1 | 16 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_02_05-track_day3 | 16_04_02-HC3 | HC3 | 10 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 16_19_13-HC1 | HC1 | 8 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 16_25_38-LT1 | LT1 | 18 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_07-TFC_cond | 16_40_40-HC2 | HC2 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 16_57_49-CNO1 | CNO1 | 12 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 17_07_15-CNO2 | CNO2 | 10 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 17_14_57-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_07-TFC_cond | 17_26_10-HC3 | HC3 | 8 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_07-TFC_cond | 17_35_25-TFC_cond | TFC_cond | 27 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_07-TFC_cond | 17_59_59-HC4 | HC4 | 10 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_02_09-TFC_test_B | 16_16_42-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_02_09-TFC_test_B | 16_22_27-LT1 | LT1 | 15 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_09-TFC_test_B | 16_34_55-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_02_09-TFC_test_B | 16_44_35-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_09-TFC_test_B | 17_02_26-HC3 | HC3 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 13_06_07-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 13_11_53-LT1 | LT1 | 13 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 13_23_06-HC2 | HC2 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_12-TFC_test_A | 13_32_18-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_12-TFC_test_A | 13_39_41-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 14_53_05-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 14_58_30-LT1 | LT1 | 13 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 15_09_42-HC2 | HC2 | 6 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_15-TFC_test_B_1wk | 15_18_44-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_15-TFC_test_B_1wk | 15_35_58-HC3 | HC3 | 8 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 15_30_15-HC1 | HC1 | 11 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 15_38_53-LT1 | LT1 | 14 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 16_00_42-HC2 | HC2 | 8 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_02_16-TFC_test_A_1wk | 16_10_03-TFC_test_A_1wk | TFC_test_A_1wk | 8 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_02_16-TFC_test_A_1wk | 16_18_38-HC3 | HC3 | 9 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G20 -- 0 of 31 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_03_01-monitor1 | 13_32_31-HC1 | HC1 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2022_03_09-monitor2 | 14_53_06-HC1 | HC1 | 12 | everything else | pending |  |  |  |  |  |  |  |
| 2022_03_14-track_day1 | 14_15_30-HC1 | HC1 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_03_14-track_day1 | 14_22_00-LT1 | LT1 | 13 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_03_14-track_day1 | 14_32_59-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_03_16-track_day2 | 13_38_06-HC1 | HC1 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_03_16-track_day2 | 13_43_39-LT1 | LT1 | 15 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_03_16-track_day2 | 13_56_07-HC2 | HC2 | 8 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_03_18-track_day3 | 14_03_26-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_03_18-track_day3 | 14_09_10-LT1 | LT1 | 14 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_03_18-track_day3 | 14_21_08-HC2 | HC2 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 13_25_47-HC1 | HC1 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 13_31_20-LT1 | LT1 | 14 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_22-TFC_cond | 13_43_11-HC2 | HC2 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 14_11_51-CNO1 | CNO1 | 6 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 14_16_55-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_22-TFC_cond | 14_28_23-HC3 | HC3 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 14_37_19-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_22-TFC_cond | 15_02_37-HC4 | HC4 | 6 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_24-TFC_test_B | 12_44_44-HC1 | HC1 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_03_24-TFC_test_B | 12_51_39-LT1 | LT1 | 14 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_24-TFC_test_B | 13_03_29-HC2 | HC2 | 11 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_03_24-TFC_test_B | 13_21_06-TFC_test_B | TFC_test_B | 19 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_24-TFC_test_B | 13_39_07-HC3 | HC3 | 8 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 16_02_48-HC1 | HC1 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 16_08_55-LT1 | LT1 | 14 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 16_21_16-HC2 | HC2 | 9 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 16_33_21-TFC_test_A | TFC_test_A | 7 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_26-TFC_test_A | 16_40_36-HC3 | HC3 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 14_35_59-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 14_41_18-LT1 | LT1 | 13 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 14_53_11-HC2 | HC2 | 11 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 15_06_19-TFC_test_B_1wk | TFC_test_B_1wk | 19 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_29-TFC_test_B_1wk | 15_24_36-HC3 | HC3 | 9 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 13_09_18-HC1 | HC1 | 7 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 13_15_54-LT1 | LT1 | 16 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 13_30_11-HC2 | HC2 | 13 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 13_45_31-TFC_test_A_1wk | TFC_test_A_1wk | 9 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_31-TFC_test_A_1wk | 13_54_45-HC3 | HC3 | 10 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

## G21 -- 0 of 32 batch sessions done

| day | session | type | videos | group | status | finished | wall_h | units | YrA | frames_checked | on_MINISCOPE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022_03_01-monitor1 | 14_04_19-HC1 | HC1 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2022_03_09-monitor2 | 15_24_21-HC1 | HC1 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2022_03_09-monitor2 | 16_22_52-HC2 | HC2 | 7 | everything else | pending |  |  |  |  |  |  |  |
| 2022_03_14-track_day1 | 15_06_01-HC1 | HC1 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_03_14-track_day1 | 15_11_27-LT1 | LT1 | 13 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_03_14-track_day1 | 15_22_39-HC2 | HC2 | 7 | track day 1 | pending |  |  |  |  |  |  |  |
| 2022_03_16-track_day2 | 14_38_52-HC1 | HC1 | 6 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_03_16-track_day2 | 14_44_31-LT1 | LT1 | 14 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_03_16-track_day2 | 14_56_16-HC2 | HC2 | 7 | track day 2 | pending |  |  |  |  |  |  |  |
| 2022_03_18-track_day3 | 14_48_58-HC1 | HC1 | 7 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_03_18-track_day3 | 14_54_16-LT1 | LT1 | 14 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_03_18-track_day3 | 15_05_58-HC2 | HC2 | 8 | track day 3 | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 16_26_15-HC1 | HC1 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 16_32_01-LT1 | LT1 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_22-TFC_cond | 16_43_06-HC2 | HC2 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 17_11_11-CNO1 | CNO1 | 11 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 17_19_40-LT2 | LT2 | 13 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_22-TFC_cond | 17_31_06-HC3 | HC3 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_22-TFC_cond | 17_39_30-TFC_cond | TFC_cond | 26 | TFC_cond days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_22-TFC_cond | 18_04_08-HC4 | HC4 | 7 | TFC_cond days | pending |  |  |  |  |  |  |  |
| 2022_03_24-TFC_test_B | 14_06_57-HC1 | HC1 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_03_24-TFC_test_B | 14_12_17-LT1 | LT1 | 16 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_24-TFC_test_B | 14_25_23-HC2 | HC2 | 7 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_03_24-TFC_test_B | 14_35_02-TFC_test_B | TFC_test_B | 18 | TFC_test_B days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_24-TFC_test_B | 14_52_11-HC3 | HC3 | 11 | TFC_test_B days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 17_19_25-HC1 | HC1 | 7 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 17_25_15-LT1 | LT1 | 14 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 17_37_21-HC2 | HC2 | 11 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_26-TFC_test_A | 17_50_05-TFC_test_A | TFC_test_A | 9 | TFC_test_A days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_26-TFC_test_A | 17_59_52-HC3 | HC3 | 8 | TFC_test_A days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 16_04_25-HC1 | HC1 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 16_10_00-LT1 | LT1 | 14 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 16_21_36-HC2 | HC2 | 8 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_29-TFC_test_B_1wk | 16_30_54-TFC_test_B_1wk | TFC_test_B_1wk | 18 | TFC_test_B_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_29-TFC_test_B_1wk | 16_48_17-HC3 | HC3 | 7 | TFC_test_B_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 14_44_01-HC1 | HC1 | 8 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 14_49_58-LT1 | LT1 | 14 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 15_02_13-HC2 | HC2 | 8 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |
| 2022_03_31-TFC_test_A_1wk | 15_12_05-TFC_test_A_1wk | TFC_test_A_1wk | 7 | TFC_test_A_1wk days | production |  |  |  | Y |  | Y (source) |  |
| 2022_03_31-TFC_test_A_1wk | 15_20_37-HC3 | HC3 | 13 | TFC_test_A_1wk days | pending |  |  |  |  |  |  |  |

# Data-exclusion audit (2026-10-13)

**Why:** a technical rule ("exposure-matched trials") dropped the first conditioning trial from every
Figure 2 analysis from 2026-08-22 until 2026-10-13. VS's rule since then: **never throw away data;
every rule that drops data is an explicit decision by VS.** This report lists every rule found that
drops trials, windows, frames, cells, sessions or mice, so each can be decided. Nothing below has been
changed except the trial-0 rule itself (§0).

Method: two read-only code audits (Figure 2 modules; everything else), with the top findings checked
by hand. Sizes marked "not measured" need a run to quantify. "05-12" means the rule is in the first
imported snapshot of the repo, so its true origin is earlier.

## Status (2026-10-14): fixed as follows (VS: "just fix everything")

| item | fix |
|---|---|
| §0 trial 0 | kept everywhere (2026-10-13) |
| 1.1 QC sentence | Methods now say every ROI was analysed and no cell-quality filter was applied |
| 2.1 G15 in PSTH / tone-cell analyses | reinstated (the C/YrA mismatch was fixed 2026-09-21) |
| 2.2–2.3 G07/G15 in cross-registered analyses | hard-coded lists removed. Analyses that do not need the missing session include them (place-field analyses, firing-rate changes, isomap, PCA pipeline 1); and every analysis defined on cells registered across all three of TFC_cond, Test_B, Test_B_1wk (or the Test_A trio) leaves out a mouse lacking one of the sessions, by checking the sessions (`utilities.mapping_sessions_missing`), with a printed note: the cross-registered place-field figures and firing-rate changes, PCA pipelines 2–4, three-session UMAPs, the R export. **Decision (VS, 2026-10-14):** restricted rather than narrowing G07/G15 to their two-session cells: G07 (285 cells) and G15 (155) would have roughly doubled the hM4D group's cells in the Test_B figures on a looser inclusion rule. Both mice stay in every per-session analysis |
| 2.4 G07 LT PV "outlier" | included |
| 2.5 silent `continue` in `sp_rates_lmm` | now prints which mouse has no session in the table |
| 2.6 loader per-mouse errors | the load stops after the error summary if any mouse failed (montage-plot errors excepted) |
| firing-rate changes: unmatched mapping rows skipped with a warning | now a hard failure (every mapped id exists, per the registry check) |
| 3.1 decoder speed mask | the fraction of frames removed is printed once per session |
| 3.2 the 180 s window | documented (decoder and spatial masks) |
| 3.3–3.5, 4.3–4.7, 5.x | unchanged: documented, logged or definitional |
| 4.1–4.2 remaining window rules | check the coverage tables after the rerun |

`caban/main.py` and `caban/analyses.py` (the old scripts `sections.py` was built from; not run by
the notebook) were not changed and still carry the old exclusions.

## 0. Fixed: the first conditioning trial (2026-10-13)

`epoch_modulation.retained_trials` and `sp_rates_lmm.restrict_to_exposure_matched_trials` dropped
trial 0 because its trace interval is 15 s (tone 1 at 185 s instead of 180 s). Now kept everywhere,
its trace averaged over its 15 s. Methods text, METHODS templates and docs updated. **Every Figure 2
conditioning result from 2026-08-22 to 2026-10-13 was computed on trials 2–5 and must be rerun.**

## 1. Text that misstates what was done (no data lost)

| # | Finding | Where | Decision needed |
|---|---|---|---|
| 1.1 | The paper's Methods say an automated QC pass "applied at load, excluded ROIs" (skewness, plateau, sparsity, minimum events). The filter only writes separate copies (`S_filt`, `C_filt`, `YrA_filt`); **every Figure 2 analysis reads the unfiltered `S`/`C`/`YrA`** (`sessions.py:1366`). No analysis module reads the filtered copies. | `docs/paper/paper_figure2.md` (QC sentence); `sessions.py:1118-1262` | Correct the text (all cells used), or apply the filter (drops cells: VS's call) |

## 2. Mice excluded

| # | Rule | Where | Used by | Status | Decision needed |
|---|---|---|---|---|---|
| 2.1 | **G15 skipped** ("G15 disabled due to s.C and s.YrA shape mismatch!! Fix then re-instate") | `analysis.py:2714, 3006, 3880, 4166` (+ `10266`, `11295`) | PSTH, PSTH shuffle, tone-responsive cell plots | **The mismatch was fixed 2026-09-21** (YrA aligned to S/C at load), so the stated reason no longer holds | Reinstate G15 |
| 2.2 | G07 and G15 excluded from cross-registered PCA, isomap, PV, within-group place fields, clustering (`EXCLUDE_MICE_CROSSREG`, `skip_mice=['G15']`) | `population.py:87`; `analyses.py:287-289, 4421, 4439`; `analysis.py:5314, 5519, 10250` | Crossreg population analyses | Reason given: they lack the full test-session set (G07: Test_B not recorded, disk full; G15: fewer test days) | Keep them in every analysis that does not need the missing session |
| 2.3 | `mice_per_group_Test_B_B_1wk` excludes G07, G15 | `loader.py:526` | Firing-rate-change plots | As 2.2 | As 2.2 |
| 2.4 | G07 dropped from LT PV correlation as an "outlier" (bottom arm truncated); `INCLUDE_OUTLIERS=False` hardcoded | `analyses.py:458` | LT PV figures | Comment only | Include, or a VS decision |
| 2.5 | Silent `if mouse not in sessions: continue` — a mouse that fails to load vanishes without a message | `sp_rates_lmm.py` ~751 (`_iter_event_windows`) | Every Figure 2 rate/amplitude result | No loss today (all 17 load); violates CLAUDE.md | Make it a hard failure |
| 2.6 | Loader catches per-mouse errors and continues (reported once at the end of loading) | `loader.py:1323-1327`, `1148-1288` | All analyses | Reported, but a failed mouse is missing from every figure | Hard failure, or keep with the report |
| 2.7 | `DEBUG` collapses the cohort to G05 | `loader.py:517` | All, only when `cfg.DEBUG` | Developer switch, default off | None |

## 3. Frames excluded

| # | Rule | Where | Used by | Size | Decision needed |
|---|---|---|---|---|---|
| 3.1 | **Decoder speed mask**: frames with smoothed speed < 2.0 removed from training and test (`use_speed=True` default, repeated ~12 times) | `decoder.py:619, 1300, 5855-5937`; `analyses.py:802, 2120-2549` | All position decoders, incl. TFC → Test_B transfer | Not measured, never reported | Keep (standard for place decoding) but report the fraction, or drop |
| 3.2 | **Pre-tone window**: only frames before tone 1 (else the first `first_n_sec=180` s) | `decoder.py:5855-5937`; `spatial.py:763, 917-971`; `place_cell_rates.py:237-272`; `locomotion.py:54` | Position decoders, place-cell maps, PV, remapping, place-cell rates, locomotion | Everything after tone 1 never enters these analyses | Design choice (pre-tone = no conditioning confound), but the 180 s cutoff is undocumented |
| 3.3 | Spatial bins with < 4 frames (0.2 s) occupancy invalid | `spatial.py:994`; `analyses.py:3769, 3891, 3927` | Place-cell maps, PV | Justified in docstring | Confirm |
| 3.4 | Speed tuning: bins with < 3 s occupancy dropped for the whole session (≥ 4 bins kept); cells with < 10 events or < 2 occupied bins excluded; coverage threshold lowered to 0.75 to fit G06 | `speed_tuning.py:105-121, 672` | Speed-tuning figure | Exclusion counts printed | Confirm |
| 3.5 | Unimaged timestamp rows dropped; frame-rate re-timing for listed sessions | `sessions.py:711-731, 700-727` | All | Logged per session; these remove no imaged data | None |

## 4. Trials and windows (Figure 2)

| # | Rule | Where | Used by | Size | Decision needed |
|---|---|---|---|---|---|
| 4.1 | Windows other than the trace must be 20 s ± 0.5 s (same rule family as trial 0) | `sp_rates_lmm.py:989-1054` | Rate, fraction active, joint epoch tests | Should catch only a recording ending mid-final-trial; per-mouse counts not checked | Check the coverage table after the rerun |
| 4.2 | Trial lost when a window is absent or the post-shock window does not fit | `epoch_modulation.py` `retained_trials` | All per-cell lanes | Written to `trial_coverage.csv` | Check after the rerun |
| 4.3 | Trials without a `post_shock_late` window dropped from the early-vs-late post-shock contrast | `sp_rates_lmm.py:918-951, 1446` | Descriptive contrast | Coverage reported | Confirm |
| 4.4 | `post_shock_late` left out of the rate model and joint profile | `sp_rates_lmm.py:354-365, 554-598` | Rate model, joint test | Documented (raggedness, sampler speed) | Confirm |
| 4.5 | **Shock (2 s) window excluded from all population models** | `sp_rates_lmm.py:354`; `paper_figure2.md` | All population models | Rationale: 67–91% of cells have no event in 2 s | Confirm (it is analysed per cell in `epoch_modulation`) |
| 4.6 | Amplitude endpoint uses only cells with ≥ 1 event in the epoch | `sp_rates_lmm.py:2764` | Amplitude panels | Definitional (no events, no amplitude); rate and fraction use all cells | None |
| 4.7 | G09's last post-shock window accepted at 387 of 400 frames | `epoch_analysis.py:71-77` | All conditioning analyses | Kept, not dropped | Already VS's decision (2026-10-01) |

## 5. Cells excluded

| # | Rule | Where | Used by | Size | Decision needed |
|---|---|---|---|---|---|
| 5.1 | Cells with no YrA trace, or a constant YrA or C trace | `epoch_modulation.py:198-252` | Per-cell lanes | 14 + 2 of 9,531 cells (0.17%), reported per mouse | Confirm (no trace = nothing to analyse) |
| 5.2 | YrA-only units dropped at alignment | `sessions.py:818-895` | All YrA uses | Logged | None |
| 5.3 | Decoder N-scaling keeps only N where every group has ≥ 2 mice | `decoder.py:11535` | Decoder N-scaling plot | Not stated | Confirm |
| 5.4 | Mice with location truncated to the PC length (G09, G21: video chunk 19.avi missing) | `population.py:88, 1009` | Crossreg PCA location overlays | Documented data fault | None |

## Not covered

Epoch-window code in `TFC.py` / `CFC.py`; `event_locked_responsiveness.py`, `engram.py`,
`population_coupling.py` beyond grep (clean); the sizes of 3.1–3.2 and 4.1–4.2, which need a run.

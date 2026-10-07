# Outstanding items after the Minian batch — with recommendations

Written 2026-10-07, after the batch finished (484 of 492 done, 8 failed) and every done session was
synced to MINISCOPE ([../reports/minian_batch_status.md](../reports/minian_batch_status.md)). It gathers
what is still open from [minian_open_items.md](minian_open_items.md) ("item N" below),
[figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) ("Fig. 2 plan") and
[handover_razer_batch_and_review.md](handover_razer_batch_and_review.md) (the 2026-10-02 update at its
top). Those files keep the detail; this one is the working list. Close items here as they are done,
with the commit.

**Already done, though [minian_open_items.md](minian_open_items.md) still lists them** (E6 updates it):
the Razer track steps 1–5 (the batch, 09-28 → 10-06); item 1, the `max_proj` replay check (`e293ed5`;
128 of 128 pass on both drives, `05062da`); item 2, the G05 NaN-motion recovery (`76013ae`); item 3, the
G21 timing fix (`044b4f1`; caches rebuilt, Figure 2 re-run 10-01/02); item 7, the loader reading the
recomputed YrA (`24fe3a9`); item 12's index part (`0e6657b`); item 14, Tailscale.

## A. Failed and excluded batch sessions

| # | item | where | recommendation |
|---|---|---|---|
| A1 | **The two LT1s with a gap mid-recording**: G06 `2021_10_22-TFC_test_A/12_18_46-LT1`, G05 `2021_09_09-TFC_test_A_1wk/16_11_07-LT1` | item 15 | **Checked 2026-10-07: not analysis sessions** — no existing result depends on them. They sit in the loader's `dpath_Test_A_LT1` / `dpath_Test_A_1wk_LT1` tables, but the loader builds no session from those tables, nothing else reads them, and neither ever had Minian output. **Recommendation: park them until D1.** If a grouping of test-day LT1s is chosen, run them on their finalised files (broken files in `session_queue.UNUSED_VIDEOS`; first check that the batch runner honours that table) and pair timestamps around the gap through the §15.1 mapping ([yra_recompute_plan.md](yra_recompute_plan.md) §15.1). |
| A2 | **Three more sessions with unfinished video files**: G07 `2021_10_20-TFC_test_B/16_25_43-HC3`, G11 `2021_12_02-TFC_test_A_1wk/15_32_02-HC2`, G18 `2022_02_07-TFC_cond/14_03_44-HC1` | item 15 | Leave out G07 HC3 (at most ~100 frames). Run G11 HC2 (3,000 frames) and G18 HC1 (4,000 frames) on their finalised files only if an HC analysis needs them; otherwise leave them out and note it. |
| A3 | **Two unfinished recordings**: G07 `2021_10_20-TFC_test_B/15_49_59-TFC_test_B`, G14 `2022_01_18-TFC_test_B_1wk/14_52_35-TFC_test_B_1wk-borked` | item 10; Fig. 2 plan §3 | Check that neither is a session the loader reads. It already reads other sessions for those test days, so these are most likely spares. If so, leave them out. |
| A4 | **G18 `2022_02_07-TFC_cond/14_57_23-CNO2-borked`**: the YrA step fails in `compute_trace` (`ValueError: All arrays must be instances of SparseArray`, after a divide-by-zero warning) | handover, 10-02 update | Debug on the Razer (probably an all-zero footprint), then `mr.resume_after_notebook`, with no notebook re-run. Small. It matters only if per-cell CNO analyses happen (D1). |
| A5 | **The runner's pre-check** catches only sessions where every `.avi` is unfinished | item 15 | Make it flag *any* unfinished file. Cheap, and it stops a repeat on future data. |
| A6 | **The 34 stub sessions and G09 `2021_11_08-TFC_cond/19_28_33-HC3`'s partial output** | item 10 | Leave them out. Look at 3–4 stubs to confirm they are bare timestamps, then close the item. |

## B. Correctness of the existing analysis

| # | item | where | recommendation |
|---|---|---|---|
| B1 | **Frame rate: 19.76 fps measured, 20 assumed; G05 `2021_09_03-TFC_test_A/16_34_42-TFC_test_A` recorded at 25** | item 13; Fig. 2 plan §2.6 (f), §2.7 item 4 | **Fix it rather than just document it**: use each session's rate from its timestamps. G05 Test_A's rates are about 19 % low, a real error in one hM3D mouse. The 1.2 % offset elsewhere is harmless for group comparisons, but windows built as seconds × 20 run 20.24 s while the timestamp-derived epochs run 20.0 s. Decide before more write-up, then do a before/after comparison as in the Fig. 2 plan §2.5. |
| | **B1 decided (VS, 2026-10-07): one constant, `MINISCOPE_FPS = 19.76`, with a loud ±0.1 % check per session and G05 Test_A as the one listed exception — plan: [miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md).** (Earlier the same day: fix everything that assumes 20 fps.) Evidence, checked the same day over every `Miniscope/timeStamps.csv` on MINISCOPE (G05–G21, sessions ≥ 2,000 frames): **655 sessions set to 20 fps run at 19.760 fps** (median; 1–99 % 19.754–19.763). **The rate is constant along each recording**: the rate in every 1,000-frame block lies within 19.720–19.767 (1–99 %; the median session's SD over blocks is 0.001 fps), and first vs second half differ by −0.013 to +0.006 fps. Single frame intervals jitter 48–53 ms (ms-quantised PC stamps) around 50.61 ms. Dropped frames are rare: 539 of 655 sessions have none; most of the rest have 1–2 (a 100–150 ms interval), and G09 `TFC_cond` has 19. **Not the PC clock**: the behaviour camera, stamped by the same PC, reads 15.00–15.03 fps (nominal 15) over 140 TFC_cond-day sessions, so the 1.2 % shortfall belongs to the Miniscope sensor. Two sessions are set to 25 fps (24.64 measured): G05 Test_A (known) and G15 `2022_01_11-TFC_cond/18_39_20-HC4` (new). Odd: G18 `14_57_23-CNO2-borked` (304 gaps, rate 27; ties in with A4), G11 `2021_11_27-TFC_test_A/17_11_20-HC2` (17 gaps). Scope: `MINISCOPE_FPS = 20` (`caban/utilities.py:16`) is used 145 times in ~20 modules, plus any literal 20 or frame windows sized at 20 fps (e.g. G09's 400-frame windows). |
| B2 | **Loader guard**: assert `YrA_idx == S_idx` (0 rows moved) before aligning | Fig. 2 plan §2.6 (d) | Do it: about 10 lines, and it turns a quiet misalignment into a loud failure. |
| B3 | **The parked collaborator notice** (`9015c23` on `review/timing-and-yra-fixes`) | handover, 10-02 update | Don't send it yet. Its numbers predate the YrA promotion and the `sp_rates_lmm` model change, so check them and regenerate `behaviour_params` first, after B1. |

## C. Figure 2 ([figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) §2.6–2.7)

| # | item | recommendation |
|---|---|---|
| C1 | (b) The event-proximal omnibus test at ≥ 20,000 draws | Do it before quoting its P (0.095). Run it on the Razer. |
| C2 | (c) G09 sensitivity check with that trial dropped | Do it; it's a cheap robustness check for the text. |
| C3 | (e) Re-run Fig. 2c, the sample traces (it plots YrA) | VS's notebook run. Quick. |
| C4 | O.8: sort panel K on held-out trials | Yes. It's honest by construction, and panel O already does it for its own analysis ([../docs/epoch_modulation.md](../docs/epoch_modulation.md) §O.8). |
| C5 | O.2: split-half stability (Q2) | Report it as supplementary. It's the only remaining answer to the per-neuron question. |
| C6 | Delete the old cache `ds_cache-20260512-exported-YrA.pkl` | After the Figure 2 write-up is final. |

## D. New sessions and cross-registration ([crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md); Fig. 2 plan §4)

| # | item | recommendation |
|---|---|---|
| D1 | Which new groupings (item 4; crossreg plan §6) | Keep groupings 1–7 exactly as published. Add the whole TFC_cond day (HC1, LT1, CNO1, CNO2, LT2, HC2, TFC_cond, HC3) if per-cell before/after CNO is wanted, which is the most valuable new question. |
| D2 | Groupings registry (item 5; crossreg plan §3) | Build it now; it needs only folder names. Explain first why grouping 1's `mappings_crossreg1.pkl` exists in one mouse only. |
| D3 | Cross-registration runner and its check (crossreg plan §5), then the loader reading the registry (§4) | After D2. Check it by re-deriving one published grouping and comparing it with production. |

## E. Infrastructure and housekeeping

| # | item | recommendation |
|---|---|---|
| E1 | Item 11: a second copy of MINISCOPE | Wanted. MINIRAZER now holds the raw data and all results, but both drives sit in one flat, and MINISCOPE is 90 % full. Get one off-site or at the lab. |
| E2 | Item 6: gate-run links and scratch on FUTROLA (G06 `09_52_24-HC1`, 6.6 GB; G10 `16_32_14-TFC_cond`, 25 GB) | Remove them; both checks are recorded. Confirm on screen where the live copies are first. |
| E3 | Item 8: the notebook's videos drop one frame in five | Leave the protected notebook alone. The videos are only a visual check, so document it in the Methods text (E4). |
| E4 | Item 9: Methods text for the Minian procedure (`analysis_methods_templates/`) | Write it now: about 1 h, unblocked. |
| E5 | Item 12: saved output in `notebooks/recompute_yra.ipynb` | Clear it; every report is also in each session's `YrA_recompute.json`. |
| E6 | Bring [minian_open_items.md](minian_open_items.md) up to date | Mark the done items with their commits, point to this plan, and add A4 and A5. |

## Suggested order

B1 → C1–C3 and B2 (the Figure 2 loose ends) → D2–D3, with E4 and E6 in the gaps; A1 waits for D1.
(A1 was first until 2026-10-07, when it turned out to touch no existing analysis.) The decisions on B1
and D1 are VS's; once they are made, the code and the runs on the Razer are Claude's.

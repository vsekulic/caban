# Plan: the Miniscope frame rate is 19.76 fps, not 20

Outstanding item B1 ([figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) §6.B); old open item 13
([minian_open_items.md](minian_open_items.md)). Inventory of every place the rate is used:
[fps_inventory_scratch.md](fps_inventory_scratch.md).

## 1. Decisions (VS, 2026-10-07)

1. **One constant: `MINISCOPE_FPS = 19.76`** (`caban/utilities.py:16`), the median measured rate.
   Not a per-session rate threaded through ~194 sites: that much new plumbing is itself a risk of
   quiet errors. Per-session rates vary by 0.05 % (19.754–19.763, 1–99 %), which is negligible.
2. **A loud check where sessions load:** each session's rate measured from its own
   `Miniscope/timeStamps.csv` must be within ±0.1 % of `MINISCOPE_FPS`, or loading raises, naming
   the session and both rates.
3. **One listed exception with its own rate: G05 `2021_09_03-TFC_test_A/16_34_42-TFC_test_A`**
   (metadata `frameRate: 25`, measured 24.64 fps). Keep in mind G15
   `2022_01_11-TFC_cond/18_39_20-HC4` (also 25 / 24.64), which is in no analysis today but may be
   used. Note its other settings differ too (`ewl` 4 vs 22, `gain` 2, `led0` 12), so check the
   imaging plane before using it.
4. **All caches are rebuilt** after the change (they would be rebuilt anyway).

## 2. The evidence (2026-10-07)

From every `Miniscope/timeStamps.csv` on MINISCOPE, G05–G21, sessions ≥ 2,000 frames:
- 655 sessions set to 20 fps run at **19.760 fps** (median; 1–99 % 19.754–19.763).
- **Constant along each recording:** every 1,000-frame block runs at 19.720–19.767 fps (1–99 %),
  and first vs second half differ by −0.013 to +0.006 fps.
- **The sensor, not the PC clock:** the behaviour camera, stamped by the same PC, reads 15.00–15.03
  fps (nominal 15) over 140 sessions.
- Two sessions set to 25 fps run at 24.64. The ratio 24.64 / 25 = 0.986 is close to 19.76 / 20 = 0.988:
  the same slow sensor clock at a 25 fps setting. Both metadata files record `frameRate: 25`, so the
  acquisition software was set to 25 for them. How is unknown; possibly an accidental change of the
  GUI control.
- Minian's `C`/`S` do **not** depend on the rate: the notebook's `FRAMERATE = 20` only sets the
  review video's playback speed, and `noise_freq` is a fraction of the sampling rate.

## 3. Steps

1. **The constant and the check.** `MINISCOPE_FPS = 19.76`; `MINISCOPE_FRAME_MS` follows (it is
   derived). The check goes where `BehaviourSession.get_timestamps` reads `timeStamps.csv`
   (`caban/sessions.py:547`), with the exception table next to `session_queue`'s other per-session
   tables (`UNUSED_VIDEOS`, `TIMESTAMPS_WITHOUT_VIDEO_FROM`). The rate is measured as frames over
   time, after dropping the pre-start frame 0, as in the 2026-10-07 scan.
2. **G05 Test_A: re-timed at load onto the 19.76 fps grid** (VS, 2026-10-08; replaces "pass its own
   rate to the sites it uses", which would have touched about a dozen Test_A analysis paths). See §5.
3. **Review the 26 frame-defined windows** (inventory category W-f). A window written directly in
   frames does not change with the constant (400 frames stays 400 = 20.24 s). For each one, decide:
   meant in frames (leave it), or meant in seconds (rewrite as seconds × `MINISCOPE_FPS`).
4. **Known follow-ons:**
   - G09's accepted short window (`SHORT_POST_SHOCK_WINDOWS`, 387 frames,
     `caban/epoch_analysis.py:88`) is an exact count. A 20 s window becomes 395 frames, so it fails
     on purpose. Re-derive it.
   - `caban/isomap.py`'s cache key includes the rate. Check that it changes and nothing stale is reused.
   - `caban/config.py:298`'s inline import of the constant (breaks the imports rule). Move it to the top.
   - The silent constant-rate fallbacks in `caban/decoder.py` / `caban/spatial.py` (inventory
     "Surprises" b). Make them hard failures.
5. **Rebuild all caches** (`ds_cache.pkl` and the derived per-analysis caches).
6. **Before/after comparison of the paper numbers**, as for Figure 2
   ([figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) §2.5). Expected: rates
   up 1.2 % for every group alike, comparisons unchanged; seconds-defined windows 1.2 % shorter;
   G05 Test_A's rates up 23 % (they are 19 % low now: 20 / 24.64 = 0.81), the one mouse-level change.
   `scientific-code-reviewer` and `statistics-checker` before anything is written up.

## 4. Status

- 2026-10-07: decisions recorded.
- **2026-10-08, step 1 done:**
  - `MINISCOPE_FPS = 19.76` and `MINISCOPE_FPS_TOLERANCE = 0.001` (`caban/utilities.py`).
  - `session_queue.MINISCOPE_FPS_EXCEPTIONS` lists G05 Test_A at 24.64.
  - `caban.sessions.check_miniscope_frame_rate` runs in `get_timestamps` for every session and stores
    `miniscope_fps_measured`.
  - Run over every session on MINISCOPE (G05–G21, ≥ 2,000 frames): 653 pass and 4 fail, none of them
    built by the loader: G05 `TFC_test_B/15_20_05-iso` (stub), G11 `TFC_test_A/17_11_20-HC2`
    (17 gaps, 19.687), G15 `TFC_cond/18_39_20-HC4` (24.64, deliberately unlisted) and G18
    `CNO2-borked` (27.0).
  - Fallout of the constant becoming a float, fixed:
    - 5 population-vector `bin_frames` and 2 spatial shuffle ranges now round to whole frames
      (`int(round(...))`); they would have crashed on fractional slices.
    - 4 dead blocks in `caban/analysis.py`'s PSTH functions deleted: they computed shock-onset frames
      as seconds × rate, then overwrote them with the timestamp-derived `s.shock_onsets`.
  - Unchanged, and fine: isomap's down-sampling factor is still exactly 2 (19.76 / 9.88).
  - Changes results, to see in the before/after comparison (step 6):
    - `filter_abnormal_cells`' plateau limit `int(15 s × fps)` goes 300 → 296 frames, so the
      abnormal-cell filter, and hence the cell sets, can move slightly;
    - every `int(s × fps)` site now truncates (e.g. 10 s → 197 frames).
  - G05 Test_A still ran on the constant until step 2 (below).
- **2026-10-09, step 4 done.**
  - **G09's accepted short window** (`epoch_analysis.SHORT_POST_SHOCK_WINDOWS`) stays at 387 frames,
    because that number is set by where imaging stops. The full window went 400 → 395 frames, so it
    is now 8 frames short instead of 13; comment updated. `sp_rates_lmm.restrict_to_exposure_matched_trials`
    (tolerance 0.5 s): full windows now read 19.990 s (395 / 19.76), and timestamp-defined trace
    windows about 20.0 s (they read 19.76 s at the old constant); G09's 19.585 s would now even pass
    the tolerance by itself.
  - **isomap:** its cache keys include `MINISCOPE_FPS`, so old caches are not reused; its downsample
    factor is still exactly 2.
  - **`caban/config.py`:** the inline import of `MINISCOPE_FPS` moved to the top of the module (no
    cycle: `caban.utilities` imports nothing from caban). `bin_width_frames` = int(10 s × 19.76) = 197.
  - **Constant fallbacks removed:**
    - `decoder._infer_dt_sec` (about 30 callers) and `spatial._infer_dt_sec_spatial` return
      1 / `MINISCOPE_FPS`. They used the *median* of the ms-rounded timestamp differences (51 ms, i.e.
      19.61 fps, about 0.8 % off the true 50.6 ms), with silent 1/30 s and 1/20 s fallbacks.
    - The LT 1D decoder's `default_fps=20` fallback is gone. The plotting function now reads `dt_sec`
      from the results, or fails, instead of assuming 20 fps; nothing passed `default_fps`.
  - **The 8 "unsure" sites:** none involves the frame rate. `look_ahead = int(20 / bin_width)` is
    seconds over bins; `seg_window` counts track segments; Minian's `noise_freq` / `wnd` are relative;
    `PERIOD_FRAMES` is outstanding item B4, out of this scope.
- **2026-10-09, step 3 done.**
  - The trial-average windows (`population.PERIOD_FRAMES`, 20/20/20/2/20 frames) stay in frames:
    changing them to seconds is out of the frame-rate scope (VS). A change to 20 s / 2 s was made
    (`6451414`) and reverted the same day. VS meant them as seconds, so this is open, separately.
  - The other frame-defined windows stay in frames (VS: their counts would not change), with a
    comment giving the time: spatial `min_occupancy_frames=4` (0.20 s), the selectivity heatmap
    display bin (5 frames, 0.25 s; label now 2 decimals), the sample trace (1,200 frames, 60.7 s), the
    decoder time bins (15 frames, 0.76 s). The PSTH defaults in `caban/analysis.py`
    (`frames_lookaround` / `frames_save`) are left as they are, uncommented.
  - The continuous PCA's binning (`PCA_FRAMES_PER_BIN = 1`, i.e. none, though its comment and the
    Methods text say 1 s) is **left as it is**: out of the frame-rate scope (VS, 2026-10-09; a change
    to 1 s bins was made and reverted the same day).
- **2026-10-08, step 2 done:** G05 Test_A is re-timed at load (§5). Tested on its real data (§5.3).
  Event counts: keep the re-timed ones (VS, 2026-10-09; §5.4).

## 5. Re-timing G05 Test_A (step 2)

### 5.1 Why

Test_A has no tones: the whole session is one "context" period in frames, and its frames feed about a
dozen analyses (binned rates and activity, proportional activities / event rates / amplitudes, cell
activity distributions, event-locked responsiveness, freezing-tuned cells, population coupling,
place-field and fluorescence maps, the per-cell lanes). Every one converts frames to time with the one
constant. Rather than pass G05's rate through all of them, the session is moved **once, at load**, onto
the standard grid, so the constant is right for it too. A window written in frames then also means the
same time in G05 as in every other mouse.

### 5.2 What the code does (`caban/sessions.py`, `caban/session_queue.py`)

- `session_queue.MINISCOPE_FPS_EXCEPTIONS` lists the session with its recording rate (24.64 fps).
  `check_miniscope_frame_rate` checks it against that rate (±0.1 %).
- `find_exp_boundaries`, step 0b: the recorded timestamps are replaced by an evenly spaced grid at
  `MINISCOPE_FPS` from the first to the last recorded stamp (`miniscope_retiming_grid`). This happens
  before the experiment boundaries are looked up, so the boundaries, the bounded timestamps and the
  velocity/location matching all come out on the new grid by themselves. Kept on the session:
  `miniscope_retimed`, `n_frames_recorded`, `tstamp_miniscope_recorded`, `tstamp_miniscope_retimed`.
- `get_CS_matrices`: after checking that `C` has one frame per recorded stamp, the **full** recordings
  are re-timed:
  - `S` is **summed** into the new frames (each recorded frame to the nearest new frame in time;
    `retime_sum`), so spike totals and rates are kept exactly;
  - `C` and `YrA` are linearly interpolated at the new frame times (`retime_interp`).
  The existing check that `C` has one frame per imaged row then holds on the new grid.
- The on-disk caches stay as recorded; re-timing runs in memory on every load, including the lazy
  reloads of `S_full` / `C_full` / `YrA_full` in `__getattr__`.
- A pickled session from before the change (no `miniscope_retimed`) refuses to load and asks for a
  `ds_cache.pkl` rebuild.
- Not supported, and raising: re-timing a session that also has timestamp rows without imaging.

### 5.3 Tests (2026-10-08, real data)

- **Grid:** 7,615 recorded frames → 6,107 at exactly 19.760000 fps over the same 309.0 s.
- **`S` totals:** kept for every one of the 318 cells (max abs difference 7e-12 against totals up to
  50,915).
- **`C`:** round-trip correlation (re-timed, interpolated back to the recorded times, against the
  original) ≥ 0.9994, median 0.99993; range kept.
- **Experiment boundaries** (via `find_exp_boundaries` on the real timestamps): frames [96, 6002].
  Duration by frames / 19.76 = 298.887 s, equal to the duration from the timestamps, and matching the
  behaviour camera's 298.864 s. Before, frames / constant gave about 375 s for this session. The
  grid spacing is exactly 50.607 ms.
- **A normal session is untouched:** G06 Test_A is not re-timed; its duration by frames / 19.76 is
  298.836 s against 298.865 s by timestamps (0.01 %).
- Not yet run end to end: `get_CS_matrices` on the real session, which happens in the cache rebuild
  (step 5), where its checks (`C` vs recorded stamps, `C` vs grid) apply.

### 5.4 Event counts change — decided: keep the re-timed counts (VS, 2026-10-09)

Events (`find_spikes_ca_S`, `thres = 2`) are local peaks of `S` above a fixed amplitude. Summing
24.64 fps frames into 19.76 fps ones raises per-frame values, so G05 Test_A's event count goes from
22,995 to 30,281 (+32 %; all 318 cells change). Neither version is out of line with the other 15
mice's Test_A sessions:

| | event rate (/cell/s) | `S` per cell per second |
|---|---|---|
| G05 as recorded (24.64 fps) | 0.234 | 18.9 |
| G05 re-timed (19.76 fps) | 0.308 | 18.9 (kept exactly) |
| other mice: median (IQR) | 0.224 (0.182–0.246) | 17.8 |
| other mice: range | 0.113–0.314 | |

The re-timed count is what this recording would give if its frames were 19.76 fps long, like every
other mouse's, so it is arguably the more comparable one. But the threshold was never calibrated for
the frame rate, and the data cannot say which is right. **Decided (VS, 2026-10-09): keep the re-timed
counts**, i.e. events are detected on the re-timed `S`, as the code does. Note it in the Methods text
(§5.5) and the before/after comparison (step 6).

### 5.5 Methods text (for item E4)

"One session (mouse G05, context test A) was recorded at 24.6 frames/s instead of 19.76; at load its
deconvolved activity was summed, and its calcium and residual traces linearly interpolated, onto a
19.76 frames/s time base spanning the same recording, so that all time-based measures are computed
identically to the other sessions."

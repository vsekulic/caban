# Plan: the Miniscope frame rate is 19.76 fps, not 20

Outstanding item B1 ([outstanding_items_plan.md](outstanding_items_plan.md)); old open item 13
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
2. **G05 Test_A's own rate (decision 3).** The one part that is more than one line. First trace
   which of the inventory's sites the Test_A analyses pass through for this session: tone windows
   are found by timestamp and need nothing; rates (spikes / (frames / fps)) and any
   seconds-to-frames window do. Then pass the session's rate to exactly those, and make every other
   use of `MINISCOPE_FPS` on this session fail loudly. Design after the trace; bring it to VS.
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

- 2026-10-07: decisions recorded; nothing changed in code yet.

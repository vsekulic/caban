# Miniscope fps inventory (scratch)

Read-only grep inventory, 2026-10-07. Scope: `caban/*.py`, `scripts/*.py`, notebook code cells. Branch feat/yra-unit-alignment.
Counts are sites (one line or one tight cluster of lines in one function). Commented-out code and docstring-only mentions are counted as M unless noted.
Legend: R rate, W-s window in seconds -> frames, W-f window in frames (implicit 20 fps), T time axis/label, F filter/smoothing, B behaviour cam / alignment, M metadata/logging/cache key, ? unsure.
"Reach" column: S = session object in scope, A = only arrays/frame indices, C = module/config constant, P = parameter already passed.

## Module table

| module | live/legacy | R | W-s | W-f | T | F | B | M | ? | total |
|---|---|---|---|---|---|---|---|---|---|---|
| caban/utilities.py | live | 5 | 1 | 0 | 1 | 1 | 1 | 2 | 0 | 11 |
| caban/sessions.py | live | 0 | 0 | 0 | 2 | 3 | 3 | 1 | 0 | 9 |
| caban/spatial.py | live | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 6 |
| caban/single_unit_common.py | live | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 3 |
| caban/engram.py | live | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| caban/epoch_analysis.py | live | 2 | 19 | 1 | 1 | 0 | 0 | 0 | 0 | 23 |
| caban/epoch_modulation.py | live | 0 | 1 | 1 | 4 | 0 | 0 | 4 | 0 | 10 |
| caban/epoch_sequence.py | live | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| caban/event_locked_responsiveness.py | live | 0 | 3 | 0 | 1 | 1 | 0 | 0 | 0 | 5 |
| caban/sp_rates_lmm.py | live | 1 | 4 | 1 | 5 | 0 | 0 | 1 | 0 | 12 |
| caban/analysis.py | live | 6 | 11 | 7 | 11 | 0 | 0 | 5 | 4 | 44 |
| caban/sections.py | live | 0 | 4 | 6 | 2 | 2 | 1 | 3 | 0 | 18 |
| caban/isomap.py | live | 0 | 0 | 8 | 0 | 3 | 0 | 4 | 1 | 16 |
| caban/population.py | live | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 5 |
| caban/place_cell_rates.py | live | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| caban/speed_tuning.py | live | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| caban/locomotion.py | live | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| caban/config.py | live | 0 | 1 | 0 | 0 | 0 | 1 | 2 | 0 | 4 |
| caban/loader.py | live | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| caban/decoder.py | live | 0 | 3 | 0 | 7 | 0 | 0 | 0 | 0 | 10 |
| caban/minian_runner.py | live | 0 | 0 | 0 | 3 | 0 | 0 | 2 | 0 | 5 |
| notebooks/run_pipeline.ipynb (root) | live driver | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| notebooks/minian_pipeline_BASELINE.ipynb | live (Minian template) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| caban/main.py | legacy | not tabulated (4 grep hits) | | | | | | | | 4 |
| caban/analyses.py | legacy | not tabulated (2 hits) | | | | | | | | 2 |
| caban/debug_snippets.py, debug_snippets3.py | legacy | not tabulated (4 + 8 hits) | | | | | | | | 12 |
| caban/decoder_good.py, decoder_copy.py | legacy | not tabulated (3 + 3 hits, `default_fps`/fps copies) | | | | | | | | 6 |
| caban/bak_old.py, debug_PSTH*.py, debug_snippets2.py, TFC.py, CFC.py, nature_comm_style.py, plotting.py, pca_state_metrics.py, freezing_tuned_cells.py, population_coupling.py, roi.py, pipeline.py, engram_sanity.py, session_*.py, yra_recompute.py, minian_gate.py, scripts/*.py | none / legacy | 0 fps hits | | | | | | | | 0 |

Live totals (R/W-s/W-f/T/F/B/M/?): R 22, W-s 56, W-f 26, T 39, F 12, B 6, M 25, ? 8 (approx. 194 sites; `MINISCOPE_FPS` has ~145 textual uses, the extra are `1/20`, `dt_sec`, literal frame windows and 20 s-derived constants).

### Liveness
- Entry points: root `run_pipeline.ipynb` (143 cells) imports caban.config, loader, pipeline, sections, roi, analysis, utilities, sessions, single_unit_common, pca_state_metrics, epoch_analysis, decoder, sp_rates_lmm, spatial, engram, engram_sanity, population, isomap, place_cell_rates, locomotion, speed_tuning. `notebooks/*.ipynb` (recompute_yra, run_minian_pipeline, BASELINE) import only minian_runner / yra_recompute.
- sections.py imports analysis, decoder, epoch_*, event_locked, freezing_tuned_cells, population(_coupling), isomap, roi, sp_rates_lmm, locomotion, place_cell_rates, speed_tuning, engram(_sanity). loader.py imports analysis, config, engram, roi, sessions, utilities, decoder.
- Legacy / dead (nothing imports them; `main`, `analyses` import each other / are standalone): main.py, analyses.py, debug_snippets.py, debug_snippets2.py, debug_snippets3.py (only named in a population.py docstring), debug_PSTH.py, debug_PSTH_2.py, decoder_good.py, decoder_copy.py, bak_old.py. TFC.py and CFC.py are not imported anywhere (zero fps hits). Root-level `SSTT_bursts*.py`, `fix_decoder.py`, `plot_lt_*.py` are outside the requested scope and were not read.
- `scripts/*.py`: no fps hits at all (Minian batch tooling only).

## utilities.py (live)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| utilities.py:16 | M | `MINISCOPE_FPS = 20` | the constant | C (replace) |
| utilities.py:17 | B | `BEHAVCAM_FPS = 15` | behaviour camera rate, nominal | C |
| utilities.py:19 | F | `SMOOTH_SIGMA = 4` | Gaussian sigma in frames for velocity smoothing (used sessions.py:1487-1493) | C; frames of two cameras |
| utilities.py:21 | T | `MINISCOPE_FRAME_MS = 1000 / MINISCOPE_FPS` | frame period in ms | C; used by spatial.py:113,131, locomotion.py:50, analysis.py:13102 |
| utilities.py:348 | R | `(end-beg)/MINISCOPE_FPS` | period length s in get_avg_sp_rate_in_period | A (frame idx only; callers in sessions.py hold S) |
| utilities.py:364 | R | same | period length in get_avg_activity_in_period | A |
| utilities.py:399 | W-s | `n_eligible < MINISCOPE_FPS` | refuse < 1 s of eligible frames | A |
| utilities.py:403 | M | `n_eligible / MINISCOPE_FPS` | error message seconds | A |
| utilities.py:460 | R | `counts / (n_eligible / FPS)` | per-cell event rate from eligible frames | A |
| utilities.py:494 | R | `duration_s = n_eligible / FPS` | rate denominator | A |
| utilities.py:537 | R | `n_frames / float(FPS)` | mean transient rate duration | A (engram path) |

## sessions.py (live; sessions are built here)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| sessions.py:31 | F | `fps=MINISCOPE_FPS` default arg of filter_abnormal_cells | frame rate for plateau filter | P (add session rate at call site) |
| sessions.py:90 | F | `int(thre_plateau * fps)` | plateau max duration 15 s -> frames | P |
| sessions.py:405 | B | `self.tstamp_tol = 70` ms | Miniscope<->behavcam nearest-timestamp tolerance (1.4 frames at 20 fps) | S (ms; fps independent but frame-interval-relative) |
| sessions.py:956 | M | `np.diff(exp_fnum)[0] / FPS` | duration inside a `'''` block (dead code) | n/a |
| sessions.py:1409 | B | comment: velocities assigned to miniscope frames by nearest behavcam timestamp | alignment | S |
| sessions.py:1487-1493 | F | `gaussian_filter(..., sigma=SMOOTH_SIGMA)` | velocity/location smoothing, 4 frames on each camera | S (behavcam sigma 4 frames = 0.27 s, miniscope 0.2 s) |
| sessions.py:1498-1517 | T | `ts2frame`, `t2frame` | seconds/ms -> frame by Miniscope timestamps within tstamp_tol; already timestamp-based | S (method) |
| sessions.py:1578-1595, 1786-1787 | B | `*_onsets_def * 1000` then nearest timestamp | tone/shock onset seconds -> miniscope frame via timestamps; already timestamp-based | S |
| sessions.py:588-631 | B | imaged-row alignment | timestamp rows <-> C frames; no rate | S |

Session sets `self.tstamp_miniscope` (ms) in `get_timestamps` (sessions.py:547-581); that array is the natural source of a per-session rate. Pickle `__getstate__/__setstate__` at sessions.py:422-438.

## spatial.py (live)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| spatial.py:113, 131 | R | `occupancy[...] += MINISCOPE_FRAME_MS` | occupancy time (ms) per visited bin; denominator of firing-rate maps | S? (OccupancyAnalysis init has `sess`; verify) |
| spatial.py:452, 466 | W-s | `np.arange(5*FPS, 5*60*FPS)` | circular-shift range 5 s to 5 min for shuffles | A/S |
| spatial.py:757-765 | T | `_infer_dt_sec_spatial`, fallback `1/FPS` | frame dt from timestamps (median), silent fallback to constant | S (already per-session) |
| spatial.py:922 | W-f | `min_occupancy_frames=4` | min occupancy per bin in frames (0.2 s at 20 fps) | P |
| spatial.py:306,339 | (not counted) | `SMOOTH_LOC_SIGMA=4` | spatial-bin smoothing, not temporal | n/a |

## single_unit_common.py, engram.py, epoch_sequence.py, event_locked_responsiveness.py, population_coupling, freezing_tuned_cells

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| single_unit_common.py:163, 237 | W-s | `int(round(5 * FPS))` | min shift for circular-shift nulls | A (signal arrays); session available to callers |
| single_unit_common.py:198 | F | `smooth_sigma_frames=20.0` | peak-contrast Gaussian sigma = 1 s at 20 fps (default; caller passes sigma at event_locked:139) | P |
| engram.py:67 | R | `n_frames / float(FPS)` | per-cell transient rate (Hz) over full session | A (S matrix only) |
| epoch_sequence.py:156 | F | `gaussian_filter1d(sigma=SMOOTH_S * FPS)` | 0.5 s smoothing | A |
| epoch_sequence.py:92,97 | W-s | `WINDOW_END_S=40`, `SMOOTH_S=0.5` | window and smoothing in s (frames via epoch_modulation._window_frames) | C |
| event_locked_responsiveness.py:82, 90 | W-s | `fps = FPS; int(round(onset_window_s*fps))` | onset window frames | S |
| event_locked_responsiveness.py:139 | F | `peak_smooth_s * FPS` | smoothing sigma | S |
| event_locked_responsiveness.py:157-159 | W-s | `pre_f, post_f = round(s*fps)` | peri-event window frames | S |
| event_locked_responsiveness.py:180 | T | `(np.arange(win)-pre_f)/fps` | time axis | S |
| population_coupling.py, freezing_tuned_cells.py | none | no rate; bin widths in frames passed in | | |

## epoch_analysis.py (live; the central window builder)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| epoch_analysis.py:75, 103 | W-s | `TRACE_MATCHED_WINDOW_S=20.0`; `round(S*FPS)` | trace-matched window 20 s -> 400 frames (`trace_matched_window_frames`) | C (no session arg; needs one) |
| epoch_analysis.py:83-95 | W-f | `SHORT_POST_SHOCK_WINDOWS = {('G09','TFC_cond',3): 387}` | exact accepted short post-shock window in frames | C (keyed to 20 fps frame counts; re-derive per session) |
| epoch_analysis.py:256-327 | W-s | `fps = MINISCOPE_FPS`; lines 272, 273, 286, 326, 327 | peri-shock / pre-shock / post-shock window offsets | S (session in scope) |
| epoch_analysis.py:397, 404 | R | `(offset-onset)/FPS` | window duration for rate exposure | A |
| epoch_analysis.py:1412-1461 | W-s | `fps = FPS`; 1416, 1426, 1446, 1447, 1451, 1461 | pre/post-tone windows, pseudo-tone windows | S |
| epoch_analysis.py:2476 | R | `pop_trace * FPS / n_cells` | population rate (events/cell/s) | A |
| epoch_analysis.py:2626-2632 | W-s | seven `int(round(x_s * FPS))` | peri/baseline/response windows in the shock-trace routine | P (function params in s) |
| epoch_analysis.py:2893 | T | `(arange - peri_pre_f)/float(FPS)` | time vector | P |

## epoch_modulation.py (live)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| epoch_modulation.py:339 | W-s | `_window_frames(duration_s) = round(s*FPS)` | the single seconds->frames helper for this module | C (needs session arg) |
| epoch_modulation.py:427, 500, 3415 | M | `/ FPS` in messages | window length in s for messages | A |
| epoch_modulation.py:740, 3162 | M | text "~40 frames at 20 Hz", "5 frames at 20 Hz" | prose in methods text | n/a (text) |
| epoch_modulation.py:911, 2801 | T | `(arange(pre_f+post_f)-pre_f)/FPS` | time axis | A |
| epoch_modulation.py:3164 | W-f | `SELECTIVITY_HEATMAP_BIN_FRAMES = 5` | display bin 0.25 s | C |
| epoch_modulation.py:3418, 3529 | T | `pre_f / FPS`, `BIN_FRAMES / FPS` | label/axis seconds | A |
| epoch_modulation.py:133 | M | comment: 396-frame intervals instead of 400 | documents quantisation | n/a |

## sp_rates_lmm.py (live)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| sp_rates_lmm.py:608 | W-s | `RECALL_WINDOW_S = 20.0` | recall window s (frames via epoch_analysis) | C |
| sp_rates_lmm.py:665 | W-s | `round(post_tone_duration_s*FPS)` | post-tone window frames | S |
| sp_rates_lmm.py:769 | R | `exposure_s = (offset-onset)/FPS` | negative-binomial exposure | S |
| sp_rates_lmm.py:1035 | W-f | `round(w['exposure_seconds']*FPS) == short` | compares exposure to accepted short window frame count (387) | S; couples with epoch_analysis:83 |
| sp_rates_lmm.py:9418 | M | label `frames, {FPS} fps` | prints constant | C |
| sp_rates_lmm.py:9581 | W-s | `pad = round(pad_seconds*FPS)` | example trace padding | S |
| sp_rates_lmm.py:9583, 9588, 9599 | T | `(arange - onset)/FPS` | trace time axes | A |
| sp_rates_lmm.py:9670 | W-s | `EXAMPLE_TRACE_PAD_S * FPS` | padding | S |
| sp_rates_lmm.py:9672, 9676 | T | `/ FPS` | time axis, axvspan | A |

## analysis.py (live; 13.6k lines, many near-duplicate PSTH functions)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| analysis.py:472, 488, 494, 503 | T | `len(bins)*bin_width/20/60` | **literal 20**: bin_width (frames) -> minutes on x axis | A (bin_width only; no session) |
| analysis.py:543 | R | `np.diff(s.miniscope_exp_fnum)[0]/FPS` in generate_interneuron_cutoff | session duration s for firing rate | S |
| analysis.py:929 | R | `S.shape[1]/float(FPS)` | duration | A |
| analysis.py:2107-2108, 2189-2190 | R | `(period[1]-period[0])/FPS` | LT period length s | S |
| analysis.py:2689, 2957, 3419, 3844, 4131 | W-f | `frames_lookaround=40, frames_save=200` | PSTH half-width 2 s, saved width 10 s | P |
| analysis.py:4337, 4579 | W-f | `frames_lookaround=20, frames_save=300` | PSTH windows | P |
| analysis.py:2693 | W-f | call with frames_lookaround=... | literal in caller | P |
| analysis.py:3026, 3482, 3905, 4194 | W-s | `onsets.append(s.shock_onsets_def[i]*FPS)`; `onsets[i] *= FPS` | shock onset s -> frame by multiplication (not by timestamps) | S (s.ts2frame / s.shock_onsets exist) |
| analysis.py:2732, 2734 | M | commented-out copies of the above | | n/a |
| analysis.py:10072, 10141, 10330, 11340, 12629 | W-s | `bin_frames = bin_width * FPS` | bin width s -> frames | S/P |
| analysis.py:10331, 11341, 12630 | ? | `look_ahead = int(20 / bin_width)` | 20 s lookahead in bins; seconds only, no rate | n/a |
| analysis.py:10826, 10868, 11072, 11130, 11758, 11814 | T | `min(1300, floor(C.shape[1]/FPS))` then `s.ts2frame(xtick_seconds*1000)` | x tick seconds from frame count, mapped back by timestamps (inconsistent mix) | S |
| analysis.py:11860 | R | `(bound[1]-bound[0])/FPS` | period length s | S |
| analysis.py:11971 | W-s | `bin_len = FPS*bin_len_s` | bin length | P |
| analysis.py:12747, 12993, 13022 | ? | `seg_window=25` | segment window; unclear units (frames? 1.25 s) | P |
| analysis.py:13092, 13102, 13606 | T | `searchsorted(ts, first_n_sec*1000)`; `dt_sec = MINISCOPE_FRAME_MS/1000`; `tstamp_miniscope[:n]/1000` | class dt from constant (13102), others from timestamps | S |
| analysis.py:575, 605, 619, 1226, 12060, 12242 | M | "Firing Rate (Hz)", "Avg. spikes/s" labels | units text | n/a |

## sections.py (live; 9k lines)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| sections.py:296 | M | docstring: "19.76 fps (G05 Test_A at 24.6)" | `dump_behaviour_params` already knows | n/a |
| sections.py:343 | B | `_frame_time_ms(s, v)` from timestamps | frame -> time by timestamps; already per-session | S |
| sections.py:379 | M | `MINISCOPE_FPS = {..} (BEHAVCAM_FPS = ..)` | provenance record | C |
| sections.py:1847 | W-f | `len_trace=1200` | sample trace length 60 s | P |
| sections.py:2538-2540 | W-f | `time_bin_frames_orig/true_2D/pvt = 15` | decode chunk 0.75 s | C/P |
| sections.py:3340 | W-f | `_zcra_time_bin_frames = 15` | decode chunk | C |
| sections.py:3510 | W-f | `TIME_BIN_FRAMES = 15` | shared decode chunk | C |
| sections.py:6533-6534 | W-f | `PCA_FRAMES_PER_BIN = 1`; comment "1 s/bin at 20 Hz" | PCA bin; comment says 1 s, value says 1 frame | C |
| sections.py:7017, 7063 | W-s | `round(FPS*bin_size_s)` | frames_per_bin | C |
| sections.py:7067 | M | error message | | n/a |
| sections.py:7966, 8408 | F | `dff_rolling_window_s=60.0` -> `round(s*FPS)` | rolling-percentile dF/F baseline window | S (per mouse C) |
| sections.py:8298, 8303 | T | `round(val*FPS)`, `round(val*60*FPS)` | scale-bar seconds/min -> frames | A/S |
| sections.py:3608, 3621 | (not counted) | `optimization_first_n_sec=180.0` | 180 s pretone mask; done by timestamps in analysis.py:13092 / place_cell_rates.build_frame_mask | S |

## isomap.py, population.py

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| isomap.py:63 | F | `DOWNSAMPLE_HZ = FPS / 2` | temporal downsample target 10 Hz | C (becomes 9.88; see design notes) |
| isomap.py:306-312 | F | `FPS % DOWNSAMPLE_HZ != 0` raise; `int(FPS // DOWNSAMPLE_HZ)` | integer block-mean factor | C (float modulo risk) |
| isomap.py:567, 1408 | M | `round(NN_CUTOFF,6), DOWNSAMPLE_HZ, MINISCOPE_FPS` | **disk cache key** (NN summary JSON) | C |
| isomap.py:973, 1491 | M | `"MINISCOPE_FPS": MINISCOPE_FPS` | provenance written to results | C |
| isomap.py:1955 | F | docstring: native FPS, divide by factor | | n/a |
| isomap.py:1962-1964, 1993-1997 | W-f | `PERIOD_FRAMES[k] // factor` | trial window widths 20/20/20/2/20 frames divided by downsample factor | C (imported from population) |
| isomap.py:44 | ? | `from population import PERIOD_FRAMES` | units unclear (see population) | |
| population.py:47-53 | W-f | `PERIOD_FRAMES = {pre_tone:20, tone:20, post_tone:20, shock:2, post_shock:20}` | canonical trial period widths | C |
| population.py:164 | R | `frames_per_bin / float(FPS)` | bin seconds | C |
| population.py:341-359 | W-f | slices built from PERIOD_FRAMES | trial-average layout | C |
| population.py:973, 981 | W-s | `(FPS*20) // frames_per_bin` | 20 s extension in bins | C |
| population.py:341 / isomap.py:1962 | ? | semantics | in population these are PCA bins (1 s each at frames_per_bin=20?), in isomap they are divided by a native-frame downsample factor; units look inconsistent, check before touching | |

## place_cell_rates.py, speed_tuning.py, locomotion.py

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| place_cell_rates.py:303 | R | `duration_s = n_eligible/FPS` | eligible time | S |
| place_cell_rates.py:307 | W-s | `n_eligible < FPS` | min 1 s guard | S |
| speed_tuning.py:288, 293, 297 | R | `seconds[b] = n/FPS`, sum check, `round(sum*FPS)` | time per speed bin (exposure) | S/A |
| locomotion.py:50 | T | `_DT_SEC = MINISCOPE_FRAME_MS/1000` | module-level frame dt for locomotion metrics | C (module constant; must become per session) |

## config.py, loader.py, pipeline.py

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| config.py:41 | B | `behavcam_fps: int = 15` | config field | C |
| config.py:42-45 | M | comment/default `bin_width_seconds=10.0` | | C |
| config.py:298-299 | W-s | `bin_width_frames = int(FPS*bin_width_seconds)` (inline import, violates CLAUDE.md) | one frame bin width for the whole cfg = 200 | C (shared across all sessions) |
| loader.py:474 | W-s | `BIN_WIDTH = cfg.bin_width_frames` | passes the shared bin to analyses | C |
| pipeline.py | none | `PCA_FRAMES_PER_BIN` pass-through | | |

## decoder.py (live; already mostly per-session)

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| decoder.py:610-621 | T | `_infer_dt_sec(LT)`: median diff of `tstamp_miniscope`; silent fallback `1/30` | per-session dt | S (already) |
| decoder.py:2654-2672 | T | `default_fps=20.0`; `dt_sec = 1.0/default_fps` fallback | dt with constant fallback | P |
| decoder.py:27947, 28034, 28059 | T | `default_fps=20.0`, `1.0/default_fps` | dt fallback when stored block lacks dt | P (cached dicts) |
| decoder.py:10168, 10240 | T | `dt_sec ... else 1.0/20.0` | class default dt | P |
| decoder.py:6311, 8364 | W-s | `block_size = int(10.0/dt_sec)` | 10 s CV blocks | P (per session dt) |
| decoder.py:9988 | W-s | `(60.0/dt_sec)/w` | chunks per minute | P |

## minian_runner.py and notebooks

| file:line | cat | expression | computes | reach |
|---|---|---|---|---|
| minian_runner.py:229 | M | `"FRAMERATE": FRAMERATE` parameter dict | passed to Minian template | C (constant in module) |
| minian_runner.py:747-800 | T | `report_session(..., framerate)`; `duration_min = n/framerate/60` | report duration | P (taken from parameters["FRAMERATE"]) |
| minian_runner.py:828-832 | T | `np.arange(n)/framerate/60` | minutes axis | P |
| minian_runner.py:926-927, 1192 | M | report text, call | | P |
| BASELINE.ipynb cell 14 | M | `FRAMERATE = 20` | **checked 2026-10-07:** used only by `generate_videos(..., framerate=FRAMERATE)` (cell 287) — playback speed of the review video. `noise_freq` (0.06) is a normalised frequency (fraction of the sampling rate), so CNMF never uses Hz: `C`/`S` do not depend on the nominal rate | none needed |
| BASELINE.ipynb cells 14, 121, 214, 270 | ? | `noise_freq: 0.06`, `wnd: 10/15` | Minian temporal/spatial params as cycles/frame-relative | outside caban/*.py; already-run data, not re-runnable cheaply |
| run_pipeline.ipynb cell 27 | M | print "(frames; 20 Hz, so 1 frame = 50 ms)" | text only | n/a |
| yra_recompute.py, minian_gate.py, session_*.py | none | no rate; `wnd=15` is a spatial disk radius | | |

## BEHAVCAM_FPS (brief)

Only three sites: utilities.py:17 (definition), config.py:41 (`behavcam_fps: int = 15`), sections.py:379 (provenance print). Nothing computes with either; behaviour-camera conversions go through timestamps (sessions.py:608, 1409, 1578-1595) and `tstamp_tol = 70 ms`. Fine as is; nominal 15 vs measured 15.02 is not used numerically.

## Design notes

**Where sessions are built.** `BehaviourSession.__init__` in caban/sessions.py (tstamp_tol at :405, `get_timestamps` at :547-581 loads `Miniscope_df` / `BehavCam_df` through `saver_timestamps`; the Miniscope timestamp array `tstamp_miniscope` (ms) is the only per-session time base). `TraceFearCondSession` derives tone/shock onset frames from timestamps (:1578-1595). caban/loader.py (`load_all_mice`) builds the dataset and passes `cfg.bin_width_frames` (:474). A per-session rate can be computed once in `get_timestamps` (median diff of imaged rows, after the unimaged-row drop at :588) and stored as `session.fps` / `session.frame_ms`.

**Who has the session.** Already per-session (timestamp-based): sessions.ts2frame/t2frame, `_infer_dt_sec` (decoder.py:610), `_infer_dt_sec_spatial` (spatial.py:757), `_frame_time_ms` (sections.py:343), analysis.py:13092 and :13606. These show the codebase partially solved this; but several keep a silent fallback to a constant (decoder.py:621 uses 1/30, spatial.py:765 and decoder.py:2670-2672 use 20).
Functions that receive only arrays and therefore need the rate threaded in: utilities.get_avg_sp_rate_in_period / get_avg_activity_in_period (:348, :364), the eligible-frame rate helpers (:399-537), engram per-cell rate (engram.py:67), analysis.py:472-503 (literal 20), single_unit_common shuffles (:163, :237), epoch_modulation._window_frames (:339), epoch_analysis.trace_matched_window_frames (:103). Everything in epoch_analysis windows (:256-327, :1412-1461) already holds a session. Module-level constants that will need to become per-session values: locomotion._DT_SEC, utilities.MINISCOPE_FRAME_MS, config.bin_width_frames, isomap.DOWNSAMPLE_HZ, population.PERIOD_FRAMES.

**Riskiest spots.**
1. isomap.py:567 and :1408 (and :973, :1491): `DOWNSAMPLE_HZ` and `MINISCOPE_FPS` are in the on-disk NN-summary cache key; `DOWNSAMPLE_HZ = FPS/2` and `FPS % DOWNSAMPLE_HZ != 0` (:306-312) is a float modulo, `19.76 % 9.88` may not be exactly 0 and would raise or give factor 1. A per-session rate also makes the cache key session-dependent; stale caches from the 20 fps run would be silently reused unless the key changes.
2. epoch_analysis.py:83-95 plus sp_rates_lmm.py:1035: `SHORT_POST_SHOCK_WINDOWS = {('G09','TFC_cond',3): 387}` is an exact frame count derived at 20 fps (400 frames = 20 s); the comparison at sp_rates_lmm:1035 re-derives frames from seconds with the same constant. With a per-session rate the expected count changes (about 395 at 19.76) and the exception would raise or misfire; window frame counts also differ between sessions, so any pooled or cross-session comparison of frame-count windows changes.
3. config.py:298-299 and loader.py:474: `bin_width_frames = int(FPS * 10)` is one number (200) in `cfg`, shared by all sessions and by every analysis using `BIN_WIDTH` (analysis.py:10072+, sections.py:1795); per-session seconds-per-bin would differ by 1.2 % and by 19 % for the 24.64 fps sessions. The inline import at config.py:298 also breaks the project's import rule.
4. population.py:47-53 / isomap.py:1962-1997: `PERIOD_FRAMES` (20/20/20/2/20) is a module-level frame constant shared by every session and used both as PCA bins and, divided by a native downsample factor, as frames; units look inconsistent between the two modules (see note above).
5. analysis.py:472-503 (`bin_width/20/60`), analysis.py:3026, 3482, 3905, 4194 (`shock_onsets_def * FPS` instead of timestamp lookup) and the PSTH defaults `frames_lookaround=40/20`, `frames_save=200/300` (analysis.py:2689-4579): hard-coded literal 20 and frame-defined PSTH windows in a 13.6k-line module with many near-duplicate functions; easy to miss one.
Also: sections.py `TIME_BIN_FRAMES = 15` (:3510) and the 15-frame decode chunks (:2538-2540, :3340) are frame counts shared across sessions in cross-session decoding (cached decoder results carry `dt_sec` from `default_fps=20.0` fallbacks, decoder.py:27947-28059).

**Surprises.** (a) `MINISCOPE_FRAME_MS` (`= 1000 / MINISCOPE_FPS`, utilities.py:21) is derived, not a second constant, but it is module-level and missed by a `MINISCOPE_FPS` grep (spatial.py:113, :131 occupancy; locomotion.py:50; analysis.py:13102). (b) Several modules already infer dt from timestamps but with constant fallbacks (decoder 1/30, spatial 1/FPS, decoder default_fps=20), contradicting the no-silent-fallback rule. (c) config.py imports inline. (d) ~~Minian estimated S/C assuming 20 fps~~ — **wrong (checked 2026-10-07):** the BASELINE notebook's `FRAMERATE = 20` only sets the review video's playback rate; `noise_freq` is normalised to the sampling rate, so `C`/`S` are frame-based and unaffected. (e) analysis.py ticks mix constant-rate seconds (`C.shape[1]/FPS`) with timestamp lookups (`ts2frame`).

**Not covered.** Legacy files were counted only by grep hits, not read; root-level scripts (`SSTT_bursts*.py`, `fix_decoder.py`, `plot_lt_*.py`, `_tmp_*.py`) outside scope; per-site "reach" in analysis.py and sections.py marked S/P by inspection of the nearby signature, not verified for every one of the ~44 + 18 sites; population/isomap PERIOD_FRAMES units unresolved; counts are approximate (+/- a few) because several lines hold more than one expression.

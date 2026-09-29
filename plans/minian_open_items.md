# Open items: Minian batch, YrA, cross-registration — to go through one by one

Written 2026-09-27 at the end of a long session, as the agenda for the next chat(s). Each item is
self-contained: what, why, where it is documented, what it needs from VS, rough effort.

## ✅ The MINIRAZER copy is done and verified (2026-09-28)

Finished 08:06, verified 08:28: 863,373 of 863,375 files identical to MINISCOPE; the other 2 are the
gate-run links of item 6. See [razer_runner_plan.md](razer_runner_plan.md) §5. From here on only the
Razer processes sessions; every command on it goes through `~/bin/logrun` into `~/minian.log`.

**Work in this repo happens in one chat at a time** (VS, 2026-09-27): a second chat working the items
in parallel rewrote shared code and plans; its 5 commits are parked, unmerged, on the branch
`review/timing-and-yra-fixes` (the `max_proj` check, the G05 recovery, a loader timing change that made
`ds_cache.pkl` unloadable, a collaborator notice, frame-rate findings), to be reviewed one by one.

## After the copy: the Razer track (continues [razer_runner_plan.md](razer_runner_plan.md))

These follow in order and are Claude's to run, reporting as they go:

1. **Verify the copy** — file list and sizes against MINISCOPE (the standard VS accepted for 4-MINISCOPE).
2. **Dry run on the Razer** — `scripts/run_minian_batch.py --dry-run` should list the same queue as the
   Mac: 492 sessions, **5 DONE** (G06 HC1, G13 HC2, and the 3 G05 sessions the Mac ran on 2026-09-27:
   `track_day0/15_10_26-LT1`, `track_day1-tests/14_16_58-LT1`, `track_day2/14_41_34-LT1`) and
   **487 to run** (G05 `track_day3/16_08_43-LT1`, which failed when FUTROLA was unplugged, was cleared
   back to pending).
3. **G10 production check on the Razer** (Phase 3.3) — `PATH` re-run of G10 `16_32_14-TFC_cond`,
   compared with production by `minian_gate`; on the Mac it matched to solver precision.
4. **One short real session**, then memory with 1, 2, 3 streams (Phase 3.4).
5. **The batch** (Phase 4) — 2–3 `screen` streams on disjoint selections; results synced back to
   MINISCOPE by a sync script (still to write; it must replay the `minian` → `minian-ORIG` renames).

---

## Items to go through (not blocked by the copy unless noted)

### 1. A `max_proj` replay check for every YrA recompute — build
[yra_recompute_plan.md](yra_recompute_plan.md) §14.2. The recompute proves its movie replay only for the
28 sessions with an old `YrA` export; for the rest nothing independent checks it (NaN motion is now
caught; finite-but-wrong motion would not be). Compare the replayed movie's maximum over frames with the
`max_proj` production saved. Tolerance: expect ±1 grey level at a few pixels (cv2 5.0 vs 4.5 — G06 HC1's
backfill showed 5 pixels over 50 frames). Then re-check all 128 production recomputes — **after the
copy, on the Razer**, where the data will be local. Effort: ~1 h code + the re-check run.
**Code built 2026-09-27** ([yra_recompute_plan.md](yra_recompute_plan.md) §14.2, "Built"): hard
check max |diff| ≤ 1; re-check with `yr.verify_all(items, stop_on_error=False)` on the Razer.

### 2. Recover the two G05 NaN-motion sessions — build + run
[yra_recompute_plan.md](yra_recompute_plan.md) §14.1. G05 `2021_09_01-TFC_test_B` `15_37_05-LT1` and
`16_20_07-TFC_test_B`: their saved `motion.zarr` is NaN in every frame, so their `YrA` cannot be
recomputed (the only 2 of 130 not done). Plan: run the template cut after `max_proj`, prove it by an exact
`max_proj` match with production, store `motion_reestimated.zarr`, recompute `YrA` with it. `A`/`C`/`S`
untouched. Runs on the Razer after the copy. Effort: ~1–2 h code + ~30 min compute.

### 3. G21 TFC_test_B timing gap in the analysis loader — VS decision
[yra_recompute_plan.md](yra_recompute_plan.md) §15.1. Production never read `11.avi`–`13.avi` (unfinished
recordings), so `C` frames from 11,000 on were recorded at timestamp rows 14,000+; `sessions.py` pairs
`C` frame *i* with timestamp row *i*, so those 3,856 frames are 150 s early in every analysis of that
session, and the experiment's end boundary may point past `C`. **Existing, published pipeline.** Decide:
fix the loader (map frames to timestamps through the files actually read), and check which published
G21 Test_B results move. The only such gap among the 130 (G15's missing file is at the end: harmless).

### 4. Which new cross-registration groupings — VS decision
[crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) §6. Scientific choice; candidates: the
whole TFC_cond day incl. HC and CNO (per-cell before/after CNO); LT1 across track days (place-cell
stability); HC with the same day's LT/TFC session. Recommended: leave groupings 1–7 exactly as published.

### 5. The groupings registry — build
[crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) §3. A tracked file saying which sessions
form each grouping, seeded from today's `minian_crossreg*` folder names and proven to reproduce them
(17 mice × 7 groupings). Needs only directory listings — fine during the copy. Prerequisite for the
crossreg runner (§5 there) and for ever renaming output folders to plain `minian` (§4 there).
Oddity to explain first: grouping 1's mapping file exists as `mappings_crossreg1.pkl` in one mouse only.

### 6. Leftover gate-run links and scratch — VS decision, then cleanup
G06 `2021_10_18-TFC_cond/09_52_24-HC1` and G10 `2021_11_23-TFC_cond/16_32_14-TFC_cond` still hold a
`minian_intermediate` symlink to FUTROLA (old linked-scratch style), and FUTROLA keeps their scratch
(6.6 GB and 25 GB). Both gate checks are finished and recorded. Remove links + scratch, or keep.
(Removing the links also silences their robocopy errors on any later sync.)

### 7. Switch analyses to the recomputed `YrA` — after the batch
[yra_recompute_plan.md](yra_recompute_plan.md) §11.3. `sessions.py:739` still loads the old `YrA.zarr`,
caches it in `npy_files/<type>/YrA/*_YrA_full.pkl` and repairs the unit order at load time. Promotion:
the loader reads `YrA_recomputed.zarr` where it exists, and the old `YrA` pickles are invalidated so the
cache rebuilds. Design needed; do after the recompute reports are reviewed.

### 8. The notebook's videos drop one frame in five — VS decision
[minian_batch_runner_plan.md](minian_batch_runner_plan.md) §13. `minian.visualization.write_video` pipes
raw frames without an input rate (ffmpeg assumes 25 fps, resamples to 20): `minian.mp4`/`minian_mc.mp4`
hold 80 % of the frames — true of every 2021 video too. Arrays unaffected. Options: leave (videos are a
visual check), or fix in the protected template (a deliberate template change, new md5). Related, not
built: the two runner-made videos of §7 (`minian_raw_traces.mp4`, `minian_preprocessing.mp4`), which
must declare the input rate.

### 9. Paper Methods text — write
[minian_batch_runner_plan.md](minian_batch_runner_plan.md) §7a. A Nature-style Methods section for the
whole Minian procedure, numbers taken from a run's `parameters.json` (now available: G13 HC2, G10
TFC_cond, the G05 runs), copied into every `minian_run/` by `_copy_analysis_methods_template`. Template
does not exist yet in `analysis_methods_templates/`. Effort: ~1 h.

### 10. Stubs and partial output — VS decision
34 stub sessions (bare timestamps, `iso`, `HC1a`, `HC2b`, `LT1a`) are listed and left out of the queue;
G09 `2021_11_08-TFC_cond/19_28_33-HC3` holds partial output (and all-NaN motion) and is left out too.
Eyeball them ([local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §2) and decide which, if any,
join the batch.

**Test sessions processed but not for analysis by default** (VS, 2026-09-28): G05
`2021_09_07-TFC_test_B_1wk-redux/16_22_40-TFC_test_B_1wk` — most likely a test recorded when the bedding
was changed. Processed by the batch like any session; whether any analysis uses it is decided
separately. Any loader or crossreg grouping that picks sessions by type must not include it silently.

### 11. A second copy of MINISCOPE — VS decision
After Phase 2, MINIRAZER (NTFS) holds a copy of MINISCOPE's raw data, but it becomes the *working* drive
and will diverge in outputs. [local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §5.1 planned
an APFS `MINISCOPE-BAK`. Decide whether MINIRAZER counts as the backup of the raw data or a proper
backup drive is still wanted.

### 12. Housekeeping
- `notebooks/recompute_yra.ipynb` has uncommitted saved output (the batch log): clear outputs or keep
  locally (every report is also in each `YrA_recompute.json`).
- `plans/0_INDEX_plans.md`: add [razer_runner_plan.md](razer_runner_plan.md),
  [crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) and this file (`/index plans`).
- Renaming production output folders to plain `minian`
  ([crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) §4) waits for item 5.

### 14. Tailscale: reach the Razer from anywhere — ✅ done 2026-09-28 (`ssh minastirith`)
`ssh minastirith` works only on the home network (192.168.3.10). Tailscale gives both machines private
addresses reachable from anywhere, with no ports opened to the internet. **First step of the handover**
([handover_razer_batch_and_review.md](handover_razer_batch_and_review.md) §8). Steps:
1. **Account**: one Tailscale account (free Personal plan), with two-factor login — anyone in it can
   reach the Razer.
2. **Mac**: install the Tailscale app (tailscale.com/download or the App Store), log in.
3. **Razer, Windows side** (not inside WSL — WSL's mirrored networking shares Windows' interfaces,
   Tailscale's included): install Tailscale for Windows, log in with the same account; in its settings
   turn on **Run unattended** (stays connected when nobody is logged in); in the admin console
   (login.tailscale.com) **disable key expiry** for the Razer (keys otherwise expire after months).
4. **Test from the Mac**: `tailscale status` lists both machines; `ssh vsekulic@minastirith` (Tailscale's
   name for the Razer, or its 100.x.y.z address) reaches WSL's sshd — the existing Windows firewall rule
   "WSL SSH" (port 22) should cover it; if not, allow port 22 on the Tailscale interface. Log the test on
   the Razer with `logrun`.
5. **Switch `ssh minastirith`** to it: in `~/.ssh/config`, `HostName minastirith` (or the 100.x address) in
   place of `192.168.3.10` — then it works at home and away alike. Keep a backup of the file first.
6. Check the lab/university network's rules on VPN-style software; Tailscale falls back to its relays
   (slower, still working) where UDP is blocked.
Cost: free (Personal plan). Alternative considered: Cloudflare Tunnel — free, but needs a domain on
Cloudflare (~$10/year), `cloudflared` on both machines and access rules, and routes everything through
Cloudflare; built for publishing services, overkill for two personal machines (VS chose Tailscale,
2026-09-28).

## Suggested order

First: 14 (Tailscale). During the copy: 5 (registry, directory listings only), 1 and 2 (code only), 9, and the decisions
3, 4, 6, 8, 10, 11. After the copy: the Razer track, then 1's re-check and 2's runs on the Razer.
After the batch: 7, then the crossreg runner.

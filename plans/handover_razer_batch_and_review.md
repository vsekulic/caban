# Handover: the Minian batch on the Razer, the YrA recompute, and the parked review branch

> ## ▶ UPDATE 2026-10-02 — start here (supersedes §3/§6/§8 where they differ)
>
> **Running, autonomous (Razer `minastirith`, `screen minian_batch`):** the Minian batch by experiment day
> (`scripts/run_minian_batch.py --by-day`, one stream, 6 workers, staged through the copier —
> [session_staging_copier_plan.md](session_staging_copier_plan.md)). 2026-10-01 10:36: 185 of 492 done, 5 failed
> (unreadable recordings G07 TFC_test_B and HC3, G14 TFC_test_B_1wk-borked, G18 TFC_cond HC1; G18 CNO2-borked
> failed in its YrA step — VS decides all), ~4–6 days left. Status: `ssh minastirith tail -f minian.log`
> (20-min `status` lines) and `~/minian_batch_status.md` (snapshots: [reports/minian_batch_status.md](../reports/minian_batch_status.md)).
> Stop switch `~/minian_stop`. **Rules (memory):** heavy compute only on the Razer (osgiliath crashed once); never
> run `caffeinate` (VS does); every Razer command via `~/bin/logrun`, but read `minian.log` with plain ssh (logrun
> echoes it into itself); queue follow-ups on the Razer, not as Mac-side waits.
>
> **Done since 2026-09-28:** review branch fully handled — `96137b9`→`e293ed5` (max_proj check), `5c3a1bd`→`044b4f1`
> (loader timing; forced the cache rebuild), `d2b74ce`→`081be37` (G05 motion recovery, hardened), `7fcac0f`
> rejected, `9015c23` (collaborator notice) **parked, not sent** — verify its numbers and regenerate
> `behaviour_params` before it goes out. Loader reads the recomputed YrA (`24fe3a9`): all **130** production
> recomputes carry a passing max_proj check on both drives (G05's two NaN-motion sessions recovered, `76013ae`;
> `NO_RECOMPUTED_YRA` empty). Sync MINIRAZER→MINISCOPE built (`scripts/sync_manifest.py` + `sync_to_miniscope.py`);
> 104 batch sessions synced, **~80 waiting** (needs MINISCOPE on osgiliath at home — light, ~30 GB).
> G09's last post-shock window accepted at 387 of 400 frames (`fc8e37e`, VS). Analysis cache rebuilt on
> osgiliath 2026-10-01 (`ds_cache.pkl`; old one `ds_cache-20260512-exported-YrA.pkl` kept until the write-up).
>
> **Figure 2 per-cell work — next steps** ([figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md)):
> VS re-ran the per-cell analyses (2026-10-01); the before/after comparison is in that plan, §2.5 ("Done
> 2026-10-02"): nulls hold; YrA shock Inh-vs-Ctl weakens (−0.097→−0.079); event-proximal omnibus P 0.151→0.095;
> Fig. 2 main unchanged except G09's corrected exposure (post-shock rate P 0.098→0.097, rate interaction F
> 1.544→1.533, P 0.237→0.240). **Next:** (1) VS re-runs "Initial checks → Sample traces" (Fig. 2c plots YrA);
> (2) `scientific-code-reviewer` + `statistics-checker` on the re-run (incl. the event-proximal move) — before any
> write-up; (3) write-up: new results section in [../docs/epoch_modulation.md](../docs/epoch_modulation.md) (old §R
> kept as superseded), the three number changes in [../docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md),
> Methods notes (recomputed YrA; G09 window). Panel → file → notebook-section map:
> [../docs/paper/figure_panel_map.md](../docs/paper/figure_panel_map.md). Reference outputs:
> `~/data/vsekulic/OF_test/plots/CURRENT-20260929-exported-YrA` (epoch_modulation from 2026-09-18).
> **Later:** the frame-rate decision (20 vs 19.76 fps, open item 13); crossreg registry/runner and new groupings
> after the batch ([crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md)).


Written 2026-09-28 ~10:30 JST, at the end of a very long chat (2026-09-26 → 28), for the chat that
continues it. **Read this whole file first, then the sources in §1, before doing anything.** Nothing
here replaces them; it tells you where the truth is and what exists only in the old conversation.

## 0. Rules VS set in this work — follow them exactly

- **One chat works in this repo at a time.** A second chat worked the open items in parallel on
  2026-09-27, rewrote shared code and plans, and made `ds_cache.pkl` unloadable (§6). This chat is now
  the only one.
- **Every command on the Razer goes through `~/bin/logrun "<title>" <command…>`** — including
  read-only checks and file copies (after an `scp`, log a line such as `logrun "copied X from the Mac"
  md5sum X`). It appends to `~/minian.log` and also shows the output; VS follows it with
  `ssh minastirith tail -f minian.log`. If something was done without it, backfill a note.
- **Before any delete or unregister, confirm on screen where the live copy is.** (On 2026-09-27 Claude
  told VS to delete a WSL disk image assuming another copy existed; it did not — the old Ubuntu was lost.)
- **Link every plan/report/markdown file** you mention: `[plans/x.md](plans/x.md)`, plus the section
  name for a section — never a bare "plan §13" (global rule in `~/.claude/CLAUDE.md`).
- **Give PowerShell/Windows commands as one copyable block, a blank line between commands.**
- Project rules in `CLAUDE.md` (repo root): no silent skips/fallbacks (hard-fail), all imports at top,
  no numeric-sequence names, no duplicated code, METHODS files, the reload snippet after code changes.
- VS's preferences seen here: `screen` (not tmux); exact, measured claims; commands explained before
  being run; plans committed and pushed; don't touch MINISCOPE's data without a clear reason.

## 1. Read these (in this order)

Plans (`plans/`):
1. [minian_open_items.md](minian_open_items.md) — the agenda: 12+ items, each self-contained.
2. [razer_runner_plan.md](razer_runner_plan.md) — the Razer: decisions (§2), steps (§3), and the
   **record (§5)**: every phase done so far with its numbers.
3. [minian_batch_runner_plan.md](minian_batch_runner_plan.md) — the runner itself; **§13** is the
   implementation record: G10 gate on the Mac, findings, every later decision (selection, YrA step,
   re-encode preset, the 50 plain-`minian/` sessions, …).
4. [yra_recompute_plan.md](yra_recompute_plan.md) — status line (open items), **§12** (G10 gate),
   **§14** (NaN motion), **§15** (unfinished recordings; the G21 timing gap, §15.1).
5. [local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) — the parent plan: §2 inventory,
   §3.1 (never re-run the 130 production sessions), §3.4, §5 (drives), §6 (gate), §8 (order).
6. [crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) — cross-registration by a registry;
   all output folders to be plain `minian`.
7. [yra_unit_alignment_plan.md](yra_unit_alignment_plan.md) — the load-time YrA aligner (implemented).
8. [session_staging_copier_plan.md](session_staging_copier_plan.md) — the copier (2026-09-28): every run
   works on a verified local copy; states, verification, recovery.

Code (`caban/`): `minian_runner.py` (the runner), `minian_gate.py` (production/gate comparison),
`session_queue.py` (scan, set-aside, queue), `yra_recompute.py` (YrA recompute), `sessions.py`
(analysis loader; `_align_YrA_to_S_units`, timestamps), and `scripts/run_minian_batch.py`.
Notebooks: `notebooks/run_minian_pipeline.ipynb` (runner), `notebooks/recompute_yra.ipynb`,
`notebooks/minian_pipeline_BASELINE.ipynb` (the protected template — read-only, md5 of cell sources
`5eb502c4a0252de26fc6170626d8ad00`). Env specs: `envs/`. Memory: `~/.claude/projects/-Users-vsekulic-code-caban/memory/`.

Live data to read (don't trust summaries — re-measure):
- On the Mac: each session's `Miniscope/minian_run/run.json`, `YrA_recompute.json` inside the output
  folder, `MINISCOPE/README.md` and `MINISCOPE/_provenance/`.
- On the Razer: `~/minian.log` (everything done there, timestamped), `~/gate_g05_tfc_cond.log`,
  `C:\Users\vlads\robocopy_minirazer*.log`.

## 2. The machines and drives

| | Mac (`osgiliath`, M5, 32 GB) | Razer Blade 16 (`minastirith`) |
|---|---|---|
| role | analysis; drives the Razer over SSH | the Minian batch, 24/7 |
| data | **MINISCOPE** (APFS, 5 TB, USB) at `/Volumes/MINISCOPE` | **MINIRAZER** (NTFS, 5 TB, was 4-MINISCOPE) = `E:` = `/mnt/e` in WSL |
| scratch | FUTROLA (`/Volumes/FUTROLA/minian_scratch`) | `~/minian_scratch` (WSL ext4 on the 2 TB `D:`) |
| envs | `caban`, `minian-native` (osx-arm64) | same names, linux-64 (`envs/*-linux.yml`) |

- Razer: i9-13950HX (24 cores/32 threads), 64 GB (WSL sees 54 GB), Windows 11 Home 25H2, WSL 2.7.10,
  Ubuntu 24.04 at `D:\WSL\Ubuntu`, mirrored networking, **Ethernet 192.168.3.10** (home LAN; the Mac is
  192.168.3.3). **`ssh minastirith`** from the Mac (`~/.ssh/config`, key `~/.ssh/id_ed25519`; backup of the
  old config `~/.ssh/config.bak-20260927`). Not reachable from outside the home LAN until Tailscale is set up (§8 step 0).
- **WSL keep-alive**: Windows scheduled task "WSL keep-alive" (at logon: `conhost --headless wsl.exe -d
  <distro> --exec /bin/sleep infinity`, restarts every minute). **SSH does NOT keep WSL alive** — only
  `wsl.exe` clients do (WSL powered off with an SSH session attached). After a Windows reboot someone
  must log in (auto sign-in not enabled — VS's call). Power: never sleep on AC, lid does nothing, USB
  selective suspend off, updates paused — [razer_runner_plan.md](razer_runner_plan.md) Phase 0 step 6.
- In WSL: Miniforge `~/miniforge3`; repos `~/code/caban` (branch `feat/yra-unit-alignment`),
  `~/code/minian_vsekulic` (`vsekulic_v4` @ `a3216ae`); kernel `minian-native` registered; helper
  scripts in `~/bin/` (`logrun`, `minirazer_copy.sh`, `minirazer_verify.sh`, `clear_failed_run.py`,
  `compare_to_production.py`). Windows programs from WSL need full paths (`/mnt/c/Windows/System32/…`)
  and `cd /mnt/c` first. The Mac share has a stored Windows credential (`cmdkey /add:192.168.3.3`).
- **MINIRAZER = MINISCOPE as of 2026-09-28 08:06** (robocopy; verified 08:28: 863,373 of 863,375 files
  identical in size+timestamp; the other 2 are the gate-run links below). **From then on only the Razer
  processes sessions** — never run the runner or YrA recompute against MINISCOPE on the Mac, or the
  copies diverge. Results go back to MINISCOPE through the sync script (not written yet, §4).
- Divergence already on MINIRAZER only: G05 `2021_08_30-TFC_cond/18_22_57-TFC_cond` now has the Razer
  production-check run (`minian/`, `minian_run/`, production's `minian.mp4` renamed `minian-ORIG.mp4`,
  `minian_set_aside.json`, `minian_run-failed-20260928T085013/` from the first attempt).

## 3. State of the work (2026-09-28 ~10:30)

**Runner / batch** ([minian_batch_runner_plan.md](minian_batch_runner_plan.md),
[razer_runner_plan.md](razer_runner_plan.md)):
- Queue = sessions needing Minian: never processed (442) + processed once but never cross-registered
  (50, re-run with old output set aside as `minian-ORIG`) = **492; 5 DONE, 487 to run** (dry run on the
  Razer, 2026-09-28). DONE: G06 `09_52_24-HC1` (Mac gate run), G13 `16_35_20-HC2` (Mac dry run), G05
  `track_day0/15_10_26-LT1`, `track_day1-tests/14_16_58-LT1`, `track_day2/14_41_34-LT1` (Mac, 2026-09-27).
  34 stubs and G09 `19_28_33-HC3` (partial output, all-NaN motion) are excluded and listed.
- Checks passed: **Mac** G10 TFC_cond vs production (C within 2e-6); **Razer** G05 TFC_cond vs production
  (570/570 units, same order, C 1.0e-5, S 6.9e-5, **A 1.2e-3** — larger than the Mac's 4e-6, no unit or
  trace affected; VS may want it localised).
- **Next on the Razer, in order**: (1) a pre-flight import check in the runner (the first Razer attempt
  died 10 min in on a missing `sk-video` hidden by the notebook's `%%capture` import cell); (2) one short
  session, then memory with 1/2/3 streams (G05 TFC_cond peaked 12.2 GB with 45 GB free); (3) ~~backfill
  the YrA recompute for the 3 G05 Mac sessions~~ (not needed: done on the Mac during their runs — razer plan §5);
  (4) the sync script; (5) the batch — `scripts/run_minian_batch.py` in `screen`, disjoint selections.
- Selection (VS's design): `LABELS` (regex in `<mouse>/<day>/<session>`), `MICE`, `SESSION_TYPES` — AND
  across, OR within, over the sessions needing Minian; `PATH` = exact sessions, processed or not, alone.

**YrA recompute** ([yra_recompute_plan.md](yra_recompute_plan.md)): **128 of 130** production sessions
recomputed, outputs `YrA_recomputed.zarr` + `YrA_recompute.json` inside each `minian_crossreg*` folder
(moved there from `~/cbp-db`, now deleted). Missing: the 2 G05 TFC_test_B NaN-motion sessions (§14.1).
G15/G21 TFC_test_B handled by `UNUSED_VIDEOS`. Runner sessions get it automatically (inside `minian/`).
**No analysis reads the recomputed YrA yet** (`sessions.py:739` loads the old `YrA.zarr`; switching is
open item 7). The notebook's own `YrA.zarr` can hold wrong units (`YrA.sel(unit_id=mask)`) — reproduced
in single runs (G10: 343/346 vs 344/347; G05 TFC_cond: 70 vs 71).

**Cross-registration**: planned only ([crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md)).

## 4. The sync script — ✅ built 2026-09-28 (`scripts/sync_manifest.py` + `scripts/sync_to_miniscope.py`; razer plan §5); the requirements it meets

Carry each finished session's results MINIRAZER → MINISCOPE: new `minian/` (incl. `YrA_recomputed.zarr`),
`minian_run/`, `minian.mp4`, `minian_mc.mp4`, `minian_set_aside.json`, any `minian_run-failed-*`. First
**replay each `minian_set_aside.json`'s renames on MINISCOPE** (`minian` → `minian-ORIG`, videos, …) —
rsync cannot express renames and would otherwise duplicate the old output. The Mac reads NTFS natively
(plug MINIRAZER into the Mac) or reads over the network from the Razer. Never copy back raw data.

Status of every session: [reports/minian_batch_status.md](../reports/minian_batch_status.md)
(`scripts/minian_status_report.py`).

## 5. Open items

[minian_open_items.md](minian_open_items.md) is the list: `max_proj` replay check (1), G05 NaN-motion
recovery (2), G21 timing gap (3, VS decision), new crossreg groupings (4, VS), groupings registry (5),
gate-run links + FUTROLA scratch cleanup (6, VS: G06 HC1 and G10 TFC_cond `minian_intermediate` symlinks,
6.6 + 25 GB on FUTROLA), switch analyses to recomputed YrA (7), videos drop 1 frame in 5 (8, VS), paper
Methods (9), stubs (10, VS), a second MINISCOPE copy (11, VS), housekeeping (12). Items 1–3 and a new 13
have **code already written on the review branch** (§6). Item 13 (only on the review branch's copy of the
list): the camera runs at 19.76 fps, not 20; G05 `2021_09_03-TFC_test_A/16_34_42-TFC_test_A` at 24.6 fps.

## 6. The parked review branch `review/timing-and-yra-fixes` — review before anything is merged

**Review status (2026-09-29):** `96137b9` merged as `e293ed5` and `5c3a1bd` as `044b4f1` (each
cherry-picked onto the copier code, conflicts resolved, `/code-review` + scientific review, fixes
applied; the `max_proj` check tested on a copy of G05 TFC_cond LT1). `7fcac0f` **rejected**, not
merged: it would run the recompute notebook on the Mac against MINISCOPE, which the sync now forbids
(MINISCOPE only receives verified copies from MINIRAZER). `9015c23` (collaborator notice) waits for VS
(sent? needed?) before its numbers are verified. `d2b74ce` (G05 NaN-motion recovery) is recall-only
and waits until after Figure 2's per-cell block
([figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md)).

The other chat's 5 commits, unpushed, moved off `feat/yra-unit-alignment` (reset to `7f217bd`) on
2026-09-27 evening. Review each with `/code-review` on its diff; the loader/timing ones also with the
`scientific-code-reviewer` and `statistics-checker` agents (VS's rule before any result is written up).

| commit | what | likely verdict |
|---|---|---|
| `96137b9` | `max_proj` replay check in the YrA recompute; `verify_all` re-checks old recomputes; notebook cell | promising (item 1); check tolerance (≤1 grey level) — the Razer replay is bit-exact, the Mac's ±1 at 5 px |
| `d2b74ce` | `reestimate_motion` in the runner (template cut after `max_proj`, output to scratch, `motion_reestimated.zarr` written only on an exact `max_proj` match); `motion_name` in the YrA recompute | promising (item 2); +198 lines in the runner — test on the Razer, not the Mac |
| `7fcac0f` | open-items reminder: "Run All `recompute_yra.ipynb` **on the Mac** after the copy" | **reject**: it would write MINISCOPE after the snapshot; run on the Razer instead |
| `5c3a1bd` | **loader change** (`sessions.py`, `loader.py`, `sections.py`, `session_queue.py`): map timestamp rows to C frames through the imaged rows (G21, G15, G09); G21 tone 3 excluded; `write_behaviour_params` from timestamps (19.76 fps); **old pickled sessions refuse to load → `ds_cache.pkl` must be rebuilt** | the finding is real; the change touches every analysis — careful review; do not rebuild the 9.3 GB cache casually |
| `9015c23` | `reports/collaborator_notice_behaviour_timing.md` + G21 old-vs-new comparison | a draft for other people: verify every number first; confirm nothing was sent |

`ds_cache.pkl` (`/Users/vsekulic/data/vsekulic/OF_test/npy_files/`, 9.3 GB, 12 May) is intact and loads
with the current branch; a Time Machine local snapshot of 2026-09-27 15:36 also exists.

## 7. Incidents and lessons (not in any plan)

- **FUTROLA unplugged mid-run** (2026-09-27) → G05 `track_day3/16_08_43-LT1` failed; cleared back to
  pending (`minian_run-failed-*` kept). Scratch drives must stay attached while a runner runs.
- **Safari at 34 GB** filled the Mac's swap (52 GB); memory is the Mac's limit, the reason for the Razer.
- **WSL move half-failed** (`wsl --manage --move`: image copied to D:, registry still pointing at C:) and
  the D: image was then deleted on Claude's wrong instruction; fresh Ubuntu installed.
- **Full 310-pin env solve stalled** 9 min on Linux; the working spec pins the ~45 numerically relevant
  packages (Mac versions) and lets the solver fill the rest — then dry-run with `mamba` first.
- **conda operations on `minian-native` reinstate bokeh 2.4.3**: re-run
  `envs/minian-native-linux-postinstall.sh` after any (it re-applies the server's jinja2 patch, md5-checked).
- Adding `netcdf4` on linux-64 would move opencv 4.5.0 → 4.5.3 — rejected; opencv stays 4.5.0.
- robocopy from the spinning MINISCOPE: `/MT:2` 111 MB/s vs `/MT:16` 66 MB/s.
- Operational: foreground `sleep` is blocked for Claude — wait with a single logged command on the Razer
  run in the background (`while screen -ls | grep -q X; do sleep 60; done`); an apostrophe inside a
  single-quoted `ssh minastirith '…'` heredoc breaks the Mac shell's quoting (happened twice); filter the
  harmless `param.Dimension` warning flood (`grep -v param.Dimension`); `ssh minastirith` + `logrun` output
  is already on screen, no need to `tail` the log.
- The other chat claimed the copy was done when it was at 47 %: **measure, don't trust summaries.**

## 8. First moves for the new chat

0. **Tailscale — done 2026-09-28**: both machines on VS's tailnet (`taildef906.ts.net`, MagicDNS);
   the Razer is `minastirith` (100.78.198.70), the Mac `osgiliath`; Run unattended on, key expiry off;
   SSH tested by name and address (logged). The SSH alias is now **`ssh minastirith`** (was `razer`;
   `~/.ssh/config.bak-20260928`). Setup as it was specified, for reference: set up Tailscale (VS, 2026-09-28), so the Razer can be driven from anywhere, not only the
   home LAN. Needs VS at both machines (~10 min). Steps (also in
   [minian_open_items.md](minian_open_items.md) item 14):
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
1. Read §1. Run `git status`/`git log -5` (branch `feat/yra-unit-alignment` at the latest pushed commit).
2. `ssh minastirith '~/bin/logrun "new chat: state check" bash -c "screen -ls; df -h /mnt/e; tail -5 ~/minian.log"'`
   (works from any network over Tailscale; checked 2026-09-28).
3. Ask VS which to take first: the Razer track (§3 "Next on the Razer") or the review branch (§6).

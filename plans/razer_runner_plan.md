# Plan: run the Minian batch on the Razer Blade 16, driven from the Mac

Status: **Phases 0–2 done; Phase 3 production check passed** (2026-09-28); next: one short
session and the 1–3-stream memory test, the sync script, then the batch. Every command on the
Razer is logged in `~/minian.log` there (`~/razer_setup.log` until 2026-09-28) (`ssh minastirith tail -f minian.log`).
Depends on: `plans/minian_batch_runner_plan.md` (the runner, the G10 gate, the 492-session queue),
`plans/local_minian_pipeline_plan.md` §5 (MINISCOPE is the only APFS copy; drive rules).

## 1. Why

The Mac runs one runner stream at a time: memory, not speed, is the limit (a 26-file session peaks
at ~13 GB RSS; the Mac has 32 GB shared with everything else, and swap reached 52 GB on 2026-09-27).
The Razer has **64 GB**, sits idle, and can run **2–3 streams 24/7**, freeing the Mac. Expected
throughput ≈ 2–3× the Mac's single stream: the ~490 remaining sessions in days, not 1–2 weeks.

## 2. Decisions so far (VS, 2026-09-27)

- **Driven from the Mac, in this Claude Code session**, over `ssh` into the Razer's WSL2 Ubuntu —
  not via VS Code Remote-SSH (a separate workspace, a separate chat). Batches run headless in `screen`
  (VS's preference over tmux) on the Razer and survive the Mac sleeping or VS Code disconnecting.
- **WSL2 (Linux), not native Windows.** Production ran on Linux (the RIKEN server); its env export
  `~/code/minian_vsekulic/minian_vsekulic_cbp-db.yaml` is a linux-64 build with exact pins, so the
  Razer can run production's own environment.
- **Working drive: 4-MINISCOPE, renamed `MINIRAZER`**, 5 TB, already NTFS, repurposed, plugged into the Razer. Verified 2026-09-27: every file
  on it is on MINISCOPE with the same size (10,738 research files + drive-root files); the
  consolidation had verified G26/G27 and small folders byte for byte and G24/G25/G30/G31 by a 2 %
  video sample (`MINISCOPE/_provenance/verify_4.txt`). VS accepts that; 4-MINISCOPE-BAK mirrors it.
- **NTFS, not exFAT** (journaled; the Mac reads NTFS natively). MINISCOPE (APFS) stays untouched and
  becomes the second copy of the raw data.
- **The Razer works on its own copy, not over the network** (VS, 2026-09-27). Considered: the Razer
  reading and writing MINISCOPE on the Mac over SMB, scratch only local — no copy, no sync, one
  source of truth. Rejected because the Mac would have to stay parked, awake and connected for the
  whole batch. The Mac is needed only once, for the ~10 h copy (Phase 2), and again for syncs.
- **Scratch on the Razer's internal 2 TB SSD** (1.5 TB free; the Steam library stays). A stream
  needs ≤ ~25 GB scratch, so 3 streams + the Linux system fit in < 200 GB.

## 3. Step by step

### Phase 0 — on the Razer, by VS (~30–45 min)

1. **What exists**: in PowerShell, `wsl --version` and `wsl -l -v`; Settings → System → About for
   the CPU and Windows version. Anything of value in the existing Ubuntu? If not, it is replaced.
2. **Put Ubuntu on the 2 TB SSD** (say it is `D:`). *Done 2026-09-27 as a fresh install: the
   in-place move failed half-way (disk image moved to `D:`, WSL's record left pointing at `C:`) and
   the moved image was then deleted on a wrong instruction from Claude — lesson: before any delete
   or unregister, confirm where the live copy is.* The options were:
   - *move the existing one* (WSL ≥ 2.3): `wsl --shutdown` then
     `wsl --manage Ubuntu --move D:\WSL\Ubuntu`; or
   - *fresh*: `wsl --unregister Ubuntu` (deletes it), then
     `wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu`; or, on older WSL,
     `wsl --export` to a `.tar` / `wsl --unregister` / `wsl --import Ubuntu D:\WSL\Ubuntu <tar>`.
3. **WSL settings** — the WSL Settings app edits the same `C:\Users\<you>\.wslconfig`:
   memory **56 GB** (leaves ~8 GB for Windows); processors **all**; swap **14–16 GB** (a cushion,
   not a working area), swap file **`D:\WSL\swap.vhdx`** (off the system drive); networking mode
   **mirrored** (Windows 11 22H2+; the Mac reaches Linux at the Razer's own IP); auto memory
   reclaim **gradual**; sparse VHD **on**. Then `wsl --shutdown`. As a file:
   ```
   [wsl2]
   networkingMode=mirrored
   memory=56GB
   swap=16GB
   swapFile=D:\\WSL\\swap.vhdx
   [experimental]
   autoMemoryReclaim=gradual
   sparseVhd=true
   ```
4. **systemd and SSH** inside Ubuntu: `printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf`,
   `wsl --shutdown`, reopen, `sudo apt update && sudo apt install -y openssh-server screen`,
   `sudo systemctl enable --now ssh`.
5. **Firewall** (PowerShell, admin):
   `New-NetFirewallRule -DisplayName "WSL SSH" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow`.
6. **24/7 settings** (administrator PowerShell, except where a Settings click is given):
   - Never sleep or hibernate on AC, and never power down idle USB devices (USB selective
     suspend — MINIRAZER is an external drive in constant use):
     ```
     powercfg /change standby-timeout-ac 0
     powercfg /change hibernate-timeout-ac 0
     powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
     powercfg /setactive SCHEME_CURRENT
     ```
   - Closing the lid does nothing on AC (or Control Panel → Power Options → *Choose what closing
     the lid does* → *When plugged in: Do nothing*):
     ```
     powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
     powercfg /setactive SCHEME_CURRENT
     ```
   - No update restarts: Settings → Windows Update → *Pause updates* for the longest period
     (usually 5 weeks); *Advanced options* → "Get me up to date" **off**, *Active hours* as wide
     as allowed.
   - Network adapter stays awake: Device Manager → *Network adapters* → the Ethernet adapter →
     *Properties* → *Power Management* → untick "Allow the computer to turn off this device to
     save power" (an idle-dropped link does not stop a batch, but cuts SSH).
   - On the charger throughout; any Razer Synapse battery-care mode balanced or performance.
   - Check: `powercfg /query SCHEME_CURRENT SUB_SLEEP` — the AC values read `0x00000000`.
7. **Send**: the Razer's Ethernet IP, the Ubuntu username, CPU, Windows version, `wsl --version`.

### Phase 1 — environment, by Claude over SSH (~1–2 h)

1. SSH key from the Mac; check connection, memory and disk inside WSL.
2. Keep WSL alive with no console open (a Windows scheduled task at logon, if needed — WSL can stop
   an idle distro); confirm `sshd` survives a Windows reboot.
3. Miniforge; `minian-native` from the server export (drop only packages absent on conda-forge
   linux-64 today, recorded); ffmpeg; register the kernel.
4. A `caban` env for the runner and YrA recompute (the Mac's `caban` env has no env file: built
   from what `minian_runner`, `minian_gate`, `yra_recompute` import, with papermill).
5. Clone `caban` (**needs VS's OK to push `feat/yra-unit-alignment`, 41 commits ahead of GitHub**)
   and the Minian fork (`vsekulic_v4`, up to date on GitHub).

### Phase 2 — the working drive (~10–12 h unattended)

1. 4-MINISCOPE into the Razer (its own USB port, no hub — `local_minian_pipeline_plan.md` §5.3);
   quick-format NTFS, label **`MINIRAZER`**.
2. The Mac shares MINISCOPE over SMB, read-only (System Settings → General → Sharing → File
   Sharing). The Razer pulls it with `robocopy /MIR /MT:16` (Windows-native: fastest for ~830k
   files) — ~3.9 TB at gigabit (~110 MB/s): ~10 h.
3. Verify: file list and sizes (the standard VS set for 4-MINISCOPE), by `robocopy /L` or `rsync -n`.
4. **From the snapshot on, only the Razer processes sessions** — the Mac's runner stays stopped, so
   the two copies cannot diverge. (The Mac's run so far: 3 G05 sessions DONE; G05 track_day3 LT1
   cleared back to pending.)

### Phase 3 — runner changes and checks (~2–3 h, much of it waiting)

1. Paths as settings: data root (e.g. `/mnt/e/SSTCa2`), fork directory, scratch root — no
   `/Volumes/...` assumptions.
2. **Scratch without a symlink**: NTFS through WSL's `/mnt` cannot hold one reliably; instead the
   runner edits the parameter cell's `intpath` to the scratch folder (one more recorded edit).
   `minian_gate` then reads intermediates from the scratch root, not through the session link.
3. **Gate on the Razer**: G10 TFC_cond via `PATH`, compared with production by `minian_gate`
   (as on the Mac: identical `motion`/`max_proj`, same 799 units, ~1e-5). A different CPU and BLAS
   may differ at solver level — the gate says whether it matters.
4. Dry run on one short session; check memory with 1, 2, 3 streams.

### Phase 4 — the batch

1. 2–3 streams, each in its own `screen` session, each a disjoint selection (by `MICE` / `SESSION_TYPES`), `N_WORKERS` 4–6.
2. Monitoring from the Mac through this session: queue status, per-session memory and wall time.
3. **Results back to MINISCOPE**, from the Mac, periodically: the Mac mounts the Razer's share
   (or reads 4-MINISCOPE directly once it is back on the Mac). A sync script first applies each
   session's `minian_set_aside.json` renames (`minian` → `minian-ORIG`, …) on MINISCOPE, then
   copies the new `minian/`, `minian_run/` and videos — rsync alone cannot express those renames.

## 4. Open

- Keeping WSL alive unattended — checked in Phase 1.
- Whether the Razer gate passes as exactly as the Mac's; if not, what tolerance VS accepts.

## 5. Record

**Phase 0** (VS, 2026-09-27): Razer `minastirith`, i9-13950HX (24 cores / 32 threads), 64 GB, Windows 11
Home 25H2, WSL 2.7.10; fresh Ubuntu 24.04 on `D:\WSL\Ubuntu`; mirrored networking; Ethernet
192.168.3.10. The Mac reaches it as `ssh minastirith` (`~/.ssh/config`, key `~/.ssh/id_ed25519`).

**Phase 1** (Claude over SSH, 2026-09-27), all logged by `~/bin/logrun`:
- Miniforge (conda 26.7.2). **`minian-native`**: `envs/minian-native-linux.yml` pins the 45 numerically
  relevant packages at the Mac `minian-native` versions (python 3.8.15, numpy 1.20.2, dask 2021.2.0,
  xarray 0.17.0, opencv 4.5.0, cvxpy 1.2.1, pyfftw 0.12.0, pymetis 2020.1, …; OpenBLAS) and leaves the
  rest to the solver — pinning all 310 Mac versions stalled the solver for 9 min and was stopped. Then
  `envs/minian-native-linux-postinstall.sh`: the PyPI set at the Mac versions and the server's jinja2
  patch, both patched files md5-identical to the server's.
- **`caban`**: `envs/caban-runner-linux.yml`, the runner's imports at the Mac `caban` versions.
- Repos cloned from GitHub into `~/code/` (caban `feat/yra-unit-alignment`, the fork `vsekulic_v4`
  at `a3216ae`); the template re-made read-only and accepted by `load_template` (md5 `5eb502c4…`).
- Smoke test: papermill (caban) → `minian-native` kernel, cwd the fork → `minian` imported from the fork.
- **Keep-alive**: WSL powered itself off ~1 min after the last WSL window closed, **with an SSH
  session connected** — SSH does not keep WSL alive, only `wsl.exe` clients do. Fix (VS): Windows
  scheduled task **"WSL keep-alive"** — at logon, `conhost.exe --headless wsl.exe -d <distro> --exec
  /bin/sleep infinity`, restart every minute if it stops. Test: 3 min with no window and no SSH, WSL
  stayed up. Caveat: it starts at *logon*, so after a Windows reboot someone must log in (or automatic
  sign-in is enabled — VS's call).

**Phase 2 started** 2026-09-27 12:22: `~/bin/minirazer_copy.sh` in `screen` `minirazer_copy` —
`robocopy \\192.168.3.3\MINISCOPE E:\ /E /COPY:DAT /DCOPY:DAT /MT:2` (tests: 66 MB/s at `/MT:16`,
**111 MB/s at `/MT:2`** — the source is a spinning USB disk), macOS housekeeping excluded, robocopy's
log `C:\Users\vlads\robocopy_minirazer.log`; progress every 10 min in `razer_setup.log` (now `~/minian.log`). The Mac is
kept awake by `caffeinate`; the share is read-only to Windows through a stored credential (`cmdkey`).

**Phase 3 code** (2026-09-27, on the Mac; end-to-end test waits for the copy — a Mac run now would
change MINISCOPE mid-copy):
- Scratch without a link, on every machine: `set_aside_minian_output(..., link_scratch=False)` creates
  the scratch folder only, and `build_run_notebook(..., scratch=)` sets the parameter cell's `intpath`
  to it (a recorded edit; the notebook's `MINIAN_INTERMEDIATE` follows). The kernel's final `intpath`
  is checked. Runs from before keep their links (`run_scratch` tells them apart); `minian_gate` reads
  the scratch from `run.json`.
- Defaults per machine: data roots MINISCOPE → MINIRAZER (`/mnt/e/SSTCa2`) → backups, or
  `$CABAN_DATA_ROOTS`; scratch FUTROLA on macOS, `~/minian_scratch` on Linux, or `$CABAN_SCRATCH_ROOT`.
- `scripts/run_minian_batch.py`: the notebook's Choose + Run cells as a command for `screen`
  (unattended; `--dry-run` lists the queue).

**Phase 2 done** 2026-09-28. Copy finished 08:06 (13 h 7 min, avg 91 MB/s): 3.901 TB, 859,083 files;
robocopy exit 11 only because of three known items — the gate-run `minian_intermediate` symlinks in
G06 `09_52_24-HC1` and G10 `16_32_14-TFC_cond` (links to FUTROLA; not followable over SMB) and the
macOS folder `.DocumentRevisions-V100-bad-1`. **Verification** (`~/bin/minirazer_verify.sh`,
robocopy `/L` with the copy's settings, 3 min): of 863,375 files, **863,373 identical in size and
timestamp, 0 mismatched or missing**; the 2 listed are those two links. From here on the Mac and
MINISCOPE are not needed by the Razer; only the Razer processes sessions.

**Phase 3.2 — production check on the Razer: passed** (2026-09-28). G05 `2021_08_30-TFC_cond/18_22_57-TFC_cond`
(26 videos, production, never runner-processed) instead of G10: on MINIRAZER G10 already carries the Mac's
gate re-run, whose set-aside renames a second run would collide with. Compared with production
(`minian_crossreg1_crossreg2_crossreg4_crossreg6_crossreg7`) by `minian_gate` and array by array
(`~/bin/compare_to_production.py` on the Razer): `motion` and `max_proj` identical; 570 = 570 units, same
`unit_id`s in the same order, all matched, median corr(C) 1.000; largest relative differences C 1.0e-5,
S 6.9e-5, A 1.2e-3 (A: the Mac's G10 gave 4e-6 — plausibly Intel vs Apple-silicon solver paths; no unit,
match or trace affected); the notebook's `YrA` again holds unit 70 where C has 71, as production's did.
The YrA step's replayed movie equals the run's own `Y_fm_chk` bit for bit (the Mac: ±1 at 5 pixels).
Wall 1.44 h (notebook 66 min, YrA 7 min, re-encode 13 min); peak RSS 12.2 GB, 45 GB still free.
- **First attempt failed**: `minian-native` lacked `sk-video` (my reduced spec dropped it); the notebook's
  import cell runs under `%%capture`, which hid the ImportError until cell 87. Fixed by adding the Mac's
  missing packages at the Mac versions (`envs/minian-native-linux.yml`), refusing `netcdf4`, which would
  have moved opencv to 4.5.3. **Pre-flight import check added** (2026-09-28): `minian_runner.preflight_kernel`
  makes every top-level import of the template and of the appended parameters cell (cell magics such as
  `%%capture` stripped), reads the recorded packages' versions and refuses a `minian` not from the fork — in a
  fresh `minian-native` kernel started in the fork, before the session is touched (~5 s; recorded in
  `run.json` as `preflight`). `run_minian_batch.py` runs it once before the queue. Tested on the Mac and the Razer (and a Razer dry run: 492 queued, 5 DONE, 487 to run, unchanged): passes on
  the real template; a missing package hidden under `%%capture` and a `minian` outside the fork both fail.
- ~~**Also found**: the 3 G05 sessions run on the Mac on 2026-09-27 have no recomputed `YrA`~~ —
  **wrong** (corrected 2026-09-28): all three were recomputed on the Mac during their runs (sidecars written
  2026-09-27 09:21–10:14 JST, 50-frame movie check; track_day1-tests LT1 differs by ≤ 1 grey level at 1,163
  pixels, the Mac's cv2). The Phase 4 "backfill" found them complete and changed nothing.

**Phase 3.4 — one short session, then 3 streams** (2026-09-28, driven from the Mac over Tailscale):
- *One stream*: G05 `track_day0/15_44_41-HC2` (6 videos, 5,201 frames), 6 workers: done in 19.5 min (notebook
  15 min, YrA 1 min, re-encode 3.3 min), peak RSS 6.6 GB, ≥ 50.9 GB always available; 167 units; the YrA
  replay equals the run's `Y_fm_chk` bit for bit; non-overlapping units r = 1.0 against the notebook's `YrA`.
- *Three streams at once* (6 workers each), the three longest pending LT sessions:

  | session | videos / frames | units | notebook | total | peak RSS | result |
  |---|---|---|---|---|---|---|
  | G11 `2021_11_25-TFC_test_B/16_47_19-LT1` (old plain `minian/` → `minian-ORIG`) | 19 / 18,281 | 748 | 2.40 h | 2.77 h | 20.9 GB | done |
  | G11 `2021_11_30-TFC_test_B_1wk/17_25_43-LT1` | 18 / 17,613 | 911 | 2.47 h | 2.82 h | 11.8 GB | done |
  | G14 `2022_01_15-TFC_test-A/17_14_14-LT1` | 18 / 17,683 | 403 | 2.15 h | — | 10.3 GB | **failed in the YrA step** |

  System-wide (`~/streams_3b.csv`, 1-min samples): at most 25.8 GB used, ≥ 29.2 GB available; load 10–16 of
  32 threads. **Memory is not the limit; throughput is**: by the single run's rate (~3.4 min per 1,000
  frames), each notebook would take ~50–60 min alone, so 3 streams ran each ~2.5× slower — about 1.2× the
  throughput of one stream. CPU was not saturated; the shared input (`/mnt/e`, the USB drive through WSL)
  or the scratch disk is the likely bottleneck — not yet measured (drvfs reads do not appear in `/proc/diskstats`).
- *The G14 failure*: `yra_recompute.load_avi_ffmpeg` got 3 frames too many from `17.avi` (253,589,504
  bytes = 686 frames; header and the notebook's `C`: 683) and stopped, as designed. Not reproduced: the
  exact command gives 683 every time, in both envs' ffmpeg (8.1.2, 4.3.2), and 3 concurrent passes over all
  18 G14 videos under the 2 remaining streams gave identical md5s and frame counts. Cause open. G14 left
  as failed, scratch kept (a full replay-vs-`Y_fm_chk` check is still possible); VS to decide before
  `resume_after_notebook`.

**Phase 3.4b — why 3 streams did not help** (2026-09-28, all read-only, scripts in `~/bin/` on the Razer):
- *Per-cell times* (papermill cell metadata; `cell_timings.py`), per 1,000 frames, single run vs the three
  concurrent ones: 148 s vs 457 s (×3.1) — i.e. three streams did the work of one. Not only the HDD cells:
  loading the videos (cell 32) ×5.3, but motion estimation (cell 82, scratch + CPU only) ×3.4, spatial
  updates ×2.3–2.9 (temporal updates ×13–15 also scale with the 4–5× larger unit count).
- *MINIRAZER* is a **WD My Passport HDD on USB** (`Get-PhysicalDisk`; scratch `D:` = Samsung 990 PRO NVMe).
  Sequential reads through `/mnt/e` (`hdd_read_test.sh`, sessions not read since the copy): **1 reader
  80 MB/s; 3 readers 19 MB/s each, 58 MB/s in total** — seek contention.
- *CPU* (`cpu_scaling.py`: the notebook's median-5 + tophat-15 on 608×608 frames, one thread per process,
  no disk): 1 process 209–215 frames/s; **6 processes 1,000 in total; 18 processes 860–1,050 in total**.
  Under 18: WSL steal 0 %, hypervisor logical-processor run time ~60 % (all scheduled), clock ~2.6 GHz (120 %
  of the 2.2 GHz base; `cpu_sustained.sh`); a second consecutive pass fell to 710. Windows power plan
  Balanced with the "Best performance" overlay. So **the CPU's total throughput is power-limited at about
  6 busy cores**: more processes lower the clock and land on E-cores. One stream at 6 workers already uses it.
- Consequence: **run one stream**, 6 workers. Throughput can only rise by raising the CPU's power budget
  (a Razer/BIOS performance mode — VS, at the machine) — not by more streams.

**G14's bad read, investigated** (2026-09-28; scripts in `~/bin/` on the Razer, all logged):
- *The notebook's output is sound*: replaying the motion-corrected movie from the `.avi` files
  (`yra_recompute`, the YrA step's own code; `g14_full_replay_check.py`) equals the run's saved `Y_fm_chk`
  in **all 17,683 frames (0 pixels differ)**, and the replay's max over frames equals the notebook's
  `max_proj` exactly. So every notebook read of the videos was right; the 686-frame read (13:32) was in the
  YrA step only, and its byte-count check stopped it. The whole replay took 108 s with nothing else running.
- *Not reproduced*: 5 later full reads of `17.avi` correct; then 20 min of 3 decode loops (the recompute's
  exact ffmpeg command, `-v warning`) + 2 background readers (`decode_stress.py`): 899 decodes, 379 of
  `17.avi`, all correct, no ffmpeg warning — though repeated decodes of the same 3 files were likely served
  from the Windows file cache, so the USB read path was only lightly exercised.
- *Windows System log* (`hw_errors.ps1`, `root_port_1b.ps1`, `nvme_map.ps1`): **one `disk` event 11,
  "controller error on \Device\Harddisk2" = the WD My Passport (MINIRAZER), at 14:33:00** — during the full
  replay above, which still came out exact (the read was retried). Nothing logged near 13:32. Separately,
  **44 WHEA-17 corrected PCIe errors** (13:00–15:00 only, none since yesterday before that) on root port
  00:1B.0 (Intel 7A44) = the link to the CA6 NVMe, **`C:`, the Windows system drive** — not scratch (`D:`,
  Samsung 990 PRO on 1D) and not MINIRAZER; corrected, off the data path, worth watching. (Also: the Razer
  Chroma Stream Server service crashes every 5 min, 335 times today — noise.)
- *Conclusion*: cause **not established**. Best supported: a transient fault on MINIRAZER's USB read path,
  which demonstrably has them (the 14:33 controller error), surfacing under the 3-stream load — but no
  event ties it to 13:32. The checks that exist catch a changed frame count; a read that changed pixel
  values without changing the count would pass them, except for the replay-vs-`Y_fm_chk` comparison (50
  frames today).
- *Decision* (VS, 2026-09-28): don't dig further — watch the drive. Guard every run with the
  **full-frame check** (the YrA step compares the replayed movie with the notebook's `Y_fm_chk` in
  every frame and refuses unless equal) and run on local copies through **the copier**:
  [session_staging_copier_plan.md](session_staging_copier_plan.md).

**Phase 4 — the batch started** (2026-09-28 16:04, `screen minian_batch`, `~/bin/minian_batch.sh`): one
stream, 6 workers, staged through the copier, in the priority order of
[local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §8 — TFC, LT, CNO, HC (one
`run_minian_batch.py --session-types <type>` after the other). First, the **YrA backfill** of the three G05
sessions run on the Mac (`~/bin/backfill_yra.py`): a no-op — their recomputes already existed (see the
correction under Phase 3.2); the numbers it printed (462, 180, 202 units; corr(C, YrA) median 0.32 / 0.56 /
0.49) are those recomputes'. Queue at launch: 482 pending (after the 3.4 runs, G14 and the
copier test). Still to do alongside: the sync script MINIRAZER → MINISCOPE; the power-mode check (VS).

**Phase 4 — power settings and the worker count settled** (2026-09-28 evening). VS confirmed Windows "Best
performance" and Razer Synapse "High performance" (on AC). With the batch running (one stream, 6 workers) the
work sits on the 8 P-cores, one thread per core, at ~165 % of nominal (~3.6 GHz; per-logical-processor
counters, `per_lp.ps1`) — no efficiency-mode (EcoQoS) problem. A clean test at a session boundary (stop
switch, `cpu_ab_test.sh`; batch paused 21:05:56–21:08:09, nothing lost): the preprocessing ops at **6
processes 920 frames/s in total, 12 → 701, 18 → 966**, every process at CPU time / wall 1.00 and 12.4 / 18.6
logical processors busy. More busy cores lower every core's clock (sustained power limit), so the total does
not rise. **One stream, 6 workers stays.** (The per-LP clock split by P/E in this run is not trustworthy —
Windows' `Processor Information` numbering evidently differs from the hypervisor's; only the frames/s are.)
The earlier "power-limited at ~6 cores" (Phase 3.4b) stands in substance; its clock figures were averages
over idle logical processors too.

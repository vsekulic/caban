# Plan: run the Minian batch on the Razer Blade 16, driven from the Mac

Status: **planning**, written 2026-09-27. Nothing done on the Razer yet.
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
3. **`C:\Users\<you>\.wslconfig`**:
   ```
   [wsl2]
   networkingMode=mirrored
   memory=56GB
   ```
   (`mirrored` needs Windows 11 22H2+; it lets the Mac reach Linux at the Razer's own IP.)
4. **systemd and SSH** inside Ubuntu: `printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf`,
   `wsl --shutdown`, reopen, `sudo apt update && sudo apt install -y openssh-server screen`,
   `sudo systemctl enable --now ssh`.
5. **Firewall** (PowerShell, admin):
   `New-NetFirewallRule -DisplayName "WSL SSH" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow`.
6. **24/7 settings**: never sleep when plugged in; closing the lid does nothing; pause Windows
   Update restarts.
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

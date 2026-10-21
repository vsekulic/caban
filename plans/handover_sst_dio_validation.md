# Handover (2026-10-21): SST DIO-GCaMP6f validation — a separate chat, in parallel

For the chat that validates the DREADDs on SST cells. **Read this whole file first**, then
[sst_dio_dreadd_validation_plan.md](sst_dio_dreadd_validation_plan.md) (the stub plan, cohort audit
and open questions) and [figure2_redesign_plan.md](figure2_redesign_plan.md) §3, §4.2 and §7
(why this is now panel **d** of Figure 2 v2).

## 0. Rules (VS) — follow exactly (also in memory)

- **This chat runs in parallel with the Figure 2 v2 chat**, by VS's choice. A parallel chat broke
  `ds_cache.pkl` loading in September, so:
  - **Work in your own git worktree on your own branch** (`EnterWorktree`, branch e.g.
    `feat/sst-dio-validation`). Never commit in the main checkout (`feat/yra-unit-alignment`).
  - **Do not edit shared files**: `caban/sections.py`, `caban/loader.py`, `caban/sessions.py`,
    `caban/figure2_composite.py`, `caban/sp_rates_lmm.py`, `caban/epoch_modulation.py`,
    `run_pipeline.ipynb`, and never write `ds_cache.pkl`. New code goes in new modules
    (e.g. `caban/sst_dio_validation.py`) and a new notebook. If a shared file must change, write
    what is needed in your plan and leave it for VS to hand to the main chat.
  - `caban/minian_runner.py` / `caban/session_queue.py`: the main chat is not working in them now.
    Changes there are allowed but small and additive (e.g. extending the mouse scope), on your
    branch; say so in the commit message.
  - Razer `screen` names start with `sst_` so they cannot collide with the main chat's.
  - **On the Razer, use your own clone**, e.g. `git clone ~/code/caban ~/code/caban_sst` then
    `git checkout -b feat/sst-dio-validation` there (push your branch from it). **Never `git checkout`
    or commit in `~/code/caban` on the Razer**: the main chat's runs `git pull` there on
    `feat/yra-unit-alignment`, and a branch switch would silently run the other chat's code.
  - The Razer `caban` conda env is shared: install nothing into it without telling VS; if you need a
    package, make your own env (e.g. `conda create --clone caban -n caban_sst`).
  - Also off limits (main chat's, added 2026-10-21): `caban/homecage_cno.py`, `caban/shock_motion.py`,
    `scripts/run_homecage_cno.py`, `scripts/export_tfc_*`.
  - Analyses of Minian outputs run wholly on the Razer (VS, 2026-10-21); plots are rsynced into the
    Mac's `plots/CURRENT/<your folder>`.
- **Never throw away data.** No rule may drop trials, windows, frames, cells, sessions or mice;
  any exclusion goes to VS as a decision, keep-the-data option first.
- Every Razer command through `~/bin/logrun "<title>" …`; read `~/minian.log` with plain
  `ssh minastirith 'grep …'`. Heavy compute on the Razer (minastirith), never the Mac. Caches on
  FUTROLA (`/Volumes/FUTROLA/caban_cache/`), plots on the Mac, MINISCOPE read-only (writes only
  through the verified sync). Never `caffeinate`; tell VS when the Mac must stay awake.
- No silent fallbacks (hard-fail); imports at the top; descriptive names. Display order Ctl, Exc,
  Inh — here only Exc (hM3D) and Inh (hM4D). Pastel box-and-strip style; Nature style for any
  paper panel (`nature-figures` skill).
- Every major analysis gets a METHODS template in `analysis_methods_templates/`, copied into its
  plot folder at runtime. `scientific-code-reviewer` and `statistics-checker` before any result is
  written up.
- After launching anything on the Razer, check it within ~20 min. Don't edit a notebook VS has
  open; give a cell to paste.

## 1. State (audited 2026-10-20 on the Razer)

- **No Minian output exists for any DIO mouse.** Raw video is on the Razer at
  `/mnt/e/SSTCa2/<mouse>/` (and on MINISCOPE). The Razer batch covered G05–G21 only:
  `caban/minian_runner.py` `MOUSE_RANGE = ("G05", "G21")`; 520 of 520 batch sessions are done and
  synced, so the Razer is free.
- **Cohort:** G22-ST875-hM3D-DIOGC, G23-ST856-hM3D-DIOGC (2023-03), G26-ST894-hM4D-DIOGC,
  G27-ST895-hM4D-DIOGC (2023-05). No mCherry-DIO control. (G03/G04, the 2021 pilots, are in the
  stub plan §2.1 as context.)
- **Conditioning days** (the sessions that matter first):
  - G22 `2023_03_20-TFC_cond`: HC1, LT1, HC2, **CNO**, LT2, HC3, TFC1, TFC2 (conditioning split in
    two — find out why), HC4.
  - G23 `2023_03_20-TFC_cond`: HC1, LT1, HC2, **CNO**, LT2, HC3, TFC_cond, HC4.
  - G26 `2023_05_22-TFC_cond`: HC1, LT1, HC2, **CNO1**, **CNO2**, LT2, HC3,
    TFC_cond-borked (shocker off; restarted), TFC_cond, HC4.
  - G27 `2023_05_22-TFC_cond`: HC1, LT1, HC2, **CNO1**, **CNO2**, LT2, HC3, TFC_cond, HC4.
  - Each G22/G23 day also has ~15–18 bare-timestamp folders before the named sessions (focus or
    setup recordings, one .avi each); check before queueing anything. G22 `2023_03_09-HC2` is an
    electrowetting-lens focus sweep.
- The main cohort's pre/post-CNO analysis of **pyramidal** cells (Figure 2 v2 panel e) is the main
  chat's job; this chat is the **SST** cells only.

## 2. What to do, in order

1. **Parameter gate.** The Minian parameters are one frozen set tuned on dense CA1 pyramidal
   layers ([local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §3.2). SST cells are
   sparse, larger and fire differently. Run one conditioning-day session per mouse (e.g. HC2, a
   short homecage) with the frozen set (`scripts/run_minian_batch.py --path … --keep-scratch`),
   inspect seeds, footprints, merges and traces, and only then decide whether any parameter
   changes. Any change is a documented, per-cohort parameter set, not an edit of the frozen one.
   Present the gate results to VS before the batch.
2. **Queue.** Extend the runner's scope to the DIO mice (small, additive change on your branch),
   label the sessions (the bare-timestamp folders must be listed and decided, not skipped
   silently), dry-run (`--dry-run`), then run the conditioning-day sessions first.
3. **Analysis design, then `statistics-checker` before running it.** Within cell, within animal:
   pre-CNO (HC1/HC2, and LT1 in context) vs post-CNO (CNO / CNO1 → CNO2 → LT2 → HC3). Expected:
   hM3D raises SST event rate (and/or amplitude), hM4D lowers it. With n = 2 per DREADD and no
   control, it is a demonstration shown per animal and per cell, with the pre-injection homecage
   drift (HC1 → HC2) as the only bound on time and handling effects. Cross-registration within the
   day is needed for "within cell"; session-level per-animal means are the fallback that needs none.
4. **Panel.** Figure 2 v2 panel **d**, Nature style, drawn at its box size:
   `caban.figure2_composite.panel_figsize('d', caban.figure2_composite.LAYOUT_V2_MM)`
   (123 × 36.8 mm). Write it into your own plots folder under `PLOTS_DIR` and tell VS the path; the
   main chat moves `d` from `PENDING_V2` to `PANEL_FILES_V2` and adds its section to
   `FIGURE2_V2_PANEL_SECTIONS` in `caban/sections.py` (a shared file).

## 3. Open questions to settle with VS early

The stub plan §3 lists them: CNMF-E parameters; the primary comparison; no DIO-mCherry control;
n = 2 per DREADD; whether G03/G04 are usable; whether cross-registration is needed. Add: what G22's
TFC1/TFC2 split is, and the injection time on each day (CNO1 starts right after injection per
G26's `notes.txt`).

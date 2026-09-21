# Plan: recompute YrA from the existing footprints instead of re-running Minian

Status: **approach approved 2026-09-21** (§10); execution not started.
Written 2026-09-21.
Depends on: `plans/yra_unit_alignment_plan.md` (implemented, commit `485f296`) — the load-time
aligner stays regardless, as a guard; this plan is about removing the need for it.
Out of scope: freeze scores / `freeze_data`; any change to crossreg mappings or cell identity.

## 1. Why this exists

Two questions forced it:

1. **Conditioning YrA is row-sheared against S/C** in all 17 mice, and in 8 mice the unit id sets
   differ by 1–4 cells (14 total). Currently handled in memory by `_align_YrA_to_S_units`.
2. **Recall sessions have essentially no YrA at all**, so no YrA-based recall analysis is possible.

## 2. Evidence gathered 2026-09-21

### 2.1 The shear is in the Minian output, not in the caching

The backup drives (`/Volumes/1a-MINISCOPE-BAK`, `/Volumes/1b-MINISCOPE-BAK`) file conditioning
`YrA.zarr` at the `Miniscope/` level; the CBP server (`/Volumes/vsekulic/data/vsekulic/OF_test`)
files it *inside* `minian_crossreg*` next to `A/C/S`. **The two are the same export**: the server's
`YrA.zarr` `unit_id` is byte-identical to the cached `YrA_idx` in all 17 mice, with identical shear
counts (G05 36, G08 124, G10 127, G18 325, G20 374…). Re-reading from either location changes
nothing.

### 2.2 The shear is a *sort*, and the run was near-deterministic

History (from VS): Minian was first run over everything producing `A/C/S`; roughly a year later the
notebook was edited to also save `YrA` and everything was re-run. Measured:

| | result |
|---|---|
| `S_idx` sorted ascending? | **never** (0/17) |
| `YrA_idx` sorted ascending? | **always** (17/17) |
| id sets equal | 9/17 |
| of those 9, `YrA_idx == sorted(S_idx)` | **9/9, exactly** |

So for **G06, G07, G09, G11, G13, G15, G16, G19, G21** the second run reproduced the cell set
*exactly* and the only difference is that YrA was written sorted. For **G05, G08, G10, G12, G14,
G17, G18, G20** the re-run additionally diverged by 1–4 marginal cells (14 in, 14 out). VS's
recollection that an early G05 re-run "looked the same" is close but G05 is in fact one of the 8
divergent mice (1 unit).

This matters: the divergence is small and confined to threshold-marginal cells, which is consistent
with the same parameters and an essentially deterministic pipeline, not a parameter change.

### 2.3 `A.zarr` is correctly aligned

`A.zarr`'s `unit_id` **equals `S.zarr`'s exactly** (checked G05, G08, G10, G13, G18, G20). Cell
identity was never in question — only the YrA export step reordered (and slightly re-set) the units.
This is the lever the whole plan turns on.

### 2.4 Recall YrA inventory (CBP server, all 17 mice × 4 recall sessions)

Every mouse has `A/C/S` in a `minian_crossreg*` folder for every recall session. `YrA.zarr` exists
only for:

| session | mice | location | usable with crossreg S/C? |
|---|---|---|---|
| Test_A | G10 | `minian/` | no — different run |
| Test_B | G10, G11 | G10 `minian_crossreg*/`, G11 `minian/` | G10 only |
| Test_A_1wk | G10 | `minian/` | no |
| Test_B_1wk | G10 | `minian_crossreg*/` | yes |

Two usable session-mice, both hM3D. Useless for any DREADD group comparison.

## 3. Decision

**Recompute YrA from the existing `A`, `C`, `b`, `f` and a regenerated `Y`. Do not re-run CNMF-E and
do not re-run cross-registration.**

Rationale: a full Minian re-run re-derives footprints, hence new unit ids, hence new crossreg
mappings — invalidating every longitudinal result (engram overlap, PV correlations, crossreg
decoding, place-field crossreg, epoch-modulation cell sets) and every cached S/C/PF/decoder/engram
artifact. That is a paper-scale redo. Recomputing YrA alone touches nothing but YrA.

## 4. The computation

From `minian/cnmf.py::compute_trace` (read at `/Volumes/vsekulic/minian_vsekulic`, lines 584-660),
the definition is exact. With `⟨·,·⟩` the pixel-wise inner product over `(height, width)`:

```
B(t)      = f(t) · b                                   background movie
Ybs(t)    = Y(t) − B(t)                                background-subtracted movie
AtA[i,j]  = ⟨A_i, A_j⟩                                 footprint overlap (unit × unit)
A_norm    = diag( 1 / ⟨A_i, A_i⟩ )                     footprint energy normalization

YrA_i(t)  = max( 0 ,  C_i(t) + [ ⟨Ybs(t), A_i⟩ − Σ_j C_j(t)·AtA[i,j] ] / ⟨A_i, A_i⟩ )
```

In words: project the background-subtracted movie onto footprint *i*, subtract the crosstalk that
every unit's denoised activity contributes through footprint overlap, normalize by footprint energy,
add the unit's own denoised trace back, and clip at zero.

Two properties matter here:

- **`Y` is unavoidable.** The bracketed term is the residual — the part of the data the model does
  *not* explain. If `Y` were exactly `A·C + b·f` the bracket would vanish and `YrA` would equal `C`.
  There is no route to YrA from `A/C/b/f` alone; the information simply is not in them.
- **`unit_id` is inherited from `A`, by construction.** `compute_trace` does `uid =
  A.coords["unit_id"]` and stamps that straight onto the output DataArray. Since `A_idx == S_idx`
  exactly (§2.3), a recomputed YrA is aligned with S/C **by construction** — not by a later repair.

## 5. Inputs: what exists, what must be regenerated

| input | status |
|---|---|
| `A.zarr`, `C.zarr`, `b.zarr`, `f.zarr` | **present** in every `minian_crossreg*` folder, all mice, all sessions |
| `motion.zarr` | **present** — motion is *applied*, never re-estimated |
| `Y` | **absent** except G10 TFC_cond (`minian_intermediate/Y_fm_chk.zarr`, `Y_hw_chk.zarr`) |

`Y` is regenerated by replaying preprocessing, all parameters recorded in
`pipeline-WORKING-TFC_cond-*.ipynb`:

```
varr      = load_videos(dpath, pattern=r"[0-9]+\.avi$", dtype=np.uint8,
                        downsample=dict(frame=1, height=1, width=1),
                        downsample_strategy="subset")
varr_ref  = varr.where(~varr.frame.isin(del_frames), drop=True).astype('uint8')
varr_ref  = varr_ref - varr_ref.min("frame")            # glow removal
varr_ref  = denoise(varr_ref, method="median", ksize=5)
varr_ref  = remove_background(varr_ref, method="tophat", wnd=15)
Y         = apply_transform(varr_ref, motion, fill=0)   # motion loaded, not estimated
YrA       = compute_trace(Y, A, b, C, f)
```

## 6. Risks, in descending order

1. **`del_frames` is per-session and lives in the notebook.** Dropping the wrong frames shifts the
   time axis and silently corrupts everything. **Mitigation:** it is recoverable independently — the
   `frame` coordinate saved in `C.zarr` / `motion.zarr` is the post-deletion frame set, so
   `del_frames = set(range(n_avi_frames)) − set(C.frame)`. Derive it that way and cross-check against
   the notebook rather than trusting either alone.
2. **`Y` may not reproduce bit-for-bit**, so recomputed conditioning YrA may differ slightly from the
   current one, moving existing YrA-based results (epoch modulation, Fig 2C traces). **This is the
   decision point, not a bug:** a YrA computed against the true footprints is more correct than a
   sorted one from a divergent run, but the numbers will move. **Mitigation:** G10 TFC_cond retains
   `Y_fm_chk`/`Y_hw_chk`, so the replay can be validated exactly on that mouse *before* committing to
   the rest (§7.1).
3. **Compute cost.** 17 mice × 5 sessions ≈ 85 replays of load + denoise + tophat + transform over
   ~25k frames each. Large but embarrassingly parallel and far cheaper than CNMF-E.
4. **`clip(0)`.** Recomputed YrA is non-negative by definition. Anything treating YrA as a signed
   fluorescence residual should be checked — `analysis.py` uses `F0 = nanmedian(YrA)` then
   `(YrA − F0)/F0`, which is fine.
5. **G17's conditioning session directory on the server is misnamed** `918_26_30-TFC_cond` (stray
   leading `9`) where `dpath_TFC_cond['G17']` says `18_26_30-TFC_cond`. Per VS this was deliberate:
   the `9` forced the correct ordering of the TFC_cond / Test_B / Test_B_1wk timestamped directories
   during cross-registration, which sorts sessions by directory name. The specifics are not recalled
   and are **not worth digging up now** — but it means the rename must NOT be "fixed" blindly, since
   the crossreg mappings that every existing result depends on were produced under that ordering.
   For the recompute, resolve the path with a glob (`*TFC_cond`) rather than the literal map entry,
   and leave the directory alone.

## 7. Verification

### 7.1 Gate: validate the replay on G10 before touching anything else
G10 TFC_cond has saved `Y_fm_chk`. Run the §5 replay for G10 and compare:
1. regenerated `Y` vs saved `Y_fm_chk` — expect exact equality, or quantify the difference;
2. `compute_trace(Y_saved, A, b, C, f)` vs the existing `YrA.zarr`, matched **by unit id** — this
   isolates "is the formula/input set right" from "does Y replay exactly";
3. the recomputed `unit_id` must equal `S_idx` elementwise, with no sort.

**Do not proceed past this gate if (2) does not come out essentially identical on the 672 shared
units.** That would mean the existing YrA was produced from something other than these A/C/b/f.

### 7.2 Per session, after recompute
- `YrA_idx == S_idx` elementwise, so `_align_YrA_to_S_units` reports **0 rows moved, 0 NaN rows,
  0 YrA-only** for every mouse and session. That is the whole point: the aligner becomes a no-op.
- The 14 previously missing conditioning cells now carry real traces; `resolve_shared_cells` should
  report `dropped['YrA']['missing'] == []` and the analysed set rises from 9,515 toward 9,531.
- Per-cell corr(C, YrA) matched by row position should now equal the by-id value (≈0.3–0.6), since
  the two orders coincide.

### 7.3 Downstream
- Re-run `build_modulation_table` on conditioning and diff against the current reference. Expect
  small movement from the 14 restored cells plus any Y-replay difference; **quantify it rather than
  assuming it is nil**, and record it in `docs/epoch_modulation.md`.

## 8. Changes needed in `caban`

Modest, because the alignment work is already done:

- `loader.py:562-576` — `mouse_path_prefix` is Windows-only (`DRIVE_1a='I'`, `DRIVE_1b='H'`,
  `"I:\\data\vsekulic\OF_test\"`). Needs a POSIX branch for `/Volumes/vsekulic/data/vsekulic/OF_test`
  (server) and/or the backup mounts.
- `sessions.py:513-523` + the YrA load block — `set_minian_output_dir` globs
  `Miniscope/minian_crossreg*` and YrA is looked up only inside it. Once YrA is re-exported into the
  crossreg folder this is correct as-is; until then it never finds the `Miniscope/`-level files.
- `zarr.load` → `zarr.open_group`. Confirmed: zarr 3.1.6 raises
  `NotImplementedError: loading groups not yet supported`; `open_group` works and yields
  `YrA (570, 25837) float64` + `unit_id (570,)`.
- Recall sessions gain a YrA where they had none, so any code branching on `YrA is None` changes
  behaviour for them. Audit those branches before the reload.
- `_align_YrA_to_S_units` **stays** as a guard. It should simply never have anything to do.

Per `CLAUDE.md`: no new analysis, so no new METHODS template; no config flag; imports at top; the
recompute must hard-fail on any inconsistency rather than skipping a session.

## 9. Reload

VS's preference is a clean full reload rather than side scripts, and under this plan that is the
right call: the YrA caches change for ~5× more sessions, so a fresh `load_all_mice(use_cache=False)`
writing a new `ds_cache.pkl` is cleaner than patching the existing 8.7 GB one. Note this also swaps
raw S/C in for the cached S/C, which O.5 measured as differing by ≤1.4×10⁻⁴ relative — so expect
every downstream number to move at that order, independently of anything YrA.

## 10. Settled 2026-09-21

**Recompute everything, conditioning included**, and use G10 as the validation gate (§7.1). The
principled choice: correct footprint correspondence for every cell in every session, rather than 8
mice patched in memory. §R numbers will move; §7.3 quantifies and records the movement rather than
assuming it is nil.

## 11. How the recompute runs

A **parameterized, idempotent, one-mouse-at-a-time notebook**, not a one-off script.

### 11.1 Shape

- `notebooks/recompute_yra.ipynb`, with a single parameter cell (`MOUSE = 'G05'`, `SESSIONS = [...]`,
  `DRY_RUN = True`). Runnable interactively, or headless per mouse via papermill so progress can be
  tailed from a log.
- All real logic lives in `caban/yra_recompute.py` so it is importable, testable and diffable; the
  notebook is a thin driver plus the report. (Notebooks diff badly — keep them thin.)
- One mouse per invocation, by design: it gives a natural checkpoint to inspect the report before
  committing to the next.

### 11.2 Idempotency

Each output gets a provenance sidecar `YrA_recompute.json` next to it, recording the Minian commit,
the parameter dicts, `del_frames`, input array hashes and a completion flag. A session is skipped
when the sidecar exists, its inputs still hash the same, and the output's `unit_id` matches `S_idx`.
Re-running the notebook is therefore free for finished sessions and resumes cleanly after a crash.

### 11.3 Non-destructive output

Write **`YrA_recomputed.zarr`** alongside the existing `YrA.zarr` — never overwrite. Promotion to the
name the loader reads is a separate, explicit step after the reports are reviewed. This keeps the
whole operation reversible and lets old and new be compared directly.

### 11.4 Do not persist Y

`Y` is 608x608x~26k float64 = **~77 GB per session** (38 GB as float32). With ~103 GB free locally,
persisting it is not viable across 85 sessions. Every step in §5 is a lazy dask operation and
`compute_trace` consumes `Y` through `tensordot`, so the chain streams from the `.avi` files without
materializing. Deviates from the stock Minian notebook, which saves `varr_ref` and `Y_fm_chk` to
`intpath` — that is a performance choice, not a correctness one. The G10 gate measures whether the
streaming version is fast enough; if not, persist as float32 and delete between sessions.

### 11.5 Parallelism

**Per-session serial, dask-parallel within.** Do not run mice concurrently: each session is already
memory- and I/O-bound through its dask graph, and concurrency would defeat the "inspect before
continuing" checkpoint. `n_workers` is the knob.

### 11.6 Per-session report, printed before the next session

Hard assertions (abort on failure):
- `YrA_new.unit_id == S_idx` elementwise, **not** merely as sets and not sorted;
- `YrA_new.shape == C.shape`; `YrA_new.frame == C.frame` elementwise;
- no NaN, no inf.

Reported for review (no auto-abort, but the reason to go one mouse at a time):
- median and 5th/95th percentile of per-cell `corr(C_i, YrA_new_i)`;
- **against the old YrA, matched by unit id** — per-cell correlation, expected ~1.0; the distribution
  of this is the single most informative number in the whole exercise, because it separates "Y
  replayed faithfully" from "Y drifted";
- rows where old and new disagree, with unit ids;
- fraction of samples hitting the `clip(0)` floor, per cell and overall;
- for the 8 divergent mice: confirmation that the previously missing 14 unit ids now carry traces;
- `del_frames` as derived from `C.frame` vs as read from the notebook — **printed side by side**, and
  a hard failure if they disagree (§6.1).

### 11.7 Order of execution

1. G10 TFC_cond alone, against the saved `Y_fm_chk` — the §7.1 gate. Nothing else runs until it
   passes.
2. The remaining 8 conditioning mice with divergent id sets (G05, G08, G12, G14, G17, G18, G20),
   where the payoff is largest and the risk best understood.
3. The 8 clean conditioning mice.
4. Recall sessions, where there is no old YrA to compare against — so they lean entirely on the
   checks that do not need one.

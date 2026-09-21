# Plan: recompute YrA from the existing footprints instead of re-running Minian

Status: **assessment complete, not yet approved.** Written 2026-09-21.
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
   leading `9`) where `dpath_TFC_cond['G17']` says `18_26_30-TFC_cond`. Fix the directory or the map
   before any batch run.

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

## 10. Open question to settle before starting

**Do conditioning results get recomputed, or only recall?** Recomputing conditioning YrA is the
principled choice (correct footprint correspondence for all 570-1049 cells, not 8 mice patched in
memory), but it moves already-written §R results. Recomputing only recall leaves conditioning on the
aligner and avoids that churn. §7.1's G10 gate answers how much movement is actually at stake, so
decide after it, not before.

# Plan: recompute YrA from the existing footprints instead of re-running Minian

Status: **128 of 130 production sessions recomputed (2026-09-27).** **Open, for VS:**
**(1)** §14.1 — the 2 G05 TFC_test_B sessions with all-NaN saved motion need re-estimated motion;
**(2)** §14.2 — a `max_proj` replay check, to prove the replay for sessions with no old `YrA`
and re-check all 128; **(3)** §15.1 — G21 TFC_test_B frames after 10,999 are paired with timestamps
3,000 frames early by the analysis loader (existing pipeline; loader fix + check of published results).
Earlier: approach approved 2026-09-21 (§10); **§7.1 G10 gate run and passed 2026-09-22**
(§12). §11 implemented in `caban/yra_recompute.py` + `notebooks/recompute_yra.ipynb`.
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

> **2026-09-22:** check (1) could not be run *on G10* — its `minian_intermediate/`
> is on the unmounted CBP server. Checks (2) and (3) were, and (2) isolates the replay
> just as well; see §12.
>
> **But check (1) is runnable after all, on other mice.** A later scan of the backup
> drives (an earlier `find` had been depth-limited and missed them) turned up **9**
> `minian_intermediate/` directories, **7 with `Y_fm_chk.zarr` and `Y_hw_chk.zarr`**:
>
> | session | arrays kept | final output |
> |---|---|---|
> | G06 `2021_10_18-TFC_cond` / `09_52_24-HC1` | **27** | `minian/` |
> | G06 `2021_10_18-TFC_cond` / `10_34_09-CNO1` | **27** | `minian/` |
> | G06 `2021_10_18-TFC_cond` / `10_49_52-CNO2` | **27** | `minian/` |
> | G06 `2021_10_18-TFC_cond` / `11_26_52-HC2` | **27** | `minian/` |
> | G03 `2021_07_16` / `13_02_08` | 14 | `minian/` |
> | G04 `2021_07_27` / `11_47_50` | 4 | `minian/` |
> | G04 `2021_07_27` / `12_13_07` | 5 | `minian/` |
>
> The four G06 sessions carry a saved `Y_fm_chk` **and** `A`/`C`/`b`/`f` in `minian/`,
> so `compare_replayed_movie` can do the exact frame-for-frame comparison there. They
> are HC/CNO sessions outside the 130, which is irrelevant for validating a replay.
> Worth running: it would upgrade §12 from "inferred from zero-overlap units" to
> "verified directly against the saved movie".
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

### 11.1b Dependencies: vendor, do not install Minian

**The recompute does not need the Minian package, and definitely not its 2021 environment.** It
needs five operations, four of which are thin wrappers:

| step | what it is | needs |
|---|---|---|
| `load_videos` | read `.avi`s in natsorted order | cv2 |
| `denoise(method="median", ksize=5)` | `cv2.medianBlur(fm, 5)` per frame via `xr.apply_ufunc` | cv2 |
| `remove_background(method="tophat", wnd=15)` | `cv2.morphologyEx(fm, MORPH_TOPHAT, disk(15))` per frame | cv2, `skimage.morphology.disk` |
| `apply_transform(varr, motion, fill=0)` | `sitk.TranslationTransform(2, -shift[::-1])` + `sitk.Resample(..., sitkLinear, 0)` | **SimpleITK** |
| `compute_trace` | the §4 equation | numpy, xarray, dask, sparse |

Only `apply_transform` has real substance — rigid translation with **linear subpixel
interpolation**. Integer-shifting instead would silently change Y, so this one must be reproduced
exactly, not approximated.

The `caban` env already has cv2 5.0, xarray 2026.4, zarr 3.1.6, sparse 0.17, skimage 0.26,
numpy 2.4, scipy, pandas, natsort, tifffile, numba. **Missing only `dask` and `SimpleITK`**, both
with current arm64 builds. (`medpy`, `rechunker`, `ffmpeg-python` are also absent but serve only
Minian paths this never touches: anisotropic denoise, chunked saving, video export.)

**Decision: vendor the five functions into `caban/yra_recompute.py`**, verbatim, each with a
provenance comment naming the Minian commit and file it came from, adapted where the modern
xarray/dask API requires. Installing Minian as-is is not an alternative *within* the caban env
anyway — it pins numpy 1.20 against the env's 2.4 — so it would mean a second environment for no
gain.

The residual risk is plumbing drift, not algorithms: `xr.apply_ufunc(..., dask="parallelized",
output_dtypes=...)` semantics moved between xarray 0.16 and 2026.4, and numpy 2.x casting is
stricter. Both surface as an exception or an obvious mismatch on the G10 gate, not as a silent
numerical shift.

**This makes §7.1's G10 gate do double duty**: matching the existing `YrA.zarr` at r ~ 1.0 proves
*both* that Y replays faithfully *and* that the vendored implementation is correct — after which
Minian need never be installed. A mismatch is the signal to fall back to a real Minian install
(`plans/cbp_server_migration_plan.md` §4), which is why that work is deferred rather than dropped.

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

## 12. G10 TFC_cond gate: run and result, 2026-09-22

Run through `caban/yra_recompute.py` (see §11): 799 units x 25,995 frames, 679 s
end to end on 6 threads, streaming from the 26 `.avi` files with no `Y` on disk.
Output at `~/cbp-db/yra_recomputed/G10/2021_11_23-TFC_cond/16_32_14-TFC_cond/YrA_recomputed.zarr` with its
`YrA_recompute.json` sidecar. The BAK drives are mounted read-only, so the output
is staged locally rather than written beside `A`/`C`/`S`.

> **Moved 2026-09-26** (VS): recomputed `YrA` now lives beside the `A`/`C`/`S` it was computed
> from — `YrA_recomputed.zarr` + `YrA_recompute.json` inside the session's `minian_crossreg*`
> folder on MINISCOPE (`discover_sessions` sets `output_dir = minian_dir`). The G10 and G05 LT1
> outputs were moved there and still pass `completed_run` against the MINISCOPE inputs.
> `~/cbp-db/yra_recomputed` no longer exists.

### 12.1 Hard checks — all passed

| check | result |
|---|---|
| `del_frames` derived from `C.frame` vs the notebook | `[]` vs `[]`. 26 `.avi`s hold exactly 25,995 frames and `C.frame` is `0..25994` contiguous; `pipeline-WORKING-TFC_cond-1.ipynb` (whose live `dpath` **is** this session) has its `del_frames` block commented out and its one live deletion guarded by a G25 `dpath`. |
| `YrA_new.unit_id == S_idx` elementwise, unsorted | yes — and `S_idx` is itself unsorted, so this is the aligner becoming a no-op, not a coincidence |
| shape and `frame` against `C` | equal |
| NaN / inf | none |
| `corr(C_i, YrA_new_i)` | median 0.507, p5 0.191, p95 0.851 — inside the 0.3–0.6 band §7.2 predicted |
| `clip(0)` floor | 0.04 % of all samples |

### 12.2 Against the existing `YrA.zarr`, matched by unit id

797 of 799 units are shared; `[344, 347]` are new-only and `[343, 346]` old-only,
which is G10's share of the §2.2 divergence. Overall per-cell r: median 0.9943,
p5 0.917, min 0.700 — **not** the flat ~1.0 §7.1 expected. Split by how much each
footprint overlaps the others, it resolves completely:

| off-diagonal overlap `Σ_{j≠i}⟨A_i,A_j⟩ / ⟨A_i,A_i⟩` | units | median r | min r |
|---|---|---|---|
| **none** | 13 | **0.999999877** | 0.999999 |
| 0–1 % | 9 | 0.999999427 | 0.999999 |
| 1–10 % | 32 | 0.999853 | 0.985 |
| >10 % | 743 | 0.993413 | 0.700 |

### 12.3 Why that is a pass, not a near-miss

In the §4 equation the `j == i` term of the crosstalk sum cancels `C_i` **exactly**,
because `AtA[i,i]/⟨A_i,A_i⟩ = 1`. So

```
YrA_i(t) = max( 0 ,  ⟨Ybs(t), A_i⟩/⟨A_i,A_i⟩  −  Σ_{j≠i} C_j(t)·AtA[i,j]/⟨A_i,A_i⟩ )
```

and a unit whose footprint overlaps nothing has `YrA_i` determined by the movie and
its own footprint alone — no `C` at all. Those 13 units reproduce the 2021 export to
r = 0.999999877. **That is the whole gate**: it says the replayed `Y` is right, `b`/`f`
are right, and the vendored `compute_trace` is right, which per §11.1b is the point at
which Minian need never be installed.

The disagreement on the rest rises monotonically with the overlap ratio — the one
coefficient by which `C` can enter — which is the signature of a different `C`, not a
different `Y`.

### 12.4 The different `C`: a correction to §2.2

Reading `pipeline-WORKING-TFC_cond-1.ipynb` end to end settles where that different
`C` comes from, and it is more specific than "the re-run diverged":

```
cell 267:  A, b, f  <- second spatial update            (saved to intpath)
cell 276:  YrA = compute_trace(Y_fm_chk, A, b, C_chk, f)  <- C_chk is PRE-update
cell 277:  C_new, S_new, ..., mask = update_temporal(A, C, YrA=YrA, ...)
cell 289:  C, S <- C_new, S_new
cell 304:  A, C, S saved; YrA saved as YrA.sel(unit_id=mask)
```

The exported `YrA` is computed from the `C` that existed **before** the second
temporal update, while the exported `C.zarr` is the one that update produced. The two
were never meant to correspond. Add that the run which wrote `YrA.zarr` is not the run
whose `A`/`C`/`S` sit in `minian_crossreg*` (hence the 2-unit difference, and hence the
residual ~0.08 % even at zero overlap), and every part of the discrepancy is accounted
for.

This also makes the recompute strictly more correct than the export, beyond the
alignment argument of §1: a recomputed `YrA` is the residual of *these* footprints
against *these* traces, which is what `YrA` is supposed to mean.

### 12.5 What §6.2 said would need deciding

§6.2 reserved judgment on conditioning numbers moving. They will: median r 0.994
against the old export, and the `>10 %`-overlap tail reaches r = 0.70. The movement is
real and is in the direction of correctness, but §7.3 should quantify it on
`build_modulation_table` before anything is rewritten.

### 12.6 Independent confirmation on a second session

G05 `2021_08_30-TFC_cond / 16_47_02-LT1`, 292 units x 12,431 frames, 137 s. A
different mouse, a different session type, and — unlike G10 — **identical unit sets**:
`in new only []`, `in old only []`. The same structure appears:

| overlap | units | median r |
|---|---|---|
| **none** | 5 | **0.999999294** |
| 0–1 % | 8 | 0.999998729 |
| 1–10 % | 9 | 0.999841 |
| >10 % | 270 | 0.996018 |

This is worth more than a repeat. With the unit sets identical, "the two runs
disagreed about which cells exist" is eliminated as an explanation for this session,
leaving only the pre- vs post-second-temporal-update `C` of §12.4. The overlap
gradient is therefore that difference and nothing else.

## 13. What building §11 turned up

Findings from implementing §11, kept because they change the plan rather than
describe the code. The notebook's own shape is documented in the notebook.

### 13.1 The queue is 130 sessions, not 85

§6.3 estimated "17 mice x 5 sessions ~ 85". Walking both drives for folders that have
numbered `.avi` *and* one complete `minian_crossreg*` finds **130**, 6-8 per mouse
across all 17, `(mouse, day, session)` unique for every one: LT1 30, LT2 17,
TFC_cond 17, Test_A 17, Test_B 16, Test_A_1wk 16, Test_B_1wk 16, LT1b 1. The other 976
`Miniscope` folders have no complete Minian output at all -- the HC and CNO sessions
never put through CNMF-E.

**28 of the 130 already have an exported `YrA.zarr`**, more than §2.4's recall
inventory implied: the LT sessions have them too, so there is more to compare against
than expected. ~137 s for a 13-file LT session, ~680 s for a 26-file conditioning
session, so the whole queue is on the order of a day, serial.

### 13.2 `del_frames` is answered corpus-wide, retiring §6.1

§6.1 is the top risk in this plan and §11.6 asked for the notebook value per session.
Scanning **all 60 `*.ipynb` in the Minian mirror** (including `.ipynb_checkpoints/`
and `prev/`) for every `del_frames` assignment and its guarding `dpath ==` test
returns exactly one non-empty value anywhere:

- **G25** `2023_04_05-TFC_cond / 16_00_27-TFC_cond`, 27 frames — an SGFR1 mouse
  outside the G05-G21 cohort, whose data is not on these drives.

Every other assignment in every notebook is the literal `del_frames = []`, and in the
TFC_cond notebooks the deletion block is commented out entirely. The notebook-side
answer for this cohort is **no frames dropped, anywhere**, recorded once in
`yr.NOTEBOOK_DEL_FRAMES` with its provenance.

This does not weaken the check: the derivation from the saved `frame` coordinate still
runs independently, both values are still printed side by side, and a disagreement is
still a hard failure. What changed is that the expected answer is now known in
advance, so a disagreement is a real signal rather than a typo.

### 13.3 G16 has a session with two Minian outputs, and one of them does not fit

`G16-ST731-mCherry/2022_01_26-TFC_test_B/14_23_45-LT1/Miniscope` holds two complete
`minian_crossreg*` folders:

| folder | A / C / S | frames |
|---|---|---|
| `minian_crossreg2_crossreg3` | 883 units | **17,853** |
| `minian_crossreg3` | 788 units | **13,923** |

The session's 14 `.avi` files hold 13,923 frames and `timeStamps.csv` has 13,923 rows,
so **`minian_crossreg3` is the one that belongs to this session**; the other has more
frames than the session has and belongs to something else. That is the override the
recompute uses, and picking wrong is not silently possible -- `derive_deleted_frames`
raises on a `frame` coordinate running past the end of the videos, which is exactly
what the other folder does.

`sessions.set_minian_output_dir` takes `glob.glob(...)[0]` with its
`assert len(glob_results) == 1` commented out, and here that returns
`minian_crossreg2_crossreg3` — the one that does not fit. **It is harmless today**:
`dpath_Test_B_LT1` has only a `G05` entry, so `caban` never loads this session. Worth
knowing before anything adds Test_B LT1 sessions to the loader, and worth restoring
that assert.

## 14. NaN motion in two production sessions (2026-09-26)

The batch wrote an **all-zero** `YrA` for G05 `2021_09_01-TFC_test_B/15_37_05-LT1` and marked it
complete: its saved `motion.zarr` is NaN in **every** frame (12,511 of 12,511), a NaN shift moves every
pixel out of frame, so the replayed `Y` is zero and so is `YrA`. A scan of every output folder on
MINISCOPE finds three with all-NaN motion:

| folder | NaN frames |
|---|---|
| G05 `2021_09_01-TFC_test_B/15_37_05-LT1/…/minian_crossreg3` (production) | 12,511 / 12,511 |
| G05 `2021_09_01-TFC_test_B/16_20_07-TFC_test_B/…/minian_crossreg2_crossreg3_crossreg4_crossreg6` (production) | 24,714 / 24,714 |
| G09 `2021_11_08-TFC_cond/19_28_33-HC3/…/minian` (the partial-output session, not production) | 6,062 / 6,062 |

Production's movie for the two G05 sessions was not zero — their `C` and `max_proj` (max 61) are
real — so the saved `motion.zarr` is not the motion that was applied. The recompute now refuses NaN
motion before replaying and before accepting an earlier output, and refuses to write a `YrA` whose
correlation with `C` is undefined for every unit (commit `c83cd4b`). The bad G05 LT1 output was deleted.

**Was it only NaN motion?** Yes, as far as the outputs show: the five recomputes on disk all have a
clip floor of 0–0.24 % and median corr(C, YrA) 0.42–0.62, and the four with an old `YrA` agree with it
at r ≈ 0.9999999 on non-overlapping units — the replay check of §12.3. The bug was a missing guard.
But a session with **no** old `YrA` (G05 LT2; every recall session) has no independent check of its
replay: finite-but-wrong motion would still pass. §14.2 closes that.

### 14.1 Recovering the two sessions (VS, 2026-09-26: plan it)

Re-estimate motion the way production did, prove it against production, replay with it.
`A`/`C`/`S` and the saved `motion.zarr` are never touched.

1. **Re-estimate with the notebook itself**, not re-implemented code: run the protected template
   (`notebooks/minian_pipeline_BASELINE.ipynb`) through `caban.minian_runner`'s machinery, **cut
   after the cell that saves `max_proj`** (baseline cell 104) — loading, glow removal, denoise,
   background removal, `estimate_motion`, `apply_transform`, `max_proj`; no CNMF. One more recorded
   edit: `minian_ds_path` points at a scratch folder, so nothing is written into the session. About
   10–20 min per session.
2. **Prove it**: the re-estimated `max_proj` must equal production's saved `max_proj` exactly
   (G06 and G10: identical in the gate). That shows the re-estimated motion reproduces the movie
   production's `A`/`C`/`S` came from.
3. **Keep it**: copy it into the production folder as `motion_reestimated.zarr`, beside the NaN
   `motion.zarr`, with a sidecar naming the run, the template md5 and the `max_proj` check.
4. **Recompute `YrA`** with it: `recompute_session_yra` gains a `motion_name` argument (default
   `"motion"`) that also enters the input hash, so the sidecar records which motion was used.

Alternative with no new code: re-run the whole notebook on each session via the runner's `PATH`
(about 1 h each) and take its `motion.zarr` — but that leaves a second full CNMF output beside
production, against the "one `minian` folder per session" goal (`crossreg_batch_runner_plan.md` §4).

### 14.2 A replay check for every session: `max_proj`

Every production folder saves `max_proj` = max over frames of the motion-corrected movie. The replay
can compute the same reduction in the same dask pass as `YrA` (nearly free), and compare. This proves
the replay for sessions with no old `YrA`, and catches finite-but-wrong motion anywhere.
Tolerance to calibrate on the five existing recomputes before it becomes a hard check: the recompute
applies motion with `caban`'s cv2 5.0, which differed from production by ±1 grey level in 132 of
5,972 frames of G06 (`minian_gate.py`, `MOTION_APPLIED_NOTE`) — so expect `max_proj` to match to
within 1 grey level at a small fraction of pixels, not bit for bit.

Order: 14.2 first (it also re-checks everything already written), then 14.1.

## 15. Unfinished recordings, and a timing gap in G21 TFC_test_B (2026-09-27)

The overnight batch finished **126 of 130**. The two §14 NaN-motion sessions failed as expected. Two more
failed on frame counting: **G15 `2022_01_13-TFC_test_B/16_19_13-TFC_test_B`** and **G21
`2022_03_24-TFC_test_B/14_35_02-TFC_test_B`** contain `.avi` files whose header was never finalised
(`ffprobe` `nb_frames` N/A), which `probe_video` did not handle.

| session | files without a header count | frames in the files that have one | production `C` frames | `timeStamps.csv` rows |
|---|---|---|---|---|
| G15 TFC_test_B | `19.avi` (865 decodable frames) — **at the end** | 19,000 | 19,000 | 19,866 |
| G21 TFC_test_B | `11.avi` (593 decodable), `12.avi`, `13.avi` (14 KB, empty) — **in the middle** | 14,856 | 14,856 | 17,856 |

Production never read those files — Minian's `load_avi_lazy` needs the header count — so its `C` is
exactly the other files. `yra_recompute.UNUSED_VIDEOS` now lists them per session (explicitly, not by
rule: an unreadable file anywhere else is still a hard failure), the replay leaves them out, and the
replay's frame-count check against `C` re-proves the list every run. Both sessions then recomputed
cleanly (corr(C, YrA) median 0.37 and 0.43 — in step with `C`, which a misplaced gap would break).
**130 of 132 accounted for: 128 done, the 2 NaN-motion sessions pending §14.1.**

A scan of all 130 production sessions for `C` frames ≠ timestamp rows finds only these, plus G09
`2021_11_08-TFC_cond/18_54_05-TFC_cond` (19,000 frames, 19,586 timestamps, every file readable —
the extra timestamps have no video frames, presumably at the end).

### 15.1 Open: G21 TFC_test_B's frames after 10,999 are 3,000 timestamps late in the analyses

`sessions.py` (`get_timestamps`, `find_exp_boundaries`) uses the `timeStamps.csv` row number as the
frame index into `C`. In G21 TFC_test_B, `C` frame 11,000 was recorded at timestamp row 14,000
(files 11–13 missing, 1,000 timestamp rows each), so every frame from 11,000 to 14,855 is paired with
a timestamp **3,000 frames (150 s at 20 fps) too early**, and an experiment boundary past row 14,855
points beyond the end of `C`. This is in the existing, published pipeline, independent of the YrA work,
and affects one conditioning-test session. Not fixed here: the fix belongs in the loader (map `C`
frames to timestamp rows through the files actually read), and whether any published G21 Test_B
result moves has to be checked. G15's unused file is at the end, so its alignment is unaffected.

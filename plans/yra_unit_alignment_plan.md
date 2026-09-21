# Plan: align YrA to S/C unit order at load time

Status: **implemented 2026-09-21** on `feat/yra-unit-alignment` (designed and implemented same day,
from `f11f600`). See §7 for what was verified and where the implementation departed from this plan.
Out of scope: anything to do with freeze scores / `freeze_data` (being reworked separately).

## 1. Problem

Minian exported `S.zarr`, `C.zarr` and `YrA.zarr` independently. For the conditioning session
(`TFC_cond`), in the raw exports and therefore in the caches:

- `S_idx` and `C_idx` are identical (same unit IDs, same order) in every session checked.
- `YrA_idx` is always sorted ascending; `S_idx`/`C_idx` are not. So in **all 17 mice** some rows
  of `session.YrA` hold a different cell than the same row of `session.S` / `session.C`
  (5–374 rows per mouse).
- In **8 mice** (G05, G08, G10, G12, G14, G17, G18, G20) the ID *sets* also differ: 14 units in
  total are in S/C but not in YrA, and 14 are in YrA but not in S/C. Trace correlations confirm
  these are genuinely different cells (r ≤ 0.16 on every cross-pair), not renumbered duplicates.
- Other cached sessions (LT1: G05/G08/G10/G11; LT2: G10/G11) have equal ID sets, differing only in
  order. Recall sessions have no YrA cache.

Evidence that matching by unit ID is right: C–YrA per-cell correlation is median r ≈ 0.3–0.57 when
matched by ID, vs ≈ 0.0–0.12 matched by row position (random-pair baseline ≈ 0.01). On 2026-09-18
the caches were checked against the raw Minian drives: cached YrA is byte-identical to raw; cached
S/C have the raw IDs and order. Background: `docs/epoch_modulation.md` §A.4 and open item O.5.

`caban/epoch_modulation.py` already matches by unit ID (`_unit_id_rows`, `resolve_shared_cells`),
so its results stand. Every other reader indexes YrA by S-derived row position and is wrong on the
misaligned rows. **No fix has been made yet** (verified 2026-09-21: `sessions.py`, `analysis.py`,
`engram_sanity.py`, `utilities.py` unchanged since 2026-08-28).

## 2. Decision

Fix once, at load time: rebuild `YrA_full` in memory so that row *i* is the same unit as row *i*
of S and C.

- **S and C are never modified** — no rows dropped, no NaNs. Crossreg mappings and all S/C-based
  results are unchanged.
- A unit that is in S/C but has no YrA export gets an **all-NaN row in YrA only**.
- A unit that is only in YrA has no S/C row to attach to; it is dropped from the in-memory YrA and
  reported.
- The on-disk YrA caches stay raw (they are verified equal to the Minian exports). Alignment
  happens in memory on every load.
- Rejected alternatives: intersecting S, C and YrA (shifts S/C row indices in 8 mice → crossreg
  mappings and existing results move); hard error until YrA is re-exported (user decided on
  2026-09-18 not to regenerate YrA).

## 3. Audit of every YrA reader (why NaN rows are safe, and where they are not)

| Reader | Live? | After the fix |
|---|---|---|
| `sessions.py:733` `YrA_pyr` | dead — inside a `'''` block | nothing to do |
| `analysis.py:630` `YrA_pyr` in `set_interneuron_cutoff` | dead — references undefined `self` | nothing to do |
| `process_PSTH_simple`, `process_PSTH_shuffle1`, `process_PSTH_shuffle_sep`, `process_PSTH_shuffle_sep_S` | no live caller (only a `'''` block in `sections.py` ~1909) | rows become correct; leave alone |
| `process_PSTH_hist` (`use_YrA`, `X_hist_use='YrA'`) | YrA options default off; no live caller enables them | rows become correct; leave alone |
| `engram_sanity.py:134`, `utilities.py:1184` | live; per-cell trace plots | correct cell; a NaN cell simply draws no YrA line. No change |
| `sessions.py:826-827` `YrA_filt`, `YrA_idx_filt` | live when the cell filter is enabled | become correct with no edit |
| `plot_sample_traces` / `plot_sample_traces2` (Fig 2C; G10, G14, G16 — G10 and G14 are affected mice) | live (`sections.py:1863-1868`) | arithmetic already nan-safe, **but `selection_mode` could randomly pick a NaN cell** → handle (§4.3) |
| `process_PSTH_shuffle` (`analysis.py:2913`, called at `sections.py:1906`) | **live** | stacks YrA-based `C_responses_save` into `group_PSTH*`, then `np.mean(..., 0)` → **one NaN cell blanks a whole group trace** → handle (§4.3) |
| `_run_abnormal_cell_filter` with `cell_filter_signal='YrA'` | only if configured (default is `'C'`) | NaN through the skew/peak checks is undefined → handle (§4.1) |
| `Session.__getattr__('YrA_full')` lazy reload (`sessions.py:441-447`) | live after the loader releases `YrA_full` (`loader.py:230`) | **would hand back the raw, unaligned matrix** → handle (§4.1) |
| `epoch_modulation._unit_id_rows` / `resolve_shared_cells` | live | once `YrA_idx == S_idx`, the 14 NaN cells stop looking "missing" (NaN SD is not `== 0`) and would leak into the analysis → handle (§4.2) |

## 4. Changes

### 4.1 `caban/sessions.py`
- New method `_align_YrA_to_S_units(YrA_raw, YrA_raw_idx)`:
  - hard-fail (clear message) if `C_idx` and `S_idx` are not elementwise equal, or if either
    `S_idx` or `YrA_raw_idx` contains duplicates, or if `YrA_raw` is not a floating dtype;
  - `aligned = np.full((len(S_idx), YrA_raw.shape[1]), np.nan, dtype=YrA_raw.dtype)`; fill rows by
    unit-ID lookup;
  - set `self.YrA_has_trace` (bool mask over S rows), `self.YrA_missing_unit_ids` (in S/C, not in
    YrA), `self.YrA_only_unit_ids` (in YrA, not in S/C — dropped);
  - print one line per session: number of rows whose position changed, the missing IDs, the
    YrA-only IDs;
  - return `aligned`. Caller sets `self.YrA_full = aligned` and
    `self.YrA_idx = np.asarray(self.S_idx).copy()`.
- Call it immediately after the YrA load block (`sessions.py:614-626`), before the session-bounds
  slice at line 631. Both branches (pickle cache, zarr) go through it; the zarr branch still saves
  the **raw** arrays to the cache first.
- `__getattr__('YrA_full')`: load both `YrA_full` and `YrA_idx` from `saver_YrA` and pass them
  through the same aligner, so a released-then-reloaded matrix is aligned too.
- `_run_abnormal_cell_filter`: when `signal_name == 'YrA'`, add submask
  `has_YrA_trace = self.YrA_has_trace`, run `filter_abnormal_cells` only over rows that have a
  trace, and AND the submask into `good_mask`. The existing kept/rejected print covers reporting.

### 4.2 `caban/epoch_modulation.py`
- `_unit_id_rows(session, 'YrA')`: omit IDs in `session.YrA_missing_unit_ids`, so
  `resolve_shared_cells` keeps reporting them under `dropped['YrA']['missing']` and the analysed
  cell set stays exactly as it is now. Update the docstring: alignment now happens at load; this
  lookup stays ID-based as a second guard.

### 4.3 `caban/analysis.py`
- `process_PSTH_shuffle`: once the row list is known (full mapping or `indeces`), take
  `has_YrA = s.YrA_has_trace[rows]`; exclude `~has_YrA` cells from the YrA-based
  `C_responses_save` stacks (`group_PSTH`, `group_PSTH_all`, `group_PSTH_vel`) and print, per
  mouse, how many were excluded and their unit IDs. The C-based significance test is untouched.
- `plot_sample_traces2` and `plot_sample_traces` (same pattern): in `selection_mode`, draw
  `cell_random` only from rows with `YrA_has_trace`; when `selections` are supplied, raise if a
  selected row has no YrA trace.

### 4.4 Docs
- `docs/epoch_modulation.md` §A.4 and O.5: record that alignment is done at load, the NaN-row
  convention, and the three new session attributes. Close O.5.

No new analysis → no new METHODS template. No config flag: the alignment is always on.

Project rules that apply (see `CLAUDE.md`): imports at top of file; no silent skips — every
exclusion above is printed with unit IDs, every inconsistency raises; no duplicated logic — one
aligner, called from both load paths.

## 5. Verification

Before editing, save a reference from the current code: per-mouse `resolve_shared_cells` output and
the `build_modulation_table` result (both signals) for `TFC_cond`.

1. Load `TFC_cond` for all 17 mice: `YrA.shape[0] == S.shape[0]`, `YrA_idx == S_idx`; NaN rows
   total 14 across G05/G08/G10/G12/G14/G17/G18/G20 and 0 in the other 9 mice.
2. Row correctness: per-row corr(C, YrA) on the previously misaligned rows is now ≈ 0.3–0.57
   median (was ≈ 0.0–0.12 by row position).
3. `epoch_modulation`: `shared` and `dropped` per mouse equal the saved reference (9,515 cells
   total); `build_modulation_table` numerically identical to the reference.
4. Lazy path: release `YrA_full`, re-access it, confirm it equals the aligned matrix.
5. Fig 2C: every row in `selections_paper` has `YrA_has_trace`; regenerate and compare with the
   existing panel (those cells were already on correctly aligned rows, so it should not change).
6. Run the `process_PSTH_shuffle` section: no NaN in `group_PSTH`; exclusion lines print for the
   affected mice.
7. LT1/LT2 sessions with YrA (G05/G08/G10/G11; G10/G11): alignment reorders only, zero NaN rows.
8. End with the `importlib.reload()` snippet: `sessions` → `epoch_modulation` → `analysis` →
   `sections`.

## 6. Known, separate, not part of this change
- `zarr.load(...)` at `sessions.py:596-624` raises `NotImplementedError: loading groups not yet
  supported` under the installed zarr v3; `zarr.open_group` works. Only bites when a cache is
  missing on this machine.
- sp_rates ΔF/F axis-label bug in `analysis.py` — being handled separately.

## 7. Implementation notes (2026-09-21)

### Verified against the cached S/C/YrA arrays on this machine
Steps 1, 2 and 7 of §5 were run directly over the per-session caches, reproducing the numbers this
plan was written from: TFC_cond has **14 NaN rows across exactly G05/G08/G10/G12/G14/G17/G18/G20**
and 0 in the other 9 mice; **14 YrA-only units dropped**; 5 (G13) to 374 (G20) rows moved per mouse,
in all 17; per-row corr(C, YrA) on the previously misaligned rows is **0.20–0.57 matched by unit id
vs 0.003–0.123 matched by row position**, with id winning in every mouse. LT1 (G05/G06/G08/G10/G11)
and LT2 (G10/G11) reorder only, **0 NaN rows**. The aligner's hard-fail branches, idempotency, the
lazy `YrA_full` reload, `__setstate__`, `_unit_id_rows` and the `analysis.py` helpers have unit
tests of their own.

**Still to run in the live session** (they need the full `ds` and interactive plotting): §5.3
(`epoch_modulation` reference comparison — expected unchanged, since `_unit_id_rows` omits the same
14 ids that `dropped['YrA']['missing']` already carried), §5.5 (Fig 2C) and §5.6
(`process_PSTH_shuffle`).

### Departure 1: `__setstate__` re-aligns (not in §3's audit)
§3 missed that `ds_cache.pkl` pickles the session-bounded `self.YrA` directly — `__getstate__` prunes
only `YrA_full`. Loading the existing 8.7 GB cache would therefore have bypassed the fix entirely.
`__setstate__` now puts `self.YrA` through the same aligner. This also forced a correctness fix:
`YrA_has_trace` is derived from **row content** (`np.isnan(row).all()`), not from the id lookup
alone, because after the first pass an absent unit *is* a NaN row under an id that now matches — the
id-only version silently cleared `YrA_missing_unit_ids` on the second pass and would have let the 14
cells back into `epoch_modulation`.

### Departure 2: `group_PSTH_all` changes for the first mouse of each group
`group_PSTH_all[group] = C_responses_save` assigned a **reference**, so the later
`C_responses_save /= len(bout_onsets)` silently rescaled the first mouse of each group; subsequent
mice went through `np.vstack`, which copies. Restricting rows to `has_YrA` copies, so that aliasing
is gone and every mouse is now on the same scale. Pre-existing bug, fixed incidentally, commented at
the site. §5.6's comparison should expect this.

### Unchanged, as designed
S and C are never touched. The on-disk YrA caches stay raw. The C-based significance test in
`process_PSTH_shuffle`, `tot_cells`, `frac_tots` and the C-derived `group_PSTH_subtract` are
untouched. A read-only indexing audit of the whole diff found no row-identity or NaN-leakage defect,
and confirmed §3's verdict that `engram_sanity.py:134` and `utilities.py:1184` become correct with
no edit.

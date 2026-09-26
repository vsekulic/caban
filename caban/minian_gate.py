"""The gate: compare a runner re-run with the 2021 run it re-derives, stage by stage.

`plans/local_minian_pipeline_plan.md` §6. Four G06 sessions kept all 27 intermediate
arrays of their 2021 run, so the question is not only "is the final answer similar?"
but "at which stage does the automated run first diverge?". A gate run
(`caban.minian_runner.run_session` with ``keep_scratch=True``) leaves:

    minian_intermediate-ORIG/, minian-ORIG/    the 2021 run
    minian_intermediate -> scratch, minian/    the re-run

Stages, in pipeline order (§6.1). The first three should reproduce; from seed
initialisation on, CNMF-E is not deterministic and the comparison is distributional
(§6.2):

1. ``varr``, ``varr_ref`` -- loading, denoise, tophat: **exact** (uint8 arithmetic).
2. motion correction -- ``motion`` re-estimated vs 2021, and ``Y_fm_chk`` both ways:
   2021's motion *applied* to the re-run's ``varr_ref`` (a diagnostic only -- see
   :data:`MOTION_APPLIED_NOTE`) and the re-run's own ``Y_fm_chk`` (exact only if the
   re-estimated motion is).
3. ``sn_spatial``, ``max_res`` -- noise estimation: floating-point tolerance.
4. ``A_init``/``C_init``, 5. ``A_mrg``/``C_mrg``, 6. final ``A``/``C`` -- units matched
   by mutual nearest footprint centroid; counts, match rate, matched-trace correlation,
   and the unmatched units drawn, not summarised away.

Most sessions kept no intermediates; there the reference is the production output
(``minian_crossreg*``) and the gate compares ``motion``, ``max_proj`` and the final
units only -- see :func:`run_gate`.

The §6.2 thresholds are the plan's proposals, "to be calibrated on the four G06
sessions before being fixed" -- so they are reported as provisional flags, and every
number behind them is in the report.
"""

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from caban import minian_runner as mr
from caban import session_queue as sq
from caban import yra_recompute as yr

# Stage 3: max |new - old| relative to max |old|.
NOISE_RTOL = 1e-6
# §6.2 proposals (provisional).
MATCH_MAX_DIST_PX = 2.0
MIN_MATCHED_FRACTION = 0.95
MIN_MEDIAN_MATCHED_CORR = 0.9
MAX_UNIT_COUNT_CHANGE = 0.05

# Stage 2, motion applied: computed here, in caban's env, with the vendored
# yr.apply_transform -- so cv2 5.0, not the 2021 run's cv2. On 2021 data alone it
# already differs from 2021's Y_fm_chk by +-1 grey level in 132 of 5,972 frames
# (16,615 of 2.2e9 values, G06 09_52_24-HC1, 2026-09-26). It therefore localises a
# re-estimated-Y_fm_chk mismatch (estimate vs apply) but cannot itself block.
MOTION_APPLIED_NOTE = ("computed in the caban env (cv2 5.0); 2021 data against itself "
                       "already differs by +-1 in 132/5972 frames -- diagnostic only")

UNIT_STAGES = (
    ("seed initialisation", "A_init", "C_init", "intermediate"),
    ("initial merge", "A_mrg", "C_mrg", "intermediate"),
    ("final", "A", "C", "final"),
)

# Figure colours: categorical slots 1-3, fixed order.
COLOR_MATCHED = "#2a78d6"
COLOR_OLD_ONLY = "#eb6834"
COLOR_NEW_ONLY = "#1baf7a"


def gate_dirs(session_dir: str, reference_final: Optional[str] = None,
              new_final: Optional[str] = None) -> dict:
    """The directories a gate compares; hard-fails unless the ones it needs exist.

    ``reference_final`` is the output to judge against: by default the set-aside
    ``minian-ORIG/``, for a production session its ``minian_crossreg*`` folder (which
    holds the CNMF-E ``A``/``C``/``S`` unchanged -- parent plan §3.4). ``new_final``
    defaults to the re-run's ``minian/``. Intermediate stages need the reference run's
    ``minian_intermediate-ORIG/``; most sessions never kept one, and then
    ``old_intermediate`` is ``None`` and only the final output is compared.
    """
    old_intermediate = os.path.join(session_dir, sq.set_aside_name(sq.SCRATCH_DIR_NAME))
    has_intermediates = os.path.isdir(old_intermediate)
    dirs = {
        "old_intermediate": old_intermediate if has_intermediates else None,
        "new_intermediate": os.path.join(session_dir, sq.SCRATCH_DIR_NAME) if has_intermediates else None,
        "old_final": reference_final or os.path.join(session_dir, sq.set_aside_name(mr.OUTPUT_NAME)),
        "new_final": new_final or os.path.join(session_dir, mr.OUTPUT_NAME),
    }
    missing = [d for d in dirs.values() if d is not None and not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError("{}: missing {} -- run the session (keep_scratch=True when "
                                "comparing intermediates) first".format(session_dir, missing))
    return dirs


def _check_coords(new: xr.DataArray, old: xr.DataArray, name: str, skip=("unit_id",)) -> None:
    if new.dims != old.dims:
        raise ValueError("{}: dims {} vs {}".format(name, new.dims, old.dims))
    for dim in new.dims:
        if dim in skip:
            continue
        if not np.array_equal(new.coords[dim].values, old.coords[dim].values):
            raise ValueError("{}: coordinate {} differs ({} vs {} values)".format(
                name, dim, new.sizes[dim], old.sizes[dim]))


def compare_exact(new: xr.DataArray, old: xr.DataArray, name: str) -> dict:
    _check_coords(new, old, name)
    differ = (new != old)
    n_differ = int(differ.sum().compute())
    return {
        "stage_array": name,
        "kind": "exact",
        "n_values": int(np.prod(new.shape)),
        "n_differ": n_differ,
        "frames_differ": int(differ.any([d for d in new.dims if d != "frame"]).sum().compute())
        if "frame" in new.dims else None,
        "max_abs_diff": float(abs(new.astype(float) - old.astype(float)).max().compute()),
        "passed": n_differ == 0,
    }


def compare_close(new: xr.DataArray, old: xr.DataArray, name: str, rtol: float = NOISE_RTOL) -> dict:
    _check_coords(new, old, name)
    max_abs = float(abs(new - old).max().compute())
    scale = float(abs(old).max().compute())
    rel = max_abs / scale if scale > 0 else float("inf")
    return {"stage_array": name, "kind": "close", "max_abs_diff": max_abs,
            "max_abs_old": scale, "max_rel_diff": rel, "rtol": rtol, "passed": rel <= rtol}


def compare_motion(new: xr.DataArray, old: xr.DataArray) -> dict:
    _check_coords(new, old, "motion")
    diff = abs(new - old).values
    per_frame = diff.max(axis=1)
    return {
        "stage_array": "motion",
        "kind": "exact",
        "max_abs_diff_px": float(diff.max()),
        "frames_differ": int((per_frame > 0).sum()),
        "frames_differ_over_0_1_px": int((per_frame > 0.1).sum()),
        "n_frames": int(new.sizes["frame"]),
        "passed": bool(diff.max() == 0),
    }


def footprint_centroids(A: xr.DataArray) -> np.ndarray:
    """(unit, [height, width]) intensity-weighted centroid of each footprint, in pixels."""
    mass = A.sum(["height", "width"])
    rows = (A * A.coords["height"]).sum(["height", "width"]) / mass
    cols = (A * A.coords["width"]).sum(["height", "width"]) / mass
    return np.stack([rows.compute().values, cols.compute().values], axis=1)


def match_units(cent_old: np.ndarray, cent_new: np.ndarray, max_dist: float) -> np.ndarray:
    """Pairs ``(i_old, i_new)`` that are each other's nearest centroid, within ``max_dist``."""
    dist_on, nearest_new = cKDTree(cent_new).query(cent_old)
    _, nearest_old = cKDTree(cent_old).query(cent_new)
    pairs = [(i, j) for i, j in enumerate(nearest_new)
             if nearest_old[j] == i and dist_on[i] <= max_dist]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def compare_units(A_new, C_new, A_old, C_old, name: str, max_dist: float = MATCH_MAX_DIST_PX):
    """§6.2: count, centroid match rate, matched-trace correlation, the unmatched units."""
    _check_coords(C_new, C_old, name + " C")
    cent_old, cent_new = footprint_centroids(A_old), footprint_centroids(A_new)
    pairs = match_units(cent_old, cent_new, max_dist)
    n_old, n_new = len(cent_old), len(cent_new)
    old_ids, new_ids = A_old.coords["unit_id"].values, A_new.coords["unit_id"].values
    # C indexed by A's unit_id, never assumed to share A's order.
    corr = yr._per_cell_correlation(C_old.sel(unit_id=old_ids[pairs[:, 0]]).values,
                                    C_new.sel(unit_id=new_ids[pairs[:, 1]]).values) \
        if len(pairs) else np.array([])
    shifts = np.linalg.norm(cent_old[pairs[:, 0]] - cent_new[pairs[:, 1]], axis=1) \
        if len(pairs) else np.array([])
    count_change = (n_new - n_old) / n_old
    matched_fraction = len(pairs) / n_old
    median_corr = float(np.nanmedian(corr)) if len(corr) else float("nan")
    return {
        "stage_array": name,
        "kind": "distributional",
        "n_old": n_old,
        "n_new": n_new,
        "count_change": count_change,
        "n_matched": int(len(pairs)),
        "matched_fraction_of_old": matched_fraction,
        "matched_fraction_of_new": len(pairs) / n_new,
        "centroid_shift_px": yr._summary(shifts),
        "matched_corr_C": yr._summary(corr),
        "unmatched_old_unit_ids": sorted(set(old_ids.tolist()) - set(old_ids[pairs[:, 0]].tolist())),
        "unmatched_new_unit_ids": sorted(set(new_ids.tolist()) - set(new_ids[pairs[:, 1]].tolist())),
        "max_dist_px": max_dist,
        "provisional_checks": {
            "count_within_5pct": abs(count_change) <= MAX_UNIT_COUNT_CHANGE,
            "matched_95pct_of_old": matched_fraction >= MIN_MATCHED_FRACTION,
            "median_matched_corr_over_0_9": median_corr > MIN_MEDIAN_MATCHED_CORR,
        },
        "_pairs": pairs,
    }


def plot_unit_match(A_new, A_old, background: np.ndarray, result: dict, path: str) -> None:
    """Every footprint outlined: matched, 2021-only and re-run-only in their own colours."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(background, cmap="gray")
    pairs = result["_pairs"]
    groups = (
        (A_old.isel(unit_id=pairs[:, 0]) if len(pairs) else None, COLOR_MATCHED,
         "matched ({})".format(len(pairs))),
        (A_old.sel(unit_id=result["unmatched_old_unit_ids"]), COLOR_OLD_ONLY,
         "reference only ({})".format(len(result["unmatched_old_unit_ids"]))),
        (A_new.sel(unit_id=result["unmatched_new_unit_ids"]), COLOR_NEW_ONLY,
         "new only ({})".format(len(result["unmatched_new_unit_ids"]))),
    )
    for footprints, color, label in groups:
        if footprints is None or footprints.sizes["unit_id"] == 0:
            ax.plot([], [], color=color, lw=1.5, label=label)
            continue
        values = footprints.compute().values
        for unit in values:
            ax.contour(unit, levels=[0.3 * unit.max()], colors=[color], linewidths=0.8)
        ax.plot([], [], color=color, lw=1.5, label=label)
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_title("{}: footprints at 30 % of peak, matched by mutual nearest centroid "
                 "within {} px".format(result["stage_array"], result["max_dist_px"]),
                 fontsize=10, loc="left")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_gate(session_dir: str, reference_final: Optional[str] = None,
             new_final: Optional[str] = None, tag: str = "gate",
             out_dir: Optional[str] = None, stop_at_first_mismatch: bool = True) -> dict:
    """Compare a run with its reference, in pipeline order; write ``<tag>_report.json``.

    See :func:`gate_dirs` for ``reference_final`` / ``new_final``. Without reference
    intermediates the intermediate stages are listed as not available in the report and
    only ``motion``, ``max_proj`` (the maximum over the motion-corrected movie, so an
    exact check of preprocessing and motion correction together) and the final units are
    compared. ``tag`` prefixes the report and figures, so several comparisons of one
    session sit side by side.

    With ``stop_at_first_mismatch`` (§6.1) the reproducible stages stop the gate at the
    first failure -- everything downstream inherits that divergence. The unit stages are
    distributional and always all reported.
    """
    dirs = gate_dirs(session_dir, reference_final, new_final)
    out_dir = out_dir or os.path.join(session_dir, mr.RUN_DIR_NAME)
    has_intermediates = dirs["old_intermediate"] is not None
    new = lambda n: yr.open_minian_array(dirs["new_intermediate"], n)
    old = lambda n: yr.open_minian_array(dirs["old_intermediate"], n)
    new_final = lambda n: yr.open_minian_array(dirs["new_final"], n)
    old_final = lambda n: yr.open_minian_array(dirs["old_final"], n)
    stages = []
    report = {"session_dir": session_dir, "dirs": dirs, "stages": stages,
              "intermediate_stages": "compared" if has_intermediates else
              "not available: the reference run kept no minian_intermediate",
              "first_mismatch": None, "stopped_early": False}
    print("  intermediate stages: {}".format(report["intermediate_stages"]))

    def record(result):
        stages.append(result)
        passed = result.get("passed", True)
        blocking = result.get("blocking", True)
        if result["kind"] == "distributional":
            checks = result["provisional_checks"]
            verdict = "{}/{} provisional §6.2 checks met ({})".format(
                sum(checks.values()), len(checks),
                ", ".join(k for k, v in checks.items() if not v) or "all")
        else:
            verdict = "pass" if passed else ("MISMATCH" if blocking else "differs (diagnostic)")
        print("  {:26s} {}".format(result["stage_array"], verdict))
        if not passed and blocking and report["first_mismatch"] is None:
            report["first_mismatch"] = result["stage_array"]
        return passed or not blocking

    def write():
        path = os.path.join(out_dir, "{}_report.json".format(tag))
        clean = dict(report, stages=[{k: v for k, v in s.items() if not k.startswith("_")}
                                     for s in stages])
        with open(path, "w") as fh:
            json.dump(clean, fh, indent=2, default=str)
        return path

    reproducible = []
    if has_intermediates:
        reproducible += [
            lambda: compare_exact(new("varr"), old("varr"), "varr"),
            lambda: compare_exact(new("varr_ref"), old("varr_ref"), "varr_ref"),
            lambda: dict(compare_exact(
                yr.apply_transform(new("varr_ref"), old_final("motion")).astype(float),
                old("Y_fm_chk"), "Y_fm_chk"), stage_array="Y_fm_chk (2021 motion)",
                blocking=False, note=MOTION_APPLIED_NOTE),
        ]
    reproducible.append(lambda: compare_motion(new_final("motion"), old_final("motion")))
    if has_intermediates:
        reproducible += [
            lambda: dict(compare_exact(new("Y_fm_chk"), old("Y_fm_chk"), "Y_fm_chk"),
                         stage_array="Y_fm_chk (re-estimated)"),
            lambda: compare_close(new("sn_spatial"), old("sn_spatial"), "sn_spatial"),
            lambda: compare_close(new("max_res"), old("max_res"), "max_res"),
        ]
    reproducible.append(lambda: compare_exact(new_final("max_proj"), old_final("max_proj"), "max_proj"))
    for compare in reproducible:
        if not record(compare()) and stop_at_first_mismatch:
            report["stopped_early"] = True
            print("  stopped at the first mismatch; report: {}".format(write()))
            return report

    background = old_final("max_proj").values
    for label, a_name, c_name, where in UNIT_STAGES:
        if where == "intermediate" and not has_intermediates:
            continue
        src_new = dirs["new_" + where]
        src_old = dirs["old_" + where]
        A_new, C_new = yr.open_minian_array(src_new, a_name), yr.open_minian_array(src_new, c_name)
        A_old, C_old = yr.open_minian_array(src_old, a_name), yr.open_minian_array(src_old, c_name)
        result = compare_units(A_new, C_new, A_old, C_old, "{} ({})".format(a_name, label))
        figure = "{}_units_{}.png".format(tag, a_name)
        plot_unit_match(A_new, A_old, background, result, os.path.join(out_dir, figure))
        result["figure"] = figure
        record(result)
        print("    {} -> {} units, {} matched ({:.0%} of reference), median matched corr(C) {:.3f}"
              .format(result["n_old"], result["n_new"], result["n_matched"],
                      result["matched_fraction_of_old"], result["matched_corr_C"].get("median", np.nan)))
    print("  report: {}".format(write()))
    return report

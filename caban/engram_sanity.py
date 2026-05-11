"""caban.engram_sanity — engram-cell sanity / inspection plots.

Produces a directory tree per engram type and per classification mode:

    <save_root>/engram_<etype>/<mode>/histograms/<mouse>.png
    <save_root>/engram_<etype>/<mode>/cells/<mouse>_<above|below>_rank<i>_cell<idx>.png

Where:
    etype  in {'encoding', 'recall'}
        encoding -> engram threshold computed on TFC_cond
        recall   -> engram threshold computed on Test_B
        Both use the cross-registered cells of ``mapping`` so the
        score distribution corresponds to the cells that survive the
        crossreg filter (which is what downstream PCA actually sees).
    mode   in {'permouse', 'ctlthresh_z', 'ctlthresh_p50'}
        Same modes used by caban.main / caban.population.

Per cell, one figure with traces (S red / C orange / YrA light-grey) on
the left and the cell's ROI (A matrix slice) on the right.

All imports live at module top.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from caban.engram import ENGRAM_REFERENCE


# Spike-detection threshold (matches per-session ``thres = 2`` in
# caban.sessions). Drawn on the cell-trace panel as the line above
# which deconvolved-S peaks are counted as transients.
_SPIKE_THRES = 2.0


_TRACE_FIGSIZE = (15, 3)
_HIST_FIGSIZE = (8, 4)
_DPI = 200


# ---------------------------------------------------------------------------
# Score transform
# ---------------------------------------------------------------------------

def _apply_mode(raw_score, mode, ext_norm):
    """Return (transformed_score, cutoff) for the chosen engram mode."""
    if mode == "permouse":
        mu = float(np.mean(raw_score))
        sigma = float(np.std(raw_score))
        if sigma <= 0:
            raise RuntimeError(
                f"_apply_mode permouse: zero SD for raw_score "
                f"(n={raw_score.size})."
            )
        return (raw_score - mu) / sigma, 0.0
    if mode == "ctlthresh_z":
        if ext_norm is None or ext_norm[0] != "zscore":
            raise ValueError(
                f"ctlthresh_z requires ext_norm=('zscore', mu, sigma); got {ext_norm!r}"
            )
        _, mu, sigma = ext_norm
        return (raw_score - float(mu)) / float(sigma), 0.0
    if mode == "ctlthresh_p50":
        if ext_norm is None or ext_norm[0] != "absolute":
            raise ValueError(
                f"ctlthresh_p50 requires ext_norm=('absolute', cutoff); got {ext_norm!r}"
            )
        _, cutoff = ext_norm
        return raw_score, float(cutoff)
    raise ValueError(f"Unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------

def _plot_histogram(scores, cutoff, title, xlabel, save_path):
    fig, ax = plt.subplots(figsize=_HIST_FIGSIZE)
    ax.hist(scores, bins=80, color="0.5", edgecolor="0.2")
    ax.axvline(cutoff, color="red", linestyle="--",
               label=f"threshold={cutoff:.3g}")
    n_above = int(np.sum(scores > cutoff))
    n_total = int(scores.size)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{title}\n{n_above}/{n_total} above threshold "
                 f"({100.0*n_above/max(1, n_total):.1f}%)")
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=_DPI)
    plt.close(fig)


_HIST_XLABEL_BY_MODE = {
    "permouse":      "Avg transient rate (z-score, per-mouse)",
    "ctlthresh_z":   "Avg transient rate (z-score, mCherry-pooled)",
    "ctlthresh_p50": "Avg transient rate (Hz)",
}


def _safe_get_A(session):
    """Return A or None — tries get_A_matrix() if not yet loaded."""
    A = getattr(session, "A", None)
    if A is not None:
        return A
    if hasattr(session, "get_A_matrix"):
        session.get_A_matrix()
        return getattr(session, "A", None)
    return None


def _plot_cell(session, cell_idx_into_S, raw_score_value,
               transformed_score_value, cutoff, threshold_y,
               global_y_max, title, save_path):
    """One figure: traces on left, A ROI on right.

    The dotted horizontal line is the *spike-detection threshold* used
    by ``find_spikes_ca`` (``thres = 2``) — peaks above this line are
    counted as transients, and the per-cell average transient rate is
    the engram-classification statistic. ``threshold_y`` (the engram
    score cutoff in rate units) is unused on the per-frame trace axis
    and is intentionally not drawn here.
    """
    fig = plt.figure(figsize=_TRACE_FIGSIZE)
    gs = GridSpec(1, 2, width_ratios=[5, 1], figure=fig, wspace=0.15)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])

    S_row = session.S[cell_idx_into_S, :]
    C_row = session.C[cell_idx_into_S, :] if hasattr(session, "C") else None
    YrA_row = session.YrA[cell_idx_into_S, :] if (
        hasattr(session, "YrA") and getattr(session, "YrA", None) is not None
    ) else None

    t = np.arange(S_row.shape[0])
    if YrA_row is not None:
        ax_t.plot(t, YrA_row, color="0.75", linewidth=0.5, label="YrA")
    if C_row is not None:
        ax_t.plot(t, C_row, color="orange", linewidth=0.7, label="C")
    ax_t.plot(t, S_row, color="red", linewidth=0.5, label="S")
    ax_t.set_xlim(t[0], t[-1])
    ax_t.set_ylim(0, max(global_y_max, _SPIKE_THRES * 1.5))
    ax_t.axhline(_SPIKE_THRES, color="black", linestyle=":", linewidth=1.0,
                 label=f"spike thr={_SPIKE_THRES:.2g}")
    ax_t.set_xlabel("Frame")
    ax_t.set_ylabel("Trace")
    ax_t.legend(loc="upper right", fontsize=7, ncol=4, framealpha=0.7)

    A = _safe_get_A(session)
    if A is not None and cell_idx_into_S < A.shape[0]:
        ax_a.imshow(A[cell_idx_into_S], cmap="magma", aspect="equal")
        ax_a.set_xticks([]); ax_a.set_yticks([])
        ax_a.set_title("ROI A", fontsize=9)
    else:
        ax_a.text(0.5, 0.5, "A not available", ha="center", va="center",
                  transform=ax_a.transAxes, fontsize=9)
        ax_a.set_xticks([]); ax_a.set_yticks([])

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _global_S_max(sessions_by_name, mice):
    """Max of session.S across every (session_name, mouse) we'll plot."""
    g = 0.0
    for name, sess_dict in sessions_by_name.items():
        for m in mice:
            if m not in sess_dict:
                continue
            S = sess_dict[m].S
            if S.size:
                v = float(np.max(S))
                if v > g:
                    g = v
    if g <= 0:
        raise RuntimeError("Global S max is non-positive.")
    return g


def _threshold_in_raw_units(mode, ext_norm, raw_score):
    """Express the threshold in *raw_score* units (same units as the cell
    plot can render along y). For permouse: mu+0*sigma = mean. For
    ctlthresh_z: mu (since raw_thr corresponds to score=0). For
    ctlthresh_p50: the absolute cutoff itself.
    """
    if mode == "permouse":
        return float(np.mean(raw_score))
    if mode == "ctlthresh_z":
        _, mu, _sigma = ext_norm
        return float(mu)
    if mode == "ctlthresh_p50":
        _, cutoff = ext_norm
        return float(cutoff)
    raise ValueError(mode)


def plot_engram_sanity(
    engram_id,                 # {mouse: {etype: {mode: bool_mask}}}
    engram_rates,              # {etype: {mouse: ndarray rate}}
    engram_norms,              # {etype: {mode: ext_norm | None}}
    ref_sessions_by_etype,     # {'encoding': TFC_cond, 'recall': Test_B}
    mouse_groups,
    save_root,
    other_sessions_by_name=None,  # unused; kept for caller-side symmetry
    n_cells=5,
    modes=("permouse", "ctlthresh_z", "ctlthresh_p50"),
    etypes=("encoding", "recall"),
):
    """Histogram + cell-trace plots driven by the unified engram identity.

    Consumes the outputs of ``caban.engram.build_engram_identity``:

    * ``engram_id``    : per-mouse, per-etype, per-mode boolean mask over
                         FULL reference-session rows.
    * ``engram_rates`` : per-etype, per-mouse raw transient-rate (Hz)
                         vector over FULL reference-session rows.
    * ``engram_norms`` : per-etype, per-mode external-norm spec used by
                         the ctlthresh_* modes (or ``None``).

    Cell indices in the histograms / trace plots are FULL row indices
    into the reference session's ``S`` matrix.
    """
    mice = list(mouse_groups.keys())
    global_y_max = _global_S_max(
        {f"_ref_{e}": ref_sessions_by_etype[e] for e in etypes
         if e in ref_sessions_by_etype},
        mice,
    )

    for etype in etypes:
        if etype not in ref_sessions_by_etype:
            raise KeyError(f"plot_engram_sanity: missing ref sessions for "
                           f"etype={etype!r}")
        ref_sessions = ref_sessions_by_etype[etype]
        rates_by_mouse = engram_rates[etype]

        for mode in modes:
            ext_norm = engram_norms[etype][mode]
            etype_dir = os.path.join(save_root, f"engram_{etype}", mode)
            hist_dir = os.path.join(etype_dir, "histograms")
            cells_dir = os.path.join(etype_dir, "cells")
            n_done = 0

            for m, raw in rates_by_mouse.items():
                if m not in ref_sessions:
                    print(f"  [engram-sanity] {etype}/{mode}: mouse {m} has "
                          f"no reference session; skipping.", flush=True)
                    continue
                score, cutoff = _apply_mode(raw, mode, ext_norm)
                title = (f"{m} ({mouse_groups[m]}) | etype={etype} | "
                         f"mode={mode} | n={raw.size}")
                _plot_histogram(
                    score, cutoff, title,
                    _HIST_XLABEL_BY_MODE[mode],
                    os.path.join(hist_dir, f"{m}.png"),
                )

                thr_raw_y = _threshold_in_raw_units(mode, ext_norm, raw)
                order = np.argsort(score)
                below_pick = order[:n_cells]
                above_pick = order[::-1][:n_cells]
                ref_sess = ref_sessions[m]

                for rank, ci in enumerate(above_pick):
                    cell_in_S = int(ci)
                    fname = f"{m}_above_rank{rank:02d}_cell{cell_in_S}.png"
                    cell_title = (
                        f"{m} ({mouse_groups[m]}) | etype={etype} | "
                        f"mode={mode} | cell {cell_in_S} | "
                        f"raw={raw[ci]:.3g} score={score[ci]:.3g} "
                        f"cutoff={cutoff:.3g} | ABOVE"
                    )
                    _plot_cell(ref_sess, cell_in_S, raw[ci], score[ci],
                               cutoff, thr_raw_y, global_y_max, cell_title,
                               os.path.join(cells_dir, fname))

                for rank, ci in enumerate(below_pick):
                    cell_in_S = int(ci)
                    fname = f"{m}_below_rank{rank:02d}_cell{cell_in_S}.png"
                    cell_title = (
                        f"{m} ({mouse_groups[m]}) | etype={etype} | "
                        f"mode={mode} | cell {cell_in_S} | "
                        f"raw={raw[ci]:.3g} score={score[ci]:.3g} "
                        f"cutoff={cutoff:.3g} | BELOW"
                    )
                    _plot_cell(ref_sess, cell_in_S, raw[ci], score[ci],
                               cutoff, thr_raw_y, global_y_max, cell_title,
                               os.path.join(cells_dir, fname))
                n_done += 1
            print(f"  [engram-sanity] etype={etype} mode={mode}: "
                  f"wrote {n_done} mice -> {etype_dir}", flush=True)


print("caban.engram_sanity.py loaded.")

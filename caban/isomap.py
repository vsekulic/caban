"""Isomap manifold analysis (Wilson-lab style) for SSTCa2.

Implements three phases:

1. Per-session independent Isomap manifolds on full ``sess.S`` for
   TFC_cond / Test_B / Test_B_1wk. Sweeps NN ∈ [NN_RANGE[0], NN_RANGE[1]]
   in steps of NN_STEPS, stops when reconstruction-error improvement is
   < NN_CUTOFF for two consecutive NNs.
2. Cross-session paper-style embedding: fit Isomap on TFC_cond (cells
   restricted to ``mapping_TFC_cond_Test_B_Test_B_1wk``) then project
   Test_B / Test_B_1wk through the same embedding.
3. Cross-manifold similarity score vs the TFC_cond reference, computed
   on trial-averaged event-phase trajectories with a 5-degree rotation
   search (paper's similarity metric).

All output goes under ``<PLOTS_DIR>/TFC_Isomap/``.
"""
from __future__ import annotations

import os
import sys
import time
import json
try:
    import msvcrt  # Windows-only; killswitch keyboard polling.
except ImportError:
    msvcrt = None  # no-op on POSIX (killswitch helpers below handle this).
import hashlib
import gc
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import Isomap
from joblib import Parallel, delayed
from types import SimpleNamespace
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from SSTCa2_utilities import get_S_indeces_crossreg, MINISCOPE_FPS
from SSTCa2_population import PERIOD_FRAMES
from SSTCa2_decoder import _copy_analysis_methods_template


# ---------------------------------------------------------------------------
# Top-level parameters
# ---------------------------------------------------------------------------

NN_STEPS = 5
NN_RANGE = (5, 100)
NN_CUTOFF = 0.10            # 10% plateau threshold
SS_ROTATE = 5               # degrees per rotation step in similarity search
ISOMAP_DIMS = 2
POST_SHOCK_SEC = 20         # colored window after shock offset (paper analog)

# Temporal downsampling before Isomap fit. Default = MINISCOPE_FPS (no
# downsampling). Set to 10 to reproduce the paper's 30Hz->10Hz step at
# our 20Hz native rate (block-mean by factor 2). Must evenly divide
# MINISCOPE_FPS.
DOWNSAMPLE_HZ = MINISCOPE_FPS / 2

# Parallelism. With OUTER=1 / INNER=-1 we run one Isomap fit at a
# time but let sklearn's kNN search (BallTree/KDTree on the coordinate
# input) parallelize across ALL logical processors. At T~10k that
# kNN stage uses every core for several seconds per fit, which is the
# single biggest source of multi-core utilisation in the pipeline.
# MKL further parallelises the dense LAPACK eigendecomp at the end of
# each fit. The Dijkstra geodesic step in between is single-threaded
# scipy Cython and unavoidable.
#
# OUTER>1 was tried; it caps at ~4-8 workers and showed less total CPU
# utilisation than OUTER=1/INNER=-1 because each worker had to share
# joblib's process pool and BLAS threads. Empirically OUTER=1/INNER=-1
# saturates the machine best.
#
# IMPORTANT: this requires the inner Isomap to receive the coordinate
# matrix X (metric='minkowski'), NOT a precomputed distance matrix.
# Passing a precomputed (T,T) D matrix bypasses BallTree/KDTree and
# forces single-threaded argpartition kNN extraction — which kills the
# main source of inner parallelism. The independent per-session fit
# must therefore use the coordinate path, even though it means
# redundant pairwise distance
# work across the NN sweep.
ISOMAP_N_JOBS_OUTER = 1
ISOMAP_N_JOBS_INNER = -1

# Eigen solver passed through to Isomap (its inner KernelPCA). 'arpack'
# does iterative Lanczos for just the top d=2 eigenpairs — O(T²·d·iter)
# vs dense O(T³). At T~12.8k this is the difference between ~5s and
# ~60s of pure eigendecomp per fit; dense was empirically dominating
# wall time (~240s/fit observed). ARPACK eigenvectors have arbitrary
# signs from random init; we canonicalise them post-fit so embeddings
# are reproducible across runs (see _canonicalize_isomap_signs).
ISOMAP_EIGEN_SOLVER = "arpack"

# Memory budget for the parallel NN sweep, in GB. The number of
# concurrent Isomap fits is capped to keep total estimated memory
# (T² dense matrices × ~3 per fit) below this budget. With OUTER=1
# only one fit is in flight at a time, so this only matters if you
# raise OUTER>1. Set lower if baseline RAM usage from previously
# loaded data is high.
ISOMAP_MEM_BUDGET_GB = 8.0


def _estimate_per_fit_gb(T: int) -> float:
    """Rough peak memory per Isomap fit at T samples (geodesic + kernel
    + workspace, all T×T float64)."""
    return 3.0 * (T * T * 8) / (1024 ** 3)


def _cap_workers_for_memory(requested: int, T: int) -> int:
    """Cap requested workers to fit ISOMAP_MEM_BUDGET_GB at T samples."""
    per_fit = _estimate_per_fit_gb(T)
    if per_fit <= 0:
        return max(1, requested)
    allowed = max(1, int(ISOMAP_MEM_BUDGET_GB // per_fit))
    return max(1, min(requested, allowed))

# Interactive abort prompt. After each NN fit prints its err, display a
# 3..2..1 countdown; if ENTER is pressed within the window, raise
# IsomapAbort to stop the entire pipeline. Useful for sanity-checking
# parameters early without waiting for the full sweep / all sessions.
ISOMAP_ABORT_PROMPT = True
ISOMAP_ABORT_SECONDS = 10


class IsomapAbort(Exception):
    """Raised when the user presses ENTER during the abort countdown."""


def _abort_countdown(seconds: int = ISOMAP_ABORT_SECONDS) -> None:
    """Print a countdown; abort via IsomapAbort if ENTER is pressed.

    Windows-only (uses ``msvcrt.kbhit``/``getch``). No-op when
    ``ISOMAP_ABORT_PROMPT`` is False or stdin is not a TTY.
    """
    if not ISOMAP_ABORT_PROMPT:
        return
    if not sys.stdin.isatty():
        return
    # Drain any buffered keystrokes from the prior fit window.
    while msvcrt is not None and msvcrt.kbhit():
        msvcrt.getch()
    end_t = time.time() + seconds
    last_shown = -1
    print("    [ENTER=abort] ", end="", flush=True)
    while True:
        remaining = end_t - time.time()
        if remaining <= 0:
            break
        cur = int(remaining) + 1
        if cur != last_shown:
            print(f"{cur}.. ", end="", flush=True)
            last_shown = cur
        if msvcrt is not None and msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b"\r", b"\n"):
                print("ABORTED by user", flush=True)
                raise IsomapAbort("user pressed ENTER during countdown")
        time.sleep(0.05)
    print("continue", flush=True)

_METHODS_TEMPLATE = "isomap_methods.txt"

# Group → (label, color). mCherry → Ctl, hM3D → Exc, hM4D → Inh.
GROUP_LABEL = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}
GROUP_COLOR = {"Ctl": "black", "Exc": "red", "Inh": "blue"}
GROUP_ORDER = ("Ctl", "Exc", "Inh")

SESSION_ORDER = ("TFC_cond", "Test_B", "Test_B_1wk")

# Phase color map (per user request).
# post_tone is intentionally low-alpha + small marker so it doesn't
# dominate the scatter when isi/pre_tone/shock points are sparser.
PHASE_COLORS = {
    "pre_tone":   (0.60, 0.95, 0.60, 0.85),  # light green
    "tone":       (0.10, 0.20, 0.85, 0.85),  # blue
    "post_tone":  (0.55, 0.75, 0.95, 0.30),  # light blue, transparent
    "shock":      (0.85, 0.10, 0.10, 0.95),  # red
    "post_shock": (0.95, 0.55, 0.55, 0.85),  # light red
    "isi":        (0.80, 0.80, 0.80, 0.50),  # light grey
}
# Per-phase scatter marker size multiplier. post_tone shrunk so dense
# clouds of post_tone frames don't drown out the rarer phases.
PHASE_SIZE_MULT = {
    "pre_tone":   1.0,
    "tone":       1.0,
    "post_tone":  0.45,
    "shock":      1.6,
    "post_shock": 1.2,
    "isi":        1.0,
}
PHASE_ORDER_FOR_LEGEND = (
    "pre_tone", "tone", "post_tone", "shock", "post_shock", "isi"
)
# Plot phases in this z-order so post_tone (largest in count) is drawn
# first/underneath and rarer phases (shock, tone) sit on top.
PHASE_DRAW_ORDER = (
    "isi", "post_tone", "pre_tone", "post_shock", "tone", "shock"
)

# Top-level switch: label each tone/shock trial on the manifold grids by
# drawing a thin time-ordered polyline through that trial's embedded
# frames and placing a small 't{k}' / 's{k}' tag at the point of maximal
# distance from the embedding centroid. Useful for testing whether the
# "arms" of the manifold correspond to individual tone-shock pairs. Set
# to False if it makes the plot too cluttered.
LABEL_TRIAL_ARMS = True
# Additional switch: also label post_tone ('pt{k}') and post_shock
# ('ps{k}') trials. Because labeling all four phases at once would be
# very cluttered, when this is True we emit TWO additional variants of
# every plot that already had trial-arm labels: one with post_shock
# added (suffix '_with_post_shock'), one with post_tone added (suffix
# '_with_post_tone'). The base tone+shock plot is always produced.
# Has no effect when LABEL_TRIAL_ARMS is False.
LABEL_POST_TRIAL_ARMS = True
TRIAL_ARM_LINE_KW = {"color": "black", "linewidth": 0.5, "alpha": 0.55,
                     "zorder": 5}
TRIAL_ARM_LABEL_FONTSIZE = 7
# Default labeling target list: (prefix, phase_name).
TRIAL_ARM_TARGETS_BASE = (("t", "tone"), ("s", "shock"))


def _arm_label_variants():
    """Yield ``(fname_suffix, targets_tuple)`` for each plot variant.

    - If ``LABEL_TRIAL_ARMS`` is False: a single variant with no
      labeling (empty targets, empty suffix).
    - Else: always emit the base tone+shock variant. If
      ``LABEL_POST_TRIAL_ARMS`` is also True, additionally emit
      ``_with_post_shock`` and ``_with_post_tone`` variants.
    """
    if not LABEL_TRIAL_ARMS:
        return [("", ())]
    variants = [("", TRIAL_ARM_TARGETS_BASE)]
    if LABEL_POST_TRIAL_ARMS:
        variants.append(("_with_post_shock",
                         TRIAL_ARM_TARGETS_BASE + (("ps", "post_shock"),)))
        variants.append(("_with_post_tone",
                         TRIAL_ARM_TARGETS_BASE + (("pt", "post_tone"),)))
    return variants


def _with_suffix(fname: str, suffix: str) -> str:
    """Insert ``suffix`` before the extension of ``fname``."""
    if not suffix:
        return fname
    root, ext = os.path.splitext(fname)
    return f"{root}{suffix}{ext}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_per_neuron_01(S: np.ndarray, *, label: str = "",
                             drop_silent: bool = False) -> np.ndarray:
    """Rescale each row (neuron) of S to [0, 1] within the session.

    Silent cells (zero range over the whole session) cannot be
    normalised meaningfully. By default we RAISE — the crossreg fit
    needs cell-by-cell alignment across sessions, so silently dropping
    a cell in one session and not another would break that contract.

    The independent per-session fit can pass
    ``drop_silent=True`` to drop silent cells with a loud warning,
    since cell counts there are not coordinated across sessions.
    """
    if S.ndim != 2:
        raise ValueError(f"_normalize_per_neuron_01: expected 2D, got {S.shape}")
    smin = S.min(axis=1, keepdims=True)
    smax = S.max(axis=1, keepdims=True)
    rng = smax - smin
    bad_mask = (rng[:, 0] <= 0)
    if np.any(bad_mask):
        bad_idx = np.where(bad_mask)[0].tolist()
        tag = f"[{label}] " if label else ""
        if not drop_silent:
            raise RuntimeError(
                f"_normalize_per_neuron_01 {tag}: {len(bad_idx)} cell(s) "
                f"with zero range (indices="
                f"{bad_idx[:10]}{'...' if len(bad_idx) > 10 else ''}); "
                f"refusing to normalise without explicit drop_silent=True."
            )
        print(f"  *** _normalize_per_neuron_01 {tag}DROPPING "
              f"{len(bad_idx)} silent cell(s) with zero range "
              f"(indices={bad_idx[:10]}{'...' if len(bad_idx) > 10 else ''}); "
              f"keeping {int(np.sum(~bad_mask))}/{S.shape[0]} cells.",
              flush=True)
        keep = ~bad_mask
        S = S[keep, :]
        smin = smin[keep, :]
        rng = rng[keep, :]
        if S.shape[0] == 0:
            raise RuntimeError(
                f"_normalize_per_neuron_01 {tag}: all cells silent; "
                f"nothing left to normalise."
            )
    return (S - smin) / rng


def _downsample_factor() -> int:
    """Block-mean factor (frames) implied by DOWNSAMPLE_HZ vs MINISCOPE_FPS."""
    if MINISCOPE_FPS % DOWNSAMPLE_HZ != 0:
        raise RuntimeError(
            f"DOWNSAMPLE_HZ={DOWNSAMPLE_HZ} must evenly divide "
            f"MINISCOPE_FPS={MINISCOPE_FPS}"
        )
    return int(MINISCOPE_FPS // DOWNSAMPLE_HZ)


def _block_mean_rows(S: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a (cells, T) matrix along time by integer ``factor``.

    Trailing frames that don't fill a full block are dropped (paper does
    likewise when downsampling).
    """
    if factor == 1:
        return S
    N, T = S.shape
    T_keep = (T // factor) * factor
    if T_keep == 0:
        raise RuntimeError(
            f"_block_mean_rows: T={T} too short to downsample by {factor}"
        )
    S = S[:, :T_keep]
    return S.reshape(N, T_keep // factor, factor).mean(axis=2)


def _downsample_session(sess, S_norm: np.ndarray):
    """Return (S_ds, sess_ds) with frame-indexed event lists rescaled.

    ``sess_ds`` is a SimpleNamespace duck-typing ``sess`` for the attributes
    used downstream (tone_onsets, tone_offsets, shock_onsets, shock_offsets).
    Indices are integer-floored after scaling and clipped to [0, T_ds].
    """
    factor = _downsample_factor()
    if factor == 1:
        return S_norm, sess
    S_ds = _block_mean_rows(S_norm, factor)
    T_ds = S_ds.shape[1]

    def _scale(arr):
        if arr is None or len(arr) == 0:
            return list(arr) if arr is not None else []
        idx = np.asarray(arr, dtype=int) // factor
        idx = np.clip(idx, 0, T_ds)
        return idx.tolist()

    def _ds_pos(arr):
        """Block-mean a 1D position vector to length T_ds. Returns None
        if ``arr`` is missing/empty."""
        if arr is None:
            return None
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return None
        T_keep = (a.shape[0] // factor) * factor
        if T_keep == 0:
            return None
        a = a[:T_keep]
        return a.reshape(T_keep // factor, factor).mean(axis=1)

    def _pick_pos(*names):
        for n in names:
            v = getattr(sess, n, None)
            if v is not None and np.asarray(v).size > 0:
                return v
        return None

    sess_ds = SimpleNamespace(
        tone_onsets=_scale(getattr(sess, "tone_onsets", []) or []),
        tone_offsets=_scale(getattr(sess, "tone_offsets", []) or []),
        shock_onsets=_scale(getattr(sess, "shock_onsets", []) or []),
        shock_offsets=_scale(getattr(sess, "shock_offsets", []) or []),
        loc_X_miniscope=_ds_pos(_pick_pos("loc_X_miniscope_smooth",
                                          "loc_X_miniscope")),
        loc_Y_miniscope=_ds_pos(_pick_pos("loc_Y_miniscope_smooth",
                                          "loc_Y_miniscope")),
    )
    return S_ds, sess_ds


def _phase_index_from_session(sess, T: int) -> np.ndarray:
    """Build per-frame phase labels (length T).

    Categories (string ids):
      'pre_tone' (initial tail before first tone), 'tone', 'post_tone'
      (between tone offset and shock onset, or rest of trial if no shock),
      'shock', 'post_shock' (POST_SHOCK_SEC after shock offset),
      'isi' (everything else: ISIs after post_shock and final tail).
    """
    phase = np.array(["isi"] * T, dtype=object)

    tone_on = np.asarray(getattr(sess, "tone_onsets", []) or [], dtype=int)
    tone_off = np.asarray(getattr(sess, "tone_offsets", []) or [], dtype=int)
    shock_on = np.asarray(getattr(sess, "shock_onsets", []) or [], dtype=int)
    shock_off = np.asarray(getattr(sess, "shock_offsets", []) or [], dtype=int)
    post_shock_len = int(POST_SHOCK_SEC * DOWNSAMPLE_HZ)

    if len(tone_on) != len(tone_off):
        raise RuntimeError(
            f"tone_onsets ({len(tone_on)}) vs tone_offsets ({len(tone_off)}) mismatch"
        )
    if len(shock_on) != len(shock_off):
        raise RuntimeError(
            f"shock_onsets ({len(shock_on)}) vs shock_offsets ({len(shock_off)}) mismatch"
        )

    # Pre-tone tail = [0, first tone onset)
    if len(tone_on) > 0:
        phase[0:tone_on[0]] = "pre_tone"

    has_shock = len(shock_on) > 0
    for i, (t_on, t_off) in enumerate(zip(tone_on, tone_off)):
        phase[t_on:t_off] = "tone"
        if has_shock:
            s_on = shock_on[i]
            s_off = shock_off[i]
            phase[t_off:s_on] = "post_tone"
            phase[s_on:s_off] = "shock"
            ps_end = min(T, s_off + post_shock_len)
            phase[s_off:ps_end] = "post_shock"
            # remaining gap to next tone (or session end) stays 'isi'
        else:
            # no shocks → mark post_tone as up to next tone (or session end)
            next_on = tone_on[i + 1] if (i + 1) < len(tone_on) else T
            phase[t_off:next_on] = "post_tone"

    return phase


def _color_array_from_phase(phase_idx: np.ndarray) -> np.ndarray:
    """Map a length-T phase id array to an (T, 4) RGBA color array."""
    colors = np.zeros((len(phase_idx), 4), dtype=float)
    for ph, rgba in PHASE_COLORS.items():
        m = phase_idx == ph
        colors[m] = rgba
    return colors


def _phase_legend_handles():
    return [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=PHASE_COLORS[p], markeredgecolor='none',
               markersize=6, label=p)
        for p in PHASE_ORDER_FOR_LEGEND
    ]


# ---------------------------------------------------------------------------
# Isomap fit with NN sweep
# ---------------------------------------------------------------------------

# Module-level cache root, set by run_isomap_pipeline. Caches are tiny
# JSON files storing ONLY the NN sweep summary (chosen NN + errors).
# On a hit we skip the sweep and refit a single Isomap at nn_chosen —
# this avoids pickling multi-GB geodesic / kernel matrices.
_CACHE_DIR: str | None = None
_LOG_DIR: str | None = None
# Similarity cache: stores per-mouse JSON with all 6 row-lists
# (main + 4 per-trial + vs_position) so that on rerun we can skip the
# expensive iso refits + rotation sweeps and reload similarity
# DataFrames directly. Keyed by (mouse, group); invalidated when the
# independent-fit NN trio changes (which happens whenever the Y
# sidecar cache is invalidated).
_SIM_CACHE_DIR: str | None = None


def _sim_cache_path(mouse: str, group: str, *,
                    mode: str = "anchored") -> str | None:
    if _SIM_CACHE_DIR is None:
        return None
    suffix = "" if mode == "anchored" else f"_{mode}"
    return os.path.join(_SIM_CACHE_DIR,
                        f"{mouse}_{group}__similarity{suffix}.json")


def _independent_nn_signature(independent_results: dict) -> dict:
    """Per-session NN dict used as the similarity cache invalidation key."""
    return {s: int(p["NN"]) for s, p in independent_results.items()}


def _try_load_sim_cache(mouse: str, group: str, independent_results: dict, *,
                        mode: str = "anchored"):
    """Return cached similarity payload or None.

    On a NN-signature mismatch (i.e. the independent-fit NN sweep
    landed on a different NN than last time, meaning inputs changed)
    we treat it as a miss and force recomputation.
    """
    path = _sim_cache_path(mouse, group, mode=mode)
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    expected = _independent_nn_signature(independent_results)
    cached_sig = data.get("nn_signature")
    if cached_sig != expected:
        print(f"  [similarity cache:{mode}] {mouse}: NN signature mismatch "
              f"(cached={cached_sig}, current={expected}) -- recomputing",
              flush=True)
        return None
    # Forward-compat invalidation: if the set of per_trial keys cached
    # on disk doesn't match the trial_types the pipeline expects today,
    # force recomputation so newly added trial_types (e.g. "peri_tone")
    # actually get populated rather than silently defaulting to empty.
    expected_per_trial_keys = expected_per_trial_keys_for_mode()
    cached_keys = set((data.get("per_trial") or {}).keys())
    if expected_per_trial_keys is not None and cached_keys != expected_per_trial_keys:
        missing = expected_per_trial_keys - cached_keys
        extra = cached_keys - expected_per_trial_keys
        print(f"  [similarity cache:{mode}] {mouse}: per_trial key mismatch "
              f"(missing={sorted(missing)}, extra={sorted(extra)}) -- recomputing",
              flush=True)
        return None
    print(f"  [similarity cache:{mode}] {mouse}: HIT -> reusing similarity rows "
          f"from {path}", flush=True)
    return data


# Set by run_isomap_pipeline so cache loaders can validate that on-disk
# per_trial keys match what the current pipeline expects to populate.
_EXPECTED_PER_TRIAL_KEYS: frozenset[str] | None = None


def expected_per_trial_keys_for_mode() -> frozenset[str] | None:
    return _EXPECTED_PER_TRIAL_KEYS


def _save_sim_cache(mouse: str, group: str, *, independent_results: dict,
                    main_rows: list, per_trial_rows: dict,
                    vs_pos_rows: list,
                    mode: str = "anchored") -> None:
    path = _sim_cache_path(mouse, group, mode=mode)
    if path is None:
        return
    os.makedirs(_SIM_CACHE_DIR, exist_ok=True)
    payload = {
        "mouse": mouse,
        "group": group,
        "mode": mode,
        "nn_signature": _independent_nn_signature(independent_results),
        "main": list(main_rows),
        "per_trial": {tt: list(rows) for tt, rows in per_trial_rows.items()},
        "vs_position": list(vs_pos_rows),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    print(f"  [similarity cache:{mode}] {mouse}: saved -> {path}", flush=True)


def _isomap_cache_path(X: np.ndarray, label: str) -> str | None:
    """Return cache file path for a given (X, label) or None if disabled."""
    if _CACHE_DIR is None:
        return None
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.array(X.shape, dtype=np.int64).tobytes())
    params = (
        ISOMAP_DIMS, NN_STEPS, NN_RANGE[0], NN_RANGE[1],
        round(NN_CUTOFF, 6), DOWNSAMPLE_HZ, MINISCOPE_FPS,
    )
    h.update(repr(params).encode("utf-8"))
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in label)
    digest = h.hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{safe}__{digest}.json")


def _try_load_nn_summary(cache_path: str | None):
    """Load the tiny JSON sweep summary if present.

    Returns dict with keys ``nn_chosen``, ``nn_values``, ``errors``, or
    None on miss / unreadable / schema mismatch.
    """
    if cache_path is None or not os.path.isfile(cache_path):
        return None
    with open(cache_path, "r") as f:
        obj = json.load(f)
    required = {"nn_chosen", "nn_values", "errors"}
    if not required.issubset(obj.keys()):
        raise RuntimeError(
            f"_try_load_nn_summary: cache {cache_path} missing keys "
            f"{required - set(obj.keys())}"
        )
    return obj


def _save_nn_summary(cache_path: str | None, payload: dict):
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, cache_path)
    print(f"  [Isomap cache] saved NN summary -> {cache_path}", flush=True)


# Embedding sidecar (Y.npy alongside the JSON). Storing the embedding
# itself lets us skip the per-cache-hit Isomap refit entirely — which
# is the dominant cost (Dijkstra geodesic + dense eigendecomp ~30-60s
# per fit at T~10k). The sidecar is tiny (T × ISOMAP_DIMS × 8 bytes,
# ~200 KB at T=12,827 / d=2) so well under any disk budget.

def _y_sidecar_path(cache_path: str | None) -> str | None:
    if cache_path is None:
        return None
    if not cache_path.endswith(".json"):
        raise RuntimeError(
            f"_y_sidecar_path: unexpected cache path {cache_path!r} "
            f"(must end in .json)"
        )
    return cache_path[:-len(".json")] + "__Y.npy"


def _save_y_sidecar(cache_path: str | None, Y: np.ndarray) -> None:
    sidecar = _y_sidecar_path(cache_path)
    if sidecar is None:
        return
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    tmp = sidecar + ".tmp"
    np.save(tmp, np.ascontiguousarray(Y, dtype=np.float64))
    # np.save appends .npy if missing; rename the actual file written.
    written = tmp if os.path.exists(tmp) else tmp + ".npy"
    os.replace(written, sidecar)
    print(f"  [Isomap cache] saved Y sidecar -> {sidecar} "
          f"(shape={Y.shape})", flush=True)


def _try_load_y_sidecar(cache_path: str | None) -> np.ndarray | None:
    sidecar = _y_sidecar_path(cache_path)
    if sidecar is None or not os.path.isfile(sidecar):
        return None
    Y = np.load(sidecar)
    if Y.ndim != 2 or Y.shape[1] != ISOMAP_DIMS:
        raise RuntimeError(
            f"_try_load_y_sidecar: sidecar {sidecar} has shape {Y.shape}, "
            f"expected (T, {ISOMAP_DIMS})"
        )
    return Y


# Note: we deliberately do NOT persist the fitted Isomap estimator.
# At T~12,827 sklearn's Isomap holds a dense ``dist_matrix_`` of T*T*8
# bytes ~= 1.3 GB per session, blowing through both disk budget and
# RAM. The similarity computation instead refits at p['NN'] on first iso.transform() use
# (cost ~50-110s per session, paid once per pipeline run); the iso is
# then dropped immediately when the mouse finishes (see
# run_isomap_pipeline) so peak memory stays bounded by one mouse's
# worth of fitted Isomaps rather than the whole cohort.


def _canonicalize_isomap_signs(iso) -> None:
    """Force a deterministic sign convention on Isomap eigenpairs.

    ARPACK eigenvectors come out with arbitrary signs (the iteration
    starts from a random vector). For each axis, we flip the sign so
    that the sample with the largest absolute value has a positive
    coordinate. The flip is applied in-place to both ``iso.embedding_``
    and the underlying ``iso.kernel_pca_.eigenvectors_`` so subsequent
    ``iso.transform(X_new)`` calls inherit the same orientation — this
    is what keeps the crossreg fit's Test_B / Test_B_1wk projections aligned to
    the TFC_cond reference manifold across reruns.
    """
    Y = iso.embedding_
    eigvecs = iso.kernel_pca_.eigenvectors_
    for k in range(Y.shape[1]):
        idx = int(np.argmax(np.abs(Y[:, k])))
        if Y[idx, k] < 0:
            Y[:, k] *= -1.0
            eigvecs[:, k] *= -1.0


def _fit_isomap_with_nn_search(X: np.ndarray, *, label: str = "",
                               precomputed: bool = False):
    """Fit Isomap with a NN sweep; stop when error plateaus.

    Caches only the NN sweep summary (chosen NN + per-NN errors) as a
    small JSON file under ``_CACHE_DIR``. On a hit, the sweep is
    skipped and a single Isomap is refit at ``nn_chosen`` — the
    Isomap object itself (which holds T×T geodesic / kernel matrices)
    is never pickled.

    Parameters
    ----------
    X : (T, N) coordinate matrix, OR (T, T) precomputed pairwise
        distance matrix when ``precomputed=True``.
    label : human-readable + path-safe identifier (used for cache filename
            and print messages).
    precomputed : if True, ``X`` is interpreted as a (T, T) distance
        matrix and the inner Isomap fits use ``metric='precomputed'``.
        This is the path used by the independent fit to share one set
        of pairwise distances across the entire NN sweep — the dominant
        speedup. The crossreg fit keeps coordinate input because it must
        call ``iso.transform(X_new)`` on test sessions.
    -------
    iso, Y, nn_chosen, nn_values, errors
    """
    if X.ndim != 2:
        raise ValueError(f"_fit_isomap_with_nn_search expects 2D, got {X.shape}")
    n_samples = X.shape[0]
    if precomputed and X.shape[0] != X.shape[1]:
        raise ValueError(
            f"_fit_isomap_with_nn_search[{label}]: precomputed=True needs "
            f"square distance matrix, got {X.shape}"
        )
    if n_samples < NN_RANGE[0] + 1:
        raise RuntimeError(
            f"_fit_isomap_with_nn_search[{label}]: only {n_samples} samples; "
            f"need at least {NN_RANGE[0] + 1}."
        )

    metric = "precomputed" if precomputed else "minkowski"

    def _new_isomap(nn: int) -> Isomap:
        return Isomap(
            n_neighbors=nn,
            n_components=ISOMAP_DIMS,
            n_jobs=ISOMAP_N_JOBS_INNER,
            eigen_solver=ISOMAP_EIGEN_SOLVER,
            metric=metric,
        )

    cache_path = _isomap_cache_path(X, label)
    cached = _try_load_nn_summary(cache_path)
    if cached is not None:
        nn_chosen = int(cached["nn_chosen"])
        nn_values = [int(v) for v in cached["nn_values"]]
        errors = [float(v) for v in cached["errors"]]
        Y_cached = _try_load_y_sidecar(cache_path)
        if Y_cached is not None:
            if Y_cached.shape[0] != n_samples:
                raise RuntimeError(
                    f"_fit_isomap_with_nn_search[{label}]: Y sidecar T="
                    f"{Y_cached.shape[0]} but X T={n_samples}"
                )
            print(f"  [Isomap {label}] CACHE HIT (Y sidecar) -> NN={nn_chosen} "
                  f"(skipping refit, sweep length={len(errors)})",
                  flush=True)
            # Independent-fit callers do not use `iso` downstream. The
            # crossreg fit has its own higher-level cache and will not
            # reach this path on hit. Returning iso=None here is
            # intentional and safe; _ensure_independent_iso() lazily
            # refits when the similarity computation needs
            # iso.transform().
            return None, Y_cached.copy(), nn_chosen, nn_values, errors, cache_path
        print(f"  [Isomap {label}] CACHE HIT (legacy, no Y sidecar) -> NN="
              f"{nn_chosen} (refitting single Isomap from sweep length="
              f"{len(errors)})", flush=True)
        iso = _new_isomap(nn_chosen)
        iso.fit(X)
        _canonicalize_isomap_signs(iso)
        Y = np.asarray(iso.embedding_).copy()
        _save_y_sidecar(cache_path, Y)
        return iso, Y, nn_chosen, nn_values, errors, cache_path

    nn_values = list(range(NN_RANGE[0], NN_RANGE[1] + 1, NN_STEPS))
    nn_values = [nn for nn in nn_values if nn < n_samples]
    if not nn_values:
        raise RuntimeError(
            f"_fit_isomap_with_nn_search[{label}]: no valid NN values for "
            f"n_samples={n_samples}"
        )

    def _fit_one(nn):
        iso = _new_isomap(nn)
        t_fit = time.time()
        iso.fit(X)
        fit_dt = time.time() - t_fit
        _canonicalize_isomap_signs(iso)
        err = float(iso.reconstruction_error())
        return iso, err, fit_dt

    requested_workers = (os.cpu_count() or 1) if ISOMAP_N_JOBS_OUTER == -1 \
        else ISOMAP_N_JOBS_OUTER
    T_samples = X.shape[0]
    n_workers = _cap_workers_for_memory(requested_workers, T_samples)
    per_fit_gb = _estimate_per_fit_gb(T_samples)
    if n_workers != requested_workers:
        print(f"  [Isomap {label}] memory cap: requested "
              f"{requested_workers} workers, capped to {n_workers} "
              f"(T={T_samples}, ~{per_fit_gb:.2f} GB/fit, budget="
              f"{ISOMAP_MEM_BUDGET_GB:.1f} GB)", flush=True)
    print(f"  [Isomap {label}] sweeping {len(nn_values)} NN values "
          f"(workers={n_workers}, inner n_jobs="
          f"{ISOMAP_N_JOBS_INNER}, ~{per_fit_gb:.2f} GB/fit)", flush=True)

    # Per-sweep verbose log: NN, error, % improvement vs previous,
    # plateau counter, and the final plateau decision. Written to
    # <_LOG_DIR>/<label>__nn_sweep.log so each sweep is self-documenting.
    log_lines: list[str] = []
    log_lines.append(
        f"# Isomap NN sweep: {label}\n"
        f"# T={T_samples}  X.shape={tuple(X.shape)}  "
        f"NN_RANGE={NN_RANGE}  NN_STEPS={NN_STEPS}  "
        f"NN_CUTOFF={NN_CUTOFF}\n"
        f"# Plateau rule (paper methods): the algorithm ceases when "
        f"the improvement in reconstruction error is <{NN_CUTOFF*100:.0f}% "
        f"for two consecutive NNs. Chosen NN = the NN at which the "
        f"algorithm ceased (i.e. the second-consecutive-plateau NN).\n"
    )
    print(f"  [Isomap {label}] {log_lines[0].rstrip()}", flush=True)

    # Online plateau detection: pick the FIRST plateau (matches the
    # original post-hoc logic) and stop dispatching new fits once the
    # decision is locked in. Saves the tail of the sweep when we
    # already know nn_chosen.
    fits: list = []
    errors: list = []
    consec_plateau = 0
    chosen_idx: int | None = None
    stopped_early = False

    def _record(nn_idx: int, nn: int, iso_obj, err: float, fit_dt: float):
        """Append a fit, update plateau state, log + print verbosely."""
        nonlocal consec_plateau, chosen_idx
        if nn_idx != len(errors):
            raise RuntimeError(
                f"_record: nn_idx={nn_idx} but errors len={len(errors)} "
                f"— sweep dispatched out of order"
            )
        fits.append(iso_obj)
        errors.append(err)
        if nn_idx == 0:
            improvement = float("nan")
            plateau_hit = False
            line = (f"  [Isomap {label}] NN={nn:3d}  err={err:.4g}  "
                    f"fit={fit_dt:.1f}s  (baseline)")
        else:
            prev = errors[nn_idx - 1]
            improvement = (prev - err) / prev if prev > 0 else 0.0
            plateau_hit = improvement < NN_CUTOFF
            if plateau_hit:
                consec_plateau += 1
            else:
                consec_plateau = 0
            line = (f"  [Isomap {label}] NN={nn:3d}  err={err:.4g}  "
                    f"fit={fit_dt:.1f}s  improvement={improvement*100:+.1f}%  "
                    f"plateau={'YES' if plateau_hit else 'no'}  "
                    f"consec_plateau={consec_plateau}/2")
        print(line, flush=True)
        log_lines.append(line + "\n")

        if chosen_idx is None and consec_plateau >= 2:
            # Methods: "the algorithm ceased when the improvement in
            # reconstruction error plateaued, defined as <10% for two
            # consecutive NNs". The NN at which the algorithm CEASED is
            # the second-consecutive-plateau NN — i.e. the current one.
            chosen_idx = nn_idx
            decision = (
                f"  [Isomap {label}] PLATEAU REACHED at NN={nn} "
                f"(consec_plateau=2): improvement was <{NN_CUTOFF*100:.0f}% "
                f"for two consecutive NNs (NN={nn_values[nn_idx - 1]} and "
                f"NN={nn}). Algorithm ceased here; choosing NN={nn} "
                f"(err={err:.4g})."
            )
            print(decision, flush=True)
            log_lines.append(decision + "\n")

    if n_workers == 1:
        for nn_idx, nn in enumerate(nn_values):
            iso_obj, err, fit_dt = _fit_one(nn)
            _record(nn_idx, nn, iso_obj, err, fit_dt)
            if chosen_idx is not None:
                stopped_early = True
                msg = (f"  [Isomap {label}] EARLY STOP: skipping remaining "
                       f"{len(nn_values) - (nn_idx + 1)} NN values "
                       f"({nn_values[nn_idx + 1:]}).")
                print(msg, flush=True)
                log_lines.append(msg + "\n")
                break
            _abort_countdown()
    else:
        # Dispatch in chunks of n_workers so we can prompt for abort
        # between batches AND so plateau detection between batches can
        # short-circuit dispatching the rest of the sweep.
        chunk = max(1, n_workers)
        nn_idx = 0
        with Parallel(n_jobs=n_workers, backend="loky",
                      verbose=5) as parallel:
            for start in range(0, len(nn_values), chunk):
                batch = nn_values[start:start + chunk]
                batch_res = parallel(delayed(_fit_one)(nn) for nn in batch)
                for nn, (iso_obj, err, fit_dt) in zip(batch, batch_res):
                    _record(nn_idx, nn, iso_obj, err, fit_dt)
                    nn_idx += 1
                    if chosen_idx is not None:
                        stopped_early = True
                        break
                if chosen_idx is not None:
                    remaining = nn_values[nn_idx:]
                    if remaining:
                        msg = (f"  [Isomap {label}] EARLY STOP: skipping "
                               f"remaining {len(remaining)} NN values "
                               f"({remaining}).")
                        print(msg, flush=True)
                        log_lines.append(msg + "\n")
                    break
                if start + chunk < len(nn_values):
                    _abort_countdown()
        _abort_countdown()

    if not errors:
        raise RuntimeError(
            f"_fit_isomap_with_nn_search[{label}]: sweep produced no fits"
        )

    if chosen_idx is None:
        # No plateau hit within the sweep — pick the NN with lowest error
        # (which is the last one if monotone decreasing, otherwise the
        # global min). This matches the original "fall through" fallback.
        chosen_idx = int(np.argmin(np.asarray(errors)))
        decision = (
            f"  [Isomap {label}] NO PLATEAU within sweep "
            f"({len(errors)}/{len(nn_values)} NN values fit); choosing "
            f"NN={nn_values[chosen_idx]} (err={errors[chosen_idx]:.4g}) "
            f"as the lowest-error fit."
        )
        print(decision, flush=True)
        log_lines.append(decision + "\n")

    iso = fits[chosen_idx]
    # Canonicalization was already applied to every sweep fit, so the
    # chosen iso is already in canonical orientation.
    Y = np.asarray(iso.embedding_).copy()
    # Truncate nn_values to only those actually fit (early-stop case)
    # so the cached/returned arrays line up 1-to-1 with `errors`.
    nn_values_run = nn_values[:len(errors)]
    nn_chosen = nn_values_run[chosen_idx]
    summary = (
        f"  [Isomap {label}] -> chose NN={nn_chosen} "
        f"(err={errors[chosen_idx]:.4g}, sweep length={len(errors)}/"
        f"{len(nn_values)}, early_stop={stopped_early})"
    )
    print(summary, flush=True)
    log_lines.append(summary + "\n")

    # Persist the verbose log to <_LOG_DIR>/<safe_label>__nn_sweep.log so
    # each sweep decision is self-documenting and reproducible without
    # having to re-run the pipeline.
    if _LOG_DIR is not None:
        safe = "".join(c if (c.isalnum() or c in "._-") else "_"
                       for c in label)
        log_path = os.path.join(_LOG_DIR, f"{safe}__nn_sweep.log")
        os.makedirs(_LOG_DIR, exist_ok=True)
        tmp = log_path + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(log_lines)
        os.replace(tmp, log_path)
        print(f"  [Isomap {label}] wrote sweep log -> {log_path}",
              flush=True)

    _save_nn_summary(cache_path, {
        "label": label,
        "X_shape": list(X.shape),
        "nn_chosen": int(nn_chosen),
        "nn_values": [int(v) for v in nn_values_run],
        "errors": [float(v) for v in errors],
        "stopped_early": bool(stopped_early),
        "nn_values_planned": [int(v) for v in nn_values],
        "params": {
            "ISOMAP_DIMS": ISOMAP_DIMS,
            "NN_STEPS": NN_STEPS,
            "NN_RANGE": list(NN_RANGE),
            "NN_CUTOFF": NN_CUTOFF,
            "DOWNSAMPLE_HZ": DOWNSAMPLE_HZ,
            "MINISCOPE_FPS": MINISCOPE_FPS,
        },
    })
    _save_y_sidecar(cache_path, Y)
    return iso, Y, nn_chosen, nn_values_run, errors, cache_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_by_phase(ax, Y, phase_idx, *, base_size=4, marker='o',
                      label_session: str | None = None):
    """Scatter points grouped by phase so per-phase size + alpha apply.

    Returns the list of PathCollection artists (mainly so the caller can
    set zorder if needed). ``label_session`` is unused for legend (the
    legend is composed elsewhere) but kept for future hooks.
    """
    for ph in PHASE_DRAW_ORDER:
        m = (phase_idx == ph)
        if not np.any(m):
            continue
        rgba = PHASE_COLORS[ph]
        s = base_size * PHASE_SIZE_MULT.get(ph, 1.0)
        ax.scatter(Y[m, 0], Y[m, 1],
                   c=[rgba], s=s, marker=marker, linewidths=0)


def _add_corner_scalebar(ax, length_data: float, *, loc='upper left',
                         label: str | None = None):
    """Draw a small L-shaped scalebar (x-arm + y-arm of equal data length)
    in a corner of ``ax``. Replaces full axes / ticks for the side-by-side
    panels where we just need a unit indicator.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    pad_x = xspan * 0.04
    pad_y = yspan * 0.04
    if loc == 'upper left':
        x0 = xlim[0] + pad_x
        y0 = ylim[1] - pad_y
        x1 = x0 + length_data
        y1 = y0 - length_data
        ha = 'left'
    elif loc == 'lower left':
        x0 = xlim[0] + pad_x
        y0 = ylim[0] + pad_y + length_data
        x1 = x0 + length_data
        y1 = y0 - length_data
        ha = 'left'
    else:
        raise ValueError(f"_add_corner_scalebar: unsupported loc={loc!r}")
    ax.plot([x0, x1], [y0, y0], color='black', lw=1.4, solid_capstyle='butt')
    ax.plot([x0, x0], [y0, y1], color='black', lw=1.4, solid_capstyle='butt')
    if label:
        ax.text(x0 + length_data * 0.5, y0 + yspan * 0.012, label,
                ha='center', va='bottom', fontsize=7)


def _plot_recon_error(nn_values, errors, nn_chosen, mouse, sess_name, save_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(nn_values, errors, marker='o', color='black')
    ax.axvline(nn_chosen, color='red', ls='--', label=f"chosen NN={nn_chosen}")
    ax.set_xlabel("Nearest neighbours (NN)")
    ax.set_ylabel("Isomap reconstruction error")
    ax.set_title(f"{mouse} — {sess_name} — recon error")
    ax.legend()
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"recon_error_{sess_name}.png"), dpi=150)
    plt.close(fig)


def _plot_manifold_2d(Y, phase_idx, title, fname, save_dir, *,
                      marker_size=4, label_targets=()):
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_by_phase(ax, Y, phase_idx, base_size=marker_size)
    if label_targets:
        _label_trial_arms(ax, Y, phase_idx, targets=label_targets)
    ax.set_xlabel("Isomap 1")
    ax.set_ylabel("Isomap 2")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(handles=_phase_legend_handles(), loc='best', fontsize=7,
              frameon=True, ncol=2)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _label_trial_arms(ax, Y, phase_idx, *, targets=TRIAL_ARM_TARGETS_BASE,
                      line_kwargs=None, fontsize=None):
    """Annotate each trial of selected phases on a 2D embedding axis.

    ``targets`` is an iterable of ``(prefix, phase_name)`` pairs (e.g.
    ``[("t","tone"), ("s","shock")]``). Identifies contiguous runs of
    each named phase in ``phase_idx`` as separate trials, draws a thin
    time-ordered polyline through that trial's embedded frames, and
    places a ``{prefix}{k}`` label at the point of maximal distance from
    the embedding centroid (i.e. the "arm tip" if the trial pulls out
    one direction).

    No-op when fewer than 2 frames are available for a trial.
    """
    Y = np.asarray(Y)
    phase = np.asarray(phase_idx)
    if Y.shape[0] != phase.shape[0]:
        raise RuntimeError(
            f"_label_trial_arms: Y/phase length mismatch {Y.shape[0]} vs {phase.shape[0]}"
        )
    if line_kwargs is None:
        line_kwargs = TRIAL_ARM_LINE_KW
    if fontsize is None:
        fontsize = TRIAL_ARM_LABEL_FONTSIZE
    if not targets:
        return
    centroid = Y.mean(axis=0)
    for label_prefix, target in targets:
        is_target = (phase == target)
        if not np.any(is_target):
            continue
        # Find boundaries of contiguous True runs.
        diff = np.diff(is_target.astype(np.int8))
        starts = list(np.where(diff == 1)[0] + 1)
        ends = list(np.where(diff == -1)[0] + 1)
        if is_target[0]:
            starts.insert(0, 0)
        if is_target[-1]:
            ends.append(len(is_target))
        for k, (s, e) in enumerate(zip(starts, ends), start=1):
            sub = Y[s:e]
            if sub.shape[0] < 2:
                continue
            ax.plot(sub[:, 0], sub[:, 1], **line_kwargs)
            d = np.linalg.norm(sub - centroid, axis=1)
            j = int(np.argmax(d))
            ax.text(sub[j, 0], sub[j, 1], f"{label_prefix}{k}",
                    fontsize=fontsize, color="black", weight="bold",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.75))


def _plot_manifold_grid(Ys_by_session, phases_by_session, mouse, group, save_dir,
                      *, marker_size=4, fname="manifolds_grid_indep.png",
                      title_tag="Independent manifolds",
                      label_targets=()):
    """Side-by-side per-session manifolds with a SHARED data scale.

    Axes ticks/spines are removed; a small L-shaped scalebar in the top-
    left of each panel indicates units (length = nice round number near
    20% of the shared x-range).

    ``fname`` and ``title_tag`` let callers reuse this helper for the
    crossreg grid (where panels share an Isomap basis by construction
    and so are aligned without any rotation).
    """
    sess_present = [s for s in SESSION_ORDER if s in Ys_by_session]
    if not sess_present:
        return

    # Shared data range across sessions (square aspect, equal extent on
    # both axes so visual sizes are comparable across panels).
    all_xy = np.concatenate([Ys_by_session[s] for s in sess_present], axis=0)
    x_lo, x_hi = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_lo, y_hi = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    pad = 0.05 * max(x_hi - x_lo, y_hi - y_lo)
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo) + pad
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    # Pick a 1-2-5 nice scalebar length ~20% of the axis range.
    raw = 0.20 * (xlim[1] - xlim[0])
    exp = np.floor(np.log10(raw))
    base = raw / (10 ** exp)
    nice = 1.0 if base < 1.5 else (2.0 if base < 3.5 else 5.0)
    sb_len = nice * (10 ** exp)

    n = len(sess_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.4),
                             squeeze=False)
    axes = axes[0]
    for ax, sess_name in zip(axes, sess_present):
        Y = Ys_by_session[sess_name]
        ph = phases_by_session[sess_name]
        _scatter_by_phase(ax, Y, ph, base_size=marker_size)
        if label_targets:
            _label_trial_arms(ax, Y, ph, targets=label_targets)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(sess_name, fontsize=11)
        # Strip axis chrome — the scalebar carries the unit info.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Only TFC_cond gets the scalebar (shared scale across panels).
        if sess_name == "TFC_cond":
            _add_corner_scalebar(ax, sb_len, loc='upper left',
                                 label=f"{sb_len:g}")

    # One shared phase legend at the top.
    fig.legend(handles=_phase_legend_handles(),
               loc='upper center', ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"{mouse} ({GROUP_LABEL[group]}) — {title_tag} "
                 f"(shared scale; scalebar = {sb_len:g} Isomap units)",
                 fontsize=11, y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _plot_manifold_grid_aligned(Ys_by_session, phases_by_session, deg_by_session,
                              mouse, group, save_dir, *, marker_size=4,
                              fname="manifolds_grid_indep_aligned.png",
                              label_targets=()):
    """Independent-manifold grid with each non-TFC_cond session rotated
    by its similarity-derived ``best_deg`` into the TFC_cond reference frame.

    ``best_deg`` is the rotation that maximised cosine/Pearson similarity
    between the trial-averaged trajectories (see ``_similarity_after_rotation``).
    Applying it to the *full* embedding Y is approximate — the optimum was
    found on the trial-averaged trajectory only — but it removes the
    arbitrary per-fit rotation so trajectory directionality is visually
    comparable across panels. Purely cosmetic; does not affect the
    similarity computation.

    TFC_cond is plotted unrotated (it is the reference).
    """
    sess_present = [s for s in SESSION_ORDER if s in Ys_by_session]
    if not sess_present:
        return

    # Apply rotations into a copy so caller's dict is untouched.
    Ys_rot = {}
    for s in sess_present:
        deg = 0.0 if s == "TFC_cond" else float(deg_by_session.get(s, 0.0))
        Ys_rot[s] = _rotate_2d(np.asarray(Ys_by_session[s]), deg)

    # Shared square data range across (rotated) sessions.
    all_xy = np.concatenate([Ys_rot[s] for s in sess_present], axis=0)
    x_lo, x_hi = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_lo, y_hi = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    pad = 0.05 * max(x_hi - x_lo, y_hi - y_lo)
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo) + pad
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    raw = 0.20 * (xlim[1] - xlim[0])
    exp = np.floor(np.log10(raw))
    base = raw / (10 ** exp)
    nice = 1.0 if base < 1.5 else (2.0 if base < 3.5 else 5.0)
    sb_len = nice * (10 ** exp)

    n = len(sess_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.4), squeeze=False)
    axes = axes[0]
    for ax, sess_name in zip(axes, sess_present):
        Y = Ys_rot[sess_name]
        ph = phases_by_session[sess_name]
        _scatter_by_phase(ax, Y, ph, base_size=marker_size)
        if label_targets:
            _label_trial_arms(ax, Y, ph, targets=label_targets)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        if sess_name == "TFC_cond":
            ax.set_title(f"{sess_name} (reference)", fontsize=11)
        else:
            deg = float(deg_by_session.get(sess_name, 0.0))
            ax.set_title(f"{sess_name} (rotated {deg:.0f}\u00b0)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if sess_name == "TFC_cond":
            _add_corner_scalebar(ax, sb_len, loc='upper left',
                                 label=f"{sb_len:g}")

    fig.legend(handles=_phase_legend_handles(),
               loc='upper center', ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"{mouse} ({GROUP_LABEL[group]}) — Independent manifolds, "
                 f"similarity-aligned (shared scale; scalebar = {sb_len:g})",
                 fontsize=11, y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _plot_manifold_overlay(Ys_by_session, phases_by_session, mouse, save_dir,
                            *, fname="manifold_overlay_crossreg.png",
                            label_targets=()):
    """Overlay multiple sessions in a single 2D plot.

    Per-session marker shapes distinguish sessions; phase colours match
    the per-session plots.
    """
    markers = {"TFC_cond": "o", "Test_B": "s", "Test_B_1wk": "^"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for sess_name in SESSION_ORDER:
        if sess_name not in Ys_by_session:
            continue
        Y = Ys_by_session[sess_name]
        phase_idx = phases_by_session[sess_name]
        _scatter_by_phase(ax, Y, phase_idx, base_size=4,
                          marker=markers.get(sess_name, 'o'))
        if label_targets:
            _label_trial_arms(ax, Y, phase_idx, targets=label_targets)
    ax.set_xlabel("Isomap 1 (TFC_cond reference)")
    ax.set_ylabel("Isomap 2 (TFC_cond reference)")
    ax.set_title(f"{mouse} — crossreg manifold overlay")
    ax.set_aspect('equal', adjustable='datalim')

    sess_handles = [
        Line2D([0], [0], marker=markers[s], linestyle='',
               markerfacecolor='lightgrey', markeredgecolor='black',
               markersize=7, label=s)
        for s in SESSION_ORDER if s in Ys_by_session
    ]
    leg1 = ax.legend(handles=sess_handles, loc='upper left', fontsize=8,
                     title='session', frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=_phase_legend_handles(), loc='upper right', fontsize=7,
              title='phase', frameon=True, ncol=2)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Independent manifolds — per-session Isomap fits (no cross-session
# coupling)
# ---------------------------------------------------------------------------

def _fit_independent_manifolds(mouse, group, sessions_by_name, save_dir):
    """Returns dict[sess_name] -> {'Y': (T,2), 'phase': str-array, 'NN': int,
                                   'sess': sess, 'S_norm': (N,T)}."""
    results = {}
    for sess_name in SESSION_ORDER:
        sess = sessions_by_name.get(sess_name)
        if sess is None:
            print(f"  [independent] {mouse}: missing {sess_name} — skipping", flush=True)
            continue
        S = np.asarray(sess.S, dtype=float)        # (N, T)
        S_norm = _normalize_per_neuron_01(
            S, label=f"{mouse}/{sess_name}", drop_silent=True,
        )
        S_ds, sess_ds = _downsample_session(sess, S_norm)
        T = S_ds.shape[1]
        X = S_ds.T                                  # (T_ds, N)
        print(f"  [independent] {mouse}/{sess_name}: T_native={S_norm.shape[1]} "
              f"T_ds={T} N={S.shape[0]} (DS factor={_downsample_factor()})",
              flush=True)
        # Pass coordinate X (NOT precomputed D). With metric='minkowski'
        # sklearn uses BallTree/KDTree for the kNN step, parallelised
        # via INNER n_jobs across all logical processors — this is the
        # main source of multi-core utilisation in the independent fit. Switching
        # to a precomputed distance matrix bypasses that path and forces
        # single-threaded argpartition kNN extraction, which empirically
        # killed parallelism without saving meaningful wall time.
        iso, Y, nn_chosen, nn_values, errors, cache_path = \
            _fit_isomap_with_nn_search(
                X, label=f"{mouse}/{sess_name}/indep"
            )
        phase_idx = _phase_index_from_session(sess_ds, T)

        _plot_recon_error(nn_values, errors, nn_chosen, mouse, sess_name, save_dir)
        for suffix, targets in _arm_label_variants():
            _plot_manifold_2d(
                Y, phase_idx,
                title=f"{mouse} ({GROUP_LABEL[group]}) — {sess_name} (NN={nn_chosen})",
                fname=_with_suffix(f"manifold_{sess_name}_indep.png", suffix),
                save_dir=save_dir,
                label_targets=targets,
            )

        results[sess_name] = {
            "iso": iso, "Y": Y, "phase": phase_idx, "NN": nn_chosen,
            "sess": sess, "sess_ds": sess_ds,
            "S_norm": S_norm, "S_ds": S_ds,
            "nn_values": nn_values, "errors": errors,
            "cache_path": cache_path,
        }
    # Side-by-side grid of all available sessions on a SHARED scale.
    Ys_grid = {s: results[s]["Y"] for s in results}
    phases_grid = {s: results[s]["phase"] for s in results}
    for suffix, targets in _arm_label_variants():
        _plot_manifold_grid(
            Ys_grid, phases_grid, mouse, group, save_dir,
            fname=_with_suffix("manifolds_grid_indep.png", suffix),
            label_targets=targets,
        )
    return results


# ---------------------------------------------------------------------------
# Crossreg fit — single Isomap on TFC_cond crossreg cells; project
# Test_B / Test_B_1wk into that frame via iso.transform().
# ---------------------------------------------------------------------------

def _crossreg_cache_paths(mouse: str, X_by_name: dict, idx_basis: np.ndarray,
                          *, anchor_session: str = "TFC_cond"):
    """Return (json_path, {sess_name: y_npy_path}, {sess_name: phase_npy_path}).

    Cache key hashes the per-session normalised + downsampled X arrays
    (the actual Isomap inputs) plus the crossreg cell indices, the
    anchor_session, and sweep params. None if caching disabled.
    """
    if _CACHE_DIR is None:
        return None, {}, {}
    h = hashlib.sha1()
    h.update(anchor_session.encode("utf-8"))
    h.update(np.ascontiguousarray(idx_basis).tobytes())
    h.update(np.array(idx_basis.shape, dtype=np.int64).tobytes())
    for name in sorted(X_by_name.keys()):
        h.update(name.encode("utf-8"))
        X = X_by_name[name]
        h.update(np.ascontiguousarray(X).tobytes())
        h.update(np.array(X.shape, dtype=np.int64).tobytes())
    params = (
        ISOMAP_DIMS, NN_STEPS, NN_RANGE[0], NN_RANGE[1],
        round(NN_CUTOFF, 6), DOWNSAMPLE_HZ, MINISCOPE_FPS,
    )
    h.update(repr(params).encode("utf-8"))
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in mouse)
    safe_basis = "".join(c if (c.isalnum() or c in "._-") else "_"
                         for c in anchor_session)
    digest = h.hexdigest()[:16]
    base = os.path.join(_CACHE_DIR,
                        f"crossreg_basis_{safe_basis}__{safe}__{digest}")
    json_path = base + ".json"
    y_paths = {name: f"{base}__Y_{name}.npy" for name in X_by_name}
    phase_paths = {name: f"{base}__phase_{name}.npy" for name in X_by_name}
    return json_path, y_paths, phase_paths


def _try_load_crossreg_cache(json_path, y_paths, phase_paths):
    """Load cached crossreg-fit result. Return dict or None on miss.

    Hard-fails if the JSON is present but sidecars are missing /
    inconsistent (rather than silently refalling-through to a refit).
    """
    if json_path is None or not os.path.isfile(json_path):
        return None
    with open(json_path, "r") as f:
        meta = json.load(f)
    required = {"sessions", "nn_chosen", "nn_values", "errors", "n_cells"}
    if not required.issubset(meta.keys()):
        raise RuntimeError(
            f"_try_load_crossreg_cache: {json_path} missing keys "
            f"{required - set(meta.keys())}"
        )
    Ys = {}
    phases = {}
    for name in meta["sessions"]:
        yp = y_paths.get(name)
        pp = phase_paths.get(name)
        if yp is None or pp is None:
            raise RuntimeError(
                f"_try_load_crossreg_cache: no path entry for session {name!r}"
            )
        if not (os.path.isfile(yp) and os.path.isfile(pp)):
            raise RuntimeError(
                f"_try_load_crossreg_cache: sidecar missing for {name} "
                f"(Y={yp}, phase={pp})"
            )
        Ys[name] = np.load(yp)
        phases[name] = np.load(pp, allow_pickle=True)
    return {
        "Ys": Ys,
        "phases": phases,
        "NN": int(meta["nn_chosen"]),
        "nn_values": [int(v) for v in meta["nn_values"]],
        "errors": [float(v) for v in meta["errors"]],
        "n_cells": int(meta["n_cells"]),
        "anchor_session": meta.get("anchor_session", "TFC_cond"),
    }


def _save_crossreg_cache(json_path, y_paths, phase_paths, payload: dict) -> None:
    if json_path is None:
        return
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    sessions = list(payload["Ys"].keys())
    for name in sessions:
        yp = y_paths[name]
        pp = phase_paths[name]
        np.save(yp, np.ascontiguousarray(payload["Ys"][name], dtype=np.float64))
        # Phase arrays are object/string dtype; allow_pickle for round-trip.
        np.save(pp, np.asarray(payload["phases"][name]),
                allow_pickle=True)
    meta = {
        "sessions": sessions,
        "anchor_session": payload.get("anchor_session", "TFC_cond"),
        "nn_chosen": int(payload["NN"]),
        "nn_values": [int(v) for v in payload["nn_values"]],
        "errors": [float(v) for v in payload["errors"]],
        "n_cells": int(payload["n_cells"]),
        "params": {
            "ISOMAP_DIMS": ISOMAP_DIMS,
            "NN_STEPS": NN_STEPS,
            "NN_RANGE": list(NN_RANGE),
            "NN_CUTOFF": NN_CUTOFF,
            "DOWNSAMPLE_HZ": DOWNSAMPLE_HZ,
            "MINISCOPE_FPS": MINISCOPE_FPS,
        },
    }
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, json_path)
    print(f"  [crossreg cache] saved -> {json_path} (sessions={sessions})",
          flush=True)


# ---------------------------------------------------------------------------

def _fit_crossreg_manifolds(mouse, group, sessions_by_name,
                     crossreg_obj, mapping, save_dir, *,
                     anchor_session: str = "TFC_cond"):
    """Fit Isomap on ``anchor_session``'s crossreg cells, project the
    remaining sessions into that frame.

    ``anchor_session``: which session's crossreg-restricted activity is
    used as the Isomap basis (``iso.fit``); other sessions are projected
    via ``iso.transform``. Defaults to ``"TFC_cond"`` (TFC anchor).
    Pass ``"Test_B"`` to anchor on the post-encoding test session
    instead — useful for asking whether DREADD/CNO during conditioning
    produces a less-stable representation across sessions when judged
    from a post-encoding basis.
    """
    sess_basis = sessions_by_name.get(anchor_session)
    if sess_basis is None:
        raise RuntimeError(f"[crossreg] {mouse}: anchor session "
                           f"{anchor_session!r} missing")

    # crossreg cell indices into each session's full S
    idx_basis = np.asarray(
        get_S_indeces_crossreg(sess_basis, crossreg_obj, mapping), dtype=int)
    if len(idx_basis) < NN_RANGE[0]:
        raise RuntimeError(
            f"[crossreg] {mouse}: only {len(idx_basis)} crossreg cells "
            f"({anchor_session} basis), need at least {NN_RANGE[0]}."
        )

    # Build aligned, normalized S for the basis (each cell normalized
    # within its own session).
    S_basis_full = _normalize_per_neuron_01(
        np.asarray(sess_basis.S[idx_basis, :], dtype=float))
    S_basis, sess_basis_ds = _downsample_session(sess_basis, S_basis_full)
    X_basis = S_basis.T

    # Pre-build all X_* arrays (and capture downsampled sessions) so we
    # can hash them for the crossreg cache key BEFORE doing any Isomap
    # fitting. Non-basis sessions can be missing for some mice.
    X_by_name: dict[str, np.ndarray] = {anchor_session: X_basis}
    S_ds_by_name: dict[str, np.ndarray] = {anchor_session: S_basis}
    sess_ds_by_name: dict[str, object] = {anchor_session: sess_basis_ds}
    for name in SESSION_ORDER:
        if name == anchor_session:
            continue
        sess = sessions_by_name.get(name)
        if sess is None:
            print(f"  [crossreg] {mouse}: missing {name} — skipping", flush=True)
            continue
        idx = np.asarray(get_S_indeces_crossreg(sess, crossreg_obj, mapping), dtype=int)
        if len(idx) != len(idx_basis):
            raise RuntimeError(
                f"[crossreg] {mouse}: crossreg cell-count mismatch "
                f"{anchor_session}={len(idx_basis)} {name}={len(idx)}"
            )
        S_full = _normalize_per_neuron_01(np.asarray(sess.S[idx, :], dtype=float))
        S, sess_ds = _downsample_session(sess, S_full)
        X_by_name[name] = S.T
        S_ds_by_name[name] = S
        sess_ds_by_name[name] = sess_ds

    print(f"  [crossreg basis={anchor_session}] {mouse}: N={len(idx_basis)} cells "
          f"(DS factor={_downsample_factor()}, T_ds={X_basis.shape[0]})",
          flush=True)

    fit_label = f"{anchor_session}_crossreg_basis_{anchor_session}"
    title_tag = f"Crossreg manifolds ({anchor_session} basis)"
    overlay_fname = f"manifold_overlay_crossreg_basis_{anchor_session}.png"
    grid_fname = f"manifolds_grid_crossreg_basis_{anchor_session}.png"

    # ----- crossreg cache hit path: skip fit + transforms entirely -----
    json_path, y_paths, phase_paths = _crossreg_cache_paths(
        mouse, X_by_name, idx_basis, anchor_session=anchor_session,
    )
    cached = _try_load_crossreg_cache(json_path, y_paths, phase_paths)
    if cached is not None:
        print(f"  [crossreg basis={anchor_session}] {mouse}: CACHE HIT -> "
              f"NN={cached['NN']} (skipping fit+transforms, "
              f"sessions={list(cached['Ys'])})",
              flush=True)
        _plot_recon_error(cached["nn_values"], cached["errors"],
                          cached["NN"], mouse, fit_label,
                          save_dir)
        for suffix, targets in _arm_label_variants():
            _plot_manifold_overlay(
                cached["Ys"], cached["phases"],
                mouse=f"{mouse} ({GROUP_LABEL[group]})",
                save_dir=save_dir,
                fname=_with_suffix(overlay_fname, suffix),
                label_targets=targets,
            )
            _plot_manifold_grid(
                cached["Ys"], cached["phases"], mouse, group, save_dir,
                fname=_with_suffix(grid_fname, suffix),
                title_tag=title_tag,
                label_targets=targets,
            )
        return {"iso": None, "Ys": cached["Ys"], "phases": cached["phases"],
                "NN": cached["NN"], "n_cells": cached["n_cells"],
                "anchor_session": anchor_session,
                "S_ds_by_name": S_ds_by_name,
                "sess_ds_by_name": sess_ds_by_name}

    # ----- Cache miss: run the full crossreg pipeline -----
    iso, Y_basis, nn_chosen, nn_values, errors, cache_path_basis = \
        _fit_isomap_with_nn_search(
            X_basis, label=f"{mouse}/{anchor_session}/crossreg_basis_{anchor_session}"
        )
    if iso is None:
        # The lower-level Y sidecar fired even though the crossreg
        # cache missed (e.g. crossreg schema changed but per-X cache
        # is still valid). We need iso.transform() for the non-basis
        # sessions, so refit explicitly.
        print(f"  [crossreg basis={anchor_session}] {mouse}: lower-level Y "
              f"cache hit but iso required for transforms -> refitting at "
              f"NN={nn_chosen} (T={X_basis.shape[0]}, may take several "
              f"minutes)...", flush=True)
        iso = Isomap(
            n_neighbors=int(nn_chosen),
            n_components=ISOMAP_DIMS,
            n_jobs=ISOMAP_N_JOBS_INNER,
            eigen_solver=ISOMAP_EIGEN_SOLVER,
            metric="minkowski",
        )
        iso.fit(X_basis)
        _canonicalize_isomap_signs(iso)
        Y_basis = np.asarray(iso.embedding_).copy()
        print(f"  [crossreg basis={anchor_session}] {mouse}: refit at "
              f"NN={nn_chosen} done", flush=True)
    _plot_recon_error(nn_values, errors, nn_chosen, mouse, fit_label, save_dir)

    Ys = {anchor_session: Y_basis}
    phases = {anchor_session: _phase_index_from_session(
        sess_basis_ds, X_basis.shape[0])}

    for name in SESSION_ORDER:
        if name == anchor_session or name not in X_by_name:
            continue
        X = X_by_name[name]
        sess_ds = sess_ds_by_name[name]
        Y = iso.transform(X)
        Ys[name] = Y
        phases[name] = _phase_index_from_session(sess_ds, X.shape[0])

    for suffix, targets in _arm_label_variants():
        _plot_manifold_overlay(
            Ys, phases, mouse=f"{mouse} ({GROUP_LABEL[group]})",
            save_dir=save_dir,
            fname=_with_suffix(overlay_fname, suffix),
            label_targets=targets,
        )
        _plot_manifold_grid(
            Ys, phases, mouse, group, save_dir,
            fname=_with_suffix(grid_fname, suffix),
            title_tag=title_tag,
            label_targets=targets,
        )
    _save_crossreg_cache(json_path, y_paths, phase_paths, {
        "Ys": Ys,
        "phases": phases,
        "NN": nn_chosen,
        "nn_values": nn_values,
        "errors": errors,
        "n_cells": len(idx_basis),
        "anchor_session": anchor_session,
    })
    return {"iso": iso, "Ys": Ys, "phases": phases, "NN": nn_chosen,
            "n_cells": len(idx_basis),
            "anchor_session": anchor_session,
            "S_ds_by_name": S_ds_by_name,
            "sess_ds_by_name": sess_ds_by_name}


# ---------------------------------------------------------------------------
# Similarity score — cross-manifold rotation-invariant correlation
# (each session's manifold vs TFC_cond reference)
# ---------------------------------------------------------------------------

def _rotate_2d(Y, deg):
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Y @ R.T


def _similarity_after_rotation(traj_ref, traj_other):
    """Max over rotations of (corr_x + corr_y) / 2.

    Both inputs are (T_avg, 2) trajectories, **already trimmed to the same
    set of phase frames**.

    Returns ``(np.nan, np.nan)`` if either trajectory is degenerate
    (zero variance on any axis). Callers that aggregate over many pairs
    (per-trial Fisher-z) must filter out NaN explicitly and count the
    drops; single-pair callers (trial-averaged) must raise on NaN.
    """
    if traj_ref.shape != traj_other.shape:
        raise RuntimeError(
            f"_similarity_after_rotation shape mismatch {traj_ref.shape} vs {traj_other.shape}"
        )
    # Degenerate-input guard: a constant trajectory on either axis makes
    # the Pearson denominator zero, producing the NaN warnings we'd
    # otherwise see from numpy and a non-finite ``best``. Use a relative
    # tolerance against the larger trajectory so floating-point residue
    # (e.g. std ~1e-19 from iso.transform of an effectively constant
    # cell pattern) is treated as constant.
    std_ref = np.std(traj_ref, axis=0)
    std_other = np.std(traj_other, axis=0)
    scale = max(float(np.max(std_ref)), float(np.max(std_other)), 1.0)
    tol = 1e-12 * scale
    if (np.min(std_ref) <= tol) or (np.min(std_other) <= tol):
        return float("nan"), float("nan")
    best = -np.inf
    best_deg = 0
    for deg in range(0, 360, SS_ROTATE):
        rot = _rotate_2d(traj_other, deg)
        rx = float(np.corrcoef(rot[:, 0], traj_ref[:, 0])[0, 1])
        ry = float(np.corrcoef(rot[:, 1], traj_ref[:, 1])[0, 1])
        if not (np.isfinite(rx) and np.isfinite(ry)):
            continue
        sim = 0.5 * (rx + ry)
        if sim > best:
            best = sim
            best_deg = deg
    if not np.isfinite(best):
        raise RuntimeError(
            "_similarity_after_rotation: no finite similarity computed "
            f"(std_ref={std_ref}, std_other={std_other})"
        )
    return best, best_deg


def _similarity_no_rotation(traj_ref, traj_other):
    """Per-axis Pearson similarity at fixed rotation (deg=0).

    Use this for trajectories that already share a coordinate frame
    (e.g. all sessions in a crossreg fit project through the same iso).
    Returns ``(0.5 * (rx + ry), 0.0)`` so the call signature mirrors
    ``_similarity_after_rotation``. NaN on degenerate input.
    """
    if traj_ref.shape != traj_other.shape:
        raise RuntimeError(
            f"_similarity_no_rotation shape mismatch {traj_ref.shape} vs {traj_other.shape}"
        )
    std_ref = np.std(traj_ref, axis=0)
    std_other = np.std(traj_other, axis=0)
    scale = max(float(np.max(std_ref)), float(np.max(std_other)), 1.0)
    tol = 1e-12 * scale
    if (np.min(std_ref) <= tol) or (np.min(std_other) <= tol):
        return float("nan"), float("nan")
    rx = float(np.corrcoef(traj_other[:, 0], traj_ref[:, 0])[0, 1])
    ry = float(np.corrcoef(traj_other[:, 1], traj_ref[:, 1])[0, 1])
    if not (np.isfinite(rx) and np.isfinite(ry)):
        return float("nan"), float("nan")
    return 0.5 * (rx + ry), 0.0


def _pair_similarity_matrix(trajs_ref, trajs_other, *, rotate=True,
                            exclude_diag=False):
    """Return (matrix, finite_sims, n_degen) for an N_ref x N_other grid.

    ``rotate=True`` -> per-pair rotation-optimised similarity (Wilson
    primitive). ``rotate=False`` -> no rotation; both trajectories
    assumed to share a coordinate frame (e.g. crossreg/anchor mode).
    ``exclude_diag=True`` masks i==j (TFC self-pairing) with NaN and
    excludes them from ``n_degen``.
    """
    sim_fn = _similarity_after_rotation if rotate else _similarity_no_rotation
    n_ref = len(trajs_ref)
    n_other = len(trajs_other)
    matrix = np.full((n_ref, n_other), np.nan, dtype=float)
    finite_sims = []
    n_degen = 0
    for i in range(n_ref):
        for j in range(n_other):
            if exclude_diag and i == j:
                continue
            sim, _ = sim_fn(trajs_ref[i], trajs_other[j])
            if not np.isfinite(sim):
                n_degen += 1
                continue
            matrix[i, j] = sim
            finite_sims.append(sim)
    return matrix, finite_sims, n_degen


def _plot_pair_similarity_heatmap(matrix, save_path, *, title,
                                  ref_label, other_label,
                                  vmin=-1.0, vmax=1.0):
    """Save a heatmap of an N_ref x N_other per-trial similarity matrix."""
    matrix = np.asarray(matrix, dtype=float)
    n_ref, n_other = matrix.shape
    fig_w = max(3.0, 0.4 * n_other + 2.0)
    fig_h = max(2.5, 0.4 * n_ref + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="RdBu_r",
                   aspect="auto", origin="upper")
    ax.set_xlabel(f"{other_label} trial index")
    ax.set_ylabel(f"{ref_label} trial index")
    ax.set_xticks(range(n_other))
    ax.set_yticks(range(n_ref))
    ax.set_title(title)
    # Annotate each cell with the value (skip NaNs).
    for i in range(n_ref):
        for j in range(n_other):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color=("white" if abs(v) > 0.5 else "black"))
    fig.colorbar(im, ax=ax, label="similarity")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _ensure_independent_iso(p: dict, label: str):
    """Return a fitted Isomap for an independent-fit result dict.

    On Y-sidecar cache hits the independent-fit result has ``iso=None``
    because the fast path skips the refit (sklearn's fitted Isomap
    holds a dense T*T distance matrix that is too large to persist or
    to keep around for every mouse). The similarity computation needs
    ``iso.transform()`` to project trial-averaged trajectories onto
    the manifold, so on first access we refit at ``p['NN']`` and cache
    the refitted iso back into the dict. The caller
    (``run_isomap_pipeline``) is responsible for dropping the iso when
    the mouse finishes so peak RAM stays bounded by one mouse's worth
    of fitted Isomaps.
    """
    if p.get("iso") is not None:
        return p["iso"]
    nn = int(p["NN"])
    S_ds = p["S_ds"]
    X = np.asarray(S_ds, dtype=float).T
    print(f"  [similarity] {label}: refitting Isomap at NN={nn} for transforms "
          f"(no in-memory iso; T={X.shape[0]}, may take several minutes)...",
          flush=True)
    t0 = time.time()
    iso = Isomap(
        n_neighbors=nn,
        n_components=ISOMAP_DIMS,
        n_jobs=ISOMAP_N_JOBS_INNER,
        eigen_solver=ISOMAP_EIGEN_SOLVER,
        metric="minkowski",
    )
    iso.fit(X)
    _canonicalize_isomap_signs(iso)
    fit_dt = time.time() - t0
    Y_new = np.asarray(iso.embedding_)
    Y_cached = np.asarray(p["Y"])
    if Y_new.shape != Y_cached.shape:
        raise RuntimeError(
            f"_ensure_independent_iso[{label}]: refit Y shape {Y_new.shape} "
            f"differs from cached Y shape {Y_cached.shape}"
        )
    diff = float(np.max(np.abs(Y_new - Y_cached)))
    diff_rel = diff / (float(np.max(np.abs(Y_cached))) + 1e-12)
    print(f"  [similarity] {label}: iso refit took {fit_dt:.1f}s "
          f"(max|Y_refit - Y_cached|={diff:.3g}, rel={diff_rel:.2g})",
          flush=True)
    p["iso"] = iso
    return iso


def _ensure_crossreg_iso(crossreg_result: dict, label: str):
    """Return the fitted basis-session crossreg Isomap for a crossreg-fit
    result.

    On crossreg cache hits ``crossreg_result['iso']`` is None (we only
    persist Y sidecars, not the dense T*T iso). The anchor similarity
    path needs ``iso.transform()`` to project trial-window slices of
    every session onto the shared basis manifold, so on first access we
    refit at ``crossreg_result['NN']`` using the basis session's
    crossreg-restricted X and cache the iso back into the dict.
    """
    if crossreg_result.get("iso") is not None:
        return crossreg_result["iso"]
    NN = int(crossreg_result["NN"])
    anchor_session = crossreg_result.get("anchor_session", "TFC_cond")
    X_basis = np.asarray(
        crossreg_result["S_ds_by_name"][anchor_session], dtype=float).T
    print(f"  [similarity-anchor] {label}: refitting crossreg iso "
          f"(basis={anchor_session}) at NN={NN} "
          f"(no in-memory iso; pickling disabled by design)", flush=True)
    t0 = time.time()
    iso = Isomap(
        n_neighbors=NN,
        n_components=ISOMAP_DIMS,
        n_jobs=ISOMAP_N_JOBS_INNER,
        eigen_solver=ISOMAP_EIGEN_SOLVER,
        metric="minkowski",
    )
    iso.fit(X_basis)
    _canonicalize_isomap_signs(iso)
    fit_dt = time.time() - t0
    Y_new = np.asarray(iso.embedding_)
    Y_cached = np.asarray(crossreg_result["Ys"][anchor_session])
    if Y_new.shape != Y_cached.shape:
        raise RuntimeError(
            f"_ensure_crossreg_iso[{label}]: refit Y shape {Y_new.shape} "
            f"differs from cached {anchor_session} Y shape {Y_cached.shape}"
        )
    diff = float(np.max(np.abs(Y_new - Y_cached)))
    diff_rel = diff / (float(np.max(np.abs(Y_cached))) + 1e-12)
    print(f"  [similarity-anchor] {label}: iso refit took {fit_dt:.1f}s "
          f"(max|Y_refit - Y_cached|={diff:.3g}, rel={diff_rel:.2g})",
          flush=True)
    crossreg_result["iso"] = iso
    return iso


def _crossreg_to_independent_shaped(crossreg_result: dict, mouse_label: str) -> dict:
    """Adapt a crossreg-fit result into an independent-fit-shaped dict.

    Every session entry shares the SAME ``iso`` object (the basis
    session's crossreg Isomap from the crossreg fit; basis is read from
    ``crossreg_result['anchor_session']``), so the existing
    ``_similarity_*`` code paths transparently project trial windows
    from every session onto that shared manifold via
    ``iso.transform()``. Y / phase / S_ds /
    sess_ds are per-session (each one's anchor-frame projection of its
    own crossreg-restricted, normalized, downsampled S).
    """
    iso_basis = _ensure_crossreg_iso(crossreg_result, mouse_label)
    NN = int(crossreg_result["NN"])
    out = {}
    for name in crossreg_result["Ys"]:
        if name not in crossreg_result["S_ds_by_name"]:
            raise RuntimeError(
                f"_crossreg_to_independent_shaped[{mouse_label}]: session {name!r} "
                f"in Ys but missing from S_ds_by_name"
            )
        if name not in crossreg_result["sess_ds_by_name"]:
            raise RuntimeError(
                f"_crossreg_to_independent_shaped[{mouse_label}]: session {name!r} "
                f"in Ys but missing from sess_ds_by_name"
            )
        out[name] = {
            "iso": iso_basis,
            "Y": crossreg_result["Ys"][name],
            "phase": crossreg_result["phases"][name],
            "NN": NN,
            "S_ds": crossreg_result["S_ds_by_name"][name],
            "sess_ds": crossreg_result["sess_ds_by_name"][name],
        }
    return out

def _build_trial_avg_traj(sess, S_norm, trial_subset, iso, factor):
    """Build a trial-averaged 2D trajectory from per-trial windows.

    ``sess`` and ``S_norm`` MUST be in downsampled units. PERIOD_FRAMES
    values are at native MINISCOPE_FPS so we divide by ``factor``. Uses
    only the phases common to all sessions: pre_tone + tone + post_tone.

    Lifted from a closure inside ``_similarity_session_level`` so
    adjacent-mode callers can reuse it.
    """
    n_trials = len(sess.tone_onsets)
    pre_tone = PERIOD_FRAMES["pre_tone"] // factor
    tone = PERIOD_FRAMES["tone"] // factor
    post_tone = PERIOD_FRAMES["post_tone"] // factor
    if pre_tone == 0 or tone == 0 or post_tone == 0:
        raise RuntimeError(
            f"_build_trial_avg_traj: PERIOD_FRAMES // {factor} produced a zero "
            f"window (pre={pre_tone}, tone={tone}, post={post_tone})"
        )
    n_cells = S_norm.shape[0]
    avg = np.zeros((n_cells, pre_tone + tone + post_tone), dtype=float)
    trials = trial_subset if trial_subset is not None else range(n_trials)
    for cell in range(n_cells):
        pre_buf, tone_buf, post_buf = [], [], []
        for t in trials:
            t_on = sess.tone_onsets[t]
            t_off = sess.tone_offsets[t]
            pre_buf.append(S_norm[cell, t_on - pre_tone:t_on])
            tone_buf.append(S_norm[cell, t_on:t_on + tone])
            post_buf.append(S_norm[cell, t_off:t_off + post_tone])
        avg[cell, 0:pre_tone] = np.mean(pre_buf, axis=0)
        avg[cell, pre_tone:pre_tone + tone] = np.mean(tone_buf, axis=0)
        avg[cell, pre_tone + tone:] = np.mean(post_buf, axis=0)
    return iso.transform(avg.T)

def _trial_windows_for_type(sess, trial_type, factor):
    """Return list of (start, end) frame index pairs for a trial type.

    Returns empty list if not applicable (e.g. shock+post_shock on Test_B).
    Anchored at onset with canonical PERIOD_FRAMES widths so all trials
    of a type are equal-length.
    """
    pre_tone = PERIOD_FRAMES["pre_tone"] // factor
    tone = PERIOD_FRAMES["tone"] // factor
    post_tone = PERIOD_FRAMES["post_tone"] // factor
    shock = PERIOD_FRAMES["shock"] // factor
    post_shock = PERIOD_FRAMES["post_shock"] // factor
    if pre_tone == 0 or tone == 0 or post_tone == 0:
        raise RuntimeError(
            f"_trial_windows_for_type[{trial_type}]: PERIOD_FRAMES // {factor} "
            f"produced a zero window")
    if trial_type == "tone":
        return [(int(sess.tone_onsets[t]),
                 int(sess.tone_onsets[t]) + tone)
                for t in range(len(sess.tone_onsets))]
    if trial_type == "peri_tone":
        return [(int(sess.tone_onsets[t]) - pre_tone,
                 int(sess.tone_onsets[t]) + tone + post_tone)
                for t in range(len(sess.tone_onsets))]
    if trial_type == "post_tone":
        return [(int(sess.tone_offsets[t]),
                 int(sess.tone_offsets[t]) + post_tone)
                for t in range(len(sess.tone_offsets))]
    if trial_type == "post_shock":
        offs = list(getattr(sess, "shock_offsets", []) or [])
        if not offs:
            return []
        return [(int(offs[t]), int(offs[t]) + post_shock)
                for t in range(len(offs))]
    if trial_type == "tone+post_tone":
        return [(int(sess.tone_onsets[t]),
                 int(sess.tone_onsets[t]) + tone + post_tone)
                for t in range(len(sess.tone_onsets))]
    if trial_type == "shock+post_shock":
        shocks = list(getattr(sess, "shock_onsets", []) or [])
        offs = list(getattr(sess, "shock_offsets", []) or [])
        if not shocks or not offs:
            return []
        return [(int(shocks[t]), int(shocks[t]) + shock + post_shock)
                for t in range(len(shocks))]
    raise RuntimeError(
        f"_trial_windows_for_type: unknown trial_type {trial_type!r}")


def _build_per_trial_trajs(sess, S_norm, iso, trial_type, factor):
    """Return list of (T_trial, 2) per-trial manifold trajectories."""
    trajs = []
    windows = _trial_windows_for_type(sess, trial_type, factor)
    if not windows:
        return trajs
    T = S_norm.shape[1]
    expected_len = windows[0][1] - windows[0][0]
    for k, (s, e) in enumerate(windows):
        if s < 0 or e > T:
            raise RuntimeError(
                f"_build_per_trial_trajs[{trial_type}]: trial {k} window "
                f"[{s}, {e}) out of bounds (T={T})")
        seg = S_norm[:, s:e]
        if seg.shape[1] != expected_len:
            raise RuntimeError(
                f"_build_per_trial_trajs[{trial_type}]: trial {k} length "
                f"{seg.shape[1]} != expected {expected_len}")
        trajs.append(iso.transform(seg.T))
    return trajs


def _fisher_z_mean(rs):
    rs = np.asarray([r for r in rs if np.isfinite(r)], dtype=float)
    if rs.size == 0:
        return float("nan")
    rs = np.clip(rs, -0.999999, 0.999999)
    z = np.arctanh(rs)
    return float(np.tanh(np.mean(z)))


def _similarity_session_level(mouse, group, independent_results, *,
                              reference_for_session=None, rotate=True,
                              basis="TFC_cond"):
    """Compute similarity for each session vs a reference session.

    ``basis``: the session that plays the role of the reference frame.
    Its split-half is the within-session row; non-basis sessions are
    compared to it by default. Used to support both TFC-anchored
    (``basis="TFC_cond"``) and Test_B-anchored (``basis="Test_B"``)
    modes.

    ``reference_for_session``: dict mapping non-basis session names
    to the session whose trajectory should be used as reference.
    Defaults to anchored mode where every non-TFC_cond session is
    compared against TFC_cond's even-trial trajectory. Adjacent mode is
    obtained by passing ``{"Test_B": "TFC_cond", "Test_B_1wk": "Test_B"}``.

    ``rotate``: if True, per-pair rotation-optimised similarity (Wilson
    primitive). If False, no rotation — use for crossreg/anchor mode
    where all sessions share TFC_cond's iso so 2D axes are already
    aligned and a rotation sweep can only inflate noise.

    Strategy
    --------
    * Build trial-averaged 2D trajectories from each session's *independent*
      Phase-1 manifold (so neurons may differ across sessions — we compare
      trajectories in their respective 2D embeddings, both being projections
      from the same trial-averaged S into 2D).
    * For TFC_cond self, use even/odd trial split: build two trajectories
      from disjoint trial subsets so similarity is non-trivial.
    * Use only the phases common to all sessions (Test_B / Test_B_1wk lack
      shock & post_shock). Common = pre_tone, tone, post_tone.

    Returns: list of {mouse, group, session, similarity, best_deg}.
    """
    if reference_for_session is None:
        reference_for_session = {}
    rows = []

    if basis not in independent_results:
        print(f"  [similarity] {mouse}: basis {basis!r} missing — skipping similarity", flush=True)
        return rows

    factor = _downsample_factor()

    # Each independent-fit result holds its already-fitted Isomap
    # (key 'iso'); we re-use it to project trial-averaged trajectories
    # into 2D. On cache hits iso is initially None —
    # _ensure_independent_iso refits lazily and caches the result back
    # into the dict.
    p_basis = independent_results[basis]
    iso_basis = _ensure_independent_iso(p_basis, f"{mouse}/{basis}")

    # Basis split-half: even/odd trials. All time-domain ops use the
    # downsampled session/S so frame indices match.
    sess_basis = p_basis["sess_ds"]
    S_basis = p_basis["S_ds"]
    n_trials_basis = len(sess_basis.tone_onsets)
    if n_trials_basis < 2:
        raise RuntimeError(f"[similarity] {mouse}: {basis} has <2 trials")
    even_trials = np.arange(0, n_trials_basis, 2)
    odd_trials = np.arange(1, n_trials_basis, 2)
    traj_basis_even = _build_trial_avg_traj(sess_basis, S_basis, even_trials,
                                            iso_basis, factor)
    traj_basis_odd = _build_trial_avg_traj(sess_basis, S_basis, odd_trials,
                                           iso_basis, factor)
    sim_fn = _similarity_after_rotation if rotate else _similarity_no_rotation
    sim, deg = sim_fn(traj_basis_even, traj_basis_odd)
    rows.append({"mouse": mouse, "group": GROUP_LABEL[group],
                 "session": basis, "similarity": sim, "best_deg": deg})

    # Build full-trial trajectories for non-basis sessions (one per
    # session, reused as either "other" or as "reference" for the
    # adjacent-mode caller).
    full_trajs = {basis: traj_basis_even}
    for name in SESSION_ORDER:
        if name == basis or name not in independent_results:
            continue
        p = independent_results[name]
        iso_s = _ensure_independent_iso(p, f"{mouse}/{name}")
        full_trajs[name] = _build_trial_avg_traj(
            p["sess_ds"], p["S_ds"], None, iso_s, factor)

    # Cross-session rows: for each non-basis session, compare its full
    # trajectory against its reference session's trajectory (default =
    # basis's even-split, anchored mode).
    for name in SESSION_ORDER:
        if name == basis or name not in full_trajs:
            continue
        ref_name = reference_for_session.get(name, basis)
        if ref_name not in full_trajs:
            raise RuntimeError(
                f"[similarity] {mouse}/{name}: reference {ref_name!r} unavailable "
                f"(have {list(full_trajs)})")
        traj_ref = full_trajs[ref_name]
        traj_other = full_trajs[name]
        if traj_other.shape != traj_ref.shape:
            raise RuntimeError(
                f"[similarity] {mouse}/{name}: traj shape {traj_other.shape} "
                f"vs ref ({ref_name}) {traj_ref.shape}")
        sim, deg = sim_fn(traj_ref, traj_other)
        rows.append({"mouse": mouse, "group": GROUP_LABEL[group],
                     "session": name, "similarity": sim,
                     "best_deg": deg, "reference": ref_name})

    return rows


def _similarity_per_trial(mouse, group, independent_results, *,
                                 trial_type="tone",
                                 reference_for_session=None,
                                 rotate=True,
                                 basis="TFC_cond"):
    """Per-trial Fisher-z averaged similarity (preserves trial-to-trial
    variability that ``_similarity_session_level`` averages out).

    ``basis``: the session that plays the role of the reference frame
    (split-half row + default reference for non-basis sessions). See
    ``_similarity_session_level`` for usage with ``basis="Test_B"``.

    ``trial_type`` selects which segment defines a "trial":

    - ``"tone"``: ``[t_on, t_on + tone]`` strict tone window.
    - ``"peri_tone"``: ``[t_on - pre_tone, t_on + tone + post_tone]``
      window centered loosely on each tone (the previous "tone"
      definition; renamed to be honest about its content).
    - ``"post_tone"``: ``[t_off, t_off + post_tone]`` window starting
      at each tone offset.
    - ``"tone+post_tone"``: ``[t_on, t_off + post_tone]`` window from
      tone onset through end of post-tone period.
    - ``"shock+post_shock"``: ``[s_on, s_off + post_shock]`` window
      from shock onset through end of post-shock. Only valid for
      sessions with shocks (TFC_cond); cross-session comparison is
      skipped because Test_B / Test_B_1wk have no shocks.

    ``reference_for_session``: dict mapping non-basis session names
    to the session whose per-trial trajectories should be used as
    reference. Defaults to anchored mode (everything compared against
    ``basis``). Adjacent mode (TFC basis) is obtained by passing
    ``{"Test_B": "TFC_cond", "Test_B_1wk": "Test_B"}``.

    ``rotate``: if True, per-pair rotation-optimised similarity (use for
    independent fits where each session has its own iso axes). If False,
    use deg=0 directly (use for crossreg/anchor mode where all sessions
    share the basis's iso, so axes are already aligned and rotation can
    only inflate noise).

    Strategy
    --------
    For each trial in the reference session and each trial in the
    comparison session, build a per-trial trajectory by projecting that
    single trial's cells x ``T_trial`` frames into the session's
    independent-fit manifold via ``iso.transform``. Compute the
    similarity between every (ref trial i, other trial j) pair (Wilson
    primitive when ``rotate=True``). Fisher-z transform, average, then
    inverse-transform back to a correlation. For self-comparisons
    (same session as reference), exclude i==j to avoid trivial 1.0
    self-pairs.

    Returns
    -------
    (rows, matrices)
        ``rows``: list of {mouse, group, session, similarity, n_pairs,
        reference}.
        ``matrices``: dict[session_name -> {"matrix": ndarray (N_ref,
        N_other), "reference": str}] for downstream heatmap plotting.
    """
    if reference_for_session is None:
        reference_for_session = {}
    rows: list[dict] = []
    matrices: dict[str, dict] = {}
    if basis not in independent_results:
        print(f"  [similarity-trial:{trial_type}] {mouse}: basis {basis!r} missing — skipping",
              flush=True)
        return rows, matrices

    factor = _downsample_factor()

    # Build per-trial trajectories for every session that has applicable
    # trials. Cached in ``trajs_by_session`` so the adjacent-mode caller
    # can use the same trajectories as either reference or other.
    trajs_by_session: dict[str, list] = {}
    for name in SESSION_ORDER:
        if name not in independent_results:
            continue
        p = independent_results[name]
        iso_s = _ensure_independent_iso(p, f"{mouse}/{name}")
        ts = _build_per_trial_trajs(
            p["sess_ds"], p["S_ds"], iso_s, trial_type, factor)
        if ts:
            trajs_by_session[name] = ts

    if basis not in trajs_by_session:
        print(f"  [similarity-trial:{trial_type}] {mouse}: {basis} has no "
              f"applicable trials — skipping", flush=True)
        return rows, matrices

    # Basis self: all i!=j pairs (always anchored to itself).
    trajs_basis = trajs_by_session[basis]
    matrix, sims, n_degen = _pair_similarity_matrix(
        trajs_basis, trajs_basis, rotate=rotate, exclude_diag=True)
    if n_degen:
        print(f"  [similarity-trial:{trial_type}] {mouse}/{basis}: dropped "
              f"{n_degen}/{len(trajs_basis) * (len(trajs_basis) - 1)} "
              f"degenerate (zero-variance) trial pairs", flush=True)
    if sims:
        rows.append({"mouse": mouse, "group": GROUP_LABEL[group],
                     "session": basis,
                     "similarity": _fisher_z_mean(sims),
                     "n_pairs": len(sims),
                     "reference": basis})
        matrices[basis] = {"matrix": matrix, "reference": basis}

    # Other sessions: pair every (ref trial i, other trial j).
    for name in SESSION_ORDER:
        if name == basis:
            continue
        if name not in trajs_by_session:
            print(f"  [similarity-trial:{trial_type}] {mouse}/{name}: no "
                  f"applicable trials — skipping", flush=True)
            continue
        ref_name = reference_for_session.get(name, basis)
        if ref_name not in trajs_by_session:
            raise RuntimeError(
                f"[similarity-trial:{trial_type}] {mouse}/{name}: reference "
                f"{ref_name!r} unavailable (have {list(trajs_by_session)})")
        trajs_ref = trajs_by_session[ref_name]
        trajs_other = trajs_by_session[name]
        matrix, sims, n_degen = _pair_similarity_matrix(
            trajs_ref, trajs_other, rotate=rotate, exclude_diag=False)
        if n_degen:
            print(f"  [similarity-trial:{trial_type}] {mouse}/{name}: dropped "
                  f"{n_degen}/{len(trajs_ref) * len(trajs_other)} "
                  f"degenerate (zero-variance) trial pairs", flush=True)
        if not sims:
            raise RuntimeError(
                f"[similarity-trial:{trial_type}] {mouse}/{name}: ALL "
                f"{len(trajs_ref) * len(trajs_other)} trial pairs degenerate "
                f"(zero variance) — cell subset cannot resolve per-trial "
                f"trajectories on this manifold"
            )
        rows.append({"mouse": mouse, "group": GROUP_LABEL[group],
                     "session": name,
                     "similarity": _fisher_z_mean(sims),
                     "n_pairs": len(sims),
                     "reference": ref_name})
        matrices[name] = {"matrix": matrix, "reference": ref_name}

    return rows, matrices


def _save_per_trial_matrices_for_mouse(matrices, save_dir, *,
                                       mouse, group, trial_type,
                                       comparison_tag=""):
    """Save N_ref x N_other per-trial similarity matrices as CSV + PNG.

    ``matrices`` is the dict returned by ``_similarity_per_trial`` (or
    ``_similarity_post_tone_vs_TFC_post_shock``): keyed by session name,
    value = {"matrix": ndarray, "reference": str}.

    Output layout (one file per session)::

        {save_dir}/per_trial_matrices/{trial_type}/
            {mouse}_{group}_{session}{tag}.csv
            {mouse}_{group}_{session}{tag}.png

    ``comparison_tag``: optional suffix (e.g. ``"_vs_prev"``) used by
    anchor mode to disambiguate adjacent-extra vs. anchored rows.
    """
    if not matrices:
        return
    out_dir = os.path.join(save_dir, "per_trial_matrices", trial_type)
    os.makedirs(out_dir, exist_ok=True)
    group_label = GROUP_LABEL.get(group, group)
    for session_name, payload in matrices.items():
        matrix = np.asarray(payload["matrix"], dtype=float)
        ref_name = payload["reference"]
        base = (f"{mouse}_{group_label}_{session_name}"
                f"{comparison_tag}")
        # CSV: rows = ref trial idx (label = ref_name), cols = other.
        ref_label = (f"{ref_name}_post_shock"
                     if ref_name == "TFC_cond_post_shock" else ref_name)
        other_label = (f"{session_name}_post_tone"
                       if ref_name == "TFC_cond_post_shock"
                       else session_name)
        df = pd.DataFrame(
            matrix,
            index=[f"{ref_label}__t{i}" for i in range(matrix.shape[0])],
            columns=[f"{other_label}__t{j}" for j in range(matrix.shape[1])],
        )
        df.to_csv(os.path.join(out_dir, f"{base}.csv"))
        title = (f"{mouse} ({group_label}) — {trial_type}\n"
                 f"{ref_label} (rows) vs {other_label} (cols)")
        _plot_pair_similarity_heatmap(
            matrix, os.path.join(out_dir, f"{base}.png"),
            title=title, ref_label=ref_label, other_label=other_label,
        )


def _write_per_trial_matrix_group_averages(sim_dir, mouse_groups):
    """Write per-group average heatmaps of the per-trial similarity matrices.

    Scans ``{sim_dir}/per_trial_matrices/{trial_type}/`` for the
    per-mouse CSVs written by ``_save_per_trial_matrices_for_mouse``,
    groups them by (group, session, optional ``_vs_prev`` tag), Fisher-z
    averages the matrices across mice, and writes one PNG + one CSV per
    (group, session, tag) into the same trial_type folder.

    Output filenames::

        {trial_type}/group_avg__{group_label}_{session}{tag}.{csv,png}

    Mice with shapes that don't match the modal shape for a given
    (group, session, tag) raise a hard error so a silent shape mismatch
    can't corrupt the average.
    """
    root = os.path.join(sim_dir, "per_trial_matrices")
    if not os.path.isdir(root):
        return
    label_to_group = {GROUP_LABEL.get(g, g): g for g in set(mouse_groups.values())}
    for trial_type in sorted(os.listdir(root)):
        tt_dir = os.path.join(root, trial_type)
        if not os.path.isdir(tt_dir):
            continue
        # Skip our own outputs from a prior run.
        csv_files = sorted(f for f in os.listdir(tt_dir)
                           if f.endswith(".csv")
                           and not f.startswith("group_avg__"))
        # Bucket by (group_label, session, tag).
        buckets: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
        for fname in csv_files:
            stem = fname[:-len(".csv")]
            # Filename: {mouse}_{group_label}_{session}{tag}
            # session may contain underscores (TFC_cond, Test_B, Test_B_1wk).
            tag = ""
            if stem.endswith("_vs_prev"):
                tag = "_vs_prev"
                stem_no_tag = stem[:-len("_vs_prev")]
            else:
                stem_no_tag = stem
            parts = stem_no_tag.split("_", 2)
            if len(parts) < 3:
                continue
            mouse, group_label, session = parts
            if group_label not in label_to_group:
                continue
            buckets.setdefault((group_label, session, tag), []).append(
                (mouse, os.path.join(tt_dir, fname)))
        for (group_label, session, tag), entries in buckets.items():
            mats = []
            mice_used = []
            ref_label = None
            other_label = None
            for mouse, path in entries:
                df = pd.read_csv(path, index_col=0)
                arr = df.values.astype(float)
                if ref_label is None and len(df.index):
                    ref_label = str(df.index[0]).rsplit("__t", 1)[0]
                if other_label is None and len(df.columns):
                    other_label = str(df.columns[0]).rsplit("__t", 1)[0]
                mats.append(arr)
                mice_used.append(mouse)
            # Mice may have different trial counts (e.g. one mouse with
            # 4 tones vs others with 5). Pad each matrix with NaN to the
            # bucket's max (rows, cols) and use nanmean. Same-indexed
            # trials are still paired correctly because we pad the
            # trailing rows/cols only.
            max_r = max(m.shape[0] for m in mats)
            max_c = max(m.shape[1] for m in mats)
            padded = np.full((len(mats), max_r, max_c), np.nan, dtype=float)
            for k, m in enumerate(mats):
                padded[k, :m.shape[0], :m.shape[1]] = m
            # Fisher-z mean across mice (clip to avoid arctanh(±1)=inf).
            clipped = np.clip(padded, -0.999999, 0.999999)
            with warnings.catch_warnings():
                # All-NaN slices are expected at padded cells where no
                # mouse had both that row and that column; nanmean->NaN
                # is the correct answer.
                warnings.filterwarnings("ignore",
                                        message="Mean of empty slice",
                                        category=RuntimeWarning)
                with np.errstate(invalid="ignore"):
                    z = np.arctanh(clipped)
                    z_mean = np.nanmean(z, axis=0)
                    avg = np.tanh(z_mean)
            ref_label = ref_label or "ref"
            other_label = other_label or session
            base = f"group_avg__{group_label}_{session}{tag}"
            df_avg = pd.DataFrame(
                avg,
                index=[f"{ref_label}__t{i}" for i in range(avg.shape[0])],
                columns=[f"{other_label}__t{j}" for j in range(avg.shape[1])],
            )
            df_avg.to_csv(os.path.join(tt_dir, f"{base}.csv"))
            shapes = sorted({m.shape for m in mats})
            shape_str = (f"shape={shapes[0]}" if len(shapes) == 1
                         else f"shapes={shapes} padded to ({max_r},{max_c})")
            title = (f"{group_label} group avg (n={len(mats)}, {shape_str}) — "
                     f"{trial_type}\n{ref_label} (rows) vs {other_label} (cols), "
                     f"Fisher-z mean")
            _plot_pair_similarity_heatmap(
                avg, os.path.join(tt_dir, f"{base}.png"),
                title=title, ref_label=ref_label, other_label=other_label,
            )


def _similarity_post_tone_vs_TFC_post_shock(mouse, group, independent_results, *,
                                            rotate=True):
    """Similarity between each session's per-trial post-tone trajectories
    and TFC_cond's per-trial post-shock trajectories.

    Tests whether the trace-interval ensemble at retrieval reactivates the
    shock-period ensemble of the conditioning session. The reference is
    fixed to TFC_cond/post_shock for all 3 session columns:

      - TFC_cond:    post_tone(TFC_cond)    vs post_shock(TFC_cond)   (within)
      - Test_B:      post_tone(Test_B)      vs post_shock(TFC_cond)
      - Test_B_1wk:  post_tone(Test_B_1wk)  vs post_shock(TFC_cond)

    Per-trial trajectories are projected through each session's own iso
    in ``independent_results[name]`` (or the shared TFC_cond crossreg iso
    when ``independent_results`` is a crossreg-shaped dict). Pair-wise
    sims are Fisher-z averaged. ``rotate=True`` runs the per-pair
    rotation sweep; ``rotate=False`` is for crossreg/anchor frames where
    rotation is not meaningful.

    Returns
    -------
    (rows, matrices)
        ``rows``: list of {mouse, group, session, similarity, n_pairs, reference}.
        ``matrices``: dict[session_name -> {"matrix": ndarray, "reference": str}].
    """
    rows: list[dict] = []
    matrices: dict[str, dict] = {}
    if "TFC_cond" not in independent_results:
        print(f"  [similarity-pt-vs-TFCps] {mouse}: TFC_cond missing — skipping",
              flush=True)
        return rows, matrices

    factor = _downsample_factor()

    # Reference: TFC_cond post-shock trajectories.
    p_tfc = independent_results["TFC_cond"]
    iso_tfc = _ensure_independent_iso(p_tfc, f"{mouse}/TFC_cond")
    trajs_ps = _build_per_trial_trajs(
        p_tfc["sess_ds"], p_tfc["S_ds"], iso_tfc, "post_shock", factor)
    if not trajs_ps:
        print(f"  [similarity-pt-vs-TFCps] {mouse}: TFC_cond has no post_shock "
              f"trials — skipping", flush=True)
        return rows, matrices

    for name in SESSION_ORDER:
        if name not in independent_results:
            continue
        p = independent_results[name]
        iso_s = _ensure_independent_iso(p, f"{mouse}/{name}")
        trajs_pt = _build_per_trial_trajs(
            p["sess_ds"], p["S_ds"], iso_s, "post_tone", factor)
        if not trajs_pt:
            print(f"  [similarity-pt-vs-TFCps] {mouse}/{name}: no post_tone "
                  f"trials — skipping", flush=True)
            continue
        matrix, sims, n_degen = _pair_similarity_matrix(
            trajs_ps, trajs_pt, rotate=rotate, exclude_diag=False)
        if n_degen:
            print(f"  [similarity-pt-vs-TFCps] {mouse}/{name}: dropped "
                  f"{n_degen}/{len(trajs_ps) * len(trajs_pt)} "
                  f"degenerate trial pairs", flush=True)
        if not sims:
            raise RuntimeError(
                f"[similarity-pt-vs-TFCps] {mouse}/{name}: ALL "
                f"{len(trajs_ps) * len(trajs_pt)} trial pairs degenerate"
            )
        rows.append({"mouse": mouse, "group": GROUP_LABEL[group],
                     "session": name,
                     "similarity": _fisher_z_mean(sims),
                     "n_pairs": len(sims),
                     "reference": "TFC_cond_post_shock"})
        matrices[name] = {"matrix": matrix, "reference": "TFC_cond_post_shock"}
    return rows, matrices


def _similarity_vs_position(mouse, group, independent_results):
    """Wilson-faithful similarity: rotate the manifold trajectory and
    correlate against the animal's physical (x, y) position.

    For each session that has both ``loc_X_miniscope`` and
    ``loc_Y_miniscope`` (downsampled), correlate the *full* manifold
    embedding ``Y`` (no trial averaging) against the position trace
    after the optimal 2D rotation. Returns the rotation-maximised
    average of (corr_x, corr_y).

    Returns: list of {mouse, group, session, similarity, best_deg, n_frames}.
    """
    rows = []
    for name in SESSION_ORDER:
        if name not in independent_results:
            continue
        p = independent_results[name]
        sess_ds = p.get("sess_ds")
        if sess_ds is None:
            print(f"  [similarity-pos] {mouse}/{name}: no sess_ds — skipping",
                  flush=True)
            continue
        x = getattr(sess_ds, "loc_X_miniscope", None)
        y = getattr(sess_ds, "loc_Y_miniscope", None)
        if x is None or y is None:
            print(f"  [similarity-pos] {mouse}/{name}: position missing — skipping",
                  flush=True)
            continue
        Y = np.asarray(p["Y"], dtype=float)
        T_min = min(Y.shape[0], len(x), len(y))
        if T_min < 10:
            raise RuntimeError(
                f"[similarity-pos] {mouse}/{name}: T_min={T_min} too short")
        pos = np.column_stack([np.asarray(x[:T_min], dtype=float),
                               np.asarray(y[:T_min], dtype=float)])
        manifold = Y[:T_min, :]
        sim, deg = _similarity_after_rotation(pos, manifold)
        rows.append({"mouse": mouse, "group": GROUP_LABEL[group],
                     "session": name, "similarity": sim,
                     "best_deg": deg, "n_frames": int(T_min)})
    return rows


def _plot_similarity_by_group(df: pd.DataFrame, save_dir: str, *,
                              fname="similarity_score_by_group.png",
                              title="Cross-manifold similarity vs TFC_cond reference"):
    """Paper Fig 1M-style line plot."""
    if df.empty:
        print("[Isomap] similarity dataframe empty — skipping group plot")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    x_pos = np.arange(len(SESSION_ORDER))

    # individual mouse traces (light grey)
    for mouse in df["mouse"].unique():
        sub = df[df["mouse"] == mouse].set_index("session").reindex(SESSION_ORDER)
        if sub["similarity"].isna().any():
            continue
        ax.plot(x_pos, sub["similarity"].values, color="lightgrey",
                lw=0.8, alpha=0.7, marker='o', ms=3, zorder=1)

    # group means ± SEM
    for grp in GROUP_ORDER:
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        means, sems = [], []
        for sess in SESSION_ORDER:
            vals = gdf.loc[gdf["session"] == sess, "similarity"].values
            if len(vals) == 0:
                means.append(np.nan); sems.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        ax.errorbar(x_pos, means, yerr=sems, color=GROUP_COLOR[grp],
                    lw=2, marker='o', ms=6, capsize=4, label=grp, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(SESSION_ORDER)
    ax.set_ylabel("Similarity score")
    ax.set_xlabel("Session")
    ax.set_title(title)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _plot_similarity_bars_by_group(
    df: pd.DataFrame, save_dir: str, *,
    fname="similarity_score_bars_by_group.png",
    title="Cross-manifold similarity vs TFC_cond reference"):
    """Grouped-bar version: 3 bars (one per group) at each session,
    with mean ± SEM and group means connected across sessions.
    """
    if df.empty:
        print("[Isomap] similarity dataframe empty — skipping bar plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(SESSION_ORDER))
    n_groups = len(GROUP_ORDER)
    bar_w = 0.8 / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w

    for gi, grp in enumerate(GROUP_ORDER):
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        means, sems, ns = [], [], []
        for sess in SESSION_ORDER:
            vals = gdf.loc[gdf["session"] == sess, "similarity"].values
            if len(vals) == 0:
                means.append(np.nan); sems.append(np.nan); ns.append(0)
            else:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
                ns.append(len(vals))
        means_arr = np.asarray(means, dtype=float)
        sems_arr = np.asarray(sems, dtype=float)
        bar_x = x_pos + offsets[gi]
        ax.bar(bar_x, means_arr, width=bar_w * 0.9,
               color=GROUP_COLOR[grp], alpha=0.55, edgecolor="k",
               linewidth=0.6, label=grp, zorder=2)
        ax.errorbar(bar_x, means_arr, yerr=sems_arr, fmt="none",
                    ecolor="k", capsize=3, lw=1.0, zorder=3)
        # Connect group means across sessions through bar tops.
        ax.plot(bar_x, means_arr, color=GROUP_COLOR[grp],
                lw=1.5, marker="o", ms=4, zorder=4)

        # individual mouse points jittered within the bar
        rng = np.random.default_rng(42 + gi)
        for si, sess in enumerate(SESSION_ORDER):
            vals = gdf.loc[gdf["session"] == sess, "similarity"].values
            if len(vals) == 0:
                continue
            jitter = (rng.random(len(vals)) - 0.5) * (bar_w * 0.5)
            ax.scatter(np.full(len(vals), bar_x[si]) + jitter, vals,
                       s=12, color="k", alpha=0.6, zorder=5,
                       linewidths=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(SESSION_ORDER)
    ax.set_ylabel("Similarity score (mean ± SEM)")
    ax.set_xlabel("Session")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


# Anchor-mode column order: standard sessions vs the basis session,
# optionally followed by an extra adjacent-style comparison column. Two
# variants are supported, keyed by the canonical comparison label that
# appears in the data:
#   * vs_TFC   -> TFC_cond basis. 4 columns: 3 standard + extra row
#                 Test_B_1wk vs Test_B (anchor-adjacent).
#   * vs_TestB -> Test_B basis. 3 columns: Test_B, Test_B_1wk, then a
#                 dashed-line separator and TFC_cond (vs Test_B). No
#                 extra adjacent comparison.
ANCHOR_X_ORDER_BY_BASIS = {
    "vs_TFC": [
        ("TFC_cond", "vs_TFC"),
        ("Test_B", "vs_TFC"),
        ("Test_B_1wk", "vs_TFC"),
        ("Test_B_1wk", "vs_prev"),
    ],
    "vs_TestB": [
        ("Test_B", "vs_TestB"),
        ("Test_B_1wk", "vs_TestB"),
        ("TFC_cond", "vs_TestB"),
    ],
}
ANCHOR_X_LABELS_BY_BASIS = {
    "vs_TFC": [
        "TFC_cond",
        "Test_B",
        "Test_B_1wk\n(vs TFC)",
        "Test_B_1wk\n(vs Test_B)",
    ],
    "vs_TestB": [
        "Test_B",
        "Test_B_1wk",
        "TFC_cond\n(vs Test_B)",
    ],
}
# x-position of the dashed separator line, expressed as a fractional
# index between two columns (e.g. 2.5 = between idx 2 and 3).
ANCHOR_DIVIDER_BY_BASIS = {
    "vs_TFC": 2.5,
    "vs_TestB": 1.5,
}


def _anchor_axis_for_df(df: pd.DataFrame, label: str) -> tuple[list, list, float]:
    """Pick the (x_order, x_labels, divider_x) triple matching the
    canonical anchor label present in ``df['comparison']``.
    """
    if "comparison" not in df.columns:
        raise RuntimeError(f"{label}: df missing 'comparison' column")
    present = set(df["comparison"].unique())
    for canonical in ("vs_TFC", "vs_TestB"):
        if canonical in present:
            return (ANCHOR_X_ORDER_BY_BASIS[canonical],
                    ANCHOR_X_LABELS_BY_BASIS[canonical],
                    ANCHOR_DIVIDER_BY_BASIS[canonical])
    raise RuntimeError(
        f"{label}: df['comparison'] has no canonical anchor label "
        f"(found {sorted(present)!r}; expected one of vs_TFC, vs_TestB)"
    )


# Back-compat aliases (still referenced in a couple of callers below).
ANCHOR_X_ORDER = ANCHOR_X_ORDER_BY_BASIS["vs_TFC"]
ANCHOR_X_LABELS = ANCHOR_X_LABELS_BY_BASIS["vs_TFC"]


def _plot_similarity_by_group_anchor(
    df: pd.DataFrame, save_dir: str, *,
    fname="similarity_score_by_group.png",
    title="Anchor-fit cross-manifold similarity (TFC_cond crossreg frame)"):
    """Anchor-mode group plot with the extra Test_B_1wk-vs-Test_B column.

    Expects a ``comparison`` column with values ``"vs_TFC"`` or
    ``"vs_prev"`` in addition to the standard ``session`` column.
    """
    if df.empty:
        print("[Isomap] anchor similarity dataframe empty — skipping group plot")
        return
    ANCHOR_X_ORDER, ANCHOR_X_LABELS, _div_x = _anchor_axis_for_df(
        df, "_plot_similarity_by_group_anchor")
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(ANCHOR_X_ORDER))

    # individual mouse traces (light grey) — only mice with all 4 cells.
    for mouse in df["mouse"].unique():
        sub = df[df["mouse"] == mouse]
        vals = []
        for sess, comp in ANCHOR_X_ORDER:
            row = sub[(sub["session"] == sess) & (sub["comparison"] == comp)]
            if len(row) != 1:
                vals = None
                break
            vals.append(float(row["similarity"].iloc[0]))
        if vals is None:
            continue
        ax.plot(x_pos, vals, color="lightgrey",
                lw=0.8, alpha=0.7, marker='o', ms=3, zorder=1)

    # group means ± SEM
    for grp in GROUP_ORDER:
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        means, sems = [], []
        for sess, comp in ANCHOR_X_ORDER:
            vals = gdf.loc[
                (gdf["session"] == sess) & (gdf["comparison"] == comp),
                "similarity"].values
            if len(vals) == 0:
                means.append(np.nan); sems.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
        ax.errorbar(x_pos, means, yerr=sems, color=GROUP_COLOR[grp],
                    lw=2, marker='o', ms=6, capsize=4, label=grp, zorder=3)

    # Visual separator between anchored standard columns and the
    # extra adjacent comparison column.
    ax.axvline(_div_x, color="0.5", ls="--", lw=0.8, alpha=0.6, zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ANCHOR_X_LABELS)
    ax.set_ylabel("Similarity score")
    ax.set_xlabel("Session / comparison")
    ax.set_title(title)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _plot_similarity_bars_by_group_anchor(
    df: pd.DataFrame, save_dir: str, *,
    fname="similarity_score_bars_by_group.png",
    title="Anchor-fit cross-manifold similarity (TFC_cond crossreg frame)"):
    """Anchor-mode grouped-bar variant. Three bars per (session, comparison)
    column with means connected across columns per group.
    """
    if df.empty:
        print("[Isomap] anchor similarity dataframe empty — skipping bar plot")
        return
    ANCHOR_X_ORDER, ANCHOR_X_LABELS, _div_x = _anchor_axis_for_df(
        df, "_plot_similarity_bars_by_group_anchor")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(ANCHOR_X_ORDER))
    n_groups = len(GROUP_ORDER)
    bar_w = 0.8 / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w

    for gi, grp in enumerate(GROUP_ORDER):
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        means, sems = [], []
        for sess, comp in ANCHOR_X_ORDER:
            vals = gdf.loc[
                (gdf["session"] == sess) & (gdf["comparison"] == comp),
                "similarity"].values
            if len(vals) == 0:
                means.append(np.nan); sems.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
        means_arr = np.asarray(means, dtype=float)
        sems_arr = np.asarray(sems, dtype=float)
        bar_x = x_pos + offsets[gi]
        ax.bar(bar_x, means_arr, width=bar_w * 0.9,
               color=GROUP_COLOR[grp], alpha=0.55, edgecolor="k",
               linewidth=0.6, label=grp, zorder=2)
        ax.errorbar(bar_x, means_arr, yerr=sems_arr, fmt="none",
                    ecolor="k", capsize=3, lw=1.0, zorder=3)
        ax.plot(bar_x, means_arr, color=GROUP_COLOR[grp],
                lw=1.5, marker="o", ms=4, zorder=4)

        rng = np.random.default_rng(42 + gi)
        for si, (sess, comp) in enumerate(ANCHOR_X_ORDER):
            vals = gdf.loc[
                (gdf["session"] == sess) & (gdf["comparison"] == comp),
                "similarity"].values
            if len(vals) == 0:
                continue
            jitter = (rng.random(len(vals)) - 0.5) * (bar_w * 0.5)
            ax.scatter(np.full(len(vals), bar_x[si]) + jitter, vals,
                       s=12, color="k", alpha=0.6, zorder=5,
                       linewidths=0)

    ax.axvline(_div_x, color="0.5", ls="--", lw=0.8, alpha=0.6, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ANCHOR_X_LABELS)
    ax.set_ylabel("Similarity score (mean ± SEM)")
    ax.set_xlabel("Session / comparison")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _plot_similarity_box_by_group(
    df: pd.DataFrame, save_dir: str, *,
    fname="similarity_score_box_by_group.png",
    title="Cross-manifold similarity vs TFC_cond reference"):
    """Grouped-box version: 3 boxplots per session (median, IQR, whiskers)
    plus jittered individual mice; group means connected across sessions.
    """
    if df.empty:
        print("[Isomap] similarity dataframe empty — skipping box plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(SESSION_ORDER))
    n_groups = len(GROUP_ORDER)
    bar_w = 0.8 / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w

    for gi, grp in enumerate(GROUP_ORDER):
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        data_per_session, means, present_x = [], [], []
        for si, sess in enumerate(SESSION_ORDER):
            vals = gdf.loc[gdf["session"] == sess, "similarity"].values
            if len(vals) == 0:
                continue
            data_per_session.append(vals)
            means.append(float(np.mean(vals)))
            present_x.append(x_pos[si] + offsets[gi])
        if not data_per_session:
            continue
        bp = ax.boxplot(
            data_per_session, positions=present_x, widths=bar_w * 0.8,
            patch_artist=True, manage_ticks=False, showfliers=False,
            medianprops=dict(color="k", lw=1.4),
            boxprops=dict(facecolor=GROUP_COLOR[grp], alpha=0.55,
                          edgecolor="k", lw=0.8),
            whiskerprops=dict(color="k", lw=0.8),
            capprops=dict(color="k", lw=0.8),
        )
        # Connect group means across sessions through box centers.
        ax.plot(present_x, means, color=GROUP_COLOR[grp],
                lw=1.5, marker="D", ms=5, mec="k", mew=0.6,
                zorder=4, label=grp)

        rng = np.random.default_rng(42 + gi)
        for px, vals in zip(present_x, data_per_session):
            jitter = (rng.random(len(vals)) - 0.5) * (bar_w * 0.5)
            ax.scatter(np.full(len(vals), px) + jitter, vals,
                       s=12, color="k", alpha=0.6, zorder=5,
                       linewidths=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(SESSION_ORDER)
    ax.set_ylabel("Similarity score (box: median/IQR; ◆: mean)")
    ax.set_xlabel("Session")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def _plot_similarity_box_by_group_anchor(
    df: pd.DataFrame, save_dir: str, *,
    fname="similarity_score_box_by_group.png",
    title="Anchor-fit cross-manifold similarity (TFC_cond crossreg frame)"):
    """Anchor-mode grouped-box variant."""
    if df.empty:
        print("[Isomap] anchor similarity dataframe empty — skipping box plot")
        return
    ANCHOR_X_ORDER, ANCHOR_X_LABELS, _div_x = _anchor_axis_for_df(
        df, "_plot_similarity_box_by_group_anchor")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(ANCHOR_X_ORDER))
    n_groups = len(GROUP_ORDER)
    bar_w = 0.8 / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w

    for gi, grp in enumerate(GROUP_ORDER):
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        data_per_col, means, present_x = [], [], []
        for ci, (sess, comp) in enumerate(ANCHOR_X_ORDER):
            vals = gdf.loc[
                (gdf["session"] == sess) & (gdf["comparison"] == comp),
                "similarity"].values
            if len(vals) == 0:
                continue
            data_per_col.append(vals)
            means.append(float(np.mean(vals)))
            present_x.append(x_pos[ci] + offsets[gi])
        if not data_per_col:
            continue
        ax.boxplot(
            data_per_col, positions=present_x, widths=bar_w * 0.8,
            patch_artist=True, manage_ticks=False, showfliers=False,
            medianprops=dict(color="k", lw=1.4),
            boxprops=dict(facecolor=GROUP_COLOR[grp], alpha=0.55,
                          edgecolor="k", lw=0.8),
            whiskerprops=dict(color="k", lw=0.8),
            capprops=dict(color="k", lw=0.8),
        )
        ax.plot(present_x, means, color=GROUP_COLOR[grp],
                lw=1.5, marker="D", ms=5, mec="k", mew=0.6,
                zorder=4, label=grp)

        rng = np.random.default_rng(42 + gi)
        for px, vals in zip(present_x, data_per_col):
            jitter = (rng.random(len(vals)) - 0.5) * (bar_w * 0.5)
            ax.scatter(np.full(len(vals), px) + jitter, vals,
                       s=12, color="k", alpha=0.6, zorder=5,
                       linewidths=0)

    ax.axvline(_div_x, color="0.5", ls="--", lw=0.8, alpha=0.6, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ANCHOR_X_LABELS)
    ax.set_ylabel("Similarity score (box: median/IQR; ◆: mean)")
    ax.set_xlabel("Session / comparison")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistics on similarity DataFrames
# ---------------------------------------------------------------------------

def _fisher_z(r):
    """Element-wise Fisher z-transform with safe clipping."""
    r = np.asarray(r, dtype=float)
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def _stats_one_csv(csv_path: str, out_dir: str, *, fisher_z: bool = True):
    """Run mixed ANOVA + per-group RM-ANOVA + LMM on one similarity CSV.

    Expects columns: ``mouse, group, session, similarity``. If a
    ``comparison`` column is present (anchor mode), only rows whose
    comparison equals the canonical anchor label for this CSV are used
    so that the within-factor has exactly the 3 standard sessions
    (matching Wilson Fig 1M form). The canonical label is auto-detected
    from the values present (``"vs_TFC"`` for the TFC_cond anchor,
    ``"vs_TestB"`` for the Test_B anchor); the extra row label
    (``"vs_prev"`` / ``"vs_Test_B_1wk"``) is excluded.

    Writes:
      - mixed_anova.csv
      - rm_anova_per_group.csv      (one Wilson-style RM-ANOVA per group)
      - lmm_summary.txt + lmm_anova_typeII.csv
      - pairwise_session_per_group.csv (paired t-tests with Holm)
      - summary.txt (key F/p lines)
    """
    df = pd.read_csv(csv_path)
    required = {"mouse", "group", "session", "similarity"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"_stats_one_csv: {csv_path} missing columns {missing}"
        )

    if "comparison" in df.columns:
        # Canonical anchor labels (3-session within-factor) vs. extra
        # adjacent-style rows that we always want to drop here.
        _CANONICAL = ("vs_TFC", "vs_TestB")
        present = [c for c in _CANONICAL if c in set(df["comparison"].unique())]
        if len(present) != 1:
            raise RuntimeError(
                f"_stats_one_csv: {csv_path} comparison column has "
                f"{sorted(set(df['comparison'].unique()))!r}; expected "
                f"exactly one canonical label from {_CANONICAL}"
            )
        df = df[df["comparison"] == present[0]].copy()

    # Keep only the canonical 3 sessions, keep finite values, drop dups.
    df = df[df["session"].isin(SESSION_ORDER)].copy()
    df = df[np.isfinite(df["similarity"].astype(float))].copy()
    df = df.drop_duplicates(subset=["mouse", "session"], keep="first")

    # Drop mice that don't have all 3 sessions (RM-ANOVA requires complete cases).
    counts = df.groupby("mouse")["session"].nunique()
    complete_mice = counts[counts == len(SESSION_ORDER)].index.tolist()
    n_dropped = int((counts < len(SESSION_ORDER)).sum())
    df = df[df["mouse"].isin(complete_mice)].copy()
    if df.empty:
        print(f"[Stats] {csv_path}: no complete-case mice — skipping.")
        return

    df["session"] = pd.Categorical(df["session"], categories=list(SESSION_ORDER),
                                   ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=list(GROUP_ORDER),
                                 ordered=False)

    if fisher_z:
        df["similarity_stat"] = _fisher_z(df["similarity"].values)
        dv = "similarity_stat"
        dv_note = "Fisher-z(similarity)"
    else:
        df["similarity_stat"] = df["similarity"].astype(float)
        dv = "similarity_stat"
        dv_note = "similarity (raw r)"

    os.makedirs(out_dir, exist_ok=True)
    summary_lines = []
    summary_lines.append(f"CSV:           {csv_path}")
    summary_lines.append(f"DV:            {dv_note}")
    summary_lines.append(
        f"n mice (complete cases): {len(complete_mice)}  "
        f"(dropped {n_dropped} with missing session)"
    )
    grp_counts = (df.drop_duplicates("mouse")["group"]
                    .value_counts().reindex(GROUP_ORDER, fill_value=0))
    summary_lines.append(f"group counts:  " +
                         ", ".join(f"{g}={int(grp_counts[g])}" for g in GROUP_ORDER))
    summary_lines.append("")

    # ---- 1) Mixed-design ANOVA (group × session) ----
    try:
        mixed = pg.mixed_anova(data=df, dv=dv, within="session",
                               between="group", subject="mouse")
        mixed.to_csv(os.path.join(out_dir, "mixed_anova.csv"), index=False)
        summary_lines.append("== Mixed ANOVA (group × session) ==")
        for _, r in mixed.iterrows():
            src = r.get("Source", "")
            F = r.get("F", np.nan)
            df1 = r.get("DF1", r.get("ddof1", np.nan))
            df2 = r.get("DF2", r.get("ddof2", np.nan))
            p = r.get("p-unc", np.nan)
            p_gg = r.get("p-GG-corr", np.nan)
            np2 = r.get("np2", np.nan)
            line = (f"  {src:<14s}  F({df1:g},{df2:g})={F:.3f}, "
                    f"p={p:.4g}, np2={np2:.3f}")
            if pd.notna(p_gg):
                line += f", p_GG={p_gg:.4g}"
            summary_lines.append(line)
        summary_lines.append("")
    except Exception as e:
        summary_lines.append(f"[mixed_anova FAILED] {e}\n")

    # ---- 2) Wilson-style RM-ANOVA, one per DREADD group ----
    rm_rows = []
    summary_lines.append("== Per-group RM-ANOVA (Wilson-style: session) ==")
    for grp in GROUP_ORDER:
        gdf = df[df["group"] == grp].copy()
        if gdf["mouse"].nunique() < 2:
            summary_lines.append(f"  {grp}: <2 mice, skipped")
            continue
        try:
            rm = pg.rm_anova(data=gdf, dv=dv, within="session",
                             subject="mouse", detailed=True,
                             correction="auto")
            rm.insert(0, "group", grp)
            rm_rows.append(rm)
            sess_row = rm[rm["Source"] == "session"]
            if not sess_row.empty:
                r = sess_row.iloc[0]
                # pingouin returns df under varying column names depending
                # on version / correction setting. Fall back to the
                # canonical RM df: df1=k-1, df2=(n-1)*(k-1).
                n_mice = int(gdf["mouse"].nunique())
                k_sess = int(gdf["session"].nunique())
                df1_default = max(k_sess - 1, 0)
                df2_default = max((n_mice - 1) * (k_sess - 1), 0)
                df1 = next((float(r[k]) for k in ("ddof1", "DF1", "df1")
                            if k in r.index and pd.notna(r[k])), df1_default)
                df2 = next((float(r[k]) for k in ("ddof2", "DF2", "df2")
                            if k in r.index and pd.notna(r[k])), df2_default)
                F = float(r.get("F", np.nan))
                p = float(r.get("p-unc", np.nan))
                p_gg = r.get("p-GG-corr", np.nan)
                np2 = r.get("np2", r.get("ng2", np.nan))
                line = (f"  {grp:<8s}  F({df1:g},{df2:g})={F:.3f}, "
                        f"p={p:.4g}")
                if pd.notna(np2):
                    line += f", np2={float(np2):.3f}"
                if pd.notna(p_gg):
                    line += f", p_GG={float(p_gg):.4g}"
                summary_lines.append(line)
        except Exception as e:
            summary_lines.append(f"  {grp}: [rm_anova FAILED] {e}")
    if rm_rows:
        pd.concat(rm_rows, ignore_index=True).to_csv(
            os.path.join(out_dir, "rm_anova_per_group.csv"), index=False)
    summary_lines.append("")

    # ---- 3) Pairwise paired t-tests within group (Holm-corrected) ----
    try:
        pw_all = []
        for grp in GROUP_ORDER:
            gdf = df[df["group"] == grp].copy()
            if gdf["mouse"].nunique() < 2:
                continue
            pw = pg.pairwise_tests(data=gdf, dv=dv, within="session",
                                   subject="mouse", padjust="holm",
                                   parametric=True)
            pw.insert(0, "group", grp)
            pw_all.append(pw)
        if pw_all:
            pd.concat(pw_all, ignore_index=True).to_csv(
                os.path.join(out_dir, "pairwise_session_per_group.csv"),
                index=False)
    except Exception as e:
        summary_lines.append(f"[pairwise_tests FAILED] {e}")

    # ---- 4) LMM: similarity ~ group * session + (1|mouse) ----
    # Try a few optimisers / start points; if MixedLM is singular (RE
    # variance collapses to ~0 because between-mouse variance is small
    # after Fisher-z), fall back to OLS with mouse as a fixed effect.
    lmm_ok = False
    for _method in ("lbfgs", "powell", "cg", "nm"):
        try:
            md = smf.mixedlm(f"{dv} ~ C(group) * C(session)", df,
                             groups=df["mouse"])
            res = md.fit(reml=False, method=_method)
            with open(os.path.join(out_dir, "lmm_summary.txt"), "w") as f:
                f.write(str(res.summary()))
            lmm_ok = True
            break
        except Exception:
            continue
    if not lmm_ok:
        with open(os.path.join(out_dir, "lmm_summary.txt"), "w") as f:
            f.write("MixedLM singular for all optimisers. "
                    "Between-mouse variance ≈ 0 after Fisher-z; the "
                    "OLS-with-mouse-FE type-II ANOVA below is the "
                    "appropriate omnibus test.\n")
    try:
        ols_res = smf.ols(f"{dv} ~ C(group) * C(session) + C(mouse)",
                          data=df).fit()
        anova_tbl = anova_lm(ols_res, typ=2)
        anova_tbl.to_csv(os.path.join(out_dir, "lmm_anova_typeII.csv"))
        summary_lines.append("== Fixed-effects type-II ANOVA "
                             "(OLS with mouse FE) ==")
        for term, row in anova_tbl.iterrows():
            if term == "Residual":
                continue
            F = row.get("F", np.nan)
            p = row.get("PR(>F)", np.nan)
            df_n = row.get("df", np.nan)
            summary_lines.append(
                f"  {term:<28s}  F={F:.3f}, df={df_n:g}, p={p:.4g}")
    except Exception as e:
        summary_lines.append(f"[OLS ANOVA FAILED] {e}")
    summary_lines.append("")

    # ---- 5) Between-group descriptives + per-session group contrast ----
    summary_lines.append("== Per-session means ± SEM (back-transformed to r) ==")
    desc_rows = []
    for sess in SESSION_ORDER:
        sub = df[df["session"] == sess]
        line = f"  {sess:<14s}  "
        for grp in GROUP_ORDER:
            vals = sub.loc[sub["group"] == grp, dv].values
            if vals.size == 0:
                line += f"{grp}=n/a   "
                continue
            mean_dv = float(np.mean(vals))
            sem_dv = (float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                      if vals.size > 1 else 0.0)
            mean_r = float(np.tanh(mean_dv)) if fisher_z else mean_dv
            sem_r = (float(np.tanh(mean_dv + sem_dv) -
                           np.tanh(mean_dv)) if fisher_z else sem_dv)
            line += f"{grp}={mean_r:+.3f}±{sem_r:.3f} (n={vals.size})  "
            desc_rows.append(dict(session=sess, group=grp,
                                  mean_r=mean_r, sem_r=sem_r,
                                  mean_dv=mean_dv, sem_dv=sem_dv,
                                  n=int(vals.size)))
        summary_lines.append(line)
    if desc_rows:
        pd.DataFrame(desc_rows).to_csv(
            os.path.join(out_dir, "per_session_group_means.csv"),
            index=False)
    summary_lines.append("")

    # Independent-samples between-group test at each session
    # (Exc>Ctl>Inh hypothesis: report all 3 pairs, Holm within session).
    summary_lines.append(
        "== Between-group tests at each session "
        "(Welch t, Holm within session) ==")
    bg_rows = []
    for sess in SESSION_ORDER:
        sub = df[df["session"] == sess]
        try:
            pw = pg.pairwise_tests(data=sub, dv=dv, between="group",
                                   parametric=True, padjust="holm")
            pw.insert(0, "session", sess)
            bg_rows.append(pw)
            for _, r in pw.iterrows():
                a = r.get("A", "?")
                b = r.get("B", "?")
                T = r.get("T", np.nan)
                dof = r.get("dof", np.nan)
                p = r.get("p-unc", np.nan)
                p_corr = r.get("p-corr", np.nan)
                hg = r.get("hedges", np.nan)
                summary_lines.append(
                    f"  {sess:<14s} {a} vs {b}: t({dof:.1f})={T:+.2f}, "
                    f"p={p:.4g}, p_holm={p_corr:.4g}, g={hg:+.2f}")
        except Exception as e:
            summary_lines.append(f"  {sess}: [between-group FAILED] {e}")
    if bg_rows:
        pd.concat(bg_rows, ignore_index=True).to_csv(
            os.path.join(out_dir, "between_group_per_session.csv"),
            index=False)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print(f"[Stats] -> {out_dir}\n", flush=True)


def run_similarity_stats(sim_dir: str, *, fisher_z: bool = True):
    """Run mixed ANOVA / RM-ANOVA / LMM on every similarity CSV in *sim_dir*.

    Walks ``similarity_score.csv``, ``similarity_score_per_trial_*.csv``
    and ``similarity_score_vs_position.csv``. Per-CSV outputs go under
    ``<sim_dir>/stats/<csv_stem>/``.
    """
    if not os.path.isdir(sim_dir):
        print(f"[Stats] missing sim_dir: {sim_dir}")
        return
    patterns = [
        "similarity_score.csv",
        "similarity_score_per_trial_*.csv",
        "similarity_score_vs_position.csv",
    ]
    csvs = []
    for pat in patterns:
        csvs.extend(sorted(glob.glob(os.path.join(sim_dir, pat))))
    if not csvs:
        print(f"[Stats] no similarity CSVs in {sim_dir}")
        return
    stats_root = os.path.join(sim_dir, "stats")
    os.makedirs(stats_root, exist_ok=True)
    _copy_analysis_methods_template(
        "similarity_stats_methods.txt", stats_root)
    print(f"\n[Stats] {sim_dir}: {len(csvs)} CSV(s) "
          f"(Fisher-z={fisher_z})", flush=True)
    for csv_path in csvs:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(stats_root, stem)
        print(f"\n[Stats] === {stem} ===")
        _stats_one_csv(csv_path, out_dir, fisher_z=fisher_z)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def _run_anchor_block(
    *,
    mouse,
    group,
    crossreg_result,
    basis: str,
    sim_dir_anc: str,
    mode_tag: str,
    standard_comparison_label: str,
    extra_ref_for_session: dict,
    extra_comparison_label: str,
    sim_rows_anchor: list,
    sim_rows_per_trial_anchor: dict,
    sim_rows_vs_position_anchor: list,
    sim_rows_pt_vs_TFCps_anchor: list,
):
    """Run anchor-mode similarity for one mouse against one crossreg basis.

    ``crossreg_result`` is the output of ``_fit_crossreg_manifolds`` for
    the given ``basis``. The standard 3 (or 2) rows produced by
    ``_similarity_session_level`` are tagged
    ``comparison=standard_comparison_label`` (e.g. ``'vs_TFC'`` or
    ``'vs_TestB'``). One extra row defined by
    ``extra_ref_for_session`` (e.g. ``{"Test_B_1wk": "Test_B"}`` for the
    TFC anchor, ``{"TFC_cond": "Test_B_1wk"}`` for the TestB anchor) is
    tagged ``comparison=extra_comparison_label`` and merged into the
    same accumulators.
    """
    if crossreg_result is None:
        print(f"  [similarity-anchor basis={basis}] {mouse}: no crossreg result"
              f" — skipping", flush=True)
        return
    crossreg_shaped = _crossreg_to_independent_shaped(
        crossreg_result, mouse_label=f"{mouse}/{basis}")
    sim_cached_anc = _try_load_sim_cache(
        mouse, group, crossreg_shaped, mode=mode_tag)
    if sim_cached_anc is not None:
        sim_rows_anchor.extend(sim_cached_anc["main"])
        for tt in sim_rows_per_trial_anchor.keys():
            sim_rows_per_trial_anchor[tt].extend(
                sim_cached_anc["per_trial"].get(tt, []))
        sim_rows_vs_position_anchor.extend(sim_cached_anc["vs_position"])
    else:
        # Anchored rows: every non-basis session vs the basis. rotate=False
        # because every session shares the basis-session's iso (axes are
        # already aligned by construction).
        anc_rows = _similarity_session_level(
            mouse, group, crossreg_shaped, rotate=False, basis=basis)
        for r in anc_rows:
            r["comparison"] = standard_comparison_label

        # Extra row(s): pull the requested override comparisons (only
        # when extra_ref_for_session is non-empty).
        if extra_ref_for_session:
            adj_extra = _similarity_session_level(
                mouse, group, crossreg_shaped,
                reference_for_session=extra_ref_for_session,
                rotate=False, basis=basis)
            for r in adj_extra:
                sess = r.get("session")
                ref = r.get("reference")
                if sess in extra_ref_for_session and ref == extra_ref_for_session[sess]:
                    r2 = dict(r)
                    r2["comparison"] = extra_comparison_label
                    anc_rows.append(r2)
        sim_rows_anchor.extend(anc_rows)

        anc_pt_rows: dict[str, list[dict]] = {}
        for tt in sim_rows_per_trial_anchor.keys():
            rows_anc, mats_anc = _similarity_per_trial(
                mouse, group, crossreg_shaped, trial_type=tt,
                rotate=False, basis=basis)
            for r in rows_anc:
                r["comparison"] = standard_comparison_label
            if extra_ref_for_session:
                rows_adj_extra, mats_adj_extra = _similarity_per_trial(
                    mouse, group, crossreg_shaped, trial_type=tt,
                    reference_for_session=extra_ref_for_session,
                    rotate=False, basis=basis)
                for r in rows_adj_extra:
                    sess = r.get("session")
                    ref = r.get("reference")
                    if (sess in extra_ref_for_session
                            and ref == extra_ref_for_session[sess]):
                        r2 = dict(r)
                        r2["comparison"] = extra_comparison_label
                        rows_anc.append(r2)
            else:
                mats_adj_extra = {}
            anc_pt_rows[tt] = rows_anc
            sim_rows_per_trial_anchor[tt].extend(rows_anc)
            _save_per_trial_matrices_for_mouse(
                mats_anc, sim_dir_anc, mouse=mouse, group=group,
                trial_type=tt)
            # Save extra-comparison matrices only for the override
            # session(s) so we don't overwrite the standard ones.
            extra_only = {s: mats_adj_extra[s]
                          for s in extra_ref_for_session
                          if s in mats_adj_extra}
            if extra_only:
                _save_per_trial_matrices_for_mouse(
                    extra_only, sim_dir_anc, mouse=mouse, group=group,
                    trial_type=tt,
                    comparison_tag=f"_{extra_comparison_label}")

        anc_vs_pos_rows = _similarity_vs_position(
            mouse, group, crossreg_shaped)
        sim_rows_vs_position_anchor.extend(anc_vs_pos_rows)

        _save_sim_cache(
            mouse, group, independent_results=crossreg_shaped,
            main_rows=anc_rows, per_trial_rows=anc_pt_rows,
            vs_pos_rows=anc_vs_pos_rows,
            mode=mode_tag,
        )
    # Anchor variant of post_tone-vs-TFC_post_shock metric. Uses the
    # basis-session crossreg-cell iso for every session; rotate=False
    # because all share that iso.
    anc_pt_vs_ps_rows, anc_pt_vs_ps_mats = (
        _similarity_post_tone_vs_TFC_post_shock(
            mouse, group, crossreg_shaped, rotate=False))
    sim_rows_pt_vs_TFCps_anchor.extend(anc_pt_vs_ps_rows)
    _save_per_trial_matrices_for_mouse(
        anc_pt_vs_ps_mats, sim_dir_anc, mouse=mouse, group=group,
        trial_type="post_tone_vs_TFC_post_shock")
    # Drop the heavy iso held by crossreg_result before next basis/mouse.
    crossreg_result.pop("iso", None)


def run_isomap_pipeline(
    TFC_cond,
    Test_B,
    Test_B_1wk,
    mouse_groups,
    mice_per_group_Test_B_B_1wk,
    TFC_B_B_1wk_crossreg,
    mapping_TFC_cond_Test_B_Test_B_1wk,
    PLOTS_DIR,
    NPY_SAVE_PATH=None,
    auto_close=True,
):
    """Run the full Isomap pipeline (Phases 1–3).

    Parameters mirror ``run_population_pca_pipeline`` for consistency.
    Outputs land under ``PLOTS_DIR`` (already the per-analysis root, e.g.
    ``<root>/TFC_Isomap``).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _copy_analysis_methods_template(_METHODS_TEMPLATE, PLOTS_DIR)

    # Configure on-disk cache for fitted Isomaps. Cache is keyed by a hash
    # of (X bytes, X.shape, sweep params, DOWNSAMPLE_HZ) so toggling
    # parameters or input data invalidates entries automatically.
    global _CACHE_DIR, _LOG_DIR, _SIM_CACHE_DIR
    if NPY_SAVE_PATH is not None:
        _CACHE_DIR = os.path.join(NPY_SAVE_PATH, "isomap_cache")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _LOG_DIR = os.path.join(NPY_SAVE_PATH, "isomap_logs")
        os.makedirs(_LOG_DIR, exist_ok=True)
        _SIM_CACHE_DIR = os.path.join(NPY_SAVE_PATH, "isomap_sim_cache")
        os.makedirs(_SIM_CACHE_DIR, exist_ok=True)
        print(f"*** Isomap cache dir:     {_CACHE_DIR}", flush=True)
        print(f"*** Isomap log dir:       {_LOG_DIR}", flush=True)
        print(f"*** Isomap sim cache dir: {_SIM_CACHE_DIR}", flush=True)
    else:
        _CACHE_DIR = None
        _LOG_DIR = None
        _SIM_CACHE_DIR = None
        print("*** Isomap cache: DISABLED (no NPY_SAVE_PATH provided)", flush=True)

    # Mice eligible for the crossreg fit = those with crossreg
    # mappings available. ``mice_per_group_Test_B_B_1wk`` only covers
    # DREADD treatment groups, not all tracked mice (e.g. G10 has
    # crossreg but no DREADD assignment).
    crossreg_mice = set(TFC_B_B_1wk_crossreg.keys())

    # Output directory layout (defined before the per-mouse loop so
    # per-trial matrix heatmaps can be written incrementally).
    sim_dir = os.path.join(PLOTS_DIR, "similarity")
    sim_dir_adj = os.path.join(PLOTS_DIR, "similarity_adjacent")
    sim_dir_anc_TFC = os.path.join(PLOTS_DIR, "similarity_anchor_TFC")
    sim_dir_anc_TestB = os.path.join(PLOTS_DIR, "similarity_anchor_TestB")
    for _d in (sim_dir, sim_dir_adj, sim_dir_anc_TFC, sim_dir_anc_TestB):
        os.makedirs(_d, exist_ok=True)


    sim_rows: list[dict] = []
    sim_rows_per_trial: dict[str, list[dict]] = {
        "tone": [],
        "peri_tone": [],
        "post_tone": [],
        "tone+post_tone": [],
        "shock+post_shock": [],
    }
    # Publish expected per_trial keys so _try_load_sim_cache can
    # invalidate stale caches that predate newly added trial_types.
    global _EXPECTED_PER_TRIAL_KEYS
    _EXPECTED_PER_TRIAL_KEYS = frozenset(sim_rows_per_trial.keys())
    sim_rows_vs_position: list[dict] = []
    # Adjacent-mode accumulators: each non-TFC session is compared
    # against the most recent prior session (Test_B vs TFC_cond,
    # Test_B_1wk vs Test_B) instead of always against TFC_cond.
    sim_rows_adjacent: list[dict] = []
    sim_rows_per_trial_adjacent: dict[str, list[dict]] = {
        "tone": [],
        "peri_tone": [],
        "post_tone": [],
        "tone+post_tone": [],
        "shock+post_shock": [],
    }
    ADJACENT_REFERENCE = {"Test_B": "TFC_cond", "Test_B_1wk": "Test_B"}
    # Anchor-mode accumulators: every session lives in the basis
    # session's crossreg-cell Isomap frame. There are two anchor
    # variants:
    #   * anchor_TFC: TFC_cond crossreg basis (encoding-anchored);
    #     standard 3 rows tagged comparison='vs_TFC' plus a 4th row
    #     Test_B_1wk-vs-Test_B tagged 'vs_prev'.
    #   * anchor_TestB: Test_B crossreg basis (post-encoding-anchored);
    #     standard 3 rows tagged comparison='vs_TestB' plus a 4th row
    #     TFC_cond-vs-Test_B_1wk tagged 'vs_Test_B_1wk' (the forward
    #     analog of vs_prev for this basis).
    sim_rows_anchor_TFC: list[dict] = []
    sim_rows_per_trial_anchor_TFC: dict[str, list[dict]] = {
        "tone": [],
        "peri_tone": [],
        "post_tone": [],
        "tone+post_tone": [],
        "shock+post_shock": [],
    }
    sim_rows_vs_position_anchor_TFC: list[dict] = []
    sim_rows_anchor_TestB: list[dict] = []
    sim_rows_per_trial_anchor_TestB: dict[str, list[dict]] = {
        "tone": [],
        "peri_tone": [],
        "post_tone": [],
        "tone+post_tone": [],
        "shock+post_shock": [],
    }
    sim_rows_vs_position_anchor_TestB: list[dict] = []
    ANCHOR_ADJACENT_REFERENCE = {"Test_B_1wk": "Test_B"}
    # New metric: each session's per-trial post_tone trajectories vs
    # TFC_cond's per-trial post_shock trajectories (fixed reference).
    # Anchor-agnostic for the non-anchor mode -> one accumulator covers
    # both similarity/ and similarity_adjacent/. The anchor variants use
    # their respective basis-session crossreg-cell iso.
    sim_rows_pt_vs_TFCps: list[dict] = []
    sim_rows_pt_vs_TFCps_anchor_TFC: list[dict] = []
    sim_rows_pt_vs_TFCps_anchor_TestB: list[dict] = []
    # independent_by_mouse holds only the SLIM per-session results
    # (Y, phase, NN, sweep curve). The heavy fields (iso, S_ds, S_norm,
    # sess, sess_ds) are dropped immediately after the similarity
    # computation finishes for the mouse so peak RAM stays bounded by
    # ~1 mouse's worth of fitted Isomaps (each Isomap holds a dense T*T
    # distance matrix ~1.3 GB at T=12,827; keeping all 11 mice * 3
    # sessions in memory blows up).
    independent_by_mouse: dict[str, dict] = {}
    SLIM_KEYS = ("Y", "phase", "NN", "nn_values", "errors")

    for mouse, group in mouse_groups.items():
        if group not in GROUP_LABEL:
            raise RuntimeError(
                f"run_isomap_pipeline: unknown group {group!r} for mouse {mouse}"
            )
        sub_dir = os.path.join(PLOTS_DIR, mouse)
        os.makedirs(sub_dir, exist_ok=True)
        print(f"\n*** Isomap pipeline: {mouse} ({GROUP_LABEL[group]}) -> {sub_dir}",
              flush=True)

        sessions = {
            "TFC_cond": TFC_cond.get(mouse),
            "Test_B": Test_B.get(mouse),
            "Test_B_1wk": Test_B_1wk.get(mouse),
        }

        # Independent per-session manifolds (full dict with
        # iso/S_ds/sess/etc.; lives only for this mouse's iteration).
        independent_results = _fit_independent_manifolds(mouse, group, sessions, sub_dir)

        # Crossreg overlays (also feed anchor similarity below). One
        # fit per anchor basis: TFC_cond (encoding-anchored, always
        # done) and Test_B (post-encoding-anchored, used to ask whether
        # CNO/encoding produces a less-stable representation when
        # judged from a post-encoding basis).
        crossreg_result = None
        crossreg_result_TestB = None
        if mouse in crossreg_mice and mouse in TFC_B_B_1wk_crossreg:
            crossreg_result = _fit_crossreg_manifolds(
                mouse, group, sessions,
                TFC_B_B_1wk_crossreg[mouse],
                mapping_TFC_cond_Test_B_Test_B_1wk,
                sub_dir,
                anchor_session="TFC_cond",
            )
            crossreg_result_TestB = _fit_crossreg_manifolds(
                mouse, group, sessions,
                TFC_B_B_1wk_crossreg[mouse],
                mapping_TFC_cond_Test_B_Test_B_1wk,
                sub_dir,
                anchor_session="Test_B",
            )
        else:
            print(f"  [crossreg] {mouse}: not in crossreg set — skipping overlay",
                  flush=True)

        # Similarity computation (only meaningful if all 3 sessions present).
        # This may lazy-refit Isomaps via _ensure_independent_iso and
        # stash them back into independent_results[sess]['iso'] — those
        # are the heavy objects we drop next. Cached on disk (per-mouse
        # JSON keyed by NN signature) so reruns skip the iso refits
        # entirely.
        sim_cached = _try_load_sim_cache(mouse, group, independent_results)
        if sim_cached is not None:
            new_rows = sim_cached["main"]
            sim_rows.extend(new_rows)
            for tt in sim_rows_per_trial.keys():
                sim_rows_per_trial[tt].extend(
                    sim_cached["per_trial"].get(tt, []))
            sim_rows_vs_position.extend(sim_cached["vs_position"])
        else:
            new_rows = _similarity_session_level(mouse, group, independent_results)
            sim_rows.extend(new_rows)

            # Per-trial parallel metrics: Fisher-z averaged similarity
            # over all (TFC_cond trial i, other-session trial j) pairs.
            # Preserves trial-to-trial variability that
            # ``_similarity_session_level`` averages out. Run once per
            # trial-type definition. Also produces an N_ref x N_other
            # similarity matrix per (mouse, session) for at-a-glance
            # inspection of trial-level structure.
            pt_rows: dict[str, list[dict]] = {}
            for tt in sim_rows_per_trial.keys():
                rows_tt, mats_tt = _similarity_per_trial(
                    mouse, group, independent_results, trial_type=tt)
                pt_rows[tt] = rows_tt
                sim_rows_per_trial[tt].extend(rows_tt)
                _save_per_trial_matrices_for_mouse(
                    mats_tt, sim_dir, mouse=mouse, group=group,
                    trial_type=tt)

            # Wilson-faithful: correlate manifold vs physical (x, y)
            # position. Uses cached Y + downsampled position trace;
            # no iso.transform.
            vs_pos_rows = _similarity_vs_position(
                mouse, group, independent_results)
            sim_rows_vs_position.extend(vs_pos_rows)

            _save_sim_cache(
                mouse, group, independent_results=independent_results,
                main_rows=new_rows, per_trial_rows=pt_rows,
                vs_pos_rows=vs_pos_rows,
            )

        # New metric: post_tone(session) vs post_shock(TFC_cond).
        # Run unconditionally (no cache); reference-fixed so the result
        # is anchor-agnostic for non-anchor mode -> written to both
        # similarity/ and similarity_adjacent/.
        pt_vs_ps_rows, pt_vs_ps_mats = _similarity_post_tone_vs_TFC_post_shock(
            mouse, group, independent_results)
        sim_rows_pt_vs_TFCps.extend(pt_vs_ps_rows)
        _save_per_trial_matrices_for_mouse(
            pt_vs_ps_mats, sim_dir, mouse=mouse, group=group,
            trial_type="post_tone_vs_TFC_post_shock")

        # Adjacent-mode similarity: same metrics but with each non-TFC
        # session compared against its most-recent prior session.
        # Reuses the same iso (lazy-refit & cached on
        # independent_results[sess]) so this only adds rotation-sweep
        # work; no extra iso fits beyond what the anchored block
        # already triggered (or what this block triggers if the
        # anchored cache was a hit and adjacent isn't).
        sim_cached_adj = _try_load_sim_cache(mouse, group, independent_results,
                                             mode="adjacent")
        if sim_cached_adj is not None:
            sim_rows_adjacent.extend(sim_cached_adj["main"])
            for tt in sim_rows_per_trial_adjacent.keys():
                sim_rows_per_trial_adjacent[tt].extend(
                    sim_cached_adj["per_trial"].get(tt, []))
        else:
            adj_rows = _similarity_session_level(
                mouse, group, independent_results,
                reference_for_session=ADJACENT_REFERENCE)
            sim_rows_adjacent.extend(adj_rows)
            adj_pt_rows: dict[str, list[dict]] = {}
            for tt in sim_rows_per_trial_adjacent.keys():
                rows_tt, mats_tt = _similarity_per_trial(
                    mouse, group, independent_results, trial_type=tt,
                    reference_for_session=ADJACENT_REFERENCE)
                adj_pt_rows[tt] = rows_tt
                sim_rows_per_trial_adjacent[tt].extend(rows_tt)
                _save_per_trial_matrices_for_mouse(
                    mats_tt, sim_dir_adj, mouse=mouse, group=group,
                    trial_type=tt)
            _save_sim_cache(
                mouse, group, independent_results=independent_results,
                main_rows=adj_rows, per_trial_rows=adj_pt_rows,
                vs_pos_rows=[],  # vs_position is anchor-agnostic
                mode="adjacent",
            )

        # ----- Anchor-mode similarity (crossreg frame) -----
        # Same metrics as the anchored block above but run on an
        # independent-shaped view of each crossreg fit (every session
        # shares the basis-session's crossreg-cell iso). Two anchor
        # variants are run per mouse: TFC_cond basis (encoding) and
        # Test_B basis (post-encoding).
        _run_anchor_block(
            mouse=mouse, group=group,
            crossreg_result=crossreg_result,
            basis="TFC_cond",
            sim_dir_anc=sim_dir_anc_TFC,
            mode_tag="anchor_TFC",
            standard_comparison_label="vs_TFC",
            extra_ref_for_session=ANCHOR_ADJACENT_REFERENCE,
            extra_comparison_label="vs_prev",
            sim_rows_anchor=sim_rows_anchor_TFC,
            sim_rows_per_trial_anchor=sim_rows_per_trial_anchor_TFC,
            sim_rows_vs_position_anchor=sim_rows_vs_position_anchor_TFC,
            sim_rows_pt_vs_TFCps_anchor=sim_rows_pt_vs_TFCps_anchor_TFC,
        )
        _run_anchor_block(
            mouse=mouse, group=group,
            crossreg_result=crossreg_result_TestB,
            basis="Test_B",
            sim_dir_anc=sim_dir_anc_TestB,
            mode_tag="anchor_TestB",
            standard_comparison_label="vs_TestB",
            extra_ref_for_session={},
            extra_comparison_label="",
            sim_rows_anchor=sim_rows_anchor_TestB,
            sim_rows_per_trial_anchor=sim_rows_per_trial_anchor_TestB,
            sim_rows_vs_position_anchor=sim_rows_vs_position_anchor_TestB,
            sim_rows_pt_vs_TFCps_anchor=sim_rows_pt_vs_TFCps_anchor_TestB,
        )
        # Drop heavy crossreg state before next mouse.
        for _cr in (crossreg_result, crossreg_result_TestB):
            if _cr is not None:
                _cr.pop("iso", None)
                _cr.pop("S_ds_by_name", None)
                _cr.pop("sess_ds_by_name", None)

        # Similarity-aligned independent-fit grid: rotate each
        # non-TFC_cond Y by its best_deg into TFC_cond's frame for
        # visual trajectory comparison. Cosmetic only.
        deg_by_session = {r["session"]: r["best_deg"] for r in new_rows}
        if deg_by_session:
            Ys_grid = {s: independent_results[s]["Y"] for s in independent_results}
            phases_grid = {s: independent_results[s]["phase"] for s in independent_results}
            for suffix, targets in _arm_label_variants():
                _plot_manifold_grid_aligned(
                    Ys_grid, phases_grid, deg_by_session,
                    mouse, group, sub_dir,
                    fname=_with_suffix(
                        "manifolds_grid_indep_aligned.png", suffix),
                    label_targets=targets,
                )

        # ----- per-mouse cleanup: free heavy fields, keep slim summary -----
        slim = {}
        for sess_name, p in independent_results.items():
            slim[sess_name] = {k: p[k] for k in SLIM_KEYS if k in p}
            # Explicitly drop the heaviest references so they are
            # reclaimable as soon as `independent_results` and
            # `sessions` go out of scope at the end of this iteration.
            for heavy in ("iso", "S_ds", "S_norm", "sess", "sess_ds",
                          "cache_path"):
                p.pop(heavy, None)
        independent_by_mouse[mouse] = slim
        del independent_results, sessions
        if crossreg_result is not None:
            del crossreg_result
        if crossreg_result_TestB is not None:
            del crossreg_result_TestB
        gc.collect()
        print(f"  [Isomap] {mouse}: heavy state freed; "
              f"slim summary kept ({list(slim.keys())})", flush=True)
        # Abort window between mice — on fully-cached runs this is the
        # only place a user can break out cleanly without Ctrl-C.
        _abort_countdown()

    # Group plot + CSV — all similarity outputs in a 'similarity/' subdir
    # to avoid cluttering PLOTS_DIR. (Output dirs already created above
    # so per-trial matrix heatmaps could be written during the per-mouse
    # loop; ``os.makedirs`` is idempotent.)
    os.makedirs(sim_dir, exist_ok=True)

    sim_df = pd.DataFrame(sim_rows)
    if not sim_df.empty:
        sim_df.to_csv(os.path.join(sim_dir, "similarity_score.csv"), index=False)
    _plot_similarity_by_group(sim_df, sim_dir)
    _plot_similarity_bars_by_group(sim_df, sim_dir)
    _plot_similarity_box_by_group(sim_df, sim_dir)

    # One CSV + plot per per-trial type.
    sim_df_per_trial_by_type = {}
    for tt, rows in sim_rows_per_trial.items():
        df_tt = pd.DataFrame(rows)
        sim_df_per_trial_by_type[tt] = df_tt
        # Filename-safe tag (replace '+' with '_plus_').
        tag = tt.replace("+", "_plus_")
        if not df_tt.empty:
            df_tt.to_csv(
                os.path.join(sim_dir, f"similarity_score_per_trial_{tag}.csv"),
                index=False)
        _plot_similarity_by_group(
            df_tt, sim_dir,
            fname=f"similarity_score_per_trial_{tag}_by_group.png",
            title=f"Per-trial similarity (Fisher-z mean) — trial = {tt}",
        )
        _plot_similarity_bars_by_group(
            df_tt, sim_dir,
            fname=f"similarity_score_per_trial_{tag}_bars_by_group.png",
            title=f"Per-trial similarity (Fisher-z mean) — trial = {tt}",
        )
        _plot_similarity_box_by_group(
            df_tt, sim_dir,
            fname=f"similarity_score_per_trial_{tag}_box_by_group.png",
            title=f"Per-trial similarity (Fisher-z mean) — trial = {tt}",
        )

    sim_df_vs_position = pd.DataFrame(sim_rows_vs_position)
    if not sim_df_vs_position.empty:
        sim_df_vs_position.to_csv(
            os.path.join(sim_dir, "similarity_score_vs_position.csv"),
            index=False)
    _plot_similarity_by_group(
        sim_df_vs_position, sim_dir,
        fname="similarity_score_vs_position_by_group.png",
        title="Manifold vs physical (x, y) similarity (Wilson-style)",
    )
    _plot_similarity_bars_by_group(
        sim_df_vs_position, sim_dir,
        fname="similarity_score_vs_position_bars_by_group.png",
        title="Manifold vs physical (x, y) similarity (Wilson-style)",
    )
    _plot_similarity_box_by_group(
        sim_df_vs_position, sim_dir,
        fname="similarity_score_vs_position_box_by_group.png",
        title="Manifold vs physical (x, y) similarity (Wilson-style)",
    )

    # ---- Adjacent-mode outputs (mirrors anchored, separate folder) ----
    os.makedirs(sim_dir_adj, exist_ok=True)

    sim_df_adjacent = pd.DataFrame(sim_rows_adjacent)
    if not sim_df_adjacent.empty:
        sim_df_adjacent.to_csv(
            os.path.join(sim_dir_adj, "similarity_score.csv"), index=False)
    _plot_similarity_by_group(
        sim_df_adjacent, sim_dir_adj,
        title="Adjacent-session similarity (Test_B vs TFC_cond, "
              "Test_B_1wk vs Test_B)",
    )
    _plot_similarity_bars_by_group(
        sim_df_adjacent, sim_dir_adj,
        title="Adjacent-session similarity (Test_B vs TFC_cond, "
              "Test_B_1wk vs Test_B)",
    )
    _plot_similarity_box_by_group(
        sim_df_adjacent, sim_dir_adj,
        title="Adjacent-session similarity (Test_B vs TFC_cond, "
              "Test_B_1wk vs Test_B)",
    )

    sim_df_per_trial_by_type_adjacent = {}
    for tt, rows in sim_rows_per_trial_adjacent.items():
        df_tt = pd.DataFrame(rows)
        sim_df_per_trial_by_type_adjacent[tt] = df_tt
        tag = tt.replace("+", "_plus_")
        if not df_tt.empty:
            df_tt.to_csv(
                os.path.join(sim_dir_adj,
                             f"similarity_score_per_trial_{tag}.csv"),
                index=False)
        _plot_similarity_by_group(
            df_tt, sim_dir_adj,
            fname=f"similarity_score_per_trial_{tag}_by_group.png",
            title=f"Adjacent-session per-trial similarity "
                  f"(Fisher-z mean) — trial = {tt}",
        )
        _plot_similarity_bars_by_group(
            df_tt, sim_dir_adj,
            fname=f"similarity_score_per_trial_{tag}_bars_by_group.png",
            title=f"Adjacent-session per-trial similarity "
                  f"(Fisher-z mean) — trial = {tt}",
        )
        _plot_similarity_box_by_group(
            df_tt, sim_dir_adj,
            fname=f"similarity_score_per_trial_{tag}_box_by_group.png",
            title=f"Adjacent-session per-trial similarity "
                  f"(Fisher-z mean) — trial = {tt}",
        )

    # ---- Anchor-mode outputs (crossreg frame, with extra fourth bar) ----
    # Two anchor variants: TFC_cond basis (vs_TFC + vs_prev) and
    # Test_B basis (vs_TestB + vs_Test_B_1wk).
    sim_df_anchor_TFC = pd.DataFrame(sim_rows_anchor_TFC)
    sim_df_per_trial_by_type_anchor_TFC: dict = {}
    sim_df_vs_position_anchor_TFC = pd.DataFrame(sim_rows_vs_position_anchor_TFC)
    sim_df_anchor_TestB = pd.DataFrame(sim_rows_anchor_TestB)
    sim_df_per_trial_by_type_anchor_TestB: dict = {}
    sim_df_vs_position_anchor_TestB = pd.DataFrame(sim_rows_vs_position_anchor_TestB)
    _ANCHOR_OUTPUTS = (
        ("TFC", sim_dir_anc_TFC,
         sim_df_anchor_TFC, sim_rows_per_trial_anchor_TFC,
         sim_df_per_trial_by_type_anchor_TFC,
         sim_df_vs_position_anchor_TFC,
         "TFC_cond"),
        ("TestB", sim_dir_anc_TestB,
         sim_df_anchor_TestB, sim_rows_per_trial_anchor_TestB,
         sim_df_per_trial_by_type_anchor_TestB,
         sim_df_vs_position_anchor_TestB,
         "Test_B"),
    )
    for (_lbl, _dir, _df_main, _pt_rows, _pt_dfs_out,
         _df_vs_pos, _basis_label) in _ANCHOR_OUTPUTS:
        os.makedirs(_dir, exist_ok=True)
        _title = (f"Anchor-fit cross-manifold similarity "
                  f"({_basis_label} crossreg frame)")
        if not _df_main.empty:
            _df_main.to_csv(
                os.path.join(_dir, "similarity_score.csv"), index=False)
        _plot_similarity_by_group_anchor(_df_main, _dir, title=_title)
        _plot_similarity_bars_by_group_anchor(_df_main, _dir, title=_title)
        _plot_similarity_box_by_group_anchor(_df_main, _dir, title=_title)

        for tt, rows in _pt_rows.items():
            df_tt = pd.DataFrame(rows)
            _pt_dfs_out[tt] = df_tt
            tag = tt.replace("+", "_plus_")
            if not df_tt.empty:
                df_tt.to_csv(
                    os.path.join(_dir,
                                 f"similarity_score_per_trial_{tag}.csv"),
                    index=False)
            _pt_title = (f"Anchor-fit per-trial similarity "
                         f"({_basis_label} basis, Fisher-z mean) "
                         f"— trial = {tt}")
            _plot_similarity_by_group_anchor(
                df_tt, _dir,
                fname=f"similarity_score_per_trial_{tag}_by_group.png",
                title=_pt_title)
            _plot_similarity_bars_by_group_anchor(
                df_tt, _dir,
                fname=f"similarity_score_per_trial_{tag}_bars_by_group.png",
                title=_pt_title)
            _plot_similarity_box_by_group_anchor(
                df_tt, _dir,
                fname=f"similarity_score_per_trial_{tag}_box_by_group.png",
                title=_pt_title)

        if not _df_vs_pos.empty:
            _df_vs_pos.to_csv(
                os.path.join(_dir, "similarity_score_vs_position.csv"),
                index=False)
        _vp_title = (f"Anchor-fit manifold vs physical (x, y) similarity "
                     f"({_basis_label} basis)")
        _plot_similarity_by_group(
            _df_vs_pos, _dir,
            fname="similarity_score_vs_position_by_group.png",
            title=_vp_title)
        _plot_similarity_bars_by_group(
            _df_vs_pos, _dir,
            fname="similarity_score_vs_position_bars_by_group.png",
            title=_vp_title)
        _plot_similarity_box_by_group(
            _df_vs_pos, _dir,
            fname="similarity_score_vs_position_box_by_group.png",
            title=_vp_title)

    # ---- New metric: post-tone(session) vs post-shock(TFC_cond) ----
    # Reference is fixed to TFC_cond/post_shock for all 3 session columns.
    # Same data written to similarity/ and similarity_adjacent/ (own-iso
    # frame); each anchor variant uses its basis-session crossreg-cell iso.
    sim_df_pt_vs_TFCps = pd.DataFrame(sim_rows_pt_vs_TFCps)
    sim_df_pt_vs_TFCps_anchor_TFC = pd.DataFrame(sim_rows_pt_vs_TFCps_anchor_TFC)
    sim_df_pt_vs_TFCps_anchor_TestB = pd.DataFrame(sim_rows_pt_vs_TFCps_anchor_TestB)
    _PT_VS_TFCPS_FNAME = "similarity_score_post_tone_vs_TFC_post_shock"
    _PT_VS_TFCPS_TITLE = (
        "Per-trial post-tone vs post-shock(TFC_cond)\n"
        "(reactivation of shock ensemble during trace interval)"
    )
    for _df_, _dir_, _suffix_ in (
        (sim_df_pt_vs_TFCps, sim_dir, ""),
        (sim_df_pt_vs_TFCps, sim_dir_adj, ""),
        (sim_df_pt_vs_TFCps_anchor_TFC, sim_dir_anc_TFC,
         " (TFC_cond anchor)"),
        (sim_df_pt_vs_TFCps_anchor_TestB, sim_dir_anc_TestB,
         " (Test_B anchor)"),
    ):
        if _df_.empty:
            continue
        _df_.to_csv(os.path.join(_dir_, f"{_PT_VS_TFCPS_FNAME}.csv"),
                    index=False)
        _plot_similarity_by_group(
            _df_, _dir_,
            fname=f"{_PT_VS_TFCPS_FNAME}_by_group.png",
            title=_PT_VS_TFCPS_TITLE + _suffix_,
        )
        _plot_similarity_bars_by_group(
            _df_, _dir_,
            fname=f"{_PT_VS_TFCPS_FNAME}_bars_by_group.png",
            title=_PT_VS_TFCPS_TITLE + _suffix_,
        )
        _plot_similarity_box_by_group(
            _df_, _dir_,
            fname=f"{_PT_VS_TFCPS_FNAME}_box_by_group.png",
            title=_PT_VS_TFCPS_TITLE + _suffix_,
        )

    # ------------------------------------------------------------------
    # Statistics: mixed ANOVA + per-group RM-ANOVA + LMM, automatically
    # for every similarity CSV in all three directories.
    # ------------------------------------------------------------------
    for _stats_dir in (sim_dir, sim_dir_adj, sim_dir_anc_TFC, sim_dir_anc_TestB):
        _write_per_trial_matrix_group_averages(_stats_dir, mouse_groups)
        run_similarity_stats(_stats_dir, fisher_z=True)

    print(f"\n*** Isomap pipeline complete -> {PLOTS_DIR}\n", flush=True)
    return {
        "independent_by_mouse": independent_by_mouse,
        "similarity_df": sim_df,
        "similarity_df_per_trial_by_type": sim_df_per_trial_by_type,
        "similarity_df_vs_position": sim_df_vs_position,
        "similarity_df_adjacent": sim_df_adjacent,
        "similarity_df_per_trial_by_type_adjacent":
            sim_df_per_trial_by_type_adjacent,
        "similarity_df_anchor_TFC": sim_df_anchor_TFC,
        "similarity_df_per_trial_by_type_anchor_TFC":
            sim_df_per_trial_by_type_anchor_TFC,
        "similarity_df_vs_position_anchor_TFC": sim_df_vs_position_anchor_TFC,
        "similarity_df_anchor_TestB": sim_df_anchor_TestB,
        "similarity_df_per_trial_by_type_anchor_TestB":
            sim_df_per_trial_by_type_anchor_TestB,
        "similarity_df_vs_position_anchor_TestB": sim_df_vs_position_anchor_TestB,
        "PLOTS_DIR": PLOTS_DIR,
    }

print("isomap.py loaded")
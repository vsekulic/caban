"""caban.population — Population PCA trajectory analyses.

OOP refactor of the PCA blocks from ``caban.debug_snippets3.py`` (sections
"PCA", "Trial-averaged PCA", "Crossreg PCA - AVG", "Crossreg PCA - FULL"
tone/shock + post-tone/shock).

Entry point
-----------
``run_population_pca_pipeline`` — full per-mouse pipeline; called from
``caban.main.py`` after the cross-epoch decoder analyses.

Design
------
* All imports at module top.
* No silent skips: hard-fails when prerequisites are missing.
* Method-2 in the FULL crossreg pipelines fits PCA on the *concatenated*
  z-scored matrix ``S_tot.T`` (the prototype erroneously fitted on
  ``S_TFC_cond_n.T``; this is the documented bugfix).
* Engram-cell selection always uses the *raw* TFC_cond activity matrix as
  reference (the prototype AVG path used z-scored S, which is degenerate
  because z-scored row sums are ~0; FULL paths already used raw S).
"""

import os
import shutil
import gc
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from caban.utilities import (
    get_S_indeces_crossreg, MINISCOPE_FPS, Saver,
)

from caban.pca_state_metrics import run_pca_state_metrics_pipeline


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

PERIOD_FRAMES = {
    "pre_tone": 20,
    "tone": 20,
    "post_tone": 20,
    "shock": 2,
    "post_shock": 20,
}

# Period order and total length per session type
_AVG_PERIODS_TFC = ("pre_tone", "tone", "post_tone", "shock", "post_shock")
_AVG_PERIODS_TEST = ("pre_tone", "tone", "post_tone")

PERIOD_COLORS = {
    "pre_tone": "black",
    "tone": "blue",
    "post_tone": "grey",
    "shock": "red",
    "post_shock": "green",
}

# Per-mouse 3D view angles (elev, azim) for the FULL toneshock 2D-loc + 3D-PCA
# combined plots. Keyed by trajectory colouring mode.
MOUSE_ELEV_AZIM = {
    "G05": {"Time": (18, 4),   "Location": (19, 16)},
    "G06": {"Time": (17, 16),  "Location": (14, 10)},
    "G08": {"Time": (36, 11),  "Location": (30, 10)},
    "G09": {"Time": (27, 12),  "Location": (30, 10)},
    "G10": {"Time": (27, 30),  "Location": (24, 20)},
    "G11": {"Time": (24, 34),  "Location": (25, 22)},
    "G12": {"Time": (30, 10),  "Location": (30, 10)},
    "G13": {"Time": (30, 10),  "Location": (28, 12)},
    "G14": {"Time": (25, 14),  "Location": (24, 24)},
    "G16": {"Time": (30, 10),  "Location": (30, 10)},
    "G17": {"Time": (24, 22),  "Location": (21, 18)},
    "G18": {"Time": (25, 23),  "Location": (30, 10)},
    "G19": {"Time": (13, 11),  "Location": (13, 21)},
    "G20": {"Time": (24, -33), "Location": (26, -23)},
    "G21": {"Time": (3, 25),   "Location": (3, 25)},
}

EXCLUDE_MICE_CROSSREG = ("G07", "G15")
LOC_TRUNCATE_MICE = ("G09", "G21")  # Miniscope chunk 19.avi missing

MARKER_SIZE = 2
MARKER_ALPHA = 0.6
MARKER_SIZE_TONE = 5
MARKER_SIZE_SHOCK = 5
AZIM_ROTATION = 70

_METHODS_TEMPLATE_FILENAME = "population_pca_methods.txt"
_METHODS_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "analysis_methods_templates"
)

# NPY cache subdir under NPY_SAVE_PATH
_CACHE_ROOT = "Population_PCA"
_CACHE_SUBDIR_AVG = "trial_averaged"
_CACHE_SUBDIR_CROSSREG_AVG = "crossreg_avg"
_CACHE_SUBDIR_CROSSREG_FULL = "crossreg_full"


def _make_saver(npy_save_path, subdir):
    """Build a Saver under <NPY_SAVE_PATH>/Population_PCA/<subdir>, or None."""
    if npy_save_path is None:
        return None
    return Saver(parent_path=npy_save_path, subdirs=[_CACHE_ROOT, subdir])


def _cache_get(saver, key):
    """Return cached value for *key* if present, else None.

    DISABLED: the population-PCA pickles grew large enough to trigger
    MemoryError on load (and the speedup over recomputing PCA was
    marginal in practice). This function now always returns None so the
    pipeline always recomputes; remove the body below to re-enable.
    """
    return None


def _cache_put(saver, key, value):
    """No-op: see _cache_get docstring."""
    return


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _copy_methods_template(dest_dir):
    """Copy the population PCA METHODS template into *dest_dir* (no-op if present)."""
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, _METHODS_TEMPLATE_FILENAME)
    if os.path.isfile(dest_path):
        return
    src_path = os.path.join(_METHODS_TEMPLATES_DIR, _METHODS_TEMPLATE_FILENAME)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Population PCA METHODS template missing: {src_path}")
    shutil.copy2(src_path, dest_path)


def _write_run_config(dest_dir, *, pca_kind, normalize, binarize,
                       frames_per_bin, smoothing_active, smoothing_sigma,
                       cache_suffix):
    """Write a per-run RUN_CONFIG.txt sidecar into *dest_dir*.

    Always overwrites. Records the actual preprocessing values used by
    this PCA invocation so each output dir is self-documenting.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if normalize and binarize:
        mode = "INVALID(normalize+binarize)"
    elif normalize:
        mode = "normalize"
    elif binarize:
        mode = "binarize"
    else:
        mode = "default"
    bin_seconds = float(frames_per_bin) / float(MINISCOPE_FPS)
    text = (
        "Population PCA - RUN CONFIG\n"
        "===========================\n"
        f"pca_kind         : {pca_kind}\n"
        f"mode             : {mode}\n"
        f"normalize        : {bool(normalize)}\n"
        f"binarize         : {bool(binarize)}\n"
        f"frames_per_bin   : {int(frames_per_bin)}"
        + ("  (trial-averaged PCAs always use frames_per_bin=1)\n"
           if pca_kind in ("PCA_avg", "PCA_crossreg_avg") else "\n")
        + f"bin_seconds      : {bin_seconds:.4f}\n"
        f"smoothing_active : {bool(smoothing_active)}"
        + ("  (auto-disabled when frames_per_bin > 1)\n"
           if (frames_per_bin or 1) > 1 else "\n")
        + f"smoothing_sigma  : {float(smoothing_sigma)} frames"
        + ("  (only used when smoothing_active=True)\n")
        + f"cache_suffix     : {cache_suffix}\n"
        "\n"
        "See population_pca_methods.txt in this directory for definitions of\n"
        "each preprocessing mode and the binning / smoothing rules.\n"
    )
    with open(os.path.join(dest_dir, "RUN_CONFIG.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def _zscore_neurons(S):
    """Z-score each neuron's trace along time. NaN-safe."""
    mu = S.mean(axis=1, keepdims=True)
    sigma = S.std(axis=1, keepdims=True)
    return np.nan_to_num((S - mu) / sigma)


def _binarize_from_spikes(shape, full_idx, session_S_spikes):
    """Build a (n_rows, n_frames) binary spike-peak matrix.

    Parameters
    ----------
    shape : tuple
        (n_rows, n_frames) of the output matrix.
    full_idx : array-like[int]
        Full-session row index for each output row (length == shape[0]).
    session_S_spikes : dict[int, list[int]]
        ``session.S_spikes[full_i]`` -> list of frame indices where row
        ``full_i`` has a detected spike peak.

    Returns
    -------
    M : ndarray (shape, dtype=float)
        ``M[i, t] = 1.0`` iff ``t in session_S_spikes[full_idx[i]]``.

    Raises ``KeyError`` if any ``full_idx[i]`` is missing from
    ``session_S_spikes`` (no silent skips).
    """
    n_rows, n_frames = shape
    full_idx = np.asarray(full_idx, dtype=int)
    if full_idx.size != n_rows:
        raise ValueError(
            f"_binarize_from_spikes: full_idx size {full_idx.size} != n_rows {n_rows}"
        )
    M = np.zeros(shape, dtype=float)
    for i, full_i in enumerate(full_idx):
        full_i = int(full_i)
        if full_i not in session_S_spikes:
            raise KeyError(
                f"_binarize_from_spikes: row index {full_i} missing from S_spikes"
            )
        for t in session_S_spikes[full_i]:
            if 0 <= int(t) < n_frames:
                M[i, int(t)] = 1.0
    return M


def _peakval_normalize(S, full_idx, session_S_peakval):
    """Per-cell, per-session amplitude normalization (in place: returns new array).

    Each row of ``S`` is divided by ``max(|S_peakval[full_idx[i]]|)``.
    Cells with empty peakval lists pass through unchanged.

    Raises ``KeyError`` if any ``full_idx[i]`` is missing from
    ``session_S_peakval``.
    """
    full_idx = np.asarray(full_idx, dtype=int)
    n_rows = S.shape[0]
    if full_idx.size != n_rows:
        raise ValueError(
            f"_peakval_normalize: full_idx size {full_idx.size} != n_rows {n_rows}"
        )
    out = S.astype(float, copy=True)
    for i, full_i in enumerate(full_idx):
        full_i = int(full_i)
        if full_i not in session_S_peakval:
            raise KeyError(
                f"_peakval_normalize: row index {full_i} missing from S_peakval"
            )
        peakvals = np.asarray(session_S_peakval[full_i], dtype=float)
        if peakvals.size == 0:
            continue
        denom = float(np.max(np.abs(peakvals)))
        if denom == 0.0:
            continue
        out[i, :] = out[i, :] / denom
    return out


def _bin_time_axis(S, frames_per_bin):
    """Sum-reduce S along the time axis into ``frames_per_bin``-frame bins.

    Trailing frames that don't fill a complete bin are dropped. Returns the
    original matrix unchanged when ``frames_per_bin <= 1``.
    """
    fpb = int(frames_per_bin)
    if fpb < 1:
        raise ValueError(f"_bin_time_axis: frames_per_bin must be >= 1, got {fpb}")
    if fpb == 1:
        return S
    n_rows, n_frames = S.shape
    n_bins = n_frames // fpb
    if n_bins == 0:
        raise ValueError(
            f"_bin_time_axis: not enough frames ({n_frames}) for one bin of {fpb}"
        )
    truncated = S[:, : n_bins * fpb]
    return truncated.reshape(n_rows, n_bins, fpb).sum(axis=2)


def _bin_event_times(times, frames_per_bin):
    """Map a list of frame indices to bin indices via integer division.

    Returns the original list when ``frames_per_bin <= 1``.
    """
    fpb = int(frames_per_bin)
    if fpb <= 1:
        return list(times)
    return [int(int(t) // fpb) for t in times]


def _calculate_overlap(ax, PCs):
    """Heuristic 2D-projection overlap count for view-angle search."""
    proj = ax.get_proj()
    PCs_h = np.hstack([PCs, np.ones((PCs.shape[0], 1))])
    proj_pts = PCs_h @ proj.T
    proj_pts /= proj_pts[:, 3].reshape(-1, 1)
    hist, _, _ = np.histogram2d(proj_pts[:, 0], proj_pts[:, 1], bins=50)
    return int(np.sum(hist > 1))


def find_best_view(PCs):
    """Grid-search (elev, azim) minimising 2D projection overlap."""
    best_elev, best_azim = 30, 45
    min_overlap = float("inf")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        for elev in range(0, 90, 10):
            for azim in range(0, 360, 10):
                ax.view_init(elev=elev, azim=azim)
                ov = _calculate_overlap(ax, PCs)
                if ov < min_overlap:
                    min_overlap = ov
                    best_elev = elev
                    best_azim = azim
    finally:
        plt.close(fig)
    return best_elev, best_azim


def _trial_average(S, sess, has_shock):
    """Trial-average S across tone (and shock if has_shock) windows.

    Returns
    -------
    S_avg : ndarray (cells, total_frames)
        Concatenated period-averaged matrix (5 periods if has_shock else 3).
    period_slices : dict[str, slice]
        Slice into the time axis for each period.
    """
    pre_tone = PERIOD_FRAMES["pre_tone"]
    tone = PERIOD_FRAMES["tone"]
    post_tone = PERIOD_FRAMES["post_tone"]
    shock = PERIOD_FRAMES["shock"]
    post_shock = PERIOD_FRAMES["post_shock"]

    if has_shock:
        total = pre_tone + tone + post_tone + shock + post_shock
        periods = _AVG_PERIODS_TFC
    else:
        total = pre_tone + tone + post_tone
        periods = _AVG_PERIODS_TEST

    S_avg = np.zeros((S.shape[0], total))
    period_slices = {}
    cursor = 0
    for name in periods:
        period_slices[name] = slice(cursor, cursor + PERIOD_FRAMES[name])
        cursor += PERIOD_FRAMES[name]

    if len(sess.tone_onsets) == 0 or len(sess.tone_offsets) == 0:
        raise RuntimeError(f"Session {sess!r} has no tone events for trial averaging.")
    if has_shock and (len(sess.shock_onsets) == 0 or len(sess.shock_offsets) == 0):
        raise RuntimeError(f"Session {sess!r} marked has_shock=True but has no shock events.")

    for cell in range(S.shape[0]):
        pre_tone_avg, tone_avg, post_tone_avg = [], [], []
        shock_avg, post_shock_avg = [], []

        for tone_onset in sess.tone_onsets:
            pre_tone_avg.append(S[cell, tone_onset - pre_tone:tone_onset])
            tone_avg.append(S[cell, tone_onset:tone_onset + tone])
            post_tone_avg.append(
                S[cell, tone_onset + tone:tone_onset + tone + post_tone]
            )
        if has_shock:
            for shock_onset in sess.shock_onsets:
                shock_avg.append(S[cell, shock_onset:shock_onset + shock])
                post_shock_avg.append(
                    S[cell, shock_onset + shock:shock_onset + shock + post_shock]
                )

        S_avg[cell, period_slices["pre_tone"]] = np.mean(pre_tone_avg, axis=0)
        S_avg[cell, period_slices["tone"]] = np.mean(tone_avg, axis=0)
        S_avg[cell, period_slices["post_tone"]] = np.mean(post_tone_avg, axis=0)
        if has_shock:
            S_avg[cell, period_slices["shock"]] = np.mean(shock_avg, axis=0)
            S_avg[cell, period_slices["post_shock"]] = np.mean(post_shock_avg, axis=0)

    return S_avg, period_slices


def _save_fig(fig, save_dir, fname, want_svg, auto_close):
    os.makedirs(save_dir, exist_ok=True)
    _copy_methods_template(save_dir)
    fig.savefig(os.path.join(save_dir, f"{fname}.png"), format="png", dpi=600)
    if want_svg:
        svg_dir = os.path.join(save_dir, "svg")
        os.makedirs(svg_dir, exist_ok=True)
        fig.savefig(os.path.join(svg_dir, f"{fname}.svg"), format="svg")
    if auto_close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Base PCA class
# ---------------------------------------------------------------------------

class PopulationPCA:
    """Base class — single-session population PCA primitives."""

    def __init__(self, session, mouse, mouse_group,
                 normalize=False, binarize=False):
        if not hasattr(session, "S"):
            raise AttributeError(f"Session for mouse {mouse!r} has no .S matrix.")
        if normalize and binarize:
            raise ValueError("normalize and binarize are mutually exclusive")
        self.session = session
        self.mouse = mouse
        self.mouse_group = mouse_group
        self.normalize = bool(normalize)
        self.binarize = bool(binarize)
        S_raw = session.S
        if self.binarize:
            S_raw = _binarize_from_spikes(
                S_raw.shape, np.arange(S_raw.shape[0]), session.S_spikes
            )
        elif self.normalize:
            S_raw = _peakval_normalize(
                S_raw, np.arange(S_raw.shape[0]), session.S_peakval
            )
        self.S_raw = S_raw
        self.S_norm = None
        self.pca = None

    def z_normalize(self):
        if self.binarize:
            self.S_norm = self.S_raw.astype(float, copy=True)
        else:
            self.S_norm = _zscore_neurons(self.S_raw)
        return self.S_norm

    def apply_smoothing(self, sigma):
        if self.S_norm is None:
            raise RuntimeError("Call z_normalize() before apply_smoothing().")
        self.S_norm = gaussian_filter1d(self.S_norm, sigma=sigma, axis=1)
        return self.S_norm

    def fit_pca(self, fit_data, n_components=3):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(fit_data.T)
        return self.pca

    def transform(self, data):
        if self.pca is None:
            raise RuntimeError("Call fit_pca() before transform().")
        return self.pca.transform(data.T)


# ---------------------------------------------------------------------------
# Trial-averaged PCA (single session, TFC_cond)
# ---------------------------------------------------------------------------

class TrialAveragedPCA(PopulationPCA):
    """Single-session trial-averaged PCA (PCA_avg)."""

    def __init__(self, session, mouse, mouse_group,
                 normalize=False, binarize=False):
        super().__init__(session, mouse, mouse_group,
                         normalize=normalize, binarize=binarize)
        self.S_avg = None
        self.period_slices = None
        self.PCs = None
        self.best_elev = None
        self.best_azim = None

    def compute(self, n_components=3, saver=None):
        cache_key = (
            f"pca_avg-n{n_components}"
            f"-norm{int(self.normalize)}-bin{int(self.binarize)}"
            f"-{self.mouse}"
        )
        cached = _cache_get(saver, cache_key)
        if cached is not None:
            self.S_norm = cached["S_norm"]
            self.S_avg = cached["S_avg"]
            self.period_slices = cached["period_slices"]
            self.pca = cached["pca"]
            self.PCs = cached["PCs"]
            self.best_elev = cached["best_elev"]
            self.best_azim = cached["best_azim"]
            return self.PCs

        self.z_normalize()
        self.S_avg, self.period_slices = _trial_average(
            self.S_norm, self.session, has_shock=True
        )
        self.fit_pca(self.S_avg, n_components=n_components)
        self.PCs = self.transform(self.S_avg)
        self.best_elev, self.best_azim = find_best_view(self.PCs)

        _cache_put(saver, cache_key, {
            "S_norm": self.S_norm,
            "S_avg": self.S_avg,
            "period_slices": self.period_slices,
            "pca": self.pca,
            "PCs": self.PCs,
            "best_elev": self.best_elev,
            "best_azim": self.best_azim,
        })
        return self.PCs

    def free_heavy(self):
        """Drop large per-cell matrices to free RAM. Keeps lightweight
        outputs needed downstream (``PCs``, ``period_slices``,
        ``best_elev``/``best_azim``, ``mouse``, ``mouse_group``).
        Safe to call after ``plot()``."""
        self.S_norm = None
        self.S_avg = None
        self.S_raw = None
        self.session = None
        gc.collect()

    def plot(self, save_dir, want_svg=False, auto_close=True):
        if self.PCs is None:
            raise RuntimeError("Call compute() before plot().")
        _write_run_config(
            save_dir,
            pca_kind="PCA_avg",
            normalize=self.normalize,
            binarize=self.binarize,
            frames_per_bin=1,
            smoothing_active=False,
            smoothing_sigma=0.0,
            cache_suffix=(
                f"-norm{int(self.normalize)}-bin{int(self.binarize)}"
            ),
        )
        PCs = self.PCs
        best_elev, best_azim = self.best_elev, self.best_azim

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for name, sl in self.period_slices.items():
            ax.scatter(
                PCs[sl, 0], PCs[sl, 1], PCs[sl, 2],
                color=PERIOD_COLORS[name],
                s=100 if name == "shock" else 50,
                label=f"{name.replace('_', '-').capitalize()} Period",
            )
        ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color="gray", alpha=0.5,
                label="Trajectory")
        ax.scatter(PCs[0, 0], PCs[0, 1], PCs[0, 2], color="black", s=100,
                   marker="o", label="Start", zorder=10)
        ax.scatter(PCs[-1, 0], PCs[-1, 1], PCs[-1, 2], color="black", s=100,
                   facecolors="none", edgecolors="black", label="End", zorder=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.view_init(elev=best_elev, azim=best_azim)
        ax.set_title(f"{self.mouse} {self.mouse_group} trial-averaged PCA")
        ax.legend()

        fname = f"PCA_avg-{self.mouse_group}-{self.mouse}"
        _save_fig(fig, save_dir, fname, want_svg, auto_close)


# ---------------------------------------------------------------------------
# Crossreg PCA base
# ---------------------------------------------------------------------------

_SESS_NAMES = ("TFC_cond", "Test_B", "Test_B_1wk")


class CrossregPCA:
    """Cross-registered, multi-session PCA base."""

    def __init__(self, sessions, mouse, mouse_group, crossreg, mapping,
                 normalize=False, binarize=False):
        if mouse in EXCLUDE_MICE_CROSSREG:
            raise ValueError(f"Mouse {mouse} is excluded from crossreg analyses.")
        for name in _SESS_NAMES:
            if name not in sessions:
                raise KeyError(f"sessions missing required key {name!r}")
        if normalize and binarize:
            raise ValueError("normalize and binarize are mutually exclusive")
        self.mouse = mouse
        self.mouse_group = mouse_group
        self.normalize = bool(normalize)
        self.binarize = bool(binarize)
        self.sessions = {n: sessions[n] for n in _SESS_NAMES}
        self.crossreg = crossreg
        self.mapping = mapping

        # raw cross-registered S
        self.S_raw = {}
        # full-session row indices used to build each S_raw[name] (so the
        # unified engram identity, which lives in full-session row-index
        # space, can be intersected against the crossreg subset).
        self.S_full_idx = {}
        for name, sess in self.sessions.items():
            idx = get_S_indeces_crossreg(sess, crossreg, mapping)
            full_idx = np.asarray(idx, dtype=int)
            S_sub = sess.S[full_idx, :]
            if self.binarize:
                S_sub = _binarize_from_spikes(
                    S_sub.shape, full_idx, sess.S_spikes
                )
            elif self.normalize:
                S_sub = _peakval_normalize(
                    S_sub, full_idx, sess.S_peakval
                )
            self.S_raw[name] = S_sub
            self.S_full_idx[name] = full_idx
        self.S_norm = None
        self.engram_mask = None
        self.engram_skipped = False
        self.pca = None
        self.PCs = None

    def z_normalize(self):
        if self.binarize:
            self.S_norm = {n: s.astype(float, copy=True)
                           for n, s in self.S_raw.items()}
        else:
            self.S_norm = {n: _zscore_neurons(s) for n, s in self.S_raw.items()}
        return self.S_norm

    def apply_smoothing(self, sigma):
        if self.S_norm is None:
            raise RuntimeError("Call z_normalize() before apply_smoothing().")
        for n in self.S_norm:
            self.S_norm[n] = gaussian_filter1d(self.S_norm[n], sigma=sigma, axis=1)
        return self.S_norm

    def select_engram(self, engram_full_idx_in_ref, which_engram="encoding",
                      allow_empty=False):
        """Subset all S_norm matrices to engram cells.

        Parameters
        ----------
        engram_full_idx_in_ref : iterable[int]
            Row indices into the *reference session's full S* (TFC_cond
            for which_engram='encoding', Test_B for 'recall') that the
            unified engram pipeline (``caban.engram``) classified as
            engram for this mouse/mode. Cells in this list that are not
            cross-registered onto the current ``mapping`` are dropped.
        which_engram : {'encoding', 'recall'}
            Selects the reference session for the intersection.
        allow_empty : bool
            When True, an empty selection is permitted (the S_norm
            matrices become 0-row); callers should check
            ``self.engram_mask.size`` and skip downstream PCA fits.
        """
        if self.S_norm is None:
            raise RuntimeError("Call z_normalize() before select_engram().")
        if which_engram == "encoding":
            ref_name = "TFC_cond"
        elif which_engram == "recall":
            ref_name = "Test_B"
        else:
            raise ValueError(f"Unknown which_engram {which_engram!r}")
        eng_set = set(int(i) for i in engram_full_idx_in_ref)
        ref_full = self.S_full_idx[ref_name]
        # positional indices into S_norm[ref_name] rows where the
        # corresponding full-session cell is in the engram set.
        mask = np.array(
            [pos for pos, full_i in enumerate(ref_full)
             if int(full_i) in eng_set],
            dtype=int,
        )
        if mask.size == 0 and not allow_empty:
            raise RuntimeError(
                f"select_engram: 0 cells after intersecting engram set "
                f"({len(eng_set)} cells) with crossreg subset "
                f"(ref={ref_name}, n={ref_full.size})."
            )
        self.engram_mask = mask
        for n in self.S_norm:
            self.S_norm[n] = self.S_norm[n][mask, :]
        return mask

    def free_heavy(self):
        """Drop large per-cell matrices to free RAM. Keeps lightweight
        outputs needed downstream (``PCs``, ``engram_mask``, ``mouse``,
        ``mouse_group``, ``period_slices`` if set). Safe to call after
        ``run()`` + plotting + state-metric extraction.

        ``self.sessions`` is replaced with lightweight stubs that
        preserve only the event-timing attributes consumed by
        ``caban.pca_state_metrics`` (tone/shock onsets/offsets); the
        original session objects (which hold full S/C/YrA matrices) are
        dropped.
        """
        self.S_norm = None
        self.S_raw = None
        if self.sessions is not None:
            self.sessions = {
                name: SimpleNamespace(
                    tone_onsets=list(getattr(s, "tone_onsets", []) or []),
                    tone_offsets=list(getattr(s, "tone_offsets", []) or []),
                    shock_onsets=list(getattr(s, "shock_onsets", []) or []),
                    shock_offsets=list(getattr(s, "shock_offsets", []) or []),
                )
                for name, s in self.sessions.items()
            }
        self.crossreg = None
        if hasattr(self, "S_avg"):
            self.S_avg = None
        gc.collect()

    def fit_pca(self, method, n_components=3, source=None):
        """Fit PCA. *source* dict (name → matrix) defaults to self.S_norm.

        method == 1 : fit on ``source['TFC_cond']``.
        method == 2 : fit on the time-axis-concatenation of all three matrices.
        """
        if source is None:
            source = self.S_norm
        if source is None:
            raise RuntimeError("No data to fit PCA on (z_normalize first).")

        if method == 1:
            fit_data = source["TFC_cond"]
        elif method == 2:
            fit_data = np.hstack([source[n] for n in _SESS_NAMES])
        else:
            raise ValueError(f"Unknown PCA method: {method!r}")

        self.pca = PCA(n_components=n_components)
        self.pca.fit(fit_data.T)
        self.PCs = {n: self.pca.transform(source[n].T) for n in _SESS_NAMES}
        return self.PCs


# ---------------------------------------------------------------------------
# Crossreg AVG PCA
# ---------------------------------------------------------------------------

class CrossregAvgPCA(CrossregPCA):
    """Cross-registered, trial-averaged PCA across the three sessions."""

    def __init__(self, sessions, mouse, mouse_group, crossreg, mapping,
                 normalize=False, binarize=False):
        super().__init__(sessions, mouse, mouse_group, crossreg, mapping,
                         normalize=normalize, binarize=binarize)
        self.S_avg = None
        self.period_slices = None  # populated for TFC_cond (5 periods)

    def _build_avg(self):
        S_avg = {}
        period_slices = None
        for name, sess in self.sessions.items():
            has_shock = (name == "TFC_cond")
            S_avg_i, slices_i = _trial_average(
                self.S_norm[name], sess, has_shock=has_shock
            )
            S_avg[name] = S_avg_i
            if has_shock:
                period_slices = slices_i
        self.S_avg = S_avg
        self.period_slices = period_slices

    def run(self, methods, n_components, want_engram,
            save_dir_base, plots_dir, want_svg, auto_close, saver=None,
            engram_full_idx_in_ref=None, which_engram="encoding"):
        cache_key = (
            f"crossreg_avg-engram{want_engram}"
            f"-which{which_engram}"
            f"-eng_n{0 if engram_full_idx_in_ref is None else len(engram_full_idx_in_ref)}"
            f"-norm{int(self.normalize)}-bin{int(self.binarize)}"
            f"-n{n_components}-{self.mouse}"
        )
        cached = _cache_get(saver, cache_key)
        if cached is not None:
            self.S_norm = cached["S_norm"]
            self.S_avg = cached["S_avg"]
            self.period_slices = cached["period_slices"]
            self.engram_mask = cached.get("engram_mask")
            pca_by_method = cached["pca_by_method"]
            PCs_by_method = cached["PCs_by_method"]
        else:
            self.z_normalize()
            if want_engram:
                if engram_full_idx_in_ref is None:
                    raise RuntimeError(
                        "CrossregAvgPCA.run(want_engram=True) requires "
                        "engram_full_idx_in_ref."
                    )
                self.select_engram(engram_full_idx_in_ref,
                                   which_engram=which_engram,
                                   allow_empty=True)
                if self.engram_mask is not None and self.engram_mask.size == 0:
                    print(f"  [PCA engram skip] CrossregAvgPCA {self.mouse} "
                          f"({self.mouse_group}) which={which_engram}: "
                          f"0 engram cells passed -> skipping mouse for this mode",
                          flush=True)
                    self.engram_skipped = True
                    return
            self._build_avg()
            pca_by_method = {}
            PCs_by_method = {}
            for method in methods:
                self.fit_pca(method=method, n_components=n_components,
                             source=self.S_avg)
                pca_by_method[method] = self.pca
                PCs_by_method[method] = self.PCs
            _cache_put(saver, cache_key, {
                "S_norm": self.S_norm,
                "S_avg": self.S_avg,
                "period_slices": self.period_slices,
                "engram_mask": self.engram_mask,
                "pca_by_method": pca_by_method,
                "PCs_by_method": PCs_by_method,
            })

        out_subdir = (
            "PCA_crossreg_avg_engram" if want_engram else "PCA_crossreg_avg"
        )
        save_dir = os.path.join(plots_dir, out_subdir)
        _write_run_config(
            save_dir,
            pca_kind="PCA_crossreg_avg",
            normalize=self.normalize,
            binarize=self.binarize,
            frames_per_bin=1,
            smoothing_active=False,
            smoothing_sigma=0.0,
            cache_suffix=(
                f"-norm{int(self.normalize)}-bin{int(self.binarize)}"
            ),
        )

        for method in methods:
            if method not in PCs_by_method:
                raise KeyError(
                    f"Cached PCs for method {method} missing for {self.mouse}; "
                    f"delete cache file and re-run."
                )
            self.pca = pca_by_method[method]
            self.PCs = PCs_by_method[method]
            PCs = self.PCs

            all_PCs = np.vstack([PCs[n] for n in _SESS_NAMES])
            x_lim = (all_PCs[:, 0].min(), all_PCs[:, 0].max())
            y_lim = (all_PCs[:, 1].min(), all_PCs[:, 1].max())
            z_lim = (all_PCs[:, 2].min(), all_PCs[:, 2].max())
            tb_stack = np.vstack([PCs["Test_B"], PCs["Test_B_1wk"]])
            x_lim_tb = (tb_stack[:, 0].min(), tb_stack[:, 0].max())
            y_lim_tb = (tb_stack[:, 1].min(), tb_stack[:, 1].max())
            z_lim_tb = (tb_stack[:, 2].min(), tb_stack[:, 2].max())

            plot_types = ["full"]
            if method == 1:
                plot_types.append("zoom")

            for plot_type in plot_types:
                fig, axes = plt.subplots(
                    1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"}
                )
                for ax, name in zip(axes, _SESS_NAMES):
                    pcs = PCs[name]
                    has_shock = (name == "TFC_cond")
                    periods = _AVG_PERIODS_TFC if has_shock else _AVG_PERIODS_TEST
                    for pname in periods:
                        sl = self.period_slices[pname]
                        ax.scatter(
                            pcs[sl, 0], pcs[sl, 1], pcs[sl, 2],
                            color=PERIOD_COLORS[pname],
                            s=100 if pname == "shock" else 50,
                            label=f"{pname.replace('_', '-').capitalize()} Period",
                        )
                    ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], color="gray",
                            alpha=0.5, label="Trajectory")
                    ax.scatter(pcs[0, 0], pcs[0, 1], pcs[0, 2], color="black",
                               s=200, marker="o", label="Start", zorder=10)
                    ax.scatter(pcs[-1, 0], pcs[-1, 1], pcs[-1, 2], color="black",
                               s=200, facecolors="none", edgecolors="black",
                               label="End", zorder=10)
                    if plot_type == "full":
                        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
                    else:
                        ax.set_xlim(x_lim_tb); ax.set_ylim(y_lim_tb); ax.set_zlim(z_lim_tb)
                    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
                    ax.set_title(name)
                    ax.legend(fontsize="small")

                fig.suptitle(
                    f"{self.mouse} {self.mouse_group} PCA crossreg "
                    f"trial-avg ({plot_type})"
                )
                fname = (
                    f"PCA_crossreg_avg-want_engram_{want_engram}"
                    f"-n_components{n_components}-method{method}-{plot_type}"
                    f"-{self.mouse_group}-{self.mouse}"
                )
                _save_fig(fig, save_dir, fname, want_svg, auto_close)


# ---------------------------------------------------------------------------
# Crossreg FULL PCA (full time course)
# ---------------------------------------------------------------------------

class CrossregFullPCA(CrossregPCA):
    """Cross-registered, full-time-course PCA with tone/shock highlighting."""

    def __init__(self, sessions, mouse, mouse_group, crossreg, mapping,
                 event_window, normalize=False, binarize=False,
                 frames_per_bin=1):
        super().__init__(sessions, mouse, mouse_group, crossreg, mapping,
                         normalize=normalize, binarize=binarize)
        if event_window not in ("toneshock", "posttoneshock"):
            raise ValueError(f"Unknown event_window: {event_window!r}")
        self.event_window = event_window
        fpb = int(frames_per_bin)
        if fpb < 1:
            raise ValueError(f"frames_per_bin must be >= 1, got {fpb}")
        self.frames_per_bin = fpb

        # Time-bin the per-session S_raw and event timing now (before
        # z_normalize / smoothing / PCA fit see them). Trial-averaged
        # subclasses do not call this code path.
        if self.frames_per_bin > 1:
            self._apply_time_binning()

    def _apply_time_binning(self):
        """Sum-bin S_raw along time, and remap tone/shock onsets/offsets.

        Replaces ``self.sessions[name]`` with SimpleNamespace stubs whose
        tone/shock indices are bin-space (so ``_tone_windows`` /
        ``_shock_windows`` and downstream state-metrics work in bin-space
        coordinates). The original session objects are released.
        """
        fpb = self.frames_per_bin
        for name in list(self.S_raw.keys()):
            self.S_raw[name] = _bin_time_axis(self.S_raw[name], fpb)
        new_sessions = {}
        for name, s in self.sessions.items():
            stub = SimpleNamespace(
                tone_onsets=_bin_event_times(
                    list(getattr(s, "tone_onsets", []) or []), fpb),
                tone_offsets=_bin_event_times(
                    list(getattr(s, "tone_offsets", []) or []), fpb),
                shock_onsets=_bin_event_times(
                    list(getattr(s, "shock_onsets", []) or []), fpb),
                shock_offsets=_bin_event_times(
                    list(getattr(s, "shock_offsets", []) or []), fpb),
            )
            # Carry over per-frame location traces, mean-binned to match
            # the new time axis (consumed by ``_plot_2dloc_3dpca``).
            for loc_attr in ("loc_X_miniscope_smooth", "loc_Y_miniscope_smooth"):
                arr = getattr(s, loc_attr, None)
                if arr is None:
                    setattr(stub, loc_attr, None)
                    continue
                arr = np.asarray(arr, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(
                        f"_apply_time_binning: {loc_attr} expected 1D, got {arr.shape}"
                    )
                n_bins = arr.shape[0] // fpb
                if n_bins == 0:
                    setattr(stub, loc_attr, arr[:0])
                else:
                    setattr(stub, loc_attr,
                            arr[: n_bins * fpb].reshape(n_bins, fpb).mean(axis=1))
            new_sessions[name] = stub
        self.sessions = new_sessions

    # ---- event window helpers ----------------------------------------

    def _tone_windows(self, sess):
        """Yield (onset, offset) pairs for tone highlight windows."""
        # Window extension is 20 s; convert to bin-units when binning is on.
        ext = max(1, int((MINISCOPE_FPS * 20) // max(1, self.frames_per_bin)))
        if self.event_window == "toneshock":
            return list(zip(sess.tone_onsets, sess.tone_offsets))
        # posttoneshock: 20 s after tone offset
        offs = np.asarray(sess.tone_offsets)
        return list(zip(offs.tolist(), (offs + ext).tolist()))

    def _shock_windows(self, sess):
        ext = max(1, int((MINISCOPE_FPS * 20) // max(1, self.frames_per_bin)))
        if self.event_window == "toneshock":
            return list(zip(sess.shock_onsets, sess.shock_offsets))
        offs = np.asarray(sess.shock_offsets)
        return list(zip(offs.tolist(), (offs + ext).tolist()))

    def _gather_event_PCs(self):
        """Concatenate PCs over all tone/shock event windows for axis limits."""
        chunks = []
        for name in _SESS_NAMES:
            pcs = self.PCs[name]
            sess = self.sessions[name]
            for onset, offset in self._tone_windows(sess):
                chunks.append(pcs[onset:offset])
        for onset, offset in self._shock_windows(self.sessions["TFC_cond"]):
            chunks.append(self.PCs["TFC_cond"][onset:offset])
        return np.vstack(chunks)

    # ---- 2D location + 3D PCA combined plot (toneshock only) ---------

    def _plot_2dloc_3dpca(self, sess_name, save_dir, want_svg, auto_close,
                          method, only_tone_shock, preset_angles,
                          auto_angle_adjust, axis_limits):
        sess = self.sessions[sess_name]
        pcs = self.PCs[sess_name]

        x = np.asarray(sess.loc_X_miniscope_smooth)
        y = np.asarray(sess.loc_Y_miniscope_smooth)
        if self.mouse in LOC_TRUNCATE_MICE:
            x = x[:pcs.shape[0]]
            y = y[:pcs.shape[0]]
        elif x.shape[0] < pcs.shape[0] or y.shape[0] < pcs.shape[0]:
            raise RuntimeError(
                f"{self.mouse}/{sess_name}: location arrays "
                f"(x={x.shape}, y={y.shape}) shorter than PCs={pcs.shape[0]} frames."
            )
        else:
            x = x[:pcs.shape[0]]
            y = y[:pcs.shape[0]]

        for trajectory_type in ("Time", "Location"):
            if trajectory_type == "Time":
                values = np.linspace(0, 1, len(x))
                colorbar_label = "Time (frames)"
            else:
                values = np.sqrt(x ** 2 + y ** 2)
                colorbar_label = "Distance from Origin"

            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection="3d")

            sc = ax1.scatter(x, y, c=values, cmap="viridis", s=1)
            fig.colorbar(sc, ax=ax1, label=colorbar_label)
            ax1.set_xlabel("X Position"); ax1.set_ylabel("Y Position")

            ax2.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], color="gray",
                     alpha=0.5, label="Trajectory")
            ax2.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=values,
                        cmap="viridis", alpha=0.6)
            x_lim, y_lim, z_lim = axis_limits[
                "tone_shock" if only_tone_shock else "full"
            ]
            ax2.set_xlim(x_lim); ax2.set_ylim(y_lim); ax2.set_zlim(z_lim)
            ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")

            if preset_angles:
                if self.mouse not in MOUSE_ELEV_AZIM:
                    raise KeyError(
                        f"No MOUSE_ELEV_AZIM preset for {self.mouse!r}; set "
                        f"preset_angles=False to fall back to find_best_view."
                    )
                elev, azim = MOUSE_ELEV_AZIM[self.mouse][trajectory_type]
                ax2.view_init(elev=elev, azim=azim)
            elif auto_angle_adjust:
                elev, azim = find_best_view(pcs)
                ax2.view_init(elev=elev, azim=azim)

            fig.suptitle(
                f"Mouse trajectory in 2D and PC space {self.mouse} "
                f"{self.mouse_group} {sess_name} by {trajectory_type}"
            )
            fname = (
                f"trajectory-2Dloc_and_PCA_crossreg_full-method{method}"
                f"-type-{trajectory_type}-{self.mouse_group}-{self.mouse}"
                f"-{sess_name}"
            )
            _save_fig(fig, save_dir, fname, want_svg, auto_close)

    # ---- per-session 1×3 PCA panel -----------------------------------

    def _plot_three_panel(self, save_dir, want_svg, auto_close,
                          method, want_engram, only_tone_shock,
                          auto_angle_adjust, axis_limits):
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"}
        )
        for ax, name in zip(axes, _SESS_NAMES):
            pcs = self.PCs[name]
            sess = self.sessions[name]

            if not only_tone_shock and self.event_window == "toneshock":
                ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2],
                           c=np.arange(pcs.shape[0]), cmap="viridis",
                           alpha=0.6, s=MARKER_SIZE, label="Trajectory")
            if self.event_window == "toneshock":
                ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], color="gray",
                        alpha=0.5, label="Trajectory")
            elif not only_tone_shock:
                ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], color="gray",
                        alpha=0.6, label="Trajectory")

            tone_windows = self._tone_windows(sess)
            tone_marker = "^" if self.event_window == "toneshock" else "s"
            for i, (onset, offset) in enumerate(tone_windows):
                if i == 0:
                    color_dark = "darkblue" if self.event_window == "toneshock" else "blue"
                    ax.scatter(pcs[onset:offset, 0], pcs[onset:offset, 1],
                               pcs[onset:offset, 2], color=color_dark,
                               s=MARKER_SIZE_TONE, marker=tone_marker)
                    if self.event_window == "toneshock":
                        ax.plot(pcs[onset:offset, 0], pcs[onset:offset, 1],
                                pcs[onset:offset, 2], color="blue", alpha=0.6,
                                linewidth=0.5, label="Tone Trajectory")
                elif i == len(tone_windows) - 1:
                    color_light = (
                        "lightblue" if self.event_window == "toneshock"
                        else "lightskyblue"
                    )
                    ax.scatter(pcs[onset:offset, 0], pcs[onset:offset, 1],
                               pcs[onset:offset, 2], color=color_light,
                               s=MARKER_SIZE_TONE, marker=tone_marker)
                    if self.event_window == "toneshock":
                        ax.plot(pcs[onset:offset, 0], pcs[onset:offset, 1],
                                pcs[onset:offset, 2], color="lightblue",
                                alpha=0.6, linewidth=0.5, label="Tone Trajectory")

            if name == "TFC_cond":
                shock_windows = self._shock_windows(sess)
                shock_marker = "x" if self.event_window == "toneshock" else "o"
                for i, (onset, offset) in enumerate(shock_windows):
                    if i == 0:
                        color_dark = "darkred" if self.event_window == "toneshock" else "red"
                        ax.scatter(pcs[onset:offset, 0], pcs[onset:offset, 1],
                                   pcs[onset:offset, 2], color=color_dark,
                                   s=MARKER_SIZE_SHOCK, marker=shock_marker)
                        if self.event_window == "toneshock":
                            ax.plot(pcs[onset:offset, 0], pcs[onset:offset, 1],
                                    pcs[onset:offset, 2], color="darkred",
                                    alpha=0.6, linewidth=0.5,
                                    label="Shock Trajectory")
                    elif i == len(shock_windows) - 1:
                        color_light = (
                            "lightcoral" if self.event_window == "toneshock"
                            else "salmon"
                        )
                        ax.scatter(pcs[onset:offset, 0], pcs[onset:offset, 1],
                                   pcs[onset:offset, 2], color=color_light,
                                   s=MARKER_SIZE_SHOCK, marker=shock_marker)
                        if self.event_window == "toneshock":
                            ax.plot(pcs[onset:offset, 0], pcs[onset:offset, 1],
                                    pcs[onset:offset, 2], color="lightcoral",
                                    alpha=0.6, linewidth=0.5,
                                    label="Shock Trajectory")

            ax.scatter(pcs[0, 0], pcs[0, 1], pcs[0, 2], color="black", s=200,
                       marker="o", label="Start", zorder=10)
            ax.scatter(pcs[-1, 0], pcs[-1, 1], pcs[-1, 2], color="black", s=200,
                       facecolors="none", edgecolors="black", label="End",
                       zorder=10)

            x_lim, y_lim, z_lim = axis_limits[
                "tone_shock" if only_tone_shock else "full"
            ]
            ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            elev, azim = ax.elev, ax.azim
            ax.view_init(elev=elev, azim=azim + AZIM_ROTATION)
            if auto_angle_adjust:
                elev_b, azim_b = find_best_view(pcs)
                ax.view_init(elev=elev_b, azim=azim_b)
            ax.set_title(name)

        fig.suptitle(
            f"{self.mouse} {self.mouse_group} PCA crossreg full time course "
            f"(method {method})"
        )
        if self.event_window == "toneshock":
            subdir = "PCA_crossreg_full_engram_toneshock" if want_engram else "PCA_crossreg_full_toneshock"
        else:
            subdir = "PCA_crossreg_full_posttoneshock_engram" if want_engram else "PCA_crossreg_full_posttoneshock"
        fname = (
            f"PCA_crossreg_full-want_engram-{want_engram}"
            f"-only_tone_shock_{only_tone_shock}-method{method}"
            f"-{self.mouse_group}-{self.mouse}"
        )
        _save_fig(fig, save_dir, fname, want_svg, auto_close)
        return subdir

    # ---- top-level run -----------------------------------------------

    def run(self, methods, n_components, want_engram,
            use_gaussian_smoothing, smoothing_sigma,
            only_tone_shock, preset_angles, auto_angle_adjust,
            plots_dir, want_svg, auto_close, saver=None,
            engram_full_idx_in_ref=None, which_engram="encoding"):
        cache_key = (
            f"crossreg_full-{self.event_window}"
            f"-engram{want_engram}"
            f"-which{which_engram}"
            f"-eng_n{0 if engram_full_idx_in_ref is None else len(engram_full_idx_in_ref)}"
            f"-smooth{use_gaussian_smoothing}_sigma{smoothing_sigma}"
            f"-norm{int(self.normalize)}-bin{int(self.binarize)}"
            f"-fpb{self.frames_per_bin}"
            f"-n{n_components}-{self.mouse}"
        )
        cached = _cache_get(saver, cache_key)
        if cached is not None:
            self.S_norm = cached["S_norm"]
            self.engram_mask = cached.get("engram_mask")
            pca_by_method = cached["pca_by_method"]
            PCs_by_method = cached["PCs_by_method"]
        else:
            self.z_normalize()
            if use_gaussian_smoothing and self.frames_per_bin <= 1:
                self.apply_smoothing(smoothing_sigma)
            elif use_gaussian_smoothing and self.frames_per_bin > 1:
                print(
                    f"  [PCA smoothing skip] CrossregFullPCA "
                    f"({self.event_window}) {self.mouse}: "
                    f"frames_per_bin={self.frames_per_bin} > 1, "
                    f"suppressing sigma={smoothing_sigma}",
                    flush=True,
                )
            if want_engram:
                if engram_full_idx_in_ref is None:
                    raise RuntimeError(
                        "CrossregFullPCA.run(want_engram=True) requires "
                        "engram_full_idx_in_ref."
                    )
                self.select_engram(engram_full_idx_in_ref,
                                   which_engram=which_engram,
                                   allow_empty=True)
                if self.engram_mask is not None and self.engram_mask.size == 0:
                    print(f"  [PCA engram skip] CrossregFullPCA "
                          f"({self.event_window}) {self.mouse} "
                          f"({self.mouse_group}) which={which_engram}: "
                          f"0 engram cells passed -> skipping mouse for this mode",
                          flush=True)
                    self.engram_skipped = True
                    return
            pca_by_method = {}
            PCs_by_method = {}
            for method in methods:
                self.fit_pca(method=method, n_components=n_components)
                pca_by_method[method] = self.pca
                PCs_by_method[method] = self.PCs
            _cache_put(saver, cache_key, {
                "S_norm": self.S_norm,
                "engram_mask": self.engram_mask,
                "pca_by_method": pca_by_method,
                "PCs_by_method": PCs_by_method,
            })

        if self.event_window == "toneshock":
            subdir = "PCA_crossreg_full_engram_toneshock" if want_engram else "PCA_crossreg_full_toneshock"
        else:
            subdir = "PCA_crossreg_full_posttoneshock_engram" if want_engram else "PCA_crossreg_full_posttoneshock"
        save_dir = os.path.join(plots_dir, subdir)
        _smoothing_active = bool(use_gaussian_smoothing) and self.frames_per_bin <= 1
        _write_run_config(
            save_dir,
            pca_kind=f"PCA_crossreg_full_{self.event_window}",
            normalize=self.normalize,
            binarize=self.binarize,
            frames_per_bin=self.frames_per_bin,
            smoothing_active=_smoothing_active,
            smoothing_sigma=smoothing_sigma,
            cache_suffix=(
                f"-norm{int(self.normalize)}-bin{int(self.binarize)}"
                f"-fpb{self.frames_per_bin}"
            ),
        )

        for method in methods:
            if method not in PCs_by_method:
                raise KeyError(
                    f"Cached PCs for method {method} missing for {self.mouse}; "
                    f"delete cache file and re-run."
                )
            self.pca = pca_by_method[method]
            self.PCs = PCs_by_method[method]
            PCs = self.PCs

            all_PCs = np.vstack([PCs[n] for n in _SESS_NAMES])
            full_lims = (
                (all_PCs[:, 0].min(), all_PCs[:, 0].max()),
                (all_PCs[:, 1].min(), all_PCs[:, 1].max()),
                (all_PCs[:, 2].min(), all_PCs[:, 2].max()),
            )
            event_PCs = self._gather_event_PCs()
            ts_lims = (
                (event_PCs[:, 0].min(), event_PCs[:, 0].max()),
                (event_PCs[:, 1].min(), event_PCs[:, 1].max()),
                (event_PCs[:, 2].min(), event_PCs[:, 2].max()),
            )
            axis_limits = {"full": full_lims, "tone_shock": ts_lims}

            # 2D location + 3D PCA combined plot — toneshock pipeline only
            if self.event_window == "toneshock":
                for sess_name in _SESS_NAMES:
                    self._plot_2dloc_3dpca(
                        sess_name, save_dir, want_svg, auto_close,
                        method=method, only_tone_shock=only_tone_shock,
                        preset_angles=preset_angles,
                        auto_angle_adjust=auto_angle_adjust,
                        axis_limits=axis_limits,
                    )

            self._plot_three_panel(
                save_dir, want_svg, auto_close,
                method=method, want_engram=want_engram,
                only_tone_shock=only_tone_shock,
                auto_angle_adjust=auto_angle_adjust,
                axis_limits=axis_limits,
            )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_population_pca_pipeline(
    TFC_cond,
    Test_B,
    Test_B_1wk,
    mouse_groups,
    TFC_B_B_1wk_crossreg,
    mapping_TFC_cond_Test_B_Test_B_1wk,
    PLOTS_DIR,
    NPY_SAVE_PATH=None,
    auto_close=True,
    want_svg=False,
    smoothing_sigma=1.5,
    use_gaussian_smoothing=True,
    want_engram=True,
    only_tone_shock=False,
    preset_angles=True,
    auto_angle_adjust=False,
    avg_methods=(1, 2),
    full_methods=(2,),
    n_components=3,
    engram_idx_by_mouse=None,
    which_engram="encoding",
    normalize=False,
    binarize=False,
    frames_per_bin=1,
):
    """Run the complete population PCA analysis pipeline.

    Engram cells are supplied via ``engram_idx_by_mouse``: a dict
    ``{mouse: ndarray of full reference-session row indices}`` produced by
    the unified ``caban.engram.build_engram_identity`` pipeline, already
    resolved for the target mode and ``which_engram``. Pass ``None`` (or
    ``want_engram=False``) to disable engram filtering.

    Pipelines run, per mouse:
      1. ``TrialAveragedPCA`` on TFC_cond → ``PCA_avg/``
      2. ``CrossregAvgPCA``  (methods 1 & 2) → ``PCA_crossreg_avg[_engram]/``
      3. ``CrossregFullPCA(event_window='toneshock')``     (method 2) →
         ``PCA_crossreg_full_toneshock[_engram]/``
      4. ``CrossregFullPCA(event_window='posttoneshock')`` (method 2) →
         ``PCA_crossreg_full_posttoneshock[_engram]/``

    Mice in ``EXCLUDE_MICE_CROSSREG`` (G07, G15) are skipped for crossreg
    analyses but are still included in pipeline (1).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    saver_avg = _make_saver(NPY_SAVE_PATH, _CACHE_SUBDIR_AVG)
    saver_crossreg_avg = _make_saver(NPY_SAVE_PATH, _CACHE_SUBDIR_CROSSREG_AVG)
    saver_crossreg_full = _make_saver(NPY_SAVE_PATH, _CACHE_SUBDIR_CROSSREG_FULL)
    if want_engram and engram_idx_by_mouse is None:
        raise RuntimeError(
            "run_population_pca_pipeline(want_engram=True) requires "
            "engram_idx_by_mouse — pass the unified engram identity for the "
            "chosen (etype, mode)."
        )
    results = {}

    # --- (1) single-session trial-averaged PCA on TFC_cond ----------------
    for m, group in mouse_groups.items():
        if m not in TFC_cond:
            raise KeyError(f"TFC_cond is missing mouse {m!r}")
        print(f"*** PCA_avg: processing {m} {group} ...", end="", flush=True)
        avg = TrialAveragedPCA(TFC_cond[m], m, group,
                                normalize=normalize, binarize=binarize)
        avg.compute(n_components=n_components, saver=saver_avg)
        avg.plot(
            save_dir=os.path.join(PLOTS_DIR, "PCA_avg"),
            want_svg=want_svg,
            auto_close=auto_close,
        )
        avg.free_heavy()
        results.setdefault(m, {})["avg"] = avg
        print("done.", flush=True)

    # --- (2-4) crossreg pipelines -----------------------------------------
    for m, group in mouse_groups.items():
        if m in EXCLUDE_MICE_CROSSREG:
            continue
        for src, name in ((TFC_cond, "TFC_cond"), (Test_B, "Test_B"),
                          (Test_B_1wk, "Test_B_1wk")):
            if m not in src:
                raise KeyError(f"{name} is missing mouse {m!r}")
        sessions = {
            "TFC_cond": TFC_cond[m],
            "Test_B": Test_B[m],
            "Test_B_1wk": Test_B_1wk[m],
        }
        if m not in TFC_B_B_1wk_crossreg:
            raise KeyError(f"TFC_B_B_1wk_crossreg is missing mouse {m!r}")
        crossreg = TFC_B_B_1wk_crossreg[m]

        # (2) crossreg AVG
        print(f"*** PCA_crossreg_avg: processing {m} {group} ", end="", flush=True)
        avg_x = CrossregAvgPCA(sessions, m, group, crossreg,
                                mapping_TFC_cond_Test_B_Test_B_1wk,
                                normalize=normalize, binarize=binarize)
        avg_x.run(
            methods=avg_methods,
            n_components=n_components,
            want_engram=want_engram,
            save_dir_base=PLOTS_DIR,
            plots_dir=PLOTS_DIR,
            want_svg=want_svg,
            auto_close=auto_close,
            saver=saver_crossreg_avg,
            engram_full_idx_in_ref=(
                None if not want_engram else engram_idx_by_mouse.get(m)
            ),
            which_engram=which_engram,
        )
        if not getattr(avg_x, "engram_skipped", False):
            avg_x.free_heavy()
            results.setdefault(m, {})["crossreg_avg"] = avg_x
        print("done.", flush=True)

        # (3) crossreg FULL — tone/shock
        print(f"*** PCA_crossreg_full_toneshock: processing {m} {group} ",
              end="", flush=True)
        full_ts = CrossregFullPCA(
            sessions, m, group, crossreg,
            mapping_TFC_cond_Test_B_Test_B_1wk,
            event_window="toneshock",
            normalize=normalize, binarize=binarize,
            frames_per_bin=frames_per_bin,
        )
        full_ts.run(
            methods=full_methods,
            n_components=n_components,
            want_engram=want_engram,
            use_gaussian_smoothing=use_gaussian_smoothing,
            smoothing_sigma=smoothing_sigma,
            only_tone_shock=only_tone_shock,
            preset_angles=preset_angles,
            auto_angle_adjust=auto_angle_adjust,
            plots_dir=PLOTS_DIR,
            want_svg=want_svg,
            auto_close=auto_close,
            saver=saver_crossreg_full,
            engram_full_idx_in_ref=(
                None if not want_engram else engram_idx_by_mouse.get(m)
            ),
            which_engram=which_engram,
        )
        if not getattr(full_ts, "engram_skipped", False):
            full_ts.free_heavy()
            results.setdefault(m, {})["crossreg_full_toneshock"] = full_ts
        print("done.", flush=True)

        # (4) crossreg FULL — post-tone/shock
        print(f"*** PCA_crossreg_full_posttoneshock: processing {m} {group} ",
              end="", flush=True)
        full_pts = CrossregFullPCA(
            sessions, m, group, crossreg,
            mapping_TFC_cond_Test_B_Test_B_1wk,
            event_window="posttoneshock",
            normalize=normalize, binarize=binarize,
            frames_per_bin=frames_per_bin,
        )
        full_pts.run(
            methods=full_methods,
            n_components=n_components,
            want_engram=want_engram,
            use_gaussian_smoothing=False,  # post-event pipeline does not smooth
            smoothing_sigma=smoothing_sigma,
            only_tone_shock=only_tone_shock,
            preset_angles=False,  # no preset angles for post-event panels
            auto_angle_adjust=auto_angle_adjust,
            plots_dir=PLOTS_DIR,
            want_svg=want_svg,
            auto_close=auto_close,
            saver=saver_crossreg_full,
            engram_full_idx_in_ref=(
                None if not want_engram else engram_idx_by_mouse.get(m)
            ),
            which_engram=which_engram,
        )
        if not getattr(full_pts, "engram_skipped", False):
            full_pts.free_heavy()
            results.setdefault(m, {})["crossreg_full_posttoneshock"] = full_pts
        print("done.", flush=True)
        gc.collect()

    return results


def run_pca_state_metrics_from_results(
    results,
    mouse_groups,
    PLOTS_DIR,
    want_svg=False,
    auto_close=True,
):
    """Compute joint-PCA state-space metrics from a populated results dict.

    Designed to be called *after* ``run_population_pca_pipeline`` so that
    expensive PCA fits don't have to be redone each time the metric panel
    is iterated on. Reads each mouse's ``CrossregFullPCA`` (toneshock,
    method 2) from ``results[mouse]['crossreg_full_toneshock']`` and writes
    the per-mouse table and per-metric Nature-style boxplots
    + ``__stats.txt`` files into ``<PLOTS_DIR>/PCA_state_metrics/``.

    Parameters
    ----------
    results : dict
        Output of ``run_population_pca_pipeline``.
    mouse_groups : dict[str, str]
        Mouse-id → group string (mCherry / hM3D / hM4D).
    PLOTS_DIR : str
        Same plot root passed to ``run_population_pca_pipeline``.
    want_svg, auto_close
        Plotting options.

    Returns
    -------
    dict
        Whatever ``run_pca_state_metrics_pipeline`` returns
        (``per_mouse_df``, ``group_stats``).
    """
    full_pca_by_mouse = {
        m: results[m]["crossreg_full_toneshock"]
        for m in results
        if isinstance(results.get(m), dict)
        and "crossreg_full_toneshock" in results[m]
    }
    if not full_pca_by_mouse:
        raise RuntimeError(
            "results dict has no 'crossreg_full_toneshock' entries; "
            "run run_population_pca_pipeline() first."
        )
    groups_subset = {m: mouse_groups[m] for m in full_pca_by_mouse}
    state_dir = os.path.join(PLOTS_DIR, "PCA_state_metrics")
    print(f"*** PCA state metrics: writing to {state_dir}", flush=True)
    print(f"*** PCA state metrics: {len(full_pca_by_mouse)} mice", flush=True)
    out = run_pca_state_metrics_pipeline(
        full_pca_by_mouse=full_pca_by_mouse,
        mouse_groups=groups_subset,
        plots_dir=PLOTS_DIR,
        want_svg=want_svg,
        auto_close=auto_close,
    )
    print(f"*** PCA state metrics: done. Output in {state_dir}", flush=True)
    return out

print("caban.population.py loaded.")
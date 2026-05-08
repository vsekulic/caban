from random import shuffle
import numpy as np
from SSTCa2_utilities import *
from scipy.ndimage import gaussian_filter
from numpy.random import default_rng
import time
from sklearn import mixture
from sklearn.cluster import KMeans
from numpy import linalg
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import statsmodels.formula.api as smf
import pandas as pd
import os
from patsy import dmatrix
from scipy.stats import norm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import chi2

@dataclass
class LapSegmentation:
    lap_idx: np.ndarray            # (T,) lap id per sample, -1 invalid/unassigned
    lap_dir: np.ndarray            # (T,) direction sign estimate: +1/-1/0
    turn_mask: np.ndarray          # (T,) True where a turnaround is detected
    valid_mask: np.ndarray         # (T,) True where sample is usable (not a huge jump, finite, etc.)
    jump_mask: np.ndarray          # (T,) True where a large discontinuity was detected
    lap_start_inds: np.ndarray     # (n_laps,)
    lap_end_inds: np.ndarray       # (n_laps,)
    n_laps: int

    def lap_inds(self, k: int) -> np.ndarray:
        return np.where(self.lap_idx == k)[0]

    def even_mask(self) -> np.ndarray:
        return (self.lap_idx >= 0) & (self.lap_idx % 2 == 0)

    def odd_mask(self) -> np.ndarray:
        return (self.lap_idx >= 0) & (self.lap_idx % 2 == 1)

class Laps1D:
    """
    1D lap segmentation for linear track with:
      - laps defined by end-zone arrivals (left<->right)
      - back-and-forth within a lap allowed
      - robust handling of large discontinuities in x(t)
    """
    def __init__(
        self,
        x: np.ndarray,
        t: Optional[np.ndarray] = None,
        *,
        endzone_eps: Optional[float] = None,
        endzone_frac: float = 0.05,
        smooth_win: int = 9,
        min_vel_abs: float = 1e-6,
        min_turn_separation: int = 5,
        require_start_in_endzone: bool = True,
        jump_thresh: Optional[float] = None,
        jump_thresh_frac_of_range: float = 0.25,
    ):
        """
        Args:
            x: 1D position (T,)
            t: optional time in ms or s (T,) — only used for velocity scale; sign works regardless.
            endzone_eps: absolute endzone width. If None, uses endzone_frac * range.
            endzone_frac: fraction of track range for endzones (default 5%).
            smooth_win: boxcar smoothing window for velocity.
            min_vel_abs: small velocities treated as 0.
            min_turn_separation: suppress turns closer than this many samples.
            require_start_in_endzone: if True, lap 0 starts at first time entering an endzone.
            jump_thresh: absolute threshold for discontinuities in x between samples. If None,
                        uses jump_thresh_frac_of_range * range.
            jump_thresh_frac_of_range: default 0.25 * range.
        """
        self.x = np.asarray(x, dtype=float)
        self.t = None if t is None else np.asarray(t, dtype=float)
        if self.t is not None and self.t.shape != self.x.shape:
            raise ValueError("t must match x shape")

        self.T = self.x.size
        self.xmin = np.nanmin(self.x)
        self.xmax = np.nanmax(self.x)
        self.xrange = self.xmax - self.xmin

        if endzone_eps is None:
            endzone_eps = endzone_frac * self.xrange
        self.endzone_eps = float(endzone_eps)

        if jump_thresh is None:
            jump_thresh = jump_thresh_frac_of_range * self.xrange
        self.jump_thresh = float(jump_thresh)

        self.smooth_win = int(smooth_win)
        self.min_vel_abs = float(min_vel_abs)
        self.min_turn_separation = int(min_turn_separation)
        self.require_start_in_endzone = bool(require_start_in_endzone)

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        w = max(1, int(self.smooth_win))
        if w <= 1:
            return y
        if w % 2 == 0:
            w += 1
        k = np.ones(w) / w
        ypad = np.pad(y, (w//2, w//2), mode="reflect")
        return np.convolve(ypad, k, mode="valid")

    def _compute_jump_mask(self) -> np.ndarray:
        x = self.x
        dx = np.diff(x, prepend=x[0])
        jump_mask = np.abs(dx) > self.jump_thresh
        # also treat non-finite x as jump/invalid
        jump_mask |= ~np.isfinite(x)
        return jump_mask

    def _velocity(self, valid_mask: np.ndarray) -> np.ndarray:
        x = self.x.copy()

        # For velocity sign we want to suppress huge jumps; easiest is to "freeze" x at jump points
        # so dx across the discontinuity is ~0 instead of gigantic.
        x_f = x.copy()
        jump = ~valid_mask
        x_f[jump] = np.nan

        # forward-fill NaNs (simple)
        for i in range(1, x_f.size):
            if np.isnan(x_f[i]):
                x_f[i] = x_f[i-1]
        # if starts NaN, back-fill
        if np.isnan(x_f[0]):
            first_ok = np.where(~np.isnan(x_f))[0]
            if first_ok.size > 0:
                x_f[:first_ok[0]] = x_f[first_ok[0]]
            else:
                x_f[:] = 0.0

        if self.t is None:
            v = np.diff(x_f, prepend=x_f[0])
        else:
            dt = np.diff(self.t, prepend=self.t[0])
            dt[dt == 0] = np.nan
            v = np.diff(x_f, prepend=x_f[0]) / dt
            v[np.isnan(v)] = 0.0

        return v

    def segment(self) -> LapSegmentation:
        x = self.x
        jump_mask = self._compute_jump_mask()
        valid_mask = ~jump_mask

        # End-zones
        left_zone = x <= (self.xmin + self.endzone_eps)
        right_zone = x >= (self.xmax - self.endzone_eps)

        # Direction sign from robust velocity
        v = self._velocity(valid_mask)
        v_s = self._smooth(v)
        lap_dir = np.sign(v_s)
        lap_dir[np.abs(v_s) < self.min_vel_abs] = 0
        lap_dir[~valid_mask] = 0  # don't trust sign at jumps

        # Turnarounds: sign changes in lap_dir ignoring zeros + invalids
        turn_mask = np.zeros(self.T, dtype=bool)
        last_dir = 0
        last_turn_i = -10**9
        for i in range(self.T):
            if not valid_mask[i]:
                continue
            d = int(lap_dir[i])
            if d == 0:
                continue
            if last_dir != 0 and d != last_dir:
                if i - last_turn_i >= self.min_turn_separation:
                    turn_mask[i] = True
                    last_turn_i = i
            last_dir = d

        # Lap state machine
        lap_idx = np.full(self.T, -1, dtype=int)
        lap_start_inds = []
        lap_end_inds = []

        # Find start index
        i0 = 0
        if self.require_start_in_endzone:
            while i0 < self.T and (not left_zone[i0]) and (not right_zone[i0]):
                i0 += 1
            if i0 >= self.T:
                return LapSegmentation(lap_idx, lap_dir, turn_mask, valid_mask, jump_mask,
                                       np.array([], int), np.array([], int), 0)

        # Determine initial target endzone
        if left_zone[i0] and (not right_zone[i0]):
            target = "right"
        elif right_zone[i0] and (not left_zone[i0]):
            target = "left"
        else:
            # if ambiguous, infer from direction if possible
            target = "right" if lap_dir[i0] >= 0 else "left"

        lap = 0
        lap_start_inds.append(i0)

        for i in range(i0, self.T):
            # we still assign lap_idx even if invalid, but you’ll later mask by seg.valid_mask
            lap_idx[i] = lap

            if target == "right":
                if right_zone[i]:
                    lap_end_inds.append(i)
                    lap += 1
                    if i + 1 < self.T:
                        lap_start_inds.append(i + 1)
                    target = "left"
            else:
                if left_zone[i]:
                    lap_end_inds.append(i)
                    lap += 1
                    if i + 1 < self.T:
                        lap_start_inds.append(i + 1)
                    target = "right"

        # Drop trailing incomplete lap
        if len(lap_end_inds) < len(lap_start_inds):
            incomplete_id = len(lap_end_inds)
            lap_idx[lap_idx == incomplete_id] = -1
            lap_start_inds = lap_start_inds[:len(lap_end_inds)]
            lap = incomplete_id

        lap_start_inds = np.array(lap_start_inds, dtype=int)
        lap_end_inds = np.array(lap_end_inds, dtype=int)

        return LapSegmentation(
            lap_idx=lap_idx,
            lap_dir=lap_dir,
            turn_mask=turn_mask,
            valid_mask=valid_mask,
            jump_mask=jump_mask,
            lap_start_inds=lap_start_inds,
            lap_end_inds=lap_end_inds,
            n_laps=int(lap),
        )
    @staticmethod
    def plot_laps(
        x: np.ndarray,
        seg: LapSegmentation,
        *,
        title: str = "1D laps (x vs lap index)",
        linewidth: float = 1.5,
        alpha: float = 0.9,
        show_turn_carets: bool = True,
        show_endzone_carets: bool = True,
        caret_stride: int = 1,
        turn_caret_size: float = 18,
        end_caret_size: float = 40,
    ):
        """
        Plot x-position vs lap index.

        Color code:
        - red   : rightward movement (lap_dir > 0)
        - black : leftward movement  (lap_dir < 0)
        """
        x = np.asarray(x, dtype=float)
        lap_idx = seg.lap_idx

        valid = (lap_idx >= 0)
        if not np.any(valid):
            raise ValueError("No valid laps to plot.")

        n_laps = seg.n_laps
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * max(1, n_laps))))

        for lap in range(n_laps):
            inds = np.where(lap_idx == lap)[0]
            if inds.size < 2:
                continue

            # Only keep valid samples for plotting
            inds = inds[seg.valid_mask[inds]]
            if inds.size < 2:
                continue

            # Split into contiguous direction segments
            dirs = seg.lap_dir[inds]
            split_points = np.where(np.diff(dirs) != 0)[0] + 1
            segments = np.split(inds, split_points)

            for seg_inds in segments:
                if seg_inds.size < 2:
                    continue
                d = seg.lap_dir[seg_inds[0]]
                if d == 0:
                    continue

                color = "red" if d > 0 else "black"
                ax.plot(
                    x[seg_inds],
                    np.full(seg_inds.shape, lap),
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )

            # ---- turn carets ----
            if show_turn_carets:
                turn_inds = inds[seg.turn_mask[inds]]
                if caret_stride > 1:
                    turn_inds = turn_inds[::caret_stride]

                left_inds = turn_inds[seg.lap_dir[turn_inds] < 0]
                right_inds = turn_inds[seg.lap_dir[turn_inds] > 0]

                if left_inds.size:
                    ax.scatter(
                        x[left_inds],
                        np.full(left_inds.shape, lap),
                        marker="<",
                        s=turn_caret_size,
                        color="black",
                        alpha=0.9,
                        linewidths=0,
                    )
                if right_inds.size:
                    ax.scatter(
                        x[right_inds],
                        np.full(right_inds.shape, lap),
                        marker=">",
                        s=turn_caret_size,
                        color="red",
                        alpha=0.9,
                        linewidths=0,
                    )

            # ---- lap end caret ----
            if show_endzone_carets and lap < seg.lap_end_inds.size:
                end_i = seg.lap_end_inds[lap]
                if seg.valid_mask[end_i]:
                    d = seg.lap_dir[end_i]
                    if d != 0:
                        color = "red" if d > 0 else "black"
                        marker = ">" if d > 0 else "<"
                        ax.scatter(
                            [x[end_i]],
                            [lap],
                            marker=marker,
                            s=end_caret_size,
                            color=color,
                            alpha=0.95,
                            linewidths=0,
                        )

        ax.set_xlabel("Track position (1D, px)")
        ax.set_ylabel("Lap #")
        ax.set_title(title)
        ax.set_yticks(np.arange(n_laps))
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return fig, ax

def attach_lap_segmentations(
    LT1_group: dict,
    LT2_group: dict,
    *,
    endzone_frac: float = 0.05,
    smooth_win: int = 9,
    min_turn_separation: int = 5,
    jump_thresh_frac_of_range: float = 0.25,
    require_start_in_endzone: bool = True,
    verbose: bool = True,
    plot_debug: bool = False,
    PLOTS_DIR: str = "",
    session_str: str = "",
    mapping: str = "",
    mouse_groups: dict = {},
    auto_close: bool = True,
):
    """
    For each mouse in LT1_group (and matching in LT2_group if available),
    compute LapSegmentation and assign to LT?.ls.

    Returns:
        dict with per-mouse summary (n_laps_LT1, n_laps_LT2, etc.)
    """
    summary = {}

    mice_all = sorted(set(LT1_group.keys()) | set(LT2_group.keys()))
    
    save_path = os.path.join(PLOTS_DIR, f"lap_segmentation_{session_str}_mapping_{mapping}") 
    os.makedirs(save_path, exist_ok=True)

    for mouse in mice_all:
        info = {}

        # ---- LT1 ----
        if mouse in LT1_group:
            LT1 = LT1_group[mouse]
            x1 = np.asarray(LT1.miniscope_loc_1d_px, dtype=float)
            t1 = getattr(LT1, "tstamp_miniscope", None)  # already subset + starts at 0
            laps1 = Laps1D(
                x1,
                t=t1,
                endzone_eps=None,
                endzone_frac=endzone_frac,
                smooth_win=smooth_win,
                min_turn_separation=min_turn_separation,
                require_start_in_endzone=require_start_in_endzone,
                jump_thresh=None,
                jump_thresh_frac_of_range=jump_thresh_frac_of_range,
            )
            seg1 = laps1.segment()
            LT1.ls = seg1
            info["LT1_n_laps"] = seg1.n_laps
            info["LT1_valid_frac"] = float(np.mean(seg1.valid_mask)) if seg1.valid_mask.size else np.nan
            info["LT1_jump_count"] = int(np.sum(seg1.jump_mask))

            if verbose:
                print(f"[{mouse}] LT1: n_laps={seg1.n_laps}  valid={info['LT1_valid_frac']:.3f}  jumps={info['LT1_jump_count']}")

            fig, ax = Laps1D.plot_laps(x1, seg1, title=f"{mouse} LT1 laps")
            plt.show()
            savefile = os.path.join(
                save_path,
                f"lap_segmentation_{mouse}_{mouse_groups[mouse]}_{session_str}_crossreg_{mapping}_laps_LT1.png",
            )
            fig.savefig(savefile, dpi=300)
            plt.show()
            if auto_close:
                plt.close(fig)   

        # ---- LT2 ----
        if mouse in LT2_group:
            LT2 = LT2_group[mouse]
            x2 = np.asarray(LT2.miniscope_loc_1d_px, dtype=float)
            t2 = getattr(LT2, "tstamp_miniscope", None)
            laps2 = Laps1D(
                x2,
                t=t2,
                endzone_eps=None,
                endzone_frac=endzone_frac,
                smooth_win=smooth_win,
                min_turn_separation=min_turn_separation,
                require_start_in_endzone=require_start_in_endzone,
                jump_thresh=None,
                jump_thresh_frac_of_range=jump_thresh_frac_of_range,
            )
            seg2 = laps2.segment()
            LT2.ls = seg2
            info["LT2_n_laps"] = seg2.n_laps
            info["LT2_valid_frac"] = float(np.mean(seg2.valid_mask)) if seg2.valid_mask.size else np.nan
            info["LT2_jump_count"] = int(np.sum(seg2.jump_mask))

            if verbose:
                print(f"[{mouse}] LT2: n_laps={seg2.n_laps}  valid={info['LT2_valid_frac']:.3f}  jumps={info['LT2_jump_count']}")

            fig, ax = Laps1D.plot_laps(x2, seg2, title=f"{mouse} LT2 laps")
            plt.show()
            savefile = os.path.join(
                save_path,
                f"lap_segmentation_{mouse}_{mouse_groups[mouse]}_{session_str}_crossreg_{mapping}_laps_LT2.png",
            )
            fig.savefig(savefile, dpi=300)
            plt.show()
            if auto_close:
                plt.close(fig)                

        summary[mouse] = info

    return summary

def _infer_dt_sec(LT) -> float:
    """
    Infer dt (seconds) from LT.tstamp_miniscope. In your code it is in ms.
    Falls back to 1/30 if something weird happens.
    """
    if hasattr(LT, "tstamp_miniscope") and LT.tstamp_miniscope is not None and len(LT.tstamp_miniscope) > 10:
        ts = np.asarray(LT.tstamp_miniscope, dtype=float)
        d = np.diff(ts)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            dt_ms = np.median(d)
            # Your pipeline uses ms timestamps
            return float(dt_ms) / 1000.0
    return 1.0 / 30.0

def _get_speed_mask(LT, min_speed: float = 2.0) -> np.ndarray:
    """
    Use miniscope velocity smoothing if present. If not, return all True.
    """
    T = LT.S.shape[1]
    if hasattr(LT, "velocities_miniscope_smooth") and LT.velocities_miniscope_smooth is not None:
        v = np.asarray(LT.velocities_miniscope_smooth, dtype=float)
        v = v[:T]
        return v >= min_speed
    return np.ones(T, dtype=bool)

def get_mapped_S_from_unit_ids(LT, unit_ids: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Convert unit_id list into indices into LT.S using LT.get_S_indeces(),
    return S_sub (n_cells, T) and the indices used.
    Filters out missing/NaN unit_ids defensively.
    """
    unit_ids = [int(u) for u in unit_ids if u is not None and np.isfinite(u)]
    if len(unit_ids) == 0:
        return np.zeros((0, LT.S.shape[1])), []

    inds = LT.get_S_indeces(unit_ids)  # indices into LT.S
    S_sub = LT.S[inds, :]
    return S_sub, inds

def get_mapped_S_for_single_session(LT, df_mapping, col_name):
    """
    LT: session object (LT1 or LT2)
    df_mapping: output of LT1.crossreg.get_mappings_cells(...)
    col_name: which mapping column to use (LT.session_group)

    Returns:
        S_sub (n_cells, T), inds (indices into LT.S), unit_ids (mapped unit ids)
    """
    unit_ids = df_mapping[col_name].astype(float).astype(int).tolist()
    S_sub, inds = get_mapped_S_from_unit_ids(LT, unit_ids)
    return S_sub, inds, unit_ids

class BayesianDecoder1D:
    """
    Poisson Bayesian decoding for 1D position.

    Trains tuning curves lambda_i(x) from spike/event counts (S) and occupancy.
    Decodes by MAP: argmax_x P(x | k) where k are binned event counts.
    """

    def __init__(
        self,
        n_pos_bins: int = 60,
        time_bin_frames: int = 5,
        rate_smooth_eps: float = 1e-6,
        use_occupancy_prior: bool = True,
    ):
        self.n_pos_bins = int(n_pos_bins)
        self.time_bin_frames = int(time_bin_frames)
        self.rate_smooth_eps = float(rate_smooth_eps)
        self.use_occupancy_prior = bool(use_occupancy_prior)

        # learned
        self.bin_edges = None           # (B+1,)
        self.bin_centers = None         # (B,)
        self.lam = None                 # (N,B) rates in events/sec
        self.log_prior = None           # (B,)
        self.dt_sec = None              # scalar
        self.fitted = False

    def _make_pos_bins(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        edges = np.linspace(xmin, xmax, self.n_pos_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    def fit(
        self,
        S: np.ndarray,          # (N,T)
        x: np.ndarray,          # (T,)
        mask: np.ndarray,       # (T,) boolean frames used for training
        dt_sec: float,
        bin_edges: Optional[np.ndarray] = None,
    ):
        S = np.asarray(S, dtype=float)
        x = np.asarray(x, dtype=float)
        mask = np.asarray(mask, dtype=bool)

        assert S.ndim == 2
        N, T = S.shape
        assert x.shape[0] >= T
        x = x[:T]
        mask = mask[:T]

        # Bin edges: either provided (for cross-session consistency) or inferred from training x
        if bin_edges is None:
            bin_edges, bin_centers = self._make_pos_bins(x[mask])
        else:
            bin_edges = np.asarray(bin_edges, dtype=float)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        B = bin_centers.size

        # Assign each frame to a position bin
        pos_bin = np.digitize(x, bin_edges) - 1
        pos_bin = np.clip(pos_bin, 0, B - 1)

        # Occupancy per bin (seconds)
        occ = np.zeros(B, dtype=float)
        for b in range(B):
            occ[b] = np.sum(mask & (pos_bin == b)) * dt_sec

        # Spike/event counts per neuron per bin
        counts = np.zeros((N, B), dtype=float)
        for b in range(B):
            idx = np.where(mask & (pos_bin == b))[0]
            if idx.size:
                counts[:, b] = np.sum(S[:, idx], axis=1)

        # Rate estimate (events/sec), with epsilon smoothing and occupancy floor
        occ_safe = np.maximum(occ, dt_sec)  # at least one frame worth
        lam = counts / occ_safe[None, :]
        lam = np.maximum(lam, self.rate_smooth_eps)

        # Prior
        if self.use_occupancy_prior:
            p = occ_safe / np.sum(occ_safe)
        else:
            p = np.ones(B, dtype=float) / B
        log_prior = np.log(np.maximum(p, 1e-12))

        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.lam = lam
        self.log_prior = log_prior
        self.dt_sec = float(dt_sec)
        self.fitted = True

    def decode(
        self,
        S: np.ndarray,          # (N,T)
        x: np.ndarray,          # (T,)
        mask: np.ndarray,       # (T,) boolean frames to decode/evaluate
    ) -> Dict[str, Any]:
        """
        Decode position using Poisson Bayes with non-overlapping chunks of length self.time_bin_frames.

        Returns dict with:
        decoded_pos, decoded_bin
        true_pos, true_bin
        K (n_chunks, N) event counts per chunk
        Ksum (n_chunks,) total event count per chunk
        idx_use (n_chunks, w) indices of frames per chunk (useful for debugging)
        """
        assert self.fitted, "Call fit() first."

        S = np.asarray(S, dtype=float)
        x = np.asarray(x, dtype=float)
        mask = np.asarray(mask, dtype=bool)

        N, T = S.shape
        x = x[:T]
        mask = mask[:T]

        B = self.bin_centers.size
        dt = self.dt_sec
        w = int(self.time_bin_frames)

        # Precompute terms
        log_lam = np.log(self.lam)              # (N,B)
        lam_dt = self.lam * dt * w              # (N,B) expected count over chunk duration

        # Build time bins from masked frames
        idx_all = np.where(mask)[0]
        if idx_all.size < w:
            return {
                "decoded_pos": np.array([]),
                "decoded_bin": np.array([], dtype=int),
                "true_pos": np.array([]),
                "true_bin": np.array([], dtype=int),
                "bin_centers": self.bin_centers,
                "K": np.zeros((0, N), dtype=float),
                "Ksum": np.array([]),
                "idx_use": np.zeros((0, w), dtype=int),
            }

        n_chunks = idx_all.size // w
        idx_use = idx_all[: n_chunks * w].reshape(n_chunks, w)

        # Counts per chunk: K (n_chunks, N)
        # S[:, idx_use] gives (N, n_chunks, w); sum over w -> (N, n_chunks); transpose -> (n_chunks, N)
        K = np.sum(S[:, idx_use], axis=2).T
        Ksum = K.sum(axis=1)

        # Log posterior per chunk:
        # log P(x|K) ∝ sum_i (K_i * log lam_i(x) - lam_i(x)*dt*w) + log_prior(x)
        log_post = (K @ log_lam) - np.sum(lam_dt[None, :, :], axis=1) + self.log_prior[None, :]

        decoded_bin = np.argmax(log_post, axis=1)
        decoded_pos = self.bin_centers[decoded_bin]

        # Ground truth per chunk: mean x in chunk
        true_pos = np.mean(x[idx_use], axis=1)
        true_bin = np.digitize(true_pos, self.bin_edges) - 1
        true_bin = np.clip(true_bin, 0, B - 1)

        return {
            "decoded_pos": decoded_pos,
            "decoded_bin": decoded_bin,
            "true_pos": true_pos,
            "true_bin": true_bin,
            "bin_centers": self.bin_centers,
            "K": K,
            "Ksum": Ksum,
            "idx_use": idx_use,
        }

    @staticmethod
    def abs_error(decoded_pos: np.ndarray, true_pos: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(decoded_pos) - np.asarray(true_pos))

def build_decode_mask(LT, *, use_speed: bool = True, min_speed: float = 2.0) -> np.ndarray:
    """
    Base mask for decoding/training: valid frames & (optionally) speed threshold.
    """
    T = LT.S.shape[1]
    m = np.ones(T, dtype=bool)

    if hasattr(LT, "ls") and LT.ls is not None:
        m &= LT.ls.valid_mask[:T]
        m &= (LT.ls.lap_idx[:T] >= 0)

    if use_speed:
        m &= _get_speed_mask(LT, min_speed=min_speed)

    return m

def even_lap_mask(LT) -> np.ndarray:
    T = LT.S.shape[1]
    return (LT.ls.lap_idx[:T] >= 0) & ((LT.ls.lap_idx[:T] % 2) == 0)

def odd_lap_mask(LT) -> np.ndarray:
    T = LT.S.shape[1]
    return (LT.ls.lap_idx[:T] >= 0) & ((LT.ls.lap_idx[:T] % 2) == 1)

def decode_within_LT1_even_odd(
    LT1,
    *,
    n_pos_bins: int = 60,
    time_bin_frames: int = 5,
    min_laps: int = 4,
    use_speed: bool = True,
    min_speed: float = 2.0,
) -> Dict[str, Any]:

    if LT1.ls.n_laps < min_laps:
        return {"ok": False, "reason": f"LT1 has only {LT1.ls.n_laps} laps (<{min_laps})"}

    dt = _infer_dt_sec(LT1)

    x = np.asarray(LT1.miniscope_loc_1d_px, dtype=float)[:LT1.S.shape[1]]
    S = LT1.S

    base = build_decode_mask(LT1, use_speed=use_speed, min_speed=min_speed)

    train_even = base & even_lap_mask(LT1)
    test_odd = base & odd_lap_mask(LT1)

    train_odd = base & odd_lap_mask(LT1)
    test_even = base & even_lap_mask(LT1)

    dec = BayesianDecoder1D(n_pos_bins=n_pos_bins, time_bin_frames=time_bin_frames)

    # even -> odd
    dec.fit(S, x, train_even, dt_sec=dt)
    out_eo = dec.decode(S, x, test_odd)
    err_eo = dec.abs_error(out_eo["decoded_pos"], out_eo["true_pos"])

    # odd -> even (reuse same bin_edges for perfect comparability)
    dec2 = BayesianDecoder1D(n_pos_bins=n_pos_bins, time_bin_frames=time_bin_frames)
    dec2.fit(S, x, train_odd, dt_sec=dt, bin_edges=dec.bin_edges)
    out_oe = dec2.decode(S, x, test_even)
    err_oe = dec2.abs_error(out_oe["decoded_pos"], out_oe["true_pos"])

    return {
        "ok": True,
        "dt": dt,
        "bin_edges": dec.bin_edges,
        "even_to_odd": {"out": out_eo, "abs_err": err_eo},
        "odd_to_even": {"out": out_oe, "abs_err": err_oe},
        "n_laps": LT1.ls.n_laps,
        "n_cells": S.shape[0],
    }

def decode_LT1_to_LT2_mapped(
    LT1,
    LT2,
    *,
    mapping: str = "full",
    n_pos_bins: int = 60,
    time_bin_frames: int = 5,
    min_laps_train: int = 2,
    min_laps_test: int = 2,
    use_speed: bool = True,
    min_speed: float = 2.0,
) -> Dict[str, Any]:

    if LT1.ls.n_laps < min_laps_train:
        return {"ok": False, "reason": f"LT1 has only {LT1.ls.n_laps} laps (<{min_laps_train})"}
    if LT2.ls.n_laps < min_laps_test:
        return {"ok": False, "reason": f"LT2 has only {LT2.ls.n_laps} laps (<{min_laps_test})"}

    # Get cross-registered unit-id mapping between the two sessions
    df_mapping = LT1.crossreg.get_mappings_cells(mapping_type=mapping)
    cells_LT1 = df_mapping[LT1.session_group].astype(float).astype(int).tolist()
    cells_LT2 = df_mapping[LT2.session_group].astype(float).astype(int).tolist()

    # Subset S to mapped cells
    S1, inds1 = get_mapped_S_from_unit_ids(LT1, cells_LT1)
    S2, inds2 = get_mapped_S_from_unit_ids(LT2, cells_LT2)

    if S1.shape[0] == 0 or S2.shape[0] == 0:
        return {"ok": False, "reason": "No mapped cells found."}

    if S1.shape[0] != S2.shape[0]:
        n = min(S1.shape[0], S2.shape[0])
        S1 = S1[:n]
        S2 = S2[:n]

    # dt per miniscope frame (seconds)
    dt1 = _infer_dt_sec(LT1)
    dt2 = _infer_dt_sec(LT2)

    # Trim x to match S matrices
    x1 = np.asarray(LT1.miniscope_loc_1d_px, dtype=float)[: S1.shape[1]]
    x2 = np.asarray(LT2.miniscope_loc_1d_px, dtype=float)[: S2.shape[1]]

    # Base decode masks
    base1 = build_decode_mask(LT1, use_speed=use_speed, min_speed=min_speed)[: S1.shape[1]]
    base2 = build_decode_mask(LT2, use_speed=use_speed, min_speed=min_speed)[: S2.shape[1]]

    # --- Train decoder on LT1 (learns LT1 bin_edges) ---
    dec = BayesianDecoder1D(n_pos_bins=n_pos_bins, time_bin_frames=time_bin_frames)

    # IMPORTANT: your fit signature is (S, x, mask, dt_sec, bin_edges=None)
    dec.fit(S1, x1, base1, dt1)

    # (Optional but explicit) ensure transfer uses LT1 binning
    bin_edges_train = np.asarray(dec.bin_edges, float)

    # --- Decode LT2 using LT1-tuned decoder (same bin edges automatically, because decoder stores them) ---
    out = dec.decode(S2, x2, base2)
    err = BayesianDecoder1D.abs_error(out["decoded_pos"], out["true_pos"])

    return {
        "ok": True,
        "bin_edges": bin_edges_train,
        "dt_train": float(dt1),
        "dt_test": float(dt2),
        "out": out,
        "abs_err": err,
        "n_cells_mapped": int(S1.shape[0]),
        "LT1_n_laps": int(LT1.ls.n_laps),
        "LT2_n_laps": int(LT2.ls.n_laps),
    }


def run_decoding_all_mice_pipeline(
    LT1_group: dict,
    LT2_group: dict,
    *,
    mapping: str = "full",
    min_laps_within: int = 4,          # kept but not used here
    min_laps_within_per_dir: int = 2,
    min_laps_transfer: int = 2,
    n_pos_bins: int = 60,
    time_bin_frames: int = 5,
    use_speed: bool = True,
    min_speed: float = 2.0,
    PLOTS_DIR: str = "",
    session_str: str = "",
    mouse_groups: dict = None,
    plot_debug: bool = True,
    auto_close: bool = True,
    rng_seed: int = 0,
):
    """
    Runs:
      1) within-session decoding (FULL cells) for LT1 and LT2
      2) within-session decoding (MAPPED subset) for LT1 and LT2
      3) cross-session LT1 -> LT2 decoding (MAPPED subset)

    Notes:
      - "FULL" means using LT.S directly (all cells in that session).
      - "MAPPED" means using the same crossreg mapping used for LT1->LT2 transfer.

    Requires:
      decode_within_LT_by_direction(..., S_override=...) support.
    """
    if mouse_groups is None:
        mouse_groups = {}

    save_path = os.path.join(PLOTS_DIR, f"bayes_decoding_{session_str}_mapping_{mapping}")
    os.makedirs(save_path, exist_ok=True)

    results = {}
    mice = sorted(set(LT1_group.keys()) & set(LT2_group.keys()))

    def _plot_block(tag, out, abs_err, title_prefix):
        # error histogram
        fig1, ax1 = plot_decoding_error_hist(
            abs_err,
            title=f"{title_prefix} | bins={n_pos_bins} tb={time_bin_frames}",
        )
        fig1.savefig(os.path.join(save_path, f"{tag}_errhist.png"), dpi=300)
        if plot_debug:
            plt.show()
        if auto_close:
            plt.close(fig1)

        # decoded vs true
        fig2, ax2 = plot_decoded_vs_true(
            out["true_pos"],
            out["decoded_pos"],
            title=f"{title_prefix} decoded vs true",
            subsample=1,
        )
        fig2.savefig(os.path.join(save_path, f"{tag}_scatter.png"), dpi=300)
        if plot_debug:
            plt.show()
        if auto_close:
            plt.close(fig2)

        # error vs Ksum + Ksum hist
        if "Ksum" in out:
            figK, axK = plot_error_vs_Ksum(abs_err, out["Ksum"], title=f"{title_prefix}: error vs Ksum")
            figK.savefig(os.path.join(save_path, f"{tag}_err_vs_Ksum.png"), dpi=300)
            if plot_debug:
                plt.show()
            if auto_close:
                plt.close(figK)

            plot_Ksum_histogram(
                out["Ksum"],
                title=f"{title_prefix} Ksum",
                savefile=os.path.join(save_path, f"{tag}_Ksum_hist.png"),
            )

    def _plot_within_by_dir(mouse, group, which, r_within):
        """
        which: string label e.g. "LT1_full", "LT1_mapped", "LT2_full", "LT2_mapped"
        r_within: output dict from decode_within_LT_by_direction
        """
        if not r_within.get("ok", False):
            return

        for dir_label in ("rightward", "leftward"):
            rdir = r_within.get(dir_label, {})
            if not rdir.get("ok", False):
                continue

            for split_key, split_pretty in [("train_to_test", "train→test"), ("test_to_train", "test→train")]:
                sub = rdir[split_key]
                out = sub["out"]
                abs_err = sub["abs_err"]

                tag = f"decoder_{mouse}_{group}_{session_str}_crossreg_{mapping}_{which}_{dir_label}_{split_key}"
                title_prefix = f"{mouse} {group} {session_str} {which} {dir_label} {split_pretty}"
                _plot_block(tag, out, abs_err, title_prefix)

    for mouse in mice:
        LT1 = LT1_group[mouse]
        LT2 = LT2_group[mouse]
        group = mouse_groups.get(mouse, "NA")

        # ------------------------------------------------------------------
        # Build mapped S matrices (same mapping used for LT1 -> LT2 transfer)
        # ------------------------------------------------------------------
        df_mapping = LT1.crossreg.get_mappings_cells(mapping_type=mapping)

        # Column names are LT1.session_group and LT2.session_group (as used in decode_LT1_to_LT2_mapped)
        cells_LT1 = df_mapping[LT1.session_group].astype(float).astype(int).tolist()
        cells_LT2 = df_mapping[LT2.session_group].astype(float).astype(int).tolist()

        S1_mapped, inds1 = get_mapped_S_from_unit_ids(LT1, cells_LT1)
        S2_mapped, inds2 = get_mapped_S_from_unit_ids(LT2, cells_LT2)

        # If mapping is empty, we still want full within-session decodes to run.
        have_mapped = (S1_mapped.shape[0] > 0) and (S2_mapped.shape[0] > 0)

        # ------------------------------------------------------------------
        # Task 1: within LT1 (FULL)
        # ------------------------------------------------------------------
        r_within1_full = decode_within_LT_by_direction(
            LT1,
            label="LT1_full",
            n_pos_bins=n_pos_bins,
            time_bin_frames=time_bin_frames,
            min_laps_per_dir=min_laps_within_per_dir,
            use_speed=use_speed,
            min_speed=min_speed,
            rng_seed=rng_seed,
            S_override=None,
        )
        _plot_within_by_dir(mouse, group, "LT1_full", r_within1_full)

        # ------------------------------------------------------------------
        # Task 1b: within LT2 (FULL)
        # ------------------------------------------------------------------
        r_within2_full = decode_within_LT_by_direction(
            LT2,
            label="LT2_full",
            n_pos_bins=n_pos_bins,
            time_bin_frames=time_bin_frames,
            min_laps_per_dir=min_laps_within_per_dir,
            use_speed=use_speed,
            min_speed=min_speed,
            rng_seed=rng_seed,
            S_override=None,
        )
        _plot_within_by_dir(mouse, group, "LT2_full", r_within2_full)

        # ------------------------------------------------------------------
        # Task 1c: within LT1 (MAPPED subset)
        # ------------------------------------------------------------------
        if have_mapped:
            r_within1_mapped = decode_within_LT_by_direction(
                LT1,
                label="LT1_mapped",
                n_pos_bins=n_pos_bins,
                time_bin_frames=time_bin_frames,
                min_laps_per_dir=min_laps_within_per_dir,
                use_speed=use_speed,
                min_speed=min_speed,
                rng_seed=rng_seed,
                S_override=S1_mapped,
            )
            _plot_within_by_dir(mouse, group, "LT1_mapped", r_within1_mapped)
        else:
            r_within1_mapped = {"ok": False, "reason": "No mapped cells found."}

        # ------------------------------------------------------------------
        # Task 1d: within LT2 (MAPPED subset)
        # ------------------------------------------------------------------
        if have_mapped:
            r_within2_mapped = decode_within_LT_by_direction(
                LT2,
                label="LT2_mapped",
                n_pos_bins=n_pos_bins,
                time_bin_frames=time_bin_frames,
                min_laps_per_dir=min_laps_within_per_dir,
                use_speed=use_speed,
                min_speed=min_speed,
                rng_seed=rng_seed,
                S_override=S2_mapped,
            )
            _plot_within_by_dir(mouse, group, "LT2_mapped", r_within2_mapped)
        else:
            r_within2_mapped = {"ok": False, "reason": "No mapped cells found."}

        # ------------------------------------------------------------------
        # Task 2: LT1 -> LT2 transfer (MAPPED subset)
        # ------------------------------------------------------------------
        r_xfer = decode_LT1_to_LT2_mapped(
            LT1,
            LT2,
            mapping=mapping,
            n_pos_bins=n_pos_bins,
            time_bin_frames=time_bin_frames,
            min_laps_train=min_laps_transfer,
            min_laps_test=min_laps_transfer,
            use_speed=use_speed,
            min_speed=min_speed,
        )

        if r_xfer.get("ok", False):
            tag = f"decoder_{mouse}_{group}_{session_str}_crossreg_{mapping}_LT1_to_LT2"
            title_prefix = f"{mouse} {group} {session_str} LT1→LT2"
            _plot_block(tag, r_xfer["out"], r_xfer["abs_err"], title_prefix)

        # ------------------------------------------------------------------
        # Print status
        # ------------------------------------------------------------------
        msg = f"[{mouse}] "
        msg += "within_LT1_full=" + ("OK" if r_within1_full.get("ok", False) else f"SKIP({r_within1_full.get('reason','')})")
        msg += " ; within_LT2_full=" + ("OK" if r_within2_full.get("ok", False) else f"SKIP({r_within2_full.get('reason','')})")
        msg += " ; within_LT1_mapped=" + ("OK" if r_within1_mapped.get("ok", False) else f"SKIP({r_within1_mapped.get('reason','')})")
        msg += " ; within_LT2_mapped=" + ("OK" if r_within2_mapped.get("ok", False) else f"SKIP({r_within2_mapped.get('reason','')})")
        msg += " ; LT1→LT2=" + ("OK" if r_xfer.get("ok", False) else f"SKIP({r_xfer.get('reason','')})")
        print(msg)

        results[mouse] = {
            "within_LT1_by_dir_full": r_within1_full,
            "within_LT2_by_dir_full": r_within2_full,
            "within_LT1_by_dir_mapped": r_within1_mapped,
            "within_LT2_by_dir_mapped": r_within2_mapped,
            "LT1_to_LT2": r_xfer,
            "n_cells_full_LT1": int(getattr(LT1.S, "shape", [0])[0]) if hasattr(LT1, "S") else None,
            "n_cells_full_LT2": int(getattr(LT2.S, "shape", [0])[0]) if hasattr(LT2, "S") else None,
            "n_cells_mapped": int(S1_mapped.shape[0]) if have_mapped else 0,
        }

    return results

def collect_group_Ksum_v2(dec_results, mouse_groups, condition, *, within_mode="full"):
    """
    condition:
      - "LT1_train_to_test", "LT1_test_to_train",
      - "LT2_train_to_test", "LT2_test_to_train",
      - "LT1_to_LT2"

    within_mode:
      - "full"  -> use within_*_by_dir_full
      - "mapped"-> use within_*_by_dir_mapped
      - "proxy" -> use within_*_by_dir (your mapped-within proxy dict)
    """
    assert within_mode in ("full", "mapped", "proxy")

    group_Ksum = {g: [] for g in set(mouse_groups.values())}
    group_Ksum.setdefault("NA", [])  # safe bucket if group missing

    def add(mouse, grp, k):
        k = np.asarray(k).ravel()
        if k.size:
            group_Ksum[grp].append(k)

    for mouse, res in dec_results.items():
        grp = mouse_groups.get(mouse, "NA")

        if condition == "LT1_to_LT2":
            blk = res.get("LT1_to_LT2", None)
            if isinstance(blk, dict) and "Ksum" in blk:
                add(mouse, grp, blk["Ksum"])
            continue

        # within-session
        if condition.startswith("LT1_"):
            sess = "LT1"
        elif condition.startswith("LT2_"):
            sess = "LT2"
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if within_mode == "proxy":
            within_key = f"within_{sess}_by_dir"
        else:
            within_key = f"within_{sess}_by_dir_{within_mode}"

        within_blk = res.get(within_key, None)
        if not isinstance(within_blk, dict):
            continue

        blk = within_blk.get(condition, None)
        if isinstance(blk, dict) and "Ksum" in blk:
            add(mouse, grp, blk["Ksum"])

    # concat
    out = {}
    for grp, parts in group_Ksum.items():
        out[grp] = np.concatenate(parts) if len(parts) else np.array([])
    return out

def plot_decoding_error_hist(err_px, title, bins=50):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(err_px, bins=bins)
    ax.set_xlabel("Absolute error (px)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax

def plot_decoded_vs_true(true_pos, decoded_pos, title, subsample=1):
    true_pos = np.asarray(true_pos)
    decoded_pos = np.asarray(decoded_pos)
    if subsample > 1 and true_pos.size > subsample:
        idx = np.arange(0, true_pos.size, subsample)
        true_pos = true_pos[idx]
        decoded_pos = decoded_pos[idx]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(true_pos, decoded_pos, s=6, alpha=0.4)
    lo = np.nanmin([true_pos.min(), decoded_pos.min()])
    hi = np.nanmax([true_pos.max(), decoded_pos.max()])
    ax.plot([lo, hi], [lo, hi], linewidth=1.5)
    ax.set_xlabel("True position (px)")
    ax.set_ylabel("Decoded position (px)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax

def plot_error_vs_Ksum(abs_err, Ksum, title):
    abs_err = np.asarray(abs_err, dtype=float)
    Ksum = np.asarray(Ksum, dtype=float)
    n = min(abs_err.size, Ksum.size)
    abs_err = abs_err[:n]
    Ksum = Ksum[:n]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(Ksum, abs_err, s=10, alpha=0.35)
    ax.set_xlabel("Total events per bin (Ksum)")
    ax.set_ylabel("Absolute error (px)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax

def plot_Ksum_histogram(
    Ksum,
    title,
    savefile,
    bins=40,
    xlim=None,
    auto_close=True,
):
    Ksum = np.asarray(Ksum, dtype=float)
    Ksum = Ksum[np.isfinite(Ksum)]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.hist(Ksum, bins=bins, color="black", alpha=0.8)
    ax.set_xlabel("Total events per bin (Ksum)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(savefile, dpi=300)

    if auto_close:
        plt.close(fig)

    return fig, ax

def collect_group_Ksum(dec_results, mouse_groups, condition):
    """
    Collect Ksum arrays across mice, grouped by manipulation.

    Supports CURRENT dec_results structure:
      - Within-session:
          res["within_LT1_by_dir_full"][dir]["train_to_test"]["out"]["Ksum"]
          res["within_LT2_by_dir_full"][dir]["test_to_train"]["out"]["Ksum"]
        (and _mapped variants, plus optional proxy keys)

      - Across-session:
          res["LT1_to_LT2"]["out"]["Ksum"]

    condition must be one of:
      "LT1_train_to_test", "LT1_test_to_train",
      "LT2_train_to_test", "LT2_test_to_train",
      "LT1_to_LT2"
    """
    group_Ksum_parts = {g: [] for g in set(mouse_groups.values())}
    group_Ksum_parts.setdefault("NA", [])

    mouse_Ksum = {}   # optional: mouse -> concatenated Ksum
    n_mice_used = {g: 0 for g in group_Ksum_parts.keys()}

    def _append(grp, mouse, arr):
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return
        group_Ksum_parts[grp].append(arr)
        mouse_Ksum[mouse] = arr
        n_mice_used[grp] = n_mice_used.get(grp, 0) + 1

    def _get_within_block(res, sess):
        """
        Choose which within dict to use.
        Priority:
          1) proxy within_{sess}_by_dir (if you created proxy dict)
          2) within_{sess}_by_dir_full
          3) within_{sess}_by_dir_mapped
        """
        for kk in (f"within_{sess}_by_dir", f"within_{sess}_by_dir_full", f"within_{sess}_by_dir_mapped"):
            d = res.get(kk, None)
            if isinstance(d, dict) and len(d) > 0:
                return d
        return None

    for mouse, res in dec_results.items():
        grp = mouse_groups.get(mouse, "NA")

        # ---- across-session ----
        if condition == "LT1_to_LT2":
            blk = res.get("LT1_to_LT2", None)
            if isinstance(blk, dict):
                out = blk.get("out", None)
                if isinstance(out, dict) and ("Ksum" in out):
                    _append(grp, mouse, out["Ksum"])
            continue

        # ---- within-session ----
        if condition.startswith("LT1_"):
            sess = "LT1"
        elif condition.startswith("LT2_"):
            sess = "LT2"
        else:
            continue

        within = _get_within_block(res, sess)
        if not isinstance(within, dict):
            continue

        # Map condition -> inner key
        if condition.endswith("train_to_test"):
            inner = "train_to_test"
        elif condition.endswith("test_to_train"):
            inner = "test_to_train"
        else:
            continue

        # Within is nested by direction: leftward/rightward (sometimes only one exists)
        for direction, dblk in within.items():
            if not isinstance(dblk, dict):
                continue
            step = dblk.get(inner, None)
            if not isinstance(step, dict):
                continue
            out = step.get("out", None)
            if not isinstance(out, dict):
                continue
            if "Ksum" not in out:
                continue
            _append(grp, mouse, out["Ksum"])

    # Concatenate per-group
    group_Ksum = {}
    for grp, parts in group_Ksum_parts.items():
        group_Ksum[grp] = np.concatenate(parts) if len(parts) else np.array([])

    return group_Ksum, mouse_Ksum, n_mice_used

def _get_out_block(res_mouse: dict, condition: str):
    """
    Return (ok, out_dict, abs_err_vec, reason)

    Supports:
      - within: res["within_LT{1,2}_by_dir(_full/_mapped)"][direction][train_to_test/test_to_train]
               where each step contains:
                 step["out"]["Ksum"]
                 step["abs_err"]   (preferred)  OR step["out"]["abs_err"] (fallback)

      - across: res["LT1_to_LT2"]["out"]["Ksum"] and res["LT1_to_LT2"]["abs_err"]
    """

    # -----------------------
    # Helper: choose within dict
    # -----------------------
    def _get_within_dict(sess: str):
        for kk in (f"within_{sess}_by_dir",
                   f"within_{sess}_by_dir_full",
                   f"within_{sess}_by_dir_mapped"):
            d = res_mouse.get(kk, None)
            if isinstance(d, dict) and len(d) > 0:
                return d, kk
        return None, None

    # -----------------------
    # Across-session
    # -----------------------
    if condition == "LT1_to_LT2":
        blk = res_mouse.get("LT1_to_LT2", None)
        if not isinstance(blk, dict):
            return False, None, None, "missing LT1_to_LT2 dict"

        out = blk.get("out", None)
        if not isinstance(out, dict):
            return False, None, None, "missing LT1_to_LT2.out"

        abs_err = blk.get("abs_err", None)
        if abs_err is None:
            # very defensive fallback
            abs_err = out.get("abs_err", None)

        if abs_err is None:
            return False, None, None, "missing LT1_to_LT2 abs_err"

        return True, out, abs_err, "ok"

    # -----------------------
    # Within-session: parse condition
    # -----------------------
    if condition.startswith("LT1_"):
        sess = "LT1"
    elif condition.startswith("LT2_"):
        sess = "LT2"
    else:
        return False, None, None, f"unknown condition format: {condition}"

    if condition.endswith("train_to_test"):
        inner = "train_to_test"
    elif condition.endswith("test_to_train"):
        inner = "test_to_train"
    else:
        return False, None, None, f"unknown within direction: {condition}"

    within, which_key = _get_within_dict(sess)
    if within is None:
        return False, None, None, f"missing within dict for {sess} (tried by_dir/_full/_mapped)"

    # -----------------------
    # Collect across directions (leftward/rightward/etc.)
    # -----------------------
    Ksum_parts = []
    err_parts = []

    for direction, dblk in within.items():
        if not isinstance(dblk, dict):
            continue

        step = dblk.get(inner, None)
        if not isinstance(step, dict):
            continue

        out = step.get("out", None)
        if not isinstance(out, dict):
            continue

        # Ksum
        Ksum = out.get("Ksum", None)
        if Ksum is None:
            continue

        # abs_err: preferred at step-level, fallback inside out
        abs_err = step.get("abs_err", None)
        if abs_err is None:
            abs_err = out.get("abs_err", None)
        if abs_err is None:
            # can't compute error summary without errors
            continue

        Ksum = np.asarray(Ksum).ravel()
        abs_err = np.asarray(abs_err).ravel()

        if Ksum.size == 0 or abs_err.size == 0:
            continue
        if Ksum.shape[0] != abs_err.shape[0]:
            # skip this direction if mismatch
            continue

        Ksum_parts.append(Ksum)
        err_parts.append(abs_err)

    if len(Ksum_parts) == 0:
        return False, None, None, f"no usable within blocks found for {condition} in {which_key}"

    Ksum_all = np.concatenate(Ksum_parts)
    err_all = np.concatenate(err_parts)

    # Return a minimal out dict that satisfies build_mouse_level_summary_from_dec_results()
    out_combined = {"Ksum": Ksum_all}

    return True, out_combined, err_all, "ok"

def filter_err_by_Ksum(abs_err, Ksum, kmin):
    abs_err = np.asarray(abs_err, float)
    Ksum = np.asarray(Ksum, float)
    n = min(abs_err.size, Ksum.size)
    abs_err = abs_err[:n]
    Ksum = Ksum[:n]
    keep = np.isfinite(abs_err) & np.isfinite(Ksum) & (Ksum >= kmin)
    return abs_err[keep], keep

def plot_group_filtered_error_hist(
    dec_results, mouse_groups, condition,
    kmin=5,
    bins=60,
    title="",
    savefile="",
    density=False,
    auto_close=True
):
    colors = {"hM3D":"red", "hM4D":"blue", "mCherry":"black"}
    group_err = {"hM3D":[], "hM4D":[], "mCherry":[]}

    for mouse, res_mouse in dec_results.items():
        group = mouse_groups.get(mouse, "NA")
        if group not in group_err:
            continue
        ok, out, abs_err, _ = _get_out_block(res_mouse, condition)
        if not ok:
            continue
        Ksum = np.asarray(out.get("Ksum", []), float)
        err_f, _ = filter_err_by_Ksum(abs_err, Ksum, kmin=kmin)
        if err_f.size:
            group_err[group].append(err_f)

    for g in group_err:
        group_err[g] = np.concatenate(group_err[g]) if len(group_err[g]) else np.array([])

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    for g in ["hM3D","hM4D","mCherry"]:
        if group_err[g].size == 0:
            continue
        ax.hist(group_err[g], bins=bins, alpha=0.45, color=colors[g],
                label=f"{g} (n={group_err[g].size})", density=density)

    ax.set_xlabel(f"Absolute error (px)  |  Ksum ≥ {kmin}")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    if savefile:
        fig.savefig(savefile, dpi=300)
    if auto_close:
        plt.close(fig)
    return fig, ax

def plot_error_vs_Kthreshold_by_group(
    dec_results, mouse_groups, condition,
    kmins=None,
    stat="median",          # "median" or "mean"
    title="",
    savefile="",
    auto_close=True
):
    if kmins is None:
        kmins = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120])

    colors = {"hM3D":"red", "hM4D":"blue", "mCherry":"black"}
    groups = ["hM3D","hM4D","mCherry"]

    # pool across mice within each group
    pooled = {g: [] for g in groups}
    pooled_K = {g: [] for g in groups}

    for mouse, res_mouse in dec_results.items():
        g = mouse_groups.get(mouse, "NA")
        if g not in pooled:
            continue
        ok, out, abs_err, _ = _get_out_block(res_mouse, condition)
        if not ok:
            continue
        Ksum = np.asarray(out.get("Ksum", []), float)
        n = min(abs_err.size, Ksum.size)
        pooled[g].append(abs_err[:n])
        pooled_K[g].append(Ksum[:n])

    for g in groups:
        pooled[g] = np.concatenate(pooled[g]) if len(pooled[g]) else np.array([])
        pooled_K[g] = np.concatenate(pooled_K[g]) if len(pooled_K[g]) else np.array([])

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)

    for g in groups:
        if pooled[g].size == 0:
            continue
        y = []
        for kmin in kmins:
            err_f, _ = filter_err_by_Ksum(pooled[g], pooled_K[g], kmin=kmin)
            if err_f.size == 0:
                y.append(np.nan)
            else:
                y.append(np.nanmedian(err_f) if stat=="median" else np.nanmean(err_f))
        ax.plot(kmins, y, marker="o", linewidth=2, label=g, color=colors[g])

    ax.set_xlabel("Ksum threshold (kmin)")
    ax.set_ylabel(f"{stat} absolute error (px)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    if savefile:
        fig.savefig(savefile, dpi=300)
    if auto_close:
        plt.close(fig)
    return fig, ax

def get_lap_id_array(LT):
    """
    Returns lap_id per miniscope frame, length T.
    Adjust here if your LapSegmentation uses different attribute names.
    """
    seg = getattr(LT, "ls", None)
    if seg is None:
        return None

    # common names
    for name in ["lap_id", "lap_ids", "lap_index", "lap_idx", "lap_per_frame"]:
        if hasattr(seg, name):
            arr = np.asarray(getattr(seg, name))
            return arr

    # if your segmentation stores laps differently, fix here.
    return None

def chunk_lap_ids_from_idx_use(lap_id_per_frame, idx_use):
    """
    idx_use: (n_chunks, w) frame indices
    Returns: lap_id_per_chunk: (n_chunks,)
    """
    lap_id_per_frame = np.asarray(lap_id_per_frame)
    idx_use = np.asarray(idx_use, int)

    lap_ids = []
    for rows in idx_use:
        vals = lap_id_per_frame[rows]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            lap_ids.append(-1)
        else:
            # majority vote
            u, c = np.unique(vals.astype(int), return_counts=True)
            lap_ids.append(int(u[np.argmax(c)]))
    return np.asarray(lap_ids, int)

def per_lap_mean_Ksum(lap_id_per_chunk, Ksum):
    lap_id_per_chunk = np.asarray(lap_id_per_chunk, int)
    Ksum = np.asarray(Ksum, float)
    out = {}
    for lap in np.unique(lap_id_per_chunk):
        if lap < 0:
            continue
        m = (lap_id_per_chunk == lap) & np.isfinite(Ksum)
        if np.any(m):
            out[lap] = float(np.mean(Ksum[m]))
    return out

def plot_mouse_per_lap_Ksum_drift(
    mouse, LT1, LT2, res_mouse, mouse_groups,
    condition_within="LT1_train_to_test",
    title="",
    savefile="",
    auto_close=True
):
    group = mouse_groups.get(mouse, "NA")

    # --- LT1 within: pool both directions if available ---
    within = res_mouse.get("within_LT1", {})
    if not within.get("ok", False):
        return None

    outs = []
    for key in ["even_to_odd", "odd_to_even"]:
        blk = within.get(key, None)
        if blk is not None and "out" in blk and "Ksum" in blk["out"]:
            outs.append(blk["out"])

    if len(outs) == 0:
        return None

    # concatenate chunks for LT1-within
    Ksum1 = np.concatenate([np.asarray(o["Ksum"], float) for o in outs])
    idx1  = np.vstack([np.asarray(o["idx_use"], int) for o in outs])

    # --- LT1→LT2 ---
    xfer = res_mouse.get("LT1_to_LT2", {})
    if not xfer.get("ok", False):
        return None
    out2 = xfer["out"]
    Ksum2 = np.asarray(out2["Ksum"], float)
    idx2  = np.asarray(out2["idx_use"], int)

    # lap ids per frame
    lap1 = get_lap_id_array(LT1)
    lap2 = get_lap_id_array(LT2)
    if lap1 is None or lap2 is None:
        return None

    # chunk→lap
    lap_id_chunk_1 = chunk_lap_ids_from_idx_use(lap1, idx1)
    lap_id_chunk_2 = chunk_lap_ids_from_idx_use(lap2, idx2)

    d1 = per_lap_mean_Ksum(lap_id_chunk_1, Ksum1)
    d2 = per_lap_mean_Ksum(lap_id_chunk_2, Ksum2)

    # plot
    fig, ax = plt.subplots(figsize=(7,5), dpi=150)

    if d1:
        laps = np.array(sorted(d1.keys()))
        ax.plot(laps, [d1[l] for l in laps], marker="o", linewidth=2, color="black", label="LT1 within")
    if d2:
        laps = np.array(sorted(d2.keys()))
        ax.plot(laps, [d2[l] for l in laps], marker="o", linewidth=2, color="red", label="LT2 (decoded LT1→LT2)")

    ax.set_xlabel("Lap #")
    ax.set_ylabel("Mean Ksum per decoded chunk")
    ax.set_title(title or f"{mouse} {group} per-lap Ksum: LT1 vs LT2")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    if savefile:
        fig.savefig(savefile, dpi=300)
    if auto_close:
        plt.close(fig)
    return fig, ax

def run_per_lap_Ksum_drift_plots(
    LT1_group, LT2_group,
    dec_results, mouse_groups,
    PLOTS_DIR, session_str, mapping,
    auto_close=True
):
    save_path = os.path.join(PLOTS_DIR, f"Ksum_per_lap_drift_{session_str}_mapping_{mapping}")
    os.makedirs(save_path, exist_ok=True)

    for mouse, res_mouse in dec_results.items():
        if mouse not in LT1_group or mouse not in LT2_group:
            continue
        LT1 = LT1_group[mouse]
        LT2 = LT2_group[mouse]
        group = mouse_groups.get(mouse, "NA")

        figax = plot_mouse_per_lap_Ksum_drift(
            mouse, LT1, LT2, res_mouse, mouse_groups,
            title=f"{mouse} {group} {session_str}: per-lap Ksum drift",
            savefile=os.path.join(save_path, f"Ksum_per_lap_{mouse}_{group}_{session_str}_mapping_{mapping}.png"),
            auto_close=auto_close
        )

def summarize_mouse_condition(abs_err, Ksum, kmin_for_err=5, lowK_thresh=1):
    abs_err = np.asarray(abs_err, float)
    Ksum = np.asarray(Ksum, float)
    n = min(abs_err.size, Ksum.size)
    abs_err = abs_err[:n]
    Ksum = Ksum[:n]

    m = np.isfinite(Ksum)
    Ksum_v = Ksum[m]

    # filtered error
    err_f, keep = filter_err_by_Ksum(abs_err, Ksum, kmin=kmin_for_err)

    out = {
        "median_Ksum": float(np.nanmedian(Ksum_v)) if Ksum_v.size else np.nan,
        "mean_Ksum": float(np.nanmean(Ksum_v)) if Ksum_v.size else np.nan,
        "frac_lowK": float(np.mean(Ksum_v < lowK_thresh)) if Ksum_v.size else np.nan,
        "median_err_filt": float(np.nanmedian(err_f)) if err_f.size else np.nan,
        "mean_err_filt": float(np.nanmean(err_f)) if err_f.size else np.nan,
        "n_bins": int(Ksum_v.size),
        "n_bins_err_filt": int(err_f.size),
    }
    return out

def build_mixedlm_dataframe(dec_results, mouse_groups, kmin_for_err=5, lowK_thresh=1):
    rows = []

    def _add_mouse_condition(mouse, group, condition, abs_err, Ksum):
        abs_err = np.asarray(abs_err, float)
        Ksum = np.asarray(Ksum, float)
        n = min(abs_err.size, Ksum.size)
        abs_err = abs_err[:n]
        Ksum = Ksum[:n]
        s = summarize_mouse_condition(abs_err, Ksum, kmin_for_err, lowK_thresh)
        rows.append({"mouse": mouse, "group": group, "condition": condition, **s})

    for mouse, res_mouse in dec_results.items():
        group = mouse_groups.get(mouse, "NA")

        # ---- within LT1 (pooled dirs) ----
        ok, out, abs_err, _ = _get_out_block(res_mouse, "LT1_train_to_test")
        if ok:
            Ksum = np.asarray(out.get("Ksum", []), float)
            _add_mouse_condition(mouse, group, "within_LT1", abs_err, Ksum)

        # ---- within LT2 (pooled dirs) ----
        ok, out, abs_err, _ = _get_out_block(res_mouse, "LT2_train_to_test")
        if ok:
            Ksum = np.asarray(out.get("Ksum", []), float)
            _add_mouse_condition(mouse, group, "within_LT2", abs_err, Ksum)

        # ---- LT1 -> LT2 ----
        xfer = res_mouse.get("LT1_to_LT2", {})
        if xfer.get("ok", False):
            out = xfer["out"]
            abs_err = np.asarray(xfer["abs_err"], float)
            Ksum = np.asarray(out.get("Ksum", []), float)
            _add_mouse_condition(mouse, group, "LT1_to_LT2", abs_err, Ksum)

    return pd.DataFrame(rows)

def fit_mixedlm_models(df, use_median=False):
    """
    Fits:
      median_Ksum ~ group * condition + (1|mouse)
      median_err_filt ~ group * condition + (1|mouse)
    if use_median=True, else fits mean_Ksum and mean_err_filt instead.
    """
    # categorical
    df = df.copy()
    df["group"] = df["group"].astype("category")
    df["condition"] = df["condition"].astype("category")

    err_col = "median_err_filt" if use_median else "mean_err_filt"

    # 1) Ksum model
    if use_median:
        m1 = smf.mixedlm("median_Ksum ~ group * condition", df, groups=df["mouse"])
    else:
        m1 = smf.mixedlm("mean_Ksum ~ group * condition", df, groups=df["mouse"])
    r1 = m1.fit(reml=False)

    # 2) Error model (filtered)
    # drop NaNs (e.g., if a mouse had no bins with Ksum>=kmin)
    df2 = df[np.isfinite(df[err_col])].copy()
    if use_median:
        m2 = smf.mixedlm("median_err_filt ~ group * condition", df2, groups=df2["mouse"])
    else:
        m2 = smf.mixedlm("mean_err_filt ~ group * condition", df2, groups=df2["mouse"])
    r2 = m2.fit(reml=False)

    return r1, r2

def run_mixedlm_pipeline(dec_results, mouse_groups, PLOTS_DIR, session_str, mapping,
                         kmin_for_err=5, lowK_thresh=1, use_median=False):
    stat_label = "median" if use_median else "mean"
    save_path = os.path.join(PLOTS_DIR, f"mixedlm_{session_str}_mapping_{mapping}_{stat_label}")
    os.makedirs(save_path, exist_ok=True)

    df = build_mixedlm_dataframe(dec_results, mouse_groups,
                                 kmin_for_err=kmin_for_err,
                                 lowK_thresh=lowK_thresh)
    df.to_csv(os.path.join(save_path, "mixedlm_mouse_level_summary.csv"), index=False)

    r1, r2 = fit_mixedlm_models(df, use_median=use_median)

    with open(os.path.join(save_path, f"mixedlm_{stat_label}_Ksum.txt"), "w") as f:
        f.write(str(r1.summary()))
    with open(os.path.join(save_path, f"mixedlm_{stat_label}_err_filt.txt"), "w") as f:
        f.write(str(r2.summary()))

    print("[OK] Saved mixedlm outputs to:", save_path)
    return df, r1, r2

def plot_Ksum_fit_mixedlm(
    *,
    LT1_group,
    LT2_group,
    dec_results,
    mouse_groups,
    PLOTS_DIR,
    session_str,
    mapping,
    kmin_for_err=5,
    lowK_thresh=1,
    use_median=False,
    auto_close=True,
):
    """
    Post-decoding:
      • Group-level Ksum plots (all conditions)
      • Error-vs-K plots (all conditions)
      • Mouse-level summaries:
            - within_LT1
            - within_LT2   <-- NEW
            - LT1_to_LT2   (via mixedlm df)
      • MixedLM fits
    """

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------
    stat_label = "median" if use_median else "mean"

    SUMMARY_DIR = os.path.join(
        PLOTS_DIR,
        f"summary_{session_str}_mapping_{mapping}_{stat_label}",
    )
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    MIXEDLM_DIR = os.path.join(
        PLOTS_DIR,
        f"mixedlm_{session_str}_mapping_{mapping}_{stat_label}",
    )
    os.makedirs(MIXEDLM_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Group Ksum + error plots (ALL conditions)
    # ------------------------------------------------------------
    for cond, nice in [
        ("LT1_train_to_test", "LT1 train→test"),
        ("LT1_test_to_train", "LT1 test→train"),
        ("LT2_train_to_test", "LT2 train→test"),   # NEW
        ("LT2_test_to_train", "LT2 test→train"),   # NEW
        ("LT1_to_LT2", "LT1→LT2"),
    ]:

        group_Ksum, _, _ = collect_group_Ksum(dec_results, mouse_groups, cond)

        if len(group_Ksum) > 0:
            plot_group_Ksum_histograms(
                group_Ksum,
                title=f"{session_str} {nice} Ksum",
                savefile=os.path.join(
                    SUMMARY_DIR,
                    f"Ksum_{cond}_by_group.png",
                ),
                auto_close=auto_close,
            )

        group_err, _, _ = collect_group_abs_err(
            dec_results,
            mouse_groups,
            cond,
            kmin_for_err=kmin_for_err,
        )

        if len(group_err) > 0:
            plot_group_error_histograms(
                group_err,
                title=f"{session_str} {nice} error (K ≥ {kmin_for_err})",
                savefile=os.path.join(
                    SUMMARY_DIR,
                    f"error_{cond}_by_group.png",
                ),
                auto_close=auto_close,
            )

    # ------------------------------------------------------------
    # 2) Mouse-level summaries
    # ------------------------------------------------------------

    # ---- within LT1 ----
    df_within_LT1 = build_mouse_level_summary_from_dec_results(
        dec_results,
        mouse_groups,
        condition="LT1_train_to_test",
        kmin_for_err=kmin_for_err,
        lowK_thresh=lowK_thresh,
    )

    df_within_LT1.to_csv(
        os.path.join(
            SUMMARY_DIR,
            "mouse_level_summary_within_LT1_all.csv",
        ),
        index=False,
    )
    print("[OK] wrote within_LT1 mouse summary:",
          df_within_LT1.shape)

    # ---- within LT2  (NEW) ----
    df_within_LT2 = build_mouse_level_summary_from_dec_results(
        dec_results,
        mouse_groups,
        condition="LT2_train_to_test",
        kmin_for_err=kmin_for_err,
        lowK_thresh=lowK_thresh,
    )

    df_within_LT2.to_csv(
        os.path.join(
            SUMMARY_DIR,
            "mouse_level_summary_within_LT2_all.csv",
        ),
        index=False,
    )
    print("[OK] wrote within_LT2 mouse summary:",
          df_within_LT2.shape)

    # ------------------------------------------------------------
    # 3) MixedLM dataframe (now includes within_LT1 + within_LT2 + LT1_to_LT2)
    # ------------------------------------------------------------
    df_mixed = build_mixedlm_dataframe(
        dec_results,
        mouse_groups,
        kmin_for_err=kmin_for_err,
        lowK_thresh=lowK_thresh,
    )

    df_mixed.to_csv(
        os.path.join(
            MIXEDLM_DIR,
            "mixedlm_mouse_level_summary.csv",
        ),
        index=False,
    )

    print("[OK] wrote MixedLM dataframe:",
          df_mixed.shape)

    # ------------------------------------------------------------
    # 4) Fit MixedLM
    # ------------------------------------------------------------
    r_Ksum, r_err = fit_mixedlm_models(df_mixed, use_median=use_median)

    with open(os.path.join(MIXEDLM_DIR, "mixedlm_results.txt"), "w") as f:
        f.write("===== Ksum model =====\n\n")
        f.write(str(r_Ksum))
        f.write("\n\n===== Error model =====\n\n")
        f.write(str(r_err))

    print("[OK] MixedLM models fitted.")

    return df_mixed, r_Ksum, r_err

def get_lap_arrays(LT):
    seg = getattr(LT, "ls", None)
    if seg is None:
        return None, None

    lap_idx = np.asarray(getattr(seg, "lap_idx", None))
    lap_dir = np.asarray(getattr(seg, "lap_dir", None))

    if lap_idx is None or lap_dir is None:
        return None, None

    return lap_idx.astype(int), lap_dir.astype(float)

def lap_direction_by_id(lap_idx, lap_dir):
    """
    Returns dict {lap_id: +1 or -1}
    """
    out = {}
    for lap in np.unique(lap_idx):
        if lap < 0:
            continue
        m = (lap_idx == lap) & np.isfinite(lap_dir)
        if not np.any(m):
            continue
        vals = lap_dir[m]

        # keep only clearly signed values
        vals = vals[np.abs(vals) > 0.1]
        if vals.size == 0:
            continue

        sign = +1 if np.sum(vals > 0) >= np.sum(vals < 0) else -1
        out[int(lap)] = sign
    return out

def decode_within_LT_by_direction(
    LT,
    *,
    label="LT",
    n_pos_bins=60,
    time_bin_frames=5,
    min_laps_per_dir=2,
    use_speed=True,
    min_speed=2.0,
    rng_seed=0,
    default_fps=20.0,
    S_override=None,          # NEW
):
    """
    Within-session decoding split by direction using LT.ls.lap_idx and LT.ls.lap_dir.
    Returns schema compatible with your run_decoding_all_mice_pipeline pooling logic.
    """

    # ---- dt_sec (per miniscope frame) ----
    t_ms = getattr(LT, "tstamp_miniscope", None)
    if t_ms is not None:
        t_ms = np.asarray(t_ms, float)
        if t_ms.size >= 3:
            dt_sec = float(np.nanmedian(np.diff(t_ms)) / 1000.0)
        else:
            dt_sec = 1.0 / float(default_fps)
    else:
        dt_sec = 1.0 / float(default_fps)

    # ---- data ----
    x = np.asarray(LT.miniscope_loc_1d_px, float)
    if S_override is None:
        S = np.asarray(LT.S, float)
    else:
        S = np.asarray(S_override, float)
    T = min(x.size, S.shape[1])
    x = x[:T]
    S = S[:, :T]

    seg = getattr(LT, "ls", None)
    if seg is None:
        return {"ok": False, "reason": f"{label}.ls missing (run attach_lap_segmentations first)"}

    lap_idx = np.asarray(getattr(seg, "lap_idx", None))
    lap_dir = np.asarray(getattr(seg, "lap_dir", None))
    if lap_idx is None or lap_dir is None:
        return {"ok": False, "reason": f"{label}.ls missing lap_idx/lap_dir"}

    lap_idx = lap_idx.astype(int)[:T]
    lap_dir = lap_dir.astype(float)[:T]

    # ---- map lap_id -> direction (majority vote) ----
    lapdir = {}
    for lap in np.unique(lap_idx):
        if lap < 0:
            continue
        m = (lap_idx == lap) & np.isfinite(lap_dir)
        if not np.any(m):
            continue
        vals = lap_dir[m]
        vals = vals[np.abs(vals) > 0.1]
        if vals.size == 0:
            continue
        lapdir[int(lap)] = +1 if np.sum(vals > 0) >= np.sum(vals < 0) else -1

    laps_R = sorted([k for k, v in lapdir.items() if v == +1])
    laps_L = sorted([k for k, v in lapdir.items() if v == -1])

    rng = np.random.RandomState(rng_seed)

    def build_mask_from_laps(laps_keep):
        laps_keep = set(int(l) for l in laps_keep)
        mask = np.isin(lap_idx, list(laps_keep))

        if use_speed:
            v = getattr(LT, "velocities_miniscope_smooth", None)
            if v is not None:
                v = np.asarray(v, float)[:T]
                mask = mask & (v >= float(min_speed))

        valid = getattr(seg, "valid_mask", None)
        if valid is not None:
            valid = np.asarray(valid, bool)[:T]
            mask = mask & valid

        # also enforce "in a lap"
        mask = mask & (lap_idx >= 0)
        return mask

    def run_dir(laps, dir_label):
        if len(laps) < min_laps_per_dir:
            return {"ok": False, "reason": f"too few laps for {label} {dir_label}: {len(laps)}"}

        laps = np.array(laps, int)
        laps_shuf = laps.copy()
        rng.shuffle(laps_shuf)

        train_laps = laps_shuf[::2]
        test_laps  = laps_shuf[1::2]

        if test_laps.size == 0:
            test_laps = train_laps[-1:]
            train_laps = train_laps[:-1]

        if train_laps.size == 0 or test_laps.size == 0:
            return {"ok": False, "reason": f"split failed for {label} {dir_label}"}

        train_mask = build_mask_from_laps(train_laps)
        test_mask  = build_mask_from_laps(test_laps)

        if train_mask.sum() < 10 * time_bin_frames or test_mask.sum() < 10 * time_bin_frames:
            return {"ok": False, "reason": f"not enough frames after mask for {label} {dir_label}"}

        # ---- train -> test ----
        dec = BayesianDecoder1D(n_pos_bins=n_pos_bins, time_bin_frames=time_bin_frames)
        dec.fit(S, x, train_mask, dt_sec)
        out_tr2te = dec.decode(S, x, test_mask)
        abs_err_tr2te = BayesianDecoder1D.abs_error(out_tr2te["decoded_pos"], out_tr2te["true_pos"])

        # ---- test -> train ----
        dec2 = BayesianDecoder1D(n_pos_bins=n_pos_bins, time_bin_frames=time_bin_frames)
        dec2.fit(S, x, test_mask, dt_sec)
        out_te2tr = dec2.decode(S, x, train_mask)
        abs_err_te2tr = BayesianDecoder1D.abs_error(out_te2tr["decoded_pos"], out_te2tr["true_pos"])

        return {
            "ok": True,
            "dir": dir_label,
            "train_laps": train_laps.tolist(),
            "test_laps": test_laps.tolist(),
            "n_laps_dir": int(len(laps)),
            "n_cells": int(S.shape[0]),
            "dt_sec": float(dt_sec),
            "bin_edges": np.asarray(dec.bin_edges),
            "train_to_test": {"out": out_tr2te, "abs_err": abs_err_tr2te},
            "test_to_train": {"out": out_te2tr, "abs_err": abs_err_te2tr},
        }

    rR = run_dir(laps_R, "rightward")
    rL = run_dir(laps_L, "leftward")

    ok_any = bool(rR.get("ok", False) or rL.get("ok", False))
    reason = "" if ok_any else f"R: {rR.get('reason','')} ; L: {rL.get('reason','')}"

    return {
        "ok": ok_any,
        "reason": reason,
        "rightward": rR,
        "leftward": rL,
        "n_laps_total": int(getattr(seg, "n_laps", -1)),
        "n_laps_R": int(len(laps_R)),
        "n_laps_L": int(len(laps_L)),
        "dt_sec": float(dt_sec),
    }

###
### Ksum, decoding error, and plots
###

# ----------------------------
# Small helpers
# ----------------------------
def _ensure_dir(path: str) -> str:
    if path is None:
        raise ValueError("_ensure_dir got path=None (bug upstream)")
    os.makedirs(path, exist_ok=True)
    return path

def _as_group_dict(groups=("hM3D", "hM4D", "mCherry")):
    return {g: [] for g in groups}

# ----------------------------
# 2) Collect per-bin arrays from dec_results (for distributions / scatter)
# ----------------------------
def collect_group_array_from_dec_results(
    dec_results: dict,
    mouse_groups: dict,
    *,
    condition: str,
    value_key: str = "Ksum",
    groups=("hM3D", "hM4D", "mCherry"),
):
    """
    Collect per-bin arrays across mice, grouped by DREADD group.

    condition options:
      - "LT1_to_LT2"
      - "LT1_train_to_test", "LT1_test_to_train"
      - "LT2_train_to_test", "LT2_test_to_train"

    value_key:
      - "Ksum" from out dict
      - "abs_err" uses abs_err from the condition block

    Returns:
      group_vals: dict(group -> concatenated 1D array)
      used_mice: dict(group -> list of mice)
      skipped: list of (mouse, reason)
    """
    group_vals = {g: [] for g in groups}
    used_mice = {g: [] for g in groups}
    skipped = []

    for mouse, res in dec_results.items():
        g = mouse_groups.get(mouse, "NA")
        if g not in group_vals:
            continue

        try:
            ok, out, abs_err, reason = _get_out_block(res, condition)
            if not ok or out is None:
                skipped.append((mouse, reason))
                continue

            if value_key == "abs_err":
                arr = np.asarray(abs_err, dtype=float)
            else:
                arr = np.asarray(out.get(value_key, []), dtype=float)

            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                skipped.append((mouse, f"{value_key} empty"))
                continue

            group_vals[g].append(arr)
            used_mice[g].append(mouse)

        except Exception as e:
            skipped.append((mouse, f"error: {type(e).__name__}: {e}"))

    for g in group_vals:
        group_vals[g] = np.concatenate(group_vals[g]) if len(group_vals[g]) else np.array([], dtype=float)

    return group_vals, used_mice, skipped


# ----------------------------
# 3) Group distribution plots
# ----------------------------

# ----------------------------
# Style + saving helpers
# ----------------------------
def _set_compact_plot_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
    })


def _savefig_tight(fig, savefile, dpi=300, pad_inches=0.25):
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    fig.savefig(savefile, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)

# ----------------------------
# Generic group histogram plotter
#   (use this for BOTH error and Ksum)
# ----------------------------
def plot_group_histograms(
    group_arrays,
    *,
    title: str,
    xlabel: str,
    savefile: str,
    bins: int = 60,
    density: bool = True,
    xlim=None,
    colors=None,
    auto_close: bool = True,
):
    """
    group_arrays: dict[str -> np.ndarray]
      e.g. {"hM3D": arr, "hM4D": arr, "mCherry": arr}

    density=True gives you the "Density" axis like your error distribution plot.
    density=False gives counts (useful for some Ksum views).
    """
    _set_compact_plot_style()

    if colors is None:
        colors = {"hM3D": "red", "hM4D": "blue", "mCherry": "black"}

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)

    order = ["hM3D", "hM4D", "mCherry"]
    for g in order:
        if g not in group_arrays:
            continue
        a = np.asarray(group_arrays[g], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue

        ax.hist(
            a,
            bins=bins,
            density=density,
            alpha=0.35,
            color=colors.get(g, "gray"),
            label=f"{g} (n={a.size})",
        )

    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="upper right")

    # tighter layout, but also reserve a bit for title
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _savefig_tight(fig, savefile, dpi=300, pad_inches=0.25)

    if auto_close:
        plt.close(fig)

    return fig, ax


# ----------------------------
# 3b) Aggregate decoded-vs-true 2D density panels across all mice per group
# ----------------------------

def _collect_decoded_true_per_group(
    dec_results: dict,
    mouse_groups: dict,
    condition: str,
    groups=("hM3D", "hM4D", "mCherry"),
    kmin: float = 0.0,
):
    """
    Collect concatenated (true_pos, decoded_pos) arrays per group.

    condition: "within_LT1", "within_LT2", or "LT1_to_LT2"

    For within-session conditions, both CV folds (train_to_test,
    test_to_train) and both directions (leftward, rightward) are
    concatenated.

    kmin : float
        If > 0, only keep decode bins whose Ksum >= kmin.

    Returns
    -------
    group_pairs : dict[str -> (true_pos_1d, decoded_pos_1d)]
    """
    group_true = {g: [] for g in groups}
    group_dec  = {g: [] for g in groups}

    def _append_filtered(g, tp, dp, Ksum):
        """Apply finite + optional Ksum filter, then append."""
        n = min(tp.size, dp.size)
        if n == 0:
            return
        tp, dp = tp[:n], dp[:n]
        m = np.isfinite(tp) & np.isfinite(dp)
        if Ksum is not None and kmin > 0:
            K = np.asarray(Ksum, float)
            K = K[:n]
            m = m & np.isfinite(K) & (K >= kmin)
        if m.sum() == 0:
            return
        group_true[g].append(tp[m])
        group_dec[g].append(dp[m])

    for mouse, res in dec_results.items():
        g = mouse_groups.get(mouse, "NA")
        if g not in group_true:
            continue

        if condition == "LT1_to_LT2":
            blk = res.get("LT1_to_LT2", {})
            if not blk.get("ok", False):
                continue
            out = blk.get("out", {})
            tp = np.asarray(out.get("true_pos", []), float)
            dp = np.asarray(out.get("decoded_pos", []), float)
            Ksum = out.get("Ksum", None)
            _append_filtered(g, tp, dp, Ksum)

        else:
            # within_LT1 or within_LT2
            sess = "LT1" if "LT1" in condition else "LT2"
            within = None
            for kk in (f"within_{sess}_by_dir",
                       f"within_{sess}_by_dir_full",
                       f"within_{sess}_by_dir_mapped"):
                d = res.get(kk)
                if isinstance(d, dict) and len(d) > 0:
                    within = d
                    break
            if within is None:
                continue

            for dir_label, dblk in within.items():
                if not isinstance(dblk, dict) or not dblk.get("ok", False):
                    continue
                for fold_key in ("train_to_test", "test_to_train"):
                    step = dblk.get(fold_key)
                    if not isinstance(step, dict):
                        continue
                    out = step.get("out")
                    if not isinstance(out, dict):
                        continue
                    tp = np.asarray(out.get("true_pos", []), float)
                    dp = np.asarray(out.get("decoded_pos", []), float)
                    Ksum = out.get("Ksum", None)
                    _append_filtered(g, tp, dp, Ksum)

    result = {}
    for g in groups:
        if group_true[g]:
            result[g] = (np.concatenate(group_true[g]),
                         np.concatenate(group_dec[g]))
        else:
            result[g] = (np.array([], float), np.array([], float))
    return result


def plot_aggregate_error_density(
    dec_results: dict,
    mouse_groups: dict,
    *,
    save_dir: str,
    session_str: str = "",
    bin_width_px: float = 10.0,
    kmin: float = 0.0,
    auto_close: bool = True,
    groups=("mCherry", "hM3D", "hM4D"),
    colors=None,
    density: bool = True,       # kept for API compat – unused now
):
    """
    3-row × 3-column figure of 2D decoded-vs-true heatmaps.

    Rows   = DREADD groups  (mCherry, hM3D, hM4D)
    Cols   = conditions     (Within LT1, Within LT2, LT1→LT2)

    Each panel is a 2D histogram (bin_width_px × bin_width_px) with a
    diagonal identity line.  Colormap intensity = number of decoded
    time-bins falling in each (true, decoded) cell.

    Parameters
    ----------
    kmin : float
        Minimum Ksum to keep a decode bin (0 = no filter).
    """
    from matplotlib.colors import LogNorm

    _set_compact_plot_style()

    if colors is None:
        colors = {"hM3D": "red", "hM4D": "blue", "mCherry": "black"}

    cond_specs = [
        ("within_LT1", "Within LT1"),
        ("within_LT2", "Within LT2"),
        ("LT1_to_LT2", "LT1 → LT2"),
    ]
    order = list(groups)

    # -- collect decoded/true per group per condition ----------------
    cond_data = {}
    for cond_key, nice in cond_specs:
        cond_data[cond_key] = _collect_decoded_true_per_group(
            dec_results, mouse_groups, cond_key, groups=groups, kmin=kmin,
        )

    # -- determine global axis range ---------------------------------
    all_vals = []
    for cond_key, _ in cond_specs:
        for g in order:
            tp, dp = cond_data[cond_key][g]
            if tp.size:
                all_vals.extend([tp.min(), tp.max(), dp.min(), dp.max()])
    if not all_vals:
        print("[WARN] plot_aggregate_error_density: no data → skip")
        return None, None, ""
    global_lo = min(all_vals)
    global_hi = max(all_vals)
    bins = np.arange(global_lo, global_hi + bin_width_px, bin_width_px)

    # -- figure layout -----------------------------------------------
    # Nature single-column = 89 mm ≈ 3.5 in; double-column = 183 mm ≈ 7.2 in
    # Use full page width (183 mm) with square panels
    panel_size = 2.0   # inches per panel (square)
    n_rows = len(order)
    n_cols = len(cond_specs)
    fig_w = panel_size * n_cols + 1.2   # +1.2 for labels + colorbar
    fig_h = panel_size * n_rows + 0.8   # +0.8 for column titles + x-labels
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        dpi=200,
        sharex=True, sharey=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    # -- pre-compute global vmax for shared colorbar -------------------
    global_max = 1
    for cond_key, _ in cond_specs:
        for g in order:
            tp, dp = cond_data[cond_key][g]
            if tp.size == 0:
                continue
            H_pre, _, _ = np.histogram2d(tp, dp, bins=[bins, bins])
            mx = int(H_pre.max())
            if mx > global_max:
                global_max = mx

    log_norm = LogNorm(vmin=1, vmax=global_max)
    last_mesh = None

    _TITLE_FS = 10
    _LABEL_FS = 9
    _TICK_FS  = 8
    _N_FS     = 7

    for ri, g in enumerate(order):
        for ci, (cond_key, nice) in enumerate(cond_specs):
            ax = axes[ri, ci]

            tp, dp = cond_data[cond_key][g]
            if tp.size == 0:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=_LABEL_FS, color="0.5")
            else:
                # 2D histogram
                H, xedges, yedges = np.histogram2d(tp, dp, bins=[bins, bins])
                H_masked = np.ma.masked_where(H == 0, H)

                last_mesh = ax.pcolormesh(
                    xedges, yedges, H_masked.T,
                    cmap="inferno",
                    norm=log_norm,
                    rasterized=True,
                )

            # identity line
            ax.plot([global_lo, global_hi], [global_lo, global_hi],
                    color="cyan", linewidth=0.8, alpha=0.7)

            # labels / titles
            if ri == 0:
                ax.set_title(nice, fontsize=_TITLE_FS, pad=4)
            if ci == 0:
                ax.set_ylabel(f"{g}\nDecoded pos (px)", fontsize=_LABEL_FS)
            if ri == n_rows - 1:
                ax.set_xlabel("True position (px)", fontsize=_LABEL_FS)

            ax.tick_params(labelsize=_TICK_FS, length=2, width=0.6)

            # n count annotation
            ax.text(0.03, 0.96, f"n={tp.size:,}", transform=ax.transAxes,
                    fontsize=_N_FS, va="top", ha="left", color="white",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"))

            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)

    fig.tight_layout(rect=[0, 0, 0.91, 1.0], h_pad=0.4, w_pad=0.4)

    # -- shared colorbar on the right --------------------------------
    if last_mesh is not None:
        cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
        cbar = fig.colorbar(last_mesh, cax=cbar_ax)
        cbar.set_label("Count", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS, length=2)

    _ensure_dir(save_dir)
    kmin_tag = f"_kmin{int(kmin)}" if kmin > 0 else ""
    savefile = os.path.join(
        save_dir,
        f"aggregate_decoded_vs_true_{session_str}{kmin_tag}.png",
    )
    _savefig_tight(fig, savefile, dpi=300, pad_inches=0.3)
    print(f"[SAVED] {savefile}")

    if auto_close:
        plt.close(fig)

    return fig, axes, savefile


# ----------------------------
# 4) Error vs Ksum scatter (LT1→LT2)
# ----------------------------
def plot_error_vs_Ksum_scatter_grouped_LT1_to_LT2(
    dec_results: dict,
    mouse_groups: dict,
    *,
    title: str,
    savefile: str,
    colors=None,
    alpha: float = 0.25,
    s: float = 10,
    max_points_per_mouse: int = None,  # None = all points
    auto_close: bool = True,
):
    """
    Scatter: each point = one decode chunk (Ksum, abs_err), colored by group.
    This is a *distributional* / mechanistic plot, not mouse-level inference.
    """
    if colors is None:
        colors = {"hM3D": "red", "hM4D": "blue", "mCherry": "black"}

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=150)

    for mouse, res in dec_results.items():
        g = mouse_groups.get(mouse, "NA")
        blk = res.get("LT1_to_LT2", {})
        if not blk.get("ok", False):
            continue

        out = blk.get("out", {})
        Ksum = np.asarray(out.get("Ksum", []), dtype=float)
        err = np.asarray(blk.get("abs_err", []), dtype=float)

        n = min(Ksum.size, err.size)
        if n == 0:
            continue
        Ksum = Ksum[:n]
        err = err[:n]

        m = np.isfinite(Ksum) & np.isfinite(err)
        Ksum = Ksum[m]
        err = err[m]
        if Ksum.size == 0:
            continue

        if max_points_per_mouse is not None and Ksum.size > max_points_per_mouse:
            idx = np.random.choice(Ksum.size, size=max_points_per_mouse, replace=False)
            Ksum = Ksum[idx]
            err = err[idx]

        ax.scatter(Ksum, err, color=colors.get(g, "gray"), alpha=alpha, s=s)

    ax.set_xlabel("Total events per decode bin (Ksum)")
    ax.set_ylabel("Absolute decoding error (pixels)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(savefile, dpi=300)
    if auto_close:
        plt.close(fig)
    return fig, ax

# ----------------------------
# 5) One-call wrapper to generate “communication” plots
# ----------------------------
def run_group_summary_plots(
    dec_results,
    mouse_groups,
    *,
    PLOTS_DIR: str,
    session_str: str,
    mapping: str,
    auto_close: bool = True,
    use_median: bool = False,
):
    """
    Produces:
      - within-LT1: dist_abs_err + dist_Ksum + mouse-level metric plots
      - within-LT2: dist_abs_err + dist_Ksum + mouse-level metric plots
      - LT1->LT2:   dist_abs_err + dist_Ksum + mouse-level metric plots (from mixedlm CSV if present)

    use_median: if True, plots use median_err_filt / median_Ksum; if False, mean_err_filt / mean_Ksum.
    CSVs always include both median and mean columns.

    Also writes:
      summary_{session_str}_mapping_{mapping}/mouse_level_summary_within_LT1_all.csv
      summary_{session_str}_mapping_{mapping}/mouse_level_summary_within_LT2_all.csv
    """
    stat_label = "median" if use_median else "mean"
    err_col  = f"{stat_label}_err_filt"
    ksum_col = f"{stat_label}_Ksum"

    save_path = os.path.join(PLOTS_DIR, f"summary_{session_str}_mapping_{mapping}_{stat_label}")
    _ensure_dir(save_path)

    # -----------------------
    # 1) Chunk-level dists: within LT1 and within LT2
    # -----------------------
    within_specs = [
        ("LT1_train_to_test", "LT1 train→test"),
        ("LT1_test_to_train", "LT1 test→train"),
        ("LT2_train_to_test", "LT2 train→test"),
        ("LT2_test_to_train", "LT2 test→train"),
    ]

    for cond, pretty in within_specs:
        # abs_err dist
        group_err, used_mice_err, skipped_err = collect_group_abs_err(dec_results, mouse_groups, cond)
        if sum(len(v) for v in used_mice_err.values()) > 0:
            plot_group_error_histograms(
                group_err,
                title=f"{session_str} {pretty}: decoding error distribution",
                savefile=os.path.join(save_path, f"dist_abs_err_{cond}_by_group.png"),
                auto_close=auto_close,
                font_scale=0.80,
            )

        # Ksum dist
        group_Ksum, used_mice_K, skipped_K = collect_group_Ksum(dec_results, mouse_groups, cond)
        if sum(len(v) for v in used_mice_K.values()) > 0:
            plot_group_Ksum_histograms(
                group_Ksum,
                title=f"{session_str} {pretty}: Ksum distribution",
                savefile=os.path.join(save_path, f"dist_Ksum_{cond}_by_group.png"),
                density=True,
                auto_close=auto_close,
                font_scale=0.80,
            )

    # -----------------------
    # 2) Chunk-level dists: LT1 -> LT2
    # -----------------------
    cond = "LT1_to_LT2"
    group_err, _, _ = collect_group_abs_err(dec_results, mouse_groups, cond)
    plot_group_error_histograms(
        group_err,
        title=f"{session_str} LT1→LT2: decoding error distribution",
        savefile=os.path.join(save_path, "dist_abs_err_LT1_to_LT2_by_group.png"),
        auto_close=auto_close,
        font_scale=0.80,
    )

    group_Ksum, _, _ = collect_group_Ksum(dec_results, mouse_groups, cond)
    plot_group_Ksum_histograms(
        group_Ksum,
        title=f"{session_str} LT1→LT2: Ksum distribution",
        savefile=os.path.join(save_path, "dist_Ksum_LT1_to_LT2_by_group.png"),
        density=True,
        auto_close=auto_close,
        font_scale=0.80,
    )

    # -----------------------
    # 3) Mouse-level summaries: within LT1 + within LT2 (computed from dec_results)
    # -----------------------
    # NOTE: these two conditions define "within session" for your summary.
    # If you prefer to average train→test and test→train, say so, and we’ll adjust.
    for sess_label, cond_within in [
        ("LT1", "LT1_train_to_test"),
        ("LT2", "LT2_train_to_test"),
    ]:
        df_within = build_mouse_level_summary_from_dec_results(
            dec_results,
            mouse_groups,
            condition=cond_within,
            kmin_for_err=5,
            lowK_thresh=1,
        )

        out_csv = os.path.join(save_path, f"mouse_level_summary_within_{sess_label}_all.csv")
        df_within.to_csv(out_csv, index=False)

        # If empty, don’t try to plot.
        if df_within.empty:
            print(f"[run_group_summary_plots] No mouse-level rows for within {sess_label} ({cond_within})")
            continue

        plot_mouse_level_metric_by_group(
            df_within,
            metric_col=err_col,
            title=f"{session_str} within-{sess_label}: mouse-level {stat_label} decoding error",
            savefile=os.path.join(save_path, f"mouse_level_{stat_label}_err_filt_within_{sess_label}_by_group.png"),
            auto_close=auto_close,
            font_scale=0.70,
            figsize=(6.6, 4.8),
        )
        plot_mouse_level_metric_by_group(
            df_within,
            metric_col=ksum_col,
            title=f"{session_str} within-{sess_label}: mouse-level {stat_label} Ksum",
            savefile=os.path.join(save_path, f"mouse_level_{stat_label}_Ksum_within_{sess_label}_by_group.png"),
            auto_close=auto_close,
            font_scale=0.70,
            figsize=(6.6, 4.8),
        )

    # -----------------------
    # 4) Mouse-level plots for LT1->LT2 from mixedlm CSV (if present)
    # -----------------------
    mixedlm_dir = os.path.join(PLOTS_DIR, f"mixedlm_{session_str}_mapping_{mapping}_{stat_label}")
    csv_path = os.path.join(mixedlm_dir, "mixedlm_mouse_level_summary.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {err_col, ksum_col}.issubset(df.columns):
            plot_mouse_level_metric_by_group(
                df,
                metric_col=err_col,
                title=f"{session_str} LT1→LT2: mouse-level {stat_label} decoding error",
                savefile=os.path.join(save_path, f"mouse_level_{stat_label}_err_filt_LT1_to_LT2_by_group.png"),
                auto_close=auto_close,
                font_scale=0.70,
                figsize=(6.6, 4.8),
            )
            plot_mouse_level_metric_by_group(
                df,
                metric_col=ksum_col,
                title=f"{session_str} LT1→LT2: mouse-level {stat_label} Ksum",
                savefile=os.path.join(save_path, f"mouse_level_{stat_label}_Ksum_LT1_to_LT2_by_group.png"),
                auto_close=auto_close,
                font_scale=0.70,
                figsize=(6.6, 4.8),
            )
        else:
            print(f"[run_group_summary_plots] CSV missing expected columns ({err_col}, {ksum_col}): {csv_path}")
    else:
        print(f"[run_group_summary_plots] mixedlm CSV not found: {csv_path}")

    print(f"[run_group_summary_plots] Saved summary plots -> {save_path}")
    return save_path


# ----------------------------
# Style helpers
# ----------------------------
def set_compact_plot_style(font_scale=0.85):
    """
    Smaller, consistent text sizing so titles/legends never clip.
    Call at top of each plotting function (or once globally).
    """
    base = 12.0
    fs = base * float(font_scale)

    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs * 1.15,
        "axes.labelsize": fs * 1.05,
        "xtick.labelsize": fs * 0.95,
        "ytick.labelsize": fs * 0.95,
        "legend.fontsize": fs * 0.95,
        "figure.titlesize": fs * 1.2,
    })


# ----------------------------
# Collectors (abs_err + Ksum)
# ----------------------------
def collect_group_abs_err(
    dec_results,
    mouse_groups,
    condition,
    kmin_for_err=5,
):
    """
    Collect abs decoding error per group for a given condition, but only
    for timebins where Ksum >= kmin_for_err.

    Returns:
        group_err: dict[group] -> list of 1D arrays (per mouse)
        per_mouse: dict[mouse] -> 1D abs_err array (thresholded)
        reasons: dict[mouse] -> str reason if skipped
    """
    import numpy as np

    group_err = {}
    per_mouse = {}
    reasons = {}

    for mouse, res_mouse in dec_results.items():
        group = mouse_groups.get(mouse, "NA")

        ok, out, abs_err, reason = _get_out_block(res_mouse, condition)

        if not ok:
            reasons[mouse] = reason
            continue

        Ksum = np.asarray(out.get("Ksum", []), float)
        abs_err = np.asarray(abs_err, float)

        n = min(Ksum.size, abs_err.size)
        if n == 0:
            reasons[mouse] = "empty Ksum/abs_err"
            continue

        Ksum = Ksum[:n]
        abs_err = abs_err[:n]

        keep = (Ksum >= float(kmin_for_err)) & np.isfinite(abs_err) & np.isfinite(Ksum)
        abs_err_f = abs_err[keep]

        if abs_err_f.size == 0:
            reasons[mouse] = f"no bins pass Ksum >= {kmin_for_err}"
            continue

        per_mouse[mouse] = abs_err_f
        group_err.setdefault(group, []).append(abs_err_f)

    return group_err, per_mouse, reasons


# ----------------------------
# Group distribution plots
# ----------------------------
def plot_group_error_histograms(
    group_err,
    *,
    title,
    savefile,
    bins=40,
    density=True,
    xlim=None,
    font_scale=1.0,   # <-- re-added
    auto_close=True,
):
    """
    group_err: dict[group] -> list of 1D arrays (per mouse)
    Concatenates per-mouse arrays within each group.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(savefile), exist_ok=True)

    # Optional font scaling
    plt.rcParams.update({
        "font.size": 10 * font_scale,
        "axes.titlesize": 11 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "legend.fontsize": 9 * font_scale,
    })

    # --- concatenate per group ---
    group_vec = {}
    for g, arr_list in group_err.items():
        if not arr_list:
            continue

        flat = []
        for a in arr_list:
            if a is None:
                continue
            a = np.asarray(a, float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                flat.append(a)

        if flat:
            group_vec[g] = np.concatenate(flat)

    if not group_vec:
        print(f"[WARN] No data for {title}")
        return

    order = ["mCherry", "hM3D", "hM4D"]
    groups = [g for g in order if g in group_vec] + [g for g in group_vec if g not in order]

    color_map = {
        "mCherry": "gray",
        "hM3D": "black",
        "hM4D": "tab:blue",
    }

    plt.figure(figsize=(7.5, 4.5), dpi=150)

    for g in groups:
        vec = group_vec[g]
        plt.hist(
            vec,
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{g} (n={vec.size})",
            color=color_map.get(g, None),
        )

    plt.title(title)
    plt.xlabel("Absolute decoding error (pixels)")
    plt.ylabel("Density" if density else "Count")
    plt.legend(frameon=False)

    if xlim is not None:
        plt.xlim(xlim)

    plt.tight_layout()
    plt.savefig(savefile)

    if auto_close:
        plt.close()


def plot_group_Ksum_histograms(
    group_Ksum,
    title,
    savefile,
    bins=50,
    xlim=None,
    density=False,
    auto_close=True,
    font_scale=0.80,
):
    """
    KEEP THIS VERSION (compact + tight saving).
    group_Ksum: dict(group-> np.array)
    """
    set_compact_plot_style(font_scale=font_scale)
    colors = {"hM3D": "red", "hM4D": "blue", "mCherry": "black"}

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    for group, Ksum in group_Ksum.items():
        Ksum = np.asarray(Ksum)
        if Ksum.size == 0:
            continue
        ax.hist(
            Ksum,
            bins=bins,
            alpha=0.4,
            label=f"{group} (n={Ksum.size})",
            color=colors.get(group, "gray"),
            density=density,
        )

    ax.set_xlabel("Total events per decode bin (Ksum)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title, pad=8)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savefile, dpi=300, bbox_inches="tight", pad_inches=0.2)

    if auto_close:
        plt.close(fig)

    return fig, ax


# ----------------------------
# Mouse-level plots (FIXED TEXT)
# ----------------------------
def plot_mouse_level_metric_by_group(
    df_mouse_level: pd.DataFrame,
    metric_col: str,
    title: str,
    savefile: str,
    *,
    group_col: str = "group",
    mouse_col: str = "mouse",
    agg_col: str = None,         # unused, kept for compatibility
    auto_close: bool = True,
    font_scale: float = 0.75,    # <-- SMALLER TEXT HERE
    figsize=(6.8, 4.8),
):
    """
    df_mouse_level must have at least columns: [mouse_col, group_col, metric_col]
    Plots per-mouse dots + mean±SEM (or mean±SD if you prefer; this uses SEM).
    """
    set_compact_plot_style(font_scale=font_scale)

    colors = {"hM3D": "red", "hM4D": "blue", "mCherry": "black"}

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    groups = [g for g in ["hM3D", "hM4D", "mCherry"] if g in set(df_mouse_level[group_col])]
    x_positions = np.arange(len(groups))

    # jittered points + summary bars
    rng = np.random.default_rng(0)
    for xi, g in enumerate(groups):
        sub = df_mouse_level[df_mouse_level[group_col] == g]
        y = sub[metric_col].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        jitter = rng.normal(loc=0.0, scale=0.04, size=y.size)
        ax.scatter(
            np.full(y.size, xi) + jitter,
            y,
            s=55,
            color=colors.get(g, "gray"),
            edgecolor="k",
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
        )

        mean = float(np.mean(y))
        sem = float(np.std(y, ddof=1) / np.sqrt(max(y.size, 1))) if y.size > 1 else 0.0
        ax.errorbar(
            [xi],
            [mean],
            yerr=[sem],
            fmt="none",
            ecolor="k",
            elinewidth=2.5,
            capsize=7,
            zorder=4,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)
    ax.set_title(title, pad=10)
    ax.set_ylabel(metric_col)
    ax.grid(True, axis="y", alpha=0.25)

    # Give extra top margin for long titles (your session strings are huge)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(savefile, dpi=300, bbox_inches="tight", pad_inches=0.2)

    if auto_close:
        plt.close(fig)

    return fig, ax

def build_mouse_level_summary_from_dec_results(
    dec_results,
    mouse_groups,
    condition: str,
    *,
    kmin_for_err: int = 5,
    lowK_thresh: int = 1,
):
    """
    Builds mouse-level summary directly from dec_results via _get_out_block().

    Returns df with:
      mouse, group, condition, median_err_filt, median_Ksum, n_err_used, n_Ksum_used
    """
    rows = []

    for mouse, res_mouse in dec_results.items():
        ok, out, abs_err, reason = _get_out_block(res_mouse, condition)
        if (not ok) or (out is None):
            continue

        Ksum = np.asarray(out.get("Ksum", []))
        abs_err = np.asarray(abs_err)

        if Ksum.size == 0 or abs_err.size == 0:
            continue
        if Ksum.shape[0] != abs_err.shape[0]:
            print(f"[build_mouse_level_summary] shape mismatch {mouse} {condition}: "
                  f"Ksum {Ksum.shape} vs err {abs_err.shape}")
            continue

        # --- match your LT1->LT2 semantics:
        # error summary uses only bins with enough events
        err_used = abs_err[Ksum >= kmin_for_err]
        # Ksum summary can optionally drop ultra-low bins
        Ksum_used = Ksum[Ksum > lowK_thresh]

        row = dict(
            mouse=mouse,
            group=mouse_groups.get(mouse, None),
            condition=condition,

            # unfiltered
            median_err = float(np.nanmedian(abs_err)) if abs_err.size else np.nan,
            mean_err   = float(np.nanmean(abs_err))   if abs_err.size else np.nan,

            # filtered           
            mean_err_filt=float(np.nanmean(err_used)) if err_used.size else np.nan,
            median_err_filt=float(np.nanmedian(err_used)) if err_used.size else np.nan,

            median_Ksum=float(np.nanmedian(Ksum_used)) if Ksum_used.size else np.nan,
            mean_Ksum=float(np.nanmean(Ksum_used)) if Ksum_used.size else np.nan,

            n_err_used=int(err_used.size),
            n_Ksum_used=int(Ksum_used.size),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["group"].notna()].copy()

    return df

# ============================================================
# MixedLM fitting helper (purely reads CSVs, no recomputation)
# ============================================================
def fit_mixedlm_from_outputs(
    PLOTS_DIR: str,
    session_str: str,
    mapping: str,
    tag: str,
    *,
    do_within_only: bool = True,
    use_median=False,
    use_filt=True
):
    """
    Fits:
      (A) 3-level model ALWAYS:
          <err_col> ~ group * condition + (1|mouse)
          where err_col = median_err_filt (use_median=True) or mean_err_filt (use_median=False)
          condition levels: within_LT1 (ref), within_LT2, LT1_to_LT2

      (B) within-only model OPTIONAL (if do_within_only=True):
          same formula but only within_LT1 vs within_LT2

    Returns:
      df_all_clean, res_3level, df_within_only_or_None, res_within_or_None
    """
    stat_label = "median" if use_median else "mean"
    
    if use_filt:
        err_col = f"{stat_label}_err_filt"
    else:
        err_col = f"{stat_label}_err"

    save_path = os.path.join(PLOTS_DIR, f"summary_{session_str}_mapping_{mapping}_{stat_label}")
    mixedlm_dir = os.path.join(PLOTS_DIR, f"mixedlm_{session_str}_mapping_{mapping}_{stat_label}")

    # ---------- within LT1 ----------
    df_within1 = pd.read_csv(os.path.join(save_path, "mouse_level_summary_within_LT1_all.csv"))[
        ["mouse", "group", err_col]
    ].copy()
    df_within1["condition"] = "within_LT1"

    # ---------- within LT2 ----------
    df_within2 = pd.read_csv(os.path.join(save_path, "mouse_level_summary_within_LT2_all.csv"))[
        ["mouse", "group", err_col]
    ].copy()
    df_within2["condition"] = "within_LT2"

    # ---------- LT1 -> LT2 ----------
    df_across = pd.read_csv(os.path.join(mixedlm_dir, "mixedlm_mouse_level_summary.csv"))[
        ["mouse", "group", err_col]
    ].copy()
    df_across["condition"] = "LT1_to_LT2"

    # ---------- combine ----------
    df_all_clean = pd.concat([df_within1, df_within2, df_across], ignore_index=True)

    # enforce reference levels
    df_all_clean["mouse"] = df_all_clean["mouse"].astype(str)
    df_all_clean["group"] = pd.Categorical(
        df_all_clean["group"].astype(str),
        categories=["mCherry", "hM3D", "hM4D"],
        ordered=False,
    )
    df_all_clean["condition"] = pd.Categorical(
        df_all_clean["condition"].astype(str),
        categories=["within_LT1", "within_LT2", "LT1_to_LT2"],
        ordered=True,
    )

    print("\n============================================================")
    print(f"[{tag}] Using {stat_label}_err_filt for MixedLM")
    print(f"[{tag}] Condition counts:")
    print(df_all_clean["condition"].value_counts(dropna=False))
    print(f"[{tag}] Rows per mouse:")
    print(df_all_clean.groupby("mouse").size().value_counts())

    # ---------- 3-level model ----------
    formula = f"{err_col} ~ group * condition"
    model_3 = smf.mixedlm(
        formula,
        df_all_clean,
        groups=df_all_clean["mouse"],
    )
    res_3 = model_3.fit(reml=False)

    print("\n============================================================")
    print(f"[{tag}] 3-LEVEL MODEL (within_LT1 ref) [{stat_label}]")
    print(res_3.summary())

    # ---------- within-only optional ----------
    df_within_only = None
    res_within = None

    if do_within_only:
        df_within_only = df_all_clean[
            df_all_clean["condition"].isin(["within_LT1", "within_LT2"])
        ].copy()

        df_within_only["condition"] = pd.Categorical(
            df_within_only["condition"].astype(str),
            categories=["within_LT1", "within_LT2"],
            ordered=True,
        )

        model_w = smf.mixedlm(
            formula,
            df_within_only,
            groups=df_within_only["mouse"],
        )
        res_within = model_w.fit(reml=False)

        print("\n============================================================")
        print(f"[{tag}] WITHIN-ONLY MODEL (LT2 vs LT1) [{stat_label}]")
        print(res_within.summary())

    return df_all_clean, res_3, df_within_only, res_within


##
# Plot MixedLM results
##

def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _design_row_for(res, group, condition):
    """
    Robust patsy-based row for "1 + group * condition", aligned to res.fe_params.
    """
    df = pd.DataFrame([{"group": group, "condition": condition}])
    df["group"] = pd.Categorical(df["group"], categories=["mCherry", "hM3D", "hM4D"], ordered=False)
    df["condition"] = pd.Categorical(df["condition"], categories=["within_LT1", "within_LT2", "LT1_to_LT2"], ordered=True)

    X = dmatrix("1 + group * condition", df, return_type="dataframe")
    fe_names = list(res.fe_params.index)
    X = X.reindex(columns=fe_names, fill_value=0.0)
    return X.values[0], fe_names


def _cov_fe(res):
    """
    Fixed-effects covariance submatrix (guarding against extra params).
    """
    fe_names = list(res.fe_params.index)
    cov = pd.DataFrame(res.cov_params(), index=res.cov_params().index, columns=res.cov_params().columns)
    return cov.loc[fe_names, fe_names].values


def _fixed_pred_ci(res, xrow, alpha=0.05):
    """
    Fixed-effect predicted mean + Wald CI.
    """
    beta = res.fe_params.values
    covb = _cov_fe(res)

    mu = float(xrow @ beta)
    var = float(xrow @ covb @ xrow.T)
    se = np.sqrt(max(var, 0.0))

    z = norm.ppf(1 - alpha / 2)
    lo, hi = mu - z * se, mu + z * se
    return mu, se, lo, hi


def _contrast_wald(res, xA, xB):
    """
    Wald z-test for (B - A) on fixed effects.
    """
    beta = res.fe_params.values
    covb = _cov_fe(res)

    L = (xB - xA)
    diff = float(L @ beta)
    var = float(L @ covb @ L.T)
    se = np.sqrt(max(var, 0.0))

    if se == 0:
        return diff, se, np.nan, 1.0

    z = diff / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return diff, se, z, p


def _abs_table(res, model_label):
    groups = ["mCherry", "hM3D", "hM4D"]
    conds = ["within_LT1", "within_LT2", "LT1_to_LT2"]
    rows = []
    for g in groups:
        for c in conds:
            x, _ = _design_row_for(res, g, c)
            mu, se, lo, hi = _fixed_pred_ci(res, x, alpha=0.05)
            rows.append({
                "model": model_label, "group": g, "condition": c,
                "mu": mu, "lo": lo, "hi": hi, "xrow": x
            })
    return pd.DataFrame(rows)


def _contrast_table(res, abs_df, model_label):
    """
    Two contrasts per group, with UNCORRECTED Wald p-values:
      1) Within-LT2 − Within-LT1
      2) LT1→LT2 − Within-LT1
    """
    groups = ["mCherry", "hM3D", "hM4D"]
    out = []
    for g in groups:
        x_w1 = abs_df[(abs_df.group == g) & (abs_df.condition == "within_LT1")]["xrow"].iloc[0]
        x_w2 = abs_df[(abs_df.group == g) & (abs_df.condition == "within_LT2")]["xrow"].iloc[0]
        x_xf = abs_df[(abs_df.group == g) & (abs_df.condition == "LT1_to_LT2")]["xrow"].iloc[0]

        # (w2 - w1)
        d1, se1, z1, p1 = _contrast_wald(res, x_w1, x_w2)
        out.append({"model": model_label, "group": g, "contrast": "Within-LT2 − Within-LT1",
                    "diff": d1, "se": se1, "p": p1})

        # (xf - w1)
        d2, se2, z2, p2 = _contrast_wald(res, x_w1, x_xf)
        out.append({"model": model_label, "group": g, "contrast": "LT1→LT2 − Within-LT1",
                    "diff": d2, "se": se2, "p": p2})

    df = pd.DataFrame(out)
    df["stars"] = df["p"].apply(p_to_stars)
    return df


def plot_grouped_full_vs_shared(
    res_full_model,
    res_shared_model,
    *,
    PLOTS_DIR,
    fname="decoding_grouped_by_mousegroup_FULL_vs_SHARED.png",
    title="Decoding errors: All neurons (session-specific) vs Cross-registered neurons (shared)\nFixed-effects predictions; Wald p-values for contrasts (uncorrected)",
    figsize=(14, 7),
    use_median=False,
):
    """
    Creates a 2×3 figure:
      Row 1: Absolute predicted decoding errors (within_LT1, within_LT2, LT1→LT2)
             x-axis = mouse group; within each group = (All neurons vs Cross-registered) bars with CI.
      Row 2: Contrast bars (Within-LT2−Within-LT1, LT1→LT2−Within-LT1), same grouping,
             includes Wald stars (uncorrected) on each bar.
    """
    stat_label = "median" if use_median else "mean"
    out_dir = os.path.join(PLOTS_DIR, f"mixedlm_prediction_plots_{stat_label}")
    os.makedirs(out_dir, exist_ok=True)
    savefile = os.path.join(out_dir, fname)

    abs_full = _abs_table(res_full_model, "All neurons (session-specific)")
    abs_shr  = _abs_table(res_shared_model, "Cross-registered neurons (shared)")

    con_full = _contrast_table(res_full_model, abs_full, "All neurons (session-specific)")
    con_shr  = _contrast_table(res_shared_model, abs_shr,  "Cross-registered neurons (shared)")

    groups = ["mCherry", "hM3D", "hM4D"]
    conds = ["within_LT1", "within_LT2", "LT1_to_LT2"]
    cond_titles = {
        "within_LT1": "Within-LT1 decoding error",
        "within_LT2": "Within-LT2 decoding error",
        "LT1_to_LT2": "LT1→LT2 decoding error\n(train on LT1, decode LT2)",
    }

    # layout
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey="row")
    fig.suptitle(title, fontsize=14)

    # x positions: group centers
    x = np.arange(len(groups)) * 1.3  # adds gap between groups
    w = 0.32                          # bar width
    x_full = x - w/2
    x_shr  = x + w/2

    # -------- Row 1: absolute (3 conditions) --------
    for j, cond in enumerate(conds):
        ax = axes[0, j]
        ax.set_title(cond_titles[cond], fontsize=11)

        subF = abs_full[abs_full.condition == cond].set_index("group").loc[groups]
        subS = abs_shr [abs_shr .condition == cond].set_index("group").loc[groups]

        muF = subF["mu"].values
        muS = subS["mu"].values
        errF = np.vstack([muF - subF["lo"].values, subF["hi"].values - muF])
        errS = np.vstack([muS - subS["lo"].values, subS["hi"].values - muS])

        ax.bar(x_full, muF, width=w, yerr=errF, capsize=3, label="All neurons (session-specific)")
        ax.bar(x_shr,  muS, width=w, yerr=errS, capsize=3, label="Cross-registered neurons (shared)")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.grid(True, axis="y", alpha=0.25)

        if j == 0:
            stat_label = "median" if use_median else "mean"
            ax.set_ylabel(f"Predicted {stat_label} decoding error")
        if j == 2:
            ax.legend(fontsize=9)

    # -------- Row 2: contrasts (2 contrasts; we’ll use only first two panels, leave third empty) --------
    contrasts = ["Within-LT2 − Within-LT1", "LT1→LT2 − Within-LT1"]

    for j, cname in enumerate(contrasts):
        ax = axes[1, j]
        ax.set_title(cname, fontsize=11)

        subF = con_full[con_full.contrast == cname].set_index("group").loc[groups]
        subS = con_shr [con_shr .contrast == cname].set_index("group").loc[groups]

        dF = subF["diff"].values
        dS = subS["diff"].values
        seF = subF["se"].values
        seS = subS["se"].values

        # ~95% CI bars
        yerrF = 1.96 * seF
        yerrS = 1.96 * seS

        barsF = ax.bar(x_full, dF, width=w, yerr=yerrF, capsize=3, label="All neurons")
        barsS = ax.bar(x_shr,  dS, width=w, yerr=yerrS, capsize=3, label="Cross-registered")

        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.grid(True, axis="y", alpha=0.25)

        if j == 0:
            ax.set_ylabel("Contrast vs Within-LT1")

        # stars (UNCORRECTED Wald p-values)
        for i, g in enumerate(groups):
            starF = subF.loc[g, "stars"]
            starS = subS.loc[g, "stars"]

            # place slightly above CI top (or below if negative)
            def _place_star(bar, star):
                if not star:
                    return
                h = bar.get_height()
                cx = bar.get_x() + bar.get_width()/2
                offset = 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 50)
                ax.text(cx, h + (offset if h >= 0 else -offset), star,
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=12, fontweight="bold")

            _place_star(barsF[i], starF)
            _place_star(barsS[i], starS)

    # third bottom panel unused
    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(savefile, dpi=200)
    plt.close(fig)

    print("[SAVED]", savefile)
    return abs_full, con_full, abs_shr, con_shr, savefile

##
## END MixedLM plotting
##


##
## Within-group contrast tests
##

import numpy as np
import pandas as pd
from scipy import stats

def wald_contrast_linear_comb(res, L, label="contrast"):
    beta = res.fe_params.values
    cov = res.cov_params().loc[res.fe_params.index, res.fe_params.index].values

    est = float(L @ beta)
    var = float(L @ cov @ L)
    se = np.sqrt(var) if var > 0 else np.nan
    z = est / se if se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return dict(label=label, est=est, se=se, z=z, p=p)

def within_group_condition_contrasts(res):
    names = list(res.fe_params.index)

    def Lvec(terms):
        L = np.zeros(len(names))
        for t, w in terms.items():
            if t not in names:
                raise KeyError(f"Term not found in fixed effects: {t}")
            L[names.index(t)] = w
        return L

    groups = ["mCherry", "hM3D", "hM4D"]

    rows = []
    for g in groups:
        # (within_LT2 - within_LT1)
        terms = {"condition[T.within_LT2]": 1.0}
        if g != "mCherry":
            terms[f"group[T.{g}]:condition[T.within_LT2]"] = 1.0
        rows.append(wald_contrast_linear_comb(
            res,
            Lvec(terms),
            label=f"{g}: within_LT2 - within_LT1"
        ))

        # (LT1_to_LT2 - within_LT1)
        terms = {"condition[T.LT1_to_LT2]": 1.0}
        if g != "mCherry":
            terms[f"group[T.{g}]:condition[T.LT1_to_LT2]"] = 1.0
        rows.append(wald_contrast_linear_comb(
            res,
            Lvec(terms),
            label=f"{g}: LT1_to_LT2 - within_LT1"
        ))

    df = pd.DataFrame(rows)
    return df


def correct_contrasts_pvalues(df_contrasts, methods=("holm", "fdr_bh")):
    """
    Apply multiple-comparisons corrections to Wald contrast p-values.

    Two correction families are computed:
      1) "all" — correcting across all contrasts in df_contrasts
      2) per-contrast-type — correcting only across groups within each
         contrast type (e.g. just the 3 "within_LT2 - within_LT1" tests).

    Parameters
    ----------
    df_contrasts : DataFrame with columns 'label' and 'p'
                   (from within_group_condition_contrasts)
    methods : tuple of correction methods passed to statsmodels multipletests.
              Default: Holm-Bonferroni and Benjamini-Hochberg FDR.

    Returns
    -------
    df_contrasts : same DataFrame with added columns:
        p_holm, p_fdr_bh, sig_holm, sig_fdr_bh           (all-contrasts family)
        p_holm_pct, p_fdr_bh_pct, sig_holm_pct, sig_fdr_bh_pct  (per-contrast-type family)
    """
    from statsmodels.stats.multitest import multipletests

    # --- 1) correct across ALL contrasts ---
    for method in methods:
        reject, pvals_corrected, _, _ = multipletests(df_contrasts["p"].values, method=method)
        col = f"p_{method}"
        df_contrasts[col] = pvals_corrected
        df_contrasts[f"sig_{method}"] = reject

    # --- 2) correct per contrast type (e.g. 3 groups within same contrast) ---
    # Extract contrast type from label: everything after ": "
    contrast_types = df_contrasts["label"].str.split(": ", n=1).str[1]
    for ctype in contrast_types.unique():
        mask = contrast_types == ctype
        if mask.sum() < 2:
            continue
        for method in methods:
            reject, pvals_corrected, _, _ = multipletests(
                df_contrasts.loc[mask, "p"].values, method=method
            )
            df_contrasts.loc[mask, f"p_{method}_pct"] = pvals_corrected
            df_contrasts.loc[mask, f"sig_{method}_pct"] = reject

    return df_contrasts


##
## Within-group contrast tests END
##


##
## Plotting LT decoder
##
 
# ----------------------------
# Helpers: fixed-effects prediction + Wald contrasts
# ----------------------------

# Display labels (requested)
GROUPS = ["mCherry", "hM3D", "hM4D"]
GROUP_LABELS = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}

CONDS  = ["within_LT1", "within_LT2", "LT1_to_LT2"]
COND_LABELS_ABS = {
    "within_LT1": "Within LT1",
    "within_LT2": "Within LT2",
    "LT1_to_LT2": "LT1 to LT2",
}

def _star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.08:  return f"p={p:.2f}"
    return ""

def _pad_axes_for_stars(ax, margin_frac=0.12):
    """Expand y-limits so star annotations don't clip."""
    y_lo, y_hi = ax.get_ylim()
    rng = y_hi - y_lo if y_hi > y_lo else 1.0
    pad = margin_frac * rng
    ax.set_ylim(y_lo - pad, y_hi + pad)

def _fe_design_vector(param_index, group, cond):
    """
    Build a fixed-effects design vector aligned to res.fe_params.index for model:
      median_err_filt ~ group * condition
    with reference: group=mCherry, condition=within_LT1
    """
    idx = list(param_index)
    x = np.zeros(len(idx), dtype=float)

    def set_if(name, val=1.0):
        if name in idx:
            x[idx.index(name)] = val

    set_if("Intercept", 1.0)

    # group main effects
    if group == "hM3D":
        set_if("group[T.hM3D]", 1.0)
    elif group == "hM4D":
        set_if("group[T.hM4D]", 1.0)

    # condition main effects
    if cond == "within_LT2":
        set_if("condition[T.within_LT2]", 1.0)
    elif cond == "LT1_to_LT2":
        set_if("condition[T.LT1_to_LT2]", 1.0)

    # interactions
    if group == "hM3D" and cond == "within_LT2":
        set_if("group[T.hM3D]:condition[T.within_LT2]", 1.0)
    if group == "hM4D" and cond == "within_LT2":
        set_if("group[T.hM4D]:condition[T.within_LT2]", 1.0)
    if group == "hM3D" and cond == "LT1_to_LT2":
        set_if("group[T.hM3D]:condition[T.LT1_to_LT2]", 1.0)
    if group == "hM4D" and cond == "LT1_to_LT2":
        set_if("group[T.hM4D]:condition[T.LT1_to_LT2]", 1.0)

    return x

def predicted_mean_sem(res, group, cond):
    """
    Returns (mean, sem) for fixed-effects prediction in given group/cond.
    SEM here = model-based standard error of the predicted mean (fixed effects).
    """
    beta = res.fe_params.values
    cov  = np.asarray(res.cov_params())
    p_fe = len(res.fe_params)
    cov  = cov[:p_fe, :p_fe]

    x = _fe_design_vector(res.fe_params.index, group, cond)
    mean = float(x @ beta)
    sem  = float(np.sqrt(x @ cov @ x))
    return mean, sem

def wald_contrast(res, group, cond_A, cond_B):
    """
    Contrast: (A - B) within a group using fixed effects.
    Returns dict with est, se, z, p.
    """
    beta = res.fe_params.values
    cov  = np.asarray(res.cov_params())
    p_fe = len(res.fe_params)
    cov  = cov[:p_fe, :p_fe]

    xA = _fe_design_vector(res.fe_params.index, group, cond_A)
    xB = _fe_design_vector(res.fe_params.index, group, cond_B)
    d  = xA - xB

    est = float(d @ beta)
    se  = float(np.sqrt(d @ cov @ d))
    z   = est / se if se > 0 else np.nan
    p   = 2*(1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(est=est, se=se, z=z, p=p)

def one_sided_yerr_by_sign(vals, errs):
    """
    yerr for matplotlib where:
      - if vals >= 0: show only upward error (lower=0, upper=err)
      - if vals <  0: show only downward error (lower=err, upper=0)
    Returns shape (2, N): [lower, upper]
    """
    vals = np.asarray(vals, dtype=float)
    errs = np.asarray(errs, dtype=float)
    upper = np.where(vals >= 0, errs, 0.0)
    lower = np.where(vals >= 0, 0.0, errs)
    return np.vstack([lower, upper])

# ----------------------------
# Plot:    2-panel figure
# ----------------------------

def plot_2panel_full_vs_shared(
    *,
    res_full_3level,
    res_shared_3level,
    PLOTS_DIR,
    out_name="decoding_2panel_FULL_vs_SHARED.png",
    show_title=False,                     # requested default False
    add_stars_on_contrasts=True,
    use_median=False
):
    """
    Panel A: absolute predicted median decoding errors (fixed effects; errorbars = SEM; one-sided, sign-aware)
    Panel B: key contrasts (fixed effects; errorbars = SEM; one-sided, sign-aware)
      - Within LT2 − Within LT1
      - (LT1 to LT2) − Within LT1

    Colors:
      - All neurons: black
      - Cross-registered neurons: dark grey fill + black outline

    Requested changes implemented:
      - NO legends at all
      - Panel B narrower (width roughly half of A)
      - Errorbars are SEM (model-based SE of predicted mean / contrast)
      - Errorbars one-sided and sign-aware (negative bars get downward-only)
      - Title optional (default off)
      - Subtitles match your wording
      - Group labels: Ctl / Exc / Inh
    """

    stat_label = "median" if use_median else "mean"
    out_dir = os.path.join(PLOTS_DIR, f"mixedlm_prediction_plots_{stat_label}")
    os.makedirs(out_dir, exist_ok=True)
    savefile = os.path.join(out_dir, out_name)

    # Figure layout: A (3 small axes) + B (2 stacked axes)
    # Panel B narrower so bar widths match panel A
    fig = plt.figure(figsize=(14.5, 5.4), dpi=160)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2.0, 0.55], wspace=0.28)

    gsA = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.22)
    axA = [fig.add_subplot(gsA[0, i]) for i in range(3)]

    gsB = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.28)
    axB1 = fig.add_subplot(gsB[0, 0])
    axB2 = fig.add_subplot(gsB[1, 0])

    # Bar geometry
    width = 0.34
    x0 = np.arange(len(GROUPS))
    x_full   = x0 - width/2
    x_shared = x0 + width/2

    # Styling
    col_full = "black"
    col_shared = "#666666"  # dark grey
    edge_shared = "black"

    def bar_one_sided_sem_signaware(ax, xpos, means, sems, *, color, edgecolor=None):
        """
        Sign-aware one-sided errorbars:
          - positive bars: only upward SEM
          - negative bars: only downward SEM
        """
        means = np.asarray(means, float)
        sems  = np.asarray(sems, float)

        yerr = one_sided_yerr_by_sign(means, sems)

        ax.bar(
            xpos, means, width=width,
            yerr=yerr, capsize=3,
            color=color,
            edgecolor=edgecolor,
            linewidth=1.0 if edgecolor else 0.0,
            error_kw=dict(lw=1.5, capthick=1.5)
        )

    # ----------------------------
    # Panel A: absolute predicted means (SEM)
    # ----------------------------
    for ci, cond in enumerate(CONDS):
        means_full, sem_full = [], []
        means_sh,   sem_sh   = [], []

        for g in GROUPS:
            m, s = predicted_mean_sem(res_full_3level, g, cond)
            means_full.append(m); sem_full.append(s)

            m, s = predicted_mean_sem(res_shared_3level, g, cond)
            means_sh.append(m); sem_sh.append(s)

        ax = axA[ci]

        bar_one_sided_sem_signaware(ax, x_full,   means_full, sem_full, color=col_full)
        bar_one_sided_sem_signaware(ax, x_shared, means_sh,   sem_sh,   color=col_shared, edgecolor=edge_shared)

        ax.set_title(COND_LABELS_ABS[cond], fontsize=13)
        ax.set_xticks(x0)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=12)
        if use_median:
            ax.set_ylabel("Predicted median decoding error" if ci == 0 else "", fontsize=12)
        else:
            ax.set_ylabel("Predicted mean decoding error" if ci == 0 else "", fontsize=12)
        ax.axhline(0, linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis='y', labelsize=11)

    # Panel labels
    fig.text(0.01, 0.98, "A", fontsize=16, fontweight="bold", va="top")
    fig.text(0.78, 0.98, "B", fontsize=16, fontweight="bold", va="top")
    def contrast_panel(ax, condA, condB, title):
        est_full, se_full, p_full = [], [], []
        est_sh,   se_sh,   p_sh   = [], [], []

        for g in GROUPS:
            c = wald_contrast(res_full_3level, g, condA, condB)
            est_full.append(c["est"]); se_full.append(c["se"]); p_full.append(c["p"])

            c = wald_contrast(res_shared_3level, g, condA, condB)
            est_sh.append(c["est"]); se_sh.append(c["se"]); p_sh.append(c["p"])

        bar_one_sided_sem_signaware(ax, x_full,   est_full, se_full, color=col_full)
        bar_one_sided_sem_signaware(ax, x_shared, est_sh,   se_sh,   color=col_shared, edgecolor=edge_shared)

        ax.set_title(title, fontsize=13)
        ax.set_xticks(x0)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=12)
        ax.set_ylabel("Contrast vs Within LT1", fontsize=12)
        ax.axhline(0, linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis='y', labelsize=11)

        # Stars: for positive bars place above; for negative bars place below
        if add_stars_on_contrasts:
            y_min, y_max = ax.get_ylim()
            rng = (y_max - y_min) if (y_max > y_min) else 1.0
            bump = 0.03 * rng

            def place_star(x, val, se, p):
                s = _star(p)
                if not s:
                    return
                is_pval = s.startswith("p=")
                fs = 9 if is_pval else 14
                if val >= 0:
                    y = val + se + bump
                    va = "bottom"
                else:
                    y = val - se - bump
                    va = "top"
                ax.text(x, y, s, ha="center", va=va, fontsize=fs, color="black")

            for i, g in enumerate(GROUPS):
                place_star(x_full[i],   est_full[i], se_full[i], p_full[i])
                place_star(x_shared[i], est_sh[i],   se_sh[i],   p_sh[i])

            # Expand limits so stars don't clip
            _pad_axes_for_stars(ax)

        return dict(full=dict(est=est_full, p=p_full), shared=dict(est=est_sh, p=p_sh))

    outB1 = contrast_panel(
        axB1,
        condA="within_LT2",
        condB="within_LT1",
        title="Within LT2 − Within LT1",
    )
    outB2 = contrast_panel(
        axB2,
        condA="LT1_to_LT2",
        condB="within_LT1",
        title="(LT1 to LT2) − Within LT1",
    )

    if show_title:
        fig.suptitle(
            "Linear track decoding errors in all neurons versus cross-registered neurons",
            fontsize=12,
            y=1.03
        )

    plt.tight_layout()
    fig.savefig(savefile, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {savefile}")
    return {"B1": outB1, "B2": outB2, "savefile": savefile}


# ----------------------------
# Plot:    within-only figure (LT1 vs LT2 only)
# ----------------------------

def plot_2panel_within_only(
    *,
    res_full_within,
    res_shared_within,
    PLOTS_DIR,
    out_name="decoding_2panel_WITHIN_ONLY.png",
    show_title=False,
    add_stars_on_contrasts=True,
    use_median=False
):
    """
    Like plot_2panel_full_vs_shared but uses the WITHIN-ONLY models
    (fitted on within_LT1 + within_LT2 only, no LT1→LT2).

    Panel A: absolute predicted decoding errors for within_LT1 and within_LT2 (2 subplots)
    Panel B: single contrast (Within LT2 − Within LT1)

    Colors:
      - All neurons: black
      - Cross-registered neurons: dark grey fill + black outline
    """

    stat_label = "median" if use_median else "mean"
    out_dir = os.path.join(PLOTS_DIR, f"mixedlm_prediction_plots_{stat_label}")
    os.makedirs(out_dir, exist_ok=True)
    savefile = os.path.join(out_dir, out_name)

    WITHIN_CONDS = ["within_LT1", "within_LT2"]
    WITHIN_COND_LABELS = {
        "within_LT1": "Within LT1",
        "within_LT2": "Within LT2",
    }

    stat_label = "median" if use_median else "mean"

    # Figure layout: A (2 subplots) + B (1 contrast)
    # Panel B narrower so bar widths match panel A
    fig = plt.figure(figsize=(11.0, 5.4), dpi=160)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2.0, 0.55], wspace=0.30)

    gsA = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.22)
    axA = [fig.add_subplot(gsA[0, i]) for i in range(2)]

    axB = fig.add_subplot(outer[1])

    # Bar geometry
    width = 0.34
    x0 = np.arange(len(GROUPS))
    x_full   = x0 - width/2
    x_shared = x0 + width/2

    # Styling
    col_full = "black"
    col_shared = "#666666"
    edge_shared = "black"

    def bar_one_sided_sem_signaware(ax, xpos, means, sems, *, color, edgecolor=None):
        means = np.asarray(means, float)
        sems  = np.asarray(sems, float)
        yerr = one_sided_yerr_by_sign(means, sems)
        ax.bar(
            xpos, means, width=width,
            yerr=yerr, capsize=3,
            color=color,
            edgecolor=edgecolor,
            linewidth=1.0 if edgecolor else 0.0,
            error_kw=dict(lw=1.5, capthick=1.5)
        )

    # ----------------------------
    # Panel A: absolute predicted means (within_LT1, within_LT2)
    # ----------------------------
    for ci, cond in enumerate(WITHIN_CONDS):
        means_full, sem_full = [], []
        means_sh,   sem_sh   = [], []

        for g in GROUPS:
            m, s = predicted_mean_sem(res_full_within, g, cond)
            means_full.append(m); sem_full.append(s)

            m, s = predicted_mean_sem(res_shared_within, g, cond)
            means_sh.append(m); sem_sh.append(s)

        ax = axA[ci]

        bar_one_sided_sem_signaware(ax, x_full,   means_full, sem_full, color=col_full)
        bar_one_sided_sem_signaware(ax, x_shared, means_sh,   sem_sh,   color=col_shared, edgecolor=edge_shared)

        ax.set_title(WITHIN_COND_LABELS[cond], fontsize=13)
        ax.set_xticks(x0)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=12)
        ax.set_ylabel(f"Predicted {stat_label} decoding error" if ci == 0 else "", fontsize=12)
        ax.axhline(0, linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis='y', labelsize=11)

    # Panel labels
    fig.text(0.01, 0.98, "A", fontsize=16, fontweight="bold", va="top")
    fig.text(0.78, 0.98, "B", fontsize=16, fontweight="bold", va="top")

    # ----------------------------
    # Panel B: single contrast (Within LT2 − Within LT1)
    # ----------------------------
    est_full, se_full, p_full = [], [], []
    est_sh,   se_sh,   p_sh   = [], [], []

    for g in GROUPS:
        c = wald_contrast(res_full_within, g, "within_LT2", "within_LT1")
        est_full.append(c["est"]); se_full.append(c["se"]); p_full.append(c["p"])

        c = wald_contrast(res_shared_within, g, "within_LT2", "within_LT1")
        est_sh.append(c["est"]); se_sh.append(c["se"]); p_sh.append(c["p"])

    bar_one_sided_sem_signaware(axB, x_full,   est_full, se_full, color=col_full)
    bar_one_sided_sem_signaware(axB, x_shared, est_sh,   se_sh,   color=col_shared, edgecolor=edge_shared)

    axB.set_title("Within LT2 \u2212 Within LT1", fontsize=13)
    axB.set_xticks(x0)
    axB.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=12)
    axB.set_ylabel("Contrast vs Within LT1", fontsize=12)
    axB.axhline(0, linewidth=0.8)
    axB.grid(True, axis="y", alpha=0.25)
    axB.tick_params(axis='y', labelsize=11)

    if add_stars_on_contrasts:
        y_min, y_max = axB.get_ylim()
        rng = (y_max - y_min) if (y_max > y_min) else 1.0
        bump = 0.03 * rng

        def place_star(x, val, se, p):
            s = _star(p)
            if not s:
                return
            is_pval = s.startswith("p=")
            fs = 9 if is_pval else 14
            if val >= 0:
                y = val + se + bump
                va = "bottom"
            else:
                y = val - se - bump
                va = "top"
            axB.text(x, y, s, ha="center", va=va, fontsize=fs, color="black")

        for i, g in enumerate(GROUPS):
            place_star(x_full[i],   est_full[i], se_full[i], p_full[i])
            place_star(x_shared[i], est_sh[i],   se_sh[i],   p_sh[i])

        # Expand limits so stars don't clip
        _pad_axes_for_stars(axB)

    if show_title:
        fig.suptitle(
            "Within-only model: LT1 vs LT2 decoding errors",
            fontsize=12,
            y=1.03
        )

    plt.tight_layout()
    fig.savefig(savefile, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {savefile}")
    return {
        "contrast": dict(full=dict(est=est_full, p=p_full), shared=dict(est=est_sh, p=p_sh)),
        "savefile": savefile,
    }



# -------------------------------------------------------------------
# Spatial decoding-error profile: mean |error| vs normalised track pos
# -------------------------------------------------------------------

def _collect_spatial_error_per_mouse(
    dec_results: dict,
    mouse_groups: dict,
    *,
    sess: str = "LT1",
    n_pos_bins: int = 20,
    kmin: float = 0.0,
    groups=("mCherry", "hM3D", "hM4D"),
):
    """
    For every mouse, collect per-direction (true_pos_norm, abs_err)
    from both CV folds, normalise position to [0, 1], and bin.

    sess can be "LT1", "LT2", or "LT1_to_LT2".
    For "LT1_to_LT2" (cross-session), there is no direction split
    so only a "combined" profile is produced per mouse.

    Returns
    -------
    mouse_profiles : dict[mouse -> dict]
        mouse_profiles[mouse]["leftward"]  = {"bin_centers": (B,), "mean_err": (B,), "n": (B,)}
        mouse_profiles[mouse]["rightward"] = {"bin_centers": (B,), "mean_err": (B,), "n": (B,)}
        mouse_profiles[mouse]["combined"]  = {"bin_centers": (B,), "mean_err": (B,), "n": (B,)}
        mouse_profiles[mouse]["group"] = str
    """
    bin_edges = np.linspace(0.0, 1.0, n_pos_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mouse_profiles = {}

    is_xfer = (sess == "LT1_to_LT2")

    for mouse, res in dec_results.items():
        g = mouse_groups.get(mouse, "NA")
        if g not in groups:
            continue

        if is_xfer:
            # ---- cross-session LT1→LT2 (flat, no direction split) ----
            blk = res.get("LT1_to_LT2", {})
            if not blk.get("ok", False):
                continue
            out = blk.get("out")
            if not isinstance(out, dict):
                continue
            tp = np.asarray(out.get("true_pos", []), float)
            dp = np.asarray(out.get("decoded_pos", []), float)
            Ksum = np.asarray(out.get("Ksum", []), float)
            ae = np.abs(dp - tp)

            n = min(tp.size, dp.size, Ksum.size)
            if n == 0:
                continue
            tp, dp, ae, Ksum = tp[:n], dp[:n], ae[:n], Ksum[:n]
            m = np.isfinite(tp) & np.isfinite(dp)
            if kmin > 0:
                m = m & (Ksum >= kmin)
            tp_f, ae_f = tp[m], ae[m]
            if tp_f.size == 0:
                continue

            pmin, pmax = tp_f.min(), tp_f.max()
            if pmax - pmin < 1e-6:
                continue
            tp_norm = (tp_f - pmin) / (pmax - pmin)
            err_norm = ae_f / (pmax - pmin)

            mean_err = np.full(n_pos_bins, np.nan)
            counts   = np.zeros(n_pos_bins, int)
            bin_idx  = np.clip(np.digitize(tp_norm, bin_edges) - 1, 0, n_pos_bins - 1)
            for b in range(n_pos_bins):
                sel = bin_idx == b
                counts[b] = sel.sum()
                if counts[b] > 0:
                    mean_err[b] = np.nanmean(err_norm[sel])

            mouse_profiles[mouse] = {
                "group": g,
                "combined": {"bin_centers": bin_centers.copy(), "mean_err": mean_err, "n": counts},
            }
            continue

        # ---- within-session (LT1 / LT2) ----
        # find the within-session dict
        within = None
        for kk in (f"within_{sess}_by_dir",
                   f"within_{sess}_by_dir_full",
                   f"within_{sess}_by_dir_mapped"):
            d = res.get(kk)
            if isinstance(d, dict) and len(d) > 0:
                within = d
                break
        if within is None:
            continue

        dir_data = {}   # dir_label -> (true_norm, err)

        for dir_label in ("leftward", "rightward"):
            dblk = within.get(dir_label)
            if not isinstance(dblk, dict) or not dblk.get("ok", False):
                continue

            tp_all, err_all = [], []
            for fold_key in ("train_to_test", "test_to_train"):
                step = dblk.get(fold_key)
                if not isinstance(step, dict):
                    continue
                out = step.get("out")
                if not isinstance(out, dict):
                    continue
                tp = np.asarray(out.get("true_pos", []), float)
                dp = np.asarray(out.get("decoded_pos", []), float)
                Ksum = np.asarray(out.get("Ksum", []), float)
                ae = np.abs(dp - tp)

                n = min(tp.size, dp.size, Ksum.size)
                if n == 0:
                    continue
                tp, dp, ae, Ksum = tp[:n], dp[:n], ae[:n], Ksum[:n]
                m = np.isfinite(tp) & np.isfinite(dp)
                if kmin > 0:
                    m = m & (Ksum >= kmin)
                tp_all.append(tp[m])
                err_all.append(ae[m])

            if not tp_all:
                continue

            tp_cat = np.concatenate(tp_all)
            err_cat = np.concatenate(err_all)

            # normalise position to [0,1]
            pmin, pmax = tp_cat.min(), tp_cat.max()
            if pmax - pmin < 1e-6:
                continue
            tp_norm = (tp_cat - pmin) / (pmax - pmin)
            # normalise error the same way (in fraction-of-track units)
            err_norm = err_cat / (pmax - pmin)

            dir_data[dir_label] = (tp_norm, err_norm)

        if not dir_data:
            continue

        prof = {"group": g}
        # per-direction binned profiles
        for dl, (tp_norm, err_norm) in dir_data.items():
            mean_err = np.full(n_pos_bins, np.nan)
            counts   = np.zeros(n_pos_bins, int)
            bin_idx  = np.digitize(tp_norm, bin_edges) - 1
            bin_idx  = np.clip(bin_idx, 0, n_pos_bins - 1)
            for b in range(n_pos_bins):
                sel = bin_idx == b
                counts[b] = sel.sum()
                if counts[b] > 0:
                    mean_err[b] = np.nanmean(err_norm[sel])
            prof[dl] = {"bin_centers": bin_centers.copy(), "mean_err": mean_err, "n": counts}

        # combined: merge both directions
        all_tp = np.concatenate([d[0] for d in dir_data.values()])
        all_err = np.concatenate([d[1] for d in dir_data.values()])
        mean_err_c = np.full(n_pos_bins, np.nan)
        counts_c   = np.zeros(n_pos_bins, int)
        bin_idx_c  = np.clip(np.digitize(all_tp, bin_edges) - 1, 0, n_pos_bins - 1)
        for b in range(n_pos_bins):
            sel = bin_idx_c == b
            counts_c[b] = sel.sum()
            if counts_c[b] > 0:
                mean_err_c[b] = np.nanmean(all_err[sel])
        prof["combined"] = {"bin_centers": bin_centers.copy(), "mean_err": mean_err_c, "n": counts_c}

        mouse_profiles[mouse] = prof

    return mouse_profiles


def plot_spatial_decoding_error(
    dec_results: dict,
    mouse_groups: dict,
    *,
    save_dir: str,
    session_str: str = "",
    n_pos_bins: int = 20,
    kmin: float = 0.0,
    auto_close: bool = True,
    groups=("mCherry", "hM3D", "hM4D"),
    colors=None,
    sessions=("LT1", "LT2"),
):
    """
    Plot mean decoding error vs normalised track position.

    Produces:
      1) Per-mouse plots: leftward & rightward (one figure per mouse per session)
         For LT1_to_LT2: single combined bar chart per mouse (no direction split).
      2) Group-average plots: leftward, rightward, and combined
         (one figure per session, with one line per group)
         For LT1_to_LT2: combined-only group plot.
    """
    _set_compact_plot_style()
    _ensure_dir(save_dir)

    if colors is None:
        colors = {"hM3D": "red", "hM4D": "blue", "mCherry": "black"}

    group_labels = {"hM3D": "hM3D (Gq)", "hM4D": "hM4D (Gi)", "mCherry": "mCherry"}

    for sess in sessions:
        mp = _collect_spatial_error_per_mouse(
            dec_results, mouse_groups,
            sess=sess, n_pos_bins=n_pos_bins, kmin=kmin, groups=groups,
        )
        if not mp:
            print(f"[WARN] No spatial-error data for {sess}")
            continue

        is_xfer = (sess == "LT1_to_LT2")

        # ============================================================
        # 1) Per-mouse plots
        # ============================================================
        per_mouse_dir = os.path.join(save_dir, "per_mouse")
        _ensure_dir(per_mouse_dir)

        for mouse, prof in mp.items():
            g = prof["group"]

            if is_xfer:
                # Cross-session: single combined bar chart
                fig, ax = plt.subplots(figsize=(6, 3.5))
                if "combined" in prof:
                    bc = prof["combined"]["bin_centers"]
                    me = prof["combined"]["mean_err"]
                    ax.bar(bc, me, width=bc[1] - bc[0], color=colors.get(g, "gray"),
                           alpha=0.7, edgecolor="k", linewidth=0.4)
                ax.set_xlabel("Normalised track position")
                ax.set_ylabel("Mean |error| (frac of track)")
                ax.set_xlim(0, 1)
                fig.suptitle(f"{mouse} ({g})  —  LT1\u2192LT2  {session_str}", fontsize=12)
                plt.tight_layout()
                fpath = os.path.join(per_mouse_dir, f"spatial_err_{mouse}_{sess}_{session_str}.png")
                _savefig_tight(fig, fpath)
                if auto_close:
                    plt.close(fig)
                print(f"[SAVED] {fpath}")
            else:
                # Within-session: leftward + rightward side-by-side
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
                for ax_i, dl in enumerate(("leftward", "rightward")):
                    ax = axes[ax_i]
                    if dl in prof:
                        bc = prof[dl]["bin_centers"]
                        me = prof[dl]["mean_err"]
                        ax.bar(bc, me, width=bc[1] - bc[0], color=colors.get(g, "gray"),
                               alpha=0.7, edgecolor="k", linewidth=0.4)
                        ax.set_title(f"{dl.capitalize()}")
                    else:
                        ax.set_title(f"{dl.capitalize()} (no data)")
                    ax.set_xlabel("Normalised track position")
                    ax.set_xlim(0, 1)
                axes[0].set_ylabel("Mean |error| (frac of track)")
                fig.suptitle(f"{mouse} ({g})  —  {sess}  {session_str}", fontsize=12)
                plt.tight_layout()
                fpath = os.path.join(per_mouse_dir, f"spatial_err_{mouse}_{sess}_{session_str}.png")
                _savefig_tight(fig, fpath)
                if auto_close:
                    plt.close(fig)
                print(f"[SAVED] {fpath}")

        # ============================================================
        # 2) Group averages: leftward, rightward, combined
        #    (LT1_to_LT2 → combined only)
        # ============================================================
        modes = ("combined",) if is_xfer else ("leftward", "rightward", "combined")
        sess_display = "LT1\u2192LT2" if is_xfer else sess

        for mode in modes:
            fig, ax = plt.subplots(figsize=(6, 3.5))

            for g in groups:
                mice_in_group = [m for m, p in mp.items() if p["group"] == g and mode in p]
                if not mice_in_group:
                    continue
                # stack profiles → (n_mice, n_pos_bins)
                stacked = np.stack([mp[m][mode]["mean_err"] for m in mice_in_group])
                group_mean = np.nanmean(stacked, axis=0)
                group_sem  = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(len(mice_in_group))
                bc = mp[mice_in_group[0]][mode]["bin_centers"]

                ax.plot(bc, group_mean, color=colors.get(g, "gray"), linewidth=1.5,
                        label=f"{group_labels.get(g, g)} (n={len(mice_in_group)})")
                ax.fill_between(bc, group_mean - group_sem, group_mean + group_sem,
                                color=colors.get(g, "gray"), alpha=0.2)

            ax.set_xlabel("Normalised track position")
            ax.set_ylabel("Mean |error| (frac of track)")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9)
            kmin_str = f"_Kmin{kmin:.0f}" if kmin > 0 else ""
            ax.set_title(f"Group avg decoding error — {mode} — {sess_display}  {session_str}{kmin_str}")
            plt.tight_layout()
            fpath = os.path.join(save_dir, f"group_spatial_err_{mode}_{sess}_{session_str}{kmin_str}.png")
            _savefig_tight(fig, fpath)
            if auto_close:
                plt.close(fig)
            print(f"[SAVED] {fpath}")


def plot_spatial_decoding_error_heatmap(
    dec_results: dict,
    mouse_groups: dict,
    *,
    save_dir: str,
    session_str: str = "",
    n_pos_bins: int = 20,
    kmin: float = 0.0,
    auto_close: bool = True,
    groups=("mCherry", "hM3D", "hM4D"),
    cmap: str = "inferno",
    sessions=("LT1", "LT2"),
):
    """
    1-D heatmap-strip version of plot_spatial_decoding_error.

    For each session × mode (leftward / rightward / combined):
      - Per-mouse figure: one colour strip per mouse (rows grouped by group)
      - Group-average figure: one strip per group (mean across mice)

    Colour encodes group-average mean |error| (fraction of track).
    """
    _set_compact_plot_style()
    _ensure_dir(save_dir)

    group_labels = {"hM3D": "hM3D (Gq)", "hM4D": "hM4D (Gi)", "mCherry": "mCherry"}

    for sess in sessions:
        mp = _collect_spatial_error_per_mouse(
            dec_results, mouse_groups,
            sess=sess, n_pos_bins=n_pos_bins, kmin=kmin, groups=groups,
        )
        if not mp:
            print(f"[WARN] No spatial-error heatmap data for {sess}")
            continue

        is_xfer = (sess == "LT1_to_LT2")
        modes = ("combined",) if is_xfer else ("leftward", "rightward", "combined")
        sess_display = "LT1\u2192LT2" if is_xfer else sess
        kmin_str = f"_Kmin{kmin:.0f}" if kmin > 0 else ""

        # ----- determine global vmin/vmax across all groups & modes -----
        all_vals = []
        for prof in mp.values():
            for mode in modes:
                if mode in prof:
                    all_vals.append(prof[mode]["mean_err"])
        if not all_vals:
            continue
        vmin = np.nanmin(np.concatenate(all_vals))
        vmax = np.nanmax(np.concatenate(all_vals))

        # =================================================================
        # 1) Per-mouse heatmap strips (one figure per mode)
        # =================================================================
        per_mouse_dir = os.path.join(save_dir, "per_mouse")
        _ensure_dir(per_mouse_dir)

        for mode in modes:
            # collect rows: list of (mouse_label, mean_err array)
            rows = []
            for g in groups:
                mice_g = sorted([m for m, p in mp.items()
                                 if p["group"] == g and mode in p])
                for m in mice_g:
                    rows.append((f"{m} ({g})", mp[m][mode]["mean_err"]))
            if not rows:
                continue

            n_rows = len(rows)
            fig_h = max(1.5, 0.45 * n_rows + 0.8)
            fig, ax = plt.subplots(figsize=(8, fig_h))

            data = np.stack([r[1] for r in rows])           # (n_rows, n_pos_bins)
            im = ax.imshow(data, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[0, 1, n_rows - 0.5, -0.5])
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([r[0] for r in rows], fontsize=8)
            ax.set_xlabel("Normalised track position")
            ax.set_title(f"Per-mouse error — {mode} — {sess_display}  {session_str}{kmin_str}",
                         fontsize=11)
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label("Mean |error| (frac of track)", fontsize=9)
            plt.tight_layout()
            fpath = os.path.join(per_mouse_dir,
                                 f"spatial_err_heatmap_{mode}_{sess}_{session_str}{kmin_str}.png")
            _savefig_tight(fig, fpath)
            if auto_close:
                plt.close(fig)
            print(f"[SAVED] {fpath}")

        # =================================================================
        # 2) Group-average heatmap strips (one figure per mode)
        # =================================================================
        for mode in modes:
            strips = []       # (label, mean_err_1d)
            for g in groups:
                mice_g = [m for m, p in mp.items()
                          if p["group"] == g and mode in p]
                if not mice_g:
                    continue
                stacked = np.stack([mp[m][mode]["mean_err"] for m in mice_g])
                group_mean = np.nanmean(stacked, axis=0)
                strips.append((f"{group_labels.get(g, g)} (n={len(mice_g)})", group_mean))
            if not strips:
                continue

            n_strips = len(strips)
            fig_h = max(1.2, 0.6 * n_strips + 0.8)
            fig, ax = plt.subplots(figsize=(8, fig_h))

            data = np.stack([s[1] for s in strips])
            im = ax.imshow(data, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[0, 1, n_strips - 0.5, -0.5])
            ax.set_yticks(range(n_strips))
            ax.set_yticklabels([s[0] for s in strips], fontsize=10)
            ax.set_xlabel("Normalised track position")
            ax.set_title(f"Group avg error — {mode} — {sess_display}  {session_str}{kmin_str}",
                         fontsize=11)
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label("Mean |error| (frac of track)", fontsize=9)
            plt.tight_layout()
            fpath = os.path.join(save_dir,
                                 f"group_spatial_err_heatmap_{mode}_{sess}_{session_str}{kmin_str}.png")
            _savefig_tight(fig, fpath)
            if auto_close:
                plt.close(fig)
            print(f"[SAVED] {fpath}")


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

def build_stacked_df(df_full, df_shared):
    df_full = df_full.copy()
    df_shared = df_shared.copy()
    df_full["neur_set"] = "all"
    df_shared["neur_set"] = "shared"
    df = pd.concat([df_full, df_shared], axis=0, ignore_index=True)

    # enforce categorical + reference levels if you want
    df["mouse"] = df["mouse"].astype(str)
    df["group"] = pd.Categorical(df["group"].astype(str), categories=["mCherry", "hM3D", "hM4D"], ordered=False)
    df["condition"] = pd.Categorical(df["condition"].astype(str),
                                     categories=["within_LT1", "within_LT2", "LT1_to_LT2"],
                                     ordered=True)
    df["neur_set"] = pd.Categorical(df["neur_set"].astype(str), categories=["all", "shared"], ordered=True)
    return df


def fit_stacked_mixedlm(df, error_col):
    formula = f"{error_col} ~ group * condition * neur_set"
    model = smf.mixedlm(formula, df, groups=df["mouse"])
    res = model.fit(reml=False)
    return res


def _fe_design_vector(param_index, group, condition, neur_set):
    """
    Fixed-effects design vector aligned to res.fe_params.index for:
      y ~ group * condition * neur_set
    Reference levels:
      group=mCherry, condition=within_LT1, neur_set=all
    """
    idx = list(param_index)
    x = np.zeros(len(idx), dtype=float)

    def set_if(name):
        if name in idx:
            x[idx.index(name)] = 1.0

    set_if("Intercept")

    # main effects
    if group == "hM3D":
        set_if("group[T.hM3D]")
    elif group == "hM4D":
        set_if("group[T.hM4D]")

    if condition == "within_LT2":
        set_if("condition[T.within_LT2]")
    elif condition == "LT1_to_LT2":
        set_if("condition[T.LT1_to_LT2]")

    if neur_set == "shared":
        set_if("neur_set[T.shared]")

    # 2-way interactions
    if group == "hM3D" and condition == "within_LT2":
        set_if("group[T.hM3D]:condition[T.within_LT2]")
    if group == "hM4D" and condition == "within_LT2":
        set_if("group[T.hM4D]:condition[T.within_LT2]")
    if group == "hM3D" and condition == "LT1_to_LT2":
        set_if("group[T.hM3D]:condition[T.LT1_to_LT2]")
    if group == "hM4D" and condition == "LT1_to_LT2":
        set_if("group[T.hM4D]:condition[T.LT1_to_LT2]")

    if group == "hM3D" and neur_set == "shared":
        set_if("group[T.hM3D]:neur_set[T.shared]")
    if group == "hM4D" and neur_set == "shared":
        set_if("group[T.hM4D]:neur_set[T.shared]")

    if condition == "within_LT2" and neur_set == "shared":
        set_if("condition[T.within_LT2]:neur_set[T.shared]")
    if condition == "LT1_to_LT2" and neur_set == "shared":
        set_if("condition[T.LT1_to_LT2]:neur_set[T.shared]")

    # 3-way interactions
    if group == "hM3D" and condition == "within_LT2" and neur_set == "shared":
        set_if("group[T.hM3D]:condition[T.within_LT2]:neur_set[T.shared]")
    if group == "hM4D" and condition == "within_LT2" and neur_set == "shared":
        set_if("group[T.hM4D]:condition[T.within_LT2]:neur_set[T.shared]")
    if group == "hM3D" and condition == "LT1_to_LT2" and neur_set == "shared":
        set_if("group[T.hM3D]:condition[T.LT1_to_LT2]:neur_set[T.shared]")
    if group == "hM4D" and condition == "LT1_to_LT2" and neur_set == "shared":
        set_if("group[T.hM4D]:condition[T.LT1_to_LT2]:neur_set[T.shared]")

    return x


def omnibus_group_test(res, condition, neur_set):
    """
    2-df omnibus Wald test within a (condition × neur_set) cell:
      H0: Ctl = Exc = Inh
    Implemented as joint test of:
      (hM3D - mCherry) == 0 and (hM4D - mCherry) == 0
    """
    params = res.fe_params
    p_fe = len(params)

    # CRITICAL FIX: slice covariance to fixed-effects block only
    V_full = np.asarray(res.cov_params())
    V = V_full[:p_fe, :p_fe]
    beta = params.values

    x_ctl = _fe_design_vector(params.index, "mCherry", condition, neur_set)
    x_exc = _fe_design_vector(params.index, "hM3D",   condition, neur_set)
    x_inh = _fe_design_vector(params.index, "hM4D",   condition, neur_set)

    C = np.vstack([x_exc - x_ctl, x_inh - x_ctl])  # (2 x p_fe)

    diff = C @ beta                       # (2,)
    W = C @ V @ C.T                       # (2 x 2)

    stat = float(diff.T @ np.linalg.inv(W) @ diff)
    pval = float(1.0 - chi2.cdf(stat, df=2))
    return stat, pval


def run_six_omnibus_tests(res):
    cells = [
        ("within_LT1", "all"),
        ("within_LT1", "shared"),
        ("within_LT2", "all"),
        ("within_LT2", "shared"),
        ("LT1_to_LT2", "all"),
        ("LT1_to_LT2", "shared"),
    ]
    out = []
    for cond, ns in cells:
        stat, p = omnibus_group_test(res, cond, ns)
        out.append({"condition": cond, "neur_set": ns, "chi2_df2": stat, "p": p})
    return pd.DataFrame(out)



print("loaded")

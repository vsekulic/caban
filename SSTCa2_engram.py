"""SSTCa2_engram — unified engram-cell identity store.

Single source of truth for engram-cell classification. Computed ONCE
after all sessions are loaded; consumed by:
  * the per-session binned-rate engram panels (SSTCa2_main.py),
  * the Population PCA pipeline (SSTCa2_population.py),
  * the engram sanity plots (SSTCa2_engram_sanity.py).

Conventions
-----------
* Reference sessions: only TFC_cond (etype='encoding') and Test_B
  (etype='recall') are classified. Other panels project the reference
  ensemble through crossreg.
* Per-cell statistic: average transient rate (Hz) on the FULL session
  S matrix, using SSTCa2_utilities.find_spikes_ca_S with the per-session
  ``thres`` (canonically 2). Mocle et al. 2024 / Frankland convention.
* Three modes:
    - 'permouse'     : per-mouse z-score; engram iff z > 0.
    - 'ctlthresh_z'  : z-score using mu/sigma pooled over mCherry mice;
                       engram iff z > 0.
    - 'ctlthresh_p50': absolute cutoff = 50th percentile of mCherry pool;
                       engram iff rate > cutoff.
* Pools are per-etype: encoding pool = mCherry full-session TFC_cond
  rates, recall pool = mCherry full-session Test_B rates.

Public API
----------
* ``compute_full_session_rate(session)``       — ndarray (n_cells,)
* ``build_control_pool(...)``                  — ext_norm tuple
* ``classify_engram_full_session(...)``        — bool ndarray
* ``build_engram_identity(...)``               — full nested store
* ``project_engram_to_session(...)``           — crossreg projection
"""

import numpy as np
import scipy.stats

from SSTCa2_utilities import (
    MINISCOPE_FPS,
    get_actual_cells_from_df_session,
)


ENGRAM_MODES = ('permouse', 'ctlthresh_z', 'ctlthresh_p50')
ENGRAM_REFERENCE = {'encoding': 'TFC_cond', 'recall': 'Test_B'}
DEFAULT_CTL_GROUP = 'mCherry'
DEFAULT_THRESHOLD = 0.0  # z-score threshold used by 'permouse' / 'ctlthresh_z'


# ---------------------------------------------------------------------------
# Per-cell statistic
# ---------------------------------------------------------------------------

def compute_full_session_rate(session):
    """Per-cell average transient rate (Hz) over a session's full S matrix.

    Uses the cell's already-detected spike-frame dict
    (``session.S_spikes``) so this is consistent with the canonical
    per-session classifier (which uses ``find_spikes_ca_S(self.S,
    self.thres)`` at session-load time).
    """
    n_cells, n_frames = session.S.shape
    if n_frames <= 0:
        raise RuntimeError(
            f"Session {session.mouse} {session.session_type} has zero frames."
        )
    duration_s = n_frames / float(MINISCOPE_FPS)
    rates = np.zeros(n_cells, dtype=float)
    for i in range(n_cells):
        if i not in session.S_spikes:
            raise RuntimeError(
                f"Session {session.mouse} {session.session_type}: "
                f"S_spikes missing entry for cell row {i}."
            )
        rates[i] = len(session.S_spikes[i]) / duration_s
    return rates


# ---------------------------------------------------------------------------
# Control pool -> ext_norm spec
# ---------------------------------------------------------------------------

def build_control_pool(rates_by_mouse, ctl_mice, mode, percentile=50.0):
    """Pool per-cell rates of *ctl_mice* and produce an ext_norm spec.

    Parameters
    ----------
    rates_by_mouse : dict[str, ndarray]
        Per-mouse full-session transient rates.
    ctl_mice : iterable[str]
        Control-group mice whose cells form the pool.
    mode : {'zscore', 'percentile'}
    percentile : float
        Used only when ``mode == 'percentile'``.

    Returns
    -------
    spec : tuple
        ('zscore', mu, sigma) or ('absolute', cutoff).
    """
    pooled = []
    for m in ctl_mice:
        if m not in rates_by_mouse:
            raise KeyError(
                f"build_control_pool: control mouse {m!r} missing from "
                f"rates_by_mouse (have {list(rates_by_mouse)})."
            )
        pooled.append(np.asarray(rates_by_mouse[m], dtype=float).ravel())
    if not pooled:
        raise RuntimeError("build_control_pool: empty control mouse list.")
    pooled = np.concatenate(pooled)
    if pooled.size == 0:
        raise RuntimeError("build_control_pool: pooled rate distribution is empty.")
    if mode == 'zscore':
        mu = float(np.mean(pooled))
        sigma = float(np.std(pooled, ddof=0))
        if sigma <= 0:
            raise RuntimeError(
                f"build_control_pool: pooled SD is zero (n={pooled.size})."
            )
        return ('zscore', mu, sigma)
    if mode in ('percentile', 'absolute'):
        cutoff = float(np.percentile(pooled, percentile))
        return ('absolute', cutoff)
    raise ValueError(f"build_control_pool: unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Per-mode classification of a single mouse
# ---------------------------------------------------------------------------

def classify_engram_full_session(rate, mode, ext_norm=None,
                                 threshold=DEFAULT_THRESHOLD):
    """Boolean mask (length == len(rate)) marking engram cells.

    Modes:
      - 'permouse'     : z-score within this mouse, mask = (z > threshold).
      - 'ctlthresh_z'  : require ext_norm=('zscore', mu, sigma);
                         mask = ((rate - mu)/sigma > threshold).
      - 'ctlthresh_p50': require ext_norm=('absolute', cutoff);
                         mask = (rate > cutoff).
    """
    rate = np.asarray(rate, dtype=float)
    if mode == 'permouse':
        mu = float(np.mean(rate))
        sigma = float(np.std(rate, ddof=0))
        if sigma <= 0:
            raise RuntimeError(
                f"classify_engram_full_session permouse: zero SD (n={rate.size})."
            )
        score = (rate - mu) / sigma
        return score > threshold
    if mode == 'ctlthresh_z':
        if ext_norm is None or ext_norm[0] != 'zscore':
            raise ValueError(
                f"ctlthresh_z requires ext_norm=('zscore', mu, sigma); got {ext_norm!r}"
            )
        _, mu, sigma = ext_norm
        if float(sigma) <= 0:
            raise RuntimeError(
                f"ctlthresh_z: ext_norm sigma must be > 0 (got {sigma})."
            )
        score = (rate - float(mu)) / float(sigma)
        return score > threshold
    if mode == 'ctlthresh_p50':
        if ext_norm is None or ext_norm[0] != 'absolute':
            raise ValueError(
                f"ctlthresh_p50 requires ext_norm=('absolute', cutoff); got {ext_norm!r}"
            )
        _, cutoff = ext_norm
        return rate > float(cutoff)
    raise ValueError(f"classify_engram_full_session: unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Driver: build the full identity store
# ---------------------------------------------------------------------------

def build_engram_identity(ref_sessions_by_etype, mouse_groups,
                          ctl_group=DEFAULT_CTL_GROUP,
                          modes=ENGRAM_MODES,
                          threshold=DEFAULT_THRESHOLD):
    """Compute engram identity for every (mouse, etype, mode).

    Parameters
    ----------
    ref_sessions_by_etype : dict[etype, dict[mouse, BehaviourSession]]
        e.g. ``{'encoding': TFC_cond, 'recall': Test_B}``.
    mouse_groups : dict[mouse, group_label]
        Group labels (e.g. {'G05': 'hM3D', 'G06': 'mCherry', ...}).
    ctl_group : str
        Group label whose mice form the pool for ctlthresh_* modes.
    modes : iterable[str]
        Subset of ENGRAM_MODES to compute.
    threshold : float
        Z-score threshold (used by permouse / ctlthresh_z).

    Returns
    -------
    engram_id : dict[mouse, dict[etype, dict[mode, np.ndarray bool]]]
        Bool masks over each mouse's full reference-session S rows.
    rates_by_etype_mouse : dict[etype, dict[mouse, np.ndarray]]
        Per-cell transient rates that the masks were derived from
        (useful for sanity plots / diagnostics).
    ext_norms : dict[etype, dict[mode, tuple | None]]
        The classification specs used.
    """
    if not ref_sessions_by_etype:
        raise ValueError("build_engram_identity: ref_sessions_by_etype is empty.")
    ctl_mice = [m for m, g in mouse_groups.items() if g == ctl_group]
    if not ctl_mice:
        raise RuntimeError(
            f"build_engram_identity: no mice with group {ctl_group!r}."
        )

    # 1) per-mouse full-session rates per etype
    rates_by_etype_mouse = {}
    for etype, sess_dict in ref_sessions_by_etype.items():
        rates = {}
        for m, sess in sess_dict.items():
            rates[m] = compute_full_session_rate(sess)
        if not rates:
            raise RuntimeError(
                f"build_engram_identity: no mice have ref session for etype={etype!r}."
            )
        rates_by_etype_mouse[etype] = rates

    # 2) per-etype ext_norm specs (one pool per etype, drawn from ctl_mice
    #    that actually have that reference session)
    ext_norms = {}
    for etype, rates in rates_by_etype_mouse.items():
        ctl_for_et = [m for m in ctl_mice if m in rates]
        if not ctl_for_et:
            raise RuntimeError(
                f"build_engram_identity etype={etype!r}: no {ctl_group} mice "
                f"have the reference session."
            )
        ext_norms[etype] = {
            'permouse':      None,
            'ctlthresh_z':   build_control_pool(rates, ctl_for_et, mode='zscore'),
            'ctlthresh_p50': build_control_pool(rates, ctl_for_et,
                                                mode='percentile', percentile=50.0),
        }
        print(f"  [engram pool] etype={etype} {ctl_group} pool n={len(ctl_for_et)} "
              f"({ctl_for_et}) -> "
              f"ctlthresh_z={ext_norms[etype]['ctlthresh_z']}, "
              f"ctlthresh_p50={ext_norms[etype]['ctlthresh_p50']}", flush=True)

    # 3) classify per (mouse, etype, mode)
    engram_id = {}
    for etype, rates in rates_by_etype_mouse.items():
        for m, rate in rates.items():
            engram_id.setdefault(m, {})[etype] = {}
            for mode in modes:
                mask = classify_engram_full_session(
                    rate, mode, ext_norm=ext_norms[etype][mode],
                    threshold=threshold,
                )
                engram_id[m][etype][mode] = mask
                n_eng = int(mask.sum())
                print(f"  [engram] mouse={m} group={mouse_groups[m]} "
                      f"etype={etype} mode={mode}: {n_eng}/{mask.size} engram",
                      flush=True)
    return engram_id, rates_by_etype_mouse, ext_norms


# ---------------------------------------------------------------------------
# Crossreg projection
# ---------------------------------------------------------------------------

def project_engram_to_session(ref_session, ref_engram_full_idx,
                              target_session, crossreg, mapping):
    """Project a set of engram cells from *ref_session* to *target_session*.

    Parameters
    ----------
    ref_session : BehaviourSession
        Session whose full S the engram identity is defined over.
    ref_engram_full_idx : iterable[int]
        Row indices into ``ref_session.S`` selecting engram cells.
    target_session : BehaviourSession
        Session to project into.
    crossreg : Crossreg
        Cross-registration object covering BOTH sessions.
    mapping : str
        Mapping name that includes both sessions, e.g.
        'TFC_cond+Test_A' or 'TFC_cond+Test_B+Test_B_1wk'.

    Returns
    -------
    target_engram_full_idx : np.ndarray
        Row indices into ``target_session.S`` that cross-register to
        the requested reference engram cells. Cells with no
        cross-registered counterpart in *target_session* are dropped.
    """
    ref_engram_full_idx = np.asarray(list(ref_engram_full_idx), dtype=int)

    # ref-session row indices -> ref-session unit_ids
    ref_S_idx = list(ref_session.S_idx)
    ref_unit_ids = np.asarray([ref_S_idx[i] for i in ref_engram_full_idx])

    # mapping rows: one row per cross-registered cell
    df_mapping = crossreg.get_mappings_cells(mapping_type=mapping)
    ref_col = ref_session.get_df_col(with_crossreg=crossreg)
    tgt_col = target_session.get_df_col(with_crossreg=crossreg)

    ref_units_in_xreg = np.asarray(
        get_actual_cells_from_df_session(df_mapping[ref_col])
    )
    tgt_units_in_xreg = np.asarray(
        get_actual_cells_from_df_session(df_mapping[tgt_col])
    )

    mask = np.isin(ref_units_in_xreg, ref_unit_ids)
    tgt_unit_ids_engram = tgt_units_in_xreg[mask]

    if tgt_unit_ids_engram.size == 0:
        return np.array([], dtype=int)
    tgt_idx = target_session.get_S_indeces(list(tgt_unit_ids_engram))
    return np.asarray(tgt_idx, dtype=int)


print("SSTCa2_engram.py loaded.")

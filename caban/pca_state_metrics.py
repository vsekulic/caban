"""caban.pca_state_metrics — joint-PCA Euclidean state-space metrics.

Brain-dead simple distances in the joint-PCA subspace of CrossregFullPCA
(method 2, event_window='toneshock'). All metrics are raw Euclidean — no
covariance whitening, no LDA, no shuffles.

Two metric families:

* Excursion (TFC_cond)
    - excursion_from_baseline : ||centroid(shock) - centroid(baseline)||
    - excursion_from_origin   : mean ||x|| over shock frames

* Tone-to-Shock distance
    For each session in {TFC_cond, Test_B, Test_B_1wk}, distance from tones
    in that session to where the shocks were in TFC_cond:
      - dist_allTones_to_allShocks_<sess>
      - dist_allTones_to_firstShock_<sess>
      - dist_allTones_to_lastShock_<sess>
      - dist_firstTone_to_allShocks_<sess>
      - dist_lastTone_to_allShocks_<sess>

Group-level stats per metric: one-way ANOVA + Welch pairwise t-tests with
Holm correction (each mouse contributes a single scalar).

See ``analysis_methods_templates/pca_state_metrics_methods.txt`` for the
full description.

Public entry point: ``run_pca_state_metrics_pipeline``, called from
``caban.population.run_pca_state_metrics_from_results``.
"""

import os
import shutil
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind, bootstrap as scipy_bootstrap

import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import mixedlm, ols

from caban.spatial import (
    _PV_GROUP_ORDER,
    _PV_GROUP_LABELS,
    _PV_BOX_COLORS,
    _PV_DOT_COLORS,
    _PV_BOX_ALPHA,
    _PV_SCATTER_SIZE,
    _pv_p_to_star,
    _pv_draw_bracket,
)
from caban.single_unit_common import joint_wald_test


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_SECTION_EXCURSION = "Excursion (TFC_cond)"
_SECTION_TONE_TO_SHOCK = "Tone-to-Shock distance"

_TEST_SESSIONS_ALL = ("TFC_cond", "Test_B", "Test_B_1wk")

_TONE_SHOCK_VARIANTS = (
    "allTones_to_allShocks",
    "allTones_to_firstShock",
    "allTones_to_lastShock",
    "firstTone_to_allShocks",
    "lastTone_to_allShocks",
)

_METRICS_BY_SECTION = {
    _SECTION_EXCURSION: (
        "excursion_from_baseline",
        "excursion_from_origin",
    ),
    _SECTION_TONE_TO_SHOCK: tuple(
        f"dist_{variant}_{sess}"
        for sess in _TEST_SESSIONS_ALL
        for variant in _TONE_SHOCK_VARIANTS
    ),
}

# ---- Per-event metric registry (used by the per-event pipeline only) ------
# In TFC_cond, the tone-to-shock metric depends on the pairing scheme:
#   indexpaired -> one row per trial (tone_i vs shock_i)
#   allpairs    -> N_tones x N_shocks rows
# In Test_B / Test_B_1wk there are no shocks, so the test-session
# tone-to-shock distance is always against TFC_cond shocks. The
# "to_allShocks" variant is also scheme-aware:
#   indexpaired -> per-tone vs centroid(all TFC shocks)
#                  (one row per tone, name "perTone_to_allShocks")
#   allpairs    -> per-tone x per-TFC-shock all pairs
#                  (N_tones x N_TFC_shocks rows, name "allPairs_tone_shock")
# The firstShock / lastShock variants reference a single TFC shock cloud
# regardless of scheme, so they collapse to one row per tone in both.
_PEREVENT_TESTSESSION_FIXED_VARIANTS = (
    "perTone_to_firstShock",
    "perTone_to_lastShock",
)
_PEREVENT_TESTSESSION_ALLSHOCKS_BY_SCHEME = {
    "indexpaired": "perTone_to_allShocks",
    "allpairs":    "allPairs_tone_shock",
}
_PEREVENT_TESTSESSIONS = ("Test_B", "Test_B_1wk")
_PEREVENT_EXCURSION_METRICS = (
    "excursion_perShock_from_baseline_TFC_cond",
    "excursion_perFrame_from_origin_TFC_cond",
)


def _perevent_testsession_metrics(scheme):
    """Return the test-session metric names for the given pairing scheme."""
    if scheme not in _PEREVENT_TESTSESSION_ALLSHOCKS_BY_SCHEME:
        raise KeyError(f"Unknown per-event scheme {scheme!r}")
    allshock_variant = _PEREVENT_TESTSESSION_ALLSHOCKS_BY_SCHEME[scheme]
    variants = (allshock_variant,) + _PEREVENT_TESTSESSION_FIXED_VARIANTS
    return tuple(f"dist_{v}_{s}"
                 for s in _PEREVENT_TESTSESSIONS for v in variants)


_PEREVENT_TFC_METRIC_BY_SCHEME = {
    "indexpaired": "dist_paired_toneShock_TFC_cond",
    "allpairs":    "dist_allPairs_toneShock_TFC_cond",
}
_PEREVENT_SCHEMES = ("indexpaired", "allpairs")


def _perevent_metrics_by_section(scheme):
    if scheme not in _PEREVENT_TFC_METRIC_BY_SCHEME:
        raise KeyError(f"Unknown per-event scheme {scheme!r}")
    return {
        _SECTION_EXCURSION: _PEREVENT_EXCURSION_METRICS,
        _SECTION_TONE_TO_SHOCK: (
            _PEREVENT_TFC_METRIC_BY_SCHEME[scheme],
        ) + _perevent_testsession_metrics(scheme),
    }

_METHODS_TEMPLATE_FILENAME = "pca_state_metrics_methods.txt"
_METHODS_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "analysis_methods_templates"
)
_OUT_SUBDIR = "PCA_state_metrics"


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_session_masks(sess, n_frames):
    """Boolean masks (baseline, tone, shock) over the *n_frames* PC frame axis.

    A frame is in `tone` if it lies in any [tone_onset, tone_offset) window;
    in `shock` if it lies in any [shock_onset, shock_offset) window; in
    `baseline` otherwise (i.e. neither tone nor shock).

    Parameters
    ----------
    sess : Session-like object
        Must expose `tone_onsets` / `tone_offsets`; optionally
        `shock_onsets` / `shock_offsets` (treated as empty if missing).
    n_frames : int
        Frame count of the corresponding PC array; windows are clipped here.

    Returns
    -------
    masks : dict[str, ndarray(bool)]
        Keys: 'baseline', 'tone', 'shock'.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be ≥1, got {n_frames}")
    tone_mask = np.zeros(n_frames, dtype=bool)
    shock_mask = np.zeros(n_frames, dtype=bool)

    for onset, offset in zip(sess.tone_onsets, sess.tone_offsets):
        a = max(int(onset), 0)
        b = min(int(offset), n_frames)
        if b > a:
            tone_mask[a:b] = True

    shock_onsets = list(getattr(sess, "shock_onsets", []) or [])
    shock_offsets = list(getattr(sess, "shock_offsets", []) or [])
    for onset, offset in zip(shock_onsets, shock_offsets):
        a = max(int(onset), 0)
        b = min(int(offset), n_frames)
        if b > a:
            shock_mask[a:b] = True

    if np.any(tone_mask & shock_mask):
        raise RuntimeError(
            "Tone and shock windows overlap on the PC frame axis."
        )
    baseline_mask = ~(tone_mask | shock_mask)
    return {"baseline": baseline_mask, "tone": tone_mask, "shock": shock_mask}


# ---------------------------------------------------------------------------
# Window/event helpers
# ---------------------------------------------------------------------------

def _window_mask(onset, offset, n_frames):
    """Return a boolean mask for [onset, offset) clipped to [0, n_frames)."""
    a = max(int(onset), 0)
    b = min(int(offset), n_frames)
    if b <= a:
        raise ValueError(
            f"Empty window after clipping: onset={onset}, offset={offset}, "
            f"n_frames={n_frames}"
        )
    m = np.zeros(n_frames, dtype=bool)
    m[a:b] = True
    return m


def _all_windows_mask(onsets, offsets, n_frames):
    """Boolean mask for the union of all [onset, offset) windows."""
    m = np.zeros(n_frames, dtype=bool)
    for on, off in zip(onsets, offsets):
        a = max(int(on), 0)
        b = min(int(off), n_frames)
        if b > a:
            m[a:b] = True
    return m


def _centroid(X):
    """Mean over rows of X, with a non-empty assertion."""
    if X.shape[0] < 1:
        raise ValueError("Cannot take centroid of empty point set.")
    return X.mean(axis=0)


def _euclid_centroid_distance(X_a, X_b):
    """Euclidean distance between centroids of two point clouds in PCA space."""
    return float(np.linalg.norm(_centroid(X_a) - _centroid(X_b)))


def _euclid_mean_norm(X):
    """Mean Euclidean norm (||x||) over rows of X. Distance from origin."""
    if X.shape[0] < 1:
        raise ValueError("Cannot compute mean norm of empty point set.")
    return float(np.mean(np.linalg.norm(X, axis=1)))


# ---------------------------------------------------------------------------
# Per-mouse orchestrator
# ---------------------------------------------------------------------------

def compute_mouse_state_metrics(full_pca):
    """Compute the simplified metric panel for one mouse.

    Two metric families, all in joint-PCA coordinates of the toneshock
    CrossregFullPCA (method 2). All distances are raw Euclidean — no
    covariance whitening, no LDA, no shuffles.

    Excursion (TFC_cond):
      excursion_from_baseline
          ||centroid(shock_TFC) - centroid(baseline_TFC)||
      excursion_from_origin
          mean over TFC shock frames of ||x|| (distance from PCA origin).

    Tone-to-Shock distance:
      For each session in {TFC_cond, Test_B, Test_B_1wk}:
        dist_allTones_to_allShocks_<sess>
            ||centroid(all tones in <sess>) - centroid(all shocks in TFC_cond)||
        dist_allTones_to_firstShock_<sess>
            ||centroid(all tones in <sess>) - centroid(first shock in TFC_cond)||
        dist_allTones_to_lastShock_<sess>
        dist_firstTone_to_allShocks_<sess>
            ||centroid(first tone in <sess>) - centroid(all shocks in TFC_cond)||
        dist_lastTone_to_allShocks_<sess>

    The shock cloud always lives in TFC_cond — Test_B / Test_B_1wk have
    no shocks. The user-visible question is "how close do tones in each
    session sit to where the shocks were in TFC_cond?"

    Parameters
    ----------
    full_pca : CrossregFullPCA
        Must have ``.PCs`` populated (method=2 fit). Sessions: TFC_cond,
        Test_B, Test_B_1wk.

    Returns
    -------
    metrics : dict[str, float]
        Flat dict of metric_name -> scalar.
    """
    if full_pca.PCs is None:
        raise RuntimeError(
            f"CrossregFullPCA for {full_pca.mouse!r} has no PCs populated."
        )

    PCs = {name: full_pca.PCs[name] for name in full_pca.PCs}
    sessions = {name: full_pca.sessions[name] for name in PCs}

    # ---- TFC_cond data: baseline, all shocks, first/last shock --------
    cond_pcs = PCs["TFC_cond"]
    n_cond = cond_pcs.shape[0]
    cond_sess = sessions["TFC_cond"]
    cond_masks = build_session_masks(cond_sess, n_cond)
    X_baseline_cond = cond_pcs[cond_masks["baseline"]]
    X_shock_all_cond = cond_pcs[cond_masks["shock"]]
    if X_baseline_cond.shape[0] == 0:
        raise RuntimeError(f"{full_pca.mouse}: no baseline frames in TFC_cond.")
    if X_shock_all_cond.shape[0] == 0:
        raise RuntimeError(f"{full_pca.mouse}: no shock frames in TFC_cond.")

    shock_onsets = list(getattr(cond_sess, "shock_onsets", []) or [])
    shock_offsets = list(getattr(cond_sess, "shock_offsets", []) or [])
    if len(shock_onsets) < 1 or len(shock_offsets) < 1:
        raise RuntimeError(
            f"{full_pca.mouse}: TFC_cond has no shock_onsets/offsets."
        )
    X_shock_first_cond = cond_pcs[
        _window_mask(shock_onsets[0], shock_offsets[0], n_cond)
    ]
    X_shock_last_cond = cond_pcs[
        _window_mask(shock_onsets[-1], shock_offsets[-1], n_cond)
    ]

    metrics = {
        "excursion_from_baseline": _euclid_centroid_distance(
            X_shock_all_cond, X_baseline_cond
        ),
        "excursion_from_origin": _euclid_mean_norm(X_shock_all_cond),
    }

    # ---- Tone-to-Shock distance for each session ------------------------
    for sess_name in _TEST_SESSIONS_ALL:
        if sess_name not in PCs:
            raise KeyError(f"{full_pca.mouse}: PCs missing session {sess_name!r}.")
        sess_pcs = PCs[sess_name]
        n_frames = sess_pcs.shape[0]
        sess = sessions[sess_name]

        tone_onsets = list(getattr(sess, "tone_onsets", []) or [])
        tone_offsets = list(getattr(sess, "tone_offsets", []) or [])
        if len(tone_onsets) < 1 or len(tone_offsets) < 1:
            raise RuntimeError(
                f"{full_pca.mouse}: {sess_name} has no tone_onsets/offsets."
            )

        all_tones_mask = _all_windows_mask(tone_onsets, tone_offsets, n_frames)
        first_tone_mask = _window_mask(tone_onsets[0], tone_offsets[0], n_frames)
        last_tone_mask = _window_mask(tone_onsets[-1], tone_offsets[-1], n_frames)

        X_all_tones = sess_pcs[all_tones_mask]
        X_first_tone = sess_pcs[first_tone_mask]
        X_last_tone = sess_pcs[last_tone_mask]
        if X_all_tones.shape[0] == 0:
            raise RuntimeError(
                f"{full_pca.mouse}: no tone frames in {sess_name}."
            )

        metrics[f"dist_allTones_to_allShocks_{sess_name}"] = (
            _euclid_centroid_distance(X_all_tones, X_shock_all_cond)
        )
        metrics[f"dist_allTones_to_firstShock_{sess_name}"] = (
            _euclid_centroid_distance(X_all_tones, X_shock_first_cond)
        )
        metrics[f"dist_allTones_to_lastShock_{sess_name}"] = (
            _euclid_centroid_distance(X_all_tones, X_shock_last_cond)
        )
        metrics[f"dist_firstTone_to_allShocks_{sess_name}"] = (
            _euclid_centroid_distance(X_first_tone, X_shock_all_cond)
        )
        metrics[f"dist_lastTone_to_allShocks_{sess_name}"] = (
            _euclid_centroid_distance(X_last_tone, X_shock_all_cond)
        )

    return metrics
# ---------------------------------------------------------------------------
# Group-level statistics + Nature-style plot
# ---------------------------------------------------------------------------

def _anova_holm_pairs(groups_data):
    """Run one-way ANOVA + Welch t-tests with Holm correction.

    Parameters
    ----------
    groups_data : dict[str, ndarray]
        {group_name: per-mouse values}, with at least 2 mice per group.

    Returns
    -------
    omnibus : dict
        Keys: 'F', 'p' (one-way ANOVA across all groups present).
    contrasts : list[dict]
        One dict per pair: {'a','b','t','df','p_raw','p_holm','n_a','n_b'}.
        df is approximated for Welch's t (Welch-Satterthwaite).
    """
    grp_order = [g for g in _PV_GROUP_ORDER if g in groups_data
                 and len(groups_data[g]) >= 2]
    if len(grp_order) < 2:
        raise RuntimeError(
            f"Need ≥2 groups with ≥2 mice each; got {[(g, len(groups_data.get(g, []))) for g in _PV_GROUP_ORDER]}"
        )
    arrays = [np.asarray(groups_data[g], dtype=float) for g in grp_order]
    F, p_om = f_oneway(*arrays)

    pairs = list(combinations(range(len(grp_order)), 2))
    raw_ps = []
    rows = []
    for ai, bi in pairs:
        a, b = grp_order[ai], grp_order[bi]
        va, vb = groups_data[a], groups_data[b]
        res = ttest_ind(va, vb, equal_var=False)
        # Welch-Satterthwaite df
        s2a = float(np.var(va, ddof=1))
        s2b = float(np.var(vb, ddof=1))
        na, nb = len(va), len(vb)
        if s2a > 0 and s2b > 0:
            df = (s2a / na + s2b / nb) ** 2 / (
                (s2a / na) ** 2 / (na - 1) + (s2b / nb) ** 2 / (nb - 1)
            )
        else:
            df = float(na + nb - 2)
        raw_ps.append(float(res.pvalue))
        rows.append({
            "a": a, "b": b,
            "t": float(res.statistic),
            "df": float(df),
            "p_raw": float(res.pvalue),
            "n_a": int(na),
            "n_b": int(nb),
        })
    _, p_holm, _, _ = multipletests(raw_ps, method="holm")
    for k, r in enumerate(rows):
        r["p_holm"] = float(p_holm[k])
    return {"F": float(F), "p": float(p_om)}, rows


def _anova_tukey_pairs(groups_data):
    """Run one-way ANOVA + Tukey HSD post hoc.

    Parameters
    ----------
    groups_data : dict[str, ndarray]
        {group_name: per-mouse values}, with at least 2 mice per group.

    Returns
    -------
    omnibus : dict
        Keys: 'F', 'p' (one-way ANOVA across all groups present).
    contrasts : list[dict]
        One dict per pair:
        {'a','b','mean_diff','p_raw','p_holm','n_a','n_b'}.
        ``p_raw`` and ``p_holm`` both store the Tukey-adjusted p so the
        plotting/stats-text helpers (which key on p_holm/p_raw) keep
        working unchanged. The "raw" pairwise p has no separate meaning
        under Tukey HSD because the family-wise correction is built in.
    """
    grp_order = [g for g in _PV_GROUP_ORDER if g in groups_data
                 and len(groups_data[g]) >= 2]
    if len(grp_order) < 2:
        raise RuntimeError(
            f"Need >=2 groups with >=2 mice each; got "
            f"{[(g, len(groups_data.get(g, []))) for g in _PV_GROUP_ORDER]}"
        )
    arrays = [np.asarray(groups_data[g], dtype=float) for g in grp_order]
    F, p_om = f_oneway(*arrays)

    # Build long-form vectors for Tukey
    vals = np.concatenate(arrays)
    labels = np.concatenate([
        np.full(len(arr), grp, dtype=object) for grp, arr in
        zip(grp_order, arrays)
    ])
    tk = pairwise_tukeyhsd(endog=vals, groups=labels, alpha=0.05)
    # statsmodels TukeyHSDResults._results_table.data[1:] rows are
    # [g1, g2, meandiff, p-adj, lower, upper, reject]
    rows = []
    n_by_grp = {g: len(arr) for g, arr in zip(grp_order, arrays)}
    for r in tk._results_table.data[1:]:
        a, b = str(r[0]), str(r[1])
        if a not in n_by_grp or b not in n_by_grp:
            raise KeyError(
                f"Tukey row references unknown group(s): {a!r}, {b!r}"
            )
        mean_diff = float(r[2])
        p_adj = float(r[3])
        rows.append({
            "a": a, "b": b,
            "mean_diff": mean_diff,
            "p_raw": p_adj,
            "p_holm": p_adj,  # Tukey already family-wise corrected
            "n_a": int(n_by_grp[a]),
            "n_b": int(n_by_grp[b]),
        })
    return {"F": float(F), "p": float(p_om)}, rows


def _bootstrap_perm_pairs(groups_data, n_boot=10000, n_perm=10000,
                          random_state=20260427):
    """Permutation ANOVA + percentile-bootstrap pairwise CIs (no normality).

    Drop-in replacement for ``_anova_holm_pairs`` / ``_anova_tukey_pairs``
    that does not assume normality or equal variance. Recommended for
    small-n designs (n_mice ≈ 5/group) where parametric assumptions are
    suspect.

    - **Omnibus**: one-way F statistic; omnibus p estimated by permuting
      group labels (``n_perm`` shuffles), reported as
      ``(n_extreme + 1) / (n_perm + 1)`` so p > 0.
    - **Pairwise**: ``mean(a) - mean(b)`` with a percentile-bootstrap
      95% CI (``scipy.stats.bootstrap``, ``method='percentile'``,
      ``n_boot`` resamples) and a two-sided permutation p-value on
      ``|mean_a - mean_b|``. Holm-corrected across the 3 pairs.
      (BCa is unavailable for two-sample statistics in scipy; the
      percentile interval is the standard conservative replacement.)

    Returns
    -------
    omnibus : dict   {'F': observed F, 'p': permutation p}
    contrasts : list[dict]
        {'a','b','mean_diff','ci_lo','ci_hi','p_raw','p_holm',
         'n_a','n_b','n_boot','n_perm'}.
    """
    grp_order = [g for g in _PV_GROUP_ORDER if g in groups_data
                 and len(groups_data[g]) >= 2]
    if len(grp_order) < 2:
        raise RuntimeError(
            f"Need >=2 groups with >=2 mice each; got "
            f"{[(g, len(groups_data.get(g, []))) for g in _PV_GROUP_ORDER]}"
        )
    arrays = [np.asarray(groups_data[g], dtype=float) for g in grp_order]
    F_obs = float(f_oneway(*arrays).statistic)

    rs = np.random.default_rng(random_state)
    pooled = np.concatenate(arrays)
    sizes = [len(a) for a in arrays]
    n_extreme = 0
    for _ in range(int(n_perm)):
        perm = rs.permutation(pooled)
        chunks = []
        idx = 0
        for s in sizes:
            chunks.append(perm[idx:idx + s])
            idx += s
        if float(f_oneway(*chunks).statistic) >= F_obs:
            n_extreme += 1
    p_om = (n_extreme + 1) / (int(n_perm) + 1)
    omnibus = {"F": F_obs, "p": float(p_om)}

    pairs = list(combinations(range(len(grp_order)), 2))
    raw_ps = []
    rows = []
    for k, (ai, bi) in enumerate(pairs):
        a, b = grp_order[ai], grp_order[bi]
        va = np.asarray(groups_data[a], dtype=float)
        vb = np.asarray(groups_data[b], dtype=float)
        obs_diff = float(np.mean(va) - np.mean(vb))

        # Percentile bootstrap CI on the mean difference. (scipy's BCa
        # is one-sample only; percentile is the standard conservative
        # alternative for two-sample statistics.)
        boot = scipy_bootstrap(
            (va, vb),
            statistic=lambda x, y, axis=-1: np.mean(x, axis=axis)
                                            - np.mean(y, axis=axis),
            n_resamples=int(n_boot),
            method="percentile",
            confidence_level=0.95,
            random_state=np.random.default_rng(random_state + 17 * (k + 1)),
            paired=False,
            vectorized=True,
        )
        ci_lo = float(boot.confidence_interval.low)
        ci_hi = float(boot.confidence_interval.high)

        # Permutation p on |mean_diff|
        rs_pair = np.random.default_rng(random_state + 31 * (k + 1))
        pooled_ab = np.concatenate([va, vb])
        n_a = len(va)
        abs_obs = abs(obs_diff)
        n_extreme_pair = 0
        for _ in range(int(n_perm)):
            perm = rs_pair.permutation(pooled_ab)
            d = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
            if abs(d) >= abs_obs:
                n_extreme_pair += 1
        p_raw = (n_extreme_pair + 1) / (int(n_perm) + 1)
        raw_ps.append(p_raw)
        rows.append({
            "a": a, "b": b,
            "mean_diff": obs_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_raw": float(p_raw),
            "n_a": int(n_a),
            "n_b": int(len(vb)),
            "n_boot": int(n_boot),
            "n_perm": int(n_perm),
        })
    _, p_holm, _, _ = multipletests(raw_ps, method="holm")
    for k, r in enumerate(rows):
        r["p_holm"] = float(p_holm[k])
    return omnibus, rows


def _state_metric_boxplot(ax, groups_data, ylabel, title,
                          stats_fn=_anova_holm_pairs):
    """Nature-style boxplot panel without on-plot statistical text.

    Significance stars are drawn above brackets for the pairwise contrasts
    using the p-values returned by *stats_fn* (re-using ``_pv_p_to_star``
    and ``_pv_draw_bracket`` from caban.spatial). No ANOVA / contrast
    text is written on the figure — those go to the sibling stats file.

    Parameters
    ----------
    stats_fn : callable
        Function ``(groups_data) -> (omnibus, contrasts)`` whose contrast
        dicts expose 'a','b','p_raw','p_holm' (Welch+Holm or Tukey HSD).
    """
    grp_order = [g for g in _PV_GROUP_ORDER if g in groups_data
                 and len(groups_data[g]) > 0]
    if len(grp_order) == 0:
        ax.set_title(title, fontsize=9)
        return None, None

    bw = 0.60
    bp_kw = dict(patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", lw=1.2),
                 whiskerprops=dict(color="black", lw=0.6),
                 capprops=dict(color="black", lw=0.6))

    for gi, grp in enumerate(grp_order):
        vals = np.asarray(groups_data[grp], dtype=float)
        bp = ax.boxplot([vals], positions=[gi], widths=bw, **bp_kw)
        for patch in bp["boxes"]:
            patch.set_facecolor(_PV_BOX_COLORS[grp])
            patch.set_alpha(_PV_BOX_ALPHA)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        jit = np.random.default_rng(42 + gi * 100).uniform(
            -bw * 0.18, bw * 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), gi) + jit, vals,
                   color=_PV_DOT_COLORS.get(grp, "gray"),
                   s=_PV_SCATTER_SIZE, zorder=5,
                   edgecolors="white", linewidths=0.3)

    ax.set_xticks(range(len(grp_order)))
    ax.set_xticklabels([_PV_GROUP_LABELS.get(g, g) for g in grp_order],
                       fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.6)

    # Run stats (only if ≥2 groups have ≥2 mice)
    eligible = [g for g in grp_order if len(groups_data[g]) >= 2]
    if len(eligible) < 2:
        return None, None
    omnibus, contrasts = stats_fn(
        {g: np.asarray(groups_data[g], dtype=float) for g in eligible}
    )

    all_vals = np.concatenate([np.asarray(groups_data[g]) for g in grp_order
                               if len(groups_data[g]) > 0])
    y_max = float(np.nanmax(all_vals))
    y_min = float(np.nanmin(all_vals))
    y_rng = max(y_max - y_min, 1e-6)
    bdy = 0.03 * y_rng
    bgap = 2.6 * bdy
    by = y_max + 0.04 * y_rng

    grp_pos = {g: i for i, g in enumerate(grp_order)}
    for c in contrasts:
        if c["a"] not in grp_pos or c["b"] not in grp_pos:
            continue
        star, fs = _pv_p_to_star(c["p_holm"], c["p_raw"])
        if star is None:
            continue
        x1 = grp_pos[c["a"]]
        x2 = grp_pos[c["b"]]
        if x1 > x2:
            x1, x2 = x2, x1
        _pv_draw_bracket(ax, x1, x2, by, bdy, y_rng, star, fs)
        by += bgap

    return omnibus, contrasts


def _format_stats_text(metric, ylabel, omnibus, contrasts, groups_data,
                      stats_label="Welch t + Holm"):
    """Return a formatted statistics block for the sibling __stats.txt file.

    *stats_label* selects the rendering of the contrast rows. Supported:
      - "Welch t + Holm": expects per-contrast keys t, df, p_raw, p_holm.
      - "Tukey HSD":      expects per-contrast keys mean_diff, p_raw (=p_adj).
    """
    lines = []
    lines.append(f"Metric: {metric}")
    lines.append(f"Ylabel: {ylabel}")
    lines.append(f"Stats: {stats_label}")
    lines.append("")
    lines.append("Per-group n and summary (per-mouse values):")
    for g in _PV_GROUP_ORDER:
        v = np.asarray(groups_data.get(g, []), dtype=float)
        if v.size:
            lines.append(
                f"  {g:8s}  n={v.size:2d}  mean={np.mean(v):.4g}  "
                f"sd={np.std(v, ddof=1) if v.size > 1 else float('nan'):.4g}  "
                f"median={np.median(v):.4g}"
            )
        else:
            lines.append(f"  {g:8s}  n=0")
    lines.append("")
    if omnibus is not None:
        if stats_label == "Permutation + percentile bootstrap":
            lines.append(
                f"Permutation ANOVA: F={omnibus['F']:.4g}, "
                f"p_perm={omnibus['p']:.4g}"
            )
            lines.append("")
            lines.append(
                "Pairwise permutation p (Holm-corrected) + BCa 95% CI on "
                "mean diff:"
            )
            for c in contrasts:
                lines.append(
                    f"  {c['a']:8s} vs {c['b']:8s}  "
                    f"mean_diff={c['mean_diff']:+.4g}  "
                    f"95%CI=[{c['ci_lo']:+.4g}, {c['ci_hi']:+.4g}]  "
                    f"p_perm_raw={c['p_raw']:.4g}  "
                    f"p_holm={c['p_holm']:.4g}  "
                    f"(n_{c['a']}={c['n_a']}, n_{c['b']}={c['n_b']}, "
                    f"n_boot={c['n_boot']}, n_perm={c['n_perm']})"
                )
            return "\n".join(lines) + "\n"
        lines.append(
            f"One-way ANOVA: F={omnibus['F']:.4g}, p={omnibus['p']:.4g}"
        )
        lines.append("")
        if stats_label == "Tukey HSD":
            lines.append("Pairwise Tukey HSD (family-wise alpha=0.05):")
            for c in contrasts:
                lines.append(
                    f"  {c['a']:8s} vs {c['b']:8s}  "
                    f"mean_diff={c['mean_diff']:+.4g}  "
                    f"p_adj={c['p_raw']:.4g}  "
                    f"(n_{c['a']}={c['n_a']}, n_{c['b']}={c['n_b']})"
                )
            return "\n".join(lines) + "\n"
        lines.append("Pairwise Welch t-tests (Holm-corrected within metric):")
        for c in contrasts:
            lines.append(
                f"  {c['a']:8s} vs {c['b']:8s}  "
                f"t={c['t']:+.3f}, df={c['df']:.2f}  "
                f"p_raw={c['p_raw']:.4g}  p_holm={c['p_holm']:.4g}  "
                f"(n_{c['a']}={c['n_a']}, n_{c['b']}={c['n_b']})"
            )
    else:
        lines.append("Insufficient data for omnibus / pairwise stats.")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Per-event pipeline (parallel to the per-mouse-mean pipeline)
# ===========================================================================
#
# Each metric is computed at the per-event level (per trial / per tone / per
# shock-frame) instead of being collapsed to one scalar per mouse. This
# preserves within-mouse variability for inferential models that account
# for clustering.
#
# Two TFC_cond pairing schemes:
#   indexpaired : tone_i <-> shock_i, one row per trial
#   allpairs    : every (tone_i, shock_j), N_tones * N_shocks rows
# Test_B / Test_B_1wk metrics are scheme-independent and duplicated.
#
# Two stats engines per metric (parallel output files):
#   _LMM   : statsmodels MixedLM, distance ~ C(group) + (1|mouse)
#            pairwise Wald contrasts, Holm correction across the 3 pairs.
#            Falls back to OLS + cluster-robust SE on `mouse` if MixedLM
#            fails to converge or yields a singular RE variance.
#   _ANOVA : per-event one-way ANOVA + Tukey HSD, ignoring within-mouse
#            clustering (deliberately Igarashi-style; will inflate
#            significance when events within a mouse are correlated).
# ---------------------------------------------------------------------------


def _collect_event_clouds(full_pca):
    """Return per-event point-cloud dictionaries for one mouse.

    Returns
    -------
    out : dict
        Keys:
          'tfc_baseline_centroid'      : (n_pcs,) ndarray
          'tfc_shock_all_centroid'     : (n_pcs,) ndarray
          'tfc_first_shock_centroid'   : (n_pcs,) ndarray
          'tfc_last_shock_centroid'    : (n_pcs,) ndarray
          'tfc_tone_clouds'            : list of (M_i, n_pcs) ndarrays
          'tfc_shock_clouds'           : list of (M_i, n_pcs) ndarrays
          'testB_tone_clouds'          : list of (M_j, n_pcs) ndarrays
          'testB_1wk_tone_clouds'      : list of (M_j, n_pcs) ndarrays
    """
    if full_pca.PCs is None:
        raise RuntimeError(
            f"CrossregFullPCA for {full_pca.mouse!r} has no PCs populated."
        )

    PCs = {name: full_pca.PCs[name] for name in full_pca.PCs}
    sessions = {name: full_pca.sessions[name] for name in PCs}

    cond_pcs = PCs["TFC_cond"]
    n_cond = cond_pcs.shape[0]
    cond_sess = sessions["TFC_cond"]
    cond_masks = build_session_masks(cond_sess, n_cond)

    X_baseline_cond = cond_pcs[cond_masks["baseline"]]
    X_shock_all_cond = cond_pcs[cond_masks["shock"]]
    if X_baseline_cond.shape[0] == 0:
        raise RuntimeError(f"{full_pca.mouse}: no baseline frames in TFC_cond.")
    if X_shock_all_cond.shape[0] == 0:
        raise RuntimeError(f"{full_pca.mouse}: no shock frames in TFC_cond.")

    tone_onsets_cond = list(getattr(cond_sess, "tone_onsets", []) or [])
    tone_offsets_cond = list(getattr(cond_sess, "tone_offsets", []) or [])
    shock_onsets = list(getattr(cond_sess, "shock_onsets", []) or [])
    shock_offsets = list(getattr(cond_sess, "shock_offsets", []) or [])
    if len(tone_onsets_cond) < 1:
        raise RuntimeError(f"{full_pca.mouse}: TFC_cond has no tones.")
    if len(shock_onsets) < 1:
        raise RuntimeError(f"{full_pca.mouse}: TFC_cond has no shocks.")

    tfc_tone_clouds = [
        cond_pcs[_window_mask(on, off, n_cond)]
        for on, off in zip(tone_onsets_cond, tone_offsets_cond)
    ]
    tfc_shock_clouds = [
        cond_pcs[_window_mask(on, off, n_cond)]
        for on, off in zip(shock_onsets, shock_offsets)
    ]
    for i, c in enumerate(tfc_tone_clouds):
        if c.shape[0] == 0:
            raise RuntimeError(
                f"{full_pca.mouse}: TFC_cond tone trial {i} has 0 frames."
            )
    for i, c in enumerate(tfc_shock_clouds):
        if c.shape[0] == 0:
            raise RuntimeError(
                f"{full_pca.mouse}: TFC_cond shock trial {i} has 0 frames."
            )

    out = {
        "tfc_baseline_centroid":   _centroid(X_baseline_cond),
        "tfc_shock_all_centroid":  _centroid(X_shock_all_cond),
        "tfc_first_shock_centroid": _centroid(tfc_shock_clouds[0]),
        "tfc_last_shock_centroid":  _centroid(tfc_shock_clouds[-1]),
        "tfc_tone_clouds":  tfc_tone_clouds,
        "tfc_shock_clouds": tfc_shock_clouds,
        "tfc_shock_all_frames": X_shock_all_cond,
    }

    for sess_name, key in (("Test_B", "testB_tone_clouds"),
                           ("Test_B_1wk", "testB_1wk_tone_clouds")):
        if sess_name not in PCs:
            raise KeyError(f"{full_pca.mouse}: PCs missing {sess_name!r}.")
        sess_pcs = PCs[sess_name]
        n_frames = sess_pcs.shape[0]
        sess = sessions[sess_name]
        tone_on = list(getattr(sess, "tone_onsets", []) or [])
        tone_off = list(getattr(sess, "tone_offsets", []) or [])
        if len(tone_on) < 1:
            raise RuntimeError(
                f"{full_pca.mouse}: {sess_name} has no tones."
            )
        clouds = [
            sess_pcs[_window_mask(on, off, n_frames)]
            for on, off in zip(tone_on, tone_off)
        ]
        for j, c in enumerate(clouds):
            if c.shape[0] == 0:
                raise RuntimeError(
                    f"{full_pca.mouse}: {sess_name} tone {j} has 0 frames."
                )
        out[key] = clouds

    return out


def compute_mouse_event_metrics(full_pca, scheme):
    """Compute per-event metrics for one mouse under a given pairing scheme.

    Parameters
    ----------
    full_pca : CrossregFullPCA
    scheme : {'indexpaired', 'allpairs'}
        TFC_cond tone/shock pairing scheme. Test_B / Test_B_1wk metrics
        do not depend on this.

    Returns
    -------
    rows : list[dict]
        Long-form rows: {'mouse','section','metric','event_id','value'}.
    """
    if scheme not in _PEREVENT_TFC_METRIC_BY_SCHEME:
        raise KeyError(f"Unknown per-event scheme {scheme!r}")

    clouds = _collect_event_clouds(full_pca)
    mouse = full_pca.mouse
    rows = []

    # ---- Excursion (TFC_cond) -----------------------------------------
    baseline_c = clouds["tfc_baseline_centroid"]
    for i, sc in enumerate(clouds["tfc_shock_clouds"]):
        rows.append({
            "mouse": mouse,
            "section": _SECTION_EXCURSION,
            "metric": "excursion_perShock_from_baseline_TFC_cond",
            "event_id": f"shock_{i}",
            "value": float(np.linalg.norm(_centroid(sc) - baseline_c)),
        })
    shock_frames = clouds["tfc_shock_all_frames"]
    norms = np.linalg.norm(shock_frames, axis=1)
    for k, val in enumerate(norms):
        rows.append({
            "mouse": mouse,
            "section": _SECTION_EXCURSION,
            "metric": "excursion_perFrame_from_origin_TFC_cond",
            "event_id": f"frame_{k}",
            "value": float(val),
        })

    # ---- TFC_cond tone-to-shock distance (scheme-dependent) -----------
    tfc_metric = _PEREVENT_TFC_METRIC_BY_SCHEME[scheme]
    tone_clouds = clouds["tfc_tone_clouds"]
    shock_clouds = clouds["tfc_shock_clouds"]
    if scheme == "indexpaired":
        n_pairs = min(len(tone_clouds), len(shock_clouds))
        if n_pairs < 1:
            raise RuntimeError(
                f"{mouse}: indexpaired needs ≥1 trial pair; got "
                f"n_tones={len(tone_clouds)}, n_shocks={len(shock_clouds)}."
            )
        for i in range(n_pairs):
            rows.append({
                "mouse": mouse,
                "section": _SECTION_TONE_TO_SHOCK,
                "metric": tfc_metric,
                "event_id": f"trial_{i}",
                "value": float(np.linalg.norm(
                    _centroid(tone_clouds[i]) - _centroid(shock_clouds[i])
                )),
            })
    else:  # allpairs
        for i, tc in enumerate(tone_clouds):
            tc_centroid = _centroid(tc)
            for j, sc in enumerate(shock_clouds):
                rows.append({
                    "mouse": mouse,
                    "section": _SECTION_TONE_TO_SHOCK,
                    "metric": tfc_metric,
                    "event_id": f"t{i}_s{j}",
                    "value": float(np.linalg.norm(
                        tc_centroid - _centroid(sc)
                    )),
                })

    # ---- Test_B / Test_B_1wk per-tone metrics --------------------------
    # firstShock / lastShock are single ref centroids in both schemes
    # (one row per tone). The "all shocks" variant is scheme-aware:
    #   indexpaired -> tone vs centroid(all TFC shocks): one row/tone
    #   allpairs    -> tone x each TFC shock: N_tones * N_TFC_shocks rows
    fixed_ref_centroids = {
        "perTone_to_firstShock": clouds["tfc_first_shock_centroid"],
        "perTone_to_lastShock":  clouds["tfc_last_shock_centroid"],
    }
    allshock_variant = _PEREVENT_TESTSESSION_ALLSHOCKS_BY_SCHEME[scheme]
    tfc_shock_all_centroid = clouds["tfc_shock_all_centroid"]
    tfc_shock_clouds_for_allpairs = clouds["tfc_shock_clouds"]

    for sess_name, key in (("Test_B", "testB_tone_clouds"),
                           ("Test_B_1wk", "testB_1wk_tone_clouds")):
        for j, tc in enumerate(clouds[key]):
            tc_centroid = _centroid(tc)

            # firstShock / lastShock variants (one row per tone)
            for variant, ref_centroid in fixed_ref_centroids.items():
                rows.append({
                    "mouse": mouse,
                    "section": _SECTION_TONE_TO_SHOCK,
                    "metric": f"dist_{variant}_{sess_name}",
                    "event_id": f"tone_{j}",
                    "value": float(np.linalg.norm(
                        tc_centroid - ref_centroid
                    )),
                })

            # all-shocks variant (scheme-dependent)
            metric_name = f"dist_{allshock_variant}_{sess_name}"
            if scheme == "indexpaired":
                rows.append({
                    "mouse": mouse,
                    "section": _SECTION_TONE_TO_SHOCK,
                    "metric": metric_name,
                    "event_id": f"tone_{j}",
                    "value": float(np.linalg.norm(
                        tc_centroid - tfc_shock_all_centroid
                    )),
                })
            else:  # allpairs: every tone x every TFC shock
                for k, sc in enumerate(tfc_shock_clouds_for_allpairs):
                    rows.append({
                        "mouse": mouse,
                        "section": _SECTION_TONE_TO_SHOCK,
                        "metric": metric_name,
                        "event_id": f"tone_{j}_shock_{k}",
                        "value": float(np.linalg.norm(
                            tc_centroid - _centroid(sc)
                        )),
                    })

    return rows


# ---------------------------------------------------------------------------
# Per-event stats engines
# ---------------------------------------------------------------------------

def _eligible_groups_for_perevent(long_df):
    """Return ordered group list with ≥2 mice and ≥1 event per mouse."""
    grp_order = []
    for g in _PV_GROUP_ORDER:
        sub = long_df[long_df["group"] == g]
        n_mice = sub["mouse"].nunique()
        if n_mice >= 2 and len(sub) >= n_mice:  # at least 1 event/mouse
            grp_order.append(g)
    return grp_order


def _lmm_holm_pairs(long_df, value_col="value"):
    """Fit LMM ``value ~ C(group)`` with random intercept per mouse.

    Returns the standard ``(omnibus, contrasts)`` tuple. Contrast rows:
      {a, b, mean_diff, se, z, p_raw, p_holm, n_a, n_b,
       n_events_a, n_events_b, model}
    where ``model`` is 'MixedLM' on success or 'OLS+cluster_se' on
    fallback. Holm correction is applied across the 3 pairwise contrasts.
    """
    grp_order = _eligible_groups_for_perevent(long_df)
    if len(grp_order) < 2:
        raise RuntimeError(
            f"LMM needs ≥2 groups with ≥2 mice; got "
            f"{[(g, long_df.loc[long_df['group']==g, 'mouse'].nunique()) for g in _PV_GROUP_ORDER]}"
        )
    df = long_df[long_df["group"].isin(grp_order)].copy()
    df["group"] = pd.Categorical(df["group"], categories=grp_order,
                                 ordered=False)
    ref = grp_order[0]

    n_mice_by_g = {g: int(df.loc[df["group"] == g, "mouse"].nunique())
                   for g in grp_order}
    n_events_by_g = {g: int((df["group"] == g).sum()) for g in grp_order}

    formula = f"{value_col} ~ C(group, Treatment(reference='{ref}'))"
    model_label = "MixedLM"
    res = None
    try:
        md = mixedlm(formula, data=df, groups=df["mouse"])
        res = md.fit(reml=True, method="lbfgs")
        # Detect a singular RE variance: cov_re ~ 0 means RE essentially
        # collapsed; fall back to OLS+cluster.
        if hasattr(res, "cov_re"):
            cov_re_arr = np.asarray(res.cov_re)
            if cov_re_arr.size and float(cov_re_arr.flat[0]) <= 1e-12:
                raise RuntimeError("MixedLM cov_re ≈ 0 (singular).")
        if not getattr(res, "converged", True):
            raise RuntimeError("MixedLM did not converge.")
    except Exception as exc:
        print(f"[LMM] fallback to OLS+cluster_se for value_col={value_col!r}: "
              f"{exc}", flush=True)
        res = ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["mouse"].values}
        )
        model_label = "OLS+cluster_se"

    # Identify the dummy parameter names. statsmodels names them like
    # "C(group, Treatment(reference='mCherry'))[T.hM3D]". For MixedLM the
    # FE parameter index lives on `res.fe_params`; for OLS it lives on
    # `res.params`. We use whichever is present and build numeric contrast
    # vectors of length k_fe so that `t_test` / `f_test` work uniformly.
    fe_names = (list(res.fe_params.index) if hasattr(res, "fe_params")
                else list(res.params.index))
    k_fe = len(fe_names)
    coef_idx = {}
    for g in grp_order:
        if g == ref:
            continue
        match = [i for i, n in enumerate(fe_names) if n.endswith(f"[T.{g}]")]
        if len(match) != 1:
            raise RuntimeError(
                f"Could not locate FE dummy for {g!r} in {fe_names}"
            )
        coef_idx[g] = match[0]

    # Omnibus: joint Wald restriction that all non-reference dummies = 0. Shared with every
    # other joint mixed-model test in the codebase via caban.single_unit_common.joint_wald_test
    # (it works uniformly for MixedLM, whose `.f_test` has a param-vector-shape quirk involving
    # the RE variance, and for the OLS+cluster fallback).
    nonref = [g for g in grp_order if g != ref]
    nonref_names = [fe_names[coef_idx[g]] for g in nonref]
    wald = joint_wald_test(res, nonref_names, n_fixed=k_fe)
    omnibus = {"F": wald["F"], "p": wald["p"]}

    # Pairwise: Wald t_test on the appropriate linear restriction.
    pairs = list(combinations(range(len(grp_order)), 2))
    raw_ps = []
    rows = []
    for ai, bi in pairs:
        a, b = grp_order[ai], grp_order[bi]
        cvec = np.zeros(k_fe, dtype=float)
        if a == ref:
            cvec[coef_idx[b]] = -1.0   # mean(a) - mean(b) = -coef_b
        elif b == ref:
            cvec[coef_idx[a]] = +1.0   # mean(a) - mean(b) = +coef_a
        else:
            cvec[coef_idx[a]] = +1.0
            cvec[coef_idx[b]] = -1.0
        tt = res.t_test(cvec.reshape(1, -1))
        eff = float(np.asarray(tt.effect).flatten()[0])
        se = float(np.asarray(tt.sd).flatten()[0])
        z = float(np.asarray(tt.tvalue).flatten()[0])
        p_raw = float(np.asarray(tt.pvalue).flatten()[0])
        raw_ps.append(p_raw)
        rows.append({
            "a": a, "b": b,
            "mean_diff": eff,
            "se": se,
            "z": z,
            "p_raw": p_raw,
            "n_a": n_mice_by_g[a],
            "n_b": n_mice_by_g[b],
            "n_events_a": n_events_by_g[a],
            "n_events_b": n_events_by_g[b],
            "model": model_label,
        })
    _, p_holm, _, _ = multipletests(raw_ps, method="holm")
    for k, r in enumerate(rows):
        r["p_holm"] = float(p_holm[k])

    return omnibus, rows


def _perevent_anova_tukey_pairs(long_df, value_col="value"):
    """Per-event one-way ANOVA + Tukey HSD, ignoring within-mouse clustering.

    Same return shape as ``_anova_tukey_pairs`` plus ``n_events_a/b``.
    Deliberately naive (Igarashi-style); will inflate significance when
    events within a mouse are correlated. Provided for visual parity.
    """
    grp_order = _eligible_groups_for_perevent(long_df)
    if len(grp_order) < 2:
        raise RuntimeError(
            f"per-event ANOVA needs ≥2 groups with ≥2 mice; got "
            f"{[(g, long_df.loc[long_df['group']==g, 'mouse'].nunique()) for g in _PV_GROUP_ORDER]}"
        )
    arrays = [long_df.loc[long_df["group"] == g, value_col]
              .to_numpy(dtype=float) for g in grp_order]
    F, p_om = f_oneway(*arrays)

    vals = np.concatenate(arrays)
    labels = np.concatenate([np.full(len(arr), g, dtype=object)
                             for g, arr in zip(grp_order, arrays)])
    tk = pairwise_tukeyhsd(endog=vals, groups=labels, alpha=0.05)

    n_events_by_g = {g: len(arr) for g, arr in zip(grp_order, arrays)}
    n_mice_by_g = {g: int(long_df.loc[long_df["group"] == g, "mouse"].nunique())
                   for g in grp_order}
    rows = []
    for r in tk._results_table.data[1:]:
        a, b = str(r[0]), str(r[1])
        if a not in n_events_by_g or b not in n_events_by_g:
            raise KeyError(f"Tukey row references unknown group(s): {a!r}, {b!r}")
        rows.append({
            "a": a, "b": b,
            "mean_diff": float(r[2]),
            "p_raw": float(r[3]),
            "p_holm": float(r[3]),  # Tukey already family-wise adjusted
            "n_a": n_mice_by_g[a],
            "n_b": n_mice_by_g[b],
            "n_events_a": n_events_by_g[a],
            "n_events_b": n_events_by_g[b],
        })
    return {"F": float(F), "p": float(p_om)}, rows


# ---------------------------------------------------------------------------
# Per-event boxplot
# ---------------------------------------------------------------------------

def _state_metric_perevent_boxplot(ax, long_df, ylabel, title, stats_fn):
    """Boxplot on per-event values, with per-event jittered dots and a
    per-mouse-mean overlay. Stars from ``stats_fn``.

    The box summarizes the inferential unit (per-event values). Small
    per-event dots show within-mouse spread; the larger overlay dot per
    mouse marks the per-mouse mean for visual comparison with the
    legacy per-mouse-mean panel.
    """
    grp_order = [g for g in _PV_GROUP_ORDER
                 if (long_df["group"] == g).any()]
    if len(grp_order) == 0:
        ax.set_title(title, fontsize=9)
        return None, None

    bw = 0.60
    bp_kw = dict(patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", lw=1.2),
                 whiskerprops=dict(color="black", lw=0.6),
                 capprops=dict(color="black", lw=0.6))

    rng = np.random.default_rng(31)
    for gi, grp in enumerate(grp_order):
        sub = long_df[long_df["group"] == grp]
        ev_vals = sub["value"].to_numpy(dtype=float)
        bp = ax.boxplot([ev_vals], positions=[gi], widths=bw, **bp_kw)
        for patch in bp["boxes"]:
            patch.set_facecolor(_PV_BOX_COLORS[grp])
            patch.set_alpha(_PV_BOX_ALPHA)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        # Per-event dots — every event treated equally, grouped only by
        # treatment condition (no per-mouse aggregate overlay).
        jit = rng.uniform(-bw * 0.22, bw * 0.22, size=len(ev_vals))
        ax.scatter(np.full(len(ev_vals), gi) + jit, ev_vals,
                   color=_PV_DOT_COLORS.get(grp, "gray"),
                   s=_PV_SCATTER_SIZE, alpha=0.55, zorder=4,
                   edgecolors="white", linewidths=0.3)

    ax.set_xticks(range(len(grp_order)))
    ax.set_xticklabels([_PV_GROUP_LABELS.get(g, g) for g in grp_order],
                       fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.6)

    # Stats
    eligible = _eligible_groups_for_perevent(long_df)
    if len(eligible) < 2:
        return None, None
    omnibus, contrasts = stats_fn(
        long_df[long_df["group"].isin(eligible)].copy()
    )

    all_vals = long_df["value"].to_numpy(dtype=float)
    y_max = float(np.nanmax(all_vals))
    y_min = float(np.nanmin(all_vals))
    y_rng = max(y_max - y_min, 1e-6)
    bdy = 0.03 * y_rng
    bgap = 2.6 * bdy
    by = y_max + 0.04 * y_rng

    grp_pos = {g: i for i, g in enumerate(grp_order)}
    for c in contrasts:
        if c["a"] not in grp_pos or c["b"] not in grp_pos:
            continue
        star, fs = _pv_p_to_star(c["p_holm"], c["p_raw"])
        if star is None:
            continue
        x1 = grp_pos[c["a"]]
        x2 = grp_pos[c["b"]]
        if x1 > x2:
            x1, x2 = x2, x1
        _pv_draw_bracket(ax, x1, x2, by, bdy, y_rng, star, fs)
        by += bgap

    return omnibus, contrasts


def _format_perevent_stats_text(metric, ylabel, omnibus, contrasts,
                                long_df, stats_label):
    """Render the per-event sibling stats file text.

    *stats_label* ∈ {'LMM (mouse RE)', 'per-event Tukey HSD'}.
    """
    lines = []
    lines.append(f"Metric: {metric}")
    lines.append(f"Ylabel: {ylabel}")
    lines.append(f"Stats: {stats_label}")
    lines.append("")
    lines.append("Per-group counts and per-event summary:")
    for g in _PV_GROUP_ORDER:
        sub = long_df[long_df["group"] == g]
        if len(sub):
            v = sub["value"].to_numpy(dtype=float)
            lines.append(
                f"  {g:8s}  n_mice={sub['mouse'].nunique():2d}  "
                f"n_events={len(v):4d}  mean={np.mean(v):.4g}  "
                f"sd={np.std(v, ddof=1) if len(v) > 1 else float('nan'):.4g}  "
                f"median={np.median(v):.4g}"
            )
        else:
            lines.append(f"  {g:8s}  n_mice=0  n_events=0")
    lines.append("")
    if omnibus is None:
        lines.append("Insufficient data for omnibus / pairwise stats.")
        return "\n".join(lines) + "\n"

    lines.append(
        f"Omnibus (one-way): F={omnibus['F']:.4g}, p={omnibus['p']:.4g}"
    )
    lines.append("")
    if stats_label.startswith("LMM"):
        lines.append(
            "Pairwise Wald contrasts (Holm-corrected within metric):"
        )
        for c in contrasts:
            lines.append(
                f"  {c['a']:8s} vs {c['b']:8s}  "
                f"mean_diff={c['mean_diff']:+.4g}  se={c['se']:.4g}  "
                f"z={c['z']:+.3f}  "
                f"p_raw={c['p_raw']:.4g}  p_holm={c['p_holm']:.4g}  "
                f"(n_mice_{c['a']}={c['n_a']}, n_mice_{c['b']}={c['n_b']}, "
                f"n_events_{c['a']}={c['n_events_a']}, "
                f"n_events_{c['b']}={c['n_events_b']}, model={c['model']})"
            )
    else:  # per-event Tukey HSD
        lines.append("Pairwise per-event Tukey HSD (family-wise alpha=0.05):")
        lines.append(
            "  NOTE: ignores within-mouse clustering -> p values are "
            "anti-conservative when events within a mouse are correlated."
        )
        for c in contrasts:
            lines.append(
                f"  {c['a']:8s} vs {c['b']:8s}  "
                f"mean_diff={c['mean_diff']:+.4g}  p_adj={c['p_raw']:.4g}  "
                f"(n_mice_{c['a']}={c['n_a']}, n_mice_{c['b']}={c['n_b']}, "
                f"n_events_{c['a']}={c['n_events_a']}, "
                f"n_events_{c['b']}={c['n_events_b']})"
            )
    return "\n".join(lines) + "\n"


def _ylabel_for_perevent(metric):
    if metric.startswith("excursion_perShock_from_baseline"):
        return "Per-shock distance to baseline (a.u.)"
    if metric.startswith("excursion_perFrame_from_origin"):
        return "Per-frame ||x|| in shock (a.u.)"
    if metric.startswith("dist_paired_toneShock"):
        return "Per-trial tone↔shock distance (a.u.)"
    if metric.startswith("dist_allPairs_toneShock"):
        return "Tone × shock pairwise distance (a.u.)"
    if metric.startswith("dist_allPairs_tone_shock"):
        return "Per-tone × per-TFC-shock distance (a.u.)"
    if metric.startswith("dist_perTone_to_"):
        return "Per-tone distance to shock centroid (a.u.)"
    return metric


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _copy_methods_template(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, _METHODS_TEMPLATE_FILENAME)
    if os.path.isfile(dest_path):
        return
    src_path = os.path.join(_METHODS_TEMPLATES_DIR, _METHODS_TEMPLATE_FILENAME)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(
            f"PCA state-metrics METHODS template missing: {src_path}"
        )
    shutil.copy2(src_path, dest_path)


def _save_fig(fig, save_dir, fname, want_svg, auto_close):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=200,
                bbox_inches="tight")
    if want_svg:
        fig.savefig(os.path.join(save_dir, f"{fname}.svg"),
                    bbox_inches="tight")
    if auto_close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pca_state_metrics_pipeline(
    full_pca_by_mouse,
    mouse_groups,
    plots_dir,
    want_svg=False,
    auto_close=True,
):
    """Compute the per-mouse panel + group plots + stats files.

    Parameters
    ----------
    full_pca_by_mouse : dict[str, CrossregFullPCA]
        Toneshock CrossregFullPCA objects keyed by mouse id (already with
        ``.PCs`` populated by ``fit_pca(method=2)``).
    mouse_groups : dict[str, str]
        Mouse-id -> group string (must be one of mCherry / hM3D / hM4D).
    plots_dir : str
        Output directory; subdirectory ``PCA_state_metrics/`` will be created.
    want_svg, auto_close
        Plotting options.

    Returns
    -------
    summary : dict
        Keys: 'per_mouse_df', 'group_stats' (per-metric dict containing
        omnibus + contrasts).
    """
    save_dir = os.path.join(plots_dir, _OUT_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    _copy_methods_template(save_dir)

    per_mouse_rows = []
    for mouse, full_pca in full_pca_by_mouse.items():
        if mouse not in mouse_groups:
            raise KeyError(f"mouse_groups missing entry for {mouse!r}")
        group = mouse_groups[mouse]
        print(f"*** PCA state metrics: {mouse} {group} ...", flush=True)
        observed = compute_mouse_state_metrics(full_pca)
        row = {"mouse": mouse, "group": group}
        row.update(observed)
        per_mouse_rows.append(row)

    per_mouse_df = pd.DataFrame(per_mouse_rows)
    per_mouse_df.to_csv(os.path.join(save_dir, "per_mouse.csv"), index=False)

    # Group-level boxplots + stats files. Two passes:
    #   1. Welch t + Holm  (default; filename: <section>__<metric>.png)
    #   2. ANOVA + Tukey HSD (filename: <section>__<metric>_ANOVA.png)
    stats_passes = [
        ("Welch t + Holm",              _anova_holm_pairs,      ""),
        ("Tukey HSD",                   _anova_tukey_pairs,     "_ANOVA"),
        ("Permutation + percentile bootstrap", _bootstrap_perm_pairs,  "_BOOT"),
    ]
    group_stats = {}
    for stats_label, stats_fn, fname_suffix in stats_passes:
        for section, metrics in _METRICS_BY_SECTION.items():
            for metric in metrics:
                groups_data = {}
                for g in _PV_GROUP_ORDER:
                    vals = per_mouse_df.loc[per_mouse_df["group"] == g, metric] \
                        .to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    groups_data[g] = vals

                ylabel = _ylabel_for(metric)
                title = f"{section}: {metric}"
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                omnibus, contrasts = _state_metric_boxplot(
                    ax, groups_data, ylabel=ylabel, title=title,
                    stats_fn=stats_fn,
                )
                fname = f"{section.replace(' ', '_')}__{metric}{fname_suffix}"
                _save_fig(fig, save_dir, fname, want_svg, auto_close)

                stats_path = os.path.join(save_dir, f"{fname}__stats.txt")
                with open(stats_path, "w", encoding="utf-8") as fh:
                    fh.write(_format_stats_text(
                        metric, ylabel, omnibus, contrasts, groups_data,
                        stats_label=stats_label,
                    ))
                if not fname_suffix:
                    group_stats[metric] = {
                        "omnibus": omnibus, "contrasts": contrasts
                    }

    # ----------------------------------------------------------------- 
    # Per-event pipeline (parallel sub-directories per pairing scheme)
    # -----------------------------------------------------------------
    perevent_summary = {}
    for scheme in _PEREVENT_SCHEMES:
        scheme_dir = os.path.join(save_dir, f"per_event_{scheme}")
        os.makedirs(scheme_dir, exist_ok=True)
        _copy_methods_template(scheme_dir)

        # Build long-form df by concatenating per-mouse rows
        all_rows = []
        for mouse, full_pca in full_pca_by_mouse.items():
            print(f"*** PCA per-event metrics ({scheme}): {mouse} ...",
                  flush=True)
            rows = compute_mouse_event_metrics(full_pca, scheme)
            for r in rows:
                r["group"] = mouse_groups[mouse]
            all_rows.extend(rows)
        long_df = pd.DataFrame(all_rows)
        long_df.to_csv(os.path.join(scheme_dir, "per_event_long.csv"),
                       index=False)

        perevent_passes = [
            ("LMM (mouse RE)",      _lmm_holm_pairs,             "_LMM"),
            ("per-event Tukey HSD", _perevent_anova_tukey_pairs, "_ANOVA"),
        ]
        scheme_stats = {}
        metrics_by_section = _perevent_metrics_by_section(scheme)
        for stats_label, stats_fn, fname_suffix in perevent_passes:
            for section, metrics in metrics_by_section.items():
                for metric in metrics:
                    sub = long_df[long_df["metric"] == metric]
                    if sub.empty:
                        raise RuntimeError(
                            f"per-event ({scheme}): no rows for metric "
                            f"{metric!r}; mice present in long_df: "
                            f"{sorted(long_df['mouse'].unique())}"
                        )
                    ylabel = _ylabel_for_perevent(metric)
                    title = f"{section}: {metric} [{scheme}]"
                    fig, ax = plt.subplots(figsize=(3.4, 3.4))
                    omnibus, contrasts = _state_metric_perevent_boxplot(
                        ax, sub.copy(), ylabel=ylabel, title=title,
                        stats_fn=stats_fn,
                    )
                    fname = (f"{section.replace(' ', '_')}__{metric}"
                             f"_perEvent{fname_suffix}")
                    _save_fig(fig, scheme_dir, fname, want_svg, auto_close)

                    stats_path = os.path.join(
                        scheme_dir, f"{fname}__stats.txt"
                    )
                    with open(stats_path, "w", encoding="utf-8") as fh:
                        fh.write(_format_perevent_stats_text(
                            metric, ylabel, omnibus, contrasts, sub,
                            stats_label=stats_label,
                        ))
                    if fname_suffix == "_LMM":
                        scheme_stats[metric] = {
                            "omnibus": omnibus, "contrasts": contrasts,
                        }
        perevent_summary[scheme] = {
            "long_df": long_df,
            "metric_stats_lmm": scheme_stats,
        }

    return {
        "per_mouse_df": per_mouse_df,
        "group_stats": group_stats,
        "per_event": perevent_summary,
    }


def _all_metric_names():
    return [m for metrics in _METRICS_BY_SECTION.values() for m in metrics]


def _ylabel_for(metric):
    if metric.startswith("excursion_from_baseline"):
        return "Shock-to-baseline distance (a.u.)"
    if metric.startswith("excursion_from_origin"):
        return "Mean ||x|| of shock frames (a.u.)"
    if metric.startswith("dist_"):
        return "Euclidean distance (a.u.)"
    return metric


print("caban.pca_state_metrics.py loaded.")

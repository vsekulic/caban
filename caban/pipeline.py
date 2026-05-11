"""caban analysis pipeline wrappers.

Thin wrappers that take ``(ds, cfg)`` and dispatch to the existing
analysis modules (``caban.population``, ``caban.isomap``,
``caban.epoch_analysis``, ``caban.spatial``, ``caban.decoder``...).
The goal is **not** to re-implement anything — only to consolidate the
notebook-vs-loader boundary so the notebook can call clean entry points
that thread ``cfg`` values through.

What's wrapped here:
    - ``resolve_continuity_params``  — runs the velocity-stats +
      continuity-preset block and returns the resolved sigma parameters.
    - ``engram_idx_by_mouse``       — same helper main.py L5749 uses
      to convert ``engram_id`` into ``{mouse: ndarray}``.
    - ``run_population_pca``        — wraps ``run_population_pca_pipeline``
      + ``run_pca_state_metrics_from_results``.
    - ``run_isomap``                — wraps ``run_isomap_pipeline``.
    - ``run_epoch_pv``              — wraps the within-session epoch PV
      sweep (replaces the inline ``plot_epoch_pv_analysis`` block).
    - ``run_cross_session_epoch_pv``— wraps the cross-session sweep.

Decoder wrappers (``run_2D_decoder``, ``run_2D_PF_decoder``) plus their
paramset builders (``build_raw_paramset``, ``build_pf_paramset``) live
here too. They replace the global-leaning ``run_2D_decoder_all_mice``
/ ``run_2D_PF_decoder_all_mice`` wrappers that used to sit at the top
of ``caban/main.py`` and read module-level switches. Inline decoder
paradigm A–F blocks in the notebook should call ``run_2D_decoder(cfg,
raw_params, train_sessions, test_targets, **kwargs)`` instead of the
bare ``run_2D_decoder_all_mice``.
"""

from __future__ import annotations

import os
import importlib
from types import SimpleNamespace
from typing import Optional

import numpy as np

# Bootstrap decoder first to break the caban.analysis <-> caban.decoder
# circular import (same trick caban/main.py uses at lines 7-8).
import caban.decoder as _caban_decoder_bootstrap
importlib.reload(_caban_decoder_bootstrap)

from caban.utilities import (  # noqa: F401
    msg_start, msg_end, NPY_SAVE_PATH, Saver,
)
from caban.decoder import (
    BayesianDecoderParamset,
    plot_velocity_histograms,
    collect_velocity_stats,
    compute_continuity_params,
    save_continuity_params_txt,
    run_2D_decoder_all_mice as _run_2D_decoder_all_mice_base,
    run_2D_PF_decoder_all_mice as _run_2D_PF_decoder_all_mice_base,
)
from caban.population import (
    run_population_pca_pipeline,
    run_pca_state_metrics_from_results,
    EXCLUDE_MICE_CROSSREG as _PCA_EXCLUDE,
)
from caban.isomap import run_isomap_pipeline
from caban.epoch_analysis import (
    run_epoch_analysis_all_mice,
    run_cross_session_epoch_analysis_all_mice,
)
from caban.engram import ENGRAM_REFERENCE
from caban.engram_sanity import plot_engram_sanity

from caban.config import PipelineConfig


# ---------------------------------------------------------------------------
def resolve_continuity_params(ds: SimpleNamespace, cfg: PipelineConfig,
                              *, save_plot: bool = True) -> dict:
    """Resolve the continuity-constraint sigma parameters from ``cfg``.

    Runs velocity-stats collection across all open-field sessions in
    ``ds`` and, when ``cfg.continuity_preset == 'data-driven'``, computes
    the per-arena sigma parameters via ``compute_continuity_params``. For
    every other preset, returns the canonical preset constants.

    Side effects: when ``save_plot=True`` the velocity histogram figure is
    written to ``ds.PLOTS_DIR/TFC_2D_decoding/`` (matching main.py).

    Returns
    -------
    dict
        Keys: ``use_continuity_constraint``, ``continuity_sigma_k``,
        ``continuity_speed_ref``, ``continuity_exp``, ``continuity_sigma_min``,
        ``continuity_sigma_max``, ``continuity_sigma_default``,
        plus ``vel_stats_TFC`` (the raw velocity stats dict for downstream use).
    """
    out_dir = os.path.join(ds.PLOTS_DIR, "TFC_2D_decoding")
    os.makedirs(out_dir, exist_ok=True)

    vel_session_dicts = []
    vel_tags = []
    for tag, sd in [("TFC_cond", ds.TFC_cond), ("Test_A", ds.Test_A),
                    ("Test_A_1wk", ds.Test_A_1wk),
                    ("Test_B", ds.Test_B), ("Test_B_1wk", ds.Test_B_1wk)]:
        if sd:
            vel_session_dicts.append(sd)
            vel_tags.append(tag)

    if vel_session_dicts:
        if save_plot:
            vel_stats_TFC = plot_velocity_histograms(
                *vel_session_dicts,
                mouse_groups=ds.mouse_groups,
                PLOTS_DIR=out_dir,
                session_tags=tuple(vel_tags),
                auto_close=True,
            )
        else:
            vel_stats_TFC = collect_velocity_stats(ds.TFC_cond)
    else:
        vel_stats_TFC = collect_velocity_stats(ds.TFC_cond)

    preset = cfg.continuity_preset
    if preset == "data-driven":
        dd = compute_continuity_params(vel_stats_TFC, time_bin_frames=cfg.TIME_BIN_FRAMES)
        save_continuity_params_txt(
            dd, out_dir,
            tag=f"TFC data-driven (time_bin_frames={cfg.TIME_BIN_FRAMES})",
            time_bin_frames=cfg.TIME_BIN_FRAMES,
        )
        params = {
            'use_continuity_constraint': True,
            'continuity_sigma_k': dd['continuity_sigma_k'],
            'continuity_speed_ref': dd['continuity_speed_ref'],
            'continuity_exp': dd['continuity_exp'],
            'continuity_sigma_min': dd['continuity_sigma_min'],
            'continuity_sigma_max': dd['continuity_sigma_max'],
            'continuity_sigma_default': dd['continuity_sigma_default'],
        }
    elif preset == "conservative":
        params = {'use_continuity_constraint': True, 'continuity_sigma_k': 45.0,
                  'continuity_speed_ref': 20.0, 'continuity_exp': 1.0,
                  'continuity_sigma_min': 15.0, 'continuity_sigma_max': 45.0,
                  'continuity_sigma_default': 25.0}
    elif preset == "balanced":
        params = {'use_continuity_constraint': True, 'continuity_sigma_k': 60.0,
                  'continuity_speed_ref': 20.0, 'continuity_exp': 1.0,
                  'continuity_sigma_min': 20.0, 'continuity_sigma_max': 60.0,
                  'continuity_sigma_default': 30.0}
    elif preset == "aggressive":
        params = {'use_continuity_constraint': True, 'continuity_sigma_k': 80.0,
                  'continuity_speed_ref': 20.0, 'continuity_exp': 1.0,
                  'continuity_sigma_min': 25.0, 'continuity_sigma_max': 80.0,
                  'continuity_sigma_default': 40.0}
    elif preset == "custom":
        # Honor the user's hand-set values from cfg.
        params = {
            'use_continuity_constraint': cfg.use_continuity_constraint,
            'continuity_sigma_k': cfg.continuity_sigma_k,
            'continuity_speed_ref': cfg.continuity_speed_ref,
            'continuity_exp': cfg.continuity_exp,
            'continuity_sigma_min': cfg.continuity_sigma_min,
            'continuity_sigma_max': cfg.continuity_sigma_max,
            'continuity_sigma_default': cfg.continuity_sigma_default,
        }
    else:
        raise ValueError(
            f"Invalid continuity_preset={preset!r}. "
            f"Use one of: 'data-driven', 'conservative', 'balanced', "
            f"'aggressive', 'custom'."
        )

    params['vel_stats_TFC'] = vel_stats_TFC
    return params


# ---------------------------------------------------------------------------
def engram_idx_by_mouse(ds: SimpleNamespace, cfg: PipelineConfig,
                        mode: str) -> dict:
    """Build ``{mouse: ndarray of engram row indices}`` for a given mode.

    Mirrors the inline helper in main.py L5749. Honors the population-PCA
    exclude list from ``caban.population.EXCLUDE_MICE_CROSSREG``.
    """
    out = {}
    for m in ds.mouse_groups:
        if m in _PCA_EXCLUDE:
            continue
        if m not in ds.engram_id or cfg.which_engram not in ds.engram_id[m]:
            continue
        out[m] = np.where(ds.engram_id[m][cfg.which_engram][mode])[0]
    return out


# ---------------------------------------------------------------------------
def run_engram_sanity_plots(ds: SimpleNamespace, cfg: PipelineConfig,
                             *, save_root: Optional[str] = None) -> None:
    """Generate the engram-sanity panels under ``PLOTS_DIR/engram_plots``."""
    save_root = save_root or os.path.join(ds.PLOTS_DIR, "engram_plots")
    print(f"*** Engram sanity plots -> {save_root}", flush=True)
    plot_engram_sanity(
        engram_id=ds.engram_id,
        engram_rates=ds.engram_rates,
        engram_norms=ds.engram_norms,
        ref_sessions_by_etype={'encoding': ds.TFC_cond, 'recall': ds.Test_B},
        other_sessions_by_name={'Test_B_1wk': ds.Test_B_1wk},
        mouse_groups=ds.mouse_groups,
        save_root=save_root,
        n_cells=cfg.NUM_ENGRAM_PLOT_CELLS,
        modes=cfg.ENGRAM_MODES,
        etypes=('encoding', 'recall'),
    )


# ---------------------------------------------------------------------------
def run_population_pca(ds: SimpleNamespace, cfg: PipelineConfig, *,
                       pop_plots_dir: str,
                       want_engram: bool = True,
                       engram_mode: Optional[str] = None,
                       normalize: bool = False,
                       binarize: bool = False,
                       frames_per_bin: Optional[int] = None,
                       also_state_metrics: bool = True):
    """Run Population PCA + (optionally) state-metrics for one mode.

    Mirrors ``_run_population_pca_and_metrics`` from main.py L5760.
    """
    if frames_per_bin is None:
        frames_per_bin = cfg.PCA_FRAMES_PER_BIN

    if want_engram:
        if engram_mode is None:
            raise RuntimeError(
                "run_population_pca(want_engram=True) requires engram_mode."
            )
        eidx = engram_idx_by_mouse(ds, cfg, engram_mode)
        suffix = ""
        if normalize:
            suffix += " [normalize]"
        if binarize:
            suffix += " [binarize]"
        suffix += f" [fpb={frames_per_bin}]"
        print(f"\n*** Population PCA pipeline - etype={cfg.which_engram} "
              f"mode={engram_mode}{suffix} -> {pop_plots_dir}", flush=True)
    else:
        eidx = None
        suffix = ""
        if normalize:
            suffix += " [normalize]"
        if binarize:
            suffix += " [binarize]"
        suffix += f" [fpb={frames_per_bin}]"
        print(f"\n*** Population PCA pipeline - all crossreg cells"
              f"{suffix} -> {pop_plots_dir}", flush=True)

    res = run_population_pca_pipeline(
        TFC_cond=ds.TFC_cond,
        Test_B=ds.Test_B,
        Test_B_1wk=ds.Test_B_1wk,
        mouse_groups=ds.mouse_groups,
        TFC_B_B_1wk_crossreg=ds.TFC_B_B_1wk_crossreg,
        mapping_TFC_cond_Test_B_Test_B_1wk=ds.mapping_TFC_cond_Test_B_Test_B_1wk,
        PLOTS_DIR=pop_plots_dir,
        NPY_SAVE_PATH=NPY_SAVE_PATH,
        auto_close=True,
        want_engram=want_engram,
        engram_idx_by_mouse=eidx,
        which_engram=cfg.which_engram,
        normalize=normalize,
        binarize=binarize,
        frames_per_bin=frames_per_bin,
    )
    if also_state_metrics:
        run_pca_state_metrics_from_results(
            results=res,
            mouse_groups=ds.mouse_groups,
            PLOTS_DIR=pop_plots_dir,
            auto_close=True,
        )
    return res


def run_population_pca_all_modes(ds: SimpleNamespace, cfg: PipelineConfig
                                  ) -> dict:
    """Convenience: run the full Population PCA matrix (crossreg variants
    + every engram mode) the same way main.py does at L5700-5860.

    Returns a dict with the various results dicts keyed by variant.
    """
    msg_start('*** Population PCA Trajectory Analysis')
    PLOTS_DIR = ds.PLOTS_DIR

    out: dict = {}
    out['crossreg'] = run_population_pca(
        ds, cfg,
        pop_plots_dir=os.path.join(PLOTS_DIR, "Population_PCA_crossreg"),
        want_engram=False,
    )
    out['crossreg_normalize'] = run_population_pca(
        ds, cfg,
        pop_plots_dir=os.path.join(PLOTS_DIR, "Population_PCA_crossreg_normalize"),
        want_engram=False, normalize=True,
    )
    out['crossreg_binarize'] = run_population_pca(
        ds, cfg,
        pop_plots_dir=os.path.join(PLOTS_DIR, "Population_PCA_crossreg_binarize"),
        want_engram=False, binarize=True,
    )
    out['by_mode'] = {}
    for em in cfg.ENGRAM_MODES:
        plots_dir = os.path.join(
            PLOTS_DIR,
            ("Population_PCA_engram" if cfg.want_engram_pop_pca
             else "Population_PCA_noengram")
            + (f"_{cfg.which_engram}_{em}" if cfg.want_engram_pop_pca else ""),
        )
        out['by_mode'][em] = run_population_pca(
            ds, cfg,
            pop_plots_dir=plots_dir,
            want_engram=cfg.want_engram_pop_pca,
            engram_mode=em,
        )
    msg_end()
    return out


# ---------------------------------------------------------------------------
def run_isomap(ds: SimpleNamespace, cfg: PipelineConfig, *,
               plots_subdir: str = "TFC_Isomap",
               exclude_mice: tuple = ("G07", "G15")) -> dict:
    """Run the Isomap manifold pipeline (main.py L5863)."""
    msg_start('*** Isomap Manifold Analysis')
    plots_dir = os.path.join(ds.PLOTS_DIR, plots_subdir)
    iso_groups = {m: g for m, g in ds.mouse_groups.items()
                  if m not in exclude_mice}
    res = run_isomap_pipeline(
        TFC_cond=ds.TFC_cond,
        Test_B=ds.Test_B,
        Test_B_1wk=ds.Test_B_1wk,
        mouse_groups=iso_groups,
        mice_per_group_Test_B_B_1wk=ds.mice_per_group_Test_B_B_1wk,
        TFC_B_B_1wk_crossreg=ds.TFC_B_B_1wk_crossreg,
        mapping_TFC_cond_Test_B_Test_B_1wk=ds.mapping_TFC_cond_Test_B_Test_B_1wk,
        PLOTS_DIR=plots_dir,
        NPY_SAVE_PATH=NPY_SAVE_PATH,
        auto_close=True,
    )
    msg_end()
    return res


# ---------------------------------------------------------------------------
def run_epoch_pv(ds: SimpleNamespace, cfg: PipelineConfig, *,
                 matrix_types=('S', 'C'),
                 aggregations=('integral', 'mean'),
                 use_event_rate: bool = True,
                 metrics=('pearson', 'cosine'),
                 mappings=None,
                 peri_shock_pre_s: float = 10.0,
                 peri_shock_post_s: float = 10.0,
                 pre_tone_duration_s: float = 35.0,
                 mobility_filters=(None, 'mobile', 'immobile')) -> None:
    """Within-session epoch-PV similarity + RDM sweep (main.py L5560)."""
    msg_start('*** Epoch PV similarity + RDM analysis (TFC_cond)')
    if mappings is None:
        mappings = [ds.mapping_FULL]

    data_modes = [f'{mt}_{agg}' for mt in matrix_types for agg in aggregations]
    if use_event_rate:
        data_modes.append('event_rate')

    for mapping in mappings:
        for dm in data_modes:
            for mob in mobility_filters:
                run_epoch_analysis_all_mice(
                    PLOTS_DIR=ds.PLOTS_DIR,
                    mice_per_group=ds.mice_per_group,
                    TFC_cond=ds.TFC_cond,
                    TFC_cond_crossreg=ds.TFC_cond_crossreg,
                    mapping=mapping,
                    data_mode=dm,
                    metrics=metrics,
                    epochs=None,
                    peri_shock_pre_s=peri_shock_pre_s,
                    peri_shock_post_s=peri_shock_post_s,
                    pre_tone_duration_s=pre_tone_duration_s,
                    mobility_filter=mob,
                    auto_close=True,
                )
    msg_end()


def run_cross_session_epoch_pv(ds: SimpleNamespace, cfg: PipelineConfig, *,
                                matrix_types=('S', 'C'),
                                aggregations=('integral', 'mean'),
                                use_event_rate: bool = True,
                                metrics=('pearson', 'cosine'),
                                mobility_filters=(None, 'mobile', 'immobile'),
                                peri_shock_pre_s: float = 10.0,
                                peri_shock_post_s: float = 10.0,
                                pre_tone_duration_s: float = 35.0,
                                post_tone_duration_s: float = 35.0) -> None:
    """Cross-session epoch-PV similarity sweep (main.py L5630-5705)."""
    msg_start('*** Cross-session Epoch PV similarity (TFC_cond vs recall)')

    map_B = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    map_A = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    map_all = ds.mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk

    data_modes = [f'{mt}_{agg}' for mt in matrix_types for agg in aggregations]
    if use_event_rate:
        data_modes.append('event_rate')

    recall_configs = [
        ('Test_B', ds.Test_B, 'testb', [
            (map_B, ds.TFC_B_B_1wk_crossreg),
            (map_all, ds.TFC_AB_48hr_1wk_crossreg),
        ]),
        ('Test_B_1wk', ds.Test_B_1wk, 'testb', [
            (map_B, ds.TFC_B_B_1wk_crossreg),
            (map_all, ds.TFC_AB_48hr_1wk_crossreg),
        ]),
        ('Test_A', ds.Test_A, 'testa', [
            (map_A, ds.TFC_A_A_1wk_crossreg),
            (map_all, ds.TFC_AB_48hr_1wk_crossreg),
        ]),
        ('Test_A_1wk', ds.Test_A_1wk, 'testa', [
            (map_A, ds.TFC_A_A_1wk_crossreg),
            (map_all, ds.TFC_AB_48hr_1wk_crossreg),
        ]),
    ]

    for label, sessions, rtype, mappings_list in recall_configs:
        for mapping, xreg_dict in mappings_list:
            for dm in data_modes:
                for mob in mobility_filters:
                    run_cross_session_epoch_analysis_all_mice(
                        PLOTS_DIR=ds.PLOTS_DIR,
                        mice_per_group=ds.mice_per_group,
                        TFC_cond=ds.TFC_cond,
                        recall_sessions=sessions,
                        recall_label=label,
                        recall_type=rtype,
                        crossreg_dict=xreg_dict,
                        mapping=mapping,
                        data_mode=dm,
                        metrics=metrics,
                        peri_shock_pre_s=peri_shock_pre_s,
                        peri_shock_post_s=peri_shock_post_s,
                        pre_tone_duration_s=pre_tone_duration_s,
                        post_tone_duration_s=post_tone_duration_s,
                        mobility_filter=mob,
                        auto_close=True,
                    )
    msg_end()


# ===========================================================================
# 2D Bayesian decoder wrappers
# ===========================================================================
#
# These replace the global-state-leaking wrappers that used to sit at the top
# of ``caban/main.py`` (lines ~3418/3477). Call signature change:
#
#     # Old (main.py module-globals):
#     run_2D_decoder_all_mice(train, test, train_label=..., session_str=...)
#
#     # New (explicit cfg + paramset):
#     from caban.pipeline import build_raw_paramset, run_2D_decoder
#     raw_params = build_raw_paramset(cfg, continuity)
#     run_2D_decoder(cfg, raw_params, train, test,
#                    train_label=..., session_str=..., PLOTS_DIR=ds.PLOTS_DIR)
#
# Notebook decoder paradigm cells should reach for these.

def build_raw_paramset(cfg: PipelineConfig, continuity: dict
                        ) -> BayesianDecoderParamset:
    """Construct the raw-S 2D decoder paramset from cfg + resolved continuity."""
    return BayesianDecoderParamset(
        n_spatial_bins=cfg.N_SPATIAL_BINS,
        time_bin_frames=cfg.TIME_BIN_FRAMES,
        use_posterior_mean=cfg.use_posterior_mean,
        use_z_score=cfg.use_z_score,
        use_continuity_constraint=continuity['use_continuity_constraint'],
        continuity_sigma_k=continuity['continuity_sigma_k'],
        continuity_speed_ref=continuity['continuity_speed_ref'],
        continuity_exp=continuity['continuity_exp'],
        continuity_sigma_min=continuity['continuity_sigma_min'],
        continuity_sigma_max=continuity['continuity_sigma_max'],
        continuity_sigma_default=continuity['continuity_sigma_default'],
    )


def build_pf_paramset(cfg: PipelineConfig, continuity: dict
                       ) -> BayesianDecoderParamset:
    """Construct the PF 2D decoder paramset from cfg + resolved continuity."""
    return BayesianDecoderParamset(
        time_bin_frames=cfg.TIME_BIN_FRAMES,
        use_posterior_mean=cfg.use_posterior_mean,
        use_z_score=cfg.use_z_score,
        use_continuity_constraint=continuity['use_continuity_constraint'],
        continuity_sigma_k=continuity['continuity_sigma_k'],
        continuity_speed_ref=continuity['continuity_speed_ref'],
        continuity_exp=continuity['continuity_exp'],
        continuity_sigma_min=continuity['continuity_sigma_min'],
        continuity_sigma_max=continuity['continuity_sigma_max'],
        continuity_sigma_default=continuity['continuity_sigma_default'],
        place_cells_only=cfg.place_cells_only,
        use_pf_num=cfg.use_pf_num,
        use_occupancy_fallback=cfg.use_occupancy_fallback,
    )


# --- Population-curve summary helpers (moved from caban/main.py) -----------

def _count_csv_rows(csv_path: str):
    """Count data rows (excluding header) in a CSV; return None if missing."""
    try:
        if not os.path.exists(csv_path):
            return None
        with open(csv_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        return max(0, n_lines - 1)
    except Exception:
        return None


def _z_score_dir_tag(use_z_score_mode: str) -> str:
    mode = str(use_z_score_mode).strip().lower()
    if mode in ("per-session", "per_session", "per"):
        return "z-score_per"
    if mode in ("across-sessions", "across_sessions", "across"):
        return "z-score_across"
    return "z-score_none"


def _decoder_is_ridge(decoder_type_mode: str) -> bool:
    dt = str(decoder_type_mode).strip().lower()
    return dt in ("ridge", "ridge-regression", "ridge_regression")


def _print_population_curve_summary(cfg: PipelineConfig, kwargs: dict,
                                    *, pf_variant: bool) -> None:
    """End-of-run summary of population-curve output paths and row counts."""
    if not bool(kwargs.get("enable_population_curve", False)):
        return

    plots_dir = str(kwargs.get("PLOTS_DIR", ""))
    train_label = str(kwargs.get("train_label", "TFC_cond"))
    session_str = str(kwargs.get("session_str", ""))
    ztag = _z_score_dir_tag(kwargs.get("use_z_score", cfg.use_z_score))
    dec_type = kwargs.get("decoder_type", cfg.decoder_type)
    suffix = str(kwargs.get("population_curve_suffix", cfg.population_curve_suffix))
    shuffle_enabled = bool(kwargs.get("enable_shuffle_control", False))
    shuffle_suffix = "_shuffle" if shuffle_enabled else ""

    if pf_variant:
        pf_str = "pcells" if bool(kwargs.get("place_cells_only", True)) else "allcells"
        decoder_dir_tag = "ridge_2D_PF" if _decoder_is_ridge(dec_type) else "bayes_2D_PF"
        save_path = os.path.join(
            plots_dir,
            f"{decoder_dir_tag}_multitgt_train_{train_label}_{session_str}_{pf_str}_{ztag}{suffix}{shuffle_suffix}",
        )
        tables_dir = os.path.join(save_path, "population_curve_tables_pf")
        tag = "[PIPE->2D-PF]"
    else:
        decoder_dir_tag = "ridge_2D" if _decoder_is_ridge(dec_type) else "bayes_2D"
        save_path = os.path.join(
            plots_dir,
            f"{decoder_dir_tag}_multitgt_train_{train_label}_{session_str}_{ztag}{suffix}{shuffle_suffix}",
        )
        tables_dir = os.path.join(save_path, "population_curve_tables")
        tag = "[PIPE->2D]"

    repeat_csv = os.path.join(tables_dir, "population_curve_repeat_level.csv")
    summary_csv = os.path.join(tables_dir, "population_curve_summary_level.csv")
    status_txt = os.path.join(tables_dir, "population_curve_run_status.txt")
    n_repeat = _count_csv_rows(repeat_csv)
    n_summary = _count_csv_rows(summary_csv)
    status_exists = os.path.exists(status_txt)

    print(
        f"{tag} population-curve outputs: tables_dir={tables_dir} "
        f"repeat_rows={n_repeat if n_repeat is not None else 'NA'} "
        f"summary_rows={n_summary if n_summary is not None else 'NA'} "
        f"status_file={'yes' if status_exists else 'no'}"
    )


def _inject_popcurve_and_shuffle_defaults(cfg: PipelineConfig, kwargs: dict,
                                          *, opt_tag: str) -> None:
    """Inject cfg-derived population-curve and shuffle defaults into kwargs."""
    kwargs.setdefault("opt_tag", opt_tag)
    kwargs.setdefault("enable_population_curve", cfg.enable_population_curve)
    kwargs.setdefault("population_curve_n_values", cfg.population_curve_n_values)
    kwargs.setdefault("population_curve_n_values_max_shared", cfg.population_curve_n_values_max_shared)
    kwargs.setdefault("population_curve_per_pair_grid", cfg.population_curve_per_pair_grid)
    kwargs.setdefault("population_curve_n_step", cfg.population_curve_n_step)
    kwargs.setdefault("population_curve_quantile", cfg.population_curve_quantile)
    kwargs.setdefault("population_curve_repeats", cfg.population_curve_repeats)
    kwargs.setdefault("population_curve_seed", cfg.population_curve_seed)
    kwargs.setdefault("population_curve_metric", cfg.population_curve_metric)
    kwargs.setdefault("manual_killswitch", cfg.manual_killswitch)
    kwargs.setdefault("killswitch_prompt_every", cfg.killswitch_prompt_every)
    kwargs.setdefault("population_curve_print_level", cfg.population_curve_print_level)
    kwargs.setdefault("population_curve_suffix", cfg.population_curve_suffix)
    kwargs.setdefault("population_curve_load_cached", cfg.population_curve_load_cached)

    # If caller explicitly passed None for these, fall back to cfg defaults.
    for k, default in [
        ("population_curve_n_values", cfg.population_curve_n_values),
        ("population_curve_n_step", cfg.population_curve_n_step),
        ("population_curve_n_values_max_shared", cfg.population_curve_n_values_max_shared),
        ("population_curve_quantile", cfg.population_curve_quantile),
        ("population_curve_repeats", cfg.population_curve_repeats),
    ]:
        if kwargs.get(k, None) is None:
            kwargs[k] = default

    # Shuffle-control: LT-family vs TFC-family decided from labels.
    train_label = str(kwargs.get("train_label", "")).lower()
    session_str = str(kwargs.get("session_str", "")).lower()
    is_lt_family = ("lt" in train_label) or ("lt" in session_str)
    kwargs.setdefault(
        "enable_shuffle_control",
        cfg.enable_lt_shuffle_control if is_lt_family else cfg.enable_tfc_shuffle_control,
    )
    kwargs.setdefault("shuffle_type", cfg.decoder_shuffle_type)
    kwargs.setdefault("n_shuffles", cfg.decoder_shuffle_n_repeats)
    kwargs.setdefault("shuffle_seed", cfg.decoder_shuffle_seed)


def run_2D_decoder(cfg: PipelineConfig,
                   raw_params: BayesianDecoderParamset,
                   train_sessions, test_targets, **kwargs):
    """Raw-S 2D Bayesian decoder wrapper. Replaces ``caban.main.run_2D_decoder_all_mice``."""
    raw_keys = [k for k in raw_params.__slots__
                if k not in ('place_cells_only', 'use_pf_num', 'use_occupancy_fallback')]
    raw_params.inject_into(kwargs, keys=raw_keys,
                            aliases={'n_spatial_bins': ['n_x_bins', 'n_y_bins']})
    _inject_popcurve_and_shuffle_defaults(cfg, kwargs,
                                          opt_tag=cfg.optimization_param_set)

    print(
        "[PIPE->2D] population-curve config: "
        f"enable={kwargs.get('enable_population_curve')} "
        f"n_values={kwargs.get('population_curve_n_values')} "
        f"max_shared={kwargs.get('population_curve_n_values_max_shared')} "
        f"require_exact_mapping={kwargs.get('require_exact_mapping', False)} "
        f"n_step={kwargs.get('population_curve_n_step')} "
        f"quantile={kwargs.get('population_curve_quantile')} "
        f"repeats={kwargs.get('population_curve_repeats')}"
    )
    results = _run_2D_decoder_all_mice_base(train_sessions, test_targets, **kwargs)
    _print_population_curve_summary(cfg, kwargs, pf_variant=False)
    return results


def run_2D_PF_decoder(cfg: PipelineConfig,
                      pf_params: BayesianDecoderParamset,
                      train_sessions, test_targets, **kwargs):
    """PF 2D Bayesian decoder wrapper. Replaces ``caban.main.run_2D_PF_decoder_all_mice``."""
    # Exclude n_spatial_bins (PF decoder uses place-field grid) and use_z_score
    # (rate maps are on original scale; z-scoring would break the Poisson likelihood).
    pf_keys = [k for k in pf_params.__slots__
               if k not in ('n_spatial_bins', 'use_z_score')]
    pf_params.inject_into(kwargs, keys=pf_keys)
    _inject_popcurve_and_shuffle_defaults(cfg, kwargs,
                                          opt_tag=cfg.optimization_pf_param_set)

    print(
        "[PIPE->2D-PF] population-curve config: "
        f"enable={kwargs.get('enable_population_curve')} "
        f"n_values={kwargs.get('population_curve_n_values')} "
        f"max_shared={kwargs.get('population_curve_n_values_max_shared')} "
        f"require_exact_mapping={kwargs.get('require_exact_mapping', False)} "
        f"n_step={kwargs.get('population_curve_n_step')} "
        f"quantile={kwargs.get('population_curve_quantile')} "
        f"repeats={kwargs.get('population_curve_repeats')}"
    )
    results = _run_2D_PF_decoder_all_mice_base(train_sessions, test_targets, **kwargs)
    _print_population_curve_summary(cfg, kwargs, pf_variant=True)
    return results


def popcurve_dirname(cfg: PipelineConfig, name: str) -> str:
    """Append population-curve suffix to run directory names only when enabled."""
    if cfg.enable_population_curve:
        return f"{name}{cfg.population_curve_suffix}"
    return name

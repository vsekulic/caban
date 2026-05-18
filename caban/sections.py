"""caban analysis sections — one function per top-level analysis in caban/main.py.

Each function takes ``(ds, cfg)`` (plus any cross-section state via
keyword arguments) and runs one block of the historical caban/main.py
pipeline. Bodies are the verbatim main.py code with ds/cfg names
unpacked into locals at the top.

Cross-section state (``raw_params``, ``pf_params``, ``mt_A_results_2D``,
``mt_B_results_2D``, ``mt_C_results_2D``, ``lt_cont_pvt``) is returned
by the section that produces it and passed into downstream sections as
keyword arguments.
"""
from __future__ import annotations

import os, sys, json, time, pickle, traceback, importlib, warnings
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap

# Star-imports mirror caban/main.py top-of-file so every plot_/process_
# helper resolves at module scope without per-function imports.
import caban.decoder as _caban_decoder_bootstrap  # break circular import
importlib.reload(_caban_decoder_bootstrap)
from caban.utilities import *  # noqa: F401,F403
from caban.sessions import *   # noqa: F401,F403
from caban.analysis import *   # noqa: F401,F403
from caban.decoder import *    # noqa: F401,F403
from caban.decoder import _PVT_MODE_LABEL_TO_KEY, _copy_analysis_methods_template  # noqa: F401
from caban.engram import ENGRAM_REFERENCE
from caban.population import EXCLUDE_MICE_CROSSREG as _PCA_EXCLUDE
from caban.population import run_population_pca_pipeline, run_pca_state_metrics_from_results
from caban.isomap import run_isomap_pipeline
from caban.epoch_analysis import run_epoch_analysis_all_mice, run_cross_session_epoch_analysis_all_mice
from caban.engram_sanity import plot_engram_sanity

import statsmodels.api as sm  # noqa: F401
from statsmodels.regression.mixed_linear_model import MixedLM as mixedlm  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

from caban.config import PipelineConfig  # noqa: F401

# ---------------------------------------------------------------------------
# Shared helpers used by paradigm-A/B/C and decoder-paramset sections.
# Pulled out of the inline main.py bodies so they can be referenced as
# bare names from every section function.
# ---------------------------------------------------------------------------
def _popcurve_dirname(name: str, *, cfg) -> str:
    """Append population-curve suffix to run directories only when enabled."""
    if cfg.enable_population_curve:
        return f"{name}{cfg.population_curve_suffix}"
    return name


def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, *, cfg):
    """Open tee to plots_dir/output.txt, print param banner."""
    log_path = os.path.join(plots_dir, "output.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _ParadigmTee(orig_stdout, log_file)
    _print_decoder_param_banner(paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)
    return orig_stdout, log_file


class _ParadigmTee:
    """Write to both console and log file."""
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams: st.write(s)
    def flush(self):
        for st in self._streams: st.flush()


def _print_decoder_param_banner(paradigm_label, decoder_tag, paramset, param_set_name, *, cfg):
    """Print banner showing all active decoder parameters."""
    import datetime as _dt_banner
    print(f"\n{'='*70}")
    print(f"  {paradigm_label} — {decoder_tag} decoder")
    print(f"  Timestamp: {_dt_banner.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Applied optimized param set: '{param_set_name}'")
    print(f"  decoder_type: {cfg.decoder_type}")
    print(f"  encoder_period: {cfg.encoder_period}")
    _sep = "─" * 66
    print(f"  {_sep}")
    for attr in paramset.__slots__:
        val = getattr(paramset, attr, None)
        if val is not None:
            print(f"    {attr:30s} = {val}")
    print(f"{'='*70}\n")


def _stop_paradigm_log(orig_stdout, log_file, plots_dir):
    """Restore stdout and close log file."""
    sys.stdout = orig_stdout
    if log_file and not log_file.closed:
        log_file.close()
    print(f"[LOG] Paradigm output -> {os.path.join(plots_dir, 'output.txt')}")


def _make_paradigm_ABC_mapping(ds):
    """Per-target-family mapping dict used by Paradigms A/B/C."""
    return {
        "Test_A":     ds.mapping_TFC_cond_Test_A_Test_A_1wk,
        "Test_A_1wk": ds.mapping_TFC_cond_Test_A_Test_A_1wk,
        "Test_B":     ds.mapping_TFC_cond_Test_B_Test_B_1wk,
        "Test_B_1wk": ds.mapping_TFC_cond_Test_B_Test_B_1wk,
        "TFC_cond":   ds.mapping_TFC_cond_Test_B_Test_B_1wk,
        "default":    ds.mapping_TFC_cond_Test_B_Test_B_1wk,
    }


def run_rastermap_single_mouse(
    ds,
    m,
    session_name="LT1",
    *,
    n_PCs=200,
    n_clusters=100,
    locality=0.75,
    time_lag_window=5,
    grid_upsample=None,
    vmin=0,
    vmax=1.5,
    show_plot=True,
):
    """Fit and plot a Rastermap embedding for one mouse/session."""
    session_lookup = {
        "TFC_cond": ds.TFC_cond,
        "LT1": ds.TFC_cond_LT1,
        "LT2": ds.TFC_cond_LT2,
    }
    if hasattr(ds, "Test_B"):
        session_lookup["Test_B"] = ds.Test_B
    if hasattr(ds, "Test_B_1wk"):
        session_lookup["Test_B_1wk"] = ds.Test_B_1wk

    if session_name not in session_lookup:
        raise ValueError(f"Unsupported session_name={session_name!r}.")
    if m not in session_lookup[session_name]:
        raise KeyError(f"Mouse {m!r} is not available in session {session_name!r}.")
    if not hasattr(ds, "mouse_groups"):
        raise RuntimeError("Dataset object is missing mouse_groups.")
    if m not in ds.mouse_groups:
        raise RuntimeError(f"Mouse {m!r} is missing from ds.mouse_groups.")

    session = session_lookup[session_name][m]
    mouse_group = ds.mouse_groups[m]
    spks = np.asarray(session.S, dtype="float32")

    rastermap_kwargs = {
        "n_PCs": n_PCs,
        "n_clusters": n_clusters,
        "locality": locality,
        "time_lag_window": time_lag_window,
    }
    if grid_upsample is not None:
        rastermap_kwargs["grid_upsample"] = grid_upsample

    print(f"[rastermap] calling Rastermap with kwargs: {rastermap_kwargs}", flush=True)
    model = Rastermap(**rastermap_kwargs).fit(spks)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.imshow(model.X_embedding, vmin=vmin, vmax=vmax, cmap="gray_r", aspect="auto")

    def _draw_event_lines(onsets, offsets, color):
        for x in onsets:
            ax.axvline(x=float(x), color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        for x in offsets:
            ax.axvline(x=float(x), color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    if session_name == "TFC_cond":
        if not (hasattr(session, "tone_onsets") and hasattr(session, "tone_offsets")):
            raise RuntimeError("TFC_cond session is missing tone onset/offset fields.")
        if not (hasattr(session, "shock_onsets") and hasattr(session, "shock_offsets")):
            raise RuntimeError("TFC_cond session is missing shock onset/offset fields.")
        _draw_event_lines(session.tone_onsets, session.tone_offsets, "b")
        _draw_event_lines(session.shock_onsets, session.shock_offsets, "r")
    elif session_name in {"Test_B", "Test_B_1wk"}:
        if not (hasattr(session, "tone_onsets") and hasattr(session, "tone_offsets")):
            raise RuntimeError(f"{session_name} session is missing tone onset/offset fields.")
        _draw_event_lines(session.tone_onsets, session.tone_offsets, "b")

    ax.set_title(f"Rastermap embedding for {m} ({mouse_group}) [{session_name}]")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Sorted neuron bin")
    if show_plot:
        plt.show()

    return {
        "session": session,
        "spks": spks,
        "model": model,
        "embedding": model.embedding,
        "isort": model.isort,
        "X_embedding": model.X_embedding,
        "figure": fig,
        "axes": ax,
        "mouse": m,
        "mouse_group": mouse_group,
        "session_name": session_name,
    }


def run_rastermap_sweep(
    ds,
    *,
    session_l,
    n_PCs,
    n_clusters,
    locality,
    time_lag_window,
    grid_upsample=2,
    vmin=0,
    vmax=1.5,
    plot_PDF=True,
    dpi=300,
    sweep_label=None,
    close_fig=True,
):
    """Run one Rastermap parameter combo across all mice in each requested session."""
    if not session_l:
        raise RuntimeError("session_l cannot be empty.")
    if not hasattr(ds, "PLOTS_DIR"):
        raise RuntimeError("Dataset object is missing PLOTS_DIR.")
    if not hasattr(ds, "mouse_groups"):
        raise RuntimeError("Dataset object is missing mouse_groups.")

    session_lookup = {
        "TFC_cond": ds.TFC_cond,
        "LT1": ds.TFC_cond_LT1,
        "LT2": ds.TFC_cond_LT2,
    }
    if hasattr(ds, "Test_B"):
        session_lookup["Test_B"] = ds.Test_B
    if hasattr(ds, "Test_B_1wk"):
        session_lookup["Test_B_1wk"] = ds.Test_B_1wk

    for session_name in session_l:
        if session_name not in session_lookup:
            raise RuntimeError(f"Unsupported session_name in session_l: {session_name!r}")

    if sweep_label is None:
        sweep_label = "sweep_" + "__".join(session_l)

    sweep_root = os.path.join(ds.PLOTS_DIR, "rastermap", "param_sweeps", sweep_label)
    os.makedirs(sweep_root, exist_ok=True)
    _copy_analysis_methods_template("rastermap_param_sweeps_methods.txt", sweep_root)

    leaf_dir = os.path.join(
        sweep_root,
        f"n_PCs_{n_PCs}",
        f"n_clusters_{n_clusters}",
        f"locality_{locality}",
        f"time_lag_window_{time_lag_window}",
    )
    os.makedirs(leaf_dir, exist_ok=True)
    pdf_dir = None
    if plot_PDF:
        pdf_dir = os.path.join(leaf_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

    saved_files = []
    runs = 0
    for session_name in session_l:
        session_dict = session_lookup[session_name]
        mice = sorted(session_dict.keys())
        if not mice:
            raise RuntimeError(f"No mice found for session {session_name!r}.")

        for m in mice:
            result = run_rastermap_single_mouse(
                ds,
                m=m,
                session_name=session_name,
                n_PCs=n_PCs,
                n_clusters=n_clusters,
                locality=locality,
                time_lag_window=time_lag_window,
                grid_upsample=grid_upsample,
                vmin=vmin,
                vmax=vmax,
                show_plot=False,
            )

            if m not in ds.mouse_groups:
                raise RuntimeError(f"Mouse {m!r} missing from ds.mouse_groups.")
            group = ds.mouse_groups[m]
            stem = (
                f"{group}_{m}_session_{session_name}"
                f"_n_PCs_{n_PCs}"
                f"_n_clusters_{n_clusters}"
                f"_locality_{locality}"
                f"_time_lag_window_{time_lag_window}"
            )
            png_path = os.path.join(leaf_dir, f"{stem}.png")
            result["figure"].savefig(
                png_path,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            saved_files.append(png_path)

            if plot_PDF:
                if pdf_dir is None:
                    raise RuntimeError("Internal error: pdf_dir is None while plot_PDF=True.")
                pdf_path = os.path.join(pdf_dir, f"{stem}.pdf")
                result["figure"].savefig(
                    pdf_path,
                    format="pdf",
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                saved_files.append(pdf_path)

            if close_fig:
                plt.close(result["figure"])
            runs += 1

    print(
        f"[rastermap sweep] saved {len(saved_files)} files across {runs} runs -> {leaf_dir}",
        flush=True,
    )
    return {
        "sweep_root": sweep_root,
        "leaf_dir": leaf_dir,
        "runs": runs,
        "saved_files": saved_files,
    }


# ---------------------------------------------------------------------------
# Section: sp_rates  (caban/main.py L1474-1539)
# ---------------------------------------------------------------------------
def run_sp_rates(ds, cfg):
    """Analysis section: sp_rates. Originally caban/main.py L1474-1539."""
    if not (cfg.plot_sp_rates and not cfg.DEVEL_SWITCH):
        return
    # --- ds attributes ---
    LT1_exp_activity_mapping = ds.LT1_exp_activity_mapping
    LT1_exp_sp_rates_mapping = ds.LT1_exp_sp_rates_mapping
    LT2_exp_activity_mapping = ds.LT2_exp_activity_mapping
    LT2_exp_sp_rates_mapping = ds.LT2_exp_sp_rates_mapping
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_B_1wk_post_tone_activity_mapping = ds.Test_B_1wk_post_tone_activity_mapping
    Test_B_1wk_post_tone_sp_rates_mapping = ds.Test_B_1wk_post_tone_sp_rates_mapping
    Test_B_1wk_tone_activity_mapping = ds.Test_B_1wk_tone_activity_mapping
    Test_B_1wk_tone_post_tone_activity_mapping = ds.Test_B_1wk_tone_post_tone_activity_mapping
    Test_B_1wk_tone_post_tone_sp_rates_mapping = ds.Test_B_1wk_tone_post_tone_sp_rates_mapping
    Test_B_1wk_tone_sp_rates_mapping = ds.Test_B_1wk_tone_sp_rates_mapping
    Test_B_post_tone_activity_mapping = ds.Test_B_post_tone_activity_mapping
    Test_B_post_tone_sp_rates_mapping = ds.Test_B_post_tone_sp_rates_mapping
    Test_B_tone_activity_mapping = ds.Test_B_tone_activity_mapping
    Test_B_tone_post_tone_activity_mapping = ds.Test_B_tone_post_tone_activity_mapping
    Test_B_tone_post_tone_sp_rates_mapping = ds.Test_B_tone_post_tone_sp_rates_mapping
    Test_B_tone_sp_rates_mapping = ds.Test_B_tone_sp_rates_mapping
    mappings_all_LT1 = ds.mappings_all_LT1
    mappings_all_LT2 = ds.mappings_all_LT2
    mappings_all_TFC_cond = ds.mappings_all_TFC_cond
    mappings_all_Test_B = ds.mappings_all_Test_B
    mappings_all_Test_B_1wk = ds.mappings_all_Test_B_1wk
    mouse_groups = ds.mouse_groups
    plot_interneuron_cutoff = ds.plot_interneuron_cutoff
    post_shock_activity_mapping = ds.post_shock_activity_mapping
    post_shock_sp_rates_mapping = ds.post_shock_sp_rates_mapping
    shock_activity_mapping = ds.shock_activity_mapping
    shock_sp_rates_mapping = ds.shock_sp_rates_mapping
    tone_activity_mapping = ds.tone_activity_mapping
    tone_sp_rates_mapping = ds.tone_sp_rates_mapping
    # --- cfg switches ---
    DEVEL_SWITCH = cfg.DEVEL_SWITCH
    plot_sp_rates = cfg.plot_sp_rates

    # ===== verbatim body from caban/main.py =====
    if plot_sp_rates and not DEVEL_SWITCH:
        msg_start('*** Generating TFC_cond plots')
        for mapping in mappings_all_TFC_cond:
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, tone_sp_rates_mapping[mapping], 'TFC_cond', 'Tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, shock_sp_rates_mapping[mapping], 'TFC_cond', 'Shock '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, post_shock_sp_rates_mapping[mapping], 'TFC_cond', 'Post-shock '+mapping)
        msg_end()

        msg_start('*** Generating Test_B plots')
        for mapping in mappings_all_Test_B:
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_sp_rates_mapping[mapping], 'Test_B', 'Tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_post_tone_sp_rates_mapping[mapping], 'Test_B', 'Post-tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_post_tone_sp_rates_mapping[mapping], 'Test_B', 'Tones+Post-tones '+mapping)

        msg_start('*** Generating Test_B_1wk plots')
        for mapping in mappings_all_Test_B_1wk:
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_sp_rates_mapping[mapping], 'Test_B_1wk', 'Tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_post_tone_sp_rates_mapping[mapping], 'Test_B_1wk', 'Post-tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_post_tone_sp_rates_mapping[mapping], 'Test_B_1wk', 'Tones+Post-tones '+mapping)
        msg_end()

        msg_start('*** Generating LT1 plots')
        for mapping in mappings_all_LT1:
            plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT1_exp_sp_rates_mapping[mapping], 'LT1 track '+mapping)
        msg_end()

        msg_start('*** Generating LT2 plots')
        for mapping in mappings_all_LT2:
            plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT2_exp_sp_rates_mapping[mapping], 'LT2 track '+mapping)
        msg_end()

        ### Now activities (want_peakval)

        msg_start('*** Generating TFC_cond plots (activities)')
        for mapping in mappings_all_TFC_cond:
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, tone_activity_mapping[mapping], 'TFC_cond-activity', 'Tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, shock_activity_mapping[mapping], 'TFC_cond-activity', 'Shock '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, post_shock_activity_mapping[mapping], 'TFC_cond-activity', 'Post-shock '+mapping, tot_dh_incr=0.25)
        msg_end()

        msg_start('*** Generating Test_B plots (activities)')
        for mapping in mappings_all_Test_B:
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_activity_mapping[mapping], 'Test_B-activity', 'Tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_post_tone_activity_mapping[mapping], 'Test_B-activity', 'Post-tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_post_tone_activity_mapping[mapping], 'Test_B-activity', 'Tones+Post-tones '+mapping)

        msg_start('*** Generating Test_B_1wk plots (activities)')
        for mapping in mappings_all_Test_B_1wk:
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_activity_mapping[mapping], 'Test_B_1wk-activity', 'Tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_post_tone_activity_mapping[mapping], 'Test_B_1wk-activity', 'Post-tones '+mapping)
            plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_post_tone_activity_mapping[mapping], 'Test_B_1wk-activity', 'Tones+Post-tones '+mapping)
        msg_end()

        msg_start('*** Generating LT1 plots (activities)')
        for mapping in mappings_all_LT1:
            plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT1_exp_activity_mapping[mapping], 'LT1 track '+mapping, want_peakval=True)
        msg_end()

        msg_start('*** Generating LT2 plots (activities)')
        for mapping in mappings_all_LT2:
            plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT2_exp_activity_mapping[mapping], 'LT2 track '+mapping, want_peakval=True)
        msg_end()

        # Finally plot interneuron cutoff threshold plots
        plot_interneuron_cutoff(PLOTS_DIR, TFC_cond)

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: binned_sp_rates  (caban/main.py L1540-1590)
# ---------------------------------------------------------------------------
def run_binned_sp_rates(ds, cfg):
    """Analysis section: binned_sp_rates. Originally caban/main.py L1540-1590."""
    if not (cfg.plot_binned_sp_rates and not cfg.DEVEL_SWITCH):
        return
    # --- ds attributes ---
    BIN_WIDTH = ds.BIN_WIDTH
    ENGRAM_MODES = ds.ENGRAM_MODES
    PAPER_DIR = ds.PAPER_DIR
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    TFC_cond_binned_activity_mapping = ds.TFC_cond_binned_activity_mapping
    TFC_cond_binned_activity_mapping_engram = ds.TFC_cond_binned_activity_mapping_engram
    TFC_cond_binned_sp_rates_mapping = ds.TFC_cond_binned_sp_rates_mapping
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    Test_B_1wk_binned_activity_mapping = ds.Test_B_1wk_binned_activity_mapping
    Test_B_1wk_binned_activity_mapping_engram = ds.Test_B_1wk_binned_activity_mapping_engram
    Test_B_1wk_binned_sp_rates_mapping = ds.Test_B_1wk_binned_sp_rates_mapping
    Test_B_binned_activity_mapping = ds.Test_B_binned_activity_mapping
    Test_B_binned_activity_mapping_engram = ds.Test_B_binned_activity_mapping_engram
    Test_B_binned_sp_rates_mapping = ds.Test_B_binned_sp_rates_mapping
    mappings_all_TFC_cond = ds.mappings_all_TFC_cond
    mappings_all_Test_B = ds.mappings_all_Test_B
    mappings_all_Test_B_1wk = ds.mappings_all_Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    DEVEL_SWITCH = cfg.DEVEL_SWITCH
    ENGRAM_MODES = cfg.ENGRAM_MODES
    plot_binned_sp_rates = cfg.plot_binned_sp_rates

    # ===== verbatim body from caban/main.py =====
    if plot_binned_sp_rates and not DEVEL_SWITCH:
        msg_start('*** Generating binned spiking TFC_cond plots')
        for mapping in mappings_all_TFC_cond:
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_sp_rates_mapping[mapping], mapping, TFC_cond, 'TFC_cond', BIN_WIDTH)
        msg_end()

        msg_start('*** Generating binned spiking Test_B plots')
        for mapping in mappings_all_Test_B:
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_sp_rates_mapping[mapping], mapping, Test_B, 'Test_B', BIN_WIDTH)
        msg_end()

        msg_start('*** Generating binned spiking Test_B_1wk plots')
        for mapping in mappings_all_Test_B_1wk:
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_sp_rates_mapping[mapping], mapping, Test_B_1wk, 'Test_B_1wk', BIN_WIDTH)
        msg_end()

        ### Now activities (want_peakval)

        msg_start('*** Generating binned activities TFC_cond plots')
        for mapping in mappings_all_TFC_cond:
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_activity_mapping[mapping], mapping, TFC_cond, 'TFC_cond-activity', BIN_WIDTH)
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_activity_mapping[mapping], mapping, TFC_cond, 'TFC_cond-activity', BIN_WIDTH,
                                         plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'))
            for _em in ENGRAM_MODES:
                plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_activity_mapping_engram[_em][mapping], mapping, TFC_cond, 'TFC_cond-activity', BIN_WIDTH,
                                             plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'), suffix=f'_engram_{_em}')

        msg_end()

        msg_start('*** Generating binned activities Test_B plots')
        for mapping in mappings_all_Test_B:
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_activity_mapping[mapping], mapping, Test_B, 'Test_B-activity', BIN_WIDTH)
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_activity_mapping[mapping], mapping, Test_B, 'Test_B-activity', BIN_WIDTH,
                                         plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'))
            for _em in ENGRAM_MODES:
                plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_activity_mapping_engram[_em][mapping], mapping, Test_B, 'Test_B-activity', BIN_WIDTH,
                                             plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'), suffix=f'_engram_{_em}')

        msg_end()

        msg_start('*** Generating binned activities Test_B_1wk plots')
        for mapping in mappings_all_Test_B_1wk:
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_activity_mapping[mapping], mapping, Test_B_1wk, 'Test_B_1wk-activity', BIN_WIDTH)
            plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_activity_mapping[mapping], mapping, Test_B_1wk, 'Test_B_1wk-activity', BIN_WIDTH,
                                         plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'))        
            for _em in ENGRAM_MODES:
                plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_activity_mapping_engram[_em][mapping], mapping, Test_B_1wk, 'Test_B_1wk-activity', BIN_WIDTH,
                                             plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'), suffix=f'_engram_{_em}')

        msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: ROIs  (caban/main.py L1591-1598)
# ---------------------------------------------------------------------------
def run_ROIs(ds, cfg):
    """Analysis section: ROIs. Originally caban/main.py L1591-1598."""
    if not (cfg.plot_ROIs and not cfg.DEVEL_SWITCH):
        return
    # --- ds attributes ---
    LT1_ROI_mappings = ds.LT1_ROI_mappings
    LT1_ROI_mappings_peakval = ds.LT1_ROI_mappings_peakval
    LT2_ROI_mappings = ds.LT2_ROI_mappings
    LT2_ROI_mappings_peakval = ds.LT2_ROI_mappings_peakval
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond_ROI_mappings = ds.TFC_cond_ROI_mappings
    TFC_cond_ROI_mappings_peakval = ds.TFC_cond_ROI_mappings_peakval
    mappings_all_LT1 = ds.mappings_all_LT1
    mappings_all_LT2 = ds.mappings_all_LT2
    mappings_all_TFC_cond = ds.mappings_all_TFC_cond
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    DEVEL_SWITCH = cfg.DEVEL_SWITCH
    plot_ROIs = cfg.plot_ROIs

    # ===== verbatim body from caban/main.py =====
    if plot_ROIs and not DEVEL_SWITCH:
        msg_start('*** Generating ROI mapping plots for TFC_cond')
        plot_ROI_mappings(PLOTS_DIR, mouse_groups, TFC_cond_ROI_mappings, mappings_all_TFC_cond, LT1_ROI_mappings, mappings_all_LT1, LT2_ROI_mappings, mappings_all_LT2)
        plot_ROI_mappings(PLOTS_DIR, mouse_groups, TFC_cond_ROI_mappings_peakval, mappings_all_TFC_cond, \
            LT1_ROI_mappings_peakval, mappings_all_LT1, LT2_ROI_mappings_peakval, mappings_all_LT2, \
                want_peakval=True)
        msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: proportional_activities  (caban/main.py L1599-1609)
# ---------------------------------------------------------------------------
def run_proportional_activities(ds, cfg):
    """Analysis section: proportional_activities. Originally caban/main.py L1599-1609."""
    if not (cfg.plot_proportional_activities and not cfg.DEVEL_SWITCH):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    TFC_cond_LT1 = ds.TFC_cond_LT1
    TFC_cond_LT2 = ds.TFC_cond_LT2
    TFC_cond_crossreg = ds.TFC_cond_crossreg
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mice_per_group = ds.mice_per_group
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    DEVEL_SWITCH = cfg.DEVEL_SWITCH
    plot_proportional_activities = cfg.plot_proportional_activities

    # ===== verbatim body from caban/main.py =====
    if plot_proportional_activities and not DEVEL_SWITCH:
        msg_start('*** Generating proportional activities plots')
        proportional_activities(PLOTS_DIR, mice_per_group, TFC_cond, TFC_cond_LT1, TFC_cond_LT2)
        proportional_activities_TFC_B_B_1wk(PLOTS_DIR, mice_per_group, TFC_cond, Test_B, Test_B_1wk)

        proportional_activities_donut(PLOTS_DIR, mouse_groups, TFC_cond, TFC_cond_LT1, TFC_cond_LT2, ['TFC_cond','TFC_cond_LT1','TFC_cond_LT2'], crossreg_type='TFC_cond', \
            crossreg_to_use=TFC_cond_crossreg)
        proportional_activities_donut(PLOTS_DIR, mouse_groups, TFC_cond, Test_B, Test_B_1wk, ['TFC_cond', 'Test_B', 'Test_B_1wk'], crossreg_type='TFC_B_B_1wk', \
            crossreg_to_use=TFC_B_B_1wk_crossreg)
        msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: LT_firing_rate_changes  (caban/main.py L1610-1646)
# ---------------------------------------------------------------------------
def run_LT_firing_rate_changes(ds, cfg):
    """Analysis section: LT_firing_rate_changes. Originally caban/main.py L1610-1646."""
    if not (cfg.plot_LT_firing_rate_changes and not cfg.DEVEL_SWITCH):
        return
    # --- ds attributes ---
    PAPER_DIR = ds.PAPER_DIR
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    TFC_cond_LT1 = ds.TFC_cond_LT1
    TFC_cond_LT2 = ds.TFC_cond_LT2
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mice_per_group = ds.mice_per_group
    mice_per_group_Test_B_B_1wk = ds.mice_per_group_Test_B_B_1wk
    # --- cfg switches ---
    DEVEL_SWITCH = cfg.DEVEL_SWITCH
    plot_LT_firing_rate_changes = cfg.plot_LT_firing_rate_changes
    want_sample_traces_paper = cfg.want_sample_traces_paper

    # ===== verbatim body from caban/main.py =====
    if plot_LT_firing_rate_changes and not DEVEL_SWITCH:
        msg_start('*** Generating LT1->LT2 firing rate changes plots')
        LT_firing_rate_changes(PLOTS_DIR, mice_per_group, TFC_cond_LT1, TFC_cond_LT2)
        LT_firing_rate_changes(PLOTS_DIR, mice_per_group, TFC_cond_LT1, TFC_cond_LT2, use_peakval=True)
        msg_end()

        msg_start('*** Generating TFC_cond, Test_B, Test_B_1wk firing rate changes plots')
        plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B, mapping_TFC_cond_Test_B_Test_B_1wk)
        plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B, mapping_TFC_cond_Test_B_Test_B_1wk, use_peakval=True)
        plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk)
        plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, use_peakval=True)
        plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk)
        plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, use_peakval=True)
        msg_end()

    if want_sample_traces_paper and not DEVEL_SWITCH:
        msg_start('*** Generating sample traces for paper (TFC_cond) (activities)')
        selections = {
            'hM3D' : [(582, 5149), (130, 23781), (735, 6450)],
            'hM4D' : [(154, 23413), (208, 6672), (271, 14012)],
            'mCherry' : [(319, 18272), (230, 7538), (370, 4138)]
        }
        selections2 = {
            'hM3D' : [(737, 18410), (49, 4657), (434, 9302)],
            'hM4D' : [(153, 3470), (217, 24415), (37, 15678)],
            'mCherry' : [(429, 5391), (424, 796), (588, 7081)]
        }
        selections3 = {
            'hM3D' : [(382, 20196), (9, 14400), (658, 5770)],
            'hM4D' : [(340, 17241), (371, 12186), (256, 20758)],
            'mCherry' : [(67, 3699), (512, 2752), (204, 1502)]
        }
        paper_dir = get_paper_dir(PAPER_DIR, 'fig2')
        plot_sample_traces(PLOTS_DIR, {'hM3D':'G10', 'hM4D':'G14', 'mCherry':'G17'}, TFC_cond, paper_dir=paper_dir, selections=selections3, len_trace=1200)
        #plot_sample_traces(PLOTS_DIR, {'hM3D':'G10', 'hM4D':'G14', 'mCherry':'G17'}, TFC_cond, paper_dir=paper_dir, selection_mode=True, len_trace=1000)
        msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: PSTH  (caban/main.py L1647-1681)
# ---------------------------------------------------------------------------
def run_PSTH(ds, cfg):
    """Analysis section: PSTH. Originally caban/main.py L1647-1681."""
    if not (cfg.plot_PSTH and not cfg.DEVEL_SWITCH):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    TFC_cond_crossreg = ds.TFC_cond_crossreg
    mapping_FULL = ds.mapping_FULL
    mice_per_group = ds.mice_per_group
    # --- cfg switches ---
    DEVEL_SWITCH = cfg.DEVEL_SWITCH
    plot_PSTH = cfg.plot_PSTH

    # ===== verbatim body from caban/main.py =====
    if plot_PSTH and not DEVEL_SWITCH:
        msg_start('*** Generating tone PSTH')
        for mapping in [mapping_FULL]: #mappings_all_TFC_cond: #
            nonzero_active_cells = dict()
            nonzero_suppr_cells = dict()
            frac_tots_active = dict()
            frac_tots_suppr = dict()
            trapz_cells_active = dict()
            trapz_cells_suppr = dict()
            PSTH_cells_active = dict()
            PSTH_cells_suppr = dict()
            max_per_cell_active = dict()
            max_per_cell_suppr = dict()

            tot_cells, frac_tots, sig_cells, group_PSTH, group_percentiles, group_PSTH_vel = process_PSTH_shuffle(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim='shock', num_shuffles=100)

            # Old, obsolete way vvv
            '''
            for fl in [20]:
                for stim in ['tone', 'shock']:
                    for ba in [True]:
                        nonzero_active_cells[stim], frac_tots_active[stim], trapz_cells_active[stim], PSTH_cells_active[stim], max_per_cell_active[stim] = \
                            process_PSTH_simple(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim, frames_lookaround=fl, normalize=True, binary_activity=ba, binary_thresh=80)
                        nonzero_suppr_cells[stim], frac_tots_suppr[stim], trapz_cells_suppr[stim], PSTH_cells_suppr[stim], max_per_cell_suppr[stim] = \
                            process_PSTH_simple(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim, frames_lookaround=fl, normalize=True, binary_activity=ba, binary_thresh=-80, binary_flip=True)
                        nonzero_active_cells[stim], frac_tots_active[stim], trapz_cells_active[stim], PSTH_cells_active[stim], max_per_cell_active[stim] = \
                            process_PSTH_simple(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim, frames_lookaround=fl, normalize=False, binary_activity=ba, binary_thresh=80)
                        #plot_PSTH_overlay(PLOTS_DIR, TFC_cond, nonzero_active_cells, nonzero_suppr_cells, stim, mapping)
                        plot_PSTH_activities(PLOTS_DIR, frac_tots_active[stim], stim, 'Active', mapping)
                        plot_PSTH_activities(PLOTS_DIR, frac_tots_suppr[stim], stim, 'Suppressed', mapping)
                        plot_PSTH_intensities(PLOTS_DIR, trapz_cells_active[stim], stim, mapping)
                        plot_PSTH_peaks(PLOTS_DIR, PSTH_cells_active[stim], max_per_cell_active[stim], stim, mapping)
            '''
        msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: pf_and_loc  (caban/main.py L1682-1779)
# ---------------------------------------------------------------------------
def run_pf_and_loc(ds, cfg):
    """Analysis section: pf_and_loc. Originally caban/main.py L1682-1779."""
    if not (cfg.plot_pf_and_loc):
        return
    # --- ds attributes ---
    BEHAVIOUR_TYPE = ds.BEHAVIOUR_TYPE
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_A_A_1wk_crossreg = ds.TFC_A_A_1wk_crossreg
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mice_per_group = ds.mice_per_group
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    BEHAVIOUR_TYPE = cfg.BEHAVIOUR_TYPE
    plot_pf_and_loc = cfg.plot_pf_and_loc
    plot_pf_raw_maps = cfg.plot_pf_raw_maps

    # ===== verbatim body from caban/main.py =====
    if plot_pf_and_loc: # and not DEVEL_SWITCH:
        if BEHAVIOUR_TYPE == 'movement':
            if plot_pf_raw_maps:
                plot_location_map(PLOTS_DIR, mice_per_group, TFC_cond, 'TFC_cond')
            plot_fluorescence_map(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond', bin_width=34, random_width=4, want_3D=True, pcells_mice=None, \
                max_fields=45, only_fm_pcells=True, print_pcell_maps=True, plot_pf_maps=plot_pf_raw_maps) 
            if plot_pf_raw_maps:
                plot_pf_analyses(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond')
                plot_pf_analyses(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond', crossreg=TFC_B_B_1wk_crossreg, mapping=mapping_TFC_cond_Test_B_Test_B_1wk)

            if plot_pf_raw_maps:
                plot_location_map(PLOTS_DIR, mice_per_group, Test_B, 'Test_B')
            plot_fluorescence_map(PLOTS_DIR, Test_B, mouse_groups, 'Test_B', bin_width=34, random_width=4, want_3D=True, pcells_mice=None, \
                max_fields=45, only_fm_pcells=True, print_pcell_maps=True, plot_pf_maps=plot_pf_raw_maps)
            if plot_pf_raw_maps:
                plot_pf_analyses(PLOTS_DIR, Test_B, mouse_groups, 'Test_B')
                plot_pf_analyses(PLOTS_DIR, Test_B, mouse_groups, 'Test_B', crossreg=TFC_B_B_1wk_crossreg, mapping=mapping_TFC_cond_Test_B_Test_B_1wk)

            if plot_pf_raw_maps:
                plot_location_map(PLOTS_DIR, mice_per_group, Test_B_1wk, 'Test_B_1wk')
            plot_fluorescence_map(PLOTS_DIR, Test_B_1wk, mouse_groups, 'Test_B_1wk', bin_width=34, random_width=4, want_3D=True, pcells_mice=None, \
                max_fields=45, only_fm_pcells=True, print_pcell_maps=True, plot_pf_maps=plot_pf_raw_maps)
            if plot_pf_raw_maps:
                plot_pf_analyses(PLOTS_DIR, Test_B_1wk, mouse_groups, 'Test_B_1wk')
                plot_pf_analyses(PLOTS_DIR, Test_B_1wk, mouse_groups, 'Test_B_1wk', crossreg=TFC_B_B_1wk_crossreg, mapping=mapping_TFC_cond_Test_B_Test_B_1wk)

            if plot_pf_raw_maps:
                plot_location_map(PLOTS_DIR, mice_per_group, Test_A, 'Test_A')
            plot_fluorescence_map(PLOTS_DIR, Test_A, mouse_groups, 'Test_A', bin_width=34, random_width=4, want_3D=True, pcells_mice=None, \
                max_fields=45, only_fm_pcells=True, print_pcell_maps=True, plot_pf_maps=plot_pf_raw_maps)
            if plot_pf_raw_maps:
                plot_pf_analyses(PLOTS_DIR, Test_A, mouse_groups, 'Test_A')
                plot_pf_analyses(PLOTS_DIR, Test_A, mouse_groups, 'Test_A', crossreg=TFC_A_A_1wk_crossreg, mapping=mapping_TFC_cond_Test_A_Test_A_1wk)

            if plot_pf_raw_maps:
                plot_location_map(PLOTS_DIR, mice_per_group, Test_A_1wk, 'Test_A_1wk')
            plot_fluorescence_map(PLOTS_DIR, Test_A_1wk, mouse_groups, 'Test_A_1wk', bin_width=34, random_width=4, want_3D=True, pcells_mice=None, \
                max_fields=45, only_fm_pcells=True, print_pcell_maps=True, plot_pf_maps=plot_pf_raw_maps)
            if plot_pf_raw_maps:
                plot_pf_analyses(PLOTS_DIR, Test_A_1wk, mouse_groups, 'Test_A_1wk')
                plot_pf_analyses(PLOTS_DIR, Test_A_1wk, mouse_groups, 'Test_A_1wk', crossreg=TFC_A_A_1wk_crossreg, mapping=mapping_TFC_cond_Test_A_Test_A_1wk)

            # Finally, across sessions (i.e., within groups). Sessions are handled in the function.
            if plot_pf_raw_maps:
                plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_B, Test_B_1wk, mouse_groups, crossreg=None, mapping=None)
                plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_B, Test_B_1wk, mouse_groups, crossreg=TFC_B_B_1wk_crossreg, mapping=mapping_TFC_cond_Test_B_Test_B_1wk)
                plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_A, Test_A_1wk, mouse_groups, crossreg=None, mapping=None,
                    sess_names=['TFC_cond', 'Test_A', 'Test_A_1wk'], skip_mice=['G15'])
                plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_A, Test_A_1wk, mouse_groups, crossreg=TFC_A_A_1wk_crossreg, mapping=mapping_TFC_cond_Test_A_Test_A_1wk,
                    sess_names=['TFC_cond', 'Test_A', 'Test_A_1wk'], skip_mice=['G15'])

    # Ensure merged PF attributes exist (back-fills from cached .npz without re-fitting)
    for _label, _sess_dict in [('TFC_cond', TFC_cond), ('Test_B', Test_B),
                                ('Test_B_1wk', Test_B_1wk), ('Test_A', Test_A),
                                ('Test_A_1wk', Test_A_1wk)]:
        for _mouse, _sess in _sess_dict.items():
            _fm = getattr(_sess, 'fm', None)
            if _fm is None or not hasattr(_fm, 'pf'):
                continue
            _pf = _fm.pf
            if getattr(_pf, 'merged_means_', None) and _pf.merged_means_:
                continue
            _pf.merged_means_ = {}
            _pf.merged_covariances_ = {}
            _pf.merged_weights_ = {}
            for _cid, _merged in _pf.merged_means.items():
                if _cid not in _pf.model_:
                    continue
                _model = _pf.model_[_cid]
                _mu_list, _cov_list, _w_list = [], [], []
                for _comp_idxs in _merged:
                    _w = _model.weights_[_comp_idxs]
                    _w_sum = _w.sum()
                    _w_norm = _w / _w_sum
                    _mu = _w_norm @ _model.means_[_comp_idxs]
                    _cov = np.zeros((2, 2))
                    for _ci, _wi in zip(_comp_idxs, _w_norm):
                        _diff = _model.means_[_ci] - _mu
                        _cov += _wi * (_model.covariances_[_ci] + np.outer(_diff, _diff))
                    _mu_list.append(_mu)
                    _cov_list.append(_cov)
                    _w_list.append(float(_w_sum))
                _pf.merged_means_[_cid] = _mu_list
                _pf.merged_covariances_[_cid] = _cov_list
                _pf.merged_weights_[_cid] = _w_list
            print(f"  [merged PF backfill] {_label} {_mouse}: {len(_pf.merged_means_)} cells")

    # ==============================================================================
    # ====  Occupancy / trajectory / immobility analysis  =========================
    # ==============================================================================

    plot_occupancy_analysis = True

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: occupancy_analysis  (caban/main.py L1780-1818)
# ---------------------------------------------------------------------------
def run_occupancy_analysis(ds, cfg):
    """Analysis section: occupancy_analysis. Originally caban/main.py L1780-1818."""
    if not (cfg.plot_occupancy_analysis):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    TFC_cond_LT1 = ds.TFC_cond_LT1
    TFC_cond_LT2 = ds.TFC_cond_LT2
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    plot_occupancy_analysis = cfg.plot_occupancy_analysis

    # ===== verbatim body from caban/main.py =====
    if plot_occupancy_analysis:
        msg_start('*** Occupancy / trajectory / immobility analysis')

        # Collect all available session dicts.
        # LT sessions use full duration; TFC-family sessions use first 180 s.
        occupancy_sessions = {}
        if TFC_cond_LT1:
            occupancy_sessions["LT1"] = TFC_cond_LT1
        if TFC_cond_LT2:
            occupancy_sessions["LT2"] = TFC_cond_LT2
        if TFC_cond:
            occupancy_sessions["TFC_cond"] = TFC_cond
        if Test_B:
            occupancy_sessions["Test_B"] = Test_B
        if Test_B_1wk:
            occupancy_sessions["Test_B_1wk"] = Test_B_1wk
        if Test_A:
            occupancy_sessions["Test_A"] = Test_A
        if Test_A_1wk:
            occupancy_sessions["Test_A_1wk"] = Test_A_1wk

        df_occ, df_occ_stats, occ_all = run_occupancy_analysis_pipeline(
            occupancy_sessions,
            mouse_groups,
            PLOTS_DIR,
            bin_width=10.0,
            first_n_sec_tfc=180.0,
            first_n_sec_lt=None,
            speed_thresh=2.0,
            auto_close=True,
        )

        print("\n--- Occupancy summary (first rows) ---")
        print(df_occ.to_string(index=False))
        print("\n--- Group statistics ---")
        print(df_occ_stats.to_string(index=False))

        msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: LT_pfs  (caban/main.py L1819-2095)
# ---------------------------------------------------------------------------
def run_LT_pfs(ds, cfg):
    """Analysis section: LT_pfs. Originally caban/main.py L1819-2095."""
    if not (cfg.plot_LT_pfs):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond_LT1 = ds.TFC_cond_LT1
    TFC_cond_LT2 = ds.TFC_cond_LT2
    mice_per_group = ds.mice_per_group
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    plot_LT_pfs = cfg.plot_LT_pfs

    # ===== verbatim body from caban/main.py =====
    if plot_LT_pfs:
    #if plot_LT_pfs and not DEVEL_SWITCH:

        interactive_before = plt.isinteractive()
        plt.ioff()
        try:
            # ---- Place-field count filter ----
            # Set to None for all place cells, or an integer (1, 2, ...) to include
            # only cells with <= that many place fields.
            MAX_PF_COUNT = None        # None -> pfALL;  1 -> pf1 (single-field only);  2 -> pf1+pf2;  etc.
            _pf_str = "pfALL" if (MAX_PF_COUNT is None) else f"pf{int(MAX_PF_COUNT)}"
            PAPER_PLOTS = True        # True -> generate additional compact publication-quality figures

            # plot mapping of mouse positions to centerline of LT
            plot_LT_linearized(PLOTS_DIR, TFC_cond_LT1, mouse_groups, 'TFC_cond_LT1')
            plot_LT_linearized(PLOTS_DIR, TFC_cond_LT2, mouse_groups, 'TFC_cond_LT2')

            plot_location_map(PLOTS_DIR, mice_per_group, TFC_cond_LT1, 'TFC_cond_LT1')
            plot_location_map(PLOTS_DIR, mice_per_group, TFC_cond_LT2, 'TFC_cond_LT2')

            sig_responses_TFC_cond_LT1 = get_fluorescence_map(TFC_cond_LT1, "TFC_cond_LT1", NPY_SAVE_PATH, PLOTS_DIR=PLOTS_DIR, mouse_groups=mouse_groups, \
                bin_width=4.5, random_width=4, want_3D=True, max_fields=45, only_fm_pcells=True, print_pcell_maps=True, merge_distance=4)
            sig_responses_TFC_cond_LT2 = get_fluorescence_map(TFC_cond_LT2, "TFC_cond_LT2", NPY_SAVE_PATH, PLOTS_DIR=PLOTS_DIR, mouse_groups=mouse_groups, \
                bin_width=4.5, random_width=4, want_3D=True, max_fields=45, only_fm_pcells=True, print_pcell_maps=True, merge_distance=4)

            plot_pf_analyses(PLOTS_DIR, TFC_cond_LT1, mouse_groups, 'TFC_cond_LT1')
            plot_pf_analyses(PLOTS_DIR, TFC_cond_LT2, mouse_groups, 'TFC_cond_LT2')    


            # Loop through mappings and produce tilings/analyses for all
            for mapping in ['full', 'LT1+LT2', 'TFC_cond+LT1+LT2']:
                mapping_str = f"mapping_{mapping}"
                # Main PF count variants
                tiling_LT1 = plot_lt_within_session_tuning(
                    PLOTS_DIR, TFC_cond_LT1, mouse_groups,
                    f'TFC_cond_LT1_{mapping_str}',
                    max_pf_count=MAX_PF_COUNT, want_S=True, mapping=mapping)
                tiling_LT2 = plot_lt_within_session_tuning(
                    PLOTS_DIR, TFC_cond_LT2, mouse_groups,
                    f'TFC_cond_LT2_{mapping_str}',
                    max_pf_count=MAX_PF_COUNT, want_S=True, mapping=mapping)
                tiling_LT1_pf1 = plot_lt_within_session_tuning(
                    PLOTS_DIR, TFC_cond_LT1, mouse_groups,
                    f'TFC_cond_LT1_{mapping_str}',
                    max_pf_count=1, want_S=True, mapping=mapping)
                tiling_LT2_pf1 = plot_lt_within_session_tuning(
                    PLOTS_DIR, TFC_cond_LT2, mouse_groups,
                    f'TFC_cond_LT2_{mapping_str}',
                    max_pf_count=1, want_S=True, mapping=mapping)
                tiling_LT1_pf2 = plot_lt_within_session_tuning(
                    PLOTS_DIR, TFC_cond_LT1, mouse_groups,
                    f'TFC_cond_LT1_{mapping_str}',
                    max_pf_count=2, want_S=True, mapping=mapping)
                tiling_LT2_pf2 = plot_lt_within_session_tuning(
                    PLOTS_DIR, TFC_cond_LT2, mouse_groups,
                    f'TFC_cond_LT2_{mapping_str}',
                    max_pf_count=2, want_S=True, mapping=mapping)

                # Tiling metrics analyses
                plot_tiling_metrics_anova(tiling_LT1, mouse_groups, PLOTS_DIR=PLOTS_DIR,
                                          session_str=f'TFC_cond_LT1_{mapping_str}', pf_str=_pf_str, paper_plots=PAPER_PLOTS, mapping=mapping)
                plot_tiling_metrics_anova(tiling_LT2, mouse_groups, PLOTS_DIR=PLOTS_DIR,
                                          session_str=f'TFC_cond_LT2_{mapping_str}', pf_str=_pf_str, paper_plots=PAPER_PLOTS, mapping=mapping)
                plot_tiling_metrics_anova(tiling_LT1_pf1, mouse_groups, PLOTS_DIR=PLOTS_DIR,
                                          session_str=f'TFC_cond_LT1_{mapping_str}', pf_str='pf1', paper_plots=PAPER_PLOTS, mapping=mapping)
                plot_tiling_metrics_anova(tiling_LT2_pf1, mouse_groups, PLOTS_DIR=PLOTS_DIR,
                                          session_str=f'TFC_cond_LT2_{mapping_str}', pf_str='pf1', paper_plots=PAPER_PLOTS, mapping=mapping)
                plot_tiling_metrics_anova(tiling_LT1_pf2, mouse_groups, PLOTS_DIR=PLOTS_DIR,
                                          session_str=f'TFC_cond_LT1_{mapping_str}', pf_str='pf2', paper_plots=PAPER_PLOTS, mapping=mapping)
                plot_tiling_metrics_anova(tiling_LT2_pf2, mouse_groups, PLOTS_DIR=PLOTS_DIR,
                                          session_str=f'TFC_cond_LT2_{mapping_str}', pf_str='pf2', paper_plots=PAPER_PLOTS, mapping=mapping)

            # Normalised [0,1] versions (per-mouse) - full (all sig cells) and cross-reg mappings
            for _mapping in [None, 'LT1+LT2', 'TFC_cond+LT1+LT2']:
                plot_lt_within_session_tuning_normalized(PLOTS_DIR, TFC_cond_LT1, mouse_groups, 'TFC_cond_LT1', mapping=_mapping, max_pf_count=MAX_PF_COUNT, want_S=True)
                plot_lt_within_session_tuning_normalized(PLOTS_DIR, TFC_cond_LT2, mouse_groups, 'TFC_cond_LT2', mapping=_mapping, max_pf_count=MAX_PF_COUNT, want_S=True)

            # Group-averaged versions (normalised, one heatmap per group) - full and cross-reg mappings
            for _mapping in [None, 'LT1+LT2', 'TFC_cond+LT1+LT2']:
                plot_lt_within_session_tuning_group_averaged(PLOTS_DIR, TFC_cond_LT1, mouse_groups, 'TFC_cond_LT1', mapping=_mapping, max_pf_count=MAX_PF_COUNT, want_S=True)
                plot_lt_within_session_tuning_group_averaged(PLOTS_DIR, TFC_cond_LT2, mouse_groups, 'TFC_cond_LT2', mapping=_mapping, max_pf_count=MAX_PF_COUNT, want_S=True)
        finally:
            plt.close('all')
            if interactive_before:
                plt.ion()

        #pv_corr_LT1_LT2 = plot_lt_spatial_responses(PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups, 'TFC_cond_LT', mapping='LT1+LT2', want_C=True, \
        #    normalize_pairwise_per_cell=True, pairwise_cell_mode='pctl', pairwise_cell_pctl=95.0)
        #pv_corr_TFC_cond_LT1_LT2 = plot_lt_spatial_responses(PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups, 'TFC_cond_LT', mapping='TFC_cond+LT1+LT2', want_C=True, normalize_per_mouse=True, \
        #    normalize_pairwise_per_cell=True, pairwise_cell_mode='pctl', pairwise_cell_pctl=95.0)

        # ---- Outlier exclusion for PV correlation analyses ----
        INCLUDE_OUTLIERS = False
        LT_PV_OUTLIER_MICE = ['G07']  # incomplete track coverage (bottom arm truncated)

        if INCLUDE_OUTLIERS:
            mouse_groups_pv = mouse_groups
        else:
            mouse_groups_pv = {m: g for m, g in mouse_groups.items() if m not in LT_PV_OUTLIER_MICE}
            print(f"[INFO] LT PV correlation: excluding outlier mice {LT_PV_OUTLIER_MICE}. "
                  f"Remaining: {list(mouse_groups_pv.keys())}")

        pv_corr_LT1_LT2_S_global = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_global",
            mapping="LT1+LT2",
            want_S=True,
            normalize_global=True,
            global_norm_mode="pctl",
            global_norm_pctl=99.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
        ) 

        pv_corr_LT1_LT2_S_pairwise = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_pairwise",
            mapping="LT1+LT2",
            want_S=True,
            normalize_pairwise_per_cell=True,
            pairwise_cell_mode="pctl",
            pairwise_cell_pctl=95.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
        )

        pv_corr_TFC_cond_LT1_LT2_S_global = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_global",
            mapping="TFC_cond+LT1+LT2",
            want_S=True,
            normalize_global=True,
            global_norm_mode="pctl",
            global_norm_pctl=99.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
        ) 

        pv_corr_TFC_cond_LT1_LT2_S_pairwise = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_pairwise",
            mapping="TFC_cond+LT1+LT2",
            want_S=True,
            normalize_pairwise_per_cell=True,
            pairwise_cell_mode="pctl",
            pairwise_cell_pctl=95.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
        )

        # ---- Same 4 calls but sorted by LT2 PF centres ----
        pv_corr_LT1_LT2_S_global_sortLT2 = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_global",
            mapping="LT1+LT2",
            want_S=True,
            normalize_global=True,
            global_norm_mode="pctl",
            global_norm_pctl=99.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
            sort_by="LT2",
        )

        pv_corr_LT1_LT2_S_pairwise_sortLT2 = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_pairwise",
            mapping="LT1+LT2",
            want_S=True,
            normalize_pairwise_per_cell=True,
            pairwise_cell_mode="pctl",
            pairwise_cell_pctl=95.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
            sort_by="LT2",
        )

        pv_corr_TFC_cond_LT1_LT2_S_global_sortLT2 = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_global",
            mapping="TFC_cond+LT1+LT2",
            want_S=True,
            normalize_global=True,
            global_norm_mode="pctl",
            global_norm_pctl=99.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
            sort_by="LT2",
        )

        pv_corr_TFC_cond_LT1_LT2_S_pairwise_sortLT2 = plot_lt_spatial_responses(
            PLOTS_DIR, TFC_cond_LT1, TFC_cond_LT2, mouse_groups_pv, "TFC_cond_LT_pairwise",
            mapping="TFC_cond+LT1+LT2",
            want_S=True,
            normalize_pairwise_per_cell=True,
            pairwise_cell_mode="pctl",
            pairwise_cell_pctl=95.0,
            pv_use_normalized=True,
            max_pf_count=MAX_PF_COUNT,
            sort_by="LT2",
        )

        results = plot_pv_corr_anova(
            pv_corr_LT1_LT2_S_global,
            mouse_groups_pv,
            PLOTS_DIR=PLOTS_DIR,
            type='global',
            mapping='LT1+LT2',
            pf_str=_pf_str,
            title="PV correlation by group",
            ylabel="Average diagonal PV correlation",
            paper_plots=PAPER_PLOTS,
        )
        results = plot_pv_corr_anova(
            pv_corr_LT1_LT2_S_pairwise,
            mouse_groups_pv,
            PLOTS_DIR=PLOTS_DIR,
            type='pairwise',
            mapping='LT1+LT2',
            pf_str=_pf_str,
            title="PV correlation by group",
            ylabel="Average diagonal PV correlation",
            paper_plots=PAPER_PLOTS,
        )    
        results = plot_pv_corr_anova(
            pv_corr_TFC_cond_LT1_LT2_S_global,
            mouse_groups_pv,
            PLOTS_DIR=PLOTS_DIR,
            type='global',
            mapping='TFC_cond+LT1+LT2',
            pf_str=_pf_str,
            title="PV correlation by group",
            ylabel="Average diagonal PV correlation",
            paper_plots=PAPER_PLOTS,
        )
        results = plot_pv_corr_anova(
            pv_corr_TFC_cond_LT1_LT2_S_pairwise,
            mouse_groups_pv,
            PLOTS_DIR=PLOTS_DIR,
            type='pairwise',
            mapping='TFC_cond+LT1+LT2',
            pf_str=_pf_str,
            title="PV correlation by group",
            ylabel="Average diagonal PV correlation",
            paper_plots=PAPER_PLOTS,
        )

        # ============================================================
        # Zone-split PV correlation analysis (arm_A / joint / arm_B)
        # + alternative metrics (diagonal excess, specificity, decode acc, etc.)
        # ============================================================
        for pv_data, norm_type, map_label in [
            (pv_corr_LT1_LT2_S_global,             'global',   'LT1+LT2'),
            (pv_corr_LT1_LT2_S_pairwise,           'pairwise', 'LT1+LT2'),
            (pv_corr_TFC_cond_LT1_LT2_S_global,    'global',   'TFC_cond+LT1+LT2'),
            (pv_corr_TFC_cond_LT1_LT2_S_pairwise,  'pairwise', 'TFC_cond+LT1+LT2'),
        ]:
            # Zone-aware ANOVA (arm_A, joint, arm_B, full) x multiple metrics
            plot_pv_corr_anova_zones(
                pv_data, mouse_groups_pv,
                PLOTS_DIR=PLOTS_DIR,
                type=norm_type,
                mapping=map_label,
                pf_str=_pf_str,
                paper_plots=PAPER_PLOTS,
            )
            # Zone heatmap panels per mouse
            plot_pv_corr_zone_heatmaps(
                pv_data, mouse_groups_pv,
                PLOTS_DIR=PLOTS_DIR,
                type=norm_type,
                mapping=map_label,
                pf_str=_pf_str,
            )
            # Group-averaged PV correlation heatmaps (one per group)
            plot_pv_corr_group_averaged(
                pv_data, mouse_groups_pv,
                PLOTS_DIR=PLOTS_DIR,
                type=norm_type,
                mapping=map_label,
                pf_str=_pf_str,
            )
            # Group-averaged zone sub-matrix heatmaps (one figure per group)
            plot_pv_corr_zone_heatmaps_group_averaged(
                pv_data, mouse_groups_pv,
                PLOTS_DIR=PLOTS_DIR,
                type=norm_type,
                mapping=map_label,
                pf_str=_pf_str,
            )

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: LT_decoding  (caban/main.py L2096-2888)
# ---------------------------------------------------------------------------
def run_LT_decoding(ds, cfg):
    """Analysis section: LT_decoding. Originally caban/main.py L2096-2888."""
    if not (cfg.plot_LT_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond_LT1 = ds.TFC_cond_LT1
    TFC_cond_LT2 = ds.TFC_cond_LT2
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_shuffle_n_repeats = cfg.decoder_shuffle_n_repeats
    decoder_shuffle_seed = cfg.decoder_shuffle_seed
    decoder_shuffle_type = cfg.decoder_shuffle_type
    enable_lt_shuffle_control = cfg.enable_lt_shuffle_control
    plot_LT_decoding = cfg.plot_LT_decoding

    interactive_before = plt.isinteractive()
    plt.ioff()

    # ===== verbatim body from caban/main.py =====
    if plot_LT_decoding:
        # ---- Position-decoder mode switches ----
        # True  → PVT decoder and main 1D decoder run independently (original behaviour)
        # False → PVT decoder runs first; the chosen mode replaces the main 1D decoder
        # NB: mutually exclusive with `use_true_2D_decoder`
        use_separate_position_decoder = False
        run_PVT_decoder = True # if above set to True, this will be checked to see if it should run at all.
        positional_decoder_to_use = "PF (2D)"   # "S (1D)", "PF (1D)", "S (2D)", "PF (2D)", or "all"

        # ---- Crossreg-1 (triple-session) cell restriction ----
        # When True: within-LT1, within-LT2, and LT1→LT2 all use only cells
        # cross-registered across LT1+LT2+TFC_cond (from mappings_crossreg_1.csv),
        # i.e. the triple intersection.  Fewer cells than the default LT1+LT2 pair.
        # Only supported when use_separate_position_decoder == False.
        use_crossreg_1 = False

        # ---- True-2D decoder switch ----
        # When True: run full 2D Bayesian decoding for BOTH S_2D_full and
        #   PF_2D_full, each in its own top-level directory (LT_decoding_2D_S,
        #   LT_decoding_2D_PF) with the full downstream pipeline.
        # NB: mutually exclusive with `use_separate_position_decoder`, above
        use_true_2D_decoder = False
        true_2D_use_continuity = True

        # ---- Zone-level analysis metric switch ----
        # True  → use PCT (proportion-correct-within-threshold) as DV
        # False → use mean_err and median_err as DV (both run)
        use_PCT_error = True
        pct_threshold = 15.0  # cm; only used when use_PCT_error=True

        time_bin_frames_orig = 15
        time_bin_frames_true_2D = 15
        time_bin_frames_pvt = 15

        _run_all_pvt_modes = (not use_separate_position_decoder) and (str(positional_decoder_to_use).strip().lower() == "all")
        _all_pvt_mode_labels = list(_PVT_MODE_LABEL_TO_KEY.keys())

        if use_crossreg_1 and use_separate_position_decoder:
            raise ValueError("use_crossreg_1 is only supported when "
                             "use_separate_position_decoder == False")

        if use_true_2D_decoder:
            use_separate_position_decoder = False
            # LT_PLOTS_DIR set per-mode below; use a shared dir for lap seg & PVT
            _lt_dir_name = "LT_decoding_2D"
        elif use_separate_position_decoder:
            _lt_dir_name = "LT_decoding"
        elif _run_all_pvt_modes:
            # Use first mode's directory for shared preprocessing output
            _first_tag = _all_pvt_mode_labels[0].replace(" ", "").replace("(", "_").replace(")", "")
            _lt_dir_name = f"LT_decoding_{_first_tag}"
        else:
            _mode_tag = positional_decoder_to_use.replace(" ", "").replace("(", "_").replace(")", "")
            _lt_dir_name = f"LT_decoding_{_mode_tag}"

        if use_crossreg_1:
            _lt_dir_name += "_xreg1"

        LT_PLOTS_DIR = os.path.join(PLOTS_DIR, _lt_dir_name)
        os.makedirs(LT_PLOTS_DIR, exist_ok=True)

        session_str_full = "TFC_cond_LT"
        session_str_mapped_within = f"{session_str_full}_withinMAPPED"
        mapping = "LT1+LT2"
        # USE_MEDIAN is looped below over [False, True]

        print(f"\n[LT-DECODE] use_separate_position_decoder = {use_separate_position_decoder}")
        print(f"[LT-DECODE] positional_decoder_to_use     = {positional_decoder_to_use!r}")
        print(f"[LT-DECODE] run_all_pvt_modes             = {_run_all_pvt_modes}")
        print(f"[LT-DECODE] use_true_2D_decoder            = {use_true_2D_decoder}")
        print(f"[LT-DECODE] use_crossreg_1                = {use_crossreg_1}")
        print(f"[LT-DECODE] use_PCT_error                 = {use_PCT_error}")
        if use_PCT_error:
            print(f"[LT-DECODE] pct_threshold                 = {pct_threshold} cm")
        if use_true_2D_decoder:
            print(f"[LT-DECODE] time_bin_frames_true_2D        = {time_bin_frames_true_2D}")
            print(f"[LT-DECODE] true_2D_use_continuity         = {true_2D_use_continuity}")
            print(f"[LT-DECODE] n_x/y_bins auto-computed from track extent & fm.loc.bin_width")

        # ============================================================
        # -1) Velocity histograms  (per-mouse & group-pooled)
        # ============================================================
        vel_stats_LT = plot_velocity_histograms(
            TFC_cond_LT1, TFC_cond_LT2,
            mouse_groups=mouse_groups,
            PLOTS_DIR=LT_PLOTS_DIR,
            session_tags=("LT1", "LT2"),
            auto_close=True,
        )

        # Data-driven continuity params for PVT decoder
        _lt_cont_pvt = compute_continuity_params(vel_stats_LT, time_bin_frames=time_bin_frames_pvt)
        save_continuity_params_txt(_lt_cont_pvt, LT_PLOTS_DIR,
                                   tag=f"LT PVT (time_bin_frames={time_bin_frames_pvt})",
                                   time_bin_frames=time_bin_frames_pvt)
        # Data-driven continuity params for true-2D decoder (may differ in bin size)
        _lt_cont_2d  = compute_continuity_params(vel_stats_LT, time_bin_frames=time_bin_frames_true_2D)
        save_continuity_params_txt(_lt_cont_2d, LT_PLOTS_DIR,
                                   tag=f"LT true-2D (time_bin_frames={time_bin_frames_true_2D})",
                                   time_bin_frames=time_bin_frames_true_2D)

        # ============================================================
        # 0) Lap segmentation
        # ============================================================

        summary = attach_lap_segmentations(
            TFC_cond_LT1,
            TFC_cond_LT2,
            endzone_frac=0.05,               # 5% endzones
            smooth_win=9,                    # robust vel sign for turns
            min_turn_separation=5,           # suppress micro-flips
            jump_thresh_frac_of_range=0.08,  # tighter now that projection is continuous
            require_start_in_endzone=False,  # start lap 0 at frame 0; don't discard initial traversal
            verbose=True,
            plot_debug=True,                # show lap plots
            PLOTS_DIR=LT_PLOTS_DIR,
            session_str=session_str_full,
            mapping=mapping,
            mouse_groups=mouse_groups
        )

        for m, info in summary.items():
            if "LT1_n_laps" in info or "LT2_n_laps" in info:
                print(m, info)

        # ============================================================
        # 0b) Position-vs-Time LT decoder pipeline (4 modes × 3 conds)
        #     Runs BEFORE the main 1D decoder so that, when
        #     use_separate_position_decoder=False, the chosen PVT mode
        #     can be converted into dec_results and fed downstream.
        # ============================================================
        if run_PVT_decoder:
            LT_PVT_NPY_DIR = os.path.join(NPY_SAVE_PATH, "LT_decoder_position_vs_time")
            os.makedirs(LT_PVT_NPY_DIR, exist_ok=True)

            # When PVT replaces the main decoder, only the chosen mode is needed.
            # When running independently (use_separate_position_decoder), run all 4
            # so cross-mode comparison stats are meaningful.
            if use_separate_position_decoder:
                _pvt_modes = None          # all 4
            elif _run_all_pvt_modes:
                _pvt_modes = [_PVT_MODE_LABEL_TO_KEY[m] for m in _all_pvt_mode_labels]
            else:
                _pvt_mode_key = _PVT_MODE_LABEL_TO_KEY.get(
                    positional_decoder_to_use, positional_decoder_to_use
                )
                _pvt_modes = [_pvt_mode_key]

            _pvt_common_kwargs = dict(
                mapping=mapping,
                n_pos_bins=60,
                time_bin_frames=time_bin_frames_pvt,
                use_speed=True,
                min_speed=2.0,
                use_z_score="per-session",
                use_posterior_mean=True,
                use_continuity_constraint=True,
                **_lt_cont_pvt,
                n_x_bins=20,
                n_y_bins=20,
                NPY_SAVE_DIR=LT_PVT_NPY_DIR,
                session_str=session_str_full,
                mouse_groups=mouse_groups,
                plot_debug=True,
                auto_close=True,
                enable_shuffle_control=enable_lt_shuffle_control,
                shuffle_type=decoder_shuffle_type,
                n_shuffles=decoder_shuffle_n_repeats,
                shuffle_seed=decoder_shuffle_seed,
                use_crossreg_1=use_crossreg_1,
                pct_threshold=pct_threshold,
            )

            if _run_all_pvt_modes:
                # Run each mode separately into its own directory
                _all_pvt_run_results = {}   # mode_label → (pvt_results, mode_dir)
                for _ml in _all_pvt_mode_labels:
                    _mk = _PVT_MODE_LABEL_TO_KEY[_ml]
                    _mt = _ml.replace(" ", "").replace("(", "_").replace(")", "")
                    _mode_plot_dir = os.path.join(PLOTS_DIR, f"LT_decoding_{_mt}")
                    os.makedirs(_mode_plot_dir, exist_ok=True)
                    print(f"\n[LT-DECODE] Running PVT mode {_ml!r} → {_mode_plot_dir}")
                    _mode_pvt = run_LT_position_vs_time_all_mice(
                        TFC_cond_LT1, TFC_cond_LT2,
                        PLOTS_DIR=_mode_plot_dir,
                        decoder_modes=[_mk],
                        **_pvt_common_kwargs,
                    )
                    run_decoder_diagnostic_stats(
                        _mode_pvt,
                        PLOTS_DIR=_mode_plot_dir,
                        mouse_groups=mouse_groups,
                        session_str=session_str_full,
                        auto_close=True,
                    )
                    _all_pvt_run_results[_ml] = (_mode_pvt, _mode_plot_dir)
            else:
                pvt_results = run_LT_position_vs_time_all_mice(
                    TFC_cond_LT1, TFC_cond_LT2,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    decoder_modes=_pvt_modes,
                    **_pvt_common_kwargs,
                )

            # Decoder-comparison statistics (4 modes × 3 conditions × 3 groups)
            # Only meaningful when all 4 modes were run.
            if use_separate_position_decoder:
                pvt_stats = run_pvt_decoder_comparison_stats(
                    pvt_results,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    mouse_groups=mouse_groups,
                    session_str=session_str_full,
                    auto_close=True,
                )

            # Decoder diagnostic stats (group-level Ksum / frac_active / entropy)
            # (already run per-mode above when _run_all_pvt_modes)
            if not _run_all_pvt_modes:
                pvt_diag_stats = run_decoder_diagnostic_stats(
                    pvt_results,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    mouse_groups=mouse_groups,
                    session_str=session_str_full,
                    auto_close=True,
                )

        # ============================================================
        # 0c) True-2D decoder pipeline (S_2D_full + PF_2D_full)
        # ============================================================
        if use_true_2D_decoder:
            LT_2D_NPY_DIR = os.path.join(NPY_SAVE_PATH, "LT_decoder_true_2D")
            os.makedirs(LT_2D_NPY_DIR, exist_ok=True)

            true_2d_results = run_LT_true_2D_all_mice(
                TFC_cond_LT1,
                TFC_cond_LT2,
                mapping=mapping,
                time_bin_frames=time_bin_frames_true_2D,
                use_speed=True,
                min_speed=2.0,
                use_z_score="per-session",
                use_posterior_mean=True,
                use_continuity_constraint=true_2D_use_continuity,
                **_lt_cont_2d,
                PLOTS_DIR=LT_PLOTS_DIR,
                NPY_SAVE_DIR=LT_2D_NPY_DIR,
                session_str=session_str_full,
                mouse_groups=mouse_groups,
                auto_close=True,
                enable_shuffle_control=enable_lt_shuffle_control,
                shuffle_type=decoder_shuffle_type,
                n_shuffles=decoder_shuffle_n_repeats,
                shuffle_seed=decoder_shuffle_seed,
            )

        # ============================================================
        # 1) Build dec_results + output directory for each downstream run.
        #    true_2D → two runs (S_2D_full, PF_2D_full) in separate dirs.
        #    Otherwise → single run.
        # ============================================================

        _downstream_runs = []   # list of (dec_results, plots_dir)

        if use_true_2D_decoder:
            for _2d_mode in ["S_2D_full", "PF_2D_full"]:
                _enc = "S" if _2d_mode.startswith("S") else "PF"
                _2d_dir = os.path.join(PLOTS_DIR, f"LT_decoding_2D_{_enc}")
                os.makedirs(_2d_dir, exist_ok=True)
                print(f"\n[LT-DECODE] Converting true-2D mode {_2d_mode!r} → dec_results → {_2d_dir}")
                _dr = convert_true_2D_to_dec_results(true_2d_results, _2d_mode)
                _downstream_runs.append((_dr, _2d_dir))
        elif use_separate_position_decoder:
            # ---------- Original pipeline ----------
            dec_results = run_decoding_all_mice_pipeline(
                TFC_cond_LT1,
                TFC_cond_LT2,
                mapping=mapping,
                min_laps_within=2,
                min_laps_transfer=2,
                n_pos_bins=60,
                time_bin_frames=time_bin_frames_orig,
                use_speed=True,
                min_speed=2.0,
                PLOTS_DIR=LT_PLOTS_DIR,
                session_str=session_str_full,
                mouse_groups=mouse_groups,
                plot_debug=True,
                auto_close=True,
            )

            # Position-vs-time plots for the original (no continuity) decoder
            plot_position_vs_time_original_decoder(
                dec_results,
                mouse_groups,
                PLOTS_DIR=LT_PLOTS_DIR,
                session_str=session_str_full,
                auto_close=True,
            )
            _downstream_runs.append((dec_results, LT_PLOTS_DIR))
        else:
            # ---------- Use PVT decoder output ----------
            if _run_all_pvt_modes:
                for _ml, (_pvt_res, _mode_dir) in _all_pvt_run_results.items():
                    print(f"\n[LT-DECODE] Converting PVT mode {_ml!r} → dec_results → {_mode_dir}")
                    _dr = convert_pvt_to_dec_results(_pvt_res, _ml)
                    _downstream_runs.append((_dr, _mode_dir))
            else:
                print(f"\n[LT-DECODE] Converting PVT mode {positional_decoder_to_use!r} → dec_results")
                dec_results = convert_pvt_to_dec_results(pvt_results, positional_decoder_to_use)
                _downstream_runs.append((dec_results, LT_PLOTS_DIR))

        # ============================================================
        # Downstream pipeline — runs once per entry in _downstream_runs
        # ============================================================
        for dec_results, LT_PLOTS_DIR in _downstream_runs:

            # Optional sanity check:
            m0 = next(iter(dec_results.keys()))
            print("[INFO] Example mouse:", m0)
            print("[INFO] Keys:", dec_results[m0].keys())
            print("[INFO] n_cells_full_LT1:", dec_results[m0].get("n_cells_full_LT1"))
            print("[INFO] n_cells_full_LT2:", dec_results[m0].get("n_cells_full_LT2"))
            print("[INFO] n_cells_mapped:",  dec_results[m0].get("n_cells_mapped"))


            # ============================================================
            # 1b) Aggregate error-density plots (raw errors, USE_MEDIAN-independent)
            # ============================================================

            # FULL-within
            plot_aggregate_error_density(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "aggregate_error_density"),
                session_str=session_str_full,
                bin_width_px=10.0,
                kmin=0.0,
                auto_close=True,
            )

            # FULL-within, Ksum >= 5 filter
            plot_aggregate_error_density(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "aggregate_error_density"),
                session_str=session_str_full,
                bin_width_px=10.0,
                kmin=5.0,
                auto_close=True,
            )

            # MAPPED-within proxy (build inline for the density plot)
            dec_results_mapped_density = {}
            for mouse, res in dec_results.items():
                w1m = res.get("within_LT1_by_dir_mapped")
                w2m = res.get("within_LT2_by_dir_mapped")
                if isinstance(w1m, dict) and isinstance(w2m, dict):
                    r2 = dict(res)
                    r2["within_LT1_by_dir"] = w1m
                    r2["within_LT2_by_dir"] = w2m
                    dec_results_mapped_density[mouse] = r2

            plot_aggregate_error_density(
                dec_results_mapped_density, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "aggregate_error_density"),
                session_str=session_str_mapped_within,
                bin_width_px=10.0,
                kmin=0.0,
                auto_close=True,
            )

            # ============================================================
            # 1c) Spatial decoding-error profile (mean |error| vs track pos)
            # ============================================================

            # FULL-within
            plot_spatial_decoding_error(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_profile"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=0.0,
                auto_close=True,
            )

            # FULL-within, Ksum >= 5
            plot_spatial_decoding_error(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_profile"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=5.0,
                auto_close=True,
            )

            # MAPPED-within
            plot_spatial_decoding_error(
                dec_results_mapped_density, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_profile"),
                session_str=session_str_mapped_within,
                n_pos_bins=20,
                kmin=0.0,
                auto_close=True,
            )

            # LT1->LT2 cross-session (mapped cells only)
            plot_spatial_decoding_error(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_profile"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=0.0,
                auto_close=True,
                sessions=("LT1_to_LT2",),
            )

            # LT1->LT2 cross-session, Ksum >= 5
            plot_spatial_decoding_error(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_profile"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=5.0,
                auto_close=True,
                sessions=("LT1_to_LT2",),
            )

            # ============================================================
            # 1d) Spatial decoding-error HEATMAP strips
            # ============================================================

            # FULL-within
            plot_spatial_decoding_error_heatmap(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_heatmap"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=0.0,
                auto_close=True,
            )

            # FULL-within, Ksum >= 5
            plot_spatial_decoding_error_heatmap(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_heatmap"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=5.0,
                auto_close=True,
            )

            # MAPPED-within
            plot_spatial_decoding_error_heatmap(
                dec_results_mapped_density, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_heatmap"),
                session_str=session_str_mapped_within,
                n_pos_bins=20,
                kmin=0.0,
                auto_close=True,
            )

            # LT1->LT2 cross-session
            plot_spatial_decoding_error_heatmap(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_heatmap"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=0.0,
                auto_close=True,
                sessions=("LT1_to_LT2",),
            )

            # LT1->LT2 cross-session, Ksum >= 5
            plot_spatial_decoding_error_heatmap(
                dec_results, mouse_groups,
                save_dir=os.path.join(LT_PLOTS_DIR, "spatial_error_heatmap"),
                session_str=session_str_full,
                n_pos_bins=20,
                kmin=5.0,
                auto_close=True,
                sessions=("LT1_to_LT2",),
            )


            # ============================================================
            # 1e) Zone-level lmer analysis  (group × zone per condition)
            #     Zones collapsed: endpoints dropped, horiz arms merged.
            #     DV depends on use_PCT_error switch.
            # ============================================================
            # Locate the zone errors CSV produced by the PVT decoder
            _pvt_mode_key_for_zone = _PVT_MODE_LABEL_TO_KEY.get(
                positional_decoder_to_use, positional_decoder_to_use)
            _dim_lbl = "2D" if "2D" in _pvt_mode_key_for_zone else "1D"
            _enc_lbl = "PF" if "PF" in _pvt_mode_key_for_zone else "S"
            _zone_csv = os.path.join(
                LT_PLOTS_DIR, "position_vs_time", _dim_lbl, _enc_lbl,
                "decoder_zone_error",
                f"{_pvt_mode_key_for_zone}_zone_errors.csv",
            )
            if os.path.isfile(_zone_csv):
                print(f"\n[ZONE-LMER] Zone CSV found: {_zone_csv}")
                print(f"[ZONE-LMER] use_PCT_error={use_PCT_error}")
                _zone_lmer_dir = os.path.join(LT_PLOTS_DIR, "zone_lmer_analysis")
                zone_lmer_results = run_lt_zone_lmer_analysis(
                    _zone_csv,
                    _zone_lmer_dir,
                    use_PCT_error=use_PCT_error,
                    auto_close=True,
                )
            else:
                print(f"[ZONE-LMER] Zone CSV not found, skipping: {_zone_csv}")
                zone_lmer_results = None


            # ============================================================
            # 1e‑bis) Zone cross‑registration analysis suite  (Phases 1–5)
            #
            # MOVED to standalone block after plot_LT_decoding.
            # See `if enable_zone_crossreg_analysis:` at module level.
            # ============================================================


            # ============================================================
            # 1.5) Neural diagnostics: baseline (LT1) group differences
            # ============================================================
            print("\n" + "=" * 70)
            print("LT1 NEURAL DIAGNOSTICS - checking for pre-CNO group differences")
            print("="*70)

            for diag_label, diag_kwargs in [
                ("FULL (all neurons)", {}),
                ("MAPPED (cross-registered)", {"mapping": mapping, "LT2_group": TFC_cond_LT2}),
            ]:
                print(f"\n{'─'*60}")
                print(f"  {diag_label}")
                print(f"{'─'*60}")

                tag = "full" if "FULL" in diag_label else "mapped"

                df_diag = compute_LT1_neural_diagnostics(
                    LT1_group=TFC_cond_LT1,
                    mouse_groups=mouse_groups,
                    dec_results=dec_results,
                    **diag_kwargs,
                )
                print("\n--- Per-mouse diagnostics ---")
                print(df_diag.to_string(index=False))

                # Save CSV
                diag_csv = os.path.join(LT_PLOTS_DIR, f"LT1_neural_diagnostics_{tag}.csv")
                df_diag.to_csv(diag_csv, index=False)
                print(f"\n[SAVED] {diag_csv}")

                # Statistical tests
                df_tests = run_neural_diagnostics_tests(df_diag)
                print("\n--- Kruskal-Wallis group comparisons ---")
                print(df_tests.to_string(index=False))
                tests_csv = os.path.join(LT_PLOTS_DIR, f"LT1_neural_diagnostics_KW_tests_{tag}.csv")
                df_tests.to_csv(tests_csv, index=False)
                print(f"[SAVED] {tests_csv}")

                # Group means summary
                print("\n--- Group means (± SEM) ---")
                for metric in ["n_neurons", "mean_event_rate", "mean_spatial_info",
                                "mean_Ksum", "n_laps_total", "mean_speed", "track_coverage"]:
                    print(f"\n  {metric}:")
                    for g in GROUPS:
                        vals = df_diag.loc[df_diag["group"] == g, metric].dropna().values
                        if len(vals) > 0:
                            m = np.mean(vals)
                            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                            print(f"    {GROUP_LABELS[g]:>4s}: {m:.3f} ± {se:.3f}  (n={len(vals)})")

                # Plot
                diag_plot = plot_neural_diagnostics(
                    df_diag, df_tests, PLOTS_DIR=LT_PLOTS_DIR,
                    out_name=f"LT1_neural_diagnostics_{tag}.png",
                )
                print(f"[INFO] diagnostics plot: {diag_plot}")

            # ============================================================
            # 2) ORIGINAL downstream: Ksum histograms (FULL-within dec_results)
            # ============================================================

            if use_PCT_error:
                _stat_iterations = [("pct", False)]   # (stat_label, USE_MEDIAN)
            else:
                _stat_iterations = [("mean", False), ("median", True)]

            for stat_label, USE_MEDIAN in _stat_iterations:
                _log_path = os.path.join(LT_PLOTS_DIR, f"mixedlm_{stat_label}.txt")
                _log_file = open(_log_path, "w", encoding="utf-8")
                _orig_stdout = sys.stdout

                class _Tee:
                    """Write to both console and log file."""
                    def __init__(self, *streams): self._streams = streams
                    def write(self, s):
                        for st in self._streams: st.write(s)
                    def flush(self):
                        for st in self._streams: st.flush()

                sys.stdout = _Tee(_orig_stdout, _log_file)

                print(f"\n{'=' * 60}")
                print(f"  Decoding downstream: stat_label = {stat_label}  (USE_MEDIAN={USE_MEDIAN}, use_PCT_error={use_PCT_error})")
                print(f"  use_separate_position_decoder   = {use_separate_position_decoder}")
                print(f"  positional_decoder_to_use       = {positional_decoder_to_use!r}")
                print(f"  use_true_2D_decoder             = {use_true_2D_decoder}")
                print(f"  LT_PLOTS_DIR                    = {LT_PLOTS_DIR}")
                print(f"{'=' * 60}\n")

                save_path = os.path.join(
                    LT_PLOTS_DIR,
                    f"Ksum_summary_{session_str_full}_mapping_{mapping}_{stat_label}",
                )
                os.makedirs(save_path, exist_ok=True)

                for cond, fname, nice in [
                    ("LT1_train_to_test", "Ksum_LT1_within_LOLO_by_group.png", "LT1 within (LOLO)"),
                    ("LT2_train_to_test", "Ksum_LT2_within_LOLO_by_group.png", "LT2 within (LOLO)"),
                    ("LT1_to_LT2",        "Ksum_LT1_to_LT2_by_group.png",        "LT1->LT2"),
                ]:
                    group_Ksum, _, _ = collect_group_Ksum(dec_results, mouse_groups, cond)
                    plot_group_Ksum_histograms(
                        group_Ksum,
                        title=f"{session_str_full} {nice} Ksum",
                        savefile=os.path.join(save_path, fname),
                    )

                # ============================================================
                # 3) ORIGINAL downstream: stats + summary CSVs + mixedlm CSVs (FULL-within)
                # ============================================================

                df_mixed_full, r_Ksum_full, r_err_full = plot_Ksum_fit_mixedlm(
                    LT1_group=TFC_cond_LT1,
                    LT2_group=TFC_cond_LT2,
                    dec_results=dec_results,
                    mouse_groups=mouse_groups,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    session_str=session_str_full,
                    mapping=mapping,
                    lowK_thresh=1,
                    use_median=USE_MEDIAN,
                    auto_close=True,
                    pct_threshold=pct_threshold,
                    stat_label_override=stat_label if use_PCT_error else None,
                )

                out_dir_full = run_group_summary_plots(
                    dec_results=dec_results,
                    mouse_groups=mouse_groups,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    session_str=session_str_full,
                    mapping=mapping,
                    auto_close=True,
                    use_median=USE_MEDIAN,
                )
                print(f"[DONE] FULL-within summary plots written to:\n{out_dir_full}")


                # ============================================================
                # 4) Build MAPPED-WITHIN proxy dec_results
                #    so downstream code "sees" mapped-within as within_LT1/within_LT2
                # ============================================================

                dec_results_mapped_within = {}
                skipped = []
                for mouse, res in dec_results.items():
                    w1m = res.get("within_LT1_by_dir_mapped")
                    w2m = res.get("within_LT2_by_dir_mapped")
                    if not (isinstance(w1m, dict) and isinstance(w2m, dict)):
                        skipped.append(mouse)
                        continue
                    res2 = dict(res)
                    res2["within_LT1_by_dir"] = w1m
                    res2["within_LT2_by_dir"] = w2m
                    dec_results_mapped_within[mouse] = res2

                print(f"[INFO] MAPPED-within proxy: kept {len(dec_results_mapped_within)} mice, skipped {len(skipped)}")


                # ============================================================
                # 5) MAPPED-WITHIN downstream: Ksum histograms
                # ============================================================

                save_path_mw = os.path.join(
                    LT_PLOTS_DIR,
                    f"Ksum_summary_{session_str_mapped_within}_mapping_{mapping}_{stat_label}",
                )
                os.makedirs(save_path_mw, exist_ok=True)

                for cond, fname, nice in [
                    ("LT1_train_to_test", "Ksum_LT1_within_LOLO_by_group.png", "LT1 within LOLO (mapped)"),
                    ("LT2_train_to_test", "Ksum_LT2_within_LOLO_by_group.png", "LT2 within LOLO (mapped)"),
                    ("LT1_to_LT2",        "Ksum_LT1_to_LT2_by_group.png",        "LT1->LT2 (same across)"),
                ]:
                    group_Ksum, _, _ = collect_group_Ksum(dec_results_mapped_within, mouse_groups, cond)
                    plot_group_Ksum_histograms(
                        group_Ksum,
                        title=f"{session_str_mapped_within} {nice} Ksum",
                        savefile=os.path.join(save_path_mw, fname),
                    )

                # ============================================================
                # 6) MAPPED-WITHIN downstream: stats + summary CSVs + mixedlm CSVs
                # ============================================================

                df_mixed_mw, r_Ksum_mw, r_err_mw = plot_Ksum_fit_mixedlm(
                    LT1_group=TFC_cond_LT1,
                    LT2_group=TFC_cond_LT2,
                    dec_results=dec_results_mapped_within,
                    mouse_groups=mouse_groups,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    session_str=session_str_mapped_within,
                    mapping=mapping,
                    lowK_thresh=1,
                    use_median=USE_MEDIAN,
                    auto_close=True,
                    pct_threshold=pct_threshold,
                    stat_label_override=stat_label if use_PCT_error else None,
                )

                out_dir_mw = run_group_summary_plots(
                    dec_results=dec_results_mapped_within,
                    mouse_groups=mouse_groups,
                    PLOTS_DIR=LT_PLOTS_DIR,
                    session_str=session_str_mapped_within,
                    mapping=mapping,
                    auto_close=True,
                    use_median=USE_MEDIAN,
                )
                print(f"[DONE] MAPPED-within summary plots written to:\n{out_dir_mw}")


                # ============================================================
                # 8) R lmer + emmeans for BOTH variants
                # ============================================================
                if use_PCT_error:
                    err_col = "pct_correct"
                else:
                    err_col = "median_err" if USE_MEDIAN else "mean_err"

                for variant_tag, sess_str in [
                    ("full",   session_str_full),
                    ("shared", session_str_mapped_within),
                ]:
                    r_save_dir = os.path.join(
                        LT_PLOTS_DIR,
                        f"lt_decoder_mixedlm_{variant_tag}_{stat_label}")

                    df_lmer = build_lt_mixedlm_dataframe(
                        LT_PLOTS_DIR, sess_str, mapping,
                        use_median=USE_MEDIAN,
                        use_pct=use_PCT_error,
                        stat_label_override=stat_label if use_PCT_error else None)

                    print(f"\n{'='*60}")
                    print(f"  [{variant_tag}] R lmer + emmeans  ({stat_label})")
                    print(f"  N = {len(df_lmer)} rows, "
                          f"{df_lmer['mouse'].nunique()} mice")
                    print(f"{'='*60}")

                    r_results = run_r_lmer_emmeans(
                        df_lmer, err_col, r_save_dir)

                    plot_lt_decoder_emmeans(
                        r_results, r_save_dir,
                        model_tag=variant_tag,
                        use_median=USE_MEDIAN,
                        y_col=err_col,
                        metric_label=stat_label)

                # Close log tee
                sys.stdout = _orig_stdout
                _log_file.close()
                print(f"[LOG] MixedLM output saved to {_log_path}")


    # ==============================================================================
    # ====  Zone cross-registration analysis suite  (Phases 1–9)  =================
    # ==============================================================================
    #
    # Standalone gate — can run without re-running the full plot_LT_decoding block,
    # as long as plot_LT_decoding (or at least plot_LT_pfs) has already run in the
    # same session so TFC_cond_LT1 / TFC_cond_LT2 / mouse_groups exist.
    #
    # Decoder settings are redeclared here so the block is self-contained.
    # ==============================================================================

    enable_zone_crossreg_analysis = True
    zone_crossreg_n_repeats = 100          # Phase 1 subsampling repeats
    zone_crossreg_popcurve_n_step = 20     # Phase 5 grid step
    zone_crossreg_popcurve_n_repeats = 50  # Phase 5 repeats per N

    # ===== end verbatim body =====
    plt.close('all')
    if interactive_before:
        plt.ion()
    return {"lt_cont_pvt": _lt_cont_pvt, "use_PCT_error": use_PCT_error}


# ---------------------------------------------------------------------------
# Section: zone_crossreg  (caban/main.py L2884-3037)
# ---------------------------------------------------------------------------
def run_zone_crossreg(ds, cfg, *, lt_cont_pvt=None, use_PCT_error=None):
    """Analysis section: zone_crossreg. Originally caban/main.py L2884-3037."""
    if not (cfg.plot_LT_decoding and cfg.enable_zone_crossreg_analysis):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    TFC_cond_LT1 = ds.TFC_cond_LT1
    TFC_cond_LT2 = ds.TFC_cond_LT2
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    plot_LT_decoding = cfg.plot_LT_decoding
    # --- cross-section state ---
    _lt_cont_pvt = lt_cont_pvt
    if _lt_cont_pvt is None:
        raise RuntimeError("zone_crossreg requires lt_cont_pvt from run_LT_decoding")
    if use_PCT_error is None:
        raise RuntimeError("zone_crossreg requires use_PCT_error from run_LT_decoding")

    # ===== verbatim body from caban/main.py =====
    enable_zone_crossreg_analysis = True
    zone_crossreg_n_repeats = 100          # Phase 1 subsampling repeats
    zone_crossreg_popcurve_n_step = 20     # Phase 5 grid step
    zone_crossreg_popcurve_n_repeats = 50  # Phase 5 repeats per N

    if plot_LT_decoding and enable_zone_crossreg_analysis:
        # ---- Decoder / mode settings (must match plot_LT_decoding) ----
        _zcra_positional_decoder = "PF (2D)"
        _zcra_pct_threshold = 15.0
        _zcra_time_bin_frames = 15

        _zcra_pvt_mode_key = _PVT_MODE_LABEL_TO_KEY.get(
            _zcra_positional_decoder, _zcra_positional_decoder)
        _zcra_dim_lbl = "2D" if "2D" in _zcra_pvt_mode_key else "1D"
        _zcra_enc_lbl = "PF" if "PF" in _zcra_pvt_mode_key else "S"
        _zcra_mode_tag = _zcra_positional_decoder.replace(" ", "").replace("(", "_").replace(")", "")
        _zcra_lt_dir_name = f"LT_decoding_{_zcra_mode_tag}"

        _ZCRA_LT_PLOTS_DIR = os.path.join(PLOTS_DIR, _zcra_lt_dir_name)
        os.makedirs(_ZCRA_LT_PLOTS_DIR, exist_ok=True)

        _ZCRA_NPY_DIR = os.path.join(NPY_SAVE_PATH, "zone_crossreg_analysis")
        os.makedirs(_ZCRA_NPY_DIR, exist_ok=True)

        print("\n" + "=" * 70)
        print("  ZONE CROSS-REGISTRATION ANALYSIS  (Phases 1–9)")
        print("=" * 70)

        _zcra_common = dict(
            LT1_dict=TFC_cond_LT1,
            LT2_dict=TFC_cond_LT2,
            mouse_groups=mouse_groups,
            PLOTS_DIR=_ZCRA_LT_PLOTS_DIR,
            auto_close=True,
        )
        _zcra_dec_kw = dict(
            pct_threshold=_zcra_pct_threshold,
            n_pos_bins=60,
            time_bin_frames=_zcra_time_bin_frames,
            use_speed=True,
            min_speed=2.0,
            use_z_score="per-session",
            use_posterior_mean=True,
            use_continuity_constraint=True,
            **_lt_cont_pvt,
        )

        # ---- Phase 1: N-matched subsampling control ----
        print("\n--- Phase 1: N-matched subsampling control ---")
        phase1_results = run_zone_decoder_subsampling_control(
            **_zcra_common, **_zcra_dec_kw,
            n_repeats=zone_crossreg_n_repeats,
            seed=42,
            NPY_SAVE_DIR=_ZCRA_NPY_DIR,
        )

        # ---- Phase 1b: Zone lmer with random cell-subset controls ----
        print("\n--- Phase 1b: Zone lmer cell-subset controls ---")
        phase1b_results = run_zone_lmer_cell_subset_controls(
            **_zcra_common, **_zcra_dec_kw,
            use_PCT_error=use_PCT_error,
            seed=42,
            NPY_SAVE_DIR=_ZCRA_NPY_DIR,
        )

        # ---- Phase 2: Characterize excluded vs included cells ----
        print("\n--- Phase 2: Cell characterization ---")
        phase2_results = characterize_excluded_vs_included_cells(
            **_zcra_common,
        )

        # ---- Phase 3: Group × crossreg interaction ----
        print("\n--- Phase 3: Group × crossreg interaction ---")
        _zcra_base_lt_dir = _zcra_lt_dir_name.replace("_xreg1", "")
        _zone_csv_pair = os.path.join(
            PLOTS_DIR, _zcra_base_lt_dir,
            "position_vs_time", _zcra_dim_lbl, _zcra_enc_lbl,
            "decoder_zone_error",
            f"{_zcra_pvt_mode_key}_zone_errors.csv",
        )
        _zone_csv_triple = os.path.join(
            PLOTS_DIR, _zcra_base_lt_dir + "_xreg1",
            "position_vs_time", _zcra_dim_lbl, _zcra_enc_lbl,
            "decoder_zone_error",
            f"{_zcra_pvt_mode_key}_zone_errors.csv",
        )
        if os.path.isfile(_zone_csv_pair) and os.path.isfile(_zone_csv_triple):
            phase3_results = run_zone_crossreg_interaction_analysis(
                _zone_csv_pair, _zone_csv_triple,
                save_dir=os.path.join(_ZCRA_LT_PLOTS_DIR,
                                      "zone_crossreg_analysis",
                                      "phase3_interaction"),
                auto_close=True,
            )
        else:
            print(f"[Phase3] Zone CSVs not both found, skipping.")
            print(f"  pair:   {_zone_csv_pair}  exists={os.path.isfile(_zone_csv_pair)}")
            print(f"  triple: {_zone_csv_triple}  exists={os.path.isfile(_zone_csv_triple)}")
            phase3_results = None

        # ---- Phase 4: PF stability by cross-reg status ----
        print("\n--- Phase 4: PF stability by cross-reg status ---")
        phase4_results = compare_pf_stability_by_crossreg_status(
            **_zcra_common,
        )

        # ---- Phase 5: Zone population curve ----
        print("\n--- Phase 5: Zone population curve ---")
        phase5_results = run_zone_population_curve(
            **_zcra_common, **_zcra_dec_kw,
            n_step=zone_crossreg_popcurve_n_step,
            n_repeats=zone_crossreg_popcurve_n_repeats,
            seed=42,
            NPY_SAVE_DIR=_ZCRA_NPY_DIR,
        )

        # ---- Phase 6: Junction place-field enrichment & coverage ----
        print("\n--- Phase 6: Junction place-field enrichment & coverage ---")
        phase6_results = run_zone_pf_enrichment_analysis(
            **_zcra_common,
        )

        # ---- Phase 7: Zone-resolved PV correlation (triple vs excluded) ----
        print("\n--- Phase 7: Zone PV correlation (triple vs excluded) ---")
        phase7_results = run_zone_pv_corr_comparison(
            **_zcra_common,
        )

        # ---- Phase 8: PF-filtered zone PV correlation (triple vs excluded) ----
        _phase8_methods = "zone_crossreg_phase8_zone_pv_corr_pf_methods.txt"
        for _pf_label, _pf_max in [("all PF", None), ("1 PF only", 1)]:
            print(f"\n--- Phase 8: Zone PV correlation – place cells ({_pf_label}) ---")
            run_zone_pv_corr_comparison(
                **_zcra_common,
                pf_place_cells_only=True,
                max_pf_count=_pf_max,
                phase_label="phase8",
                methods_template=_phase8_methods,
            )

        # ---- Phase 9: Place-field turnover (Ziv et al.) ----
        print("\n--- Phase 9: Place-field turnover (Ziv et al.) ---")
        phase9_results = run_zone_pf_turnover_analysis(
            **_zcra_common,
            TFC_cond_dict=TFC_cond,
        )

        print("\n" + "=" * 70)
        print("  ZONE CROSS-REGISTRATION ANALYSIS COMPLETE")
        print("=" * 70)

    #
    # ---- 2D open field PF decoder parameters (apply to all paradigm "2" / PF variants) ----
    #
    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: continuity_and_paramsets  (caban/main.py L3038-3215)
# ---------------------------------------------------------------------------
def run_continuity_and_paramsets(ds, cfg):
    """Analysis section: continuity_and_paramsets. Originally caban/main.py L3038-3215."""
    if not (True):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mouse_groups = ds.mouse_groups

    # ===== verbatim body from caban/main.py =====
    use_pf_num = -1               # -1 = use all PFs; >=1 = only cells with exactly this many PFs
    use_occupancy_fallback = False  # False = PF decoder skips cells with 0 PFs (no occupancy fallback)
    place_cells_only = True         # True = PF decoder uses only place cells; False = all cells
    encoder_period = "pre-tone"    # 'pre-tone' = first 180s (before tones); 'post-shock' = ITI after shocks; 'post-tone' = ITI after tones (Test_B only)
    use_z_score = "per-session"          # 'none' | 'per-session' | 'across-sessions' | 'optimize' (perform optimization for the first three possibilities)
    TIME_BIN_FRAMES = 15          # Shared decode chunk size for all 2D paradigms
    N_SPATIAL_BINS  = 20          # Spatial resolution (n_x_bins = n_y_bins) for all 2D paradigms
    use_posterior_mean = True    # True: posterior mean decode; False: hard MAP decode (argmax)
    decoder_type = 'bayesian'    # 'bayesian' | 'ridge-regression' for paradigms A-F
    ridge_alpha = 1.0            # Regularization strength for ridge-regression decoder
    use_scoring_method = 'both'  # 'euclidean' | 'rsquared' | 'both' (for decoder summaries/stats)
    pct_threshold_2D = 15.0      # cm; threshold for PCT metric (fraction of chunks with error < this)
    run_fixed_effects_models_tfc_cross_vs_within = True  # True: run OLS/ANOVA companion models in addition to MixedLM

    # Continuity constraint (Bayesian decoders only)
    continuity_preset = "data-driven"  # "data-driven" | "conservative" | "balanced" | "aggressive" | "custom"

    # --- TFC velocity histograms & data-driven sigma computation ----------------
    # Collect velocity stats from all open-field sessions for the 2D preset block.
    _tfc_vel_session_dicts = []
    _tfc_vel_tags = []
    for _tag, _sd in [("TFC_cond", TFC_cond), ("Test_A", Test_A), ("Test_A_1wk", Test_A_1wk),
                       ("Test_B", Test_B), ("Test_B_1wk", Test_B_1wk)]:
        if _sd:
            _tfc_vel_session_dicts.append(_sd)
            _tfc_vel_tags.append(_tag)

    if _tfc_vel_session_dicts:
        vel_stats_TFC = plot_velocity_histograms(
            *_tfc_vel_session_dicts,
            mouse_groups=mouse_groups,
            PLOTS_DIR=os.path.join(PLOTS_DIR, "TFC_2D_decoding"),
            session_tags=tuple(_tfc_vel_tags),
            auto_close=True,
        )
    else:
        vel_stats_TFC = collect_velocity_stats(TFC_cond)
    # ----------------------------------------------------------------------------

    # Defaults used when continuity_preset == "custom"
    use_continuity_constraint = True
    continuity_sigma_k = 60.0          # K in sigma_t = K * (v_t / V)^d (cm)
    continuity_speed_ref = 20.0        # V (same units as miniscope velocity vector, cm/s)
    continuity_exp = 1.0               # d; 1.0 ~ linear movement model, 0.5 ~ random walk
    continuity_sigma_min = 20.0        # Lower clamp for sigma_t (cm)
    continuity_sigma_max = 60.0        # Upper clamp for sigma_t (cm)
    continuity_sigma_default = 30.0    # Fallback sigma when velocity is unavailable/NaN (cm)

    if continuity_preset == "data-driven":
        use_continuity_constraint = True
        _dd = compute_continuity_params(vel_stats_TFC, time_bin_frames=TIME_BIN_FRAMES)
        save_continuity_params_txt(_dd, os.path.join(PLOTS_DIR, "TFC_2D_decoding"),
                                   tag=f"TFC data-driven (time_bin_frames={TIME_BIN_FRAMES})",
                                   time_bin_frames=TIME_BIN_FRAMES)
        continuity_sigma_k       = _dd['continuity_sigma_k']
        continuity_speed_ref     = _dd['continuity_speed_ref']
        continuity_exp           = _dd['continuity_exp']
        continuity_sigma_min     = _dd['continuity_sigma_min']
        continuity_sigma_max     = _dd['continuity_sigma_max']
        continuity_sigma_default = _dd['continuity_sigma_default']
    elif continuity_preset == "conservative":
        use_continuity_constraint = True
        continuity_sigma_k = 45.0
        continuity_speed_ref = 20.0
        continuity_exp = 1.0
        continuity_sigma_min = 15.0
        continuity_sigma_max = 45.0
        continuity_sigma_default = 25.0
    elif continuity_preset == "balanced":
        use_continuity_constraint = True
        continuity_sigma_k = 60.0
        continuity_speed_ref = 20.0
        continuity_exp = 1.0
        continuity_sigma_min = 20.0
        continuity_sigma_max = 60.0
        continuity_sigma_default = 30.0
    elif continuity_preset == "aggressive":
        use_continuity_constraint = True
        continuity_sigma_k = 80.0
        continuity_speed_ref = 20.0
        continuity_exp = 1.0
        continuity_sigma_min = 25.0
        continuity_sigma_max = 80.0
        continuity_sigma_default = 40.0
    elif continuity_preset == "custom":
        use_continuity_constraint = True
        continuity_sigma_k        = 35.0   # baseline scale
        continuity_speed_ref      = 20.0
        continuity_exp            = 1.0
        continuity_sigma_min      = 10.0   # tight floor when stationary
        continuity_sigma_max      = 30.0   # hard ceiling ~2 bins of arena
        continuity_sigma_default  = 18.0
    else:
        raise ValueError(
            f"Invalid continuity_preset={continuity_preset!r}. "
            f"Use one of: 'data-driven', 'conservative', 'balanced', 'aggressive', 'custom'."
        )

    # Optional hyperparameter optimization for 2D Bayesian decoder.
    # Runs per-group + pooled studies, then cross-evaluates across all mice/groups.
    optimize_parameters = False
    optimization_trials = 50
    optimization_seed = 42
    optimization_first_n_sec = 180.0
    optimization_target = "within_TFC"
    optimization_use_speed = True
    optimization_min_speed = 2.0
    optimization_n_startup_trials = 12
    optimization_load_cached = False         # True: skip optimization, reload params from cache
    optimization_param_set = "mCherry"      # Which param set to apply: "baseline" (hand-set ones above) | "pooled" | "hM3D" | "hM4D" | "mCherry"
    optimization_jitter_lambda = 0.5        # Weight of temporal jitter penalty in optimizer objective (0 = no jitter penalty)

    # PF-based decoder optimizer (separate from raw 2D; uses place-field rate maps + continuity)
    optimize_pf_parameters = False
    optimization_pf_trials = 50
    optimization_pf_seed = 42
    optimization_pf_first_n_sec = 180.0
    optimization_pf_use_speed = True
    optimization_pf_min_speed = 2.0
    optimization_pf_n_startup_trials = 12
    optimization_pf_load_cached = False     # True: skip PF optimizer, reload from cache
    optimization_pf_param_set = "pooled"   # Which PF param set to apply downstream ("baseline" (hand-set ones above) | "pooled" | "hM3D" | "hM4D" | "mCherry")
    optimization_pf_jitter_lambda = 0.5    # Weight of temporal jitter penalty in PF optimizer objective (0 = no jitter penalty)

    # Population-size curve mode for cross-session 2D decoding.
    enable_population_curve = False
    population_curve_n_values = None      # Example manual grid: [10, 20, 30, 40, 50]
    population_curve_n_values_max_shared = True  # True: use shared-N cap = min availability across included pairs
    population_curve_per_pair_grid = True   # True: each pair builds its own 0..step..n_available grid (ignores global grid)
    population_curve_n_step = 20
    population_curve_quantile = 0.25      # N-cap = this quantile of matched-cell availability
    population_curve_repeats = 100
    population_curve_seed = 42
    population_curve_metric = "median_err"  # "median_err" | "mean_err"
    manual_killswitch = False
    killswitch_prompt_every = "N_step"      # prompt between N-grid values
    population_curve_print_level = "medium" # "low" | "medium" | "high"
    population_curve_suffix = "_population_curve"
    population_curve_load_cached = True    # True: skip decoder loop and reload from NPY cache


    def _popcurve_dirname(name: str) -> str:
        """Append population-curve suffix to run directories only when enabled."""
        if enable_population_curve:
            return f"{name}{population_curve_suffix}"
        return name

    # Default paramset instances — used by wrappers and call sites.
    # The optimizer blocks below will update these in-place if enabled.
    raw_params = BayesianDecoderParamset(
        n_spatial_bins=N_SPATIAL_BINS,
        time_bin_frames=TIME_BIN_FRAMES,
        use_posterior_mean=use_posterior_mean,
        use_z_score=use_z_score,
        use_continuity_constraint=use_continuity_constraint,
        continuity_sigma_k=continuity_sigma_k,
        continuity_speed_ref=continuity_speed_ref,
        continuity_exp=continuity_exp,
        continuity_sigma_min=continuity_sigma_min,
        continuity_sigma_max=continuity_sigma_max,
        continuity_sigma_default=continuity_sigma_default,
    )
    pf_params = BayesianDecoderParamset(
        time_bin_frames=TIME_BIN_FRAMES,
        use_posterior_mean=use_posterior_mean,
        use_z_score=use_z_score,
        use_continuity_constraint=use_continuity_constraint,
        continuity_sigma_k=continuity_sigma_k,
        continuity_speed_ref=continuity_speed_ref,
        continuity_exp=continuity_exp,
        continuity_sigma_min=continuity_sigma_min,
        continuity_sigma_max=continuity_sigma_max,
        continuity_sigma_default=continuity_sigma_default,
        place_cells_only=place_cells_only,
        use_pf_num=use_pf_num,
        use_occupancy_fallback=use_occupancy_fallback,
    )

    # ===== end verbatim body =====
    return {"raw_params": raw_params, "pf_params": pf_params}


# ---------------------------------------------------------------------------
# Section: optimize_raw_decoder  (caban/main.py L3216-3285)
# ---------------------------------------------------------------------------
def run_optimize_raw_decoder(ds, cfg):
    """Analysis section: optimize_raw_decoder. Originally caban/main.py L3216-3285."""
    if not (cfg.optimize_parameters and cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    encoder_period = cfg.encoder_period
    optimization_first_n_sec = cfg.optimization_first_n_sec
    optimization_jitter_lambda = cfg.optimization_jitter_lambda
    optimization_load_cached = cfg.optimization_load_cached
    optimization_min_speed = cfg.optimization_min_speed
    optimization_n_startup_trials = cfg.optimization_n_startup_trials
    optimization_seed = cfg.optimization_seed
    optimization_target = cfg.optimization_target
    optimization_trials = cfg.optimization_trials
    optimization_use_speed = cfg.optimization_use_speed
    optimize_parameters = cfg.optimize_parameters
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding

    # ===== verbatim body from caban/main.py =====
    if optimize_parameters and plot_TFC_2D_decoding:
        import json as _json

        _opt_cache_dir = os.path.join(NPY_SAVE_PATH, "2D_decoder_optimization_cache")
        _opt_cache_path = os.path.join(_opt_cache_dir, "optimized_decoder_params_grouped.json")

        _raw_defaults = raw_params.copy()  # keep a snapshot for summary printing

        # ---- Run optimizer (or reload from cache with plots/stats only) ----
        grp_result = optimize_2d_decoder_parameters_grouped(
            TFC_cond,
            mouse_groups=mouse_groups,
            PLOTS_DIR=PLOTS_DIR,
            encoder_period=encoder_period,
            paramset=raw_params,
            optimization_trials=optimization_trials,
            optimization_seed=optimization_seed,
            optimization_first_n_sec=optimization_first_n_sec,
            optimization_target=optimization_target,
            optimization_use_speed=optimization_use_speed,
            optimization_min_speed=optimization_min_speed,
            optimization_n_startup_trials=optimization_n_startup_trials,
            skip_optimization=optimization_load_cached,
            jitter_lambda=optimization_jitter_lambda,
        )
        _all_param_sets = grp_result.get("param_sets", {"baseline": _raw_defaults.to_dict()})

        # Save/update cache (always write so npy_files cache stays in sync)
        os.makedirs(_opt_cache_dir, exist_ok=True)
        with open(_opt_cache_path, "w", encoding="utf-8") as _f:
            _json.dump(_all_param_sets, _f, indent=2)
        print(f"[OPT] Cached grouped params -> {_opt_cache_path}")

        # ---- Select which param set to apply ----
        if optimization_param_set not in _all_param_sets:
            print(f"[OPT][WARN] Requested param set '{optimization_param_set}' not found in cache.")
            print(f"[OPT][WARN] Available: {list(_all_param_sets.keys())}. Falling back to 'baseline'.")
            optimization_param_set = "baseline"
            if "baseline" not in _all_param_sets:
                _all_param_sets["baseline"] = _raw_defaults.to_dict()

        applied_opt_params = _all_param_sets[optimization_param_set]
        print(f"[OPT] Applying param set: '{optimization_param_set}'")

        # Apply optimized parameters into the paramset instance
        raw_params.update_from_dict(applied_opt_params)
        # use_z_score: only override if user set it to "optimize" (delegated to Optuna)
        if str(_raw_defaults.use_z_score).strip().lower() != "optimize":
            raw_params.use_z_score = _raw_defaults.use_z_score  # keep user-set value
        else:
            print(f"[OPT] use_z_score resolved by optimizer: '{raw_params.use_z_score}'")

        # Also update legacy globals so non-decoder code still works
        N_SPATIAL_BINS = raw_params.n_spatial_bins
        TIME_BIN_FRAMES = raw_params.time_bin_frames
        use_posterior_mean = raw_params.use_posterior_mean
        use_z_score = raw_params.use_z_score
        use_continuity_constraint = raw_params.use_continuity_constraint
        continuity_sigma_k = raw_params.continuity_sigma_k
        continuity_speed_ref = raw_params.continuity_speed_ref
        continuity_exp = raw_params.continuity_exp
        continuity_sigma_min = raw_params.continuity_sigma_min
        continuity_sigma_max = raw_params.continuity_sigma_max
        continuity_sigma_default = raw_params.continuity_sigma_default

        raw_params.print_summary(_raw_defaults, label=f"raw 2D, param_set='{optimization_param_set}'")

    # ---------------------------------------------------------------------------
    #  PF-based decoder parameter optimization
    # ---------------------------------------------------------------------------
    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: optimize_pf_decoder  (caban/main.py L3286-3537)
# ---------------------------------------------------------------------------
def run_optimize_pf_decoder(ds, cfg):
    """Analysis section: optimize_pf_decoder. Originally caban/main.py L3286-3537."""
    if not (cfg.optimize_pf_parameters and cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    encoder_period = cfg.encoder_period
    optimization_pf_first_n_sec = cfg.optimization_pf_first_n_sec
    optimization_pf_jitter_lambda = cfg.optimization_pf_jitter_lambda
    optimization_pf_load_cached = cfg.optimization_pf_load_cached
    optimization_pf_min_speed = cfg.optimization_pf_min_speed
    optimization_pf_n_startup_trials = cfg.optimization_pf_n_startup_trials
    optimization_pf_seed = cfg.optimization_pf_seed
    optimization_pf_trials = cfg.optimization_pf_trials
    optimization_pf_use_speed = cfg.optimization_pf_use_speed
    optimize_pf_parameters = cfg.optimize_pf_parameters
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding

    # ===== verbatim body from caban/main.py =====
    if optimize_pf_parameters and plot_TFC_2D_decoding:
        import json as _json_pf

        _pf_opt_cache_dir = os.path.join(NPY_SAVE_PATH, "2D_pf_decoder_optimization_cache")
        _pf_opt_cache_path = os.path.join(_pf_opt_cache_dir, "optimized_pf_decoder_params_grouped.json")

        _pf_defaults = pf_params.copy()

        grp_result_pf = optimize_2d_pf_decoder_parameters_grouped(
            TFC_cond,
            mouse_groups=mouse_groups,
            PLOTS_DIR=PLOTS_DIR,
            encoder_period=encoder_period,
            paramset=pf_params,
            optimization_trials=optimization_pf_trials,
            optimization_seed=optimization_pf_seed,
            optimization_first_n_sec=optimization_pf_first_n_sec,
            optimization_use_speed=optimization_pf_use_speed,
            optimization_min_speed=optimization_pf_min_speed,
            optimization_n_startup_trials=optimization_pf_n_startup_trials,
            skip_optimization=optimization_pf_load_cached,
            jitter_lambda=optimization_pf_jitter_lambda,
        )
        _all_pf_param_sets = grp_result_pf.get("param_sets", {"baseline": _pf_defaults.to_dict()})

        # Save/update cache (always write so npy_files cache stays in sync)
        os.makedirs(_pf_opt_cache_dir, exist_ok=True)
        with open(_pf_opt_cache_path, "w", encoding="utf-8") as _f:
            _json_pf.dump(_all_pf_param_sets, _f, indent=2)
        print(f"[OPT-PF] Cached PF params -> {_pf_opt_cache_path}")

        # Select which PF param set to apply
        if optimization_pf_param_set not in _all_pf_param_sets:
            print(f"[OPT-PF][WARN] Requested PF param set '{optimization_pf_param_set}' not found.")
            print(f"[OPT-PF][WARN] Available: {list(_all_pf_param_sets.keys())}. Falling back to 'baseline'.")
            optimization_pf_param_set = "baseline"
            if "baseline" not in _all_pf_param_sets:
                _all_pf_param_sets["baseline"] = _pf_defaults.to_dict()

        applied_pf_opt_params = _all_pf_param_sets[optimization_pf_param_set]
        print(f"[OPT-PF] Applying PF param set: '{optimization_pf_param_set}'")

        # Apply ALL optimized PF params — fully independent from raw_params
        pf_params.update_from_dict(applied_pf_opt_params)
        # use_z_score: only override if user set it to "optimize"
        if str(_pf_defaults.use_z_score).strip().lower() != "optimize":
            pf_params.use_z_score = _pf_defaults.use_z_score
        else:
            print(f"[OPT-PF] use_z_score resolved by PF optimizer: '{pf_params.use_z_score}'")

        pf_params.print_summary(_pf_defaults, label=f"PF, pf_param_set='{optimization_pf_param_set}'")

    #encoder_period = "post-shock"
    #encoder_period = "post-tone" # for Test_B only; post-shock for everything else (to maximize mice with .fm)

    # Ensure all downstream multi-target 2D decoder calls use current continuity parameters.
    _run_2D_decoder_all_mice_base = run_2D_decoder_all_mice

    def _count_csv_rows(csv_path: str):
        """Count data rows (excluding header) in a CSV file; return None if missing/unreadable."""
        try:
            if not os.path.exists(csv_path):
                return None
            with open(csv_path, "r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            return max(0, n_lines - 1)
        except Exception:
            return None


    def _main_z_score_dir_tag(use_z_score_mode: str) -> str:
        """Main-local mirror of decoder z-score directory naming."""
        mode = str(use_z_score_mode).strip().lower()
        if mode in ("per-session", "per_session", "per"):
            return "z-score_per"
        if mode in ("across-sessions", "across_sessions", "across"):
            return "z-score_across"
        return "z-score_none"


    def _main_decoder_is_ridge(decoder_type_mode: str) -> bool:
        """Main-local mirror of decoder ridge mode detector."""
        dt = str(decoder_type_mode).strip().lower()
        return dt in ("ridge", "ridge-regression", "ridge_regression")


    def _print_population_curve_summary_from_kwargs(kwargs: dict, *, pf_variant: bool):
        """Print a concise end-of-run summary of curve output paths and row counts."""
        if not bool(kwargs.get("enable_population_curve", False)):
            return

        plots_dir = str(kwargs.get("PLOTS_DIR", ""))
        train_label_local = str(kwargs.get("train_label", "TFC_cond"))
        session_str_local = str(kwargs.get("session_str", ""))
        ztag = _main_z_score_dir_tag(kwargs.get("use_z_score", use_z_score))
        dec_type_local = kwargs.get("decoder_type", decoder_type)
        suffix_local = str(kwargs.get("population_curve_suffix", "_population_curve"))
        _shuffle_enabled_local = bool(kwargs.get("enable_shuffle_control", False))
        _shuffle_suffix_local = "_shuffle" if _shuffle_enabled_local else ""

        if pf_variant:
            pf_str_local = "pcells" if bool(kwargs.get("place_cells_only", True)) else "allcells"
            decoder_dir_tag = "ridge_2D_PF" if _main_decoder_is_ridge(dec_type_local) else "bayes_2D_PF"
            save_path_local = os.path.join(
                plots_dir,
                f"{decoder_dir_tag}_multitgt_train_{train_label_local}_{session_str_local}_{pf_str_local}_{ztag}{suffix_local}{_shuffle_suffix_local}",
            )
            tables_dir = os.path.join(save_path_local, "population_curve_tables_pf")
            tag = "[MAIN->2D-PF]"
        else:
            decoder_dir_tag = "ridge_2D" if _main_decoder_is_ridge(dec_type_local) else "bayes_2D"
            save_path_local = os.path.join(
                plots_dir,
                f"{decoder_dir_tag}_multitgt_train_{train_label_local}_{session_str_local}_{ztag}{suffix_local}{_shuffle_suffix_local}",
            )
            tables_dir = os.path.join(save_path_local, "population_curve_tables")
            tag = "[MAIN->2D]"

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

    def run_2D_decoder_all_mice(train_sessions, test_targets, **kwargs):
        # Inject all optimized decoder params from the raw-2D paramset
        # Exclude PF-only keys that the raw-S decoder doesn't accept.
        _raw_keys = [k for k in raw_params.__slots__
                     if k not in ('place_cells_only', 'use_pf_num', 'use_occupancy_fallback')]
        raw_params.inject_into(kwargs, keys=_raw_keys, aliases={'n_spatial_bins': ['n_x_bins', 'n_y_bins']})
        kwargs.setdefault("opt_tag", optimization_param_set)
        kwargs.setdefault("enable_population_curve", enable_population_curve)
        kwargs.setdefault("population_curve_n_values", population_curve_n_values)
        kwargs.setdefault("population_curve_n_values_max_shared", population_curve_n_values_max_shared)
        kwargs.setdefault("population_curve_per_pair_grid", population_curve_per_pair_grid)
        kwargs.setdefault("population_curve_n_step", population_curve_n_step)
        kwargs.setdefault("population_curve_quantile", population_curve_quantile)
        kwargs.setdefault("population_curve_repeats", population_curve_repeats)
        kwargs.setdefault("population_curve_seed", population_curve_seed)
        kwargs.setdefault("population_curve_metric", population_curve_metric)
        kwargs.setdefault("manual_killswitch", manual_killswitch)
        kwargs.setdefault("killswitch_prompt_every", killswitch_prompt_every)
        kwargs.setdefault("population_curve_print_level", population_curve_print_level)
        kwargs.setdefault("population_curve_suffix", population_curve_suffix)
        kwargs.setdefault("population_curve_load_cached", population_curve_load_cached)

        # If a caller passed None explicitly, fall back to global defaults.
        if kwargs.get("population_curve_n_values", None) is None:
            kwargs["population_curve_n_values"] = population_curve_n_values
        if kwargs.get("population_curve_n_step", None) is None:
            kwargs["population_curve_n_step"] = population_curve_n_step
        if kwargs.get("population_curve_n_values_max_shared", None) is None:
            kwargs["population_curve_n_values_max_shared"] = population_curve_n_values_max_shared
        if kwargs.get("population_curve_quantile", None) is None:
            kwargs["population_curve_quantile"] = population_curve_quantile
        if kwargs.get("population_curve_repeats", None) is None:
            kwargs["population_curve_repeats"] = population_curve_repeats

        _train_label = str(kwargs.get("train_label", "")).lower()
        _session_str = str(kwargs.get("session_str", "")).lower()
        _is_lt_family = ("lt" in _train_label) or ("lt" in _session_str)
        kwargs.setdefault("enable_shuffle_control", enable_lt_shuffle_control if _is_lt_family else enable_tfc_shuffle_control)
        kwargs.setdefault("shuffle_type", decoder_shuffle_type)
        kwargs.setdefault("n_shuffles", decoder_shuffle_n_repeats)
        kwargs.setdefault("shuffle_seed", decoder_shuffle_seed)

        print(
            "[MAIN->2D] population-curve config: "
            f"enable={kwargs.get('enable_population_curve')} "
            f"n_values={kwargs.get('population_curve_n_values')} "
            f"max_shared={kwargs.get('population_curve_n_values_max_shared')} "
            f"require_exact_mapping={kwargs.get('require_exact_mapping', False)} "
            f"n_step={kwargs.get('population_curve_n_step')} "
            f"quantile={kwargs.get('population_curve_quantile')} "
            f"repeats={kwargs.get('population_curve_repeats')}"
        )
        results = _run_2D_decoder_all_mice_base(train_sessions, test_targets, **kwargs)
        _print_population_curve_summary_from_kwargs(kwargs, pf_variant=False)
        return results


    _run_2D_PF_decoder_all_mice_base = run_2D_PF_decoder_all_mice

    def run_2D_PF_decoder_all_mice(train_sessions, test_targets, **kwargs):
        # Inject all optimized PF-decoder params from the PF paramset.
        # Exclude n_spatial_bins (PF decoder uses place-field grid) and
        # use_z_score (PF rate maps are precomputed from original-scale data;
        # z-scoring the spike counts breaks the Poisson likelihood because
        # K becomes negative while λ stays in original scale).
        _pf_keys = [k for k in pf_params.__slots__
                     if k not in ('n_spatial_bins', 'use_z_score')]
        pf_params.inject_into(kwargs, keys=_pf_keys)
        kwargs.setdefault("opt_tag", optimization_pf_param_set)
        kwargs.setdefault("enable_population_curve", enable_population_curve)
        kwargs.setdefault("population_curve_n_values", population_curve_n_values)
        kwargs.setdefault("population_curve_n_values_max_shared", population_curve_n_values_max_shared)
        kwargs.setdefault("population_curve_per_pair_grid", population_curve_per_pair_grid)
        kwargs.setdefault("population_curve_n_step", population_curve_n_step)
        kwargs.setdefault("population_curve_quantile", population_curve_quantile)
        kwargs.setdefault("population_curve_repeats", population_curve_repeats)
        kwargs.setdefault("population_curve_seed", population_curve_seed)
        kwargs.setdefault("population_curve_metric", population_curve_metric)
        kwargs.setdefault("manual_killswitch", manual_killswitch)
        kwargs.setdefault("killswitch_prompt_every", killswitch_prompt_every)
        kwargs.setdefault("population_curve_print_level", population_curve_print_level)
        kwargs.setdefault("population_curve_suffix", population_curve_suffix)
        kwargs.setdefault("population_curve_load_cached", population_curve_load_cached)

        # If a caller passed None explicitly, fall back to global defaults.
        if kwargs.get("population_curve_n_values", None) is None:
            kwargs["population_curve_n_values"] = population_curve_n_values
        if kwargs.get("population_curve_n_step", None) is None:
            kwargs["population_curve_n_step"] = population_curve_n_step
        if kwargs.get("population_curve_n_values_max_shared", None) is None:
            kwargs["population_curve_n_values_max_shared"] = population_curve_n_values_max_shared
        if kwargs.get("population_curve_quantile", None) is None:
            kwargs["population_curve_quantile"] = population_curve_quantile
        if kwargs.get("population_curve_repeats", None) is None:
            kwargs["population_curve_repeats"] = population_curve_repeats

        _train_label = str(kwargs.get("train_label", "")).lower()
        _session_str = str(kwargs.get("session_str", "")).lower()
        _is_lt_family = ("lt" in _train_label) or ("lt" in _session_str)
        kwargs.setdefault("enable_shuffle_control", enable_lt_shuffle_control if _is_lt_family else enable_tfc_shuffle_control)
        kwargs.setdefault("shuffle_type", decoder_shuffle_type)
        kwargs.setdefault("n_shuffles", decoder_shuffle_n_repeats)
        kwargs.setdefault("shuffle_seed", decoder_shuffle_seed)

        print(
            "[MAIN->2D-PF] population-curve config: "
            f"enable={kwargs.get('enable_population_curve')} "
            f"n_values={kwargs.get('population_curve_n_values')} "
            f"max_shared={kwargs.get('population_curve_n_values_max_shared')} "
            f"require_exact_mapping={kwargs.get('require_exact_mapping', False)} "
            f"n_step={kwargs.get('population_curve_n_step')} "
            f"quantile={kwargs.get('population_curve_quantile')} "
            f"repeats={kwargs.get('population_curve_repeats')}"
        )
        results = _run_2D_PF_decoder_all_mice_base(train_sessions, test_targets, **kwargs)
        _print_population_curve_summary_from_kwargs(kwargs, pf_variant=True)
        return results



    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: paradigm_A  (caban/main.py L3783-4016)
# ---------------------------------------------------------------------------
def run_paradigm_A(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_A. Originally caban/main.py L3783-4016."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    N_SPATIAL_BINS = cfg.N_SPATIAL_BINS
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_A requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_A requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _paradigm_ABC_mapping = _make_paradigm_ABC_mapping(ds)
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_A_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"A1_TFC_2D_multi_target_encode_{encoder_period}"))
        os.makedirs(MT_A_PLOTS_DIR, exist_ok=True)

        # Build target dicts - only mice that have each session
        tfc_mice_for_mt = {m: TFC_cond[m] for m in TFC_cond
                           if m in Test_A or m in Test_B or m in Test_B_1wk or m in Test_A_1wk}
        targets_A = {}
        if Test_B:
            targets_A["Test_B"] = {m: Test_B[m] for m in Test_B if m in tfc_mice_for_mt}
        if Test_B_1wk:
            targets_A["Test_B_1wk"] = {m: Test_B_1wk[m] for m in Test_B_1wk if m in tfc_mice_for_mt}
        if Test_A:
            targets_A["Test_A"] = {m: Test_A[m] for m in Test_A if m in tfc_mice_for_mt}
        if Test_A_1wk:
            targets_A["Test_A_1wk"] = {m: Test_A_1wk[m] for m in Test_A_1wk if m in tfc_mice_for_mt}

        if targets_A:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_A_PLOTS_DIR, "Paradigm A", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm A (2D): Train TFC_cond -> {list(targets_A.keys())}")
            print(f"  {len(tfc_mice_for_mt)} mice")
            print(f"{'='*60}\n")

            mt_A_results_2D = run_2D_decoder_all_mice(
                tfc_mice_for_mt, targets_A,
                train_label="TFC_cond",
                mapping=_paradigm_ABC_mapping,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_A_PLOTS_DIR, session_str="paradigmA",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_A_results_2D, mouse_groups,
                    PLOTS_DIR=MT_A_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmA",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            # Spatial error heatmaps (2D raw-S, Paradigm A)
            plot_decoder_spatial_error_heatmaps(
                mt_A_results_2D, mouse_groups,
                PLOTS_DIR=MT_A_PLOTS_DIR,
                train_label="TFC_cond", session_str="paradigmA",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm A)
            plot_mobility_seconds_boxplots(
                mt_A_results_2D, mouse_groups,
                PLOTS_DIR=MT_A_PLOTS_DIR,
                train_label="TFC_cond", decoder_tag="2D",
                session_str="paradigmA", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm A)
            plot_xy_time_per_mouse(
                mt_A_results_2D,
                PLOTS_DIR=MT_A_PLOTS_DIR,
                train_label="TFC_cond",
                session_str="paradigmA",
                decoder_tag="2D",
                auto_close=True,
            )

            _stop_paradigm_log(_plog_orig, _plog_file, MT_A_PLOTS_DIR)

        # ==================================================================
        # ====  PARADIGM A-BEHAV: Error vs Behaviour (freezing, speed, coverage)
        # ====  Three sub-cases with different condition pooling strategies
        # ==================================================================
        MT_ABEHAV_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"A_Behav_error_vs_behavior_encode_{encoder_period}"))
        os.makedirs(MT_ABEHAV_DIR, exist_ok=True)

        _abehav_common = dict(
            mouse_groups=mouse_groups,
            train_sessions=tfc_mice_for_mt,
            test_targets=targets_A,
            PLOTS_DIR=MT_ABEHAV_DIR,
            train_label="TFC_cond",
            min_speed=2.0,
            first_n_sec=180.0,
            n_x_bins=N_SPATIAL_BINS, n_y_bins=N_SPATIAL_BINS,
            encoder_period=encoder_period,
            auto_close=True,
        )

        # A-Behav-Separate: each condition analysed individually
        try:
            run_error_vs_behavior_analysis(
                mt_A_results_2D, pooling_mode="separate", **_abehav_common)
        except Exception as e:
            print(f"[A-Behav-Separate] Failed: {e}")
            import traceback; traceback.print_exc()

        # A-Behav-Pooled: pool 48hr+1wk per context (cross_Test_A, cross_Test_B)
        try:
            run_error_vs_behavior_analysis(
                mt_A_results_2D, pooling_mode="pooled", **_abehav_common)
        except Exception as e:
            print(f"[A-Behav-Pooled] Failed: {e}")
            import traceback; traceback.print_exc()

        # A-Behav-Cross_session: all cross-session conditions merged
        try:
            run_error_vs_behavior_analysis(
                mt_A_results_2D, pooling_mode="cross_session", **_abehav_common)
        except Exception as e:
            print(f"[A-Behav-Cross_session] Failed: {e}")
            import traceback; traceback.print_exc()

        # --- 2D PF-based decoder ---
        MT_A_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"A1_TFC_2D_PF_multi_target_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_A_PF_PLOTS_DIR, exist_ok=True)

        tfc_mice_with_fm = {m: TFC_cond[m] for m in tfc_mice_for_mt
                            if hasattr(TFC_cond[m], 'fm') and TFC_cond[m].fm is not None}

        if tfc_mice_with_fm and targets_A:
            targets_A_pf = {k: {m: v[m] for m in v if m in tfc_mice_with_fm}
                            for k, v in targets_A.items()}
            # Remove empty targets
            targets_A_pf = {k: v for k, v in targets_A_pf.items() if v}

            if targets_A_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_A_PF_PLOTS_DIR, "Paradigm A", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm A (PF): Train TFC_cond PF -> {list(targets_A_pf.keys())}")
                print(f"  {len(tfc_mice_with_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_A_results_PF = run_2D_PF_decoder_all_mice(
                    tfc_mice_with_fm, targets_A_pf,
                    train_label="TFC_cond",
                    mapping=_paradigm_ABC_mapping,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_A_PF_PLOTS_DIR, session_str="paradigmA",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_A_results_PF, mouse_groups,
                        PLOTS_DIR=MT_A_PF_PLOTS_DIR,
                        train_label="TFC_cond", session_str="paradigmA_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                # Spatial error heatmaps (PF decoder, Paradigm A)
                plot_decoder_spatial_error_heatmaps(
                    mt_A_results_PF, mouse_groups,
                    PLOTS_DIR=MT_A_PF_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmA_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm A)
                plot_mobility_seconds_boxplots(
                    mt_A_results_PF, mouse_groups,
                    PLOTS_DIR=MT_A_PF_PLOTS_DIR,
                    train_label="TFC_cond", decoder_tag="2D_PF",
                    session_str="paradigmA_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm A)
                plot_xy_time_per_mouse(
                    mt_A_results_PF,
                    PLOTS_DIR=MT_A_PF_PLOTS_DIR,
                    train_label="TFC_cond",
                    session_str="paradigmA_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )

                _stop_paradigm_log(_plog_orig, _plog_file, MT_A_PF_PLOTS_DIR)

                # ==============================================================
                # A-Behav (PF decoder): Error vs Behaviour
                # ==============================================================
                MT_ABEHAV_PF_DIR = os.path.join(
                    PLOTS_DIR,
                    _popcurve_dirname(
                        f"A_Behav_PF_error_vs_behavior_use_pf_num_{use_pf_num}"
                        f"_occup_{use_occupancy_fallback}_encode_{encoder_period}"
                    ))
                os.makedirs(MT_ABEHAV_PF_DIR, exist_ok=True)

                _abehav_pf_common = dict(
                    mouse_groups=mouse_groups,
                    train_sessions=tfc_mice_with_fm,
                    test_targets=targets_A_pf,
                    PLOTS_DIR=MT_ABEHAV_PF_DIR,
                    train_label="TFC_cond",
                    min_speed=2.0,
                    first_n_sec=180.0,
                    n_x_bins=N_SPATIAL_BINS, n_y_bins=N_SPATIAL_BINS,
                    encoder_period=encoder_period,
                    auto_close=True,
                )

                for _pf_pm in ["separate", "pooled", "cross_session"]:
                    try:
                        run_error_vs_behavior_analysis(
                            mt_A_results_PF, pooling_mode=_pf_pm,
                            **_abehav_pf_common)
                    except Exception as e:
                        print(f"[A-Behav-PF-{_pf_pm}] Failed: {e}")
                        import traceback; traceback.print_exc()


    # ==============================================================================
    # ====  PARADIGM B: Train Test_B -> Decode TFC_cond + Test_B_1wk + Test_A + Test_A_1wk  =====
    # ==============================================================================

    # ===== end verbatim body =====
    return {"mt_A_results_2D": mt_A_results_2D}


# ---------------------------------------------------------------------------
# Section: paradigm_B  (caban/main.py L4017-4171)
# ---------------------------------------------------------------------------
def run_paradigm_B(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_B. Originally caban/main.py L4017-4171."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_B requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_B requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _paradigm_ABC_mapping = _make_paradigm_ABC_mapping(ds)
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_B_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"B1_TestB_2D_multi_target_encode_{encoder_period}"))
        os.makedirs(MT_B_PLOTS_DIR, exist_ok=True)

        testb_mice_for_mt = {m: Test_B[m] for m in Test_B
                             if m in TFC_cond or m in Test_A or m in Test_B_1wk or m in Test_A_1wk}
        targets_B = {}
        if TFC_cond:
            targets_B["TFC_cond"] = {m: TFC_cond[m] for m in TFC_cond if m in testb_mice_for_mt}
        if Test_B_1wk:
            targets_B["Test_B_1wk"] = {m: Test_B_1wk[m] for m in Test_B_1wk if m in testb_mice_for_mt}
        if Test_A:
            targets_B["Test_A"] = {m: Test_A[m] for m in Test_A if m in testb_mice_for_mt}
        if Test_A_1wk:
            targets_B["Test_A_1wk"] = {m: Test_A_1wk[m] for m in Test_A_1wk if m in testb_mice_for_mt}

        if targets_B:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_B_PLOTS_DIR, "Paradigm B", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm B (2D): Train Test_B -> {list(targets_B.keys())}")
            print(f"  {len(testb_mice_for_mt)} mice")
            print(f"{'='*60}\n")

            mt_B_results_2D = run_2D_decoder_all_mice(
                testb_mice_for_mt, targets_B,
                train_label="Test_B",
                mapping=_paradigm_ABC_mapping,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_B_PLOTS_DIR, session_str="paradigmB",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_B_results_2D, mouse_groups,
                    PLOTS_DIR=MT_B_PLOTS_DIR,
                    train_label="Test_B", session_str="paradigmB",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            # Spatial error heatmaps (2D raw-S, Paradigm B)
            plot_decoder_spatial_error_heatmaps(
                mt_B_results_2D, mouse_groups,
                PLOTS_DIR=MT_B_PLOTS_DIR,
                train_label="Test_B", session_str="paradigmB",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm B)
            plot_mobility_seconds_boxplots(
                mt_B_results_2D, mouse_groups,
                PLOTS_DIR=MT_B_PLOTS_DIR,
                train_label="Test_B", decoder_tag="2D",
                session_str="paradigmB", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm B)
            plot_xy_time_per_mouse(
                mt_B_results_2D,
                PLOTS_DIR=MT_B_PLOTS_DIR,
                train_label="Test_B",
                session_str="paradigmB",
                decoder_tag="2D",
                auto_close=True,
            )

            _stop_paradigm_log(_plog_orig, _plog_file, MT_B_PLOTS_DIR)

        # --- 2D PF-based decoder ---
        MT_B_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"B1_TestB_2D_PF_multi_target_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_B_PF_PLOTS_DIR, exist_ok=True)

        testb_mice_with_fm = {m: Test_B[m] for m in testb_mice_for_mt
                              if hasattr(Test_B[m], 'fm') and Test_B[m].fm is not None}

        if testb_mice_with_fm and targets_B:
            targets_B_pf = {k: {m: v[m] for m in v if m in testb_mice_with_fm}
                            for k, v in targets_B.items()}
            targets_B_pf = {k: v for k, v in targets_B_pf.items() if v}

            if targets_B_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_B_PF_PLOTS_DIR, "Paradigm B", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm B (PF): Train Test_B PF -> {list(targets_B_pf.keys())}")
                print(f"  {len(testb_mice_with_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_B_results_PF = run_2D_PF_decoder_all_mice(
                    testb_mice_with_fm, targets_B_pf,
                    train_label="Test_B",
                    mapping=_paradigm_ABC_mapping,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_B_PF_PLOTS_DIR, session_str="paradigmB",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_B_results_PF, mouse_groups,
                        PLOTS_DIR=MT_B_PF_PLOTS_DIR,
                        train_label="Test_B", session_str="paradigmB_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                # Spatial error heatmaps (PF decoder, Paradigm B)
                plot_decoder_spatial_error_heatmaps(
                    mt_B_results_PF, mouse_groups,
                    PLOTS_DIR=MT_B_PF_PLOTS_DIR,
                    train_label="Test_B", session_str="paradigmB_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm B)
                plot_mobility_seconds_boxplots(
                    mt_B_results_PF, mouse_groups,
                    PLOTS_DIR=MT_B_PF_PLOTS_DIR,
                    train_label="Test_B", decoder_tag="2D_PF",
                    session_str="paradigmB_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm B)
                plot_xy_time_per_mouse(
                    mt_B_results_PF,
                    PLOTS_DIR=MT_B_PF_PLOTS_DIR,
                    train_label="Test_B",
                    session_str="paradigmB_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )

                _stop_paradigm_log(_plog_orig, _plog_file, MT_B_PF_PLOTS_DIR)


    # ==============================================================================
    # ====  PARADIGM C: Train Test_A -> Decode TFC_cond + Test_B + Test_B_1wk + Test_A_1wk  =====
    # ==============================================================================

    # ===== end verbatim body =====
    return {"mt_B_results_2D": mt_B_results_2D}


# ---------------------------------------------------------------------------
# Section: paradigm_C  (caban/main.py L4172-4327)
# ---------------------------------------------------------------------------
def run_paradigm_C(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_C. Originally caban/main.py L4172-4327."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_C requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_C requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _paradigm_ABC_mapping = _make_paradigm_ABC_mapping(ds)
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_C_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"C1_TestA_2D_multi_target_encode_{encoder_period}"))
        os.makedirs(MT_C_PLOTS_DIR, exist_ok=True)

        testa_mice_for_mt = {m: Test_A[m] for m in Test_A
                             if m in TFC_cond or m in Test_B or m in Test_B_1wk or m in Test_A_1wk}
        targets_C = {}
        if TFC_cond:
            targets_C["TFC_cond"] = {m: TFC_cond[m] for m in TFC_cond if m in testa_mice_for_mt}
        if Test_B:
            targets_C["Test_B"] = {m: Test_B[m] for m in Test_B if m in testa_mice_for_mt}
        if Test_B_1wk:
            targets_C["Test_B_1wk"] = {m: Test_B_1wk[m] for m in Test_B_1wk if m in testa_mice_for_mt}
        if Test_A_1wk:
            targets_C["Test_A_1wk"] = {m: Test_A_1wk[m] for m in Test_A_1wk if m in testa_mice_for_mt}

        if targets_C:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_C_PLOTS_DIR, "Paradigm C", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm C (2D): Train Test_A -> {list(targets_C.keys())}")
            print(f"  {len(testa_mice_for_mt)} mice")
            print(f"{'='*60}\n")

            mt_C_results_2D = run_2D_decoder_all_mice(
                testa_mice_for_mt, targets_C,
                train_label="Test_A",
                mapping=_paradigm_ABC_mapping,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_C_PLOTS_DIR, session_str="paradigmC",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_C_results_2D, mouse_groups,
                    PLOTS_DIR=MT_C_PLOTS_DIR,
                    train_label="Test_A", session_str="paradigmC",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            # Spatial error heatmaps (2D raw-S, Paradigm C)
            plot_decoder_spatial_error_heatmaps(
                mt_C_results_2D, mouse_groups,
                PLOTS_DIR=MT_C_PLOTS_DIR,
                train_label="Test_A", session_str="paradigmC",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm C)
            plot_mobility_seconds_boxplots(
                mt_C_results_2D, mouse_groups,
                PLOTS_DIR=MT_C_PLOTS_DIR,
                train_label="Test_A", decoder_tag="2D",
                session_str="paradigmC", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm C)
            plot_xy_time_per_mouse(
                mt_C_results_2D,
                PLOTS_DIR=MT_C_PLOTS_DIR,
                train_label="Test_A",
                session_str="paradigmC",
                decoder_tag="2D",
                auto_close=True,
            )

            _stop_paradigm_log(_plog_orig, _plog_file, MT_C_PLOTS_DIR)

        # --- 2D PF-based decoder ---
        MT_C_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"C1_TestA_2D_PF_multi_target_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_C_PF_PLOTS_DIR, exist_ok=True)

        testa_mice_with_fm = {m: Test_A[m] for m in testa_mice_for_mt
                              if hasattr(Test_A[m], 'fm') and Test_A[m].fm is not None}

        if testa_mice_with_fm and targets_C:
            targets_C_pf = {k: {m: v[m] for m in v if m in testa_mice_with_fm}
                            for k, v in targets_C.items()}
            targets_C_pf = {k: v for k, v in targets_C_pf.items() if v}

            if targets_C_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_C_PF_PLOTS_DIR, "Paradigm C", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm C (PF): Train Test_A PF -> {list(targets_C_pf.keys())}")
                print(f"  {len(testa_mice_with_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_C_results_PF = run_2D_PF_decoder_all_mice(
                    testa_mice_with_fm, targets_C_pf,
                    train_label="Test_A",
                    mapping=_paradigm_ABC_mapping,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_C_PF_PLOTS_DIR, session_str="paradigmC",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_C_results_PF, mouse_groups,
                        PLOTS_DIR=MT_C_PF_PLOTS_DIR,
                        train_label="Test_A", session_str="paradigmC_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                # Spatial error heatmaps (PF decoder, Paradigm C)
                plot_decoder_spatial_error_heatmaps(
                    mt_C_results_PF, mouse_groups,
                    PLOTS_DIR=MT_C_PF_PLOTS_DIR,
                    train_label="Test_A", session_str="paradigmC_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm C)
                plot_mobility_seconds_boxplots(
                    mt_C_results_PF, mouse_groups,
                    PLOTS_DIR=MT_C_PF_PLOTS_DIR,
                    train_label="Test_A", decoder_tag="2D_PF",
                    session_str="paradigmC_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm C)
                plot_xy_time_per_mouse(
                    mt_C_results_PF,
                    PLOTS_DIR=MT_C_PF_PLOTS_DIR,
                    train_label="Test_A",
                    session_str="paradigmC_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )

                _stop_paradigm_log(_plog_orig, _plog_file, MT_C_PF_PLOTS_DIR)


    # ==============================================================================
    # ====  PARADIGM D1: Train TFC_cond -> Decode Test_A + Test_A_1wk             ==
    # ====  (3-session crossreg: TFC_cond+Test_A+Test_A_1wk -> more cells)        ==
    # ==============================================================================

    # ===== end verbatim body =====
    return {"mt_C_results_2D": mt_C_results_2D}


# ---------------------------------------------------------------------------
# Section: paradigm_D1  (caban/main.py L4328-4477)
# ---------------------------------------------------------------------------
def run_paradigm_D1(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_D1. Originally caban/main.py L4328-4477."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_D1 requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_D1 requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_D1_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"D1_TFC_2D_multi_target_D1_encode_{encoder_period}"))
        os.makedirs(MT_D1_PLOTS_DIR, exist_ok=True)

        tfc_mice_for_d1 = {m: TFC_cond[m] for m in TFC_cond
                           if m in Test_A or m in Test_A_1wk}
        targets_D1 = {}
        if Test_A:
            targets_D1["Test_A"] = {m: Test_A[m] for m in Test_A if m in tfc_mice_for_d1}
        if Test_A_1wk:
            targets_D1["Test_A_1wk"] = {m: Test_A_1wk[m] for m in Test_A_1wk if m in tfc_mice_for_d1}

        if targets_D1:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_D1_PLOTS_DIR, "Paradigm D1", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm D1 (2D): Train TFC_cond -> {list(targets_D1.keys())}")
            print(f"  {len(tfc_mice_for_d1)} mice")
            print(f"{'='*60}\n")

            mt_D1_results_2D = run_2D_decoder_all_mice(
                tfc_mice_for_d1, targets_D1,
                train_label="TFC_cond",
                mapping=mapping_TFC_cond_Test_A_Test_A_1wk,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_D1_PLOTS_DIR, session_str="paradigmD1",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_D1_results_2D, mouse_groups,
                    PLOTS_DIR=MT_D1_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmD1",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            plot_decoder_spatial_error_heatmaps(
                mt_D1_results_2D, mouse_groups,
                PLOTS_DIR=MT_D1_PLOTS_DIR,
                train_label="TFC_cond", session_str="paradigmD1",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm D1)
            plot_mobility_seconds_boxplots(
                mt_D1_results_2D, mouse_groups,
                PLOTS_DIR=MT_D1_PLOTS_DIR,
                train_label="TFC_cond", decoder_tag="2D",
                session_str="paradigmD1", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm D1)
            plot_xy_time_per_mouse(
                mt_D1_results_2D,
                PLOTS_DIR=MT_D1_PLOTS_DIR,
                train_label="TFC_cond",
                session_str="paradigmD1",
                decoder_tag="2D",
                auto_close=True,
            )

            _stop_paradigm_log(_plog_orig, _plog_file, MT_D1_PLOTS_DIR)

        # --- 2D PF-based decoder ---
        MT_D1_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"D1_TFC_2D_PF_multi_target_D1_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_D1_PF_PLOTS_DIR, exist_ok=True)

        tfc_mice_d1_fm = {m: TFC_cond[m] for m in tfc_mice_for_d1
                          if hasattr(TFC_cond[m], 'fm') and TFC_cond[m].fm is not None}

        if tfc_mice_d1_fm and targets_D1:
            targets_D1_pf = {k: {m: v[m] for m in v if m in tfc_mice_d1_fm}
                             for k, v in targets_D1.items()}
            targets_D1_pf = {k: v for k, v in targets_D1_pf.items() if v}

            if targets_D1_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_D1_PF_PLOTS_DIR, "Paradigm D1", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm D1 (PF): Train TFC_cond PF -> {list(targets_D1_pf.keys())}")
                print(f"  {len(tfc_mice_d1_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_D1_results_PF = run_2D_PF_decoder_all_mice(
                    tfc_mice_d1_fm, targets_D1_pf,
                    train_label="TFC_cond",
                    mapping=mapping_TFC_cond_Test_A_Test_A_1wk,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_D1_PF_PLOTS_DIR, session_str="paradigmD1",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_D1_results_PF, mouse_groups,
                        PLOTS_DIR=MT_D1_PF_PLOTS_DIR,
                        train_label="TFC_cond", session_str="paradigmD1_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                plot_decoder_spatial_error_heatmaps(
                    mt_D1_results_PF, mouse_groups,
                    PLOTS_DIR=MT_D1_PF_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmD1_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm D1)
                plot_mobility_seconds_boxplots(
                    mt_D1_results_PF, mouse_groups,
                    PLOTS_DIR=MT_D1_PF_PLOTS_DIR,
                    train_label="TFC_cond", decoder_tag="2D_PF",
                    session_str="paradigmD1_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm D1)
                plot_xy_time_per_mouse(
                    mt_D1_results_PF,
                    PLOTS_DIR=MT_D1_PF_PLOTS_DIR,
                    train_label="TFC_cond",
                    session_str="paradigmD1_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )

                _stop_paradigm_log(_plog_orig, _plog_file, MT_D1_PF_PLOTS_DIR)


    # ==============================================================================
    # ====  PARADIGM D2: Train Test_A -> Decode Test_A_1wk + TFC_cond             ==
    # ====  (3-session crossreg: TFC_cond+Test_A+Test_A_1wk -> more cells)        ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: paradigm_D2  (caban/main.py L4478-4627)
# ---------------------------------------------------------------------------
def run_paradigm_D2(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_D2. Originally caban/main.py L4478-4627."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_D2 requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_D2 requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_D2_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"D2_TestA_2D_multi_target_D2_encode_{encoder_period}"))
        os.makedirs(MT_D2_PLOTS_DIR, exist_ok=True)

        testa_mice_for_d2 = {m: Test_A[m] for m in Test_A
                             if m in TFC_cond or m in Test_A_1wk}
        targets_D2 = {}
        if Test_A_1wk:
            targets_D2["Test_A_1wk"] = {m: Test_A_1wk[m] for m in Test_A_1wk if m in testa_mice_for_d2}
        if TFC_cond:
            targets_D2["TFC_cond"] = {m: TFC_cond[m] for m in TFC_cond if m in testa_mice_for_d2}

        if targets_D2:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_D2_PLOTS_DIR, "Paradigm D2", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm D2 (2D): Train Test_A -> {list(targets_D2.keys())}")
            print(f"  {len(testa_mice_for_d2)} mice")
            print(f"{'='*60}\n")

            mt_D2_results_2D = run_2D_decoder_all_mice(
                testa_mice_for_d2, targets_D2,
                train_label="Test_A",
                mapping=mapping_TFC_cond_Test_A_Test_A_1wk,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_D2_PLOTS_DIR, session_str="paradigmD2",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_D2_results_2D, mouse_groups,
                    PLOTS_DIR=MT_D2_PLOTS_DIR,
                    train_label="Test_A", session_str="paradigmD2",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            plot_decoder_spatial_error_heatmaps(
                mt_D2_results_2D, mouse_groups,
                PLOTS_DIR=MT_D2_PLOTS_DIR,
                train_label="Test_A", session_str="paradigmD2",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm D2)
            plot_mobility_seconds_boxplots(
                mt_D2_results_2D, mouse_groups,
                PLOTS_DIR=MT_D2_PLOTS_DIR,
                train_label="Test_A", decoder_tag="2D",
                session_str="paradigmD2", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm D2)
            plot_xy_time_per_mouse(
                mt_D2_results_2D,
                PLOTS_DIR=MT_D2_PLOTS_DIR,
                train_label="Test_A",
                session_str="paradigmD2",
                decoder_tag="2D",
                auto_close=True,
            )

            _stop_paradigm_log(_plog_orig, _plog_file, MT_D2_PLOTS_DIR)

        # --- 2D PF-based decoder ---
        MT_D2_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"D2_TestA_2D_PF_multi_target_D2_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_D2_PF_PLOTS_DIR, exist_ok=True)

        testa_mice_d2_fm = {m: Test_A[m] for m in testa_mice_for_d2
                            if hasattr(Test_A[m], 'fm') and Test_A[m].fm is not None}

        if testa_mice_d2_fm and targets_D2:
            targets_D2_pf = {k: {m: v[m] for m in v if m in testa_mice_d2_fm}
                             for k, v in targets_D2.items()}
            targets_D2_pf = {k: v for k, v in targets_D2_pf.items() if v}

            if targets_D2_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_D2_PF_PLOTS_DIR, "Paradigm D2", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm D2 (PF): Train Test_A PF -> {list(targets_D2_pf.keys())}")
                print(f"  {len(testa_mice_d2_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_D2_results_PF = run_2D_PF_decoder_all_mice(
                    testa_mice_d2_fm, targets_D2_pf,
                    train_label="Test_A",
                    mapping=mapping_TFC_cond_Test_A_Test_A_1wk,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_D2_PF_PLOTS_DIR, session_str="paradigmD2",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_D2_results_PF, mouse_groups,
                        PLOTS_DIR=MT_D2_PF_PLOTS_DIR,
                        train_label="Test_A", session_str="paradigmD2_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                plot_decoder_spatial_error_heatmaps(
                    mt_D2_results_PF, mouse_groups,
                    PLOTS_DIR=MT_D2_PF_PLOTS_DIR,
                    train_label="Test_A", session_str="paradigmD2_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm D2)
                plot_mobility_seconds_boxplots(
                    mt_D2_results_PF, mouse_groups,
                    PLOTS_DIR=MT_D2_PF_PLOTS_DIR,
                    train_label="Test_A", decoder_tag="2D_PF",
                    session_str="paradigmD2_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm D2)
                plot_xy_time_per_mouse(
                    mt_D2_results_PF,
                    PLOTS_DIR=MT_D2_PF_PLOTS_DIR,
                    train_label="Test_A",
                    session_str="paradigmD2_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )

                _stop_paradigm_log(_plog_orig, _plog_file, MT_D2_PF_PLOTS_DIR)


    # ==============================================================================
    # ====  PARADIGM E1: Train TFC_cond -> Decode Test_B + Test_B_1wk             ==
    # ====  (3-session crossreg: TFC_cond+Test_B+Test_B_1wk -> more cells)        ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: paradigm_E1  (caban/main.py L4628-4775)
# ---------------------------------------------------------------------------
def run_paradigm_E1(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_E1. Originally caban/main.py L4628-4775."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_E1 requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_E1 requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_E1_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"E1_TFC_2D_multi_target_E1_encode_{encoder_period}"))
        os.makedirs(MT_E1_PLOTS_DIR, exist_ok=True)

        tfc_mice_for_e1 = {m: TFC_cond[m] for m in TFC_cond
                           if m in Test_B or m in Test_B_1wk}
        targets_E1 = {}
        if Test_B:
            targets_E1["Test_B"] = {m: Test_B[m] for m in Test_B if m in tfc_mice_for_e1}
        if Test_B_1wk:
            targets_E1["Test_B_1wk"] = {m: Test_B_1wk[m] for m in Test_B_1wk if m in tfc_mice_for_e1}

        if targets_E1:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_E1_PLOTS_DIR, "Paradigm E1", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm E1 (2D): Train TFC_cond -> {list(targets_E1.keys())}")
            print(f"  {len(tfc_mice_for_e1)} mice")
            print(f"{'='*60}\n")

            mt_E1_results_2D = run_2D_decoder_all_mice(
                tfc_mice_for_e1, targets_E1,
                train_label="TFC_cond",
                mapping=mapping_TFC_cond_Test_B_Test_B_1wk,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_E1_PLOTS_DIR, session_str="paradigmE1",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_E1_results_2D, mouse_groups,
                    PLOTS_DIR=MT_E1_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmE1",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            plot_decoder_spatial_error_heatmaps(
                mt_E1_results_2D, mouse_groups,
                PLOTS_DIR=MT_E1_PLOTS_DIR,
                train_label="TFC_cond", session_str="paradigmE1",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm E1)
            plot_mobility_seconds_boxplots(
                mt_E1_results_2D, mouse_groups,
                PLOTS_DIR=MT_E1_PLOTS_DIR,
                train_label="TFC_cond", decoder_tag="2D",
                session_str="paradigmE1", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm E1)
            plot_xy_time_per_mouse(
                mt_E1_results_2D,
                PLOTS_DIR=MT_E1_PLOTS_DIR,
                train_label="TFC_cond",
                session_str="paradigmE1",
                decoder_tag="2D",
                auto_close=True,
            )
            _stop_paradigm_log(_plog_orig, _plog_file, MT_E1_PLOTS_DIR)

        # --- 2D PF-based decoder ---
        MT_E1_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"E1_TFC_2D_PF_multi_target_E1_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_E1_PF_PLOTS_DIR, exist_ok=True)

        tfc_mice_e1_fm = {m: TFC_cond[m] for m in tfc_mice_for_e1
                          if hasattr(TFC_cond[m], 'fm') and TFC_cond[m].fm is not None}

        if tfc_mice_e1_fm and targets_E1:
            targets_E1_pf = {k: {m: v[m] for m in v if m in tfc_mice_e1_fm}
                             for k, v in targets_E1.items()}
            targets_E1_pf = {k: v for k, v in targets_E1_pf.items() if v}

            if targets_E1_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_E1_PF_PLOTS_DIR, "Paradigm E1", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm E1 (PF): Train TFC_cond PF -> {list(targets_E1_pf.keys())}")
                print(f"  {len(tfc_mice_e1_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_E1_results_PF = run_2D_PF_decoder_all_mice(
                    tfc_mice_e1_fm, targets_E1_pf,
                    train_label="TFC_cond",
                    mapping=mapping_TFC_cond_Test_B_Test_B_1wk,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_E1_PF_PLOTS_DIR, session_str="paradigmE1",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_E1_results_PF, mouse_groups,
                        PLOTS_DIR=MT_E1_PF_PLOTS_DIR,
                        train_label="TFC_cond", session_str="paradigmE1_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                plot_decoder_spatial_error_heatmaps(
                    mt_E1_results_PF, mouse_groups,
                    PLOTS_DIR=MT_E1_PF_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmE1_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm E1)
                plot_mobility_seconds_boxplots(
                    mt_E1_results_PF, mouse_groups,
                    PLOTS_DIR=MT_E1_PF_PLOTS_DIR,
                    train_label="TFC_cond", decoder_tag="2D_PF",
                    session_str="paradigmE1_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm E1)
                plot_xy_time_per_mouse(
                    mt_E1_results_PF,
                    PLOTS_DIR=MT_E1_PF_PLOTS_DIR,
                    train_label="TFC_cond",
                    session_str="paradigmE1_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )
                _stop_paradigm_log(_plog_orig, _plog_file, MT_E1_PF_PLOTS_DIR)


    # ==============================================================================
    # ====  PARADIGM E2: Train Test_B -> Decode Test_B_1wk + TFC_cond             ==
    # ====  (3-session crossreg: TFC_cond+Test_B+Test_B_1wk -> more cells)        ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: paradigm_E2  (caban/main.py L4776-4924)
# ---------------------------------------------------------------------------
def run_paradigm_E2(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_E2. Originally caban/main.py L4776-4924."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_E2 requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_E2 requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # --- 2D raw-S decoder ---
        MT_E2_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"E2_TestB_2D_multi_target_E2_encode_{encoder_period}"))
        os.makedirs(MT_E2_PLOTS_DIR, exist_ok=True)

        testb_mice_for_e2 = {m: Test_B[m] for m in Test_B
                             if m in TFC_cond or m in Test_B_1wk}
        targets_E2 = {}
        if Test_B_1wk:
            targets_E2["Test_B_1wk"] = {m: Test_B_1wk[m] for m in Test_B_1wk if m in testb_mice_for_e2}
        if TFC_cond:
            targets_E2["TFC_cond"] = {m: TFC_cond[m] for m in TFC_cond if m in testb_mice_for_e2}

        if targets_E2:
            _plog_orig, _plog_file = _start_paradigm_log(
                MT_E2_PLOTS_DIR, "Paradigm E2", "raw-S 2D", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm E2 (2D): Train Test_B -> {list(targets_E2.keys())}")
            print(f"  {len(testb_mice_for_e2)} mice")
            print(f"{'='*60}\n")

            mt_E2_results_2D = run_2D_decoder_all_mice(
                testb_mice_for_e2, targets_E2,
                train_label="Test_B",
                mapping=mapping_TFC_cond_Test_B_Test_B_1wk,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_E2_PLOTS_DIR, session_str="paradigmE2",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_E2_results_2D, mouse_groups,
                    PLOTS_DIR=MT_E2_PLOTS_DIR,
                    train_label="Test_B", session_str="paradigmE2",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            plot_decoder_spatial_error_heatmaps(
                mt_E2_results_2D, mouse_groups,
                PLOTS_DIR=MT_E2_PLOTS_DIR,
                train_label="Test_B", session_str="paradigmE2",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm E2)
            plot_mobility_seconds_boxplots(
                mt_E2_results_2D, mouse_groups,
                PLOTS_DIR=MT_E2_PLOTS_DIR,
                train_label="Test_B", decoder_tag="2D",
                session_str="paradigmE2", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm E2)
            plot_xy_time_per_mouse(
                mt_E2_results_2D,
                PLOTS_DIR=MT_E2_PLOTS_DIR,
                train_label="Test_B",
                session_str="paradigmE2",
                decoder_tag="2D",
                auto_close=True,
            )
            _stop_paradigm_log(_plog_orig, _plog_file, MT_E2_PLOTS_DIR)

        # --- 2D PF-based decoder ---
        MT_E2_PF_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"E2_TestB_2D_PF_multi_target_E2_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_E2_PF_PLOTS_DIR, exist_ok=True)

        testb_mice_e2_fm = {m: Test_B[m] for m in testb_mice_for_e2
                            if hasattr(Test_B[m], 'fm') and Test_B[m].fm is not None}

        if testb_mice_e2_fm and targets_E2:
            targets_E2_pf = {k: {m: v[m] for m in v if m in testb_mice_e2_fm}
                             for k, v in targets_E2.items()}
            targets_E2_pf = {k: v for k, v in targets_E2_pf.items() if v}

            if targets_E2_pf:
                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_E2_PF_PLOTS_DIR, "Paradigm E2", "PF 2D", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm E2 (PF): Train Test_B PF -> {list(targets_E2_pf.keys())}")
                print(f"  {len(testb_mice_e2_fm)} mice with .fm")
                print(f"{'='*60}\n")

                mt_E2_results_PF = run_2D_PF_decoder_all_mice(
                    testb_mice_e2_fm, targets_E2_pf,
                    train_label="Test_B",
                    mapping=mapping_TFC_cond_Test_B_Test_B_1wk,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_E2_PF_PLOTS_DIR, session_str="paradigmE2",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_E2_results_PF, mouse_groups,
                        PLOTS_DIR=MT_E2_PF_PLOTS_DIR,
                        train_label="Test_B", session_str="paradigmE2_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                plot_decoder_spatial_error_heatmaps(
                    mt_E2_results_PF, mouse_groups,
                    PLOTS_DIR=MT_E2_PF_PLOTS_DIR,
                    train_label="Test_B", session_str="paradigmE2_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm E2)
                plot_mobility_seconds_boxplots(
                    mt_E2_results_PF, mouse_groups,
                    PLOTS_DIR=MT_E2_PF_PLOTS_DIR,
                    train_label="Test_B", decoder_tag="2D_PF",
                    session_str="paradigmE2_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm E2)
                plot_xy_time_per_mouse(
                    mt_E2_results_PF,
                    PLOTS_DIR=MT_E2_PF_PLOTS_DIR,
                    train_label="Test_B",
                    session_str="paradigmE2_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )
                _stop_paradigm_log(_plog_orig, _plog_file, MT_E2_PF_PLOTS_DIR)


    # ==============================================================================
    # ====  PARADIGM F: Train TFC_cond -> Decode pooled (A+A_1wk) and (B+B_1wk)  ==
    # ====  F1 = occupancy-based 2D decoder                                       ==
    # ====  F2 = place-field GMM-based 2D decoder                                 ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: paradigm_F  (caban/main.py L4925-5127)
# ---------------------------------------------------------------------------
def run_paradigm_F(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: paradigm_F. Originally caban/main.py L4925-5127."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    decoder_type = cfg.decoder_type
    encoder_period = cfg.encoder_period
    optimization_param_set = cfg.optimization_param_set
    optimization_pf_param_set = cfg.optimization_pf_param_set
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    ridge_alpha = cfg.ridge_alpha
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    use_z_score = cfg.use_z_score
    if raw_params is None:
        raise RuntimeError(f"paradigm_F requires raw_params (build via caban.pipeline.build_*_paramset)")
    if pf_params is None:
        raise RuntimeError(f"paradigm_F requires pf_params (build via caban.pipeline.build_*_paramset)")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n
    _start_paradigm_log_impl = globals()['_start_paradigm_log']
    def _start_paradigm_log(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name):  # noqa: F811
        return _start_paradigm_log_impl(plots_dir, paradigm_label, decoder_tag, paramset, param_set_name, cfg=cfg)

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        # ---- F1: Occupancy-based 2D decoder, pooled targets --------------------
        MT_F1_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"F1_TFC_2D_pooled_F1_encode_{encoder_period}"))
        os.makedirs(MT_F1_PLOTS_DIR, exist_ok=True)

        # Mice that have TFC_cond and at least one pair to pool
        tfc_mice_for_f = {m: TFC_cond[m] for m in TFC_cond}

        # Build pooled target dicts: each value is {mouse: [sess_24h, sess_1wk]}
        pooled_targets_F = {}

        # A + A_1wk
        mice_A_pooled = {m: [Test_A[m], Test_A_1wk[m]]
                         for m in tfc_mice_for_f
                         if m in Test_A and m in Test_A_1wk}
        if mice_A_pooled:
            pooled_targets_F["A_plus_A1wk"] = mice_A_pooled

        # B + B_1wk
        mice_B_pooled = {m: [Test_B[m], Test_B_1wk[m]]
                         for m in tfc_mice_for_f
                         if m in Test_B and m in Test_B_1wk}
        if mice_B_pooled:
            pooled_targets_F["B_plus_B1wk"] = mice_B_pooled

        if pooled_targets_F:
            # Mice with at least one pooled target
            tfc_mice_f1 = {m: TFC_cond[m] for m in tfc_mice_for_f
                           if any(m in pt for pt in pooled_targets_F.values())}

            _plog_orig, _plog_file = _start_paradigm_log(
                MT_F1_PLOTS_DIR, "Paradigm F1", "raw-S 2D pooled", raw_params, optimization_param_set)
            print(f"\n{'='*60}")
            print(f"  Paradigm F1 (2D): Train TFC_cond -> pooled targets "
                  f"{list(pooled_targets_F.keys())}")
            print(f"  {len(tfc_mice_f1)} mice")
            print(f"{'='*60}\n")

            # Use the correct 3-session mapping for each pooled target
            pooled_mapping_F = {}
            if "A_plus_A1wk" in pooled_targets_F:
                pooled_mapping_F["A_plus_A1wk"] = mapping_TFC_cond_Test_A_Test_A_1wk
            if "B_plus_B1wk" in pooled_targets_F:
                pooled_mapping_F["B_plus_B1wk"] = mapping_TFC_cond_Test_B_Test_B_1wk

            mt_F1_results = run_2D_pooled_decoder_all_mice(
                tfc_mice_f1, pooled_targets_F,
                train_label="TFC_cond",
                mapping=pooled_mapping_F,
                n_x_bins=raw_params.n_spatial_bins, n_y_bins=raw_params.n_spatial_bins,
                time_bin_frames=raw_params.time_bin_frames,
                use_posterior_mean=raw_params.use_posterior_mean,
                decoder_type=decoder_type, ridge_alpha=ridge_alpha,
                use_speed=True, min_speed=2.0, first_n_sec=180.0,
                use_continuity_constraint=raw_params.use_continuity_constraint,
                continuity_sigma_k=raw_params.continuity_sigma_k,
                continuity_speed_ref=raw_params.continuity_speed_ref,
                continuity_exp=raw_params.continuity_exp,
                continuity_sigma_min=raw_params.continuity_sigma_min,
                continuity_sigma_max=raw_params.continuity_sigma_max,
                continuity_sigma_default=raw_params.continuity_sigma_default,
                encoder_period=encoder_period,
                PLOTS_DIR=MT_F1_PLOTS_DIR, session_str="paradigmF1",
                mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                use_z_score=raw_params.use_z_score,
            )

            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_decoder_stats_and_plots(
                    mt_F1_results, mouse_groups,
                    PLOTS_DIR=MT_F1_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmF1",
                    decoder_tag="2D",
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                )

            plot_decoder_spatial_error_heatmaps(
                mt_F1_results, mouse_groups,
                PLOTS_DIR=MT_F1_PLOTS_DIR,
                train_label="TFC_cond", session_str="paradigmF1",
                decoder_tag="2D", kmin=5.0, auto_close=True,
            )

            # Mobility seconds boxplots (2D, Paradigm F1)
            plot_mobility_seconds_boxplots(
                mt_F1_results, mouse_groups,
                PLOTS_DIR=MT_F1_PLOTS_DIR,
                train_label="TFC_cond", decoder_tag="2D",
                session_str="paradigmF1", auto_close=True,
            )

            # X/Y position vs time per mouse (2D, Paradigm F1)
            plot_xy_time_per_mouse(
                mt_F1_results,
                PLOTS_DIR=MT_F1_PLOTS_DIR,
                train_label="TFC_cond",
                session_str="paradigmF1",
                decoder_tag="2D",
                auto_close=True,
            )
            _stop_paradigm_log(_plog_orig, _plog_file, MT_F1_PLOTS_DIR)

        # ---- F2: PF-based 2D decoder, pooled targets --------------------------
        MT_F2_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"F1_TFC_2D_PF_pooled_F2_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(MT_F2_PLOTS_DIR, exist_ok=True)

        tfc_mice_f2_fm = {m: TFC_cond[m] for m in tfc_mice_for_f
                          if hasattr(TFC_cond[m], 'fm') and TFC_cond[m].fm is not None}

        if tfc_mice_f2_fm and pooled_targets_F:
            pooled_targets_F_pf = {}
            for tgt_name, tgt_dict in pooled_targets_F.items():
                sub = {m: v for m, v in tgt_dict.items() if m in tfc_mice_f2_fm}
                if sub:
                    pooled_targets_F_pf[tgt_name] = sub

            if pooled_targets_F_pf:
                tfc_mice_f2 = {m: TFC_cond[m] for m in tfc_mice_f2_fm
                               if any(m in pt for pt in pooled_targets_F_pf.values())}

                _plog_orig, _plog_file = _start_paradigm_log(
                    MT_F2_PLOTS_DIR, "Paradigm F2", "PF 2D pooled", pf_params, optimization_pf_param_set)
                print(f"\n{'='*60}")
                print(f"  Paradigm F2 (PF): Train TFC_cond PF -> pooled targets "
                      f"{list(pooled_targets_F_pf.keys())}")
                print(f"  {len(tfc_mice_f2)} mice with .fm")
                print(f"{'='*60}\n")

                mt_F2_results = run_2D_PF_pooled_decoder_all_mice(
                    tfc_mice_f2, pooled_targets_F_pf,
                    train_label="TFC_cond",
                    mapping=pooled_mapping_F,
                    time_bin_frames=pf_params.time_bin_frames,
                    use_posterior_mean=pf_params.use_posterior_mean,
                    decoder_type=decoder_type, ridge_alpha=ridge_alpha, use_speed=True, min_speed=2.0,
                    first_n_sec=180.0,
                    place_cells_only=pf_params.place_cells_only,
                    use_pf_num=pf_params.use_pf_num,
                    use_occupancy_fallback=pf_params.use_occupancy_fallback,
                    use_continuity_constraint=pf_params.use_continuity_constraint,
                    continuity_sigma_k=pf_params.continuity_sigma_k,
                    continuity_speed_ref=pf_params.continuity_speed_ref,
                    continuity_exp=pf_params.continuity_exp,
                    continuity_sigma_min=pf_params.continuity_sigma_min,
                    continuity_sigma_max=pf_params.continuity_sigma_max,
                    continuity_sigma_default=pf_params.continuity_sigma_default,
                    encoder_period=encoder_period,
                    PLOTS_DIR=MT_F2_PLOTS_DIR, session_str="paradigmF2",
                    mouse_groups=mouse_groups, plot_debug=True, auto_close=True,
                    use_z_score="none",  # PF decoder: no z-score (rate maps are precomputed from original-scale data)
                )

                for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                    run_decoder_stats_and_plots(
                        mt_F2_results, mouse_groups,
                        PLOTS_DIR=MT_F2_PLOTS_DIR,
                        train_label="TFC_cond", session_str="paradigmF2_PF",
                        decoder_tag="2D_PF",
                        use_scoring_method=use_scoring_method,
                        use_median=USE_MEDIAN,
                        pct_threshold=pct_threshold_2D,
                        stat_label_override=stat_label if stat_label == "pct" else None,
                        use_z_score=use_z_score, auto_close=True,
                    )

                plot_decoder_spatial_error_heatmaps(
                    mt_F2_results, mouse_groups,
                    PLOTS_DIR=MT_F2_PLOTS_DIR,
                    train_label="TFC_cond", session_str="paradigmF2_PF",
                    decoder_tag="2D_PF", kmin=5.0, auto_close=True,
                )

                # Mobility seconds boxplots (PF, Paradigm F2)
                plot_mobility_seconds_boxplots(
                    mt_F2_results, mouse_groups,
                    PLOTS_DIR=MT_F2_PLOTS_DIR,
                    train_label="TFC_cond", decoder_tag="2D_PF",
                    session_str="paradigmF2_PF", auto_close=True,
                )

                # X/Y position vs time per mouse (PF, Paradigm F2)
                plot_xy_time_per_mouse(
                    mt_F2_results,
                    PLOTS_DIR=MT_F2_PLOTS_DIR,
                    train_label="TFC_cond",
                    session_str="paradigmF2_PF",
                    decoder_tag="2D_PF",
                    auto_close=True,
                )
                _stop_paradigm_log(_plog_orig, _plog_file, MT_F2_PLOTS_DIR)


    # ==============================================================================
    # ====  TFC cross-session vs within-session MixedLM                          ==
    # ====  Compares (TFC->TestB) − Within-TestB and (TFC->TestA) − Within-TestA   ==
    # ====  across Ctl / Exc / Inh groups using paradigm A, B, C results         ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: mixedlm_cross_vs_within  (caban/main.py L5128-5155)
# ---------------------------------------------------------------------------
def run_mixedlm_cross_vs_within(ds, cfg, *, mt_A_results_2D=None, mt_B_results_2D=None, mt_C_results_2D=None):
    """Analysis section: mixedlm_cross_vs_within. Originally caban/main.py L5128-5155."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    encoder_period = cfg.encoder_period
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    run_fixed_effects_models_tfc_cross_vs_within = cfg.run_fixed_effects_models_tfc_cross_vs_within
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    if mt_A_results_2D is None:
        raise RuntimeError(f"mixedlm_cross_vs_within requires mt_A_results_2D from upstream paradigm section")
    if mt_B_results_2D is None:
        raise RuntimeError(f"mixedlm_cross_vs_within requires mt_B_results_2D from upstream paradigm section")
    if mt_C_results_2D is None:
        raise RuntimeError(f"mixedlm_cross_vs_within requires mt_C_results_2D from upstream paradigm section")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        TFC_MLM_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"TFC_cross_vs_within_mixedlm_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(TFC_MLM_PLOTS_DIR, exist_ok=True)

        try:
            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_tfc_cross_vs_within_analysis(
                    mt_A_results_2D, mt_B_results_2D, mt_C_results_2D,
                    mouse_groups=mouse_groups,
                    PLOTS_DIR=TFC_MLM_PLOTS_DIR,
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    run_fixed_effects_models=run_fixed_effects_models_tfc_cross_vs_within,
                    auto_close=True,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                )
        except NameError as e:
            print(f"[TFC-MixedLM] Skipping - paradigm A/B/C result not available: {e}")


    # ==============================================================================
    # ====  TFC cross-session vs within-TFC_cond baseline MixedLM               ==
    # ====  All contrasts referenced to Within TFC_cond                          ==
    # ====  across Ctl / Exc / Inh groups using paradigm A, B, C results        ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: mixedlm_vs_tfc_cond  (caban/main.py L5156-5181)
# ---------------------------------------------------------------------------
def run_mixedlm_vs_tfc_cond(ds, cfg, *, mt_A_results_2D=None, mt_B_results_2D=None, mt_C_results_2D=None):
    """Analysis section: mixedlm_vs_tfc_cond. Originally caban/main.py L5156-5181."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    encoder_period = cfg.encoder_period
    pct_threshold_2D = cfg.pct_threshold_2D
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    run_fixed_effects_models_tfc_cross_vs_within = cfg.run_fixed_effects_models_tfc_cross_vs_within
    use_occupancy_fallback = cfg.use_occupancy_fallback
    use_pf_num = cfg.use_pf_num
    use_scoring_method = cfg.use_scoring_method
    if mt_A_results_2D is None:
        raise RuntimeError(f"mixedlm_vs_tfc_cond requires mt_A_results_2D from upstream paradigm section")
    if mt_B_results_2D is None:
        raise RuntimeError(f"mixedlm_vs_tfc_cond requires mt_B_results_2D from upstream paradigm section")
    if mt_C_results_2D is None:
        raise RuntimeError(f"mixedlm_vs_tfc_cond requires mt_C_results_2D from upstream paradigm section")
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        TFC_MLM_COND_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname(f"TFC_cross_vs_withinCond_mixedlm_use_pf_num_{use_pf_num}_occup_{use_occupancy_fallback}_encode_{encoder_period}"))
        os.makedirs(TFC_MLM_COND_PLOTS_DIR, exist_ok=True)

        try:
            for stat_label, USE_MEDIAN in [("mean", False), ("median", True), ("pct", False)]:
                run_tfc_cross_vs_within_cond_baseline_analysis(
                    mt_A_results_2D, mt_B_results_2D, mt_C_results_2D,
                    mouse_groups=mouse_groups,
                    PLOTS_DIR=TFC_MLM_COND_PLOTS_DIR,
                    use_scoring_method=use_scoring_method,
                    use_median=USE_MEDIAN,
                    run_fixed_effects_models=run_fixed_effects_models_tfc_cross_vs_within,
                    auto_close=True,
                    pct_threshold=pct_threshold_2D,
                    stat_label_override=stat_label if stat_label == "pct" else None,
                )
        except NameError as e:
            print(f"[TFC-MixedLM-CondBase] Skipping - paradigm A/B/C result not available: {e}")


    # ==============================================================================
    # ====  2D Population Vector (PV) Correlation: TFC_cond pre-tone vs Tests    ==
    # ==============================================================================

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: pv_correlation_2d  (caban/main.py L5182-5562)
# ---------------------------------------------------------------------------
def run_pv_correlation_2d(ds, cfg, *, raw_params=None, pf_params=None):
    """Analysis section: pv_correlation_2d. Originally caban/main.py L5182-5562."""
    if not (cfg.plot_TFC_2D_decoding):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    plot_TFC_2D_decoding = cfg.plot_TFC_2D_decoding
    use_z_score = cfg.use_z_score
    # _popcurve_dirname closure capturing cfg
    def _popcurve_dirname(_n):  # noqa: F811
        if cfg.enable_population_curve:
            return f"{_n}{cfg.population_curve_suffix}"
        return _n

    raw_n_bins = int(raw_params.n_spatial_bins) if raw_params is not None else 7
    pf_n_bins = int(pf_params.n_spatial_bins) if pf_params is not None else raw_n_bins

    # ===== verbatim body from caban/main.py =====
    if plot_TFC_2D_decoding:

        PV_2D_PLOTS_DIR = os.path.join(PLOTS_DIR, _popcurve_dirname("PV_2D_correlation"))
        os.makedirs(PV_2D_PLOTS_DIR, exist_ok=True)

        # Build test session dicts for PV correlation
        pv_test_sessions = {}
        if Test_A:
            pv_test_sessions["Test_A"] = Test_A
        if Test_A_1wk:
            pv_test_sessions["Test_A_1wk"] = Test_A_1wk
        if Test_B:
            pv_test_sessions["Test_B"] = Test_B
        if Test_B_1wk:
            pv_test_sessions["Test_B_1wk"] = Test_B_1wk

        if pv_test_sessions:
            msg_start('*** 2D Population Vector Correlation')

            pv_2D_mappings = {
                "Test_A":     mapping_TFC_cond_Test_A_Test_A_1wk,
                "Test_A_1wk": mapping_TFC_cond_Test_A_Test_A_1wk,
                "Test_B":     mapping_TFC_cond_Test_B_Test_B_1wk,
                "Test_B_1wk": mapping_TFC_cond_Test_B_Test_B_1wk,
            }

            pv_2D_results = run_2D_pv_correlation_pipeline(
                TFC_cond,
                pv_test_sessions,
                mouse_groups,
                mappings=pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                n_bins=raw_n_bins,
                smooth_sigma=1.0,
                min_occupancy_frames=4,
                first_n_sec=180.0,
                use_z_score=use_z_score,
                auto_close=True,
            )

            # ---- Delta scores: 1wk - 48h per family ----
            for ntype, nlabel in [("raw", "raw"), ("z", "z")]:
                deltas = compute_pv_delta_scores(
                    pv_2D_results, mouse_groups,
                    metric_name="frac_best_match_same_bin", norm_type=ntype,
                )
                plot_pv_delta_scores(
                    deltas, PV_2D_PLOTS_DIR, n_bins=raw_n_bins,
                    metric_label="Frac best=same", norm_label=nlabel,
                    auto_close=True,
                )

            # ---- Mixed model: metric ~ group * target * delay + (1|mouse) ----
            for ntype in ["raw", "z"]:
                run_pv_mixed_model(
                    pv_2D_results, mouse_groups, PV_2D_PLOTS_DIR, n_bins=raw_n_bins,
                    metric_name="frac_best_match_same_bin", norm_type=ntype,
                    auto_close=True,
                )

            # ---- Spatial information (Skaggs, bits/spike) ----
            pv_all_sessions = {"TFC_cond": TFC_cond}
            pv_all_sessions.update(pv_test_sessions)

            msg_start('*** Spatial Information (Skaggs)')
            si_results = compute_spatial_information(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                n_bins=raw_n_bins, smooth_sigma=1.0, min_occupancy_frames=4,
                first_n_sec=180.0, auto_close=True,
            )
            msg_end()

            # ---- Place field stability (per-neuron rate-map r) ----
            msg_start('*** Place Field Stability')
            stab_results = compute_place_field_stability(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                n_bins=raw_n_bins, smooth_sigma=1.0, min_occupancy_frames=4,
                first_n_sec=180.0, auto_close=True,
            )
            msg_end()

            # ---- Population dimensionality (PCA participation ratio) ----
            msg_start('*** Population Dimensionality (PR)')
            dim_results = compute_population_dimensionality(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                first_n_sec=180.0, auto_close=True,
            )
            msg_end()

            # ---- PF-based Spatial Information (all PF neurons) ----
            msg_start('*** Spatial Information - PF (all PF neurons)')
            si_pf_all = compute_spatial_information_PF(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=None, auto_close=True,
            )
            msg_end()

            # ---- PF-based Spatial Information (single-PF neurons) ----
            msg_start('*** Spatial Information - PF (1 PF only)')
            si_pf_1 = compute_spatial_information_PF(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=1, auto_close=True,
            )
            msg_end()

            # ---- PF-based Field Stability (all PF neurons) ----
            msg_start('*** PF Stability (all PF neurons)')
            stab_pf_all = compute_place_field_stability_PF(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=None, auto_close=True,
            )
            msg_end()

            # ---- PF-based Field Stability (single-PF neurons) ----
            msg_start('*** PF Stability (1 PF only)')
            stab_pf_1 = compute_place_field_stability_PF(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=1, auto_close=True,
            )
            msg_end()

            # ---- PF Centroid Shift (single-PF neurons) ----
            msg_start('*** PF Centroid Shift (1 PF)')
            shift_pf_1 = compute_pf_centroid_shift(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=1, auto_close=True,
            )
            msg_end()

            # ---- PF Centroid Shift (all PF neurons) ----
            msg_start('*** PF Centroid Shift (all PF)')
            shift_pf_all = compute_pf_centroid_shift(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=None, auto_close=True,
            )
            msg_end()

            # ---- PF-filtered 2D PV Correlation (all PF neurons) ----
            msg_start('*** 2D PV Correlation - PF (all PF neurons)')
            pv_2D_PF_all = run_2D_pv_correlation_pipeline_PF(
                TFC_cond,
                pv_test_sessions,
                mouse_groups,
                mappings=pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                n_bins=pf_n_bins,
                smooth_sigma=1.0,
                min_occupancy_frames=4,
                first_n_sec=180.0,
                max_pf_count=None,
                use_z_score=use_z_score,
                auto_close=True,
            )
            # Delta scores for PF-all
            for ntype, nlabel in [("raw", "raw"), ("z", "z")]:
                deltas_pf = compute_pv_delta_scores(
                    pv_2D_PF_all, mouse_groups,
                    metric_name="frac_best_match_same_bin", norm_type=ntype,
                )
                plot_pv_delta_scores(
                    deltas_pf, PV_2D_PLOTS_DIR, n_bins=pf_n_bins,
                    metric_label="Frac best=same (PF)",
                    norm_label=nlabel, auto_close=True,
                )
            # Mixed model for PF-all
            for ntype in ["raw", "z"]:
                run_pv_mixed_model(
                    pv_2D_PF_all, mouse_groups, PV_2D_PLOTS_DIR, n_bins=pf_n_bins,
                    metric_name="frac_best_match_same_bin", norm_type=ntype,
                    auto_close=True, dir_suffix="_PF",
                )
            msg_end()

            # ---- PF-filtered 2D PV Correlation (single-PF neurons) ----
            msg_start('*** 2D PV Correlation - PF (1 PF only)')
            pv_2D_PF_1 = run_2D_pv_correlation_pipeline_PF(
                TFC_cond,
                pv_test_sessions,
                mouse_groups,
                mappings=pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                n_bins=pf_n_bins,
                smooth_sigma=1.0,
                min_occupancy_frames=4,
                first_n_sec=180.0,
                max_pf_count=1,
                use_z_score=use_z_score,
                auto_close=True,
            )
            # Delta scores for PF-1
            for ntype, nlabel in [("raw", "raw"), ("z", "z")]:
                deltas_pf1 = compute_pv_delta_scores(
                    pv_2D_PF_1, mouse_groups,
                    metric_name="frac_best_match_same_bin", norm_type=ntype,
                )
                plot_pv_delta_scores(
                    deltas_pf1, PV_2D_PLOTS_DIR, n_bins=pf_n_bins,
                    metric_label="Frac best=same (PF npf1)",
                    norm_label=nlabel, auto_close=True,
                )
            # Mixed model for PF-1
            for ntype in ["raw", "z"]:
                run_pv_mixed_model(
                    pv_2D_PF_1, mouse_groups, PV_2D_PLOTS_DIR, n_bins=pf_n_bins,
                    metric_name="frac_best_match_same_bin", norm_type=ntype,
                    auto_close=True, dir_suffix="_PF_npf1",
                )
            msg_end()

            # ---- Pooled PF stability (all PF neurons) ----
            msg_start('*** PF Stability POOLED (all PF neurons)')
            compute_place_field_stability_PF_pooled(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=None, auto_close=True,
            )
            msg_end()

            # ---- Pooled PF stability (single-PF neurons) ----
            msg_start('*** PF Stability POOLED (1 PF only)')
            compute_place_field_stability_PF_pooled(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=1, auto_close=True,
            )
            msg_end()

            # ---- Pooled PF centroid shift (all PF neurons) ----
            msg_start('*** PF Centroid Shift POOLED (all PF neurons)')
            compute_pf_centroid_shift_pooled(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=None, auto_close=True,
            )
            msg_end()

            # ---- Pooled PF centroid shift (single-PF neurons) ----
            msg_start('*** PF Centroid Shift POOLED (1 PF only)')
            compute_pf_centroid_shift_pooled(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                max_pf_count=1, auto_close=True,
            )
            msg_end()

            # ---- PF-filtered dimensionality (all PF neurons) ----
            msg_start('*** Dimensionality PF (all PF neurons)')
            compute_population_dimensionality_PF(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                first_n_sec=180.0, max_pf_count=None, auto_close=True,
            )
            msg_end()

            # ---- PF-filtered dimensionality (single-PF neurons) ----
            msg_start('*** Dimensionality PF (1 PF only)')
            compute_population_dimensionality_PF(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                first_n_sec=180.0, max_pf_count=1, auto_close=True,
            )
            msg_end()

            # ---- Place Field Turnover (Ziv et al. style) ----
            pv_pf_turnover_specs = []
            if "Test_A" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "TFC_cond",
                    "sess2_label": "Test_A",
                    "mapping": mapping_TFC_cond_Test_A_Test_A_1wk,
                    "output_key": "Test_A",
                    "display_label": "TFC→Test_A",
                })
            if "Test_A_1wk" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "TFC_cond",
                    "sess2_label": "Test_A_1wk",
                    "mapping": mapping_TFC_cond_Test_A_Test_A_1wk,
                    "output_key": "Test_A_1wk",
                    "display_label": "TFC→Test_A_1wk",
                })
            if "Test_B" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "TFC_cond",
                    "sess2_label": "Test_B",
                    "mapping": mapping_TFC_cond_Test_B_Test_B_1wk,
                    "output_key": "Test_B",
                    "display_label": "TFC→Test_B",
                })
            if "Test_B_1wk" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "TFC_cond",
                    "sess2_label": "Test_B_1wk",
                    "mapping": mapping_TFC_cond_Test_B_Test_B_1wk,
                    "output_key": "Test_B_1wk",
                    "display_label": "TFC→Test_B_1wk",
                })
            if "Test_A" in pv_all_sessions and "Test_A_1wk" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "Test_A",
                    "sess2_label": "Test_A_1wk",
                    "mapping": mapping_TFC_cond_Test_A_Test_A_1wk,
                    "output_key": "Test_A_to_Test_A_1wk",
                    "display_label": "Test_A→Test_A_1wk",
                })
            if "Test_B" in pv_all_sessions and "Test_B_1wk" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "Test_B",
                    "sess2_label": "Test_B_1wk",
                    "mapping": mapping_TFC_cond_Test_B_Test_B_1wk,
                    "output_key": "Test_B_to_Test_B_1wk",
                    "display_label": "Test_B→Test_B_1wk",
                })
            if "Test_A" in pv_all_sessions and "Test_B" in pv_all_sessions:
                pv_pf_turnover_specs.append({
                    "sess1_label": "Test_A",
                    "sess2_label": "Test_B",
                    "mapping": mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk,
                    "output_key": "Test_A_to_Test_B",
                    "display_label": "Test_A→Test_B",
                })
            msg_start('*** PF Turnover')
            turnover_results = compute_pf_turnover(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                auto_close=True,
                comparison_specs=pv_pf_turnover_specs,
            )
            msg_end()

            # ---- Place Field Turnover Pooled (neuron-level) ----
            msg_start('*** PF Turnover POOLED')
            turnover_pooled = compute_pf_turnover_pooled(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                auto_close=True,
                comparison_specs=pv_pf_turnover_specs,
            )
            msg_end()

            # ---- PF Turnover Example Gallery ----
            msg_start('*** PF Turnover Example Gallery')
            plot_pf_turnover_examples(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR,
                n_examples=3,
                auto_close=True,
                comparison_specs=pv_pf_turnover_specs,
            )
            msg_end()

            # ---- PF Turnover Example Gallery VS (3-session rows) ----
            msg_start('*** PF Turnover Example Gallery VS')
            plot_pf_turnover_examples_VS(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR, n_examples=3, auto_close=True,
            )
            msg_end()

            # ---- Cross-registration sanity check ----
            msg_start('*** Crossreg Sanity Check')
            crossreg_sanity_check(
                pv_all_sessions, mouse_groups, pv_2D_mappings,
                PLOTS_DIR=PV_2D_PLOTS_DIR, n_per_group=10, auto_close=True,
            )
            msg_end()


    # ─────────────────────────────────────────────────────────────────────────────
    # Epoch population-vector similarity and RDM analysis (TFC conditioning)
    # ─────────────────────────────────────────────────────────────────────────────

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: epoch_pv_within  (caban/main.py L5563-5631)
# ---------------------------------------------------------------------------
def run_epoch_pv_within(ds, cfg):
    """Analysis section: epoch_pv_within. Originally caban/main.py L5563-5631."""
    if not (cfg.plot_epoch_pv_analysis):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_cond = ds.TFC_cond
    TFC_cond_crossreg = ds.TFC_cond_crossreg
    mapping_FULL = ds.mapping_FULL
    mice_per_group = ds.mice_per_group
    # --- cfg switches ---
    plot_epoch_pv_analysis = cfg.plot_epoch_pv_analysis

    # ===== verbatim body from caban/main.py =====
    if plot_epoch_pv_analysis:
        msg_start('*** Epoch PV similarity + RDM analysis (TFC_cond)')

        # ── Switches ──────────────────────────────────────────────────────────
        # Matrix types to loop over: 'S' (deconvolved) and/or 'C' (raw calcium)
        epoch_pv_matrix_types = ['S', 'C']

        # Aggregations to loop over: 'integral' sums the signal over the epoch
        # (total calcium load); 'mean' divides by epoch length (normalised rate).
        # For cosine and Pearson similarity these give identical results; they
        # differ for Mahalanobis or when comparing across unequal-duration epochs.
        epoch_pv_aggregations = ['integral', 'mean']

        # Event-rate mode: when True, also runs spike-count event-rate PV.
        # Event rate is S-based (peaks of deconvolved trace), so it runs once
        # regardless of matrix type or aggregation.
        epoch_pv_use_event_rate = True

        # Similarity metrics to compute
        epoch_pv_metrics = ('pearson', 'cosine')

        # Mappings: 'full' uses all cells per session; add crossreg mapping
        # strings (e.g. 'TFC_cond+Test_B+Test_B_1wk') for matched-cell subsets.
        epoch_pv_mappings = [mapping_FULL]

        # Epoch timing parameters
        epoch_pv_peri_shock_pre_s  = 10.0   # seconds before shock onset
        epoch_pv_peri_shock_post_s = 10.0   # seconds after shock onset
        epoch_pv_pre_tone_dur_s    = 35.0   # pre-tone baseline duration

        # Mobility filter: None = all frames, 'mobile' = speed >= 2 cm/s,
        # 'immobile' = speed < 2 cm/s.  Loops over all three by default.
        epoch_pv_mobility_filters = [None, 'mobile', 'immobile']

        # ── Build data_mode list from switches ────────────────────────────────
        # One mode per (matrix_type, aggregation) pair; event_rate appended once.
        _ep_data_modes = [f'{mt}_{agg}'
                          for mt in epoch_pv_matrix_types
                          for agg in epoch_pv_aggregations]
        if epoch_pv_use_event_rate:
            _ep_data_modes.append('event_rate')

        # ── Run ───────────────────────────────────────────────────────────────
        for _ep_mapping in epoch_pv_mappings:
            for _ep_data_mode in _ep_data_modes:
                for _ep_mob in epoch_pv_mobility_filters:
                    epoch_pv_results = run_epoch_analysis_all_mice(
                        PLOTS_DIR=PLOTS_DIR,
                        mice_per_group=mice_per_group,
                        TFC_cond=TFC_cond,
                        TFC_cond_crossreg=TFC_cond_crossreg,
                        mapping=_ep_mapping,
                        data_mode=_ep_data_mode,
                        metrics=epoch_pv_metrics,
                        epochs=None,                              # all EPOCH_NAMES
                        peri_shock_pre_s=epoch_pv_peri_shock_pre_s,
                        peri_shock_post_s=epoch_pv_peri_shock_post_s,
                        pre_tone_duration_s=epoch_pv_pre_tone_dur_s,
                        mobility_filter=_ep_mob,
                        auto_close=True,
                    )

        msg_end()


    # ─────────────────────────────────────────────────────────────────────────────
    # Cross-session epoch PV similarity (TFC_cond vs recall sessions)
    # ─────────────────────────────────────────────────────────────────────────────

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: epoch_pv_cross  (caban/main.py L5632-5708)
# ---------------------------------------------------------------------------
def run_epoch_pv_cross(ds, cfg):
    """Analysis section: epoch_pv_cross. Originally caban/main.py L5632-5708."""
    if not (cfg.plot_cross_session_epoch_pv_analysis):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_AB_48hr_1wk_crossreg = ds.TFC_AB_48hr_1wk_crossreg
    TFC_A_A_1wk_crossreg = ds.TFC_A_A_1wk_crossreg
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_A = ds.Test_A
    Test_A_1wk = ds.Test_A_1wk
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk
    mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mice_per_group = ds.mice_per_group
    # --- cfg switches ---
    plot_cross_session_epoch_pv_analysis = cfg.plot_cross_session_epoch_pv_analysis

    # ===== verbatim body from caban/main.py =====
    if plot_cross_session_epoch_pv_analysis:
        msg_start('*** Cross-session Epoch PV similarity (TFC_cond vs recall)')

        # ── Local mapping aliases (self-contained when running block alone) ───
        _xep_map_B   = mapping_TFC_cond_Test_B_Test_B_1wk           # crossreg 4
        _xep_map_A   = mapping_TFC_cond_Test_A_Test_A_1wk           # crossreg 7
        _xep_map_all = mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk  # crossreg 6

        # ── Switches ──────────────────────────────────────────────────────────
        xep_matrix_types   = ['S', 'C']
        xep_aggregations   = ['integral', 'mean']
        xep_use_event_rate = True
        xep_metrics        = ('pearson', 'cosine')
        xep_mobility_filters = [None, 'mobile', 'immobile']

        # Epoch timing parameters (shared with within-session)
        xep_peri_shock_pre_s  = 10.0
        xep_peri_shock_post_s = 10.0
        xep_pre_tone_dur_s    = 35.0
        xep_post_tone_dur_s   = 35.0   # post-tone / post-pseudo-tone window

        # ── Build data_mode list ──────────────────────────────────────────────
        _xep_data_modes = [f'{mt}_{agg}'
                           for mt in xep_matrix_types
                           for agg in xep_aggregations]
        if xep_use_event_rate:
            _xep_data_modes.append('event_rate')

        # ── Recall session configurations ─────────────────────────────────────
        # Each entry: (label, sessions_dict, recall_type,
        #              [(mapping_str, crossreg_dict), ...])
        _xep_recall_configs = [
            ('Test_B', Test_B, 'testb', [
                (_xep_map_B,   TFC_B_B_1wk_crossreg),
                (_xep_map_all, TFC_AB_48hr_1wk_crossreg),
            ]),
            ('Test_B_1wk', Test_B_1wk, 'testb', [
                (_xep_map_B,   TFC_B_B_1wk_crossreg),
                (_xep_map_all, TFC_AB_48hr_1wk_crossreg),
            ]),
            ('Test_A', Test_A, 'testa', [
                (_xep_map_A,   TFC_A_A_1wk_crossreg),
                (_xep_map_all, TFC_AB_48hr_1wk_crossreg),
            ]),
            ('Test_A_1wk', Test_A_1wk, 'testa', [
                (_xep_map_A,   TFC_A_A_1wk_crossreg),
                (_xep_map_all, TFC_AB_48hr_1wk_crossreg),
            ]),
        ]

        # ── Run ───────────────────────────────────────────────────────────────
        for _label, _sessions, _rtype, _mappings in _xep_recall_configs:
            for _mapping, _xreg_dict in _mappings:
                for _dm in _xep_data_modes:
                    for _mob in xep_mobility_filters:
                        run_cross_session_epoch_analysis_all_mice(
                            PLOTS_DIR=PLOTS_DIR,
                            mice_per_group=mice_per_group,
                            TFC_cond=TFC_cond,
                            recall_sessions=_sessions,
                            recall_label=_label,
                            recall_type=_rtype,
                            crossreg_dict=_xreg_dict,
                            mapping=_mapping,
                            data_mode=_dm,
                            metrics=xep_metrics,
                            peri_shock_pre_s=xep_peri_shock_pre_s,
                            peri_shock_post_s=xep_peri_shock_post_s,
                            pre_tone_duration_s=xep_pre_tone_dur_s,
                            post_tone_duration_s=xep_post_tone_dur_s,
                            mobility_filter=_mob,
                            auto_close=True,
                        )

        msg_end()

    # ---- Population PCA Trajectory Analysis ----
    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: population_pca  (caban/main.py L5709-5861)
# ---------------------------------------------------------------------------
def run_population_pca(ds, cfg):
    """Analysis section: population_pca. Originally caban/main.py L5709-5861."""
    if not (True):
        return
    # --- ds attributes ---
    ENGRAM_MODES = ds.ENGRAM_MODES
    NPY_SAVE_PATH = ds.NPY_SAVE_PATH
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    engram_id = ds.engram_id
    engram_norms = ds.engram_norms
    engram_rates = ds.engram_rates
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    ENGRAM_MODES = cfg.ENGRAM_MODES
    NUM_ENGRAM_PLOT_CELLS = cfg.NUM_ENGRAM_PLOT_CELLS

    # ===== verbatim body from caban/main.py =====
    msg_start('*** Population PCA Trajectory Analysis')
    from caban.population import (
        run_population_pca_pipeline,
        run_pca_state_metrics_from_results,
        EXCLUDE_MICE_CROSSREG as _PCA_EXCLUDE,
    )

    # Top-level PCA temporal-binning default (frames per bin) — applied to
    # CrossregFullPCA only (trial-averaged variants always run at frame
    # resolution). Default 1 s/bin at MINISCOPE_FPS=20 Hz.
    PCA_FRAMES_PER_BIN = 1 # MINISCOPE_FPS * 1

    want_engram_pop_pca = True

    # Engram-type switch for the PCA pipeline.
    #   'encoding' : engram identity defined on TFC_cond full-session cells.
    #   'recall'   : engram identity defined on Test_B  full-session cells.
    which_engram = 'encoding'

    # --- engram sanity plots (both etypes, all modes, all mice) -----------------
    from caban.engram_sanity import plot_engram_sanity
    _engram_plots_root = os.path.join(PLOTS_DIR, "engram_plots")
    print(f"*** Engram sanity plots -> {_engram_plots_root}", flush=True)
    plot_engram_sanity(
        engram_id=engram_id,
        engram_rates=engram_rates,
        engram_norms=engram_norms,
        ref_sessions_by_etype={'encoding': TFC_cond, 'recall': Test_B},
        other_sessions_by_name={'Test_B_1wk': Test_B_1wk},
        mouse_groups=mouse_groups,
        save_root=_engram_plots_root,
        n_cells=NUM_ENGRAM_PLOT_CELLS,
        modes=ENGRAM_MODES,
        etypes=('encoding', 'recall'),
    )

    # --- run the Population PCA pipeline for the chosen engram-type, all modes --
    # Build {mouse: ndarray of full reference-session S row indices} per mode,
    # resolved against the unified engram identity.
    _REF_NAME = ENGRAM_REFERENCE[which_engram]
    def _engram_idx_by_mouse(mode):
        out = {}
        for _m in mouse_groups:
            if _m in _PCA_EXCLUDE:
                continue
            if _m not in engram_id or which_engram not in engram_id[_m]:
                continue
            out[_m] = np.where(engram_id[_m][which_engram][mode])[0]
        return out


    def _run_population_pca_and_metrics(pop_plots_dir, want_engram, engram_mode=None,
                                         normalize=False, binarize=False,
                                         frames_per_bin=PCA_FRAMES_PER_BIN):
        if want_engram:
            if engram_mode is None:
                raise RuntimeError(
                    "_run_population_pca_and_metrics(want_engram=True) requires engram_mode."
                )
            engram_idx_by_mouse = _engram_idx_by_mouse(engram_mode)
            _suffix = ""
            if normalize: _suffix += " [normalize]"
            if binarize:  _suffix += " [binarize]"
            _suffix += f" [fpb={frames_per_bin}]"
            print(f"\n*** Population PCA pipeline - etype={which_engram} mode={engram_mode}"
                  f"{_suffix} -> {pop_plots_dir}", flush=True)
        else:
            engram_idx_by_mouse = None
            _suffix = ""
            if normalize: _suffix += " [normalize]"
            if binarize:  _suffix += " [binarize]"
            _suffix += f" [fpb={frames_per_bin}]"
            print(f"\n*** Population PCA pipeline - all crossreg cells"
                  f"{_suffix} -> {pop_plots_dir}", flush=True)

        res = run_population_pca_pipeline(
            TFC_cond=TFC_cond,
            Test_B=Test_B,
            Test_B_1wk=Test_B_1wk,
            mouse_groups=mouse_groups,
            TFC_B_B_1wk_crossreg=TFC_B_B_1wk_crossreg,
            mapping_TFC_cond_Test_B_Test_B_1wk=mapping_TFC_cond_Test_B_Test_B_1wk,
            PLOTS_DIR=pop_plots_dir,
            NPY_SAVE_PATH=NPY_SAVE_PATH,
            auto_close=True,
            want_engram=want_engram,
            engram_idx_by_mouse=engram_idx_by_mouse,
            which_engram=which_engram,
            normalize=normalize,
            binarize=binarize,
            frames_per_bin=frames_per_bin,
        )
        run_pca_state_metrics_from_results(
            results=res,
            mouse_groups=mouse_groups,
            PLOTS_DIR=pop_plots_dir,
            auto_close=True,
        )
        return res


    # Also run the same PCA pipeline on all cross-registered neurons (no engram filter).
    POP_PCA_CROSSREG_PLOTS_DIR = os.path.join(PLOTS_DIR, "Population_PCA_crossreg")
    pop_pca_results_crossreg = _run_population_pca_and_metrics(
        pop_plots_dir=POP_PCA_CROSSREG_PLOTS_DIR,
        want_engram=False,
    )

    # Same all-crossreg PCA but with per-cell-per-session peakval normalization
    # (compensates session-level peakval inflation, e.g. CNO in TFC_cond).
    POP_PCA_CROSSREG_NORMALIZE_DIR = os.path.join(PLOTS_DIR, "Population_PCA_crossreg_normalize")
    pop_pca_results_crossreg_normalize = _run_population_pca_and_metrics(
        pop_plots_dir=POP_PCA_CROSSREG_NORMALIZE_DIR,
        want_engram=False,
        normalize=True,
    )

    # Same all-crossreg PCA but with binary spike-peak input (1.0 at S_spikes
    # frames, 0.0 elsewhere). z-scoring is skipped automatically; after binning
    # values are integer spike counts per bin.
    POP_PCA_CROSSREG_BINARIZE_DIR = os.path.join(PLOTS_DIR, "Population_PCA_crossreg_binarize")
    pop_pca_results_crossreg_binarize = _run_population_pca_and_metrics(
        pop_plots_dir=POP_PCA_CROSSREG_BINARIZE_DIR,
        want_engram=False,
        binarize=True,
    )

    pop_pca_results_by_mode = {}
    for _em in ENGRAM_MODES:
        POP_PCA_PLOTS_DIR = os.path.join(
            PLOTS_DIR,
            ("Population_PCA_engram" if want_engram_pop_pca else "Population_PCA_noengram")
            + (f"_{which_engram}_{_em}" if want_engram_pop_pca else ""),
        )
        _res = _run_population_pca_and_metrics(
            pop_plots_dir=POP_PCA_PLOTS_DIR,
            want_engram=want_engram_pop_pca,
            engram_mode=_em,
        )
        pop_pca_results_by_mode[_em] = _res

    # Backward-compat: keep the existing names pointing to the per-mouse mode so
    # any downstream cells that reference pop_pca_results / POP_PCA_PLOTS_DIR
    # continue to work unchanged.
    pop_pca_results = pop_pca_results_by_mode['permouse']
    POP_PCA_PLOTS_DIR = os.path.join(
        PLOTS_DIR,
        ("Population_PCA_engram" if want_engram_pop_pca else "Population_PCA_noengram")
        + (f"_{which_engram}_permouse" if want_engram_pop_pca else ""),
    )
    msg_end()

    # ---- Isomap manifold pipeline (Wilson-lab style) ----
    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: isomap  (caban/main.py L5862-5882)
# ---------------------------------------------------------------------------
def run_isomap(ds, cfg):
    """Analysis section: isomap. Originally caban/main.py L5862-5882."""
    if not (True):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mice_per_group_Test_B_B_1wk = ds.mice_per_group_Test_B_B_1wk
    mouse_groups = ds.mouse_groups

    # ===== verbatim body from caban/main.py =====
    msg_start('*** Isomap Manifold Analysis')
    from caban.isomap import run_isomap_pipeline
    ISOMAP_PLOTS_DIR = os.path.join(PLOTS_DIR, "TFC_Isomap")
    # G07 and G15 lack the full B-test set (TFC_cond + Test_B + Test_B_1wk),
    # so exclude them from the Isomap pipeline.
    _isomap_mouse_groups = {m: g for m, g in mouse_groups.items()
                            if m not in ("G07", "G15")}
    isomap_results = run_isomap_pipeline(
        TFC_cond=TFC_cond,
        Test_B=Test_B,
        Test_B_1wk=Test_B_1wk,
        mouse_groups=_isomap_mouse_groups,
        mice_per_group_Test_B_B_1wk=mice_per_group_Test_B_B_1wk,
        TFC_B_B_1wk_crossreg=TFC_B_B_1wk_crossreg,
        mapping_TFC_cond_Test_B_Test_B_1wk=mapping_TFC_cond_Test_B_Test_B_1wk,
        PLOTS_DIR=ISOMAP_PLOTS_DIR,
        NPY_SAVE_PATH=NPY_SAVE_PATH,
        auto_close=True,
    )
    msg_end()

    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: population_vectors  (caban/main.py L5883-5937)
# ---------------------------------------------------------------------------
def run_population_vectors(ds, cfg):
    """Analysis section: population_vectors. Originally caban/main.py L5883-5937."""
    if not (cfg.plot_population_vectors):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mice_per_group = ds.mice_per_group
    mouse_groups = ds.mouse_groups
    # --- cfg switches ---
    perform_agglomerative_clustering = cfg.perform_agglomerative_clustering
    plot_population_vectors = cfg.plot_population_vectors
    process_for_R = cfg.process_for_R

    # ===== verbatim body from caban/main.py =====
    if plot_population_vectors: #and not DEVEL_SWITCH:
        #plot_pop_vectors(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond')
        if process_for_R:
            for mouse in [m for m in mouse_groups if m not in ['G07', 'G15']]:
                process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                            binarize=False, normalize=False, normalize_full=False, spk_cutoff=2)
                process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                            binarize=True, normalize=False, spk_cutoff=2)
                process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                            binarize=False, normalize=True, spk_cutoff=2)
                process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                            binarize=False, normalize=False, normalize_full=True, spk_cutoff=2)

        if perform_agglomerative_clustering:
            sess_all = [TFC_cond, Test_B, Test_B_1wk]
            sess_all_use = [TFC_cond, Test_B, Test_B_1wk]
            sess_label_all = ['TFC_cond', 'Test_B', 'Test_B_1wk']
            PV_sess = dict()
            use_silhouette = True
            saver_PV_sess= Saver(parent_path=NPY_SAVE_PATH, subdirs=['PV_sess'])
            if saver_PV_sess.check_exists('PV_sess'):
                PV_sess = saver_PV_sess.load('PV_sess')
            else:
                for Ca_act_type in ['full', 'mov', 'imm']:
                    PV_sess[Ca_act_type] = dict()
                    for sess, sess_label in zip(sess_all, sess_label_all):
                        PV_sess[Ca_act_type][sess_label] = dict()
                        for only_crossreg in [True, False]:
                            only_crossreg_str = get_only_crossreg_str(only_crossreg)
                            for transpose_wanted in [True, False]:
                                transpose_str = get_transpose_str(transpose_wanted)
                                PV_sess[Ca_act_type][sess_label][only_crossreg_str] = dict()
                                print('\n*** WORKING cluster_pop_vectors() for {}, only_crossreg = {}, use_silhouette = {}, transpose_wanted = {}'.format(Ca_act_type, only_crossreg, use_silhouette, transpose_wanted))
                                PV_group = cluster_pop_vectors(PLOTS_DIR, sess, sess_label, mice_per_group, transpose_wanted=transpose_wanted, auto_close=True, \
                                    crossreg=TFC_B_B_1wk_crossreg, Ca_act_type=Ca_act_type, only_crossreg=only_crossreg, use_silhouette=use_silhouette, bin_width=1, \
                                    sess_all=sess_all)
                                PV_sess[Ca_act_type][sess_label][only_crossreg_str][transpose_str] = PV_group
                    saver_PV_sess.save(PV_sess[Ca_act_type], 'PV_sess_{}'.format(Ca_act_type))

            for only_crossreg in [True, False]:
                process_pop_vectors(PLOTS_DIR, PV_sess, only_crossreg, crossreg_str='TFC_B_B_1wk', plot_type='boxplot', want_scatter=True, auto_close=True, \
                    use_silhouette=use_silhouette, frac_type_l=[1/2], force_calc=True, cohens_thresh=0.1)

        '''
            [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
                cluster_pop_vectors(PLOTS_DIR, TFC_cond, 'TFC_cond', mice_per_group, transpose_wanted=False, auto_close=True, bin_width=2, \
                    spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)
            [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
                cluster_pop_vectors(PLOTS_DIR, Test_B, 'Test_B', mice_per_group, transpose_wanted=False, auto_close=True, bin_width=2, \
                    spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)
            [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
                cluster_pop_vectors(PLOTS_DIR, Test_B_1wk, 'Test_B_1wk', mice_per_group, transpose_wanted=False, auto_close=True, bin_width=2, \
                    spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)
        '''

    # ===== end verbatim body =====

# ---------------------------------------------------------------------------
# Section: UMAP
# ---------------------------------------------------------------------------
def run_umap(ds, cfg):
    """Analysis section: UMAP population embeddings for TFC_cond, Test_B, Test_B_1wk."""
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from umap import UMAP
    from mpl_toolkits.mplot3d import Axes3D
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    mapping_TFC_cond_Test_B_Test_B_1wk = ds.mapping_TFC_cond_Test_B_Test_B_1wk
    mouse_groups = ds.mouse_groups
    # --- 1. Single-session TFC_cond UMAP ---
    print("*** UMAP (1) Single-session TFC_cond\n")
    for m in mouse_groups:
        print(f" {m}...", end='', flush=True)
        sess = TFC_cond[m]
        S = sess.S
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        for i, norm_str in enumerate(['Normalized', 'Non-normalized']):
            if norm_str == 'Normalized':
                S_normalized = np.nan_to_num((S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True))
            else:
                S_normalized = S
            umap = UMAP(n_components=2, n_neighbors=150, n_jobs=-1)
            embedding = umap.fit_transform(S_normalized.T)
            first_shock_idx = 0
            last_shock_idx = len(sess.shock_onsets)-1
            first_tone_idx = 0
            last_tone_idx = len(sess.tone_onsets)-1
            for j, (title_str, shock_idx_to_use, tone_idx_to_use) in enumerate(zip(['First', 'Last'], [first_shock_idx, last_shock_idx], [first_tone_idx, last_tone_idx])):
                ax = axes[i * 2 + j]
                sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(S.shape[1]), cmap='viridis', s=2, alpha=1)
                ax.scatter(embedding[sess.tone_onsets[tone_idx_to_use]:sess.tone_offsets[tone_idx_to_use], 0], embedding[sess.tone_onsets[tone_idx_to_use]:sess.tone_offsets[tone_idx_to_use], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
                ax.scatter(embedding[sess.shock_onsets[shock_idx_to_use]:sess.shock_offsets[shock_idx_to_use], 0], embedding[sess.shock_onsets[shock_idx_to_use]:sess.shock_offsets[shock_idx_to_use], 1], color='red', label='Shock Period', s=8, alpha=1, marker='x')
                ax.set_title(f'UMAP Embedding {title_str} {norm_str}')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize='small')
        plt.colorbar(sc, ax=axes, label='Frame index', shrink=0.7)
        plt.suptitle(f'{m} {mouse_groups[m]} UMAP embedding (n_neighbors=150)')
        plt.show()
        save_path = os.path.join(PLOTS_DIR, 'UMAP', f'{m}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'UMAP-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
        plt.close()
    print('done.')

    # --- 2. Cross-registered UMAP (separate fit per session) ---
    print("*** UMAP (2) Cross-registered UMAP (separate fit per session)\n")
    for m in mouse_groups:
        if m in ['G07', 'G15']:
            continue
        print(f" {m}...", end='', flush=True)
        S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        sess_TFC_cond = TFC_cond[m]
        sess_Test_B = Test_B[m]
        sess_Test_B_1wk = Test_B_1wk[m]
        S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
        S_Test_B = sess_Test_B.S[S_i_Test_B, :]
        S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]
        S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
        S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
        S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))
        umap_TFC_cond = UMAP(n_components=2, n_jobs=-1)
        umap_Test_B = UMAP(n_components=2, n_jobs=-1)
        umap_Test_B_1wk = UMAP(n_components=2, n_jobs=-1)
        embedding_TFC_cond = umap_TFC_cond.fit_transform(S_TFC_cond_n.T)
        embedding_Test_B = umap_Test_B.fit_transform(S_Test_B_n.T)
        embedding_Test_B_1wk = umap_Test_B_1wk.fit_transform(S_Test_B_1wk_n.T)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, first_last_str in enumerate(['First', 'Last']):
            ax = axes[i, 0]
            sc = ax.scatter(embedding_TFC_cond[:, 0], embedding_TFC_cond[:, 1], c=np.arange(S_TFC_cond.shape[1]), cmap='viridis', s=2, alpha=1)
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.tone_onsets[i]:sess_TFC_cond.tone_offsets[i], 0], embedding_TFC_cond[sess_TFC_cond.tone_onsets[i]:sess_TFC_cond.tone_offsets[i], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.shock_onsets[i]:sess_TFC_cond.shock_offsets[i], 0], embedding_TFC_cond[sess_TFC_cond.shock_onsets[i]:sess_TFC_cond.shock_offsets[i], 1], color='red', label='Shock Period', s=8, alpha=1, marker='x')
            ax.set_title(f'TFC_cond {first_last_str}')
            ax = axes[i, 1]
            sc = ax.scatter(embedding_Test_B[:, 0], embedding_Test_B[:, 1], c=np.arange(S_Test_B.shape[1]), cmap='viridis', s=2, alpha=1)
            ax.scatter(embedding_Test_B[sess_Test_B.tone_onsets[i]:sess_Test_B.tone_offsets[i], 0], embedding_Test_B[sess_Test_B.tone_onsets[i]:sess_Test_B.tone_offsets[i], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
            ax.set_title(f'Test_B {first_last_str}')
            ax = axes[i, 2]
            sc = ax.scatter(embedding_Test_B_1wk[:, 0], embedding_Test_B_1wk[:, 1], c=np.arange(S_Test_B_1wk.shape[1]), cmap='viridis', s=2, alpha=1)
            ax.scatter(embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[i]:sess_Test_B_1wk.tone_offsets[i], 0], embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[i]:sess_Test_B_1wk.tone_offsets[i], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
            ax.set_title(f'Test_B_1wk {first_last_str}')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize='small')
        plt.colorbar(sc, ax=axes, label='Frame index', shrink=0.7)
        plt.suptitle(f'{m} {mouse_groups[m]} UMAP embedding (crossreg, separate fit)')
        plt.show()
        save_path = os.path.join(PLOTS_DIR, 'UMAP', 'crossreg', f'{m}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'UMAP-crossreg-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
        plt.close()
    print('done.')

    # --- 3. Cross-registered UMAP (fit on TFC_cond, transform recall) ---
    print("*** UMAP (3) Cross-registered UMAP (fit on TFC_cond, transform recall)\n")
    for m in mouse_groups:
        if m in ['G07', 'G15']:
            continue
        print(f" {m}...", end='', flush=True)
        S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        sess_TFC_cond = TFC_cond[m]
        sess_Test_B = Test_B[m]
        sess_Test_B_1wk = Test_B_1wk[m]
        S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
        S_Test_B = sess_Test_B.S[S_i_Test_B, :]
        S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]
        S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
        S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
        S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))
        umap_TFC_cond = UMAP(n_components=2, n_jobs=-1)
        embedding_TFC_cond = umap_TFC_cond.fit_transform(S_TFC_cond_n.T)
        embedding_Test_B = umap_TFC_cond.transform(S_Test_B_n.T)
        embedding_Test_B_1wk = umap_TFC_cond.transform(S_Test_B_1wk_n.T)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, first_last_str in enumerate(['First', 'Last']):
            ax = axes[i, 0]
            sc = ax.scatter(embedding_TFC_cond[:, 0], embedding_TFC_cond[:, 1], c=np.arange(S_TFC_cond.shape[1]), cmap='viridis', s=2, alpha=1)
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.tone_onsets[i]:sess_TFC_cond.tone_offsets[i], 0], embedding_TFC_cond[sess_TFC_cond.tone_onsets[i]:sess_TFC_cond.tone_offsets[i], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.shock_onsets[i]:sess_TFC_cond.shock_offsets[i], 0], embedding_TFC_cond[sess_TFC_cond.shock_onsets[i]:sess_TFC_cond.shock_offsets[i], 1], color='red', label='Shock Period', s=8, alpha=1, marker='x')
            ax.set_title(f'TFC_cond {first_last_str}')
            ax = axes[i, 1]
            sc = ax.scatter(embedding_Test_B[:, 0], embedding_Test_B[:, 1], c=np.arange(S_Test_B.shape[1]), cmap='viridis', s=2, alpha=1)
            ax.scatter(embedding_Test_B[sess_Test_B.tone_onsets[i]:sess_Test_B.tone_offsets[i], 0], embedding_Test_B[sess_Test_B.tone_onsets[i]:sess_Test_B.tone_offsets[i], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
            ax.set_title(f'Test_B {first_last_str}')
            ax = axes[i, 2]
            sc = ax.scatter(embedding_Test_B_1wk[:, 0], embedding_Test_B_1wk[:, 1], c=np.arange(S_Test_B_1wk.shape[1]), cmap='viridis', s=2, alpha=1)
            ax.scatter(embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[i]:sess_Test_B_1wk.tone_offsets[i], 0], embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[i]:sess_Test_B_1wk.tone_offsets[i], 1], color='blue', label='Tone Period', s=10, alpha=1, marker='^')
            ax.set_title(f'Test_B_1wk {first_last_str}')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize='small')
        plt.colorbar(sc, ax=axes, label='Frame index', shrink=0.7)
        plt.suptitle(f'{m} {mouse_groups[m]} UMAP embedding (crossreg, TFC_cond fit)')
        plt.show()
        save_path = os.path.join(PLOTS_DIR, 'UMAP', 'crossreg-TFC', f'{m}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'UMAP-crossreg-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
        plt.close()
    print('done.')

    # --- 4. Concatenated-session UMAP block ---
    print("*** UMAP (4) Concatenated-session UMAP\n")
    for m in mouse_groups:
        if m in ['G07', 'G15']:
            continue
        print(f" {m}...", end='', flush=True)
        S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        sess_TFC_cond = TFC_cond[m]
        sess_Test_B = Test_B[m]
        sess_Test_B_1wk = Test_B_1wk[m]
        S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
        S_Test_B = sess_Test_B.S[S_i_Test_B, :]
        S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]
        combined_S = np.hstack((S_TFC_cond, S_Test_B, S_Test_B_1wk))
        S_normalized = (combined_S - combined_S.mean(axis=1, keepdims=True)) / combined_S.std(axis=1, keepdims=True)
        umap = UMAP(n_components=2, n_jobs=-1)
        embedding = umap.fit_transform(S_normalized.T)
        frames_TFC_cond = S_TFC_cond.shape[1]
        frames_Test_B = S_Test_B.shape[1]
        frames_Test_B_1wk = S_Test_B_1wk.shape[1]
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc1 = ax.scatter(embedding[:frames_TFC_cond, 0], embedding[:frames_TFC_cond, 1], np.arange(frames_TFC_cond), c=np.arange(frames_TFC_cond), cmap='viridis', label='TFC_cond')
        sc2 = ax.scatter(embedding[frames_TFC_cond:frames_TFC_cond + frames_Test_B, 0], embedding[frames_TFC_cond:frames_TFC_cond + frames_Test_B, 1], np.arange(frames_Test_B) + frames_TFC_cond, c=np.arange(frames_Test_B), cmap='plasma', label='Test_B')
        sc3 = ax.scatter(embedding[frames_TFC_cond + frames_Test_B:, 0], embedding[frames_TFC_cond + frames_Test_B:, 1], np.arange(frames_Test_B_1wk) + frames_TFC_cond + frames_Test_B, c=np.arange(frames_Test_B_1wk), cmap='inferno', label='Test_B_1wk')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('Time')
        plt.legend()
        plt.title('Combined 3D UMAP Embedding with Time Progression')
        plt.show()
        save_path = os.path.join(PLOTS_DIR, 'UMAP', 'concat', f'{m}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'UMAP-concat-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
        plt.close()
    print('done.')

# ---------------------------------------------------------------------------
# Section: Cross-registered PCA+UMAP
# ---------------------------------------------------------------------------
def run_crossreg_pca_umap(
    ds,
    cfg,
    *,
    bin_size_s=1,
    z_score_sess="TFC_cond",
    fit_sess="TFC_cond",
    n_pca_components=30,
    umap_n_neighbors=30,
    umap_min_dist=0.1,
    umap_metric="cosine",
    random_state=None,          # None allows parallel UMAP; fixed seed may force serial behavior
    umap_n_jobs=-1,             # -1 = all cores
    set_parallel_threads=True,
    skip_mice=("G07", "G15"),
    session_order=("TFC_cond", "Test_B", "Test_B_1wk"),
    save_embeddings=True,
    auto_close=True,
    verbose=False,
):
    """Cross-registered PCA→UMAP embeddings for TFC_cond/Test_B/Test_B_1wk.

    Time binning is applied to S before z-scoring/PCA/UMAP using
    frames_per_bin = round(MINISCOPE_FPS * bin_size_s).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from umap import UMAP

    # ---- Parallelization setup ----
    if umap_n_jobs in (None, -1):
        n_threads = os.cpu_count() or 1
        umap_n_jobs_eff = -1
    else:
        n_threads = int(umap_n_jobs)
        umap_n_jobs_eff = int(umap_n_jobs)

    if set_parallel_threads:
        for var in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ[var] = str(n_threads)

        try:
            import numba
            numba.set_num_threads(n_threads)
        except Exception:
            pass

    if random_state is not None and umap_n_jobs_eff != 1:
        print(
            "[WARN] UMAP with random_state set may override parallel execution. "
            "For maximum cores, use random_state=None."
        )

    valid_sessions = {"TFC_cond", "Test_B", "Test_B_1wk"}
    if bin_size_s < 0:
        raise ValueError(f"bin_size_s must be >= 0, got {bin_size_s!r}")
    
    # bin_size_s=0 means no binning; use original frame resolution.
    # bin_size_s > 0 means time-bin at that resolution in seconds.
    frames_per_bin = None
    if bin_size_s > 0:
        frames_per_bin = int(round(MINISCOPE_FPS * float(bin_size_s)))
        if frames_per_bin < 1:
            raise ValueError(
                f"Computed frames_per_bin must be >= 1, got {frames_per_bin} "
                f"from MINISCOPE_FPS={MINISCOPE_FPS} and bin_size_s={bin_size_s}."
            )

    if z_score_sess not in valid_sessions:
        raise ValueError(f"z_score_sess must be one of {sorted(valid_sessions)}, got {z_score_sess!r}")
    if fit_sess not in valid_sessions:
        raise ValueError(f"fit_sess must be one of {sorted(valid_sessions)}, got {fit_sess!r}")

    PLOTS_DIR = ds.PLOTS_DIR
    mouse_groups = ds.mouse_groups
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    mapping = ds.mapping_TFC_cond_Test_B_Test_B_1wk

    sess_dicts = {
        "TFC_cond": TFC_cond,
        "Test_B": Test_B,
        "Test_B_1wk": Test_B_1wk,
    }

    tag = (
        f"zscore_{z_score_sess}"
        f"__bin_size_s_{float(bin_size_s):g}s"
        f"__fit_{fit_sess}"
        f"__pca{n_pca_components}"
        f"__umapN{umap_n_neighbors}_{umap_metric}"
    )
    root_save_path = os.path.join(PLOTS_DIR, "UMAP", "crossreg_PCA_UMAP", tag)
    os.makedirs(root_save_path, exist_ok=True)

    print("*** Cross-registered PCA→UMAP")
    print(f"    z-score reference : {z_score_sess}")
    print(f"    PCA/UMAP fit      : {fit_sess}")
    print(f"    bin_size_s        : {bin_size_s}" + (f" (no binning)" if bin_size_s == 0 else ""))
    if frames_per_bin is not None:
        print(f"    frames_per_bin    : {frames_per_bin}")
    print(f"    PCA components    : {n_pca_components}")
    print(f"    UMAP              : n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric={umap_metric}")
    print(f"    UMAP n_jobs       : {umap_n_jobs}")
    print(f"    thread target     : {n_threads}")
    print(f"    random_state      : {random_state}")
    print(f"    output            : {root_save_path}\n")

    def _crossreg_indices(mouse):
        return {
            "TFC_cond": get_S_indeces_crossreg(TFC_cond[mouse], TFC_B_B_1wk_crossreg[mouse], mapping),
            "Test_B": get_S_indeces_crossreg(Test_B[mouse], TFC_B_B_1wk_crossreg[mouse], mapping),
            "Test_B_1wk": get_S_indeces_crossreg(Test_B_1wk[mouse], TFC_B_B_1wk_crossreg[mouse], mapping),
        }

    def _bin_matrix_time(S_in, frames_per_bin, *, mouse, session_name):
        n_cells, n_frames = S_in.shape
        usable_frames = (n_frames // frames_per_bin) * frames_per_bin
        dropped_frames = n_frames - usable_frames
        if usable_frames <= 0:
            raise ValueError(
                f"{mouse} {session_name}: cannot bin {n_frames} frames with "
                f"frames_per_bin={frames_per_bin}."
            )

        if dropped_frames > 0 and verbose:
            print(
                f"\n    [INFO] {mouse} {session_name}: dropped {dropped_frames} trailing "
                f"frames for binning (frames_per_bin={frames_per_bin})."
            )

        S_trim = S_in[:, :usable_frames]
        n_bins = usable_frames // frames_per_bin
        S_binned = S_trim.reshape(n_cells, n_bins, frames_per_bin).mean(axis=2)
        return S_binned, usable_frames, dropped_frames

    def _event_slice(sess, event_name, event_idx, frames_per_bin, n_bins):
        on_name = f"{event_name}_onsets"
        off_name = f"{event_name}_offsets"
        if not hasattr(sess, on_name) or not hasattr(sess, off_name):
            return None
        onsets = getattr(sess, on_name)
        offsets = getattr(sess, off_name)
        if onsets is None or offsets is None or len(onsets) == 0 or len(offsets) == 0:
            return None
        if event_idx >= len(onsets) or event_idx >= len(offsets):
            return None

        onset = int(onsets[event_idx])
        offset = int(offsets[event_idx])
        if offset <= onset:
            return None

        # If no binning (frames_per_bin is None), use frame indices directly
        if frames_per_bin is None:
            start_bin = onset
            stop_bin = offset
        else:
            start_bin = int(np.floor(onset / frames_per_bin))
            stop_bin = int(np.ceil(offset / frames_per_bin))

        start_bin = max(0, min(start_bin, n_bins))
        stop_bin = max(0, min(stop_bin, n_bins))
        if stop_bin <= start_bin:
            return None
        return slice(start_bin, stop_bin)

    def _all_event_slices(sess, event_name, frames_per_bin, n_bins):
        on_name = f"{event_name}_onsets"
        off_name = f"{event_name}_offsets"
        if not hasattr(sess, on_name) or not hasattr(sess, off_name):
            return []
        onsets = getattr(sess, on_name)
        offsets = getattr(sess, off_name)
        if onsets is None or offsets is None:
            return []

        n_events = min(len(onsets), len(offsets))
        slices = []
        for i in range(n_events):
            sl = _event_slice(sess, event_name, i, frames_per_bin, n_bins)
            if sl is not None:
                slices.append(sl)
        return slices

    def _scatter_session(ax, emb, sess, session_name, first_last_idx, n_frames):
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=np.arange(n_frames), cmap="viridis", s=2, alpha=1,
        )

        tone_sl = _event_slice(sess, "tone", first_last_idx, frames_per_bin, n_frames)
        if tone_sl is not None:
            ax.scatter(
                emb[tone_sl, 0], emb[tone_sl, 1],
                color="blue", label="Tone Period", s=10, alpha=1, marker="^",
            )

        shock_sl = _event_slice(sess, "shock", first_last_idx, frames_per_bin, n_frames)
        if shock_sl is not None:
            ax.scatter(
                emb[shock_sl, 0], emb[shock_sl, 1],
                color="red", label="Shock Period", s=8, alpha=1, marker="x",
            )

        ax.set_title(session_name)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        return sc

    results = {}

    for mouse in mouse_groups:
        if mouse in skip_mice:
            print(f" {mouse} skipped (listed in skip_mice)")
            continue
        if any(mouse not in sess_dicts[s] for s in session_order):
            print(f" {mouse} skipped (missing one of {session_order})")
            continue
        if mouse not in TFC_B_B_1wk_crossreg:
            print(f" {mouse} skipped (missing TFC_B_B_1wk_crossreg)")
            continue

        print(f" {mouse} ({mouse_groups[mouse]})...", end="", flush=True)

        try:
            idx = _crossreg_indices(mouse)
            sess = {s: sess_dicts[s][mouse] for s in session_order}
            S_raw = {
                s: np.asarray(sess[s].S[idx[s], :], dtype=np.float64)
                for s in session_order
            }

            S = {}
            binned_usable_frames = {}
            binned_dropped_frames = {}
            for s in session_order:
                if frames_per_bin is None:
                    # No binning: use original S directly
                    S[s] = S_raw[s]
                    binned_usable_frames[s] = S_raw[s].shape[1]
                    binned_dropped_frames[s] = 0
                else:
                    # Apply time binning
                    S[s], binned_usable_frames[s], binned_dropped_frames[s] = _bin_matrix_time(
                        S_raw[s],
                        frames_per_bin,
                        mouse=mouse,
                        session_name=s,
                    )

            # Z-score reference session defines mean/std for all sessions.
            mu = np.nanmean(S[z_score_sess], axis=1, keepdims=True)
            sigma = np.nanstd(S[z_score_sess], axis=1, keepdims=True)
            sigma[~np.isfinite(sigma) | (sigma == 0)] = 1.0

            X = {
                s: np.nan_to_num((S[s] - mu) / sigma)
                for s in session_order
            }

            n_components_eff = int(
                min(n_pca_components, X[fit_sess].shape[0], X[fit_sess].shape[1])
            )
            if n_components_eff < 2:
                print(f" skipped (too few PCA components: {n_components_eff})")
                continue

            pca = PCA(
                n_components=n_components_eff,
                svd_solver="auto",
                random_state=random_state,
            )

            Z = {}
            Z[fit_sess] = pca.fit_transform(X[fit_sess].T)
            for s in session_order:
                if s != fit_sess:
                    Z[s] = pca.transform(X[s].T)

            reducer = UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=random_state,
                n_jobs=umap_n_jobs_eff,
            )

            U = {}
            U[fit_sess] = reducer.fit_transform(Z[fit_sess])
            for s in session_order:
                if s != fit_sess:
                    U[s] = reducer.transform(Z[s])

            evr = pca.explained_variance_ratio_
            evr_sum = float(np.nansum(evr))

            fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=False)
            last_sc = None

            for row, event_idx in enumerate([0, -1]):
                row_label = "First" if row == 0 else "Last"

                for col, s in enumerate(session_order):
                    ax = axes[row, col]
                    sess_obj = sess[s]

                    if event_idx == -1:
                        tone_onsets = getattr(sess_obj, "tone_onsets", [])
                        idx_to_plot = max(0, len(tone_onsets) - 1) if len(tone_onsets) else 0
                    else:
                        idx_to_plot = event_idx

                    last_sc = _scatter_session(
                        ax,
                        U[s],
                        sess_obj,
                        f"{s} {row_label}",
                        idx_to_plot,
                        S[s].shape[1],
                    )

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if not handles:
                for ax in axes.ravel():
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        break

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(0, 0.5),
                    fontsize="small",
                )

            if last_sc is not None:
                fig.colorbar(last_sc, ax=axes, label="Frame index", shrink=0.7)

            bin_str = "no binning" if bin_size_s == 0 else f"bin={float(bin_size_s):g}s (fpb={frames_per_bin})"
            fig.suptitle(
                f"{mouse} {mouse_groups[mouse]} PCA→UMAP crossreg | "
                f"z-score={z_score_sess}, fit={fit_sess}, "
                f"{bin_str}, "
                f"PCA n={n_components_eff}, EVR={evr_sum:.3f}"
            )

            mouse_save_path = os.path.join(root_save_path, mouse)
            os.makedirs(mouse_save_path, exist_ok=True)

            fig_path = os.path.join(
                mouse_save_path,
                f"PCA-UMAP-crossreg-{mouse_groups[mouse]}-{mouse}-{tag}.png",
            )

            plt.savefig(fig_path, format="png", dpi=600, bbox_inches="tight")
            plt.show()

            if auto_close:
                plt.close(fig)

            # Separate view: all binned time points with all tone/shock periods.
            fig_all, axes_all = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=False)
            last_sc_all = None

            for col, s in enumerate(session_order):
                ax = axes_all[col]
                emb = U[s]
                sess_obj = sess[s]
                n_bins = emb.shape[0]

                last_sc_all = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=np.arange(n_bins),
                    cmap="viridis",
                    s=2,
                    alpha=1,
                )

                tone_slices = _all_event_slices(sess_obj, "tone", frames_per_bin, n_bins)
                for i, sl in enumerate(tone_slices):
                    ax.scatter(
                        emb[sl, 0],
                        emb[sl, 1],
                        color="blue",
                        s=9,
                        alpha=0.9,
                        marker="^",
                        label="Tone Period" if i == 0 else None,
                    )

                shock_slices = _all_event_slices(sess_obj, "shock", frames_per_bin, n_bins)
                for i, sl in enumerate(shock_slices):
                    ax.scatter(
                        emb[sl, 0],
                        emb[sl, 1],
                        color="red",
                        s=8,
                        alpha=0.9,
                        marker="x",
                        label="Shock Period" if i == 0 else None,
                    )

                ax.set_title(f"{s} all binned points")
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")

            handles_all, labels_all = axes_all[0].get_legend_handles_labels()
            if not handles_all:
                for ax in axes_all:
                    handles_all, labels_all = ax.get_legend_handles_labels()
                    if handles_all:
                        break

            if handles_all:
                fig_all.legend(
                    handles_all,
                    labels_all,
                    loc="center left",
                    bbox_to_anchor=(0, 0.5),
                    fontsize="small",
                )

            if last_sc_all is not None:
                fig_all.colorbar(last_sc_all, ax=axes_all, label="Binned frame index", shrink=0.8)

            bin_str_all = "no binning" if bin_size_s == 0 else f"bin={float(bin_size_s):g}s (fpb={frames_per_bin})"
            fig_all.suptitle(
                f"{mouse} {mouse_groups[mouse]} PCA→UMAP crossreg | all points | "
                f"z-score={z_score_sess}, fit={fit_sess}, {bin_str_all}"
            )

            fig_all_path = os.path.join(
                mouse_save_path,
                f"PCA-UMAP-crossreg-ALL-BINS-{mouse_groups[mouse]}-{mouse}-{tag}.png",
            )
            plt.savefig(fig_all_path, format="png", dpi=600, bbox_inches="tight")
            plt.show()

            if auto_close:
                plt.close(fig_all)

            if save_embeddings:
                npz_path = os.path.join(
                    mouse_save_path,
                    f"PCA-UMAP-crossreg-{mouse_groups[mouse]}-{mouse}-{tag}.npz",
                )

                np.savez_compressed(
                    npz_path,
                    bin_size_s=float(bin_size_s),
                    frames_per_bin=int(frames_per_bin),
                    z_score_sess=z_score_sess,
                    fit_sess=fit_sess,
                    session_order=np.array(session_order),
                    pca_explained_variance_ratio=evr,
                    pca_explained_variance_ratio_sum=evr_sum,
                    zscore_mu=mu.squeeze(),
                    zscore_sigma=sigma.squeeze(),
                    **{f"usable_frames_{s}": int(binned_usable_frames[s]) for s in session_order},
                    **{f"dropped_frames_{s}": int(binned_dropped_frames[s]) for s in session_order},
                    **{f"PCA_{s}": Z[s] for s in session_order},
                    **{f"UMAP_{s}": U[s] for s in session_order},
                )

            results[mouse] = {
                "group": mouse_groups[mouse],
                "bin_size_s": float(bin_size_s),
                "frames_per_bin": int(frames_per_bin) if frames_per_bin is not None else None,
                "z_score_sess": z_score_sess,
                "fit_sess": fit_sess,
                "session_order": tuple(session_order),
                "pca": pca,
                "umap": reducer,
                "pca_scores": Z,
                "umap_embeddings": U,
                "explained_variance_ratio": evr,
                "explained_variance_ratio_sum": evr_sum,
                "zscore_mu": mu,
                "zscore_sigma": sigma,
                "fig_path": fig_path,
                "fig_all_bins_path": fig_all_path,
            }

            print(f" done. PCA EVR={evr_sum:.3f}")

        except Exception as exc:
            print(f" FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print("done.")
    return results

# ---------------------------------------------------------------------------
# Section: population_vector_distances  (caban/main.py L5938-6367)
# ---------------------------------------------------------------------------
def run_population_vector_distances(ds, cfg):
    """Analysis section: population_vector_distances. Originally caban/main.py L5938-6367."""
    if not (cfg.plot_population_vector_distances):
        return
    # --- ds attributes ---
    PLOTS_DIR = ds.PLOTS_DIR
    TFC_B_B_1wk_crossreg = ds.TFC_B_B_1wk_crossreg
    TFC_cond = ds.TFC_cond
    Test_B = ds.Test_B
    Test_B_1wk = ds.Test_B_1wk
    add_significance_bars = ds.add_significance_bars
    mice_per_group = ds.mice_per_group
    # --- cfg switches ---
    plot_population_vector_distances = cfg.plot_population_vector_distances

    # ===== verbatim body from caban/main.py =====
    if plot_population_vector_distances:
        #
        # Population vector distance calculations
        #
        saver_PV_dist = Saver(parent_path=NPY_SAVE_PATH, subdirs=['PV_dist'])

        if saver_PV_dist.check_exists('PV_dist') and saver_PV_dist.check_exists('PV_data_df_dict'):
            PV_dist = saver_PV_dist.load('PV_dist') 
            PV_data_df_dict = saver_PV_dist.load('PV_data_df_dict')
            results = saver_PV_dist.load('PV_stats_results')

            plot_only_mode = True
            save_PV_dist = False
        else:
            PV_dist = dict()
            PV_data_df_dict = dict()
            results = dict()

            plot_only_mode = False
            save_PV_dist = True

        auto_close = True
        want_scatter = False
        #plot_only_mode = True
        use_median = True
        num_shuffles=1
        cov_type = 'lw'
        plot_bounds = True
        redo_stats = True

        ### BRANCH 1
        #PV_data_df_dict = dict()
        for binary_PV_type, dist_type in [ \
            ('non_binary', 'mahalanobis'), \
            #('binary', 'cosine'), \
            #('binary', 'jaccard'), \
            ]:

            if binary_PV_type == 'binary':
                want_binary_PV = True
                use_log = False
            else:
                want_binary_PV = False
                use_log = True
            if binary_PV_type not in PV_data_df_dict and save_PV_dist:
                PV_data_df_dict[binary_PV_type] = dict()

            ### BRANCH 2
            #for dist_type in ['cosine']: #['cosine', 'mahalanobis', 'jaccard']:
            if dist_type == 'cosine' or dist_type == 'jaccard':
                dist_lower_bound = 0.5
                dist_upper_bound = 0.95
            if dist_type == 'mahalanobis':
                dist_lower_bound = 0.1
                dist_upper_bound = 2
            if dist_type not in PV_data_df_dict[binary_PV_type] and save_PV_dist:
                PV_data_df_dict[binary_PV_type][dist_type] = dict() 

            ### BRANCH 3
            for shuffle_type in ['by_time']: # ['by_time', 'by_cells']:
                if shuffle_type not in PV_data_df_dict[binary_PV_type][dist_type] and save_PV_dist:
                    PV_data_df_dict[binary_PV_type][dist_type][shuffle_type] = dict()   

                ### BRANCH 4
                for PV_use_B_1wk_type in ['B', 'B_1wk']:
                    if PV_use_B_1wk_type == 'B_1wk':
                        PV_use_B_1wk = True
                    else:
                        PV_use_B_1wk = False

                    if not plot_only_mode:
                        timer = Timer()
                        timer.start()
                        PV_dist_curr = pop_vectors_dist(PLOTS_DIR, TFC_cond, 'TFC_cond', mice_per_group, \
                                auto_close=True, bin_width=1, spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg, \
                                sess_all_use=[TFC_cond, Test_B, Test_B_1wk], want_binary_PV=want_binary_PV, num_shuffles=num_shuffles, \
                                close_dist_bound_plots=True, PV_use_B_1wk=PV_use_B_1wk, dist_type=dist_type, shuffle_type=shuffle_type, \
                                dist_lower_bound=dist_lower_bound, dist_upper_bound=dist_upper_bound, \
                                cov_type=cov_type, plot_bounds=plot_bounds)
                        timer_str = timer.end()

                        if save_PV_dist:
                            assign_dict(PV_dist, [binary_PV_type, dist_type, shuffle_type, PV_use_B_1wk_type, 'PV_dist'], PV_dist_curr)
                            assign_dict(PV_dist, [binary_PV_type, dist_type, shuffle_type, PV_use_B_1wk_type, 'timer'], timer)

                    data_shuffle_norm = dict()
                    for shuffle_normalization in [True, False]:
                        PV_data_df = plot_group_PV_stats(PLOTS_DIR, PV_dist, auto_close=auto_close, use_log=use_log, binary_PV_type=binary_PV_type, \
                            shuffle_normalization=shuffle_normalization, dist_type=dist_type, shuffle_type=shuffle_type, PV_use_B_1wk_type=PV_use_B_1wk_type, \
                            want_scatter=want_scatter, plot_type='violinplot', use_median=use_median)
                        data_shuffle_norm[shuffle_normalization] = PV_data_df
                    if save_PV_dist:
                        PV_data_df_dict[binary_PV_type][dist_type][shuffle_type][PV_use_B_1wk_type] = data_shuffle_norm

        if save_PV_dist:
            if not plot_only_mode:
                saver_PV_dist.save(PV_dist, 'PV_dist')
            saver_PV_dist.save(PV_data_df_dict, 'PV_data_df_dict')

        # Do stats
        comp_mapping = {
            'pre': '$pre$',
            'CS' : '$CS$',
            'CS_first' : '$CS_{first}$',
            'CS_last' : '$CS_{last}$',
            'trace' : '$trace$',
            'trace_first' : '$trace_{first}$',
            'trace_last' : '$trace_{last}$',
            'US' : '$US$',
            'US_first' : '$US_{first}$',
            'US_last' : '$US_{last}$',
            'post_US' : '$post_{US}$',
            'post_US_first' : '$post_{US_{first}}$',
            'post_US_last' : '$post_{US_{last}}$',
            'pre_B' : '$pre_{B}$',
            'CS_B' : '$CS_{B}$',
            'CS_B_first' : '$CS_{B,first}$',
            'CS_B_last' : '$CS_{B,last}$',
            'trace_B' : '$trace_{B}$',
            'trace_B_first' : '$trace_{B,first}$',
            'trace_B_last' : '$trace_{B,}$'
        }    
        if redo_stats:
            results = {}
        for binary_PV_type, dist_type in [ \
            ('non_binary', 'mahalanobis'), \
            #('binary', 'cosine'), \
            #('binary', 'jaccard'), \
            ]:
            #dist_type = 'mahalanobis'
            #binary_PV_type = 'binary'

            # True means data normalized by shuffle ('real_raw' means raw data in calc_type column)
            df_B = PV_data_df_dict[binary_PV_type][dist_type]['by_time']['B'][True]
            df_B_1wk = PV_data_df_dict[binary_PV_type][dist_type]['by_time']['B_1wk'][True]

            df = pd.concat([df_B, df_B_1wk])

            df_B_real = df_B[(df_B['binary_PV_type']==binary_PV_type) & \
                (df_B['dist_type']==dist_type) & (df_B['shuffle_type']=='by_time') & \
                (df_B['PV_use_B_1wk_type']=='B') & (df_B['calc_type']=='real')].copy()
            df_B_1wk_real = df_B_1wk[(df_B_1wk['binary_PV_type']==binary_PV_type) & \
                (df_B_1wk['dist_type']==dist_type) & (df_B_1wk['shuffle_type']=='by_time') & \
                (df_B_1wk['PV_use_B_1wk_type']=='B_1wk') & (df_B_1wk['calc_type']=='real')].copy()

            # Ensure 'group' and 'mouse_index' are categorical
            #comparisons = [('hM3D', 'hM4D'), ('hM3D', 'mCherry'), ('hM4D', 'mCherry')]
            comparisons = [('hM3D', 'hM4D'), ('hM4D', 'mCherry'), ('hM3D', 'mCherry')]
            for df, df_str in zip([df_B_real, df_B_1wk_real], ['B', 'B_1wk']):
                df['group'] = df['group'].astype('category')
                df['mouse_index'] = df['mouse_index'].astype('category')
                comparison_keys = df['comparison_key'].unique()
                split_comparison_keys = [key.split('-') for key in comparison_keys]

                for pre_PV,post_PV in split_comparison_keys:
                    data = df.loc[df['comparison_key']==pre_PV+'-'+post_PV].copy()
                    scaler = StandardScaler()
                    #data['PV_dist'] = scaler.fit_transform(data[['PV_dist']])     
                    data.loc[:, 'PV_dist'] = scaler.fit_transform(data[['PV_dist']])
                    model = mixedlm("PV_dist ~ C(group, Treatment(reference='mCherry'))", data, groups=data["mouse_index"])
                    result = model.fit(method='powell')

                    if save_PV_dist or redo_stats:
                        assign_dict(results, [df_str, pre_PV, post_PV, 'result'], result)

                    # Check if any of the specified p-values are < 0.05
                    p_values_to_check = [
                        result.pvalues['C(group, Treatment(reference=\'mCherry\'))[T.hM3D]'],
                        result.pvalues['C(group, Treatment(reference=\'mCherry\'))[T.hM4D]'],
                        result.pvalues['Intercept']
                    ]
                    any_significant = any(p < 0.05 for p in p_values_to_check)
                    if save_PV_dist or redo_stats:
                        assign_dict(results, [binary_PV_type, dist_type, df_str, pre_PV, post_PV, 'result_any_significance'], any_significant)            

                    # Perform pairwise comparisons with Holm-Sidak correction
                    p_values_dict = {}
                    groups = data['group']
                    distances = data['PV_dist']
                    p_values_list = []
                    for group1, group2 in comparisons:
                        group1_data = distances[groups == group1]
                        group2_data = distances[groups == group2]
                    
                        # Perform t-test
                        t_stat, p_value, _ = sm.stats.ttest_ind(group1_data, group2_data)
                        p_values_list.append(p_value)
                
                    if save_PV_dist or redo_stats:
                        assign_dict(results, [binary_PV_type, dist_type, df_str, pre_PV, post_PV, 'p_values_posthoc'], p_values_list)

                    # Holm-Sidak correction
                    reject, pvals_corrected, _, _ = multipletests(p_values_list, method='holm-sidak')
                    if save_PV_dist or redo_stats:
                        assign_dict(results, [binary_PV_type, dist_type, df_str, pre_PV, post_PV, 'multipletests_correction', 'reject'], reject)
                        assign_dict(results, [binary_PV_type, dist_type, df_str, pre_PV, post_PV, 'multipletests_correction', 'pvals_corrected'], pvals_corrected)

                    #assign_dict(\
                    #    result.params['C(group, Treatment(reference=\'mCherry\'))[T.hM3D]'])
                    #assign_dict(results, [df_str, pre_PV, post_PV, 'lme', 'coef_hM3D'], result.params['C(group, Treatment(reference=\'mCherry\'))[T.hM3D]'])

        if save_PV_dist or redo_stats:
            saver_PV_dist.save(results, 'PV_stats_results')

        ##
        ## PLOT stats results
        ##
        #for binary_PV_type, dist_type in zip(['non_binary', 'binary'], ['mahalanobis', 'cosine']):
        for binary_PV_type, dist_type in [ \
            ('non_binary', 'mahalanobis'), \
            #('binary', 'cosine'), \
            #('binary', 'jaccard'), \
            ]:

            for df_str in ['B', 'B_1wk']:

                # Iterate through each pre_PV
                results_df = results[binary_PV_type][dist_type][df_str]
                for pre_PV in results_df.keys():
                    post_PV_keys = results_df[pre_PV].keys()
                
                    # Prepare the data for plotting
                    plot_data = []
                    for post_PV in post_PV_keys:
                        # Extract PV_dist from df_B
                        df_use = df_B if df_str == 'B' else df_B_1wk
                        df_filtered = df_use[(df_use['comparison_key'] == f'{pre_PV}-{post_PV}') & (df_use['calc_type'] == 'real')]
                        for group in ['hM3D', 'hM4D', 'mCherry']:
                            group_data = df_filtered[df_filtered['group'] == group]['PV_dist'].values
                            for value in group_data:
                                plot_data.append({
                                    'pre_PV': pre_PV,
                                    'post_PV': post_PV,
                                    'group': group,
                                    'PV_dist': value
                                })
                
                    plot_df = pd.DataFrame(plot_data)
                
                    # Create subplots with shared x-axis
                    fig, axes = plt.subplots(1, len(post_PV_keys), figsize=(15, 6), sharey=False, sharex=False)
                
                    if len(post_PV_keys) == 1:
                        axes = [axes]  # Ensure axes is iterable when there's only one subplot
                
                    # Create the violin plots
                    #palette = {'hM3D': 'red', 'hM4D': 'blue', 'mCherry': 'darkgrey'}
                    palette = {'hM3D' : my_colours['my_r'], 'hM4D': my_colours['my_b'], 'mCherry': my_colours['my_h']}
                    for ax, post_PV in zip(axes, post_PV_keys):
                        ax_vp = sns.violinplot(x='group', y='PV_dist', data=plot_df[plot_df['post_PV'] == post_PV], ax=ax, palette=palette)
                        ax.set_title(f'{comp_mapping[pre_PV]}-\n{comp_mapping[post_PV]}')#, loc='center')
                        #ax.set_xlabel('Group')
                        if ax == axes[0]:
                            ax.set_ylabel('PV_dist')
                        else:
                            ax.set_ylabel('')
                        ax.set_xlabel('')

                        # Add significance bars
                        if 'result_any_significance' in results_df[pre_PV][post_PV] and results_df[pre_PV][post_PV]:
                            max_val = plot_df[plot_df['post_PV'] == post_PV]['PV_dist'].max()
                            #max_val = max([np.max(coll.get_paths()[0].vertices[:, 1]) for coll in ax.collections])
                            P = list(results_df[pre_PV][post_PV]['multipletests_correction']['pvals_corrected'])
                            G = [[0,1],[1,2],[0, 2]]
                            sigstar(ax,G,P,props={'sigbar_sep_amt':0.05})
                        if 'multipletests_correction' in results[df_str][pre_PV][post_PV]:
                            comparisons = [(0, 1), (0, 2), (1, 2)]  # Assuming the order is hM3D, hM4D, mCherry
                            pvals_corrected = results[df_str][pre_PV][post_PV]['multipletests_correction']['pvals_corrected']
                            y_max = plot_df[plot_df['post_PV'] == post_PV]['PV_dist'].max()
                            add_significance_bars(ax, comparisons, pvals_corrected, y_max)

                        # Remove borders to merge subplots
                        #if ax != axes[0]:
                        #    ax.spines['left'].set_visible(False)
                        #    ax.yaxis.set_visible(False)
                        #if ax != axes[-1]:
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        #ax.tick_params(axis='x', labelrotation=45)
                        ax.set_xticklabels(['hM3D', 'hM4D', 'mCherry'], rotation=45)

                    max_ylim = max(ax.get_ylim()[1] for ax in axes)

                    # Set the ylim for all subplots to the largest y-value
                    for ax in axes:
                        if dist_type == 'cosine':
                            ax.set_ylim(-3.0, 3.0)
                        if dist_type == 'jaccard':
                            ax.set_ylim(0.8, 1.2)
                        #else:
                        ax.set_ylim(axes[0].get_ylim()[0], max_ylim)

                    # Adjust subplot spacing
                    plt.subplots_adjust(wspace=0.5)
                    #plt.subplot_tool()

                    #plt.suptitle(f'Violin Plots for {pre_PV}')
                    ###plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.show()

                    path_dir = os.path.join(PLOTS_DIR, 'PV_dist', 'PV_stats', f'{binary_PV_type}_{dist_type}', 'by_group')
                    os.makedirs(path_dir, exist_ok=True)
                    plt.savefig(os.path.join(path_dir, f'PV_dist_violin_{pre_PV}_{df_str}.png'), format='png', dpi=300)
                    if auto_close:
                        plt.close(fig)


        ##
        ## Across B, B_1wk
        ##
        ## Do stats first.

        df = pd.concat([df_B_real, df_B_1wk_real])
        df['group'] = df['group'].astype('category')
        df['mouse_index'] = df['mouse_index'].astype('category')

        comparison_keys = df['comparison_key'].unique()
        split_comparison_keys = [key.split('-') for key in comparison_keys]

        results_t = {}
        for pre_PV,post_PV in split_comparison_keys:
            data = df.loc[df['comparison_key']==pre_PV+'-'+post_PV].copy()
            scaler = StandardScaler()
            data.loc[:, 'PV_dist'] = scaler.fit_transform(data[['PV_dist']])

            model = mixedlm("PV_dist ~ C(group, Treatment(reference='mCherry')) * C(PV_use_B_1wk_type)", data, groups=data["mouse_index"])
            result = model.fit(method='powell')
            assign_dict(results_t, [pre_PV, post_PV, 'result'], result)

            any_significant = any(p < 0.05 for p in result.pvalues.values)
            assign_dict(results_t, [pre_PV, post_PV, 'result_any_significance'], any_significant)

            p_values_list = []
            for grp in ['hM3D', 'hM4D', 'mCherry']:
                time_pt1_data = data[ \
                    (data['group'] == grp) & \
                    (data['PV_use_B_1wk_type'] == 'B') & \
                    (data['calc_type'] == 'real')]['PV_dist']
                time_pt2_data = data[ \
                    (data['group'] == grp) & \
                    (data['PV_use_B_1wk_type'] == 'B_1wk') & \
                    (data['calc_type'] == 'real')]['PV_dist']
                t_stat, p_value, _ = sm.stats.ttest_ind(time_pt1_data, time_pt2_data)
                p_values_list.append(p_value)

                assign_dict(results_t, [pre_PV, post_PV, 'p_values_posthoc'], p_values_list)

            # Holm-Sidak correction
            reject, pvals_corrected, _, _ = multipletests(p_values_list, method='holm-sidak')
            assign_dict(results_t, [pre_PV, post_PV, 'multipletests_correction', 'reject'], reject)
            assign_dict(results_t, [pre_PV, post_PV, 'multipletests_correction', 'pvals_corrected'], pvals_corrected)

        ## HERE:

        # Determine group names based on the time point
        groups = ['hM3D', 'hM4D', 'mCherry']
        for pre_PV in results[df_str].keys():
            post_PV_keys = results[df_str][pre_PV].keys()
            plot_data = []

            for df_str in ['B', 'B_1wk']:
                group_time_labels = [f"{group}_{df_str}" for group in groups]

                # Prepare the data for plotting
                for post_PV in post_PV_keys:
                    df_use = df_B if df_str == 'B' else df_B_1wk
                    df_filtered = df_use[(df_use['comparison_key'] == f'{pre_PV}-{post_PV}') & (df_use['calc_type'] == 'real')]
                
                    # Populate plot_data with entries per group in group_time_labels
                    for group_time in group_time_labels:
                        group, time_suffix = group_time.split('_', 1)
                        group_data = df_filtered[df_filtered['group'] == group]['PV_dist'].values
                        for value in group_data:
                            plot_data.append({
                                'pre_PV': pre_PV,
                                'post_PV': post_PV,
                                'group_time': group_time,
                                'PV_dist': value
                            })
            
            plot_df = pd.DataFrame(plot_data)
        
            # Create subplots with shared x-axis
            fig, axes = plt.subplots(1, len(post_PV_keys), figsize=(15, 10), sharey=False, sharex=False)
        
            if len(post_PV_keys) == 1:
                axes = [axes]  # Ensure axes is iterable when there's only one subplot
        
            palette = {
                'hM3D_B': my_colours['my_r'], 'hM3D_B_1wk': my_colours['my_r_dark'],
                'hM4D_B': my_colours['my_b'], 'hM4D_B_1wk': my_colours['my_b_dark'],
                'mCherry_B': my_colours['my_h'], 'mCherry_B_1wk': my_colours['my_h_dark']
            }

            for ax, post_PV in zip(axes, post_PV_keys):
                ax_vp = sns.violinplot(x='group_time', y='PV_dist', data=plot_df[plot_df['post_PV'] == post_PV], ax=ax, palette=palette, order=palette.keys())
                ax.set_title(f'{comp_mapping[pre_PV]}-\n{comp_mapping[post_PV]}')
                if ax == axes[0]:
                    ax.set_ylabel('PV_dist')
                else:
                    ax.set_ylabel('')
                ax.set_xlabel('')

                # Set x-tick labels to match the six groups with time points
                #ax.set_xticklabels(group_time_labels, rotation=45)

                # Add significance bars if available
                if 'result_any_significance' in results[df_str][pre_PV][post_PV] and results[df_str][pre_PV][post_PV]['result_any_significance']:
                    max_val = plot_df[plot_df['post_PV'] == post_PV]['PV_dist'].max()
                    P = list(results[df_str][pre_PV][post_PV]['multipletests_correction']['pvals_corrected'])
                    G = [[0,1],[1,2],[0,2]]
                    sigstar(ax, G, P, props={'sigbar_sep_amt': 0.05})

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='x', rotation=45)

            max_ylim = max(ax.get_ylim()[1] for ax in axes)

            for ax in axes:
                ax.set_ylim(axes[0].get_ylim()[0], max_ylim)

            plt.subplots_adjust(wspace=0.5)
            plt.show()
        
            path_dir = os.path.join(PLOTS_DIR, 'PV_dist', 'PV_stats', 'by_group_time')
            os.makedirs(path_dir, exist_ok=True)
            plt.savefig(os.path.join(path_dir, f'PV_dist_violin_{pre_PV}_{df_str}.png'), format='png', dpi=300)


    # ===== end verbatim body =====


# ---------------------------------------------------------------------------
# Section: binned_activities  (caban/main.py L6368-6371)
# ---------------------------------------------------------------------------
def run_binned_activities(ds, cfg):
    """Analysis section: binned_activities. Originally caban/main.py L6368-6371."""
    if not (cfg.plot_binned_activities):
        return
    # --- cfg switches ---
    plot_binned_activities = cfg.plot_binned_activities

    # ===== verbatim body from caban/main.py =====
    if plot_binned_activities:
        print('here')
        pass

    # ===== end verbatim body =====




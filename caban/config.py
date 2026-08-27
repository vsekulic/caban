"""caban pipeline configuration.

Single dataclass holding every user-tunable switch that used to live as a
module-level constant in caban/main.py. Construct one in the notebook,
edit fields directly, and pass it into the pipeline / loader entry points.

Defaults mirror the values set at the top of caban/main.py.

Note: a few "config" values in caban/main.py are actually computed at
runtime from the loaded sessions (e.g. the data-driven continuity sigma
parameters resolved from velocity stats). Those are NOT stored here as
defaults — the dataclass only holds the *user-facing* knob
(`continuity_preset`). The pipeline resolves the runtime values from
that knob after the dataset is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Any


@dataclass
class PipelineConfig:
    # Top-level output directory for all plots. If None, set to a unique timestamped path in __post_init__.
    PLOTS_DIR: Optional[str] = None
    PLOTS_DIR_SINGULAR: bool = False  # If True, all plot types go in the same dir instead of by date/time
    # ------------------------------------------------------------------
    # Top-level run-control switches
    # ------------------------------------------------------------------
    DEBUG: bool = False
    DEVEL_SWITCH: bool = True  # When True, dev-mode skip of bulky plot blocks
    LOCAL_DATA: bool = True
    # Root directory for paper-ready plots. If None, loader resolves this to
    # <PLOTS_DIR>/paper_plots on the active run.
    PAPER_PLOTS: Optional[str] = None

    # ------------------------------------------------------------------
    # Behaviour / binning
    # ------------------------------------------------------------------
    behavcam_fps: int = 15
    # BIN_WIDTH = MINISCOPE_FPS * 10 by default — resolved in __post_init__
    # when bin_width_seconds is set instead.
    bin_width_frames: Optional[int] = None
    bin_width_seconds: float = 10.0
    BEHAVIOUR_TYPE: Optional[str] = "movement"  # 'movement' | 'immobility' | None

    # ------------------------------------------------------------------
    # Engram identity
    # ------------------------------------------------------------------
    ENGRAM_MODES: Tuple[str, ...] = ("permouse", "ctlthresh_z", "ctlthresh_p50")
    NUM_ENGRAM_PLOT_CELLS: int = 7
    # Engram-type for binned-rate engram panels.
    #   'encoding' = reference is TFC_cond (Mocle convention)
    #   'recall'   = reference is Test_B
    WHICH_ENGRAM_FOR_BINNED: str = "encoding"
    # Engram-type for the Population PCA pipeline.
    which_engram: str = "encoding"
    want_engram_pop_pca: bool = True
    # Frames per bin for the Population PCA temporal binning.
    PCA_FRAMES_PER_BIN: int = 1

    # ------------------------------------------------------------------
    # Coarse "section" plot switches (mirror caban/main.py)
    # ------------------------------------------------------------------
    plot_sample_cell: bool = False
    plot_sp_rates: bool = True
    plot_binned_sp_rates: bool = True
    plot_ROIs: bool = True
    plot_proportional_activities: bool = True
    plot_LT_firing_rate_changes: bool = True
    want_sample_traces_paper: bool = True
    plot_PSTH: bool = True
    plot_population_vectors: bool = True
    plot_population_vector_distances: bool = True
    perform_agglomerative_clustering: bool = True
    process_for_R: bool = False
    plot_pf_and_loc: bool = True
    plot_pf_raw_maps: bool = False  # skip heavy per-cell FM/3D plots
    plot_LT_pfs: bool = True
    plot_LT_decoding: bool = True
    plot_TFC_2D_decoding: bool = True
    plot_binned_activities: bool = True
    plot_epoch_pv_analysis: bool = True
    plot_cross_session_epoch_pv_analysis: bool = True
    plot_occupancy_analysis: bool = True
    plot_freeze_mobility_verification: bool = True
    enable_zone_crossreg_analysis: bool = True

    # Navigation-aware single cell analyses: decompose the whole-session spike rate /
    # activity by cell class (place vs non-place) and frame class (movement vs immobility),
    # and test the locomotion covariate directly. All output lands under
    # PLOTS_DIR/navigation_aware_single_cell/ and nothing existing is modified.
    plot_locomotion_comparison: bool = True
    plot_place_cell_properties: bool = True
    plot_place_cell_rates: bool = True
    plot_rate_vs_locomotion: bool = True
    # Speed-binned robustness check on the above: movement_only is a binary 2 cm/s threshold,
    # but hippocampal firing is graded with speed, so matched mean speed does not guarantee
    # matched speed distributions. speed_lag_frames probes deconvolution latency (S event
    # frames lag true spikes while the speed kernel is symmetric); 0 is the reported analysis.
    plot_speed_tuning: bool = True
    speed_lag_frames: int = 0

    # Single-unit response analyses (per-cell drill-downs beyond per-mouse means)
    plot_cell_activity_distributions: bool = True
    plot_event_locked_responsiveness: bool = True
    plot_epoch_modulation: bool = True
    # The hierarchical cell-level COMPANION to run_epoch_modulation, written into
    # <signal>/hierarchical_cells/ beside that analysis's own panels. On by default like every
    # other switch here, so a normal pipeline run produces it.
    # ** It is still the slowest thing in this section: ~8 min (YrA) + ~17 min (C). ** Almost all
    # of that is the single full cell-level statsmodels fit kept as an independent cross-check
    # (246 s and 807 s respectively); the randomization itself is a few minutes. Set this False
    # for a quick pass -- the mouse-level analysis is the paper-facing one and is complete
    # without it.
    epoch_modulation_hierarchical_cells: bool = True
    # The EVENT-PROXIMAL companion to run_epoch_modulation, written into
    # <signal>/event_proximal/ beside the hierarchical one. It recomputes the same modulation
    # index over a short window (3 s) from each event onset instead of over the full 20 s
    # epochs, to test whether the epoch means dilute a brief event-locked response. Same cells,
    # same trials, same standardization and same pre-tone baseline, so the two lanes are a
    # paired comparison. It carries its own hierarchical lane, so it roughly doubles this
    # analysis's runtime; set False for a quick pass.
    epoch_modulation_event_proximal: bool = True
    plot_epoch_sequence: bool = True
    plot_freezing_tuned_cells: bool = True
    plot_population_coupling: bool = True
    plot_sp_rates_lmm: bool = True

    # --- sp_rates_lmm analysis parameters ---
    # Number of Monte Carlo draws for the mouse-label permutation tests (see
    # caban.sp_rates_lmm.mouse_label_permutation_test). ~5.7M distinct 5/6/6 relabellings
    # exist for this dataset, so this is Monte Carlo, not exact enumeration.
    sp_rates_lmm_n_perm: int = 20000
    sp_rates_lmm_seed: int = 0

    # --- Single-unit response analysis parameters ---
    # Circular-shift shuffle count (higher = smoother z-scored p-values; slower).
    single_unit_n_shuffles: int = 1000
    single_unit_seed: int = 0
    # Signal for the activity-trace analyses: 'C' (denoised calcium, dense
    # transients) or 'S' (deconvolved spikes, rare).
    single_unit_signal: str = 'C'
    # Baseline window (seconds) immediately before each tone, used as the reference
    # for event-locked responsiveness. Shorter than the 35 s epoch-PV baseline so it
    # sits closer to stimulus onset.
    single_unit_baseline_s: float = 10.0
    # Event-locked response metrics to compute (each produces its own panels/tables):
    #   'peak'         : peak of the (smoothed) trace anywhere in the full epoch,
    #                    minus baseline — sensitive to phasic responses at any latency.
    #   'onset_window' : mean over the first single_unit_onset_window_s of the epoch,
    #                    minus baseline — targets the phasic onset response.
    #   'epoch_mean'   : mean over the whole epoch, minus baseline (dilutes phasic
    #                    responses; kept for completeness).
    single_unit_response_metrics: tuple = ('peak', 'onset_window')
    single_unit_onset_window_s: float = 5.0
    # Gaussian smoothing (seconds) applied before peak detection in the 'peak' metric.
    single_unit_peak_smooth_s: float = 1.0

    # ------------------------------------------------------------------
    # Decoder shuffle-control null model (shared by LT and TFC decoders)
    # ------------------------------------------------------------------
    enable_lt_shuffle_control: bool = True
    enable_tfc_shuffle_control: bool = False
    decoder_shuffle_type: str = "circular_time_shift"
    decoder_shuffle_n_repeats: int = 5
    decoder_shuffle_seed: int = 42

    # ------------------------------------------------------------------
    # 2D decoder (raw S) — shared across paradigms A-F
    # ------------------------------------------------------------------
    encoder_period: str = "pre-tone"  # 'pre-tone' | 'post-shock' | 'post-tone'
    use_z_score: str = "per-session"  # 'none' | 'per-session' | 'across-sessions' | 'optimize'
    TIME_BIN_FRAMES: int = 15
    N_SPATIAL_BINS: int = 20
    use_posterior_mean: bool = True
    decoder_type: str = "bayesian"     # 'bayesian' | 'ridge-regression'
    ridge_alpha: float = 1.0
    use_scoring_method: str = "both"   # 'euclidean' | 'rsquared' | 'both'
    pct_threshold_2D: float = 15.0
    run_fixed_effects_models_tfc_cross_vs_within: bool = True

    # ------------------------------------------------------------------
    # PF-based 2D decoder
    # ------------------------------------------------------------------
    use_pf_num: int = -1
    use_occupancy_fallback: bool = False
    place_cells_only: bool = True

    # ------------------------------------------------------------------
    # Continuity constraint (Bayesian decoders only).
    # The preset string is the single user-facing knob. The numeric
    # sigma parameters are resolved at runtime by the pipeline from the
    # preset (and, when preset='data-driven', from session velocity
    # stats). The values below are only used when preset='custom'.
    # ------------------------------------------------------------------
    continuity_preset: str = "data-driven"   # 'data-driven'|'conservative'|'balanced'|'aggressive'|'custom'
    use_continuity_constraint: bool = True
    continuity_sigma_k: float = 60.0
    continuity_speed_ref: float = 20.0
    continuity_exp: float = 1.0
    continuity_sigma_min: float = 20.0
    continuity_sigma_max: float = 60.0
    continuity_sigma_default: float = 30.0

    # ------------------------------------------------------------------
    # Hyperparameter optimization — raw-S 2D decoder
    # ------------------------------------------------------------------
    optimize_parameters: bool = False
    optimization_trials: int = 50
    optimization_seed: int = 42
    optimization_first_n_sec: float = 180.0
    optimization_target: str = "within_TFC"
    optimization_use_speed: bool = True
    optimization_min_speed: float = 2.0
    optimization_n_startup_trials: int = 12
    optimization_load_cached: bool = False
    optimization_param_set: str = "mCherry"
    optimization_jitter_lambda: float = 0.5

    # ------------------------------------------------------------------
    # Hyperparameter optimization — PF 2D decoder
    # ------------------------------------------------------------------
    optimize_pf_parameters: bool = False
    optimization_pf_trials: int = 50
    optimization_pf_seed: int = 42
    optimization_pf_first_n_sec: float = 180.0
    optimization_pf_use_speed: bool = True
    optimization_pf_min_speed: float = 2.0
    optimization_pf_n_startup_trials: int = 12
    optimization_pf_load_cached: bool = False
    optimization_pf_param_set: str = "pooled"
    optimization_pf_jitter_lambda: float = 0.5

    # ------------------------------------------------------------------
    # Population-curve sweep (decoder N-of-cells scaling)
    # ------------------------------------------------------------------
    enable_population_curve: bool = False
    population_curve_n_values: Optional[list] = None
    population_curve_n_values_max_shared: bool = True
    population_curve_per_pair_grid: bool = True
    population_curve_n_step: int = 20
    population_curve_quantile: float = 0.25
    population_curve_repeats: int = 100
    population_curve_seed: int = 42
    population_curve_metric: str = "median_err"
    manual_killswitch: bool = False
    killswitch_prompt_every: str = "N_step"
    population_curve_print_level: str = "medium"
    population_curve_suffix: str = "_population_curve"
    population_curve_load_cached: bool = True

    # ------------------------------------------------------------------
    # Abnormal-cell QC filter (applied at Minian load in get_CS_matrices).
    # Produces NEW filtered matrices S_filt/C_filt/YrA_filt (good cells only)
    # alongside the untouched originals S/C/YrA. Each sub-check has its own
    # enable flag; the master switch turns the whole pass on/off.
    # ------------------------------------------------------------------
    filter_abnormal_cells: bool = True          # master on/off
    cell_filter_skew_enabled: bool = True        # right-skewness check
    cell_filter_sparsity_enabled: bool = True    # hyperactivity / interneuron check
    cell_filter_plateau_enabled: bool = True     # plateau-artifact check
    cell_filter_silent_enabled: bool = True      # silent-cell (min-peaks) check
    cell_filter_sphericity_enabled: bool = False  # ROI shape (sphericity) check
    cell_filter_plot_diagnostics: bool = True    # render ROI + trace montages
    cell_filter_signal: str = "C"                # 'C' | 'S' | 'YrA' — signal driving the QC
    cell_filter_thre_skew: float = 1.5
    cell_filter_thre_plateau: float = 15.0       # seconds
    cell_filter_min_peaks: int = 3
    cell_filter_thre_sphericity: float = 0.5     # keep cells with roundness >= threshold

    # ------------------------------------------------------------------
    # Free-form extension slot for ad-hoc experimentation in the notebook
    # without having to subclass.
    # ------------------------------------------------------------------
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        import os
        from datetime import datetime
        if self.PLOTS_DIR is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.PLOTS_DIR = os.path.join("plots", ts)
        # Resolve bin_width_frames if not set explicitly.
        if self.bin_width_frames is None:
            # Lazy import so importing caban.config doesn't drag in the
            # whole pipeline. caban.utilities defines MINISCOPE_FPS.
            from caban.utilities import MINISCOPE_FPS
            self.bin_width_frames = int(MINISCOPE_FPS * self.bin_width_seconds)

        # Validate enumerated string fields up front so typos fail fast.
        _validate_choice("BEHAVIOUR_TYPE", self.BEHAVIOUR_TYPE,
                         (None, "movement", "immobility"))
        _validate_choice("WHICH_ENGRAM_FOR_BINNED", self.WHICH_ENGRAM_FOR_BINNED,
                         ("encoding", "recall"))
        _validate_choice("which_engram", self.which_engram,
                         ("encoding", "recall"))
        _validate_choice("encoder_period", self.encoder_period,
                         ("pre-tone", "post-shock", "post-tone"))
        _validate_choice("use_z_score", self.use_z_score,
                         ("none", "per-session", "across-sessions", "optimize"))
        _validate_choice("decoder_type", self.decoder_type,
                         ("bayesian", "ridge-regression", "ridge", "ridge_regression"))
        _validate_choice("use_scoring_method", self.use_scoring_method,
                         ("euclidean", "rsquared", "both"))
        _validate_choice("continuity_preset", self.continuity_preset,
                         ("data-driven", "conservative", "balanced", "aggressive", "custom"))
        _validate_choice("population_curve_metric", self.population_curve_metric,
                         ("median_err", "mean_err"))
        _validate_choice("population_curve_print_level", self.population_curve_print_level,
                         ("low", "medium", "high"))
        _validate_choice("cell_filter_signal", self.cell_filter_signal,
                         ("C", "S", "YrA"))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def export_globals(self) -> dict:
        """Return a dict suitable for ``globals().update(...)`` so that
        script-style code (e.g. ``caban/analyses.py``) which references
        bare names like ``plot_PSTH`` or ``DEVEL_SWITCH`` can find them.
        Extras keys are promoted to the top level."""
        out = {}
        for k, v in asdict(self).items():
            if k == "extras":
                # Promote extras keys to top-level so the notebook can stash
                # ad-hoc bare names there.
                out.update(v)
                continue
            out[k] = v
        return out

    def __repr__(self) -> str:  # compact repr that hides extras when empty
        parts = []
        for k, v in asdict(self).items():
            if k == "extras" and not v:
                continue
            parts.append(f"{k}={v!r}")
        return "PipelineConfig(\n  " + ",\n  ".join(parts) + "\n)"


def _validate_choice(name: str, value: Any, choices: Tuple[Any, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"PipelineConfig.{name}={value!r} is not one of {choices!r}"
        )

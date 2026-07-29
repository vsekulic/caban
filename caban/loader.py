"""caban dataset loader.

Reproduces the data-loading / engram / PF-backfill block of
``caban/main.py`` (originally lines 1-1773) inside a single callable
``load_all_mice(cfg)`` that returns a ``SimpleNamespace`` exposing every
top-level name the analysis blocks expect. The notebook is expected to
do::

    cfg = PipelineConfig()
    ds  = load_all_mice(cfg)
    globals().update(vars(ds))         # bring TFC_cond, Test_B, ... into scope

so that subsequent analysis cells reference bare names (``TFC_cond``,
``Test_B``, ``engram_id``, ``mappings_all_TFC_cond``, ...) without any
prefix.

This module is intentionally thin: it preserves the load / engram
build-up logic verbatim. Refactoring into smaller helpers / removing
dead code is left for follow-up patches once output parity is verified.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from collections import Counter
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import numpy as np

# Project imports — same star-imports used by caban/main.py so every
# top-level name (CrossRegMapping, TraceFearCondSession, msg_start, ...) is
# in scope here.
# Bootstrap caban.decoder first to break the caban.analysis <-> decoder
# circular import (mirrors caban/main.py lines 7-8).
import importlib
import caban.decoder as _caban_decoder_bootstrap
importlib.reload(_caban_decoder_bootstrap)
from caban.utilities import *  # noqa: F401,F403
from caban.sessions import *   # noqa: F401,F403
from caban.analysis import *   # noqa: F401,F403

from caban.engram import (
    build_engram_identity,
    project_engram_to_session,
    ENGRAM_REFERENCE,
)

from caban.roi import (
    plot_abnormal_cell_filter_roi_montage,
    plot_abnormal_cell_filter_trace_montage,
    emit_abnormal_cell_filter_montages,
    emit_abnormal_cell_filter_montages_one_mouse,
    ensure_abnormal_cell_filter_dirs,
)

from caban.config import PipelineConfig


# ---------------------------------------------------------------------------
# Module-level helpers that analysis cells expect to be available as
# bare top-level names. They are re-exposed via the returned Dataset.
# ---------------------------------------------------------------------------
def _extract_ts(dpath_val: str) -> str:
    """Extract the ``HH_MM_SS`` timestamp prefix from a session dpath."""
    return os.path.basename(dpath_val).split('-')[0]


def plot_interneuron_cutoff(PLOTS_DIR, session, mice_per_group, auto_close=True):
    """Backward-compatible wrapper for interneuron cutoff plotting."""
    return generate_interneuron_cutoff(PLOTS_DIR, session, mice_per_group, auto_close=auto_close)  # noqa: F405


def add_significance_bars(ax, comparisons, p_values, y_max,
                          step_fraction=0.08, bar_height_fraction=0.015):
    """Draw pairwise significance bars for categorical x positions."""
    if ax is None or comparisons is None or p_values is None:
        return
    if len(comparisons) == 0 or len(p_values) == 0:
        return

    y_low, y_high = ax.get_ylim()
    y_range = max(y_high - y_low, 1e-9)
    step = y_range * step_fraction
    bar_height = y_range * bar_height_fraction
    base_y = max(float(y_max), y_low)

    def _p_to_stars(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return 'ns'

    n_drawn = 0
    for i, (pair, p_val) in enumerate(zip(comparisons, p_values)):
        if pair is None or len(pair) != 2:
            continue
        if p_val is None:
            continue
        try:
            if np.isnan(float(p_val)):
                continue
        except Exception:
            continue

        x1, x2 = pair
        y = base_y + step * (i + 1)
        ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y],
                color='black', linewidth=1.2)
        ax.text((x1 + x2) / 2.0, y + bar_height, _p_to_stars(float(p_val)),
                ha='center', va='bottom', fontsize=10)
        n_drawn += 1

    if n_drawn > 0:
        top_needed = base_y + step * (len(comparisons) + 2)
        ax.set_ylim(y_low, max(y_high, top_needed))


def _apply_test_b_tone_onset_override(sess, tone_onsets_frames):
    """Apply a manual Test_B tone-onset correction and rebuild dependent windows."""
    if not tone_onsets_frames:
        return

    # Test_B tones are fixed to 20 seconds.
    tone_duration_frames = int(20 * MINISCOPE_FPS)

    sess.tone_onsets = [int(x) for x in tone_onsets_frames]
    exp_stop = int(sess.miniscope_exp_fnum[sess.stop_idx])
    sess.tone_offsets = [min(int(on) + tone_duration_frames, exp_stop) for on in sess.tone_onsets]

    sess.post_tone_onsets = []
    sess.post_tone_offsets = []
    sess.tone_post_tone_onsets = []
    sess.tone_post_tone_offsets = []
    for i in range(len(sess.tone_onsets)):
        sess.post_tone_onsets.append(sess.tone_offsets[i])
        sess.tone_post_tone_onsets.append(sess.tone_onsets[i])
        if i == len(sess.tone_onsets) - 1:
            sess.post_tone_offsets.append(exp_stop)
            sess.tone_post_tone_offsets.append(exp_stop)
        else:
            sess.post_tone_offsets.append(sess.tone_onsets[i + 1])
            sess.tone_post_tone_offsets.append(sess.tone_onsets[i + 1])

    # Rebuild labels/bounds to keep downstream epoch helpers consistent.
    sess.period_bounds = []
    sess.find_period_bounds()


# ---------------------------------------------------------------------------
def _resolve_plots_dir(plots_dir: Optional[str], plots_dir_singular: bool = False) -> str:
    """Mirror caban/main.py's host-aware PLOTS_DIR choice."""
    if plots_dir is not None:
        return plots_dir
    if MAIN_DRIVE == '' and os.path.isdir('/Users/vsekulic/data/vsekulic/OF_test'):  # noqa: F405
        base = '/Users/vsekulic/data/vsekulic/OF_test/plots'
    else:
        base = MAIN_DRIVE + '\\data\\vsekulic\\OF_test\\plots'  # noqa: F405
    if plots_dir_singular:
        return os.path.join(base, 'CURRENT')
    return os.path.join(base, datetime.now().strftime('%Y-%m-%d %H_%M_%S'))


def _resolve_paper_dir(
    paper_dir: Optional[str],
    *,
    cfg: Optional[PipelineConfig] = None,
    plots_dir: Optional[str] = None,
) -> str:
    if paper_dir is not None:
        return paper_dir

    if cfg is not None and cfg.PAPER_PLOTS is not None:
        return cfg.PAPER_PLOTS

    if plots_dir is None:
        raise RuntimeError("Cannot resolve PAPER_DIR: plots_dir is None.")
    return os.path.join(plots_dir, '0-PAPER_PLOTS')


# ===========================================================================
def _deep_size(obj, seen=None):
    """Recursive byte size of an object: numpy arrays count nbytes, containers recurse."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if hasattr(obj, '__dict__'):
        return sum(_deep_size(v, seen) for v in obj.__dict__.values())
    if isinstance(obj, dict):
        return sum(_deep_size(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_deep_size(v, seen) for v in obj)
    return sys.getsizeof(obj)


def report_ds_memory(ds, top_n: int = 15, min_mb: float = 1.0) -> None:
    """Print total deep size of ``ds`` and its top-N largest top-level attributes."""
    total = _deep_size(ds)
    print(f'ds total: {total / 1e9:.2f} GB')
    sizes = [(_deep_size(v), k) for k, v in ds.__dict__.items()]
    sizes.sort(reverse=True)
    print(f'top {top_n} attributes (>= {min_mb:.0f} MB):')
    threshold = min_mb * 1e6
    for sz, name in sizes[:top_n]:
        if sz < threshold:
            break
        print(f'  {sz / 1e9:6.2f} GB  {name}')


def report_cache_compression(ds, top_n: int = 10) -> None:
    """Print a compact summary of what the cache pruned or can lazily restore."""
    print('cache compression report:')
    print(f'  format version: {getattr(ds, "cache_format_version", "unknown")}')

    session_counts = Counter()
    prunable_counts = Counter()
    missing_counts = Counter()
    tracked_fields = [
        'C_full', 'S_full', 'YrA_full',
        'S_spikes', 'S_peakval',
        'S_mov', 'S_imm', 'C_mov', 'C_imm', 'YrA_mov', 'YrA_imm',
    ]
    tracked_present = Counter()
    tracked_missing = Counter()
    largest_sessions = []
    seen = set()

    def _iter_session_objects(path, value):
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if hasattr(value, '_cache_pruned_fields') and hasattr(value, '__dict__'):
            yield path, value
            return

        if isinstance(value, dict):
            for key, child in value.items():
                yield from _iter_session_objects(f'{path}.{key}', child)
            return

        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                yield from _iter_session_objects(f'{path}[{idx}]', child)
            return

    for attr_name, attr_value in ds.__dict__.items():
        for obj_path, obj_value in _iter_session_objects(attr_name, attr_value):
            session_counts[type(obj_value).__name__] += 1
            prunable_fields = sorted(obj_value._cache_pruned_fields)
            missing_fields = [
                field_name for field_name in obj_value._cache_pruned_fields
                if field_name not in obj_value.__dict__
            ]
            for field_name in prunable_fields:
                prunable_counts[field_name] += 1
            for field_name in missing_fields:
                missing_counts[field_name] += 1
            for field_name in tracked_fields:
                if field_name in obj_value.__dict__:
                    tracked_present[field_name] += 1
                else:
                    tracked_missing[field_name] += 1
            largest_sessions.append((obj_path, _deep_size(obj_value), len(missing_fields), type(obj_value).__name__))

    if session_counts:
        session_summary = ', '.join(f'{count} {class_name}' for class_name, count in sorted(session_counts.items()))
        print(f'  session objects: {session_summary}')
    else:
        print('  session objects: none found')

    if prunable_counts:
        print(f'  prunable fields (top {top_n}):')
        for field_name, count in prunable_counts.most_common(top_n):
            print(f'    {field_name}: {count}')

    if missing_counts:
        print(f'  missing in cache (top {top_n}):')
        for field_name, count in missing_counts.most_common(top_n):
            print(f'    {field_name}: {count}')
    else:
        print('  missing in cache: none detected')

    if tracked_present or tracked_missing:
        print('  tracked heavy fields (present/missing):')
        for field_name in tracked_fields:
            present_n = tracked_present.get(field_name, 0)
            missing_n = tracked_missing.get(field_name, 0)
            print(f'    {field_name}: {present_n}/{missing_n}')

    if largest_sessions:
        print(f'  largest cached sessions (top {top_n}):')
        largest_sessions.sort(key=lambda item: item[1], reverse=True)
        for attr_name, size_bytes, pruned_count, class_name in largest_sessions[:top_n]:
            print(f'    {size_bytes / 1e9:6.2f} GB  {attr_name} [{class_name}, pruned={pruned_count}]')


CACHE_FORMAT_VERSION = 2


def _compact_ndarray(arr: np.ndarray) -> np.ndarray:
    """Downcast dense numeric arrays to smaller dtypes when it is safe to do so."""
    if arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.int64 and arr.size > 0:
        int32_info = np.iinfo(np.int32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_min >= int32_info.min and arr_max <= int32_info.max:
            return arr.astype(np.int32, copy=False)
    if arr.dtype == np.uint64 and arr.size > 0:
        uint32_info = np.iinfo(np.uint32)
        if arr.max() <= uint32_info.max:
            return arr.astype(np.uint32, copy=False)
    return arr


def _compact_cache_value(value, seen=None):
    """Recursively compact numpy-heavy objects in-place for smaller cache payloads."""
    if seen is None:
        seen = {}

    value_id = id(value)
    if value_id in seen:
        return seen[value_id]

    if isinstance(value, np.ndarray):
        compact = _compact_ndarray(value)
        seen[value_id] = compact
        return compact

    if value.__class__.__module__.startswith('scipy.sparse'):
        seen[value_id] = value
        return value

    if isinstance(value, dict):
        seen[value_id] = value
        for key in list(value.keys()):
            value[key] = _compact_cache_value(value[key], seen)
        return value

    if isinstance(value, list):
        seen[value_id] = value
        for idx in range(len(value)):
            value[idx] = _compact_cache_value(value[idx], seen)
        return value

    if isinstance(value, tuple):
        compact = tuple(_compact_cache_value(item, seen) for item in value)
        seen[value_id] = compact
        return compact

    if isinstance(value, set):
        compact = {_compact_cache_value(item, seen) for item in value}
        seen[value_id] = compact
        return compact

    if isinstance(value, SimpleNamespace) or hasattr(value, '__dict__'):
        seen[value_id] = value
        for attr_name, attr_value in vars(value).items():
            setattr(value, attr_name, _compact_cache_value(attr_value, seen))
        return value

    seen[value_id] = value
    return value


def _load_cache_payload(cache_path: str):
    with open(cache_path, 'rb') as f:
        payload = pickle.load(f)

    is_versioned = isinstance(payload, dict) and payload.get('cache_format_version') == CACHE_FORMAT_VERSION
    if is_versioned:
        payload = payload['ds']

    return _compact_cache_value(payload), is_versioned


def _dump_cache_payload(cache_path: str, ds) -> None:
    payload = {
        'cache_format_version': CACHE_FORMAT_VERSION,
        'ds': _compact_cache_value(ds),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ===========================================================================
def load_all_mice(
    cfg: Optional[PipelineConfig] = None,
    *,
    plots_dir: Optional[str] = None,
    paper_dir: Optional[str] = None,
    use_cache: bool = True,
    cache_path: Optional[str] = None,
    report_memory: bool = True,
) -> SimpleNamespace:
    """Build the loaded dataset state and return it.

    Parameters
    ----------
    cfg
        Pipeline configuration. If None, a default ``PipelineConfig()`` is used.
    plots_dir, paper_dir
        Optional overrides. If neither ``paper_dir`` nor
        ``cfg.PAPER_PLOTS`` is set, paper plots default to
        ``<PLOTS_DIR>/paper_plots``.
    use_cache
        If True (default), load ``ds`` from ``cache_path`` when it exists and
        otherwise write a fresh build to ``cache_path``. If False, always
        rebuild and never touch the pickle.
    cache_path
        Path to the compact ``ds`` cache. Defaults to ``<NPY_SAVE_PATH>/ds_cache.pkl``.
    report_memory
        If True (default), print a deep-size breakdown of ``ds`` after load.

    Returns
    -------
    SimpleNamespace
        Namespace whose attributes are every top-level name that
        ``caban/main.py`` produced after the per-mouse load loop, the
        unified engram pass, and the PF-merged backfill. Pass through
        ``globals().update(vars(ds))`` to land them in the notebook.
    """
    if cfg is None:
        cfg = PipelineConfig()

    if cache_path is None:
        cache_path = os.path.join(NPY_SAVE_PATH, 'ds_cache.pkl')  # noqa: F405

    print(f'cache path : {cache_path}')
    print(f'  exists   : {os.path.exists(cache_path)}')
    if os.path.exists(cache_path):
        print(f'  size     : {os.path.getsize(cache_path) / 1e9:.2f} GB')

    if use_cache and os.path.exists(cache_path):
        print('loading ds from pickle (skipping fresh build)...')
        ds, is_versioned = _load_cache_payload(cache_path)
        ds.cfg = cfg
        # Always use cfg.PLOTS_DIR as the single source of truth
        cfg.PLOTS_DIR = _resolve_plots_dir(plots_dir, plots_dir_singular=cfg.PLOTS_DIR_SINGULAR)
        cfg.PAPER_DIR = _resolve_paper_dir(paper_dir, cfg=cfg, plots_dir=cfg.PLOTS_DIR)
        print(f'*** PLOTS_DIR: {cfg.PLOTS_DIR}')
        print(f'*** PAPER_DIR: {cfg.PAPER_DIR}')
        ds.cache_format_version = CACHE_FORMAT_VERSION
        if not hasattr(ds, 'NPY_SAVE_PATH'):
            ds.NPY_SAVE_PATH = NPY_SAVE_PATH  # noqa: F405
        if not is_versioned:
            print('rewriting legacy cache into compact format...')
            _dump_cache_payload(cache_path, ds)
        if report_memory:
            print()
            report_ds_memory(ds)
            report_cache_compression(ds)
        return ds

    # Resolve PLOTS_DIR/PAPER_DIR up front (mirror the cache-load path above) so
    # a fresh build writes to the same singular CURRENT location instead of the
    # timestamped relative fallback that PipelineConfig.__post_init__ assigns
    # when PLOTS_DIR is left at its default of None.
    cfg.PLOTS_DIR = _resolve_plots_dir(plots_dir, plots_dir_singular=cfg.PLOTS_DIR_SINGULAR)
    cfg.PAPER_DIR = _resolve_paper_dir(paper_dir, cfg=cfg, plots_dir=cfg.PLOTS_DIR)
    print(f'*** PLOTS_DIR: {cfg.PLOTS_DIR}')
    print(f'*** PAPER_DIR: {cfg.PAPER_DIR}')

    ds = _build_dataset(cfg, plots_dir=plots_dir, paper_dir=paper_dir)
    ds = _compact_cache_value(ds)
    ds.cache_format_version = CACHE_FORMAT_VERSION

    if use_cache:
        print(f'writing ds_cache.pkl -> {cache_path}')
        _dump_cache_payload(cache_path, ds)

    if report_memory:
        print()
        report_ds_memory(ds)
        report_cache_compression(ds)

    return ds


# ===========================================================================
def _build_dataset(
    cfg: PipelineConfig,
    *,
    plots_dir: Optional[str] = None,
    paper_dir: Optional[str] = None,
) -> SimpleNamespace:
    """Internal: build a fresh ``ds`` namespace from scratch (no cache logic)."""

    DEBUG = cfg.DEBUG
    LOCAL_DATA = cfg.LOCAL_DATA
    BIN_WIDTH = cfg.bin_width_frames
    BEHAVIOUR_TYPE = cfg.BEHAVIOUR_TYPE
    ENGRAM_MODES = tuple(cfg.ENGRAM_MODES)
    plot_sample_cell = cfg.plot_sample_cell
    plot_ROIs = cfg.plot_ROIs

    # Bundle the abnormal-cell QC knobs into a single dict threaded into every
    # session constructor (consumed by BehaviourSession._run_abnormal_cell_filter).
    cell_filter_params = {
        'filter_abnormal_cells': cfg.filter_abnormal_cells,
        'cell_filter_skew_enabled': cfg.cell_filter_skew_enabled,
        'cell_filter_sparsity_enabled': cfg.cell_filter_sparsity_enabled,
        'cell_filter_plateau_enabled': cfg.cell_filter_plateau_enabled,
        'cell_filter_silent_enabled': cfg.cell_filter_silent_enabled,
        'cell_filter_sphericity_enabled': cfg.cell_filter_sphericity_enabled,
        'cell_filter_signal': cfg.cell_filter_signal,
        'cell_filter_thre_skew': cfg.cell_filter_thre_skew,
        'cell_filter_thre_plateau': cfg.cell_filter_thre_plateau,
        'cell_filter_min_peaks': cfg.cell_filter_min_peaks,
        'cell_filter_thre_sphericity': cfg.cell_filter_thre_sphericity,
    }
    cell_filter_plot_diagnostics = cfg.cell_filter_plot_diagnostics
    data_dir = 'minian_crossreg1'
    crossreg_file_TFC_cond = 'mappings_crossreg_1.csv'
    crossreg_file_4 = 'mappings_crossreg_4.csv'
    crossreg_file_6 = 'mappings_crossreg_6.csv'
    crossreg_file_7 = 'mappings_crossreg_7.csv'

    # Always use cfg.PLOTS_DIR as the single source of truth
    PLOTS_DIR = cfg.PLOTS_DIR
    PAPER_DIR = _resolve_paper_dir(paper_dir, cfg=cfg, plots_dir=cfg.PLOTS_DIR)

    # -----------------------------------------------------------------------
    # Mouse metadata (verbatim from caban/main.py L126-840)
    # -----------------------------------------------------------------------
    mouse_groups = {
        'G05': 'hM3D', 'G06': 'hM4D', 'G07': 'hM4D', 'G08': 'mCherry',
        'G09': 'mCherry', 'G10': 'hM3D', 'G11': 'hM3D', 'G12': 'mCherry',
        'G13': 'mCherry', 'G14': 'hM4D', 'G15': 'hM4D', 'G16': 'mCherry',
        'G17': 'mCherry', 'G18': 'hM3D', 'G19': 'hM3D', 'G20': 'hM4D',
        'G21': 'hM4D',
    }
    if DEBUG:
        mouse_groups = {'G05': 'hM3D'}
    mouse_list = list(mouse_groups.keys())

    mice_per_group: dict = {}
    mice_per_group_Test_B_B_1wk: dict = {}
    for mouse, group in mouse_groups.items():
        mice_per_group.setdefault(group, []).append(mouse)
        mice_per_group_Test_B_B_1wk.setdefault(group, [])
        if mouse not in ['G07', 'G15']:
            mice_per_group_Test_B_B_1wk[group].append(mouse)

    mice_skip_LT: list = []

    # Drive mount points
    DRIVE_1a = 'I'
    DRIVE_1b = 'H'
    mouse_drive_prefix = {
        'G05': DRIVE_1a, 'G06': DRIVE_1a, 'G07': DRIVE_1a, 'G08': DRIVE_1a,
        'G09': DRIVE_1a, 'G10': DRIVE_1a, 'G11': DRIVE_1a,
        'G12': DRIVE_1b, 'G13': DRIVE_1b, 'G14': DRIVE_1b, 'G15': DRIVE_1b,
        'G16': DRIVE_1b, 'G17': DRIVE_1b, 'G18': DRIVE_1b, 'G19': DRIVE_1b,
        'G20': DRIVE_1b, 'G21': DRIVE_1b,
    }
    mouse_path_prefix: dict = {}
    for mouse in mouse_drive_prefix:
        if LOCAL_DATA:
            mouse_path_prefix[mouse] = mouse_drive_prefix[mouse] + ":\\\\data\\vsekulic\\OF_test\\"
        else:
            mouse_path_prefix[mouse] = "\\\\cbp-db.bnf.brain.riken.jp\\vsekulic\\data\\vsekulic\\OF_test\\"

    dpath_mouse = {
        'G05': 'G05-ST637_hM3D', 'G06': 'G06-ST688_hM4D', 'G07': 'G07-ST689_hM4D',
        'G08': 'G08-ST701_mCherry', 'G09': 'G09-ST702_mCherry', 'G10': 'G10-ST703_hM3D',
        'G11': 'G11-ST705_hM3D', 'G12': 'G12-ST709-mCherry', 'G13': 'G13-ST710-mCherry',
        'G14': 'G14-ST719-hM4D', 'G15': 'G15-ST721-hM4D', 'G16': 'G16-ST731-mCherry',
        'G17': 'G17-ST741-mCherry', 'G18': 'G18-ST734-hM3D', 'G19': 'G19-ST735-hM3D',
        'G20': 'G20-ST760-hM4D', 'G21': 'G21-ST762-hM4D',
    }

    # ---- TFC_cond -------------------------------------------------------
    dpath_TFC_cond_day = {
        'G05': '2021_08_30-TFC_cond', 'G06': '2021_10_18-TFC_cond',
        'G07': '2021_10_18-TFC_cond', 'G08': '2021_11_08-TFC_cond',
        'G09': '2021_11_08-TFC_cond', 'G10': '2021_11_23-TFC_cond',
        'G11': '2021_11_23-TFC_cond', 'G12': '2022_01_03-TFC_cond',
        'G13': '2022_01_03-TFC_cond', 'G14': '2022_01_11-TFC_cond',
        'G15': '2022_01_11-TFC_cond', 'G16': '2022_01_24-TFC_cond',
        'G17': '2022_01_24-TFC_cond', 'G18': '2022_02_07-TFC_cond',
        'G19': '2022_02_07-TFC_cond', 'G20': '2022_03_22-TFC_cond',
        'G21': '2022_03_22-TFC_cond',
    }
    dpath_TFC_cond = {
        'G05': '18_22_57-TFC_cond', 'G06': '11_42_00-TFC_cond',
        'G07': '15_14_32-TFC_cond', 'G08': '15_13_23-TFC_cond',
        'G09': '18_54_05-TFC_cond', 'G10': '16_32_14-TFC_cond',
        'G11': '19_24_52-TFC_cond', 'G12': '15_12_38-TFC_cond',
        'G13': '17_13_06-TFC_cond', 'G14': '16_00_53-TFC_cond',
        'G15': '18_10_56-TFC_cond', 'G16': '16_06_44-TFC_cond',
        'G17': '18_26_30-TFC_cond', 'G18': '15_29_25-TFC_cond',
        'G19': '17_35_25-TFC_cond', 'G20': '14_37_19-TFC_cond',
        'G21': '17_39_30-TFC_cond',
    }
    TFC_cond_exp_frames = {
        'G05': [75, 19550], 'G06': [135, 19617], 'G07': [83, 19564],
        'G08': [67, 19549], 'G09': [51, 14440], 'G10': [72, 19553],
        'G11': [48, 19530], 'G12': [43, 19524], 'G13': [44, 19527],
        'G14': [676, 20159], 'G15': [37, 19519], 'G16': [43, 19525],
        'G17': [38, 19519], 'G18': [65, 19548], 'G19': [80, 19562],
        'G20': [37, 19519], 'G21': [34, 19516],
    }
    dpath_TFC_cond_LT1 = {
        'G05': '16_47_02-LT1', 'G06': '10_14_41-LT1', 'G07': '13_25_42-LT1b',
        'G08': '13_22_42-LT1', 'G09': '16_53_07-LT1', 'G10': '14_51_37-LT1',
        'G11': '17_51_25-LT1', 'G12': '14_11_51-LT1', 'G13': '16_10_53-LT1',
        'G14': '14_53_32-LT1', 'G15': '17_05_25-LT1', 'G16': '14_59_15-LT1',
        'G17': '17_38_16-LT1', 'G18': '14_19_24-LT1', 'G19': '16_25_38-LT1',
        'G20': '13_31_20-LT1', 'G21': '16_32_01-LT1',
    }
    dpath_TFC_cond_LT2 = {
        'G05': '17_44_51-LT2', 'G06': '11_05_51-LT2', 'G07': '14_35_40-LT2',
        'G08': '14_43_03-LT2', 'G09': '18_08_37-LT2', 'G10': '15_53_29-LT2',
        'G11': '18_55_26-LT2', 'G12': '14_52_29-LT2', 'G13': '16_53_52-LT2',
        'G14': '15_39_23-LT2', 'G15': '17_51_17-LT2', 'G16': '15_47_20-LT2',
        'G17': '19_02_10-LT2', 'G18': '15_10_08-LT2', 'G19': '17_14_57-LT2',
        'G20': '14_16_55-LT2', 'G21': '17_19_40-LT2',
    }
    LT1_exp_frames = {
        'G05': [0, -1], 'G06': [0, -1], 'G07': [0, -1], 'G08': [442, -1],
        'G09': [314, -1], 'G10': [249, -1], 'G11': [240, -1], 'G12': [370, -1],
        'G13': [304, -1], 'G14': [275, -1], 'G15': [220, -1], 'G16': [582, -1],
        'G17': [200, -1], 'G18': [0, -1], 'G19': [288, -1], 'G20': [246, -1],
        'G21': [188, -1],
    }
    LT2_exp_frames = {
        'G05': [0, -1], 'G06': [0, -1], 'G07': [0, -1], 'G08': [291, -1],
        'G09': [252, -1], 'G10': [227, -1], 'G11': [248, -1], 'G12': [430, -1],
        'G13': [229, -1], 'G14': [352, -1], 'G15': [261, -1], 'G16': [200, -1],
        'G17': [326, -1], 'G18': [264, -1], 'G19': [244, -1], 'G20': [226, -1],
        'G21': [201, -1],
    }

    # ---- Test_B / Test_B_1wk -------------------------------------------
    dpath_Test_B_day = {
        'G05': '2021_09_01-TFC_test_B', 'G06': '2021_10_20-TFC_test_B',
        'G07': '2021_10_20-TFC_test_B', 'G08': '2021_11_10-TFC_test_B',
        'G09': '2021_11_10-TFC_test_B', 'G10': '2021_11_25-TFC_test_B',
        'G11': '2021_11_25-TFC_test_B', 'G12': '2022_01_05-TFC_test_B',
        'G13': '2022_01_05-TFC_test_B', 'G14': '2022_01_13-TFC_test_B',
        'G15': '2022_01_13-TFC_test_B', 'G16': '2022_01_26-TFC_test_B',
        'G17': '2022_01_26-TFC_test_B', 'G18': '2022_02_09-TFC_test_B',
        'G19': '2022_02_09-TFC_test_B', 'G20': '2022_03_24-TFC_test_B',
        'G21': '2022_03_24-TFC_test_B',
    }
    dpath_Test_B = {
        'G05': '16_20_07-TFC_test_B', 'G06': '13_32_28-TFC_test_B',
        'G07': '15_49_59-TFC_test_B', 'G08': '14_45_42-TFC_test_B',
        'G09': '17_12_18-TFC_test_B', 'G10': '15_28_34-TFC_test_B',
        'G11': '17_17_33-TFC_test_B', 'G12': '14_28_46-TFC_test_B',
        'G13': '15_53_23-TFC_test_B', 'G14': '15_05_26-TFC_test_B',
        'G15': '16_19_13-TFC_test_B', 'G16': '14_47_09-TFC_test_B',
        'G17': '16_14_47-TFC_test_B', 'G18': '15_25_42-TFC_test_B',
        'G19': '16_44_35-TFC_test_B', 'G20': '13_21_06-TFC_test_B',
        'G21': '14_35_02-TFC_test_B',
    }
    Test_B_exp_frames = {
        'G05': [56, 13537], 'G06': [751, 14231], 'G07': [],
        'G08': [44, 13524], 'G09': [86, 13565], 'G10': [129, 13610],
        'G11': [58, 13539], 'G12': [35, 13516], 'G13': [43, 13523],
        'G14': [39, 13520], 'G15': [37, 13518], 'G16': [37, 13515],
        'G17': [70, 13551], 'G18': [32, 13512], 'G19': [42, 13523],
        'G20': [36, 13520], 'G21': [43, 13524],
    }
    dpath_Test_B_LT1 = {'G05': '12_45_49-LT1'}
    Test_B_LT1_exp_frames = {'G05': [0, -1]}

    dpath_Test_B_1wk_day = {
        'G05': '2021_09_06-TFC_test_B_1wk', 'G06': '2021_10_25-TFC_test_B_1wk',
        'G07': '2021_10_25-TFC_test_B_1wk', 'G08': '2021_11_15-TFC_test_B_1wk',
        'G09': '2021_11_15-TFC_test_B_1wk', 'G10': '2021_11_30-TFC_test_B_1wk',
        'G11': '2021_11_30-TFC_test_B_1wk', 'G12': '2022_01_10-TFC_test_B_1wk',
        'G13': '2022_01_10-TFC_test_B_1wk', 'G14': '2022_01_18-TFC_test_B_1wk',
        'G15': '', 'G16': '2022_01_31-TFC_test_B_1wk',
        'G17': '2022_01_31-TFC_test_B_1wk', 'G18': '2022_02_15-TFC_test_B_1wk',
        'G19': '2022_02_15-TFC_test_B_1wk', 'G20': '2022_03_29-TFC_test_B_1wk',
        'G21': '2022_03_29-TFC_test_B_1wk',
    }
    dpath_Test_B_1wk = {
        'G05': '16_51_41-TFC_test_B_1wk', 'G06': '12_23_30-TFC_test_B_1wk',
        'G07': '15_19_51-TFC_test_B_1wk', 'G08': '14_58_20-TFC_test_B_1wk',
        'G09': '17_15_18-TFC_test_B_1wk', 'G10': '16_07_16-TFC_test_B_1wk',
        'G11': '17_57_20-TFC_test_B_1wk', 'G12': '15_31_10-TFC_test_B_1wk',
        'G13': '16_50_10-TFC_test_B_1wk', 'G14': '15_10_50-TFC_test_B_1wk',
        'G15': '', 'G16': '14_08_02-TFC_test_B_1wk',
        'G17': '16_06_22-TFC_test_B_1wk', 'G18': '13_49_23-TFC_test_B_1wk',
        'G19': '15_18_44-TFC_test_B_1wk', 'G20': '15_06_19-TFC_test_B_1wk',
        'G21': '16_30_54-TFC_test_B_1wk',
    }
    Test_B_1wk_exp_frames = {
        'G05': [56, 13537], 'G06': [50, 13531], 'G07': [49, 13529],
        'G08': [467, 13947], 'G09': [52, 13533], 'G10': [53, 13533],
        'G11': [49, 13529], 'G12': [40, 13520], 'G13': [43, 13524],
        'G14': [40, 13520], 'G15': [], 'G16': [42, 13523],
        'G17': [39, 13520], 'G18': [35, 13516], 'G19': [36, 13517],
        'G20': [52, 13532], 'G21': [45, 13525],
    }
    dpath_Test_B_1wk_LT1 = {'G05': '15_58_07-LT1'}
    Test_B_1wk_LT1_exp_frames = {'G05': [0, -1]}

    test_unit_id = {'G05': 22, 'G06': 413}
    period_overrides = {'G09': [0, 1, 2, 3]}
    test_b_tone_onset_overrides = {'G21': [3600, 8400, 11793]}

    # ---- Test_A / Test_A_1wk -------------------------------------------
    dpath_Test_A_day = {
        'G05': '2021_09_03-TFC_test_A', 'G06': '2021_10_22-TFC_test_A',
        'G07': '2021_10_22-TFC_test_A', 'G08': '2021_11_12-TFC_test_A',
        'G09': '2021_11_12-TFC_test_A', 'G10': '2021_11_27-TFC_test_A',
        'G11': '2021_11_27-TFC_test_A', 'G12': '2022_01_07-TFC_test_A',
        'G13': '2022_01_07-TFC_test_A', 'G14': '2022_01_15-TFC_test-A',
        'G15': '2022_01_15-TFC_test_A', 'G16': '2022_01_28-TFC_test_A',
        'G17': '2022_01_28-TFC_test_A', 'G18': '2022_02_12-TFC_test_A',
        'G19': '2022_02_12-TFC_test_A', 'G20': '2022_03_26-TFC_test_A',
        'G21': '2022_03_26-TFC_test_A',
    }
    dpath_Test_A = {
        'G05': '16_34_42-TFC_test_A', 'G06': '12_58_00-TFC_test_A',
        'G07': '15_26_13-TFC_test_A', 'G08': '14_38_53-TFC_test_A',
        'G09': '16_33_31-TFC_test_A', 'G10': '15_34_25-TFC_test_A',
        'G11': '17_00_04-TFC_test_A', 'G12': '16_08_09-TFC_test_A',
        'G13': '17_09_27-TFC_test_A', 'G14': '17_39_32-TFC_test_A',
        'G15': '18_45_26-TFC_test_A', 'G16': '14_01_26-TFC_test_A',
        'G17': '15_02_16-TFC_test_A', 'G18': '12_19_08-TFC_test_A',
        'G19': '13_32_18-TFC_test_A', 'G20': '16_33_21-TFC_test_A',
        'G21': '17_50_05-TFC_test_A',
    }
    Test_A_exp_frames = {
        'G05': [74, 4558], 'G06': [94, 4576], 'G07': [59, 4543],
        'G08': [38, 4521], 'G09': [55, 4539], 'G10': [43, 4527],
        'G11': [43, 4527], 'G12': [37, 4520], 'G13': [39, 4523],
        'G14': [36, 4520], 'G15': [44, 4528], 'G16': [50, 4533],
        'G17': [53, 4537], 'G18': [36, 4520], 'G19': [32, 4516],
        'G20': [35, 4519], 'G21': [38, 4522],
    }
    dpath_Test_A_LT1 = {
        'G05': '15_29_57-LT1', 'G06': '12_18_46-LT1', 'G07': '14_41_53-LT1',
        'G08': '13_35_08-LT1', 'G09': '15_51_11-LT1', 'G10': '15_08_43-LT1',
        'G11': '16_32_25-LT1', 'G12': '15_43_53-LT1', 'G13': '16_49_21-LT1',
        'G14': '17_14_14-LT1', 'G15': '18_22_53-LT1', 'G16': '13_38_27-LT1',
        'G17': '14_42_59-LT1', 'G18': '11_56_41-LT1', 'G19': '13_11_53-LT1',
        'G20': '16_08_55-LT1', 'G21': '17_25_15-LT1',
    }
    Test_A_LT1_exp_frames = {
        'G05': [0, -1], 'G06': [216, -1], 'G07': [332, -1], 'G08': [519, -1],
        'G09': [260, -1], 'G10': [298, -1], 'G11': [316, -1], 'G12': [266, -1],
        'G13': [225, -1], 'G14': [0, -1], 'G15': [256, -1], 'G16': [295, -1],
        'G17': [246, -1], 'G18': [272, -1], 'G19': [260, -1], 'G20': [437, -1],
        'G21': [274, -1],
    }
    dpath_Test_A_1wk_day = {
        'G05': '2021_09_09-TFC_test_A_1wk', 'G06': '2021_10_27-TFC_test_A_1wk',
        'G07': '2021_10_27-TFC_test_A_1wk', 'G08': '2021_11_17-TFC_test_A_1wk',
        'G09': '2021_11_17-TFC_test_A_1wk', 'G10': '2021_12_02-TFC_test_A_1wk',
        'G11': '2021_12_02-TFC_test_A_1wk', 'G12': '2022_01_12-TFC_test_A_1wk',
        'G13': '2022_01_12-TFC_test_A_1wk', 'G14': '2022_01_21-TFC_test_A_1wk',
        'G15': '', 'G16': '2022_02_02-TFC_test_A_1wk',
        'G17': '2022_02_02-TFC_test_A_1wk', 'G18': '2022_02_16-TFC_test_A_1wk',
        'G19': '2022_02_16-TFC_test_A_1wk', 'G20': '2022_03_31-TFC_test_A_1wk',
        'G21': '2022_03_31-TFC_test_A_1wk',
    }
    dpath_Test_A_1wk = {
        'G05': '17_32_00-TFC_test_A_1wk', 'G06': '14_45_19-TFC_test_A_1wk',
        'G07': '16_49_38-TFC_test_A_1wk', 'G08': '15_24_18-TFC_test_A_1wk',
        'G09': '17_47_11-TFC_test_A_1wk', 'G10': '13_59_52-TFC_test_A_1wk',
        'G11': '15_20_57-TFC_test_A_1wk', 'G12': '14_03_29-TFC_test_A_1wk',
        'G13': '15_07_47-TFC_test_A_1wk', 'G14': '16_13_09-TFC_test_A_1wk',
        'G15': '', 'G16': '14_49_27-TFC_test_A_1wk',
        'G17': '16_00_11-TFC_test_A_1wk', 'G18': '14_47_39-TFC_test_A_1wk',
        'G19': '16_10_03-TFC_test_A_1wk', 'G20': '13_45_31-TFC_test_A_1wk',
        'G21': '15_12_05-TFC_test_A_1wk',
    }
    Test_A_1wk_exp_frames = {
        'G05': [56, 4540], 'G06': [135, 4619], 'G07': [526, 5010],
        'G08': [44, 4527], 'G09': [0, 4534], 'G10': [35, 4518],
        'G11': [40, 4523], 'G12': [38, 4521], 'G13': [34, 4518],
        'G14': [37, 4520], 'G15': [], 'G16': [37, 4520],
        'G17': [54, 4539], 'G18': [32, 4516], 'G19': [36, 4520],
        'G20': [46, 4530], 'G21': [47, 4531],
    }
    dpath_Test_A_1wk_LT1 = {
        'G05': '16_11_07-LT1', 'G06': '14_01_05-LT1', 'G07': '16_06_09-LT1',
        'G08': '14_39_00-LT1', 'G09': '16_55_57-LT1', 'G10': '13_34_50-LT1',
        'G11': '14_57_08-LT1', 'G12': '13_43_14-LT1', 'G13': '14_48_49-LT1',
        'G14': '15_48_10-LT1', 'G15': '', 'G16': '14_24_38-LT1',
        'G17': '15_35_03-LT1', 'G18': '14_26_40-LT1', 'G19': '15_38_53-LT1',
        'G20': '13_15_54-LT1', 'G21': '14_49_58-LT1',
    }
    Test_A_1wk_LT1_exp_frames = {
        'G05': [0, -1], 'G06': [485, -1], 'G07': [0, -1], 'G08': [264, -1],
        'G09': [220, -1], 'G10': [301, -1], 'G11': [329, -1], 'G12': [256, -1],
        'G13': [242, -1], 'G14': [297, -1], 'G15': [], 'G16': [201, -1],
        'G17': [250, -1], 'G18': [225, -1], 'G19': [335, -1], 'G20': [203, -1],
        'G21': [180, -1],
    }

    # -----------------------------------------------------------------------
    # Crossreg group definitions
    # -----------------------------------------------------------------------
    TFC_cond_crossreg_groups: dict = {}
    for _m in mouse_groups:
        _grp = {}
        _lt1 = dpath_TFC_cond_LT1.get(_m, '')
        if _lt1:
            _grp['LT1'] = _extract_ts(_lt1)
        _lt2 = dpath_TFC_cond_LT2.get(_m, '')
        if _lt2:
            _grp['LT2'] = _extract_ts(_lt2)
        _tfc = dpath_TFC_cond.get(_m, '')
        if _tfc:
            _grp['TFC_cond'] = _extract_ts(_tfc)
        TFC_cond_crossreg_groups[_m] = _grp

    TFC_B_B_1wk_crossreg_groups: dict = {}
    for _m in mouse_groups:
        _grp = {}
        _tfc = dpath_TFC_cond.get(_m, '')
        if _tfc:
            _grp['TFC_cond'] = _extract_ts(_tfc)
        _tb = dpath_Test_B.get(_m, '')
        if _tb:
            _grp['Test_B'] = _extract_ts(_tb)
        _tb1 = dpath_Test_B_1wk.get(_m, '')
        if _tb1:
            _grp['Test_B_1wk'] = _extract_ts(_tb1)
        TFC_B_B_1wk_crossreg_groups[_m] = _grp

    TFC_AB_48hr_1wk_crossreg_groups: dict = {}
    for _m in TFC_B_B_1wk_crossreg_groups:
        TFC_AB_48hr_1wk_crossreg_groups[_m] = dict(TFC_B_B_1wk_crossreg_groups[_m])
        _ta = dpath_Test_A.get(_m, '')
        if _ta:
            TFC_AB_48hr_1wk_crossreg_groups[_m]['Test_A'] = _extract_ts(_ta)
        _ta1 = dpath_Test_A_1wk.get(_m, '')
        if _ta1:
            TFC_AB_48hr_1wk_crossreg_groups[_m]['Test_A_1wk'] = _extract_ts(_ta1)

    TFC_A_A_1wk_crossreg_groups: dict = {}
    for _m in TFC_B_B_1wk_crossreg_groups:
        _tfc_ts = TFC_B_B_1wk_crossreg_groups[_m]['TFC_cond']
        _grp = {'TFC_cond': _tfc_ts}
        _ta = dpath_Test_A.get(_m, '')
        if _ta:
            _grp['Test_A'] = _extract_ts(_ta)
        _ta1 = dpath_Test_A_1wk.get(_m, '')
        if _ta1:
            _grp['Test_A_1wk'] = _extract_ts(_ta1)
        TFC_A_A_1wk_crossreg_groups[_m] = _grp

    # -----------------------------------------------------------------------
    # Save paths
    # -----------------------------------------------------------------------
    TFC_cond_savepath = os.path.join(NPY_SAVE_PATH, 'TFC_cond')          # noqa: F405
    Test_A_savepath = os.path.join(NPY_SAVE_PATH, 'Test_A')              # noqa: F405
    Test_A_1wk_savepath = os.path.join(NPY_SAVE_PATH, 'Test_A_1wk')      # noqa: F405
    Test_B_savepath = os.path.join(NPY_SAVE_PATH, 'Test_B')              # noqa: F405
    Test_B_1wk_savepath = os.path.join(NPY_SAVE_PATH, 'Test_B_1wk')      # noqa: F405
    for _p in (TFC_cond_savepath, Test_A_savepath, Test_A_1wk_savepath,
               Test_B_savepath, Test_B_1wk_savepath, PLOTS_DIR):
        os.makedirs(_p, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build absolute paths
    # -----------------------------------------------------------------------
    dpath_TFC_cond_crossreg_file: dict = {}
    dpath_TFC_B_B_1wk_crossreg_file: dict = {}
    dpath_TFC_AB_48hr_1wk_crossreg_file: dict = {}
    dpath_TFC_A_A_1wk_crossreg_file: dict = {}
    for mouse in mouse_list:
        dpath_TFC_cond_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_TFC_cond_day[mouse])
        dpath_TFC_cond[mouse] = os.path.join(dpath_TFC_cond_day[mouse], dpath_TFC_cond[mouse])
        dpath_TFC_cond_LT1[mouse] = os.path.join(dpath_TFC_cond_day[mouse], dpath_TFC_cond_LT1[mouse])
        dpath_TFC_cond_LT2[mouse] = os.path.join(dpath_TFC_cond_day[mouse], dpath_TFC_cond_LT2[mouse])
        dpath_TFC_cond_crossreg_file[mouse] = os.path.join(dpath_TFC_cond_day[mouse], crossreg_file_TFC_cond)

        dpath_Test_A_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_Test_A_day[mouse])
        dpath_Test_A[mouse] = os.path.join(dpath_Test_A_day[mouse], dpath_Test_A[mouse])
        dpath_Test_A_1wk_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_Test_A_1wk_day[mouse])
        dpath_Test_A_1wk[mouse] = os.path.join(dpath_Test_A_1wk_day[mouse], dpath_Test_A_1wk[mouse])

        dpath_Test_B_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_Test_B_day[mouse])
        dpath_Test_B[mouse] = os.path.join(dpath_Test_B_day[mouse], dpath_Test_B[mouse])
        dpath_Test_B_1wk_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_Test_B_1wk_day[mouse])
        dpath_Test_B_1wk[mouse] = os.path.join(dpath_Test_B_1wk_day[mouse], dpath_Test_B_1wk[mouse])

        dpath_TFC_B_B_1wk_crossreg_file[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], crossreg_file_4)
        dpath_TFC_AB_48hr_1wk_crossreg_file[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], crossreg_file_6)
        dpath_TFC_A_A_1wk_crossreg_file[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], crossreg_file_7)

    # -----------------------------------------------------------------------
    # Session and crossreg slots
    # -----------------------------------------------------------------------
    TFC_cond: dict = {}
    TFC_cond_LT1: dict = {}
    TFC_cond_LT2: dict = {}
    TFC_cond_crossreg: dict = {}
    Test_A: dict = {}
    Test_A_1wk: dict = {}
    Test_B: dict = {}
    Test_B_1wk: dict = {}
    TFC_B_B_1wk_crossreg: dict = {}
    TFC_AB_48hr_1wk_crossreg: dict = {}
    TFC_A_A_1wk_crossreg: dict = {}

    # ---- Mapping label constants ---------------------------------------
    mapping_FULL = 'full'
    mapping_LT1_LT2_TFC_cond = 'LT1+LT2+TFC_cond'
    mapping_TFC_cond = 'TFC_cond'
    mapping_LT2_TFC_cond = 'LT2+TFC_cond'
    mapping_LT1_TFC_cond = 'LT1+TFC_cond'
    mappings_all_TFC_cond = [
        mapping_LT1_LT2_TFC_cond, mapping_TFC_cond,
        mapping_LT2_TFC_cond, mapping_LT1_TFC_cond, mapping_FULL,
    ]

    mapping_TFC_cond_Test_B_Test_B_1wk = 'TFC_cond+Test_B+Test_B_1wk'
    mapping_TFC_cond_Test_B = 'TFC_cond+Test_B'
    mapping_TFC_cond_Test_B_1wk = 'TFC_cond+Test_B_1wk'
    mapping_Test_B_Test_B_1wk = 'Test_B+Test_B_1wk'

    mapping_TFC_cond_Test_A_Test_A_1wk = 'TFC_cond+Test_A+Test_A_1wk'
    mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk = (
        'TFC_cond+Test_A+Test_A_1wk+Test_B+Test_B_1wk'
    )

    mappings_all_Test_B = [
        mapping_TFC_cond_Test_B_Test_B_1wk, mapping_TFC_cond_Test_B,
        mapping_Test_B_Test_B_1wk, mapping_FULL,
    ]
    mappings_all_Test_B_G15 = [mapping_TFC_cond_Test_B, mapping_FULL]
    mappings_all_Test_B_1wk = [
        mapping_TFC_cond_Test_B_Test_B_1wk, mapping_TFC_cond_Test_B_1wk,
        mapping_Test_B_Test_B_1wk, mapping_FULL,
    ]
    mappings_all_Test_B_1wk_G07 = [mapping_TFC_cond_Test_B_1wk, mapping_FULL]

    mapping_LT1 = 'LT1'
    mapping_LT2 = 'LT2'
    mapping_LT1_LT2 = 'LT1+LT2'
    mappings_all_LT1 = [
        mapping_LT1, mapping_LT1_TFC_cond, mapping_LT1_LT2_TFC_cond,
        mapping_LT1_LT2, mapping_FULL,
    ]
    mappings_all_LT2 = [
        mapping_LT2, mapping_LT2_TFC_cond, mapping_LT1_LT2_TFC_cond,
        mapping_LT1_LT2, mapping_FULL,
    ]

    mappings_all_Test_A = [mapping_FULL]
    mappings_all_Test_A_1wk = [mapping_FULL]

    # -----------------------------------------------------------------------
    # Per-mapping accumulator dicts
    # -----------------------------------------------------------------------
    # TFC_cond
    tone_sp_rates_mapping: dict = {}
    shock_sp_rates_mapping: dict = {}
    post_shock_sp_rates_mapping: dict = {}
    tone_activity_mapping: dict = {}
    shock_activity_mapping: dict = {}
    post_shock_activity_mapping: dict = {}
    post_tone_activity_mapping: dict = {}
    TFC_cond_binned_sp_rates_mapping: dict = {}
    TFC_cond_binned_activity_mapping: dict = {}
    TFC_cond_binned_activity_mapping_engram = {em: {} for em in ENGRAM_MODES}
    TFC_cond_ROI_mappings: dict = {}
    TFC_cond_ROI_mappings_peakval: dict = {}
    for mapping in mappings_all_TFC_cond:
        tone_sp_rates_mapping[mapping] = {}
        shock_sp_rates_mapping[mapping] = {}
        post_shock_sp_rates_mapping[mapping] = {}
        tone_activity_mapping[mapping] = {}
        shock_activity_mapping[mapping] = {}
        post_shock_activity_mapping[mapping] = {}
        post_tone_activity_mapping[mapping] = {}
        TFC_cond_binned_sp_rates_mapping[mapping] = {}
        TFC_cond_binned_activity_mapping[mapping] = {}
        for _em in ENGRAM_MODES:
            TFC_cond_binned_activity_mapping_engram[_em][mapping] = {}
        TFC_cond_ROI_mappings[mapping] = {}
        TFC_cond_ROI_mappings_peakval[mapping] = {}

    # Test A / A_1wk
    Test_A_exp_sp_rates_mapping: dict = {}
    Test_A_exp_activity_mapping: dict = {}
    Test_A_1wk_exp_sp_rates_mapping: dict = {}
    Test_A_1wk_exp_activity_mapping: dict = {}
    Test_A_binned_sp_rates_mapping: dict = {}
    Test_A_binned_activity_mapping: dict = {}
    Test_A_binned_activity_mapping_engram = {em: {} for em in ENGRAM_MODES}
    Test_A_ROI_mappings: dict = {}
    Test_A_ROI_mappings_peakval: dict = {}
    Test_A_1wk_binned_sp_rates_mapping: dict = {}
    Test_A_1wk_binned_activity_mapping: dict = {}
    Test_A_1wk_binned_activity_mapping_engram = {em: {} for em in ENGRAM_MODES}
    Test_A_1wk_ROI_mappings: dict = {}
    Test_A_1wk_ROI_mappings_peakval: dict = {}
    for mapping in mappings_all_Test_A:
        Test_A_exp_sp_rates_mapping[mapping] = {}
        Test_A_exp_activity_mapping[mapping] = {}
        Test_A_binned_sp_rates_mapping[mapping] = {}
        Test_A_binned_activity_mapping[mapping] = {}
        for _em in ENGRAM_MODES:
            Test_A_binned_activity_mapping_engram[_em][mapping] = {}
        Test_A_ROI_mappings[mapping] = {}
        Test_A_ROI_mappings_peakval[mapping] = {}
    for mapping in mappings_all_Test_A_1wk:
        Test_A_1wk_exp_sp_rates_mapping[mapping] = {}
        Test_A_1wk_exp_activity_mapping[mapping] = {}
        Test_A_1wk_binned_sp_rates_mapping[mapping] = {}
        Test_A_1wk_binned_activity_mapping[mapping] = {}
        for _em in ENGRAM_MODES:
            Test_A_1wk_binned_activity_mapping_engram[_em][mapping] = {}
        Test_A_1wk_ROI_mappings[mapping] = {}
        Test_A_1wk_ROI_mappings_peakval[mapping] = {}

    # Test B / B_1wk
    Test_B_tone_sp_rates_mapping: dict = {}
    Test_B_post_tone_sp_rates_mapping: dict = {}
    Test_B_tone_post_tone_sp_rates_mapping: dict = {}
    Test_B_tone_activity_mapping: dict = {}
    Test_B_post_tone_activity_mapping: dict = {}
    Test_B_tone_post_tone_activity_mapping: dict = {}
    Test_B_1wk_tone_sp_rates_mapping: dict = {}
    Test_B_1wk_post_tone_sp_rates_mapping: dict = {}
    Test_B_1wk_tone_post_tone_sp_rates_mapping: dict = {}
    Test_B_1wk_tone_activity_mapping: dict = {}
    Test_B_1wk_post_tone_activity_mapping: dict = {}
    Test_B_1wk_tone_post_tone_activity_mapping: dict = {}
    Test_B_binned_sp_rates_mapping: dict = {}
    Test_B_binned_activity_mapping: dict = {}
    Test_B_binned_activity_mapping_engram = {em: {} for em in ENGRAM_MODES}
    Test_B_ROI_mappings: dict = {}
    Test_B_ROI_mappings_peakval: dict = {}
    Test_B_1wk_binned_sp_rates_mapping: dict = {}
    Test_B_1wk_binned_activity_mapping: dict = {}
    Test_B_1wk_binned_activity_mapping_engram = {em: {} for em in ENGRAM_MODES}
    Test_B_1wk_ROI_mappings: dict = {}
    Test_B_1wk_ROI_mappings_peakval: dict = {}
    for mapping in mappings_all_Test_B:
        Test_B_tone_sp_rates_mapping[mapping] = {}
        Test_B_post_tone_sp_rates_mapping[mapping] = {}
        Test_B_tone_post_tone_sp_rates_mapping[mapping] = {}
        Test_B_tone_activity_mapping[mapping] = {}
        Test_B_post_tone_activity_mapping[mapping] = {}
        Test_B_tone_post_tone_activity_mapping[mapping] = {}
        Test_B_binned_sp_rates_mapping[mapping] = {}
        Test_B_binned_activity_mapping[mapping] = {}
        for _em in ENGRAM_MODES:
            Test_B_binned_activity_mapping_engram[_em][mapping] = {}
        Test_B_ROI_mappings[mapping] = {}
        Test_B_ROI_mappings_peakval[mapping] = {}
    for mapping in mappings_all_Test_B_1wk:
        Test_B_1wk_tone_sp_rates_mapping[mapping] = {}
        Test_B_1wk_post_tone_sp_rates_mapping[mapping] = {}
        Test_B_1wk_tone_post_tone_sp_rates_mapping[mapping] = {}
        Test_B_1wk_tone_activity_mapping[mapping] = {}
        Test_B_1wk_post_tone_activity_mapping[mapping] = {}
        Test_B_1wk_tone_post_tone_activity_mapping[mapping] = {}
        Test_B_1wk_binned_sp_rates_mapping[mapping] = {}
        Test_B_1wk_binned_activity_mapping[mapping] = {}
        for _em in ENGRAM_MODES:
            Test_B_1wk_binned_activity_mapping_engram[_em][mapping] = {}
        Test_B_1wk_ROI_mappings[mapping] = {}
        Test_B_1wk_ROI_mappings_peakval[mapping] = {}

    # LT1 / LT2
    LT1_exp_sp_rates_mapping: dict = {}
    LT2_exp_sp_rates_mapping: dict = {}
    LT1_exp_activity_mapping: dict = {}
    LT2_exp_activity_mapping: dict = {}
    LT1_ROI_mappings: dict = {}
    LT2_ROI_mappings: dict = {}
    LT1_ROI_mappings_peakval: dict = {}
    LT2_ROI_mappings_peakval: dict = {}
    for mapping in mappings_all_LT1:
        LT1_exp_sp_rates_mapping[mapping] = {}
        LT1_exp_activity_mapping[mapping] = {}
        LT1_ROI_mappings[mapping] = {}
        LT1_ROI_mappings_peakval[mapping] = {}
    for mapping in mappings_all_LT2:
        LT2_exp_sp_rates_mapping[mapping] = {}
        LT2_exp_activity_mapping[mapping] = {}
        LT2_ROI_mappings[mapping] = {}
        LT2_ROI_mappings_peakval[mapping] = {}

    # -----------------------------------------------------------------------
    # Per-mouse loading loop
    # -----------------------------------------------------------------------
    _load_errors: list = []

    # Create the abnormal-cell QC montage dirs up front (so they exist from the
    # very start of the build) and emit each mouse's montages incrementally as
    # it finishes inside the loop below.
    _cf_emit = cfg.filter_abnormal_cells and cell_filter_plot_diagnostics
    if _cf_emit:
        _cf_trace_dir, _cf_roi_dir = ensure_abnormal_cell_filter_dirs(PLOTS_DIR)

    for mouse in mouse_list:
        msg_start('*** Processing mouse ' + mouse + '\n')  # noqa: F405
        try:
            # Crossreg objects
            TFC_cond_crossreg[mouse] = CrossRegMapping(  # noqa: F405
                mouse, dpath_TFC_cond_crossreg_file[mouse], crossreg_type=1,
                groups_mappings=TFC_cond_crossreg_groups[mouse],
                savepath=TFC_cond_savepath,
            )
            TFC_B_B_1wk_crossreg[mouse] = CrossRegMapping(  # noqa: F405
                mouse, dpath_TFC_B_B_1wk_crossreg_file[mouse], crossreg_type=4,
                groups_mappings=TFC_B_B_1wk_crossreg_groups[mouse],
                savepath=TFC_cond_savepath,
            )
            TFC_AB_48hr_1wk_crossreg[mouse] = CrossRegMapping(  # noqa: F405
                mouse, dpath_TFC_AB_48hr_1wk_crossreg_file[mouse], crossreg_type=6,
                groups_mappings=TFC_AB_48hr_1wk_crossreg_groups[mouse],
                savepath=TFC_cond_savepath,
            )
            TFC_A_A_1wk_crossreg[mouse] = CrossRegMapping(  # noqa: F405
                mouse, dpath_TFC_A_A_1wk_crossreg_file[mouse], crossreg_type=7,
                groups_mappings=TFC_A_A_1wk_crossreg_groups[mouse],
                savepath=TFC_cond_savepath,
            )

            # TFC cond + LT
            period_override = period_overrides.get(mouse, [])
            TFC_cond[mouse] = TraceFearCondSession(  # noqa: F405
                mouse, dpath_TFC_cond[mouse],
                session_bounds=TFC_cond_exp_frames[mouse],
                period_override=period_override,
                plot_sample_cell=plot_sample_cell, data_dir=data_dir,
                crossreg=TFC_cond_crossreg[mouse], savepath=TFC_cond_savepath,
                behaviour_type=BEHAVIOUR_TYPE, behaviour_condition=mouse_groups[mouse],
                cell_filter_params=cell_filter_params,
            )

            wanted_behaviour = None if mouse in mice_skip_LT else BEHAVIOUR_TYPE
            TFC_cond_LT1[mouse] = LinearTrackSession(  # noqa: F405
                mouse, dpath_TFC_cond_LT1[mouse],
                session_bounds=LT1_exp_frames[mouse], LT_type='LT1',
                data_dir=data_dir, crossreg=TFC_cond_crossreg[mouse],
                savepath=TFC_cond_savepath, behaviour_type=wanted_behaviour,
                behaviour_condition=mouse_groups[mouse],
                cell_filter_params=cell_filter_params,
            )
            TFC_cond_LT2[mouse] = LinearTrackSession(  # noqa: F405
                mouse, dpath_TFC_cond_LT2[mouse],
                session_bounds=LT2_exp_frames[mouse], LT_type='LT2',
                data_dir=data_dir, crossreg=TFC_cond_crossreg[mouse],
                savepath=TFC_cond_savepath, behaviour_type=wanted_behaviour,
                behaviour_condition=mouse_groups[mouse],
                cell_filter_params=cell_filter_params,
            )
            TFC = TFC_cond[mouse]
            LT1 = TFC_cond_LT1[mouse]
            LT2 = TFC_cond_LT2[mouse]

            for mapping in mappings_all_TFC_cond:
                TFC.process_avg_sp_rates_mapping(mapping)
                TFC.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                tone_sp_rates_mapping[mapping][mouse] = TFC.tone_sp_rates_mapping[mapping]
                shock_sp_rates_mapping[mapping][mouse] = TFC.shock_sp_rates_mapping[mapping]
                post_shock_sp_rates_mapping[mapping][mouse] = TFC.post_shock_sp_rates_mapping[mapping]
                tone_activity_mapping[mapping][mouse] = TFC.tone_activity_mapping[mapping]
                shock_activity_mapping[mapping][mouse] = TFC.shock_activity_mapping[mapping]
                post_shock_activity_mapping[mapping][mouse] = TFC.post_shock_activity_mapping[mapping]
                post_tone_activity_mapping[mapping][mouse] = TFC.post_tone_activity_mapping[mapping]
                TFC_cond_binned_sp_rates_mapping[mapping][mouse] = TFC.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
                TFC_cond_binned_activity_mapping[mapping][mouse] = TFC.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, want_peakval=True)
                if plot_ROIs:
                    try:
                        TFC.get_A_matrix()
                        TFC_cond_ROI_mappings[mapping][mouse] = TFC.get_ROI_mapping(mapping, want_peakval=False)
                        TFC_cond_ROI_mappings_peakval[mapping][mouse] = TFC.get_ROI_mapping(mapping, want_peakval=True)
                    except Exception as _roi_e:
                        _load_errors.append((mouse, f'ROI TFC_cond mapping={mapping}', str(_roi_e), traceback.format_exc()))
                        print(f"  [ROI] {mouse} TFC_cond mapping={mapping}: {_roi_e}")

            for mapping in mappings_all_LT1:
                LT1.process_avg_sp_rates_mapping(mapping)
                LT1.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                LT1_exp_sp_rates_mapping[mapping][mouse] = LT1.exp_sp_rates_mapping[mapping]
                LT1_exp_activity_mapping[mapping][mouse] = LT1.exp_activity_mapping[mapping]
                if plot_ROIs:
                    try:
                        LT1.get_A_matrix()
                        LT1_ROI_mappings[mapping][mouse] = LT1.get_ROI_mapping(mapping, want_peakval=False)
                        LT1_ROI_mappings_peakval[mapping][mouse] = LT1.get_ROI_mapping(mapping, want_peakval=True)
                    except Exception as _roi_e:
                        _load_errors.append((mouse, f'ROI LT1 mapping={mapping}', str(_roi_e), traceback.format_exc()))
                        print(f"  [ROI] {mouse} LT1 mapping={mapping}: {_roi_e}")

            for mapping in mappings_all_LT2:
                LT2.process_avg_sp_rates_mapping(mapping)
                LT2.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                LT2_exp_sp_rates_mapping[mapping][mouse] = LT2.exp_sp_rates_mapping[mapping]
                LT2_exp_activity_mapping[mapping][mouse] = LT2.exp_activity_mapping[mapping]
                if plot_ROIs:
                    try:
                        LT2.get_A_matrix()
                        LT2_ROI_mappings[mapping][mouse] = LT2.get_ROI_mapping(mapping, want_peakval=False)
                        LT2_ROI_mappings_peakval[mapping][mouse] = LT2.get_ROI_mapping(mapping, want_peakval=True)
                    except Exception as _roi_e:
                        _load_errors.append((mouse, f'ROI LT2 mapping={mapping}', str(_roi_e), traceback.format_exc()))
                        print(f"  [ROI] {mouse} LT2 mapping={mapping}: {_roi_e}")

            # Test A
            Test_A[mouse] = TestASession(  # noqa: F405
                mouse, dpath_Test_A[mouse],
                session_bounds=Test_A_exp_frames[mouse],
                plot_sample_cell=plot_sample_cell,
                crossreg=TFC_A_A_1wk_crossreg[mouse], savepath=Test_A_savepath,
                session_group=TFC_A_A_1wk_crossreg[mouse].mappings_labels['Test_A'],
                behaviour_type=BEHAVIOUR_TYPE, behaviour_condition=mouse_groups[mouse],
                cell_filter_params=cell_filter_params,
            )
            Test_A[mouse].crossreg_full = TFC_AB_48hr_1wk_crossreg[mouse]
            A = Test_A[mouse]
            for mapping in mappings_all_Test_A:
                A.process_avg_sp_rates_mapping(mapping)
                A.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                Test_A_exp_sp_rates_mapping[mapping][mouse] = A.exp_sp_rates_mapping[mapping]
                Test_A_exp_activity_mapping[mapping][mouse] = A.exp_activity_mapping[mapping]
                Test_A_binned_sp_rates_mapping[mapping][mouse] = A.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
                Test_A_binned_activity_mapping[mapping][mouse] = A.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, want_peakval=True)
                if plot_ROIs:
                    try:
                        A.get_A_matrix()
                        Test_A_ROI_mappings[mapping][mouse] = A.get_ROI_mapping(mapping, want_peakval=False)
                        Test_A_ROI_mappings_peakval[mapping][mouse] = A.get_ROI_mapping(mapping, want_peakval=True)
                    except Exception as _roi_e:
                        _load_errors.append((mouse, f'ROI Test_A mapping={mapping}', str(_roi_e), traceback.format_exc()))
                        print(f"  [ROI] {mouse} Test_A mapping={mapping}: {_roi_e}")

            # Test A 1wk
            if mouse not in ['G15'] and dpath_Test_A_1wk_day.get(mouse, '') != '':
                Test_A_1wk[mouse] = TestASession(  # noqa: F405
                    mouse, dpath_Test_A_1wk[mouse],
                    session_bounds=Test_A_1wk_exp_frames[mouse],
                    plot_sample_cell=plot_sample_cell,
                    crossreg=TFC_A_A_1wk_crossreg[mouse], savepath=Test_A_1wk_savepath,
                    session_group=TFC_A_A_1wk_crossreg[mouse].mappings_labels['Test_A_1wk'],
                    is_1wk=True, behaviour_type=BEHAVIOUR_TYPE,
                    behaviour_condition=mouse_groups[mouse],
                    cell_filter_params=cell_filter_params,
                )
                Test_A_1wk[mouse].crossreg_full = TFC_AB_48hr_1wk_crossreg[mouse]
                A_1wk = Test_A_1wk[mouse]
                for mapping in mappings_all_Test_A_1wk:
                    A_1wk.process_avg_sp_rates_mapping(mapping)
                    A_1wk.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                    Test_A_1wk_exp_sp_rates_mapping[mapping][mouse] = A_1wk.exp_sp_rates_mapping[mapping]
                    Test_A_1wk_exp_activity_mapping[mapping][mouse] = A_1wk.exp_activity_mapping[mapping]
                    Test_A_1wk_binned_sp_rates_mapping[mapping][mouse] = A_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
                    Test_A_1wk_binned_activity_mapping[mapping][mouse] = A_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, want_peakval=True)
                    if plot_ROIs:
                        try:
                            A_1wk.get_A_matrix()
                            Test_A_1wk_ROI_mappings[mapping][mouse] = A_1wk.get_ROI_mapping(mapping, want_peakval=False)
                            Test_A_1wk_ROI_mappings_peakval[mapping][mouse] = A_1wk.get_ROI_mapping(mapping, want_peakval=True)
                        except Exception as _roi_e:
                            _load_errors.append((mouse, f'ROI Test_A_1wk mapping={mapping}', str(_roi_e), traceback.format_exc()))
                            print(f"  [ROI] {mouse} Test_A_1wk mapping={mapping}: {_roi_e}")

            # Test B
            if mouse not in ['G07']:
                Test_B[mouse] = TestBSession(  # noqa: F405
                    mouse, dpath_Test_B[mouse],
                    session_bounds=Test_B_exp_frames[mouse],
                    plot_sample_cell=plot_sample_cell,
                    crossreg=TFC_B_B_1wk_crossreg[mouse], savepath=Test_B_savepath,
                    session_group=TFC_B_B_1wk_crossreg[mouse].mappings_labels['Test_B'],
                    behaviour_type=BEHAVIOUR_TYPE, behaviour_condition=mouse_groups[mouse],
                    cell_filter_params=cell_filter_params,
                )
                Test_B[mouse].crossreg_full = TFC_AB_48hr_1wk_crossreg[mouse]
                B = Test_B[mouse]
                _tb_onsets_override = test_b_tone_onset_overrides.get(mouse)
                if _tb_onsets_override:
                    _apply_test_b_tone_onset_override(B, _tb_onsets_override)
                mappings_all_list = mappings_all_Test_B_G15 if mouse == 'G15' else mappings_all_Test_B
                for mapping in mappings_all_list:
                    B.process_avg_sp_rates_mapping(mapping)
                    B.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                    Test_B_tone_sp_rates_mapping[mapping][mouse] = B.tone_sp_rates_mapping[mapping]
                    Test_B_post_tone_sp_rates_mapping[mapping][mouse] = B.post_tone_sp_rates_mapping[mapping]
                    Test_B_tone_post_tone_sp_rates_mapping[mapping][mouse] = B.tone_post_tone_sp_rates_mapping[mapping]
                    Test_B_tone_activity_mapping[mapping][mouse] = B.tone_activity_mapping[mapping]
                    Test_B_post_tone_activity_mapping[mapping][mouse] = B.post_tone_activity_mapping[mapping]
                    Test_B_tone_post_tone_activity_mapping[mapping][mouse] = B.tone_post_tone_activity_mapping[mapping]
                    Test_B_binned_sp_rates_mapping[mapping][mouse] = B.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
                    Test_B_binned_activity_mapping[mapping][mouse] = B.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, want_peakval=True)
                    if plot_ROIs:
                        try:
                            B.get_A_matrix()
                            Test_B_ROI_mappings[mapping][mouse] = B.get_ROI_mapping(mapping, want_peakval=False)
                            Test_B_ROI_mappings_peakval[mapping][mouse] = B.get_ROI_mapping(mapping, want_peakval=True)
                        except Exception as _roi_e:
                            _load_errors.append((mouse, f'ROI Test_B mapping={mapping}', str(_roi_e), traceback.format_exc()))
                            print(f"  [ROI] {mouse} Test_B mapping={mapping}: {_roi_e}")

            # Test B 1wk
            if mouse not in ['G15']:
                Test_B_1wk[mouse] = TestBSession(  # noqa: F405
                    mouse, dpath_Test_B_1wk[mouse],
                    session_bounds=Test_B_1wk_exp_frames[mouse],
                    plot_sample_cell=plot_sample_cell,
                    crossreg=TFC_B_B_1wk_crossreg[mouse], savepath=Test_B_1wk_savepath,
                    session_group=TFC_B_B_1wk_crossreg[mouse].mappings_labels['Test_B_1wk'],
                    is_1wk=True, behaviour_type=BEHAVIOUR_TYPE,
                    behaviour_condition=mouse_groups[mouse],
                    cell_filter_params=cell_filter_params,
                )
                Test_B_1wk[mouse].crossreg_full = TFC_AB_48hr_1wk_crossreg[mouse]
                B_1wk = Test_B_1wk[mouse]
                mappings_all_list = mappings_all_Test_B_1wk_G07 if mouse == 'G07' else mappings_all_Test_B_1wk
                for mapping in mappings_all_list:
                    B_1wk.process_avg_sp_rates_mapping(mapping)
                    B_1wk.process_avg_sp_rates_mapping(mapping, want_peakval=True)
                    Test_B_1wk_tone_sp_rates_mapping[mapping][mouse] = B_1wk.tone_sp_rates_mapping[mapping]
                    Test_B_1wk_post_tone_sp_rates_mapping[mapping][mouse] = B_1wk.post_tone_sp_rates_mapping[mapping]
                    Test_B_1wk_tone_post_tone_sp_rates_mapping[mapping][mouse] = B_1wk.tone_post_tone_sp_rates_mapping[mapping]
                    Test_B_1wk_tone_activity_mapping[mapping][mouse] = B_1wk.tone_activity_mapping[mapping]
                    Test_B_1wk_post_tone_activity_mapping[mapping][mouse] = B_1wk.post_tone_activity_mapping[mapping]
                    Test_B_1wk_tone_post_tone_activity_mapping[mapping][mouse] = B_1wk.tone_post_tone_activity_mapping[mapping]
                    Test_B_1wk_binned_sp_rates_mapping[mapping][mouse] = B_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
                    Test_B_1wk_binned_activity_mapping[mapping][mouse] = B_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, want_peakval=True)
                    if plot_ROIs:
                        try:
                            B_1wk.get_A_matrix()
                            Test_B_1wk_ROI_mappings[mapping][mouse] = B_1wk.get_ROI_mapping(mapping, want_peakval=False)
                            Test_B_1wk_ROI_mappings_peakval[mapping][mouse] = B_1wk.get_ROI_mapping(mapping, want_peakval=True)
                        except Exception as _roi_e:
                            _load_errors.append((mouse, f'ROI Test_B_1wk mapping={mapping}', str(_roi_e), traceback.format_exc()))
                            print(f"  [ROI] {mouse} Test_B_1wk mapping={mapping}: {_roi_e}")

            # Emit this mouse's abnormal-cell QC montages incrementally (the
            # output dirs were created before the loop) so a long or interrupted
            # build still yields montages for every mouse that finished.
            if _cf_emit:
                try:
                    _cf_session_by_label = {
                        'TFC_cond': TFC_cond.get(mouse),
                        'LT1': TFC_cond_LT1.get(mouse),
                        'LT2': TFC_cond_LT2.get(mouse),
                        'Test_A': Test_A.get(mouse),
                        'Test_A_1wk': Test_A_1wk.get(mouse),
                        'Test_B': Test_B.get(mouse),
                        'Test_B_1wk': Test_B_1wk.get(mouse),
                    }
                    _, _cf_errs = emit_abnormal_cell_filter_montages_one_mouse(
                        mouse, _cf_session_by_label, _cf_trace_dir, _cf_roi_dir,
                        cfg.cell_filter_sphericity_enabled)
                    _load_errors.extend(_cf_errs)
                except Exception as _cf_e:
                    _load_errors.append((mouse, 'cell_filter_montage', str(_cf_e),
                                         traceback.format_exc()))
                    print(f"  [cell_filter] {mouse}: {_cf_e}")

            msg_end()  # noqa: F405

        except Exception as _e:
            _load_errors.append((mouse, type(_e).__name__, str(_e), traceback.format_exc()))
            print(f"  *** ERROR processing {mouse}: {type(_e).__name__}: {_e}")
            print(f"      Skipping rest of {mouse}, will report at end.")
            continue

    if _load_errors:
        print("\n" + "=" * 80)
        print(f"  LOADING ERRORS SUMMARY: {len(_load_errors)} mouse/mice had errors")
        print("=" * 80)
        for (m, etype, emsg, etb) in _load_errors:
            print(f"\n  Mouse {m}: {etype}: {emsg}")
            print("  Traceback:")
            for tb_line in etb.strip().split('\n'):
                print(f"    {tb_line}")
        print("=" * 80 + "\n")
    else:
        print("\n  All mice loaded successfully.\n")

    # -----------------------------------------------------------------------
    # Abnormal-cell QC diagnostic montages were emitted incrementally inside the
    # per-mouse loop (see above), into dirs created before the loop started.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Unified engram identity pass
    # -----------------------------------------------------------------------
    WHICH_ENGRAM_FOR_BINNED = cfg.WHICH_ENGRAM_FOR_BINNED
    print(f"\n*** Building unified engram identity (which_engram_for_binned="
          f"{WHICH_ENGRAM_FOR_BINNED!r})", flush=True)

    engram_id, engram_rates, engram_norms = build_engram_identity(
        ref_sessions_by_etype={'encoding': TFC_cond, 'recall': Test_B},
        mouse_groups=mouse_groups,
    )

    def _project_engram_full_idx(target_session, target_session_name, mouse,
                                 etype, mode):
        ref_name = ENGRAM_REFERENCE[etype]
        if mouse not in engram_id or etype not in engram_id[mouse]:
            return None
        mask = engram_id[mouse][etype][mode]
        ref_full_idx = np.where(mask)[0]
        if target_session_name == ref_name:
            return ref_full_idx
        if target_session_name in ('Test_B', 'Test_B_1wk'):
            if mouse not in TFC_B_B_1wk_crossreg:
                return None
            crossreg = TFC_B_B_1wk_crossreg[mouse]
            proj_mapping = ('TFC_cond+Test_B' if target_session_name == 'Test_B'
                            else 'TFC_cond+Test_B_1wk')
        elif target_session_name in ('Test_A', 'Test_A_1wk'):
            if mouse not in TFC_A_A_1wk_crossreg:
                return None
            crossreg = TFC_A_A_1wk_crossreg[mouse]
            proj_mapping = ('TFC_cond+Test_A' if target_session_name == 'Test_A'
                            else 'TFC_cond+Test_A_1wk')
        elif target_session_name == 'TFC_cond':
            if mouse not in TFC_B_B_1wk_crossreg:
                return None
            crossreg = TFC_B_B_1wk_crossreg[mouse]
            proj_mapping = 'TFC_cond+Test_B'
        else:
            return None
        ref_session = (TFC_cond if ref_name == 'TFC_cond' else Test_B)[mouse]
        try:
            return project_engram_to_session(
                ref_session, ref_full_idx, target_session, crossreg, proj_mapping,
            )
        except Exception as _e:
            print(f"  [engram-proj] mouse={mouse} ref={ref_name} -> "
                  f"target={target_session_name} via {proj_mapping!r}: {_e}",
                  flush=True)
            return None

    def _populate_unified_engram_panels(session_dict, session_name, mappings,
                                        target_dict, label):
        for mode in ENGRAM_MODES:
            for m, sess in session_dict.items():
                engram_full_idx = _project_engram_full_idx(
                    sess, session_name, m, WHICH_ENGRAM_FOR_BINNED, mode,
                )
                if engram_full_idx is None:
                    print(f"  [engram] {label} mouse={m} mode={mode}: no projection "
                          f"available (crossreg/ref missing) -> skipping", flush=True)
                    continue
                for mapping in mappings:
                    if (mapping != 'full'
                            and not all(part in sess.crossreg.mappings_labels
                                        for part in mapping.split('+'))):
                        print(f"  [engram] {label} mapping={mapping} mouse={m} "
                              f"mode={mode}: mouse crossreg lacks one or more "
                              f"mapping components -> skipping", flush=True)
                        continue
                    target_dict[mode][mapping][m] = sess.process_binned_sp_rates_mapping(
                        mapping, BIN_WIDTH, want_peakval=True, with_engram=True,
                        engram_cell_indices=list(engram_full_idx),
                    )

    print("\n*** Filling engram-binned panels using unified identity"
          f" (etype={WHICH_ENGRAM_FOR_BINNED!r})", flush=True)
    _populate_unified_engram_panels(TFC_cond, 'TFC_cond', mappings_all_TFC_cond,
                                    TFC_cond_binned_activity_mapping_engram, 'TFC_cond')
    _populate_unified_engram_panels(Test_A, 'Test_A', mappings_all_Test_A,
                                    Test_A_binned_activity_mapping_engram, 'Test_A')
    _populate_unified_engram_panels(Test_A_1wk, 'Test_A_1wk', mappings_all_Test_A_1wk,
                                    Test_A_1wk_binned_activity_mapping_engram, 'Test_A_1wk')
    _populate_unified_engram_panels(Test_B, 'Test_B', mappings_all_Test_B,
                                    Test_B_binned_activity_mapping_engram, 'Test_B')
    _populate_unified_engram_panels(Test_B_1wk, 'Test_B_1wk', mappings_all_Test_B_1wk,
                                    Test_B_1wk_binned_activity_mapping_engram, 'Test_B_1wk')
    print("    ...done.\n", flush=True)

    # -----------------------------------------------------------------------
    # PF-merged attribute backfill
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Pack everything into a Dataset namespace.
    # -----------------------------------------------------------------------
    ds = SimpleNamespace(
        cfg=cfg,
        # paths / dirs
        NPY_SAVE_PATH=NPY_SAVE_PATH,
        PLOTS_DIR=PLOTS_DIR,
        PAPER_DIR=PAPER_DIR,
        TFC_cond_savepath=TFC_cond_savepath,
        Test_A_savepath=Test_A_savepath,
        Test_A_1wk_savepath=Test_A_1wk_savepath,
        Test_B_savepath=Test_B_savepath,
        Test_B_1wk_savepath=Test_B_1wk_savepath,
        data_dir=data_dir,
        crossreg_file_TFC_cond=crossreg_file_TFC_cond,
        crossreg_file_4=crossreg_file_4,
        crossreg_file_6=crossreg_file_6,
        crossreg_file_7=crossreg_file_7,
        # behaviour / config echo (some analysis cells reference these as bare names)
        BEHAVIOUR_TYPE=BEHAVIOUR_TYPE,
        BIN_WIDTH=BIN_WIDTH,
        ENGRAM_MODES=ENGRAM_MODES,
        WHICH_ENGRAM_FOR_BINNED=WHICH_ENGRAM_FOR_BINNED,
        DEBUG=DEBUG,
        LOCAL_DATA=LOCAL_DATA,
        # group/mouse metadata
        mouse_groups=mouse_groups,
        mouse_list=mouse_list,
        mice_per_group=mice_per_group,
        mice_per_group_Test_B_B_1wk=mice_per_group_Test_B_B_1wk,
        mice_skip_LT=mice_skip_LT,
        mouse_drive_prefix=mouse_drive_prefix,
        mouse_path_prefix=mouse_path_prefix,
        dpath_mouse=dpath_mouse,
        test_unit_id=test_unit_id,
        period_overrides=period_overrides,
        test_b_tone_onset_overrides=test_b_tone_onset_overrides,
        # dpath / exp_frames dicts (post-join, absolute)
        dpath_TFC_cond_day=dpath_TFC_cond_day,
        dpath_TFC_cond=dpath_TFC_cond,
        dpath_TFC_cond_LT1=dpath_TFC_cond_LT1,
        dpath_TFC_cond_LT2=dpath_TFC_cond_LT2,
        TFC_cond_exp_frames=TFC_cond_exp_frames,
        LT1_exp_frames=LT1_exp_frames,
        LT2_exp_frames=LT2_exp_frames,
        dpath_Test_A_day=dpath_Test_A_day,
        dpath_Test_A=dpath_Test_A,
        Test_A_exp_frames=Test_A_exp_frames,
        dpath_Test_A_LT1=dpath_Test_A_LT1,
        Test_A_LT1_exp_frames=Test_A_LT1_exp_frames,
        dpath_Test_A_1wk_day=dpath_Test_A_1wk_day,
        dpath_Test_A_1wk=dpath_Test_A_1wk,
        Test_A_1wk_exp_frames=Test_A_1wk_exp_frames,
        dpath_Test_A_1wk_LT1=dpath_Test_A_1wk_LT1,
        Test_A_1wk_LT1_exp_frames=Test_A_1wk_LT1_exp_frames,
        dpath_Test_B_day=dpath_Test_B_day,
        dpath_Test_B=dpath_Test_B,
        Test_B_exp_frames=Test_B_exp_frames,
        dpath_Test_B_LT1=dpath_Test_B_LT1,
        Test_B_LT1_exp_frames=Test_B_LT1_exp_frames,
        dpath_Test_B_1wk_day=dpath_Test_B_1wk_day,
        dpath_Test_B_1wk=dpath_Test_B_1wk,
        Test_B_1wk_exp_frames=Test_B_1wk_exp_frames,
        dpath_Test_B_1wk_LT1=dpath_Test_B_1wk_LT1,
        Test_B_1wk_LT1_exp_frames=Test_B_1wk_LT1_exp_frames,
        # crossreg group definitions + file paths
        TFC_cond_crossreg_groups=TFC_cond_crossreg_groups,
        TFC_B_B_1wk_crossreg_groups=TFC_B_B_1wk_crossreg_groups,
        TFC_AB_48hr_1wk_crossreg_groups=TFC_AB_48hr_1wk_crossreg_groups,
        TFC_A_A_1wk_crossreg_groups=TFC_A_A_1wk_crossreg_groups,
        dpath_TFC_cond_crossreg_file=dpath_TFC_cond_crossreg_file,
        dpath_TFC_B_B_1wk_crossreg_file=dpath_TFC_B_B_1wk_crossreg_file,
        dpath_TFC_AB_48hr_1wk_crossreg_file=dpath_TFC_AB_48hr_1wk_crossreg_file,
        dpath_TFC_A_A_1wk_crossreg_file=dpath_TFC_A_A_1wk_crossreg_file,
        # session dicts
        TFC_cond=TFC_cond,
        TFC_cond_LT1=TFC_cond_LT1,
        TFC_cond_LT2=TFC_cond_LT2,
        TFC_cond_crossreg=TFC_cond_crossreg,
        Test_A=Test_A,
        Test_A_1wk=Test_A_1wk,
        Test_B=Test_B,
        Test_B_1wk=Test_B_1wk,
        TFC_B_B_1wk_crossreg=TFC_B_B_1wk_crossreg,
        TFC_AB_48hr_1wk_crossreg=TFC_AB_48hr_1wk_crossreg,
        TFC_A_A_1wk_crossreg=TFC_A_A_1wk_crossreg,
        # mapping label constants
        mapping_FULL=mapping_FULL,
        mapping_LT1=mapping_LT1, mapping_LT2=mapping_LT2,
        mapping_LT1_LT2=mapping_LT1_LT2,
        mapping_TFC_cond=mapping_TFC_cond,
        mapping_LT1_LT2_TFC_cond=mapping_LT1_LT2_TFC_cond,
        mapping_LT2_TFC_cond=mapping_LT2_TFC_cond,
        mapping_LT1_TFC_cond=mapping_LT1_TFC_cond,
        mapping_TFC_cond_Test_B=mapping_TFC_cond_Test_B,
        mapping_TFC_cond_Test_B_1wk=mapping_TFC_cond_Test_B_1wk,
        mapping_TFC_cond_Test_B_Test_B_1wk=mapping_TFC_cond_Test_B_Test_B_1wk,
        mapping_Test_B_Test_B_1wk=mapping_Test_B_Test_B_1wk,
        mapping_TFC_cond_Test_A_Test_A_1wk=mapping_TFC_cond_Test_A_Test_A_1wk,
        mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk=mapping_TFC_cond_Test_A_Test_A_1wk_Test_B_Test_B_1wk,
        # mappings_all_* lists
        mappings_all_TFC_cond=mappings_all_TFC_cond,
        mappings_all_LT1=mappings_all_LT1,
        mappings_all_LT2=mappings_all_LT2,
        mappings_all_Test_A=mappings_all_Test_A,
        mappings_all_Test_A_1wk=mappings_all_Test_A_1wk,
        mappings_all_Test_B=mappings_all_Test_B,
        mappings_all_Test_B_G15=mappings_all_Test_B_G15,
        mappings_all_Test_B_1wk=mappings_all_Test_B_1wk,
        mappings_all_Test_B_1wk_G07=mappings_all_Test_B_1wk_G07,
        # TFC_cond per-mapping accumulators
        tone_sp_rates_mapping=tone_sp_rates_mapping,
        shock_sp_rates_mapping=shock_sp_rates_mapping,
        post_shock_sp_rates_mapping=post_shock_sp_rates_mapping,
        tone_activity_mapping=tone_activity_mapping,
        shock_activity_mapping=shock_activity_mapping,
        post_shock_activity_mapping=post_shock_activity_mapping,
        post_tone_activity_mapping=post_tone_activity_mapping,
        TFC_cond_binned_sp_rates_mapping=TFC_cond_binned_sp_rates_mapping,
        TFC_cond_binned_activity_mapping=TFC_cond_binned_activity_mapping,
        TFC_cond_binned_activity_mapping_engram=TFC_cond_binned_activity_mapping_engram,
        TFC_cond_ROI_mappings=TFC_cond_ROI_mappings,
        TFC_cond_ROI_mappings_peakval=TFC_cond_ROI_mappings_peakval,
        # Test_A
        Test_A_exp_sp_rates_mapping=Test_A_exp_sp_rates_mapping,
        Test_A_exp_activity_mapping=Test_A_exp_activity_mapping,
        Test_A_binned_sp_rates_mapping=Test_A_binned_sp_rates_mapping,
        Test_A_binned_activity_mapping=Test_A_binned_activity_mapping,
        Test_A_binned_activity_mapping_engram=Test_A_binned_activity_mapping_engram,
        Test_A_ROI_mappings=Test_A_ROI_mappings,
        Test_A_ROI_mappings_peakval=Test_A_ROI_mappings_peakval,
        # Test_A_1wk
        Test_A_1wk_exp_sp_rates_mapping=Test_A_1wk_exp_sp_rates_mapping,
        Test_A_1wk_exp_activity_mapping=Test_A_1wk_exp_activity_mapping,
        Test_A_1wk_binned_sp_rates_mapping=Test_A_1wk_binned_sp_rates_mapping,
        Test_A_1wk_binned_activity_mapping=Test_A_1wk_binned_activity_mapping,
        Test_A_1wk_binned_activity_mapping_engram=Test_A_1wk_binned_activity_mapping_engram,
        Test_A_1wk_ROI_mappings=Test_A_1wk_ROI_mappings,
        Test_A_1wk_ROI_mappings_peakval=Test_A_1wk_ROI_mappings_peakval,
        # Test_B
        Test_B_tone_sp_rates_mapping=Test_B_tone_sp_rates_mapping,
        Test_B_post_tone_sp_rates_mapping=Test_B_post_tone_sp_rates_mapping,
        Test_B_tone_post_tone_sp_rates_mapping=Test_B_tone_post_tone_sp_rates_mapping,
        Test_B_tone_activity_mapping=Test_B_tone_activity_mapping,
        Test_B_post_tone_activity_mapping=Test_B_post_tone_activity_mapping,
        Test_B_tone_post_tone_activity_mapping=Test_B_tone_post_tone_activity_mapping,
        Test_B_binned_sp_rates_mapping=Test_B_binned_sp_rates_mapping,
        Test_B_binned_activity_mapping=Test_B_binned_activity_mapping,
        Test_B_binned_activity_mapping_engram=Test_B_binned_activity_mapping_engram,
        Test_B_ROI_mappings=Test_B_ROI_mappings,
        Test_B_ROI_mappings_peakval=Test_B_ROI_mappings_peakval,
        # Test_B_1wk
        Test_B_1wk_tone_sp_rates_mapping=Test_B_1wk_tone_sp_rates_mapping,
        Test_B_1wk_post_tone_sp_rates_mapping=Test_B_1wk_post_tone_sp_rates_mapping,
        Test_B_1wk_tone_post_tone_sp_rates_mapping=Test_B_1wk_tone_post_tone_sp_rates_mapping,
        Test_B_1wk_tone_activity_mapping=Test_B_1wk_tone_activity_mapping,
        Test_B_1wk_post_tone_activity_mapping=Test_B_1wk_post_tone_activity_mapping,
        Test_B_1wk_tone_post_tone_activity_mapping=Test_B_1wk_tone_post_tone_activity_mapping,
        Test_B_1wk_binned_sp_rates_mapping=Test_B_1wk_binned_sp_rates_mapping,
        Test_B_1wk_binned_activity_mapping=Test_B_1wk_binned_activity_mapping,
        Test_B_1wk_binned_activity_mapping_engram=Test_B_1wk_binned_activity_mapping_engram,
        Test_B_1wk_ROI_mappings=Test_B_1wk_ROI_mappings,
        Test_B_1wk_ROI_mappings_peakval=Test_B_1wk_ROI_mappings_peakval,
        # LT
        LT1_exp_sp_rates_mapping=LT1_exp_sp_rates_mapping,
        LT2_exp_sp_rates_mapping=LT2_exp_sp_rates_mapping,
        LT1_exp_activity_mapping=LT1_exp_activity_mapping,
        LT2_exp_activity_mapping=LT2_exp_activity_mapping,
        LT1_ROI_mappings=LT1_ROI_mappings,
        LT2_ROI_mappings=LT2_ROI_mappings,
        LT1_ROI_mappings_peakval=LT1_ROI_mappings_peakval,
        LT2_ROI_mappings_peakval=LT2_ROI_mappings_peakval,
        # engram pass outputs
        engram_id=engram_id,
        engram_rates=engram_rates,
        engram_norms=engram_norms,
        # helper functions used downstream
        plot_interneuron_cutoff=plot_interneuron_cutoff,
        add_significance_bars=add_significance_bars,
        # error log
        _load_errors=_load_errors,
    )
    return ds

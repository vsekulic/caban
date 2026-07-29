import os
import traceback

import numpy as np
import matplotlib.pyplot as plt
from caban.utilities import *
from caban.sessions import *
from caban.spatial import *
from caban.plotting import *


def _resolve_session(session_or_sessions, mouse):
    if hasattr(session_or_sessions, "items"):
        if mouse not in session_or_sessions:
            raise KeyError(f"Mouse '{mouse}' is not present in the provided session collection.")
        return session_or_sessions[mouse]

    session = session_or_sessions
    session_mouse = getattr(session, "mouse", None)
    if mouse is not None and session_mouse is not None and session_mouse != mouse:
        raise ValueError(
            f"Requested mouse '{mouse}', but the provided session object belongs to '{session_mouse}'."
        )
    return session


def _save_figure_outputs(fig, output_paths):
    for output_path in output_paths:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")


def _build_A_matrix_paths(plots_dir, paper_plots_dir, session_label, mouse):
    filename = f"A_matrix_{session_label}_{mouse}.png"
    return [
        os.path.join(plots_dir, filename),
        os.path.join(paper_plots_dir, filename),
    ]

def _to_numpy(x):
    """
    Accepts numpy arrays or xarray.DataArray.
    Handles dask-backed xarray by calling .compute().
    """
    if hasattr(x, "compute") and hasattr(x, "values"):
        return x.compute().values
    return np.asarray(x)

def normalize_img(img, p_low=1, p_high=99.8):
    """
    Robust grayscale normalization for raw miniscope images.
    """
    img = _to_numpy(img).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(img, [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0, 1)

    return img

def prepare_raw_image(raw, projection="mean"):
    """
    raw can be:
        2D: height x width
        3D: frame x height x width

    projection:
        "mean", "max", or "std"
    """
    raw = _to_numpy(raw).astype(np.float32)

    if raw.ndim == 2:
        raw_img = raw

    elif raw.ndim == 3:
        if projection == "mean":
            raw_img = np.nanmean(raw, axis=0)
        elif projection == "max":
            raw_img = np.nanmax(raw, axis=0)
        elif projection == "std":
            raw_img = np.nanstd(raw, axis=0)
        else:
            raise ValueError("projection must be 'mean', 'max', or 'std'")

    else:
        raise ValueError(f"raw should be 2D or 3D, got shape {raw.shape}")

    return normalize_img(raw_img)

def prepare_A_array(A, unit_axis=0):
    """
    Converts A to shape:
        n_units x height x width

    For MiniAn xarray A, this expects dims:
        unit_id, height, width

    If your A is numpy:
        unit_axis=0 means A.shape = n_units x height x width
        unit_axis=-1 means A.shape = height x width x n_units
    """
    if hasattr(A, "dims"):
        # xarray / MiniAn case
        dims = list(A.dims)

        if "height" not in dims or "width" not in dims:
            raise ValueError(f"Could not find height/width dims in A.dims={A.dims}")

        non_spatial_dims = [d for d in dims if d not in ["height", "width"]]

        if len(non_spatial_dims) != 1:
            raise ValueError(
                f"A has dims {A.dims}. Select one session/animal/etc. first, "
                "so that only unit_id, height, width remain."
            )

        unit_dim = "unit_id" if "unit_id" in dims else non_spatial_dims[0]
        A = A.transpose(unit_dim, "height", "width")
        A = A.compute().values.astype(np.float32)

    else:
        A = np.asarray(A, dtype=np.float32)

        if A.ndim != 3:
            raise ValueError(
                f"A should be 3D, got shape {A.shape}. "
                "Expected n_units x height x width or height x width x n_units."
            )

        if unit_axis != 0:
            A = np.moveaxis(A, unit_axis, 0)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A

def normalize_each_footprint(A):
    """
    Normalize each ROI footprint independently before summing.
    This prevents a few bright components from dominating the display.
    """
    A = A.copy().astype(np.float32)

    mins = A.reshape(A.shape[0], -1).min(axis=1)[:, None, None]
    maxs = A.reshape(A.shape[0], -1).max(axis=1)[:, None, None]
    denom = maxs - mins

    A = np.divide(
        A - mins,
        denom,
        out=np.zeros_like(A, dtype=np.float32),
        where=denom > 0,
    )

    return A

def make_A_projection(
    A,
    unit_axis=0,
    normalize_units=True,
    projection="sum",
    clip_percentile=99.5,
    gamma=0.7,
):
    """
    Collapse A into one 2D footprint image.

    projection:
        "sum" = denser areas become brighter
        "max" = each pixel shows strongest ROI footprint; less saturation in crowded fields
    """
    A = prepare_A_array(A, unit_axis=unit_axis)

    if normalize_units:
        A = normalize_each_footprint(A)

    if projection == "sum":
        A_img = A.sum(axis=0)
    elif projection == "max":
        A_img = A.max(axis=0)
    else:
        raise ValueError("projection must be 'sum' or 'max'")

    positive = A_img[A_img > 0]

    if positive.size == 0:
        return np.zeros_like(A_img, dtype=np.float32)

    hi = np.percentile(positive, clip_percentile)

    if hi <= 0:
        hi = positive.max()

    A_img = np.clip(A_img / hi, 0, 1)
    A_img = A_img ** gamma

    return A_img.astype(np.float32)

def make_offwhite_overlay(
    raw_img,
    A_img,
    color=(1.0, 0.93, 0.78),
    alpha_scale=0.85,
):
    """
    Composite off-white A footprints on top of grayscale raw image.

    color:
        RGB tuple. This default is a warm off-white.
    """
    raw_rgb = np.dstack([raw_img, raw_img, raw_img])

    overlay_rgb = np.zeros_like(raw_rgb)
    overlay_rgb[..., 0] = color[0]
    overlay_rgb[..., 1] = color[1]
    overlay_rgb[..., 2] = color[2]

    alpha = np.clip(A_img * alpha_scale, 0, 1)
    alpha = alpha[..., None]

    composite = raw_rgb * (1 - alpha) + overlay_rgb * alpha
    composite = np.clip(composite, 0, 1)

    return composite

def plot_A_vs_raw(
    raw,
    A,
    unit_axis=0,
    raw_projection="mean",
    A_projection="sum",
    normalize_units=True,
    A_clip_percentile=99.5,
    gamma=0.7,
    alpha_scale=0.85,
    offwhite=(1.0, 0.93, 0.78),
    figsize=(14, 5),
    save_path=None,
):
    """
    Make a 3-panel figure:
        raw image | extracted A footprints | raw + A overlay
    """
    raw_img = prepare_raw_image(raw, projection=raw_projection)

    A_img = make_A_projection(
        A,
        unit_axis=unit_axis,
        normalize_units=normalize_units,
        projection=A_projection,
        clip_percentile=A_clip_percentile,
        gamma=gamma,
    )

    composite = make_offwhite_overlay(
        raw_img,
        A_img,
        color=offwhite,
        alpha_scale=alpha_scale,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title(f"Raw data: {raw_projection} projection")
    axes[0].axis("off")

    axes[1].imshow(A_img, cmap="gray")
    axes[1].set_title("Extracted A footprints")
    axes[1].axis("off")

    axes[2].imshow(composite)
    axes[2].set_title("Raw + A overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, axes


def plot_session_A_matrix(
    session_or_sessions,
    mouse="G10",
    *,
    plots_dir,
    paper_plots_dir,
    session_label=None,
    unit_axis=0,
    normalize_units=True,
    A_projection="sum",
    A_clip_percentile=99.5,
    gamma=0.7,
    figsize=(6, 6),
    show=True,
):
    session = _resolve_session(session_or_sessions, mouse)
    session_label = session_label or getattr(session, "session_type", "session")

    A_img = make_A_projection(
        session.A,
        unit_axis=unit_axis,
        normalize_units=normalize_units,
        projection=A_projection,
        clip_percentile=A_clip_percentile,
        gamma=gamma,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(A_img, cmap="gray")
    ax.set_title(f"{session_label} A matrix: {mouse}")
    ax.axis("off")
    plt.tight_layout()

    output_paths = _build_A_matrix_paths(plots_dir, paper_plots_dir, session_label, mouse)
    _save_figure_outputs(fig, output_paths)

    if show:
        plt.show()

    return fig, ax, output_paths


# ----------------------------------------------------------------------
# Abnormal-cell QC filter diagnostics (ROI montage + trace montage)
# ----------------------------------------------------------------------

def _crop_to_bbox(mask, pad=1):
    """Crop a 2D boolean/uint mask to the bounding box of its nonzero pixels."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    return mask[y0:y1 + 1, x0:x1 + 1]


def _save_roi_montage_pages(footprints, labels, save_dir, fname_prefix,
                            cols=14, max_per_fig=140, title=''):
    """Render a paginated dense grid of cropped ROI thumbnails."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    n = len(footprints)
    if n == 0:
        return paths
    for page_start in range(0, n, max_per_fig):
        chunk_fp = footprints[page_start:page_start + max_per_fig]
        chunk_lb = labels[page_start:page_start + max_per_fig]
        ncells = len(chunk_fp)
        rows = int(np.ceil(ncells / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.9))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for i, (fp, lb) in enumerate(zip(chunk_fp, chunk_lb)):
            ax = axes[i]
            ax.imshow(fp, cmap="gray", interpolation="nearest")
            ax.set_title(lb, fontsize=4, pad=1)
            ax.axis("off")
        page = page_start // max_per_fig + 1
        if title:
            fig.suptitle(f"{title} (page {page})", fontsize=8)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{fname_prefix}_p{page}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_abnormal_cell_filter_roi_montage(session, save_dir, mouse,
                                          session_label=None, cols=14,
                                          max_per_fig=140):
    """
    Dense ROI thumbnail montages of cell footprints. When the sphericity check
    was run, cells are split into a sphericity-REJECTED set and an INCLUDED set,
    and each tile is labelled with the cell's original S-extraction row index
    and its sphericity value. When the sphericity check was NOT run, every cell
    is shown in a single ACCEPTED montage (no rejected set) and tiles are
    labelled with the row index only.

    Requires the session to have run the abnormal-cell filter (so that
    ``cell_filter_submasks`` is populated).
    """
    session_label = session_label or getattr(session, "session_type", "session")
    submasks = getattr(session, "cell_filter_submasks", None)
    if submasks is None:
        return []  # abnormal-cell filter did not run
    has_sphericity = "is_round" in submasks
    sphericity = getattr(session, "cell_sphericity", None) if has_sphericity else None

    A = session.A
    A_idx = np.asarray(session.A_idx)
    S_idx = np.asarray(session.S_idx)
    uid_to_A = {int(u): i for i, u in enumerate(A_idx)}

    def collect(rows):
        fps, labels = [], []
        for r in rows:
            uid = int(S_idx[r])
            a = uid_to_A.get(uid)
            if a is None:
                continue
            fps.append(_crop_to_bbox(np.asarray(A[a]) > 0))
            if sphericity is not None:
                labels.append(f"{r}\n{sphericity[r]:.2f}")
            else:
                labels.append(f"{r}")
        return fps, labels

    paths = []
    if has_sphericity:
        is_round = submasks["is_round"]
        rej_fp, rej_lb = collect(np.where(~is_round)[0])
        paths += _save_roi_montage_pages(
            rej_fp, rej_lb, save_dir, f"{mouse}_{session_label}_rejected",
            cols=cols, max_per_fig=max_per_fig,
            title=f"{mouse} {session_label}: sphericity REJECTED")
        inc_fp, inc_lb = collect(np.where(is_round)[0])
        paths += _save_roi_montage_pages(
            inc_fp, inc_lb, save_dir, f"{mouse}_{session_label}_included",
            cols=cols, max_per_fig=max_per_fig,
            title=f"{mouse} {session_label}: sphericity INCLUDED")
    else:
        all_fp, all_lb = collect(np.arange(len(S_idx)))
        paths += _save_roi_montage_pages(
            all_fp, all_lb, save_dir, f"{mouse}_{session_label}_accepted",
            cols=cols, max_per_fig=max_per_fig,
            title=f"{mouse} {session_label}: ROIs ACCEPTED (sphericity off)")
    return paths


def plot_abnormal_cell_filter_trace_montage(session, save_dir, mouse,
                                            session_label=None, per_page=40,
                                            page_size=(8.5, 11)):
    """
    Densely-stacked traces of EXCLUDED cells (all rejection reasons), one trace
    per row spanning the full session length on an 8.5x11 page. Each trace is
    labelled with the cell's original S-extraction row index and the failed
    check(s). Paginated.
    """
    session_label = session_label or getattr(session, "session_type", "session")
    bad = getattr(session, "bad_cell_indices", None)
    if bad is None or len(bad) == 0:
        return []
    signal_name = getattr(session, "cell_filter_signal_name", "C")
    sig = np.asarray(getattr(session, signal_name), dtype=float)
    submasks = getattr(session, "cell_filter_submasks", None)
    os.makedirs(save_dir, exist_ok=True)

    def reason(r):
        if submasks is None:
            return ""
        tags = []
        if not submasks["is_skewed"][r]:
            tags.append("skew")
        if not submasks["is_sparse"][r]:
            tags.append("active")
        if not submasks["no_plateaus"][r]:
            tags.append("plateau")
        if not submasks["no_silent"][r]:
            tags.append("silent")
        if "is_round" in submasks and not submasks["is_round"][r]:
            tags.append("shape")
        return ",".join(tags)

    bad = list(bad)
    n_frames = sig.shape[1]
    x = np.arange(n_frames)
    paths = []
    for page_start in range(0, len(bad), per_page):
        chunk = bad[page_start:page_start + per_page]
        fig, ax = plt.subplots(figsize=page_size)
        for i, r in enumerate(chunk):
            trace = sig[r]
            tmin, tmax = trace.min(), trace.max()
            denom = tmax - tmin
            norm = (trace - tmin) / denom if denom > 0 else np.zeros_like(trace)
            offset = len(chunk) - 1 - i  # stack top-down
            ax.plot(x, norm * 0.9 + offset, lw=0.3, color="k")
            ax.text(-0.01 * n_frames, offset + 0.45, f"{r} [{reason(r)}]",
                    fontsize=4, ha="right", va="center")
        ax.set_xlim(0, n_frames)
        ax.set_ylim(-0.5, len(chunk))
        ax.set_yticks([])
        ax.set_xlabel("frame")
        page = page_start // per_page + 1
        ax.set_title(
            f"{mouse} {session_label}: excluded cells ({signal_name}) page {page}",
            fontsize=8)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{mouse}_{session_label}_excluded_p{page}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


ABNORMAL_CELL_FILTER_SESSION_ATTRS = (
    "TFC_cond", "TFC_cond_LT1", "TFC_cond_LT2",
    "Test_A", "Test_A_1wk", "Test_B", "Test_B_1wk",
)


def ensure_abnormal_cell_filter_dirs(plots_dir, verbose=True):
    """
    Create (and return) the abnormal-cell QC montage output directories.

    Returns
    -------
    (trace_dir, roi_dir) : tuple[str, str]
        ``<plots_dir>/abnormal_cell_filter/{traces,roi}`` (both created).
    """
    parent_dir = os.path.join(plots_dir, "abnormal_cell_filter")
    trace_dir = os.path.join(parent_dir, "traces")
    roi_dir = os.path.join(parent_dir, "roi")
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)
    if verbose:
        print(f"*** [cell_filter] QC montage dirs ready under "
              f"{os.path.abspath(parent_dir)}", flush=True)
        print(f"    [cell_filter] trace dir = {os.path.abspath(trace_dir)} "
              f"(exists={os.path.isdir(trace_dir)})", flush=True)
        print(f"    [cell_filter] roi dir   = {os.path.abspath(roi_dir)} "
              f"(exists={os.path.isdir(roi_dir)})", flush=True)
    return trace_dir, roi_dir


def emit_abnormal_cell_filter_montages_one_mouse(
        mouse, session_by_label, trace_dir, roi_dir,
        sphericity_enabled, verbose=True):
    """
    Emit the QC montages for a single mouse's sessions.

    Parameters
    ----------
    mouse : str
    session_by_label : dict[str, BehaviourSession | None]
        Maps a session label (e.g. 'TFC_cond') to that mouse's session (or None
        if the mouse has no such session).
    trace_dir, roi_dir : str
        Output dirs (use :func:`ensure_abnormal_cell_filter_dirs`).
    sphericity_enabled : bool
        Informational only (the ROI montage is always emitted).

    Returns
    -------
    (n_pages, errors) : tuple[int, list]
        Pages written and ``(mouse, label, message, traceback)`` per failure.
    """
    n_pages = 0
    errors = []
    for label, sess in session_by_label.items():
        if sess is None:
            continue
        good = vars(sess).get("good_cell_indices", None)
        if good is None:
            if verbose:
                print(f"    [cell_filter] {label:>12} {mouse}: no filter results",
                      flush=True)
            continue
        bad = vars(sess).get("bad_cell_indices", None)
        n_good = len(good) if hasattr(good, "__len__") else good
        n_bad = len(bad) if hasattr(bad, "__len__") else bad
        if verbose:
            print(f"    [cell_filter] {label:>12} {mouse}: good={n_good} bad={n_bad}",
                  flush=True)
        try:
            n_pages += len(
                plot_abnormal_cell_filter_trace_montage(sess, trace_dir, mouse))
            n_pages += len(
                plot_abnormal_cell_filter_roi_montage(sess, roi_dir, mouse))
        except Exception as exc:
            errors.append((mouse, label, str(exc), traceback.format_exc()))
            print(f"  [cell_filter] {mouse} {label}: {exc}", flush=True)
    return n_pages, errors


def emit_abnormal_cell_filter_montages(session_dicts, plots_dir,
                                       sphericity_enabled, verbose=True):
    """
    Write the abnormal-cell QC montages for a collection of session dicts.

    Parameters
    ----------
    session_dicts : dict[str, dict[str, BehaviourSession]]
        Maps a session label (e.g. 'TFC_cond') to a {mouse: session} dict.
    plots_dir : str
        Root plots directory; montages go under
        ``<plots_dir>/abnormal_cell_filter/{traces,roi}``.
    sphericity_enabled : bool
        Informational only: the ROI montage is always emitted -- when sphericity
        is off every cell is shown as ACCEPTED; when on, cells are split into
        REJECTED/INCLUDED sets.
    verbose : bool
        Print a progress banner and per-session errors.

    Returns
    -------
    (n_pages, errors) : tuple[int, list]
        Number of montage pages written and a list of
        ``(mouse, session_label, message, traceback)`` for any failures.
    """
    trace_dir, roi_dir = ensure_abnormal_cell_filter_dirs(plots_dir, verbose=verbose)
    if verbose:
        mode = "rejected/included" if sphericity_enabled else "all-accepted"
        print(f"*** [cell_filter] emitting QC montages (+roi: {mode})", flush=True)

    n_pages = 0
    n_sessions_with_data = 0
    n_sessions_seen = 0
    errors = []
    mice = sorted({m for sdict in session_dicts.values() for m in sdict})
    for mouse in mice:
        session_by_label = {label: sdict.get(mouse)
                            for label, sdict in session_dicts.items()}
        for sess in session_by_label.values():
            n_sessions_seen += 1
            if sess is not None and vars(sess).get("good_cell_indices", None) is not None:
                n_sessions_with_data += 1
        m_pages, m_errs = emit_abnormal_cell_filter_montages_one_mouse(
            mouse, session_by_label, trace_dir, roi_dir,
            sphericity_enabled, verbose=verbose)
        n_pages += m_pages
        errors.extend(m_errs)

    if verbose:
        print(f"    [cell_filter] inspected {n_sessions_seen} session(s); "
              f"{n_sessions_with_data} carried filter results.", flush=True)
        if n_sessions_with_data == 0:
            print("*** [cell_filter] no sessions carry filter results "
                  "(good_cell_indices is None on all) -- nothing written. "
                  "This usually means ds was loaded from a pre-filter cache; "
                  "rebuild with load_all_mice(cfg, use_cache=False).", flush=True)
        else:
            print(f"*** [cell_filter] wrote {n_pages} montage page(s) from "
                  f"{n_sessions_with_data} session(s).", flush=True)
    return n_pages, errors


def emit_abnormal_cell_filter_montages_for_ds(ds, cfg, verbose=True):
    """Convenience wrapper: build the session-dict map from a loaded ``ds``."""
    session_dicts = {}
    for attr in ABNORMAL_CELL_FILTER_SESSION_ATTRS:
        sdict = getattr(ds, attr, None)
        if sdict:
            session_dicts[attr] = sdict
    return emit_abnormal_cell_filter_montages(
        session_dicts, cfg.PLOTS_DIR, cfg.cell_filter_sphericity_enabled,
        verbose=verbose)
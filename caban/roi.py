import os

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
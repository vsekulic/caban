import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from SSTCa2_decoder import _LT_1D_CM_PER_PX, _LT_1D_DISTANCE_UNIT

def plot_lt_spatial_responses(PLOTS_DIR, LT1_group, LT2_group, mouse_groups, session_str, mapping=None):
    '''
    Plot spatial responses for cross-registered cells between LT1 and LT2, sorted according to LT1.
    Uses CENTERLINE arclength (behavioral cross-registration anchor) for the x-axis in BOTH sessions,
    so LT2 is plotted in the same behavioral coordinate system as LT1.

    Assumptions:
      - LT1.centerline and LT2.centerline are arrays of shape (N,2) in pixel coordinates [x,y].
      - LT1.loc_X_miniscope_smooth / loc_Y_miniscope_smooth are miniscope XY in pixel coordinates.
      - LT1.fm.pf.model_[cell_id].means_ are in (row, col) bin-coordinates.
      - LT1.fm.pf.merged_means[cell_id] gives merged mean groups (list of lists of mean indices).
      - LT1.fm.sig_responses[cell_id] is list of [row, col, value] (row=vertical, col=horizontal).
      - "use_sig_responses" default False (your new default).
    '''
    if mapping is None:
        raise ValueError("plot_lt_spatial_responses: mapping must not be None")

    # -------------------------
    # Helpers: centerline arclength + projection
    # -------------------------
    def centerline_arclength(centerline_xy):
        cl = np.asarray(centerline_xy, dtype=float)  # (N,2) [x,y]
        d = np.sqrt(np.sum(np.diff(cl, axis=0)**2, axis=1))  # (N-1,)
        s = np.concatenate([[0.0], np.cumsum(d)])            # (N,)
        return cl, s

    def project_to_centerline_s(xy_t, cl_xy, cl_s):
        xy = np.asarray(xy_t, dtype=float)
        valid = np.isfinite(xy).all(axis=1)
        s_t = np.full((xy.shape[0],), np.nan, dtype=float)
        if np.sum(valid) == 0:
            return s_t

        xv = xy[valid]  # (Tv,2)
        d2 = ((xv[:, None, :] - cl_xy[None, :, :])**2).sum(axis=2)  # (Tv,N)
        nn = np.argmin(d2, axis=1)  # (Tv,)
        s_t[valid] = cl_s[nn]
        return s_t

    # -------------------------
    # Helper: compute "turn" lines in arclength s, using an X threshold in XY space.
    # Returns (turn1_s, turn2_s, x_thresh, mask_top_turn, mask_bot_turn, valid_mask)
    # -------------------------
    def compute_turn_lines_s(loc_X, loc_Y, s_t, arm_cutoff_perc=0.8, band_frac=0.02):
        x = np.asarray(loc_X, dtype=float)
        y = np.asarray(loc_Y, dtype=float)
        s = np.asarray(s_t, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
        xv, yv, sv = x[valid], y[valid], s[valid]

        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        x_thresh = x_min + float(arm_cutoff_perc) * (x_max - x_min)

        # split top/bottom by y
        y_lo_arm = np.nanpercentile(yv, 40)
        y_hi_arm = np.nanpercentile(yv, 60)
        mask_top = (yv >= y_hi_arm)
        mask_bot = (yv <= y_lo_arm)

        x_band = float(band_frac) * (x_max - x_min)
        mask_xband = np.abs(xv - x_thresh) <= x_band

        mask_top_turn = mask_top & mask_xband
        mask_bot_turn = mask_bot & mask_xband

        # Fallback widen
        if np.sum(mask_top_turn) < 10 or np.sum(mask_bot_turn) < 10:
            band_frac2 = 0.04
            x_band = float(band_frac2) * (x_max - x_min)
            mask_xband = np.abs(xv - x_thresh) <= x_band
            mask_top_turn = mask_top & mask_xband
            mask_bot_turn = mask_bot & mask_xband

        if np.sum(mask_top_turn) < 10 or np.sum(mask_bot_turn) < 10:
            raise RuntimeError(
                f"Too few points near x_thresh={x_thresh:.2f} for top/bottom arms "
                f"(top={np.sum(mask_top_turn)}, bot={np.sum(mask_bot_turn)}). "
                f"Try increasing band_frac or adjusting arm_cutoff_perc."
            )

        turn_top_s = float(np.nanmedian(sv[mask_top_turn]))
        turn_bot_s = float(np.nanmedian(sv[mask_bot_turn]))

        turns = np.sort([turn_top_s, turn_bot_s])
        return float(turns[0]), float(turns[1]), float(x_thresh), mask_top_turn, mask_bot_turn, valid

    for mouse, group in mouse_groups.items():
        LT1 = LT1_group[mouse]
        LT2 = LT2_group[mouse]

        # Your mapping-based S extraction (kept, though you currently use LT*.S below)
        [S1, S1_spikes, S1_peakval, S1_idx] = LT1.get_S_mapping(mapping)
        [S2, S2_spikes, S2_peakval, S2_idx] = LT2.get_S_mapping(mapping)

        # Miniscope XY and "old" 1D (we won't use old 1D for plotting anymore)
        loc_X_LT1 = LT1.loc_X_miniscope_smooth
        loc_Y_LT1 = LT1.loc_Y_miniscope_smooth
        loc_X_LT2 = LT2.loc_X_miniscope_smooth
        loc_Y_LT2 = LT2.loc_Y_miniscope_smooth

        # Cross-registered cell IDs
        df_mapping = LT1.crossreg.get_mappings_cells(mapping_type=mapping)
        cells_LT1 = df_mapping[LT1.session_group].astype(float).astype(int).tolist()
        cells_LT2 = df_mapping[LT2.session_group].astype(float).astype(int).tolist()

        # Build mapping dict LT1->LT2 (used for paired plotting)
        lt1_to_lt2 = {c1: c2 for c1, c2 in zip(cells_LT1, cells_LT2)}

        # -------------------------
        # Parameters
        # -------------------------
        sort_reverse = False           # True => far end is "start"
        num_pfs_filtered = -1          # -1 => do not filter by PF count
        use_sig_responses = False      # YOUR DEFAULT NOW
        n_pos_bins = 120
        smooth_bins = 3
        normalize_per_cell = True

        # "turn" parameters
        arm_cutoff_perc = 0.8
        band_frac = 0.02

        # -------------------------
        # Centerline anchor: compute arclength s(t) for BOTH sessions
        # -------------------------
        cl1_xy, cl1_s = centerline_arclength(LT1.centerline)
        cl2_xy, cl2_s = centerline_arclength(LT2.centerline)

        XY1 = np.column_stack([loc_X_LT1, loc_Y_LT1])
        XY2 = np.column_stack([loc_X_LT2, loc_Y_LT2])

        s1 = project_to_centerline_s(XY1, cl1_xy, cl1_s)  # (T1,)
        s2 = project_to_centerline_s(XY2, cl2_xy, cl2_s)  # (T2,)

        # -------------------------
        # Use FRACTION-ALONG-CENTERLINE as the common behavioral coordinate.
        # LT1 defines the axis; LT2 is mapped onto LT1 via fractional position.
        # -------------------------
        LT1_len = float(cl1_s[-1])
        LT2_len = float(cl2_s[-1])
        if LT1_len <= 0 or LT2_len <= 0:
            raise RuntimeError("Centerline length is zero/invalid for LT1 or LT2.")

        # Convert per-frame arclength -> fraction (0..1)
        f1 = s1 / LT1_len
        f2 = s2 / LT2_len

        # Map LT2 fraction onto LT1 arclength axis
        s1_axis = s1 * _LT_1D_CM_PER_PX   # px-arclength -> cm
        s2_axis = f2 * LT1_len * _LT_1D_CM_PER_PX  # LT2 projected into LT1 arclength, in cm

        # Define bin edges over the FULL LT1 axis (not intersection)
        s_min_axis = float(np.nanmin(s1_axis[np.isfinite(s1_axis)]))
        s_max_axis = float(np.nanmax(s1_axis[np.isfinite(s1_axis)]))

        bin_edges = np.linspace(s_min_axis, s_max_axis, n_pos_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        track_ref = float(s_max_axis) if sort_reverse else float(s_min_axis)

        # -------------------------
        # Sorting LT1 cells using sig_responses-centroid per merged PF group (your latest logic)
        # but then mapping that centroid to nearest miniscope XY, then to arclength s1.
        # -------------------------

        # For converting (row_bin, col_bin) -> pixel XY
        bin_w = float(LT1.fm.loc.bin_width)
        min_x = float(getattr(LT1.fm.loc, "MIN_X"))
        min_y = float(getattr(LT1.fm.loc, "MIN_Y"))

        sorted_entries = []
        # (cell_id_LT1, chosen_grp_i, chosen_ctr_xy_px, nn_idx, nn_xy, nn_s)

        for cell_id in cells_LT1:
            merged = LT1.fm.pf.merged_means.get(cell_id, None)
            model  = LT1.fm.pf.model_.get(cell_id, None)
            sig    = LT1.fm.sig_responses.get(cell_id, None)

            if merged is None or model is None or not hasattr(model, "means_"):
                continue

            merged_means = merged
            means_rc = np.asarray(model.means_, dtype=float)  # (row, col)

            # PF-count filtering (unless disabled)
            if num_pfs_filtered != -1 and len(merged_means) != int(num_pfs_filtered):
                continue

            # mean_index -> merged_group_index
            mean_to_group = {}
            for g_idx, grp in enumerate(merged_means):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)

            # If sig missing/empty: fall back to means
            use_sig = (sig is not None and len(sig) > 0)

            pts_by_group = [[] for _ in range(len(merged_means))]

            if use_sig:
                sig_arr = np.asarray(sig, dtype=float)
                rc = sig_arr[:, :2]  # (Nsig,2) = [row, col]

                pred_mean_idx = np.asarray(model.predict(rc), dtype=int)
                for (r, c), m_idx in zip(rc, pred_mean_idx):
                    g = mean_to_group.get(int(m_idx), None)
                    if g is None:
                        continue
                    pts_by_group[g].append((float(r), float(c)))

            pf_centers = []  # (grp_i, ctr_xy_px)
            for grp_i, grp in enumerate(merged_means):
                grp = list(grp)
                if len(grp) == 0:
                    continue

                pts = pts_by_group[grp_i]
                if len(pts) > 0:
                    pts = np.asarray(pts, dtype=float)   # [row, col]
                    ctr_rc = pts.mean(axis=0)
                else:
                    ctr_rc = means_rc[grp].mean(axis=0)

                ctr_row, ctr_col = float(ctr_rc[0]), float(ctr_rc[1])
                ctr_x = min_x + (ctr_col + 0.5) * bin_w
                ctr_y = min_y + (ctr_row + 0.5) * bin_w
                ctr_xy_px = np.array([ctr_x, ctr_y], dtype=float)

                pf_centers.append((grp_i, ctr_xy_px))

            if len(pf_centers) == 0:
                continue

            # For each PF center, find nearest miniscope sample (x,y) and take its arclength s1
            pf_candidates = []
            for grp_i, ctr_xy in pf_centers:
                d2 = np.sum((XY1 - ctr_xy[None, :])**2, axis=1)
                nn_idx = int(np.nanargmin(d2))
                nn_xy = XY1[nn_idx]
                nn_s = float(s1[nn_idx]) if np.isfinite(s1[nn_idx]) else np.nan
                pf_candidates.append((grp_i, ctr_xy, nn_idx, nn_xy, nn_s))

            # drop NaN s candidates
            pf_candidates = [c for c in pf_candidates if np.isfinite(c[-1])]
            if len(pf_candidates) == 0:
                continue

            # Choose PF
            if num_pfs_filtered == -1:
                chosen = min(pf_candidates, key=lambda x: abs(x[-1] - track_ref))
            else:
                chosen = min(pf_candidates, key=lambda x: np.sum((XY1[x[2]] - x[1])**2))

            chosen_grp_i, chosen_ctr_xy, nn_idx, nn_xy, nn_s = chosen
            # Also require it maps to an LT2 cell for paired plotting
            if cell_id not in lt1_to_lt2:
                continue

            sorted_entries.append((cell_id, chosen_grp_i, chosen_ctr_xy, nn_idx, nn_xy, nn_s))

        if len(sorted_entries) == 0:
            raise RuntimeError("No cells survived sorting criteria; check PF detection, sig_responses, and mappings.")

        sorted_entries.sort(key=lambda x: x[-1], reverse=bool(sort_reverse))
        sorted_LT1 = [cell_id for (cell_id, *_rest) in sorted_entries]

        # -------------------------
        # Build paired LT1->LT2 list in LT1-sorted order (prevents row shifting)
        # -------------------------
        paired = [(c1, lt1_to_lt2[c1]) for c1 in sorted_LT1 if c1 in lt1_to_lt2]
        if len(paired) == 0:
            raise RuntimeError("No paired cross-registered cells after sorting.")

        sorted_LT1_paired = [c1 for c1, c2 in paired]
        sorted_LT2_paired = [c2 for c1, c2 in paired]

        # -------------------------
        # Extract S for each session and reindex rows by the paired order
        # -------------------------
        S_LT1 = np.asarray(LT1.S, dtype=float)  # (num_cells, T1)
        S_LT2 = np.asarray(LT2.S, dtype=float)  # (num_cells, T2)
        n_cells_1, T1 = S_LT1.shape
        n_cells_2, T2 = S_LT2.shape

        # Build cell_id -> row index maps (assume "cell id == row" when plausible; else map from cells_LT*)
        if all(isinstance(c, (int, np.integer)) for c in cells_LT1) and (len(cells_LT1) > 0) and (max(cells_LT1) < n_cells_1):
            cell_to_row_1 = {int(c): int(c) for c in cells_LT1}
        else:
            cell_to_row_1 = {cell_id: i for i, cell_id in enumerate(cells_LT1)}

        if all(isinstance(c, (int, np.integer)) for c in cells_LT2) and (len(cells_LT2) > 0) and (max(cells_LT2) < n_cells_2):
            cell_to_row_2 = {int(c): int(c) for c in cells_LT2}
        else:
            cell_to_row_2 = {cell_id: i for i, cell_id in enumerate(cells_LT2)}

        rows1 = [cell_to_row_1[c] for c in sorted_LT1_paired if c in cell_to_row_1]
        rows2 = [cell_to_row_2[c] for c in sorted_LT2_paired if c in cell_to_row_2]

        # Ensure same count; paired lists should already match, but be defensive
        n_pair = min(len(rows1), len(rows2))
        rows1 = rows1[:n_pair]
        rows2 = rows2[:n_pair]
        sorted_LT1_paired = sorted_LT1_paired[:n_pair]
        sorted_LT2_paired = sorted_LT2_paired[:n_pair]

        S1_sorted = S_LT1[rows1, :]
        S2_sorted = S_LT2[rows2, :]

        # -------------------------
        # Optional sig-response filtering (OFF by default)
        # NOTE: This version filters per-session separately.
        # -------------------------
        if use_sig_responses:
            def filter_S_by_sig(LT, S_in, loc_X, loc_Y, cells_session):
                sig_radius_bins = 1
                fallback_to_full_if_missing = True

                bin_w_loc = float(LT.fm.loc.bin_width)
                min_x_loc = float(getattr(LT.fm.loc, "MIN_X"))
                min_y_loc = float(getattr(LT.fm.loc, "MIN_Y"))

                x_raw = np.asarray(loc_X, dtype=float)
                y_raw = np.asarray(loc_Y, dtype=float)
                valid_xy = np.isfinite(x_raw) & np.isfinite(y_raw)

                x_bin = np.full_like(x_raw, fill_value=-1, dtype=int)
                y_bin = np.full_like(y_raw, fill_value=-1, dtype=int)
                x_bin[valid_xy] = np.floor((x_raw[valid_xy] - min_x_loc) / bin_w_loc).astype(int)
                y_bin[valid_xy] = np.floor((y_raw[valid_xy] - min_y_loc) / bin_w_loc).astype(int)

                nx, ny = LT.fm.fluorescence_map_occup.shape[:2]
                x_bin = np.clip(x_bin, 0, nx - 1)
                y_bin = np.clip(y_bin, 0, ny - 1)

                S_filt = np.full_like(S_in, np.nan, dtype=float)

                idx_valid_frames = np.where(valid_xy)[0]
                if idx_valid_frames.size == 0:
                    raise RuntimeError("No valid XY frames to match sig_responses against.")

                xv = x_bin[idx_valid_frames]
                yv = y_bin[idx_valid_frames]
                r2 = int(sig_radius_bins) ** 2

                # Here cells_session are the cell IDs corresponding to rows in S_in
                for row_i, cell_id in enumerate(cells_session):
                    sig = LT.fm.sig_responses.get(cell_id, None)
                    if sig is None or len(sig) == 0:
                        if fallback_to_full_if_missing:
                            S_filt[row_i, :] = S_in[row_i, :]
                        continue

                    targets = np.array([(int(s[0]), int(s[1])) for s in sig], dtype=int)  # (K,2) row,col
                    ty = targets[:, 0][None, :]
                    tx = targets[:, 1][None, :]

                    # Convert frame bins to (row,col) too:
                    # x_bin is col, y_bin is row in your convention for sig_responses
                    frame_row = yv
                    frame_col = xv

                    d2 = (frame_row[:, None] - ty)**2 + (frame_col[:, None] - tx)**2
                    min_d2 = np.min(d2, axis=1)
                    keep_mask = (min_d2 <= r2)
                    keep_frames = idx_valid_frames[keep_mask]

                    if keep_frames.size == 0:
                        nearest_per_target = []
                        for k in range(targets.shape[0]):
                            d2k = (frame_row - targets[k, 0])**2 + (frame_col - targets[k, 1])**2
                            j = int(np.argmin(d2k))
                            nearest_per_target.append(int(idx_valid_frames[j]))
                        keep_frames = np.array(sorted(set(nearest_per_target)), dtype=int)

                    if keep_frames.size > 0:
                        S_filt[row_i, keep_frames] = S_in[row_i, keep_frames]

                return S_filt

            # Filter in the paired row-space
            S1_sorted = filter_S_by_sig(LT1, S1_sorted, loc_X_LT1, loc_Y_LT1, sorted_LT1_paired)
            S2_sorted = filter_S_by_sig(LT2, S2_sorted, loc_X_LT2, loc_Y_LT2, sorted_LT2_paired)

        print("LT1_len, LT2_len:", LT1_len, LT2_len)
        print("s1_axis finite:", np.isfinite(s1_axis).sum(), "/", s1_axis.size)
        print("s2_axis finite:", np.isfinite(s2_axis).sum(), "/", s2_axis.size)
        print("axis range:", s_min_axis, s_max_axis)

        # -------------------------
        # Compute tuning curves for a session given (S_sorted, s(t), bin_edges)
        # IMPORTANT: we DROp frames whose s is outside shared range.
        # -------------------------
        def tuning_from_S_and_s(S_sorted, s_t, bin_edges, smooth_bins=3):
            s = np.asarray(s_t, dtype=float)
            valid = np.isfinite(s)
            s_valid = s[valid]
            S_valid = S_sorted[:, valid]

            # Keep only within shared range
            in_range = (s_valid >= bin_edges[0]) & (s_valid <= bin_edges[-1])
            s_use = s_valid[in_range]
            S_use = S_valid[:, in_range]

            # digitize into shared bins
            bin_idx = np.digitize(s_use, bin_edges) - 1
            n_bins = len(bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            tuning = np.full((S_use.shape[0], n_bins), np.nan, dtype=float)
            for b in range(n_bins):
                m = (bin_idx == b)
                if not np.any(m):
                    continue
                tuning[:, b] = np.nanmean(S_use[:, m], axis=1)

            # spatial smoothing
            if smooth_bins is not None and smooth_bins > 1:
                k = np.ones(int(smooth_bins), dtype=float) / float(smooth_bins)
                pad = int(smooth_bins) // 2
                tuning_pad = np.pad(tuning, ((0, 0), (pad, pad)), mode="edge")
                tuning = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, tuning_pad)

            return tuning
        
        tuning1 = tuning_from_S_and_s(S1_sorted, s1_axis, bin_edges, smooth_bins=smooth_bins)
        tuning2 = tuning_from_S_and_s(S2_sorted, s2_axis, bin_edges, smooth_bins=smooth_bins)

        # Normalize per cell (for visualization)
        tuning1_plot = tuning1.copy()
        tuning2_plot = tuning2.copy()
        if normalize_per_cell:
            r1 = np.nanmax(tuning1_plot, axis=1, keepdims=True); r1[r1 == 0] = np.nan
            r2 = np.nanmax(tuning2_plot, axis=1, keepdims=True); r2[r2 == 0] = np.nan
            tuning1_plot = tuning1_plot / r1
            tuning2_plot = tuning2_plot / r2

        # -------------------------
        # Turn lines for BOTH sessions in ARCLENGTH s (for overlaid dashed lines)
        # -------------------------
        turn1_s_LT1, turn2_s_LT1, x_thresh1, mtop1, mbot1, valid1 = compute_turn_lines_s(
            loc_X_LT1, loc_Y_LT1, s1, arm_cutoff_perc=arm_cutoff_perc, band_frac=band_frac
        )
        turn1_s_LT2, turn2_s_LT2, x_thresh2, mtop2, mbot2, valid2 = compute_turn_lines_s(
            loc_X_LT2, loc_Y_LT2, s2, arm_cutoff_perc=arm_cutoff_perc, band_frac=band_frac
        )

        # Convert LT2 turn locations (in LT2 arclength) into LT1-axis coordinate, then to cm
        turn1_s_LT1 = turn1_s_LT1 * _LT_1D_CM_PER_PX
        turn2_s_LT1 = turn2_s_LT1 * _LT_1D_CM_PER_PX
        turn1_s_LT2 = (turn1_s_LT2 / LT2_len) * LT1_len * _LT_1D_CM_PER_PX
        turn2_s_LT2 = (turn2_s_LT2 / LT2_len) * LT1_len * _LT_1D_CM_PER_PX

        # -------------------------
        # Plot two subplots (LT1 + LT2) with shared behavioral axis (arclength)
        # No colorbar; NaNs shown as black (not white slabs).
        # -------------------------
        cmap = mpl.cm.get_cmap("viridis").copy()
        cmap.set_bad(color="black")

        vmin, vmax = (0.0, 1.0) if normalize_per_cell else (None, None)

        fig, axes = plt.subplots(
            1, 2, figsize=(8.2, 7.5), dpi=200, sharey=True,
            gridspec_kw={"wspace": 0.08}
        )

        x_left, x_right = float(bin_edges[0]), float(bin_edges[-1])

        axes[0].imshow(
            tuning1_plot,
            aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[x_left, x_right, tuning1_plot.shape[0], 1]
        )
        axes[0].set_title(f"{LT1.session_group}")
        axes[0].set_xlabel(f"Track arclength ({_LT_1D_DISTANCE_UNIT})")
        axes[0].set_ylabel("Place cells")
        axes[0].set_yticks([1, tuning1_plot.shape[0]])
        axes[0].set_yticklabels(["1", f"{tuning1_plot.shape[0]}"])
        axes[0].axvline(turn1_s_LT1, linestyle="--", linewidth=1.5)
        axes[0].axvline(turn2_s_LT1, linestyle="--", linewidth=1.5)

        axes[1].imshow(
            tuning2_plot,
            aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[x_left, x_right, tuning2_plot.shape[0], 1]
        )
        axes[1].set_title(f"{LT2.session_group}")
        axes[1].set_xlabel(f"Track arclength ({_LT_1D_DISTANCE_UNIT})")
        axes[1].set_yticks([1, tuning2_plot.shape[0]])
        axes[1].set_yticklabels(["1", f"{tuning2_plot.shape[0]}"])
        axes[1].axvline(turn1_s_LT2, linestyle="--", linewidth=1.5)
        axes[1].axvline(turn2_s_LT2, linestyle="--", linewidth=1.5)

        plt.tight_layout()
        plt.show()

        # -------------------------
        # Sanity plots in XY (optional): show x_thresh and points used
        # (Kept similar to your earlier sanity check)
        # -------------------------
        # LT1 sanity
        x1 = np.asarray(loc_X_LT1, dtype=float)
        y1 = np.asarray(loc_Y_LT1, dtype=float)
        v1 = valid1
        xv1, yv1 = x1[v1], y1[v1]

        plt.figure(figsize=(10, 3), dpi=150)
        plt.scatter(xv1, yv1, s=4, alpha=0.25)
        plt.axvline(x_thresh1, linestyle="--", linewidth=2)
        # highlight points used (optional; you said you don't care about colors much, but leaving it informative)
        plt.scatter(xv1[mtop1], yv1[mtop1], s=10, alpha=0.8, label="Top-arm near x_thresh")
        plt.scatter(xv1[mbot1], yv1[mbot1], s=10, alpha=0.8, label="Bottom-arm near x_thresh")
        plt.title(f"LT1 turn cutoff sanity check (arm_cutoff_perc={arm_cutoff_perc:.2f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper left", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.show()

        # LT2 sanity
        x2 = np.asarray(loc_X_LT2, dtype=float)
        y2 = np.asarray(loc_Y_LT2, dtype=float)
        v2 = valid2
        xv2, yv2 = x2[v2], y2[v2]

        plt.figure(figsize=(10, 3), dpi=150)
        plt.scatter(xv2, yv2, s=4, alpha=0.25)
        plt.axvline(x_thresh2, linestyle="--", linewidth=2)
        plt.scatter(xv2[mtop2], yv2[mtop2], s=10, alpha=0.8, label="Top-arm near x_thresh")
        plt.scatter(xv2[mbot2], yv2[mbot2], s=10, alpha=0.8, label="Bottom-arm near x_thresh")
        plt.title(f"LT2 turn cutoff sanity check (arm_cutoff_perc={arm_cutoff_perc:.2f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper left", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.show()

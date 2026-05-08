
import os
import numpy as np
import matplotlib.pyplot as plt
from SSTCa2_decoder import _LT_1D_CM_PER_PX, _LT_1D_DISTANCE_UNIT

def plot_lt_spatial_responses(PLOTS_DIR, LT1_group, LT2_group, mouse_groups, session_str, mapping=None):
    '''
    Plot spatial responses for cross-registered cells between LT1 and LT2, sorted according to LT1.
    Produces a 2-panel figure:
      - Left: LT1 tuning curves (rows sorted by LT1 PF center)
      - Right: LT2 tuning curves for the SAME cells, shown in the SAME row order (LT1-sorted)
    '''
    if mapping is None:
        raise ValueError("plot_lt_spatial_responses: mapping must not be None")

    for mouse, group in mouse_groups.items():
        LT1 = LT1_group[mouse]
        LT2 = LT2_group[mouse]

        # If you use these elsewhere; not required for current plotting blocks
        [S1, S1_spikes, S1_peakval, S1_idx] = LT1.get_S_mapping(mapping)
        [S2, S2_spikes, S2_peakval, S2_idx] = LT2.get_S_mapping(mapping)

        # -------------------------
        # Inputs
        # -------------------------
        loc_X_LT1 = LT1.loc_X_miniscope_smooth
        loc_Y_LT1 = LT1.loc_Y_miniscope_smooth
        loc_1d_LT1 = LT1.miniscope_loc_1d_px * _LT_1D_CM_PER_PX

        loc_X_LT2 = LT2.loc_X_miniscope_smooth
        loc_Y_LT2 = LT2.loc_Y_miniscope_smooth
        loc_1d_LT2 = LT2.miniscope_loc_1d_px * _LT_1D_CM_PER_PX

        df_mapping = LT1.crossreg.get_mappings_cells(mapping_type=mapping)
        cells_LT1 = df_mapping[LT1.session_group].astype(float).astype(int).tolist()
        cells_LT2 = df_mapping[LT2.session_group].astype(float).astype(int).tolist()

        # -------------------------
        # Global toggles / defaults
        # -------------------------
        use_sig_responses = False  # default from now on (as you requested)

        # Sorting parameters
        sort_reverse = False          # True => max loc_1d is "start"
        num_pfs_filtered = -1         # -1 => do NOT filter by PF count (keep all)

        # Tuning curve parameters
        n_pos_bins = 120          # tweak (e.g., 60-200)
        smooth_bins = 3           # spatial smoothing width in bins; set 1 to disable
        normalize_per_cell = True # for display (each row max -> 1)

        # Turn-line parameters
        arm_cutoff_perc = 0.8     # define "turn begins" at this fraction along x-range
        band_frac = 0.02          # band around x_thresh as a fraction of x-range (fallback widens)

        # -------------------------
        # Build helper arrays
        # -------------------------
        XY_LT1 = np.column_stack([loc_X_LT1, loc_Y_LT1])  # (T, 2)

        # Track reference point (beginning or end)
        track_ref = np.nanmax(loc_1d_LT1) if sort_reverse else np.nanmin(loc_1d_LT1)

        # For converting (row_bin, col_bin) -> pixel XY (used only for sorting PF centers)
        bin_w = float(LT1.fm.loc.bin_width)
        min_x = float(getattr(LT1.fm.loc, "MIN_X"))
        min_y = float(getattr(LT1.fm.loc, "MIN_Y"))

        # -------------------------
        # SORTING: produce sorted_LT1
        # Uses PF centers anchored to sig-response centroids (when available), else model means
        # -------------------------
        sorted_entries = []
        # tuples:
        # (cell_id, chosen_pf_group_idx, chosen_pf_center_xy_px, nn_time_idx, nn_xy, nn_loc1d)

        for cell_id in cells_LT1:
            merged = LT1.fm.pf.merged_means.get(cell_id, None)
            model  = LT1.fm.pf.model_.get(cell_id, None)
            sig    = LT1.fm.sig_responses.get(cell_id, None)

            if merged is None or model is None or not hasattr(model, "means_"):
                continue

            merged_means = merged
            means_rc = np.asarray(model.means_, dtype=float)  # (row, col)

            # PF-count filtering (unless disabled)
            if num_pfs_filtered != -1:
                if len(merged_means) != int(num_pfs_filtered):
                    continue

            # mean_index -> merged_group_index
            mean_to_group = {}
            for g_idx, grp in enumerate(merged_means):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)

            # Collect sig points by merged group (sig is [row, col, c], ignore c)
            use_sig = (sig is not None and len(sig) > 0)
            pts_by_group = [[] for _ in range(len(merged_means))]

            if use_sig:
                sig = np.asarray(sig, dtype=float)
                rc = sig[:, :2]  # (Nsig,2) [row,col]
                pred_mean_idx = np.asarray(model.predict(rc), dtype=int)

                for (r, c), m_idx in zip(rc, pred_mean_idx):
                    g = mean_to_group.get(int(m_idx), None)
                    if g is None:
                        continue
                    pts_by_group[g].append((float(r), float(c)))

            # PF centers: centroid of sig points per merged group (fallback to means_rc group centroid)
            pf_centers = []  # list of (grp_i, ctr_xy_px)
            for grp_i, grp in enumerate(merged_means):
                grp = list(grp)
                if len(grp) == 0:
                    continue

                pts = pts_by_group[grp_i]
                if len(pts) > 0:
                    pts = np.asarray(pts, dtype=float)   # [row,col]
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

            # For each PF center, find nearest miniscope (x,y) sample, then map to 1D
            pf_candidates = []
            for grp_i, ctr_xy in pf_centers:
                d2 = np.sum((XY_LT1 - ctr_xy[None, :])**2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_xy = XY_LT1[nn_idx]
                nn_loc1d = float(loc_1d_LT1[nn_idx])
                pf_candidates.append((grp_i, ctr_xy, nn_idx, nn_xy, nn_loc1d))

            if len(pf_candidates) == 0:
                continue

            # PF selection logic
            if num_pfs_filtered == -1:
                chosen = min(pf_candidates, key=lambda x: abs(x[-1] - track_ref))
            else:
                chosen = min(pf_candidates, key=lambda x: np.sum((XY_LT1[x[2]] - x[1])**2))

            chosen_grp_i, chosen_ctr_xy, nn_idx, nn_xy, nn_loc1d = chosen
            sorted_entries.append((cell_id, chosen_grp_i, chosen_ctr_xy, nn_idx, nn_xy, nn_loc1d))

        # Sort by LT1 linearized position
        sorted_entries.sort(key=lambda x: x[-1], reverse=bool(sort_reverse))
        sorted_LT1 = [cell_id for (cell_id, *_rest) in sorted_entries]

        # -------------------------
        # Turn-lines for LT1 (computed once; used for LT1 subplot)
        # -------------------------
        x1 = np.asarray(loc_X_LT1, dtype=float)
        y1 = np.asarray(loc_Y_LT1, dtype=float)
        p1d1 = np.asarray(loc_1d_LT1, dtype=float)

        valid1_xy = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(p1d1)
        xv1, yv1, p1dv1 = x1[valid1_xy], y1[valid1_xy], p1d1[valid1_xy]

        x1_min, x1_max = np.nanmin(xv1), np.nanmax(xv1)
        x1_thresh = x1_min + float(arm_cutoff_perc) * (x1_max - x1_min)

        y1_lo_arm = np.nanpercentile(yv1, 40)
        y1_hi_arm = np.nanpercentile(yv1, 60)
        mask1_top = yv1 >= y1_hi_arm
        mask1_bot = yv1 <= y1_lo_arm

        x1_band = band_frac * (x1_max - x1_min)
        mask1_xband = np.abs(xv1 - x1_thresh) <= x1_band
        mask1_top_turn = mask1_top & mask1_xband
        mask1_bot_turn = mask1_bot & mask1_xband

        if np.sum(mask1_top_turn) < 10 or np.sum(mask1_bot_turn) < 10:
            x1_band = 0.04 * (x1_max - x1_min)
            mask1_xband = np.abs(xv1 - x1_thresh) <= x1_band
            mask1_top_turn = mask1_top & mask1_xband
            mask1_bot_turn = mask1_bot & mask1_xband

        #if np.sum(mask1_top_turn) < 10 or np.sum(mask1_bot_turn) < 10:
        #    raise RuntimeError(
        #        f"Too few points near LT1 x_thresh={x1_thresh:.2f} "
        #        f"(top={np.sum(mask1_top_turn)}, bot={np.sum(mask1_bot_turn)})."
        #    )

        turn1_top_1d = float(np.nanmedian(p1dv1[mask1_top_turn]))
        turn1_bot_1d = float(np.nanmedian(p1dv1[mask1_bot_turn]))
        turns1 = np.sort([turn1_top_1d, turn1_bot_1d])
        lt1_turn_a, lt1_turn_b = float(turns1[0]), float(turns1[1])

        # -------------------------
        # Turn-lines for LT2 (computed once; used for LT2 subplot)
        # -------------------------
        x2 = np.asarray(loc_X_LT2, dtype=float)
        y2 = np.asarray(loc_Y_LT2, dtype=float)
        p1d2 = np.asarray(loc_1d_LT2, dtype=float)

        valid2_xy = np.isfinite(x2) & np.isfinite(y2) & np.isfinite(p1d2)
        xv2, yv2, p1dv2 = x2[valid2_xy], y2[valid2_xy], p1d2[valid2_xy]

        x2_min, x2_max = np.nanmin(xv2), np.nanmax(xv2)
        x2_thresh = x2_min + float(arm_cutoff_perc) * (x2_max - x2_min)

        y2_lo_arm = np.nanpercentile(yv2, 40)
        y2_hi_arm = np.nanpercentile(yv2, 60)
        mask2_top = yv2 >= y2_hi_arm
        mask2_bot = yv2 <= y2_lo_arm

        x2_band = band_frac * (x2_max - x2_min)
        mask2_xband = np.abs(xv2 - x2_thresh) <= x2_band
        mask2_top_turn = mask2_top & mask2_xband
        mask2_bot_turn = mask2_bot & mask2_xband

        if np.sum(mask2_top_turn) < 10 or np.sum(mask2_bot_turn) < 10:
            x2_band = 0.04 * (x2_max - x2_min)
            mask2_xband = np.abs(xv2 - x2_thresh) <= x2_band
            mask2_top_turn = mask2_top & mask2_xband
            mask2_bot_turn = mask2_bot & mask2_xband

        if np.sum(mask2_top_turn) < 10 or np.sum(mask2_bot_turn) < 10:
            raise RuntimeError(
                f"Too few points near LT2 x_thresh={x2_thresh:.2f} "
                f"(top={np.sum(mask2_top_turn)}, bot={np.sum(mask2_bot_turn)})."
            )

        turn2_top_1d = float(np.nanmedian(p1dv2[mask2_top_turn]))
        turn2_bot_1d = float(np.nanmedian(p1dv2[mask2_bot_turn]))
        turns2 = np.sort([turn2_top_1d, turn2_bot_1d])
        lt2_turn_a, lt2_turn_b = float(turns2[0]), float(turns2[1])

        # -------------------------
        # Build LT1 tuning plot (rows sorted by sorted_LT1)
        # -------------------------
        S_LT1 = np.asarray(LT1.C)            # (num_cells, time)
        pos1 = np.asarray(loc_1d_LT1, float) # (time,)
        n_cells_S1, T1 = S_LT1.shape

        # Row indices for LT1
        if all(isinstance(c, (int, np.integer)) for c in sorted_LT1) and max(sorted_LT1) < n_cells_S1:
            sorted_row_idx_LT1 = [int(c) for c in sorted_LT1]
        else:
            cell_to_row_LT1 = {cell_id: i for i, cell_id in enumerate(cells_LT1)}
            sorted_row_idx_LT1 = [cell_to_row_LT1[c] for c in sorted_LT1 if c in cell_to_row_LT1]

        S1_sorted = S_LT1[sorted_row_idx_LT1, :]

        valid_pos1 = np.isfinite(pos1)
        pos1_valid = pos1[valid_pos1]
        S1_valid = S1_sorted[:, valid_pos1]

        pos1_min = np.nanmin(pos1_valid)
        pos1_max = np.nanmax(pos1_valid)

        bin_edges1 = np.linspace(pos1_min, pos1_max, n_pos_bins + 1)
        bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])

        bin_idx1 = np.digitize(pos1_valid, bin_edges1) - 1
        bin_idx1 = np.clip(bin_idx1, 0, n_pos_bins - 1)

        tuning1 = np.full((S1_valid.shape[0], n_pos_bins), np.nan, float)
        for b in range(n_pos_bins):
            m = (bin_idx1 == b)
            if not np.any(m):
                continue
            tuning1[:, b] = np.nanmean(S1_valid[:, m], axis=1)

        if smooth_bins is not None and smooth_bins > 1:
            k = np.ones(smooth_bins, float) / smooth_bins
            pad = smooth_bins // 2
            tuning1_pad = np.pad(tuning1, ((0, 0), (pad, pad)), mode="edge")
            tuning1 = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, tuning1_pad)

        tuning1_plot = tuning1.copy()
        if normalize_per_cell:
            row_max = np.nanmax(tuning1_plot, axis=1, keepdims=True)
            row_max[row_max == 0] = np.nan
            tuning1_plot = tuning1_plot / row_max

        # -------------------------
        # Build LT2 tuning plot in the SAME cell order as LT1 (via mapping)
        # -------------------------
        S_LT2 = np.asarray(LT2.C)
        pos2 = np.asarray(loc_1d_LT2, float)
        n_cells_S2, T2 = S_LT2.shape

        lt1_to_lt2 = {c1: c2 for c1, c2 in zip(cells_LT1, cells_LT2)}
        sorted_LT2 = [lt1_to_lt2[c1] for c1 in sorted_LT1 if c1 in lt1_to_lt2]

        # Row indices for LT2
        if len(sorted_LT2) == 0:
            raise RuntimeError("sorted_LT2 is empty after mapping LT1->LT2. Check crossreg mapping lists.")

        if all(isinstance(c, (int, np.integer)) for c in sorted_LT2) and max(sorted_LT2) < n_cells_S2:
            sorted_row_idx_LT2 = [int(c) for c in sorted_LT2]
        else:
            cell_to_row_LT2 = {cell_id: i for i, cell_id in enumerate(cells_LT2)}
            sorted_row_idx_LT2 = [cell_to_row_LT2[c] for c in sorted_LT2 if c in cell_to_row_LT2]

        S2_sorted = S_LT2[sorted_row_idx_LT2, :]

        valid_pos2 = np.isfinite(pos2)
        pos2_valid = pos2[valid_pos2]
        S2_valid = S2_sorted[:, valid_pos2]

        pos2_min = np.nanmin(pos2_valid)
        pos2_max = np.nanmax(pos2_valid)

        bin_edges2 = np.linspace(pos2_min, pos2_max, n_pos_bins + 1)
        bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

        bin_idx2 = np.digitize(pos2_valid, bin_edges2) - 1
        bin_idx2 = np.clip(bin_idx2, 0, n_pos_bins - 1)

        tuning2 = np.full((S2_valid.shape[0], n_pos_bins), np.nan, float)
        for b in range(n_pos_bins):
            m = (bin_idx2 == b)
            if not np.any(m):
                continue
            tuning2[:, b] = np.nanmean(S2_valid[:, m], axis=1)

        if smooth_bins is not None and smooth_bins > 1:
            k = np.ones(smooth_bins, float) / smooth_bins
            pad = smooth_bins // 2
            tuning2_pad = np.pad(tuning2, ((0, 0), (pad, pad)), mode="edge")
            tuning2 = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, tuning2_pad)

        tuning2_plot = tuning2.copy()
        if normalize_per_cell:
            row_max = np.nanmax(tuning2_plot, axis=1, keepdims=True)
            row_max[row_max == 0] = np.nan
            tuning2_plot = tuning2_plot / row_max

        # Ensure same number of rows (in case of partial mapping)
        n_rows = min(tuning1_plot.shape[0], tuning2_plot.shape[0])
        tuning1_plot = tuning1_plot[:n_rows, :]
        tuning2_plot = tuning2_plot[:n_rows, :]
        
        # -------------------------
        # Two-panel plot (no colorbar; robust extent; NaNs not white)
        # -------------------------
        import matplotlib as mpl

        # Make NaNs show as black instead of white
        cmap = mpl.cm.get_cmap("viridis").copy()
        cmap.set_bad(color="black")   # or set_bad((0,0,0,0)) for transparent

        # Use bin EDGES for correct extent (not centers)
        x1_left, x1_right = float(bin_edges1[0]), float(bin_edges1[-1])
        x2_left, x2_right = float(bin_edges2[0]), float(bin_edges2[-1])

        fig, axes = plt.subplots(
            1, 2,
            figsize=(8.2, 7.5),
            dpi=200,
            sharey=True,
            gridspec_kw={"wspace": 0.08}  # tighter gap
        )

        # If normalized_per_cell=True, force consistent scaling
        vmin, vmax = (0.0, 1.0) if normalize_per_cell else (None, None)

        im1 = axes[0].imshow(
            tuning1_plot,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            extent=[x1_left, x1_right, tuning1_plot.shape[0], 1],
        )
        axes[0].set_title(f"{LT1.session_group}")
        axes[0].set_xlabel(f"Linearized track position ({_LT_1D_DISTANCE_UNIT})")
        axes[0].set_ylabel("Place cells")
        axes[0].set_yticks([1, tuning1_plot.shape[0]])
        axes[0].set_yticklabels(["1", f"{tuning1_plot.shape[0]}"])
        axes[0].axvline(lt1_turn_a, linestyle="--", linewidth=1.5)
        axes[0].axvline(lt1_turn_b, linestyle="--", linewidth=1.5)

        im2 = axes[1].imshow(
            tuning2_plot,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            extent=[x2_left, x2_right, tuning2_plot.shape[0], 1],
        )
        axes[1].set_title(f"{LT2.session_group}")
        axes[1].set_xlabel(f"Linearized track position ({_LT_1D_DISTANCE_UNIT})")
        axes[1].set_yticks([1, tuning2_plot.shape[0]])
        axes[1].set_yticklabels(["1", f"{tuning2_plot.shape[0]}"])
        axes[1].axvline(lt2_turn_a, linestyle="--", linewidth=1.5)
        axes[1].axvline(lt2_turn_b, linestyle="--", linewidth=1.5)

        plt.tight_layout()
        plt.show()


        # -------------------------
        # Optional sanity plot in XY for LT1 turn threshold (kept from your prior code)
        # -------------------------
        plt.figure(figsize=(10, 3), dpi=150)
        plt.scatter(xv1, yv1, s=4, alpha=0.25)
        plt.axvline(x1_thresh, linestyle="--", linewidth=2)
        plt.scatter(xv1[mask1_top_turn], yv1[mask1_top_turn], s=10, alpha=0.8, label="LT1 top-arm near x_thresh")
        plt.scatter(xv1[mask1_bot_turn], yv1[mask1_bot_turn], s=10, alpha=0.8, label="LT1 bottom-arm near x_thresh")
        plt.title("LT1 turn cutoff sanity check (x-threshold + selected points)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper left", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.show()

        # Optional sanity plot in XY for LT2 turn threshold
        plt.figure(figsize=(10, 3), dpi=150)
        plt.scatter(xv2, yv2, s=4, alpha=0.25)
        plt.axvline(x2_thresh, linestyle="--", linewidth=2)
        plt.scatter(xv2[mask2_top_turn], yv2[mask2_top_turn], s=10, alpha=0.8, label="LT2 top-arm near x_thresh")
        plt.scatter(xv2[mask2_bot_turn], yv2[mask2_bot_turn], s=10, alpha=0.8, label="LT2 bottom-arm near x_thresh")
        plt.title("LT2 turn cutoff sanity check (x-threshold + selected points)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper left", fontsize=8, frameon=False)
        plt.tight_layout()
        plt.show()


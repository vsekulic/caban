"""One-time script to fix the duplicated within-session CV block."""
import re

with open('SSTCa2_decoder.py', 'r', encoding='utf-8') as f:
    content = f.read()

# -----------------------------------------------------------------------
# Fix 1: duplicated within-CV block in run_2D_multi_target_all_mice
# -----------------------------------------------------------------------
# The bad block has the old call immediately followed by the new call.
OLD_MULTI = (
    "        within = _within_session_2D_cv(S_tr, x_tr, y_tr, mask_tr, dt_train,\n"
    "                                        n_x_bins=n_x_bins, n_y_bins=n_y_bins,\n"
    "                                        time_bin_frames=time_bin_frames,\n"
    "                                        use_posterior_mean=use_posterior_mean)\n"
    "        mouse_res[\"within_train\"] = within\n"
    "                            within = _within_session_2D_cv(S_tr, x_tr, y_tr, mask_tr, dt_train,\n"
    "                                            n_x_bins=n_x_bins, n_y_bins=n_y_bins,\n"
    "                                            time_bin_frames=time_bin_frames,\n"
    "                                            use_posterior_mean=use_posterior_mean,\n"
    "                                            decoder_type=decoder_type,\n"
    "                                            ridge_alpha=ridge_alpha)\n"
    "                            mouse_res[\"within_train\"] = within"
)
NEW_MULTI = (
    "        within = _within_session_2D_cv(S_tr, x_tr, y_tr, mask_tr, dt_train,\n"
    "                                        n_x_bins=n_x_bins, n_y_bins=n_y_bins,\n"
    "                                        time_bin_frames=time_bin_frames,\n"
    "                                        use_posterior_mean=use_posterior_mean,\n"
    "                                        decoder_type=decoder_type,\n"
    "                                        ridge_alpha=ridge_alpha)\n"
    "        mouse_res[\"within_train\"] = within"
)

if OLD_MULTI in content:
    content = content.replace(OLD_MULTI, NEW_MULTI, 1)
    print('Fixed: run_2D_multi_target_all_mice within-CV block')
else:
    print('WARNING: run_2D_multi_target_all_mice block NOT found')

# -----------------------------------------------------------------------
# Fix 2: duplicated within-CV block in run_2D_pooled_target_all_mice
# -----------------------------------------------------------------------
OLD_POOL = (
    "        within = _within_session_2D_cv(S_tr, x_tr, y_tr, mask_tr, dt_train,\n"
    "                                        n_x_bins=n_x_bins, n_y_bins=n_y_bins,\n"
    "                                        time_bin_frames=time_bin_frames,\n"
    "                                        use_posterior_mean=use_posterior_mean)\n"
    "        mouse_res[\"within_train\"] = within\n"
    "\n"
    "        if within.get(\"ok\", False):"
)
# NOTE: this pattern might also appear in run_2D_pooled_target_all_mice -
# that function's within-CV call doesn't yet have decoder_type threaded through.
# Let's just check whether it still has the old-style call.
if OLD_POOL in content:
    # Replace ONLY the run_2D_pooled_target_all_mice instance
    NEW_POOL = (
        "        within = _within_session_2D_cv(S_tr, x_tr, y_tr, mask_tr, dt_train,\n"
        "                                        n_x_bins=n_x_bins, n_y_bins=n_y_bins,\n"
        "                                        time_bin_frames=time_bin_frames,\n"
        "                                        use_posterior_mean=use_posterior_mean,\n"
        "                                        decoder_type=decoder_type,\n"
        "                                        ridge_alpha=ridge_alpha)\n"
        "        mouse_res[\"within_train\"] = within\n"
        "\n"
        "        if within.get(\"ok\", False):"
    )
    content = content.replace(OLD_POOL, NEW_POOL, 1)
    print('Fixed: run_2D_pooled_target_all_mice within-CV block')
else:
    print('INFO: run_2D_pooled_target_all_mice block pattern not found (may already be correct)')

with open('SSTCa2_decoder.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done.')

mapping = 'TFC_cond+Test_B+Test_B_1wk'
mouse = 'G05'
[S_TFC_cond, S_spikes_TFC_cond, S_peakval_TFC_cond, S_idx_TFC_cond] = \
    TFC_cond[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])
[S_Test_B, S_spikes_Test_B, S_peakval_Test_B, S_idx_Test_B] = \
    Test_B[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])
[S_Test_B_1wk, S_spikes_Test_B_1wk, S_peakval_Test_B_1wk, S_idx_Test_B_1wk] = \
    Test_B_1wk[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])

TFC_cond_indeces_into_S = TFC_cond[mouse].get_S_indeces(S_idx_TFC_cond)
Test_B_indeces_into_S = Test_B[mouse].get_S_indeces(S_idx_Test_B)
Test_B_1wk_indeces_into_S = Test_B_1wk[mouse].get_S_indeces(S_idx_Test_B_1wk)

i=1
my_sigma = 2

i_TFC_cond = TFC_cond_indeces_into_S[i]
i_Test_B = Test_B_indeces_into_S[i]
i_Test_B_1wk = Test_B_1wk_indeces_into_S[i]

# for    cell selection instead of crossreg
'''
j = 200
i_TFC_cond = j
i_Test_B = j
i_Test_B_1wk = j
'''


mouse='G09'

S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)
S_i_Test_B = get_S_indeces_crossreg(Test_B[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)
S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)

my_sigma = 1
cell_=12

TFC_cond[mouse].fm.check_fluorescence_map()
fm_TFC_cond = TFC_cond[mouse].fm
#fm_cells_TFC_cond = TFC_cond[mouse].fm.fluorescence_map[:,:,np.where(S_i_TFC_cond[cell_])[0][0]]
fm_cells_TFC_cond = TFC_cond[mouse].fm.fluorescence_map[:,:,S_i_TFC_cond[cell_]]
fm_cells_TFC_cond_gauss = gaussian_filter(fm_cells_TFC_cond, my_sigma)

Test_B[mouse].fm.check_fluorescence_map()
fm_Test_B = Test_B[mouse].fm
#fm_cells_Test_B = Test_B[mouse].fm.fluorescence_map[:,:,np.where(S_i_Test_B[cell_])[0][0]]
fm_cells_Test_B = Test_B[mouse].fm.fluorescence_map[:,:,S_i_Test_B[cell_]]
fm_cells_Test_B_gauss = gaussian_filter(fm_cells_Test_B, my_sigma)

Test_B_1wk[mouse].fm.check_fluorescence_map()
fm_Test_B_1wk = Test_B_1wk[mouse].fm
#fm_cells_Test_B_1wk = Test_B_1wk[mouse].fm.fluorescence_map[:,:,np.where(S_i_Test_B_1wk[cell_])[0][0]]
fm_cells_Test_B_1wk = Test_B_1wk[mouse].fm.fluorescence_map[:,:,S_i_Test_B_1wk[cell_]]
fm_cells_Test_B_1wk_gauss = gaussian_filter(fm_cells_Test_B_1wk, my_sigma)

plt.figure()
#plt.imshow(fm_cells_TFC_cond_gauss)
plt.imshow(fm_cells_TFC_cond)
plt.title('TFC_cond mouse {} cell {} crossreg_cell {}'.format(mouse, S_i_TFC_cond[cell_], cell_))

plt.figure()
#plt.imshow(fm_cells_Test_B_gauss)
plt.imshow(fm_cells_Test_B)
plt.title('Test_B mouse {} cell {} crossreg_cell {}'.format(mouse, S_i_Test_B[cell_], cell_))

plt.figure()
#plt.imshow(fm_cells_Test_B_1wk_gauss)
plt.imshow(fm_cells_Test_B_1wk)
plt.title('Test_B_1wk mouse {} cell {} crossreg_cell {}'.format(mouse, S_i_Test_B_1wk[cell_], cell_))


stats_TFC_B = {'hM3D': [], 'hM4D': [], 'mCherry': []}
pval_TFC_B = {'hM3D': [], 'hM4D': [], 'mCherry': []}
stats_TFC_B_1wk = {'hM3D': [], 'hM4D': [], 'mCherry': []}
pval_TFC_B_1wk = {'hM3D': [], 'hM4D': [], 'mCherry': []}
stats_B_B_1wk = {'hM3D': [], 'hM4D': [], 'mCherry': []}
pval_B_B_1wk = {'hM3D': [], 'hM4D': [], 'mCherry': []}

single_pf_corr_TFC_B = {'hM3D': [], 'hM4D': [], 'mCherry': []}
single_pf_corr_TFC_B_1wk = {'hM3D': [], 'hM4D': [], 'mCherry': []}
single_pf_corr_B_B_1wk = {'hM3D': [], 'hM4D': [], 'mCherry': []}

single_pf_corr_TFC_B_pval = {'hM3D': [], 'hM4D': [], 'mCherry': []}
single_pf_corr_TFC_B_1wk_pval = {'hM3D': [], 'hM4D': [], 'mCherry': []}
single_pf_corr_B_B_1wk_pval = {'hM3D': [], 'hM4D': [], 'mCherry': []}

want_engram = False

for mouse, group in mouse_groups.items():
    if mouse in ['G07', 'G15']:
        continue
    print('*** Processing PFS for mouse {}...'.format(mouse))
    
    S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B = get_S_indeces_crossreg(Test_B[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)    

    if want_engram:
        engram_thresh=0
        [S_TFC_cond, S_spikes_TFC_cond, S_peakval_TFC_cond, S_idx_TFC_cond] = \
            TFC_cond[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])
        S_TFC_cond_engram_mask, S_TFC_cond_engram_indices, S_TFC_cond_engram, S_TFC_cond_engram_spikes, S_TFC_cond_engram_peakval = \
            get_engram_cells(S_TFC_cond, S_spikes_TFC_cond, S_peakval_TFC_cond, zscore_thresh=engram_thresh)        
        engram_indices_into_S_TFC_cond = np.intersect1d(S_TFC_cond_engram_indices, S_i_TFC_cond)

        S_i_engram = np.where(np.isin(S_i_TFC_cond, engram_indeces_into_S_TFC_cond))[0]
        S_i_engram_TFC_cond = np.array(S_i_TFC_cond)[S_i_engram]
        S_i_engram_Test_B = np.array(S_i_Test_B)[S_i_engram]
        S_i_engram_Test_B_1wk = np.array(S_i_Test_B_1wk)[S_i_engram]

    TFC_cond[mouse].fm.check_fluorescence_map()
    Test_B[mouse].fm.check_fluorescence_map()
    Test_B_1wk[mouse].fm.check_fluorescence_map()

    for i in range(len(S_i_engram if want_engram else S_i_TFC_cond)):
        if want_engram:
            fm_cells_TFC_cond = TFC_cond[mouse].fm.fluorescence_map[:,:,S_i_engram_TFC_cond[i]]
            fm_cells_Test_B = Test_B[mouse].fm.fluorescence_map[:,:,S_i_engram_Test_B[i]]
            fm_cells_Test_B_1wk = Test_B_1wk[mouse].fm.fluorescence_map[:,:,S_i_engram_Test_B_1wk[i]]

        else:
            fm_cells_TFC_cond = TFC_cond[mouse].fm.fluorescence_map[:,:,S_i_TFC_cond[i]]
            fm_cells_Test_B = Test_B[mouse].fm.fluorescence_map[:,:,S_i_Test_B[i]]
            fm_cells_Test_B_1wk = Test_B_1wk[mouse].fm.fluorescence_map[:,:,S_i_Test_B_1wk[i]]

        #fm_cells_TFC_cond = gaussian_filter(fm_cells_TFC_cond, my_sigma)
        #fm_cells_Test_B = gaussian_filter(fm_cells_Test_B, my_sigma)
        #fm_cells_Test_B_1wk = gaussian_filter(fm_cells_Test_B_1wk, my_sigma)

        result = stats.pearsonr(fm_cells_TFC_cond.ravel(), fm_cells_Test_B.ravel())
        if ~np.isnan(result.statistic) and result.pvalue < 0.01:
            single_pf_corr_TFC_B[group].append(result.statistic)
            pval_TFC_B[group].append(result.pvalue)

        result = stats.pearsonr(fm_cells_TFC_cond.ravel(), fm_cells_Test_B_1wk.ravel())
        if ~np.isnan(result.statistic) and result.pvalue < 0.01:
            stats_TFC_B_1wk[group].append(result.statistic)
            pval_TFC_B_1wk[group].append(result.pvalue)

        result = stats.pearsonr(fm_cells_Test_B.ravel(), fm_cells_Test_B_1wk.ravel())
        if ~np.isnan(result.statistic) and result.pvalue < 0.01:
            stats_B_B_1wk[group].append(result.statistic)
            pval_B_B_1wk[group].append(result.pvalue)

        # Track stability of single place field cells
        TFC_chk = B_chk = B_1wk_chk = False
        if S_i_TFC_cond[i] in TFC_cond[mouse].fm.pf.merged_means.keys():
            num_pfs_TFC_cond = TFC_cond[mouse].fm.pf.merged_means[S_i_TFC_cond[i]]
            TFC_chk = True
        if S_i_Test_B[i] in Test_B[mouse].fm.pf.merged_means.keys():
            num_pfs_Test_B = Test_B[mouse].fm.pf.merged_means[S_i_Test_B[i]]
            B_chk = True
        if S_i_Test_B_1wk[i] in Test_B_1wk[mouse].fm.pf.merged_means.keys():
            num_pfs_Test_B_1wk = Test_B_1wk[mouse].fm.pf.merged_means[S_i_Test_B_1wk[i]]
            B_1wk_chk = True
        if TFC_chk and B_chk:
            result = stats.pearsonr(fm_cells_TFC_cond.ravel(), fm_cells_Test_B.ravel())
            if ~np.isnan(result.statistic) and result.pvalue < 0.01:
                stats_TFC_B[group].append(result.statistic)
                pval_TFC_B[group].append(result.pvalue)
        if TFC_chk and B_1wk_chk:
            result = stats.pearsonr(fm_cells_TFC_cond.ravel(), fm_cells_Test_B_1wk.ravel())
            if ~np.isnan(result.statistic) and result.pvalue < 0.01:
                single_pf_corr_TFC_B_1wk[group].append(result.statistic)
                single_pf_corr_TFC_B_1wk_pval[group].append(result.pvalue)
        if B_chk and B_1wk_chk:
            result = stats.pearsonr(fm_cells_Test_B.ravel(), fm_cells_Test_B_1wk.ravel())
            if ~np.isnan(result.statistic) and result.pvalue < 0.01:
                single_pf_corr_B_B_1wk[group].append(result.statistic)
                single_pf_corr_B_B_1wk_pval[group].append(result.pvalue)


stats_checks = [[stats_TFC_B, stats_TFC_B_1wk, stats_B_B_1wk], \
                [single_pf_corr_TFC_B, single_pf_corr_TFC_B_1wk, single_pf_corr_B_B_1wk]]
title_checks = [['TFC+B', 'TFC+B_1wk', 'B+B_1wk'],['TFC+B single PFs', 'TFC+B_1wk single PFs', 'B+B_1wk single PFs']]

for stats_check, title_check, pfs_str in zip(stats_checks, title_checks, ['all pfs', 'single pfs']):
    for stats_measure, title_str in zip(stats_check, title_check):
        #for stats_measure, title_str in zip([stats_TFC_B, stats_TFC_B_1wk, stats_B_B_1wk], ['TFC+B', 'TFC+B_1wk', 'B+B_1wk']):
        '''
        stats_clean = {}
        for group in ['hM3D', 'hM4D', 'mCherry']:
            stats_measure[group] = np.array(stats_measure[group])
            stats_clean[group] = stats_measure[group][~np.isnan(stats_measure[group])]
        '''
        plt.figure(figsize=(4,3))
        bp = plt.boxplot([stats_measure['hM3D'], stats_measure['hM4D'], stats_measure['mCherry']], \
            notch=True, patch_artist=True, positions=[0.5,1,1.5])
        for p, c in zip(bp['boxes'], ['red','blue','black']):
            plt.setp(p,facecolor=c)
        plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])
        plt.title(title_str)

    for group in ['hM3D', 'hM4D', 'mCherry']:
        plt.figure(figsize=(4,3))
        bp = plt.boxplot([stats_check[0][group], stats_check[1][group], stats_check[2][group]], \
            notch=True, patch_artist=True, positions=[0.5,1,1.5])

        #bp = plt.boxplot([stats_TFC_B[group], stats_TFC_B_1wk[group], stats_B_B_1wk[group]], \
        #    notch=True, patch_artist=True, positions=[0.5,1,1.5])
        for p, c in zip(bp['boxes'], ['black','grey','gainsboro']):
            plt.setp(p,facecolor=c)
        plt.xticks([0.5,1,1.5],['TFC+B','TFC+B_1wk','B+B_1wk'])
        plt.title('{} {}'.format(group, pfs_str))





pcells_mice_TFC_cond_LT2=dict()
session=TFC_cond_LT2
session_str='TFC_cond_LT2'

cells=np.array([]); 
bin_width=4.5; random_width=4; want_3D=True; pcells_mice=pcells_mice_TFC_cond_LT2; max_fields=45; only_fm_pcells=True; print_pcell_maps=True; merge_distance=4

save_path = os.path.join(PLOTS_DIR, 'fluorescence_maps_{}fields_{}'.format(max_fields, session_str)) #'{}_{}_fluorescence-map.png'.format(session_str, m))
os.makedirs(save_path, exist_ok=True)
SAVE_PATH = save_path

loc_bounds = None
if len(session) > 0:
    min_xs = []
    max_xs = []
    min_ys = []
    max_ys = []
    for m, sess in session.items():
        loc_X = sess.loc_X_behavcam_smooth
        loc_Y = sess.loc_Y_behavcam_smooth
        min_xs.append(np.nanmin(loc_X))
        max_xs.append(np.nanmax(loc_X))
        min_ys.append(np.nanmin(loc_Y))
        max_ys.append(np.nanmax(loc_Y))
    loc_bounds = {
        'MIN_X': float(np.min(min_xs)),
        'MAX_X': float(np.max(max_xs)),
        'MIN_Y': float(np.min(min_ys)),
        'MAX_Y': float(np.max(max_ys))
    }
    
mouse='G05'
sess=session[mouse]
random_pcells_per_field=5









##


for mouse_id in range(6, 9):
    mouse = f"G{mouse_id:02d}"
    print(mouse)
    sess = session[mouse]
    print(f"confirming session: {sess.mouse}")

    print("plot_fluorescence_map_helper: we were given {} max_fields!".format(max_fields))

    # Get fluorescence traces
    S_mov = sess.S_mov
    S_imm = sess.S_imm
    S_peakval = sess.S_peakval
    #[S_mov_sp, S_mov_pkval] = find_spikes_ca_S(S_mov, sess.thres, want_peakval=True)

    # Process mouse location and establish bins
    loc = Location_XY(mouse, mouse_groups[mouse], sess, bin_width, loc_bounds=loc_bounds)
    [loc_X, loc_Y, min_x, min_y, max_x, max_y, binned_X, binned_Y, num_bins_x, num_bins_y] = loc.get_loc_data()
    occupancy = loc.get_occupancy_map(PLOTS_DIR, session_str)

    # Get cells to plot
    num_cells = S_mov.shape[0]
    if not cells.any(): 
        if random_width > 0: # do random cells in grid with width defined by random_width

            # first check if we saved pickles of previous runs, and just reuse those cells to save time.
            fm = FluorescenceMap(loc, S_mov, cells, save_path=SAVE_PATH, mouse=mouse, to_pickle=True, sess=sess, load_num_cells=random_width*random_width)

            if fm.cells.any():
                cells = fm.cells
            else:
                # shuffle calculations. (If want to forcibly 'reroll' cell #'s just delete the .npy files)
                rng = default_rng()
                # sampling without replacement of cell indices; don't want to plot duplicates
                cells = rng.choice(num_cells, size=random_width*random_width, replace=False) 
        else:
            # get all cells
            cells = range(S_mov.shape[0])
    
    # Get all cells for pcell analysis.
    cells_pcells = np.array(range(S_mov.shape[0]))

    #
    # Plot fluorescence maps during:
    #
    if not only_fm_pcells: # this switch allows us to skip a lot of the FM's that were already plotted and behaviour-based so don't change between analysis runs.
            
        # 1. Movement (no Gaussian smoothing)
        fluorescence_map = FluorescenceMap(loc, S_mov, cells).get_map()
        plot_fluorescence_map_plotter(fluorescence_map, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'movement_raw', S=S_mov)
        #plot_fluorescence_map_plotter(fluorescence_map / np.max(fluorescence_map), cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'movement_norm')

        # 1. Movement-Gaussian smoothed (as is everything from now on)
        fluorescence_map = FluorescenceMap(loc, gaussian_filter(S_mov, sigma=SMOOTH_LOC_SIGMA), cells).get_map()
        plot_fluorescence_map_plotter(fluorescence_map, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'movement', S=S_mov)
        #plot_fluorescence_map_plotter(fluorescence_map / np.max(fluorescence_map), cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'movement_norm')

        # 2. Immobility
        fluorescence_map = FluorescenceMap(loc, S_imm, cells).get_map()
        plot_fluorescence_map_plotter(fluorescence_map, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'immobility', S=S_imm)    
        #plot_fluorescence_map_plotter(fluorescence_map / np.max(fluorescence_map), cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'immobility_norm')    

        # 3. First 3 min of TFC_cond
        if isinstance(sess, TraceFearCondSession):
            first_3min = range(0,sess.tone_onsets[0])
            fluorescence_map = FluorescenceMap(loc, S_mov[:,first_3min], cells).get_map()
            plot_fluorescence_map_plotter(fluorescence_map, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'first_3min', S=S_mov[:,first_3min])    
            #plot_fluorescence_map_plotter(fluorescence_map / np.max(fluorescence_map), cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'immobility_norm')    
        
        # 4a. Occupancy-corrected Movement
        fm = FluorescenceMap(loc, S_mov, cells, save_path=SAVE_PATH, mouse=mouse, to_pickle=True, sess=sess, load_num_cells=len(cells), max_fields=max_fields)
        [fluorescence_map_occup, occup_map] = fm.generate_occupancy_map()
        plot_fluorescence_map_plotter(fluorescence_map_occup, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'movement+occup', S=S_mov, want_3D=want_3D)

        # 4b. Plot place cells for this subset of occupancy-corrected movement.
        fm.get_shuffled_responses(num_shifts=500)
        for percentile in [99.0, 99.3, 99.5, 99.7, 99.9]:
            fm.get_significant_response_profiles(percentile=percentile)
        # Not needed now vvv
        #plot_fluorescence_map_plotter(occup_map, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'OCCUP only', S=S_mov)

    # 5. Now obtain all pcells (don't plot)
    fm = FluorescenceMap(loc, S_mov, cells_pcells, save_path=SAVE_PATH, mouse=mouse, to_pickle=True, sess=sess, load_num_cells=num_cells, print_pcell_maps=print_pcell_maps, \
        max_fields=max_fields)
    [fluorescence_map_occup, occup_map] = fm.generate_occupancy_map()
    fm.get_shuffled_responses(num_shifts=500)
    percentile = 99.0
    sig_responses = fm.get_significant_response_profiles(percentile=percentile)
    #plot_fluorescence_map_plotter(fluorescence_map / np.max(fluorescence_map), cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, 'movement_norm')
    pcells_mice[mouse] = sig_responses
    sess.fm = fm
    sess.loc = loc
    sess.sig_responses = sig_responses

    # 6. Plot examples of cells with all numbers of found significant responses.
    pcells_num_fields = dict()
    for k,v in sig_responses.items():
        num_fields = len(v)
        if num_fields not in pcells_num_fields:
            pcells_num_fields[num_fields] = [k]
        else:
            pcells_num_fields[num_fields].append(k)
    for num_fields, pcells_with_fields in pcells_num_fields.items():
        rng = default_rng()
        cell_indeces = rng.choice(len(pcells_with_fields), size=min(len(pcells_with_fields),random_pcells_per_field), replace=False) 
        for cell_idx in cell_indeces:
            cell_id = pcells_with_fields[cell_idx]
            fm.save_map(cell_id, sig_responses[cell_id], percentile, num_fields=str(num_fields))

            fig, ax = plt.subplots(1,1, subplot_kw={"projection": "3d"})
            X = np.arange(0, fluorescence_map_occup.shape[0])
            Y = np.arange(0, fluorescence_map_occup.shape[1])
            Z = fluorescence_map_occup[:,:,cell_id]
            X, Y = np.meshgrid(X, Y)
            ax.plot_surface(X, Y, np.transpose(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_title('cell {}'.format(cell_id))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout()
            #fig.suptitle('Mouse {} ({}) - {}, {}'.format(mouse, mouse_groups[mouse], session_str, condition))
            fig.suptitle('Mouse {} ({})'.format(mouse, mouse_groups[mouse]))

            #plt.subplot_tool(targetfig=fig)
            #fig_3D.subplots_adjust(top=0.93)
            fig.savefig(os.path.join(fm.save_path, 'pcells_{}_max_fields_{}_num_fields_{}_cell_{}_perc_{}_3D.png'.format(mouse, max_fields, num_fields, cell_id, percentile)), format='png', dpi=300)
            plt.close(fig)

    # 7. Find actual place fields now
    fm.find_place_fields(sess, method='iterative_gauss', merge_distance=merge_distance)








###
SAVE
###

import pickle
from pathlib import Path
save_path_pcells_mice_TFC_cond_LT1 = Path(NPY_SAVE_PATH) / "pcells_mice_TFC_cond_LT1.pkl.gz"    
with gzip.open(save_path_pcells_mice_TFC_cond_LT1, 'wb') as f:
    pickle.dump(pcells_mice_TFC_cond_LT1, f, protocol=pickle.HIGHEST_PROTOCOL)

###
LOAD
###
import pickle
import gzip
from pathlib import Path

load_path_pcells_mice_TFC_cond_LT1 = Path(NPY_SAVE_PATH) / "pcells_mice_TFC_cond_LT1.pkl.gz"
with gzip.open(load_path_pcells_mice_TFC_cond_LT1, "rb") as f:
    pcells_mice_TFC_cond_LT1_foo = pickle.load(f)




###
SAVE-FANCY
###

import os
import pickle
import gzip
import tempfile
from pathlib import Path

# Optional progress bar (works if tqdm is installed; otherwise disabled)
try:
    from tqdm.auto import tqdm   # auto picks best renderer (terminal vs notebook)
except ImportError:
    tqdm = None
except ImportError:
    tqdm = None


class TqdmWriter:
    def __init__(self, file_obj, desc="saving"):
        self._f = file_obj
        self._pbar = None
        if tqdm is not None:
            self._pbar = tqdm(
                total=None,
                unit="B",
                unit_scale=True,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
                mininterval=0.2,
            )

    def write(self, data):
        n = self._f.write(data)
        if self._pbar is not None:
            self._pbar.update(n)
        return n

    def flush(self):
        self._f.flush()

    def close(self):
        if self._pbar is not None:
            self._pbar.close()


def save_pickle_gz_atomic(obj, final_path, *, protocol=pickle.HIGHEST_PROTOCOL, fsync=True, desc="saving"):
    """
    Atomically write obj to final_path as a gzipped pickle.
    Writes to a temp file in the same directory, then os.replace() to finalize.
    If interrupted, final file is left untouched.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd = None
    tmp_path = None

    try:
        # Temp file in same dir => atomic replace is reliable
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=final_path.name + ".tmp-",
            suffix=".gz",
            dir=str(final_path.parent),
        )
        tmp_path = Path(tmp_name)

        with os.fdopen(tmp_fd, "wb") as raw_f:
            tmp_fd = None  # now owned by raw_f

            with gzip.GzipFile(fileobj=raw_f, mode="wb") as gz_f:
                writer = TqdmWriter(gz_f, desc=desc)
                pickle.dump(obj, writer, protocol=protocol)
                writer.flush()
                writer.close()

            raw_f.flush()
            if fsync:
                os.fsync(raw_f.fileno())

        os.replace(str(tmp_path), str(final_path))

    except Exception:
        # Clean up temp file on any failure
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def load_pickle_gz(path):
    path = Path(path)
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# -------------------------
# YOUR PATHS + SAVE / LOAD
# -------------------------

save_path_pcells_mice_TFC_cond_LT1 = Path(NPY_SAVE_PATH) / "pcells_mice_TFC_cond_LT1.pkl.gz"
save_path_TFC_cond_LT1 = Path(NPY_SAVE_PATH) / "TFC_cond_LT1.pkl.gz"

# SAVE (atomic)
save_pickle_gz_atomic(pcells_mice_TFC_cond_LT1, save_path_pcells_mice_TFC_cond_LT1)
save_pickle_gz_atomic(TFC_cond_LT1, save_path_TFC_cond_LT1)

# LOAD
pcells_mice_TFC_cond_LT1 = load_pickle_gz(save_path_pcells_mice_TFC_cond_LT1)












### OLD SORTING


        sorted_entries = []
        # Will store tuples:
        # (cell_id, chosen_pf_group_idx, chosen_pf_center_xy, nn_time_idx, nn_xy, nn_loc1d)

        for cell_id in cells_LT1:
            merged = LT1.fm.pf.merged_means.get(cell_id, None)
            model  = LT1.fm.pf.model_.get(cell_id, None)

            if merged is None or model is None or not hasattr(model, "means_"):
                continue

            merged_means = merged
            means_xy = np.asarray(model.means_)  # (n_means, 2)

            # ---- PF-count filtering (unless disabled) ----
            if num_pfs_filtered != -1:
                if len(merged_means) != int(num_pfs_filtered):
                    continue

            # ---- Compute PF centers ----
            pf_centers = []
            for grp_i, grp in enumerate(merged_means):
                grp = list(grp)
                if len(grp) == 0:
                    continue
                ctr_xy = means_xy[grp].mean(axis=0)
                pf_centers.append((grp_i, ctr_xy))

            if len(pf_centers) == 0:
                continue

            # ---- For each PF center, find nearest miniscope (x,y) sample ----
            pf_candidates = []
            for grp_i, ctr_xy in pf_centers:
                d2 = np.sum((XY_LT1 - ctr_xy[None, :])**2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_xy = XY_LT1[nn_idx]
                nn_loc1d = float(loc_1d_LT1[nn_idx])

                pf_candidates.append(
                    (grp_i, ctr_xy, nn_idx, nn_xy, nn_loc1d)
                )

            if len(pf_candidates) == 0:
                continue

            # ---- PF selection logic ----
            if num_pfs_filtered == -1:
                # Choose PF closest to track beginning (or end if sort_reverse=True)
                chosen = min(
                    pf_candidates,
                    key=lambda x: abs(x[-1] - track_ref)
                )
            else:
                # Original behavior: PF whose center is closest in 2D space
                chosen = min(
                    pf_candidates,
                    key=lambda x: np.sum((XY_LT1[x[2]] - x[1])**2)
                )

            chosen_grp_i, chosen_ctr_xy, nn_idx, nn_xy, nn_loc1d = chosen

            sorted_entries.append(
                (cell_id, chosen_grp_i, chosen_ctr_xy, nn_idx, nn_xy, nn_loc1d)
            )

        # ---- Sort cells by linear-track position ----
        sorted_entries.sort(key=lambda x: x[-1], reverse=bool(sort_reverse))

        # Final output
        sorted_LT1 = [cell_id for (cell_id, *_rest) in sorted_entries]
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

# for random cell selection instead of crossreg
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


#
# try logarithmic scale plots
#

fig, axes = plt.subplots(2,3)
i = 0
target_shape = 1.0
target_loc = 0.0 
target_scale = 1.0

for m,group in zip(['G05','G10','G14'], ['Exc','Ctl','Inh']):
    sess=TFC_cond
    s=sess[m]
    peaks = np.concatenate(list(s.S_peakval.values()))
    ax0 = axes[0,i]
    ax0.hist(peaks, bins=50)
    ax0.set_yscale('log')
    ax0.set_title('{} ({})'.format(m, group))
    print('{} {}'.format(m, group))
    shape, loc, scale = sp.stats.lognorm.fit(peaks)
    print(' ({},{},{})'.format(shape,loc,scale))
    #peaks_standardized = (np.log(peaks) - np.log(scale) - loc) / shape
    #peaks_norm = np.exp(peaks_standardized*target_shape+ np.log(target_scale) + target_loc)
    peaks_norm = np.exp((np.log(peaks) - np.log(scale) - loc) / shape * target_shape + np.log(target_scale) + target_loc)

    ax1 = axes[1,i]
    ax1.hist(peaks_norm, bins=50, color='r')
    ax1.set_yscale('log')
    ax1.set_title('{} ({}) scaled'.format(m,group))
    i+=1

# part deux

import scipy as sp, numpy as np
from scipy.stats import lognorm
fig, axes = plt.subplots(2,3)
i = 0
target_shape = 1.0
target_loc = 0.0 
target_scale = 1.0

for m,group in zip(['G05','G10','G14'], ['Exc','Ctl','Inh']):
    sess=TFC_cond
    s=sess[m]
    peaks = np.concatenate(list(s.S_peakval.values()))
    ax0 = axes[0,i]
    n, bins, patches = ax0.hist(peaks, bins=1000)
    ax0.set_xlim([0,200])
    #ax0.set_yscale('log')
    ax0.set_title('{} ({})'.format(m, group))
    print('{} {}'.format(m, group))
    shape, loc, scale = sp.stats.lognorm.fit(n)
    #x_frozen = np.linspace(lognorm.ppf(0.0, scale), lognorm.ppf(max(peaks), scale), 100)     
    ax0.plot(bins, lognorm.pdf(n, shape, loc=loc, scale=scale), 'k-', lw=5, alpha=0.6)
    #ax0.plot(x_frozen, lognorm.pdf(x_frozen, scale), 'k-', lw=5, alpha=0.6)
    print(' ({},{},{})'.format(shape,loc,scale))
    #peaks_standardized = (np.log(peaks) - np.log(scale) - loc) / shape
    #peaks_norm = np.exp(peaks_standardized*target_shape+ np.log(target_scale) + target_loc)
    peaks_norm = np.exp((np.log(peaks) - np.log(scale) - loc) / shape * target_shape + np.log(target_scale) + target_loc)

    ax1 = axes[1,i]
    ax1.hist(peaks_norm, bins=1000, color='r')
    ax1.set_xlim([0,2])
    #ax1.set_yscale('log')
    ax1.set_title('{} ({}) scaled'.format(m,group))
    i+=1
fig.tight_layout()



# z-score
sess=Test_B
m='G05'
s=sess[m]
peaks = np.concatenate(list(s.S_peakval.values()))

    sess=TFC_cond
    s=sess[m]
    peaks = np.concatenate(list(s.S_peakval.values()))
    ax0 = axes[0,i]
    n, bins, patches = ax0.hist(peaks, bins=1000)
    ax0.set_xlim([0,200])
    #ax0.set_yscale('log')
    ax0.set_title('{} ({})'.format(m, group))
    print('{} {}'.format(m, group))
    shape, loc, scale = sp.stats.lognorm.fit(n)
    #x_frozen = np.linspace(lognorm.ppf(0.0, scale), lognorm.ppf(max(peaks), scale), 100)     
    ax0.plot(bins, lognorm.pdf(n, shape, loc=loc, scale=scale), 'k-', lw=5, alpha=0.6)
    #ax0.plot(x_frozen, lognorm.pdf(x_frozen, scale), 'k-', lw=5, alpha=0.6)
    print(' ({},{},{})'.format(shape,loc,scale))
    #peaks_standardized = (np.log(peaks) - np.log(scale) - loc) / shape
    #peaks_norm = np.exp(peaks_standardized*target_shape+ np.log(target_scale) + target_loc)
    peaks_norm = np.exp((np.log(peaks) - np.log(scale) - loc) / shape * target_shape + np.log(target_scale) + target_loc)

    ax1 = axes[1,i]
    ax1.hist(peaks_norm, bins=1000, color='r')
    ax1.set_xlim([0,2])
    #ax1.set_yscale('log')
    ax1.set_title('{} ({}) scaled'.format(m,group))
    i+=1
fig.tight_layout()


# cross-registration engrams
m='G21'
mapping='TFC_cond+Test_B+Test_B_1wk'
[S_TFC, S_spikes_TFC, S_peakval_TFC, S_idx_TFC] = TFC_cond[m].get_S_mapping(mapping, with_peakval=True,  with_crossreg=TFC_B_B_1wk_crossreg[m])
[S_B, S_spikes_B, S_peakval_B, S_idx_B] = Test_B[m].get_S_mapping(mapping, with_peakval=True,  with_crossreg=TFC_B_B_1wk_crossreg[m])
[S_B_1wk, S_spikes_B_1wk, S_peakval_B_1wk, S_idx_B_1wk] = Test_B_1wk[m].get_S_mapping(mapping, with_peakval=True,  with_crossreg=TFC_B_B_1wk_crossreg[m])

S_mask_TFC_eng, S_indices_TFC_eng, S_TFC_eng, S_spikes_TFC_eng, S_peakval_TFC_eng = get_engram_cells(S_TFC, S_spikes_TFC, S_peakval_TFC)
S_mask_B_eng, S_indices_B_eng, S_B_eng, S_spikes_B_eng, S_peakval_B_eng = get_engram_cells(S_B, S_spikes_B, S_peakval_B)
S_mask_B_1wk_eng, S_indices_B_1wk_eng, S_B_1wk_eng, S_spikes_B_1wk_eng, S_peakval_B_1wk_eng = get_engram_cells(S_B_1wk, S_spikes_B_1wk, S_peakval_B_1wk)

print([x for x, m1, m2, m3 in zip(range(len(S_mask_TFC_eng)), S_mask_TFC_eng, S_mask_B_eng, S_mask_B_1wk_eng) if m1 and m2 and m3])
print([x for x, m1, m2 in zip(range(len(S_mask_B_eng)), S_mask_B_eng, S_mask_B_1wk_eng) if m1 and m2])
print([x for x, m1, m2, m3 in zip(range(len(S_mask_TFC_eng)), S_mask_TFC_eng, S_mask_B_eng, S_mask_B_1wk_eng) if m1])

[x for x, m1, m2 in zip(range(len(S_mask_B_1wk_eng)), S_mask_B_eng, S_mask_B_1wk_eng) if m1 and m2]
[x for x, m1, m2 in zip(range(len(S_mask_B_eng)), S_mask_B_eng, S_mask_B_1wk_eng) if m1 and m2]


from sklearn import mixture

sess=TFC_cond['G05']
cell=10
[sess.S_spikes_orig, sess.S_peakval_orig] = find_spikes_ca_S(sess.S_orig, sess.thres, want_peakval=True)
plt.figure()
plt.plot(sess.S[cell],'b--',alpha=0.7)
plt.plot(sess.S_orig[cell],'r-.',alpha=0.7)
for i in range(len(sess.S_spikes_orig[cell])):
    plt.plot([sess.S_spikes_orig[cell][i]], [0],'r+')
for i in range(len(sess.S_spikes[cell])):
    plt.plot([sess.S_spikes[cell][i]], [0],'b*')
plt.plot(sess.S_spikes[cell],sess.S_peakval[cell],'b+')
plt.plot(sess.S_spikes_orig[cell],sess.S_peakval_orig[cell],'r*')

'''
For engrams
'''
#mouse='G14'
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import pingouin as pg

# The shuffle dicts have to be separate per engram/low_activity groups to match the number of
# cells in each (which varies)
avg_TFC_cond_groups_engram = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_groups_engram = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_engram = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

avg_TFC_cond_groups_engram_shuffle = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_groups_engram_shuffle = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_engram_shuffle = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

avg_TFC_cond_groups_low_activity = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_groups_low_activity = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_low_activity = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

avg_TFC_cond_groups_low_activity_shuffle = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_groups_low_activity_shuffle = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_low_activity_shuffle = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

mapping='TFC_cond+Test_B+Test_B_1wk'
engram_type = 'encoding' # 'encoding' or 'recall'
engram_thresh=0
plot_individual = False
want_correlations = False

corr_sess_engram = {'TFC_cond': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_sess_low_activity = {'TFC_cond': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_sess_engram_cross = {'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))},
                          'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_sess_low_activity_cross = {'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))},
                          'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_shuffle_engram = {'TFC_cond': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_shuffle_low_activity = {'TFC_cond': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
                      'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))},
                      'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_shuffle_engram_cross = {'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}
corr_shuffle_low_activity_cross = {'Test_B': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}, 
             'Test_B_1wk': {'hM3D': np.empty((0,)), 'hM4D': np.empty((0,)), 'mCherry': np.empty((0,))}}

for mouse, group in mouse_groups.items():
    print('{} '.format(mouse),end='')
    if mouse in ['G07', 'G15']:
        continue
    [S_TFC_cond, S_spikes_TFC_cond, S_peakval_TFC_cond, S_idx_TFC_cond] = \
        TFC_cond[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])
    [S_Test_B, S_spikes_Test_B, S_peakval_Test_B, S_idx_Test_B] = \
        Test_B[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])
    [S_Test_B_1wk, S_spikes_Test_B_1wk, S_peakval_Test_B_1wk, S_idx_Test_B_1wk] = \
        Test_B_1wk[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])

    S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B = get_S_indeces_crossreg(Test_B[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[mouse], TFC_B_B_1wk_crossreg[mouse], mapping_TFC_cond_Test_B_Test_B_1wk)    


    for shuffle_type in ['real', 'shuffle']:
        for act_type in ['engram', 'low_activity']:
            print('{} {} '.format(shuffle_type, act_type),end='')

            # Engram
            if act_type == 'engram' and engram_type == 'encoding':
                [S_i_TFC_cond, S_i_Test_B, S_i_Test_B_1wk] = \
                    get_engram_crossreg(mouse, TFC_cond, Test_B, Test_B_1wk, TFC_B_B_1wk_crossreg, mapping_TFC_cond_Test_B_Test_B_1wk, engram_type='encoding', want_low_activity=False)                
            if act_type == 'engram' and engram_type == 'recall':
                [S_i_TFC_cond, S_i_Test_B, S_i_Test_B_1wk] = \
                    get_engram_crossreg(mouse, TFC_cond, Test_B, Test_B_1wk, TFC_B_B_1wk_crossreg, mapping_TFC_cond_Test_B_Test_B_1wk, engram_type='recall', want_low_activity=False)                

            # Non-engram (low activity)
            if act_type == 'low_activity' and engram_type == 'encoding':
                [S_i_TFC_cond, S_i_Test_B, S_i_Test_B_1wk] = \
                    get_engram_crossreg(mouse, TFC_cond, Test_B, Test_B_1wk, TFC_B_B_1wk_crossreg, mapping_TFC_cond_Test_B_Test_B_1wk, engram_type='encoding', want_low_activity=True)
            if act_type == 'low_activity' and engram_type == 'recall':
                [S_i_TFC_cond, S_i_Test_B, S_i_Test_B_1wk] = \
                    get_engram_crossreg(mouse, TFC_cond, Test_B, Test_B_1wk, TFC_B_B_1wk_crossreg, mapping_TFC_cond_Test_B_Test_B_1wk, engram_type='recall', want_low_activity=True)                

            if shuffle_type == 'shuffle':
                if engram_type == 'encoding':
                    S_mask=random.sample(range(1,len(S_TFC_cond)), len(S_i_TFC_cond))
                if engram_type == 'recall':
                    S_mask=random.sample(range(1,len(S_Test_B)), len(S_i_Test_B))
            #avg_TFC_cond = np.sum(S_TFC_cond[S_mask,:],1)/S_TFC_cond.shape[1]
            avg_TFC_cond = np.sum(TFC_cond[mouse].S[S_i_TFC_cond,:],1)/len(S_i_TFC_cond)
            avg_Test_B = np.sum(Test_B[mouse].S[S_i_Test_B,:],1)/len(S_i_Test_B)
            avg_Test_B_1wk = np.sum(Test_B_1wk[mouse].S[S_i_Test_B_1wk,:],1)/len(S_i_Test_B_1wk)

            if act_type == 'engram':
                if shuffle_type == 'real':
                    avg_TFC_cond_groups_engram[group] = np.append(avg_TFC_cond_groups_engram[group], avg_TFC_cond)
                    avg_Test_B_groups_engram[group] = np.append(avg_Test_B_groups_engram[group], avg_Test_B)
                    avg_Test_B_1wk_groups_engram[group] = np.append(avg_Test_B_1wk_groups_engram[group], avg_Test_B_1wk)
                if shuffle_type == 'shuffle':
                    avg_TFC_cond_groups_engram_shuffle[group] = np.append(avg_TFC_cond_groups_engram_shuffle[group], avg_TFC_cond)
                    avg_Test_B_groups_engram_shuffle[group] = np.append(avg_Test_B_groups_engram_shuffle[group], avg_Test_B)
                    avg_Test_B_1wk_groups_engram_shuffle[group] = np.append(avg_Test_B_1wk_groups_engram_shuffle[group], avg_Test_B_1wk)
            if act_type == 'low_activity':
                if shuffle_type == 'real':
                    avg_TFC_cond_groups_low_activity[group] = np.append(avg_TFC_cond_groups_low_activity[group], avg_TFC_cond)
                    avg_Test_B_groups_low_activity[group] = np.append(avg_Test_B_groups_low_activity[group], avg_Test_B)
                    avg_Test_B_1wk_groups_low_activity[group] = np.append(avg_Test_B_1wk_groups_low_activity[group], avg_Test_B_1wk)
                if shuffle_type == 'shuffle':
                    avg_TFC_cond_groups_low_activity_shuffle[group] = np.append(avg_TFC_cond_groups_low_activity_shuffle[group], avg_TFC_cond)
                    avg_Test_B_groups_low_activity_shuffle[group] = np.append(avg_Test_B_groups_low_activity_shuffle[group], avg_Test_B)
                    avg_Test_B_1wk_groups_low_activity_shuffle[group] = np.append(avg_Test_B_1wk_groups_low_activity_shuffle[group], avg_Test_B_1wk)

            if plot_individual:
                plt.figure()
                for i in range(len(S_i_TFC_cond)-1):
                    plt.plot([avg_TFC_cond[i], avg_Test_B[i], avg_Test_B_1wk[i]])
                plt.title('{} {} {}'.format(mouse, act_type, shuffle_type))

            if want_correlations:
                engram_sessions = {'TFC_cond': TFC_cond[mouse].S[S_i_TFC_cond,:], 
                                'Test_B' : Test_B[mouse].S[S_i_Test_B,:], 
                                'Test_B_1wk': Test_B_1wk[mouse].S[S_i_Test_B_1wk,:]}

                for correlation_type in ['auto', 'cross']:         
                    print('{} '.format(correlation_type),end='')
                    if correlation_type == 'auto':
                        engram_sessions_keys = engram_sessions.keys()
                    if correlation_type == 'cross':
                        engram_sessions_keys = ['Test_B', 'Test_B_1wk']

                    min_len = np.min([engram_sessions[k].shape[1] for k in engram_sessions_keys])

                    for sess_str in engram_sessions_keys:
                        S_engram_first = S_engram_second = engram_sessions[sess_str]
                        first_str = second_str = sess_str
                        if correlation_type == 'cross':
                            S_engram_first = engram_sessions['TFC_cond']
                            first_str = 'TFC_cond'

                        print('{}<->{} '.format(first_str, second_str),end='')

                        S_engram_mouse = np.zeros(S_engram_first.shape[0])                
                        for i in range(S_engram_first.shape[0]):
                            for j in range(S_engram_second.shape[0]):
                                if j == i:
                                    continue # don't do auto-correlation on a neuron...
                                #FOO
                                binned_first = S_engram_first[i,range(min_len-(min_len%MINISCOPE_FPS))].reshape(-1,MINISCOPE_FPS).mean(axis=1)
                                binned_second = S_engram_second[j,range(min_len-(min_len%MINISCOPE_FPS))].reshape(-1,MINISCOPE_FPS).mean(axis=1)
                                corr = pg.corr(binned_first, binned_second)
                                #corr = pg.corr(S_engram_first[i,range(min_len)], S_engram_second[j,range(min_len)])
                                corr_val = corr['r'].iloc[0]
                                if ~np.isnan(corr_val):
                                    S_engram_mouse[j] = corr_val
                                else:
                                    print('***NaN corr found found!! i={} j={} act_type {} sess_str {} correlation_type {} shuffle_type {} mouse {} group {}'.format(i, j, act_type, sess_str, correlation_type, shuffle_type, mouse, group))
                        #corr_mean = np.mean(S_engram_mouse)

                        if shuffle_type == 'real':
                            if act_type == 'engram' and correlation_type == 'auto':
                                corr_sess_engram[sess_str][group] = np.concatenate((corr_sess_engram[sess_str][group], S_engram_mouse))
                            elif act_type == 'engram' and correlation_type == 'cross':
                                corr_sess_engram_cross[sess_str][group] = np.concatenate((corr_sess_engram_cross[sess_str][group], S_engram_mouse))
                            elif act_type == 'low_activity' and correlation_type == 'auto':
                                corr_sess_low_activity[sess_str][group] = np.concatenate((corr_sess_low_activity[sess_str][group], S_engram_mouse))
                            elif act_type == 'low_activity' and correlation_type == 'cross':
                                corr_sess_low_activity_cross[sess_str][group] = np.concatenate((corr_sess_low_activity_cross[sess_str][group], S_engram_mouse))
                        if shuffle_type == 'shuffle':
                            if act_type == 'engram' and correlation_type == 'auto':
                                corr_shuffle_engram[sess_str][group] = np.concatenate((corr_shuffle_engram[sess_str][group], S_engram_mouse))
                            elif act_type == 'engram' and correlation_type == 'cross':
                                corr_shuffle_engram_cross[sess_str][group] = np.concatenate((corr_shuffle_engram_cross[sess_str][group], S_engram_mouse))
                            elif act_type == 'low_activity' and correlation_type == 'auto':
                                corr_shuffle_low_activity[sess_str][group] = np.concatenate((corr_shuffle_low_activity[sess_str][group], S_engram_mouse))
                            elif act_type == 'low_activity' and correlation_type == 'cross':
                                corr_shuffle_low_activity_cross[sess_str][group] = np.concatenate((corr_shuffle_low_activity_cross[sess_str][group], S_engram_mouse))

    print('done with mouse {}.\n'.format(mouse))

for act_type in ['engram', 'low_activity', 'engram-shuffle', 'low_activity-shuffle']:
    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    
    for ax_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
        if act_type == 'engram':
            act_groups = [avg_TFC_cond_groups_engram, avg_Test_B_groups_engram, avg_Test_B_1wk_groups_engram]
        if act_type == 'low_activity':
            act_groups = [avg_TFC_cond_groups_low_activity, avg_Test_B_groups_low_activity, avg_Test_B_1wk_groups_low_activity]
        if act_type == 'engram-shuffle':
            act_groups = [avg_TFC_cond_groups_engram_shuffle, avg_Test_B_groups_engram_shuffle, avg_Test_B_1wk_groups_engram_shuffle]
        if act_type == 'low_activity-shuffle':
            act_groups = [avg_TFC_cond_groups_low_activity_shuffle, avg_Test_B_groups_low_activity_shuffle, avg_Test_B_1wk_groups_low_activity_shuffle]
        ax = axs[ax_i]
        for i in range(len(act_groups[0][group])-1):
            ax.plot([act_groups[0][group][i], act_groups[1][group][i], act_groups[2][group][i]], color='0.8', alpha=0.5)
        ax.plot([np.mean(act_groups[0][group]), np.mean(act_groups[1][group]), np.mean(act_groups[2][group])], color='black')
        ax.set_xticks([0,1,2],labels=['TFC','Test B','Test B +1wk'])        
        ax.set_ylim([0,2])
        ax.set_title('{} {}'.format(group, act_type))
    fig.tight_layout()

rm_data = {}
for group in ['hM3D', 'hM4D', 'mCherry']:
    rm_data[group] = {}
    rm_data[group]['engram'] = pd.DataFrame({
        #'subject': np.tile(np.arange(1, len(avg_TFC_cond_groups_engram[group])+1),6),
        'subject': np.concatenate((np.tile(np.arange(1, len(avg_TFC_cond_groups_engram[group])+1),3),
                                  np.tile(np.arange(len(avg_TFC_cond_groups_engram[group])+1, 2*len(avg_TFC_cond_groups_engram[group])+1),3))),
        'type': np.concatenate((np.repeat('real', len(avg_TFC_cond_groups_engram[group])),
                               np.repeat('real', len(avg_Test_B_groups_engram[group])),
                               np.repeat('real', len(avg_Test_B_1wk_groups_engram[group])),
                               np.repeat('shuffle', len(avg_TFC_cond_groups_engram_shuffle[group])),
                               np.repeat('shuffle', len(avg_Test_B_groups_engram_shuffle[group])),
                               np.repeat('shuffle', len(avg_Test_B_1wk_groups_engram_shuffle[group])))),
        'time': np.concatenate((np.repeat('TFC_cond', len(avg_TFC_cond_groups_engram[group])),
                               np.repeat('Test_B', len(avg_Test_B_groups_engram[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_engram[group])),
                               np.repeat('TFC_cond', len(avg_TFC_cond_groups_engram_shuffle[group])),
                               np.repeat('Test_B', len(avg_Test_B_groups_engram_shuffle[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_engram_shuffle[group])))),
        'value': np.concatenate((avg_TFC_cond_groups_engram[group],
                                 avg_Test_B_groups_engram[group],
                                 avg_Test_B_1wk_groups_engram[group],
                                 avg_TFC_cond_groups_engram_shuffle[group],
                                 avg_Test_B_groups_engram_shuffle[group],
                                 avg_Test_B_1wk_groups_engram_shuffle[group]))
    })
    #rm_data[group]['engram']['cell'] = np.arange(1,len(rm_data[group]['engram'])+1)
    rm_data[group]['engram']['type'].astype('category')
    rm_data[group]['engram']['type'].astype('category')

    rm_data[group]['low_activity'] = pd.DataFrame({
        #'subject': np.tile(np.arange(1, len(avg_TFC_cond_groups_low_activity[group])+1),6),
        'subject': np.concatenate((np.tile(np.arange(1, len(avg_TFC_cond_groups_low_activity[group])+1),3),
                                  np.tile(np.arange(len(avg_TFC_cond_groups_low_activity[group])+1, 2*len(avg_TFC_cond_groups_low_activity[group])+1),3))),
        'type': np.concatenate((np.repeat('real', len(avg_TFC_cond_groups_low_activity[group])),
                               np.repeat('real', len(avg_Test_B_groups_low_activity[group])),
                               np.repeat('real', len(avg_Test_B_1wk_groups_low_activity[group])),
                               np.repeat('shuffle', len(avg_TFC_cond_groups_low_activity_shuffle[group])),
                               np.repeat('shuffle', len(avg_Test_B_groups_low_activity_shuffle[group])),
                               np.repeat('shuffle', len(avg_Test_B_1wk_groups_low_activity_shuffle[group])))),
        'time': np.concatenate((np.repeat('TFC_cond', len(avg_TFC_cond_groups_low_activity[group])),
                               np.repeat('Test_B', len(avg_Test_B_groups_low_activity[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_low_activity[group])),
                               np.repeat('TFC_cond', len(avg_TFC_cond_groups_low_activity_shuffle[group])),
                               np.repeat('Test_B', len(avg_Test_B_groups_low_activity_shuffle[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_low_activity_shuffle[group])))),
        'value': np.concatenate((avg_TFC_cond_groups_low_activity[group],
                                 avg_Test_B_groups_low_activity[group],
                                 avg_Test_B_1wk_groups_low_activity[group],
                                 avg_TFC_cond_groups_low_activity_shuffle[group],
                                 avg_Test_B_groups_low_activity_shuffle[group],
                                 avg_Test_B_1wk_groups_low_activity_shuffle[group]))
    })
    #rm_data[group]['engram']['cell'] = np.arange(1,len(rm_data[group]['engram'])+1)
    rm_data[group]['low_activity']['type'].astype('category')
    rm_data[group]['low_activity']['type'].astype('category')    
    #rm_data[group]['engram'].index.name='cell'
    #rm_data[group]['engram'].reindex(['cell','time','value'],axis=1)    
    '''
    anova_rm = AnovaRM(data, 'value', 'subject', within=['time','type'])
    res = anova_rm.fit()
    print(res)
    '''

    '''
    anova_rm = pg.rm_anova(dv='value', within=['type', 'time'], subject='subject', data=rm_data[group]['engram'], detailed=True)    
    print(anova_rm)
    posthoc = pg.pairwise_tests(dv='value', within='time', subject='subject', data=rm_data[group]['engram'], padjust='bonferroni')
    print(posthoc)
    posthoc = pg.pairwise_tests(dv='value', within='type', subject='subject', data=rm_data[group]['engram'], padjust='bonferroni')
    print(posthoc)
    
    anova_results = pg.mixed_anova(dv='value', within='time', between='type', subject='subject', data=rm_data[group]['engram'])

    data=rm_data['hM3D']['engram']
    pg.rm_anova(dv='value', within=['type','time'], subject='subject', data=data)
    stats_out = pg.pairwise_tests(dv='value', within=['type','time'], subject='subject', data=data, padjust='sidak')

    data=rm_data['hM3D']['engram']
    pg.rm_anova(dv='value', within=['time','type'], subject='subject', data=data)
    stats_out = pg.pairwise_tests(dv='value', within=['time','type'], subject='subject', data=data, padjust='sidak')
    '''

plot_it = True
stats_groups = {'hM3D': {'engram':None, 'low_activity':None},
                    'hM4D': {'engram':None, 'low_activity':None},
                    'mCherry': {'engram':None, 'low_activity':None}}
posthoc_time_groups = {'hM3D': {'engram':None, 'low_activity':None},
                    'hM4D': {'engram':None, 'low_activity':None},
                    'mCherry': {'engram':None, 'low_activity':None}}
for group in ['hM3D', 'hM4D', 'mCherry']:
    for type in ['engram', 'low_activity']:
        data=rm_data[group][type]

        #
        # Perform one-way ANOVA on real and shuffle separately.
        #
        data_real = data[data['type'] == 'real']
        data_shuffle = data[data['type'] == 'shuffle']

        stats_out_real = pg.pairwise_tests(dv='value', within='time', subject='subject', data=data_real, padjust='sidak')
        stats_out_shuffle = pg.pairwise_tests(dv='value', within='time', subject='subject', data=data_shuffle, padjust='sidak')

        for anova_str, data_anova in zip(['real','shuffle'], [data_real, data_shuffle]):
            model = ols('value ~ C(time)', data=data_anova).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print('*** {} {} {} 1-way ANOVA on time'.format(group, type, anova_str))
            print(anova_table)

            # If ANOVA is significant, perform post-hoc tests
            if anova_table['PR(>F)'][0] < 0.05:
                comp = mc.MultiComparison(data_real['value'], data_real['time'])
                post_hoc_res = comp.tukeyhsd()
                print(post_hoc_res)

        # Perform mixed anova
        aov = pg.mixed_anova(data=data, dv='value', within='time', subject='subject', between='type')
        stats_out = pg.pairwise_tests(dv='value', within='time', subject='subject', between='type', data=data, padjust='sidak')
        posthoc_time = pg.pairwise_tests(data=data, dv='value', within='time', subject='subject', padjust='sidak')

        stats_groups[group][type] = stats_out
        posthoc_time_groups[group][type] = posthoc_time
        print('*** {} {}'.format(group, type))
        print(stats_out)
        print('\n')
        if plot_it:
            plt.figure()
            sns.set()
            sns.pointplot(data=data, x='time', y='value', hue='type', dodge=True, markers=['o', 's'],
                    capsize=.1, errwidth=1, palette='colorblind')
            plt.title('{} {}'.format(group,type))

#
# Bar plots for mean of engram & low-activity activities vs shuffle
#
for group in ['hM3D', 'hM4D', 'mCherry']:
    for act_type in ['engram', 'low_activity']:
        if act_type == 'engram':
            real_groups = [avg_TFC_cond_groups_engram, avg_Test_B_groups_engram, avg_Test_B_1wk_groups_engram]
            shuffle_groups = [avg_TFC_cond_groups_engram_shuffle, avg_Test_B_groups_engram_shuffle, avg_Test_B_1wk_groups_engram_shuffle]
        if act_type == 'low_activity':
            real_groups = [avg_TFC_cond_groups_low_activity, avg_Test_B_groups_low_activity, avg_Test_B_1wk_groups_low_activity]
            shuffle_groups = [avg_TFC_cond_groups_low_activity_shuffle, avg_Test_B_groups_low_activity_shuffle, avg_Test_B_1wk_groups_low_activity_shuffle]                        

        fig, ax = plt.subplots(figsize=(12,6))
        fig.suptitle('{} {}'.format(group, act_type))

        num_sessions = 3
        tot_sessions = num_sessions*2 # *2 b/c have real & shuffle, so duplicated.
        x = range(tot_sessions)
        means = np.zeros(tot_sessions) 
        sems = np.zeros(tot_sessions)
        errbars = np.zeros((2,tot_sessions))
        sess_idx = 0
        for i in range(num_sessions):
            for group_data in real_groups, shuffle_groups:
                means[sess_idx] = np.mean(group_data[i][group])
                sems[sess_idx] = (np.std(group_data[i][group]) / np.sqrt(len(group_data[i][group])))
                errbars[1,sess_idx] = sems[sess_idx]
                sess_idx += 1
        ax.bar(x, means, yerr=errbars, color=np.array(list(zip(group_colours.values(),[0.5,0.5,0.5]))).ravel()) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
        ax.set_xticks(range(6))
        ax.set_xticklabels(['TFC-real','TFC-shuffle','B-real','B-shuffle','B+1wk-real','B+1wk-shuffle'])

        df = stats_groups[group][act_type]
        P_real_vs_shuffle = df[(df['Contrast']=='time * type') & (df['A']=='real') & (df['B']=='shuffle')]['p-corr'].iloc[:].tolist()

        df = posthoc_time_groups[group][act_type]
        P_groups = df['p-corr'].tolist()

        #G = [[0,1],[2,3],[4,5],[0,2],[0,4],[2,4]]
        #P = P_real_vs_shuffle + P_groups
        G = [[0,1],[2,3],[4,5]]
        P = P_real_vs_shuffle
        sigstar(ax,G,P)

#
# Compare time points across groups
#
want_bar = False
want_boxplot = True
want_scatter = False
want_violinplot = False
for act_type in ['engram', 'low_activity']:
    if act_type == 'engram':
        real_groups = [avg_TFC_cond_groups_engram, avg_Test_B_groups_engram, avg_Test_B_1wk_groups_engram]
    if act_type == 'low_activity':
        real_groups = [avg_TFC_cond_groups_low_activity, avg_Test_B_groups_low_activity, avg_Test_B_1wk_groups_low_activity]

    fig, axs = plt.subplots(1,3,figsize=(12,6))
    fig.suptitle('{}'.format(act_type))

    for g_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
        ax = axs[g_i]

        tot_x = 3 # 3 groups x 3 sessions + 2 blank in between
        x = range(tot_x)
        means = np.zeros(tot_x) 
        sems = np.zeros(tot_x)
        errbars = np.zeros((2,tot_x))  

        x_scatter = []
        y_scatter = []
        for i in range(3):
            this_data = real_groups[i][group]
            x_scatter.append(np.zeros(len(this_data))+i)
            y_scatter.append(this_data)
            means[i] = np.mean(this_data)
            sems[i] = (np.std(this_data) / np.sqrt(len(this_data)))
            errbars[1,i] = sems[i]

        df = posthoc_time_groups[group][act_type]
        P = df['p-corr'].tolist()
        G = [[0,1],[0,2],[1,2]]

        if want_bar:
            ax.bar(x, means, yerr=errbars, color=np.repeat(group_colours[group],3)) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
        if want_boxplot:
            bp = ax.boxplot([real_groups[0][group], real_groups[1][group], real_groups[2][group]], \
                notch=True, patch_artist=True, positions=range(3), showfliers=False, widths=0.7, showmeans=True, \
                meanprops=dict(color='black'), meanline=True)
            for p, c in zip(bp['boxes'], np.repeat(group_colours[group],3)):
                plt.setp(p,facecolor=c,alpha=0.5)
            for median in bp['medians']:
                plt.setp(median,color='black')
            #for mean in bp['means']:
            #    plt.setp(mean,color='black')
        if want_scatter:
            x_scatter = np.array(x_scatter).flatten()
            y_scatter = np.array(y_scatter).flatten()
            ax.scatter(x_scatter, y_scatter, marker='o', edgecolors='k')
        if want_violinplot:
            sns.violinplot([real_groups[0][group], real_groups[1][group], real_groups[2][group]], \
                color=group_colours[group], ax=ax)

        ax.set_xticks(range(3))
        ax.set_xticklabels(['TFC','B+48h','B+1wk'],rotation=45)
        sigstar(ax,G,P)
        ax.set_ylim([0,4])

aov_data = {'engram': {}, 'low_activity': {}}
for time_key, time_groups in zip(['TFC_cond', 'Test_B', 'Test_B_1wk'], \
        [avg_TFC_cond_groups_engram, avg_Test_B_groups_engram, avg_Test_B_1wk_groups_engram]):
    aov_data['engram'][time_key] = pd.DataFrame({
        'subject': np.arange(1, len(time_groups['hM3D']) + len(time_groups['hM4D']) + len(time_groups['mCherry'])+1),
        'group': np.concatenate((np.repeat('hM3D', len(time_groups['hM3D'])),
                               np.repeat('hM4D', len(time_groups['hM4D'])),
                               np.repeat('mCherry', len(time_groups['mCherry'])))),
        'value': np.concatenate((time_groups['hM3D'],
                                 time_groups['hM4D'],
                                 time_groups['mCherry']))
    })
for time_key, time_groups in zip(['TFC_cond', 'Test_B', 'Test_B_1wk'], \
        [avg_TFC_cond_groups_low_activity, avg_Test_B_groups_low_activity, avg_Test_B_1wk_groups_low_activity]):
    aov_data['low_activity'][time_key] = pd.DataFrame({
        'subject': np.arange(1, len(time_groups['hM3D']) + len(time_groups['hM4D']) + len(time_groups['mCherry'])+1),
        'group': np.concatenate((np.repeat('hM3D', len(time_groups['hM3D'])),
                               np.repeat('hM4D', len(time_groups['hM4D'])),
                               np.repeat('mCherry', len(time_groups['mCherry'])))),
        'value': np.concatenate((time_groups['hM3D'],
                                 time_groups['hM4D'],
                                 time_groups['mCherry']))
    })

stats_time = {'TFC_cond': {'engram':None, 'low_activity':None},
                    'Test_B': {'engram':None, 'low_activity':None},
                    'Test_B_1wk': {'engram':None, 'low_activity':None}}
posthoc_groups = {'TFC_cond': {'engram':None, 'low_activity':None},
                    'Test_B': {'engram':None, 'low_activity':None},
                    'Test_B_1wk': {'engram':None, 'low_activity':None}}
for time_group in ['TFC_cond', 'Test_B', 'Test_B_1wk']:
    for type in ['engram', 'low_activity']:
        data=aov_data[type][time_group]
        anova_table = pg.anova(data=data, dv='value', between='group')
        print('*** {} {} 1-way ANOVA on group'.format(time_group, type))
        print(anova_table)
        if anova_table['p-unc'].iloc[0] < 0.05:
            print('*** SIGNIFICANT. Multcompare:')
            comp = mc.MultiComparison(data['value'], data['group'])
            post_hoc_res = comp.tukeyhsd()
            print(post_hoc_res)

        # Perform mixed anova
        posthoc = pg.pairwise_tests(data=data, dv='value', between='group', padjust='sidak')

        posthoc_groups[time_group][type] = posthoc

#
# Compare groups across time points
#
want_bar = False
want_boxplot = True
want_scatter = False
want_violinplot = False
for act_type in ['engram', 'low_activity']:
    if act_type == 'engram':
        real_groups = [avg_TFC_cond_groups_engram, avg_Test_B_groups_engram, avg_Test_B_1wk_groups_engram]
    if act_type == 'low_activity':
        real_groups = [avg_TFC_cond_groups_low_activity, avg_Test_B_groups_low_activity, avg_Test_B_1wk_groups_low_activity]

    fig, axs = plt.subplots(1,3,figsize=(12,6))
    fig.suptitle('{}'.format(act_type))

    sess_keys = ['TFC_cond', 'Test_B', 'Test_B_1wk']
    for t_i, sess_groups in enumerate(real_groups):
        sess_str = sess_keys[t_i]
        ax = axs[t_i]

        tot_x = 3 # 3 groups x 3 sessions + 2 blank in between
        x = range(tot_x)
        means = np.zeros(tot_x) 
        sems = np.zeros(tot_x)
        errbars = np.zeros((2,tot_x))  

        x_scatter = []
        y_scatter = []
        for g_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
            this_data = real_groups[t_i][group]
            x_scatter.append(np.zeros(len(this_data))+i)
            y_scatter.append(this_data)
            means[g_i] = np.mean(this_data)
            sems[g_i] = (np.std(this_data) / np.sqrt(len(this_data)))
            errbars[1,g_i] = sems[g_i]

        df = posthoc_groups[sess_str][act_type]
        P = df['p-corr'].tolist()
        G = [[0,1],[0,2],[1,2]]

        if want_bar:
            ax.bar(x, means, yerr=errbars, color=group_colours.values()) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
        if want_boxplot:
            bp = ax.boxplot([real_groups[t_i]['hM3D'], real_groups[t_i]['hM4D'], real_groups[t_i]['mCherry']], \
                notch=True, patch_artist=True, positions=range(3), showfliers=False, widths=0.7, showmeans=True, \
                meanprops=dict(color='black'), meanline=True)
            for p, c in zip(bp['boxes'], group_colours.values()):
                plt.setp(p,facecolor=c,alpha=0.5)
            for median in bp['medians']:
                plt.setp(median,color='black')
            #for mean in bp['means']:
            #    plt.setp(mean,color='black')
        if want_scatter:
            x_scatter = np.array(x_scatter).flatten()
            y_scatter = np.array(y_scatter).flatten()
            ax.scatter(x_scatter, y_scatter, marker='o', edgecolors='k')
        if want_violinplot:
            sns.violinplot([real_groups[t_i]['hM3D'], real_groups[t_i]['hM4D'], real_groups[t_i]['mCherry']], \
                color=group_colours.values(), ax=ax)

        ax.set_xticks(range(3))
        ax.set_xticklabels(['Exc','Inh','Ctl'],rotation=45)
        ax.set_title(sess_str)
        sigstar(ax,G,P)
        if act_type == 'engram':
            ax.set_ylim([0,4])
        if act_type == 'low-activity':
            ax.set_ylim([0,3])


# The shuffle dicts have to be separate per engram/low_activity groups to match the number of
# cells in each (which varies)
avg_Test_B_groups_engram_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_engram_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

avg_Test_B_groups_engram_shuffle_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_engram_shuffle_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

avg_Test_B_groups_low_activity_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_low_activity_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

avg_Test_B_groups_low_activity_shuffle_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
avg_Test_B_1wk_groups_low_activity_shuffle_recall = {'hM3D':[], 'hM4D':[], 'mCherry':[]}

mapping='Test_B+Test_B_1wk'
plot_individual = False
for mouse, group in mouse_groups.items():
    if mouse in ['G07', 'G15']:
        continue
    [S_Test_B, S_spikes_Test_B, S_peakval_Test_B, S_idx_Test_B] = \
        Test_B[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])
    [S_Test_B_1wk, S_spikes_Test_B_1wk, S_peakval_Test_B_1wk, S_idx_Test_B_1wk] = \
        Test_B_1wk[mouse].get_S_mapping(mapping, with_peakval=True, with_crossreg=TFC_B_B_1wk_crossreg[mouse])

    for shuffle_type in ['real', 'shuffle']:
        for act_type in ['engram', 'low_activity']:
            # Engram
            if act_type == 'engram':
                S_Test_B_engram_mask_recall, S_Test_B_engram_indices_recall, S_Test_B_engram_recall, S_Test_B_engram_spikes_recall, S_Test_B_engram_peakval_recall = \
                    get_engram_cells(S_Test_B, S_spikes_Test_B, S_peakval_Test_B)
            # Non-engram (low activity)
            if act_type == 'low_activity':
                S_Test_B_engram_mask_recall, S_Test_B_engram_indices_recall, S_Test_B_engram_recall, S_Test_B_engram_spikes_recall, S_Test_B_engram_peakval_recall = \
                    get_engram_cells(S_Test_B, S_spikes_Test_B, S_peakval_Test_B, zscore_thresh=0, want_low_activity=True)

            if shuffle_type == 'shuffle':
                S_mask=random.sample(range(1,S_Test_B.shape[0]), len(S_Test_B_engram_indices_recall))
            if shuffle_type == 'real':
                S_mask = S_Test_B_engram_mask_recall
            avg_Test_B = np.sum(S_Test_B[S_mask,:],1)/S_Test_B.shape[1]
            avg_Test_B_1wk = np.sum(S_Test_B_1wk[S_mask,:],1)/S_Test_B_1wk.shape[1]

            if act_type == 'engram':
                if shuffle_type == 'real':
                    avg_Test_B_groups_engram_recall[group] = np.append(avg_Test_B_groups_engram_recall[group], avg_Test_B)
                    avg_Test_B_1wk_groups_engram_recall[group] = np.append(avg_Test_B_1wk_groups_engram_recall[group], avg_Test_B_1wk)
                if shuffle_type == 'shuffle':
                    avg_Test_B_groups_engram_shuffle_recall[group] = np.append(avg_Test_B_groups_engram_shuffle_recall[group], avg_Test_B)
                    avg_Test_B_1wk_groups_engram_shuffle_recall[group] = np.append(avg_Test_B_1wk_groups_engram_shuffle_recall[group], avg_Test_B_1wk)
            if act_type == 'low_activity':
                if shuffle_type == 'real':
                    avg_Test_B_groups_low_activity_recall[group] = np.append(avg_Test_B_groups_low_activity_recall[group], avg_Test_B)
                    avg_Test_B_1wk_groups_low_activity_recall[group] = np.append(avg_Test_B_1wk_groups_low_activity_recall[group], avg_Test_B_1wk)
                if shuffle_type == 'shuffle':
                    avg_Test_B_groups_low_activity_shuffle_recall[group] = np.append(avg_Test_B_groups_low_activity_shuffle_recall[group], avg_Test_B)
                    avg_Test_B_1wk_groups_low_activity_shuffle_recall[group] = np.append(avg_Test_B_1wk_groups_low_activity_shuffle_recall[group], avg_Test_B_1wk)

            if plot_individual:
                plt.figure()
                for i in range(len(S_Test_B_engram_indices_recall)-1):
                    plt.plot([avg_Test_B[i], avg_Test_B_1wk[i]])
                plt.title('{} {} {}'.format(mouse, act_type, shuffle_type))

for act_type in ['engram', 'low_activity', 'engram-shuffle', 'low_activity-shuffle']:
    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)

    for ax_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
        if act_type == 'engram':
            act_groups = [avg_Test_B_groups_engram_recall, avg_Test_B_1wk_groups_engram_recall]
        if act_type == 'low_activity':
            act_groups = [avg_Test_B_groups_low_activity_recall, avg_Test_B_1wk_groups_low_activity_recall]
        if act_type == 'engram-shuffle':
            act_groups = [avg_Test_B_groups_engram_shuffle_recall, avg_Test_B_1wk_groups_engram_shuffle_recall]
        if act_type == 'low_activity-shuffle':
            act_groups = [avg_Test_B_groups_low_activity_shuffle_recall, avg_Test_B_1wk_groups_low_activity_shuffle_recall]
        ax = axs[ax_i]
        for i in range(len(act_groups[0][group])-1):
            ax.plot([act_groups[0][group][i], act_groups[1][group][i]], color='0.8', alpha=0.5)
        ax.plot([np.mean(act_groups[0][group]), np.mean(act_groups[1][group])], color='black')
        ax.set_xticks([0,1],labels=['Test B','Test B +1wk'])        
        ax.set_ylim([0,2])
        ax.set_title('{} {}'.format(group, act_type))
    fig.tight_layout()

rm_data_recall = {}
for group in ['hM3D', 'hM4D', 'mCherry']:
    rm_data_recall[group] = {}
    rm_data_recall[group]['engram'] = pd.DataFrame({
        #'subject': np.tile(np.arange(1, len(avg_TFC_cond_groups_engram[group])+1),6),
        'subject': np.concatenate((np.tile(np.arange(1, len(avg_Test_B_groups_engram_recall[group])+1),2),
                                  np.tile(np.arange(len(avg_Test_B_groups_engram_recall[group])+1, 2*len(avg_Test_B_groups_engram_recall[group])+1),2))),
        'type': np.concatenate((np.repeat('real', len(avg_Test_B_groups_engram_recall[group])),
                               np.repeat('real', len(avg_Test_B_1wk_groups_engram_recall[group])),
                               np.repeat('shuffle', len(avg_Test_B_groups_engram_shuffle_recall[group])),
                               np.repeat('shuffle', len(avg_Test_B_1wk_groups_engram_shuffle_recall[group])))),
        'time': np.concatenate((np.repeat('Test_B', len(avg_Test_B_groups_engram_recall[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_engram_recall[group])),
                               np.repeat('Test_B', len(avg_Test_B_groups_engram_shuffle_recall[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_engram_shuffle_recall[group])))),
        'value': np.concatenate((avg_Test_B_groups_engram_recall[group],
                                 avg_Test_B_1wk_groups_engram_recall[group],
                                 avg_Test_B_groups_engram_shuffle_recall[group],
                                 avg_Test_B_1wk_groups_engram_shuffle_recall[group]))
    })
    #rm_data[group]['engram']['cell'] = np.arange(1,len(rm_data[group]['engram'])+1)
    rm_data_recall[group]['engram']['type'].astype('category')
    rm_data_recall[group]['engram']['type'].astype('category')

    rm_data_recall[group]['low_activity'] = pd.DataFrame({
        #'subject': np.tile(np.arange(1, len(avg_TFC_cond_groups_low_activity[group])+1),6),
        'subject': np.concatenate((np.tile(np.arange(1, len(avg_Test_B_groups_low_activity_recall[group])+1),2),
                                  np.tile(np.arange(len(avg_Test_B_groups_low_activity_recall[group])+1, 2*len(avg_Test_B_groups_low_activity_recall[group])+1),2))),
        'type': np.concatenate((np.repeat('real', len(avg_Test_B_groups_low_activity_recall[group])),
                               np.repeat('real', len(avg_Test_B_1wk_groups_low_activity_recall[group])),
                               np.repeat('shuffle', len(avg_Test_B_groups_low_activity_shuffle_recall[group])),
                               np.repeat('shuffle', len(avg_Test_B_1wk_groups_low_activity_shuffle_recall[group])))),
        'time': np.concatenate((np.repeat('Test_B', len(avg_Test_B_groups_low_activity_recall[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_low_activity_recall[group])),
                               np.repeat('Test_B', len(avg_Test_B_groups_low_activity_shuffle_recall[group])),
                               np.repeat('Test_B_1wk', len(avg_Test_B_1wk_groups_low_activity_shuffle_recall[group])))),
        'value': np.concatenate((avg_Test_B_groups_low_activity_recall[group],
                                 avg_Test_B_1wk_groups_low_activity_recall[group],
                                 avg_Test_B_groups_low_activity_shuffle_recall[group],
                                 avg_Test_B_1wk_groups_low_activity_shuffle_recall[group]))
    })
    #rm_data[group]['engram']['cell'] = np.arange(1,len(rm_data[group]['engram'])+1)
    rm_data_recall[group]['low_activity']['type'].astype('category')
    rm_data_recall[group]['low_activity']['type'].astype('category')    
    #rm_data[group]['engram'].index.name='cell'
    #rm_data[group]['engram'].reindex(['cell','time','value'],axis=1)    
    '''
    anova_rm = AnovaRM(data, 'value', 'subject', within=['time','type'])
    res = anova_rm.fit()
    print(res)
    '''

    '''
    anova_rm = pg.rm_anova(dv='value', within=['type', 'time'], subject='subject', data=rm_data[group]['engram'], detailed=True)    
    print(anova_rm)
    posthoc = pg.pairwise_tests(dv='value', within='time', subject='subject', data=rm_data[group]['engram'], padjust='bonferroni')
    print(posthoc)
    posthoc = pg.pairwise_tests(dv='value', within='type', subject='subject', data=rm_data[group]['engram'], padjust='bonferroni')
    print(posthoc)
    
    anova_results = pg.mixed_anova(dv='value', within='time', between='type', subject='subject', data=rm_data[group]['engram'])

    data=rm_data['hM3D']['engram']
    pg.rm_anova(dv='value', within=['type','time'], subject='subject', data=data)
    stats_out = pg.pairwise_tests(dv='value', within=['type','time'], subject='subject', data=data, padjust='sidak')

    data=rm_data['hM3D']['engram']
    pg.rm_anova(dv='value', within=['time','type'], subject='subject', data=data)
    stats_out = pg.pairwise_tests(dv='value', within=['time','type'], subject='subject', data=data, padjust='sidak')
    '''

plot_it = True
stats_groups_recall = {'hM3D': {'engram':None, 'low_activity':None},
                    'hM4D': {'engram':None, 'low_activity':None},
                    'mCherry': {'engram':None, 'low_activity':None}}
posthoc_time_groups_recall = {'hM3D': {'engram':None, 'low_activity':None},
                    'hM4D': {'engram':None, 'low_activity':None},
                    'mCherry': {'engram':None, 'low_activity':None}}
for group in ['hM3D', 'hM4D', 'mCherry']:
    for type in ['engram', 'low_activity']:
        data=rm_data_recall[group][type]

        #
        # Perform one-way ANOVA on real and shuffle separately.
        #
        data_real = data[data['type'] == 'real']
        data_shuffle = data[data['type'] == 'shuffle']
        
        data_real_B = data[data['type']=='real'][data['time'] == 'Test_B']
        data_real_B_1wk = data[data['type']=='real'][data['time'] == 'Test_B_1wk']
        data_shuffle_B = data[data['type']=='real'][data['time'] == 'Test_B']
        data_shuffle_B_1wk = data[data['type']=='real'][data['time'] == 'Test_B_1wk']

        stats_out_real = pg.pairwise_tests(dv='value', within='time', subject='subject', data=data_real, padjust='sidak')
        stats_out_shuffle = pg.pairwise_tests(dv='value', within='time', subject='subject', data=data_shuffle, padjust='sidak')

        for anova_str, data_anova in zip(['real','shuffle'], [data_real, data_shuffle]):
            model = ols('value ~ C(time)', data=data_anova).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print('*** {} {} {} 1-way ANOVA on time'.format(group, type, anova_str))
            print(anova_table)

            # If ANOVA is significant, perform post-hoc tests
            if anova_table['PR(>F)'][0] < 0.05:
                comp = mc.MultiComparison(data_real['value'], data_real['time'])
                post_hoc_res = comp.tukeyhsd()
                print(post_hoc_res)

        #stats_out = pg.ttest(data_real_B, data_real_B_1wk)
        # Perform mixed anova
        aov = pg.mixed_anova(data=data, dv='value', within='time', subject='subject', between='type')
        stats_out = pg.pairwise_tests(dv='value', within='time', subject='subject', between='type', data=data, padjust='sidak')
        posthoc_time = pg.pairwise_tests(data=data, dv='value', within='time', subject='subject', padjust='sidak')

        stats_groups_recall[group][type] = stats_out
        posthoc_time_groups_recall[group][type] = posthoc_time
        print('*** {} {}'.format(group, type))
        print(stats_out)
        print('\n')
        if plot_it:
            plt.figure()
            sns.set()
            sns.pointplot(data=data, x='time', y='value', hue='type', dodge=True, markers=['o', 's'],
                    capsize=.1, errwidth=1, palette='colorblind')
            plt.title('{} {}'.format(group,type))

#
# Bar plots for mean of engram & low-activity activities vs shuffle
#
for group in ['hM3D', 'hM4D', 'mCherry']:
    for act_type in ['engram', 'low_activity']:
        if act_type == 'engram':
            real_groups_recall = [avg_Test_B_groups_engram_recall, avg_Test_B_1wk_groups_engram_recall]
            shuffle_groups_recall = [avg_Test_B_groups_engram_shuffle_recall, avg_Test_B_1wk_groups_engram_shuffle_recall]
        if act_type == 'low_activity':
            real_groups_recall = [avg_Test_B_groups_low_activity_recall, avg_Test_B_1wk_groups_low_activity_recall]
            shuffle_groups_recall = [avg_Test_B_groups_low_activity_shuffle_recall, avg_Test_B_1wk_groups_low_activity_shuffle_recall]                        

        fig, ax = plt.subplots(figsize=(12,6))
        fig.suptitle('{} {}'.format(group, act_type))

        num_sessions = len(real_groups_recall)
        tot_sessions = num_sessions*2 # *2 b/c have real & shuffle, so duplicated.
        x = range(tot_sessions)
        means = np.zeros(tot_sessions) 
        sems = np.zeros(tot_sessions)
        errbars = np.zeros((2,tot_sessions))
        sess_idx = 0
        for i in range(num_sessions):
            for group_data in real_groups_recall, shuffle_groups_recall:
                means[sess_idx] = np.mean(group_data[i][group])
                sems[sess_idx] = (np.std(group_data[i][group]) / np.sqrt(len(group_data[i][group])))
                errbars[1,sess_idx] = sems[sess_idx]
                sess_idx += 1
        ax.bar(x, means, yerr=errbars, color=np.array(list(zip(np.tile(np.array(group_colours[group]),3),[0.5,0.5,0.5]))).ravel()) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
        ax.set_xticks(range(tot_sessions))
        ax.set_xticklabels(['B-real','B-shuffle','B+1wk-real','B+1wk-shuffle'])

        df = stats_groups_recall[group][act_type]
        P_real_vs_shuffle_recall = df[(df['Contrast']=='time * type') & (df['A']=='real') & (df['B']=='shuffle')]['p-corr'].iloc[:].tolist()

        df = posthoc_time_groups_recall[group][act_type]
        P_groups_recall = df['p-unc'].tolist()

        G = [[0,1],[2,3]]
        P = P_real_vs_shuffle_recall
        sigstar(ax,G,P)

#
# Compare time points across groups
#
want_bar = False
want_boxplot = True
want_scatter = False
want_violinplot = False
for act_type in ['engram', 'low_activity']:
    if act_type == 'engram':
        real_groups = [avg_Test_B_groups_engram_recall, avg_Test_B_1wk_groups_engram_recall]
    if act_type == 'low_activity':
        real_groups = [avg_Test_B_groups_low_activity_recall, avg_Test_B_1wk_groups_low_activity_recall]

    fig, axs = plt.subplots(1,3,figsize=(12,6))
    fig.suptitle('{}'.format(act_type))

    for g_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
        ax = axs[g_i]

        tot_x = 3 # 3 groups x 3 sessions + 2 blank in between
        x = range(tot_x)
        means = np.zeros(tot_x) 
        sems = np.zeros(tot_x)
        errbars = np.zeros((2,tot_x))  

        x_scatter = []
        y_scatter = []
        for i in range(len(real_groups)):
            this_data = real_groups[i][group]
            x_scatter.append(np.zeros(len(this_data))+i)
            y_scatter.append(this_data)
            means[i] = np.mean(this_data)
            sems[i] = (np.std(this_data) / np.sqrt(len(this_data)))
            errbars[1,i] = sems[i]

        df = posthoc_time_groups_recall[group][act_type]
        P = df['p-unc'].tolist()
        G = [[0,1]]

        if want_bar:
            ax.bar(x, means, yerr=errbars, color=np.repeat(group_colours[group],3)) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
        if want_boxplot:
            bp = ax.boxplot([real_groups[0][group], real_groups[1][group]], \
                notch=True, patch_artist=True, positions=range(len(real_groups)), showfliers=False, widths=0.7, showmeans=True, \
                meanprops=dict(color='black'), meanline=True)
            for p, c in zip(bp['boxes'], np.repeat(group_colours[group],len(real_groups))):
                plt.setp(p,facecolor=c,alpha=0.5)
            for median in bp['medians']:
                plt.setp(median,color='black')
            #for mean in bp['means']:
            #    plt.setp(mean,color='black')
        if want_scatter:
            x_scatter = np.array(x_scatter).flatten()
            y_scatter = np.array(y_scatter).flatten()
            ax.scatter(x_scatter, y_scatter, marker='o', edgecolors='k')
        if want_violinplot:
            sns.violinplot([real_groups[0][group], real_groups[1][group]], \
                color=group_colours[group], ax=ax)

        ax.set_xticks(range(len(real_groups)))
        ax.set_xticklabels(['B+48h','B+1wk'],rotation=45)
        sigstar(ax,G,P)
        ax.set_ylim([0,4])


aov_data_recall = {'engram': {}, 'low_activity': {}}
for time_key, time_groups in zip(['Test_B', 'Test_B_1wk'], \
        [avg_Test_B_groups_engram_recall, avg_Test_B_1wk_groups_engram_recall]):
    aov_data_recall['engram'][time_key] = pd.DataFrame({
        'subject': np.arange(1, len(time_groups['hM3D']) + len(time_groups['hM4D']) + len(time_groups['mCherry'])+1),
        'group': np.concatenate((np.repeat('hM3D', len(time_groups['hM3D'])),
                               np.repeat('hM4D', len(time_groups['hM4D'])),
                               np.repeat('mCherry', len(time_groups['mCherry'])))),
        'value': np.concatenate((time_groups['hM3D'],
                                 time_groups['hM4D'],
                                 time_groups['mCherry']))
    })
for time_key, time_groups in zip(['Test_B', 'Test_B_1wk'], \
        [avg_Test_B_groups_low_activity_recall, avg_Test_B_1wk_groups_low_activity_recall]):
    aov_data_recall['low_activity'][time_key] = pd.DataFrame({
        'subject': np.arange(1, len(time_groups['hM3D']) + len(time_groups['hM4D']) + len(time_groups['mCherry'])+1),
        'group': np.concatenate((np.repeat('hM3D', len(time_groups['hM3D'])),
                               np.repeat('hM4D', len(time_groups['hM4D'])),
                               np.repeat('mCherry', len(time_groups['mCherry'])))),
        'value': np.concatenate((time_groups['hM3D'],
                                 time_groups['hM4D'],
                                 time_groups['mCherry']))
    })

stats_time_recall = {'Test_B': {'engram':None, 'low_activity':None},
                    'Test_B_1wk': {'engram':None, 'low_activity':None}}
posthoc_groups_recall = {'Test_B': {'engram':None, 'low_activity':None},
                    'Test_B_1wk': {'engram':None, 'low_activity':None}}
for time_group in ['Test_B', 'Test_B_1wk']:
    for type in ['engram', 'low_activity']:
        data=aov_data_recall[type][time_group]
        anova_table = pg.anova(data=data, dv='value', between='group')
        print('*** {} {} 1-way ANOVA on group'.format(time_group, type))
        print(anova_table)
        if anova_table['p-unc'].iloc[0] < 0.05:
            print('*** SIGNIFICANT. Multcompare:')
            comp = mc.MultiComparison(data['value'], data['group'])
            post_hoc_res = comp.tukeyhsd()
            print(post_hoc_res)

        # Perform mixed anova
        posthoc = pg.pairwise_tests(data=data, dv='value', between='group', padjust='sidak')
        posthoc_groups_recall[time_group][type] = posthoc

#
# Compare groups across time points
#
want_bar = False
want_boxplot = True
want_scatter = False
want_violinplot = False
for act_type in ['engram', 'low_activity']:
    if act_type == 'engram':
        real_groups = [avg_Test_B_groups_engram_recall, avg_Test_B_1wk_groups_engram_recall]
    if act_type == 'low_activity':
        real_groups = [avg_Test_B_groups_low_activity_recall, avg_Test_B_1wk_groups_low_activity_recall]

    fig, axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle('{}'.format(act_type))

    sess_keys = ['Test_B', 'Test_B_1wk']
    for t_i, sess_groups in enumerate(real_groups):
        sess_str = sess_keys[t_i]
        ax = axs[t_i]

        tot_x = 3 # 3 groups x 3 sessions + 2 blank in between
        x = range(tot_x)
        means = np.zeros(tot_x) 
        sems = np.zeros(tot_x)
        errbars = np.zeros((2,tot_x))  

        x_scatter = []
        y_scatter = []
        for g_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
            this_data = real_groups[t_i][group]
            x_scatter.append(np.zeros(len(this_data))+i)
            y_scatter.append(this_data)
            means[g_i] = np.mean(this_data)
            sems[g_i] = (np.std(this_data) / np.sqrt(len(this_data)))
            errbars[1,g_i] = sems[g_i]

        df = posthoc_groups_recall[sess_str][act_type]
        P = df['p-corr'].tolist()
        G = [[0,1],[0,2],[1,2]]

        if want_bar:
            ax.bar(x, means, yerr=errbars, color=group_colours.values()) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
        if want_boxplot:
            bp = ax.boxplot([real_groups[t_i]['hM3D'], real_groups[t_i]['hM4D'], real_groups[t_i]['mCherry']], \
                notch=True, patch_artist=True, positions=range(3), showfliers=False, widths=0.7, showmeans=True, \
                meanprops=dict(color='black'), meanline=True)
            for p, c in zip(bp['boxes'], group_colours.values()):
                plt.setp(p,facecolor=c,alpha=0.5)
            for median in bp['medians']:
                plt.setp(median,color='black')
            #for mean in bp['means']:
            #    plt.setp(mean,color='black')
        if want_scatter:
            x_scatter = np.array(x_scatter).flatten()
            y_scatter = np.array(y_scatter).flatten()
            ax.scatter(x_scatter, y_scatter, marker='o', edgecolors='k')
        if want_violinplot:
            sns.violinplot([real_groups[t_i]['hM3D'], real_groups[t_i]['hM4D'], real_groups[t_i]['mCherry']], \
                color=group_colours.values(), ax=ax)

        ax.set_xticks(range(3))
        ax.set_xticklabels(['Exc','Inh','Ctl'],rotation=45)
        ax.set_title(sess_str)
        sigstar(ax,G,P)
        if act_type == 'engram':
            ax.set_ylim([0,4])
        if act_type == 'low-activity':
            ax.set_ylim([0,3])

#
# Pearson's correlation of binned neuronal activities
# 
if want_correlations:
    rm_data_corr = {}
    for group in ['hM3D', 'hM4D', 'mCherry']:
        rm_data_corr[group] = {}
        rm_data_corr[group]['engram'] = pd.DataFrame({
            #'subject': np.tile(np.arange(1, len(avg_TFC_cond_groups_engram[group])+1),6),
            'subject': np.concatenate((np.tile(np.arange(1, len(corr_sess_engram['TFC_cond'][group])+1),3),
                                    np.tile(np.arange(len(corr_sess_engram['TFC_cond'][group])+1, 2*len(corr_sess_engram['TFC_cond'][group])+1),3))),
            'type': np.concatenate((np.repeat('real', len(corr_sess_engram['TFC_cond'][group])),
                                np.repeat('real', len(corr_sess_engram['Test_B'][group])),
                                np.repeat('real', len(corr_sess_engram['Test_B_1wk'][group])),
                                np.repeat('shuffle', len(corr_shuffle_engram['TFC_cond'][group])),
                                np.repeat('shuffle', len(corr_shuffle_engram['Test_B'][group])),
                                np.repeat('shuffle', len(corr_shuffle_engram['Test_B_1wk'][group])))),
            'time': np.concatenate((np.repeat('TFC_cond', len(corr_sess_engram['TFC_cond'][group])),
                                np.repeat('Test_B', len(corr_sess_engram['Test_B'][group])),
                                np.repeat('Test_B_1wk', len(corr_sess_engram['Test_B_1wk'][group])),
                                np.repeat('TFC_cond', len(corr_shuffle_engram['TFC_cond'][group])),
                                np.repeat('Test_B', len(corr_shuffle_engram['Test_B'][group])),
                                np.repeat('Test_B_1wk', len(corr_shuffle_engram['Test_B_1wk'][group])))),
            'value': np.concatenate((corr_sess_engram['TFC_cond'][group],
                                    corr_sess_engram['Test_B'][group],
                                    corr_sess_engram['Test_B_1wk'][group],
                                    corr_shuffle_engram['TFC_cond'][group],
                                    corr_shuffle_engram['Test_B'][group],
                                    corr_shuffle_engram['Test_B_1wk'][group]))
        })
        rm_data_corr[group]['engram']['type'].astype('category')

        rm_data_corr[group]['low_activity'] = pd.DataFrame({
            #'subject': np.tile(np.arange(1, len(avg_TFC_cond_groups_engram[group])+1),6),
            'subject': np.concatenate((np.tile(np.arange(1, len(corr_sess_low_activity['TFC_cond'][group])+1),3),
                                    np.tile(np.arange(len(corr_sess_low_activity['TFC_cond'][group])+1, 2*len(corr_sess_low_activity['TFC_cond'][group])+1),3))),
            'type': np.concatenate((np.repeat('real', len(corr_sess_low_activity['TFC_cond'][group])),
                                np.repeat('real', len(corr_sess_low_activity['Test_B'][group])),
                                np.repeat('real', len(corr_sess_low_activity['Test_B_1wk'][group])),
                                np.repeat('shuffle', len(corr_shuffle_low_activity['TFC_cond'][group])),
                                np.repeat('shuffle', len(corr_shuffle_low_activity['Test_B'][group])),
                                np.repeat('shuffle', len(corr_shuffle_low_activity['Test_B_1wk'][group])))),
            'time': np.concatenate((np.repeat('TFC_cond', len(corr_sess_low_activity['TFC_cond'][group])),
                                np.repeat('Test_B', len(corr_sess_low_activity['Test_B'][group])),
                                np.repeat('Test_B_1wk', len(corr_sess_low_activity['Test_B_1wk'][group])),
                                np.repeat('TFC_cond', len(corr_shuffle_low_activity['TFC_cond'][group])),
                                np.repeat('Test_B', len(corr_shuffle_low_activity['Test_B'][group])),
                                np.repeat('Test_B_1wk', len(corr_shuffle_low_activity['Test_B_1wk'][group])))),
            'value': np.concatenate((corr_sess_low_activity['TFC_cond'][group],
                                    corr_sess_low_activity['Test_B'][group],
                                    corr_sess_low_activity['Test_B_1wk'][group],
                                    corr_shuffle_low_activity['TFC_cond'][group],
                                    corr_shuffle_low_activity['Test_B'][group],
                                    corr_shuffle_low_activity['Test_B_1wk'][group]))
        })
        rm_data_corr[group]['low_activity']['type'].astype('category')

    plot_it = True
    stats_groups_corr = {'hM3D': {'engram':None, 'low_activity':None},
                        'hM4D': {'engram':None, 'low_activity':None},
                        'mCherry': {'engram':None, 'low_activity':None}}
    posthoc_time_groups_corr = {'hM3D': {'engram':None, 'low_activity':None},
                        'hM4D': {'engram':None, 'low_activity':None},
                        'mCherry': {'engram':None, 'low_activity':None}}
    for group in ['hM3D', 'hM4D', 'mCherry']:
        for type in ['engram', 'low_activity']:
            data=rm_data_corr[group][type]

            #
            # Perform one-way ANOVA on real and shuffle separately.
            #
            data_real = data[data['type'] == 'real']
            data_shuffle = data[data['type'] == 'shuffle']

            stats_out_real = pg.pairwise_tests(dv='value', within='time', subject='subject', data=data_real, padjust='sidak')
            stats_out_shuffle = pg.pairwise_tests(dv='value', within='time', subject='subject', data=data_shuffle, padjust='sidak')

            for anova_str, data_anova in zip(['real','shuffle'], [data_real, data_shuffle]):
                model = ols('value ~ C(time)', data=data_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                print('*** {} {} {} 1-way ANOVA on time'.format(group, type, anova_str))
                print(anova_table)

                # If ANOVA is significant, perform post-hoc tests
                if anova_table['PR(>F)'][0] < 0.05:
                    comp = mc.MultiComparison(data_real['value'], data_real['time'])
                    post_hoc_res = comp.tukeyhsd()
                    print(post_hoc_res)

            # Perform mixed anova
            aov = pg.mixed_anova(data=data, dv='value', within='time', subject='subject', between='type')
            stats_out = pg.pairwise_tests(dv='value', within='time', subject='subject', between='type', data=data, padjust='sidak')
            posthoc_time = pg.pairwise_tests(data=data, dv='value', within='time', subject='subject', padjust='sidak')

            stats_groups_corr[group][type] = stats_out
            posthoc_time_groups_corr[group][type] = posthoc_time
            print('*** {} {}'.format(group, type))
            print(stats_out)
            print('\n')
            if plot_it:
                plt.figure()
                sns.set()
                sns.pointplot(data=data, x='time', y='value', hue='type', dodge=True, markers=['o', 's'],
                        capsize=.1, errwidth=1, palette='colorblind')
                plt.title('{} {}'.format(group,type))

    #
    # Bar plots for mean of engram & low-activity correlations vs shuffle
    #
    for group in ['hM3D', 'hM4D', 'mCherry']:
        for act_type in ['engram', 'low_activity']:
            if act_type == 'engram':
                real_groups = [corr_sess_engram['TFC_cond'], corr_sess_engram['Test_B'], corr_sess_engram['Test_B_1wk']]
                #shuffle_groups = [corr_shuffle_engram['TFC_cond'], corr_shuffle_engram['Test_B'], corr_shuffle_engram['Test_B_1wk']]
                shuffle_groups = [corr_shuffle_engram['TFC_cond'], corr_shuffle_engram['Test_B'], corr_shuffle_engram['Test_B_1wk']]
            if act_type == 'low_activity':
                real_groups = [corr_sess_low_activity['TFC_cond'], corr_sess_low_activity['Test_B'], corr_sess_low_activity['Test_B_1wk']]
                shuffle_groups = [corr_shuffle_low_activity['TFC_cond'], corr_shuffle_low_activity['Test_B'], corr_shuffle_low_activity['Test_B_1wk']]

            fig, ax = plt.subplots(figsize=(12,6))
            fig.suptitle('{} {}'.format(group, act_type))

            num_sessions = 3
            tot_sessions = num_sessions*2 # *2 b/c have real & shuffle, so duplicated.
            x = range(tot_sessions)
            means = np.zeros(tot_sessions) 
            sems = np.zeros(tot_sessions)
            errbars = np.zeros((2,tot_sessions))
            sess_idx = 0
            for i in range(num_sessions):
                for group_data in real_groups, shuffle_groups:
                    means[sess_idx] = np.mean(group_data[i][group])
                    sems[sess_idx] = (np.std(group_data[i][group]) / np.sqrt(len(group_data[i][group])))
                    errbars[1,sess_idx] = sems[sess_idx]
                    sess_idx += 1
            ax.bar(x, means, yerr=errbars, color=np.array(list(zip(group_colours.values(),[0.5,0.5,0.5]))).ravel()) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
            ax.set_xticks(range(6))
            ax.set_xticklabels(['TFC-real','TFC-shuffle','B-real','B-shuffle','B+1wk-real','B+1wk-shuffle'])

            df = stats_groups_corr[group][act_type]
            P_real_vs_shuffle = df[(df['Contrast']=='time * type') & (df['A']=='real') & (df['B']=='shuffle')]['p-corr'].iloc[:].tolist()

            df = posthoc_time_groups_corr[group][act_type]
            P_groups = df['p-corr'].tolist()

            #G = [[0,1],[2,3],[4,5],[0,2],[0,4],[2,4]]
            #P = P_real_vs_shuffle + P_groups
            G = [[0,1],[2,3],[4,5]]
            P = P_real_vs_shuffle
            sigstar(ax,G,P)

    #
    # Compare time points across groups
    #
    want_bar = False
    want_boxplot = True
    want_scatter = False
    want_violinplot = False
    for act_type in ['engram', 'low_activity']:
        if act_type == 'engram':
            real_groups = [corr_sess_engram['TFC_cond'], corr_sess_engram['Test_B'], corr_sess_engram['Test_B_1wk']]
        if act_type == 'low_activity':
            real_groups = [corr_sess_low_activity['TFC_cond'], corr_sess_low_activity['Test_B'], corr_sess_low_activity['Test_B_1wk']]

        fig, axs = plt.subplots(1,3,figsize=(12,6))
        fig.suptitle('{}'.format(act_type))

        for g_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
            ax = axs[g_i]

            tot_x = 3 # 3 groups x 3 sessions + 2 blank in between
            x = range(tot_x)
            means = np.zeros(tot_x) 
            sems = np.zeros(tot_x)
            errbars = np.zeros((2,tot_x))  

            x_scatter = []
            y_scatter = []
            for i in range(3):
                this_data = real_groups[i][group]
                x_scatter.append(np.zeros(len(this_data))+i)
                y_scatter.append(this_data)
                means[i] = np.mean(this_data)
                sems[i] = (np.std(this_data) / np.sqrt(len(this_data)))
                errbars[1,i] = sems[i]

            df = posthoc_time_groups_corr[group][act_type]
            P = df['p-corr'].tolist()
            G = [[0,1],[0,2],[1,2]]

            if want_bar:
                ax.bar(x, means, yerr=errbars, color=np.repeat(group_colours[group],3)) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
            if want_boxplot:
                bp = ax.boxplot([real_groups[0][group], real_groups[1][group], real_groups[2][group]], \
                    notch=True, patch_artist=True, positions=range(3), showfliers=False, widths=0.7, showmeans=True, \
                    meanprops=dict(color='black'), meanline=True)
                for p, c in zip(bp['boxes'], np.repeat(group_colours[group],3)):
                    plt.setp(p,facecolor=c,alpha=0.5)
                for median in bp['medians']:
                    plt.setp(median,color='black')
                #for mean in bp['means']:
                #    plt.setp(mean,color='black')
            if want_scatter:
                x_scatter = np.array(x_scatter).flatten()
                y_scatter = np.array(y_scatter).flatten()
                ax.scatter(x_scatter, y_scatter, marker='o', edgecolors='k')
            if want_violinplot:
                sns.violinplot([real_groups[0][group], real_groups[1][group], real_groups[2][group]], \
                    color=group_colours[group], ax=ax)

            ax.set_xticks(range(3))
            ax.set_xticklabels(['TFC','B+48h','B+1wk'],rotation=45)
            sigstar(ax,G,P)
            ax.set_ylim([-0.2,0.5])

    aov_data_corr = {'engram': {}, 'low_activity': {}}
    for time_key, time_groups in zip(['TFC_cond', 'Test_B', 'Test_B_1wk'], \
            [corr_sess_engram['TFC_cond'], corr_sess_engram['Test_B'], corr_sess_engram['Test_B_1wk']]):
        aov_data_corr['engram'][time_key] = pd.DataFrame({
            'subject': np.arange(1, len(time_groups['hM3D']) + len(time_groups['hM4D']) + len(time_groups['mCherry'])+1),
            'group': np.concatenate((np.repeat('hM3D', len(time_groups['hM3D'])),
                                np.repeat('hM4D', len(time_groups['hM4D'])),
                                np.repeat('mCherry', len(time_groups['mCherry'])))),
            'value': np.concatenate((time_groups['hM3D'],
                                    time_groups['hM4D'],
                                    time_groups['mCherry']))
        })
    for time_key, time_groups in zip(['TFC_cond', 'Test_B', 'Test_B_1wk'], \
            [corr_sess_low_activity['TFC_cond'], corr_sess_low_activity['Test_B'], corr_sess_low_activity['Test_B_1wk']]):
        aov_data_corr['low_activity'][time_key] = pd.DataFrame({
            'subject': np.arange(1, len(time_groups['hM3D']) + len(time_groups['hM4D']) + len(time_groups['mCherry'])+1),
            'group': np.concatenate((np.repeat('hM3D', len(time_groups['hM3D'])),
                                np.repeat('hM4D', len(time_groups['hM4D'])),
                                np.repeat('mCherry', len(time_groups['mCherry'])))),
            'value': np.concatenate((time_groups['hM3D'],
                                    time_groups['hM4D'],
                                    time_groups['mCherry']))
        })

    stats_time_corr = {'TFC_cond': {'engram':None, 'low_activity':None},
                        'Test_B': {'engram':None, 'low_activity':None},
                        'Test_B_1wk': {'engram':None, 'low_activity':None}}
    posthoc_groups_corr = {'TFC_cond': {'engram':None, 'low_activity':None},
                        'Test_B': {'engram':None, 'low_activity':None},
                        'Test_B_1wk': {'engram':None, 'low_activity':None}}
    for time_group in ['TFC_cond', 'Test_B', 'Test_B_1wk']:
        for type in ['engram', 'low_activity']:
            data=aov_data_corr[type][time_group]
            anova_table = pg.anova(data=data, dv='value', between='group')
            print('*** {} {} 1-way ANOVA on group'.format(time_group, type))
            print(anova_table)
            if anova_table['p-unc'].iloc[0] < 0.05:
                print('*** SIGNIFICANT. Multcompare:')
                comp = mc.MultiComparison(data['value'], data['group'])
                post_hoc_res = comp.tukeyhsd()
                print(post_hoc_res)

            # Perform mixed anova
            posthoc = pg.pairwise_tests(data=data, dv='value', between='group', padjust='sidak')

            posthoc_groups_corr[time_group][type] = posthoc

    #
    # Compare groups across time points
    #
    want_bar = False
    want_boxplot = True
    want_scatter = False
    want_violinplot = False
    for act_type in ['engram', 'low_activity']:
        if act_type == 'engram':
            real_groups = [corr_sess_engram['TFC_cond'], corr_sess_engram['Test_B'], corr_sess_engram['Test_B_1wk']]
        if act_type == 'low_activity':
            real_groups = [corr_sess_low_activity['TFC_cond'], corr_sess_low_activity['Test_B'], corr_sess_low_activity['Test_B_1wk']]

        fig, axs = plt.subplots(1,3,figsize=(12,6))
        fig.suptitle('{}'.format(act_type))

        sess_keys = ['TFC_cond', 'Test_B', 'Test_B_1wk']
        for t_i, sess_groups in enumerate(real_groups):
            sess_str = sess_keys[t_i]
            ax = axs[t_i]

            tot_x = 3 # 3 groups x 3 sessions + 2 blank in between
            x = range(tot_x)
            means = np.zeros(tot_x) 
            sems = np.zeros(tot_x)
            errbars = np.zeros((2,tot_x))  

            x_scatter = []
            y_scatter = []
            for g_i, group in enumerate(['hM3D', 'hM4D', 'mCherry']):
                this_data = real_groups[t_i][group]
                x_scatter.append(np.zeros(len(this_data))+i)
                y_scatter.append(this_data)
                means[g_i] = np.mean(this_data)
                sems[g_i] = (np.std(this_data) / np.sqrt(len(this_data)))
                errbars[1,g_i] = sems[g_i]

            df = posthoc_groups_corr[sess_str][act_type]
            P = df['p-corr'].tolist()
            G = [[0,1],[0,2],[1,2]]

            if want_bar:
                ax.bar(x, means, yerr=errbars, color=group_colours.values()) #color=np.array([[x,x] for x in group_colours.values()]).ravel())
            if want_boxplot:
                bp = ax.boxplot([real_groups[t_i]['hM3D'], real_groups[t_i]['hM4D'], real_groups[t_i]['mCherry']], \
                    notch=True, patch_artist=True, positions=range(3), showfliers=False, widths=0.7, showmeans=True, \
                    meanprops=dict(color='black'), meanline=True)
                for p, c in zip(bp['boxes'], group_colours.values()):
                    plt.setp(p,facecolor=c,alpha=0.5)
                for median in bp['medians']:
                    plt.setp(median,color='black')
                #for mean in bp['means']:
                #    plt.setp(mean,color='black')
            if want_scatter:
                x_scatter = np.array(x_scatter).flatten()
                y_scatter = np.array(y_scatter).flatten()
                ax.scatter(x_scatter, y_scatter, marker='o', edgecolors='k')
            if want_violinplot:
                sns.violinplot([real_groups[t_i]['hM3D'], real_groups[t_i]['hM4D'], real_groups[t_i]['mCherry']], \
                    color=group_colours.values(), ax=ax)

            ax.set_xticks(range(3))
            ax.set_xticklabels(['Exc','Inh','Ctl'],rotation=45)
            ax.set_title(sess_str)
            sigstar(ax,G,P)
            #if act_type == 'engram':
            #    ax.set_ylim([-1,1])
            #if act_type == 'low-activity':
            #    ax.set_ylim([-1,1])


'''
For fitting of place fields
'''

#_BEGIN_##############################################################################
import time
from sklearn import mixture
from sklearn.cluster import KMeans
from numpy import linalg

m_ = 'G05'
#cell_ = 512
#cell_ = 406
#cell_ = 241
#cell_ = 129
#cell_ = 372
cell_ = 373
#cell_ = 65
#cell_ = 521

#m_ = 'G21'
#cell_ = 340
#cell_ = 443

n_comp = 4

pcells = pcells_mice[m_][cell_]
fm = TFC_cond[m_].fm.fluorescence_map_occup

st = time.time()
data = []
sample_weights = []
for entry in pcells:
    data.append([entry[0], entry[1]])
    sample_weights.append(entry[2])
data = np.array(data)
et = time.time()
print('took {}'.format(et-st))

max_intensity = np.max(fm[:,:,cell_])
increments = 10
intensity_increments = max_intensity / increments
st = time.time()
sample_weights = []
data = []
for entry in pcells:
    for i in range(int(max_intensity // intensity_increments)):
        data.append([entry[0], entry[1]])
        sample_weights.append(entry[2])
data = np.array(data)
sample_weights = np.array(sample_weights)
et = time.time()
print('took {}'.format(et-st))

'''
st = time.time()
data1 = [[entry[0], entry[1]] for entry in pcell_vals for pcell_vals in pcells.values()]
data1 = np.array(data1)
et = time.time()
print('took {}'.format(et-st))
'''

#data = fm[:,:,340]
x = np.linspace(-0.5,fm.shape[0]-0.5)
y = np.linspace(-0.5,fm.shape[1]-0.5)

m,n = data.shape
R,C = np.mgrid[:m,:n]
out = np.column_stack((C.ravel(),R.ravel(), data.ravel()))
#gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(data)
#n_comp = 2
#w_conc_prior = (1./n_comp)/100
w_conc_prior = 1e-3
gmm = mixture.BayesianGaussianMixture(n_components=n_comp, covariance_type='full', \
    weight_concentration_prior=w_conc_prior).fit(data)
kmeans = KMeans(n_clusters=n_comp, random_state=0).fit(data, sample_weight=sample_weights)

X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)
plt.figure()
plt.imshow(fm[:,:,cell_], cmap='viridis')
plt.contour(Y,X,Z)
plt.scatter(gmm.means_[:,1], gmm.means_[:,0], marker='X', color='k')

# Thresholded gmm
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)
plt.figure()
plt.imshow(fm[:,:,cell_], cmap='viridis')
plt.contour(Y,X,Z)
#plt.scatter(gmm.means_[:,1], gmm.means_[:,0], marker='X', color='k')
#thres = 0.09
thres = np.max(gmm.weights_) / 3
desired_means_y = [gmm.means_[i,1] for i in range(len(gmm.weights_)) if gmm.weights_[i] > thres]
desired_means_x = [gmm.means_[i,0] for i in range(len(gmm.weights_)) if gmm.weights_[i] > thres]
plt.scatter(desired_means_y, desired_means_x, marker='X', color='k')
for i in range(len(gmm.means_)):
    print(i)
    plt.annotate(str(i), (gmm.means_[i,1], gmm.means_[i,0]), color='w')

plt.figure()
colours=['white', 'black', 'red', 'sienna', 'darkorange', 'gold', 'chartreuse', 'darkgreen', 'aqua', 'deepskyblue', \
    'blue', 'blueviolet', 'purple', 'deeppink']
plt.imshow(fm[:,:,cell_], cmap='viridis')
plt.scatter(data[:,1], data[:,0], color=[colours[i] for i in kmeans.labels_])

w, v = linalg.eig(gmm.covariances_[0]); print(np.pi*w[0]*w[1])

# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
def plot_results(X, Y_, means, covariances, index, title):
    plt.figure()
    splot = plt.subplot(1,1,1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        
dpgmm = mixture.BayesianGaussianMixture(n_components=n_comp, covariance_type='full', \
    weight_concentration_prior=w_conc_prior).fit(data)
plot_results(
    data,
    dpgmm.predict(data),
    dpgmm.means_,
    dpgmm.covariances_,
    1,
    "Bayesian Gaussian Mixture with a Dirichlet process prior",
)
#_END_##############################################################################

plt.figure()
plt.plot(sess.loc_X_miniscope_smooth-sess.loc.min_x, -(sess.loc_Y_miniscope_smooth-sess.loc.min_y), lw=0.1)
X_spikes = [sess.loc_X_miniscope_smooth[x]-sess.loc.min_x for x in range(len(sess.loc_X_miniscope_smooth)) if x in sess.S_spikes[cell_]]
Y_spikes = [-(sess.loc_Y_miniscope_smooth[x]-sess.loc.min_y) for x in range(len(sess.loc_Y_miniscope_smooth)) if x in sess.S_spikes[cell_]]
plt.scatter(X_spikes, Y_spikes, c=sess.S_peakval[cell_].astype(int), cmap='jet')
plt.title('Spikes')

plt.figure()
S_positive_idx = [i for i in range(sess.S[cell_].shape[0]) if sess.S_movement[cell_,i] > 0]
S_positive = [s for s in sess.S_movement[cell_] if s > 0]
plt.plot(sess.loc_X_miniscope_smooth-sess.loc.min_x, -(sess.loc_Y_miniscope_smooth-sess.loc.min_y), lw=0.1)
X_values = [sess.loc_X_miniscope_smooth[x]-sess.loc.min_x for x in range(len(sess.loc_X_miniscope_smooth)) if x in S_positive_idx]
Y_values = [-(sess.loc_Y_miniscope_smooth[x]-sess.loc.min_y) for x in range(len(sess.loc_Y_miniscope_smooth)) if x in S_positive_idx]
plt.scatter(X_values, Y_values, c=np.array(S_positive).astype(int), cmap='jet')
plt.title('Fluorescence values during movement')

plt.figure()
S_positive_idx = [i for i in range(sess.S[cell_].shape[0]) if sess.S_immobility[cell_,i] > 0]
S_positive = [s for s in sess.S_immobility[cell_] if s > 0]
plt.plot(sess.loc_X_miniscope_smooth-sess.loc.min_x, -(sess.loc_Y_miniscope_smooth-sess.loc.min_y), lw=0.1)
X_values = [sess.loc_X_miniscope_smooth[x]-sess.loc.min_x for x in range(len(sess.loc_X_miniscope_smooth)) if x in S_positive_idx]
Y_values = [-(sess.loc_Y_miniscope_smooth[x]-sess.loc.min_y) for x in range(len(sess.loc_Y_miniscope_smooth)) if x in S_positive_idx]
plt.scatter(X_values, Y_values, c=np.array(S_positive).astype(int), cmap='jet')
plt.title('Fluorescence values during immobility')




###
## ROUGH WORKING PSTH 
###

    #frames_lookaround = 20 # 20 frames before and after tone/shock onsets

    #snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    snippet_len = frames_lookaround * 2
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    group_PSTH['hM3D'] = np.zeros(snippet_len)
    group_PSTH['hM4D'] = np.zeros(snippet_len)
    group_PSTH['mCherry'] = np.zeros(snippet_len)

    sig_cells = dict()
    sig_cells['hM3D'] = dict()
    sig_cells['hM4D'] = dict()
    sig_cells['mCherry'] = dict()

    frac_tots = dict()
    frac_tots['hM3D'] = []
    frac_tots['hM4D'] = []
    frac_tots['mCherry'] = []

    rng = default_rng()

    print('*** MAPPING: {}'.format(mapping_type))
    for group, mice in mice_per_group.items():

        print('\nin {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            print(' {}'.format(mouse), end='')
            s = session[mouse]
            crossreg = crossreg_mice[mouse]

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                onsets = s.shock_onsets
                offsets = s.shock_offsets

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                C = s.C
            else:
                indeces = []
                df_mapping = crossreg.get_mappings_cells(mapping_type=mapping_type)
                s_df_col = s.get_df_col()
                for i in range(len(df_mapping)):
                    cell_s = int(float(df_mapping[s_df_col].iloc[i]))
                    try:
                        #idx = np.where(s.C_zarr['unit_id']==cell_s)[0][0]
                        idx = np.where(s.C_idx==cell_s)[0][0]
                        indeces.append(idx)
                    except IndexError as error:
                        # sometimes the minian-saved zarr files don't contain the cells from the crossreg mapping;
                        # so just skip this row
                        print('***WARNING: could not find {} ({})'.format(cell_s, s.session_type))
                        continue
                C = s.C[indeces,:]

            sig_cells[group][mouse] = []

            sig_cells_total_mouse = np.ndarray(0)
            for on in onsets:
                period = range(on - frames_lookaround, on + frames_lookaround)
                post = np.mean(C[:, range(on, on + frames_lookaround)],1)
                pre = np.mean(C[:, range(on - frames_lookaround, on)],1)
                C_binary = (post - pre) / (post + pre)
                ##C_binary = post - pre
                #C_binary = np.mean(C[:, range(on, on + frames_lookaround)],1) - \
                #    np.mean(C[:, range(on - frames_lookaround, on)],1)

                # Allocate matrix where we have the response-values for all of the shuffles (columns) for each cell (rows).
                C_shuffles = np.zeros((C.shape[0], len(period), num_shuffles))
                C_shuffles_binary = np.zeros((C.shape[0], num_shuffles))
                #C_period_values = np.zeros((C.shape[0], len(period) * 2, num_shuffles)) # for entire onset,offset period
                for i in range(num_shuffles):
                    shuffle_times = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround), num_shuffles)
                    C_rolled = np.roll(C, shuffle_times, axis=0)
                    ##
                    post_shuffle = np.mean(C_rolled[:, range(shuffle_times[i], shuffle_times[i] + frames_lookaround)])
                    pre_shuffle = np.mean(C_rolled[:, range(shuffle_times[i] - frames_lookaround, shuffle_times[i])])
                    C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                    ##
                    C_shuffles[:,:,i] = C_rolled[:,period]

                    '''
                    half_period = int(len(period)/2)
                    post_shuffle = np.mean(C_shuffles[:, range(half_period, len(period)), i],1)
                    pre_shuffle = np.mean(C_shuffles[:, range(0, half_period), i],1)
                    C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                    '''
                    #np.nan_to_num(C_shuffles_binary,copy=False)
                    ##C_shuffles_binary[:,i] = post_shuffle - pre_shuffle
                C_percentile = np.percentile(C_shuffles, percentile, axis=(1,2)) # Will compute percentiles along 0th axis, i.e., cells
                #C_percentile = np.percentile(C_shuffles_binary, percentile, axis=1)
                sig_cells_mouse = np.where((C_binary > C_percentile)==True)[0]
                sig_cells_total_mouse = np.append(sig_cells_total_mouse, sig_cells_mouse)
                ###frac_tots[group].append(len(sig_cells_mouse) / C.shape[0])
                #frac_tots[group].append(len(sig_cells_mouse))
                sig_cells[group][mouse].append(sig_cells_mouse)

            frac_tots[group].append(len(np.unique(sig_cells_total_mouse)) / C.shape[0])
    group_tots = dict()


###
# Plot trace with onsets, offsets
##
cell_id=9
mouse='G05'

cell_id=293
mouse='G06'
plt.plot(session[mouse].C[cell_id,:])
for i in session[mouse].shock_onsets:
    plt.plot([i,i], [0, 100],'r')
for i in session[mouse].shock_offsets:
    plt.plot([i,i], [0, 100],'r')


on_num=0
f_tot=dict()
for group in ['hM3D', 'hM4D', 'mCherry']:
    f_tot[group] = frac_tots[group][on_num]

plot_PSTH_activities(PLOTS_DIR, frac_tots, 'shock', 'Active_FOO', mapping)



####
# PSTH checkpoint, averaging all responses before comparing shuffles
####


    #snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    snippet_len = frames_lookaround * 2
    save_pre = frames_lookaround + frames_save
    save_post = frames_lookaround + frames_save
    #save_len = frames_lookaround * 2 + frames_save * 2
    save_len = save_pre + save_post
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    #group_PSTH['hM3D'] = np.array([])
    #group_PSTH['hM4D'] = np.array([])
    #group_PSTH['mCherry'] = np.array([])

    sig_cells = dict()
    sig_cells['hM3D'] = dict()
    sig_cells['hM4D'] = dict()
    sig_cells['mCherry'] = dict()

    frac_tots = dict()
    frac_tots['hM3D'] = []
    frac_tots['hM4D'] = []
    frac_tots['mCherry'] = []

    rng = default_rng()

    print('*** MAPPING: {}'.format(mapping_type))
    for group, mice in mice_per_group.items():

        print('\nin {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            print(' {}'.format(mouse), end='')
            s = session[mouse]
            crossreg = crossreg_mice[mouse]

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                onsets = s.shock_onsets
                offsets = s.shock_offsets

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                C = s.C
            else:
                indeces = []
                df_mapping = crossreg.get_mappings_cells(mapping_type=mapping_type)
                s_df_col = s.get_df_col()
                for i in range(len(df_mapping)):
                    cell_s = int(float(df_mapping[s_df_col].iloc[i]))
                    try:
                        #idx = np.where(s.C_zarr['unit_id']==cell_s)[0][0]
                        idx = np.where(s.C_idx==cell_s)[0][0]
                        indeces.append(idx)
                    except IndexError as error:
                        # sometimes the minian-saved zarr files don't contain the cells from the crossreg mapping;
                        # so just skip this row
                        print('***WARNING: could not find {} ({})'.format(cell_s, s.session_type))
                        continue
                C = s.C[indeces,:]

            C_responses = np.zeros((C.shape[0], frames_lookaround*2))
            C_responses_save = np.zeros((C.shape[0], save_len))
            sig_cells_total_mouse = np.ndarray(0)
            for on in onsets:
                period = range(on - frames_lookaround, on + frames_lookaround)
                #period_save = range(on - frames_lookaround, on + save_post)
                period_save = range(on - save_pre, on + save_post)
                C_responses = np.add(C_responses, C[:,period])
                C_responses_save = np.add(C_responses_save, C[:,period_save])
            C_responses /= len(onsets)
            C_responses_save /= len(onsets)

            post = np.mean(C_responses[:, range(frames_lookaround, len(period))],1)
            pre = np.mean(C_responses[:, range(0, frames_lookaround)],1)
            #C_binary = (post - pre) / (post + pre)
            C_binary = post - pre

            #C_binary = np.mean(C[:, range(on, on + frames_lookaround)],1) - \
            #    np.mean(C[:, range(on - frames_lookaround, on)],1)

            # Allocate matrix where we have the response-values for all of the shuffles (columns) for each cell (rows).
            C_shuffles = np.zeros((C.shape[0], len(period), num_shuffles))
            C_shuffles_binary = np.zeros((C.shape[0], num_shuffles))
            #C_period_values = np.zeros((C.shape[0], len(period) * 2, num_shuffles)) # for entire onset,offset period
            for i in range(num_shuffles):
                shuffle_times = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround), num_shuffles)
                C_rolled = np.roll(C, shuffle_times, axis=0)

                '''
                C_rolled_responses = np.zeros((C_rolled.shape[0], frames_lookaround*2))
                for on in onsets:
                    period = range(on - frames_lookaround, on + frames_lookaround)
                    C_rolled_responses = np.add(C_rolled_responses, C_rolled[:,period])
                C_rolled_responses /= len(onsets)
                '''

                C_rolled_responses = C_rolled[:, range(shuffle_times[i] - frames_lookaround, shuffle_times[i] + frames_lookaround)]
                post_shuffle = np.mean(C_rolled_responses[:, range(frames_lookaround, len(period))],1)
                pre_shuffle = np.mean(C_rolled_responses[:, range(0, frames_lookaround)],1)
                #C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                C_shuffles_binary[:,i] = post_shuffle - pre_shuffle

                ##
                '''
                post_shuffle = np.mean(C_rolled[:, range(shuffle_times[i], shuffle_times[i] + frames_lookaround)])
                pre_shuffle = np.mean(C_rolled[:, range(shuffle_times[i] - frames_lookaround, shuffle_times[i])])
                #C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                C_shuffles_binary[:,i] = post_shuffle - pre_shuffle
                #C_shuffles_binary[:,i] = np.mean(C_rolled[:,range(shuffle_times[i]-int(len(period)/2))])
                '''
                ##
                '''
                C_shuffles[:,:,i] = C_rolled[:,period]
                half_period = int(len(period)/2)
                post_shuffle = np.mean(C_shuffles[:, range(half_period, len(period)), i],1)
                pre_shuffle = np.mean(C_shuffles[:, range(0, half_period), i],1)
                C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                '''
                #np.nan_to_num(C_shuffles_binary,copy=False)
                ##C_shuffles_binary[:,i] = post_shuffle - pre_shuffle
            #C_percentile = np.percentile(C_shuffles, percentile, axis=(1,2)) # Will compute percentiles along 0th axis, i.e., cells
            C_percentile = np.percentile(C_shuffles_binary, percentile, axis=1)
            sig_cells_mouse = np.where((C_binary > C_percentile)==True)[0]
            sig_cells_total_mouse = np.append(sig_cells_total_mouse, sig_cells_mouse)
            frac_tots[group].append(len(sig_cells_mouse) / C.shape[0])
            #frac_tots[group].append(len(sig_cells_mouse))
            sig_cells[group][mouse] = sig_cells_mouse

            if group not in group_PSTH.keys():
                group_PSTH[group] = C_responses_save[sig_cells_mouse,:]
            else:
                group_PSTH[group] = np.vstack((group_PSTH[group], C_responses_save[sig_cells_mouse,:]))

        #frac_tots[group].append(len(np.unique(sig_cells_total_mouse)) / C.shape[0])




###
### PLOT GROUP-AVERAGED PSTH TRACES
###


fig, axs = plt.subplots(1, 3, figsize=(9,6), sharey=True, sharex=True)
group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
ylabel_set = False
for group, ax in zip(['hM3D', 'hM4D', 'mCherry'], axs.flat):
    #ax.plot(session_group_PSTH[group], color='b', lw=1)
    group_mean = np.mean(group_PSTH[group],0)
    if shaded == 'sem':
        group_shaded = np.std(group_PSTH[group], 0) / np.sqrt(group_PSTH[group].shape[0])
    if shaded == 'sd':
        group_shaded = np.std(group_PSTH[group], 0)
    ax.plot(group_mean, color='b', lw=1)
    ax.fill_between(range(save_len), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
    #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
    ax.set_title('{} {}'.format(group, group_PSTH[group].shape[0]), size='medium')
    #ax.set_xticks([0,100])
    #ax.set_xticklabels([0, 5])
    ax.set_xlabel('Time (s)')
    if not ylabel_set:
        ax.set_ylabel('$\Delta$F/F (arbitrary units)')
        ylabel_set = True





## Plots and stats for PF measures
stats.kstest(meas_num_pfs['hM3D'], meas_num_pfs['hM4D']) # 2.6728e-09
stats.kstest(meas_num_pfs['mCherry'], meas_num_pfs['hM3D']) # 0.002729
stats.kstest(meas_num_pfs['mCherry'], meas_num_pfs['hM4D']) # 0.09271
plt.figure(figsize=(4,3))
bp = plt.boxplot([meas_num_pfs['hM3D'], meas_num_pfs['hM4D'], meas_num_pfs['mCherry']], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])

## not as clean as including all, don't use this one vvv ; use above full data
hM3D = meas_num_pfs['hM3D']
hM4D = meas_num_pfs['hM4D']
mCherry = meas_num_pfs['mCherry']
hM3D_filter = np.where(meas_num_pfs['hM3D'] > 1)[0]
hM4D_filter = np.where(meas_num_pfs['hM4D'] > 1)[0]
mCherry_filter = np.where(meas_num_pfs['mCherry'] > 1)[0]
stats.kstest(hM3D[hM3D_filter], hM4D[hM4D_filter]) # 0.0155
stats.kstest(mCherry[mCherry_filter], hM3D[hM3D_filter]) # 0.1513
stats.kstest(mCherry[mCherry_filter], hM4D[hM4D_filter]) # 0.0258
plt.figure(figsize=(4,3))
bp = plt.boxplot([hM3D[hM3D_filter], hM4D[hM4D_filter], mCherry[mCherry_filter]], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])

stats.kstest(meas_pf_size['hM3D'], meas_pf_size['hM4D']) # 0.9732
stats.kstest(meas_pf_size['hM3D'], meas_pf_size['mCherry']) # 0.6284
stats.kstest(meas_pf_size['hM4D'], meas_pf_size['mCherry']) # 0.3301
plt.figure(figsize=(4,3))
bp = plt.boxplot([meas_pf_size['hM3D'], meas_pf_size['hM4D'], meas_pf_size['mCherry']], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])

hM3D = meas_pf_size['hM3D']
hM4D = meas_pf_size['hM4D']
mCherry = meas_pf_size['mCherry']
hM3D_filter = np.where(meas_pf_size['hM3D'] > 15)[0]
hM4D_filter = np.where(meas_pf_size['hM4D'] > 15)[0]
mCherry_filter = np.where(meas_pf_size['mCherry'] > 15)[0]
stats.kstest(hM3D[hM3D_filter], hM4D[hM4D_filter]) # 0.4037
stats.kstest(mCherry[mCherry_filter], hM3D[hM3D_filter]) # 0.6527
stats.kstest(mCherry[mCherry_filter], hM4D[hM4D_filter]) # 0.5167
plt.figure(figsize=(4,3))
bp = plt.boxplot([hM3D[hM3D_filter], hM4D[hM4D_filter], mCherry[mCherry_filter]], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])

stats.kstest(np.where(meas_pf_size['hM3D'] > 50)[0], np.where(meas_pf_size['hM4D'] > 50)[0]) # 4.6698e-05
stats.kstest(np.where(meas_pf_size['hM3D'] > 50)[0], np.where(meas_pf_size['mCherry'] > 50)[0]) # 0.09790 WOAH!!
stats.kstest(np.where(meas_pf_size['hM4D'] > 50)[0], np.where(meas_pf_size['mCherry'] > 50)[0]) # 5.4731e-05 
stats.kstest(np.where(meas_pf_size['hM3D'] < 50)[0], np.where(meas_pf_size['hM4D'] < 50)[0]) # 0.6645
stats.kstest(np.where(meas_pf_size['hM3D'] < 50)[0], np.where(meas_pf_size['mCherry'] < 50)[0]) # 2.00584e-27
stats.kstest(np.where(meas_pf_size['hM4D'] < 50)[0], np.where(meas_pf_size['mCherry'] < 50)[0]) # 8.636e-25

stats.kstest(meas_spatial_selectivity['hM3D'], meas_spatial_selectivity['hM4D']) # 6.1296e-30
stats.kstest(meas_spatial_selectivity['mCherry'], meas_spatial_selectivity['hM3D']) # 3.6186e-05
stats.kstest(meas_spatial_selectivity['mCherry'], meas_spatial_selectivity['hM4D']) # 1.1844e-11
hM3D = meas_spatial_selectivity['hM3D']
hM4D = meas_spatial_selectivity['hM4D']
mCherry = meas_spatial_selectivity['mCherry']
plt.figure(figsize=(4,3))
bp = plt.boxplot([hM3D, hM4D, mCherry], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])

stats.kstest(meas_compactness_pf['hM3D'], meas_compactness_pf['hM4D']) # 0.8906
stats.kstest(meas_compactness_pf['hM3D'], meas_compactness_pf['mCherry']) # 0.11916
stats.kstest(meas_compactness_pf['hM4D'], meas_compactness_pf['mCherry']) # 0.40511
hM3D = meas_compactness_pf['hM3D']
hM4D = meas_compactness_pf['hM4D']
mCherry = meas_compactness_pf['mCherry']
plt.figure(figsize=(4,3))
bp = plt.boxplot([hM3D, hM4D, mCherry], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])


hM3D = meas_compactness_pf['hM3D']
hM4D = meas_compactness_pf['hM4D']
mCherry = meas_compactness_pf['mCherry']
hM3D_filter = np.where(meas_compactness_pf['hM3D'] < 1)[0]
hM4D_filter = np.where(meas_compactness_pf['hM4D'] < 1)[0]
mCherry_filter = np.where(meas_compactness_pf['mCherry'] < 1)[0]
stats.kstest(hM3D[hM3D_filter], hM4D[hM4D_filter]) # 0.878
stats.kstest(mCherry[mCherry_filter], hM3D[hM3D_filter]) # 0.0138
stats.kstest(mCherry[mCherry_filter], hM4D[hM4D_filter]) # 0.0367
plt.figure(figsize=(4,3))
bp = plt.boxplot([hM3D[hM3D_filter], hM4D[hM4D_filter], mCherry[mCherry_filter]], \
    notch=True, patch_artist=True, positions=[0.5,1,1.5])
for p, c in zip(bp['boxes'], ['red','blue','black']):
    plt.setp(p,facecolor=c)
plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])


'''
For getting crossreg mappings for calculating PVs for skmeans in R
'''
def process_mice_for_R(mouse, TFC_cond=TFC_cond, Test_B=Test_B, Test_B_1wk=Test_B_1wk, mapping=mapping_TFC_cond_Test_B_Test_B_1wk, \
                       binarize=False, normalize=True, normalize_full=False, spk_cutoff=2):

#from sklearn.preprocessing import PowerTransformer
    sess_TFC = TFC_cond[mouse]
    sess_Test_B = Test_B[mouse]
    sess_Test_B_1wk = Test_B_1wk[mouse]

    df_mapping = sess_Test_B.crossreg.get_mappings_cells(mapping_type=mapping)
    s_df_col_TFC = sess_TFC.get_df_col()
    s_df_col_Test_B = sess_Test_B.get_df_col()
    s_df_col_Test_B_1wk = sess_Test_B_1wk.get_df_col()

    indeces_TFC = []
    indeces_Test_B = []
    indeces_Test_B_1wk = []

    for i in range(len(df_mapping)):
        cell_s_TFC = int(float(df_mapping[s_df_col_TFC].iloc[i]))
        cell_s_Test_B = int(float(df_mapping[s_df_col_Test_B].iloc[i]))
        cell_s_Test_B_1wk = int(float(df_mapping[s_df_col_Test_B_1wk].iloc[i]))

        try:
            #idx = np.where(s.C_zarr['unit_id']==cell_s)[0][0]
            idx_TFC = np.where(sess_TFC.S_idx==cell_s_TFC)[0][0]
            indeces_TFC.append(idx_TFC)
        except IndexError as error:
            print('***WARNING: could not find {} ({})'.format(cell_s_TFC, sess_TFC.session_type))
            # Since it seems to mainly happen during TFC_cond, skip the rest so we have equal number
            # of cross-registered cells in the end.
            continue
        try:
            idx_Test_B = np.where(sess_Test_B.S_idx==cell_s_Test_B)[0][0]
            indeces_Test_B.append(idx_Test_B)
        except IndexError as error:
            print('***WARNING: could not find {} ({})'.format(cell_s_Test_B, sess_Test_B.session_type))
        try:
            idx_Test_B_1wk = np.where(sess_Test_B_1wk.S_idx==cell_s_Test_B_1wk)[0][0]
            indeces_Test_B_1wk.append(idx_Test_B_1wk)
        except IndexError as error:
            print('***WARNING: could not find {} ({})'.format(cell_s_Test_B_1wk, sess_Test_B_1wk.session_type))

    S_TFC = sess_TFC.S[indeces_TFC,:]
    print('*** TFC {} tone_onsets {}'.format(mouse, sess_TFC.tone_onsets))
    print('*** TFC {} tone_offsets {}'.format(mouse, sess_TFC.tone_offsets))
    print('*** TFC {} shock_onsets {}'.format(mouse, sess_TFC.shock_onsets))

    S_Test_B = sess_Test_B.S[indeces_Test_B,:]
    print('*** Test_B {} tone_onsets {}'.format(mouse, sess_Test_B.tone_onsets))
    print('*** Test_B {} tone_offsets {}'.format(mouse, sess_Test_B.tone_offsets))

    S_Test_B_1wk = sess_Test_B_1wk.S[indeces_Test_B_1wk,:]
    print('*** Test_B_1wk {} tone_onsets {}'.format(mouse, sess_Test_B_1wk.tone_onsets))
    print('*** Test_B_1wk {} tone_offsets {}'.format(mouse, sess_Test_B_1wk.tone_offsets))

    bin_frames = bin_width * MINISCOPE_FPS
    upper=1
    lower=0

    PV_TFC = np.zeros((S_TFC.shape[0], math.floor(S_TFC.shape[1]/bin_frames)))
    PV_Test_B = np.zeros((S_Test_B.shape[0], math.floor(S_Test_B.shape[1]/bin_frames)))
    PV_Test_B_1wk = np.zeros((S_Test_B_1wk.shape[0], math.floor(S_Test_B_1wk.shape[1]/bin_frames)))

    curr_frame = 0
    for i in range(PV_TFC.shape[1]):
        if binarize:
            PV_TFC[:,i] = np.where(np.sum(S_TFC[:,curr_frame:curr_frame+bin_frames],1) >= spk_cutoff, upper, lower)
            if np.sum(PV_TFC[:,i])==0:
                PV_TFC[0,i] = -1
        else:
            # Average calcium activity/frame
            PV_TFC[:,i] = np.sum(S_TFC[:,curr_frame:curr_frame+bin_frames],1) / bin_frames
        curr_frame += bin_frames
    curr_frame = 0

    # For C
    sess_TFC.C_sess = sess_TFC.C[:,sess_TFC.miniscope_exp_fnum[sess_TFC.start_idx]:sess_TFC.miniscope_exp_fnum[sess_TFC.stop_idx]]    
    C_TFC = sess_TFC.C_sess[indeces_TFC,:]
    PV_TFC_C = np.zeros((C_TFC.shape[0], math.floor(C_TFC.shape[1]/bin_frames)))
    curr_frame = 0
    for i in range(PV_TFC_C.shape[1]):
        if binarize:
            PV_TFC_C[:,i] = np.where(np.sum(C_TFC[:,curr_frame:curr_frame+bin_frames],1) >= spk_cutoff, upper, lower)
            if np.sum(PV_TFC_C[:,i])==0:
                PV_TFC_C[0,i] = -1
        else:
            # Average calcium activity/frame
            PV_TFC_C[:,i] = np.sum(C_TFC[:,curr_frame:curr_frame+bin_frames],1) / bin_frames
        curr_frame += bin_frames
    curr_frame = 0

    for i in range(PV_Test_B.shape[1]):
        if binarize:
            PV_Test_B[:,i] = np.where(np.sum(S_Test_B[:,curr_frame:curr_frame+bin_frames],1) >= spk_cutoff, upper, lower)
            if np.sum(PV_Test_B[:,i])==0:
                PV_Test_B[0,i] = -1        
        else:
            PV_Test_B[:,i] = np.sum(S_Test_B[:,curr_frame:curr_frame+bin_frames],1) / bin_frames
        curr_frame += bin_frames    
    curr_frame = 0
    for i in range(PV_Test_B_1wk.shape[1]):
        if binarize:
            PV_Test_B_1wk[:,i] = np.where(np.sum(S_Test_B_1wk[:,curr_frame:curr_frame+bin_frames],1) >= spk_cutoff, upper, lower)
            if np.sum(PV_Test_B_1wk[:,i])==0:
                PV_Test_B_1wk[0,i] = -1     
        else:
            PV_Test_B_1wk[:,i] = np.sum(S_Test_B_1wk[:,curr_frame:curr_frame+bin_frames],1) / bin_frames
        curr_frame += bin_frames    

    if normalize and not binarize:
        print('*** normalizing...')
        pt_TFC = PowerTransformer()
        PV_TFC = pt_TFC.fit_transform(PV_TFC)
        pt_Test_B = PowerTransformer()
        PV_Test_B = pt_Test_B.fit_transform(PV_Test_B)
        pt_Test_B_1wk = PowerTransformer()
        PV_Test_B_1wk = pt_Test_B_1wk.fit_transform(PV_Test_B_1wk)

    if normalize_full and not binarize:
        PV_c = np.concatenate((PV_TFC, PV_Test_B, PV_Test_B_1wk), axis=1)
        pt = PowerTransformer()
        PV_t = pt.fit_transform(PV_c)
        PV_TFC = PV_t[:,0:PV_TFC.shape[1]]
        PV_Test_B = PV_t[:,PV_TFC.shape[1]+1:PV_TFC.shape[1]+PV_Test_B.shape[1]+1]
        PV_Test_B_1wk = PV_t[:,PV_TFC.shape[1]+PV_Test_B.shape[1]:]

    df = pd.DataFrame(PV_TFC)
    if binarize:
        df.to_csv("C:\data\PV_{}_TFC_crossreg_binarized.csv".format(mouse))
    else:
        df.to_csv("C:\data\PV_{}_TFC_crossreg.csv".format(mouse))
    df = pd.DataFrame()
    df['tone_onsets'] = sess_TFC.tone_onsets
    df['tone_offsets'] = sess_TFC.tone_offsets
    df['shock_onsets'] = sess_TFC.shock_onsets
    df['shock_offsets'] = sess_TFC.shock_offsets
    df.to_csv("C:\data\PV_{}_stims.csv".format(mouse))

    df = pd.DataFrame(PV_Test_B)
    if binarize:
        df.to_csv("C:\data\PV_{}_Test_B_crossreg_binarized.csv".format(mouse))
    else:
        df.to_csv("C:\data\PV_{}_Test_B_crossreg.csv".format(mouse))
    df = pd.DataFrame()
    df['tone_onsets'] = sess_Test_B.tone_onsets
    df['tone_offsets'] = sess_Test_B.tone_offsets
    df.to_csv("C:\data\PV_{}_Test_B_stims.csv".format(mouse))

    df = pd.DataFrame(PV_Test_B_1wk)
    if binarize:
        df.to_csv("C:\data\PV_{}_Test_B_1wk_crossreg_binarized.csv".format(mouse))
    else:
        df.to_csv("C:\data\PV_{}_Test_B_1wk_crossreg.csv".format(mouse))
    df = pd.DataFrame()
    df['tone_onsets'] = sess_Test_B_1wk.tone_onsets
    df['tone_offsets'] = sess_Test_B_1wk.tone_offsets
    df.to_csv("C:\data\PV_{}_Test_B_1wk_stims.csv".format(mouse))

    print('...done.')

'''
Calculate Mahalonobis distance
'''
# todo

'''
Plot 2D representation of all cells during recording (y-axis is cell id, x-axis are frames)
'''

S = sess.S
C = sess.C
# just use imshow
plt.imshow(S[:,1:1000])
plt.imshow(C[:,1:1000])


sess_TFC.C_sess = sess_TFC.C[:,sess_TFC.miniscope_exp_fnum[sess_TFC.start_idx]:sess_TFC.miniscope_exp_fnum[sess_TFC.stop_idx]]    
C_TFC = sess_TFC.C_sess[indeces_TFC,:]

i=1; plt.figure(); plt.plot(PV_TFC[i,:]); plt.plot(PV_TFC_C[i,:],'r')
S_TFC_z = stats.zscore(PV_TFC)
C_TFC_z = stats.zscore(PV_TFC_C)
i=1; plt.figure(); plt.plot(S_TFC_z[i,:]); plt.plot(C_TFC_z[i,:],'r')
i=1; plt.figure(); plt.plot(PV_TFC[i,:]); plt.plot(PV_TFC_C[i,:],'r')

####
# Rasters & clustering of PVs
####

'''
mice_per_group['hM3D']
['G05', 'G10', 'G11', 'G18', 'G19']
mice_per_group['hM4D']
['G06', 'G07', 'G14', 'G15', 'G20', 'G21']
mice_per_group['mCherry']
['G08', 'G09', 'G12', 'G13', 'G16', 'G17']
'''

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

    binary_C_mice = dict()
    dend_thresh_mice = dict()
    labels_mice = dict()

    #group=mouse_groups[mouse]
    #s=TFC_cond[mouse]
    s=session[mouse]
    C=s.S_full
    #C=s.S
    plt.figure(figsize=(8,6))
    inc = 0
    max_C = np.max(C)
    binary_C = np.copy(C)
    for i in range(C.shape[0]):
        binary_C[i, binary_C[i,:] != 0] = 1
        plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
        inc += 1
    binary_C_mice[mouse] = binary_C

    for i in s.shock_onsets:
        plt.axvline(i, c='r', ls='-', lw=0.5)
    for i in s.tone_onsets:
        plt.axvline(i, c='b', ls='-', lw=0.5) 
    for i in s.shock_offsets:
        plt.axvline(i, c='r', ls='-', lw=0.5)
    for i in s.tone_offsets:
        plt.axvline(i, c='b', ls='-', lw=0.5)
    plt.xlim((0, binary_C.shape[1]))
    plt.ylim((0, binary_C.shape[0]))
    xtick_seconds = np.concatenate(([0], s.tone_onsets_def, [1300]))
    plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
    plt.xlabel('Time (s)')
    plt.ylabel('Cell #')
    plt.title('Calcium transients {}'.format(mouse))
    filename = 'raster_{}_{}.png'.format(group, mouse)
    dir_name = 'PV'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    #binary_C = np.transpose(binary_C)
    # hierarchical clustering with Ward
    '''
    binary_C_concat = np.concatenate((binary_C[:,s.tone_onsets[0]:s.tone_offsets[0]], binary_C[:,s.tone_onsets[1]:s.tone_offsets[1]], \
        binary_C[:,s.tone_onsets[2]:s.tone_offsets[2]], binary_C[:,s.tone_onsets[3]:s.tone_offsets[3]], \
        binary_C[:,s.tone_onsets[4]:s.tone_offsets[4]]),axis=1)
    '''

    if transpose_wanted:
        binary_C_fit = np.copy(binary_C).transpose()
    else:
        binary_C_fit = binary_C

        ## hM3D
        dend_thresh_mice['G05'] = 94.0
        dend_thresh_mice['G10'] = 120.0
        dend_thresh_mice['G11'] = 120.0
        dend_thresh_mice['G18'] = 105.0
        dend_thresh_mice['G19'] = 83.0

        ## hM4D
        dend_thresh_mice['G06'] = 80.0
        dend_thresh_mice['G07'] = 76.0 # 65.0
        dend_thresh_mice['G14'] = 66.0 # 55.0 # G14
        dend_thresh_mice['G15'] = 70.0 # =50.1 # G15
        dend_thresh_mice['G20'] = 87.0 
        dend_thresh_mice['G21'] = 62.0

        ## mCherry
        dend_thresh_mice['G08'] = 85.0 # 60.0
        dend_thresh_mice['G09'] = 89.0 # 60.0
        dend_thresh_mice['G12'] = 70.0 # 60.0
        dend_thresh_mice['G13'] = 54.0 # 60.0
        dend_thresh_mice['G16'] = 93.0 # 60.0
        dend_thresh_mice['G17'] = 121.0 # 60.0


    ##binary_C_fit = binary_C_concat
    Z = linkage(binary_C_fit, 'ward')
    plt.figure()
    dendrogram(Z)

    dend_thresh = dend_thresh_mice[mouse]

    plt.axhline(dend_thresh, c='k', ls='--', lw=1)
    filename = 'dendrogram_{}_{}.png'.format(group, mouse)
    dir_name = 'PV'
    plt.title('Dendrogram {}'.format(mouse))
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    clustering = AgglomerativeClustering(distance_threshold=dend_thresh, n_clusters=None, linkage='ward')
    #clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
    ##clustering.fit(binary_C)
    labels = clustering.fit_predict(binary_C_fit)
    num_clusters = len(np.unique(labels))
    print("Num of clusters: {}".format(num_clusters))
    labels_mice[mouse] = labels

    if not transpose_wanted:
        ind = np.argsort(labels)
        plt.figure(figsize=(8,6))
        inc = 0
        current_label = labels[0]
        for i in ind:
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
            inc += 1.0
            if labels[i] != current_label:
                current_label = labels[i]
                plt.axhline(inc,c='b',lw=0.5)
        for i in s.shock_onsets:
            plt.axvline(i, c='r', ls='-', lw=0.5)
        for i in s.tone_onsets:
            plt.axvline(i, c='b', ls='-', lw=0.5) 
        for i in s.shock_offsets:
            plt.axvline(i, c='r', ls='-', lw=0.5)
        for i in s.tone_offsets:
            plt.axvline(i, c='b', ls='-', lw=0.5)
        plt.xlim((0, binary_C.shape[1]))
        plt.ylim((0, binary_C.shape[0]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def, [1300]))
        plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        plt.xlabel('Time (s)')
        plt.ylabel('Cell #')
        plt.title('Calcium transients {} - sorted by cluster (tot {} clusters)'.format(mouse, num_clusters))

        filename = 'raster_{}_{}_num_clusters_{}.png'.format(group, mouse, num_clusters)
        dir_name = 'PV'
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

    else:
        ## Transpose (PV across time)
        '''
        plt.figure(figsize=(8,6))
        inc = 0
        for i in ind:
            plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
            inc += 1.0
        '''

        #
        # Plot sorted by PV cluster times
        #
        ind = np.argsort(labels)
        plt.figure(figsize=(8,6))
        inc = 0
        for i in range(C.shape[0]):
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(C.shape[1]), binary_C[i,ind]+inc,'k',lw=0.1)
            inc += 1.0
        current_label = labels[0]
        for i in range(len(ind)):
            if labels[ind[i]] != current_label:
                current_label = labels[ind[i]]
                plt.axvline(i,c='b',lw=0.5)
        for i in s.shock_onsets:
            plt.axvline(i, c='r', ls='-', lw=1)
        for i in s.tone_onsets:
            plt.axvline(i, c='b', ls='-', lw=1) 
        for i in s.shock_offsets:
            plt.axvline(i, c='r', ls='-', lw=1)
        for i in s.tone_offsets:
            plt.axvline(i, c='b', ls='-', lw=1)
        plt.xlim((0, binary_C.shape[1]))
        plt.ylim((0, binary_C.shape[0]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def, [1300]))
        plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        plt.title('Calcium transients sorted by PV {} (tot {} clusters)'.format(mouse, num_clusters))
        filename = 'raster_transpose_sorted_PV_{}_{}_num_clusters_{}.png'.format(group, mouse, num_clusters)
        dir_name = 'PV'
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

        #
        # Normal plot
        #
        num_clusters = len(np.unique(labels))
        cmap = plt.get_cmap('viridis')
        labels_colours = cmap(np.linspace(0, 1, num_clusters))
        plt.figure(figsize=(8,6))
        inc = 0
        for i in range(C.shape[0]):
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
            inc += 1.0
        for i in range(len(labels)):
            plt.fill_between([i, i+1], [0, C.shape[0]], alpha=0.1, color=labels_colours[labels[i]])

        for i in s.shock_onsets:
            plt.axvline(i, c='r', ls='-', lw=1)
        for i in s.tone_onsets:
            plt.axvline(i, c='b', ls='-', lw=1) 
        for i in s.shock_offsets:
            plt.axvline(i, c='r', ls='-', lw=1)
        for i in s.tone_offsets:
            plt.axvline(i, c='b', ls='-', lw=1)
        plt.xlim((0, binary_C.shape[1]))
        plt.ylim((0, binary_C.shape[0]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def, [1300]))
        plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        plt.xlabel('Time (s)')
        plt.ylabel('Cell #')
        plt.title('Calcium transients with highlighted PVs {} (tot {} clusters)'.format(mouse, num_clusters))
        filename = 'raster_transpose_unsorted_PV_{}_{}_num_clusters_{}.png'.format(group, mouse, num_clusters)
        dir_name = 'PV'
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()
    
    return [binary_C_mice, dend_thresh_mice, labels_mice]


        num_clusters = len(np.unique(labels))
        cmap = plt.get_cmap('viridis')
        labels_colours = cmap(np.linspace(0, 1, num_clusters))
        plt.figure(figsize=(8,6))
        inc = 0
        for i in range(binary_C_fit.shape[0]):
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(binary_C_fit.shape[1]), binary_C_fit[i,:]+inc,'k',lw=0.1)
            inc += 1.0
        for i in range(len(labels)):
            plt.fill_between([i, i+1], 0, binary_C_fit.shape[0], alpha=0.2, color=labels_colours[labels[i]])

        for i in s.shock_onsets:
            plt.axvline(i/40, c='r', ls='-', lw=1)
        for i in s.tone_onsets:
            plt.axvline(i/40, c='b', ls='-', lw=1) 
        for i in s.shock_offsets:
            plt.axvline(i/40, c='r', ls='-', lw=1)
        for i in s.tone_offsets:
            plt.axvline(i/40, c='b', ls='-', lw=1)

        inc=0
        current_label = labels[0]
        for i in range(PV.shape[0]):
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(PV.shape[1]), PV[i,ind]+inc,'k',lw=0.1)
            inc += 1.0
        current_label = labels[0]
        for i in range(len(ind)):
            if labels[ind[i]] != current_label:
                current_label = labels[ind[i]]
                plt.axvline(i,c='b',lw=0.5)


        saver = Saver(parent_path=savepath, subdirs=['crossreg'], prefix='{}_crossreg_{}'.format(mouse, crossreg_type))
        self.saver = saver

        if saver.check_exists('mappings_df'):
            self.mappings_df = saver.load('mappings_df')
            df = self.mappings_df
        else:
            with open(dpath_mappings) as crossreg_file:
                df = pd.read_csv(crossreg_file)
            self.mappings_df = df
            saver.save(df, 'mappings_df')
#
# Plot histogram of "velocity-weighted calcium transients" (actually S matrix)
#

###/begin
weighted_act_group = {}
for group, mice in mice_per_group.items():
    plt.figure()
    plt.title('{} velocity-transients'.format(group))    
    weighted_act_group[group] = []
    for m in mice:
        print('{}..'.format(m),end='')
        if m == 'G09':
            continue
        s = TFC_cond[m]
        S = s.S
        vel = s.velocities_miniscope_smooth
        for i in range(S.shape[0]):
            weighted_S = np.true_divide(S[i,:], vel)
            weighted_S[weighted_S == np.inf] = 0
            weighted_S = np.nan_to_num(weighted_S)
            [frameidx, peakval] = find_spikes_ca(weighted_S, 1, plotit=False, want_peakval=True)
            weighted_act_group[group].append(peakval)
            plt.scatter(vel[frameidx], peakval, s=0.5, c='k')

weighted_hM3D = [item for sublist in weighted_act_group['hM3D'] for item in sublist]
weighted_hM4D = [item for sublist in weighted_act_group['hM4D'] for item in sublist]
weighted_mCherry = [item for sublist in weighted_act_group['mCherry'] for item in sublist]

plt.figure()
plt.hist(weighted_hM3D,bins=1000,color='red',alpha=0.5)
plt.hist(weighted_hM4D,bins=1000,color='blue',alpha=0.5)
plt.hist(weighted_mCherry,bins=1000,color='black',alpha=0.5)
###/end

weighted_act = s.velocities_miniscope_smooth * C[10,:]
plt.figure()
plt.plot(C[10,:],'r')
plt.plot(s.velocities_miniscope_smooth,'b')
plt.plot(weighted_act,'g')

weighted_C = np.true_divide(C[10,:], s.velocities_miniscope_smooth)
weighted_C[weighted_C == np.inf] = 0
weighted_C = np.nan_to_num(weighted_C)
[frameidx, peakval, fig] = find_spikes_ca(weighted_C, 2, plotit=True, want_peakval=True)

####
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for digit in digits.target_names:
        plt.scatter(
            *X_red[y == digit].T,
            marker=f"${digit}$",
            s=50,
            c=plt.cm.nipy_spectral(labels[y == digit] / 10),
            alpha=0.5,
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(linkage='ward', n_clusters=10)
clustering.fit(binary_C)
plot_clustering(binary_C, clustering.labels_, "Ward linkage")
####

# for manual PSTH 
crossreg_mice=TFC_cond_crossreg
session=TFC_cond
mapping_type=mapping
stim='shock'
frames_lookaround=40
frames_save=200
shaded='sem'
num_shuffles=100
auto_close=False
percentile=95.0

# PV
session=TFC_cond
session_str='TFC cond'

session=Test_B
session_str='Test B'
mouse='G05'
group='hM3D'
transpose_wanted=False
auto_close=False
bin_width=2
spk_cutoff=2

dend_thresh_mice = dict()

    if transpose_wanted:

        if session_type == 'TFC_cond':
            ## hM3D
            dend_thresh_mice['G05'] = 11.5
            dend_thresh_mice['G10'] = 12.4
            dend_thresh_mice['G11'] = 12.0
            dend_thresh_mice['G18'] = 13.4
            dend_thresh_mice['G19'] = 8.15

            ## hM4D
            dend_thresh_mice['G06'] = 8.4
            dend_thresh_mice['G07'] = 9.01
            dend_thresh_mice['G14'] = 7.7
            dend_thresh_mice['G15'] = 7.63
            dend_thresh_mice['G20'] = 10.75
            dend_thresh_mice['G21'] = 8.76

            ## mCherry
            dend_thresh_mice['G08'] = 8.10
            dend_thresh_mice['G09'] = 9.85
            dend_thresh_mice['G12'] = 8.31
            dend_thresh_mice['G13'] = 4.77
            dend_thresh_mice['G16'] = 11.5
            dend_thresh_mice['G17'] = 12.5

        elif session_type == 'Test_B':

            # Test B
            pass
    else:

        if session_type == 'TFC_cond':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 92.7 #
                dend_thresh_mice['G10'] = 120.0 #
                dend_thresh_mice['G11'] = 120.0 #
                dend_thresh_mice['G18'] = 105.0 #
                dend_thresh_mice['G19'] = 100.0 #

                ## hM4D
                dend_thresh_mice['G06'] = 80.0 #
                dend_thresh_mice['G07'] = 115.0 #
                dend_thresh_mice['G14'] = 83.0 #
                dend_thresh_mice['G15'] = 82.0 #
                dend_thresh_mice['G20'] = 120.0 #
                dend_thresh_mice['G21'] = 62.0 #

                ## mCherry
                dend_thresh_mice['G08'] = 101.0 #
                dend_thresh_mice['G09'] = 90.0 #
                dend_thresh_mice['G12'] = 82.0 #
                dend_thresh_mice['G13'] = 54.0 #
                dend_thresh_mice['G16'] = 106.0 #
                dend_thresh_mice['G17'] = 114.0 #

            elif Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 91.0 #
                dend_thresh_mice['G10'] = 110.0 #
                dend_thresh_mice['G11'] = 116.0 #
                dend_thresh_mice['G18'] = 104.0 #
                dend_thresh_mice['G19'] = 71.0 # 

                ## hM4D
                dend_thresh_mice['G06'] = 68.0 #
                dend_thresh_mice['G07'] = 94.0 #
                dend_thresh_mice['G14'] = 60.0 # 
                dend_thresh_mice['G15'] = 50.0 #
                dend_thresh_mice['G20'] = 87.0 #
                dend_thresh_mice['G21'] = 48.0 #

                ## mCherry
                dend_thresh_mice['G08'] = 74.0 #
                dend_thresh_mice['G09'] = 80.0 #
                dend_thresh_mice['G12'] = 76.0 #
                dend_thresh_mice['G13'] = 50.0 #
                dend_thresh_mice['G16'] = 73.0 #
                dend_thresh_mice['G17'] = 108.0 #

            elif Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 30.4 #
                dend_thresh_mice['G10'] = 40.0 #
                dend_thresh_mice['G11'] = 36.9 #
                dend_thresh_mice['G18'] = 49.0 #
                dend_thresh_mice['G19'] = 52.0 #

                ## hM4D
                dend_thresh_mice['G06'] = 24.0 #
                dend_thresh_mice['G07'] = 40.0 #
                dend_thresh_mice['G14'] = 28.0 #
                dend_thresh_mice['G15'] = 30.0 #
                dend_thresh_mice['G20'] = 43.4 #
                dend_thresh_mice['G21'] = 29.28 #

                ## mCherry
                dend_thresh_mice['G08'] = 45.2 #
                dend_thresh_mice['G09'] = 36.1 #
                dend_thresh_mice['G12'] = 35.9 #
                dend_thresh_mice['G13'] = 23.96 #
                dend_thresh_mice['G16'] = 36.75 #
                dend_thresh_mice['G17'] = 86.7 #

        elif session_type == 'Test_B':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 70.0 ##
                dend_thresh_mice['G10'] = 81.2 ##
                dend_thresh_mice['G11'] = 111.6 ##
                dend_thresh_mice['G18'] = 104.8 ##
                dend_thresh_mice['G19'] = 76.6 ##

                ## hM4D
                dend_thresh_mice['G06'] = 74.4 #
                dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
                dend_thresh_mice['G14'] = 77.4 #
                dend_thresh_mice['G15'] = 71.4 # Have to use 'TFC_cond+Test_B' mapping since Test_B_1wk not recorded. Or don't include!
                dend_thresh_mice['G20'] = 82.8 #
                dend_thresh_mice['G21'] = 73.2 #

                ## mCherry
                dend_thresh_mice['G08'] = 155.6 #
                dend_thresh_mice['G09'] = 77.1 #
                dend_thresh_mice['G12'] = 106.8 # 
                dend_thresh_mice['G13'] = 69.6 #
                dend_thresh_mice['G16'] = 83.0 #
                dend_thresh_mice['G17'] = 82.3 #

            if Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 60.6 #
                dend_thresh_mice['G10'] = 69.8 #
                dend_thresh_mice['G11'] = 95.1 #
                dend_thresh_mice['G18'] = 94.1 #
                dend_thresh_mice['G19'] = 62.3 #

                ## hM4D
                dend_thresh_mice['G06'] = 60.6 #
                dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
                dend_thresh_mice['G14'] = 75.1 #
                dend_thresh_mice['G15'] = 62.7 #
                dend_thresh_mice['G20'] = 72.8 #
                dend_thresh_mice['G21'] = 25.37 #

                ## mCherry
                dend_thresh_mice['G08'] = 66.0 #
                dend_thresh_mice['G09'] = 70.2 #
                dend_thresh_mice['G12'] = 71.6 #
                dend_thresh_mice['G13'] = 55.3 #
                dend_thresh_mice['G16'] = 68.7 #
                dend_thresh_mice['G17'] = 82.8 #
        
            if Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 48.0 #
                dend_thresh_mice['G10'] = 45.0 #
                dend_thresh_mice['G11'] = 76.1 #
                dend_thresh_mice['G18'] = 56.1 #
                dend_thresh_mice['G19'] = 52.7 #

                ## hM4D
                dend_thresh_mice['G06'] = 35.8 #
                dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
                dend_thresh_mice['G14'] = 37.6 #
                dend_thresh_mice['G15'] = 44.6 #
                dend_thresh_mice['G20'] = 38.3 #
                dend_thresh_mice['G21'] = 66.5 #

                ## mCherry
                dend_thresh_mice['G08'] = 99.3 #
                dend_thresh_mice['G09'] = 44.6 #
                dend_thresh_mice['G12'] = 80.8 #
                dend_thresh_mice['G13'] = 44.1 #
                dend_thresh_mice['G16'] = 52.7 #
                dend_thresh_mice['G17'] = 57.6 #

        elif session_type == 'Test_B_1wk':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 87.8 #
                dend_thresh_mice['G10'] = 102.5 #
                dend_thresh_mice['G11'] = 121.4 #
                dend_thresh_mice['G18'] = 124.6 #
                dend_thresh_mice['G19'] = 85.6 #

                ## hM4D
                dend_thresh_mice['G06'] = 74.5 #
                dend_thresh_mice['G07'] = 98.7 ## Skip since no Test_B # NOT TRUE! CAN DO!
                dend_thresh_mice['G14'] = 68.8 #
                dend_thresh_mice['G15'] = None ## Also skip since no Test_B_1wk
                dend_thresh_mice['G20'] = 87.3 #
                dend_thresh_mice['G21'] = 91.4 #

                ## mCherry
                dend_thresh_mice['G08'] = 84.5 #
                dend_thresh_mice['G09'] = 78.8 #
                dend_thresh_mice['G12'] = 81.5 #
                dend_thresh_mice['G13'] = 40.1 #
                dend_thresh_mice['G16'] = 99.0 #
                dend_thresh_mice['G17'] = 150.4 #

            if Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 81.3 #
                dend_thresh_mice['G10'] = 106.6 #
                dend_thresh_mice['G11'] = 120.1 #
                dend_thresh_mice['G18'] = 120.0 #
                dend_thresh_mice['G19'] = 63.3 #

                ## hM4D
                dend_thresh_mice['G06'] = 56.8 #
                dend_thresh_mice['G07'] = 85.0 ## Skip since no Test_B # NOT TRUE! CAN DO!
                dend_thresh_mice['G14'] = 61.1 #
                dend_thresh_mice['G15'] = None # Also skip since no Test_B_1wk
                dend_thresh_mice['G20'] = 79.0 #
                dend_thresh_mice['G21'] = 67.6 #

                ## mCherry
                dend_thresh_mice['G08'] = 74.7 #
                dend_thresh_mice['G09'] = 70.9 #
                dend_thresh_mice['G12'] = 72.2 #
                dend_thresh_mice['G13'] = 32.1 #
                dend_thresh_mice['G16'] = 82.1 #
                dend_thresh_mice['G17'] = 114.4 #
        
            if Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 62.5 #
                dend_thresh_mice['G10'] = 49.5 #
                dend_thresh_mice['G11'] = 42.3 #
                dend_thresh_mice['G18'] = 66.8 #
                dend_thresh_mice['G19'] = 48.5 #

                ## hM4D
                dend_thresh_mice['G06'] = 27.3 #
                dend_thresh_mice['G07'] = 58.8 # Skip since no Test_B # NOT TRUE! CAN DO!
                dend_thresh_mice['G14'] = 31.1 #
                dend_thresh_mice['G15'] = None # Also skip since no Test_B_1wk
                dend_thresh_mice['G20'] = 33.4 #
                dend_thresh_mice['G21'] = 71.4 #

                ## mCherry
                dend_thresh_mice['G08'] = 41.6 #
                dend_thresh_mice['G09'] = 47.2 #
                dend_thresh_mice['G12'] = 53.1 #
                dend_thresh_mice['G13'] = 15.63 #
                dend_thresh_mice['G16'] = 32.3 #
                dend_thresh_mice['G17'] = 107.2 #

#########################
        OLD
#########################
        
    if transpose_wanted:

        if session_type == 'TFC_cond':
            ## hM3D
            dend_thresh_mice['G05'] = 11.5
            dend_thresh_mice['G10'] = 12.4
            dend_thresh_mice['G11'] = 12.0
            dend_thresh_mice['G18'] = 13.4
            dend_thresh_mice['G19'] = 8.15

            ## hM4D
            dend_thresh_mice['G06'] = 8.4
            dend_thresh_mice['G07'] = 9.01
            dend_thresh_mice['G14'] = 7.7
            dend_thresh_mice['G15'] = 7.63
            dend_thresh_mice['G20'] = 10.75
            dend_thresh_mice['G21'] = 8.76

            ## mCherry
            dend_thresh_mice['G08'] = 8.10
            dend_thresh_mice['G09'] = 9.85
            dend_thresh_mice['G12'] = 8.31
            dend_thresh_mice['G13'] = 4.77
            dend_thresh_mice['G16'] = 11.5
            dend_thresh_mice['G17'] = 12.5

        elif session_type == 'Test_B':

            # Test B
            pass
    else:

        if session_type == 'TFC_cond':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 94.0
                dend_thresh_mice['G10'] = 120.0
                dend_thresh_mice['G11'] = 120.0
                dend_thresh_mice['G18'] = 105.0
                dend_thresh_mice['G19'] = 83.0

                ## hM4D
                dend_thresh_mice['G06'] = 80.0
                dend_thresh_mice['G07'] = 76.0 # 65.0
                dend_thresh_mice['G14'] = 66.0 # 55.0 # G14
                dend_thresh_mice['G15'] = 70.0 # =50.1 # G15
                dend_thresh_mice['G20'] = 87.0 
                dend_thresh_mice['G21'] = 62.0

                ## mCherry
                dend_thresh_mice['G08'] = 85.0 # 60.0
                dend_thresh_mice['G09'] = 89.0 # 60.0
                dend_thresh_mice['G12'] = 70.0 # 60.0
                dend_thresh_mice['G13'] = 54.0 # 60.0
                dend_thresh_mice['G16'] = 93.0 # 60.0
                dend_thresh_mice['G17'] = 121.0 # 60.0

            elif Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 94.0 #
                dend_thresh_mice['G10'] = 108.7 #   
                dend_thresh_mice['G11'] = 115.4 #
                dend_thresh_mice['G18'] = 97.9 #
                dend_thresh_mice['G19'] = 74.9 #

                ## hM4D
                dend_thresh_mice['G06'] = 66.85 #
                dend_thresh_mice['G07'] = 76.0 #
                dend_thresh_mice['G14'] = 78.48 #
                dend_thresh_mice['G15'] = 64.4 #
                dend_thresh_mice['G20'] = 106.5 #
                dend_thresh_mice['G21'] = 56.73 #

                ## mCherry
                dend_thresh_mice['G08'] = 89.4 #
                dend_thresh_mice['G09'] = 85.6 #
                dend_thresh_mice['G12'] = 72.1 #
                dend_thresh_mice['G13'] = 49.81 #
                dend_thresh_mice['G16'] = 80.2 #
                dend_thresh_mice['G17'] = 99.8 #

            elif Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 30.6 #
                dend_thresh_mice['G10'] = 39.52 #
                dend_thresh_mice['G11'] = 36.75 #
                dend_thresh_mice['G18'] = 97.9 
                dend_thresh_mice['G19'] = 74.9 

                ## hM4D
                dend_thresh_mice['G06'] = 66.85 
                dend_thresh_mice['G07'] = 76.0 
                dend_thresh_mice['G14'] = 78.48 
                dend_thresh_mice['G15'] = 64.4 
                dend_thresh_mice['G20'] = 106.5 
                dend_thresh_mice['G21'] = 56.73 

                ## mCherry
                dend_thresh_mice['G08'] = 89.4 
                dend_thresh_mice['G09'] = 85.6 
                dend_thresh_mice['G12'] = 72.1 
                dend_thresh_mice['G13'] = 49.81 
                dend_thresh_mice['G16'] = 80.2 
                dend_thresh_mice['G17'] = 99.8 

        elif session_type == 'Test_B':

            ## hM3D
            dend_thresh_mice['G05'] = 80.3
            dend_thresh_mice['G10'] = 79.2
            dend_thresh_mice['G11'] = 117.3
            dend_thresh_mice['G18'] = 103.3
            dend_thresh_mice['G19'] = 76.0

            ## hM4D
            dend_thresh_mice['G06'] = 120.2
            dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
            dend_thresh_mice['G14'] = 82.5
            dend_thresh_mice['G15'] = 76.5 # Have to use 'TFC_cond+Test_B' mapping since Test_B_1wk not recorded. Or don't include!
            dend_thresh_mice['G20'] = 106.5
            dend_thresh_mice['G21'] = 73.2

            ## mCherry
            dend_thresh_mice['G08'] = 110.3
            dend_thresh_mice['G09'] = 77.4
            dend_thresh_mice['G12'] = 127.4
            dend_thresh_mice['G13'] = 69.6
            dend_thresh_mice['G16'] = 82.7
            dend_thresh_mice['G17'] = 78.0

        elif session_type == 'Test_B_1wk':

            ## hM3D
            dend_thresh_mice['G05'] = 108.7
            dend_thresh_mice['G10'] = 95.4
            dend_thresh_mice['G11'] = 97.6
            dend_thresh_mice['G18'] = 127.9
            dend_thresh_mice['G19'] = 83.9

            ## hM4D
            dend_thresh_mice['G06'] = 72.5
            dend_thresh_mice['G07'] = 119.4 # Skip since no Test_B
            dend_thresh_mice['G14'] = 66.6
            dend_thresh_mice['G15'] = None # Also skip since no Test_B_1wk
            dend_thresh_mice['G20'] = 82.8
            dend_thresh_mice['G21'] = 94.0

            ## mCherry
            dend_thresh_mice['G08'] = 84.5
            dend_thresh_mice['G09'] = 78.1
            dend_thresh_mice['G12'] = 88.6
            dend_thresh_mice['G13'] = 33.6
            dend_thresh_mice['G16'] = 79.5
            dend_thresh_mice['G17'] = 106.2
#########################


        for i in range(len(S_idx)):
            plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
            inc += 1


class PopulationVector:
    def __init__(self, mouse, group, binary_C, labels, frac_labels, labels_tot):
        self.mouse = mouse
        self.group = group
        self.binary_C = binary_C
        self.labels = labels
        self.frac_labels = frac_labels # Fraction of label neurons that is re-activated (i.e. in crossreg)
        self.labels_tot = labels_tot

        # Because not returning it from cluster_pop_vectors_helper(), so just lazily calculate it afterwards..
        self.reactivated_crossreg = {} # Number of label neurons that are re-activated (from crossreg)
        for label in range(len(labels_tot)):
            self.reactivated_crossreg[label] = int(frac_labels[label] * labels_tot[label])


for sess in ['TFC_cond', 'Test_B', 'Test_B_1wk']:
    for group in ['hM3D', 'hM4D', 'mCherry']:
        for v in PV_sess['full'][sess][group]:
            v.mouse
            v.is_classified=False
            classify_pop_vector(v)
            plot_raster_clusters_sorted(v, use_classify_colours=True, Ca_act_type=Ca_act_type, plots_dir=plots_dir, auto_close=True)

v=PV_sess['full']['TFC_cond']['hM3D'][2]
v.mouse
v.is_classified=False
classify_pop_vector(v)
plot_raster_clusters_sorted(v, use_classify_colours=True, Ca_act_type=Ca_act_type, plots_dir=plots_dir, auto_close=True)

##
## HISTORICAL
##
## Manual fits of dend_thresh
##
    dend_thresh_mice = dict()

    if transpose_wanted:

        if session_type == 'TFC_cond':
            ## hM3D
            dend_thresh_mice['G05'] = 11.5
            dend_thresh_mice['G10'] = 12.4
            dend_thresh_mice['G11'] = 12.0
            dend_thresh_mice['G18'] = 13.4
            dend_thresh_mice['G19'] = 8.15

            ## hM4D
            dend_thresh_mice['G06'] = 8.4
            dend_thresh_mice['G07'] = 9.01
            dend_thresh_mice['G14'] = 7.7
            dend_thresh_mice['G15'] = 7.63
            dend_thresh_mice['G20'] = 10.75
            dend_thresh_mice['G21'] = 8.76

            ## mCherry
            dend_thresh_mice['G08'] = 8.10
            dend_thresh_mice['G09'] = 9.85
            dend_thresh_mice['G12'] = 8.31
            dend_thresh_mice['G13'] = 4.77
            dend_thresh_mice['G16'] = 11.5
            dend_thresh_mice['G17'] = 12.5

        elif session_type == 'Test_B':

            # Test B
            pass
    else:

        if session_type == 'TFC_cond':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 92.7 #
                dend_thresh_mice['G10'] = 120.0 #
                dend_thresh_mice['G11'] = 120.0 #
                dend_thresh_mice['G18'] = 105.0 #
                dend_thresh_mice['G19'] = 100.0 #

                ## hM4D
                dend_thresh_mice['G06'] = 80.0 #
                dend_thresh_mice['G07'] = 115.0 #
                dend_thresh_mice['G14'] = 83.0 #
                dend_thresh_mice['G15'] = 82.0 #
                dend_thresh_mice['G20'] = 120.0 #
                dend_thresh_mice['G21'] = 62.0 #

                ## mCherry
                dend_thresh_mice['G08'] = 101.0 #
                dend_thresh_mice['G09'] = 90.0 #
                dend_thresh_mice['G12'] = 82.0 #
                dend_thresh_mice['G13'] = 54.0 #
                dend_thresh_mice['G16'] = 106.0 #
                dend_thresh_mice['G17'] = 114.0 #

            elif Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 91.0 #
                dend_thresh_mice['G10'] = 110.0 #
                dend_thresh_mice['G11'] = 116.0 #
                dend_thresh_mice['G18'] = 104.0 #
                dend_thresh_mice['G19'] = 71.0 # 

                ## hM4D
                dend_thresh_mice['G06'] = 68.0 #
                dend_thresh_mice['G07'] = 94.0 #
                dend_thresh_mice['G14'] = 60.0 # 
                dend_thresh_mice['G15'] = 50.0 #
                dend_thresh_mice['G20'] = 87.0 #
                dend_thresh_mice['G21'] = 48.0 #

                ## mCherry
                dend_thresh_mice['G08'] = 74.0 #
                dend_thresh_mice['G09'] = 80.0 #
                dend_thresh_mice['G12'] = 76.0 #
                dend_thresh_mice['G13'] = 50.0 #
                dend_thresh_mice['G16'] = 73.0 #
                dend_thresh_mice['G17'] = 108.0 #

            elif Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 30.4 #
                dend_thresh_mice['G10'] = 40.0 #
                dend_thresh_mice['G11'] = 36.9 #
                dend_thresh_mice['G18'] = 49.0 #
                dend_thresh_mice['G19'] = 52.0 #

                ## hM4D
                dend_thresh_mice['G06'] = 24.0 #
                dend_thresh_mice['G07'] = 40.0 #
                dend_thresh_mice['G14'] = 28.0 #
                dend_thresh_mice['G15'] = 30.0 #
                dend_thresh_mice['G20'] = 43.4 #
                dend_thresh_mice['G21'] = 29.28 #

                ## mCherry
                dend_thresh_mice['G08'] = 45.2 #
                dend_thresh_mice['G09'] = 36.1 #
                dend_thresh_mice['G12'] = 35.9 #
                dend_thresh_mice['G13'] = 23.96 #
                dend_thresh_mice['G16'] = 36.75 #
                dend_thresh_mice['G17'] = 86.7 #

        elif session_type == 'Test_B':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 70.0 ##
                dend_thresh_mice['G10'] = 81.2 ##
                dend_thresh_mice['G11'] = 111.6 ##
                dend_thresh_mice['G18'] = 104.8 ##
                dend_thresh_mice['G19'] = 76.6 ##

                ## hM4D
                dend_thresh_mice['G06'] = 74.4 #
                dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
                dend_thresh_mice['G14'] = 77.4 #
                dend_thresh_mice['G15'] = 71.4 # Have to use 'TFC_cond+Test_B' mapping since Test_B_1wk not recorded. Or don't include!
                dend_thresh_mice['G20'] = 82.8 #
                dend_thresh_mice['G21'] = 73.2 #

                ## mCherry
                dend_thresh_mice['G08'] = 155.6 #
                dend_thresh_mice['G09'] = 77.1 #
                dend_thresh_mice['G12'] = 106.8 # 
                dend_thresh_mice['G13'] = 69.6 #
                dend_thresh_mice['G16'] = 83.0 #
                dend_thresh_mice['G17'] = 82.3 #

            if Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 60.6 #
                dend_thresh_mice['G10'] = 69.8 #
                dend_thresh_mice['G11'] = 95.1 #
                dend_thresh_mice['G18'] = 94.1 #
                dend_thresh_mice['G19'] = 62.3 #

                ## hM4D
                dend_thresh_mice['G06'] = 60.6 #
                dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
                dend_thresh_mice['G14'] = 75.1 #
                dend_thresh_mice['G15'] = 62.7 #
                dend_thresh_mice['G20'] = 72.8 #
                dend_thresh_mice['G21'] = 25.37 #

                ## mCherry
                dend_thresh_mice['G08'] = 66.0 #
                dend_thresh_mice['G09'] = 70.2 #
                dend_thresh_mice['G12'] = 71.6 #
                dend_thresh_mice['G13'] = 55.3 #
                dend_thresh_mice['G16'] = 68.7 #
                dend_thresh_mice['G17'] = 82.8 #
        
            if Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 48.0 #
                dend_thresh_mice['G10'] = 45.0 #
                dend_thresh_mice['G11'] = 76.1 #
                dend_thresh_mice['G18'] = 56.1 #
                dend_thresh_mice['G19'] = 52.7 #

                ## hM4D
                dend_thresh_mice['G06'] = 35.8 #
                dend_thresh_mice['G07'] = None # Wasn't recorded (G07/Test_B)
                dend_thresh_mice['G14'] = 37.6 #
                dend_thresh_mice['G15'] = 44.6 #
                dend_thresh_mice['G20'] = 38.3 #
                dend_thresh_mice['G21'] = 66.5 #

                ## mCherry
                dend_thresh_mice['G08'] = 99.3 #
                dend_thresh_mice['G09'] = 44.6 #
                dend_thresh_mice['G12'] = 80.8 #
                dend_thresh_mice['G13'] = 44.1 #
                dend_thresh_mice['G16'] = 52.7 #
                dend_thresh_mice['G17'] = 57.6 #

        elif session_type == 'Test_B_1wk':

            if Ca_act_type == 'full':

                ## hM3D
                dend_thresh_mice['G05'] = 87.8 #
                dend_thresh_mice['G10'] = 102.5 #
                dend_thresh_mice['G11'] = 121.4 #
                dend_thresh_mice['G18'] = 124.6 #
                dend_thresh_mice['G19'] = 85.6 #

                ## hM4D
                dend_thresh_mice['G06'] = 74.5 #
                dend_thresh_mice['G07'] = 98.7 ## Skip since no Test_B # NOT TRUE! CAN DO!
                dend_thresh_mice['G14'] = 68.8 #
                dend_thresh_mice['G15'] = None ## Also skip since no Test_B_1wk
                dend_thresh_mice['G20'] = 87.3 #
                dend_thresh_mice['G21'] = 91.4 #

                ## mCherry
                dend_thresh_mice['G08'] = 84.5 #
                dend_thresh_mice['G09'] = 78.8 #
                dend_thresh_mice['G12'] = 81.5 #
                dend_thresh_mice['G13'] = 40.1 #
                dend_thresh_mice['G16'] = 99.0 #
                dend_thresh_mice['G17'] = 150.4 #

            if Ca_act_type == 'mov':

                ## hM3D
                dend_thresh_mice['G05'] = 81.3 #
                dend_thresh_mice['G10'] = 106.6 #
                dend_thresh_mice['G11'] = 120.1 #
                dend_thresh_mice['G18'] = 120.0 #
                dend_thresh_mice['G19'] = 63.3 #

                ## hM4D
                dend_thresh_mice['G06'] = 56.8 #
                dend_thresh_mice['G07'] = 85.0 ## Skip since no Test_B # NOT TRUE! CAN DO!
                dend_thresh_mice['G14'] = 61.1 #
                dend_thresh_mice['G15'] = None # Also skip since no Test_B_1wk
                dend_thresh_mice['G20'] = 79.0 #
                dend_thresh_mice['G21'] = 67.6 #

                ## mCherry
                dend_thresh_mice['G08'] = 74.7 #
                dend_thresh_mice['G09'] = 70.9 #
                dend_thresh_mice['G12'] = 72.2 #
                dend_thresh_mice['G13'] = 32.1 #
                dend_thresh_mice['G16'] = 82.1 #
                dend_thresh_mice['G17'] = 114.4 #
        
            if Ca_act_type == 'imm':

                ## hM3D
                dend_thresh_mice['G05'] = 62.5 #
                dend_thresh_mice['G10'] = 49.5 #
                dend_thresh_mice['G11'] = 42.3 #
                dend_thresh_mice['G18'] = 66.8 #
                dend_thresh_mice['G19'] = 48.5 #

                ## hM4D
                dend_thresh_mice['G06'] = 27.3 #
                dend_thresh_mice['G07'] = 58.8 # Skip since no Test_B # NOT TRUE! CAN DO!
                dend_thresh_mice['G14'] = 31.1 #
                dend_thresh_mice['G15'] = None # Also skip since no Test_B_1wk
                dend_thresh_mice['G20'] = 33.4 #
                dend_thresh_mice['G21'] = 71.4 #

                ## mCherry
                dend_thresh_mice['G08'] = 41.6 #
                dend_thresh_mice['G09'] = 47.2 #
                dend_thresh_mice['G12'] = 53.1 #
                dend_thresh_mice['G13'] = 15.63 #
                dend_thresh_mice['G16'] = 32.3 #
                dend_thresh_mice['G17'] = 107.2 #


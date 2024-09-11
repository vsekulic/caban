import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import scipy.stats as stats
import os
import itertools
import pandas as pd
import seaborn as sns
import random
import statsmodels.stats.multicomp as mc
from caban.utilities import *
from caban.sessions import *
from caban.spatial import *
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.preprocessing import PowerTransformer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score

group_colours = {
    'hM3D' : 'r',
    'hM4D' : 'b',
    'mCherry' : 'k'
}
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

PVALS = [0.05, 0.01, 0.001]

class PopulationVector:
    def __init__(self, mouse, group, session, binary_C, labels, frac_labels, labels_tot, dend_thresh, x_dend=None, y_ss=None, y_n_clusters=None, only_crossreg=False, PV_dist=None):
        self.mouse = mouse
        self.group = group
        self.session = session
        self.binary_C = binary_C
        self.labels = labels
        self.frac_labels = frac_labels # Fraction of label neurons that is re-activated (i.e. in crossreg)
        self.labels_tot = labels_tot
        self.x_dend = x_dend
        self.y_ss = y_ss
        self.y_n_clusters = y_n_clusters
        self.dend_thresh = dend_thresh
        self.only_crossreg = only_crossreg
        self.PV_dist = PV_dist

        self.is_classified = False

        # Because not returning it from cluster_pop_vectors_helper(), so just lazily calculate it afterwards..
        self.reactivated_crossreg = {} # Number of label neurons that are re-activated (from crossreg)
        if not only_crossreg: # only compute if we did not only use crossreg..
            for label in range(len(labels_tot)):
                self.reactivated_crossreg[label] = int(frac_labels[label] * labels_tot[label])

def plot_session_sp_rates(PLOTS_DIR, mouse_groups, sp_rates, session_type, title_str, figsize=(8,4)):
    '''
    Plot bars for each of the behaviour periods specified in sp_rates, grouped according to mouse_groups.

    mouse_groups - dict of mouse : group
    sp_rates - dict of group : list of sp_rates 

    The lists of sp_rates should be the same length across all groups. The function is agnostic as to the number and types of
    groups; all bar plots will be made "correctly" and "prettily".
    '''
    # Pre-allocate groups dict placeholders for average firing rates per group
    groups_avg = dict()
    mice_per_group = dict()
    period_weights = dict()
    if 'TFC_cond' in session_type:
        default_num_periods = 5
    elif 'Test_B' in session_type or 'Test_B_1wk' in session_type:
        default_num_periods = 3

    for mouse in mouse_groups.keys():
        group = mouse_groups[mouse]
        groups_avg[group] = []

        if group not in mice_per_group:
            mice_per_group[group] = 1
        else:
            mice_per_group[group] += 1
    for group in groups_avg.keys():
        groups_avg[group] = np.array(0)

    # Per-period number of mice, needed as sometimes the last period wasn't included in analysis due to error in recording
    # So we detect that further down
    for group in mice_per_group.keys():
        period_weights[group] = np.zeros(default_num_periods)

    #x = [0,1,2,3,4]
    mice = sp_rates.keys()
    for m in mice:
        mouse_group = mouse_groups[m]
        periods_avg = groups_avg[mouse_group]
        periods = sp_rates[m]
        x = range(len(periods))

        # Build up per-period weights
        for p in range(len(periods)):
            period_weights[mouse_group][p] += 1

        if not periods_avg.any():
            periods_avg = np.array(periods)
        else:
            for i in x:
                periods_avg[i] += periods[i]
        groups_avg[mouse_group] = periods_avg

    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    group_num = 0
    group_tot = len(groups_avg.keys())

    ax.spines[['right','top']].set_visible(False)
    for group in sorted(groups_avg.keys()):
        group_avg = groups_avg[group] 
        #group_avg = group_avg / mice_per_group[group]
        group_avg = group_avg / period_weights[group]
        x = np.arange(len(group_avg))
        ax.bar(x + ((group_num)*width)/group_tot, group_avg[x], label=group, color=group_colours[group], width=width/group_tot)
        group_num += 1
    ax.legend()
    plt.title(title_str)
    os.makedirs(os.path.join(PLOTS_DIR, 'sp_rates', '{}_sp_rates'.format(session_type)), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'sp_rates', '{}_sp_rates'.format(session_type), '{}_sp_rates-'.format(session_type)+title_str+'.png'), format='png', dpi=300)
    plt.close()

    # Plot average rates and do anova
    num_groups = len(groups_avg)
    x = range(num_groups)
    means = np.zeros(num_groups)
    stds = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))
    weighted_averages = dict()
    for group, idx in zip(['hM3D', 'hM4D', 'mCherry'], range(3)):
        weighted_averages[group] = groups_avg[group] / period_weights[group]
        means[idx] = np.mean(weighted_averages[group])
        stds[idx] = np.std(weighted_averages[group])
        errbars[1,idx] = stds[idx]
    fig, ax = plt.subplots(figsize=(3,6))
    ax.spines[['right','top']].set_visible(False)
    ax.bar(x, means, yerr=errbars, color=group_colours.values())
    do_anova1_plot(weighted_averages['hM3D'], weighted_averages['hM4D'], weighted_averages['mCherry'], ax, means+stds)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Exc', 'Inh', 'Ctl'], size='medium')
    plt.suptitle(title_str + ' avg')
    os.makedirs(os.path.join(PLOTS_DIR, 'sp_rates', '{}_sp_rates'.format(session_type)), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'sp_rates', '{}_sp_rates'.format(session_type), '{}_sp_rates-'.format(session_type)+title_str+'-avg.png'), format='png', dpi=300)
    plt.close()

def plot_LT_sp_rates(PLOTS_DIR, mouse_groups, sp_rates, title_str, figsize=(8,8), with_peakval=False):
    
    # Pre-allocate groups dict placeholders for average firing rates per group
    groups_avg = dict()
    mice_per_group = dict()
    for mouse in mouse_groups.keys():
        group = mouse_groups[mouse]
        groups_avg[group] = [0]

        if group not in mice_per_group:
            mice_per_group[group] = 1
        else:
            mice_per_group[group] += 1
    for group in groups_avg.keys():
        groups_avg[group] = np.array(0.0)

    mice = sp_rates.keys()
    for m in mice:
        mouse_group = mouse_groups[m]
        sp_rate = np.array(sp_rates[m][0])
        groups_avg[mouse_group] += sp_rate

    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    group_num = 0
    group_tot = len(groups_avg.keys())

    for group in sorted(groups_avg.keys()):
        group_avg = groups_avg[group] 
        #group_avg = group_avg / mice_per_group[group]
        group_avg = group_avg / mice_per_group[group]
        ax.bar(((group_num)*width)/group_tot, group_avg, label=group, color=group_colours[group], width=width/group_tot)
        group_num += 1
    ax.legend()
    plt.title(title_str)
    if with_peakval:
        dir_str = 'LT_activity'
    else:
        dir_str = 'LT_sp_rates'
    os.makedirs(os.path.join(PLOTS_DIR, dir_str), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_str, dir_str+'-'+title_str+'.png'), format='png', dpi=300)
    plt.close()

def plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, binned_sp_rates, mapping, Session, session_type, bin_width, figsize=(8,3.5), want_close=True,
                                 plot_bars=True, paper_dir=None, suffix=''):
    bin_lengths_set = set()
    for mouse in binned_sp_rates.keys():
        bin_lengths_set.add(len(binned_sp_rates[mouse]))
    num_bins = max(bin_lengths_set)
    
    if session_type == 'TFC_cond' or session_type == 'TFC_cond-activity':
        want_shocks = True
        num_periods = 5
    else:
        want_shocks = False
        num_periods = 3

    groups_set = set()
    for mouse in mouse_groups.keys():
        groups_set.add(mouse_groups[mouse])
    groups_set = {'hM3D', 'mCherry', 'hM4D'}

    # Find average tone/shock onsets/offsets for all mice to overlay on top of histogram
    tone_onsets = np.zeros(num_periods)
    tone_offsets = np.zeros(num_periods)
    if want_shocks:
        shock_onsets = np.zeros(num_periods)
        shock_offsets = np.zeros(num_periods)
    num_mice = 0
    for mouse in Session.keys():
        if len(Session[mouse].tone_offsets) != num_periods:
            continue
        print(mouse)
        print(Session[mouse].tone_onsets)
        print(Session[mouse].tone_offsets)
        if want_shocks:
            print(Session[mouse].shock_onsets)
            print(Session[mouse].shock_offsets)
        print("---")
        tone_onsets += np.array(Session[mouse].tone_onsets)
        tone_offsets += np.array(Session[mouse].tone_offsets)
        if want_shocks:
            shock_onsets += np.array(Session[mouse].shock_onsets)
            shock_offsets += np.array(Session[mouse].shock_offsets)
        print(tone_onsets)
        print(tone_offsets)
        if want_shocks:
            print(shock_onsets)
            print(shock_offsets)
        print('---')
        num_mice += 1
    tone_onsets = tone_onsets / num_mice / bin_width
    tone_offsets = tone_offsets / num_mice / bin_width
    if want_shocks:
        shock_onsets = shock_onsets / num_mice / bin_width
        shock_offsets = shock_offsets / num_mice / bin_width

    groups_bins = dict()
    mice_per_group = dict()
    norm_vector = dict()
    sem_bins = dict()
    norm_sem_vector = dict()
    norm_groups = dict()
    for group in groups_set:
        groups_bins[group] = np.zeros(num_bins)
        mice_per_group[group] = 0
        norm_groups[group] = 0
        sem_bins[group] = np.zeros((list(mouse_groups.values()).count(group), num_bins))
        norm_sem_vector[group] = np.zeros(num_bins)

    group_i = {'hM3D':0, 'hM4D':0, 'mCherry':0}
    for mouse in binned_sp_rates.keys():

        # Pad mice that had truncated recordings to bin_lengths_set with zeroes
        #print(type(binned_sp_rates[mouse]))
        #print(binned_sp_rates[mouse])
        #print(len(binned_sp_rates[mouse]))
        bins = np.array(binned_sp_rates[mouse])
        group = mouse_groups[mouse]
        norm_groups[group] += 1
        if len(bins) < num_bins:
            #print('gonna pad {} now... {} {} subtr: {}'.format(mouse, num_bins, len(bins), num_bins-len(bins)))

            # np.pad() doesn't modify in place anymore (online docs out of date)
            bins = np.pad(bins, [(0, num_bins - len(bins))], mode='constant', constant_values=0)
        #max_val = np.max([np.max(bins), max_val])
        sem_bins[group][group_i[group], :] = bins
        #print(bins)
        groups_bins[group] += bins
        mice_per_group[group] += 1
        group_i[group] += 1

    plt.figure(figsize=figsize)
    plt.gca().spines[['right','top']].set_visible(False)
    for group in groups_bins.keys():
        bins = groups_bins[group] / norm_groups[group]#/ max_val
        if plot_bars:
            plt.bar(range(0,len(bins)), bins, color=group_colours[group], alpha=0.1)
        else:
            #x = range(0,len(bins))
            x = np.linspace(0, len(bins)*bin_width/20/60, num=len(bins))
            sem_group = np.std(sem_bins[group],0) / np.sqrt(np.sum(sem_bins[group],0))
            plt.plot(x, bins, color=group_colours[group])
            plt.fill_between(x, bins - sem_group, bins + sem_group, color=group_colours[group], alpha=0.1)
            plt.xticks(np.arange(1,np.floor(len(bins)*bin_width/20/60)+1,2))
        if not paper_dir:
            plt.title('Average binned firing rates for '+mapping)

    multiplier = 1
    if not plot_bars:
        multiplier = bin_width/20/60
    [plt.axvline(x, c='b', ls='--') for x in tone_onsets*multiplier]
    [plt.axvline(x, c='b', ls='--') for x in tone_offsets*multiplier]
    if want_shocks:
        [plt.axvline(x, c='r', ls='--') for x in shock_onsets*multiplier]
        [plt.axvline(x, c='r', ls='--') for x in shock_offsets*multiplier]

    if not plot_bars:
        plt.xlim([0,len(bins)*bin_width/20/60])
    [min_y, max_y] = plt.gca().get_ylim()
    range_y = max_y - min_y
    plt.yticks([min_y, range_y*0.25, range_y*0.5, range_y*0.75, range_y],[0,25,50,75,100])
    plt.ylabel(r'Normalized $\Delta$F/F (%)')
    plt.xlabel('Time (min)')

    plt.tight_layout()
    filename = '{}_binned_sp_rates_mapping-'.format(session_type)+mapping+suffix+'.png'
    os.makedirs(os.path.join(PLOTS_DIR, 'sp_rates', '{}_binned_sp_rates_mapping'.format(session_type)), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'sp_rates', '{}_binned_sp_rates_mapping'.format(session_type), filename), format='png', dpi=300, transparent=True)

    if paper_dir:
        path_name = os.path.join(paper_dir, filename)
        print('*** Plotting paper_dir {}'.format(path_name))
        plt.savefig(path_name, format='png', dpi=300)    
    if want_close:
        plt.close()

def plot_two_ROI_mappings(PLOTS_DIR, mouse, group, session1_ROI_mappings, mapping1, session2_ROI_mappings, mapping2, want_peakval=False):
    A = session1_ROI_mappings[mapping1][mouse]
    B = session2_ROI_mappings[mapping2][mouse]
    if not want_peakval: # Do not set masks if want peakvals as the passed in ROIs should be the A_peakval ones.
        A = (A>0).astype(int)
        B = (B>0).astype(int)

    plt.figure()
    plt.imshow(A, cmap='Reds', alpha=0.5)
    plt.imshow(B, cmap='Blues', alpha=0.5)
    plt.title('{} {} ROIs {} (R) vs {} (B)'.format(mouse, group, mapping1, mapping2))
    #plt.savefig(os.path.join(PLOTS_DIR, 'ROI-{}-{}-{}-{}.svg'.format(mapping1, mapping2, mouse, group)), format='svg', dpi=300)

    if want_peakval:
        save_path = os.path.join(PLOTS_DIR, group, 'peakval')
    else:
        save_path = os.path.join(PLOTS_DIR, group)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'ROI-{}-{}-{}-{}.png'.format(mapping1, mapping2, mouse, group)), format='png', dpi=300)
    plt.close()

def plot_three_ROI_mappings(PLOTS_DIR, mouse, group, session1_ROI_mappings, mapping1, session2_ROI_mappings, mapping2, \
    session3_ROI_mappings, mapping3, want_peakval=False):
    A = session1_ROI_mappings[mapping1][mouse]
    B = session2_ROI_mappings[mapping2][mouse]
    C = session3_ROI_mappings[mapping3][mouse]
    if not want_peakval: # Do not set masks if want peakvals as the passed in ROIs should be the A_peakval ones.
        A = (A>0).astype(int)
        B = (B>0).astype(int)
        C = (C>0).astype(int)

    plt.figure()
    plt.imshow(A, cmap='Reds', alpha=0.5)
    plt.imshow(B, cmap='Blues', alpha=0.5)
    plt.imshow(C, cmap='Greys', alpha=0.5)
    plt.title('{} {} ROIs {} (R) vs {} (U) vs {} (B)'.format(mouse, group, mapping1, mapping2, mapping3))
    #plt.savefig(os.path.join(PLOTS_DIR, 'ROI-{}-{}-{}-{}.svg'.format(mapping1, mapping2, mouse, group)), format='svg', dpi=300)

    if want_peakval:
        save_path = os.path.join(PLOTS_DIR, 'ROIs', group, 'peakval')
    else:
        save_path = os.path.join(PLOTS_DIR, 'ROIs', group)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'ROI-{}-{}-{}-{}-{}.png'.format(mapping1, mapping2, mapping3, mouse, group)), format='png', dpi=300)
    plt.close()

def plot_ROI_mappings(PLOTS_DIR, mouse_groups, TFC_cond_ROI_mappings, TFC_cond_mappings, LT1_ROI_mappings, LT1_mappings, LT2_ROI_mappings, LT2_mappings, want_peakval=False):
    
    for mouse, group in mouse_groups.items():
        for A, B in itertools.combinations(TFC_cond_ROI_mappings,2):
            plot_two_ROI_mappings(PLOTS_DIR, mouse, group, TFC_cond_ROI_mappings, A, TFC_cond_ROI_mappings, B, want_peakval=want_peakval)
    
    for mouse, group in mouse_groups.items():
        plot_two_ROI_mappings(PLOTS_DIR, mouse, group, LT1_ROI_mappings, 'LT1', LT2_ROI_mappings, 'LT2', want_peakval=want_peakval)
        plot_two_ROI_mappings(PLOTS_DIR, mouse, group, LT1_ROI_mappings, 'LT1+LT2', LT2_ROI_mappings, 'LT1+LT2', want_peakval=want_peakval)
        plot_three_ROI_mappings(PLOTS_DIR, mouse, group, LT1_ROI_mappings, 'LT1', LT2_ROI_mappings, 'LT2', LT1_ROI_mappings, 'LT1+LT2', want_peakval=want_peakval)

def do_anova1_plot(group_hM3D, group_hM4D, group_mCherry, ax, heights, annotate=True):
    f_oneway = stats.f_oneway(group_hM3D, group_hM4D, group_mCherry)
    print(f_oneway)
    if np.any(f_oneway.pvalue < PVALS):
        mc_fracs = np.concatenate((group_hM3D, group_hM4D, group_mCherry),dtype='float64')
        mc_groups = np.concatenate((['hM3D'] * len(group_hM3D), ['hM4D'] * len(group_hM4D), ['mCherry'] * len(group_mCherry)))
        df = pd.DataFrame({'groups':mc_groups, 'fracs':mc_fracs})
        comp = mc.MultiComparison(df['fracs'], df['groups'])
        post_hoc_res = comp.tukeyhsd()
        print(post_hoc_res.summary())
        pairs = [[0,1],[0,2],[1,2]]
        tot_dh = 0.05
        for p in np.where(post_hoc_res.reject==True)[0]:
            first = pairs[p][0]
            second = pairs[p][1]
            pval_str = get_pval_str(post_hoc_res.pvalues[p])
            if annotate:
                barplot_annotate_brackets(ax, first, second, pval_str, range(3), heights, barh=0, dh=tot_dh)
            tot_dh += 0.05 # was 0.1 ! NB!

def proportional_activities_helper(PLOTS_DIR, mice_per_group, first, second, third, sessions=[], session_names=[], \
    figsize=(10,4), title_str='', filename='', auto_close=True):
    session_strs = dict()
    for i in range(len(session_names)):
        session_strs[i] = session_names[i]
    num_comparisons = len(session_names)

    active_cells = dict()
    group_totals = dict()
    fracs_per_group = dict()
    for group, mice in mice_per_group.items():

        active_cells[group] = np.zeros((len(mice), num_comparisons)) 
        for m in range(len(mice)):
            mouse = mice[m]
            if sessions:
                for i in range(len(sessions)):
                    try:
                        session_str = sessions[i]
                        session_obj = None
                        if first[mouse].session_type in session_str:
                            session_obj = first[mouse]
                        if second[mouse].session_type in session_str:
                            session_obj = second[mouse]
                        if third[mouse].session_type in session_str:
                            session_obj = third[mouse]
                        [S, S_spikes, S_peakval, S_idx] = session_obj.get_S_mapping(session_str)
                    except:
                        S_spikes = []
                    active_cells[group][m, i] += len(S_spikes)
                    #active_cells[group][m, i] += len(np.where(np.mean(zscore(S,1),1)>0)[0])
        
        fracs_per_group[group] = np.zeros((len(mice), num_comparisons))
        for m in range(len(mice)):
            for i in range(num_comparisons):
                active_cells_sum = np.sum(active_cells[group][m,:])
                if active_cells_sum > 0:
                    fracs_per_group[group][m,i] = active_cells[group][m,i] / np.sum(active_cells[group][m,:])
                else:
                    fracs_per_group[group][m,i] = active_cells[group][m,i]
        group_totals[group] = np.sum(active_cells[group],0) # Summate along all mice, resulting in 3-tuple

    fig, axs = plt.subplots(1,num_comparisons, figsize=figsize, sharey='row')
    group_totals_l = list(group_totals)
    max_y = 0
    num_groups = len(group_totals)   

    x = range(num_groups)
    means = np.zeros(num_groups)
    stds = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))

    for i, ax in zip(range(num_comparisons), axs.flat):

        for group in group_totals.keys():
            idx = group_totals_l.index(group)

            fracs = fracs_per_group[group][:,i]
            means[idx] = np.mean(fracs)
            stds[idx] = np.std(fracs)
            #errbar = np.zeros((2,1))
            errbars[1,idx] = stds[idx]

            if means[idx]+stds[idx] > max_y:
                max_y = means[idx]+stds[idx]+0.01
            #ax.bar(group_totals_l.index(group), mean, yerr=errbars, label=group, color=group_colours[group])
            #print(i, group, group_totals_l.index(group), frac)

        print(session_strs[i], stats.f_oneway(fracs_per_group['hM3D'][:,i], fracs_per_group['hM4D'][:,i], fracs_per_group['mCherry'][:,i]))

        ax.bar(x, means, yerr=errbars, label=session_strs[i], color=group_colours.values())

        do_anova1_plot(fracs_per_group['hM3D'][:,i], fracs_per_group['hM4D'][:,i], fracs_per_group['mCherry'][:,i], ax, means+stds)
        
        ax.set_xticks(range(3))
        ax.set_ylim([0,max(max_y, 0.3)])
        ax.set_xticklabels(['Exc', 'Inh', 'Ctl'], size='medium')
        ax.set_title(session_strs[i], size='medium')
    plt.suptitle(title_str)
    plt.subplots_adjust(left=0.07, bottom=0.08, right=0.97, top=0.84, wspace=0.21)
    os.makedirs(os.path.join(PLOTS_DIR, 'proportional_activities'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'proportional_activities', filename), format='png', dpi=300)
    if auto_close:
        plt.close()

def proportional_activities(PLOTS_DIR, mice_per_group, TFC_cond, TFC_cond_LT1, TFC_cond_LT2):
    '''
    Plot two ways. First is fraction of each of LT1, LT2, TFC_cond cells wrt all others, regardless of cross-registration. This
    means cells may be counted twice or three times depending on whether they are active in the other sessions. But gives overall sense of
    "activity per session".

    The second way is to plot relative cross-registration activities. I.e., LT1 only cells as a % of all other mappings, and so on.
    '''

    sessions = ['LT1', 'LT2', 'TFC_cond']
    session_names = ['LT1', 'LT2', 'TFC']
    proportional_activities_helper(PLOTS_DIR, mice_per_group, TFC_cond_LT1, TFC_cond_LT2, TFC_cond, \
        sessions=sessions, session_names=session_names, \
        figsize=(6,4), title_str='Fraction of active cells across sessions', filename='frac-active-sessions.png')

    sessions = ['LT1', 'LT2', 'LT1+LT2', 'LT1+TFC_cond', 'LT2+TFC_cond', 'LT1+LT2+TFC_cond', 'TFC_cond']
    session_names = ['LT1', 'LT2', 'LT1+LT2', 'LT1+TFC', 'LT2+TFC', 'LT1+LT2+TFC', 'TFC']
    proportional_activities_helper(PLOTS_DIR, mice_per_group, TFC_cond, TFC_cond_LT1, TFC_cond_LT2, \
            sessions=sessions, session_names=session_names, \
            figsize=(10,4), title_str='Fraction of active cells across mappings', filename='frac-active-mappings.png')

def proportional_activities_TFC_B_B_1wk(PLOTS_DIR, mice_per_group, TFC_cond, Test_B, Test_B_1wk):
    '''
    See proportional_activities(). 
    '''
    sessions = ['TFC_cond', 'Test_B', 'Test_B_1wk']
    session_names = ['TFC', '48hr', '1wk']
    proportional_activities_helper(PLOTS_DIR, mice_per_group, TFC_cond, Test_B, Test_B_1wk, \
        sessions=sessions, session_names=session_names, \
        figsize=(6,4), title_str='Fraction of active cells across sessions', filename='frac-active-sessions-TFC_B_B_1wk.png')
    
    sessions = ['TFC_cond', 'Test_B', 'Test_B_1wk', 'TFC_cond+Test_B', 'TFC_cond+Test_B_1wk', 'Test_B+Test_B_1wk', 'TFC_cond+Test_B+Test_B_1wk']
    session_names = ['TFC', '48hr', '1wk', 'TFC+48hr', 'TFC+1wk', '48hr+1wk', 'TFC+48hr+1wk']
    proportional_activities_helper(PLOTS_DIR, mice_per_group, TFC_cond, Test_B, Test_B_1wk, \
            sessions=sessions, session_names=session_names, \
            figsize=(10,4), title_str='Fraction of active cells across mappings', filename='frac-active-mappings-TFC_B_B_1wk.png')

def proportional_activities_donut(PLOTS_DIR, mouse_groups, first, second, third, session_names, crossreg_type='TFC_cond', crossreg_to_use=None):
    '''
    As above but as donut plots.

    crossreg_type must be specified as either:
        'TFC_cond' - TFC,LT1,LT2
        'TFC_B_B_1wk' - TFC,Test B, Test B +1wk

    crossreg_to_use must also be specified and matching with crossreg_type.
    '''

    first_groups = {'hM3D':None, 'hM4D':None, 'mCherry':None}
    second_groups = {'hM3D':None, 'hM4D':None, 'mCherry':None}
    third_groups = {'hM3D':None, 'hM4D':None, 'mCherry':None}

    first_groups_labels = {}
    second_groups_labels = {}
    third_groups_labels = {}

    for g in ['hM3D', 'hM4D', 'mCherry']:
        if crossreg_type == 'TFC_cond':
            first_groups[g] = {'TFC_cond':0, 'TFC_cond+LT1':0, 'TFC_cond+LT2':0, 'TFC_cond+LT1+LT2':0, 'tot':0}
            first_groups_labels[g] = ['TFC', 'TFC+LT1', 'TFC+LT2', 'TFC+LT1+LT2']
            second_groups[g] = {'LT1':0, 'TFC_cond+LT1':0, 'LT1+LT2':0, 'TFC_cond+LT1+LT2':0, 'tot':0}
            second_groups_labels[g] = ['LT1', 'TFC+LT1', 'LT1+LT2', 'TFC+LT1+LT2']
            third_groups[g] = {'LT2':0, 'TFC_cond+LT2':0, 'LT1+LT2':0, 'TFC_cond+LT1+LT2':0, 'tot':0}
            third_groups_labels[g] = ['LT2', 'TFC+LT2', 'LT1+LT2', 'TFC+LT1+LT2']
        elif crossreg_type == 'TFC_B_B_1wk':
            first_groups[g] = {'TFC_cond':0, 'TFC_cond+Test_B':0, 'TFC_cond+Test_B_1wk':0, 'TFC_cond+Test_B+Test_B_1wk':0, 'tot':0}
            first_groups_labels[g] = ['TFC', 'TFC+B', 'TFC+B_1wk', 'TFC+B+B_1wk']
            second_groups[g] = {'Test_B':0, 'TFC_cond+Test_B':0, 'Test_B+Test_B_1wk':0, 'TFC_cond+Test_B+Test_B_1wk':0, 'tot':0}
            second_groups_labels[g] = ['B', 'TFC+B', 'B+B_1wk', 'TFC+B+B_1wk']
            third_groups[g] = {'Test_B_1wk':0, 'TFC_cond+Test_B_1wk':0, 'Test_B+Test_B_1wk':0, 'TFC_cond+Test_B+Test_B_1wk':0, 'tot':0}
            third_groups_labels[g] = ['B_1wk', 'TFC+B_1wk', 'B+B_1wk', 'TFC+B+B_1wk']

    for m, g in mouse_groups.items():
        for mapping in first_groups[g].keys():
            try:
                if crossreg_type == 'TFC_cond':
                    [S, S_spikes, S_peakval, S_idx] = first[m].get_S_mapping(mapping, with_peakval=True)
                elif crossreg_type == 'TFC_B_B_1wk':
                    [S, S_spikes, S_peakval, S_idx] = first[m].get_S_mapping(mapping, with_crossreg=crossreg_to_use[m], with_peakval=True)
            except:
                pass
            else:
                first_groups[g][mapping] += len(S_idx)
        try:
            first_groups[g]['tot'] += first[m].S.shape[0]
        except:
            pass
        for mapping in second_groups[g].keys():
            try:
                if crossreg_type == 'TFC_cond':
                    [S, S_spikes, S_peakval, S_idx] = second[m].get_S_mapping(mapping, with_peakval=True)
                elif crossreg_type == 'TFC_B_B_1wk':
                    [S, S_spikes, S_peakval, S_idx] = second[m].get_S_mapping(mapping, with_crossreg=crossreg_to_use[m], with_peakval=True)
            except:
                pass
            else:
                second_groups[g][mapping] += len(S_idx)
        try:
            second_groups[g]['tot'] += second[m].S.shape[0]
        except:
            pass
        for mapping in third_groups[g].keys():
            try:
                if crossreg_type == 'TFC_cond':
                    [S, S_spikes, S_peakval, S_idx] = third[m].get_S_mapping(mapping, with_peakval=True)
                elif crossreg_type == 'TFC_B_B_1wk':
                    [S, S_spikes, S_peakval, S_idx] = third[m].get_S_mapping(mapping, with_crossreg=crossreg_to_use[m], with_peakval=True)
            except:
                pass
            else:
                third_groups[g][mapping] += len(S_idx)
        try:
            third_groups[g]['tot'] += third[m].S.shape[0]
        except:
            pass

    for g in [first_groups, second_groups, third_groups]:
        for m in g.values():
            for mapping_key, num_cells in m.items():
                m[mapping_key] = [num_cells, num_cells/m['tot']]

    for i, (groups_dict, groups_labels_dict, sess_name) in enumerate(zip([first_groups, second_groups, third_groups], \
            [first_groups_labels, second_groups_labels, third_groups_labels], session_names)):
        fig, axs = plt.subplots(1,3, figsize=(8,4), subplot_kw=dict(aspect="equal"))
        for group,ax in zip(['hM3D', 'hM4D', 'mCherry'], axs):
            if 'tot' in groups_dict[group].keys():
                del groups_dict[group]['tot']
            data = [perc for tot,perc in groups_dict[group].values()]
            labels = ['{:.1f}%'.format(f*100) for f in data]
            ptr_labels = list(groups_labels_dict[group])
            wedges, texts = ax.pie(data, labels=labels, labeldistance=0.5, wedgeprops=dict(width=0.5))

            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-"),
                    bbox=bbox_props, zorder=0, va="center")

            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = f"angle,angleA=0,angleB={ang}"
                kw["arrowprops"].update({"connectionstyle": connectionstyle, "color": "black", "lw": 1})
                kw["bbox"].update({"boxstyle": "round,pad=0.3", "edgecolor": "black", "facecolor":"white"})
                kw["fontsize"] = 8
                ax.annotate(ptr_labels[i], xy=(x, y), xytext=((1+i*0.15)*np.sign(x), (1+i*0.3)*y), #xytext=(1.35*np.sign(x), 1.4*y), 
                            horizontalalignment=horizontalalignment, **kw)
            ax.set_title(group)
        fig.suptitle(sess_name)

def LT_firing_rate_changes(PLOTS_DIR, mice_per_group, TFC_cond_LT1, TFC_cond_LT2, use_peakval=False):

    LT_group_values = dict() # can be either spike rates or average spike intensities, depending on use_peakval

    for group, mice in mice_per_group.items():

        LT1_values = np.array([])
        LT2_values = np.array([])

        for m in range(len(mice)):
            mouse = mice[m]
            LT1 = TFC_cond_LT1[mouse]
            LT2 = TFC_cond_LT2[mouse]

            #[S_LT1, S_spikes_LT1, S_idx_LT1] = LT1.get_S_mapping('LT1+LT2')
            #[S_LT2, S_spikes_LT2, S_idx_LT2] = LT2.get_S_mapping('LT1+LT2')

            period_LT1 = [LT1.miniscope_exp_fnum[LT1.start_idx], LT1.miniscope_exp_fnum[LT1.stop_idx]]
            period_LT2 = [LT2.miniscope_exp_fnum[LT2.start_idx], LT2.miniscope_exp_fnum[LT2.stop_idx]]
            period_LT1_length = (period_LT1[1] - period_LT1[0]) / MINISCOPE_FPS
            period_LT2_length = (period_LT2[1] - period_LT2[0]) / MINISCOPE_FPS

            df_mapping = LT1.crossreg.get_mappings_cells(mapping_type='LT1+LT2')

            for i in range(len(df_mapping)):
                cell_LT1 = int(float(df_mapping[LT1.session_group].iloc[i]))
                cell_LT2 = int(float(df_mapping[LT2.session_group].iloc[i]))
                #LT1_idx = np.where(LT1.S_zarr['unit_id']==cell_LT1)[0][0]
                #LT2_idx = np.where(LT2.S_zarr['unit_id']==cell_LT2)[0][0]
                LT1_idx = np.where(LT1.S_idx==cell_LT1)[0][0]
                LT2_idx = np.where(LT2.S_idx==cell_LT2)[0][0]
                spikes_in_LT1 = LT1.S_spikes[LT1_idx]
                spikes_in_LT2 = LT2.S_spikes[LT2_idx]

                period_spikes_in_LT1 = get_spikes_in_period(spikes_in_LT1, period_LT1)
                period_spikes_in_LT2 = get_spikes_in_period(spikes_in_LT2, period_LT2)

                if use_peakval:
                    LT1_values = np.append(LT1_values, my_mean(np.take(LT1.S_peakval[LT1_idx], period_spikes_in_LT1)))
                    LT2_values = np.append(LT2_values, my_mean(np.take(LT2.S_peakval[LT2_idx], period_spikes_in_LT2)))
                else:
                    LT1_values = np.append(LT1_values, len(period_spikes_in_LT1) / period_LT1_length)
                    LT2_values = np.append(LT2_values, len(period_spikes_in_LT2) / period_LT2_length)

            LT_group_values[group] = np.reshape(np.concatenate((LT1_values, LT2_values)), (2, len(LT1_values)))

    fig, axs = plt.subplots(1, 3, figsize=(8,4), sharey='row')
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        ax.plot(LT_group_values[group], color='0.8', alpha=0.2)
        ax.plot([0, 1], LT_group_values[group].sum(axis=1)/LT_group_values[group].shape[1], \
            color=group_colours[group], marker='.', lw=1, ms=2)
        ax.set_xticks(range(2))
        #ax.set_ylim([0,max(max_y, 0.3)])
        ax.set_xticklabels(['LT1', 'LT2'], size='medium')
        ax.set_title(group, size='medium')

    if use_peakval:
        plt.suptitle('Per-cell linear track mean spike intensity changes (LT1->LT2)')
        filename = 'LT_peakval_changes-hist.png'
    else:
        plt.suptitle('Per-cell linear track firing rate changes (LT1->LT2)')
        filename = 'LT_firing_rate_changes-hist.png'
    os.makedirs(os.path.join(PLOTS_DIR, 'LT_firing_rate_changes'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'LT_firing_rate_changes', filename), format='png', dpi=300)
    plt.close()

    plt.figure()
    for group in mice_per_group.keys():
        d1=np.diff(LT_group_values[group],axis=0)
        e1=d1.reshape((d1.shape[1],1))
        f1=e1.flatten()
        plt.hist(x=f1, color=group_colours[group], density=True, alpha=0.3, bins=100)

    if use_peakval:
        plt.suptitle('Per-cell linear track average spike peak changes (LT1->LT2)')
        filename = 'LT_average_peakval_changes.png'
    else:
        plt.suptitle('Per-cell linear track average firing rate changes (LT1->LT2)')
        filename = 'LT_average_firing_rage_changes.png'
    os.makedirs(os.path.join(PLOTS_DIR, 'LT_firing_rate_changes'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'LT_firing_rate_changes', filename), format='png', dpi=300)
    plt.close()

def plot_firing_rate_changes(PLOTS_DIR, mice_per_group, crossreg_mice, session1, session2, mapping_type, use_peakval=False):

    session_group_values = dict()  # can be either spike rates or average spike intensities, depending on use_peakval

    for group, mice in mice_per_group.items():

        s1_values = np.array([])
        s2_values = np.array([])

        for m in range(len(mice)):
            mouse = mice[m]
            s1 = session1[mouse]
            s2 = session2[mouse]
            crossreg = crossreg_mice[mouse]

            period_s1 = [s1.miniscope_exp_fnum[s1.start_idx], s1.miniscope_exp_fnum[s1.stop_idx]]
            period_s2 = [s2.miniscope_exp_fnum[s2.start_idx], s2.miniscope_exp_fnum[s2.stop_idx]]
            period_s1_length = (period_s1[1] - period_s1[0]) / MINISCOPE_FPS
            period_s2_length = (period_s2[1] - period_s2[0]) / MINISCOPE_FPS

            df_mapping = crossreg.get_mappings_cells(mapping_type=mapping_type)
            s1_df_col = s1.get_df_col()
            s2_df_col = s2.get_df_col()
            #s1_unit_column = df_mapping[s1.get_df_col()]
            #s2_unit_column = df_mapping[s2.get_df_col()]
            #s1_S_idx = get_actual_cells_from_df_session(s1_unit_column)
            #s2_S_idx = get_actual_cells_from_df_session(s2_unit_column)
                
            for i in range(len(df_mapping)):
                cell_s1 = int(float(df_mapping[s1_df_col].iloc[i]))
                cell_s2 = int(float(df_mapping[s2_df_col].iloc[i]))
                try:
                    #s1_idx = np.where(s1.S_zarr['unit_id']==cell_s1)[0][0]
                    #s2_idx = np.where(s2.S_zarr['unit_id']==cell_s2)[0][0]
                    s1_idx = np.where(s1.S_idx==cell_s1)[0][0]
                    s2_idx = np.where(s2.S_idx==cell_s2)[0][0]
                except IndexError as error:
                    # sometimes the minian-saved zarr files don't contain the cells from the crossreg mapping;
                    # so just skip this row
                    print('***WARNING: could not find {} ({}) or {} ({})'.format(cell_s1, s1.session_type, cell_s2, s2.session_type))
                    continue
                spikes_in_s1 = s1.S_spikes[s1_idx]
                spikes_in_s2 = s2.S_spikes[s2_idx]

                period_spikes_in_s1 = get_spikes_in_period(spikes_in_s1, period_s1)
                period_spikes_in_s2 = get_spikes_in_period(spikes_in_s2, period_s2)

                if use_peakval:
                    s1_values = np.append(s1_values, my_mean(np.take(s1.S_peakval[s1_idx], period_spikes_in_s1)))
                    s2_values = np.append(s2_values, my_mean(np.take(s2.S_peakval[s2_idx], period_spikes_in_s2)))
                else:
                    s1_values = np.append(s1_values, len(period_spikes_in_s1) / period_s1_length)
                    s2_values = np.append(s2_values, len(period_spikes_in_s2) / period_s2_length)

            session_group_values[group] = np.reshape(np.concatenate((s1_values, s2_values)), (2, len(s1_values)))

    fig, axs = plt.subplots(1, 3, figsize=(8,4), sharey='row')
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        ax.plot(session_group_values[group], color='0.8', alpha=0.2)
        ax.plot([0, 1], session_group_values[group].sum(axis=1)/session_group_values[group].shape[1], \
            color=group_colours[group], marker='.', lw=1, ms=2)
        ax.set_xticks(range(2))
        #ax.set_ylim([0,max(max_y, 0.3)])
        ax.set_xticklabels([s1.session_str, s2.session_str], size='medium')
        ax.set_title(group, size='medium')


    if use_peakval:
        plt.suptitle('Per-cell linear track mean spike intensity changes ({}->{})'.format(s1.session_str, s2.session_str))
        filename = '{}_{}_peakval_changes-hist.png'.format(s1.session_type, s2.session_type)
    else:
        plt.suptitle('Per-cell linear track firing rate changes ({}->{})'.format(s1.session_str, s2.session_str))
        filename = '{}_{}_firing_rate_changes-hist.png'.format(s1.session_type, s2.session_type)
    os.makedirs(os.path.join(PLOTS_DIR, 'firing_rate_changes'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'firing_rate_changes', filename), format='png', dpi=300)
    plt.close()

    plt.figure()
    diffs_per_group = dict()
    for group in mice_per_group.keys():
        d1=np.diff(session_group_values[group],axis=0)
        e1=d1.reshape((d1.shape[1],1))
        f1=e1.flatten()
        diffs_per_group[group] = f1
        plt.hist(x=f1, color=group_colours[group], density=True, alpha=0.3, bins=100)
    do_anova1_plot(diffs_per_group['hM3D'], diffs_per_group['hM4D'], diffs_per_group['mCherry'], ax, 0, annotate=False)

    if use_peakval:
        plt.suptitle('Per-cell linear track average spike peak changes ({}->{})'.format(s1.session_str, s2.session_str))
        filename = '{}_{}_average_peakval_changes.png'.format(s1.session_type, s2.session_type)
    else:
        plt.suptitle('Per-cell linear track average firing rate changes ({}->{})'.format(s1.session_str, s2.session_str))
        filename = '{}_{}_average_firing_rage_changes.png'.format(s1.session_type, s2.session_type)
    os.makedirs(os.path.join(PLOTS_DIR, 'firing_rate_changes'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'firing_rate_changes', filename), format='png', dpi=300)
    plt.close()

def plot_sample_traces(PLOTS_DIR, mice_to_use, session, paper_dir=None, selection_mode=False, len_trace=100, desired_spikes=2, \
                       cells_per_mouse=3, selections=None):

    if selection_mode:
        selections = {'hM3D':[], 'hM4D':[], 'mCherry':[]}
        for group, m in mice_to_use.items():
            print('*** Selecting from group {}'.format(group))
            sess = session[m]
            C = sess.C
            S_spikes = sess.S_spikes
            S_peakval = sess.S_peakval

            for cell in range(cells_per_mouse):
                print('*** selecting cell {}'.format(cell))

                not_satisfied=True
                while not_satisfied:
                    cell_random = random.randrange(C.shape[0])
                    sample_t = True
                    print('sampling times: ', end='')
                    kill_num_max = 20
                    kill_num = 0                    
                    while sample_t:
                        t_idx_random = random.randrange(C.shape[1]-len_trace)
                        print('{} '.format(t_idx_random), end='')                        
                        spk_indeces = np.logical_and(sess.S_spikes[cell_random] >= t_idx_random, sess.S_spikes[cell_random] < t_idx_random+len_trace)
                        spk_times = S_spikes[cell_random][spk_indeces]
                        if len(spk_times) > desired_spikes:
                            sample_t = False
                        print('({} spk) '.format(len(spk_times)), end='')
                        kill_num += 1
                        if kill_num > kill_num_max:
                            print('*** kill switch engaged; breaking loop')
                            break
                    print('done.')

                    width=5
                    height=2
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(width,height)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    max_val = np.max(C[cell_random,t_idx_random:t_idx_random+len_trace])
                    ax.plot(C[cell_random, t_idx_random:t_idx_random+len_trace]/max_val)
                    #ax.scatter(spk_times-t_idx_random, sess.S[cell_random][spk_times], s=80, facecolor='none', edgecolors='k')
                    max_val = np.max(sess.S[cell_random,t_idx_random:t_idx_random+len_trace])
                    ax.scatter(spk_times-t_idx_random, S_peakval[cell_random][spk_indeces]/max_val, s=80, facecolor='none', edgecolors='k')
                    ax.plot(sess.S[cell_random, t_idx_random:t_idx_random+len_trace]/max_val,'r')

                    x=input('good? (y/n) ')
                    if x == 'y':
                        not_satisfied = False
                    plt.close()
                selections[group].append((cell_random, t_idx_random))
    else:
        # Better have passed selections then...
        if not selections:
            raise Exception('*** Error: selections_mode set to False but selections not provided')
        print('*** Skipping selections mode...')

    width=8
    height=3
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width,height)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    y_offset = 0
    height_scale = 2
    my_lw = 1.0
    marker_size=40
    group_colours = {'hM3D': my_colours['my_r'], 'hM4D': my_colours['my_b'], 'mCherry': my_colours['my_k']}
    for group, sel in selections.items():
        print('*** group {}'.format(group))
        m = mice_to_use[group]
        sess = session[m]
        C = sess.C
        S_spikes = sess.S_spikes
        S_peakval = sess.S_peakval

        c_group = group_colours[group]
        for (cell_random, t_idx_random) in sel:
            print('   cell {}, t_idx_random {} '.format(cell_random, t_idx_random), end='')
            max_val = np.max(C[cell_random,t_idx_random:t_idx_random+len_trace])
            print('[C..', end='')
            ax.plot(y_offset + (C[cell_random, t_idx_random:t_idx_random+len_trace]/(height_scale*max_val)), c=c_group, lw=my_lw, alpha=0.5)
            print('] ', end='')
            #ax.scatter(spk_times-t_idx_random, sess.S[cell_random][spk_times], s=80, facecolor='none', edgecolors='k')
            max_val = np.max(sess.S[cell_random,t_idx_random:t_idx_random+len_trace])
            spk_indeces = np.logical_and(sess.S_spikes[cell_random] >= t_idx_random, sess.S_spikes[cell_random] < t_idx_random+len_trace)
            spk_times = S_spikes[cell_random][spk_indeces]
            print('[scatter..', end='')
            ax.scatter(spk_times-t_idx_random, y_offset + (S_peakval[cell_random][spk_indeces]/(height_scale*max_val)), s=marker_size, facecolor='none', edgecolors='k')
            print('] [S..', end='')
            ax.plot(y_offset + (sess.S[cell_random, t_idx_random:t_idx_random+len_trace]/(height_scale*max_val)), c=c_group, lw=my_lw, alpha=1.0)        
            print(']')

            y_offset += 1/height_scale
    os.makedirs(os.path.join(PLOTS_DIR, 'plot_sample_traces'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_sample_traces', 'plot_sample_traces.png'), format='png', dpi=300)
    if paper_dir:
        path_name = os.path.join(paper_dir, 'plot_sample_traces.png')
        print('*** Plotting paper_dir {}'.format(path_name))
        plt.savefig(path_name, format='png', dpi=300)
    plt.close()
    return selections

def process_PSTH_shuffle(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim='shock', frames_lookaround=40, \
    frames_save=200, shaded='sem', num_shuffles=100, percentile=95.0, auto_close=True):
    '''
    'stim' should be strings corresponding to either 'tone' or 'shock'. Caller must make sure the session type matches
    the expected stim.

    Returns the cells that were plotted as a dict mapped from groups -> list of cell id's.

    This function calculates PSTH by averaging across all onset/trials.
    '''
    # PSTH_2 no averaging
    #snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    print('***PSTH_shuffle AVERAGED')
    snippet_len = frames_lookaround * 2
    save_post = frames_lookaround + frames_save
    save_len = frames_lookaround * 2 + frames_save
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    group_PSTH_all = dict() # PSTH for all responses not just significant ones    
    group_PSTH_vel = dict() # velocities for corresponding PSTH's
    group_PSTH_subtract = dict() # PSTH of sig cells with baseline of bouts subtracted to get stim response only
    group_percentiles = dict()

    tot_cells = dict()
    tot_cells['hM3D'] = dict()
    tot_cells['hM4D'] = dict()
    tot_cells['mCherry'] = dict()

    sig_cells = dict()
    sig_cells['hM3D'] = dict()
    sig_cells['hM4D'] = dict()
    sig_cells['mCherry'] = dict()

    frac_tots = dict()
    frac_tots['hM3D'] = dict()
    frac_tots['hM4D'] = dict()
    frac_tots['mCherry'] = dict()

    rng = default_rng()

    print('*** MAPPING: {}'.format(mapping_type))
    for group, mice in mice_per_group.items():

        print('\nin {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            #if mouse in ['G07', 'G14', 'G15', 'G20', 'G12']:
            #    continue
            print('{} {}'.format(mouse, group))
            s = session[mouse]
            crossreg = crossreg_mice[mouse]
            frac_tots[group][mouse] = []
            sig_cells[group][mouse] = []
            tot_cells[group][mouse] = []

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                #onsets = s.shock_onsets
                #offsets = s.shock_offsets
                onsets = []
                for i in range(len(s.shock_onsets)):
                    # Iterate through defined shock onsets but use def values (e.g. G09 only has 4 shocks)
                    onsets.append(s.shock_onsets_def[i]*MINISCOPE_FPS)
                onsets = s.shock_onsets

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                #C = s.C
                #C = s.S_orig
                C = s.S
                #C = s.S_imm
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
                C = s.S[indeces,:]
            tot_cells[group][mouse].append(C.shape[0])
            V = s.velocities_miniscope_smooth

            C_responses = np.zeros((C.shape[0], frames_lookaround*2))
            C_responses_save = np.zeros((C.shape[0], save_len))
            V_save = np.zeros(save_len)
            sig_cells_total_mouse = np.ndarray(0)

            #onsets_wanted = [onsets[0]]
            #onsets_wanted = [onsets[-1]]
            onsets_wanted = onsets
            for on,on_num in zip(onsets_wanted,range(len(onsets_wanted))):
                print('on {}'.format(on), end='')
                #on = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround))
                period = range(on - frames_lookaround, on + frames_lookaround)
                period_save = range(on - frames_lookaround, min(on + save_post, C.shape[1]))
                #C_responses = C_responses + C[:,period]
                C_responses = C_responses + C[:,period]
                if len(period_save) < C_responses_save.shape[1]: # for G09
                    C_responses_save = C_responses_save + np.pad(C[:,period_save], ((0,0),(0,C_responses_save.shape[1] - len(period_save))))
                    V_save = V_save + np.pad(V[period_save], (0, len(V_save) - len(period_save)))
                else:
                    C_responses_save = C_responses_save + C[:,period_save]
                    V_save = V_save + V[period_save]

            C_responses /= len(onsets_wanted)
            C_responses_save /= len(onsets_wanted)
            V_save /= len(onsets_wanted)

            post = np.mean(C_responses[:, range(frames_lookaround, len(period))],1)
            pre = np.mean(C_responses[:, range(0, frames_lookaround)],1)
            '''
            post_save = np.mean(C_responses_save[:, range(frames_lookaround, len(period))],1)
            pre_save = np.mean(C_responses_save[:, range(0, frames_lookaround)],1)            
            C_responses_save = post_save - pre_save
            C_responses = post - pre
            '''
            
            #C_binary = (post - pre) / (post + pre)
            
            C_binary = post - pre

            #C_binary = np.mean(C[:, range(on, on + frames_lookaround)],1) - \
            #    np.mean(C[:, range(on - frames_lookaround, on)],1)

            # Allocate matrix where we have the response-values for all of the shuffles (columns) for each cell (rows).
            C_shuffles = np.zeros((C.shape[0], len(period), num_shuffles))
            C_shuffles_binary = np.zeros((C.shape[0], num_shuffles))
            #C_period_values = np.zeros((C.shape[0], len(period) * 2, num_shuffles)) # for entire onset,offset period
            for i in range(num_shuffles):
                print('.'.format(on),end='')                
                shuffle_times = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround), num_shuffles)
                C_rolled = np.roll(C, shuffle_times, axis=0)

                '''
                C_rolled_responses = np.zeros((C_rolled.shape[0], frames_lookaround*2))
                for on in onsets:
                    period = range(on - frames_lookaround, on + frames_lookaround)
                    C_rolled_responses = np.add(C_rolled_responses, C_rolled[:,period])
                C_rolled_responses /= len(onsets)
                '''

                C_rolled_responses = np.zeros((C.shape[0], frames_lookaround*2))
                for on in onsets:
                    period = range(on - frames_lookaround, on + frames_lookaround)
                    #C_responses = C_responses + C[:,period]
                    C_rolled_responses = C_rolled_responses + C_rolled[:,period]   
                C_rolled_responses /= len(onsets_wanted)

                #C_rolled_responses = C_rolled[:, range(shuffle_times[i] - frames_lookaround, shuffle_times[i] + frames_lookaround)]
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
            frac_tots[group][mouse].append(len(sig_cells_mouse) / C.shape[0])
            #frac_tots[group].append(len(sig_cells_mouse))
            sig_cells[group][mouse].append(sig_cells_mouse)

            if group not in group_PSTH.keys():
                group_PSTH[group] = C_responses_save[sig_cells_mouse,:]
                group_PSTH_all[group] = C_responses_save
                group_PSTH_vel[group] = V_save
                group_percentiles[group] = dict()
            else:
                group_PSTH[group] = np.vstack((group_PSTH[group], C_responses_save[sig_cells_mouse,:]))
                group_PSTH_all[group] = np.vstack((group_PSTH_all[group], C_responses_save))
                group_PSTH_vel[group] = np.vstack((group_PSTH_vel[group], V_save))
            group_percentiles[group][mouse] = C_percentile
            print('')

            # Get random bouts
            '''
            plt.plot(s.velocities_miniscope_smooth,'r')
            plt.plot(np.diff(s.velocities_miniscope_smooth))
            '''
            V_diff = np.diff(s.velocities_miniscope_smooth)
            bout_onsets = np.where(V_diff >= 2.0)[0]
            C_bout = np.zeros((C.shape[0], frames_lookaround*2))
            C_bout_save = np.zeros((C.shape[0], save_len))
            for bout in bout_onsets:
                # Use same as PSTH
                period = range(bout - frames_lookaround, min(bout + frames_lookaround, C.shape[1]))
                period_save = range(bout - frames_lookaround, min(bout + save_post, C.shape[1]))
                
                if len(period) < C_bout.shape[1]: # for G09
                    C_bout = C_bout + np.pad(C[:,period], ((0,0),(0,C_bout.shape[1] - len(period))))
                else:
                    C_bout = C_bout + C[:,period]

                if len(period_save) < C_bout_save.shape[1]: # for G09
                    C_bout_save = C_bout_save + np.pad(C[:,period_save], ((0,0),(0,C_bout_save.shape[1] - len(period_save))))
                else:
                    C_bout_save = C_bout_save + C[:,period_save]

            C_bout /= len(bout_onsets)
            C_responses_save /= len(bout_onsets)

            #C_subtract_save = C_responses_save - C_bout_save
            C_subtract = np.copy(C_responses)
            C_subtract_save = np.copy(C_responses_save)
            for i in range(C_subtract.shape[0]):
                C_subtract[i,:] -= np.mean(C_bout[i,:])
                C_subtract[i,C_subtract[i,:]<0] = 0
                C_subtract_save[i,:] -= np.mean(C_bout[i,:])
                C_subtract_save[i,C_subtract_save[i,:]<0] = 0

            if group not in group_PSTH_subtract.keys():
                group_PSTH_subtract[group] = C_subtract[sig_cells_mouse,:]                
            else:
                group_PSTH_subtract[group] = np.vstack((group_PSTH_subtract[group], C_subtract[sig_cells_mouse,:]))

    #frac_tots[group].append(len(np.unique(sig_cells_total_mouse)) / C.shape[0])
    plt.figure()
    plt.plot(np.mean(group_PSTH['hM3D'],0))
    plt.title('hM3D')
    plt.figure()
    plt.plot(np.mean(group_PSTH['hM4D'],0))
    plt.title('hM4D')
    plt.figure()
    plt.plot(np.mean(group_PSTH['mCherry'],0))
    plt.title('mCherry')
    if auto_close:
        plt.close()

    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['hM3D'],0))
    plt.title('hM3D - velocities')
    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['hM4D'],0))
    plt.title('hM4D - velocities')
    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['mCherry'],0))
    plt.title('mCherry - velocities')
    if auto_close:
        plt.close()

    plt.figure()
    plt.plot(np.mean(group_PSTH_subtract['hM3D'],0))
    plt.title('hM3D - subtract')
    plt.figure()
    plt.plot(np.mean(group_PSTH_subtract['hM4D'],0))
    plt.title('hM4D - subtract')
    plt.figure()
    plt.plot(np.mean(group_PSTH_subtract['mCherry'],0))
    plt.title('mCherry - subtract')
    if auto_close:
        plt.close()

    print('tot cells hM3D ', [x for x in tot_cells['hM3D'].values()])
    print('tot cells hM4D ', [x for x in tot_cells['hM4D'].values()])
    print('tot cells mCherry ', [x for x in tot_cells['mCherry'].values()])
    print('frac tots hM3D ', [x for x in frac_tots['hM3D'].values()])
    print('frac tots hM4D ', [x for x in frac_tots['hM4D'].values()])
    print('frac tots mCherry ', [x for x in frac_tots['mCherry'].values()])
    print('frac tots hM3D mean ', [np.mean(x) for x in frac_tots['hM3D'].values()])
    print('frac tots hM4D mean ', [np.mean(x) for x in frac_tots['hM4D'].values()])
    print('frac tots mCherry mean ', [np.mean(x) for x in frac_tots['mCherry'].values()])
    print('sig cells hM3D ', [len(x[0]) for x in sig_cells['hM3D'].values()])
    print('sig cells hM4D ', [len(x[0]) for x in sig_cells['hM4D'].values()])
    print('sig_cells mCherry ', [len(x[0]) for x in sig_cells['mCherry'].values()])

    group_tots = dict()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        min_val = np.mean(np.min(group_PSTH[group],1))
        group_mean = np.mean(group_PSTH[group],0) - min_val
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH[group], 0) / np.sqrt(group_PSTH[group].shape[0]) - min_val
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH[group], 0) - min_val
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(snippet_len+frames_save), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
        ax.set_title('{}'.format(group), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH for {} for mapping {}'.format(stim, mapping_type))
    filename = 'PSTH_{}_{}_{}.png'.format(stim, mapping_type, frames_lookaround)
    dir_name = 'PSTH'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        min_val = np.mean(np.min(group_PSTH_all[group],1))
        group_mean = np.mean(group_PSTH_all[group],0) - min_val
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH_all[group], 0) / np.sqrt(group_PSTH_all[group].shape[0]) - min_val
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH_all[group], 0) - min_val
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(snippet_len+frames_save), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
        ax.set_title('{}'.format(group), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH (ALL) for {} for mapping {}'.format(stim, mapping_type))
    filename = 'PSTH_all_{}_{}_{}.png'.format(stim, mapping_type, frames_lookaround)
    dir_name = 'PSTH'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        min_val = np.mean(np.min(group_PSTH_subtract[group],1))
        group_mean = np.mean(group_PSTH_subtract[group],0) - min_val
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH_subtract[group], 0) / np.sqrt(group_PSTH_subtract[group].shape[0]) - min_val
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH_subtract[group], 0) - min_val
        ax.plot(group_mean, color='b', lw=1)
        #ax.fill_between(range(snippet_len+frames_save), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        ax.fill_between(range(snippet_len), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        ax.set_title('{}'.format(group), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH (SUBTRACT) for {} for mapping {}'.format(stim, mapping_type))
    filename = 'PSTH_subtract_{}_{}_{}.png'.format(stim, mapping_type, frames_lookaround)
    dir_name = 'PSTH'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        for mouse in mice_per_group[group]:
            ax.hist(group_percentiles[group][mouse], bins=30, color='skyblue', edgecolor='black')
        ax.set_title('{}'.format(group), size='medium')
    if auto_close:
        plt.close()

    for group in mice_per_group:
        plt.figure()
        plt.title('{} velocity-transients'.format(group))
        for m in mice_per_group[group]:
            print('{}..'.format(m),end='')
            s=session[m]
            if mapping_type=='full':
                C = s.S_mov
            V = s.velocities_miniscope_smooth
            V_sub = V[range(C.shape[1])]
            for i in range(C.shape[0]):
                #plot_range = C[i,range(C.shape[1])]>2
                plot_range = V[range(C.shape[1])]>2
                plt.scatter(V_sub[plot_range], C[i,plot_range], s=0.5, c='k')
                
    '''
    for group in ['hM3D', 'hM4D']:
        plt.figure()
        plt.title('{}-hM3D velocity-transients'.format(group))
        for m in mice_per_group[group]:
            print('{}..'.format(m),end='')
            s=session[m]
            if mapping_type=='full':
                C = s.S_mov
            V = s.velocities_miniscope_smooth
            V_sub = V[range(C.shape[1])]
            for i in range(C.shape[0]):
                plot_range = C[i,range(C.shape[1])]>2               
                plt.scatter(V_sub[plot_range], C[i,plot_range], s=0.5, c='k')
    '''

    return tot_cells, frac_tots, sig_cells, group_PSTH, group_percentiles, group_PSTH_vel
    #return nonzero_cells, frac_tots, trapz_cells, PSTH_cells, max_per_cell

def process_PSTH_shuffle_sep(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim='shock', frames_lookaround=40, \
    frames_save=200, shaded='sem', num_shuffles=100, percentile=95.0, auto_close=True):
    '''
    'stim' should be strings corresponding to either 'tone' or 'shock'. Caller must make sure the session type matches
    the expected stim.

    Returns the cells that were plotted as a dict mapped from groups -> list of cell id's.

    This function calculates PSTH by considering each onset period separately (as opposed to averaged).
    '''
    # PSTH_2 no averaging
    #snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    print('***PSTH_shuffle SEPARATE')
    snippet_len = frames_lookaround * 2
    save_post = frames_lookaround + frames_save
    save_len = frames_lookaround * 2 + frames_save
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    group_PSTH_all = dict() # PSTH for all responses not just significant ones    
    group_PSTH_vel = dict() # velocities for corresponding PSTH's
    group_percentiles = dict()

    tot_cells = dict()
    tot_cells['hM3D'] = dict()
    tot_cells['hM4D'] = dict()
    tot_cells['mCherry'] = dict()

    sig_cells = dict()
    sig_cells['hM3D'] = dict()
    sig_cells['hM4D'] = dict()
    sig_cells['mCherry'] = dict()

    frac_tots = dict()
    frac_tots['hM3D'] = dict()
    frac_tots['hM4D'] = dict()
    frac_tots['mCherry'] = dict()

    rng = default_rng()

    print('*** MAPPING: {}'.format(mapping_type))
    for group, mice in mice_per_group.items():

        print('\nin {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            #if mouse in ['G07', 'G14', 'G15', 'G20', 'G12']:
            #    continue
            print('{} {}'.format(mouse, group))
            s = session[mouse]
            crossreg = crossreg_mice[mouse]
            frac_tots[group][mouse] = []
            sig_cells[group][mouse] = []
            tot_cells[group][mouse] = []

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                #onsets = s.shock_onsets
                #offsets = s.shock_offsets
                onsets = s.shock_onsets_def.copy()
                for i in range(len(onsets)):
                    onsets[i] *= MINISCOPE_FPS
                onsets = s.shock_onsets

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                #C = s.C
                #C = s.S_orig
                C = s.S
                #C = s.S_imm
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
                C = s.S[indeces,:]
            tot_cells[group][mouse].append(C.shape[0])
            V = s.velocities_miniscope_smooth

            C_responses = np.zeros((C.shape[0], frames_lookaround*2))
            C_responses_save = np.zeros((C.shape[0], save_len))
            V_save = np.zeros(save_len)
            sig_cells_total_mouse = np.ndarray(0)

            #onsets_wanted = [onsets[0]]
            #onsets_wanted = [onsets[-1]]
            onsets_wanted = onsets
            for on,on_num in zip(onsets_wanted,range(len(onsets_wanted))):
                print('on {}'.format(on), end='')
                #on = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround))
                period = range(on - frames_lookaround, on + frames_lookaround)
                period_save = range(on - frames_lookaround, on + save_post)
                #C_responses = C_responses + C[:,period]
                C_responses = C_responses + C[:,period]
                C_responses_save = C_responses_save + C[:,period_save]
                V_save = V_save + V[period_save]

                post = np.mean(C_responses[:, range(frames_lookaround, len(period))],1)
                pre = np.mean(C_responses[:, range(0, frames_lookaround)],1)
                '''
                post_save = np.mean(C_responses_save[:, range(frames_lookaround, len(period))],1)
                pre_save = np.mean(C_responses_save[:, range(0, frames_lookaround)],1)            
                C_responses_save = post_save - pre_save
                C_responses = post - pre
                '''
                #C_binary = (post - pre) / (post + pre)
                C_binary = post - pre

                #C_binary = np.mean(C[:, range(on, on + frames_lookaround)],1) - \
                #    np.mean(C[:, range(on - frames_lookaround, on)],1)

                # Allocate matrix where we have the response-values for all of the shuffles (columns) for each cell (rows).
                C_shuffles = np.zeros((C.shape[0], len(period), num_shuffles))
                C_shuffles_binary = np.zeros((C.shape[0], num_shuffles))
                #C_period_values = np.zeros((C.shape[0], len(period) * 2, num_shuffles)) # for entire onset,offset period
                for i in range(num_shuffles):
                    print('.'.format(on),end='')
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
                frac_tots[group][mouse].append(len(sig_cells_mouse) / C.shape[0])
                #frac_tots[group].append(len(sig_cells_mouse))
                sig_cells[group][mouse].append(sig_cells_mouse)

                if group not in group_PSTH.keys():
                    group_PSTH[group] = C_responses_save[sig_cells_mouse,:]
                    group_PSTH_all[group] = C_responses_save
                    group_PSTH_vel[group] = V_save
                    group_percentiles[group] = dict()                    
                else:
                    group_PSTH[group] = np.vstack((group_PSTH[group], C_responses_save[sig_cells_mouse,:]))
                    group_PSTH_all[group] = np.vstack((group_PSTH_all[group], C_responses_save))
                    group_PSTH_vel[group] = np.vstack((group_PSTH_vel[group], V_save))  
                group_percentiles[group][mouse] = C_percentile
                print('')

        #frac_tots[group].append(len(np.unique(sig_cells_total_mouse)) / C.shape[0])
    plt.figure()
    plt.plot(np.mean(group_PSTH['hM3D'],0))
    plt.title('hM3D')
    plt.figure()
    plt.plot(np.mean(group_PSTH['hM4D'],0))
    plt.title('hM4D')
    plt.figure()
    plt.plot(np.mean(group_PSTH['mCherry'],0))
    plt.title('mCherry')

    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['hM3D'],0))
    plt.title('hM3D - velocities')
    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['hM4D'],0))
    plt.title('hM4D - velocities')
    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['mCherry'],0))
    plt.title('mCherry velocities')

    print('tot cells hM3D ', [x for x in tot_cells['hM3D'].values()])
    print('tot cells hM4D ', [x for x in tot_cells['hM4D'].values()])
    print('tot cells mCherry ', [x for x in tot_cells['mCherry'].values()])
    print('frac tots hM3D ', [x for x in frac_tots['hM3D'].values()])
    print('frac tots hM4D ', [x for x in frac_tots['hM4D'].values()])
    print('frac tots mCherry ', [x for x in frac_tots['mCherry'].values()])
    print('frac tots hM3D mean ', [np.mean(x) for x in frac_tots['hM3D'].values()])
    print('frac tots hM4D mean ', [np.mean(x) for x in frac_tots['hM4D'].values()])
    print('frac tots mCherry mean ', [np.mean(x) for x in frac_tots['mCherry'].values()])
    print('sig cells hM3D ', [len(x) for x in sig_cells['hM3D'].values()])
    print('sig cells hM4D ', [len(x) for x in sig_cells['hM4D'].values()])
    print('sig_cells mCherry ', [len(x) for x in sig_cells['mCherry'].values()])

    group_tots = dict()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        min_val = np.mean(np.min(group_PSTH[group],1))
        group_mean = np.mean(group_PSTH[group],0) - min_val
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH[group], 0) / np.sqrt(group_PSTH[group].shape[0]) - min_val
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH[group], 0) - min_val
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(snippet_len+frames_save), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
        ax.set_title('{}'.format(group), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH for {} for mapping {}'.format(stim, mapping_type))
    filename = 'PSTH_{}_{}_{}.png'.format(stim, mapping_type, frames_lookaround)
    dir_name = 'PSTH'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        min_val = np.mean(np.min(group_PSTH[group],1))
        group_mean = np.mean(group_PSTH_all[group],0) - min_val
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH_all[group], 0) / np.sqrt(group_PSTH_all[group].shape[0]) - min_val
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH_all[group], 0) - min_val
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(snippet_len+frames_save), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
        ax.set_title('{}'.format(group), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH (ALL) for {} for mapping {}'.format(stim, mapping_type))
    filename = 'PSTH_all_{}_{}_{}.png'.format(stim, mapping_type, frames_lookaround)
    dir_name = 'PSTH'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True, sharex=True)
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        for mouse in mice_per_group[group]:
            ax.hist(group_percentiles[group][mouse], bins=30, color='skyblue', edgecolor='black')
        ax.set_title('{}'.format(group), size='medium')

    return tot_cells, frac_tots, sig_cells
    #return nonzero_cells, frac_tots, trapz_cells, PSTH_cells, max_per_cell

def process_PSTH_simple(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim, frames_lookaround=20, \
    frames_save=300, normalize=False, binary_activity=False, binary_thresh=0, binary_flip=False, shaded='sem'):
    '''
    'stim' should be strings corresponding to either 'tone' or 'shock'. Caller must make sure the session type matches
    the expected stim.

    Returns the cells that were plotted as a dict mapped from groups -> list of cell id's.
    '''

    #frames_lookaround = 20 # 20 frames before and after tone/shock onsets

    snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    session_group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    session_group_PSTH['hM3D'] = np.zeros(snippet_len)
    session_group_PSTH['hM4D'] = np.zeros(snippet_len)
    session_group_PSTH['mCherry'] = np.zeros(snippet_len)

    PSTH_cells = dict()
    PSTH_cells['hM3D'] = []
    PSTH_cells['hM4D'] = []
    PSTH_cells['mCherry'] = []

    inserted_PSTH = dict()
    inserted_PSTH['hM3D'] = False
    inserted_PSTH['hM4D'] = False
    inserted_PSTH['mCherry'] = False

    C_binary_vals = []
    C_binary_vals_all = []

    nonzero_cells = dict()
    nonzero_cells['hM3D'] = dict()
    nonzero_cells['hM4D'] = dict()
    nonzero_cells['mCherry'] = dict()
    nonzero_cell_count = 0

    trapz_cells = dict()
    tot_cells = dict()

    max_val = 0
    min_val = 9999

    max_per_cell = dict()
    max_per_cell['hM3D'] = []
    max_per_cell['hM4D'] = []
    max_per_cell['mCherry'] = []

    session_group_PSTH_per_mice = dict()
    for group, mice in mice_per_group.items():
        session_group_PSTH_per_mice[group] = dict()
        trapz_cells[group] = []

        print('in {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            s = session[mouse]
            crossreg = crossreg_mice[mouse]
            nonzero_cells[group][mouse] = [] # set()

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
                #C = s.C_zarr['C'][indeces,:]
                C = s.C[indeces,:]
            tot_cells[mouse] = C.shape[0]

            nonzero_cells[group][mouse] = dict()
            C_mouse = np.zeros(snippet_len)
            for on, off in zip(onsets, offsets):
                if on not in nonzero_cells[group][mouse]:
                    nonzero_cells[group][mouse][on] = []
                period = range(on - frames_lookaround, on + frames_lookaround)
                save_period = range(on - frames_lookaround, on + frames_save)
                if on + frames_save > C.shape[1]:
                    continue
                for cell in range(C.shape[0]):
                    if np.sum(C[cell,period]) > 0:
                        if binary_activity:
                            C_binary = np.mean(C[cell, range(on, on + frames_lookaround)]) - \
                                np.mean(C[cell, range(on - frames_lookaround, on)])
                            C_binary_vals_all.append(C_binary)
                            #print(C_binary)
                            if binary_flip:
                                if C_binary >= binary_thresh:
                                    continue
                            else:
                                if C_binary <= binary_thresh:
                                    #print('mouse {} group {} cell {} had negative binary activity function'.format(m, group, cell))
                                    continue
                            C_binary_vals.append(C_binary)
                        if normalize:
                            #print(np.sum(C[cell,period]))
                            C_norm = C[cell, save_period]# C[cell, period]
                            #print(C[cell,period])
                            #C_norm = C_norm - np.mean(C[cell, range(on - frames_lookaround,on)])
                            C_norm = C_norm - C[cell, on-frames_lookaround]
                            if np.isnan(C_norm).any():
                                print('mouse {} group {} cell {} had nan in C_norm'.format(m, group, cell))
                                continue
                            #print(C_norm.shape, C_mouse.shape)
                            C_mouse = np.add(C_mouse, C_norm)
                            if not inserted_PSTH[group]:
                                PSTH_cells[group] = C_norm
                                inserted_PSTH[group] = True
                            else:
                                PSTH_cells[group] = np.vstack((PSTH_cells[group], C_norm))
                            max_per_cell[group].append(np.max(C_norm))

                            # Since normalization was done for small period prior to onset, depending on length of save_period, 
                            # if the cell goes back to negative values afterwards, the integral will be negative, so let's bump up all
                            # y-values of the integral by the minimum if it happens to be negative. This only affects the integrals
                            # and the integrals are based on the shape, so it doesn't change the result, only makes it correctly
                            # calculated.
                            if np.min(C_norm) < 0:
                                trapz_cells[group].append(np.trapz(C_norm - np.min(C_norm)) / np.max(C_norm))
                            else:
                                trapz_cells[group].append(np.trapz(C_norm) / np.max(C_norm))
                            #print(C_mouse)
                        else:
                            C_mouse = np.add(C_mouse, C[cell,save_period])#period])
                            if np.min(C[cell,save_period]) < 0:
                                trapz_cells[group].append(np.trapz(C[cell,save_period] - np.min(C[cell,save_period])) / np.max(C[cell,save_period]))
                            else:
                                trapz_cells[group].append(np.trapz(C[cell,save_period]) / np.max(C[cell,save_period]))
                            if not inserted_PSTH[group]:
                                PSTH_cells[group] = C[cell, save_period]
                                inserted_PSTH[group] = True
                            else:
                                PSTH_cells[group] = np.vstack((PSTH_cells[group], C[cell,save_period]))#period]))
                            max_per_cell[group].append(np.max(C[cell,save_period]))
                        nonzero_cell_count += 1
                        nonzero_cells[group][mouse][on].append(cell) #update((cell,))
                C_mouse = C_mouse / nonzero_cell_count
            C_mouse = C_mouse / len(onsets)
            
            max_mouse = np.max(C_mouse)
            min_mouse = np.max(C_mouse)
            if max_mouse > max_val:
                max_val = max_mouse
            if min_mouse < min_val:
                min_val = min_mouse

            session_group_PSTH_per_mice[group][mouse] = C_mouse
            #print(C_mouse)

            session_group_PSTH[group] = np.add(session_group_PSTH[group], C_mouse)
            #print(session_group_PSTH[group])
        session_group_PSTH[group] = session_group_PSTH[group] / len(session_group_PSTH[group])

    range_val = max_val - min_val

    group_tots = dict()
    frac_tots = dict()
    frac_tots['hM3D'] = []
    frac_tots['hM4D'] = []
    frac_tots['mCherry'] = []
    for group in nonzero_cells.keys():
        group_tots[group] = 0
        for mouse, onsets in nonzero_cells[group].items():
            for on, cells in onsets.items():
                group_tots[group] += len(cells) 
                frac_tots[group].append(len(cells) / tot_cells[mouse]) 

    fig, axs = plt.subplots(1, 3, figsize=(9,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        group_mean = np.mean(PSTH_cells[group],0)
        if shaded == 'sem':
            group_shaded = np.std(PSTH_cells[group], 0) / np.sqrt(PSTH_cells[group].shape[0])
        if shaded == 'sd':
            group_shaded = np.std(PSTH_cells[group], 0)
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(snippet_len), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
        ax.set_title('{} {}'.format(group, group_tots[group]), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH for {} for mapping {} ({} total cells)'.format(stim, mapping_type, nonzero_cell_count))
    filename = 'PSTH_{}_{}_{}_norm{}_binary{}.png'.format(stim, mapping_type, frames_lookaround, normalize, binary_activity)
    if binary_flip:
        dir_name = 'PSTH_flip'
    else:
        dir_name = 'PSTH'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    plt.close()

    if C_binary_vals:
        plt.figure()
        plt.hist(C_binary_vals, color='b', density=False, alpha=0.3, bins=100)
        plt.axvline(binary_thresh, c='r', ls='-')
        plt.suptitle('Binary activities for {} for mapping {}'.format(stim, mapping_type))
        filename = 'PSTH_{}_{}_{}_norm{}_binary{}_hist.png'.format(stim, mapping_type, frames_lookaround, normalize, binary_activity)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        plt.close()

        plt.figure()
        plt.hist(C_binary_vals_all, color='b', density=False, alpha=0.3, bins=100, range=(-250,250))
        plt.axvline(binary_thresh, c='r', ls='-')
        plt.suptitle('Binary activities for {} for mapping {}'.format(stim, mapping_type))
        filename = 'PSTH_{}_{}_{}_norm{}_binary{}_hist_all.png'.format(stim, mapping_type, frames_lookaround, normalize, binary_activity)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        plt.close()

    #print('*** PSTH: max_val is {}'.format(mav_val))
    #for group in PSTH_cells.keys():
    #    trapz_cells[group] = np.trapz(PSTH_cells[group]/max_val, axis=1)

    return nonzero_cells, frac_tots, trapz_cells, PSTH_cells, max_per_cell

def process_PSTH_simple_S(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim, frames_lookaround=20, \
    frames_save=300, normalize=False, binary_activity=False, binary_thresh=0, binary_flip=False, shaded='sem'):
    '''
    'stim' should be strings corresponding to either 'tone' or 'shock'. Caller must make sure the session type matches
    the expected stim.

    Returns the cells that were plotted as a dict mapped from groups -> list of cell id's.
    '''

    #frames_lookaround = 20 # 20 frames before and after tone/shock onsets

    snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    session_group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    session_group_PSTH['hM3D'] = np.zeros(snippet_len)
    session_group_PSTH['hM4D'] = np.zeros(snippet_len)
    session_group_PSTH['mCherry'] = np.zeros(snippet_len)

    PSTH_cells = dict()
    PSTH_cells['hM3D'] = []
    PSTH_cells['hM4D'] = []
    PSTH_cells['mCherry'] = []

    inserted_PSTH = dict()
    inserted_PSTH['hM3D'] = False
    inserted_PSTH['hM4D'] = False
    inserted_PSTH['mCherry'] = False

    S_binary_vals = []
    S_binary_vals_all = []

    nonzero_cells = dict()
    nonzero_cells['hM3D'] = dict()
    nonzero_cells['hM4D'] = dict()
    nonzero_cells['mCherry'] = dict()
    nonzero_cell_count = 0

    trapz_cells = dict()
    tot_cells = dict()

    max_val = 0
    min_val = 9999

    max_per_cell = dict()
    max_per_cell['hM3D'] = []
    max_per_cell['hM4D'] = []
    max_per_cell['mCherry'] = []

    session_group_PSTH_per_mice = dict()
    for group, mice in mice_per_group.items():
        session_group_PSTH_per_mice[group] = dict()
        trapz_cells[group] = []

        print('in {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            s = session[mouse]
            crossreg = crossreg_mice[mouse]
            nonzero_cells[group][mouse] = [] # set()

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                onsets = s.shock_onsets
                offsets = s.shock_offsets

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                S = s.S
            else:
                indeces = []
                df_mapping = crossreg.get_mappings_cells(mapping_type=mapping_type)
                s_df_col = s.get_df_col()
                for i in range(len(df_mapping)):
                    cell_s = int(float(df_mapping[s_df_col].iloc[i]))
                    try:
                        #idx = np.where(s.C_zarr['unit_id']==cell_s)[0][0]
                        idx = np.where(s.S_idx==cell_s)[0][0]
                        indeces.append(idx)
                    except IndexError as error:
                        # sometimes the minian-saved zarr files don't contain the cells from the crossreg mapping;
                        # so just skip this row
                        print('***WARNING: could not find {} ({})'.format(cell_s, s.session_type))
                        continue
                #C = s.C_zarr['C'][indeces,:]
                S = s.S[indeces,:]
            tot_cells[mouse] = S.shape[0]

            nonzero_cells[group][mouse] = dict()
            S_mouse = np.zeros(snippet_len)
            for on, off in zip(onsets, offsets):
                if on not in nonzero_cells[group][mouse]:
                    nonzero_cells[group][mouse][on] = []
                period = range(on - frames_lookaround, on + frames_lookaround)
                save_period = range(on - frames_lookaround, on + frames_save)
                if on + frames_save > S.shape[1]:
                    continue
                for cell in range(S.shape[0]):
                    if np.sum(S[cell,period]) > 0:
                        if binary_activity:
                            S_binary = np.mean(S[cell, range(on, on + frames_lookaround)]) - \
                                np.mean(S[cell, range(on - frames_lookaround, on)])
                            S_binary_vals_all.append(S_binary)
                            #print(C_binary)
                            if binary_flip:
                                if S_binary >= binary_thresh:
                                    continue
                            else:
                                if S_binary <= binary_thresh:
                                    #print('mouse {} group {} cell {} had negative binary activity function'.format(m, group, cell))
                                    continue
                            S_binary_vals.append(S_binary)
                        if normalize:
                            #print(np.sum(C[cell,period]))
                            S_norm = S[cell, save_period]# C[cell, period]
                            #print(C[cell,period])
                            #C_norm = C_norm - np.mean(C[cell, range(on - frames_lookaround,on)])
                            S_norm = S_norm - S[cell, on-frames_lookaround]
                            if np.isnan(S_norm).any():
                                print('mouse {} group {} cell {} had nan in S_norm'.format(m, group, cell))
                                continue
                            #print(C_norm.shape, C_mouse.shape)
                            S_mouse = np.add(S_mouse, S_norm)
                            if not inserted_PSTH[group]:
                                PSTH_cells[group] = S_norm
                                inserted_PSTH[group] = True
                            else:
                                PSTH_cells[group] = np.vstack((PSTH_cells[group], S_norm))
                            max_per_cell[group].append(np.max(S_norm))

                            # Since normalization was done for small period prior to onset, depending on length of save_period, 
                            # if the cell goes back to negative values afterwards, the integral will be negative, so let's bump up all
                            # y-values of the integral by the minimum if it happens to be negative. This only affects the integrals
                            # and the integrals are based on the shape, so it doesn't change the result, only makes it correctly
                            # calculated.
                            if np.min(S_norm) < 0:
                                trapz_cells[group].append(np.trapz(S_norm - np.min(S_norm)) / np.max(S_norm))
                            else:
                                trapz_cells[group].append(np.trapz(S_norm) / np.max(S_norm))
                            #print(C_mouse)
                        else:
                            S_mouse = np.add(S_mouse, S[cell,save_period])#period])
                            if np.min(S[cell,save_period]) < 0:
                                trapz_cells[group].append(np.trapz(S[cell,save_period] - np.min(S[cell,save_period])) / np.max(S[cell,save_period]))
                            else:
                                trapz_cells[group].append(np.trapz(S[cell,save_period]) / np.max(S[cell,save_period]))
                            if not inserted_PSTH[group]:
                                PSTH_cells[group] = S[cell, save_period]
                                inserted_PSTH[group] = True
                            else:
                                PSTH_cells[group] = np.vstack((PSTH_cells[group], S[cell,save_period]))#period]))
                            max_per_cell[group].append(np.max(S[cell,save_period]))
                        nonzero_cell_count += 1
                        nonzero_cells[group][mouse][on].append(cell) #update((cell,))
                S_mouse = S_mouse / nonzero_cell_count
            S_mouse = S_mouse / len(onsets)
            
            max_mouse = np.max(S_mouse)
            min_mouse = np.max(S_mouse)
            if max_mouse > max_val:
                max_val = max_mouse
            if min_mouse < min_val:
                min_val = min_mouse

            session_group_PSTH_per_mice[group][mouse] = S_mouse
            #print(C_mouse)

            session_group_PSTH[group] = np.add(session_group_PSTH[group], S_mouse)
            #print(session_group_PSTH[group])
        session_group_PSTH[group] = session_group_PSTH[group] / len(session_group_PSTH[group])

    range_val = max_val - min_val

    group_tots = dict()
    frac_tots = dict()
    frac_tots['hM3D'] = []
    frac_tots['hM4D'] = []
    frac_tots['mCherry'] = []
    for group in nonzero_cells.keys():
        group_tots[group] = 0
        for mouse, onsets in nonzero_cells[group].items():
            for on, cells in onsets.items():
                group_tots[group] += len(cells) 
                frac_tots[group].append(len(cells) / tot_cells[mouse]) 

    fig, axs = plt.subplots(1, 3, figsize=(9,6), sharey=True, sharex=True)
    group_colours = {'hM3D':'r', 'hM4D':'b', 'mCherry':'k'}
    ylabel_set = False
    for group, ax in zip(mice_per_group.keys(), axs.flat):
        #ax.plot(session_group_PSTH[group], color='b', lw=1)
        group_mean = np.mean(PSTH_cells[group],0)
        if shaded == 'sem':
            group_shaded = np.std(PSTH_cells[group], 0) / np.sqrt(PSTH_cells[group].shape[0])
        if shaded == 'sd':
            group_shaded = np.std(PSTH_cells[group], 0)
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(snippet_len), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        #ax.plot([frames_lookaround, frames_lookaround],[range_val*0.75, range_val*0.9],c='r',ls='-')
        ax.set_title('{} {}'.format(group, group_tots[group]), size='medium')
        ax.set_xticks([0,100,200,300])
        ax.set_xticklabels([0, 5, 10, 15])
        ax.set_xlabel('Time (s)')
        if not ylabel_set:
            ax.set_ylabel('$\Delta$F/F (arbitrary units)')
            ylabel_set = True
    plt.suptitle('PSTH for {} for mapping {} S ({} total cells)'.format(stim, mapping_type, nonzero_cell_count))
    filename = 'PSTH_{}_{}_{}_norm{}_binary{}_S.png'.format(stim, mapping_type, frames_lookaround, normalize, binary_activity)
    if binary_flip:
        dir_name = 'PSTH_flip_S'
    else:
        dir_name = 'PSTH_S'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    plt.close()

    if S_binary_vals:
        plt.figure()
        plt.hist(S_binary_vals, color='b', density=False, alpha=0.3, bins=100)
        plt.axvline(binary_thresh, c='r', ls='-')
        plt.suptitle('Binary activities for {} for mapping {} S'.format(stim, mapping_type))
        filename = 'PSTH_{}_{}_{}_norm{}_binary{}_S_hist.png'.format(stim, mapping_type, frames_lookaround, normalize, binary_activity)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        plt.close()

        plt.figure()
        plt.hist(S_binary_vals_all, color='b', density=False, alpha=0.3, bins=100, range=(-250,250))
        plt.axvline(binary_thresh, c='r', ls='-')
        plt.suptitle('Binary activities for {} for mapping {} S'.format(stim, mapping_type))
        filename = 'PSTH_{}_{}_{}_norm{}_binary{}_S_hist_all.png'.format(stim, mapping_type, frames_lookaround, normalize, binary_activity)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        plt.close()

    #print('*** PSTH: max_val is {}'.format(mav_val))
    #for group in PSTH_cells.keys():
    #    trapz_cells[group] = np.trapz(PSTH_cells[group]/max_val, axis=1)

    return nonzero_cells, frac_tots, trapz_cells, PSTH_cells, max_per_cell

def plot_PSTH_overlay(PLOTS_DIR, session, nonzero_active_cells, nonzero_suppr_cells, stim, mapping):
    for m, s in session.items():
        print('*** getting {} ROI'.format(m))
        s.get_A_matrix()

    for group in nonzero_active_cells[stim].keys():
        for mouse, onsets in nonzero_active_cells[stim][group].items():
            tot_active_cells = []
            tot_suppr_cells = []
            for on, active_cells in onsets.items():
                suppr_cells = nonzero_suppr_cells[stim][group][mouse][on]
                tot_active_cells = np.append(tot_active_cells, active_cells)
                tot_suppr_cells = np.append(tot_suppr_cells, suppr_cells)
            A_active = session[mouse].A[tot_active_cells.astype(int)]
            A_suppr = session[mouse].A[tot_suppr_cells.astype(int)]

            plt.figure()
            plt.imshow(np.sum(A_active,0), cmap='Reds', alpha=0.7)
            plt.imshow(np.sum(A_suppr,0), cmap='Blues', alpha=0.7)
            plt.title('{} {} ROIs {}'.format(group, mouse, stim))

            save_path = os.path.join(PLOTS_DIR, 'PSTH_ROIs_{}'.format(stim))
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'PSTH_ROI_{}_{}_{}_{}.png'.format(group, mouse, stim, mapping)), format='png', dpi=300)
            plt.close()

def plot_PSTH_activities(PLOTS_DIR, frac_tots, stim, data_type_str, mapping):
    num_groups = len(frac_tots)
    x = range(num_groups)
    means = np.zeros(num_groups)
    sems = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))
    for group, idx in zip(['hM3D', 'hM4D', 'mCherry'], range(3)):
        means[idx] = np.mean(frac_tots[group]) * 100
        #stds[idx] = np.std(frac_tots[group])
        sems[idx] = (np.std(frac_tots[group]) * 100) / np.sqrt(len(frac_tots[group]))
        errbars[1,idx] = sems[idx]
    fig, ax = plt.subplots(figsize=(3,6))
    ax.bar(x, means, yerr=errbars, color=group_colours.values())
    print('*** anova prep: {} #hM3D {} #hM4D {} #mCherry'.format(len(frac_tots['hM3D']), len(frac_tots['hM4D']), len(frac_tots['mCherry'])))
    do_anova1_plot(frac_tots['hM3D'], frac_tots['hM4D'], frac_tots['mCherry'], ax, means+sems)
    ax.set_xticks(range(3))
    ax.set_ylabel('% CA1 PCs activated by {}'.format(stim), size='large')
    ax.set_xticklabels(['Exc', 'Inh', 'Ctl'], size='large')
    plt.suptitle('{} cells ({})'.format(data_type_str, stim), size='large')
    plt.subplots_adjust(left=0.21, bottom=0.09, right=0.90, top=0.90, wspace=0.20, hspace=0.20)
    os.makedirs(os.path.join(PLOTS_DIR, 'PSTH_activities'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'PSTH_activities', 'PSTH_{}_{}_{}.png'.format(data_type_str, stim, mapping)), format='png', dpi=300)
    plt.close()

def plot_PSTH_intensities(PLOTS_DIR, trapz_cells, stim, mapping):

    num_groups = len(trapz_cells)
    x = range(num_groups)
    means = np.zeros(num_groups)
    sems = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))
    for group, idx in zip(['hM3D', 'hM4D', 'mCherry'], range(3)):
        means[idx] = np.mean(trapz_cells[group])
        #stds[idx] = np.std(frac_tots[group])
        sems[idx] = np.std(trapz_cells[group]) / np.sqrt(len(trapz_cells[group]))
        errbars[1,idx] = sems[idx]
    fig, ax = plt.subplots(figsize=(3,6))
    ax.bar(x, means, yerr=errbars, color=group_colours.values())
    print('*** anova prep: {} #hM3D {} #hM4D {} #mCherry'.format(len(trapz_cells['hM3D']), len(trapz_cells['hM4D']), len(trapz_cells['mCherry'])))
    do_anova1_plot(trapz_cells['hM3D'], trapz_cells['hM4D'], trapz_cells['mCherry'], ax, means+sems)
    ax.set_xticks(range(3))
    ax.set_ylabel(stim.capitalize()+r'-evoked CA1 PC intensity (% of max $\Delta$F/F)')
    ax.set_xticklabels(['Exc', 'Inh', 'Ctl'], size='medium')
    plt.suptitle(stim.capitalize()+'-evoked response strength')
    plt.subplots_adjust(left=0.19, bottom=0.09, right=0.90, top=0.90, wspace=0.20, hspace=0.20)
    os.makedirs(os.path.join(PLOTS_DIR, 'PSTH_intensities'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'PSTH_intensities', 'PSTH_intensity_{}_{}.png'.format(stim, mapping)), format='png', dpi=300)
    plt.close()

def plot_PSTH_peaks(PLOTS_DIR, PSTH_cells, max_per_cell, stim, mapping):
    
    data = dict()
    groups = ['hM3D', 'hM4D', 'mCherry']
    for group in groups:
        data[group] = np.max(PSTH_cells[group],axis=1)
    num_groups = len(PSTH_cells)
    x = range(num_groups)
    means = np.zeros(num_groups)
    sems = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))
    for group, idx in zip(groups, range(3)):
        means[idx] = np.mean(data[group])
        #stds[idx] = np.std(frac_tots[group])
        sems[idx] = np.std(data[group]) / np.sqrt(len(PSTH_cells[group]))
        errbars[1,idx] = sems[idx]
    fig, ax = plt.subplots(figsize=(3,6))
    ax.bar(x, means, yerr=errbars, color=group_colours.values())
    print('*** anova prep: {} #hM3D {} #hM4D {} #mCherry'.format(len(data['hM3D']), len(data['hM4D']), len(data['mCherry'])))
    do_anova1_plot(data['hM3D'], data['hM4D'], data['mCherry'], ax, means+sems)
    ax.set_xticks(range(3))
    ax.set_ylabel(r'$\Delta$F/F (arbitrary units)')
    ax.set_xticklabels(['Exc', 'Inh', 'Ctl'], size='medium')
    plt.suptitle('Peak '+stim.capitalize()+'-evoked response')
    plt.subplots_adjust(left=0.22, bottom=0.09, right=0.90, top=0.90, wspace=0.20, hspace=0.20)
    os.makedirs(os.path.join(PLOTS_DIR, 'PSTH_peaks'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'PSTH_peaks', 'PSTH_peak_{}_{}.png'.format(stim, mapping)), format='png', dpi=300)
    plt.close()

def plot_location_map(PLOTS_DIR, mice_per_group, session, session_str):
    os.makedirs(os.path.join(PLOTS_DIR, '{}_location_maps'.format(session_str)), exist_ok=True)
    for m, sess in session.items():
        loc_X = sess.loc_X_miniscope_smooth
        loc_Y = sess.loc_Y_miniscope_smooth

        #fig, axs = plt.subplots(2,1,figsize=(6,3))

        #axs[0].plot(loc_X)
        #axs[1].plot(loc_Y)
        plt.figure(figsize=(6,6))
        plt.subplot(4,2,(1,2))
        plt.plot(loc_X, lw=0.5)
        plt.subplot(4,2,(3,4))
        plt.plot(loc_Y,lw=0.5)
        plt.subplot(4,2,(5,8))
        plt.plot(loc_X, loc_Y, lw=0.5)
        plt.subplot_tool()
        plt.savefig(os.path.join(PLOTS_DIR, '{}_location_maps'.format(session_str), '{}_{}_location-map.png'.format(session_str, m)), format='png', dpi=300)
        plt.close()

def plot_fluorescence_map_plotter(fluorescence_map, cells, random_width, SAVE_PATH, mouse, mouse_groups, session_str, bin_width, condition, S=None, want_3D=False):
    num_cells = len(cells)
    fig, axs = plt.subplots(random_width,random_width, figsize=(2*random_width,2*random_width))
    if want_3D:    
        fig_3D, axs_3D = plt.subplots(random_width,random_width, figsize=(2*random_width,2*random_width), subplot_kw={"projection": "3d"})

    if S is not None:
        fig_S, axs_S = plt.subplots(random_width,random_width, figsize=(4*random_width,2*random_width))

    for ax, cell, i in zip(axs.flat, cells, range(num_cells)):
        ax.imshow(fluorescence_map[:,:,i], cmap='hot')#, interpolation='nearest')
        ax.set_title('cell {}'.format(cell))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    #fig.suptitle('Mouse {} ({}) - {}, {}'.format(mouse, mouse_groups[mouse], session_str, condition))
    fig.suptitle('Mouse {} ({})'.format(mouse, mouse_groups[mouse]))

    #plt.subplot_tool(targetfig=fig)
    fig.subplots_adjust(top=0.93)
    fig.savefig(os.path.join(SAVE_PATH, '{}_{}_{}_{}_bin_width_{}.png'.format(condition, mouse, mouse_groups[mouse], session_str, bin_width)), format='png', dpi=300)
    plt.close(fig)

    if want_3D:
        for ax, cell, i in zip(axs_3D.flat, cells, range(num_cells)):
            X = np.arange(0, fluorescence_map.shape[0])
            Y = np.arange(0, fluorescence_map.shape[1])
            Z = fluorescence_map[:,:,i]
            X, Y = np.meshgrid(X, Y)
            ax.plot_surface(X, Y, np.transpose(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_title('cell {}'.format(cell))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig_3D.tight_layout()
        #fig.suptitle('Mouse {} ({}) - {}, {}'.format(mouse, mouse_groups[mouse], session_str, condition))
        fig_3D.suptitle('Mouse {} ({})'.format(mouse, mouse_groups[mouse]))

        #plt.subplot_tool(targetfig=fig)
        #fig_3D.subplots_adjust(top=0.93)
        fig_3D.savefig(os.path.join(SAVE_PATH, '{}_{}_{}_{}_bin_width_{}_3D.png'.format(condition, mouse, mouse_groups[mouse], session_str, bin_width)), format='png', dpi=300)
        plt.close(fig_3D)

    if S is not None:
        for ax_S, cell, i in zip(axs_S.flat, cells, range(num_cells)):
            ax_S.plot(S[cell,:])
            ax_S.set_title('cell {}'.format(cell))
        fig_S.tight_layout()
        #fig.suptitle('Mouse {} ({}) - {}, {}'.format(mouse, mouse_groups[mouse], session_str, condition))
        fig_S.suptitle('Mouse {} ({})'.format(mouse, mouse_groups[mouse]))

        #plt.subplot_tool(targetfig=fig_S)
        fig_S.subplots_adjust(top=0.93)
        fig_S.savefig(os.path.join(SAVE_PATH, '{}_{}_{}_{}_bin_width_{}_S.png'.format(condition, mouse, mouse_groups[mouse], session_str, bin_width)), format='png', dpi=300)
        plt.close(fig_S)

def plot_fluorescence_map_helper(mouse, sess, mouse_groups, session_str, bin_width, cells=np.array([]), random_width=0, SAVE_PATH=None, \
    PLOTS_DIR=None, want_3D=False, pcells_mice=dict(), max_fields=15, random_pcells_per_field=5, only_fm_pcells=False, print_pcell_maps=False):
    '''
    plot_fluorescence_map_helper().

    Note that pcells is passed from parent's pcells_mice[m].
    '''

    print("plot_fluorescence_map_helper: we were given {} max_fields!".format(max_fields))

    # Get fluorescence traces
    S_mov = sess.S_mov
    S_imm = sess.S_imm
    S_peakval = sess.S_peakval
    #[S_mov_sp, S_mov_pkval] = find_spikes_ca_S(S_mov, sess.thres, want_peakval=True)

    # Process mouse location and establish bins
    loc = Location_XY(mouse, mouse_groups[mouse], sess, bin_width)
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
    fm.find_place_fields(sess, method='iterative_gauss')

def plot_fluorescence_map(PLOTS_DIR, session, mouse_groups, session_str, bin_width=20, random_width=0, cells=np.array([]), want_3D=False, pcells_mice=dict(), \
    max_fields=15, only_fm_pcells=False, print_pcell_maps=False):
    save_path = os.path.join(PLOTS_DIR, 'fluorescence_maps_{}fields_{}'.format(max_fields, session_str)) #'{}_{}_fluorescence-map.png'.format(session_str, m))
    os.makedirs(save_path, exist_ok=True)
    for m, sess in session.items():
        print('*** WORKING FLUORESCENCE MAPS FOR {}... '.format(m))
        plot_fluorescence_map_helper(m, sess, mouse_groups, session_str, bin_width, random_width=random_width, SAVE_PATH=save_path, cells=cells, PLOTS_DIR=PLOTS_DIR, \
            want_3D=want_3D, pcells_mice=pcells_mice, max_fields=max_fields, only_fm_pcells=only_fm_pcells, print_pcell_maps=print_pcell_maps)

def plot_pf_analyses(PLOTS_DIR, session, mouse_groups, session_str, crossreg=None, mapping=None, auto_close=True):
    save_path = os.path.join(PLOTS_DIR, 'place_fields_{}'.format(session_str))
    os.makedirs(save_path, exist_ok=True)

    meas_num_pfs = {}
    meas_pf_size = {}
    meas_compactness_pf = {}
    meas_spatial_selectivity = {}
    processed_groups = {}   # placeholder so we know we encountered each group when going through the mice. Just assign True to each group once here.

    for mouse, sess in session.items():
        group = mouse_groups[mouse]
        if mouse in ['G07', 'G15']:
            continue
        if group not in processed_groups:
            processed_groups[group] = True
            
            meas_num_pfs[group] = np.array([])
            meas_pf_size[group] = np.array([])
            meas_compactness_pf[group] = np.array([])
            meas_spatial_selectivity[group] = np.array([])
        
        pf_keys = list(sess.fm.pf.pf_size.keys()) # pf keys should be the same for all pf-related dicts, so just use this one
        if crossreg:
            S_i = get_S_indeces_crossreg(sess, crossreg[mouse], mapping)
            pf_keys = np.intersect1d(S_i, pf_keys)

        # Get length of pf_size as num_pfs
        pf_size = {k:sess.fm.pf.pf_size[k] for k in pf_keys}
        meas_num_pfs[group] = np.append(meas_num_pfs[group], [len(v) for v in pf_size.values()])

        # Just append all pf sizes, but have to do this list comprehension to 'flatten' (why don't you have a nice way of flattening a list of lists with 
        # different length sublists, python or numpy??)
        meas_pf_size[group] = np.append(meas_pf_size[group], [x for cell in list(pf_size.values()) for x in cell])

        # ditto for compactness, spatial selectivity
        compactness_pf = {k:sess.fm.pf.compactness_pf[k] for k in pf_keys}
        spatial_selectivity = {k:sess.fm.pf.spatial_selectivity[k] for k in pf_keys}
        meas_compactness_pf[group] = np.append(meas_compactness_pf[group], [x for cell in list(compactness_pf.values()) for x in cell])
        meas_spatial_selectivity[group] = np.append(meas_spatial_selectivity[group], [x for cell in list(spatial_selectivity.values()) for x in cell])

    title_str_all = ['Number of PFs', 'PF size ($cm^2$)', 'PF compactness', 'Spatial selectivity'];
    meas_all = [meas_num_pfs, meas_pf_size, meas_compactness_pf, meas_spatial_selectivity]
    filename_all = ['num_pfs', 'pf_size', 'pf_compactness', 'spatial_selectivity']
    for title_str, meas, filename in zip(title_str_all, meas_all, filename_all):
                
        plt.figure()
        for group in processed_groups.keys():
            if title_str == 'Number of PFs':
                #plt.hist(x=meas[group], color=group_colours[group], bins=100, density=True, alpha=0.3, cumulative=True, histtype='step', lw=2)
                plt.hist(x=meas[group], color=group_colours[group], bins=200, density=True, cumulative=True, histtype='step', lw=2)
            else:
                #plt.hist(x=meas[group], color=group_colours[group], bins=100, density=True, alpha=0.3, cumulative=True, histtype='step', lw=2)
                plt.hist(x=meas[group], color=group_colours[group], bins=200, density=True, cumulative=True, histtype='step', lw=2)
            plt.title(title_str)
        if crossreg:
            savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}.png'.format(filename, group, session_str, mapping))
        else:
            savefile = os.path.join(save_path, '{}_{}_{}.png'.format(filename, group, session_str))
        plt.savefig(savefile, format='png', dpi=300)
        if auto_close:
            plt.close()

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

def plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_B, Test_B_1wk, mouse_groups, crossreg=None, mapping=None, auto_close=True):
    save_path = os.path.join(PLOTS_DIR, 'place_fields_groups')
    os.makedirs(save_path, exist_ok=True)

    meas_num_pfs = {}
    meas_pf_size = {}
    meas_compactness_pf = {}
    meas_spatial_selectivity = {}
    processed_groups = {}   # placeholder so we know we encountered each group when going through the mice. Just assign True to each group once here.
    sess_names = ['TFC_cond', 'Test_B', 'Test_B_1wk']

    for mouse, group in mouse_groups.items():
        if mouse in ['G07', 'G15']:
            continue
        if group not in processed_groups:
            processed_groups[group] = True
            meas_num_pfs[group] = {}
            meas_pf_size[group] = {}
            meas_compactness_pf[group] = {}
            meas_spatial_selectivity[group] = {}
            
        for session, sess_name in zip([TFC_cond, Test_B, Test_B_1wk], sess_names):
            sess = session[mouse]
            if sess_name not in meas_num_pfs[group].keys():
                meas_num_pfs[group][sess_name] = np.array([])
                meas_pf_size[group][sess_name] = np.array([])
                meas_compactness_pf[group][sess_name] = np.array([])
                meas_spatial_selectivity[group][sess_name] = np.array([])
            
            pf_keys = list(sess.fm.pf.pf_size.keys()) # pf keys should be the same for all pf-related dicts, so just use this one
            if crossreg:
                S_i = get_S_indeces_crossreg(sess, crossreg[mouse], mapping)
                pf_keys = np.intersect1d(S_i, pf_keys)

            # Get length of pf_size as num_pfs
            pf_size = {k:sess.fm.pf.pf_size[k] for k in pf_keys}
            meas_num_pfs[group][sess_name] = np.append(meas_num_pfs[group][sess_name], [len(v) for v in pf_size.values()])

            # Just append all pf sizes, but have to do this list comprehension to 'flatten' (why don't you have a nice way of flattening a list of lists with 
            # different length sublists, python or numpy??)
            meas_pf_size[group][sess_name] = np.append(meas_pf_size[group][sess_name], [x for cell in list(pf_size.values()) for x in cell])

            # ditto for compactness, spatial selectivity
            compactness_pf = {k:sess.fm.pf.compactness_pf[k] for k in pf_keys}
            spatial_selectivity = {k:sess.fm.pf.spatial_selectivity[k] for k in pf_keys}
            meas_compactness_pf[group][sess_name] = np.append(meas_compactness_pf[group][sess_name], [x for cell in list(compactness_pf.values()) for x in cell])
            meas_spatial_selectivity[group][sess_name] = np.append(meas_spatial_selectivity[group][sess_name], [x for cell in list(spatial_selectivity.values()) for x in cell])

    title_str_all = ['Number of PFs', 'PF size ($cm^2$)', 'PF compactness', 'Spatial selectivity'];
    filename_all = ['num_pfs', 'pf_size', 'pf_compactness', 'spatial_selectivity']
    meas_all = [meas_num_pfs, meas_pf_size, meas_compactness_pf, meas_spatial_selectivity]
    for title_str, meas, filename in zip(title_str_all, meas_all, filename_all):
        sess_colours = {'TFC_cond': 'k', 'Test_B': 'gray', 'Test_B_1wk': 'gainsboro'}
        for group in processed_groups.keys():
            plt.figure()
            for sess_name in sess_names:
                if title_str == 'Number of PFs':
                    #plt.hist(x=meas[group], color=group_colours[group], bins=100, density=True, alpha=0.3, cumulative=True, histtype='step', lw=2)
                    plt.hist(x=meas[group][sess_name], color=sess_colours[sess_name], bins=200, density=True, cumulative=True, histtype='step', lw=2)
                else:
                    #plt.hist(x=meas[group], color=group_colours[group], bins=100, density=True, alpha=0.3, cumulative=True, histtype='step', lw=2)
                    plt.hist(x=meas[group][sess_name], color=sess_colours[sess_name], bins=200, density=True, cumulative=True, histtype='step', lw=2)
                plt.title('{} {}'.format(title_str, group))
            if crossreg:
                savefile = os.path.join(save_path, '{}_{}_across_sessions_crossreg_{}.png'.format(filename, group, mapping))
            else:
                savefile = os.path.join(save_path, '{}_{}_across_sessions.png'.format(filename, group))
            plt.savefig(savefile, format='png', dpi=300) 
            if auto_close:
                plt.close()               

def plot_pcells_distributions(PLOTS_DIR, session, mice_per_group, session_str, pcells_mice):
    '''
    *** INCOMPLETE ***
    '''
    os.makedirs(os.path.join(PLOTS_DIR, 'pcells_distributions_20fields_{}'.format(session_str)), exist_ok=True)

    num_responses_per_group = dict()

    for group, mice in mice_per_group.items():
        num_responses_per_group[group] = np.array([])
        for m in mice:
            num_responses = [len(x) for x in pcells_mice[m].values()]
            num_responses_per_group[group] = np.append(num_responses_per_group[group], num_responses)

    plt.figure()
    for group in mice_per_group.keys():
        plt.hist(x=num_responses_per_group[group], color=group_colours[group], density=True, alpha=0.3)

    plt.xlabel('# of significant bins')
    plt.ylabel('Density')

    #plt.suptitle('Per-cell linear track average spike peak changes (LT1->LT2)')
    filename = 'LT_average_peakval_changes.png'
    os.makedirs(os.path.join(PLOTS_DIR, 'LT_firing_rate_changes'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'LT_firing_rate_changes', filename), format='png', dpi=300)


def plot_pop_vectors(PLOTS_DIR, session, mouse_groups, session_str, bin_width=2, spk_cutoff=2):
    '''
    bin_width specified in seconds.
    '''
    save_path = os.path.join(PLOTS_DIR, 'pop_vectors_{}'.format(session_str))
    os.makedirs(save_path, exist_ok=True)

    for mouse, sess in session.items():
        group = mouse_groups[mouse]

        S = sess.S
        bin_frames = bin_width * MINISCOPE_FPS
        curr_frame = 0

        upper=1
        lower=0

        PV = np.zeros((S.shape[0], math.floor(S.shape[1]/bin_frames)))
        for i in range(PV.shape[1]):
            PV[:,i] = np.where(np.sum(S[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
            curr_frame += bin_frames


'''
For getting crossreg mappings for calculating PVs for skmeans in R
'''
def process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping, \
                       binarize=False, normalize=True, normalize_full=False, spk_cutoff=2, bin_width=1):

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
    print('*** Using bin_width {} s = bin_frames {}'.format(bin_width, bin_frames))

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

    if normalize_full and not normalize and not binarize:
        print('*** normalizing (full)...')
        PV_c = np.concatenate((PV_TFC, PV_Test_B, PV_Test_B_1wk), axis=1)
        pt = PowerTransformer()
        PV_t = pt.fit_transform(PV_c)
        PV_TFC = PV_t[:,0:PV_TFC.shape[1]]
        PV_Test_B = PV_t[:,PV_TFC.shape[1]+1:PV_TFC.shape[1]+PV_Test_B.shape[1]+1]
        PV_Test_B_1wk = PV_t[:,PV_TFC.shape[1]+PV_Test_B.shape[1]:]

    df = pd.DataFrame(PV_TFC)
    if binarize:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_TFC_crossreg_binarized.csv".format(mouse))
    elif normalize_full:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_TFC_crossreg_normalized_full.csv".format(mouse))
    elif normalize:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_TFC_crossreg_normalized.csv".format(mouse))
    else:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_TFC_crossreg.csv".format(mouse))
    df = pd.DataFrame()
    df['tone_onsets'] = sess_TFC.tone_onsets
    df['tone_offsets'] = sess_TFC.tone_offsets
    df['shock_onsets'] = sess_TFC.shock_onsets
    df['shock_offsets'] = sess_TFC.shock_offsets
    df.to_csv(MAIN_DRIVE+"\data\PV_{}_stims.csv".format(mouse))

    df = pd.DataFrame(PV_Test_B)
    if binarize:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_crossreg_binarized.csv".format(mouse))
    elif normalize_full:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_crossreg_normalized_full.csv".format(mouse))
    elif normalize:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_crossreg_normalized.csv".format(mouse))
    else:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_crossreg.csv".format(mouse))
    df = pd.DataFrame()
    df['tone_onsets'] = sess_Test_B.tone_onsets
    df['tone_offsets'] = sess_Test_B.tone_offsets
    df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_stims.csv".format(mouse))

    df = pd.DataFrame(PV_Test_B_1wk)
    if binarize:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_1wk_crossreg_binarized.csv".format(mouse))
    elif normalize_full:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_1wk_crossreg_normalized_full.csv".format(mouse))
    elif normalize:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_1wk_crossreg_normalize.csv".format(mouse))        
    else:
        df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_1wk_crossreg.csv".format(mouse))
    df = pd.DataFrame()
    df['tone_onsets'] = sess_Test_B_1wk.tone_onsets
    df['tone_offsets'] = sess_Test_B_1wk.tone_offsets
    df.to_csv(MAIN_DRIVE+"\data\PV_{}_Test_B_1wk_stims.csv".format(mouse))

    print('...done.')

def cluster_pop_vectors_helper(PLOTS_DIR, session, session_str, mouse, group, transpose_wanted=False, \
    auto_close=False, bin_width=2, spk_cutoff=2, crossreg=None, Ca_act_type='full', dend_step_size=5, minimum_cluster_size=10, \
    dend_short_circuit=False, use_silhouette=True, only_crossreg=False, sess_all=None):
    '''
    Find population vectors, either taking cells as observations and time as features (transpose==False)
    or, conversely, taking time points as observations and cells as features (typical 'population vector'; 
    transpose==True).

    Because processing times can be long, only process one mouse at a time specified by argument. Caller
    can loop over groups etc. as desired.

    Major switches:
        use_silhouette - whether to use silhouette scores to determine dend_thresh
        only_crossreg - cluster only within crossreg neurons. Set to False to cluster entire session.

    NB: transpose not fully implemented.
    '''
    s = binary_C = labels = frac_labels = labels_tot = None # for consistenty so can use PV object even for transpose (which doesn't use these yet)

    dir_name = 'PV'
    if use_silhouette:
        dir_name += '_silhouette'
    if only_crossreg:
        dir_name += '_only_crossreg'

    # To store for transpose-related calculations.
    PV_dist = {}

    s=session[mouse]
    crossreg_mouse=crossreg[mouse]
    if mouse in ['G07']:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B_1wk', with_crossreg=crossreg_mouse) # not ideal to specify manually but...
    elif mouse in ['G15']:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B', with_crossreg=crossreg_mouse) # not ideal to specify manually but...
    else:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B+Test_B_1wk', with_crossreg=crossreg_mouse) # not ideal to specify manually but...

    S_i = s.get_S_indeces(S_idx)
    if sess_all:
        s_B = sess_all[1][mouse]
        s_B_1wk = sess_all[2][mouse]
        S_i_Test_B = get_S_indeces_crossreg(s_B, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk') # maybe not good to hard-code mapping str here
        S_i_Test_B_1wk = get_S_indeces_crossreg(s_B_1wk, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk')    

    if only_crossreg:
        if Ca_act_type == 'mov':
            C=s.S_mov[S_i,:]
            if sess_all:
                C_B=s_B.S_mov[S_i_Test_B,:]
                C_B_1wk=S_B_1wk.S_mov[S_i_Test_B_1wk,:]
        elif Ca_act_type == 'imm':
            C=s.S_imm[S_i,:]
            if sess_all:
                C_B=s_B.S_imm[S_i_Test_B,:]
                C_B_1wk=s_B_1wk.S_imm[S_i_Test_B_1wk,:]
        else:
            C=s.S[S_i,:]
            if sess_all:
                C_B=s_B.S[S_i_Test_B,:]
                C_B_1wk=s_B_1wk.S[S_i_Test_B_1wk,:]
    else:
        if Ca_act_type == 'mov':
            C=s.S_mov
            if sess_all:
                C_B=s_B.S_mov
                C_B_1wk=s_B_1wk.S_mov
        elif Ca_act_type == 'imm':
            C=s.S_imm
            if sess_all:
                C_B=s_B.S_imm
                C_B_1wk=s_B_1wk.S_imm
        else:
            C=s.S
            if sess_all:
                C_B=s_B.S
                C_B_1wk=s_B_1wk.S

    max_C = np.max(C)
    transpose_str = ''
    if transpose_wanted:
        transpose_str = 'transpose_'
        bin_frames = bin_width * MINISCOPE_FPS 
        look_ahead = int(20 / bin_width) # should be 20 seconds.
        look_ahead_trunc = int(15 / bin_width) # just for TFC_cond 1st period

        PV = np.zeros((C.shape[0], math.floor(C.shape[1]/bin_frames)))
        if sess_all:
            PV_B = np.zeros((C_B.shape[0], math.floor(C_B.shape[1]/bin_frames)))
            PV_B_1wk = np.zeros((C_B_1wk.shape[0], math.floor(C_B_1wk.shape[1]/bin_frames)))
            C_calcs = [C, C_B, C_B_1wk]
            PV_calcs = [PV, PV_B, PV_B_1wk]
        else:
            C_calcs = [C]
            PV_calcs = [PV]
        
        for C_calc, PV_calc in zip(C_calcs, PV_calcs):
            curr_frame = 0
            upper = 1
            lower = 0
            #inc = 0
            for i in range(PV_calc.shape[1]):
                PV_calc[:,i] = np.where(np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
                curr_frame += bin_frames

        ''' # old, delete
        curr_frame = 0
        upper = 1
        lower = 0
        #inc = 0
        for i in range(PV.shape[1]):
            PV[:,i] = np.where(np.sum(C[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
            curr_frame += bin_frames
        #for i in range(PV.shape[0]):
        #    plt.plot(range(PV.shape[1]), PV[i,:]+inc,'k',lw=0.1)
        #    inc += 1
        '''
        plt.figure(figsize=(8,6))
        plt.imshow(PV)
        for i in s.shock_onsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=0.5)
        for i in s.tone_onsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=0.5) 
        for i in s.shock_offsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=0.5)
        for i in s.tone_offsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=0.5)
        plt.xlim((0, PV.shape[1]))
        plt.ylim((0, PV.shape[0]))
        xtick_seconds_PV = np.concatenate(([0], [s/bin_width for s in s.tone_onsets_def[0:len(s.tone_onsets)]], [np.min((1300/bin_width,PV.shape[1]))]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [1300]))
        plt.xticks(ticks=xtick_seconds_PV, labels=xtick_seconds)
        #plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        filename = '{}_{}_{}_{}{}_PV_binned.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

        #
        # Perform PV calculations
        #

        #
        # First, within-session PV comparisons
        #
        if 'TFC_cond' in sess_label:
            post_shock_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.shock_offsets])]
            PV_post_shock = np.hstack([PV[:, start:end] for start, end in post_shock_ranges])

            post_tone_20s_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_offsets])]
            first_range = np.round(s.tone_offsets[0]/bin_frames) # really ugly 
            post_tone_20s_ranges[0] = (int(first_range), int(first_range+look_ahead_trunc))
        if 'Test_B' in sess_label:            
            post_tone_20s_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_offsets])]
        PV_post_tone_ranges = np.hstack([PV[:, start:end] for start, end in post_tone_ranges])

        tone_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_onsets])]
        PV_tone_ranges = np.hstack([PV[:, start:end] for start, end in tone_ranges])

        baseline_ranges = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(len(s.tone_onsets))]
        baseline0_range = (0, look_ahead) # at beginning of recording
        baseline1_range = (tone_ranges[0][0]-look_ahead, tone_ranges[0][0]) # right before first tone
        PV_baseline = np.hstack([PV[:, start:end] for start, end in baseline_ranges])
        PV_baseline0 = PV[:, baseline0_range[0]:baseline0_range[1]]
        PV_baseline1 = PV[:, baseline1_range[0]:baseline1_range[1]]
        
        tone_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_onsets])]
        PV_post_tone_ranges = np.hstack([PV[:, start:end] for start, end in post_tone_ranges])

        if 'TFC_cond' in sess_label:
            comparisons = ['baseline0_to_post_shock', 'baseline1_to_post_shock', 'baseline0_to_post_tones_20s', 'baseline1_to_post_tones_20s']
            comparisons += ['post_tone_20s_0_to_post_shocks', 'post_tone_20s_1_to_post_shocks', 'post_tone_20s_2_to_post_shocks', 'post_tone_20s_3_to_post_shocks', 'post_tone_20s_4_to_post_shocks']
            comparisons += ['baseline0_to_tones', 'baseline1_to_tones']
            PV0s = [PV_baseline0, PV_baseline1, PV_baseline0, PV_baseline1]
            PV0s += [PV[:,start:end] for start,end in post_tone_20s_ranges]
            PV0s += [PV_baseline0, PV_baseline1]
            comp_ranges = [post_shock_ranges, post_shock_ranges, post_tone_20s_ranges, post_tone_20s_ranges]
            comp_ranges += [post_shock_ranges, post_shock_ranges, post_shock_ranges, post_shock_ranges, post_shock_ranges]
            comp_ranges += [tone_ranges, tone_ranges]
                        
            #
            # Process across-session PV comparisons (only once, hence when we are handling TFC_cond only)
            #
        for comparison, PV0, comp_range, plot_num in zip(comparisons, PV0s, comp_ranges, range(len(comparisons))):
            print('Working {}... '.format(comparison))
            PV_dist[comparison] = []
            for period in range(len(comp_range)):
                PV_dist[comparison].append([])
                (start, end) = comp_range[period]
                for i in range(PV0.shape[1]):
                    PV_dist[comparison][period].append(mahalanobis(PV0[:,i], PV[:,start:end]))
            plt.figure()
            plt.scatter(np.repeat(np.arange(len(comp_range)), PV0.shape[1]),PV_dist[comparison])
            plt.plot(range(len(comp_range)), [np.mean(per) for per in PV_dist[comparison]], 'r')
            plt.title(comparison)
            plt.xticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
            filename = '{}_{}_{}_{}{}_{}_{}.png'.format(Ca_act_type, group, mouse, transpose_str, session_str, plot_num, comparison)
            os.makedirs(os.path.join(PLOTS_DIR, dir_name, 'PV_dist'), exist_ok=True)
            plt.savefig(os.path.join(PLOTS_DIR, dir_name, 'PV_dist', filename), format='png', dpi=300)
            if auto_close:
                plt.close()

    else:

        #
        # Cell-PVs, no crossreg
        #
        plt.figure(figsize=(8,6))
        inc = 0        
        binary_C = np.copy(C)
        for i in range(C.shape[0]):
            binary_C[i, binary_C[i,:] != 0] = 1
            plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
            inc += 1
      
        if s.session_type == 'TFC_cond':
            for i in s.shock_onsets:
                plt.axvline(i, c='r', ls='-', lw=0.5)
            for i in s.shock_offsets:
                plt.axvline(i, c='r', ls='-', lw=0.5)            
        for i in s.tone_onsets:
            plt.axvline(i, c='b', ls='-', lw=0.5) 
        for i in s.tone_offsets:
            plt.axvline(i, c='b', ls='-', lw=0.5)
        plt.xlim((0, binary_C.shape[1]))
        plt.ylim((0, binary_C.shape[0]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [np.min((1300, np.floor(C.shape[1]/MINISCOPE_FPS)))]))
        plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        plt.xlabel('Time (s)')
        plt.ylabel('Cell #')
        plt.title('Calcium transients {} {} ({})'.format(session_str, mouse, Ca_act_type))
        filename = '{}_{}_{}_{}{}_raster.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

        #
        # Cell-PVs, crossreg (this session in red)
        #
        # *** Doesn't make sense to do during only_crossreg, because *all* neurons are crossreg'ed!
        #
        if not only_crossreg:
            plt.figure(figsize=(8,6))
            inc = 0
            #binary_C = np.copy(C)
            for i in range(C.shape[0]):
                binary_C[i, binary_C[i,:] != 0] = 1
            for i in range(len(S_i)):
                #if S_idx[i] < C.shape[0]: # since G06,G15 are borked, has a crossreg cell ID higher than tot number in session...
                plt.plot(range(C.shape[1]), binary_C[S_i[i],:]+inc,'r',lw=0.1)
                inc += 1
            for i in range(C.shape[0]):
                if i not in S_i:
                    plt.plot(range(C.shape[1]), binary_C[i,:]+inc,'k',lw=0.1)
                    inc += 1
        
            if s.session_type == 'TFC_cond':
                for i in s.shock_onsets:
                    plt.axvline(i, c='r', ls='-', lw=0.5)
                for i in s.shock_offsets:
                    plt.axvline(i, c='r', ls='-', lw=0.5)            
            for i in s.tone_onsets:
                plt.axvline(i, c='b', ls='-', lw=0.5) 
            for i in s.tone_offsets:
                plt.axvline(i, c='b', ls='-', lw=0.5)
            plt.xlim((0, binary_C.shape[1]))
            plt.ylim((0, binary_C.shape[0]))
            xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [np.min((1300, np.floor(C.shape[1]/MINISCOPE_FPS)))]))
            plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
            plt.xlabel('Time (s)')
            plt.ylabel('Cell #')
            plt.title('Calcium transients crossreg {} {} {}'.format(session_str, mouse, Ca_act_type))
            filename = '{}_{}_{}_{}{}_raster_crossreg.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
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
        binary_C_fit = np.copy(PV).transpose()
    else:
        binary_C_fit = binary_C

    Z = linkage(binary_C_fit, 'ward')

    # Consistency check. We need all of y_ss range to see trivially max regions when using silhouette scores.
    if use_silhouette:
        dend_short_circuit = False
    else:
        dend_short_circuit = True # no need to keep computing

    #
    # Fit the dendrogram threshold, possibly using average silhouette scores
    #
    # However, silhouette score doesn't work well for irregularly shaped or sized clusters, i.e. with most
    # messy population data. We try to get around it by only considering max avg silhouette scores for regions
    # of avg silhouette landscape where the values aren't "trivially maximum" when you go to the limit of very
    # low cluster numbers. 
    #
    # Cf.
    # https://en.wikipedia.org/wiki/Silhouette_(clustering)
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    # https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
    #
    max_dend = np.max(Z[:,2])
    max_silhouette_avg = -2
    dend_thresh = 0
    x_dend = []
    y_ss = []
    y_ss_avg = []
    y_ss_std = []
    y_n_clusters = []
    thresh_marker = ''
    #for thresh in range(dend_step_size,int(np.floor(max_dend)),dend_step_size):
    max_dend_int = int(max_dend)
    if max_dend_int <= dend_step_size:
        min_dend = 0
        dend_step = 1
    else:
        min_dend = dend_step_size
        dend_step = dend_step_size
    for thresh in range(min_dend,max_dend_int,dend_step):
        clustering = AgglomerativeClustering(distance_threshold=thresh, n_clusters=None, linkage='ward')
        labels = clustering.fit_predict(binary_C_fit)
        if use_silhouette:
            try:
                ss = silhouette_samples(binary_C_fit, labels)
            except ValueError:
                print("For {} {} thresh = {}, num labels same as num samples, skipping...".format(group, mouse, thresh))
                continue
            silhouette_avg = silhouette_score(binary_C_fit, labels)
            y_ss.append(silhouette_avg)
        x_dend.append(thresh)
        n_clusters = len(np.unique(labels))
        y_n_clusters.append(n_clusters)
        cluster_sizes = []
        cluster_ss_avg = []
        cluster_ss_std = []
        for i in range(n_clusters):
            cluster_sizes.append(len(np.where(labels==i)[0]))
            if use_silhouette:
                cluster_ss_avg.append(np.mean(ss[np.where(labels==i)[0]]))
                cluster_ss_std.append(np.std(ss[np.where(labels==i)[0]]))
        y_ss_avg.append(cluster_ss_avg)
        y_ss_std.append(cluster_ss_std)
        #if silhouette_avg >= max_silhouette_avg:
        #    max_silhouette_avg = silhouette_avg
        if all(x >= minimum_cluster_size for x in cluster_sizes) and dend_thresh == 0:
            dend_thresh = thresh
            thresh_marker = '***'
        if use_silhouette:
            print("For {} {} thresh = {}, n_clusters = {}, silhouette_avg = {} cluster_ss_avg = {},".format(group, mouse, thresh, n_clusters, silhouette_avg, cluster_ss_avg), end='')
        else:
            print("For {} {} thresh = {}, n_clusters = {},".format(group, mouse, thresh, n_clusters), end='')
        print(" cluster sizes = {} {}".format(cluster_sizes, thresh_marker))
        if dend_thresh > 0 and dend_short_circuit:
            break
        thresh_marker = ''

    ## Stuff to remove vvv
    #y_ss_diff = np.diff(y_ss)
    #dend_thresh = x_dend[np.where(y_ss_diff == np.max(y_ss_diff))[0][0] + 1]
    '''
    y_ss_nontrivial = np.array(y_ss)[np.where(np.array(y_n_clusters) >= 4)[0]]
    x_max_idx = np.where(y_ss_nontrivial == np.max(y_ss_nontrivial))[0][0]
    dend_thresh = x_dend[x_max_idx]
    '''

    if use_silhouette:
        
        # Find max avg silhouette score that isn't the "trivially max" plateau with lowest cluster numbers
        # Additionally, prefer threshold regions where the number of clusters aren't too large (can have spurious
        # local maxima). Do this by weighting the avg Silhouette score by the inverse of the number of clusters.
        y_ss_w = np.array(y_ss) * 1/np.array(y_n_clusters)
        y_ss_w_max = np.max(y_ss_w)
        y_ss_max = np.max(y_ss)
        idx=np.where(y_ss_w != y_ss_w_max)
        if len(idx[0]) == 0: # In case only one cluster, this will just grab it.
            idx = (np.array([0], dtype=int),)
        y_ss_idx = np.array(y_ss_w)[idx]
        y_ss_idx_max = np.max(y_ss_idx)
        x_dend_idx = np.array(x_dend)[idx]
        dend_thresh = x_dend_idx[np.where(y_ss_idx==y_ss_idx_max)][0]
        # Find region of max y_ss to shade in red in plot
        idx_triv = np.where(y_ss_w == np.max(y_ss_w))
        x_dend_idx_triv = np.array(x_dend)[idx_triv]

        plt.figure()
        plt.plot(x_dend, y_ss, 'k')
        plt.plot(x_dend, y_ss_w, 'r')
        plt.axvline(dend_thresh, c='k', ls='--', lw=1)
        plt.fill_between(x_dend_idx_triv, 0, np.max((y_ss_max, y_ss_w_max)), alpha=0.1, color='r')
        plt.title('Average silhouette score diffs {} {} ({})'.format(session_str, mouse, Ca_act_type))
        filename = '{}_{}_{}_{}{}_avg_silhouette_scores.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)    
        print('Selected dend_thresh {} for silhouette_avg {}'.format(dend_thresh, y_ss_max))
        if auto_close:
            plt.close()

    plt.figure()
    plt.plot(x_dend, y_n_clusters)
    plt.axvline(dend_thresh, c='k', ls='--', lw=1)
    plt.title('n_clusters per dend_thresh {} {} ({})'.format(session_str, mouse, Ca_act_type))
    filename = '{}_{}_{}_{}{}_n_clusters_per_dend_thresh.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)    
    if auto_close:
        plt.close()

    # Old way lol.. no more manual!
    #dend_thresh = dend_thresh_mice[mouse]

    plt.figure()
    dendrogram(Z)
    plt.axhline(dend_thresh, c='k', ls='--', lw=1)
    #filename = 'dendrogram_{}_{}_{}{}_{}.png'.format(session_str, transpose_str, group, mouse, Ca_act_type)
    filename = '{}_{}_{}_{}{}_dendrogram.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
    plt.title('Dendrogram {} {} ({})'.format(session_str, mouse, Ca_act_type))
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
    labels_tot = dict()
    for i in range(num_clusters):
        print('*** Label {} num: {}'.format(i, len(np.where(labels==i)[0])))
        labels_tot[i] = len(np.where(labels==i)[0])

    if not transpose_wanted:

        #
        # Cell-PVs, no crossreg
        #

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
        if s.session_type == 'TFC_cond':
            for i in s.shock_onsets:
                plt.axvline(i, c='r', ls='-', lw=0.5)
            for i in s.shock_offsets:
                plt.axvline(i, c='r', ls='-', lw=0.5)            
        for i in s.tone_onsets:
            plt.axvline(i, c='b', ls='-', lw=0.5) 
        for i in s.tone_offsets:
            plt.axvline(i, c='b', ls='-', lw=0.5)
        plt.xlim((0, binary_C.shape[1]))
        plt.ylim((0, binary_C.shape[0]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [np.min((1300, np.floor(C.shape[1]/MINISCOPE_FPS)))]))
        plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        plt.xlabel('Time (s)')
        plt.ylabel('Cell #')
        plt.title('Calcium transients {} {} - sorted by cluster (tot {} clusters) ({})'.format(session_str, mouse, num_clusters, Ca_act_type))

        #filename = 'raster_{}_{}_{}_{}_num_clusters_{}.png'.format(session_str, group, mouse, num_clusters, Ca_act_type)
        filename = '{}_{}_{}_{}_raster_num_clusters_{}.png'.format(Ca_act_type, group, mouse, session_str, num_clusters)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

        #
        # Cell-PVs, crossreg, unsorted (traces from crossreg set to red, but sorted according to labels)
        #
        # *** No need when only_crossreg because all cells are crossreg'ed, so all would be red!
        #
        frac_labels = dict()
        #labels_tot = dict() # to delete once labels_tot above works
        if not only_crossreg:
            ind = np.argsort(labels)
            plt.figure(figsize=(8,6))
            inc = 0
            current_label = labels[ind[0]]

            frac_labels[current_label] = 0
            for i in ind:
                #binary_C[i, binary_C[i,:] != 0] = 0.5
                if ind[i] in S_i:
                    plt_colour = 'r'
                    frac_labels[current_label] += 1 # increment then later divide by total
                else:
                    plt_colour = 'k'
                plt.plot(range(C.shape[1]), binary_C[i,:]+inc,plt_colour,lw=0.1)
                inc += 1.0
                if labels[i] != current_label:
                    #labels_tot[current_label] = len(np.where(labels==current_label)[0]) # to delete once labels_tot above works
                    frac_labels[current_label] /= len(np.where(labels==current_label)[0])
                    print('*** label {} frac {}'.format(current_label, frac_labels[current_label]))
                    current_label = labels[i]
                    frac_labels[current_label] = 0
                    plt.axhline(inc,c='b',lw=0.5)
            # for last label
            #labels_tot[current_label] = len(np.where(labels==current_label)[0]) # to delete once labels_tot above works
            frac_labels[current_label] /= labels_tot[current_label]
            print('*** label {} frac {}'.format(current_label, frac_labels[current_label]))
            if s.session_type == 'TFC_cond':
                for i in s.shock_onsets:
                    plt.axvline(i, c='r', ls='-', lw=0.5)
                for i in s.shock_offsets:
                    plt.axvline(i, c='r', ls='-', lw=0.5)            
            for i in s.tone_onsets:
                plt.axvline(i, c='b', ls='-', lw=0.5) 
            for i in s.tone_offsets:
                plt.axvline(i, c='b', ls='-', lw=0.5)
            plt.xlim((0, binary_C.shape[1]))
            plt.ylim((0, binary_C.shape[0]))
            xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [np.min((1300, np.floor(C.shape[1]/MINISCOPE_FPS)))]))
            plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
            plt.xlabel('Time (s)')
            plt.ylabel('Cell #')
            plt.title('Calcium transients {} {} - sorted by cluster (tot {} clusters) ({})'.format(session_str, mouse, num_clusters, Ca_act_type))

            #filename = 'raster_{}_{}_{}_{}_num_clusters_{}.png'.format(session_str, group, mouse, num_clusters, Ca_act_type)
            filename = '{}_{}_{}_{}_raster_num_clusters_{}.png'.format(Ca_act_type, group, mouse, session_str, num_clusters)
            os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
            plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
            if auto_close:
                plt.close()

        #
        # Cell-PVs, crossreg, sorted (labels incremented by 1, and label=0 set to crossreg and shown first)
        #
        # maybe not

    else:
        ## Transpose (PV across time)

        num_clusters = len(np.unique(labels))
        cmap = plt.get_cmap('viridis')
        labels_colours = cmap(np.linspace(0, 1, num_clusters))

        #
        # Plot sorted by PV cluster times (perhaps not as useful)
        #
        ind = np.argsort(labels)
        plt.figure(figsize=(8,6))
        inc=0
        current_label = labels[0]
        for i in range(PV.shape[0]):
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(PV.shape[1]), PV[i,ind]+inc,'k',lw=0.1)
            inc += 1
        current_label = labels[ind[0]]
        for i in range(len(ind)):
            plt.fill_between([i, i+1], 0, PV.shape[0], alpha=0.1, color=labels_colours[labels[ind[i]]])
            if labels[ind[i]] != current_label:
                current_label = labels[ind[i]]
                plt.axvline(i,ls='--',c='k',lw=0.2)
        '''
        for i in s.shock_onsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=1)
        for i in s.tone_onsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=1) 
        for i in s.shock_offsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=1)
        for i in s.tone_offsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=1)
        '''
        plt.xlim((0, PV.shape[1]))
        plt.ylim((0, PV.shape[0]))
        xtick_seconds_PV = np.concatenate(([0], [s/bin_width for s in s.tone_onsets_def[0:len(s.tone_onsets)]], [np.min((1300/bin_width,PV.shape[1]))]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [1300]))
        plt.xticks(ticks=xtick_seconds_PV, labels=xtick_seconds)
        plt.xlabel('Time (s)')
        plt.ylabel('Cell #')
        plt.title('Calcium transients sorted by PV {} {} (tot {} clusters) ({})'.format(session_str, mouse, num_clusters, Ca_act_type))
        #filename = 'raster_transpose_sorted_PV_{}_{}_{}_{}_num_clusters_{}.png'.format(session_str, group, mouse, num_clusters, Ca_act_type)
        filename = '{}_{}_{}_transpose_{}_raster_sorted_PV_num_clusters_{}.png'.format(Ca_act_type, group, mouse, session_str, num_clusters)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

        #
        # Normal plot (just colour PV times by cluster)
        #

        plt.figure(figsize=(8,6))
        inc = 0
        for i in range(PV.shape[0]):
            #binary_C[i, binary_C[i,:] != 0] = 0.5
            plt.plot(range(PV.shape[1]), PV[i,:]+inc,'k',lw=0.1)
            inc += 1.0
        for i in range(len(labels)):
            plt.fill_between([i, i+1], 0, PV.shape[0], alpha=0.1, color=labels_colours[labels[i]])

        for i in s.shock_onsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=1)
        for i in s.tone_onsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=1) 
        for i in s.shock_offsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=1)
        for i in s.tone_offsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=1)

        plt.xlim((0, PV.shape[1]))
        plt.ylim((0, PV.shape[0]))
        xtick_seconds_PV = np.concatenate(([0], [s/bin_width for s in s.tone_onsets_def[0:len(s.tone_onsets)]], [np.min((1300/bin_width,PV.shape[1]))]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [1300]))
        plt.xticks(ticks=xtick_seconds_PV, labels=xtick_seconds)
        plt.xlabel('Time (s)')
        plt.ylabel('Cell #')
        plt.title('Calcium transients with highlighted PVs {} {} (tot {} clusters) ({})'.format(session_str, mouse, num_clusters, Ca_act_type))
        #filename = 'raster_transpose_unsorted_PV_{}_{}_{}_{}_num_clusters_{}.png'.format(session_str, group, mouse, num_clusters, Ca_act_type)
        filename = '{}_{}_{}_transpose_{}_raster_unsorted_PV_num_clusters_{}.png'.format(Ca_act_type, group, mouse, session_str, num_clusters)
        os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()
    
    if use_silhouette:
        pv = PopulationVector(mouse, group, s, binary_C, labels, frac_labels, labels_tot, dend_thresh, x_dend=x_dend, y_ss=y_ss, y_n_clusters=y_n_clusters, only_crossreg=only_crossreg, PV_dist=PV_dist)
    else:
        pv = PopulationVector(mouse, group, s, binary_C, labels, frac_labels, labels_tot, dend_thresh, x_dend=x_dend, y_n_clusters=y_n_clusters, only_crossreg=only_crossreg, PV_dist=PV_dist) 
    return pv

def cluster_pop_vectors(PLOTS_DIR, session, session_str, mice_per_group, transpose_wanted=False, auto_close=False, bin_width=1, \
    spk_cutoff=2, crossreg=None, Ca_act_type='full', use_silhouette=True, only_crossreg=False, sess_all=None):

    session_type = list(session.values())[0].session_type
    PV_group = dict()

    if transpose_wanted:
        transpose_str = " (transpose)"
    else:
        transpose_str = ''

    for group, mice in mice_per_group.items():
        for mouse in mice:
            if group not in PV_group.keys():
                PV_group[group] = []

            if session_type == 'Test_B' and mouse in ['G07']:
                continue
            if session_type == 'Test_B_1wk' and mouse in ['G15']:
                continue

            print("*** Processing PV for {} {} {} {}{}...".format(session_str, Ca_act_type, group, mouse, transpose_str))
            PV_mouse = cluster_pop_vectors_helper(PLOTS_DIR, session, session_str, mouse, group, \
                transpose_wanted=transpose_wanted, auto_close=auto_close, bin_width=bin_width, \
                spk_cutoff=spk_cutoff, crossreg=crossreg, Ca_act_type=Ca_act_type, use_silhouette=use_silhouette, only_crossreg=only_crossreg,
                sess_all=sess_all)

            PV_group[group].append(PV_mouse)

    print("*** ...done.")
    return PV_group

def plot_raster_clusters_sorted(pv, use_classify_colours=True, frac_type=0, Ca_act_type='', plots_dir='', auto_close=True):
    print('***FOO')
    label_colours = pv.label_colours
    num_clusters = len(pv.labels_tot)
    ind = np.argsort(pv.labels)
    plt.figure(figsize=(8,6))
    print('*** {} {} {} {}'.format(pv.mouse, pv.group, pv.session, num_clusters)) 
    inc = 0
    current_label = pv.labels[0]
    for i in ind:
        if use_classify_colours:
            colour = label_colours[pv.labels[i]]
        else:
            colour = 'k'
        #binary_C[i, binary_C[i,:] != 0] = 0.5
        plt.plot(range(pv.binary_C.shape[1]), pv.binary_C[i,:]+inc,colour,lw=0.1)
        inc += 1.0
        if pv.labels[i] != current_label:
            current_label = pv.labels[i]
            plt.axhline(inc,c='b',lw=0.5)
    s = pv.session
    if s.session_type == 'TFC_cond':
        for i in s.shock_onsets:
            plt.axvline(i, c='r', ls='-', lw=0.5)
        for i in s.shock_offsets:
            plt.axvline(i, c='r', ls='-', lw=0.5)            
    for i in s.tone_onsets:
        plt.axvline(i, c='b', ls='-', lw=0.5) 
    for i in s.tone_offsets:
        plt.axvline(i, c='b', ls='-', lw=0.5)
    plt.xlim((0, pv.binary_C.shape[1]))
    plt.ylim((0, pv.binary_C.shape[0]))
    xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [np.min((1300, np.floor(pv.binary_C.shape[1]/MINISCOPE_FPS)))]))
    plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
    plt.xlabel('Time (s)')
    plt.ylabel('Cell #')
    plt.title('Calcium transients {} {} - sorted by cluster (tot {} clusters) {} frac_type {}'.format(s.session_str, pv.mouse, num_clusters, Ca_act_type, frac_type))

    if plots_dir:
        if use_classify_colours:
            prefix = 'classify_'
        if Ca_act_type:
            Ca_act_type += '_'
        filename = '{}{}{}_{}_{}_num_clusters_{}_frac_type_{}.png'.format(prefix, Ca_act_type, pv.group, pv.mouse, s.session_str, num_clusters, frac_type)
        os.makedirs(os.path.join(plots_dir), exist_ok=True)
        plt.savefig(os.path.join(plots_dir, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

def plot_raster_clusters_crossreg(self, filename='', Ca_act_type='', PLOTS_DIR='', auto_close=True):
    ind = np.argsort(labels)
    plt.figure(figsize=(8,6))
    inc = 0
    current_label = labels[ind[0]]
    frac_labels = dict()
    labels_tot = dict()
    frac_labels[current_label] = 0
    for i in ind:
        #binary_C[i, binary_C[i,:] != 0] = 0.5
        if ind[i] in S_idx:
            plt_colour = 'r'
            frac_labels[current_label] += 1 # increment then later divide by total
        else:
            plt_colour = 'k'
        plt.plot(range(C.shape[1]), binary_C[i,:]+inc,plt_colour,lw=0.1)
        inc += 1.0
        if labels[i] != current_label:
            labels_tot[current_label] = len(np.where(labels==current_label)[0])
            frac_labels[current_label] /= len(np.where(labels==current_label)[0])
            print('*** label {} frac {}'.format(current_label, frac_labels[current_label]))
            current_label = labels[i]
            frac_labels[current_label] = 0
            plt.axhline(inc,c='b',lw=0.5)
    # for last label
    labels_tot[current_label] = len(np.where(labels==current_label)[0])
    frac_labels[current_label] /= labels_tot[current_label]
    print('*** label {} frac {}'.format(current_label, frac_labels[current_label]))
    if s.session_type == 'TFC_cond':
        for i in s.shock_onsets:
            plt.axvline(i, c='r', ls='-', lw=0.5)
        for i in s.shock_offsets:
            plt.axvline(i, c='r', ls='-', lw=0.5)            
    for i in s.tone_onsets:
        plt.axvline(i, c='b', ls='-', lw=0.5) 
    for i in s.tone_offsets:
        plt.axvline(i, c='b', ls='-', lw=0.5)
    plt.xlim((0, binary_C.shape[1]))
    plt.ylim((0, binary_C.shape[0]))
    xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [np.min((1300, np.floor(C.shape[1]/MINISCOPE_FPS)))]))
    plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
    plt.xlabel('Time (s)')
    plt.ylabel('Cell #')
    plt.title('Calcium transients {} {} - sorted by cluster (tot {} clusters) ({})'.format(session_str, mouse, num_clusters, Ca_act_type))

    #filename = 'raster_{}_{}_{}_{}_num_clusters_{}.png'.format(session_str, group, mouse, num_clusters, Ca_act_type)
    filename = '{}_{}_{}_{}_raster_num_clusters_{}.png'.format(Ca_act_type, group, mouse, session_str, num_clusters)
    dir_name = 'PV'
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()        

def classify_pop_vector(pv, thres=0.5, label_activity_thresh=0.2, classify_frac_t=0, force_calc=False, verbose=False, cohens_thresh=0.3, Ca_act_type=''):
    '''
    Classify the PopulationVector by first calculating firing rates for all periods for the given session
    associated with the PV. Then, for each label, classify it as follows:
    - tonically firing - most neurons firing (coloured blue)
    - ramping up - most neurons increasing activity over time (coloured green)
    - ramping down - most neurons decreasing activity over time (coloured red)
    - default - not fitting the above (coloured default black)
    - add more later..

    Parameters:
    - classify_span = how many periods to use as beginning or end
    '''
    label_categories = ['ramping-up', 'ramping-down', 'tonic', 'default']
    if not pv.is_classified or force_calc: # the function is idempotent.
        print('*** classifying pop vectors {}...'.format(pv.mouse))
        label_firing_rates = {} # firing rate for entire PV
        period_avg_rates = {}
        label_classification = [] # classification
        label_colours = []
        period_bounds = find_period_bounds_session(pv.session)
        label_cell_period_rates = {}
        period_lengths = np.zeros(len(period_bounds))

        for label in pv.labels_tot.keys():
            label_firing_rates[label] = []
            period_avg_rates[label] = []
            label_indices = np.where(pv.labels==label)[0]
            label_cell_period_rates[label] = np.zeros((len(label_indices), len(period_bounds)))

            for i_period, (type, bound) in zip(range(len(period_bounds)), period_bounds):
                period_firing_rates = np.zeros(len(label_indices))
                period_len_s = (bound[1] - bound[0]) / MINISCOPE_FPS
                period_lengths[i_period] = period_len_s
                for i in range(len(label_indices)):
                    period_spikes = find_spikes_ca(pv.binary_C[label_indices[i], bound[0]:bound[1]], thres)
                    period_firing_rates[i] = len(period_spikes) / period_len_s

                    # record number of spikes per each neuron per each period
                    label_cell_period_rates[label][i, i_period] = len(period_spikes) 
                period_avg_rates[label].append(np.mean(period_firing_rates))
            label_firing_rates[label] = np.mean(period_avg_rates[label])

            # Classify 
            if classify_frac_t:
                if verbose:
                    print('*** using classify_frac_t {}'.format(classify_frac_t))
                num_periods = len(period_bounds)
                frac = int(num_periods * classify_frac_t)

                first_avg_rates = np.mean(period_avg_rates[label][0:frac])
                last_avg_rates = np.mean(period_avg_rates[label][num_periods-frac:num_periods])
                first_cell_avg_rates = label_cell_period_rates[label][:,0:frac]
                last_cell_avg_rates = label_cell_period_rates[label][:,num_periods-frac:num_periods]
            else:
                if verbose:
                    print('*** NOT using classify_frac_t!')
                
                first_avg_rates = period_avg_rates[label][0]
                last_avg_rates = period_avg_rates[label][len(period_avg_rates[label])-1]
                first_cell_avg_rates = label_cell_period_rates[label][:,0]
                last_cell_avg_rates = label_cell_period_rates[label][:,label_cell_period_rates[label].shape[1]-1]

            # Measure Cohen's d for distance of two distributions
            cohens_d_denom = (np.std(first_cell_avg_rates) + np.std(last_cell_avg_rates))/2
            if cohens_d_denom == 0: # if only 1 cell in the label, std is 0, so don't want div by zero, just compare means directly.
                cohens_d_denom = 1
            cohens_d = (np.mean(first_cell_avg_rates) - np.mean(last_cell_avg_rates)) / cohens_d_denom
            print('*** we have cohens_d: {} with mean(first) {} mean(last) {} std(first) {} std(last) {} [cohens_d_denom {}] [thresh {}]'.format(cohens_d, \
                np.mean(first_cell_avg_rates), np.mean(last_cell_avg_rates), np.std(first_cell_avg_rates), np.std(last_cell_avg_rates), \
                cohens_d_denom, cohens_thresh))
            if cohens_d == np.inf or cohens_d == -np.inf:
                raise Exception("inf cohen's d")
            if first_avg_rates >= label_activity_thresh or last_avg_rates >= label_activity_thresh: 
                # *something* has "high activity". Now we use Cohen's d to figure out what, and which direction.
                if np.abs(cohens_d) < cohens_thresh: # no difference between the periods
                    label_classification.append('tonic')
                    label_colours.append('b')
                elif cohens_d >= 0:
                    # mean of first period higher, so we are ramping-down.
                    label_classification.append('ramping-down')
                    label_colours.append('r')
                else: # cohens_d < 0:
                    # mean of second period higher, so we are ramping-up.
                    label_classification.append('ramping-up')
                    label_colours.append('g')                     
            else:
                label_classification.append('default')
                label_colours.append('k')

            '''
            if first_avg_rates <= label_activity_thresh and last_avg_rates >= label_activity_thresh:
                label_classification.append('ramping-up')
                label_colours.append('g')
            elif first_avg_rates >= label_activity_thresh and last_avg_rates <= label_activity_thresh:
                label_classification.append('ramping-down')
                label_colours.append('r')
            elif first_avg_rates >= label_activity_thresh and last_avg_rates >= label_activity_thresh:
                label_classification.append('tonic')
                label_colours.append('b')
            else:
                label_classification.append('default')
                label_colours.append('k')
            '''

        pv.label_firing_rates = label_firing_rates
        pv.label_classification = label_classification
        pv.label_colours = label_colours
        pv.period_bounds = period_bounds
        pv.period_avg_rates = period_avg_rates
        pv.is_classified = True
    else:
        if verbose:
            print('*** already classified, skipping..')
    return pv.label_classification

def classify_pop_vector_get_label_categories():
    '''
    Convenience function.
    '''
    return ['ramping-up', 'ramping-down', 'tonic', 'default']

def process_pop_vectors(PLOTS_DIR, PV_sess, only_crossreg, crossreg_str=None, plot_type='boxplot', want_scatter=True, auto_close=True, use_silhouette=True, \
    frac_type_l=[0,1/3,1/2], force_calc=False, cohens_thresh=0.3, Ca_act_type=None):
    '''
    Give it the output of cluster_pop_vectors() and will perform various analyses.
    '''

    dir_name = 'PV'
    if use_silhouette:
        dir_name += '_silhouette'
    if only_crossreg:
        dir_name += '_only_crossreg'

    if Ca_act_type:
        Ca_act_type_l = [Ca_act_type]
    else:
        Ca_act_type_l = PV_sess.keys()

    only_crossreg_str = get_only_crossreg_str(only_crossreg)

    # Params for slopes of period firing rates
    bin_len_s = 20 # seconds
    bin_len = MINISCOPE_FPS*bin_len_s # 20 seconds
    population_len_thresh=6
    thres=0.5

    label_categories = classify_pop_vector_get_label_categories()

    # First plot basic PV info. # of labels per group (# of PVs), number of cells per
    # PV, % of reactivated cells per PV. Total cells per group (not just in PV).
    for frac_type in frac_type_l:
        for Ca_act_type in Ca_act_type_l:
            num_labels_tot_group = dict()
            num_cells_PV_group = dict()
            frac_labels_group = dict()
            label_firing_rates_group = dict() # These are taken after PopulationVector.classify() has been called.
            tot_cells_group = dict()
            num_cells_PV_norm_group = dict()
            label_class_frac = dict()

            # For period firing rates
            PV_group_sess = PV_sess[Ca_act_type]
            sess_rand = list(PV_group_sess.keys())[0]
            num_groups = len(PV_group_sess[sess_rand][only_crossreg_str])
            max_mice_in_any_group = max([len(x) for x in PV_sess[Ca_act_type][sess_rand][only_crossreg_str].values()])

            plots_dir = os.path.join(PLOTS_DIR, dir_name, 'analysis', Ca_act_type, 'frac_type_{0:.2f}'.format(frac_type), only_crossreg_str)
            os.makedirs(plots_dir, exist_ok=True)

            # Generate info and plot across groups first.
            PV_group_sess = PV_sess[Ca_act_type]
            for sess in PV_group_sess.keys():
                num_labels_tot_group[sess] = dict()
                num_cells_PV_group[sess] = dict()
                frac_labels_group[sess] = dict()
                label_firing_rates_group[sess] = dict()
                tot_cells_group[sess] = dict()
                num_cells_PV_norm_group[sess] = dict()
                label_class_frac[sess] = dict()

                PV_group = PV_group_sess[sess][only_crossreg_str]
                for group in PV_group.keys():
                    num_labels_tot_group[sess][group] = []
                    num_cells_PV_group[sess][group] = []
                    frac_labels_group[sess][group] = []
                    label_firing_rates_group[sess][group] = []
                    tot_cells_group[sess][group] = []
                    num_cells_PV_norm_group[sess][group] = []
                    for lc in label_categories:
                        if lc not in label_class_frac[sess].keys():
                            label_class_frac[sess][lc] = dict()
                        label_class_frac[sess][lc][group] = []

                    for v in PV_group[group]:
                        filename_suffix = '{}_{}_{}'.format(Ca_act_type, group, v.session.mouse)

                        ##
                        ## Main PV analysis loop. Add things below.
                        ##
                        num_labels_tot_group[sess][group].append((len(v.labels_tot), v))

                        for m in v.frac_labels.values():
                            frac_labels_group[sess][group].append((m, v))

                        classify_pop_vector(v, classify_frac_t=frac_type, force_calc=force_calc, cohens_thresh=cohens_thresh, Ca_act_type=Ca_act_type)

                        lc_dict = dict()
                        for lc in label_categories:
                            if v.label_classification:
                                lc_dict[lc] = v.label_classification.count(lc) / len(v.label_classification)
                            else:
                                lc_dict[lc] = 0
                            label_class_frac[sess][lc][group].append(lc_dict[lc])

                        plot_raster_clusters_sorted(v, use_classify_colours=True, frac_type=frac_type, Ca_act_type=Ca_act_type, plots_dir=plots_dir, auto_close=True)

                        for l,fr in v.label_firing_rates.items():
                            label_firing_rates_group[sess][group].append((fr, v))

                        tot_cells_group[sess][group].append((v.binary_C.shape[0], v))

                        for num_neurons_label in v.labels_tot.values():
                            num_cells_PV_group[sess][group].append((num_neurons_label, v))
                            num_cells_PV_norm_group[sess][group].append(((num_neurons_label / v.binary_C.shape[0]), v))

                        ##
                        ## done.
                        ##

                data_plots = [num_labels_tot_group[sess], num_cells_PV_group[sess], frac_labels_group[sess], label_firing_rates_group[sess], tot_cells_group[sess], num_cells_PV_norm_group[sess]]
                title_plots = ['num_labels_tot', 'num_cells_PV', 'frac_labels', 'label_firing_rates', 'tot_cells', 'num_cells_PV_norm']
                ylabels = ['# neurons', '# neurons', '% neurons', 'Avg. spikes/s (Hz)', '# neurons', '# neurons']
                for lc in label_class_frac[sess].keys():
                    data_plots.append(label_class_frac[sess][lc])
                    title_plots.append(lc)
                    ylabels.append('Frac. neurons')

                for data_plot, title_plot, ylabel in zip(data_plots, title_plots, ylabels):
                    print('*{}*'.format(title_plot))
                    if title_plot == 'frac_labels' and only_crossreg:
                        print('NOPE...***')
                        continue
                    plot_data_group(list, plots_dir, data_plot, title='{} {} {}'.format(title_plot, Ca_act_type, sess), ylabel=ylabel, \
                        filename='{}_{}_{}'.format(title_plot, sess, Ca_act_type), type=plot_type, want_scatter=want_scatter, auto_close=auto_close)                   


                #
                # Plot period firing rates, label classification stats
                #
                fig_permouse, axs_permouse = plt.subplots(num_groups,max_mice_in_any_group,figsize=(num_groups*2,max_mice_in_any_group), sharex=True, sharey=True) 
                fig_pergroup, axs_pergroup = plt.subplots(1,num_groups,figsize=(num_groups*1.5,num_groups), sharex=True, sharey=True)
                for i, group in zip(range(len(PV_group)), PV_group.keys()):
                    mice = PV_group[group]
                    for j, pv in zip(range(len(mice)), mice):
                        for label in pv.labels_tot.keys():
                            label_indices = np.where(pv.labels==label)[0]
                            if len(label_indices) <= population_len_thresh:
                                print('    *** skipping small label {} ({} neurons)'.format(label, len(label_indices)))
                                continue
                            traces = pv.binary_C[label_indices,:]
                            binned_avg_firing = np.array([])
                            for trace in traces:
                                sections = np.array_split(trace, np.round(len(trace)/bin_len))
                                if not binned_avg_firing.any():
                                    binned_avg_firing = np.zeros(len(sections))
                                for idx, section in zip(range(len(sections)), sections):
                                    binned_avg_firing[idx] += len(find_spikes_ca(section, thres))/bin_len_s
                                binned_avg_firing /= len(label_indices)
                            binned_avg_firing_smoothed = savgol_filter(binned_avg_firing, int(np.round(len(binned_avg_firing)/5)), 2)
                            binned_avg_firing_smoothed = np.where(binned_avg_firing_smoothed < 0, 0, binned_avg_firing_smoothed)
                            #binned_avg_firing_smoothed = binned_avg_firing
                            last_loc = len(binned_avg_firing_smoothed)
                            last_val = binned_avg_firing_smoothed[last_loc-1]
                            axs_permouse[i,j].plot(binned_avg_firing_smoothed, lw=1.0, alpha=0.4)
                            axs_permouse[i,j].text(last_loc, last_val, str(len(label_indices)), fontsize=8)
                            axs_pergroup[i].plot(binned_avg_firing_smoothed, lw=1.0, alpha=0.4)
                            axs_pergroup[i].text(last_loc, last_val, str(len(label_indices)), fontsize=8)

                            axs_pergroup[i].set_title('{}'.format(group))
                            if (j==0):
                                axs_permouse[i,j].set_ylabel('{}'.format(group))

                        print('process_pop_vectors {} {} {} {} smoothed [{},{}] plotted'.format(Ca_act_type, sess, group, pv.mouse, i,j))
                        #RIGHT_HERE

                fig_permouse.suptitle(sess)
                fig_pergroup.suptitle(sess)
                path = 'period_firing_rates_{}_{}_{}_pergroup.png'.format(Ca_act_type, sess, group)
                fig_pergroup.savefig(os.path.join(plots_dir, path), format='png', dpi=300)            
                path = 'period_firing_rates_{}_{}_{}_permouse.png'.format(Ca_act_type, sess, group)
                fig_permouse.savefig(os.path.join(plots_dir, path), format='png', dpi=300)
                if auto_close==True:
                    plt.close()

                fig_pergroup, axs_pergroup = plt.subplots(1,num_groups,figsize=(num_groups*1.75, num_groups*1.5), sharex=True, sharey=True)
                for i, group in zip(range(len(PV_group)), PV_group.keys()):
                    mice = PV_group[group]
                    classification_mapping = {'default': 0, 'tonic': 1, 'ramping-up': 2, 'ramping-down': 3}
                    group_hist = {0: 0, 1: 0, 2: 0, 3: 0}
                    for j, pv in zip(range(len(mice)), mice):
                        for label in pv.label_classification:
                            group_hist[classification_mapping[label]] += 1
                    axs_pergroup[i].bar(list(group_hist.keys()), group_hist.values(), color=['black', 'blue', 'green', 'red'])
                    axs_pergroup[i].set_xticks([0,1,2,3],['default', 'tonic', 'up', 'down'], rotation=-45)
                    if i==0:
                        axs_pergroup[i].set_ylabel('Number of PVs', size='large')
                    axs_pergroup[i].set_title(group)
                fig_pergroup.suptitle(sess)
                plt.tight_layout()                
                path = 'classification_hist_{}_{}_{}_permouse.png'.format(Ca_act_type, sess, group)
                fig_pergroup.savefig(os.path.join(plots_dir, path), format='png', dpi=300)
                if auto_close==True:
                    plt.close()

                fig_pergroup, axs_pergroup = plt.subplots(1,num_groups,figsize=(num_groups*1.75, num_groups*1.5), sharex=True, sharey=True)
                for i, group in zip(range(len(PV_group)), PV_group.keys()):
                    mice = PV_group[group]
                    classification_mapping = {'default': 0, 'tonic': 1, 'ramping-up': 2, 'ramping-down': 3}
                    group_hist = []
                    for j, pv in zip(range(len(mice)), mice):
                        for label in pv.label_classification:
                            group_hist.append(classification_mapping[label])
                    axs_pergroup[i].hist(group_hist, density=True, stacked=True)
                    axs_pergroup[i].set_xticks([0,1,2,3],['default', 'tonic', 'up', 'down'], rotation=-45)
                    if i==0:
                        axs_pergroup[i].set_ylabel('Cumulative density', size='large')
                    axs_pergroup[i].set_title(group)
                fig_pergroup.suptitle(sess)
                plt.tight_layout()
                path = 'classification_hist_{}_{}_{}_permouse_density.png'.format(Ca_act_type, sess, group)
                fig_pergroup.savefig(os.path.join(plots_dir, path), format='png', dpi=300)
                if auto_close==True:
                    plt.close()

                #
                # WIP
                #
                '''
                classification_mapping = {'default': 0, 'tonic': 1, 'ramping-up': 2, 'ramping-down': 3}
                num_class = len(classification_mapping)
                fig, axs = plt.subplots(1,num_class,figsize=(num_class*1.75, num_class*1.5), sharey=True)
                for i, classification in zip(range(len(classification_mapping)), classification_mapping.keys()):
                    group_cl = {'hM3D': [], 'hM4D': [], 'mCherry': []}
                    for group in PV_group.keys():
                        mice = PV_group[group]
                        for pv in mice:
                            num_class = sum(1 for cl in pv.label_classification if cl == classification)
                            group_cl[group].append(num_class)
                    plot_data_group(list, plots_dir, group_cl, title=classification, ylabel='Number', filename='foo', type='boxplot', auto_close=False, want_scatter=True)

                    axs[i].hist(group_cl, density=True, stacked=True)

                    bp = axs[i].boxplot([group_cl['hM3D'], group_cl['hM4D'], group_cl['mCherry']], \
                        notch=False, patch_artist=True, positions=range(num_groups), showfliers=False, widths=0.7)
                    #RIGHT_HERE 
                    axs[i].set_xticks([0,1,2],['hM3D', 'hM4D', 'mCherry'], rotation=-45)
                    if i==0:
                        axs[i].set_ylabel('Cumulative density', size='large')
                    axs_pergroup[i].set_title(group)
                fig_pergroup.suptitle(sess)
                plt.tight_layout()
                path = 'classification_hist_{}_{}_{}_permouse_density.png'.format(Ca_act_type, sess, group)
                fig_pergroup.savefig(os.path.join(plots_dir, path), format='png', dpi=300)
                if auto_close==True:
                    plt.close()
                '''
                    
                ### work on this later/if ever
                '''
                classification_mapping = {'default': 0, 'tonic': 1, 'ramping-up': 2, 'ramping-down': 3}
                fig, axs = plt.subplots(len(classification_mapping), figsize=(num_groups,max_mice_in_any_group*2), sharex=True, sharey=True) 
                for i, group in zip(range(len(PV_group)), PV_group.keys()):
                    mice = PV_group[group]
                    for j, pv in zip(range(len(mice)), mice):
                        for label in pv.labels_tot.keys():
                            label_indices = np.where(pv.labels==label)[0]
                '''

            #
            # Then plot across-sessions.
            #
            num_labels_tot_sess = dict()
            num_cells_PV_sess = dict()
            frac_labels_sess = dict()
            label_firing_rates_sess = dict()
            tot_cells_sess = dict()
            num_cells_PV_norm_sess = dict()
            label_class_frac_sess = dict()

            for group in ['hM3D', 'hM4D', 'mCherry']:
                num_labels_tot_sess[group] = dict()
                num_cells_PV_sess[group] = dict()
                frac_labels_sess[group] = dict()
                label_firing_rates_sess[group] = dict()
                tot_cells_sess[group] = dict()
                num_cells_PV_norm_sess[group] = dict()
                label_class_frac_sess[group] = dict()

                for sess in PV_group_sess.keys():
                    num_labels_tot_sess[group][sess] = num_labels_tot_group[sess][group]
                    num_cells_PV_sess[group][sess] = num_cells_PV_group[sess][group]
                    frac_labels_sess[group][sess] = frac_labels_group[sess][group]
                    label_firing_rates_sess[group][sess] = label_firing_rates_group[sess][group]
                    tot_cells_sess[group][sess] = tot_cells_group[sess][group]
                    num_cells_PV_norm_sess[group][sess] = num_cells_PV_norm_group[sess][group]
                    for lc in label_categories:
                        print('adding '+lc)
                        if lc not in label_class_frac_sess[group]:
                            label_class_frac_sess[group][lc] = dict()
                        label_class_frac_sess[group][lc][sess] = label_class_frac[sess][lc][group]

                data_plots = [num_labels_tot_sess[group], num_cells_PV_sess[group], frac_labels_sess[group], label_firing_rates_sess[group], tot_cells_sess[group], num_cells_PV_norm_sess[group]]
                title_plots = ['per_sess_num_labels_tot', 'per_sess_num_cells_PV', 'per_sess_frac_labels', 'per_sess_label_firing_rates', 'per_sess_tot_cells', 'per_sess_num_cells_PV_norm']
                ylabels = ['# neurons', '# neurons', '% neurons', 'Avg. spikes/s (Hz)', '# neurons', '# neurons']
                for lc in label_class_frac_sess[group].keys():
                    data_plots.append(label_class_frac_sess[group][lc])
                    title_plots.append('per_sess_' + lc)
                    ylabels.append('Frac. neurons')

                for data_plot, title_plot, ylabel in zip(data_plots, title_plots, ylabels):
                    if title_plot == 'per_sess_frac_labels' and only_crossreg:
                        continue
                    plot_data_group(list, plots_dir, data_plot, title='{} {} {}'.format(title_plot, Ca_act_type, group), ylabel=ylabel, \
                        filename='{}_{}_{}'.format(title_plot, group, Ca_act_type), type=plot_type, want_scatter=want_scatter, auto_close=auto_close)            

            #
            # across-sessions, done.
            #

    '''
    #
    # Additional analysis on gathered label info.
    #
    for Ca_act_type in ['full']:
        plots_dir = os.path.join(PLOTS_DIR, 'PV', 'analysis', 'extra')
        os.makedirs(os.path.join(PLOTS_DIR, plots_dir), exist_ok=True)
        for sess in PV_group_sess.keys():
            high_firing = [v for v in label_firing_rates_group[sess][group] if v[0] > 0.3]
            for fr,v in high_firing:
                plot_raster_clusters_sorted(self, filename='', Ca_act_type='', PLOTS_DIR='', auto_close=True):

    # First check means of reactivated label neurons across groups.
    frac_labels_group = dict()
    for group in PV_group.keys():
        frac_labels_group[group] = []
        for v in PV_group[group]:
            for m in v.frac_labels.values():
                frac_labels_group[group].append(m)


    data = frac_labels_group
    groups = sorted(PV_group.keys())
    ylabel = '% reactivated'
    title = 'Mean of reactivated neurons across PVs'
    filename = 'PV_reactivated'
    '''

def plot_data_group(list_func, PLOTS_DIR, data_list, title='', ylabel='', filename='', type='boxplot', auto_close=True, want_scatter=False):
    '''
    Plot a dict() that has three entries, with keys equal to strings corresponding to
    'hM3D', 'hM4D', 'mCherry' (e.g.). Does anova1. Can plot as normal bar plot or boxplot, depending 
    on `type' argument.

    Options:
        type = either `boxplot' or `barplot'. Default boxplot.
    '''
    groups = list_func(data_list.keys())
    num_groups = len(groups)
    x = range(num_groups)
    means = np.zeros(num_groups)
    sems = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))
    data = dict()

    if want_scatter:
        x_scatter = []
        data_scatter = []
    for group, idx in zip(groups, range(num_groups)):
        # So can work in multiple places (messy, sorry)/
        if isinstance(data_list[group][0], tuple):
            data[group] = [tup[0] for tup in data_list[group]]
        else:
            data[group] = data_list[group]
        means[idx] = np.mean(data[group])
        sems[idx] = np.std(data[group]) / np.sqrt(len(data[group]))
        errbars[1,idx] = sems[idx]
        if want_scatter:
            data_scatter.append(data[group])
            x_scatter.append([x[idx]+np.random.uniform(-0.2,+0.2) for _ in range(len(data[group]))])
    if want_scatter:
        x_scatter = [i for l in x_scatter for i in l]
        data_scatter = [i for l in data_scatter for i in l]

    fig, ax = plt.subplots(1,1,figsize=(3,4))

    if type == 'boxplot':
        bp = ax.boxplot([data[groups[0]], data[groups[1]], data[groups[2]]], \
            notch=False, patch_artist=True, positions=x, showfliers=False, widths=0.7)
        for p, c in zip(bp['boxes'], ['red','blue','black']):
            plt.setp(p,facecolor=c,alpha=0.5)
        for median in bp['medians']:
            plt.setp(median,color='black')
    
    if type == 'barplot':
        ax.bar(x, means, yerr=errbars, color=group_colours.values())
    
    if want_scatter:
        ax.scatter(x_scatter, data_scatter, color='black', facecolor='white', linewidths=1, s=10)
    print('*** anova prep: {} #{} {} #{} {} #{}'.format(len(data[groups[0]]), groups[0], len(data[groups[1]]), groups[1], len(data[groups[2]]), groups[2]))

    if want_scatter:
        heights = []
        for group in groups:
            heights.append(np.max(data[group]))
        try:
            do_anova1_plot(data[groups[0]], data[groups[1]], data[groups[2]], ax, heights)
        except Exception as e:
            print('*** plot_data_group: exception; skipping ANOVA: {}'.format(repr(e)))
    else:
        try:
            do_anova1_plot(data[groups[0]], data[groups[1]], data[groups[2]], ax, means+sems)
        except Exception as e:
            print('*** plot_data_group: exception; skipping ANOVA: {}'.format(repr(e)))

    ax.set_ylabel(ylabel)
    ax.set_xticks(x, rotation=-45)
    ax.set_xticklabels(groups, size='medium')

    # Thanks, ChatGPT vvv
    current_ylim = ax.get_ylim()
    percentage_increase = 0.05
    new_ylim = (
        current_ylim[0],  # lower limit remains the same
        current_ylim[1] * (1 + percentage_increase)  # upper limit increased by the percentage
    )
    ax.set_ylim(new_ylim)

    # Set the new y-limits
    ax.set_ylim(new_ylim)        
    ax.set_title(title)
    ax.set_xlim([x[0]-1, x[len(x)-1]+1])
    plt.subplots_adjust(left=0.22, bottom=0.09, right=0.90, top=0.90, wspace=0.20, hspace=0.20)

    if filename:
        path = '{}_{}.png'.format(filename, type)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, path), format='png', dpi=300)
    if auto_close:
        plt.close()

def find_period_bounds_session(session, wanted_type='all'):
    '''
    Generate a dictionary of tuples of onset,offset frames for behaviourally relevant periods, as well as the labels.
    '''
    period_bounds = []
    for i in session.periods:
        if i == 0:
            period_bounds.append(('initial', (0, session.tone_onsets[0]-1)))
        else:
            if session.session_type == 'TFC_cond':
                period_bounds.append(('iti', (session.shock_onsets[i-1], session.tone_onsets[i]-1)))
            else:
                period_bounds.append(('post-tone', (session.tone_onsets[i-1], session.tone_onsets[i]-1)))

        period_bounds.append(('tone', (session.tone_onsets[i], session.tone_offsets[i]-1)))

        if session.session_type == 'TFC_cond':
            period_bounds.append(('post-tone', (session.tone_offsets[i], session.shock_onsets[i]-1)))
            period_bounds.append(('shock', (session.shock_onsets[i], session.shock_offsets[i]-1)))        

    if session.session_type == 'TFC_cond':
        period_bounds.append(('iti', (session.shock_offsets[max(session.periods)], session.S.shape[1]-1)))
    else:
        period_bounds.append(('post-tone', (session.tone_offsets[max(session.periods)], session.S.shape[1]-1)))

    if wanted_type == 'all':
        return period_bounds

    wanted_pb = []
    for (type, bounds) in period_bounds:
        if type == wanted_type:
            wanted_pb.append((type, bounds))
    return wanted_pb

def pf_stability_calc(TFC_cond, Test_B, Test_B_1wk, crossreg, mapping):
    '''
    Calculate place field stability (PFS) using method of Guerrero et al. 2024 (Csicsvari lab).

    We will calculate Pearson's correlation coefficient over the significant responses.
    '''

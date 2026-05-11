import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import scipy.stats as stats
import os
import logging
import itertools
import pandas as pd
import seaborn as sns
import random
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from caban.utilities import *
from caban.sessions import *
from caban.spatial import *
from caban.plotting import *
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import PowerTransformer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import pickle
from pathlib import Path
from caban.decoder import _LT_1D_CM_PER_PX, _LT_1D_DISTANCE_UNIT

group_colours = {
    'hM3D' : 'r',
    'hM4D' : 'b',
    'mCherry' : 'k'
}
plt.rcParams['font.family'] = 'sans-serif'
# Arial preferred (paper style); fall back to metric-compatible Liberation Sans
# and DejaVu Sans on Linux/headless servers where Arial is unavailable.
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
# Suppress "findfont: Generic family 'sans-serif' not found" spam when Arial
# is missing (the fallback still renders correctly).
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

PVALS = [0.05, 0.01, 0.001]

class PopulationVector:
    def __init__(self, mouse, group, session, binary_C, labels, frac_labels, labels_tot, dend_thresh, x_dend=None, y_ss=None, y_n_clusters=None, \
        only_crossreg=False, PV_dist_types=None):
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
        self.PV_dist_types = PV_dist_types

        self.is_classified = False

        # Because not returning it from cluster_pop_vectors_helper(), so just lazily calculate it afterwards..
        self.reactivated_crossreg = {} # Number of label neurons that are re-activated (from crossreg)
        if not only_crossreg: # only compute if we did not only use crossreg..
            for label in range(len(labels_tot)):
                self.reactivated_crossreg[label] = int(frac_labels[label] * labels_tot[label])

def plot_session_sp_rates(PLOTS_DIR, mouse_groups, sp_rates, session_type, title_str, figsize=(8,4), with_suptitle=False, \
    tot_dh_incr=0.1):
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
    fig, ax = plt.subplots(figsize=(1.5,3))
    ax.spines[['right','top']].set_visible(False)
    ax.bar(x, means, yerr=errbars, color=group_colours.values())
    do_anova1_plot(weighted_averages['hM3D'], weighted_averages['hM4D'], weighted_averages['mCherry'], ax, means+stds, tot_dh_incr=tot_dh_incr)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Exc', 'Inh', 'Ctl'], size='medium',  rotation=-45)

    [min_y, max_y] = ax.get_ylim()
    range_y = max_y - min_y
    plt.yticks([min_y, range_y*0.25, range_y*0.5, range_y*0.75, range_y],[0,25,50,75,100])
    plt.ylabel(r'Normalized $\Delta$F/F (%)')
    plt.xlabel('Time (min)')

    if with_suptitle:
        plt.suptitle(title_str + ' avg')
    plt.tight_layout(pad=0.5)
    #plt.tight_layout()
    #fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.12)
    os.makedirs(os.path.join(PLOTS_DIR, 'sp_rates', '{}_sp_rates'.format(session_type)), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'sp_rates', '{}_sp_rates'.format(session_type), '{}_sp_rates-'.format(session_type)+title_str+'-avg.png'), format='png', dpi=300)
    plt.close()

def plot_LT_sp_rates(PLOTS_DIR, mouse_groups, sp_rates, title_str, figsize=(8,8), want_peakval=False):
    
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
    if want_peakval:
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

def generate_interneuron_cutoff(PLOTS_DIR, session, mice_per_group, auto_close=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pyr_frate_threshold = dict()
    group_thresholds = {'hM3D': [], 'hM4D': [], 'mCherry': []}
    groups = ['mCherry', 'hM3D', 'hM4D']  # left to right

    # Gather all mice per group in order
    mice_lists = [mice_per_group.get(g, []) for g in groups]
    max_mice = max(len(m) for m in mice_lists)

    # Prepare data for all subplots
    firing_rates_all = {g: [] for g in groups}
    thresholds_all = {g: [] for g in groups}
    mice_names = {g: mice_per_group.get(g, []) for g in groups}

    for g in groups:
        for mouse in mice_per_group.get(g, []):
            s = session[mouse]
            duration = np.diff(s.miniscope_exp_fnum)[0] / MINISCOPE_FPS
            firing_rates = np.zeros(len(s.S_spikes))
            for i in range(len(firing_rates)):
                firing_rates[i] = len(s.S_spikes[i]) / duration
            pyr_frate_threshold[mouse] = np.percentile(firing_rates, s.pyr_percentile_cutoff)
            group_thresholds[g].append(pyr_frate_threshold[mouse])
            firing_rates_all[g].append(firing_rates)
            thresholds_all[g].append(pyr_frate_threshold[mouse])

    # Create big figure with subplots: rows=max_mice, cols=3
    with nature_comm_style(column_width='double'):
        fig, axs = plt.subplots(
            max_mice, 3, figsize=(3 * 2, max_mice * 1.5), sharex=True, sharey=True
        )
        if max_mice == 1:
            axs = np.expand_dims(axs, 0)  # ensure 2D

        for col, g in enumerate(groups):
            for row in range(max_mice):
                ax = axs[row, col]
                if row < len(mice_names[g]):
                    mouse = mice_names[g][row]
                    rates = firing_rates_all[g][row]
                    threshold = thresholds_all[g][row]
                    ax.hist(rates, bins=30, color='skyblue', edgecolor='k')
                    ax.axvline(threshold, color='red', linestyle='--', lw=1)
                    # Add legend with threshold value
                    ax.legend([f"-- {threshold:.2f}"], loc='upper right', fontsize=8, frameon=False)
                    ax.set_title(mouse, fontsize=10)
                else:
                    ax.axis('off')
                if row == max_mice - 1:
                    ax.set_xlabel('Firing Rate (Hz)')
                if col == 0 and row < len(mice_names[g]):
                    ax.set_ylabel('Cell Count')

        # Add column subtitles
        for col, g in enumerate(groups):
            axs[0, col].annotate(
                g, xy=(0.5, 1.18), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )

        fig.suptitle("Firing Rate Distributions for TFC conditioning", fontsize=14, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = os.path.join(PLOTS_DIR, 'pyr_int_cutoff')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "all_mice_firing_rate_distributions.png"), bbox_inches='tight')
        if auto_close:
            plt.close(fig)

    # Barplot with scatter for thresholds per group (unchanged)
    means = [np.mean(group_thresholds[g]) if group_thresholds[g] else 0 for g in groups]
    sems = [np.std(group_thresholds[g])/np.sqrt(len(group_thresholds[g])) if group_thresholds[g] else 0 for g in groups]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(3,2))
    ax.bar(x, means, yerr=sems, color=[group_colours[g] for g in groups], alpha=0.7, capsize=5)
    for i, g in enumerate(groups):
        y = group_thresholds[g]
        ax.scatter(np.full(len(y), i) + np.random.uniform(-0.1, 0.1, len(y)), y, color='k', s=30, zorder=10)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel('90th Percentile Firing Rate (Hz)')
    ax.set_title('Interneuron Cutoff Thresholds')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "group_interneuron_cutoff_barplot.png"))
    if auto_close:
        plt.close()

    return pyr_frate_threshold

def set_interneuron_cutoff(session):
    for mouse, s in session.items():
        firing_rates = s.firing_rates
        pyr_frate_threshold = np.percentile(firing_rates, 90)
        s.pyr_frate_threshold = pyr_frate_threshold
        print(f'*** Set interneuron cutoff for {mouse} {s.session_type} to {pyr_frate_threshold:.2f} Hz')

        pyr_mask = firing_rates < s.pyr_frate_threshold
        cell_ids = np.arange(self.S.shape[0]) 
        self.cell_ids_pyr = cell_ids[self.pyr_mask].tolist()
        self.cell_ids_int = cell_ids[~self.pyr_mask].tolist()
        self.S_pyr = self.S[self.pyr_mask]
        self.S_mov = self.S_mov[self.pyr_mask]
        self.S_imm = self.S_imm[self.pyr_mask]
        self.C_pyr = self.C[self.pyr_mask]
        if self.YrA_full is not None:
            self.YrA_pyr = self.YrA[self.pyr_mask]
            if self.behaviour_type == 'movement':
                self.YrA_mov = self.YrA_mov[self.pyr_mask]
                self.YrA_imm = self.YrA_imm[self.pyr_mask]          

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

def do_anova1_plot(group_hM3D, group_hM4D, group_mCherry, ax, heights, annotate=True, tot_dh_incr=0.1):
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
        tot_dh = 0.01
        for p in np.where(post_hoc_res.reject==True)[0]:
            first = pairs[p][0]
            second = pairs[p][1]
            pval_str = get_pval_str(post_hoc_res.pvalues[p])
            if annotate:
                barplot_annotate_brackets(ax, first, second, pval_str, range(3), heights, barh=0, dh=tot_dh)
            tot_dh += tot_dh_incr # was 0.1 ! NB!

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
                    [S, S_spikes, S_peakval, S_idx] = first[m].get_S_mapping(mapping, want_peakval=True)
                elif crossreg_type == 'TFC_B_B_1wk':
                    [S, S_spikes, S_peakval, S_idx] = first[m].get_S_mapping(mapping, with_crossreg=crossreg_to_use[m], want_peakval=True)
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
                    [S, S_spikes, S_peakval, S_idx] = second[m].get_S_mapping(mapping, want_peakval=True)
                elif crossreg_type == 'TFC_B_B_1wk':
                    [S, S_spikes, S_peakval, S_idx] = second[m].get_S_mapping(mapping, with_crossreg=crossreg_to_use[m], want_peakval=True)
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
                    [S, S_spikes, S_peakval, S_idx] = third[m].get_S_mapping(mapping, want_peakval=True)
                elif crossreg_type == 'TFC_B_B_1wk':
                    [S, S_spikes, S_peakval, S_idx] = third[m].get_S_mapping(mapping, with_crossreg=crossreg_to_use[m], want_peakval=True)
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

            for cell in range(cells_per_mouse):
                print('*** selecting cell {}'.format(cell))

                not_satisfied=True
                while not_satisfied:
                    cell_random = random.randrange(sess.C.shape[0])
                    sample_t = True
                    print(f'cell_random: {cell_random}; sampling times: ', end='')
                    kill_num_max = 20
                    kill_num = 0                    
                    while sample_t:
                        t_idx_random = random.randrange(sess.C.shape[1]-len_trace)
                        print('{} '.format(t_idx_random), end='')                        
                        spk_indeces = np.logical_and(sess.S_spikes[cell_random] >= t_idx_random, sess.S_spikes[cell_random] < t_idx_random+len_trace)
                        spk_times = sess.S_spikes[cell_random][spk_indeces]
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
                    C_sel = sess.C[cell_random,t_idx_random:t_idx_random+len_trace]
                    ###ax.plot(C[cell_random, t_idx_random:t_idx_random+len_trace]/max_val)
                    ax.plot(C_sel/np.max(C_sel))

                    #ax.scatter(spk_times-t_idx_random, sess.S[cell_random][spk_times], s=80, facecolor='none', edgecolors='k')

                    S_sel = sess.S[cell_random,t_idx_random:t_idx_random+len_trace]
                    max_val = np.max(S_sel)
                    #ax.scatter(spk_times-t_idx_random, S_peakval[cell_random][spk_indeces]/max_val, s=80, facecolor='none', edgecolors='k')
                    ax.scatter(spk_times-t_idx_random, sess.S_peakval[cell_random][spk_indeces]/max_val, s=80, facecolor='none', edgecolors='k')
                    ax.plot(S_sel/max_val,'r')
                    #ax.plot(sess.Y[cell_random, t_idx_random:t_idx_random+len_trace]/max_val_raw,'g')
                    YrA_sel = sess.YrA[cell_random, t_idx_random:t_idx_random+len_trace]
                    F0 = np.nanmedian(sess.YrA[cell_random,:])
                    #delta_YrA = (YrA_sel - np.mean(YrA_sel)) / np.mean(YrA_sel)
                    delta_YrA = (YrA_sel - F0) / F0
                    ax.plot(delta_YrA/np.max(delta_YrA),'grey', alpha=0.5)
                    #ax.plot(sess.YrA[cell_random, t_idx_random:t_idx_random+len_trace],'grey')

                    x=input('good? (y/n) ')
                    if x == 'y':
                        not_satisfied = False
                    plt.close()
                    if x == 'q':
                        break
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
        YrA = sess.YrA
        S_spikes = sess.S_spikes
        S_peakval = sess.S_peakval

        c_group = group_colours[group]
        for (cell_random, t_idx_random) in sel:
            print('   cell {}, t_idx_random {} '.format(cell_random, t_idx_random), end='')
            max_val = np.max(C[cell_random,t_idx_random:t_idx_random+len_trace])
            print('[C..', end='')
            ax.plot(y_offset + (C[cell_random, t_idx_random:t_idx_random+len_trace]/(height_scale*max_val)), c='grey', lw=my_lw, alpha=0.5)
            print('] ', end='')
            #ax.scatter(spk_times-t_idx_random, sess.S[cell_random][spk_times], s=80, facecolor='none', edgecolors='k')
            max_val = np.max(sess.S[cell_random,t_idx_random:t_idx_random+len_trace])
            spk_indeces = np.logical_and(sess.S_spikes[cell_random] >= t_idx_random, sess.S_spikes[cell_random] < t_idx_random+len_trace)
            spk_times = S_spikes[cell_random][spk_indeces]
            print('[scatter..', end='')
            ax.scatter(spk_times-t_idx_random, y_offset + (S_peakval[cell_random][spk_indeces]/(height_scale*max_val)), s=marker_size, facecolor='none', edgecolors='k')
            print('] [S..', end='')
            ax.plot(y_offset + (sess.S[cell_random, t_idx_random:t_idx_random+len_trace]/(height_scale*max_val)), c=c_group, lw=my_lw, alpha=1.0)        
            print('] [YrA..', end='')
            YrA_sel = sess.YrA[cell_random, t_idx_random:t_idx_random+len_trace]
            F0 = np.nanmedian(sess.YrA[cell_random,:])
            delta_YrA = (YrA_sel - F0) / F0
            max_val = np.max(delta_YrA)
            ax.plot(y_offset + delta_YrA/(height_scale*max_val), c=c_group, lw=my_lw, alpha=0.5)
            print(']')

            y_offset += 1/height_scale
    os.makedirs(os.path.join(PLOTS_DIR, 'plot_sample_traces'), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_sample_traces', 'plot_sample_traces.png'), format='png', dpi=300)
    if paper_dir:
        path_name = os.path.join(paper_dir, 'plot_sample_traces.png')
        print('*** Plotting paper_dir {}'.format(path_name))
        plt.savefig(path_name, format='png', dpi=300)
        plt.savefig(os.path.join(paper_dir, 'plot_sample_traces.svg'), format='svg')
    plt.close()
    return selections

def collapse_runs(arr):
    """
    Collapse consecutive runs of 1s in each row of a 2D array
    so that only the first 1 of each run is kept.
    """
    # arr: shape (n_rows, n_cols)
    # Prepend a zero at start of each row to catch rising edges
    padded = np.pad(arr, ((0,0),(1,0)), constant_values=0)
    # Rising edges: places where diff == 1
    rising = np.diff(padded, axis=1) == 1
    # Convert boolean mask to int
    return rising.astype(int)

def process_PSTH_hist(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim='shock', frames_lookaround=40, \
    frames_save = 200, shaded='sem', num_shuffles=100, percentile=95.0, auto_close=True, X_hist_use='S', use_YrA=False, \
    normalize_velocity=False, sharey=False, spike_onset_only=False, normalize_per_mouse=False):
    '''
    process_PSTH_hist(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim='shock', X_hist_use='S', use_YrA=False, normalize_velocity=False, sharey=True, frames_lookaround=MINISCOPE_FPS*2)
    '''
    print('***PSTH_shuffle HIST')
    snippet_len = frames_lookaround * 2
    save_post = frames_lookaround + frames_save
    save_len = frames_lookaround * 2 + frames_save

    group_PSTH_hist = dict()  # spike counts
    group_PSTH_hist_all = dict() 
    group_PSTH = dict() # calculated PSTH from traces
    group_PSTH_hist_vel = dict() # counts normalized by velocity
    group_PSTH_vel = dict() # PSTH normalized by velocity

    rng = default_rng()

    print('*** MAPPING: {}'.format(mapping_type))
    for group, mice in mice_per_group.items():

        print('\nin {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            if mouse in ['G15']:
                continue
            #if mouse in ['G07', 'G14', 'G15', 'G20', 'G12']:
            #    continue
            print('{} {}'.format(mouse, group))
            s = session[mouse]
            crossreg = crossreg_mice[mouse]

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                #onsets = s.shock_onsets
                #offsets = s.shock_offsets
                onsets = []
                for i in range(len(s.shock_onsets)):
                    # Iterate through defined shock onsets but use def values (e.g. G09 only has 4 shocks)
                    onsets.append(s.shock_onsets[i])
                    #onsets.append(s.shock_onsets_def[i]*MINISCOPE_FPS)
                #onsets = s.shock_onsets
                #onsets = s.shock_onsets_def * MINISCOPE_FPS

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                #C = s.C
                #C = s.S_orig
                ##C = s.S
                S = s.S
                C = s.C
                YrA = s.YrA
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
                #C = s.S[indeces,:]
                S = s.S[indeces,:]
                C = s.C[indeces,:]
                YrA = s.YrA[indeces,:]
            V = s.velocities_miniscope_smooth

            if X_hist_use == 'S':
                X_hist = S
            elif X_hist_use == 'C':
                X_hist = C
            elif X_hist_use == 'YrA':
                X_hist = YrA

            if use_YrA:
                X_save = YrA
            else:
                X_save = X_hist
                
            PSTH_hist = np.zeros((X_hist.shape[0], frames_lookaround*2), dtype=float)
            PSTH_save = np.zeros((X_save.shape[0], save_len))

            V_hist = np.zeros(frames_lookaround*2)
            V_save = np.zeros(save_len)
            #onsets_wanted = [onsets[0]]
            #onsets_wanted = [onsets[-1]]            
            onsets_wanted = onsets

            eps_thresh = 0.0   # activity threshold
            counts = np.zeros(frames_lookaround*2, dtype=float)
            n_onsets = len(onsets_wanted)
            for on,on_num in zip(onsets_wanted,range(n_onsets)):
                print('on {}'.format(on), end='')
                #on = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround))
                period = range(on - frames_lookaround, on + frames_lookaround)
                period_save = range(on - frames_lookaround, min(on + save_post, C.shape[1]))

                # Get and save binary responses, only during positive deflections.
                win = X_hist[:,period]
                positive_deflections = np.diff(win, axis=1, prepend=win[:,:1])
                #positive_deflections = np.diff(X_hist[:,period],prepend=0)>0
                if spike_onset_only:
                    binary_responses = collapse_runs((positive_deflections > eps_thresh).astype(int))
                else:
                    binary_responses = (positive_deflections > eps_thresh).astype(int)
                #counts += binary_responses.sum(axis=0)
                #counts = binary_responses.sum(axis=0)
                #prob = counts / binary_responses.shape[0]
                PSTH_hist += binary_responses
                if len(period_save) < PSTH_save.shape[1]: # for G09
                    PSTH_save = PSTH_save + np.pad(X_save[:,period_save], ((0,0),(0,PSTH_save.shape[1] - len(period_save))))
                    V_save = V_save + np.pad(V[period_save], (0, len(V_save) - len(period_save))) 
                else:
                    PSTH_save = PSTH_save + X_save[:,period_save]
                    V_save = V_save + V[period_save]

            PSTH_hist /= n_onsets
            if normalize_velocity:
                PSTH_hist /= V[period]+1 # No
            if normalize_per_mouse:
                PSTH_hist = PSTH_hist.mean(axis=0)
            #counts / (n_onsets * X_hist.shape[0])
            #PSTH_hist /= len(onsets_wanted)
            #V_hist /= len(onsets_wanted)
            #V_save /= len(onsets_wanted)

            if group not in group_PSTH.keys():
                group_PSTH_hist[group] = PSTH_hist
                group_PSTH_hist_all[group] = PSTH_hist
                group_PSTH_hist_vel[group] = PSTH_hist / (1+V_hist)
                group_PSTH[group] = PSTH_save
                group_PSTH_vel[group] = PSTH_save / (1+V_save)
            else:
                group_PSTH_hist[group] = np.vstack((group_PSTH_hist[group], PSTH_hist))
                group_PSTH_hist_vel[group] = np.vstack((group_PSTH_hist_vel[group], PSTH_hist / (1+V_hist)))
                group_PSTH[group] = np.vstack((group_PSTH[group], PSTH_save))
                group_PSTH_vel[group] = np.vstack((group_PSTH_vel[group], PSTH_save / (1+V_save)))
            print('')

    # Plot all group_PSTH_hist in a single figure with subplots along one row
    groups = ['hM3D', 'hM4D', 'mCherry']
    fig, axs = plt.subplots(1, len(groups), figsize=(15, 4), sharey=sharey)
    norm_group_PSTH = dict()
    for ax, group in zip(axs, groups):
        Y = group_PSTH_hist[group]
        #Y = Y - np.mean(Y[:,0:frames_lookaround], axis=1, keepdims=True)
        mean = np.mean(Y, 0)[1:]
        sem = np.nanstd(Y, axis=0, ddof=1)[1:] / np.sqrt(np.sum(~np.isnan(Y), axis=0))[1:]
        ax.plot(mean)
        ax.fill_between(np.arange(len(mean)), mean - sem, mean + sem, alpha=0.2, color='b')
        ax.axvline(x=frames_lookaround, color='r', linestyle='--')
        ax.set_title(group)
        norm_group_PSTH[group] = Y
    plt.tight_layout()

    # Perform ANOVA on pre-stimulus (0..frames_lookaround) and post-stimulus (frames_lookaround..end) AOC

    # Get group data
    groups = ['hM3D', 'hM4D', 'mCherry']
    pre_aoc = []
    post_aoc = []

    for group in groups:
        Y = norm_group_PSTH[group]
        # Pre-stimulus: bins 0..frames_lookaround (exclusive)
        #pre_aoc.append(np.trapz(Y[:, :frames_lookaround], axis=1))
        pre_aoc.append(np.trapz(Y[:, :frames_lookaround//2:frames_lookaround], axis=1))
        # Post-stimulus: bins frames_lookaround..end
        #post_aoc.append(np.trapz(Y[:, frames_lookaround:], axis=1))
        post_aoc.append(np.trapz(Y[:, frames_lookaround//2:], axis=1))

    # Pre-stimulus ANOVA
    f_pre = stats.f_oneway(*pre_aoc)
    print("Pre-stimulus ANOVA (AOC 0..frames_lookaround):", f_pre)

    # Post-stimulus ANOVA
    f_post = stats.f_oneway(*post_aoc)
    print("Post-stimulus ANOVA (AOC frames_lookaround..end):", f_post)

    # Prepare data for Tukey HSD
    import statsmodels.stats.multicomp as mc

    # Flatten and label
    pre_data = np.concatenate(pre_aoc)
    pre_labels = np.concatenate([[g]*len(a) for g, a in zip(groups, pre_aoc)])
    post_data = np.concatenate(post_aoc)
    post_labels = np.concatenate([[g]*len(a) for g, a in zip(groups, post_aoc)])

    # Tukey HSD for pre-stimulus
    tukey_pre = mc.MultiComparison(pre_data, pre_labels).tukeyhsd()
    print("Pre-stimulus Tukey HSD:\n", tukey_pre.summary())

    # Tukey HSD for post-stimulus
    tukey_post = mc.MultiComparison(post_data, post_labels).tukeyhsd()
    print("Post-stimulus Tukey HSD:\n", tukey_post.summary())

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    bar_colors = [group_colours[g] for g in groups]

    # Pre-stimulus plot
    ax = axs[0]
    means = [np.mean(a) for a in pre_aoc]
    sems = [np.std(a)/np.sqrt(len(a)) for a in pre_aoc]
    bars = ax.bar(range(3), means, yerr=sems, color=bar_colors, capsize=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['hM3D', 'hM4D', 'mCherry'])
    ax.set_title('Pre-stimulus')
    ax.set_ylabel('AOC')

    # Post-stimulus plot
    ax = axs[1]
    means = [np.mean(a) for a in post_aoc]
    sems = [np.std(a)/np.sqrt(len(a)) for a in post_aoc]
    bars = ax.bar(range(3), means, yerr=sems, color=bar_colors, capsize=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['hM3D', 'hM4D', 'mCherry'])
    ax.set_title('Post-stimulus')

    # Helper for significance bars
    def add_sig_bar(ax, x1, x2, y, pval):
        barh = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        y = y + barh
        ax.plot([x1, x1, x2, x2], [y, y+barh, y+barh, y], lw=1.2, c='k')
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = ''
        if stars:
            ax.text((x1+x2)/2, y+barh, stars, ha='center', va='bottom', color='k', fontsize=12)

    from itertools import combinations
    pairs = list(combinations(range(len(groups)), 2))  # [(0,1), (0,2), (1,2)]

    # Add significance bars for pre-stimulus
    ax = axs[0]
    y_max = max([b.get_height() for b in ax.patches])
    for i, ((x1, x2), pval) in enumerate(zip(pairs, tukey_pre.pvalues)):
        if pval < 0.05:
            add_sig_bar(ax, x1, x2, y_max + i*0.1*y_max, pval)

    # Add significance bars for post-stimulus
    ax = axs[1]
    y_max = max([b.get_height() for b in ax.patches])
    for i, ((x1, x2), pval) in enumerate(zip(pairs, tukey_post.pvalues)):
        if pval < 0.05:
            add_sig_bar(ax, x1, x2, y_max + i*0.1*y_max, pval)

    plt.tight_layout()
    plt.show()
    if auto_close:
        plt.close()

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
    #
    # NB: G15 disabled due to s.C and s.YrA shape mismatch!! Fix then re-instate.
    #
    print('***PSTH_shuffle AVERAGED')
    snippet_len = frames_lookaround * 2
    save_post = frames_lookaround + frames_save
    save_len = frames_lookaround * 2 + frames_save
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    group_PSTH_all = dict() # PSTH for all responses not just significant ones    
    group_PSTH_vel = dict() # velocities for corresponding PSTH's
    group_vel = dict()
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
            if mouse in ['G15']:
                continue
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
                ##C = s.S
                C = s.C
                YrA = s.YrA
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
                #C = s.S[indeces,:]
                C = s.C[indeces,:]
                YrA = s.YrA[indeces,:]
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
                    C_responses_save = C_responses_save + np.pad(YrA[:,period_save], ((0,0),(0,C_responses_save.shape[1] - len(period_save)))) # C
                    V_save = V_save + np.pad(V[period_save], (0, len(V_save) - len(period_save)))
                else:
                    C_responses_save = C_responses_save + YrA[:,period_save] # C
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
                group_PSTH_vel[group] = C_responses_save[sig_cells_mouse,:] / (1+V_save)
                group_PSTH_all[group] = C_responses_save
                group_vel[group] = V_save
                group_percentiles[group] = dict()
            else:
                group_PSTH[group] = np.vstack((group_PSTH[group], C_responses_save[sig_cells_mouse,:]))
                group_PSTH_all[group] = np.vstack((group_PSTH_all[group], C_responses_save)) 
                group_PSTH_vel[group] = np.vstack((group_PSTH_vel[group], C_responses_save[sig_cells_mouse,:] / (1+V_save)))
                group_vel[group] = np.vstack((group_vel[group], V_save))
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
    plt.title('hM3D - normalized by velocities')
    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['hM4D'],0))
    plt.title('hM4D - normalized by velocities')
    plt.figure()
    plt.plot(np.mean(group_PSTH_vel['mCherry'],0))
    plt.title('mCherry - normalized by velocities')
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
            if mouse not in ['G15']:
                ax.hist(group_percentiles[group][mouse], bins=30, color='skyblue', edgecolor='black')
        ax.set_title('{}'.format(group), size='medium')
    if auto_close:
        plt.close()

    for group in mice_per_group:
        plt.figure()
        plt.title('{} velocity-transients'.format(group))
        for m in mice_per_group[group]:
            if m in ['G15']:
                continue
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


def process_PSTH_shuffle1(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim='shock', frames_lookaround=40, \
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
                ##C = s.S
                C = s.YrA
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
                #C = s.S[indeces,:]
                C = s.YrA[indeces,:]
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
            if mouse in ['G15']:
                continue
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
                C_save = s.YrA
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
            C_responses_save = np.zeros((C_save.shape[0], save_len))
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
                C_responses_save = C_responses_save + C_save[:,period_save]
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
        min_val = np.min(group_PSTH[group], axis=1)  # shape: (num_cells,)
        #min_val = np.mean(group_PSTH[group][:,0:group_PSTH[group].shape[1]//2], axis=1)  # baseline as mean of pre-stim period
        group_mean = np.mean(group_PSTH[group] - min_val[:, np.newaxis], axis=0)
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH[group], 0) / np.sqrt(group_PSTH[group].shape[0])
            #group_shaded = group_shaded - min_val
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH[group], 0)
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

def process_PSTH_shuffle_sep_S(PLOTS_DIR, mice_per_group, crossreg_mice, session, mapping_type, stim='shock', frames_lookaround=40, \
    frames_save=200, shaded='sem', num_shuffles=100, percentile=95.0, auto_close=True):
    '''
    'stim' should be strings corresponding to either 'tone' or 'shock'. Caller must make sure the session type matches
    the expected stim.

    Returns the cells that were plotted as a dict mapped from groups -> list of cell id's.

    This function calculates PSTH by considering each onset period separately (as opposed to averaged).

    Furthermore, it does not perform shuffling as we already have the "activity" from S matrix. Thus just calculate binary
    activation function directly from S.
    '''
    # PSTH_2 no averaging
    #snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    print('***PSTH_shuffle SEPARATE')
    snippet_len = frames_lookaround * 2
    save_post = frames_lookaround + frames_save
    save_len = frames_lookaround * 2 + frames_save * 2
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    group_PSTH_all = dict() # PSTH for all responses not just significant ones    
    group_PSTH_vel = dict() # velocities for corresponding PSTH's

    tot_cells = dict()
    tot_cells['hM3D'] = dict()
    tot_cells['hM4D'] = dict()
    tot_cells['mCherry'] = dict()

    resp_cells = dict()
    resp_cells['hM3D'] = dict()
    resp_cells['hM4D'] = dict()
    resp_cells['mCherry'] = dict()

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
            if mouse in ['G15']:
                continue
            print('{} {}'.format(mouse, group))
            s = session[mouse]
            crossreg = crossreg_mice[mouse]
            frac_tots[group][mouse] = []
            resp_cells[group][mouse] = []
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
                C_save = s.YrA
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
            C_responses_save = np.zeros((C_save.shape[0], save_len))
            V_save = np.zeros(save_len)
            resp_cells_total_mouse = np.ndarray(0)

            #onsets_wanted = [onsets[0]]
            #onsets_wanted = [onsets[-1]]
            onsets_wanted = onsets
            for on,on_num in zip(onsets_wanted,range(len(onsets_wanted))):
                print('on {}'.format(on), end='')
                #on = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround))
                period = range(on - frames_lookaround, on + frames_lookaround)
                period_save = range(on - frames_lookaround - frames_save, on + frames_lookaround + frames_save)
                #C_responses = C_responses + C[:,period]
                C_responses = C_responses + C[:,period]
                C_responses_save = C_responses_save + C_save[:,period_save]
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

                resp_cells_mouse = np.where((C_binary > 0)==True)[0]
                resp_cells_total_mouse = np.append(resp_cells_total_mouse, resp_cells_mouse)
                frac_tots[group][mouse].append(len(resp_cells_mouse) / C.shape[0])
                resp_cells[group][mouse].append(resp_cells_mouse)

                if group not in group_PSTH.keys():
                    group_PSTH[group] = C_responses_save[resp_cells_mouse,:]
                    group_PSTH_all[group] = C_responses_save
                    group_PSTH_vel[group] = V_save
                else:
                    group_PSTH[group] = np.vstack((group_PSTH[group], C_responses_save[resp_cells_mouse,:]))
                    group_PSTH_all[group] = np.vstack((group_PSTH_all[group], C_responses_save))
                    group_PSTH_vel[group] = np.vstack((group_PSTH_vel[group], V_save))  
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
    for group, ax in zip(mice_per_group.keys(), axs):
        min_val = np.min(group_PSTH[group], axis=1)
        group_mean = np.mean(group_PSTH[group] - min_val[:, np.newaxis], axis=0)
        if shaded == 'sem':
            group_shaded = np.std(group_PSTH[group], 0) / np.sqrt(group_PSTH[group].shape[0])
        if shaded == 'sd':
            group_shaded = np.std(group_PSTH[group], 0)
        ax.plot(group_mean, color='b', lw=1)
        ax.fill_between(range(save_len), group_mean - group_shaded, group_mean + group_shaded, alpha=0.2)
        ax.axvline(x=frames_lookaround+frames_save, color='r', linestyle='--')
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
    fig.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close(fig)

    return tot_cells, frac_tots, resp_cells
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
                #C = s.C
                C = s.YrA
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
                #C = s.C[indeces,:]
                C = s.YrA[indices,:]
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
        # subplot_tool() requires an interactive GUI backend (Tk/Qt). On a
        # headless server matplotlib uses Agg and the call raises. Skip it
        # there; it's only useful when interactively tweaking margins.
        if mpl.get_backend().lower() not in ('agg', 'pdf', 'ps', 'svg', 'cairo', 'module://matplotlib_inline.backend_inline'):
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
    PLOTS_DIR=None, want_3D=False, pcells_mice=dict(), max_fields=15, random_pcells_per_field=5, only_fm_pcells=False, print_pcell_maps=False, \
    loc_bounds=None, merge_distance=4, plot_pf_maps=True):
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
    if not only_fm_pcells and plot_pf_maps: # this switch allows us to skip a lot of the FM's that were already plotted and behaviour-based so don't change between analysis runs.
            
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
    if not plot_pf_maps:
        # Skip heavy per-cell plotting; jump straight to place field detection.
        fm.find_place_fields(sess, method='iterative_gauss', merge_distance=merge_distance)
        return

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

def plot_fluorescence_map(PLOTS_DIR, session, mouse_groups, session_str, bin_width=20, random_width=0, cells=np.array([]), want_3D=False, pcells_mice=dict(), \
    max_fields=15, only_fm_pcells=False, print_pcell_maps=False, merge_distance=4, plot_pf_maps=True):

    save_path = os.path.join(PLOTS_DIR, 'fluorescence_maps_{}fields_{}'.format(max_fields, session_str)) #'{}_{}_fluorescence-map.png'.format(session_str, m))
    os.makedirs(save_path, exist_ok=True)

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

    for m, sess in session.items():
        print('*** WORKING FLUORESCENCE MAPS FOR {}... '.format(m))
        plot_fluorescence_map_helper(m, sess, mouse_groups, session_str, bin_width, random_width=random_width, SAVE_PATH=save_path, cells=cells, PLOTS_DIR=PLOTS_DIR, \
            want_3D=want_3D, pcells_mice=pcells_mice, max_fields=max_fields, only_fm_pcells=only_fm_pcells, print_pcell_maps=print_pcell_maps, \
            loc_bounds=loc_bounds, plot_pf_maps=plot_pf_maps)

def save_fm_pf(session, session_str, PLOTS_DIR):
    """
    Save fm.pf.merged_means, fm.pf.model_, and fm.sig_responses
    for all mice in `session` into a single pickle file.

    Structure:
      fm_pf[mouse]["merged_means"]
      fm_pf[mouse]["model"]
      fm_pf[mouse]["sig_responses"]
    """

    out_dir = Path(PLOTS_DIR) / "fm_pf"
    out_dir.mkdir(parents=True, exist_ok=True)

    fm_pf = {}

    for mouse, obj in session.items():
        fm_pf[mouse] = {
            "merged_means": obj.fm.pf.merged_means,
            "model": obj.fm.pf.model_,
            "sig_responses": obj.fm.sig_responses,
            "fluorescence_map_occup": obj.fm.fluorescence_map_occup,
        }

    save_path = out_dir / f"fm_pf_{session_str}.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(fm_pf, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_path

def get_fluorescence_map(session, session_str, NPY_SAVE_PATH, PLOTS_DIR, mouse_groups, bin_width, random_width, want_3D, \
    max_fields, only_fm_pcells, print_pcell_maps, merge_distance, plot_pf_maps=True):
    session_dir = os.path.join(NPY_SAVE_PATH, f'fm_{session_str}')

    if os.path.isdir(session_dir):
        load_fm(session, session_str, NPY_SAVE_PATH)
        return None

    sig_responses = dict()
    plot_fluorescence_map(PLOTS_DIR, session, mouse_groups, session_str, bin_width=bin_width, random_width=random_width, want_3D=want_3D, \
        pcells_mice=sig_responses, max_fields=max_fields, only_fm_pcells=only_fm_pcells, print_pcell_maps=print_pcell_maps, merge_distance=merge_distance, \
        plot_pf_maps=plot_pf_maps)
    save_fm(session, session_str, NPY_SAVE_PATH)
    return sig_responses

def save_fm(session, session_str, PLOTS_DIR):
    """
    Save FluorescenceMap objects per mouse to:
        PLOTS_DIR / f"fm_{session_str}" / f"{mouse}_fm.pkl"

    Temporarily strips back-pointers:
        fm.sess = None
        fm.loc.sess = None
        fm.pf.sess = None
        fm.shuffled_responses = None

    Restores them after saving so the live session is unchanged.
    """
    out_dir = Path(PLOTS_DIR) / f"fm_{session_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mouse, sess_obj in session.items():
        fm = sess_obj.fm

        # backups
        fm_sess_backup = getattr(fm, "sess", None)
        loc_sess_backup = getattr(fm.loc, "sess", None) if hasattr(fm, "loc") else None
        pf_sess_backup  = getattr(fm.pf, "sess", None) if hasattr(fm, "pf") else None
        shuf_backup     = getattr(fm, "shuffled_responses", None)

        try:
            # strip
            if hasattr(fm, "sess"):
                fm.sess = None
            if hasattr(fm, "loc") and hasattr(fm.loc, "sess"):
                fm.loc.sess = None
            if hasattr(fm, "pf") and hasattr(fm.pf, "sess"):
                fm.pf.sess = None
            if hasattr(fm, "shuffled_responses"):
                fm.shuffled_responses = None

            save_path = out_dir / f"{mouse}_fm.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(fm, f, protocol=pickle.HIGHEST_PROTOCOL)

        finally:
            # restore live object
            if hasattr(fm, "sess"):
                fm.sess = fm_sess_backup
            if hasattr(fm, "loc") and hasattr(fm.loc, "sess"):
                fm.loc.sess = loc_sess_backup
            if hasattr(fm, "pf") and hasattr(fm.pf, "sess"):
                fm.pf.sess = pf_sess_backup
            if hasattr(fm, "shuffled_responses"):
                fm.shuffled_responses = shuf_backup

    return out_dir

def load_fm(session, session_str, PLOTS_DIR, *, strict=False):
    """
    Load saved FluorescenceMap objects and insert them back into the
    corresponding LinearTrackSession objects in-place:

        session[mouse].fm = loaded_fm
        loaded_fm.sess = session[mouse]
        loaded_fm.loc.sess = session[mouse]
        loaded_fm.pf.sess = session[mouse]

    shuffled_responses remains None (by design).
    """
    fm_dir = Path(PLOTS_DIR) / f"fm_{session_str}"
    loaded = {}

    for mouse, sess_obj in session.items():
        pkl_path = fm_dir / f"{mouse}_fm.pkl"

        if not pkl_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing fm pickle for {mouse}: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            fm = pickle.load(f)

        sess_obj.fm = fm

        # reattach back-pointers
        if hasattr(fm, "sess"):
            fm.sess = sess_obj
        if hasattr(fm, "loc") and hasattr(fm.loc, "sess"):
            fm.loc.sess = sess_obj
        if hasattr(fm, "pf") and hasattr(fm.pf, "sess"):
            fm.pf.sess = sess_obj

        loaded[mouse] = fm

    return loaded

def plot_pf_analyses(PLOTS_DIR, session, mouse_groups, session_str, crossreg=None, mapping=None, auto_close=True, DEBUG=False):
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
        if DEBUG:
            print('*** saving pf analysis figure to {}'.format(savefile))
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
    if crossreg:
        savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}_stats.png'.format('num_pfs', group, session_str, mapping))
    else:
        savefile = os.path.join(save_path, '{}_{}_{}_stats.png'.format('num_pfs', group, session_str))
    if DEBUG:
        print('*** saving pf analysis figure to {}'.format(savefile))
    plt.savefig(savefile, format='png', dpi=300)
    if auto_close:
        plt.close()

    stats.kstest(meas_pf_size['hM3D'], meas_pf_size['hM4D']) # 0.9732
    stats.kstest(meas_pf_size['hM3D'], meas_pf_size['mCherry']) # 0.6284
    stats.kstest(meas_pf_size['hM4D'], meas_pf_size['mCherry']) # 0.3301
    plt.figure(figsize=(4,3))
    bp = plt.boxplot([meas_pf_size['hM3D'], meas_pf_size['hM4D'], meas_pf_size['mCherry']], \
        notch=True, patch_artist=True, positions=[0.5,1,1.5])
    for p, c in zip(bp['boxes'], ['red','blue','black']):
        plt.setp(p,facecolor=c)
    plt.xticks([0.5,1,1.5],['hM3D','hM4D','mCherry'])
    if crossreg:
        savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}_stats.png'.format('pf_size', group, session_str, mapping))
    else:
        savefile = os.path.join(save_path, '{}_{}_{}_stats.png'.format('pf_size', group, session_str))
    if DEBUG:
        print('*** saving pf analysis figure to {}'.format(savefile))   
    if auto_close:
        plt.close()

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
    if crossreg:
        savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}_stats.png'.format('pf_compactness', group, session_str, mapping))
    else:
        savefile = os.path.join(save_path, '{}_{}_{}_stats.png'.format('pf_compactness', group, session_str))
    if DEBUG:
        print('*** saving pf analysis figure to {}'.format(savefile))
    plt.savefig(savefile, format='png', dpi=300)
    if auto_close:
        plt.close()

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
    if crossreg:
        savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}_stats.png'.format('spatial_selectivity', group, session_str, mapping))
    else:
        savefile = os.path.join(save_path, '{}_{}_{}_stats.png'.format('spatial_selectivity', group, session_str))
    if DEBUG:
        print('*** saving pf analysis figure to {}'.format(savefile))
    plt.savefig(savefile, format='png', dpi=300)
    if auto_close:
        plt.close()

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
    if crossreg:
        savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}_stats.png'.format('pf_compactness', group, session_str, mapping))
    else:
        savefile = os.path.join(save_path, '{}_{}_{}_stats.png'.format('pf_compactness', group, session_str))
    if DEBUG:
        print('*** saving pf analysis figure to {}'.format(savefile))        
    plt.savefig(savefile, format='png', dpi=300)
    if auto_close:
        plt.close()

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
    if crossreg:
        savefile = os.path.join(save_path, '{}_{}_{}_crossreg_{}_stats.png'.format('pf_compactness_lt_1', group, session_str, mapping))
    else:
        savefile = os.path.join(save_path, '{}_{}_{}_stats.png'.format('pf_compactness_lt_1', group, session_str))
    if DEBUG:
        print('*** saving pf analysis figure to {}'.format(savefile))        
    plt.savefig(savefile, format='png', dpi=300)
    if auto_close:
        plt.close()

def plot_pf_analyses_within_group(PLOTS_DIR, session_1, session_2, session_3, mouse_groups, crossreg=None, mapping=None, auto_close=True, sess_names=None, skip_mice=None):
    save_path = os.path.join(PLOTS_DIR, 'place_fields_groups')
    os.makedirs(save_path, exist_ok=True)

    if sess_names is None:
        sess_names = ['TFC_cond', 'Test_B', 'Test_B_1wk']
    if skip_mice is None:
        skip_mice = ['G07', 'G15']

    _default_colours = ['k', 'gray', 'gainsboro']
    sess_colours = {name: _default_colours[i] for i, name in enumerate(sess_names)}

    meas_num_pfs = {}
    meas_pf_size = {}
    meas_compactness_pf = {}
    meas_spatial_selectivity = {}
    processed_groups = {}   # placeholder so we know we encountered each group when going through the mice. Just assign True to each group once here.

    for mouse, group in mouse_groups.items():
        if mouse in skip_mice:
            continue
        if group not in processed_groups:
            processed_groups[group] = True
            meas_num_pfs[group] = {}
            meas_pf_size[group] = {}
            meas_compactness_pf[group] = {}
            meas_spatial_selectivity[group] = {}
            
        for session, sess_name in zip([session_1, session_2, session_3], sess_names):
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
             
def plot_lt_spatial_responses_prev(
    PLOTS_DIR,
    LT1_group,
    LT2_group,
    mouse_groups,
    session_str,
    mapping=None,
    use_sig_responses=False,
    want_sanity_sanity=True,
    how_many_sanity_sanity_cells=3,
    auto_close=True,
    want_S=False,
    want_C=False,
    want_YrA=False,
    normalize_global=False,      # NEW
    normalize_per_mouse=False,   # NEW
):
    """
    Plot spatial responses for cross-registered cells between LT1 and LT2, sorted according to LT1.

    NEW normalization modes (mutually exclusive; priority: global > per_mouse > per_cell):
      - normalize_global=True:
          Heatmaps normalized by a single global vmax computed across ALL mice (LT1+LT2).
      - normalize_per_mouse=True:
          Heatmaps normalized per mouse by vmax computed across LT1+LT2 for that mouse only.
      - Else (default behavior):
          keep prior normalize_per_cell=True row-wise normalization for display.

    Returns:
      pv_corr_LT1_LT2 : dict keyed by mouse.
        pv_corr_LT1_LT2[mouse] = {
            "pv_corr": (n_bins, n_bins) PV correlation matrix,
            "bin_edges_LT1": edges used for LT1 PV bins,
            "bin_edges_LT2": edges used for LT2 PV bins,
            "bin_centers_LT1": centers,
            "bin_centers_LT2": centers,
            "sorted_LT1": list of LT1 cell ids (sorted order),
            "sorted_LT2": list of LT2 cell ids aligned to sorted_LT1,
        }
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from numpy.random import default_rng  # FIX: you use default_rng() later

    if mapping is None:
        raise ValueError("plot_lt_spatial_responses: mapping must not be None")

    if normalize_global and normalize_per_mouse:
        raise ValueError("Choose only one: normalize_global or normalize_per_mouse (not both).")

    if not (want_S or want_C or want_YrA):
        want_S = True
    S_file_str = "S" if want_S else ("C" if want_C else "YrA")

    save_path = os.path.join(PLOTS_DIR, f"lt_spatial_responses_{session_str}_mapping_{mapping}")
    os.makedirs(save_path, exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def _safe_corr(a, b):
        """Pearson corr with NaN handling; returns np.nan if too few points."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 3:
            return np.nan
        aa = a[m]
        bb = b[m]
        aa = aa - np.nanmean(aa)
        bb = bb - np.nanmean(bb)
        denom = np.sqrt(np.nansum(aa * aa) * np.nansum(bb * bb))
        if denom <= 0 or not np.isfinite(denom):
            return np.nan
        return float(np.nansum(aa * bb) / denom)

    def _smooth_rows(mat, smooth_bins):
        if smooth_bins is None or smooth_bins <= 1:
            return mat
        k = np.ones(int(smooth_bins), float) / float(smooth_bins)
        pad = int(smooth_bins) // 2
        mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode="edge")
        return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, mat_pad)

    def _normalize_per_cell(mat):
        out = mat.copy()
        row_max = np.nanmax(out, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = np.nan
        row_max[row_max == 0] = np.nan
        out = out / row_max
        return out

    def _compute_tuning(S_sorted, pos, bin_edges):
        """Mean activity per position bin for each cell."""
        pos = np.asarray(pos, float)
        valid = np.isfinite(pos)
        pos_v = pos[valid]
        S_v = S_sorted[:, valid]

        bin_idx = np.digitize(pos_v, bin_edges) - 1
        n_bins = len(bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        tuning = np.full((S_v.shape[0], n_bins), np.nan, float)
        for b in range(n_bins):
            m = (bin_idx == b)
            if not np.any(m):
                continue
            tuning[:, b] = np.nanmean(S_v[:, m], axis=1)
        return tuning

    def _compute_turn_lines_safe(
        loc_X,
        loc_Y,
        loc_1d,
        arm_cutoff_perc=0.8,
        band_fracs=(0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18),
        min_pts_per_arm=10,
    ):
        """
        Robustly compute turn lines (two 1D positions) using x-threshold + top/bottom arms.
        Never raises. Returns dict with turn1_1d/turn2_1d or None if failed.
        """
        x = np.asarray(loc_X, dtype=float)
        y = np.asarray(loc_Y, dtype=float)
        p1d = np.asarray(loc_1d, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(p1d)
        if np.sum(valid) < 50:
            return {"turn1_1d": None, "turn2_1d": None}

        xv, yv, p1dv = x[valid], y[valid], p1d[valid]
        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < 1e-6:
            return {"turn1_1d": None, "turn2_1d": None}

        x_thresh = x_min + float(arm_cutoff_perc) * (x_max - x_min)

        y_lo_arm = np.nanpercentile(yv, 40)
        y_hi_arm = np.nanpercentile(yv, 60)
        mask_top = yv >= y_hi_arm
        mask_bot = yv <= y_lo_arm

        for bf in band_fracs:
            x_band = float(bf) * (x_max - x_min)
            mask_xband = np.abs(xv - x_thresh) <= x_band
            mask_top_turn = mask_top & mask_xband
            mask_bot_turn = mask_bot & mask_xband
            n_top = int(np.sum(mask_top_turn))
            n_bot = int(np.sum(mask_bot_turn))

            if (n_top >= min_pts_per_arm) and (n_bot >= min_pts_per_arm):
                turn_top_1d = float(np.nanmedian(p1dv[mask_top_turn]))
                turn_bot_1d = float(np.nanmedian(p1dv[mask_bot_turn]))
                turns = np.sort([turn_top_1d, turn_bot_1d])
                return {
                    "turn1_1d": float(turns[0]),
                    "turn2_1d": float(turns[1]),
                    "x_thresh": float(x_thresh),
                    "x_band": float(x_band),
                    "band_frac": float(bf),
                    "xv": xv,
                    "yv": yv,
                    "mask_top_turn": mask_top_turn,
                    "mask_bot_turn": mask_bot_turn,
                    "n_top": n_top,
                    "n_bot": n_bot,
                }

        return {
            "turn1_1d": None,
            "turn2_1d": None,
            "x_thresh": float(x_thresh),
            "band_frac": None,
            "xv": xv,
            "yv": yv,
        }

    def _get_activity_matrix(LT):
        if want_S:
            return np.asarray(LT.S)
        if want_C:
            return np.asarray(LT.C)
        if want_YrA:
            return np.asarray(LT.YrA)
        return np.asarray(LT.S)

    # -------------------------
    # Parameters (kept same as your current working defaults)
    # -------------------------
    sort_reverse = False
    num_pfs_filtered = -1

    n_pos_bins = 120
    smooth_bins = 3

    # Your prior default behavior was "per-cell normalized display"
    # If we are doing global/per-mouse normalization, we should NOT also do per-cell.
    normalize_per_cell = not (normalize_global or normalize_per_mouse)

    arm_cutoff_perc = 0.8

    # -------------------------
    # PASS 1: compute and cache tunings + bin edges + sorted mapping, and global vmax if needed
    # -------------------------
    cache = {}
    global_vmax = np.nan

    for mouse, group in mouse_groups.items():
        LT1 = LT1_group[mouse]
        LT2 = LT2_group[mouse]

        loc_X_LT1 = LT1.loc_X_miniscope_smooth
        loc_Y_LT1 = LT1.loc_Y_miniscope_smooth
        loc_1d_LT1 = LT1.miniscope_loc_1d_px * _LT_1D_CM_PER_PX

        loc_X_LT2 = LT2.loc_X_miniscope_smooth
        loc_Y_LT2 = LT2.loc_Y_miniscope_smooth
        loc_1d_LT2 = LT2.miniscope_loc_1d_px * _LT_1D_CM_PER_PX

        df_mapping = LT1.crossreg.get_mappings_cells(mapping_type=mapping)
        cells_LT1 = df_mapping[LT1.session_group].astype(float).astype(int).tolist()
        cells_LT2 = df_mapping[LT2.session_group].astype(float).astype(int).tolist()

        # SORT by LT1 PF centers
        XY_LT1 = np.column_stack([loc_X_LT1, loc_Y_LT1])
        track_ref = np.nanmax(loc_1d_LT1) if sort_reverse else np.nanmin(loc_1d_LT1)

        bin_w = float(LT1.fm.loc.bin_width)
        min_x = float(getattr(LT1.fm.loc, "MIN_X"))
        min_y = float(getattr(LT1.fm.loc, "MIN_Y"))

        sorted_entries = []
        for cell_id in cells_LT1:
            merged = LT1.fm.pf.merged_means.get(cell_id, None)
            model = LT1.fm.pf.model_.get(cell_id, None)
            sig = LT1.fm.sig_responses.get(cell_id, None)

            if merged is None or model is None or not hasattr(model, "means_"):
                continue

            merged_means = merged
            means_rc = np.asarray(model.means_, dtype=float)

            if num_pfs_filtered != -1:
                if len(merged_means) != int(num_pfs_filtered):
                    continue

            mean_to_group = {}
            for g_idx, grp in enumerate(merged_means):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)

            pts_by_group = [[] for _ in range(len(merged_means))]
            use_sig = (sig is not None and len(sig) > 0)

            if use_sig:
                sig = np.asarray(sig, dtype=float)
                rc = sig[:, :2]
                pred_mean_idx = np.asarray(model.predict(rc), dtype=int)

                for (r, c), m_idx in zip(rc, pred_mean_idx):
                    g = mean_to_group.get(int(m_idx), None)
                    if g is None:
                        continue
                    pts_by_group[g].append((float(r), float(c)))

            pf_centers = []
            for grp_i, grp in enumerate(merged_means):
                grp = list(grp)
                if len(grp) == 0:
                    continue

                pts = pts_by_group[grp_i]
                if len(pts) > 0:
                    pts = np.asarray(pts, dtype=float)
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

            pf_candidates = []
            for grp_i, ctr_xy in pf_centers:
                d2 = np.sum((XY_LT1 - ctr_xy[None, :]) ** 2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_loc1d = float(loc_1d_LT1[nn_idx])
                pf_candidates.append((grp_i, ctr_xy, nn_idx, nn_loc1d))

            if len(pf_candidates) == 0:
                continue

            if num_pfs_filtered == -1:
                chosen = min(pf_candidates, key=lambda x: abs(x[-1] - track_ref))
            else:
                chosen = min(pf_candidates, key=lambda x: np.sum((XY_LT1[x[2]] - x[1]) ** 2))

            chosen_grp_i, chosen_ctr_xy, nn_idx, nn_loc1d = chosen
            sorted_entries.append((cell_id, chosen_grp_i, chosen_ctr_xy, nn_idx, nn_loc1d))

        sorted_entries.sort(key=lambda x: x[-1], reverse=bool(sort_reverse))
        sorted_LT1 = [cell_id for (cell_id, *_rest) in sorted_entries]

        lt1_to_lt2 = {c1: c2 for c1, c2 in zip(cells_LT1, cells_LT2)}
        sorted_LT2 = [lt1_to_lt2[c1] for c1 in sorted_LT1 if c1 in lt1_to_lt2]
        if len(sorted_LT1) == 0 or len(sorted_LT2) == 0:
            print(f"[WARN] Empty sorted list for mouse={mouse}. Skipping.")
            continue

        # Extract activity matrices
        S_LT1 = _get_activity_matrix(LT1)
        S_LT2 = _get_activity_matrix(LT2)  # FIX: match LT1 selection

        pos1 = np.asarray(loc_1d_LT1, float)
        pos2 = np.asarray(loc_1d_LT2, float)

        n_cells_S1, _T1 = S_LT1.shape
        n_cells_S2, _T2 = S_LT2.shape

        if all(isinstance(c, (int, np.integer)) for c in sorted_LT1) and max(sorted_LT1) < n_cells_S1:
            row_idx1 = [int(c) for c in sorted_LT1]
        else:
            cell_to_row1 = {cell_id: i for i, cell_id in enumerate(cells_LT1)}
            row_idx1 = [cell_to_row1[c] for c in sorted_LT1 if c in cell_to_row1]

        if all(isinstance(c, (int, np.integer)) for c in sorted_LT2) and max(sorted_LT2) < n_cells_S2:
            row_idx2 = [int(c) for c in sorted_LT2]
        else:
            cell_to_row2 = {cell_id: i for i, cell_id in enumerate(cells_LT2)}
            row_idx2 = [cell_to_row2[c] for c in sorted_LT2 if c in cell_to_row2]

        S1_sorted = S_LT1[row_idx1, :]
        S2_sorted = S_LT2[row_idx2, :]

        n_rows = min(S1_sorted.shape[0], S2_sorted.shape[0])
        S1_sorted = S1_sorted[:n_rows, :]
        S2_sorted = S2_sorted[:n_rows, :]
        sorted_LT1 = sorted_LT1[:n_rows]
        sorted_LT2 = sorted_LT2[:n_rows]

        pos1_v = pos1[np.isfinite(pos1)]
        pos2_v = pos2[np.isfinite(pos2)]
        if pos1_v.size < 10 or pos2_v.size < 10:
            print(f"[WARN] Not enough finite positions for mouse={mouse}. Skipping.")
            continue

        bin_edges1 = np.linspace(float(np.nanmin(pos1_v)), float(np.nanmax(pos1_v)), n_pos_bins + 1)
        bin_edges2 = np.linspace(float(np.nanmin(pos2_v)), float(np.nanmax(pos2_v)), n_pos_bins + 1)
        bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
        bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

        tuning1 = _compute_tuning(S1_sorted, pos1, bin_edges1)
        tuning2 = _compute_tuning(S2_sorted, pos2, bin_edges2)

        tuning1 = _smooth_rows(tuning1, smooth_bins)
        tuning2 = _smooth_rows(tuning2, smooth_bins)

        if normalize_global:
            vmax_local = np.nanmax([np.nanmax(tuning1), np.nanmax(tuning2)])
            if np.isfinite(vmax_local):
                global_vmax = vmax_local if not np.isfinite(global_vmax) else max(global_vmax, vmax_local)

        cache[mouse] = dict(
            group=group,
            LT1=LT1,
            LT2=LT2,
            loc_X_LT1=loc_X_LT1,
            loc_Y_LT1=loc_Y_LT1,
            loc_1d_LT1=loc_1d_LT1,
            loc_X_LT2=loc_X_LT2,
            loc_Y_LT2=loc_Y_LT2,
            loc_1d_LT2=loc_1d_LT2,
            cells_LT1=cells_LT1,
            cells_LT2=cells_LT2,
            sorted_LT1=sorted_LT1,
            sorted_LT2=sorted_LT2,
            tuning1=tuning1,
            tuning2=tuning2,
            bin_edges1=bin_edges1,
            bin_edges2=bin_edges2,
            bin_centers1=bin_centers1,
            bin_centers2=bin_centers2,
        )

    if normalize_global and (not np.isfinite(global_vmax) or global_vmax <= 0):
        print("[WARN] normalize_global=True but global_vmax is not finite/positive. Falling back to per-cell normalization.")
        normalize_global = False
        normalize_per_mouse = False
        normalize_per_cell = True

    # -------------------------
    # PASS 2: plotting + PV corr + sanity
    # -------------------------
    pv_corr_LT1_LT2 = {}

    # Colormaps (NaN -> black)
    cmap_pf = mpl.cm.get_cmap("viridis").copy()
    cmap_pf.set_bad(color="black")
    cmap_pv = mpl.cm.get_cmap("viridis").copy()
    cmap_pv.set_bad(color="black")

    for mouse, payload in cache.items():
        group = payload["group"]
        LT1 = payload["LT1"]
        LT2 = payload["LT2"]

        loc_X_LT1 = payload["loc_X_LT1"]
        loc_Y_LT1 = payload["loc_Y_LT1"]
        loc_1d_LT1 = payload["loc_1d_LT1"]
        loc_X_LT2 = payload["loc_X_LT2"]
        loc_Y_LT2 = payload["loc_Y_LT2"]
        loc_1d_LT2 = payload["loc_1d_LT2"]

        cells_LT1 = payload["cells_LT1"]
        sorted_LT1 = payload["sorted_LT1"]
        sorted_LT2 = payload["sorted_LT2"]

        tuning1 = payload["tuning1"]
        tuning2 = payload["tuning2"]
        bin_edges1 = payload["bin_edges1"]
        bin_edges2 = payload["bin_edges2"]
        bin_centers1 = payload["bin_centers1"]
        bin_centers2 = payload["bin_centers2"]

        # Turn lines (safe)
        turn1_info = _compute_turn_lines_safe(
            loc_X_LT1, loc_Y_LT1, loc_1d_LT1, arm_cutoff_perc=arm_cutoff_perc
        )
        turn2_info = _compute_turn_lines_safe(
            loc_X_LT2, loc_Y_LT2, loc_1d_LT2, arm_cutoff_perc=arm_cutoff_perc
        )
        lt1_turn_a, lt1_turn_b = turn1_info.get("turn1_1d"), turn1_info.get("turn2_1d")
        lt2_turn_a, lt2_turn_b = turn2_info.get("turn1_1d"), turn2_info.get("turn2_1d")

        if lt1_turn_a is None or lt1_turn_b is None:
            print(f"[WARN] LT1 turn detection failed for mouse={mouse}. No LT1 turn lines.")
        if lt2_turn_a is None or lt2_turn_b is None:
            print(f"[WARN] LT2 turn detection failed for mouse={mouse}. No LT2 turn lines.")

        # Optional sig filtering hook (unchanged)
        if use_sig_responses:
            print("[INFO] use_sig_responses=True is not implemented in this compact full-function version.")
            print("       If you truly need it back, tell me and I’ll splice your older block in verbatim.")

        # -------------------------
        # Build plot matrices according to normalization mode
        # -------------------------
        if normalize_global:
            scale = float(global_vmax)
            tuning1_plot = tuning1 / scale
            tuning2_plot = tuning2 / scale
            vmin, vmax = 0.0, 1.0
        elif normalize_per_mouse:
            mouse_vmax = np.nanmax([np.nanmax(tuning1), np.nanmax(tuning2)])
            if not np.isfinite(mouse_vmax) or mouse_vmax <= 0:
                # fallback
                tuning1_plot = _normalize_per_cell(tuning1)
                tuning2_plot = _normalize_per_cell(tuning2)
                vmin, vmax = 0.0, 1.0
            else:
                scale = float(mouse_vmax)
                tuning1_plot = tuning1 / scale
                tuning2_plot = tuning2 / scale
                vmin, vmax = 0.0, 1.0
        else:
            # Original behavior: per-cell normalization for display
            tuning1_plot = _normalize_per_cell(tuning1) if normalize_per_cell else tuning1
            tuning2_plot = _normalize_per_cell(tuning2) if normalize_per_cell else tuning2
            vmin, vmax = (0.0, 1.0) if normalize_per_cell else (None, None)

        # -------------------------
        # PV correlation matrix (use *raw* tunings, not plot-normalized)
        # -------------------------
        n_pos_bins = tuning1.shape[1]
        pv_corr = np.full((n_pos_bins, n_pos_bins), np.nan, float)
        for i in range(n_pos_bins):
            v1 = tuning1[:, i]
            for j in range(n_pos_bins):
                v2 = tuning2[:, j]
                pv_corr[i, j] = _safe_corr(v1, v2)

        pv_corr_LT1_LT2[mouse] = {
            "pv_corr": pv_corr,
            "bin_edges_LT1": bin_edges1,
            "bin_edges_LT2": bin_edges2,
            "bin_centers_LT1": bin_centers1,
            "bin_centers_LT2": bin_centers2,
            "sorted_LT1": sorted_LT1,
            "sorted_LT2": sorted_LT2,
        }

        # -------------------------
        # Figure 1: two-panel tuning heatmaps (no colorbar)
        # -------------------------
        fig, axes = plt.subplots(
            1, 2, figsize=(8.2, 7.5), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08}
        )

        axes[0].imshow(
            tuning1_plot,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap_pf,
            vmin=vmin,
            vmax=vmax,
            extent=[float(bin_edges1[0]), float(bin_edges1[-1]), tuning1_plot.shape[0], 1],
        )
        axes[0].set_title(f"{LT1.session_group}")
        axes[0].set_xlabel(f"Linearized track position ({_LT_1D_DISTANCE_UNIT})")
        axes[0].set_ylabel("Place cells")
        axes[0].set_yticks([1, tuning1_plot.shape[0]])
        axes[0].set_yticklabels(["1", f"{tuning1_plot.shape[0]}"])
        if (lt1_turn_a is not None) and (lt1_turn_b is not None):
            axes[0].axvline(lt1_turn_a, linestyle="--", linewidth=1.5)
            axes[0].axvline(lt1_turn_b, linestyle="--", linewidth=1.5)

        axes[1].imshow(
            tuning2_plot,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap_pf,
            vmin=vmin,
            vmax=vmax,
            extent=[float(bin_edges2[0]), float(bin_edges2[-1]), tuning2_plot.shape[0], 1],
        )
        axes[1].set_title(f"{LT2.session_group}")
        axes[1].set_xlabel(f"Linearized track position ({_LT_1D_DISTANCE_UNIT})")
        axes[1].set_yticks([1, tuning2_plot.shape[0]])
        axes[1].set_yticklabels(["1", f"{tuning2_plot.shape[0]}"])
        if (lt2_turn_a is not None) and (lt2_turn_b is not None):
            axes[1].axvline(lt2_turn_a, linestyle="--", linewidth=1.5)
            axes[1].axvline(lt2_turn_b, linestyle="--", linewidth=1.5)

        norm_tag = ("global" if normalize_global else ("perMouse" if normalize_per_mouse else "perCell"))
        fig.suptitle(f"{mouse} | {session_str} | LT1-sorted crossreg {mapping} | norm={norm_tag}", y=0.99)
        fig.tight_layout()

        savefile = os.path.join(
            save_path,
            f"tuning_LT1_LT2_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}_norm-{norm_tag}.png",
        )
        fig.savefig(savefile, dpi=300)
        plt.show()
        if auto_close:
            plt.close(fig)

        # -------------------------
        # Figure 2: PV correlation matrix
        # -------------------------
        fig2, ax = plt.subplots(1, 1, figsize=(4.8, 4.2), dpi=200)
        im = ax.imshow(
            pv_corr,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            cmap=cmap_pv,
            extent=[float(bin_edges2[0]), float(bin_edges2[-1]), float(bin_edges1[-1]), float(bin_edges1[0])],
        )
        ax.set_title(f"PV corr: {LT1.session_group} vs {LT2.session_group}")
        ax.set_xlabel(f"LT2 position ({_LT_1D_DISTANCE_UNIT})")
        ax.set_ylabel(f"LT1 position ({_LT_1D_DISTANCE_UNIT})")
        fig2.tight_layout()

        savefile2 = os.path.join(
            save_path,
            f"pv_corr_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}.png",
        )
        fig2.savefig(savefile2, dpi=300)
        plt.show()
        if auto_close:
            plt.close(fig2)

        # -------------------------
        # OPTIONAL: Turn sanity plots (safe; no crash)
        # -------------------------
        if want_sanity_sanity:
            # LT1
            if (turn1_info.get("turn1_1d") is not None) and ("xv" in turn1_info):
                try:
                    xv = turn1_info["xv"]
                    yv = turn1_info["yv"]
                    x_thresh = turn1_info["x_thresh"]
                    mt = turn1_info.get("mask_top_turn", None)
                    mb = turn1_info.get("mask_bot_turn", None)

                    figS, axS = plt.subplots(1, 1, figsize=(10, 3), dpi=150)
                    axS.scatter(xv, yv, s=4, alpha=0.25)
                    axS.axvline(x_thresh, linestyle="--", linewidth=2)
                    if mt is not None:
                        axS.scatter(xv[mt], yv[mt], s=10, alpha=0.8, label="LT1 top-arm near x_thresh")
                    if mb is not None:
                        axS.scatter(xv[mb], yv[mb], s=10, alpha=0.8, label="LT1 bottom-arm near x_thresh")
                    axS.set_title("LT1 turn cutoff sanity check")
                    axS.set_xlabel("X")
                    axS.set_ylabel("Y")
                    axS.legend(loc="upper left", fontsize=8, frameon=False)
                    figS.tight_layout()

                    sfile = os.path.join(
                        save_path,
                        f"turn_sanity_LT1_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}.png",
                    )
                    figS.savefig(sfile, dpi=300)
                    plt.show()
                    if auto_close:
                        plt.close(figS)
                except Exception as e:
                    print(f"[WARN] LT1 turn sanity plot skipped (error): {e}")

            # LT2
            if (turn2_info.get("turn1_1d") is not None) and ("xv" in turn2_info):
                try:
                    xv = turn2_info["xv"]
                    yv = turn2_info["yv"]
                    x_thresh = turn2_info["x_thresh"]
                    mt = turn2_info.get("mask_top_turn", None)
                    mb = turn2_info.get("mask_bot_turn", None)

                    figS, axS = plt.subplots(1, 1, figsize=(10, 3), dpi=150)
                    axS.scatter(xv, yv, s=4, alpha=0.25)
                    axS.axvline(x_thresh, linestyle="--", linewidth=2)
                    if mt is not None:
                        axS.scatter(xv[mt], yv[mt], s=10, alpha=0.8, label="LT2 top-arm near x_thresh")
                    if mb is not None:
                        axS.scatter(xv[mb], yv[mb], s=10, alpha=0.8, label="LT2 bottom-arm near x_thresh")
                    axS.set_title("LT2 turn cutoff sanity check")
                    axS.set_xlabel("X")
                    axS.set_ylabel("Y")
                    axS.legend(loc="upper left", fontsize=8, frameon=False)
                    figS.tight_layout()

                    sfile = os.path.join(
                        save_path,
                        f"turn_sanity_LT2_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}.png",
                    )
                    figS.savefig(sfile, dpi=300)
                    plt.show()
                    if auto_close:
                        plt.close(figS)
                except Exception as e:
                    print(f"[WARN] LT2 turn sanity plot skipped (error): {e}")

        # -------------------------
        # OPTIONAL: "sanity_sanity" PF plot(s)
        # -------------------------
        if want_sanity_sanity:
            rng = default_rng()
            intersection_keys = np.intersect1d(list(LT1.fm.sig_responses.keys()), np.array(payload["cells_LT1"]))
            if intersection_keys.size > 0:
                sanity_sanity_cells = rng.choice(
                    intersection_keys,
                    size=min(how_many_sanity_sanity_cells, len(intersection_keys)),
                    replace=False,
                )
                for cell_id_plot in sanity_sanity_cells:
                    try:
                        F = LT1.fm.fluorescence_map_occup[:, :, cell_id_plot]
                        sig = LT1.fm.sig_responses.get(cell_id_plot, [])
                        sig = np.asarray(sig, dtype=float) if len(sig) > 0 else np.zeros((0, 3), dtype=float)
                        sr = sig[:, 0] if sig.shape[0] > 0 else np.array([])
                        sc = sig[:, 1] if sig.shape[0] > 0 else np.array([])

                        means_rc = np.asarray(LT1.fm.pf.model_[cell_id_plot].means_)
                        merged = LT1.fm.pf.merged_means[cell_id_plot]

                        pf_centers = []
                        for grp in merged:
                            grp = list(grp)
                            if len(grp) == 0:
                                continue
                            ctr_rc = means_rc[grp].mean(axis=0)
                            pf_centers.append(ctr_rc)
                        pf_centers = np.asarray(pf_centers) if len(pf_centers) > 0 else np.zeros((0, 2), dtype=float)

                        pfr = pf_centers[:, 0] if pf_centers.size else np.array([])
                        pfc = pf_centers[:, 1] if pf_centers.size else np.array([])

                        figC, axC = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
                        axC.imshow(F, origin="upper", aspect="auto", cmap="viridis")

                        if sig.shape[0] > 0:
                            axC.scatter(sc, sr, s=45, marker="v", alpha=0.9)

                        if pf_centers.size > 0:
                            axC.scatter(
                                pfc, pfr,
                                s=200, marker="o",
                                edgecolors="white", linewidths=2,
                                alpha=0.95
                            )

                        axC.set_title(f"cell {cell_id_plot} place fields")
                        axC.set_xlabel("Column bin (horizontal)")
                        axC.set_ylabel("Row bin (vertical)")
                        figC.tight_layout()

                        sfile = os.path.join(
                            save_path,
                            f"pf_means_sanity_{mouse}_{group}_{session_str}_crossreg_{mapping}_cell{cell_id_plot}_{S_file_str}.png",
                        )
                        figC.savefig(sfile, dpi=300)
                        plt.show()
                        if auto_close:
                            plt.close(figC)
                    except Exception as e:
                        print(f"[WARN] PF sanity plot skipped for cell={cell_id_plot}: {e}")

    return pv_corr_LT1_LT2

def fisher_z_mean_r(values, axis=0):
    """
    Fisher z mean of Pearson r values with NaNs allowed.
    - clips r to (-1+eps, 1-eps) to avoid atanh inf
    - ignores NaNs in the mean
    """
    import numpy as np
    v = np.asarray(values, float)
    v = np.clip(v, -0.999999, 0.999999)
    z = np.arctanh(v)
    z_mean = np.nanmean(z, axis=axis)
    return np.tanh(z_mean)


def pv_group_summary(pv_corr_LT1_LT2):
    """
    Compute group summaries across mice:
      - group_pv_corr: Fisher-z mean PV matrix
      - group_pv_diag: Fisher-z mean of per-mouse diagonal summary (pv_diag)
      - per_mouse_diag: dict mouse -> pv_diag
    """
    import numpy as np

    mats = []
    diags = []
    per_mouse_diag = {}
    for mouse, d in pv_corr_LT1_LT2.items():
        M = d.get("pv_corr", None)
        if M is None:
            continue
        mats.append(M)

        pv_diag = d.get("pv_diag", np.nan)
        per_mouse_diag[mouse] = pv_diag
        diags.append(pv_diag)

    if len(mats) == 0:
        return None, float("nan"), per_mouse_diag

    mats = np.stack(mats, axis=0)  # (n_mice, n_bins, n_bins)
    group_pv_corr = fisher_z_mean_r(mats, axis=0)

    diags = np.asarray(diags, float)
    group_pv_diag = fisher_z_mean_r(diags, axis=0)

    return group_pv_corr, float(group_pv_diag), per_mouse_diag

import re

def _mapping_tag(mapping):
    """
    Convert mapping into a filesystem-safe tag string.
    Examples:
      None -> "full"
      "full" -> "full"
      "LT1+LT2" -> "LT1pLT2"
      "TFC_cond+LT1+LT2" -> "TFC_condpLT1pLT2"
    """
    if mapping is None:
        m = "full"
    else:
        m = str(mapping)

    if m.strip() == "" or m == "None":
        m = "full"

    # normalize common tokens
    m = m.replace("+", "p")  # "plus"
    # replace any remaining bad characters with underscore
    m = re.sub(r"[^A-Za-z0-9._-]+", "_", m)
    return m

def _get_mapping_cells_for_mouse(LT, mouse, mapping):
    """
    mapping can be:
        - None / 'full' -> empty set meaning 'no restriction'
        - str -> mapping_type used by LT.crossreg.get_mappings_cells()
        - dict-like -> mapping[mouse] is iterable of cell ids
        - DataFrame-like -> columns include mouse and cell_id
    Returns a set of cell IDs (unit IDs) allowed by mapping.
    """
    if mapping is None or mapping == "full":
        return set()  # signals "no restriction"

    # Case 1: mapping is a mapping_type string like 'LT1+LT2'
    if isinstance(mapping, str):
        if not hasattr(LT, "crossreg") or LT.crossreg is None:
            raise AttributeError(f"{mouse}: LT.crossreg is missing but mapping='{mapping}' was requested.")
        df_map = LT.crossreg.get_mappings_cells(mapping_type=mapping)
        col = _get_session_map_col(LT)
        if col not in df_map.columns:
            raise KeyError(f"{mouse}: crossreg mapping '{mapping}' does not have column '{col}'. "
                            f"Available: {list(df_map.columns)}")
        # df entries are often floats-as-strings; normalize to int
        vals = df_map[col].dropna().astype(float).astype(int).tolist()
        return set(vals)

    # Case 2: dict-like mapping keyed by mouse
    if hasattr(mapping, "keys") and (mouse in mapping):
        return set(mapping[mouse])

    # Case 3: DataFrame-like mapping with mouse/cell_id columns
    if hasattr(mapping, "loc") and hasattr(mapping, "columns"):
        if ("mouse" in mapping.columns) and ("cell_id" in mapping.columns):
            return set(mapping.loc[mapping["mouse"] == mouse, "cell_id"].astype(int).tolist())

    # Unknown mapping format
    raise TypeError(f"Unsupported mapping type: {type(mapping)}")

def _cell_ids_to_row_idx(LT, cell_ids):
    """
    Convert unit IDs -> row indices into LT.S / LT.C / LT.YrA.
    Uses LT.S_idx if present.
    """
    if hasattr(LT, "S_idx") and LT.S_idx is not None:
        # build dict for speed
        id2row = {int(uid): i for i, uid in enumerate(np.asarray(LT.S_idx).astype(int))}
        row_idx = []
        for cid in cell_ids:
            cid_i = int(cid)
            if cid_i in id2row:
                row_idx.append(id2row[cid_i])
        return row_idx
    # fallback: assume ids are already row indices
    return [int(c) for c in cell_ids]
# ===================================================================
# Within-session place-field tiling heatmap
# ===================================================================

def plot_lt_within_session_tuning(
    PLOTS_DIR,
    LT_sessions,
    mouse_groups,
    session_str,
    max_pf_count=None,
    n_pos_bins=120,
    smooth_bins=3,
    auto_close=True,
    want_S=True,
    want_C=False,
    want_YrA=False,
    arm_cutoff_perc=0.8,
    mapping=None,
    use_sig_responses=False,
):
    """
    For each mouse, plot a **single-session** place-field tuning-curve heatmap
    sorted by each cell's own place-field centre on the linearised track.

    This demonstrates that place cells tile the full track within a session
    (no cross-registration to another session is required).

    Parameters
    ----------
    LT_sessions : dict {mouse: Session}
        One linear-track session per mouse (e.g. TFC_cond_LT1 or TFC_cond_LT2).
    mouse_groups : dict {mouse: group}
    session_str : str  – for filenames / titles (e.g. 'TFC_cond_LT1')
    n_pos_bins : int – number of spatial bins
    smooth_bins : int – Gaussian smoothing kernel width (bins)
    want_S / want_C / want_YrA : bool – which activity matrix to use
    arm_cutoff_perc : float – for turn-line detection
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    if not (want_S or want_C or want_YrA):
        want_S = True
    sig_str = "S" if want_S else ("C" if want_C else "YrA")
    pf_str = "pfALL" if (max_pf_count is None) else f"pf{int(max_pf_count)}"

    save_path = os.path.join(PLOTS_DIR, f"lt_within_session_{session_str}_{pf_str}")
    os.makedirs(save_path, exist_ok=True)

    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    tiling_results = {}  # {mouse: {"group":..., "tiling":...}}

    # --- helpers (duplicated to keep function standalone) ---
    def _smooth_rows(mat, k_size):
        if k_size is None or k_size <= 1:
            return mat
        k = np.ones(int(k_size), float) / float(k_size)
        pad = int(k_size) // 2
        mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode="edge")
        return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, mat_pad)

    def _compute_tuning_single(S_rows, pos, bin_edges):
        pos = np.asarray(pos, float)
        valid = np.isfinite(pos)
        pos_v = pos[valid]
        S_v = S_rows[:, valid]
        bin_idx = np.digitize(pos_v, bin_edges) - 1
        n_bins = len(bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        tuning = np.full((S_v.shape[0], n_bins), np.nan, float)
        for b in range(n_bins):
            m = (bin_idx == b)
            if not np.any(m):
                continue
            tuning[:, b] = np.nanmean(S_v[:, m], axis=1)
        return tuning

    def _turn_lines(loc_X, loc_Y, loc_1d, arm_cutoff_perc=0.8,
                    band_fracs=(0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18),
                    min_pts_per_arm=10):
        x = np.asarray(loc_X, dtype=float)
        y = np.asarray(loc_Y, dtype=float)
        p1d = np.asarray(loc_1d, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(p1d)
        if np.sum(valid) < 50:
            return None, None
        xv, yv, p1dv = x[valid], y[valid], p1d[valid]
        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < 1e-6:
            return None, None
        x_thresh = x_min + float(arm_cutoff_perc) * (x_max - x_min)
        y_lo_arm = np.nanpercentile(yv, 40)
        y_hi_arm = np.nanpercentile(yv, 60)
        mask_top = yv >= y_hi_arm
        mask_bot = yv <= y_lo_arm
        for bf in band_fracs:
            x_band = float(bf) * (x_max - x_min)
            mask_xband = np.abs(xv - x_thresh) <= x_band
            n_top = int(np.sum(mask_top & mask_xband))
            n_bot = int(np.sum(mask_bot & mask_xband))
            if n_top >= min_pts_per_arm and n_bot >= min_pts_per_arm:
                t_top = float(np.nanmedian(p1dv[mask_top & mask_xband]))
                t_bot = float(np.nanmedian(p1dv[mask_bot & mask_xband]))
                turns = sorted([t_top, t_bot])
                return turns[0], turns[1]
        return None, None


    for mouse, group in mouse_groups.items():
        LT = LT_sessions[mouse]

        # Activity matrix
        if want_S:
            S = np.asarray(LT.S)
        elif want_C:
            S = np.asarray(LT.C)
        else:
            S = np.asarray(LT.YrA)

        loc_X = LT.loc_X_miniscope_smooth
        loc_Y = LT.loc_Y_miniscope_smooth
        loc_1d = LT.miniscope_loc_1d_px * _LT_1D_CM_PER_PX
        pos = np.asarray(loc_1d, float)

        # Get place cells
        if use_sig_responses:
            pf_keys = set(getattr(LT.fm, "sig_responses", {}).keys())
        else:
            pf_keys = set(LT.fm.pf.merged_means.keys())

        # --- Proper mapping filter (supports mapping as str like 'LT1+LT2') ---
        mapping_cells = _get_mapping_cells_for_mouse(LT, mouse, mapping)  # empty set means "no restriction"
        if len(mapping_cells) > 0:
            pf_keys = pf_keys & mapping_cells

        if len(pf_keys) == 0:
            print(f"[WARN] {mouse}: no place fields (after mapping filter) in {session_str}. Skipping.")
            continue

        # --- Mapping filter ---
        use_mapping = (mapping is not None) and (mapping != "full")
        if use_mapping:
            mapping_cells = set()
            if hasattr(mapping, 'keys') and mouse in mapping:
                mapping_cells = set(mapping[mouse])
            elif hasattr(mapping, 'loc') and hasattr(mapping, 'columns'):
                if 'mouse' in mapping.columns and 'cell_id' in mapping.columns:
                    mapping_cells = set(mapping.loc[mapping['mouse'] == mouse, 'cell_id'])
            if mapping_cells:
                pf_keys = pf_keys & mapping_cells
        if len(pf_keys) == 0:
            print(f"[WARN] {mouse}: no place fields (after mapping filter) in {session_str}. Skipping.")
            continue

        # Sort place cells by their PF centre on the linearised track
        XY = np.column_stack([loc_X, loc_Y])
        bin_w = float(LT.fm.loc.bin_width)
        min_x = float(getattr(LT.fm.loc, "MIN_X"))
        min_y = float(getattr(LT.fm.loc, "MIN_Y"))


        sorted_entries = []
        for cell_id in pf_keys:
            merged = LT.fm.pf.merged_means.get(cell_id, None)
            model = LT.fm.pf.model_.get(cell_id, None)
            sig = LT.fm.sig_responses.get(cell_id, None)

            if merged is None or model is None or not hasattr(model, "means_"):
                continue

            # PF count filter
            if max_pf_count is not None and len(merged) > int(max_pf_count):
                continue

            means_rc = np.asarray(model.means_, dtype=float)

            # Map each gaussian mean to its merged group
            mean_to_group = {}
            for g_idx, grp in enumerate(merged):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)

            pts_by_group = [[] for _ in range(len(merged))]
            if use_sig_responses:
                use_sig = (sig is not None and len(sig) > 0)
                if use_sig:
                    sig_arr = np.asarray(sig, dtype=float)
                    rc = sig_arr[:, :2]
                    pred_mean_idx = np.asarray(model.predict(rc), dtype=int)
                    for (r, c), m_idx in zip(rc, pred_mean_idx):
                        g = mean_to_group.get(int(m_idx), None)
                        if g is None:
                            continue
                        pts_by_group[g].append((float(r), float(c)))
            # If not using sig_responses, pts_by_group remains empty

            # Find place field centres in 1D space
            pf_candidates = []
            for grp_i, grp in enumerate(merged):
                grp = list(grp)
                if len(grp) == 0:
                    continue
                pts = pts_by_group[grp_i]
                if use_sig_responses and len(pts) > 0:
                    ctr_rc = np.asarray(pts, float).mean(axis=0)
                else:
                    ctr_rc = means_rc[grp].mean(axis=0)
                ctr_x = min_x + (float(ctr_rc[1]) + 0.5) * bin_w
                ctr_y = min_y + (float(ctr_rc[0]) + 0.5) * bin_w
                ctr_xy = np.array([ctr_x, ctr_y], dtype=float)
                d2 = np.sum((XY - ctr_xy[None, :]) ** 2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_loc1d = float(loc_1d[nn_idx])
                pf_candidates.append((grp_i, nn_loc1d))

            if len(pf_candidates) == 0:
                continue

            # Use the PF closest to track start for sorting
            chosen = min(pf_candidates, key=lambda x: x[-1])
            sorted_entries.append((cell_id, chosen[-1]))

        if len(sorted_entries) == 0:
            print(f"[WARN] {mouse}: no sorted place cells in {session_str}. Skipping.")
            continue

        sorted_entries.sort(key=lambda x: x[1])
        sorted_cell_ids = [e[0] for e in sorted_entries]

        # Build row indices
        row_idx = _cell_ids_to_row_idx(LT, sorted_cell_ids)
        if len(row_idx) == 0:
            print(f"[WARN] {mouse}: none of the mapped PF cell IDs were found in LT.S_idx. Skipping.")
            continue
        S_sorted = S[row_idx, :]

        # Bin edges (single session)
        pos_v = pos[np.isfinite(pos)]
        if pos_v.size < 10:
            continue
        bin_edges = np.linspace(float(np.nanmin(pos_v)), float(np.nanmax(pos_v)), int(n_pos_bins) + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Tuning curves
        tuning = _compute_tuning_single(S_sorted, pos, bin_edges)
        tuning = _smooth_rows(tuning, smooth_bins)

        # Per-cell max normalization for display
        row_max = np.nanmax(tuning, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = np.nan
        row_max[row_max == 0] = np.nan
        tuning_norm = tuning / row_max

        # Turn lines
        turn_a, turn_b = _turn_lines(loc_X, loc_Y, loc_1d, arm_cutoff_perc=arm_cutoff_perc)

        n_cells = tuning_norm.shape[0]

        # --- Plot ---
        fig, ax = plt.subplots(1, 1, figsize=(5, 7), dpi=200)
        ax.imshow(
            tuning_norm, aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap, vmin=0, vmax=1,
            extent=[float(bin_edges[0]), float(bin_edges[-1]), n_cells, 1],
        )
        if turn_a is not None:
            ax.axvline(turn_a, linestyle="--", linewidth=1.5, color="white", alpha=0.7)
        if turn_b is not None:
            ax.axvline(turn_b, linestyle="--", linewidth=1.5, color="white", alpha=0.7)

        ax.set_title(f"{mouse} ({group}) — {session_str} [{pf_str}]\n{n_cells} place cells, sorted by PF centre",
                      fontsize=10)
        ax.set_xlabel(f"Linearised track position ({_LT_1D_DISTANCE_UNIT})")
        ax.set_ylabel("Place cells")
        ax.set_yticks([1, n_cells])
        ax.set_yticklabels(["1", f"{n_cells}"])
        fig.tight_layout()

        file_out = os.path.join(save_path, f"within_session_tuning_{mouse}_{group}_{pf_str}_{sig_str}.png")
        fig.savefig(file_out, dpi=300, bbox_inches="tight")
        print(f'[INFO] Saved within-session tuning plot for {mouse} to {file_out}')
        plt.show()
        if auto_close:
            plt.close(fig)

        # Tiling quality metrics (all sig cells, self-sorted)
        tiling = compute_tiling_metrics(tuning_norm)
        tiling_results[mouse] = {
            "group": group,
            "session_str": session_str,
            "n_cells": n_cells,
            "tiling": tiling,
        }

    return tiling_results

def _get_session_map_col(LT):
    """
    Return the column name in the crossreg mapping DF that corresponds to THIS session.
    Prefer LT.get_df_col() if it exists (you already use this pattern elsewhere),
    otherwise fall back to LT.session_group if present.
    """
    if hasattr(LT, "get_df_col"):
        try:
            return LT.get_df_col()
        except Exception:
            pass
    if hasattr(LT, "session_group"):
        return LT.session_group
    raise AttributeError("Can't determine session column for crossreg mapping (no get_df_col() or session_group).")

def plot_lt_within_session_tuning_normalized(
    PLOTS_DIR,
    LT_sessions,
    mouse_groups,
    session_str,
    mapping=None,
    filter_to_sig_cells=True,
    max_pf_count=None,
    n_pos_bins=120,
    smooth_bins=3,
    auto_close=True,
    want_S=True,
    want_C=False,
    want_YrA=False,
    arm_cutoff_perc=0.8,
):
    """
    Per-mouse within-session place-field heatmap with NORMALISED position
    axis [0, 1] (fraction of linearised track range).

    Parameters
    ----------
    mapping : str or None
        If None or 'full', use all sig_responses cells (no cross-registration
        filter).  Otherwise (e.g. 'LT1+LT2', 'TFC_cond+LT1+LT2'), restrict
        to cells present in the cross-registration mapping AND in sig_responses.
    filter_to_sig_cells : bool
        If True (default), only include mapped cells that are also in
        sig_responses.  If False, include all mapped cells.

    Saves to ``{PLOTS_DIR}/lt_within_session_norm_{session_str}_mapping_{mapping_str}/``.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    use_mapping = (mapping is not None) and (mapping != "full")
    mapping_str = mapping if use_mapping else "full"
    pf_str = "pfALL" if (max_pf_count is None) else f"pf{int(max_pf_count)}"

    if not (want_S or want_C or want_YrA):
        want_S = True
    sig_str = "S" if want_S else ("C" if want_C else "YrA")

    save_path = os.path.join(PLOTS_DIR, f"lt_within_session_norm_{session_str}_mapping_{mapping_str}_{pf_str}")
    os.makedirs(save_path, exist_ok=True)

    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    # --- helpers ---
    def _smooth_rows(mat, k_size):
        if k_size is None or k_size <= 1:
            return mat
        k = np.ones(int(k_size), float) / float(k_size)
        pad = int(k_size) // 2
        mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode="edge")
        return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, mat_pad)

    def _compute_tuning_single(S_rows, pos, bin_edges):
        pos = np.asarray(pos, float)
        valid = np.isfinite(pos)
        pos_v = pos[valid]
        S_v = S_rows[:, valid]
        bin_idx = np.digitize(pos_v, bin_edges) - 1
        n_bins = len(bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        tuning = np.full((S_v.shape[0], n_bins), np.nan, float)
        for b in range(n_bins):
            m = (bin_idx == b)
            if not np.any(m):
                continue
            tuning[:, b] = np.nanmean(S_v[:, m], axis=1)
        return tuning

    def _turn_lines(loc_X, loc_Y, loc_1d, arm_cutoff_perc=0.8,
                    band_fracs=(0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18),
                    min_pts_per_arm=10):
        x = np.asarray(loc_X, dtype=float)
        y = np.asarray(loc_Y, dtype=float)
        p1d = np.asarray(loc_1d, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(p1d)
        if np.sum(valid) < 50:
            return None, None
        xv, yv, p1dv = x[valid], y[valid], p1d[valid]
        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < 1e-6:
            return None, None
        x_thresh = x_min + float(arm_cutoff_perc) * (x_max - x_min)
        y_lo_arm = np.nanpercentile(yv, 40)
        y_hi_arm = np.nanpercentile(yv, 60)
        mask_top = yv >= y_hi_arm
        mask_bot = yv <= y_lo_arm
        for bf in band_fracs:
            x_band = float(bf) * (x_max - x_min)
            mask_xband = np.abs(xv - x_thresh) <= x_band
            n_top = int(np.sum(mask_top & mask_xband))
            n_bot = int(np.sum(mask_bot & mask_xband))
            if n_top >= min_pts_per_arm and n_bot >= min_pts_per_arm:
                t_top = float(np.nanmedian(p1dv[mask_top & mask_xband]))
                t_bot = float(np.nanmedian(p1dv[mask_bot & mask_xband]))
                turns = sorted([t_top, t_bot])
                return turns[0], turns[1]
        return None, None

    def _get_cell_ids(LT, mouse, use_mapping, mapping, filter_to_sig):
        """Return set of cell IDs to include for this mouse/session."""
        sig_keys = set(getattr(LT.fm, "sig_responses", {}).keys())
        if not use_mapping:
            return sig_keys
        # Cross-registration mapping
        _ = LT.get_S_mapping(mapping)
        df_map = LT.crossreg.get_mappings_cells(mapping_type=mapping)
        mapped_cells = set(df_map[LT.session_group].astype(float).astype(int).tolist())
        if filter_to_sig:
            mapped_cells = mapped_cells & sig_keys
        return mapped_cells
    
    for mouse, group in mouse_groups.items():
        LT = LT_sessions[mouse]

        if want_S:
            S = np.asarray(LT.S)
        elif want_C:
            S = np.asarray(LT.C)
        else:
            S = np.asarray(LT.YrA)

        loc_X = LT.loc_X_miniscope_smooth
        loc_Y = LT.loc_Y_miniscope_smooth
        loc_1d = LT.miniscope_loc_1d_px * _LT_1D_CM_PER_PX
        pos = np.asarray(loc_1d, float)

        cell_ids = _get_cell_ids(LT, mouse, use_mapping, mapping, filter_to_sig_cells)
        if len(cell_ids) == 0:
            print(f"[WARN] {mouse}: no cells for {session_str} (mapping={mapping_str}). Skipping.")
            continue

        # --- Normalise position to [0, 1] ---
        pos_finite = pos[np.isfinite(pos)]
        if pos_finite.size < 10:
            continue
        pos_lo = float(np.nanmin(pos_finite))
        pos_hi = float(np.nanmax(pos_finite))
        pos_range = pos_hi - pos_lo
        if pos_range < 1e-6:
            continue
        pos_norm = (pos - pos_lo) / pos_range   # NaN stays NaN

        # Sort place cells by PF centre (in normalised coords)
        XY = np.column_stack([loc_X, loc_Y])
        bin_w = float(LT.fm.loc.bin_width)
        min_x = float(getattr(LT.fm.loc, "MIN_X"))
        min_y = float(getattr(LT.fm.loc, "MIN_Y"))

        sorted_entries = []
        for cell_id in cell_ids:
            merged = LT.fm.pf.merged_means.get(cell_id, None)
            model = LT.fm.pf.model_.get(cell_id, None)
            sig = LT.fm.sig_responses.get(cell_id, None)
            if merged is None or model is None or not hasattr(model, "means_"):
                continue
            # PF count filter
            if max_pf_count is not None and len(merged) > int(max_pf_count):
                continue
            means_rc = np.asarray(model.means_, dtype=float)
            mean_to_group = {}
            for g_idx, grp in enumerate(merged):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)
            pts_by_group = [[] for _ in range(len(merged))]
            use_sig = (sig is not None and len(sig) > 0)
            if use_sig:
                sig_arr = np.asarray(sig, dtype=float)
                rc = sig_arr[:, :2]
                pred_mean_idx = np.asarray(model.predict(rc), dtype=int)
                for (r, c), m_idx in zip(rc, pred_mean_idx):
                    g = mean_to_group.get(int(m_idx), None)
                    if g is not None:
                        pts_by_group[g].append((float(r), float(c)))
            pf_candidates = []
            for grp_i, grp in enumerate(merged):
                grp = list(grp)
                if len(grp) == 0:
                    continue
                pts = pts_by_group[grp_i]
                ctr_rc = np.asarray(pts, float).mean(axis=0) if len(pts) > 0 else means_rc[grp].mean(axis=0)
                ctr_x = min_x + (float(ctr_rc[1]) + 0.5) * bin_w
                ctr_y = min_y + (float(ctr_rc[0]) + 0.5) * bin_w
                ctr_xy = np.array([ctr_x, ctr_y], dtype=float)
                d2 = np.sum((XY - ctr_xy[None, :]) ** 2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_loc1d_norm = float((loc_1d[nn_idx] - pos_lo) / pos_range)
                pf_candidates.append((grp_i, nn_loc1d_norm))
            if len(pf_candidates) == 0:
                continue
            chosen = min(pf_candidates, key=lambda x: x[-1])
            sorted_entries.append((cell_id, chosen[-1]))

        if len(sorted_entries) == 0:
            print(f"[WARN] {mouse}: no sorted place cells in {session_str} (mapping={mapping_str}, {pf_str}). Skipping.")
            continue

        sorted_entries.sort(key=lambda x: x[1])
        sorted_cell_ids = [e[0] for e in sorted_entries]

        if all(isinstance(c, (int, np.integer)) for c in sorted_cell_ids) and max(sorted_cell_ids) < S.shape[0]:
            row_idx = [int(c) for c in sorted_cell_ids]
        else:
            row_idx = sorted_cell_ids
        S_sorted = S[row_idx, :]

        # Bin edges on normalised axis [0, 1]
        bin_edges = np.linspace(0.0, 1.0, int(n_pos_bins) + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        tuning = _compute_tuning_single(S_sorted, pos_norm, bin_edges)
        tuning = _smooth_rows(tuning, smooth_bins)

        row_max = np.nanmax(tuning, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = np.nan
        row_max[row_max == 0] = np.nan
        tuning_norm = tuning / row_max

        # Normalised turn lines
        turn_a_px, turn_b_px = _turn_lines(loc_X, loc_Y, loc_1d, arm_cutoff_perc=arm_cutoff_perc)
        turn_a = (turn_a_px - pos_lo) / pos_range if turn_a_px is not None else None
        turn_b = (turn_b_px - pos_lo) / pos_range if turn_b_px is not None else None

        n_cells = tuning_norm.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(5, 7), dpi=200)
        ax.imshow(
            tuning_norm, aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap, vmin=0, vmax=1,
            extent=[0, 1, n_cells, 1],
        )
        if turn_a is not None:
            ax.axvline(turn_a, linestyle="--", linewidth=1.5, color="white", alpha=0.7)
        if turn_b is not None:
            ax.axvline(turn_b, linestyle="--", linewidth=1.5, color="white", alpha=0.7)

        ax.set_title(f"{mouse} ({group}) — {session_str} [{mapping_str}] [{pf_str}]\n{n_cells} place cells, sorted by PF centre",
                      fontsize=10)
        ax.set_xlabel("Normalised track position")
        ax.set_ylabel("Place cells")
        ax.set_yticks([1, n_cells])
        ax.set_yticklabels(["1", f"{n_cells}"])
        fig.tight_layout()

        fig.savefig(os.path.join(save_path, f"within_session_tuning_norm_{mouse}_{group}_{mapping_str}_{pf_str}_{sig_str}.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()
        if auto_close:
            plt.close(fig)


def plot_lt_within_session_tuning_group_averaged(
    PLOTS_DIR,
    LT_sessions,
    mouse_groups,
    session_str,
    mapping=None,
    filter_to_sig_cells=True,
    max_pf_count=None,
    n_pos_bins=120,
    smooth_bins=3,
    auto_close=True,
    want_S=True,
    want_C=False,
    want_YrA=False,
    arm_cutoff_perc=0.8,
    vmin=0.0,
    vmax=1.0,
):
    """
    Group-averaged within-session place-field heatmap on a normalised [0, 1]
    position axis.

    Parameters
    ----------
    mapping : str or None
        If None or 'full', use all sig_responses cells (no cross-registration
        filter).  Otherwise (e.g. 'LT1+LT2', 'TFC_cond+LT1+LT2'), restrict
        to cells present in the cross-registration mapping AND in sig_responses.
    filter_to_sig_cells : bool
        If True (default), only include mapped cells that are also in
        sig_responses.  If False, include all mapped cells.

    Saves to ``{PLOTS_DIR}/lt_within_session_group_avg_{session_str}_mapping_{mapping_str}/``.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    from scipy.ndimage import zoom as ndimage_zoom

    use_mapping = (mapping is not None) and (mapping != "full")
    mapping_str = mapping if use_mapping else "full"
    pf_str = "pfALL" if (max_pf_count is None) else f"pf{int(max_pf_count)}"

    if not (want_S or want_C or want_YrA):
        want_S = True
    sig_str = "S" if want_S else ("C" if want_C else "YrA")

    save_path = os.path.join(PLOTS_DIR, f"lt_within_session_group_avg_{session_str}_mapping_{mapping_str}_{pf_str}")
    os.makedirs(save_path, exist_ok=True)

    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    # --- helpers ---
    def _smooth_rows(mat, k_size):
        if k_size is None or k_size <= 1:
            return mat
        k = np.ones(int(k_size), float) / float(k_size)
        pad = int(k_size) // 2
        mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode="edge")
        return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, mat_pad)

    def _compute_tuning_single(S_rows, pos, bin_edges):
        pos = np.asarray(pos, float)
        valid = np.isfinite(pos)
        pos_v = pos[valid]
        S_v = S_rows[:, valid]
        bin_idx = np.digitize(pos_v, bin_edges) - 1
        n_bins_loc = len(bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins_loc - 1)
        tuning = np.full((S_v.shape[0], n_bins_loc), np.nan, float)
        for b in range(n_bins_loc):
            m = (bin_idx == b)
            if not np.any(m):
                continue
            tuning[:, b] = np.nanmean(S_v[:, m], axis=1)
        return tuning

    def _turn_lines(loc_X, loc_Y, loc_1d, arm_cutoff_perc=0.8,
                    band_fracs=(0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18),
                    min_pts_per_arm=10):
        x = np.asarray(loc_X, dtype=float)
        y = np.asarray(loc_Y, dtype=float)
        p1d = np.asarray(loc_1d, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(p1d)
        if np.sum(valid) < 50:
            return None, None
        xv, yv, p1dv = x[valid], y[valid], p1d[valid]
        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < 1e-6:
            return None, None
        x_thresh = x_min + float(arm_cutoff_perc) * (x_max - x_min)
        y_lo_arm = np.nanpercentile(yv, 40)
        y_hi_arm = np.nanpercentile(yv, 60)
        mask_top = yv >= y_hi_arm
        mask_bot = yv <= y_lo_arm
        for bf in band_fracs:
            x_band = float(bf) * (x_max - x_min)
            mask_xband = np.abs(xv - x_thresh) <= x_band
            n_top = int(np.sum(mask_top & mask_xband))
            n_bot = int(np.sum(mask_bot & mask_xband))
            if n_top >= min_pts_per_arm and n_bot >= min_pts_per_arm:
                t_top = float(np.nanmedian(p1dv[mask_top & mask_xband]))
                t_bot = float(np.nanmedian(p1dv[mask_bot & mask_xband]))
                turns = sorted([t_top, t_bot])
                return turns[0], turns[1]
        return None, None

    # Common cell-axis grid for interpolation (# virtual rows)
    n_cells_grid = 100

    def _get_cell_ids(LT, mouse, use_mapping, mapping, filter_to_sig):
        """Return set of cell IDs to include for this mouse/session."""
        sig_keys = set(getattr(LT.fm, "sig_responses", {}).keys())
        if not use_mapping:
            return sig_keys
        _ = LT.get_S_mapping(mapping)
        df_map = LT.crossreg.get_mappings_cells(mapping_type=mapping)
        mapped_cells = set(df_map[LT.session_group].astype(float).astype(int).tolist())
        if filter_to_sig:
            mapped_cells = mapped_cells & sig_keys
        return mapped_cells

    # ---- Collect per-mouse tuning matrices (normalised axis) ----
    group_data = {}   # group -> list of (tuning_norm, turn_a_norm, turn_b_norm)

    for mouse, group in mouse_groups.items():
        LT = LT_sessions[mouse]
        if want_S:
            S = np.asarray(LT.S)
        elif want_C:
            S = np.asarray(LT.C)
        else:
            S = np.asarray(LT.YrA)

        loc_X = LT.loc_X_miniscope_smooth
        loc_Y = LT.loc_Y_miniscope_smooth
        loc_1d = LT.miniscope_loc_1d_px * _LT_1D_CM_PER_PX
        pos = np.asarray(loc_1d, float)

        cell_ids = _get_cell_ids(LT, mouse, use_mapping, mapping, filter_to_sig_cells)
        if len(cell_ids) == 0:
            continue

        pos_finite = pos[np.isfinite(pos)]
        if pos_finite.size < 10:
            continue
        pos_lo = float(np.nanmin(pos_finite))
        pos_hi = float(np.nanmax(pos_finite))
        pos_range = pos_hi - pos_lo
        if pos_range < 1e-6:
            continue
        pos_norm = (pos - pos_lo) / pos_range

        XY = np.column_stack([loc_X, loc_Y])
        bin_w = float(LT.fm.loc.bin_width)
        min_x_loc = float(getattr(LT.fm.loc, "MIN_X"))
        min_y_loc = float(getattr(LT.fm.loc, "MIN_Y"))

        sorted_entries = []
        for cell_id in cell_ids:
            merged = LT.fm.pf.merged_means.get(cell_id, None)
            model = LT.fm.pf.model_.get(cell_id, None)
            sig = LT.fm.sig_responses.get(cell_id, None)
            if merged is None or model is None or not hasattr(model, "means_"):
                continue
            means_rc = np.asarray(model.means_, dtype=float)
            mean_to_group = {}
            for g_idx, grp in enumerate(merged):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)
            pts_by_group = [[] for _ in range(len(merged))]
            if sig is not None and len(sig) > 0:
                sig_arr = np.asarray(sig, dtype=float)
                rc = sig_arr[:, :2]
                pred_mean_idx = np.asarray(model.predict(rc), dtype=int)
                for (r, c), m_idx in zip(rc, pred_mean_idx):
                    g = mean_to_group.get(int(m_idx), None)
                    if g is not None:
                        pts_by_group[g].append((float(r), float(c)))
            pf_candidates = []
            for grp_i, grp in enumerate(merged):
                grp = list(grp)
                if len(grp) == 0:
                    continue
                pts = pts_by_group[grp_i]
                ctr_rc = np.asarray(pts, float).mean(axis=0) if len(pts) > 0 else means_rc[grp].mean(axis=0)
                ctr_x = min_x_loc + (float(ctr_rc[1]) + 0.5) * bin_w
                ctr_y = min_y_loc + (float(ctr_rc[0]) + 0.5) * bin_w
                ctr_xy = np.array([ctr_x, ctr_y], dtype=float)
                d2 = np.sum((XY - ctr_xy[None, :]) ** 2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_loc1d_norm = float((loc_1d[nn_idx] - pos_lo) / pos_range)
                pf_candidates.append((grp_i, nn_loc1d_norm))
            if len(pf_candidates) == 0:
                continue
            chosen = min(pf_candidates, key=lambda x: x[-1])
            sorted_entries.append((cell_id, chosen[-1]))

        if len(sorted_entries) == 0:
            continue

        sorted_entries.sort(key=lambda x: x[1])
        sorted_cell_ids = [e[0] for e in sorted_entries]
        if all(isinstance(c, (int, np.integer)) for c in sorted_cell_ids) and max(sorted_cell_ids) < S.shape[0]:
            row_idx = [int(c) for c in sorted_cell_ids]
        else:
            row_idx = sorted_cell_ids
        S_sorted = S[row_idx, :]

        bin_edges = np.linspace(0.0, 1.0, int(n_pos_bins) + 1)
        tuning = _compute_tuning_single(S_sorted, pos_norm, bin_edges)
        tuning = _smooth_rows(tuning, smooth_bins)

        row_max = np.nanmax(tuning, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = np.nan
        row_max[row_max == 0] = np.nan
        tuning_cell_norm = tuning / row_max   # (n_cells, n_pos_bins)

        # Resample to common cell-axis grid via zoom
        n_cells_mouse = tuning_cell_norm.shape[0]
        if n_cells_mouse < 2:
            continue
        zoom_r = n_cells_grid / n_cells_mouse
        mask_nan = np.isnan(tuning_cell_norm)
        filled = np.where(mask_nan, 0.0, tuning_cell_norm)
        resized = ndimage_zoom(filled, (zoom_r, 1.0), order=1)
        mask_resized = ndimage_zoom(mask_nan.astype(float), (zoom_r, 1.0), order=1) > 0.5
        resized[mask_resized] = np.nan

        # Turn lines (normalised)
        turn_a_px, turn_b_px = _turn_lines(loc_X, loc_Y, loc_1d, arm_cutoff_perc=arm_cutoff_perc)
        ta_n = (turn_a_px - pos_lo) / pos_range if turn_a_px is not None else None
        tb_n = (turn_b_px - pos_lo) / pos_range if turn_b_px is not None else None

        group_data.setdefault(group, []).append({
            "tuning_resampled": resized,  # (n_cells_grid, n_pos_bins)
            "turn_a": ta_n,
            "turn_b": tb_n,
            "n_cells": n_cells_mouse,
            "mouse": mouse,
        })

    if not group_data:
        print(f"[WARN] plot_lt_within_session_tuning_group_averaged: no data for {session_str}.")
        return

    # ---- Plot one figure per group ----
    _GROUP_ORDER = ["mCherry", "hM3D", "hM4D"]
    groups = [g for g in _GROUP_ORDER if g in group_data] + \
             [g for g in sorted(group_data.keys()) if g not in _GROUP_ORDER]
    for group in groups:
        entries = group_data[group]
        n_mice = len(entries)
        stack = np.array([e["tuning_resampled"] for e in entries])  # (n_mice, n_cells_grid, n_pos_bins)
        mean_tuning = np.nanmean(stack, axis=0)  # (n_cells_grid, n_pos_bins)

        # Average turn positions
        ta_list = [e["turn_a"] for e in entries if e["turn_a"] is not None]
        tb_list = [e["turn_b"] for e in entries if e["turn_b"] is not None]
        avg_ta = float(np.mean(ta_list)) if ta_list else None
        avg_tb = float(np.mean(tb_list)) if tb_list else None

        avg_ncells = float(np.mean([e["n_cells"] for e in entries]))
        mice_str = ", ".join(e["mouse"] for e in entries)

        fig, ax = plt.subplots(1, 1, figsize=(5, 7), dpi=200)
        ax.imshow(
            mean_tuning, aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[0, 1, n_cells_grid, 1],
        )
        if avg_ta is not None:
            ax.axvline(avg_ta, linestyle="--", linewidth=1.5, color="white", alpha=0.7)
        if avg_tb is not None:
            ax.axvline(avg_tb, linestyle="--", linewidth=1.5, color="white", alpha=0.7)

        ax.set_title(f"{group} (n={n_mice}, avg {avg_ncells:.0f} cells) — {session_str} [{mapping_str}] [{pf_str}]\n"
                     f"Group-averaged tuning, sorted by PF centre",
                     fontsize=10)
        ax.set_xlabel("Normalised track position")
        ax.set_ylabel("Place cells (normalised order)")
        ax.set_yticks([1, n_cells_grid])
        ax.set_yticklabels(["start", "end"])
        fig.tight_layout()

        fig.savefig(os.path.join(save_path,
                    f"within_session_tuning_group_avg_{group}_{mapping_str}_{pf_str}_{sig_str}.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()
        if auto_close:
            plt.close(fig)


def plot_lt_spatial_responses(
    PLOTS_DIR,
    LT1_group,
    LT2_group,
    mouse_groups,
    session_str,
    mapping=None,
    use_sig_responses=False,
    want_sanity_sanity=True,
    how_many_sanity_sanity_cells=3,
    auto_close=True,
    want_S=False,
    want_C=False,
    want_YrA=False,
    # -------------------------
    # Display normalization (scalar)
    # -------------------------
    normalize_global=False,
    normalize_per_mouse=False,
    global_norm_mode="pctl",      # "max" or "pctl"
    global_norm_pctl=99.0,        # only used if global_norm_mode=="pctl"
    mouse_norm_mode="max",        # "max" or "pctl" (optional symmetry; default max)
    mouse_norm_pctl=99.0,         # only if mouse_norm_mode=="pctl"
    # -------------------------
    # Re-added: pairwise per-cell normalization
    # -------------------------
    normalize_pairwise_per_cell=False,
    pairwise_cell_mode="pctl",    # "pctl" or "max"
    pairwise_cell_pctl=95.0,      # only if pairwise_cell_mode=="pctl"
    # -------------------------
    # PV corr alignment + diagonal summary
    # -------------------------
    pv_use_shared_bins=True,
    pv_shared_bins_mode="overlap",   # "overlap" or "union"
    pv_n_pos_bins=120,
    pv_smooth_bins=3,
    pv_diag_summary="mean",          # "mean" or "median"
    pv_diag_min_valid_frac=0.5,
    # -------------------------
    # NEW: use normalization for PV corr
    # -------------------------
    pv_use_normalized=False,         # if True, PV corr uses normalized tunings (same as plotted)
    # -------------------------
    # Filter LT1 sig cells
    # -------------------------
    filter_to_LT1_sig_cells=True,
    # -------------------------
    # Place-field count filter
    # -------------------------
    max_pf_count=None,           # None → all PFs;  1 → single-field only;  2 → ≤2;  etc.
    # -------------------------
    # Sort order
    # -------------------------
    sort_by="LT1",               # "LT1" or "LT2" — which session's PF centres determine row order
):
    """
    Plot spatial responses for cross-registered cells between LT1 and LT2.

    By default rows are sorted by LT1 place-field centres (sort_by='LT1').
    Set sort_by='LT2' to sort by LT2 PF centres instead and show LT1 in that order.

    Normalization pipeline (applied to tunings AFTER smoothing):
      1) optional scalar scaling (global/per-mouse; max or percentile)
      2) optional per-cell scaling:
         - if normalize_pairwise_per_cell: scale each cell by a shared factor computed from (LT1 row, LT2 row)
           either max or percentile (default pctl=95)
         - else (default if no scalar mode): per-cell max normalization for display only

    PV correlations:
      - computed from RAW tunings by default
      - if pv_use_normalized=True, uses the normalized tunings (after chosen normalization pipeline)

    Cell filtering:
      - if filter_to_LT1_sig_cells=True (default), excludes mapped LT1 cells not present in LT1.fm.sig_responses
        (and removes their paired LT2 cell).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from numpy.random import default_rng

    if mapping is None:
        raise ValueError("plot_lt_spatial_responses: mapping must not be None")

    if normalize_global and normalize_per_mouse:
        raise ValueError("Choose only one: normalize_global or normalize_per_mouse (not both).")

    if pv_shared_bins_mode not in ("overlap", "union"):
        raise ValueError("pv_shared_bins_mode must be 'overlap' or 'union'")

    if pv_diag_summary not in ("mean", "median"):
        raise ValueError("pv_diag_summary must be 'mean' or 'median'")

    if global_norm_mode not in ("max", "pctl"):
        raise ValueError("global_norm_mode must be 'max' or 'pctl'")

    if mouse_norm_mode not in ("max", "pctl"):
        raise ValueError("mouse_norm_mode must be 'max' or 'pctl'")

    if sort_by not in ("LT1", "LT2"):
        raise ValueError("sort_by must be 'LT1' or 'LT2'")

    if pairwise_cell_mode not in ("max", "pctl"):
        raise ValueError("pairwise_cell_mode must be 'max' or 'pctl'")

    if not (want_S or want_C or want_YrA):
        want_S = True
    S_file_str = "S" if want_S else ("C" if want_C else "YrA")
    pf_str = "pfALL" if (max_pf_count is None) else f"pf{int(max_pf_count)}"

    sort_tag = f"_sortby_{sort_by}" if sort_by != "LT1" else ""
    save_path = os.path.join(PLOTS_DIR, f"lt_spatial_responses_{session_str}_mapping_{mapping}_{pf_str}{sort_tag}")
    os.makedirs(save_path, exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def _safe_corr(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 3:
            return np.nan
        aa = a[m] - np.nanmean(a[m])
        bb = b[m] - np.nanmean(b[m])
        denom = np.sqrt(np.nansum(aa * aa) * np.nansum(bb * bb))
        if denom <= 0 or not np.isfinite(denom):
            return np.nan
        return float(np.nansum(aa * bb) / denom)

    def _smooth_rows(mat, smooth_bins):
        if smooth_bins is None or smooth_bins <= 1:
            return mat
        k = np.ones(int(smooth_bins), float) / float(smooth_bins)
        pad = int(smooth_bins) // 2
        mat_pad = np.pad(mat, ((0, 0), (pad, pad)), mode="edge")
        return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, mat_pad)

    def _normalize_per_cell_max(mat):
        out = mat.copy()
        row_max = np.nanmax(out, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = np.nan
        row_max[row_max == 0] = np.nan
        return out / row_max

    def _pairwise_per_cell_scale(t1, t2, mode="pctl", pctl=95.0):
        """
        Compute per-cell scale factor shared between LT1 and LT2 rows.
        Returns (t1_scaled, t2_scaled).
        Scale_i = max( stat(t1_i), stat(t2_i) ) where stat=nanmax or nanpercentile.
        """
        t1 = np.asarray(t1, float)
        t2 = np.asarray(t2, float)
        if t1.shape != t2.shape:
            raise ValueError("pairwise_per_cell requires t1 and t2 to have same shape")

        if mode == "max":
            s1 = np.nanmax(t1, axis=1)
            s2 = np.nanmax(t2, axis=1)
        else:
            s1 = np.nanpercentile(t1, float(pctl), axis=1)
            s2 = np.nanpercentile(t2, float(pctl), axis=1)

        scale = np.nanmax(np.stack([s1, s2], axis=0), axis=0)  # (n_cells,)
        scale = scale.reshape(-1, 1)
        scale[~np.isfinite(scale)] = np.nan
        scale[scale == 0] = np.nan
        return t1 / scale, t2 / scale

    def _compute_tuning(S_sorted, pos, bin_edges):
        pos = np.asarray(pos, float)
        valid = np.isfinite(pos)
        pos_v = pos[valid]
        S_v = S_sorted[:, valid]

        bin_idx = np.digitize(pos_v, bin_edges) - 1
        n_bins = len(bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        tuning = np.full((S_v.shape[0], n_bins), np.nan, float)
        for b in range(n_bins):
            m = (bin_idx == b)
            if not np.any(m):
                continue
            tuning[:, b] = np.nanmean(S_v[:, m], axis=1)
        return tuning

    def _compute_turn_lines_safe(
        loc_X,
        loc_Y,
        loc_1d,
        arm_cutoff_perc=0.8,
        band_fracs=(0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18),
        min_pts_per_arm=10,
    ):
        x = np.asarray(loc_X, dtype=float)
        y = np.asarray(loc_Y, dtype=float)
        p1d = np.asarray(loc_1d, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(p1d)
        if np.sum(valid) < 50:
            return {"turn1_1d": None, "turn2_1d": None}

        xv, yv, p1dv = x[valid], y[valid], p1d[valid]
        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < 1e-6:
            return {"turn1_1d": None, "turn2_1d": None}

        x_thresh = x_min + float(arm_cutoff_perc) * (x_max - x_min)

        y_lo_arm = np.nanpercentile(yv, 40)
        y_hi_arm = np.nanpercentile(yv, 60)
        mask_top = yv >= y_hi_arm
        mask_bot = yv <= y_lo_arm

        for bf in band_fracs:
            x_band = float(bf) * (x_max - x_min)
            mask_xband = np.abs(xv - x_thresh) <= x_band
            mask_top_turn = mask_top & mask_xband
            mask_bot_turn = mask_bot & mask_xband
            n_top = int(np.sum(mask_top_turn))
            n_bot = int(np.sum(mask_bot_turn))

            if (n_top >= min_pts_per_arm) and (n_bot >= min_pts_per_arm):
                turn_top_1d = float(np.nanmedian(p1dv[mask_top_turn]))
                turn_bot_1d = float(np.nanmedian(p1dv[mask_bot_turn]))
                turns = np.sort([turn_top_1d, turn_bot_1d])
                return {
                    "turn1_1d": float(turns[0]),
                    "turn2_1d": float(turns[1]),
                    "x_thresh": float(x_thresh),
                    "x_band": float(x_band),
                    "band_frac": float(bf),
                    "xv": xv,
                    "yv": yv,
                    "mask_top_turn": mask_top_turn,
                    "mask_bot_turn": mask_bot_turn,
                    "n_top": n_top,
                    "n_bot": n_bot,
                }

        return {"turn1_1d": None, "turn2_1d": None, "x_thresh": float(x_thresh), "band_frac": None, "xv": xv, "yv": yv}

    def _get_activity_matrix(LT):
        if want_S:
            return np.asarray(LT.S)
        if want_C:
            return np.asarray(LT.C)
        if want_YrA:
            return np.asarray(LT.YrA)
        return np.asarray(LT.S)

    def _make_bin_edges(pos1, pos2, n_bins, use_shared=True, mode="overlap"):
        pos1_v = np.asarray(pos1, float)
        pos2_v = np.asarray(pos2, float)
        pos1_v = pos1_v[np.isfinite(pos1_v)]
        pos2_v = pos2_v[np.isfinite(pos2_v)]
        if pos1_v.size < 10 or pos2_v.size < 10:
            return None, None

        lo1, hi1 = float(np.nanmin(pos1_v)), float(np.nanmax(pos1_v))
        lo2, hi2 = float(np.nanmin(pos2_v)), float(np.nanmax(pos2_v))

        if use_shared:
            if mode == "overlap":
                lo = max(lo1, lo2)
                hi = min(hi1, hi2)
            else:
                lo = min(lo1, lo2)
                hi = max(hi1, hi2)
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo) < 1e-6:
                return None, None
            edges = np.linspace(lo, hi, int(n_bins) + 1)
            return edges, edges
        else:
            return (
                np.linspace(lo1, hi1, int(n_bins) + 1),
                np.linspace(lo2, hi2, int(n_bins) + 1),
            )

    # -------------------------
    # PASS 1: compute tunings + cache + global scaling if needed
    # -------------------------
    cache = {}
    global_scale = np.nan
    global_samples = []  # for percentile global scaling (finite values)

    sort_reverse = False
    arm_cutoff_perc = 0.8

    for mouse, group in mouse_groups.items():
        LT1 = LT1_group[mouse]
        LT2 = LT2_group[mouse]

        _ = LT1.get_S_mapping(mapping)
        _ = LT2.get_S_mapping(mapping)

        loc_X_LT1 = LT1.loc_X_miniscope_smooth
        loc_Y_LT1 = LT1.loc_Y_miniscope_smooth
        loc_1d_LT1 = LT1.miniscope_loc_1d_px * _LT_1D_CM_PER_PX

        loc_X_LT2 = LT2.loc_X_miniscope_smooth
        loc_Y_LT2 = LT2.loc_Y_miniscope_smooth
        loc_1d_LT2 = LT2.miniscope_loc_1d_px * _LT_1D_CM_PER_PX

        df_mapping = LT1.crossreg.get_mappings_cells(mapping_type=mapping)
        cells_LT1_all = df_mapping[LT1.session_group].astype(float).astype(int).tolist()
        cells_LT2_all = df_mapping[LT2.session_group].astype(float).astype(int).tolist()

        # ----- sig_responses filtering (adapted for sort_by) -----
        if sort_by == "LT1":
            sig_keys = set(getattr(LT1.fm, "sig_responses", {}).keys())
            sig_label = "LT1"
        else:  # sort_by == "LT2"
            sig_keys = set(getattr(LT2.fm, "sig_responses", {}).keys())
            sig_label = "LT2"

        if filter_to_LT1_sig_cells:
            pairs = list(zip(cells_LT1_all, cells_LT2_all))
            if sort_by == "LT1":
                keep_pairs = [(c1, c2) for (c1, c2) in pairs if (c1 in sig_keys)]
            else:
                keep_pairs = [(c1, c2) for (c1, c2) in pairs if (c2 in sig_keys)]
            n_not_sig = len(pairs) - len(keep_pairs)
            if n_not_sig > 0:
                print(f"[INFO] mouse={mouse}: excluded {n_not_sig}/{len(pairs)} mapped cells not in {sig_label}.fm.sig_responses")
            if len(keep_pairs) == 0:
                print(f"[WARN] mouse={mouse}: after {sig_label}.sig filter, no mapped cells remain. Skipping.")
                continue
            cells_LT1 = [c1 for (c1, _c2) in keep_pairs]
            cells_LT2 = [c2 for (_c1, c2) in keep_pairs]
        else:
            if sort_by == "LT1":
                n_not_sig = sum((c1 not in sig_keys) for c1 in cells_LT1_all)
            else:
                n_not_sig = sum((c2 not in sig_keys) for c2 in cells_LT2_all)
            if n_not_sig > 0:
                print(f"[INFO] mouse={mouse}: mapping has {n_not_sig}/{len(cells_LT1_all)} {sig_label} cells not in {sig_label}.fm.sig_responses (not filtered)")
            cells_LT1 = cells_LT1_all
            cells_LT2 = cells_LT2_all

        # ----- Sorting by PF centres (sort_by selects the reference session) -----
        if sort_by == "LT1":
            sort_LT = LT1
            sort_cells = cells_LT1
            sort_loc_X = loc_X_LT1
            sort_loc_Y = loc_Y_LT1
            sort_loc_1d = loc_1d_LT1
        else:  # "LT2"
            sort_LT = LT2
            sort_cells = cells_LT2
            sort_loc_X = loc_X_LT2
            sort_loc_Y = loc_Y_LT2
            sort_loc_1d = loc_1d_LT2

        XY_sort = np.column_stack([sort_loc_X, sort_loc_Y])
        track_ref = np.nanmax(sort_loc_1d) if sort_reverse else np.nanmin(sort_loc_1d)

        bin_w = float(sort_LT.fm.loc.bin_width)
        min_x = float(getattr(sort_LT.fm.loc, "MIN_X"))
        min_y = float(getattr(sort_LT.fm.loc, "MIN_Y"))

        sorted_entries = []
        for cell_id in sort_cells:
            merged = sort_LT.fm.pf.merged_means.get(cell_id, None)
            model = sort_LT.fm.pf.model_.get(cell_id, None)
            sig = sort_LT.fm.sig_responses.get(cell_id, None)

            if merged is None or model is None or not hasattr(model, "means_"):
                continue

            merged_means = merged
            means_rc = np.asarray(model.means_, dtype=float)

            # PF count filter
            if max_pf_count is not None and len(merged_means) > int(max_pf_count):
                continue

            mean_to_group = {}
            for g_idx, grp in enumerate(merged_means):
                for m_idx in grp:
                    mean_to_group[int(m_idx)] = int(g_idx)

            pts_by_group = [[] for _ in range(len(merged_means))]
            use_sig = (sig is not None and len(sig) > 0)

            if use_sig:
                sig = np.asarray(sig, dtype=float)
                rc = sig[:, :2]
                pred_mean_idx = np.asarray(model.predict(rc), dtype=int)
                for (r, c), m_idx in zip(rc, pred_mean_idx):
                    g = mean_to_group.get(int(m_idx), None)
                    if g is None:
                        continue
                    pts_by_group[g].append((float(r), float(c)))

            pf_centers = []
            for grp_i, grp in enumerate(merged_means):
                grp = list(grp)
                if len(grp) == 0:
                    continue

                pts = pts_by_group[grp_i]
                if len(pts) > 0:
                    pts = np.asarray(pts, dtype=float)
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

            pf_candidates = []
            for grp_i, ctr_xy in pf_centers:
                d2 = np.sum((XY_sort - ctr_xy[None, :]) ** 2, axis=1)
                nn_idx = int(np.argmin(d2))
                nn_loc1d = float(sort_loc_1d[nn_idx])
                pf_candidates.append((grp_i, ctr_xy, nn_idx, nn_loc1d))

            if len(pf_candidates) == 0:
                continue

            chosen = min(pf_candidates, key=lambda x: abs(x[-1] - track_ref))

            chosen_grp_i, chosen_ctr_xy, nn_idx, nn_loc1d = chosen
            sorted_entries.append((cell_id, chosen_grp_i, chosen_ctr_xy, nn_idx, nn_loc1d))

        sorted_entries.sort(key=lambda x: x[-1], reverse=bool(sort_reverse))

        # Build the sorted lists for both sessions via the cross-reg mapping
        if sort_by == "LT1":
            sorted_LT1 = [cell_id for (cell_id, *_rest) in sorted_entries]
            lt1_to_lt2 = {c1: c2 for c1, c2 in zip(cells_LT1, cells_LT2)}
            sorted_LT2 = [lt1_to_lt2[c1] for c1 in sorted_LT1 if c1 in lt1_to_lt2]
        else:  # sort_by == "LT2"
            sorted_LT2 = [cell_id for (cell_id, *_rest) in sorted_entries]
            lt2_to_lt1 = {c2: c1 for c1, c2 in zip(cells_LT1, cells_LT2)}
            sorted_LT1 = [lt2_to_lt1[c2] for c2 in sorted_LT2 if c2 in lt2_to_lt1]

        if len(sorted_LT1) == 0 or len(sorted_LT2) == 0:
            print(f"[WARN] Empty sorted list for mouse={mouse}. Skipping.")
            continue

        # Extract activity matrices + rows
        S_LT1 = _get_activity_matrix(LT1)
        S_LT2 = _get_activity_matrix(LT2)

        pos1 = np.asarray(loc_1d_LT1, float)
        pos2 = np.asarray(loc_1d_LT2, float)

        n_cells_S1, _ = S_LT1.shape
        n_cells_S2, _ = S_LT2.shape

        if all(isinstance(c, (int, np.integer)) for c in sorted_LT1) and max(sorted_LT1) < n_cells_S1:
            row_idx1 = [int(c) for c in sorted_LT1]
        else:
            cell_to_row1 = {cell_id: i for i, cell_id in enumerate(cells_LT1)}
            row_idx1 = [cell_to_row1[c] for c in sorted_LT1 if c in cell_to_row1]

        if all(isinstance(c, (int, np.integer)) for c in sorted_LT2) and max(sorted_LT2) < n_cells_S2:
            row_idx2 = [int(c) for c in sorted_LT2]
        else:
            cell_to_row2 = {cell_id: i for i, cell_id in enumerate(cells_LT2)}
            row_idx2 = [cell_to_row2[c] for c in sorted_LT2 if c in cell_to_row2]

        S1_sorted = S_LT1[row_idx1, :]
        S2_sorted = S_LT2[row_idx2, :]

        n_rows = min(S1_sorted.shape[0], S2_sorted.shape[0])
        S1_sorted = S1_sorted[:n_rows, :]
        S2_sorted = S2_sorted[:n_rows, :]
        sorted_LT1 = sorted_LT1[:n_rows]
        sorted_LT2 = sorted_LT2[:n_rows]

        # Bin edges (shared by default)
        bin_edges1, bin_edges2 = _make_bin_edges(
            pos1, pos2, n_bins=pv_n_pos_bins, use_shared=pv_use_shared_bins, mode=pv_shared_bins_mode
        )
        if bin_edges1 is None or bin_edges2 is None:
            print(f"[WARN] Could not build bin edges for mouse={mouse}. Skipping.")
            continue

        bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
        bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

        tuning1 = _compute_tuning(S1_sorted, pos1, bin_edges1)
        tuning2 = _compute_tuning(S2_sorted, pos2, bin_edges2)

        tuning1 = _smooth_rows(tuning1, pv_smooth_bins)
        tuning2 = _smooth_rows(tuning2, pv_smooth_bins)

        # Collect global samples if needed for percentile scaling
        if normalize_global and global_norm_mode == "pctl":
            a = tuning1[np.isfinite(tuning1)].ravel()
            b = tuning2[np.isfinite(tuning2)].ravel()
            if a.size:
                global_samples.append(a)
            if b.size:
                global_samples.append(b)

        # Or update global max if requested
        if normalize_global and global_norm_mode == "max":
            vmax_local = np.nanmax([np.nanmax(tuning1), np.nanmax(tuning2)])
            if np.isfinite(vmax_local):
                global_scale = vmax_local if not np.isfinite(global_scale) else max(global_scale, vmax_local)

        cache[mouse] = dict(
            group=group,
            LT1=LT1,
            LT2=LT2,
            loc_X_LT1=loc_X_LT1,
            loc_Y_LT1=loc_Y_LT1,
            loc_1d_LT1=loc_1d_LT1,
            loc_X_LT2=loc_X_LT2,
            loc_Y_LT2=loc_Y_LT2,
            loc_1d_LT2=loc_1d_LT2,
            sorted_LT1=sorted_LT1,
            sorted_LT2=sorted_LT2,
            tuning1=tuning1,
            tuning2=tuning2,
            bin_edges1=bin_edges1,
            bin_edges2=bin_edges2,
            bin_centers1=bin_centers1,
            bin_centers2=bin_centers2,
        )

    # Finalize global percentile scale if requested
    if normalize_global and global_norm_mode == "pctl":
        if len(global_samples) == 0:
            print("[WARN] normalize_global+pctl requested but no finite samples found. Falling back to per-cell display.")
            normalize_global = False
        else:
            allv = np.concatenate(global_samples, axis=0)
            global_scale = float(np.nanpercentile(allv, float(global_norm_pctl)))
            if (not np.isfinite(global_scale)) or (global_scale <= 0):
                print("[WARN] global percentile scale is not finite/positive. Falling back to per-cell display.")
                normalize_global = False

    if normalize_global and ((not np.isfinite(global_scale)) or (global_scale <= 0)):
        print("[WARN] normalize_global=True but global_scale invalid. Falling back to per-cell display.")
        normalize_global = False

    # -------------------------
    # PASS 2: plotting + PV corr + sanity
    # -------------------------
    pv_corr_LT1_LT2 = {}

    cmap_pf = mpl.cm.get_cmap("viridis").copy()
    cmap_pf.set_bad(color="black")
    cmap_pv = mpl.cm.get_cmap("viridis").copy()
    cmap_pv.set_bad(color="black")

    for mouse, payload in cache.items():
        group = payload["group"]
        LT1 = payload["LT1"]
        LT2 = payload["LT2"]

        loc_X_LT1 = payload["loc_X_LT1"]
        loc_Y_LT1 = payload["loc_Y_LT1"]
        loc_1d_LT1 = payload["loc_1d_LT1"]
        loc_X_LT2 = payload["loc_X_LT2"]
        loc_Y_LT2 = payload["loc_Y_LT2"]
        loc_1d_LT2 = payload["loc_1d_LT2"]

        sorted_LT1 = payload["sorted_LT1"]
        sorted_LT2 = payload["sorted_LT2"]

        tuning1_raw = payload["tuning1"]
        tuning2_raw = payload["tuning2"]
        bin_edges1 = payload["bin_edges1"]
        bin_edges2 = payload["bin_edges2"]
        bin_centers1 = payload["bin_centers1"]
        bin_centers2 = payload["bin_centers2"]

        # Turn lines
        turn1_info = _compute_turn_lines_safe(loc_X_LT1, loc_Y_LT1, loc_1d_LT1, arm_cutoff_perc=arm_cutoff_perc)
        turn2_info = _compute_turn_lines_safe(loc_X_LT2, loc_Y_LT2, loc_1d_LT2, arm_cutoff_perc=arm_cutoff_perc)
        lt1_turn_a, lt1_turn_b = turn1_info.get("turn1_1d"), turn1_info.get("turn2_1d")
        lt2_turn_a, lt2_turn_b = turn2_info.get("turn1_1d"), turn2_info.get("turn2_1d")

        # Optional hook (unchanged)
        if use_sig_responses:
            print("[INFO] use_sig_responses=True is not implemented in this compact full-function version.")

        # -------------------------
        # Build normalized versions (for display, and optionally for PV)
        # -------------------------
        t1 = tuning1_raw.copy()
        t2 = tuning2_raw.copy()

        # (A) scalar scaling (global / per-mouse)
        norm_tag_parts = []
        vmin, vmax = None, None

        if normalize_global:
            scale = float(global_scale)
            t1 = t1 / scale
            t2 = t2 / scale
            norm_tag_parts.append(f"global-{global_norm_mode}{'' if global_norm_mode=='max' else f'{global_norm_pctl:g}'}")
        elif normalize_per_mouse:
            if mouse_norm_mode == "max":
                mouse_scale = float(np.nanmax([np.nanmax(t1), np.nanmax(t2)]))
            else:
                vals = np.concatenate([t1[np.isfinite(t1)].ravel(), t2[np.isfinite(t2)].ravel()], axis=0)
                mouse_scale = float(np.nanpercentile(vals, float(mouse_norm_pctl))) if vals.size else np.nan

            if (not np.isfinite(mouse_scale)) or (mouse_scale <= 0):
                mouse_scale = np.nan
            else:
                t1 = t1 / mouse_scale
                t2 = t2 / mouse_scale
                norm_tag_parts.append(f"perMouse-{mouse_norm_mode}{'' if mouse_norm_mode=='max' else f'{mouse_norm_pctl:g}'}")

        # (B) per-cell scaling
        if normalize_pairwise_per_cell:
            t1, t2 = _pairwise_per_cell_scale(t1, t2, mode=pairwise_cell_mode, pctl=pairwise_cell_pctl)
            norm_tag_parts.append(f"pairCell-{pairwise_cell_mode}{'' if pairwise_cell_mode=='max' else f'{pairwise_cell_pctl:g}'}")
        else:
            # default display behavior if no scalar normalization requested
            # (keeps your prior visual standard)
            if not (normalize_global or normalize_per_mouse):
                t1 = _normalize_per_cell_max(t1)
                t2 = _normalize_per_cell_max(t2)
                norm_tag_parts.append("perCell-max")

        # Set display vmin/vmax if we used per-cell or scalar normalization
        # (They’ll typically be in [0,1] but not guaranteed if you later change modes)
        vmin, vmax = 0.0, 1.0 if (normalize_global or normalize_per_mouse or normalize_pairwise_per_cell or (not (normalize_global or normalize_per_mouse))) else (None, None)

        tuning1_plot = t1
        tuning2_plot = t2

        # -------------------------
        # PV correlation matrix source (raw vs normalized)
        # -------------------------
        t1_pv = tuning1_plot if pv_use_normalized else tuning1_raw
        t2_pv = tuning2_plot if pv_use_normalized else tuning2_raw

        n_bins = t1_pv.shape[1]
        pv_corr = np.full((n_bins, n_bins), np.nan, float)
        for i in range(n_bins):
            v1 = t1_pv[:, i]
            for j in range(n_bins):
                v2 = t2_pv[:, j]
                pv_corr[i, j] = _safe_corr(v1, v2)

        diag = np.diag(pv_corr)
        finite = np.isfinite(diag)
        valid_frac = float(np.mean(finite)) if diag.size else 0.0
        if valid_frac < float(pv_diag_min_valid_frac):
            pv_diag_value = np.nan
        else:
            pv_diag_value = float(np.nanmean(diag)) if pv_diag_summary == "mean" else float(np.nanmedian(diag))

        # ---- Full-track alternative metrics ----
        pv_metrics_full = compute_pv_corr_metrics(pv_corr, bin_centers=bin_centers1, diag_summary=pv_diag_summary)

        # ---- Zone-split metrics (if turn lines found for BOTH sessions) ----
        # Use average of LT1 and LT2 turn points for robustness
        zone_metrics = {}
        t1a, t1b = lt1_turn_a, lt1_turn_b
        t2a, t2b = lt2_turn_a, lt2_turn_b
        if t1a is not None and t1b is not None and t2a is not None and t2b is not None:
            avg_turn1 = 0.5 * (t1a + t2a)
            avg_turn2 = 0.5 * (t1b + t2b)
            zone_metrics = compute_pv_corr_by_zone(pv_corr, bin_centers1, avg_turn1, avg_turn2, diag_summary=pv_diag_summary)
        elif t1a is not None and t1b is not None:
            zone_metrics = compute_pv_corr_by_zone(pv_corr, bin_centers1, t1a, t1b, diag_summary=pv_diag_summary)
        elif t2a is not None and t2b is not None:
            zone_metrics = compute_pv_corr_by_zone(pv_corr, bin_centers1, t2a, t2b, diag_summary=pv_diag_summary)
        else:
            print(f"[WARN] {mouse}: no turn lines found for either LT session; zone_metrics empty.")

        pv_corr_LT1_LT2[mouse] = {
            "pv_corr": pv_corr,
            "pv_diag": pv_diag_value,
            "pv_diag_valid_frac": valid_frac,
            "pv_diag_vec": diag,
            "pv_metrics": pv_metrics_full,
            "zone_metrics": zone_metrics,
            "bin_edges_LT1": bin_edges1,
            "bin_edges_LT2": bin_edges2,
            "bin_centers_LT1": bin_centers1,
            "bin_centers_LT2": bin_centers2,
            "sorted_LT1": sorted_LT1,
            "sorted_LT2": sorted_LT2,
            "pv_used_normalized": bool(pv_use_normalized),
            "norm_tag": "+".join(norm_tag_parts) if norm_tag_parts else "raw",
            "lt1_turn_a": lt1_turn_a, "lt1_turn_b": lt1_turn_b,
            "lt2_turn_a": lt2_turn_a, "lt2_turn_b": lt2_turn_b,
            "sort_by": sort_by,
        }

        # -------------------------
        # Figure 1: tuning heatmaps
        # -------------------------
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 7.5), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08})

        # When sort_by="LT2", put the sorting session (LT2) on the left
        if sort_by == "LT2":
            left_tuning, right_tuning = tuning2_plot, tuning1_plot
            left_edges, right_edges = bin_edges2, bin_edges1
            left_title, right_title = "LT2", "LT1"
            left_turn_a, left_turn_b = lt2_turn_a, lt2_turn_b
            right_turn_a, right_turn_b = lt1_turn_a, lt1_turn_b
        else:
            left_tuning, right_tuning = tuning1_plot, tuning2_plot
            left_edges, right_edges = bin_edges1, bin_edges2
            left_title, right_title = "LT1", "LT2"
            left_turn_a, left_turn_b = lt1_turn_a, lt1_turn_b
            right_turn_a, right_turn_b = lt2_turn_a, lt2_turn_b

        axes[0].imshow(
            left_tuning, aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap_pf, vmin=vmin, vmax=vmax,
            extent=[float(left_edges[0]), float(left_edges[-1]), left_tuning.shape[0], 1],
        )
        axes[0].set_title(f"{left_title}")
        axes[0].set_xlabel(f"Linearized track position ({_LT_1D_DISTANCE_UNIT})")
        axes[0].set_ylabel("Place cells")
        axes[0].set_yticks([1, left_tuning.shape[0]])
        axes[0].set_yticklabels(["1", f"{left_tuning.shape[0]}"])
        if (left_turn_a is not None) and (left_turn_b is not None):
            axes[0].axvline(left_turn_a, linestyle="--", linewidth=1.5)
            axes[0].axvline(left_turn_b, linestyle="--", linewidth=1.5)

        axes[1].imshow(
            right_tuning, aspect="auto", interpolation="nearest", origin="upper",
            cmap=cmap_pf, vmin=vmin, vmax=vmax,
            extent=[float(right_edges[0]), float(right_edges[-1]), right_tuning.shape[0], 1],
        )
        axes[1].set_title(f"{right_title}")
        axes[1].set_xlabel(f"Linearized track position ({_LT_1D_DISTANCE_UNIT})")
        axes[1].set_yticks([1, right_tuning.shape[0]])
        axes[1].set_yticklabels(["1", f"{right_tuning.shape[0]}"])
        if (right_turn_a is not None) and (right_turn_b is not None):
            axes[1].axvline(right_turn_a, linestyle="--", linewidth=1.5)
            axes[1].axvline(right_turn_b, linestyle="--", linewidth=1.5)

        norm_tag = "+".join(norm_tag_parts) if norm_tag_parts else "raw"
        sort_info = f" | sorted by {sort_by}" if sort_by != "LT1" else ""
        fig.suptitle(
            f"{mouse} | {session_str} | crossreg {mapping} | norm={norm_tag} | PV={'norm' if pv_use_normalized else 'raw'} | diag={pv_diag_value:.3f}{sort_info}",
            y=0.99,
        )
        fig.tight_layout()

        savefile = os.path.join(
            save_path,
            f"tuning_LT1_LT2_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}_norm-{norm_tag}_sortby-{sort_by}.png",
        )
        fig.savefig(savefile, dpi=300)
        plt.show()
        if auto_close:
            plt.close(fig)

        # -------------------------
        # Figure 2: PV corr matrix
        # -------------------------
        fig2, ax = plt.subplots(1, 1, figsize=(4.8, 4.2), dpi=200)
        ax.imshow(
            pv_corr,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            cmap=cmap_pv,
            extent=[float(bin_edges2[0]), float(bin_edges2[-1]), float(bin_edges1[-1]), float(bin_edges1[0])],
        )
        # Overlay zone boundary lines on PV matrix
        avg_ta = None
        avg_tb = None
        if lt1_turn_a is not None and lt2_turn_a is not None:
            avg_ta = 0.5 * (lt1_turn_a + lt2_turn_a)
        elif lt1_turn_a is not None:
            avg_ta = lt1_turn_a
        elif lt2_turn_a is not None:
            avg_ta = lt2_turn_a

        if lt1_turn_b is not None and lt2_turn_b is not None:
            avg_tb = 0.5 * (lt1_turn_b + lt2_turn_b)
        elif lt1_turn_b is not None:
            avg_tb = lt1_turn_b
        elif lt2_turn_b is not None:
            avg_tb = lt2_turn_b

        for tval in [avg_ta, avg_tb]:
            if tval is not None:
                ax.axhline(tval, color="white", ls="--", lw=1.0, alpha=0.7)
                ax.axvline(tval, color="white", ls="--", lw=1.0, alpha=0.7)

        diag_ex = pv_metrics_full.get("diagonal_excess", np.nan)
        spec_ix = pv_metrics_full.get("specificity_idx", np.nan)
        ax.set_title(f"PV corr ({'norm' if pv_use_normalized else 'raw'}): diag={pv_diag_value:.3f} excess={diag_ex:.3f}")
        ax.set_xlabel(f"LT2 position ({_LT_1D_DISTANCE_UNIT})")
        ax.set_ylabel(f"LT1 position ({_LT_1D_DISTANCE_UNIT})")
        fig2.tight_layout()

        savefile2 = os.path.join(
            save_path,
            f"pv_corr_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}_PV-{'norm' if pv_use_normalized else 'raw'}.png",
        )
        fig2.savefig(savefile2, dpi=300)
        plt.show()
        if auto_close:
            plt.close(fig2)

        # -------------------------
        # OPTIONAL sanity blocks (unchanged in spirit; kept compact)
        # -------------------------
        if want_sanity_sanity:
            # Turn sanity plots (LT1/LT2), hardened
            for which, info in [("LT1", turn1_info), ("LT2", turn2_info)]:
                if (info.get("turn1_1d") is not None) and ("xv" in info):
                    try:
                        xv = info["xv"]; yv = info["yv"]; x_thresh = info["x_thresh"]
                        mt = info.get("mask_top_turn", None); mb = info.get("mask_bot_turn", None)

                        figS, axS = plt.subplots(1, 1, figsize=(10, 3), dpi=150)
                        axS.scatter(xv, yv, s=4, alpha=0.25)
                        axS.axvline(x_thresh, linestyle="--", linewidth=2)
                        if mt is not None:
                            axS.scatter(xv[mt], yv[mt], s=10, alpha=0.8, label=f"{which} top-arm near x_thresh")
                        if mb is not None:
                            axS.scatter(xv[mb], yv[mb], s=10, alpha=0.8, label=f"{which} bottom-arm near x_thresh")
                        axS.set_title(f"{which} turn cutoff sanity check")
                        axS.set_xlabel("X"); axS.set_ylabel("Y")
                        axS.legend(loc="upper left", fontsize=8, frameon=False)
                        figS.tight_layout()

                        sfile = os.path.join(
                            save_path,
                            f"turn_sanity_{which}_{mouse}_{group}_{session_str}_crossreg_{mapping}_{S_file_str}.png",
                        )
                        figS.savefig(sfile, dpi=300)
                        plt.show()
                        if auto_close:
                            plt.close(figS)
                    except Exception as e:
                        print(f"[WARN] {which} turn sanity plot skipped (error): {e}")

            # PF sanity plots (LT1)
            rng = default_rng()
            sig_dict = getattr(LT1.fm, "sig_responses", {})
            sig_keys_arr = np.array(list(sig_dict.keys()), dtype=int) if len(sig_dict) else np.array([], dtype=int)
            if sig_keys_arr.size > 0:
                sanity_sanity_cells = rng.choice(
                    sig_keys_arr,
                    size=min(how_many_sanity_sanity_cells, len(sig_keys_arr)),
                    replace=False,
                )
                for cell_id_plot in sanity_sanity_cells:
                    try:
                        F = LT1.fm.fluorescence_map_occup[:, :, cell_id_plot]
                        sig = sig_dict.get(cell_id_plot, [])
                        sig = np.asarray(sig, dtype=float) if len(sig) > 0 else np.zeros((0, 3), dtype=float)
                        sr = sig[:, 0] if sig.shape[0] > 0 else np.array([])
                        sc = sig[:, 1] if sig.shape[0] > 0 else np.array([])

                        means_rc = np.asarray(LT1.fm.pf.model_[cell_id_plot].means_)
                        merged = LT1.fm.pf.merged_means[cell_id_plot]

                        pf_centers = []
                        for grp_m in merged:
                            grp_m = list(grp_m)
                            if len(grp_m) == 0:
                                continue
                            pf_centers.append(means_rc[grp_m].mean(axis=0))
                        pf_centers = np.asarray(pf_centers) if len(pf_centers) > 0 else np.zeros((0, 2), dtype=float)

                        pfr = pf_centers[:, 0] if pf_centers.size else np.array([])
                        pfc = pf_centers[:, 1] if pf_centers.size else np.array([])

                        figC, axC = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
                        axC.imshow(F, origin="upper", aspect="auto", cmap="viridis")
                        if sig.shape[0] > 0:
                            axC.scatter(sc, sr, s=45, marker="v", alpha=0.9)
                        if pf_centers.size > 0:
                            axC.scatter(pfc, pfr, s=200, marker="o", edgecolors="white", linewidths=2, alpha=0.95)

                        axC.set_title(f"cell {cell_id_plot} place fields")
                        axC.set_xlabel("Column bin (horizontal)")
                        axC.set_ylabel("Row bin (vertical)")
                        figC.tight_layout()

                        sfile = os.path.join(
                            save_path,
                            f"pf_means_sanity_{mouse}_{group}_{session_str}_crossreg_{mapping}_cell{cell_id_plot}_{S_file_str}.png",
                        )
                        figC.savefig(sfile, dpi=300)
                        plt.show()
                        if auto_close:
                            plt.close(figC)
                    except Exception as e:
                        print(f"[WARN] PF sanity plot skipped for cell={cell_id_plot}: {e}")

    return pv_corr_LT1_LT2


# =====================================================================
# Group-level boxplots for tiling quality metrics
# =====================================================================

def plot_tiling_metrics_anova(
    tiling_results,
    mouse_groups,
    PLOTS_DIR=".",
    session_str="TFC_cond_LT1",
    pf_str="pfALL",
    metric_keys=None,
    auto_close=True,
    paper_plots=False,
    mapping=None,
):
    """
    One-way ANOVA + post-hoc boxplots for within-session tiling quality
    metrics (COM ordering, diagonal band concentration, spatial info,
    sparsity, peak-to-mean).

    Parameters
    ----------
    tiling_results : dict
        Return value of plot_lt_within_session_tuning.
        tiling_results[mouse] = {"group": ..., "tiling": {...metrics...}, ...}
    mouse_groups : dict
        {mouse: group_label}
    session_str : str
        e.g. 'TFC_cond_LT1' or 'TFC_cond_LT2' — used for titles / filenames.
    metric_keys : list or None
        Which tiling metrics to plot. None → all five.
    paper_plots : bool
        If True, also produce compact publication-quality figures.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import f_oneway, ttest_ind

    map_tag = _mapping_tag(mapping)

    # --- Use in filenames too ---
    fname_prefix = f"{session_str}__map_{map_tag}__{pf_str}"

    if metric_keys is None:
        metric_keys = [
            "com_ordering",
            "diag_band_conc",
            "spatial_info",
            "sparsity",
            "peak_to_mean",
        ]

    metric_label_map = {
        "com_ordering":   "COM ordering (Spearman ρ)",
        "diag_band_conc": "Diagonal band concentration",
        "spatial_info":   "Spatial information (bits/event)",
        "sparsity":       "Sparsity (lower = sharper)",
        "peak_to_mean":   "Peak-to-mean ratio",
    }

    _paper_ylabel_map = {
        "com_ordering":   "COM ordering",
        "diag_band_conc": "Diag. band conc.",
        "spatial_info":   "Spatial info.",
        "sparsity":       "Sparsity",
        "peak_to_mean":   "Peak / mean",
    }

    GROUP_COLORS = {
        "hM3D": "#d62728",
        "hM4D": "#1f77b4",
        "mCherry": "#2c2c2c",
    }
    GROUP_ORDER = ["mCherry", "hM3D", "hM4D"]
    _paper_label = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}
    _paper_order = ["mCherry", "hM3D", "hM4D"]

    # ---- Collect records ----
    records = []
    for mouse, d in tiling_results.items():
        if mouse not in mouse_groups:
            continue
        group = mouse_groups[mouse]
        tiling = d.get("tiling", {})
        if not tiling:
            continue

        rec = {"mouse": mouse, "group": group}
        for mk in metric_keys:
            rec[mk] = tiling.get(mk, np.nan)
        records.append(rec)

    if len(records) == 0:
        print("[WARN] plot_tiling_metrics_anova: no data.")
        return None

    df = pd.DataFrame(records)

    save_dir = os.path.join(
        PLOTS_DIR,
        f"lt_within_session_{session_str}_{pf_str}",
        f"tiling_metrics_mapping_{map_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # ---- Prepare stats summary file ----
    stats_lines = []

    # ---- One boxplot per metric ----
    for mk in metric_keys:
        col = df[mk].dropna()
        if col.size == 0:
            continue

        groups_present = [g for g in GROUP_ORDER if g in df["group"].values]
        data_by_group = [df.loc[df["group"] == g, mk].dropna().values for g in groups_present]


        # ANOVA
        if len(data_by_group) >= 2 and all(len(a) >= 1 for a in data_by_group):
            try:
                F, p_anova = f_oneway(*data_by_group)
            except Exception:
                F, p_anova = np.nan, np.nan
        else:
            F, p_anova = np.nan, np.nan

        stats_lines.append(f"=== {mk} ({metric_label_map.get(mk, mk)}) ===")
        stats_lines.append(f"ANOVA: F = {F:.4f}, p = {p_anova:.4g}")

        ylabel = metric_label_map.get(mk, mk)

        fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.2), dpi=200)
        bp_data = []
        bp_labels = []
        bp_colors = []
        for g in groups_present:
            vals = df.loc[df["group"] == g, mk].dropna().values
            bp_data.append(vals)
            bp_labels.append(g)
            bp_colors.append(GROUP_COLORS.get(g, "gray"))

        bplot = ax.boxplot(bp_data, patch_artist=True, widths=0.5, showfliers=False)
        for patch, color in zip(bplot["boxes"], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for patch in bplot["medians"]:
            patch.set_color("black")

        # Overlay individual mice
        for gi, g in enumerate(groups_present):
            vals = df.loc[df["group"] == g, mk].dropna().values
            mice = df.loc[df["group"] == g].dropna(subset=[mk])["mouse"].values
            x = np.full(len(vals), gi + 1) + np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=30, color=GROUP_COLORS.get(g, "gray"), edgecolors="white", linewidths=0.5, zorder=5)
            for xi, yi, mi in zip(x, vals, mice):
                ax.annotate(mi, (xi, yi), fontsize=5, ha="center", va="bottom", alpha=0.7)

        ax.set_xticklabels(bp_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Tiling: {ylabel}\nANOVA p={p_anova:.4f} | {session_str}", fontsize=8)

        # ---- Post-hoc pairwise t-tests with Holm-Bonferroni ----
        if len(groups_present) >= 2:
            from itertools import combinations
            pairs = list(combinations(range(len(groups_present)), 2))
            raw_ps = []
            for i1, i2 in pairs:
                a1, a2 = bp_data[i1], bp_data[i2]
                if len(a1) >= 2 and len(a2) >= 2:
                    _, p_raw = ttest_ind(a1, a2, equal_var=False)
                else:
                    p_raw = np.nan
                raw_ps.append(p_raw)
            # Holm-Bonferroni
            n_comp = len(raw_ps)
            sorted_idx = np.argsort(raw_ps)
            holm_ps = np.full(n_comp, np.nan)
            for rank_i, idx in enumerate(sorted_idx):
                holm_ps[idx] = raw_ps[idx] * (n_comp - rank_i)
            holm_ps = np.minimum(holm_ps, 1.0)

            # Significance brackets on the regular boxplot
            y_max = max(np.nanmax(v) for v in bp_data if len(v) > 0)
            y_range = y_max - min(np.nanmin(v) for v in bp_data if len(v) > 0)
            y_step = y_range * 0.08
            y_cur = y_max + y_step


            stats_lines.append("Pairwise t-tests (Holm-Bonferroni corrected):")
            for ci, (i1, i2) in enumerate(pairs):
                p_holm = holm_ps[ci]
                p_raw = raw_ps[ci]
                stats_lines.append(f"  {groups_present[i1]} vs {groups_present[i2]}: p_raw={p_raw:.4g}  p_holm={p_holm:.4g}")
                print(f"  {groups_present[i1]} vs {groups_present[i2]}: "
                      f"p_raw={p_raw:.4f}  p_holm={p_holm:.4f}")

                if p_holm < 0.001:
                    star_p = "***"
                elif p_holm < 0.01:
                    star_p = "**"
                elif p_holm < 0.05:
                    star_p = "*"
                elif p_raw < 0.05:
                    star_p = "*"
                elif 0.05 <= p_raw < 0.08:
                    star_p = f"p={p_raw:.2f}"
                else:
                    continue
                x1, x2 = i1 + 1, i2 + 1
                ax.plot([x1, x1, x2, x2], [y_cur, y_cur + y_step * 0.3, y_cur + y_step * 0.3, y_cur],
                        lw=0.8, color="black")
                ax.text((x1 + x2) / 2, y_cur + y_step * 0.35, star_p,
                        ha="center", va="bottom", fontsize=7)
                y_cur += y_step * 1.5

        fig.tight_layout()

        savefile = os.path.join(save_dir, f"tiling_{mk}_{session_str}.png")
        fig.savefig(savefile, dpi=300)
        print(f"[tiling] saved: {savefile}")
        plt.show()
        if auto_close:
            plt.close(fig)

        # ---- Paper-quality plot ----
        if paper_plots:
            fig_p, ax_p = plt.subplots(1, 1, figsize=(2.4, 3.0), dpi=300)
            paper_groups = [g for g in _paper_order if g in df["group"].values]
            paper_data = [df.loc[df["group"] == g, mk].dropna().values for g in paper_groups]
            paper_colors = [GROUP_COLORS.get(g, "gray") for g in paper_groups]
            paper_labels = [_paper_label.get(g, g) for g in paper_groups]

            bp2 = ax_p.boxplot(paper_data, patch_artist=True, widths=0.45, showfliers=False)
            for patch, color in zip(bp2["boxes"], paper_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
            for patch in bp2["medians"]:
                patch.set_color("black")

            for gi, g in enumerate(paper_groups):
                vals = df.loc[df["group"] == g, mk].dropna().values
                x = np.full(len(vals), gi + 1) + np.random.default_rng(42).uniform(-0.10, 0.10, len(vals))
                ax_p.scatter(x, vals, s=22, color=GROUP_COLORS.get(g, "gray"), edgecolors="white", linewidths=0.4, zorder=5)

            ax_p.set_xticklabels(paper_labels, fontsize=9)
            ax_p.set_ylabel(_paper_ylabel_map.get(mk, mk), fontsize=9)

            # Significance brackets
            if len(paper_groups) >= 2:
                y_max = max(np.nanmax(v) for v in paper_data if len(v) > 0)
                y_range = y_max - min(np.nanmin(v) for v in paper_data if len(v) > 0)
                y_step = y_range * 0.08
                y_cur = y_max + y_step

                pairs_p = list(combinations(range(len(paper_groups)), 2))
                raw_ps_p = []
                for i1, i2 in pairs_p:
                    a1, a2 = paper_data[i1], paper_data[i2]
                    if len(a1) >= 2 and len(a2) >= 2:
                        _, prw = ttest_ind(a1, a2, equal_var=False)
                    else:
                        prw = np.nan
                    raw_ps_p.append(prw)
                n_comp_p = len(raw_ps_p)
                sorted_idx_p = np.argsort(raw_ps_p)
                holm_ps_p = np.full(n_comp_p, np.nan)
                for ri, idx in enumerate(sorted_idx_p):
                    holm_ps_p[idx] = raw_ps_p[idx] * (n_comp_p - ri)
                holm_ps_p = np.minimum(holm_ps_p, 1.0)

                for ci, (i1, i2) in enumerate(pairs_p):
                    p_holm = holm_ps_p[ci]
                    p_raw = raw_ps_p[ci]
                    if p_holm < 0.001:
                        star_p = "***"
                    elif p_holm < 0.01:
                        star_p = "**"
                    elif p_holm < 0.05:
                        star_p = "*"
                    elif p_raw < 0.05:
                        star_p = "*"
                    elif 0.05 <= p_raw < 0.08:
                        star_p = f"p={p_raw:.2f}"
                    else:
                        continue
                    x1, x2 = i1 + 1, i2 + 1
                    ax_p.plot([x1, x1, x2, x2], [y_cur, y_cur + y_step * 0.3, y_cur + y_step * 0.3, y_cur], lw=0.8, color="black")
                    ax_p.text((x1 + x2) / 2, y_cur + y_step * 0.35, star_p, ha="center", va="bottom", fontsize=7)
                    y_cur += y_step * 1.5

            ax_p.spines["top"].set_visible(False)
            ax_p.spines["right"].set_visible(False)
            fig_p.tight_layout()

            sfile_p = os.path.join(save_dir, f"paper_tiling_{mk}_{session_str}.png")
            fig_p.savefig(sfile_p, dpi=300, bbox_inches="tight")
            print(f"[tiling] saved: {sfile_p}")
            sfile_pdf = os.path.join(save_dir, f"paper_tiling_{mk}_{session_str}.pdf")
            fig_p.savefig(sfile_pdf, bbox_inches="tight")
            print(f"[tiling] saved: {sfile_pdf}")
            plt.show()
            if auto_close:
                plt.close(fig_p)

    # ---- Write stats summary to file ----
    stats_file = os.path.join(save_dir, f"tiling_stats_{session_str}.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print(f"[tiling] stats summary saved: {stats_file}")

    return df


def plot_pv_corr_anova_prev(
    pv_corr_LT1_LT2,
    mouse_groups,
    title="PV correlation by group",
    ylabel="Average diagonal PV correlation",
):
    """
    Parameters
    ----------
    pv_corr_LT1_LT2 : dict
        Output of plot_lt_spatial_responses.
        pv_corr_LT1_LT2[mouse]['pv_corr'] is a 2D matrix.

    mouse_groups : dict
        mouse -> group label (e.g. 'hM3D', 'hM4D', 'mCherry')

    Returns
    -------
    results : dict
        Contains per-mouse values, ANOVA result, and Tukey table
    """

    # -------------------------------------------------
    # 1. Extract per-mouse diagonal PV correlation
    # -------------------------------------------------
    records = []

    for mouse, d in pv_corr_LT1_LT2.items():
        if mouse not in mouse_groups:
            continue

        pv = np.asarray(d.get("pv_corr", None))
        if pv is None or pv.ndim != 2:
            continue

        diag = np.diag(pv)
        diag = diag[np.isfinite(diag)]

        if diag.size == 0:
            continue

        mean_diag = np.mean(diag)

        records.append({
            "mouse": mouse,
            "group": mouse_groups[mouse],
            "pv_corr": mean_diag
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid diagonal PV correlations found.")

    # -------------------------------------------------
    # 2. One-way ANOVA
    # -------------------------------------------------
    groups = df["group"].unique()
    group_vals = [df.loc[df["group"] == g, "pv_corr"].values for g in groups]

    if sum(len(v) > 0 for v in group_vals) < 2:
        raise RuntimeError("Not enough groups with data for ANOVA.")

    F, p = stats.f_oneway(*group_vals)

    # -------------------------------------------------
    # 3. Tukey HSD
    # -------------------------------------------------
    tukey = pairwise_tukeyhsd(
        endog=df["pv_corr"].values,
        groups=df["group"].values,
        alpha=0.05
    )

    # -------------------------------------------------
    # 4. Plot
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=200)

    order = sorted(groups)
    data = [df.loc[df["group"] == g, "pv_corr"].values for g in order]

    bp = ax.boxplot(
        data,
        positions=np.arange(len(order)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    # Box colors (match your figure style)
    colors = {
        "hM3D": "#d62728",
        "hM4D": "#1f77b4",
        "mCherry": "#2c2c2c",
    }

    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(colors.get(g, "gray"))
        patch.set_alpha(0.35)

    # Scatter individual mice + labels
    for i, g in enumerate(order):
        sub = df[df["group"] == g]
        x = np.full(len(sub), i)
        ax.scatter(
            x,
            sub["pv_corr"],
            color=colors.get(g, "black"),
            s=30,
            zorder=3
        )

        for _, r in sub.iterrows():
            ax.text(
                i + 0.03,
                r["pv_corr"],
                r["mouse"],
                fontsize=7,
                va="center"
            )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 5. Print stats (as in your panel)
    # -------------------------------------------------
    print(f"ANOVA F = {F}")
    print(f"ANOVA p = {p}")
    print("\nMultiple Comparison of Means - Tukey HSD, FWER=0.05")
    print(tukey.summary())

    return {
        "dataframe": df,
        "anova": {"F": F, "p": p},
        "tukey": tukey
    }

def plot_pv_corr_anova(
    pv_corr_LT1_LT2,
    mouse_groups,
    PLOTS_DIR=None,
    type="global",
    mapping="LT1+LT2",
    pf_str="pfALL",
    title="PV correlation by group",
    ylabel="Average diagonal PV correlation",
    auto_close=True,
    paper_plots=False,
):
    """
    Statistical comparison of population vector similarity across groups.

    Uses:
      - per-mouse stored pv_diag (mean or median diagonal already computed)
      - Fisher z transform for ANOVA + Tukey
      - raw r values for plotting

    Parameters
    ----------
    PLOTS_DIR : str or None
        Base plots directory.  When provided the figure and stats text are
        saved into the same sub-folder used by plot_lt_spatial_responses:
            {PLOTS_DIR}/lt_spatial_responses_TFC_cond_LT_{type}_mapping_{mapping}/
    type : str
        Normalization type label, e.g. 'global' or 'pairwise'.
    mapping : str
        Cross-registration mapping label, e.g. 'LT1+LT2' or 'TFC_cond+LT1+LT2'.

    Returns:
        dict with dataframe, ANOVA stats, Tukey object
    """

    # -------------------------------------------------
    # Build save path (same directory as plot_lt_spatial_responses)
    # -------------------------------------------------
    save_path = None
    if PLOTS_DIR is not None:
        session_str = f"TFC_cond_LT_{type}"
        save_path = os.path.join(
            PLOTS_DIR,
            f"lt_spatial_responses_{session_str}_mapping_{mapping}_{pf_str}",
        )
        os.makedirs(save_path, exist_ok=True)

    # -------------------------------------------------
    # 1. Build per-mouse dataframe
    # -------------------------------------------------
    records = []

    for mouse, d in pv_corr_LT1_LT2.items():

        if mouse not in mouse_groups:
            continue

        r = d.get("pv_diag", np.nan)

        if not np.isfinite(r):
            continue

        records.append({
            "mouse": mouse,
            "group": mouse_groups[mouse],
            "pv_corr": float(r)
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid PV diagonal values found.")

    df["group"] = df["group"].astype("category")

    # -------------------------------------------------
    # 2. Fisher z transform for statistics
    # -------------------------------------------------
    df["pv_z"] = np.arctanh(df["pv_corr"].clip(-0.999999, 0.999999))

    # -------------------------------------------------
    # 3. One-way ANOVA on Fisher z
    # -------------------------------------------------
    groups = df["group"].unique()
    group_vals = [df.loc[df["group"] == g, "pv_z"].values for g in groups]

    if sum(len(v) > 0 for v in group_vals) < 2:
        raise RuntimeError("Not enough groups with data for ANOVA.")

    F, p = stats.f_oneway(*group_vals)

    # -------------------------------------------------
    # 4. Tukey HSD on Fisher z
    # -------------------------------------------------
    tukey = pairwise_tukeyhsd(
        endog=df["pv_z"].values,
        groups=df["group"].values,
        alpha=0.05
    )

    # -------------------------------------------------
    # 5. Plot RAW correlations (publication-style)
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=200)

    order = sorted(groups)
    data = [df.loc[df["group"] == g, "pv_corr"].values for g in order]

    bp = ax.boxplot(
        data,
        positions=np.arange(len(order)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    colors = {
        "hM3D": "#d62728",
        "hM4D": "#1f77b4",
        "mCherry": "#2c2c2c",
    }

    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(colors.get(g, "gray"))
        patch.set_alpha(0.35)

    # scatter individual mice
    for i, g in enumerate(order):
        sub = df[df["group"] == g]
        x = np.full(len(sub), i)
        ax.scatter(
            x,
            sub["pv_corr"],
            color=colors.get(g, "black"),
            s=30,
            zorder=3
        )

        # optional mouse labels
        for _, r in sub.iterrows():
            ax.text(
                i + 0.03,
                r["pv_corr"],
                r["mouse"],
                fontsize=7,
                va="center"
            )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, f"pv_corr_anova_{type}_{mapping}.png"),
            dpi=300,
        )

    if auto_close:
        plt.close(fig)

    # -------------------------------------------------
    # 5b. Paper-quality compact plot (optional)
    # -------------------------------------------------
    if paper_plots:
        _paper_group_order = ["mCherry", "hM3D", "hM4D"]
        _paper_labels = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}

        order_p = [g for g in _paper_group_order if g in groups]
        data_p = [df.loc[df["group"] == g, "pv_corr"].values for g in order_p]

        fig_p, ax_p = plt.subplots(figsize=(2.8, 3.2), dpi=300)
        bp_p = ax_p.boxplot(
            data_p, positions=np.arange(len(order_p)), widths=0.45,
            patch_artist=True, showfliers=False,
        )
        for patch, g in zip(bp_p["boxes"], order_p):
            patch.set_facecolor(colors.get(g, "gray"))
            patch.set_alpha(0.35)
        for i, g in enumerate(order_p):
            s_p = df[df["group"] == g]
            ax_p.scatter(np.full(len(s_p), i), s_p["pv_corr"],
                         color=colors.get(g, "black"), s=25, zorder=3)

        ax_p.set_xticks(np.arange(len(order_p)))
        ax_p.set_xticklabels([_paper_labels.get(g, g) for g in order_p])
        ax_p.set_ylabel("Mean PV correlation")
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        fig_p.tight_layout()
        if save_path is not None:
            fig_p.savefig(
                os.path.join(save_path, f"pv_corr_anova_{type}_{mapping}_paper.png"),
                dpi=300,
            )
            fig_p.savefig(
                os.path.join(save_path, f"pv_corr_anova_{type}_{mapping}_paper.svg"),
            )
        if auto_close:
            plt.close(fig_p)

    # -------------------------------------------------
    # 6. Collect stats text
    # -------------------------------------------------
    stat_lines = []
    stat_lines.append("ANOVA (on Fisher z)")
    stat_lines.append(f"F = {F:.4f}")
    stat_lines.append(f"p = {p:.6f}")
    stat_lines.append("")
    stat_lines.append("Multiple Comparison of Means - Tukey HSD (on Fisher z)")
    stat_lines.append(str(tukey.summary()))
    stat_lines.append("")
    stat_lines.append("Group means (raw r):")
    stat_lines.append(
        df.groupby("group")["pv_corr"].agg(["mean", "sem", "count"]).to_string()
    )

    stat_text = "\n".join(stat_lines)
    print(f"\n{stat_text}")

    if save_path is not None:
        with open(
            os.path.join(save_path, f"pv_corr_anova_{type}_{mapping}.txt"), "w"
        ) as f:
            f.write(stat_text)

    return {
        "dataframe": df,
        "anova": {"F": F, "p": p},
        "tukey": tukey
    }


# =====================================================================
# Tiling quality metrics for sorted place-field heatmaps
# =====================================================================

def compute_tiling_metrics(tuning_sorted, band_width_frac=0.10):
    """
    Quantify how well sorted place cells tile the track.

    Parameters
    ----------
    tuning_sorted : (n_cells, n_bins) array
        Tuning-curve matrix with rows sorted by PF centre.
        Typically peak-normalised to [0, 1].
    band_width_frac : float
        Half-width of the diagonal band as a fraction of n_bins
        (default 0.10 → ±10 %).

    Returns
    -------
    dict with keys
        com_ordering       : Spearman ρ between cell rank and tuning-curve COM
        com_ordering_p     : p-value of the Spearman test
        diag_band_conc     : mean fraction of each cell's activity within ±band
                             of the expected diagonal position
        spatial_info       : mean Skaggs spatial information (bits / event),
                             computed under uniform-occupancy assumption
        sparsity           : mean Skaggs sparsity (0–1, lower = sharper)
        peak_to_mean       : mean peak / mean ratio across cells
    """
    import numpy as np
    from scipy.stats import spearmanr

    T = np.asarray(tuning_sorted, float)
    n_cells, n_bins = T.shape

    out = dict(
        com_ordering=np.nan,
        com_ordering_p=np.nan,
        diag_band_conc=np.nan,
        spatial_info=np.nan,
        sparsity=np.nan,
        peak_to_mean=np.nan,
    )

    if n_cells < 3 or n_bins < 3:
        return out

    # --- 1. Centre-of-mass ordering (Spearman ρ) -------------------------
    bin_idx = np.arange(n_bins, dtype=float)
    coms = np.full(n_cells, np.nan)
    for i in range(n_cells):
        row = T[i]
        row_sum = np.nansum(row)
        if row_sum > 0:
            coms[i] = np.nansum(row * bin_idx) / row_sum
    valid = np.isfinite(coms)
    if valid.sum() >= 3:
        rho, p = spearmanr(np.arange(n_cells)[valid], coms[valid])
        out["com_ordering"] = float(rho)
        out["com_ordering_p"] = float(p)

    # --- 2. Diagonal band concentration -----------------------------------
    hw = max(1, int(round(band_width_frac * n_bins)))
    band_fracs = np.full(n_cells, np.nan)
    for i in range(n_cells):
        expected_bin = int(round(i / max(n_cells - 1, 1) * (n_bins - 1)))
        lo = max(0, expected_bin - hw)
        hi = min(n_bins, expected_bin + hw + 1)
        row = T[i]
        total = np.nansum(row)
        if total > 0:
            band_fracs[i] = np.nansum(row[lo:hi]) / total
    finite_bf = band_fracs[np.isfinite(band_fracs)]
    if finite_bf.size > 0:
        out["diag_band_conc"] = float(np.mean(finite_bf))

    # --- 3. Spatial information (Skaggs, uniform occupancy) ---------------
    # SI_cell = Σ_i (λ_i / λ_mean) * log2(λ_i / λ_mean) * p_i
    # with uniform occupancy: p_i = 1/n_bins
    p_occ = 1.0 / n_bins
    si_vals = np.full(n_cells, np.nan)
    for i in range(n_cells):
        row = T[i].copy()
        row[~np.isfinite(row)] = 0.0
        lam_mean = np.mean(row)
        if lam_mean > 0:
            ratio = row / lam_mean
            # avoid log(0)
            safe = ratio > 0
            si = np.sum(ratio[safe] * np.log2(ratio[safe]) * p_occ)
            si_vals[i] = si
    finite_si = si_vals[np.isfinite(si_vals)]
    if finite_si.size > 0:
        out["spatial_info"] = float(np.mean(finite_si))

    # --- 4. Sparsity (Skaggs) --------------------------------------------
    # sparsity = (Σ p_i λ_i)^2 / (Σ p_i λ_i^2)
    sp_vals = np.full(n_cells, np.nan)
    for i in range(n_cells):
        row = T[i].copy()
        row[~np.isfinite(row)] = 0.0
        mean1 = np.sum(p_occ * row)
        mean2 = np.sum(p_occ * row ** 2)
        if mean2 > 0:
            sp_vals[i] = (mean1 ** 2) / mean2
    finite_sp = sp_vals[np.isfinite(sp_vals)]
    if finite_sp.size > 0:
        out["sparsity"] = float(np.mean(finite_sp))

    # --- 5. Peak-to-mean ratio --------------------------------------------
    ptm_vals = np.full(n_cells, np.nan)
    for i in range(n_cells):
        row = T[i]
        mn = np.nanmean(row)
        pk = np.nanmax(row)
        if mn > 0:
            ptm_vals[i] = pk / mn
    finite_ptm = ptm_vals[np.isfinite(ptm_vals)]
    if finite_ptm.size > 0:
        out["peak_to_mean"] = float(np.mean(finite_ptm))

    return out


def compute_pv_corr_metrics(pv_corr, bin_centers=None, diag_summary="mean"):
    """
    Compute a battery of PV-correlation quality metrics from an n_bins x n_bins
    PV correlation matrix.

    Metrics returned (all scalar per mouse):
        diag_mean       – mean of diagonal (standard PV stability)
        diag_median     – median of diagonal
        offdiag_mean    – mean of off-diagonal entries
        diagonal_excess – diag_mean - offdiag_mean  (>0 = good spatial fidelity)
        specificity_idx – (diag - offdiag) / (diag + offdiag)  in [-1, 1]
        decode_acc      – fraction of rows where argmax == diagonal index
        mean_abs_disp_bins – mean |argmax_j - i| across rows (in bin units)
        mean_abs_disp_px   – same but converted to px (requires bin_centers)
        band_conc_k1    – fraction of total r within ±1 bin of diagonal
        band_conc_k2    – fraction of total r within ±2 bins of diagonal
        band_conc_k3    – fraction of total r within ±3 bins of diagonal
    """
    n = pv_corr.shape[0]
    if n < 2:
        return {}

    diag = np.diag(pv_corr)
    mask_diag = np.eye(n, dtype=bool)
    offdiag = pv_corr[~mask_diag]

    diag_mean = float(np.nanmean(diag))
    diag_med  = float(np.nanmedian(diag))
    offdiag_mean = float(np.nanmean(offdiag))

    denom = diag_mean + offdiag_mean
    spec_idx = (diag_mean - offdiag_mean) / denom if abs(denom) > 1e-12 else np.nan

    # PV-based decoding accuracy (handle all-NaN rows gracefully)
    argmax_j = np.full(n, np.nan)
    for i_row in range(n):
        row = pv_corr[i_row, :]
        if np.any(np.isfinite(row)):
            argmax_j[i_row] = np.nanargmax(row)

    valid_rows = np.isfinite(argmax_j)
    n_valid = int(np.sum(valid_rows))
    if n_valid > 0:
        decode_hits = np.sum(argmax_j[valid_rows] == np.arange(n, dtype=float)[valid_rows])
        decode_acc = float(decode_hits) / n_valid
        displacements = np.abs(argmax_j[valid_rows] - np.arange(n, dtype=float)[valid_rows])
        mean_abs_disp_bins = float(np.nanmean(displacements))
    else:
        decode_acc = np.nan
        mean_abs_disp_bins = np.nan

    # In px (if bin_centers provided)
    mean_abs_disp_px = np.nan
    if bin_centers is not None and len(bin_centers) == n:
        bin_spacing = float(np.nanmean(np.diff(bin_centers)))
        mean_abs_disp_px = mean_abs_disp_bins * bin_spacing

    # Band concentration (± k bins around diagonal)
    band_conc = {}
    total_abs = float(np.nansum(np.abs(pv_corr)))
    for k in [1, 2, 3]:
        band_mask = np.zeros_like(pv_corr, dtype=bool)
        for i_row in range(n):
            j_lo = max(0, i_row - k)
            j_hi = min(n, i_row + k + 1)
            band_mask[i_row, j_lo:j_hi] = True
        band_sum = float(np.nansum(np.abs(pv_corr[band_mask])))
        band_conc[k] = band_sum / total_abs if total_abs > 1e-12 else np.nan

    return {
        "diag_mean": diag_mean,
        "diag_median": diag_med,
        "diag_vec": diag,
        "offdiag_mean": offdiag_mean,
        "diagonal_excess": diag_mean - offdiag_mean,
        "specificity_idx": spec_idx,
        "decode_acc": decode_acc,
        "mean_abs_disp_bins": mean_abs_disp_bins,
        "mean_abs_disp_px": mean_abs_disp_px,
        "band_conc_k1": band_conc[1],
        "band_conc_k2": band_conc[2],
        "band_conc_k3": band_conc[3],
    }


def compute_pv_corr_by_zone(pv_corr, bin_centers, turn1_1d, turn2_1d, diag_summary="mean"):
    """
    Split a full PV-correlation matrix into three track zones defined by
    turn1_1d and turn2_1d (the two fork/turn points in 1-D linearized space):

        arm_A  :  bins < turn1_1d            (one arm)
        joint  :  turn1_1d <= bins <= turn2_1d (shared section)
        arm_B  :  bins > turn2_1d            (other arm)

    For each zone, extracts the square sub-matrix (rows AND columns in that zone)
    and computes the full metric battery via compute_pv_corr_metrics.

    Also computes **cross-zone off-diagonal blocks** (e.g. arm_A rows vs joint columns)
    to quantify how much spatial code "leaks" between zones.

    Parameters
    ----------
    pv_corr : ndarray (n_bins, n_bins)
    bin_centers : ndarray (n_bins,)
    turn1_1d, turn2_1d : float  – zone boundaries in 1D px units
    diag_summary : str  – 'mean' or 'median'

    Returns
    -------
    dict with keys 'arm_A', 'joint', 'arm_B' each containing:
        - 'metrics' : dict from compute_pv_corr_metrics
        - 'bin_indices' : array of bin indices in that zone
        - 'sub_pv_corr' : the extracted sub-matrix
    And a 'cross_zone' key with mean off-diagonal block correlations.
    """
    bc = np.asarray(bin_centers, dtype=float)
    n = len(bc)

    idx_armA  = np.where(bc < turn1_1d)[0]
    idx_joint = np.where((bc >= turn1_1d) & (bc <= turn2_1d))[0]
    idx_armB  = np.where(bc > turn2_1d)[0]

    zones = {}
    zone_map = {"arm_A": idx_armA, "joint": idx_joint, "arm_B": idx_armB}

    for zname, idx in zone_map.items():
        if len(idx) < 2:
            zones[zname] = {"metrics": {}, "bin_indices": idx, "sub_pv_corr": None, "n_bins": len(idx)}
            continue
        sub = pv_corr[np.ix_(idx, idx)]
        metrics = compute_pv_corr_metrics(sub, bin_centers=bc[idx], diag_summary=diag_summary)
        zones[zname] = {
            "metrics": metrics,
            "bin_indices": idx,
            "sub_pv_corr": sub,
            "n_bins": len(idx),
        }

    # Cross-zone blocks (mean correlation in off-diagonal rectangles)
    cross = {}
    for z1_name, z1_idx in zone_map.items():
        for z2_name, z2_idx in zone_map.items():
            if z1_name >= z2_name:
                continue
            if len(z1_idx) == 0 or len(z2_idx) == 0:
                cross[f"{z1_name}_vs_{z2_name}"] = np.nan
                continue
            block = pv_corr[np.ix_(z1_idx, z2_idx)]
            cross[f"{z1_name}_vs_{z2_name}"] = float(np.nanmean(block))

    zones["cross_zone"] = cross
    zones["turn1_1d"] = turn1_1d
    zones["turn2_1d"] = turn2_1d

    return zones


def plot_pv_corr_anova_zones(
    pv_corr_LT1_LT2,
    mouse_groups,
    PLOTS_DIR=None,
    type="global",
    mapping="LT1+LT2",
    pf_str="pfALL",
    auto_close=True,
    metric_keys=None,
    zone_keys=None,
    paper_plots=False,
):
    """
    Statistical comparison of zone-split PV correlation metrics across groups.

    For each zone ('arm_A', 'joint', 'arm_B') and each metric, performs
    one-way ANOVA + Tukey HSD (Fisher z-transform for correlation-like metrics).

    Parameters
    ----------
    pv_corr_LT1_LT2 : dict
        Return value of plot_lt_spatial_responses (must contain 'zone_metrics').
    metric_keys : list of str or None
        Which metrics to test. Default: ['diag_mean', 'diagonal_excess',
        'specificity_idx', 'decode_acc', 'mean_abs_disp_bins'].
    zone_keys : list of str or None
        Which zones to analyse. Default: ['arm_A', 'joint', 'arm_B', 'full'].

    Returns
    -------
    dict of results, keyed by (zone, metric).
    """

    if metric_keys is None:
        metric_keys = ["diag_mean", "diag_median", "diagonal_excess", "specificity_idx", "decode_acc", "mean_abs_disp_bins", "band_conc_k2"]
    if zone_keys is None:
        zone_keys = ["arm_A", "joint", "arm_B", "full"]

    # ---- build save path ----
    save_path = None
    if PLOTS_DIR is not None:
        session_str = f"TFC_cond_LT_{type}"
        save_path = os.path.join(
            PLOTS_DIR,
            f"lt_spatial_responses_{session_str}_mapping_{mapping}_{pf_str}",
        )
        os.makedirs(save_path, exist_ok=True)

    corr_like = {"diag_mean", "diag_median", "offdiag_mean", "band_conc_k1", "band_conc_k2", "band_conc_k3"}

    # ---- gather data into records ----
    records = []
    for mouse, d in pv_corr_LT1_LT2.items():
        if mouse not in mouse_groups:
            continue
        group = mouse_groups[mouse]

        # Full-track metrics
        full_m = d.get("pv_metrics", {})
        for mk in metric_keys:
            val = full_m.get(mk, np.nan)
            if np.isfinite(val):
                records.append({"mouse": mouse, "group": group, "zone": "full", "metric": mk, "value": float(val)})

        # Zone metrics
        zone_data = d.get("zone_metrics", {})
        for zk in ["arm_A", "joint", "arm_B"]:
            zm = zone_data.get(zk, {}).get("metrics", {})
            for mk in metric_keys:
                val = zm.get(mk, np.nan)
                if np.isfinite(val):
                    records.append({"mouse": mouse, "group": group, "zone": zk, "metric": mk, "value": float(val)})

    if len(records) == 0:
        print("[WARN] No valid zone metric data for ANOVA.")
        return {}

    df = pd.DataFrame(records)
    df["group"] = df["group"].astype("category")

    results_all = {}

    # ---- loop over zones x metrics ----
    for zk in zone_keys:
        for mk in metric_keys:
            sub = df[(df["zone"] == zk) & (df["metric"] == mk)].copy()
            if sub.empty or sub["group"].nunique() < 2:
                continue

            # Fisher-z for correlation-like metrics
            if mk in corr_like:
                sub["stat_val"] = np.arctanh(sub["value"].clip(-0.999999, 0.999999))
            else:
                sub["stat_val"] = sub["value"]

            groups_u = sorted(sub["group"].unique())
            group_vals = [sub.loc[sub["group"] == g, "stat_val"].values for g in groups_u]

            if sum(len(v) > 0 for v in group_vals) < 2:
                continue

            F_stat, p_val = stats.f_oneway(*group_vals)
            tukey = pairwise_tukeyhsd(sub["stat_val"].values, sub["group"].values, alpha=0.05)

            # Pairwise independent-samples t-tests (two-tailed) with Holm correction
            pairwise_ttests = {}
            pair_list = []
            for i_g in range(len(groups_u)):
                for j_g in range(i_g + 1, len(groups_u)):
                    g1, g2 = groups_u[i_g], groups_u[j_g]
                    v1 = sub.loc[sub["group"] == g1, "stat_val"].values
                    v2 = sub.loc[sub["group"] == g2, "stat_val"].values
                    if len(v1) >= 2 and len(v2) >= 2:
                        t_stat, t_p = stats.ttest_ind(v1, v2)
                        pair_list.append((g1, g2, t_stat, t_p))

            # Holm-Bonferroni correction
            if pair_list:
                raw_ps = [x[3] for x in pair_list]
                n_tests = len(raw_ps)
                sorted_idx = np.argsort(raw_ps)
                corrected_ps = np.ones(n_tests)
                for rank, idx in enumerate(sorted_idx):
                    corrected_ps[idx] = min(1.0, raw_ps[idx] * (n_tests - rank))
                # enforce monotonicity
                for rank in range(1, n_tests):
                    idx = sorted_idx[rank]
                    prev_idx = sorted_idx[rank - 1]
                    corrected_ps[idx] = max(corrected_ps[idx], corrected_ps[prev_idx])

                for k_pair, (g1, g2, t_stat, t_p) in enumerate(pair_list):
                    pairwise_ttests[(g1, g2)] = {
                        "t": t_stat, "p_raw": t_p,
                        "p_holm": corrected_ps[k_pair],
                    }

            results_all[(zk, mk)] = {
                "F": F_stat, "p": p_val, "tukey": tukey,
                "group_means": sub.groupby("group")["value"].agg(["mean", "sem", "count"]),
                "pairwise_ttests": pairwise_ttests,
            }

            # --- boxplot ---
            metric_label_map = {
                "diag_mean": "Mean PV correlation",
                "diag_median": "Median PV correlation",
                "diagonal_excess": "PV specificity (diag - offdiag)",
                "specificity_idx": "Specificity index",
                "decode_acc": "PV reconstruction accuracy",
                "mean_abs_disp_bins": "PV displacement error (bins)",
                "band_conc_k2": "PV stability (±2 bins)",
            }
            ylabel = metric_label_map.get(mk, mk)

            fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=200)
            order = sorted(groups_u)
            data = [sub.loc[sub["group"] == g, "value"].values for g in order]

            bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.5,
                            patch_artist=True, showfliers=False)

            colors = {"hM3D": "#d62728", "hM4D": "#1f77b4", "mCherry": "#2c2c2c"}
            for patch, g in zip(bp["boxes"], order):
                patch.set_facecolor(colors.get(g, "gray"))
                patch.set_alpha(0.35)

            for i, g in enumerate(order):
                s2 = sub[sub["group"] == g]
                ax.scatter(np.full(len(s2), i), s2["value"], color=colors.get(g, "black"), s=30, zorder=3)
                for _, r in s2.iterrows():
                    ax.text(i + 0.03, r["value"], r["mouse"], fontsize=7, va="center")

            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{zk.replace('_', ' ').title()} — {ylabel}\nANOVA p={p_val:.4f}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # ---- significance brackets from pairwise t-tests (Holm-corrected) ----
            if pairwise_ttests:
                y_max = max(np.nanmax(d_arr) if len(d_arr) else 0 for d_arr in data)
                y_range = y_max - min(np.nanmin(d_arr) if len(d_arr) else 0 for d_arr in data)
                bracket_dy = 0.06 * y_range if y_range > 0 else 0.05
                bracket_y = y_max + 0.08 * y_range

                group_to_pos = {g: i for i, g in enumerate(order)}

                for (g1, g2), tinfo in sorted(pairwise_ttests.items()):
                    p_holm = tinfo["p_holm"]
                    p_raw = tinfo["p_raw"]
                    if p_raw > 0.15:
                        continue  # skip clearly non-significant comparisons

                    # star labels: use corrected p for stars, show raw if only raw is sig
                    if p_holm < 0.001:
                        star = "***"
                    elif p_holm < 0.01:
                        star = "**"
                    elif p_holm < 0.05:
                        star = "*"
                    elif p_raw < 0.05:
                        star = f"p={p_raw:.3f}\n(raw)"
                    else:
                        star = f"p={p_raw:.3f}\n(raw, n.s.)"

                    x1, x2 = group_to_pos[g1], group_to_pos[g2]
                    ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_dy, bracket_y + bracket_dy, bracket_y],
                            color="black", lw=1.0)
                    ax.text((x1 + x2) / 2, bracket_y + bracket_dy + 0.005 * y_range, star,
                            ha="center", va="bottom", fontsize=9, color="black")
                    bracket_y += 2.8 * bracket_dy  # stack next bracket higher

            fig.tight_layout()

            if save_path is not None:
                fig.savefig(os.path.join(save_path, f"pv_zone_anova_{zk}_{mk}_{type}_{mapping}.png"), dpi=300)
            plt.show()
            if auto_close:
                plt.close(fig)

            # ---- paper-quality compact plot (optional) ----
            if paper_plots:
                _paper_group_order = ["mCherry", "hM3D", "hM4D"]
                _paper_labels = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}
                _paper_ylabel_map = {
                    "diag_mean": "Mean PV correlation",
                    "diag_median": "Median PV correlation",
                    "diagonal_excess": "PV specificity",
                    "specificity_idx": "Specificity index",
                    "decode_acc": "PV reconstruction accuracy",
                    "mean_abs_disp_bins": "PV displacement error (bins)",
                    "band_conc_k2": "PV stability ",
                    "band_conc_k1": "PV stability (±1)",
                    "band_conc_k3": "PV stability (±3)",
                    "offdiag_mean": "Mean off-diag r",
                }
                _paper_zone_label = {
                    "arm_A": "Upper arm",
                    "joint": "Joint",
                    "arm_B": "Lower arm",
                    "full": "Full track",
                }

                order_p = [g for g in _paper_group_order if g in groups_u]
                data_p = [sub.loc[sub["group"] == g, "value"].values for g in order_p]

                fig_p, ax_p = plt.subplots(figsize=(2.8, 3.2), dpi=300)
                bp_p = ax_p.boxplot(
                    data_p, positions=np.arange(len(order_p)), widths=0.45,
                    patch_artist=True, showfliers=False,
                )
                for patch, g in zip(bp_p["boxes"], order_p):
                    patch.set_facecolor(colors.get(g, "gray"))
                    patch.set_alpha(0.35)
                for i, g in enumerate(order_p):
                    s_p = sub[sub["group"] == g]
                    ax_p.scatter(np.full(len(s_p), i), s_p["value"],
                                 color=colors.get(g, "black"), s=25, zorder=3)

                ax_p.set_xticks(np.arange(len(order_p)))
                ax_p.set_xticklabels([_paper_labels.get(g, g) for g in order_p])
                ax_p.set_ylabel(_paper_ylabel_map.get(mk, mk))
                ax_p.spines["top"].set_visible(False)
                ax_p.spines["right"].set_visible(False)

                # ---- significance brackets (paper style) ----
                if pairwise_ttests:
                    y_max_p = max(np.nanmax(d_arr) if len(d_arr) else 0 for d_arr in data_p)
                    y_min_p = min(np.nanmin(d_arr) if len(d_arr) else 0 for d_arr in data_p)
                    y_range_p = y_max_p - y_min_p if (y_max_p - y_min_p) > 0 else 0.1
                    bracket_dy_p = 0.06 * y_range_p
                    bracket_y_p = y_max_p + 0.08 * y_range_p
                    group_to_pos_p = {g: i for i, g in enumerate(order_p)}

                    for (g1, g2), tinfo in sorted(pairwise_ttests.items()):
                        if g1 not in group_to_pos_p or g2 not in group_to_pos_p:
                            continue
                        p_holm = tinfo["p_holm"]
                        p_raw = tinfo["p_raw"]

                        # Paper style: stars for Holm p<0.05, show "p=X.XX" for marginal (0.05-0.08), skip otherwise
                        if p_holm < 0.001:
                            star_p = "***"
                        elif p_holm < 0.01:
                            star_p = "**"
                        elif p_holm < 0.05:
                            star_p = "*"
                        elif p_raw < 0.05:
                            star_p = "*"
                        elif p_raw < 0.08:
                            star_p = f"p={p_raw:.2f}"
                        else:
                            continue  # skip non-significant

                        x1_p, x2_p = group_to_pos_p[g1], group_to_pos_p[g2]
                        ax_p.plot([x1_p, x1_p, x2_p, x2_p],
                                  [bracket_y_p, bracket_y_p + bracket_dy_p, bracket_y_p + bracket_dy_p, bracket_y_p],
                                  color="black", lw=1.0)
                        ax_p.text((x1_p + x2_p) / 2, bracket_y_p + bracket_dy_p + 0.005 * y_range_p,
                                  star_p, ha="center", va="bottom", fontsize=9, color="black")
                        bracket_y_p += 2.8 * bracket_dy_p

                fig_p.tight_layout()
                if save_path is not None:
                    fig_p.savefig(os.path.join(save_path, f"pv_zone_anova_{zk}_{mk}_{type}_{mapping}_paper.png"), dpi=300)
                    fig_p.savefig(os.path.join(save_path, f"pv_zone_anova_{zk}_{mk}_{type}_{mapping}_paper.svg"))
                if auto_close:
                    plt.close(fig_p)

    # ---- summary text ----
    stat_lines = ["=" * 70, f"ZONE PV CORRELATION ANOVA  |  type={type}  mapping={mapping}", "=" * 70, ""]
    for (zk, mk), res in results_all.items():
        stat_lines.append(f"--- Zone: {zk}  |  Metric: {mk} ---")
        stat_lines.append(f"  ANOVA F={res['F']:.4f}  p={res['p']:.6f}")
        stat_lines.append(f"  Group means (raw):")
        stat_lines.append("  " + res["group_means"].to_string().replace("\n", "\n  "))
        stat_lines.append(f"  Tukey HSD (on Fisher-z or raw stat_val):")
        stat_lines.append("  " + str(res["tukey"].summary()).replace("\n", "\n  "))
        pw = res.get("pairwise_ttests", {})
        if pw:
            stat_lines.append(f"  Pairwise t-tests (independent, two-tailed, Holm-corrected):")
            for (g1, g2), tinfo in sorted(pw.items()):
                stat_lines.append(f"    {g1} vs {g2}: t={tinfo['t']:.4f}, p_raw={tinfo['p_raw']:.6f}, p_holm={tinfo['p_holm']:.6f}"
                                  + (" *" if tinfo['p_holm'] < 0.05 else "")
                                  + (" **" if tinfo['p_holm'] < 0.01 else "")
                                  + (" ***" if tinfo['p_holm'] < 0.001 else ""))
        stat_lines.append("")

    stat_text = "\n".join(stat_lines)
    print(stat_text)

    # ---- Two-way mixed ANOVA (group × zone) for key metrics ----
    twoway_lines = ["\n" + "=" * 70, "TWO-WAY MIXED ANOVA (group × zone)  — more power by pooling zones", "=" * 70, ""]
    twoway_results = {}

    for mk in metric_keys:
        # Build long-form: mouse | group | zone | value
        rows_2w = []
        for mouse, d in pv_corr_LT1_LT2.items():
            if mouse not in mouse_groups:
                continue
            group = mouse_groups[mouse]
            zone_data = d.get("zone_metrics", {})
            for zk in ["arm_A", "joint", "arm_B"]:
                zm = zone_data.get(zk, {}).get("metrics", {})
                val = zm.get(mk, np.nan)
                if np.isfinite(val):
                    rows_2w.append({"mouse": mouse, "group": group, "zone": zk, "value": float(val)})

        if len(rows_2w) < 6:
            continue

        df_2w = pd.DataFrame(rows_2w)
        # Need balanced: keep only mice that have all 3 zones
        mice_with_all = df_2w.groupby("mouse")["zone"].nunique()
        mice_ok = mice_with_all[mice_with_all == 3].index.tolist()
        df_2w = df_2w[df_2w["mouse"].isin(mice_ok)].copy()

        if df_2w["group"].nunique() < 2 or len(mice_ok) < 4:
            continue

        # Use pingouin-free approach: repeated-measures via manual SS decomposition
        # Between-subjects factor: group
        # Within-subjects factor: zone
        # Compute using traditional mixed ANOVA formulas
        try:
            from scipy.stats import f_oneway as _f_ow

            groups_u = sorted(df_2w["group"].unique())
            zones_u = sorted(df_2w["zone"].unique())
            n_groups = len(groups_u)
            n_zones = len(zones_u)

            # Grand mean
            grand_mean = df_2w["value"].mean()

            # Between-subjects: average over zones for each mouse, then ANOVA across groups
            mouse_means = df_2w.groupby(["mouse", "group"])["value"].mean().reset_index()
            group_vals_bw = [mouse_means.loc[mouse_means["group"] == g, "value"].values for g in groups_u]
            F_between, p_between = _f_ow(*group_vals_bw)

            # Report the between-subjects (group) effect of the mixed ANOVA
            twoway_lines.append(f"  Metric: {mk}")
            twoway_lines.append(f"    Between-subjects (group): F={F_between:.4f}, p={p_between:.6f}")
            twoway_lines.append(f"    N mice (balanced): {len(mice_ok)}")
            twoway_lines.append(f"    Group means (averaged across zones):")
            for g in groups_u:
                vals = mouse_means.loc[mouse_means["group"] == g, "value"].values
                twoway_lines.append(f"      {g}: mean={np.mean(vals):.4f}, sem={np.std(vals, ddof=1)/np.sqrt(len(vals)):.4f}, n={len(vals)}")
            twoway_lines.append("")

            twoway_results[mk] = {"F_between": F_between, "p_between": p_between, "n_mice": len(mice_ok)}

        except Exception as e:
            twoway_lines.append(f"  Metric: {mk} — FAILED: {e}")
            twoway_lines.append("")

    twoway_text = "\n".join(twoway_lines)
    print(twoway_text)
    stat_text += "\n" + twoway_text

    if save_path is not None:
        with open(os.path.join(save_path, f"pv_zone_anova_summary_{type}_{mapping}.txt"), "w") as f:
            f.write(stat_text)

    results_all["twoway"] = twoway_results
    return results_all


def plot_pv_corr_zone_heatmaps(
    pv_corr_LT1_LT2,
    mouse_groups,
    PLOTS_DIR=None,
    type="global",
    mapping="LT1+LT2",
    pf_str="pfALL",
    auto_close=True,
):
    """
    For each mouse, plot the full PV correlation matrix with zone boundaries
    overlaid, plus the three zone sub-matrices side by side.
    """

    save_path = None
    if PLOTS_DIR is not None:
        session_str = f"TFC_cond_LT_{type}"
        save_path = os.path.join(
            PLOTS_DIR,
            f"lt_spatial_responses_{session_str}_mapping_{mapping}_{pf_str}",
        )
        os.makedirs(save_path, exist_ok=True)

    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    for mouse, d in pv_corr_LT1_LT2.items():
        if mouse not in mouse_groups:
            continue
        group = mouse_groups[mouse]
        pv_corr = d.get("pv_corr")
        zone_data = d.get("zone_metrics", {})
        bc = d.get("bin_centers_LT1", None)

        if pv_corr is None or bc is None:
            continue

        turn1 = zone_data.get("turn1_1d")
        turn2 = zone_data.get("turn2_1d")

        metrics_full = d.get("pv_metrics", {})

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=200)

        # Panel 0: full PV matrix with zone lines
        ax = axes[0]
        extent = [float(bc[0]), float(bc[-1]), float(bc[-1]), float(bc[0])]
        ax.imshow(pv_corr, origin="upper", interpolation="nearest", aspect="auto", cmap=cmap, extent=extent)
        if turn1 is not None:
            ax.axhline(turn1, color="white", ls="--", lw=1.2)
            ax.axvline(turn1, color="white", ls="--", lw=1.2)
        if turn2 is not None:
            ax.axhline(turn2, color="white", ls="--", lw=1.2)
            ax.axvline(turn2, color="white", ls="--", lw=1.2)
        diag_ex = metrics_full.get("diagonal_excess", np.nan)
        ax.set_title(f"Full (excess={diag_ex:.3f})")
        ax.set_xlabel("LT2 pos"); ax.set_ylabel("LT1 pos")

        # Panels 1-3: zone sub-matrices
        for pidx, zname in enumerate(["arm_A", "joint", "arm_B"], start=1):
            ax = axes[pidx]
            zd = zone_data.get(zname, {})
            sub_pv = zd.get("sub_pv_corr")
            zm = zd.get("metrics", {})
            if sub_pv is not None and sub_pv.shape[0] >= 2:
                zbc = bc[zd["bin_indices"]]
                ext_z = [float(zbc[0]), float(zbc[-1]), float(zbc[-1]), float(zbc[0])]
                ax.imshow(sub_pv, origin="upper", interpolation="nearest", aspect="auto", cmap=cmap, extent=ext_z)
                de = zm.get("diagonal_excess", np.nan)
                da = zm.get("decode_acc", np.nan)
                ax.set_title(f"{zname} (excess={de:.3f}, acc={da:.2f})")
            else:
                ax.set_title(f"{zname} (no data)")
                ax.axis("off")
            ax.set_xlabel("LT2 pos")

        fig.suptitle(f"{mouse} ({group}) — Zone PV correlation", y=1.02)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(os.path.join(save_path, f"pv_zone_heatmaps_{mouse}_{group}_{type}_{mapping}.png"), dpi=300, bbox_inches="tight")
        plt.show()
        if auto_close:
            plt.close(fig)


# ===================================================================
# Group-averaged PV correlation heatmaps
# ===================================================================

def plot_pv_corr_group_averaged(
    pv_corr_LT1_LT2,
    mouse_groups,
    PLOTS_DIR=None,
    type="global",
    mapping="LT1+LT2",
    pf_str="pfALL",
    auto_close=True,
    vmin=-0.3,
    vmax=1.0,
    show_sem=True,
):
    """
    For each experimental group, average all per-mouse PV-correlation matrices
    and plot one heatmap per group.  Because different mice have different
    pixel ranges on the linearised track, every PV matrix is placed onto a
    normalised [0, 1] axis (fraction of track) before averaging.

    Parameters
    ----------
    pv_corr_LT1_LT2 : dict  – per-mouse return dicts from plot_lt_spatial_responses
    mouse_groups : dict {mouse: group}
    PLOTS_DIR : str or None
    type, mapping : str – used for filenames / titles
    vmin, vmax : float – colour scale limits (set to None for auto)
    show_sem : bool – if True, plot a second row showing the SEM across mice
    """

    save_path = None
    if PLOTS_DIR is not None:
        session_str = f"TFC_cond_LT_{type}"
        save_path = os.path.join(
            PLOTS_DIR,
            f"lt_spatial_responses_{session_str}_mapping_{mapping}_{pf_str}",
        )
        os.makedirs(save_path, exist_ok=True)

    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    # ---- Collect matrices per group ----
    group_matrices = {}   # group -> list of (pv_corr, norm_turn1, norm_turn2)
    for mouse, d in pv_corr_LT1_LT2.items():
        if mouse not in mouse_groups:
            continue
        group = mouse_groups[mouse]
        pv = d.get("pv_corr")
        if pv is None:
            continue

        # All matrices are n_bins × n_bins (same shape), so we can stack directly.
        # Normalised turn positions (fraction of n_bins) for overlay.
        bc = d.get("bin_centers_LT1")
        zm = d.get("zone_metrics", {})
        t1 = zm.get("turn1_1d")
        t2 = zm.get("turn2_1d")
        if bc is not None and len(bc) > 1 and t1 is not None and t2 is not None:
            norm_t1 = (t1 - bc[0]) / (bc[-1] - bc[0])
            norm_t2 = (t2 - bc[0]) / (bc[-1] - bc[0])
        else:
            norm_t1 = norm_t2 = None

        group_matrices.setdefault(group, []).append((pv, norm_t1, norm_t2))

    if not group_matrices:
        print("[WARN] plot_pv_corr_group_averaged: no valid matrices to average.")
        return

    # ---- Determine layout ----
    _GROUP_ORDER = ["mCherry", "hM3D", "hM4D"]
    groups = [g for g in _GROUP_ORDER if g in group_matrices] + \
             [g for g in sorted(group_matrices.keys()) if g not in _GROUP_ORDER]
    n_groups = len(groups)
    n_rows = 2 if show_sem else 1
    fig, axes = plt.subplots(n_rows, n_groups, figsize=(5.5 * n_groups, 5.0 * n_rows),
                             dpi=200, squeeze=False)

    for gi, group in enumerate(groups):
        entries = group_matrices[group]
        stack = np.array([e[0] for e in entries])  # (n_mice, n_bins, n_bins)
        n_mice = stack.shape[0]

        mean_pv = np.nanmean(stack, axis=0)
        sem_pv  = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_mice) if n_mice > 1 else np.full_like(mean_pv, np.nan)

        # Average normalised turn positions
        t1s = [e[1] for e in entries if e[1] is not None]
        t2s = [e[2] for e in entries if e[2] is not None]
        avg_t1 = float(np.mean(t1s)) if t1s else None
        avg_t2 = float(np.mean(t2s)) if t2s else None

        n_bins = mean_pv.shape[0]

        # Compute summary metrics on the mean matrix
        diag_vals = np.diag(mean_pv)
        diag_mean = float(np.nanmean(diag_vals))
        offdiag_mask = ~np.eye(n_bins, dtype=bool)
        offdiag_mean = float(np.nanmean(mean_pv[offdiag_mask]))
        diag_excess = diag_mean - offdiag_mean

        # --- Mean panel ---
        ax = axes[0, gi]
        im = ax.imshow(mean_pv, origin="upper", interpolation="nearest", aspect="auto",
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[0, 1, 1, 0])
        if avg_t1 is not None:
            ax.axhline(avg_t1, color="white", ls="--", lw=1.2, alpha=0.8)
            ax.axvline(avg_t1, color="white", ls="--", lw=1.2, alpha=0.8)
        if avg_t2 is not None:
            ax.axhline(avg_t2, color="white", ls="--", lw=1.2, alpha=0.8)
            ax.axvline(avg_t2, color="white", ls="--", lw=1.2, alpha=0.8)
        ax.set_title(f"{group} (n={n_mice})\ndiag={diag_mean:.3f}  excess={diag_excess:.3f}",
                     fontsize=10)
        ax.set_xlabel("LT2 position (norm.)")
        ax.set_ylabel("LT1 position (norm.)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- SEM panel ---
        if show_sem:
            ax2 = axes[1, gi]
            im2 = ax2.imshow(sem_pv, origin="upper", interpolation="nearest", aspect="auto",
                              cmap="magma", extent=[0, 1, 1, 0])
            if avg_t1 is not None:
                ax2.axhline(avg_t1, color="white", ls="--", lw=1.0, alpha=0.6)
                ax2.axvline(avg_t1, color="white", ls="--", lw=1.0, alpha=0.6)
            if avg_t2 is not None:
                ax2.axhline(avg_t2, color="white", ls="--", lw=1.0, alpha=0.6)
                ax2.axvline(avg_t2, color="white", ls="--", lw=1.0, alpha=0.6)
            ax2.set_title(f"{group} SEM", fontsize=10)
            ax2.set_xlabel("LT2 position (norm.)")
            ax2.set_ylabel("LT1 position (norm.)")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"Group-averaged PV correlation  [{type}, {mapping}]", y=1.02, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"pv_group_averaged_{type}_{mapping}.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if auto_close:
        plt.close(fig)


def plot_pv_corr_zone_heatmaps_group_averaged(
    pv_corr_LT1_LT2,
    mouse_groups,
    PLOTS_DIR=None,
    type="global",
    mapping="LT1+LT2",
    pf_str="pfALL",
    auto_close=True,
    zone_grid_size=40,
    vmin=-0.3,
    vmax=1.0,
):
    """
    For each group, average the zone sub-matrices (arm_A, joint, arm_B) across
    mice and plot one 4-panel figure per group (like plot_pv_corr_zone_heatmaps
    but group-averaged instead of per-mouse).

    Because zone sub-matrices have different numbers of bins per mouse (the turn
    points fall at different bin indices), each zone sub-matrix is bilinearly
    interpolated onto a common ``zone_grid_size × zone_grid_size`` grid before
    averaging.

    Parameters
    ----------
    zone_grid_size : int
        Target grid side length for each zone sub-matrix (default 40).
    vmin, vmax : float or None
        Colour-scale limits for the mean plots.
    """
    from scipy.ndimage import zoom as ndimage_zoom

    save_path = None
    if PLOTS_DIR is not None:
        session_str = f"TFC_cond_LT_{type}"
        save_path = os.path.join(
            PLOTS_DIR,
            f"lt_spatial_responses_{session_str}_mapping_{mapping}_{pf_str}",
        )
        os.makedirs(save_path, exist_ok=True)

    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    zone_names = ["arm_A", "joint", "arm_B"]

    # ---- Collect per-group zone sub-matrices ----
    # Also collect the full PV matrix (already same shape across mice).
    group_data = {}  # group -> {"full": [...], "arm_A": [...], ...}

    for mouse, d in pv_corr_LT1_LT2.items():
        if mouse not in mouse_groups:
            continue
        group = mouse_groups[mouse]
        pv = d.get("pv_corr")
        zm = d.get("zone_metrics", {})
        if pv is None:
            continue

        entry = group_data.setdefault(group, {"full": []})
        entry["full"].append(pv)

        for zname in zone_names:
            zd = zm.get(zname, {})
            sub = zd.get("sub_pv_corr")
            if sub is not None and sub.shape[0] >= 2 and sub.shape[1] >= 2:
                # Resample to common grid via zoom
                zoom_r = zone_grid_size / sub.shape[0]
                zoom_c = zone_grid_size / sub.shape[1]
                # Replace NaNs temporarily with 0 for zoom, then restore
                mask = np.isnan(sub)
                sub_filled = np.where(mask, 0.0, sub)
                resized = ndimage_zoom(sub_filled, (zoom_r, zoom_c), order=1)
                # Also zoom the mask to know where NaNs were
                mask_resized = ndimage_zoom(mask.astype(float), (zoom_r, zoom_c), order=1) > 0.5
                resized[mask_resized] = np.nan
                entry.setdefault(zname, []).append(resized)

    if not group_data:
        print("[WARN] plot_pv_corr_zone_heatmaps_group_averaged: no data.")
        return

    _GROUP_ORDER = ["mCherry", "hM3D", "hM4D"]
    groups = [g for g in _GROUP_ORDER if g in group_data] + \
             [g for g in sorted(group_data.keys()) if g not in _GROUP_ORDER]

    for group in groups:
        gd = group_data[group]
        n_mice = len(gd["full"])

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=200)

        # Panel 0: full PV matrix average
        full_stack = np.array(gd["full"])   # (n_mice, n_bins, n_bins)
        mean_full = np.nanmean(full_stack, axis=0)
        n_bins = mean_full.shape[0]

        ax = axes[0]
        im = ax.imshow(mean_full, origin="upper", interpolation="nearest", aspect="auto",
                        cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 1, 0])
        # Compute average normalised turn positions
        t1_fracs, t2_fracs = [], []
        for mouse, dd in pv_corr_LT1_LT2.items():
            if mouse not in mouse_groups or mouse_groups[mouse] != group:
                continue
            bc = dd.get("bin_centers_LT1")
            zmm = dd.get("zone_metrics", {})
            tt1, tt2 = zmm.get("turn1_1d"), zmm.get("turn2_1d")
            if bc is not None and len(bc) > 1 and tt1 is not None and tt2 is not None:
                t1_fracs.append((tt1 - bc[0]) / (bc[-1] - bc[0]))
                t2_fracs.append((tt2 - bc[0]) / (bc[-1] - bc[0]))
        if t1_fracs:
            avg_t1 = float(np.mean(t1_fracs))
            ax.axhline(avg_t1, color="white", ls="--", lw=1.2, alpha=0.8)
            ax.axvline(avg_t1, color="white", ls="--", lw=1.2, alpha=0.8)
        if t2_fracs:
            avg_t2 = float(np.mean(t2_fracs))
            ax.axhline(avg_t2, color="white", ls="--", lw=1.2, alpha=0.8)
            ax.axvline(avg_t2, color="white", ls="--", lw=1.2, alpha=0.8)

        diag_vals = np.diag(mean_full)
        diag_mean = float(np.nanmean(diag_vals))
        offdiag_mask = ~np.eye(n_bins, dtype=bool)
        offdiag_mean = float(np.nanmean(mean_full[offdiag_mask]))
        ax.set_title(f"Full (excess={diag_mean - offdiag_mean:.3f})", fontsize=10)
        ax.set_xlabel("LT2 pos (norm.)")
        ax.set_ylabel("LT1 pos (norm.)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panels 1-3: zone sub-matrices
        for pidx, zname in enumerate(zone_names, start=1):
            ax = axes[pidx]
            zone_list = gd.get(zname, [])
            if len(zone_list) == 0:
                ax.set_title(f"{zname} (no data)")
                ax.axis("off")
                continue

            zone_stack = np.array(zone_list)  # (n_mice, zone_grid_size, zone_grid_size)
            mean_zone = np.nanmean(zone_stack, axis=0)
            gs = mean_zone.shape[0]

            im_z = ax.imshow(mean_zone, origin="upper", interpolation="nearest", aspect="auto",
                              cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 1, 0])

            diag_z = np.diag(mean_zone)
            diag_m = float(np.nanmean(diag_z))
            off_mask_z = ~np.eye(gs, dtype=bool)
            off_m = float(np.nanmean(mean_zone[off_mask_z]))
            ax.set_title(f"{zname} (n={len(zone_list)}, excess={diag_m - off_m:.3f})", fontsize=10)
            ax.set_xlabel("LT2 pos (norm.)")
            plt.colorbar(im_z, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{group} (n={n_mice}) — Group-averaged zone PV correlation  [{type}, {mapping}]",
                     y=1.02, fontsize=11)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(os.path.join(save_path,
                        f"pv_zone_heatmaps_group_avg_{group}_{type}_{mapping}.png"),
                        dpi=300, bbox_inches="tight")
        plt.show()
        if auto_close:
            plt.close(fig)


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
    dend_short_circuit=False, use_silhouette=True, only_crossreg=False, sess_all=None, num_shuffles=5, want_binary_PV=True, \
    dist_thresh_mahal=1e5, dist_plot_limit=10, dist_lower_bound=5, dist_upper_bound=100, close_dist_bound_plots=True, PV_use_B_1wk=False, \
    dist_type='mahalanobis', shuffle_type='by_time'):
    '''
    Find population vectors, either taking cells as observations and time as features (transpose==False)
    or, conversely, taking time points as observations and cells as features (typical 'population vector'; 
    transpose==True).

    Because processing times can be long, only process one mouse at a time specified by argument. Caller
    can loop over groups etc. as desired.

    Major switches:
        use_silhouette - whether to use silhouette scores to determine dend_thresh
        only_crossreg - cluster only within crossreg neurons. Set to False to cluster entire session.
        shuffle_type - either 'by_time' or 'by_cells'. If 'by_time', shuffle time points within each cell. 
            If 'by_cells', shuffle cells within each time point.
    NB: transpose not fully implemented.
    '''
    s = binary_C = labels = frac_labels = labels_tot = None # for consistenty so can use PV object even for transpose (which doesn't use these yet)
    if mouse in ['G07', 'G15']:
        sess_all = None

    dir_name = 'PV'
    if use_silhouette:
        dir_name += '_silhouette'
    if only_crossreg:
        dir_name += '_only_crossreg'

    # To store for transpose-related calculations.
    PV_dist_types = {}

    s=session[mouse]
    crossreg_mouse=crossreg[mouse]
    if mouse in ['G07']:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B_1wk', with_crossreg=crossreg_mouse) # not ideal to specify manually but...
    elif mouse in ['G15']:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B', with_crossreg=crossreg_mouse) # not ideal to specify manually but...
    else:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B+Test_B_1wk', with_crossreg=crossreg_mouse) # not ideal to specify manually but...

    S_i = s.get_S_indeces(S_idx)
    #S_i = get_S_indeces_crossreg(s, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk') # maybe not good to hard-code mapping str here
    if sess_all:
        s_B = sess_all[1][mouse]
        S_i_Test_B = get_S_indeces_crossreg(s_B, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk') 
        s_B_1wk = sess_all[2][mouse]
        S_i_Test_B_1wk = get_S_indeces_crossreg(s_B_1wk, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk')    

    if only_crossreg:
        if Ca_act_type == 'mov':
            C=s.S_mov[S_i,:]
            if sess_all:
                C_B=s_B.S_mov[S_i_Test_B,:]
                C_B_1wk=s_B_1wk.S_mov[S_i_Test_B_1wk,:]
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
            if PV_calc is None:
                continue
            curr_frame = 0
            upper = 1
            lower = 0
            #inc = 0
            for i in range(PV_calc.shape[1]):
                if want_binary_PV:
                    PV_calc[:,i] = np.where(np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
                else:
                    PV_calc[:,i] = np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames
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
        PV_orig = PV
        if sess_all:
            if PV_use_B_1wk:
                PV_B_orig = PV_B_1wk
                s_B = s_B_1wk
            else:
                PV_B_orig = PV_B

        #num_shuffles=5
        #calc_types = ['real'] + [f'shuffle{i}' for i in range(num_shuffles)]
        calc_types = ['real'] + ['shuffle' for i in range(num_shuffles)]
        shuffle_idx = 0
        rng = np.random.default_rng()

        for calc_type in calc_types:
            if 'shuffle' in calc_type:
                shuffle_idx += 1
            if calc_type not in PV_dist_types:
                PV_dist_types[calc_type] = {}
            PV_dist = PV_dist_types[calc_type]

            PV = PV_orig.copy()
            if sess_all:
                PV_B = PV_B_orig.copy()

            if 'shuffle' in calc_type:
                if shuffle_type == 'by_time':
                    shuffle_times = rng.choice(np.arange(PV.shape[1]), size=PV.shape[0])
                    for cell_ in range(PV.shape[0]):
                        PV[cell_,:] = np.roll(PV_orig[cell_,:], shuffle_times[cell_])
                    if sess_all:
                        shuffle_times_B = rng.choice(np.arange(PV_B.shape[1]), size=PV_B.shape[0])
                        for cell_ in range(PV_B.shape[0]):
                            PV_B[cell_,:] = np.roll(PV_B_orig[cell_,:], shuffle_times_B[cell_])
                if shuffle_type == 'by_cell':
                    shuffle_cells = rng.choice(np.arange(PV.shape[0]), size=PV.shape[1])
                    for time_ in range(PV.shape[1]):
                        PV[:,time_] = np.roll(PV_orig[:,time_], shuffle_cells[time_])
                    if sess_all:
                        shuffle_cells_B = rng.choice(np.arange(PV_B.shape[0]), size=PV_B.shape[1])
                        for time_ in range(PV_B.shape[1]):
                            PV_B[:,time_] = np.roll(PV_B_orig[:,time_], shuffle_cells_B[time_])

                #PV = np.roll(PV, tuple(shuffle_times), axis=(1,)*len(shuffle_times))
                #PV_B = np.roll(PV_B, tuple(shuffle_times_B), axis=(1,)*len(shuffle_times_B))

            comparisons = []
            PV0s = []
            PV1s = []
            comp_i0_i1 = []
            comp_range_str = []
            comp_ranges = []

            #
            # First, within-session PV comparisons
            #
            if 'TFC_cond' in session_str:
                post_shock_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.shock_offsets])]
                shock_ranges = [(int(i / bin_frames), int(j / bin_frames)) for i, j in zip(s.shock_onsets, s.shock_offsets)]
                PV_post_shock = np.hstack([PV[:, start:end] for start, end in post_shock_ranges])

                post_tone_20s_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_offsets])]
                first_range = np.round(s.tone_offsets[0]/bin_frames) # really ugly 
                post_tone_20s_ranges[0] = (int(first_range), int(first_range+look_ahead_trunc))
            if 'Test_B' in session_str:
                post_tone_20s_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_offsets])]

            tone_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_onsets])]

            #baseline_ranges = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(len(s.tone_onsets))]
            baseline_ranges = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(int(tone_ranges[0][0]/look_ahead))]
            baseline0_range = (0, look_ahead) # at beginning of recording
            baseline1_range = (tone_ranges[0][0]-look_ahead, tone_ranges[0][0]) # right before first tone

            PV_baseline0 = PV[:, baseline0_range[0]:baseline0_range[1]]
            PV_baseline1 = PV[:, baseline1_range[0]:baseline1_range[1]]

            curr_plot = 0
            range_PV = range(20)

            if 'TFC_cond' in session_str:
                '''
                comparisons += ['baseline0_to_post_shock', 'baseline1_to_post_shock', 'baseline0_to_post_tones_20s', 'baseline1_to_post_tones_20s']
                comparisons += ['post_tone_20s_0_to_post_shocks', 'post_tone_20s_1_to_post_shocks', 'post_tone_20s_2_to_post_shocks', 'post_tone_20s_3_to_post_shocks', 'post_tone_20s_4_to_post_shocks']
                comparisons += ['baseline0_to_tones', 'baseline1_to_tones']
                PV0s += [PV_baseline0, PV_baseline1, PV_baseline0, PV_baseline1]
                PV0s += [PV[:,start:end] for start,end in post_tone_20s_ranges]
                PV0s += [PV_baseline0, PV_baseline1]
                
                PV1s += [PV for i in range(len(PV0s))]
                #PV1s += [ [PV[:,start:end] for start,end in post_tone_20s_ranges] for i in range(4) ]
                #PV1s += [ [PV[:,start:end] for start,end in post_shock_ranges] for i in range(5) ]
                #PV1s += [ [PV[:,start:end] for start,end in tone_ranges] for i in range(2) ]
                comp_ranges += [post_shock_ranges, post_shock_ranges, post_tone_20s_ranges, post_tone_20s_ranges]
                comp_ranges += [post_shock_ranges for i in range(len(post_tone_20s_ranges))]
                comp_ranges += [tone_ranges, tone_ranges]
                '''
                
                #
                # Process across-session PV comparisons (only once, hence when we are handling TFC_cond only)
                #
                if sess_all:
                    tone_ranges_B = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s_B.tone_onsets])]
                    baseline0_range_B = (0, look_ahead) # at beginning of recording
                    baseline1_range_B = (tone_ranges_B[0][0]-look_ahead, tone_ranges_B[0][0]) # right before first tone
                    baseline_range_B = [(0, int(s_B.tone_onsets[0]/bin_frames))]
                    baseline_ranges_B = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(int(tone_ranges_B[0][0]/look_ahead))]
                    post_tone_20s_ranges_B = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s_B.tone_offsets])]
                    
                    PV_baseline0_B = PV_B[:, baseline0_range_B[0]:baseline0_range_B[1]]
                    PV_baseline1_B = PV_B[:, baseline1_range_B[0]:baseline1_range_B[1]]

                    '''
                    comparisons += ['TFC_baseline0-B_baselines_all', 'TFC_baseline1-B_baselines_all']
                    comparisons += ['TFC_post_shock_0-B_baseline', 'TFC_post_shock_1-B_baseline', 'TFC_post_shock_2-B_baseline', 'TFC_post_shock_3-B_baseline', 'TFC_post_shock_4-B_baseline']
                    comparisons += ['TFC_post_shock_0-B_post_tone_20s', 'TFC_post_shock_1-B_post_tone_20s', 'TFC_post_shock_2-B_post_tone_20s', 'TFC_post_shock_3-B_post_tone_20s', 'TFC_post_shock_4-B_post_tone_20s']
                    PV0s += [PV_baseline0, PV_baseline1]
                    PV0s += [PV[:,start:end] for start,end in post_shock_ranges]
                    PV0s += [PV[:,start:end] for start,end in post_shock_ranges]
                    PV1s += [PV_B for i in range(12)]
                    comp_ranges += [baseline_ranges, baseline_ranges]
                    comp_ranges += [baseline_range_B for i in range(len(post_shock_ranges))]
                    comp_ranges += [post_tone_20s_ranges_B for i in range(len(post_shock_ranges))]
                    
                    # Concatenate distances for 'TFC_post_shock_?-B_post_tone_20s' into 'TFC_post_shock_all-B_post_tone_20s'
                    PV_B_post_tone_20s_ranges = np.hstack([PV_B[:, start:end] for start, end in post_tone_20s_ranges_B])
                    comparisons.append('TFC_post_shock_all-B_post_tone_20s')
                    PV0s += [PV_post_shock]
                    PV1s += [PV_B_post_tone_20s_ranges]
                    comp_ranges += [[(0, PV_B_post_tone_20s_ranges.shape[1])]]
                    '''

                    pre=baseline_ranges
                    pre_B=baseline_ranges_B
                    CS=tone_ranges
                    CS_B=tone_ranges_B
                    trace=post_tone_20s_ranges
                    trace_B=post_tone_20s_ranges_B
                    US=shock_ranges
                    post_US=post_shock_ranges
                    #TFC_all = [pre, CS, trace, US, post_US]
                    TFC_all = [PV_range for l in [pre, CS, trace, US, post_US] for PV_range in l]
                    TFC_all_str = ['pre'] * len(pre) + ['CS'] * len(CS) + ['trace'] * len(trace) + ['US'] * len(US) + ['post_US'] * len(post_US)
                    #B_all = [pre_B, CS_B, trace_B]
                    B_all = [PV_range for l in [pre_B, CS_B, trace_B] for PV_range in l]
                    B_all_str = ['pre_B'] * len(pre_B) + ['CS_B'] * len(CS_B) + ['trace_B'] * len(trace_B)

                    for i0,i0_str in zip(range(sum([len(x) for x in TFC_all])), TFC_all_str):
                        for i1,i1_str in zip(range(sum([len(x) for x in B_all])), B_all_str):
                            PV0s.append(PV[:, TFC_all[i0][0]:TFC_all[i0][1]])
                            PV1s.append(PV_B[:, B_all[i1][0]:B_all[i1][1]])
                            comparisons.append(f'{i0_str}-{i1_str}')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append((i0_str, i1_str))

                    # Do all pairwise comparisons between post-shock and post-tone-20s in B. Put them all as PV0-PV1 pairs
                    # and add all distances together for the group. Just aggregate all mice together this way. (eep?)
                    '''
                    for i0 in range(len(post_shock_ranges)):
                        for i1 in range(len(post_tone_20s_ranges_B)):
                            PV0s.append(PV[:, post_shock_ranges[i0][0]:post_shock_ranges[i0][1]])
                            PV1s.append(PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]])
                            comparisons.append('TFC_post_shock-B_post_tone_20s')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('post_shock_ranges', 'post_tone_20s_ranges_B'))
                            #comp_ranges.append([(0, PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]].shape[1])])
                    # Now do it in reverse
                    for i0 in range(len(post_tone_20s_ranges_B)):
                        for i1 in range(len(post_shock_ranges)):
                            PV0s.append(PV_B[:, post_tone_20s_ranges_B[i0][0]:post_tone_20s_ranges_B[i0][1]])
                            PV1s.append(PV[:, post_shock_ranges[i1][0]:post_shock_ranges[i1][1]])
                            comparisons.append('B_post_tone_20s-TFC_post_shock')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('post_tone_20s_ranges_B', 'post_shock_ranges'))
                    # TFC-tone to B-post-tone-20s
                    for i0 in range(len(tone_ranges)):
                        for i1 in range(len(post_tone_20s_ranges_B)):
                            PV0s.append(PV[:, tone_ranges[i0][0]:tone_ranges[i0][1]])
                            PV1s.append(PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]])
                            comparisons.append('TFC_tone-B_post_tone_20s')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('tone_ranges', 'post_tone_20s_ranges_B'))        
                    # REVERSE
                    for i0 in range(len(post_tone_20s_ranges_B)):
                        for i1 in range(len(tone_ranges)):
                            PV0s.append(PV_B[:, post_tone_20s_ranges_B[i0][0]:post_tone_20s_ranges_B[i0][1]])
                            PV1s.append(PV[:, tone_ranges[i1][0]:tone_ranges[i1][1]])
                            comparisons.append('B_post_tone_20s-TFC_tone')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('post_tone_20s_ranges_B', 'tone_ranges'))    
                    for i0 in range(len(tone_ranges)):
                        for i1 in range(len(tone_ranges_B)):
                            PV0s.append(PV[:, tone_ranges[i0][0]:tone_ranges[i0][1]])
                            PV1s.append(PV_B[:, tone_ranges_B[i1][0]:tone_ranges_B[i1][1]])
                            comparisons.append('TFC_tone-B_tone')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('tone_ranges', 'tone_ranges_B'))    
                    # REVERSE
                    for i0 in range(len(tone_ranges_B)):
                        for i1 in range(len(tone_ranges)):
                            PV0s.append(PV_B[:, tone_ranges_B[i0][0]:tone_ranges_B[i0][1]])
                            PV1s.append(PV[:, tone_ranges[i1][0]:tone_ranges[i1][1]])
                            comparisons.append('B_tone-TFC_tone')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('tone_ranges_B', 'tone_ranges'))                                                                                
                    for i0 in range(len(baseline_ranges)):
                        for i1 in range(len(baseline_ranges_B)):
                            PV0s.append(PV[:, baseline_ranges[i0][0]:baseline_ranges[i0][1]])
                            PV1s.append(PV_B[:, baseline_ranges_B[i1][0]:baseline_ranges_B[i1][1]])
                            comparisons.append('TFC_baseline_all-B_baseline_all')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('baseline_ranges', 'baseline_ranges_B'))
                    # REVERSE
                    for i0 in range(len(baseline_ranges_B)):
                        for i1 in range(len(baseline_ranges)):
                            PV0s.append(PV_B[:, baseline_ranges_B[i0][0]:baseline_ranges_B[i0][1]])
                            PV1s.append(PV[:, baseline_ranges[i1][0]:baseline_ranges[i1][1]])
                            comparisons.append('B_baseline_all-TFC_baseline_all')
                            comp_i0_i1.append((i0,i1))
                            comp_range_str.append(('baseline_ranges_B', 'baseline_ranges'))
                    for i1 in range(len(baseline_range_B)):
                        PV0s.append(PV[:, baseline0_range[0]:baseline0_range[1]])
                        PV1s.append(PV_B[:, baseline_range_B[i1][0]:baseline_range_B[i1][1]])
                        comparisons.append('TFC_baseline0-B_baselines_all')
                        comp_i0_i1.append((0,i1))
                        comp_range_str.append(('baseline0_range', 'baseline_range_B'))
                    # REVERSE
                    for i1 in range(len(baseline_range_B)):
                        PV0s.append(PV_B[:, baseline_range_B[i1][0]:baseline_range_B[i1][1]])
                        PV1s.append(PV[:, baseline0_range[0]:baseline0_range[1]])
                        comparisons.append('B_baselines_all-TFC_baseline0')
                        comp_i0_i1.append((i1,0))
                        comp_range_str.append(('baseline_range_B', 'baseline0_range'))
                    for i1 in range(len(baseline_range_B)):
                        PV0s.append(PV[:, baseline1_range[0]:baseline1_range[1]])
                        PV1s.append(PV_B[:, baseline_range_B[i1][0]:baseline_range_B[i1][1]])
                        comparisons.append('TFC_baseline1-B_baselines_all')
                        comp_i0_i1.append((0,i1))
                        comp_range_str.append(('baseline1_range', 'baseline_range_B'))
                    # REVERSE
                    for i1 in range(len(baseline_range_B)):
                        PV0s.append(PV_B[:, baseline_range_B[i1][0]:baseline_range_B[i1][1]])
                        PV1s.append(PV[:, baseline1_range[0]:baseline1_range[1]])
                        comparisons.append('B_baselines_all-TFC_baseline1')
                        comp_i0_i1.append((i1,0))
                        comp_range_str.append(('baseline_range_B', 'baseline1_range'))
                    for i1 in range(len(post_tone_20s_ranges_B)):
                        PV0s.append(PV[:, baseline0_range[0]:baseline0_range[1]])
                        PV1s.append(PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]])
                        comparisons.append('TFC_baseline0-B_post_tone_20s')
                        comp_i0_i1.append((0,i1))
                        comp_range_str.append(('baseline0_range', 'post_tone_20s_ranges_B'))
                    # REVERSE
                    for i1 in range(len(post_tone_20s_ranges_B)):
                        PV0s.append(PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]])
                        PV1s.append(PV[:, baseline0_range[0]:baseline0_range[1]])
                        comparisons.append('B_post_tone_20s-TFC_baseline0')
                        comp_i0_i1.append((i1,0))
                        comp_range_str.append(('post_tone_20s_ranges_B', 'baseline0_range'))
                    for i1 in range(len(post_tone_20s_ranges_B)):
                        PV0s.append(PV[:, baseline1_range[0]:baseline1_range[1]])
                        PV1s.append(PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]])
                        comparisons.append('TFC_baseline1-B_post_tone_20s')
                        comp_i0_i1.append((0,i1))
                        comp_range_str.append(('baseline1_range', 'post_tone_20s_ranges_B'))
                    # REVERSE
                    for i1 in range(len(post_tone_20s_ranges_B)):
                        PV0s.append(PV_B[:, post_tone_20s_ranges_B[i1][0]:post_tone_20s_ranges_B[i1][1]])
                        PV1s.append(PV[:, baseline1_range[0]:baseline1_range[1]])
                        comparisons.append('B_post_tone_20s-TFC_baseline1')
                        comp_i0_i1.append((i1,0))
                        comp_range_str.append(('post_tone_20s_ranges_B', 'baseline1_range'))
                    '''

                    '''
                    PV0s += [PV_baseline0_B for i in range(len(baseline_ranges))]
                    PV0s += [PV_baseline1_B for i in range(len(baseline_ranges))]
                    #PV1s += [PV_baseline0_B for i in range(len(baseline_ranges))]
                    PV1s += [PV_B for i in range(len(baseline_ranges))]
                    PV1s += [PV_B for i in range(len(baseline_ranges))]
                    #PV1s += [PV_B[:, start:end] for start,end in baseline_ranges]
                    #PV1s += [PV_B[:, start:end] for start,end in baseline_ranges]
                    comp_ranges += [baseline_ranges for i in range(len(baseline_ranges))]
                    comp_ranges += [baseline_ranges for i in range(len(baseline_ranges))]
                    #comp_ranges += [baseline_ranges, baseline_ranges]
                    #comparisons_TFC_B = ['TFC-B baseline0-0', 'TFC-B baseline0-1', 'TFC-B baseline1-0', 'TFC-B baseline 1-1']
                    #PV0s_TFC_B = [PV_baseline0, PV_baseline0, PV_baseline1, PV_baseline1]
                    #PV1s_TFC_B = [PV_baseline0_B, PV_baseline1_B, PV_baseline0_B, PV_baseline1_B]
                    '''

                    '''
                    for comparison, PV0, PV1 in zip(comparisons_TFC_B, PV0s_TFC_B, PV1s_TFC_B):
                        print('Working {}... '.format(comparison))
                        PV_dist[comparison] = []
                        for period in range(len(comp_range)):
                            PV_dist[comparison].append([])
                            for i in range(PV0.shape[1]):
                                PV_dist[comparison][period].append(mahalanobis(PV0[:,i], PV1))
                        plt.figure()
                        curr_plot += 1
                        plt.scatter(np.repeat(PV0.shape[1]),PV_dist[comparison])
                        plt.plot(range(range_PV), [np.mean(per) for per in PV_dist[comparison]], 'r')
                        plt.title(comparison)
                        plt.xticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
                        filename = '{}_{}_{}_{}{}_{}_{}.png'.format(Ca_act_type, group, mouse, transpose_str, session_str, plot_num, comparison)
                        os.makedirs(os.path.join(PLOTS_DIR, dir_name, 'PV_dist'), exist_ok=True)
                        plt.savefig(os.path.join(PLOTS_DIR, dir_name, 'PV_dist', filename), format='png', dpi=300)
                        if auto_close:
                            plt.close()
                    '''

            dist_thresh_limit_plot = 0
            dist_lower_plot = 0
            dist_upper_plot = 0
            for comparison, PV0, PV1, plot_num, comp_range_str_curr, comp_i0_i1_curr, in zip(comparisons, PV0s, PV1s, range(len(comparisons)), comp_range_str, comp_i0_i1):
                print('Working {} : {}... ({}-{})'.format(plot_num, comparison, calc_type, shuffle_idx))
                if comparison not in PV_dist:
                    PV_dist[comparison] = []
                for i in range(PV0.shape[1]):
                    if dist_type == 'mahalanobis':
                        dist_mahal, cov_mahal = mahalanobis(PV0[:,i], PV1)
                        dist_mahal = dist_mahal/np.sqrt(PV0.shape[0])
                        if dist_mahal > dist_thresh_mahal:
                            print('DIST IS HUGE {}'.format(dist_mahal))
                        if dist_mahal > dist_thresh_mahal or dist_mahal < -dist_thresh_mahal:
                            if dist_thresh_limit_plot <= dist_plot_limit:
                                plot_pv_matrices_with_distance(PV0, PV1, dist_mahal, dist_type, shuffle_type, comparison, mouse, plot_num=plot_num, i=i, dir_name=os.path.join(PLOTS_DIR, dir_name, 'PV_dist', f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_thresh_mahal', '{}-{}'.format(calc_type, shuffle_idx)), auto_close=auto_close, comp_range_str=comp_range_str_curr, comp_i0_i1=comp_i0_i1_curr, calc_type_str='{}-{}'.format(calc_type, shuffle_idx))
                                dist_thresh_limit_plot += 1
                        else:
                            PV_dist[comparison].append(dist_mahal)
                        dist = dist_mahal
                    if dist_type == 'cosine':
                        cos_sim = cosine_similarity(PV0[:, i].reshape(1, -1), PV1.T).flatten()
                        dist_cosine = np.mean(1 - cos_sim) # Convert similarity to distance
                        #dist_cosine = np.mean(cos_sim) # Just use cosine similarity as distance
                        PV_dist[comparison].append(dist_cosine)
                        dist = dist_cosine

                    # Keep lower and upper bound plots open (for subsequent saving) if the switch is set
                    # and if we are dealing with real data, since don't care for shuffle (so far)
                    if 'real' in calc_type:
                        this_auto_close = close_dist_bound_plots
                    else:
                        this_auto_close = auto_close

                    if dist <= dist_lower_bound:
                        if dist_lower_plot <= dist_plot_limit:
                            plot_pv_matrices_with_distance(PV0, PV1, dist, dist_type, shuffle_type, comparison, mouse, plot_num=plot_num, i=i, dir_name=os.path.join(PLOTS_DIR, dir_name, 'PV_dist', f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_lower', '{}-{}'.format(calc_type, shuffle_idx)), auto_close=this_auto_close, comp_range_str=comp_range_str_curr, comp_i0_i1=comp_i0_i1_curr, calc_type_str='{}-{}'.format(calc_type, shuffle_idx))
                            dist_lower_plot += 1
                    if dist >= dist_upper_bound:
                        if dist_upper_plot <= dist_plot_limit:
                            plot_pv_matrices_with_distance(PV0, PV1, dist, dist_type, shuffle_type, comparison, mouse, plot_num=plot_num, i=i, dir_name=os.path.join(PLOTS_DIR, dir_name, 'PV_dist', f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_upper', '{}-{}'.format(calc_type, shuffle_idx)), auto_close=this_auto_close, comp_range_str=comp_range_str_curr, comp_i0_i1=comp_i0_i1_curr, calc_type_str='{}-{}'.format(calc_type, shuffle_idx))
                            dist_upper_plot += 1

            '''#OLD
            for comparison, PV0, PV1, comp_range, plot_num in zip(comparisons, PV0s, PV1s, comp_ranges, range(len(comparisons))):
                print('Working {} : {}... ({}-{})'.format(plot_num, comparison, calc_type, shuffle_idx))
                if comparison not in PV_dist:
                    PV_dist[comparison] = []
                    for period in range(len(comp_range)):
                        PV_dist[comparison].append([])
                for period in range(len(comp_range)):
                    (start, end) = comp_range[period]
                    for i in range(PV0.shape[1]):
                        dist_mahal, cov_mahal = mahalanobis(PV0[:,i], PV1[:,start:end])
                        PV_dist[comparison][period].append(dist_mahal/np.sqrt(PV0.shape[0]))
                #if 'shuffle' not in calc_type or (shuffle_idx == num_shuffles):
                plt.figure()
                plt.scatter(np.repeat(np.arange(len(comp_range)), PV0.shape[1]*(shuffle_idx if 'shuffle' in calc_type else 1)),PV_dist[comparison])
                plt.plot(range(len(comp_range)), [np.mean(per) for per in PV_dist[comparison]], 'r')
                calc_type_str = calc_type + f'{shuffle_idx}' if 'shuffle' in calc_type else calc_type
                plt.title(comparison+' ({})'.format(calc_type_str))
                #plt.xticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
                plt.xticks(ticks=range(len(comp_range)), labels=range(len(comp_range)))
                filename = '{}_{}_{}_{}{}_{}_{}_{}.png'.format(Ca_act_type, group, mouse, transpose_str, session_str, plot_num, comparison, calc_type_str)
                os.makedirs(os.path.join(PLOTS_DIR, dir_name, 'PV_dist'), exist_ok=True)
                plt.savefig(os.path.join(PLOTS_DIR, dir_name, 'PV_dist', filename), format='png', dpi=300)
                if auto_close:
                    plt.close()
            '''

            '''#OLD
            if sess_all:
                #if 'shuffle' not in calc_type or (shuffle_idx == num_shuffles):
                if mouse == 'G09' and 'shuffle' in calc_type: # don't bother for now
                    continue
                if mouse == 'G09':
                    pre_period1 = ['TFC_post_shock_0-B_baseline', 'TFC_post_shock_1-B_baseline', 'TFC_post_shock_2-B_baseline', 'TFC_post_shock_3-B_baseline']
                else:
                    pre_period1 = ['TFC_post_shock_0-B_baseline', 'TFC_post_shock_1-B_baseline', 'TFC_post_shock_2-B_baseline', 'TFC_post_shock_3-B_baseline', 'TFC_post_shock_4-B_baseline']                
                PV_dist_pre_period1 = [PV_dist[p][0] for p in pre_period1]
                PV_dist[pre_period1[0]+'-all_pre'] = PV_dist_pre_period1
                plt.figure(); plot_num += 1
                if mouse == 'G09':
                    plt.scatter(np.concatenate((np.repeat(0,20),np.repeat(1,20),np.repeat(2,15),np.repeat(3,20))), np.concatenate(PV_dist_pre_period1))
                else:
                    plt.scatter(np.repeat(np.arange(len(pre_period1)), len(PV_dist[pre_period1[0]][0])), PV_dist_pre_period1)
                plt.plot(range(len(pre_period1)), [np.mean(per) for per in PV_dist_pre_period1], 'r')
                plt.xticks(ticks=range(len(pre_period1)), labels=range(len(pre_period1)))              
                calc_type_str = calc_type + f'{shuffle_idx}' if 'shuffle' in calc_type else calc_type
                plt.title('{}-all_pre ({})'.format(pre_period1[0], calc_type_str))
                filename = '{}_{}_{}_{}{}_{}_{}-all_pre_{}.png'.format(Ca_act_type, group, mouse, transpose_str, session_str, plot_num, pre_period1[0], calc_type_str)
                os.makedirs(os.path.join(PLOTS_DIR, dir_name, 'PV_dist'), exist_ok=True)
                plt.savefig(os.path.join(PLOTS_DIR, dir_name, 'PV_dist', filename), format='png', dpi=300)
                if auto_close:
                    plt.close()
            '''
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
        pv = PopulationVector(mouse, group, s, binary_C, labels, frac_labels, labels_tot, dend_thresh, x_dend=x_dend, y_ss=y_ss, y_n_clusters=y_n_clusters, \
            only_crossreg=only_crossreg, PV_dist_types=PV_dist_types)
    else:
        pv = PopulationVector(mouse, group, s, binary_C, labels, frac_labels, labels_tot, dend_thresh, x_dend=x_dend, y_n_clusters=y_n_clusters, \
            only_crossreg=only_crossreg, PV_dist_types=PV_dist_types) 
    return pv

def cluster_pop_vectors(PLOTS_DIR, session, session_str, mice_per_group, transpose_wanted=False, auto_close=False, bin_width=1, \
    spk_cutoff=2, crossreg=None, Ca_act_type='full', use_silhouette=True, only_crossreg=False, sess_all_use=None, want_binary_PV=True, \
    num_shuffles=5, dist_thresh_mahal=1e5, close_dist_bound_plots=True, PV_use_B_1wk=False, dist_type='mahalanobis', shuffle_type='by_time', \
    dist_lower_bound=5, dist_upper_bound=100):

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
                sess_all=sess_all_use, want_binary_PV=want_binary_PV, num_shuffles=num_shuffles, dist_thresh_mahal=dist_thresh_mahal, \
                close_dist_bound_plots=close_dist_bound_plots, PV_use_B_1wk=PV_use_B_1wk, dist_type=dist_type, shuffle_type=shuffle_type, \
                dist_lower_bound=dist_lower_bound, dist_upper_bound=dist_upper_bound)

            PV_group[group].append(PV_mouse)

    print("*** ...done.")
    return PV_group

def pop_vectors_dist_helper(PLOTS_DIR, session, session_str, mouse, group, \
    auto_close=False, bin_width=1, spk_cutoff=2, crossreg=None, Ca_act_type='full', \
    sess_all=None, num_shuffles=5, want_binary_PV=True, \
    dist_thresh_mahal=1e5, dist_plot_limit=10, dist_lower_bound=0.1, dist_upper_bound=10, close_dist_bound_plots=True, PV_use_B_1wk=False, \
    dist_type='mahalanobis', shuffle_type='by_time', cov_type='lw', \
    plot_bounds=False):
    '''
    Calculate population vector distances using various measures. Heavily modified/"forked" from cluster_pop_vectors_helper().

    Because processing times can be long, only process one mouse at a time specified by argument. Caller
    can loop over groups etc. as desired.

    Major switches:

        shuffle_type - either 'by_time' or 'by_cells'. If 'by_time', shuffle time points within each cell. 
            If 'by_cells', shuffle cells within each time point.
    NB: transpose not fully implemented.
    '''
    s = binary_C = labels = frac_labels = labels_tot = None # for consistenty so can use PV object even for transpose (which doesn't use these yet)
    if mouse in ['G07', 'G15']:
        sess_all = None

    dir_name = 'PV_dist'

    # To store for transpose-related calculations.
    PV_dist_types = {}

    s=session[mouse]
    crossreg_mouse=crossreg[mouse]
    if mouse in ['G07']:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B_1wk', with_crossreg=crossreg_mouse) # not ideal to specify manually but...
    elif mouse in ['G15']:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B', with_crossreg=crossreg_mouse) # not ideal to specify manually but...
    else:
        [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping('TFC_cond+Test_B+Test_B_1wk', with_crossreg=crossreg_mouse) # not ideal to specify manually but...

    S_i = s.get_S_indeces(S_idx)
    #S_i = get_S_indeces_crossreg(s, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk') # maybe not good to hard-code mapping str here
    if sess_all:
        s_B = sess_all[1][mouse]
        S_i_Test_B = get_S_indeces_crossreg(s_B, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk') 
        s_B_1wk = sess_all[2][mouse]
        S_i_Test_B_1wk = get_S_indeces_crossreg(s_B_1wk, crossreg_mouse, 'TFC_cond+Test_B+Test_B_1wk')    

    if Ca_act_type == 'mov':
        C=s.S_mov[S_i,:]
        if sess_all:
            C_B=s_B.S_mov[S_i_Test_B,:]
            C_B_1wk=s_B_1wk.S_mov[S_i_Test_B_1wk,:]
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

    max_C = np.max(C)
    transpose_str = ''
    bin_frames = bin_width * MINISCOPE_FPS
    look_ahead = int(20 / bin_width) # should be 20 seconds.
    look_ahead_trunc = int(15 / bin_width) # just for TFC_cond 1st period

    PV = np.zeros((C.shape[0], math.floor(C.shape[1]/bin_frames)))
    vel = np.zeros(math.floor(len(s.velocities_miniscope_smooth)/bin_frames))
    if sess_all:
        PV_B = np.zeros((C_B.shape[0], math.floor(C_B.shape[1]/bin_frames)))
        PV_B_1wk = np.zeros((C_B_1wk.shape[0], math.floor(C_B_1wk.shape[1]/bin_frames)))
        vel_B = np.zeros(math.floor(len(s_B.velocities_miniscope_smooth)/bin_frames)) 
        vel_B_1wk = np.zeros(math.floor(len(s_B_1wk.velocities_miniscope_smooth)/bin_frames))
        C_calcs = [C, C_B, C_B_1wk]
        PV_calcs = [PV, PV_B, PV_B_1wk]
        vel_calcs = [vel, vel_B, vel_B_1wk]
        vels = [s.velocities_miniscope_smooth, s_B.velocities_miniscope_smooth, s_B_1wk.velocities_miniscope_smooth]
    else:
        C_calcs = [C]
        PV_calcs = [PV]
        vel_calcs = [vel]
        vels = [s.velocities_miniscope_smooth]

    for C_calc, PV_calc, vel_calc, vel in zip(C_calcs, PV_calcs, vel_calcs, vels):
        if PV_calc is None:
            continue
        curr_frame = 0
        upper = 1
        lower = 0
        #inc = 0
        for i in range(PV_calc.shape[1]):
            if want_binary_PV:
                PV_calc[:,i] = np.where(np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
            else:
                PV_calc[:,i] = np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames
            vel_calc[i] = np.mean(vel[curr_frame:curr_frame+bin_frames])
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
    if sess_all:
        sess_titles = ['TFC_cond', 'Test_B', 'Test_B_1wk']
        s_plots = [s, s_B, s_B_1wk]
        max_xticks = [1300, 900, 900]
    else:
        sess_titles = ['TFC_cond']
        s_plots = [s]
        max_xticks = [1300]
    
    #
    # Plot all PVs first
    #
    fig, axs = plt.subplots(1, len(PV_calcs), figsize=(20, 4))
    axs = np.atleast_1d(axs) # in case only one plot
    for ax, sess_title, PV_plot, s_plot, max_xtick in zip(axs, sess_titles, PV_calcs, s_plots, max_xticks):
        ax.imshow(PV_plot, aspect='auto', cmap='jet')
        if sess_title == 'TFC_cond':
            for i in s_plot.shock_onsets:
                ax.axvline(i/bin_frames, c='r', ls='-', lw=0.5)
        for i in s_plot.tone_onsets:
            ax.axvline(i/bin_frames, c='b', ls='-', lw=0.5) 
        if sess_title == 'TFC_cond':
            for i in s_plot.shock_offsets:
                ax.axvline(i/bin_frames, c='r', ls='-', lw=0.5)
        for i in s_plot.tone_offsets:
            ax.axvline(i/bin_frames, c='b', ls='-', lw=0.5)
        ax.set_xlim((0, PV_plot.shape[1]))
        #ax.set_ylim((0, PV_plot.shape[0]))
        xtick_seconds_PV = np.concatenate(([0], [s_plot/bin_width for s_plot in s_plot.tone_onsets_def[0:len(s_plot.tone_onsets)]], [np.min((1300/bin_width,PV_plot.shape[1]))]))
        xtick_seconds = np.concatenate(([0], s_plot.tone_onsets_def[0:len(s_plot.tone_onsets)], [max_xtick]))
        ax.set_xticks(xtick_seconds_PV)
        ax.set_xticklabels(xtick_seconds)
        ax.set_title(f'{mouse} {sess_title} {Ca_act_type}')

    plt.tight_layout()

    # Save the figure
    filename = '{}_{}_{}{}_{}_PV_binned.png'.format(Ca_act_type, group, mouse, transpose_str, '_'.join(sess_titles))
    os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
    if auto_close:
        plt.close()

    #
    # Perform PV calculations
    #
    PV_orig = PV
    if sess_all:
        if PV_use_B_1wk:
            PV_B_orig = PV_B_1wk
            s_B = s_B_1wk
        else:
            PV_B_orig = PV_B

    #num_shuffles=5
    #calc_types = ['real'] + [f'shuffle{i}' for i in range(num_shuffles)]
    calc_types = ['real'] + ['shuffle' for i in range(num_shuffles)]
    shuffle_idx = 0
    rng = np.random.default_rng()

    for calc_type in calc_types:
        if 'shuffle' in calc_type:
            shuffle_idx += 1
        if calc_type not in PV_dist_types:
            PV_dist_types[calc_type] = {}
        PV_dist_comparisons = PV_dist_types[calc_type]

        PV = PV_orig.copy()
        if sess_all:
            PV_B = PV_B_orig.copy()

        if 'shuffle' in calc_type:
            if shuffle_type == 'by_time':
                #shuffle_times = rng.choice(np.arange(PV.shape[1]), size=PV.shape[0])
                #for cell_ in range(PV.shape[0]):
                #    PV[cell_,:] = np.roll(PV_orig[cell_,:], shuffle_times[cell_])
                if sess_all:
                    shuffle_times_B = rng.choice(np.arange(PV_B.shape[1]), size=PV_B.shape[0])
                    for cell_ in range(PV_B.shape[0]):
                        PV_B[cell_,:] = np.roll(PV_B_orig[cell_,:], shuffle_times_B[cell_])
            elif shuffle_type == 'by_cells':
                #shuffle_cells = rng.choice(np.arange(PV.shape[0]), size=PV.shape[1])
                #for time_ in range(PV.shape[1]):
                #    PV[:,time_] = np.roll(PV_orig[:,time_], shuffle_cells[time_])
                if sess_all:
                    shuffle_cells_B = rng.choice(np.arange(PV_B.shape[0]), size=PV_B.shape[1])
                    for time_ in range(PV_B.shape[1]):
                        PV_B[:,time_] = np.roll(PV_B_orig[:,time_], shuffle_cells_B[time_])
            else:
                raise ValueError(f"Unknown shuffle type {shuffle_type}")

            #PV = np.roll(PV, tuple(shuffle_times), axis=(1,)*len(shuffle_times))
            #PV_B = np.roll(PV_B, tuple(shuffle_times_B), axis=(1,)*len(shuffle_times_B))

        comparisons = []
        PV0s = []
        PV1s = []
        comp_i0_i1 = []
        comp_range_str = []
        comp_ranges = []

        #
        # First, within-session PV comparisons
        #
        if 'TFC_cond' in session_str:
            post_shock_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.shock_offsets])]
            shock_ranges = [(int(i / bin_frames), int(j / bin_frames)) for i, j in zip(s.shock_onsets, s.shock_offsets)]
            PV_post_shock = np.hstack([PV[:, start:end] for start, end in post_shock_ranges])

            post_tone_20s_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_offsets])]
            first_range = np.round(s.tone_offsets[0]/bin_frames) # really ugly 
            post_tone_20s_ranges[0] = (int(first_range), int(first_range+look_ahead_trunc))
        if 'Test_B' in session_str:
            post_tone_20s_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_offsets])]

        tone_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_onsets])]

        #baseline_ranges = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(len(s.tone_onsets))]
        baseline_ranges = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(int(tone_ranges[0][0]/look_ahead))]
        baseline0_range = (0, look_ahead) # at beginning of recording
        baseline1_range = (tone_ranges[0][0]-look_ahead, tone_ranges[0][0]) # right before first tone

        PV_baseline0 = PV[:, baseline0_range[0]:baseline0_range[1]]
        PV_baseline1 = PV[:, baseline1_range[0]:baseline1_range[1]]

        curr_plot = 0
        range_PV = range(20)

        if 'TFC_cond' in session_str:
            #
            # Process across-session PV comparisons (only once, hence when we are handling TFC_cond only)
            #
            if sess_all:
                tone_ranges_B = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s_B.tone_onsets])]
                baseline0_range_B = (0, look_ahead) # at beginning of recording
                baseline1_range_B = (tone_ranges_B[0][0]-look_ahead, tone_ranges_B[0][0]) # right before first tone
                baseline_range_B = [(0, int(s_B.tone_onsets[0]/bin_frames))]
                baseline_ranges_B = [(i*look_ahead, i*look_ahead + look_ahead) for i in range(int(tone_ranges_B[0][0]/look_ahead))]
                post_tone_20s_ranges_B = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s_B.tone_offsets])]
                
                PV_baseline0_B = PV_B[:, baseline0_range_B[0]:baseline0_range_B[1]]
                PV_baseline1_B = PV_B[:, baseline1_range_B[0]:baseline1_range_B[1]]

                pre=baseline_ranges
                pre_B=baseline_ranges_B
                CS=tone_ranges
                CS_first=[tone_ranges[0]]
                CS_last=[tone_ranges[-1]]
                CS_B=tone_ranges_B
                CS_B_first=[tone_ranges_B[0]]
                CS_B_last=[tone_ranges_B[-1]]
                trace=post_tone_20s_ranges
                trace_first=[post_tone_20s_ranges[0]]
                trace_last=[post_tone_20s_ranges[-1]]
                trace_B=post_tone_20s_ranges_B
                trace_B_first=[post_tone_20s_ranges_B[0]]
                trace_B_last=[post_tone_20s_ranges_B[-1]]
                US=shock_ranges
                US_first=[shock_ranges[0]]
                US_last=[shock_ranges[-1]]
                post_US=post_shock_ranges
                post_US_first=[post_shock_ranges[0]]
                post_US_last=[post_shock_ranges[-1]]
                #TFC_all = [pre, CS, trace, US, post_US]
                TFC_all = [PV_range for l in [pre, CS, CS_first, CS_last, trace, trace_first, trace_last, US, US_first, US_last, post_US, post_US_first, post_US_last] for PV_range in l]
                TFC_all_str = ['pre'] * len(pre) + ['CS'] * len(CS) + ['CS_first'] * len(CS_first) + ['CS_last'] * len(CS_last) + \
                    ['trace'] * len(trace) + ['trace_first'] * len(trace_first) + ['trace_last'] * len(trace_last) + \
                    ['US'] * len(US) + ['US_first'] * len(US_first) + ['US_last'] * len(US_last) + \
                    ['post_US'] * len(post_US) + ['post_US_first'] * len(post_US_first) + ['post_US_last'] * len(post_US_last)
                TFC_all_post = [PV_range for l in [trace, trace_first, trace_last, CS, CS_first, CS_last, post_US, post_US_first, post_US_last] for PV_range in l]
                TFC_all_post_str = ['trace'] * len(trace) + ['trace_first'] * len(trace_first) + ['trace_last'] * len(trace_last) + \
                    ['CS'] * len(CS) + ['CS_first'] * len(CS_first) + ['CS_last'] * len(CS_last) + \
                    ['post_US'] * len(post_US) + ['post_US_first'] * len(post_US_first) + ['post_US_last'] * len(post_US_last)                    
                #B_all = [pre_B, CS_B, trace_B]
                B_all = [PV_range for l in [pre_B, CS_B, CS_B_first, CS_B_last, trace_B, trace_B_first, trace_B_last] for PV_range in l]
                B_all_str = ['pre_B'] * len(pre_B) + \
                    ['CS_B'] * len(CS_B) + ['CS_B_first'] * len(CS_B_first) + ['CS_B_last'] * len(CS_B_last) + \
                    ['trace_B'] * len(trace_B) + ['trace_B_first'] * len(trace_B_first) + ['trace_B_last'] * len(trace_B_last)

                # TFC-TFC within-session PV calculations (gulp?)
                # No need for these...
                '''
                for i0,i0_str in zip(range(sum([len(x) for x in TFC_all])), TFC_all_str):
                    for i1,i1_str in zip(range(sum([len(x) for x in TFC_all_post])), TFC_all_post_str):
                        PV0s.append(PV[:, TFC_all[i0][0]:TFC_all[i0][1]])
                        PV1s.append(PV[:, TFC_all_post[i1][0]:TFC_all_post[i1][1]])
                        comparisons.append(f'{i0_str}-{i1_str}')
                        comp_i0_i1.append((i0,i1))
                        comp_range_str.append((i0_str, i1_str))
                '''

                # TFC-B (or B_1wk) across-session PV calculations
                for i0,i0_str in zip(range(sum([len(x) for x in TFC_all])), TFC_all_str):
                    for i1,i1_str in zip(range(sum([len(x) for x in B_all])), B_all_str):
                        PV0s.append(PV[:, TFC_all[i0][0]:TFC_all[i0][1]])
                        PV1s.append(PV_B[:, B_all[i1][0]:B_all[i1][1]])
                        comparisons.append(f'{i0_str}-{i1_str}')
                        comp_i0_i1.append((i0,i1))
                        comp_range_str.append((i0_str, i1_str))

        dist_thresh_limit_plot = 0
        dist_lower_plot = 0
        dist_upper_plot = 0
        for comparison, PV0, PV1, plot_num, comp_range_str_curr, comp_i0_i1_curr, in zip(comparisons, PV0s, PV1s, range(len(comparisons)), comp_range_str, comp_i0_i1):
            dist_PV0_PV1_tot = []
            print('Working {} : {}... ({}-{})'.format(plot_num, comparison, calc_type, shuffle_idx))
            if comparison not in PV_dist_types[calc_type]:
                PV_dist_comparisons[comparison] = []
            for i in range(PV0.shape[1]):
                if dist_type == 'mahalanobis':
                    dist_mahal, cov_mahal = mahalanobis(PV0[:,i], PV1, cov_type=cov_type)
                    dist_mahal = dist_mahal/np.sqrt(PV0.shape[0])
                    if dist_mahal > dist_thresh_mahal:
                        #print('DIST IS HUGE {}'.format(dist_mahal))
                        pass
                    if dist_mahal > dist_thresh_mahal or dist_mahal < -dist_thresh_mahal:
                        if dist_thresh_limit_plot <= dist_plot_limit:
                            plot_pv_matrices_with_distance(PV0, PV1, dist_mahal, dist_type, shuffle_type, comparison, mouse, plot_num=plot_num, i=i, dir_name=os.path.join(PLOTS_DIR, dir_name, f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_thresh_mahal', '{}-{}'.format(calc_type, shuffle_idx)), auto_close=auto_close, comp_range_str=comp_range_str_curr, comp_i0_i1=comp_i0_i1_curr, calc_type_str='{}-{}'.format(calc_type, shuffle_idx))
                            dist_thresh_limit_plot += 1
                    else:
                        pass
                        #PV_dist_comparisons[comparison].append(dist_mahal)
                    dist = dist_mahal
                if dist_type == 'cosine':
                    cos_sim = cosine_similarity(PV0[:, i].reshape(1, -1), PV1.T).flatten()
                    dist_cosine = np.mean(1 - cos_sim) # Convert similarity to distance
                    #dist_cosine = np.mean(cos_sim) # Just use cosine similarity as distance
                    #PV_dist_comparisons[comparison].append(dist_cosine)
                    dist = dist_cosine
                if dist_type == 'jaccard':
                    jaccard_dist = [jaccard(PV0[:, i], PV1[:, j]) for j in range(PV1.shape[1])]
                    dist_jaccard = np.mean(jaccard_dist)
                    #PV_dist_comparisons[comparison].append(dist_jaccard)
                    dist = dist_jaccard

                dist_PV0_PV1_tot.append(dist)
            PV_dist_comparisons[comparison].append(np.mean(dist_PV0_PV1_tot))

            if plot_bounds:
                # Keep lower and upper bound plots open (for subsequent saving) if the switch is set
                # and if we are dealing with real data, since don't care for shuffle (so far)
                if 'real' in calc_type:
                    this_auto_close = close_dist_bound_plots
                else:
                    this_auto_close = auto_close

                if dist <= dist_lower_bound:
                    if dist_lower_plot <= dist_plot_limit:
                        plot_pv_matrices_with_distance(PV0, PV1, dist, dist_type, shuffle_type, comparison, mouse, plot_num=plot_num, i=i, dir_name=os.path.join(PLOTS_DIR, dir_name, f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_lower', '{}-{}'.format(calc_type, shuffle_idx)), auto_close=this_auto_close, comp_range_str=comp_range_str_curr, comp_i0_i1=comp_i0_i1_curr, calc_type_str='{}-{}'.format(calc_type, shuffle_idx))
                        dist_lower_plot += 1
                if dist >= dist_upper_bound:
                    if dist_upper_plot <= dist_plot_limit:
                        plot_pv_matrices_with_distance(PV0, PV1, dist, dist_type, shuffle_type, comparison, mouse, plot_num=plot_num, i=i, dir_name=os.path.join(PLOTS_DIR, dir_name, f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_upper', '{}-{}'.format(calc_type, shuffle_idx)), auto_close=this_auto_close, comp_range_str=comp_range_str_curr, comp_i0_i1=comp_i0_i1_curr, calc_type_str='{}-{}'.format(calc_type, shuffle_idx))
                        dist_upper_plot += 1
        
        if PV_dist_comparisons:
            #
            # Plot histograms for this mouse
            #
            real_keys = list(PV_dist_types['real'].keys())

            # Determine the number of rows and columns for the grid
            num_keys = len(real_keys)
            keys_per_plot = 20  # Approximate number of keys per plot
            num_plots = (num_keys + keys_per_plot - 1) // keys_per_plot  # Calculate the number of plots needed

            # Directory to save the plots
            output_dir = os.path.join('plots', mouse)
            output_dir = os.path.join(PLOTS_DIR, dir_name, f'PV_use_B_1wk_{PV_use_B_1wk}', 'dist_histograms', '{}-{}'.format(calc_type, shuffle_idx))
            os.makedirs(output_dir, exist_ok=True)

            # Create separate figures for each group of keys
            for plot_idx in range(num_plots):
                start_idx = plot_idx * keys_per_plot
                end_idx = min((plot_idx + 1) * keys_per_plot, num_keys)
                keys_subset = real_keys[start_idx:end_idx]

                num_cols = 4  # Number of columns in the grid
                num_rows = (len(keys_subset) + num_cols - 1) // num_cols  # Calculate the number of rows needed

                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

                # Flatten the axs array for easy iteration
                axs = axs.flatten()

                # Plot histograms for each key in the subset
                for ax, key in zip(axs, keys_subset):
                    ax.hist(PV_dist_types['real'][key], bins=100, alpha=0.75)
                    ax.set_title(key)
                    ax.set_xlabel('Distance')
                    ax.set_ylabel('Frequency')
                    ax.grid(True)

                # Remove any unused subplots
                for ax in axs[len(keys_subset):]:
                    fig.delaxes(ax)

                plt.tight_layout()
                # Save the figure to a file
                plot_filename = os.path.join(output_dir, f'PV_dist_plot_{mouse}_{plot_idx + 1}.png')
                plt.savefig(plot_filename, format='png', dpi=300)
                if auto_close:
                    plt.close()
    return PV_dist_types

def pop_vectors_dist(PLOTS_DIR, session, session_str, mice_per_group, auto_close=False, bin_width=1, \
    spk_cutoff=2, crossreg=None, Ca_act_type='full', sess_all_use=None, want_binary_PV=True, \
    num_shuffles=5, dist_thresh_mahal=1e5, close_dist_bound_plots=True, PV_use_B_1wk=False, dist_type='mahalanobis', shuffle_type='by_time', \
    dist_lower_bound=5, dist_upper_bound=100, cov_type='lw', cov_regularize=False, cov_lambda=1e-6, \
    plot_bounds=False):

    session_type = list(session.values())[0].session_type
    PV_dist = dict()

    for group, mice in mice_per_group.items():
        for mouse in mice:
            if group not in PV_dist.keys():
                PV_dist[group] = []

            if session_type == 'Test_B' and mouse in ['G07']:
                continue
            if session_type == 'Test_B_1wk' and mouse in ['G15']:
                continue

            print("*** Processing PV dist for {} {} {} {}...".format(session_str, Ca_act_type, group, mouse))
            PV_dist_mouse = pop_vectors_dist_helper(PLOTS_DIR, session, session_str, mouse, group, \
                auto_close=auto_close, bin_width=bin_width, \
                spk_cutoff=spk_cutoff, crossreg=crossreg, Ca_act_type=Ca_act_type, \
                sess_all=sess_all_use, want_binary_PV=want_binary_PV, num_shuffles=num_shuffles, dist_thresh_mahal=dist_thresh_mahal, \
                close_dist_bound_plots=close_dist_bound_plots, PV_use_B_1wk=PV_use_B_1wk, dist_type=dist_type, shuffle_type=shuffle_type, \
                dist_lower_bound=dist_lower_bound, dist_upper_bound=dist_upper_bound, cov_type=cov_type, \
                plot_bounds=plot_bounds)

            PV_dist[group].append(PV_dist_mouse)

    print("*** ...done.")
    return PV_dist

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

def plot_data_group(list_func, PLOTS_DIR, data_list, title='', ylabel='', filename='', type='boxplot', auto_close=True, want_scatter=False, \
        use_median=False, use_log=False, whisker_length=1.5):
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
    medians = np.zeros(num_groups)
    sems = np.zeros(num_groups)
    errbars = np.zeros((2,num_groups))
    data = dict()
    if use_log:
        ylabel += ' (log)'

    if want_scatter:
        x_scatter = []
        data_scatter = []
    for group, idx in zip(groups, range(num_groups)):
        # So can work in multiple places (messy, sorry)/
        if isinstance(data_list[group][0], tuple):
            data[group] = [tup[0] for tup in data_list[group]]
        else:
            data[group] = data_list[group]
        if use_log:
            data[group] = np.nan_to_num(np.log(data[group]), nan=1e-10)
        means[idx] = np.mean(data[group])
        medians[idx] = np.median(data[group])
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
            notch=False, patch_artist=True, positions=x, showfliers=False, widths=0.7, whis=whisker_length)
        for p, c in zip(bp['boxes'], ['red','blue','black']):
            plt.setp(p,facecolor=c,alpha=0.5)
        for median in bp['medians']:
            plt.setp(median,color='black')

        # Extract whisker high values
        whiskers = bp['whiskers']
        whisker_high_values = []
        for i in range(1, len(whiskers), 2):
            whisker_high = whiskers[i].get_ydata()[1]
            whisker_high_values.append(whisker_high)

    if type == 'violinplot':
        if use_median:
            showmedians=True
            showmeans=False
        else:
            showmedians=False
            showmeans=True
        #ax.violinplot([data[groups[0]], data[groups[1]], data[groups[2]]], positions=x, showmeans=showmeans, showmedians=showmedians, showextrema=False)
        ax.violinplot([data[group] for group in groups], positions=x, showmeans=showmeans, showmedians=showmedians, showextrema=False)

    if type == 'barplot':
        if use_median:
            ax.bar(x, medians, yerr=errbars, color=group_colours.values())
        else:
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
            if type == 'boxplot':
                heights = whisker_high_values
            elif type == 'barplot':
                heights = means+sems
            do_anova1_plot(data[groups[0]], data[groups[1]], data[groups[2]], ax, heights)
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
    pass

def plot_group_PV_stats(PLOTS_DIR, PV_dist, auto_close=True, use_log=False, binary_PV_type=None, shuffle_normalization=True, \
    dist_type=None, shuffle_type='SHUFFLE_UNKNOWN', PV_use_B_1wk_type=None, want_scatter=False, plot_type='boxplot', use_median=False):
    '''
    Plot across-group stats for PV distances.
    '''
    PV_dist_parent = PV_dist[binary_PV_type][dist_type][shuffle_type][PV_use_B_1wk_type]
    PV_dist_curr = PV_dist_parent['PV_dist']
    first_PV = PV_dist_curr[next(iter(PV_dist_curr))][0]
    comparisons = list(first_PV['real'].keys())
    calc_types = list(first_PV.keys())
    path_dir = os.path.join(PLOTS_DIR, 'PV_dist', f'{PV_use_B_1wk_type}', 'stats')
    os.makedirs(path_dir, exist_ok=True)
    from scipy.stats import median_abs_deviation

    flat_data = []
    comp_data = dict()
    for calc_type in calc_types:
        comp_data[calc_type] = dict()
        for comp in comparisons:
            print(f'*** {calc_type} {comp}')
            data_group = dict()
            comp_data[calc_type][comp] = dict()
            for group in PV_dist_curr.keys():
                #data[group] = [v.PV_dist[comp] for v in PV_group[group]]
                #data[group] = [dist for v in PV_group[group] for dist in v.PV_dist[comp]]
                #data[group] = [dist for v in PV_group[group] if comp in v.PV_dist_types[calc_type] for period in v.PV_dist_types[calc_type][comp] for dist in period]

                data = []
                data_indices = []
                normalized_data = []
                for idx, v in enumerate(PV_dist_curr[group]):
                    if comp in v[calc_type]:
                        for dist in v[calc_type][comp]:
                            data.append(dist)
                            normalized_data.append(dist)
                            data_indices.append(idx)  # Append the index of the v vector ('mouse_index')
                if use_log:
                    data = np.log(np.where(np.array(data) > 0, data, 1e-6))
                    normalized_data = np.log(np.where(np.array(normalized_data) > 0, normalized_data, 1e-6))
                data_group[group] = data

                if calc_type == 'real' and shuffle_normalization:
                    shuffle_data = [dist for v in PV_dist_curr[group] if comp in v['shuffle'] for dist in v['shuffle'][comp]]
                    if use_log:
                        shuffle_data = np.log(np.where(np.array(shuffle_data) > 0, shuffle_data, 1e-6))
                    #shuffle_data = zscore(shuffle_data)
                    shuffle_mean = np.mean(shuffle_data)
                    shuffle_sd = np.std(shuffle_data)
            
                    # Normalize the real data by the mean of the shuffle data
                    #normalized_data = np.array(real_data) / shuffle_mean
                    #normalized_data = np.array(real_data) - shuffle_mean
                    #####normalized_data = (np.array(real_data) - shuffle_mean) / (shuffle_sd / np.sqrt(len(real_data)))
                    #normalized_data = zscore(normalized_data)
                    #normalized_data = zscore(np.array(real_data)) / shuffle_mean

                    mean = np.median(shuffle_data)  # Use median instead of mean
                    mad = median_abs_deviation(shuffle_data)  # Use MAD instead of SD
                    #data = (data - mean) / (mad + 1e-6)  # Robust z-scoring
                    normalized_data = (normalized_data - mean) / (mad + 1e-6)  # Robust z-scoring

                    # Take the logarithm and ignore NaN values
                    '''
                    real_data_log = np.log(np.where(np.array(real_data) > 0, real_data, np.nan))
                    shuffle_data_log = np.log(np.where(np.array(shuffle_data) > 0, shuffle_data, np.nan))
                    normalized_data = real_data_log #/ shuffle_data_log
                    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
                    '''

                    # Enter data as the original data, then set data to the normalized data for subsequent plotting etc
                    for datum, mouse_idx in zip(data, data_indices):
                        flat_data.append({
                            'binary_PV_type': binary_PV_type,
                            'dist_type': dist_type,
                            'shuffle_type': shuffle_type,
                            'PV_use_B_1wk_type': PV_use_B_1wk_type,
                            'calc_type' : 'real_raw',
                            'group': group,
                            'mouse_index': mouse_idx,  # Mouse number within the group
                            'comparison_key': comp,  # e.g., 'post_US-trace_B'
                            'PV_dist': datum  # The actual distance value
                        })      
                    data = normalized_data 

                # Ensure all values are positive before applying logarithm
                #if use_log:
                #    normalized_data[normalized_data <= 0] = 1e-10  # Replace non-positive values with a small positive value
                #    normalized_data = np.log(normalized_data)  # Use np.log10(normalized_data) for base-10 logarithm
                
                data_group[group] = data    

                for datum, mouse_idx in zip(data, data_indices):
                    flat_data.append({
                        'binary_PV_type': binary_PV_type,
                        'dist_type': dist_type,
                        'shuffle_type': shuffle_type,
                        'PV_use_B_1wk_type': PV_use_B_1wk_type,
                        'calc_type' : calc_type,
                        'group': group,
                        'mouse_index': mouse_idx,  # Mouse number within the group
                        'comparison_key': comp,  # e.g., 'post_US-trace_B'
                        'PV_dist': datum  # The actual distance value
                    })                        

                comp_data[calc_type][comp][group] = data

            plot_data_group(list, path_dir, data_group, title='{}\n({}) shuffle_norm: {} {}'.format(comp, calc_type, shuffle_normalization, PV_use_B_1wk_type), ylabel='Distance', \
                filename=f'{comp}_{calc_type}_{dist_type}_shuffle_type_{shuffle_type}_shuffle_norm_{shuffle_normalization}_{PV_use_B_1wk_type}', \
                want_scatter=want_scatter, type=plot_type, auto_close=auto_close, use_median=use_median, use_log=False, whisker_length=0.2)

        groups = list(comp_data[calc_type][next(iter(comp_data[calc_type]))].keys())  # Extract groups from comp_data

        for group in groups:
            fig, ax = plt.subplots()
            data_to_plot = [comp_data[calc_type][comp][group] for comp in comparisons]
            
            ax.boxplot(data_to_plot, notch=False, patch_artist=True, showfliers=False, widths=0.7, whis=0.2)
            #ax.set_title(f'Boxplot for group: {group}')
            ax.set_title(f'PV_dist [{group}]-{calc_type}-{binary_PV_type}-{dist_type}-{shuffle_type}-{PV_use_B_1wk_type}-shuffle_norm-{shuffle_normalization}')
            ax.set_xlabel('Comparison Index')
            ax.set_ylabel('Normalized Data')
            ax.set_xticklabels(comparisons, rotation=45, ha='right')  # Set x-tick labels to comparison strings
            
            plt.tight_layout()
            plt.savefig(os.path.join(path_dir, f'boxplot_all_comps_{calc_type}_{dist_type}_shuffle_type_{shuffle_type}_shuffle_norm_{shuffle_normalization}_{PV_use_B_1wk_type}_{group}.png'), format='png', dpi=300)
            if auto_close:
                plt.close()

    stats_df = pd.DataFrame(flat_data)
    return stats_df

def calc_binned_PV(session, crossreg, mouse, group, mapping='TFC_cond+Test_B+Test_B_1wk', only_crossreg=True, bin_width=1, spk_cutoff=2, \
            want_binary_PV=False, Ca_act_type='full', sess_all=None, plot_it=False, auto_close=False, PLOTS_DIR=None, dir_name=''):

    s=session[mouse]
    crossreg_mouse=crossreg[mouse]
    [S_crossreg, S_spikes_crossreg, S_peakval, S_idx] = s.get_S_mapping(mapping, with_crossreg=crossreg_mouse) 
    S_i = get_S_indeces_crossreg(s, crossreg_mouse, mapping)
    if sess_all:
        s_B = sess_all[1][mouse]
        S_i_Test_B = get_S_indeces_crossreg(s_B, crossreg_mouse, mapping) 
        s_B_1wk = sess_all[2][mouse]
        S_i_Test_B_1wk = get_S_indeces_crossreg(s_B_1wk, crossreg_mouse, mapping)    

    if only_crossreg:
        if Ca_act_type == 'mov':
            C=s.S_mov[S_i,:]
            if sess_all:
                C_B=s_B.S_mov[S_i_Test_B,:]
                C_B_1wk=s_B_1wk.S_mov[S_i_Test_B_1wk,:]
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
        if PV_calc is None:
            continue
        curr_frame = 0
        upper = 1
        lower = 0
        #inc = 0
        for i in range(PV_calc.shape[1]):
            if want_binary_PV:
                PV_calc[:,i] = np.where(np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
            else:
                PV_calc[:,i] = np.sum(C_calc[:,curr_frame:curr_frame+bin_frames],1)/bin_frames
            curr_frame += bin_frames

    if plot_it:
        PV_plot = PV
        plt.figure(figsize=(8,6))
        plt.imshow(PV_plot)
        for i in s.shock_onsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=0.5)
        for i in s.tone_onsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=0.5) 
        for i in s.shock_offsets:
            plt.axvline(i/bin_frames, c='r', ls='-', lw=0.5)
        for i in s.tone_offsets:
            plt.axvline(i/bin_frames, c='b', ls='-', lw=0.5)
        plt.xlim((0, PV_plot.shape[1]))
        plt.ylim((0, PV_plot.shape[0]))
        xtick_seconds_PV = np.concatenate(([0], [s/bin_width for s in s.tone_onsets_def[0:len(s.tone_onsets)]], [np.min((1300/bin_width,PV_plot.shape[1]))]))
        xtick_seconds = np.concatenate(([0], s.tone_onsets_def[0:len(s.tone_onsets)], [1300]))
        plt.xticks(ticks=xtick_seconds_PV, labels=xtick_seconds)
        #plt.xticks(ticks=s.ts2frame(xtick_seconds*1000), labels=xtick_seconds) ## HERE and see new s.t2frame()
        if PLOTS_DIR:
            filename = '{}_{}_{}_{}{}_PV_binned.png'.format(Ca_act_type, group, mouse, transpose_str, session_str)
            os.makedirs(os.path.join(PLOTS_DIR, dir_name), exist_ok=True)
            plt.savefig(os.path.join(PLOTS_DIR, dir_name, filename), format='png', dpi=300)
        if auto_close:
            plt.close()

    tone_ranges = [(int(i), int(i)+look_ahead) for i in np.round([i/bin_frames for i in s.tone_onsets])]

    return PV_calcs

'''
PV_t = PV[:,tone_ranges[1][0]:tone_ranges[1][1]]
max_indices_along_columns = np.argmax(PV_t,axis=1)
sorted_row_indices = np.argsort(max_indices_along_columns)
PV_sorted = PV[sorted_row_indices[::-1]]
PV_plot = PV_sorted
'''

#
# Linear track plotting & analysis functions
#

def polyline_arclength(poly):
    seg = poly[1:] - poly[:-1]
    seglen = np.sqrt((seg ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seg, seglen

def project_points_to_polyline(points, poly):
    """
    points: (M,2)
    poly:   (N,2) ordered centerline
    Returns:
      s_proj: (M,) arc-length coordinate along polyline (pixels)
      d_perp: (M,) perpendicular distance to polyline (pixels)
      xy_proj:(M,2) projected points
      seg_id: (M,) segment used
    """
    points = np.asarray(points, float)
    poly = np.asarray(poly, float)

    s_nodes, seg, seglen = polyline_arclength(poly)
    a = poly[:-1]                 # (N-1,2)
    v = poly[1:] - poly[:-1]      # (N-1,2)
    vv = (v * v).sum(axis=1)      # (N-1,)

    M = points.shape[0]
    best_s = np.empty(M, float)
    best_d = np.empty(M, float)
    best_xy = np.empty((M, 2), float)
    best_seg = np.empty(M, int)

    for i, p in enumerate(points):
        w = p - a
        t = (w * v).sum(axis=1) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = a + (t[:, None] * v)
        d2 = ((proj - p) ** 2).sum(axis=1)
        j = int(np.argmin(d2))

        best_seg[i] = j
        best_xy[i] = proj[j]
        best_d[i] = np.sqrt(d2[j])
        best_s[i] = s_nodes[j] + t[j] * seglen[j]

    return best_s, best_d, best_xy, best_seg

def project_points_to_polyline_continuous(
    points,
    poly,
    *,
    seg_window=25,
    teleport_d_px=60.0,
    fallback_full_search=True,
):
    """
    Like project_points_to_polyline(), but enforces temporal continuity by restricting
    segment search to a window around the previous best segment.

    Args:
      points: (M,2)
      poly:   (N,2) ordered centerline
      seg_window: search +/- this many segments around previous seg_id
      teleport_d_px: if perp distance becomes huge, allow full-search fallback
      fallback_full_search: if True, do full search when continuity window fails

    Returns:
      s_proj:  (M,)
      d_perp:  (M,)
      xy_proj: (M,2)
      seg_id:  (M,)
    """
    points = np.asarray(points, float)
    poly = np.asarray(poly, float)

    # Precompute polyline arclengths
    seg = poly[1:] - poly[:-1]              # (N-1,2)
    seglen = np.sqrt((seg ** 2).sum(axis=1))# (N-1,)
    s_nodes = np.concatenate([[0.0], np.cumsum(seglen)])  # (N,)

    a = poly[:-1]                           # (N-1,2)
    v = seg                                 # (N-1,2)
    vv = (v * v).sum(axis=1)                # (N-1,)

    M = points.shape[0]
    best_s = np.empty(M, float)
    best_d = np.empty(M, float)
    best_xy = np.empty((M, 2), float)
    best_seg = np.empty(M, int)

    nseg = a.shape[0]

    prev_j = None
    for i, p in enumerate(points):
        # choose candidate segments
        if prev_j is None:
            cand = np.arange(nseg)
        else:
            lo = max(0, prev_j - seg_window)
            hi = min(nseg, prev_j + seg_window + 1)
            cand = np.arange(lo, hi)

        def eval_on_segments(cand_idx):
            aa = a[cand_idx]         # (K,2)
            vv_local = vv[cand_idx]  # (K,)
            v_local = v[cand_idx]    # (K,2)

            w = p - aa
            t = (w * v_local).sum(axis=1) / vv_local
            t = np.clip(t, 0.0, 1.0)
            proj = aa + (t[:, None] * v_local)
            d2 = ((proj - p) ** 2).sum(axis=1)
            k = int(np.argmin(d2))
            j = int(cand_idx[k])
            return j, t[k], proj[k], float(np.sqrt(d2[k]))

        j, tbest, xybest, dbest = eval_on_segments(cand)

        # if the continuity-window result is suspiciously far from the line, allow fallback
        if fallback_full_search and (dbest > teleport_d_px):
            j2, t2, xy2, d2 = eval_on_segments(np.arange(nseg))
            if d2 < dbest:
                j, tbest, xybest, dbest = j2, t2, xy2, d2

        best_seg[i] = j
        best_xy[i] = xybest
        best_d[i] = dbest
        best_s[i] = s_nodes[j] + tbest * seglen[j]

        prev_j = j

    return best_s, best_d, best_xy, best_seg

def plot_projection_sanity(
    mouse,
    session_str,
    xy,
    center,
    xy_proj,
    d_px,
    save_path=None,
    subsample=1,
    auto_close=True
):
    """
    Sanity-check plots for linearization.

    Produces TWO plots:
      1) XY trajectory + centerline + projection spokes
      2) XY trajectory color-coded by distance to centerline (d_px)

    Parameters
    ----------
    xy : (N,2) array
        Original (X,Y) positions.
    center : (M,2) array
        Centerline points.
    xy_proj : (N,2) array
        Projection of each xy onto the centerline.
    d_px : (N,) array
        Perpendicular distance to centerline (pixels).
    mouse : str
        Mouse identifier (e.g., "G05").
    session_str : str
        Session identifier (e.g., "TFC_cond_LT1").
    save_path : str or None
        Base directory where LT_projection/ will be created.
        If None, figures are not saved.
    subsample : int
        Plot every Nth point to reduce clutter.
    """

    title = f"{mouse}-{session_str}"

    # -----------------------------
    # Styling  (scoped so it does not leak into downstream plots)
    # -----------------------------
    _rc_overrides = {
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    _rc_ctx = plt.rc_context(_rc_overrides)
    _rc_ctx.__enter__()

    idx = np.arange(0, xy.shape[0], subsample)

    # ==========================================================
    # Plot 1: projection spokes
    # ==========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 12), dpi=300)

    ax1.plot(
        xy[idx, 0], xy[idx, 1],
        color="blue", alpha=0.3, linewidth=1.0
    )

    ax1.plot(
        center[:, 0], center[:, 1],
        color="red", linewidth=3.0
    )

    for i in idx:
        ax1.plot(
            [xy[i, 0], xy_proj[i, 0]],
            [xy[i, 1], xy_proj[i, 1]],
            color="red", alpha=0.2, linewidth=0.8
        )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_title(title)

    fig1.tight_layout()

    # ==========================================================
    # Plot 2: distance-to-centerline heatmap
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(12, 12), dpi=300)

    sc = ax2.scatter(
        xy[idx, 0], xy[idx, 1],
        c=d_px[idx],
        cmap="viridis",
        s=6,
        alpha=0.9
    )

    ax2.plot(
        center[:, 0], center[:, 1],
        color="red", linewidth=3.0
    )

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_title(title)

    cbar = fig2.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Distance to centerline (pixels)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig2.tight_layout()

    # ==========================================================
    # Save if requested
    # ==========================================================
    if save_path is not None:
        save_path = os.path.abspath(save_path)
        out_dir = os.path.join(save_path, "LT_projection")
        os.makedirs(out_dir, exist_ok=True)

        prefix = f"{mouse}-{session_str}"

        fig1_pdf = os.path.join(out_dir, f"{prefix}_projection_spokes.pdf")
        fig1_png = os.path.join(out_dir, f"{prefix}_projection_spokes.png")
        fig2_pdf = os.path.join(out_dir, f"{prefix}_projection_dist2center.pdf")
        fig2_png = os.path.join(out_dir, f"{prefix}_projection_dist2center.png")

        fig1.savefig(fig1_pdf, dpi=300, bbox_inches="tight")
        fig1.savefig(fig1_png, dpi=300, bbox_inches="tight")
        fig2.savefig(fig2_pdf, dpi=300, bbox_inches="tight")
        fig2.savefig(fig2_png, dpi=300, bbox_inches="tight")

        print("[OK] Saved projection sanity plots:")
        print(f"     {fig1_pdf}")
        print(f"     {fig1_png}")
        print(f"     {fig2_pdf}")
        print(f"     {fig2_png}")

    plt.show()
    if auto_close:
        plt.close(fig=fig1)
        plt.close(fig=fig2)

    _rc_ctx.__exit__(None, None, None)

def plot_LT_linearized(PLOTS_DIR, session, mouse_groups, session_str, auto_close=True):
    save_path = os.path.join(PLOTS_DIR, f'linearized_{session_str}')
    os.makedirs(save_path, exist_ok=True)
    for m, sess in session.items():
        print(f'*** WORKING LT LINEARIZATION FOR {m}... ')

        xy = np.column_stack([
            session[m].loc_X_behavcam_smooth,
            session[m].loc_Y_behavcam_smooth
        ])

        centerline_behavcam = session[m].centerline_behavcam
        #s_px, d_px, xy_proj, seg_id = project_points_to_polyline(xy, centerline_behavcam)
        s_px, d_px, xy_proj, seg_id = project_points_to_polyline_continuous(
            xy, centerline_behavcam,
            seg_window=25,
            teleport_d_px=60.0,
            fallback_full_search=True
        )        
        session[m].behavcam_loc_1d_px = s_px
        session[m].behavcam_loc_dist_to_center_px = d_px
        session[m].behavcam_loc_1d_proj = xy_proj

        plot_projection_sanity(
            m,
            'behavcam_' + session_str,
            xy,
            centerline_behavcam,
            xy_proj,
            d_px,
            save_path=save_path,
            subsample=5,  # recommended if N is large
            auto_close=auto_close
        )

        xy = np.column_stack([
            session[m].loc_X_miniscope_smooth,
            session[m].loc_Y_miniscope_smooth
        ])

        centerline_miniscope = session[m].centerline_miniscope
        #s_px, d_px, xy_proj, seg_id = project_points_to_polyline(xy, centerline_miniscope)
        s_px, d_px, xy_proj, seg_id = project_points_to_polyline_continuous(
            xy, centerline_miniscope,
            seg_window=25,
            teleport_d_px=60.0,
            fallback_full_search=True
        )        
        session[m].miniscope_loc_1d_px = s_px
        session[m].miniscope_loc_dist_to_center_px = d_px
        session[m].miniscope_loc_1d_proj = xy_proj

        plot_projection_sanity(
            m,
            'miniscope_' + session_str,
            xy,
            centerline_miniscope,
            xy_proj,
            d_px,
            save_path=save_path,
            subsample=5,  # recommended if N is large
            auto_close=auto_close
        )        


# ==============================================================================
#  Occupancy / trajectory / immobility analysis
# ==============================================================================

class OccupancyAnalysis:
    """Compute 2D occupancy heatmap and trajectory for any session.

    Works for both LT and TFC/Test sessions.  For TFC-family sessions
    the analysis is restricted to the first *first_n_sec* seconds (default
    180 s, i.e. before the first tone).

    Parameters
    ----------
    sess : BehaviourSession
        Any session object that has ``loc_X_miniscope_smooth``,
        ``loc_Y_miniscope_smooth``, ``velocities_miniscope_smooth``,
        and ``tstamp_miniscope``.
    mouse : str
        Mouse identifier (e.g. ``'G10'``).
    group : str
        Experimental group (``'hM3D'``, ``'hM4D'``, ``'mCherry'``).
    session_label : str
        Human-readable session name for titles / filenames.
    bin_width : float
        Spatial bin width in the same units as X/Y (pixels).  Default 10.
    first_n_sec : float or None
        If not None, restrict to the first *first_n_sec* seconds of the
        session.  Set to None to use the entire session (typical for LT).
    speed_thresh : float
        Speed threshold in cm/s for immobility (default 2.0).
    """

    def __init__(self, sess, mouse: str, group: str, session_label: str, *,
                 bin_width: float = 10.0, first_n_sec: float = None,
                 speed_thresh: float = 2.0):
        self.mouse = mouse
        self.group = group
        self.session_label = session_label
        self.bin_width = bin_width
        self.first_n_sec = first_n_sec
        self.speed_thresh = speed_thresh

        x = np.asarray(sess.loc_X_miniscope_smooth, dtype=float)
        y = np.asarray(sess.loc_Y_miniscope_smooth, dtype=float)
        vel = np.asarray(sess.velocities_miniscope_smooth, dtype=float)
        ts = np.asarray(sess.tstamp_miniscope, dtype=float)  # ms

        # Restrict to first_n_sec if requested
        if first_n_sec is not None:
            n_frames = int(np.searchsorted(ts, first_n_sec * 1000.0))
            n_frames = max(1, min(n_frames, len(x)))
        else:
            n_frames = len(x)

        self.x = x[:n_frames]
        self.y = y[:n_frames]
        self.vel = vel[:n_frames]
        self.ts = ts[:n_frames]
        self.n_frames = n_frames
        self.dt_sec = MINISCOPE_FRAME_MS / 1000.0  # 0.05 s per frame

        # Occupancy heatmap (time in seconds per bin)
        self.nx = max(1, int(np.ceil(np.nanmax(self.x) / bin_width)))
        self.ny = max(1, int(np.ceil(np.nanmax(self.y) / bin_width)))
        self.occ = np.zeros((self.ny, self.nx), dtype=float)
        bx = np.clip((self.x / bin_width).astype(int), 0, self.nx - 1)
        by = np.clip((self.y / bin_width).astype(int), 0, self.ny - 1)
        for i in range(n_frames):
            self.occ[by[i], bx[i]] += self.dt_sec

        # Total duration analysed (seconds)
        self.total_sec = n_frames * self.dt_sec

        # ---- Coverage metrics ----
        visited = self.occ > 0
        total_bins = self.nx * self.ny
        n_visited = int(np.sum(visited))
        self.coverage_pct = 100.0 * n_visited / total_bins
        self.n_visited_bins = n_visited
        self.total_bins = total_bins

        # Mean occupancy density (seconds per visited bin)
        self.mean_occ_density = (np.mean(self.occ[visited])
                                 if n_visited > 0 else 0.0)
        # Spatial entropy of normalised occupancy (bits)
        occ_flat = self.occ.ravel()
        p = occ_flat / occ_flat.sum() if occ_flat.sum() > 0 else occ_flat
        p_pos = p[p > 0]
        self.spatial_entropy = float(-np.sum(p_pos * np.log2(p_pos)))
        max_entropy = np.log2(total_bins) if total_bins > 0 else 1.0
        self.norm_entropy = self.spatial_entropy / max_entropy if max_entropy > 0 else 0.0

        # ---- Immobility score ----
        immobile_frames = np.sum(self.vel < speed_thresh)
        self.immobility_pct = 100.0 * immobile_frames / n_frames if n_frames > 0 else 0.0

    # convenience ----------------------------------------------------------
    def summary_dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "group": self.group,
            "session": self.session_label,
            "coverage_pct": self.coverage_pct,
            "n_visited_bins": self.n_visited_bins,
            "total_bins": self.total_bins,
            "mean_occ_density_sec": self.mean_occ_density,
            "spatial_entropy_bits": self.spatial_entropy,
            "norm_entropy": self.norm_entropy,
            "immobility_pct": self.immobility_pct,
            "total_sec": self.total_sec,
        }


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

# Nature-style rc params shared by all occupancy plots
_NATURE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
}
_MM_TO_IN = 1.0 / 25.4
_SINGLE_COL = 88 * _MM_TO_IN   # 88 mm → inches
_HALF_COL = 44 * _MM_TO_IN     # ~half column


def _save_nature(fig, save_dir, stem, dpi=300):
    """Save PNG + PDF."""
    fig.savefig(os.path.join(save_dir, f"{stem}.png"),
                dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(save_dir, f"{stem}.pdf"),
                dpi=dpi, bbox_inches='tight', pad_inches=0.02)


def plot_occupancy_trajectory(occ: OccupancyAnalysis, save_dir: str, *,
                              auto_close: bool = True):
    """Plot the raw XY trajectory as a coloured curve (time → colour)."""
    os.makedirs(save_dir, exist_ok=True)
    with mpl.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(_HALF_COL, _HALF_COL * 0.85), dpi=200)
        t_norm = np.linspace(0, 1, len(occ.x))
        sc = ax.scatter(occ.x, occ.y, c=t_norm, cmap="viridis", s=0.3,
                        alpha=0.5, rasterized=True)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        dur_str = (f"first {occ.first_n_sec:.0f}s"
                   if occ.first_n_sec is not None else "full")
        ax.set_title(f"{occ.mouse} ({occ.group}) {occ.session_label} ({dur_str})",
                     fontsize=7)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="Time", fraction=0.046, pad=0.04)
        fig.tight_layout()
        stem = f"{occ.mouse}_{occ.group}_{occ.session_label}_trajectory"
        _save_nature(fig, save_dir, stem)
    if auto_close:
        plt.close(fig)
    return fig


def plot_occupancy_heatmap(occ: OccupancyAnalysis, save_dir: str, *,
                           auto_close: bool = True):
    """Plot the binned 2D occupancy heatmap (seconds per bin)."""
    os.makedirs(save_dir, exist_ok=True)
    with mpl.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(_HALF_COL, _HALF_COL * 0.85), dpi=200)
        im = ax.imshow(occ.occ, origin="upper", cmap="hot", aspect="equal",
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Occ. (s)", fraction=0.046, pad=0.04)
        dur_str = (f"first {occ.first_n_sec:.0f}s"
                   if occ.first_n_sec is not None else "full")
        ax.set_title(f"{occ.mouse} ({occ.group}) {occ.session_label} ({dur_str})",
                     fontsize=7)
        ax.set_xlabel("X bin")
        ax.set_ylabel("Y bin")
        fig.tight_layout()
        stem = f"{occ.mouse}_{occ.group}_{occ.session_label}_occupancy_heatmap"
        _save_nature(fig, save_dir, stem)
    if auto_close:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
#  Pipeline: compute occupancy for a dict of sessions
# ---------------------------------------------------------------------------

def compute_occupancy_for_sessions(sessions: dict, mouse_groups: dict,
                                   session_label: str, *,
                                   bin_width: float = 10.0,
                                   first_n_sec: float = None,
                                   speed_thresh: float = 2.0) -> dict:
    """Return {mouse: OccupancyAnalysis} for every mouse in *sessions*."""
    results = {}
    for m, sess in sessions.items():
        group = mouse_groups.get(m, "NA")
        results[m] = OccupancyAnalysis(
            sess, m, group, session_label,
            bin_width=bin_width, first_n_sec=first_n_sec,
            speed_thresh=speed_thresh)
    return results


def plot_all_occupancy(occ_dict: dict, save_dir: str, *, auto_close: bool = True):
    """Plot trajectories + heatmaps for every mouse in *occ_dict*."""
    traj_dir = os.path.join(save_dir, "trajectories")
    heat_dir = os.path.join(save_dir, "heatmaps")
    for m, occ in occ_dict.items():
        plot_occupancy_trajectory(occ, traj_dir, auto_close=auto_close)
        plot_occupancy_heatmap(occ, heat_dir, auto_close=auto_close)


# ---------------------------------------------------------------------------
#  Group-level statistics
# ---------------------------------------------------------------------------

_GROUP_ORDER = ["mCherry", "hM3D", "hM4D"]
_GROUP_NICE  = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}
# Light fills for boxplots, darker shades for scatter points
_GROUP_COLORS_BOX  = {"mCherry": "#c8c8c8", "hM3D": "#f4b8b8", "hM4D": "#b8d4f0"}
_GROUP_COLORS_DOT  = {"mCherry": "#666666", "hM3D": "#cc4444", "hM4D": "#3a7ec0"}


def _build_occupancy_dataframe(occ_dicts: dict) -> pd.DataFrame:
    """Concatenate summary dicts from multiple session→occ_dict mappings.

    Parameters
    ----------
    occ_dicts : dict[str, dict[str, OccupancyAnalysis]]
        ``{session_label: {mouse: OccupancyAnalysis, …}, …}``
    """
    rows = []
    for sess_label, od in occ_dicts.items():
        for m, occ in od.items():
            rows.append(occ.summary_dict())
    return pd.DataFrame(rows)


def run_occupancy_group_stats(df: pd.DataFrame, save_dir: str, *,
                              auto_close: bool = True):
    """For each metric × session, run Kruskal–Wallis + Dunn post-hoc.

    Saves CSV tables and produces grouped bar + strip plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("coverage_pct",        "Coverage (%)"),
        ("norm_entropy",        "Normalised entropy"),
        ("mean_occ_density_sec","Mean occ. density (s/bin)"),
        ("immobility_pct",      "Immobility (%)"),
    ]

    sessions = df["session"].unique()
    all_stats_rows = []

    for metric_col, metric_nice in metrics:
        for sess in sessions:
            sub = df[df["session"] == sess].copy()
            groups_present = [g for g in _GROUP_ORDER if g in sub["group"].values]
            if len(groups_present) < 2:
                continue

            group_vals = [sub.loc[sub["group"] == g, metric_col].values
                          for g in groups_present]

            # Kruskal–Wallis
            if all(len(v) >= 1 for v in group_vals):
                kw_stat, kw_p = stats.kruskal(*group_vals)
            else:
                kw_stat, kw_p = np.nan, np.nan

            row_base = {"metric": metric_col, "session": sess,
                        "KW_stat": kw_stat, "KW_p": kw_p}

            # Pairwise Mann–Whitney U with Holm correction
            pairs = list(itertools.combinations(groups_present, 2))
            raw_ps = []
            pair_stats = []
            for g1, g2 in pairs:
                v1 = sub.loc[sub["group"] == g1, metric_col].values
                v2 = sub.loc[sub["group"] == g2, metric_col].values
                if len(v1) >= 1 and len(v2) >= 1:
                    u_stat, u_p = stats.mannwhitneyu(v1, v2,
                                                     alternative="two-sided")
                else:
                    u_stat, u_p = np.nan, np.nan
                raw_ps.append(u_p)
                pair_stats.append((g1, g2, u_stat, u_p))

            # Holm correction
            valid_mask = ~np.isnan(raw_ps)
            corrected = np.full(len(raw_ps), np.nan)
            if np.any(valid_mask):
                _, corr_vals, _, _ = multipletests(
                    np.array(raw_ps)[valid_mask], method="holm")
                corrected[valid_mask] = corr_vals

            for idx, (g1, g2, u_stat, u_p) in enumerate(pair_stats):
                r = dict(row_base)
                r.update({"pair": f"{_GROUP_NICE[g1]} vs {_GROUP_NICE[g2]}",
                          "U_stat": u_stat, "p_raw": u_p,
                          "p_holm": corrected[idx]})
                all_stats_rows.append(r)

    df_stats = pd.DataFrame(all_stats_rows)
    df_stats.to_csv(os.path.join(save_dir, "occupancy_stats.csv"), index=False)

    # ---- Combined figure: all sessions on x-axis per metric ----
    with mpl.rc_context(_NATURE_RC):
        for metric_col, metric_nice in metrics:
            n_sess = len(sessions)
            # ~12 mm per session cluster, min single-column width
            fig_w = max(_SINGLE_COL, n_sess * 12 * _MM_TO_IN)
            fig_h = _SINGLE_COL * 0.55
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
            sub = df[["session", "group", metric_col, "mouse"]].copy()
            sub["group_nice"] = sub["group"].map(_GROUP_NICE)

            sess_order = list(sessions)
            group_nice_order = [_GROUP_NICE[g] for g in _GROUP_ORDER
                                if g in sub["group"].values]
            pal_box = {_GROUP_NICE[g]: _GROUP_COLORS_BOX[g] for g in _GROUP_ORDER}
            pal_dot = {_GROUP_NICE[g]: _GROUP_COLORS_DOT[g] for g in _GROUP_ORDER}

            bp = sns.boxplot(data=sub, x="session", y=metric_col, hue="group_nice",
                        order=sess_order, hue_order=group_nice_order,
                        palette=pal_box, ax=ax, showfliers=False, width=0.65,
                        linewidth=0.6, fliersize=0,
                        medianprops=dict(color='black', linewidth=0.8))
            sns.stripplot(data=sub, x="session", y=metric_col, hue="group_nice",
                          order=sess_order, hue_order=group_nice_order,
                          palette=pal_dot, ax=ax, dodge=True, size=2.5,
                          alpha=0.85, edgecolor="k", linewidth=0.3)

            handles, labels = ax.get_legend_handles_labels()
            n_groups = len(group_nice_order)
            ax.legend(handles[:n_groups], labels[:n_groups],
                      loc="upper right", fontsize=6, frameon=False)

            ax.set_ylabel(metric_nice)
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=30)

            # Annotate KW p-values
            for i, sess in enumerate(sess_order):
                kw_rows = df_stats[(df_stats["metric"] == metric_col) &
                                   (df_stats["session"] == sess)]
                if len(kw_rows) > 0:
                    kw_p = kw_rows.iloc[0]["KW_p"]
                    star = _p_to_stars(kw_p)
                    if star:
                        ymax = sub.loc[sub["session"] == sess, metric_col].max()
                        ax.text(i, ymax * 1.05, star, ha="center",
                                va="bottom", fontsize=7, fontweight="bold")

            fig.tight_layout()
            stem = f"occupancy_{metric_col}_boxplot"
            _save_nature(fig, save_dir, stem)
            if auto_close:
                plt.close(fig)

    # ---- Per-session box+strip figures (one figure per session×metric) ----
    with mpl.rc_context(_NATURE_RC):
        per_sess_dir = os.path.join(save_dir, "per_session")
        os.makedirs(per_sess_dir, exist_ok=True)
        for metric_col, metric_nice in metrics:
            for sess in sessions:
                sub = df[df["session"] == sess].copy()
                sub["group_nice"] = sub["group"].map(_GROUP_NICE)
                groups_present = [_GROUP_NICE[g] for g in _GROUP_ORDER
                                  if g in sub["group"].values]
                pal_box = {_GROUP_NICE[g]: _GROUP_COLORS_BOX[g]
                           for g in _GROUP_ORDER}
                pal_dot = {_GROUP_NICE[g]: _GROUP_COLORS_DOT[g]
                           for g in _GROUP_ORDER}
                fig, ax = plt.subplots(
                    figsize=(_HALF_COL, _HALF_COL * 0.8), dpi=200)
                sns.boxplot(data=sub, x="group_nice", y=metric_col,
                            order=groups_present, palette=pal_box, ax=ax,
                            showfliers=False, width=0.5, linewidth=0.6,
                            medianprops=dict(color='black', linewidth=0.8))
                sns.stripplot(data=sub, x="group_nice", y=metric_col,
                              order=groups_present, palette=pal_dot, ax=ax,
                              size=3, alpha=0.85, edgecolor="k",
                              linewidth=0.3, jitter=0.15)
                ax.set_ylabel(metric_nice)
                ax.set_xlabel("")
                ax.set_title(sess, fontsize=7)

                # Annotate pairwise significance
                pairs_df = df_stats[(df_stats["metric"] == metric_col) &
                                    (df_stats["session"] == sess)]
                ymax = sub[metric_col].max()
                dy = (sub[metric_col].max() - sub[metric_col].min()) * 0.08
                offset = 0
                for _, pr in pairs_df.iterrows():
                    star = _p_to_stars(pr["p_holm"])
                    if star:
                        g1n, g2n = pr["pair"].split(" vs ")
                        if g1n in groups_present and g2n in groups_present:
                            x1 = groups_present.index(g1n)
                            x2 = groups_present.index(g2n)
                            y_bar = ymax + dy * (1.2 + offset)
                            ax.plot([x1, x1, x2, x2],
                                    [y_bar - dy * 0.15, y_bar, y_bar,
                                     y_bar - dy * 0.15],
                                    lw=0.6, color='k')
                            ax.text((x1 + x2) / 2, y_bar, star,
                                    ha='center', va='bottom', fontsize=6)
                            offset += 1.4

                fig.tight_layout()
                stem = f"occupancy_{metric_col}_{sess}"
                _save_nature(fig, per_sess_dir, stem)
                if auto_close:
                    plt.close(fig)

    return df_stats


def _p_to_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def run_occupancy_analysis_pipeline(
        session_dicts: dict,
        mouse_groups: dict,
        PLOTS_DIR: str,
        *,
        bin_width: float = 10.0,
        first_n_sec_tfc: float = 180.0,
        first_n_sec_lt: float = None,
        speed_thresh: float = 2.0,
        auto_close: bool = True):
    """Full pipeline: compute occupancy, plot per-mouse, group stats.

    Parameters
    ----------
    session_dicts : dict[str, dict[str, BehaviourSession]]
        ``{"TFC_cond": {mouse: sess, …}, "Test_B": {…}, …}``
        Keys are used as session labels.
    mouse_groups : dict[str, str]
        ``{mouse: group}``
    PLOTS_DIR : str
        Root plot directory.  A sub-folder ``occupancy_analysis`` is created.
    bin_width : float
        Occupancy heatmap bin width (pixels).
    first_n_sec_tfc : float or None
        Time restriction for TFC-family sessions (default 180 s).
    first_n_sec_lt : float or None
        Time restriction for LT sessions (default None = full session).
    speed_thresh : float
        Immobility threshold in cm/s.
    """
    base_dir = os.path.join(PLOTS_DIR, "occupancy_analysis")
    os.makedirs(base_dir, exist_ok=True)

    lt_labels = {"LT1", "LT2", "TFC_cond_LT1", "TFC_cond_LT2"}
    occ_all = {}   # session_label → {mouse: OccupancyAnalysis}

    for sess_label, sess_dict in session_dicts.items():
        if not sess_dict:
            continue
        is_lt = sess_label in lt_labels
        fn_sec = first_n_sec_lt if is_lt else first_n_sec_tfc

        occ_dict = compute_occupancy_for_sessions(
            sess_dict, mouse_groups, sess_label,
            bin_width=bin_width, first_n_sec=fn_sec,
            speed_thresh=speed_thresh)
        occ_all[sess_label] = occ_dict

        # Per-mouse plots
        sess_dir = os.path.join(base_dir, sess_label)
        plot_all_occupancy(occ_dict, sess_dir, auto_close=auto_close)

    # Build combined dataframe and run group stats
    df = _build_occupancy_dataframe(occ_all)
    df.to_csv(os.path.join(base_dir, "occupancy_summary.csv"), index=False)

    df_stats = run_occupancy_group_stats(df, base_dir, auto_close=auto_close)

    print(f"[Occupancy] Summary CSV : {os.path.join(base_dir, 'occupancy_summary.csv')}")
    print(f"[Occupancy] Stats CSV   : {os.path.join(base_dir, 'occupancy_stats.csv')}")

    return df, df_stats, occ_all
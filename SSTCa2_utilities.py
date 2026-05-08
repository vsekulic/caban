import os
import zarr
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from numpy import linalg
import pickle
import time
from sklearn.covariance import LedoitWolf
import random

MINISCOPE_FPS = 20
BEHAVCAM_FPS = 15
VELOCITY_THRESHOLD = 2.0 # cm/s
SMOOTH_SIGMA = 4 # for Gaussian smoothing of velocities in SSTCa2_sessions.py
SMOOTH_LOC_SIGMA = 4 # for smoothing of spatial bins in plot_fluorescence_maps(). 4 bins is 8 cm as per Fournier et al 2020. (given 2cm bins)
MINISCOPE_FRAME_MS = 1000 / MINISCOPE_FPS
MAIN_DRIVE = 'D:'
NPY_SAVE_PATH = MAIN_DRIVE+'\\data\\vsekulic\\OF_test\\npy_files'

my_colours = {
    'my_b' : (0, 0.2274, 0.8196),
    'my_r' : (0.815, 0, 0.2),
    'my_k' : (0, 0, 0),
    'my_h' : (0.3, 0.3, 0.3)
}

my_colours = {
    'my_b': (0, 0.2274, 0.8196),
    'my_b_dark': (0, 0.15, 0.6),
    
    'my_r': (0.815, 0, 0.2),
    'my_r_dark': (0.6, 0, 0.15),
    
    'my_k': (0, 0, 0),
    
    'my_h': (0.3, 0.3, 0.3),
    'my_h_dark': (0.2, 0.2, 0.2)
}
class Saver:
    def __init__(self, subdirs=[], parent_path=NPY_SAVE_PATH, prefix=''):
        self.parent_path = parent_path
        self.subdirs = subdirs
        self.prefix = prefix

        save_path = parent_path
        for p in subdirs:
            save_path = os.path.join(save_path, p)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def save(self, X, name):
        save_file = self.generate_path(name)
        if type(X) is pd.core.frame.DataFrame:
            X.to_pickle(self.as_DataFrame(save_file))
        else:
            with open(save_file, 'wb') as f_pickle:
                pickle.dump(X, f_pickle)
    
    def load(self, name):
        save_file = self.generate_path(name)
        if self.check_exists(name, type='DataFrame'):
            X = pd.read_pickle(self.as_DataFrame(save_file))
        else:
            with open(save_file, 'rb') as f_pickle:
                X = pickle.load(f_pickle)
        return X

    def as_DataFrame(self, save_file):
        return save_file.replace('.pkl', '_DataFrame.pkl')

    def check_exists(self, name, type=None):
        save_file = self.generate_path(name)
        if type is None:
            if os.path.exists(save_file) or os.path.exists(self.as_DataFrame(save_file)):
                return True
            else:
                return False
        else:
            if not type:
                if os.path.exists(save_file):
                    return True
                else:
                    return False
            if type == 'DataFrame':
                if os.path.exists(self.as_DataFrame(save_file)):
                    return True
                else:
                    return False
        
    def generate_path(self, name):
        # avoid unnecessary underline if no prefix provided
        if self.prefix:
            prefix_str = '_'
        else:
            prefix_str = ''
        return os.path.join(self.save_path, '{}{}{}.pkl'.format(self.prefix, prefix_str, name))

class Timer:
    def __init__(self):
        pass
    def start(self):
        self.start_time = time.time()
    def end(self, print_it=True, return_time=False):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.elapsed_time = elapsed_time
    
        if print_it:
            self.print()
        if return_time:
            return self.elapsed_time
        else:
            return self.print_msg
    
    def print(self):
        hours, rem = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        self.print_msg = f"*** Elapsed time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}"
        print(self.print_msg)
        return self.print_msg

def get_mappings_crossreg1(dpath):

    mappings_file = os.path.join(dpath, 'mappings_crossreg1.csv')
    with open(mappings_file) as f:
        df = pd.read_csv(mappings_file)
    return df

def get_actual_cells_from_df_session(df_col):
    '''
    df_col is a Pandas Series object, so must pass df['col_name']. Peels the onion to return a list of actual cell integers.
    E.g., c_T_in_L1L2T = get_actual_cells_from_df_session(df_mapping_all['session.2'])
    '''
    cells = []
    col_list = df_col.values.tolist()
    for val in col_list:

        # Convert the cell number (which had a .0 added to it) back to a float, then to an int.
        # Have to do it this way since calling int() on a string representation of a float only 
        # throws an error in python now (apparent python 3 change).
        cells.append(int(float(val)))
    #return sorted(cells) # OMG NO!!! Never sort this!!! Otherwise lose cross-reg assignments!!!
    return cells

def find_spikes_ca_S(S, thres, want_peakval=False):
    '''
    Batch processing of spike detection of S (output of minian). Returns a dict of indices into S mapped onto
    the list output of find_spikes_ca().
    '''
    num_rows = S.shape[0]
    frameidx_d = dict()
    peakval_d = dict()
    for i in range(num_rows):
        if want_peakval:
            [frameidx, peakval] = find_spikes_ca(S[i,:], thres, want_peakval=want_peakval)
            peakval_d[i] = peakval
        else:
            frameidx = find_spikes_ca(S[i,:], want_peakval)
        frameidx_d[i] = frameidx
    if want_peakval:
        return [frameidx_d, peakval_d]
    else:
        return frameidx_d

def find_spikes_ca(trace, thres, plotit=False, want_peakval=False):
    '''
    trace is a single row from S (as numpy ndarray, output of minian).
    thres is the arbitrary y-value unit from S.

    Returns [frameidx, peakval]. 
    '''
    peakst = np.where(trace >= thres)[0] # np.where returns "true" values in first element of 2-tuple
    peaksv = np.ones(len(trace)) * thres
    
    peaksv[peakst] = trace[peakst]
    peaksdy = np.diff(peaksv)
    
    # post-process peaksdy so that dy=0 points are removed, since this then prevents 
    # certain spikes from not being detected. So, just replace dy=0 points with the 
    # previous data point value (or next one, if at the beginning of peaksdy).
    for i in range(1, len(peaksdy)):
        if peaksdy[i] == 0:
            if i == 1:
                peaksdy[i] = peaksdy[i+1]
            else:
                peaksdy[i] = peaksdy[i-1]

    frameidx = np.where((np.append(peaksdy, 0) < 0) & (np.append(0, peaksdy) > 0))[0]
    peakval = np.transpose(peaksv[frameidx])

    if plotit:
        fig = plt.figure()
        plt.plot(trace, c='k')
        plt.plot(peaksv, c='b')
        plt.plot(frameidx, peakval, c='r', marker='o', linestyle='None');
    
    # Really ugly; reconsider...
    if plotit:
        if want_peakval:
            return [frameidx, peakval, fig]
        else:
            return [frameidx, fig]
    else:
        if want_peakval:
            return [frameidx, peakval]
        else:
            return frameidx

def get_spikes_in_period(frameidx, period):
    '''
    Return subset of frames (spikes) that lie within period. frameidx is the output of the two find_spikes_ca() or
    find_spikes_ca_S() functions, that is, either a list or dict, respectively. period is a 2-tuple corresponding to
    start and end frames, inclusive, over which to search the spike frameidx data.
    '''
    if type(frameidx) is list or isinstance(frameidx, np.ndarray):
        return np.where((frameidx >= period[0]) & (frameidx <= period[1]))[0]
    elif type(frameidx) is dict:
        frameidx_d = dict()
        for k in frameidx.keys():
            frameidx_d[k] = np.where((frameidx[k] >= period[0]) & (frameidx[k] <= period[1]))[0]
        return frameidx_d

def get_avg_sp_rate_in_period(f_spikes, beg_period, end_period):
    '''
    f_spikes is dict() of cell to frames returned by find_spikes_ca_S().

    beg_period, end_period are lists of the same length for multiple periods for which to calculate average firing rates
    for all cells in f_spikes.

    Result is returned by avg_sp_rate_tone list, of same length as beg/end_period, corresponding to the average firing rate
    of the population within each specified period.
    '''
    avg_sp_rate_period = []
    for num in range(len(beg_period)):
        avg_sp_rate = 0
        f_spikes_this_period = get_spikes_in_period(f_spikes, [beg_period[num], end_period[num]])
        period_length = (end_period[num] - beg_period[num]) / MINISCOPE_FPS
        for cell in f_spikes.keys():
            avg_sp_rate += (len(f_spikes_this_period[cell]) / period_length) # averaged over spikes in period per cell
        avg_sp_rate = avg_sp_rate / len(f_spikes_this_period) # averaged over all cells
        #avg_sp_rate = avg_sp_rate / period_length
        avg_sp_rate_period.append(avg_sp_rate)
    return avg_sp_rate_period

def get_avg_activity_in_period(S_spikes, S_peakval, beg_period, end_period):
    '''
    Similar to get_avg_sp_rate_in_period() above but use peak deconvolved spike values not just spike times.
    '''
    avg_activity_period = []
    for num in range(len(beg_period)):
        avg_activity_rate = 0
        spikes_this_period = get_spikes_in_period(S_spikes, [beg_period[num], end_period[num]])
        period_length = (end_period[num] - beg_period[num]) / MINISCOPE_FPS
        for cell in S_spikes.keys():
            avg_activity_rate += (sum(S_peakval[cell][spikes_this_period[cell]]) / period_length)
        avg_activity_rate = avg_activity_rate / len(spikes_this_period) # averaged over all cells
        #avg_sp_rate = avg_sp_rate / period_length
        avg_activity_period.append(avg_activity_rate)
    return avg_activity_period

def _engram_per_cell_score(S, S_spikes, S_peakval, use_peakval=True):
    """Per-cell summary statistic used to classify engram cells.

    Implements the Mocle et al. 2024 / Frankland convention: each cell's
    *average transient rate* (detected calcium transients per second
    over the full session). Returns a 1-D ndarray of length
    ``S.shape[0]``.

    The ``use_peakval`` / ``S_peakval`` arguments are retained for
    signature back-compat and are ignored — classification is based on
    transient counts only, regardless of amplitude.
    """
    n_cells, n_frames = S.shape
    if n_frames <= 0:
        raise RuntimeError(
            f"S has zero time samples (shape={S.shape}); cannot compute "
            "transient rate."
        )
    duration_s = n_frames / float(MINISCOPE_FPS)
    rates = np.zeros(n_cells, dtype=float)
    for cell in range(n_cells):
        if cell not in S_spikes:
            raise RuntimeError(
                f"S_spikes missing entry for cell {cell} (n_cells={n_cells})."
            )
        rates[cell] = len(S_spikes[cell]) / duration_s
    return rates


def get_engram_cells(S, S_spikes, S_peakval, use_peakval=True, zscore_thresh=0,
                     want_low_activity=False, ext_norm=None):
    '''
    Using Frankland lab method of classifying 'engram' cells in calcium imaging data (Mocle et al. 2024). Viz., calculate
    average transient rates of all neurons and perform z-score on this distribution, then assign all neurons with z-score
    greater than zero to be 'engram' cells. We return a numpy array of length S.shape[0] (number of cells) to use as an 
    index into the S and C arrays for subsequent analysis.

    The function is made to be called internally by various mapping functions, so we don't do any mapping, and assume that
    the passed ndarray is already corresponding to a desired mapping.

    Parameters
    ----------
    ext_norm : None | tuple
        Optional external classification rule, used to make the threshold
        comparable across groups when overall activity differs (e.g.
        chemogenetic experiments). Recognised forms:
          - ``('zscore', mu, sigma)`` : score = (x - mu) / sigma, with
            mu/sigma typically derived from a pooled control group.
            Threshold is still ``zscore_thresh``.
          - ``('absolute', cutoff)``  : score = x (raw per-cell score),
            with engram = (x > cutoff). Useful for percentile cutoffs
            derived from a control pool.
        ``None`` (default) reproduces the original per-mouse z-score
        behaviour.
    '''
    score_raw = _engram_per_cell_score(S, S_spikes, S_peakval,
                                       use_peakval=use_peakval)
    if ext_norm is None:
        score = sp.stats.zscore(score_raw)
        cutoff = zscore_thresh
    else:
        kind = ext_norm[0]
        if kind == "zscore":
            _, mu, sigma = ext_norm
            if sigma <= 0:
                raise RuntimeError(
                    f"ext_norm sigma must be positive (got {sigma})."
                )
            score = (score_raw - float(mu)) / float(sigma)
            cutoff = zscore_thresh
        elif kind == "absolute":
            _, cutoff_val = ext_norm
            score = score_raw
            cutoff = float(cutoff_val)
        else:
            raise ValueError(f"Unknown ext_norm kind {kind!r}")
    if want_low_activity == True:
        S_mask = score < cutoff
    else:
        S_mask = score > cutoff
    S_filtered = S[S_mask, :]
    S_spikes_filtered = {k:v for k,v,mask in zip(S_spikes.keys(), S_spikes.values(), S_mask) if mask}
    S_peakval_filtered = {k:v for k,v,mask in zip(S_peakval.keys(), S_peakval.values(), S_mask) if mask}
    S_indices = list(S_spikes_filtered.keys())
    return S_mask, S_indices, S_filtered, S_spikes_filtered, S_peakval_filtered

def get_engram_crossreg(mouse, TFC_cond, Test_B, Test_B_1wk, crossreg, mapping, engram_thresh=0, engram_type='encoding', want_low_activity=False, ext_norm=None):
    [S_TFC_cond, S_spikes_TFC_cond, S_peakval_TFC_cond, S_idx_TFC_cond] = \
        TFC_cond[mouse].get_S_mapping(mapping, want_peakval=True, with_crossreg=crossreg[mouse])
    [S_Test_B, S_spikes_Test_B, S_peakval_Test_B, S_idx_Test_B] = \
        Test_B[mouse].get_S_mapping(mapping, want_peakval=True, with_crossreg=crossreg[mouse])
    [S_Test_B_1wk, S_spikes_Test_B_1wk, S_peakval_Test_B_1wk, S_idx_Test_B_1wk] = \
        Test_B_1wk[mouse].get_S_mapping(mapping, want_peakval=True, with_crossreg=crossreg[mouse])

    S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[mouse], crossreg[mouse], mapping)
    S_i_Test_B = get_S_indeces_crossreg(Test_B[mouse], crossreg[mouse], mapping)
    S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[mouse], crossreg[mouse], mapping)    

    if engram_type == 'encoding':
        S_TFC_cond_engram_mask, S_engram_indeces, S_TFC_cond_engram, S_TFC_cond_engram_spikes, S_TFC_cond_engram_peakval = \
            get_engram_cells(S_TFC_cond, S_spikes_TFC_cond, S_peakval_TFC_cond, zscore_thresh=engram_thresh, want_low_activity=want_low_activity, ext_norm=ext_norm)
        S_i = S_i_TFC_cond
    if engram_type == 'recall':
        S_Test_B_engram_mask, S_engram_indeces, S_Test_B_engram, S_Test_B_engram_spikes, S_Test_B_engram_peakval = \
            get_engram_cells(S_Test_B, S_spikes_Test_B, S_peakval_Test_B, zscore_thresh=engram_thresh, want_low_activity=want_low_activity, ext_norm=ext_norm)
        S_i = S_i_Test_B

    engram_indeces_into_S = np.intersect1d(S_engram_indeces, S_i_TFC_cond)
    S_i_engram = np.where(np.isin(S_i, engram_indeces_into_S))[0]
    S_i_engram_TFC_cond = np.array(S_i_TFC_cond)[S_i_engram]
    S_i_engram_Test_B = np.array(S_i_Test_B)[S_i_engram]
    S_i_engram_Test_B_1wk = np.array(S_i_Test_B_1wk)[S_i_engram]

    return [S_i_engram_TFC_cond, S_i_engram_Test_B, S_i_engram_Test_B_1wk]

def get_S_indeces_crossreg(session, crossreg, mapping, want_peakval=True):
    [S, S_spikes, S_peakval, S_idx] = session.get_S_mapping(mapping, want_peakval=want_peakval, with_crossreg=crossreg)
    indeces_into_S = session.get_S_indeces(S_idx)
    return indeces_into_S

def msg_start(str):
    print(str+'...', end='', flush=True)
def msg_end():
    print('done.', flush=True)

def barplot_annotate_brackets(
    ax, num1, num2, data, center, height, yerr=None,
    dh=.05, barh=.05, fs=None, maxasterix=None, tickh=None
):
    if isinstance(data, str):
        text = data
    else:
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if not text:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
    if tickh is None:
        tickh = barh

    if abs(num1 - num2) > 1:
        y = max(height) + dh
    else:
        y = max(ly, ry) + dh

    # Horizontal bar
    ax.plot([lx, rx], [y + barh, y + barh], c='black', lw=1)

    # Vertical ticks
    ax.plot([lx, lx], [y, y + barh], c='black', lw=1)
    ax.plot([rx, rx], [y, y + barh], c='black', lw=1)

    # Text label
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    ax.text((lx + rx) / 2, y + barh, text, **kwargs)


def barplot_annotate_brackets1(ax, num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)

    Adapted from:
        https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    if abs(num1-num2) > 1:
        y = max(height) + dh
    else:
        y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

def get_pval_str(pval):
    if pval < 0.05:
        if pval < 0.01:
            if pval < 0.001:
                return '***'
            return '**'
        return '*'
    return ''

def my_mean(nparray):
    if nparray.size == 0:
        return 0
    else:
        return np.mean(nparray)

# cf. https://matthew-brett.github.io/teaching/smoothing_intro.html
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))
def fwhm2sigma(fwhm=4):
    return fwhm / np.sqrt(8 * np.log(2))

# very slow; use scipy's gaussian_filter() instead
def smooth_velocities(velocities):
    y_vals = velocities
    x_vals = np.arange(0,velocities.shape[0])
    sigma = fwhm2sigma(fwhm=4)

    smoothed_vals = np.zeros(y_vals.shape)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y_vals * kernel)
    return smoothed_vals

def plot_bgmm_covariances(X, Y_, means, covariances, index, title, save_path, m_, cell_):
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
    plt.figure()
    splot = plt.subplot(1,1,1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        print('HERE')
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 1], X[Y_ == i, 0], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell = mpl.patches.Ellipse([mean[1], mean[0]], v[1], v[0], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        plt.savefig(os.path.join(save_path, '{}_cell_{}_cov.png'.format(m_, cell_)), format='png', dpi=300)
        plt.close()

def get_only_crossreg_str(only_crossreg):    
    if only_crossreg:
        only_crossreg_str = 'only_crossreg'
    else:
        only_crossreg_str = 'all_neurons'
    return only_crossreg_str

def get_transpose_str(transpose_wanted):
    if transpose_wanted:
        transpose_str = 'transpose'
    else:
        transpose_str = 'non_transpose'
    return transpose_str

def get_paper_dir(PAPER_DIR, fig_name):
    paper_dir = os.path.join(PAPER_DIR, fig_name, 'plots')
    os.makedirs(paper_dir, exist_ok=True)
    return paper_dir

###
### SIGSTAR
###

def sigstar(ax, groups, stats, props=None):
    if not isinstance(groups, list) or (len(groups) == 2 and not isinstance(groups[0], list)):
        groups = [groups]

    if stats is None:
        stats = [0.05] * len(groups)
    elif len(stats) == 0:
        stats = [0.05] * len(groups)

    if props is None:
        props = {}

    nosort = props.get('nosort', 0)
    want_ticks = props.get('want_ticks', 0)
    sigbar_sep_amt = props.get('sigbar_sep_amt', 0.05)
    fontsize = props.get('FontSize', 16)
    max_errbar = props.get('max_errbar', 0)

    if not isinstance(groups, list):
        raise ValueError('GROUPS must be a list')
    if not isinstance(stats, list):
        raise ValueError('STATS must be a list')
    if len(stats) != len(groups):
        raise ValueError('GROUPS and STATS must be the same length')

    xlocs = np.full((len(groups), 2), np.nan)
    xtl = ax.get_xticks()

    for ii, grp in enumerate(groups):
        if isinstance(grp, list) and all(isinstance(g, str) for g in grp):
            xlocs[ii, :] = [xtl.tolist().index(g) + 1 for g in grp]
        elif isinstance(grp, list) and all(isinstance(g, (int, float)) for g in grp):
            xlocs[ii, :] = grp
        elif isinstance(grp, list) and isinstance(grp[0], str) and isinstance(grp[1], (int, float)):
            xlocs[ii, :] = [xtl.tolist().index(grp[0]) + 1, grp[1]]
        elif isinstance(grp, list) and isinstance(grp[0], (int, float)) and isinstance(grp[1], str):
            xlocs[ii, :] = [grp[0], xtl.tolist().index(grp[1]) + 1]

        xlocs[ii, :] = np.sort(xlocs[ii, :])

    if np.isnan(xlocs).any():
        raise ValueError('Some groups were not found')

    if not nosort:
        lengths = xlocs[:, 1] - xlocs[:, 0]
        sorted_indices = np.argsort(lengths)
        xlocs = xlocs[sorted_indices]
        groups = [groups[i] for i in sorted_indices]
        stats = [stats[i] for i in sorted_indices]

    H = np.ones((len(groups), 2), dtype=object)
    y = ax.get_ylim()
    yd = (y[1] - y[0]) * sigbar_sep_amt

    next_offset = 0
    for ii, grp in enumerate(groups):
        thisY = findMinY(ax, max_errbar) + yd
        if stats[ii] < 0.05:
            H[ii, :], next_offset = makeBar(ax, xlocs[ii, :], thisY + next_offset, stats[ii], fontsize)

    if want_ticks:
        yd = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
        for ii, stat in enumerate(stats):
            if stat < 0.05:
                y = H[ii, 0].get_ydata()
                y[0] -= yd
                y[3] -= yd
                H[ii, 0].set_ydata(y)

    return H

def makeBar(ax, x, y, p, fontsize, nostars=0):
    if p <= 1E-3:
        stars = '***'
    elif p <= 1E-2:
        stars = '**'
    elif p <= 0.05:
        stars = '*'
    else:
        p = np.nan
        stars = 'n.s.'

    x = np.repeat(x, 2)
    y = np.repeat(y, 4)

    H = [None, None]
    H[0] = ax.plot(x, y, '-k', linewidth=1.5)[0]

    next_offset = 0
    offset = 0.007 if not np.isnan(p) else 0.05
    next_offset = 3 if np.isnan(p) else 0

    if not nostars:
        H[1] = ax.text(np.mean(x), np.mean(y) + (plt.ylim()[1] - plt.ylim()[0]) * offset, stars,
                        horizontalalignment='center', backgroundcolor='none', fontsize=fontsize)

    return H, next_offset

def findMinY(ax, max_errbar):
    y = ax.get_ylim()[1] + max_errbar
    return y

def myRange(x):
    return np.max(x) - np.min(x)

def sanitize_XY_bounds(idx, lim):
    if idx > lim:
        return lim
    return idx

def regularize_covariance(cov_matrix, lambda_val=1e-6):
    # Add lambda to the diagonal of the covariance matrix
    identity_matrix = np.eye(cov_matrix.shape[0])
    reg_cov_matrix = cov_matrix + lambda_val * identity_matrix
    return reg_cov_matrix

def mahalanobis(x, data, cov_type='lw', pinv=False, lambda_val=0.001):
    """
    Compute the Mahalanobis Distance between a population vector (PV) and a set of PVs.

    x    : a column from a S or C matrix or binned one (i.e., one "time point")
    data : the S or C matrix, or subset thereof

    Adapted from: https://www.machinelearningplus.com/statistics/mahalanobis-distance/ 

    However, typically we have in our data that the dimensionality (number of neurons in the
    PV) is  much higher than the number of samples (time points). I.e., 
    data.shape[0] >> data.shape[1]. Thus the covariance of this matrix is typically 
    singular and hence not invertible. We can use workarounds such as regularization of the
    covariance matrix. The 'best' adopted method recommended here is Ledoit-Wolf shrinkage.

    Cf. https://scikit-learn.org/1.5/modules/covariance.html#shrunk-covariance

    cov_type can be:
      'normal' : regular covariance matrix
      'regularized' : regularized covariance matrix (using lambda_val as parameter)
      'lw' : Ledoit-Wolf covariance matrix
    
    NB:
    pinv is a flag to use the pseudo-inverse of the covariance matrix instead of the inverse.
    But rather than this, use Ledoit-Wolf shrinkage to get a regularized covariance matrix.
    """
    x_minus_mu = x - np.mean(data,1)

    # First regularize (using ridge regression)

    if cov_type == 'normal':
        cov_mat = np.cov(data)
    elif cov_type == 'regularized':
        cov_mat = regularize_covariance(np.cov(data), lambda_val=lambda_val)
    elif cov_type == 'lw':
        lw = LedoitWolf()
        cov_mat = lw.fit(data.T).covariance_

    if pinv:
        inv_cov_mat = np.linalg.pinv(cov_mat)
    else:                
        inv_cov_mat = np.linalg.inv(cov_mat)
    left_term = np.dot(x_minus_mu, inv_cov_mat)
    
    # what is computed is the squared Mahalanobis distance, so return the square root
    mahal = np.sqrt(np.dot(left_term, x_minus_mu)) 
    return mahal, cov_mat

def mahalanobis_old(x, data, regularize=True, lambda_val=0.001):
    """
    Compute the Mahalanobis Distance between a population vector (PV) and a set of PVs.

    x    : a column from a S or C matrix or binned one (i.e., one "time point")
    data : the S or C matrix, or subset thereof

    Adapted from: https://www.machinelearningplus.com/statistics/mahalanobis-distance/ 

    We use Moore-Penrose pseudo-inverse since the dimensionality (number of neurons in the
    PV) is typically much higher than the number of samples (time points).     
    
    I.e., data.shape[0] >> data.shape[1]. Thus the covariance of this matrix is typically 
    singular and hence not invertible.
    """
    x_minus_mu = x - np.mean(data,1)
    cov = np.cov(data)
    if regularize:
        inv_covmat = np.linalg.inv(regularize_covariance(cov, lambda_val=lambda_val))
    else:
        inv_covmat = np.linalg.pinv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu)
    return mahal, cov

def plot_pv_matrices_with_distance(PV0, PV1, dist, dist_type, shuffle_type, comparison, mouse, plot_num=0, i=None, dir_name=None, auto_close=True, \
    comp_range_str=None, comp_i0_i1=None, calc_type_str=''):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: imshow of PV0
    axes[0].imshow(PV0, aspect='auto', cmap='viridis')
    axes[0].set_title('PV0')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')

    # Second subplot: imshow of PV1
    axes[1].imshow(PV1, aspect='auto', cmap='viridis')
    axes[1].set_title('PV1')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')

    # Set overall title with distance, i0, i1, comparison, and mouse
    fig.suptitle(f'Mouse: {mouse}, Comparison: {comparison}, {dist_type} Distance: {dist:.2f}\n{calc_type_str} i:{i} Comparisons: ({comp_range_str[0]}, {comp_range_str[1]}), ({comp_i0_i1[0]}, {comp_i0_i1[1]})', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for suptitle
    plt.show()

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        fig.savefig(os.path.join(dir_name, f'{comparison}_mouse_{mouse}_{dist_type}_{shuffle_type}_{plot_num}_distance_{dist}.png'), format='png', dpi=300)
        if auto_close:
            plt.close(fig)

def assign_dict2(d, keys, value):
    """
    Assign a value to a nested dictionary structure based on a list of keys.

    Parameters:
    d (dict): The dictionary to assign the value to.
    keys (list): A list of keys representing the nested structure.
    value: The value to assign.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def assign_dict(d, keys, value):
    """
    Assign a value to a nested dictionary structure based on a list of keys.
    If a key already exists at any level, it appends the new value to a list.
    
    Parameters:
    d (dict): The dictionary to assign the value to.
    keys (list): A list of keys representing the nested structure.
    value: The value to assign.

    # Example usage
    d = {'a': {'b': {'c': 2}}}
    assign_dict(d, ['a', 'b', 'e'], 5)
    print(d)  # Output: {'a': {'b': {'c': [2, 1]}}}
    """
    for key in keys[:-1]:
        if key in d:
            if isinstance(d[key], dict):
                d = d[key]
            else:
                if isinstance(d[key], list):
                    d[key].append({})
                    d = d[key][-1]
                else:
                    d[key] = [d[key], {}]
                    d = d[key][-1]
        else:
            d = d.setdefault(key, {})
    
    if keys[-1] in d:
        if isinstance(d[keys[-1]], list):
            d[keys[-1]].append(value)
        else:
            d[keys[-1]] = [d[keys[-1]], value]
    else:
        d[keys[-1]] = value

def plot_random_cells(mouse, session, num_random_cells=10, vel_mask_only=False):
    """
    Plots random cells' activity for a given mouse and session.
    
    For TFC_cond only now.

    Parameters:
    mouse (str): Identifier for the mouse.
    session (dict): Dictionary containing session data for each mouse.
    num_random_cells (int, optional): Number of random cells to plot. Default is 10.
    The function selects a random subset of cells from the session data and plots their activity.
    It also overlays velocity data and event markers (tone and shock onsets/offsets) on the plots.
    Example usage:
    --------------
    plot_random_cells('mouse1', session_data, num_random_cells=10)
    """

    sess = session[mouse]
    random_cells = random.sample(range(sess.S.shape[0]), num_random_cells)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
    #fig.suptitle(f'Random Cells for Mouse {mouse} {sess.behaviour_condition}', fontsize=16)
    fig.suptitle(f'Random Cells for Mouse {mouse}', fontsize=16)
    for i, cell in enumerate(random_cells[:9]):
        ax = axes[i // 3, i % 3]
        if not vel_mask_only:
            ax.plot(sess.velocities_miniscope, 'b', alpha=0.5)
            ax.plot(sess.velocities_miniscope_smooth, 'k', alpha=0.5)

        for x in sess.tone_onsets:
            ax.axvline(x, c='b', ls='--')
        for x in sess.tone_offsets:
            ax.axvline(x, c='b', ls='--')
        for x in sess.shock_onsets:
            ax.axvline(x, c='r', ls='--')
        for x in sess.shock_offsets:
            ax.axvline(x, c='r', ls='--')
        ax.plot(sess.S[cell, :], 'r')
        ax.plot(sess.vel_mask*10, 'g', alpha=1)

        ax.text(0.95, 0.95, f'Cell {cell}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=1))
        
    plt.tight_layout()
    print(random_cells[:9])
    
def sanity_check_cells(session, mouse, cell_ids=None, num_plot=9):
    '''
    If cell_ids is None, pick num_plot random cells and plot them in a square grid.
    If cell_ids is provided (as a list or 1D numpy array), plot those cells in a square grid.
    Assumes num_plot or len(cell_ids) is a perfect square.
    '''
    s = session[mouse]
    if cell_ids is None:
        assert int(np.sqrt(num_plot)) ** 2 == num_plot, "num_plot must be a perfect square"
        cells_to_plot = random.sample(range(s.S.shape[0]), num_plot)
    else:
        if isinstance(cell_ids, np.ndarray):
            assert cell_ids.ndim == 1, "cell_ids numpy array must be 1D"
            cells_to_plot = cell_ids.tolist()
        else:
            assert isinstance(cell_ids, list), "cell_ids must be a list or 1D numpy array"
            cells_to_plot = cell_ids
        assert int(np.sqrt(len(cells_to_plot))) ** 2 == len(cells_to_plot), "len(cell_ids) must be a perfect square"

    grid_size = int(np.sqrt(len(cells_to_plot)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4 * grid_size, 3 * grid_size), sharex=True, sharey=True)
    for i, cell in enumerate(cells_to_plot):
        ax = axes[i // grid_size, i % grid_size]
        ax.plot(s.S[cell, :], 'r', label='S')
        ax.plot(s.C[cell, :], 'b', label='C')
        ax.plot(s.YrA[cell, :], 'k', label='YrA')
        ax.set_title(f'Mouse {mouse} cell {cell}')
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.show()

def reanchor_XY_to_zero(X, Y):
    '''
    Re-anchor X and Y coordinates so that the minimum X and Y values are zero. Importantly, makes a copy of the input arrays.
    This is for use in LinearTrackSession for loc_* data.
    '''
    X_new = np.array(X, copy=True)
    Y_new = np.array(Y, copy=True)

    min_X = np.nanmin(X_new)
    if not np.isnan(min_X) and min_X > 0:
        X_new = X_new - min_X
    min_Y = np.nanmin(Y_new)
    if not np.isnan(min_Y) and min_Y > 0:
        Y_new = Y_new - min_Y
    return X_new, Y_new, min_X, min_Y

def set_compact_plot_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    })

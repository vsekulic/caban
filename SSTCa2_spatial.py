from random import shuffle
import numpy as np
from SSTCa2_utilities import *
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from numpy.random import default_rng
import time
from sklearn import mixture
from numpy import linalg
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
import re
import shutil
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, mannwhitneyu, norm as _stats_norm

'''
Process mouse location and establish bins.
'''
class Location_XY:
    def __init__(self, mouse, group, sess, bin_width, DEBUG=True, loc_bounds=None):
        self.MAX_X = 613.9776984686001
        self.MAX_Y = 467.7186760984302
        self.MIN_X = 27.142475257551364
        self.MIN_Y = 1.2439073151409752

        if loc_bounds is not None:
            self.MAX_X = loc_bounds.get('MAX_X', self.MAX_X)
            self.MAX_Y = loc_bounds.get('MAX_Y', self.MAX_Y)
            self.MIN_X = loc_bounds.get('MIN_X', self.MIN_X)
            self.MIN_Y = loc_bounds.get('MIN_Y', self.MIN_Y)

        self.MAX_BINNED_X = round(self.MAX_X / bin_width)
        self.MAX_BINNED_Y = round(self.MAX_Y / bin_width)

        self.sess = sess
        self.mouse = mouse
        self.group = group
        self.bin_width = bin_width

        # Process mouse location and establish bins
        loc_X = sess.loc_X_miniscope_smooth
        loc_Y = sess.loc_Y_miniscope_smooth
        [min_x, min_y] = [np.min(loc_X), np.min(loc_Y)]
        [max_x, max_y] = [np.max(loc_X), np.max(loc_Y)]
        binned_X = (loc_X / bin_width).astype(int)
        binned_Y = (loc_Y / bin_width).astype(int)
        num_bins_x = self.MAX_BINNED_X
        num_bins_y = self.MAX_BINNED_Y
        if DEBUG:
            print('Location_XY: we have num_bins_x {}, num_bins_y {}'.format(num_bins_x, num_bins_y))
        
        self.loc_X = loc_X
        self.loc_Y = loc_Y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.binned_X = binned_X
        self.binned_Y = binned_Y
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y

    def init_orig(self, mouse, group, sess, bin_width, DEBUG=True):
        self.sess = sess
        self.mouse = mouse
        self.group = group
        self.bin_width = bin_width

        # Process mouse location and establish bins
        loc_X = sess.loc_X_miniscope_smooth
        loc_Y = sess.loc_Y_miniscope_smooth
        [min_x, min_y] = [np.min(loc_X), np.min(loc_Y)]
        [max_x, max_y] = [np.max(loc_X), np.max(loc_Y)]
        x_bin_width = int((max_x - min_x) / bin_width) # Unused in the end
        y_bin_width = int((max_y - min_y) / bin_width)
        # Make consistent square bins, so only use y_bin_width
        binned_X = ((loc_X - min_x)/ bin_width).astype(int) 
        binned_Y = ((loc_Y - min_y)/ bin_width).astype(int) 
        num_bins_x = max(binned_X)
        num_bins_y = max(binned_Y)
        if DEBUG:
            print('Location_XY: we have num_bins_x {}, num_bins_y {}'.format(num_bins_x, num_bins_y))
        
        self.loc_X = loc_X
        self.loc_Y = loc_Y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.binned_X = binned_X
        self.binned_Y = binned_Y
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
    
    def get_loc_data(self):
        return [self.loc_X, self.loc_Y, self.min_x, self.min_y, self.max_x, self.max_y, self.binned_X, self.binned_Y, self.num_bins_x, self.num_bins_y]

    '''
    Create occupancy map.
    '''
    def get_occupancy_map(self, PLOTS_DIR=None, session_str=None):
        self.occupancy = np.zeros((self.num_bins_y+1, self.num_bins_x+1))
        for i in range(len(self.loc_X)): # loc_X, loc_Y are the same legnth
            idx_x = int(self.loc_X[i]/self.bin_width)
            idx_y = int(self.loc_Y[i]/self.bin_width)
            idx_x = sanitize_XY_bounds(idx_x, self.occupancy.shape[1]-1)
            idx_y = sanitize_XY_bounds(idx_y, self.occupancy.shape[0]-1)
            self.occupancy[idx_y, idx_x] += MINISCOPE_FRAME_MS
        
        if PLOTS_DIR:
            plt.figure()
            plt.imshow(self.occupancy)
            plt.title('Occupancy (ms) Mouse {} {}'.format(self.mouse, self.group))
            save_path = os.path.join(PLOTS_DIR, '{}_occupancy'.format(session_str))
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'occupancy_{}_{}_{}.png'.format(self.mouse, self.group, session_str)), format='png', dpi=300)
            plt.close()

        return self.occupancy
    
    def get_occupancy_map_orig(self, PLOTS_DIR=None, session_str=None):
        self.occupancy = np.zeros((self.num_bins_y+1, self.num_bins_x+1))
        for i in range(len(self.loc_X)): # loc_X, loc_Y are the same legnth
            idx_x = int((self.loc_X[i]-self.min_x)/self.bin_width)
            idx_y = int((self.loc_Y[i]-self.min_y)/self.bin_width)
            self.occupancy[idx_y, idx_x] += MINISCOPE_FRAME_MS
        
        if PLOTS_DIR:
            plt.figure()
            plt.imshow(self.occupancy)
            plt.title('Occupancy (ms) Mouse {} {}'.format(self.mouse, self.group))
            save_path = os.path.join(PLOTS_DIR, '{}_occupancy'.format(session_str))
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'occupancy_{}_{}_{}.png'.format(self.mouse, self.group, session_str)), format='png', dpi=300)
            plt.close()

        return self.occupancy

class PlaceFields:
    '''
    Contains data for place fields for all cells, per mouse. The idea is that each mouse, and each behaviour, will have an associated
    PlaceFields object. In particular, each FluorescenceMap will use a PlaceFields. This was initially part of FluorescenceMap but was 
    factored out so that it can be a self-contained save/load module and hence part of FluorescenceMap.find_place_fields() functionality.
    '''
    def __init__(self, to_pickle=False, sess=None, mouse=None, num_cells=0):
        self.to_pickle=to_pickle
        self.sess=sess
        self.mouse=mouse
        self.num_cells=num_cells

        self.model_ = {}                # gmm model, indexed by cell
        self.merged_means = {}          # Index into model_.means, by cell
        self.merged_means_ = {}         # Per-PF merged mean (2,) in [row,col], indexed by cell → list
        self.merged_covariances_ = {}   # Per-PF merged covariance (2,2), indexed by cell → list
        self.merged_weights_ = {}       # Per-PF total weight (scalar), indexed by cell → list
        self.responses_means = {}       # lists sig responses' assigned means, indexed by cell 
        self.responses_pf = {}          # lists all responses belonging to each cell's pf, indexed by cell
        self.pf_size = {}               # area of pf for all pf's per cell, indexed by cell. Used in calculation of compactness_pf.
        self.compactness_pf = {}        # compactness metric of all pf's per cell, indexed by cell
        self.infield_sum_ = {}          # sum of in-field S activity of all pf's per cell, indexed by cell (for calculating means)
        self.infield_mean = {}          # mean in-field S activity of all pf's per cell, indexed by cell
        self.outfield_mean = {}         # mean out-of-field S activity, just one value per cell.
        self.spatial_selectivity = {}   # spatial selectivity of all pf's per cell, indexed by cell
        
        self.loaded = False
        if to_pickle:
            self.save_pickle_path = os.path.join(sess.savepath, sess.session_type+'-'+self.mouse+'-'+str(sess.S.shape[0])+'cells'+'-PlaceFields')
            self.load_pickle_path = self.save_pickle_path+'.npz'
            if os.path.exists(self.load_pickle_path):
                load_data = np.load(self.load_pickle_path, allow_pickle=True)
                self.loaded = True
                # What are all those [()], you say? Well numpy returns a **zero**-dimensional array where the only element is the actual data because
                # of course it would, so have to use this funny syntax to get it out. NB: load_data['foo'][0] does NOT work...
                # https://stackoverflow.com/a/51150083/3268364
                self.model_ = load_data['model_'][()]
                self.merged_means = load_data['merged_means'][()]
                if 'merged_means_' in load_data:
                    self.merged_means_ = load_data['merged_means_'][()]
                    self.merged_covariances_ = load_data['merged_covariances_'][()]
                    self.merged_weights_ = load_data['merged_weights_'][()]
                self.responses_means = load_data['responses_means'][()]
                self.responses_pf = load_data['responses_pf'][()]
                self.pf_size = load_data['pf_size'][()]
                self.compactness_pf = load_data['compactness_pf'][()]
                self.infield_sum_ = load_data['infield_sum_'][()]
                self.infield_mean = load_data['infield_mean'][()]
                self.outfield_mean = load_data['outfield_mean'][()]
                self.spatial_selectivity = load_data['spatial_selectivity'][()]
        else:
            self.pickle_path = None

    def save_data(self):
        if self.to_pickle:
            np.savez(self.save_pickle_path, \
                model_=self.model_, \
                merged_means=self.merged_means, \
                merged_means_=self.merged_means_, \
                merged_covariances_=self.merged_covariances_, \
                merged_weights_=self.merged_weights_, \
                responses_means=self.responses_means, \
                responses_pf=self.responses_pf, \
                pf_size=self.pf_size, \
                compactness_pf=self.compactness_pf, \
                infield_sum_=self.infield_sum_, \
                infield_mean=self.infield_mean, \
                outfield_mean=self.outfield_mean, \
                spatial_selectivity=self.spatial_selectivity)

class FluorescenceMap:
    def __init__(self, loc, S, cells, save_path=None, mouse=None, max_save=15, to_pickle=False, sess=None, load_num_cells=0, print_pcell_maps=True, max_fields=15, DEBUG=True):
        self.DEBUG = DEBUG
        self.loc = loc
        self.S = S
        self.cells = cells
        if cells is not None:
            self.num_cells = len(cells)
        else:
            self.num_cells = 0
        self.fluorescence_map = None
        self.fluorescence_map_occup = None
        self.mouse = mouse
        self.sess = sess
        if save_path is not None:
            self.save_path = os.path.join(save_path, 'pcells_{}'.format(mouse))
            os.makedirs(self.save_path, exist_ok=True)
        self.max_save = max_save
        self.save_counter = 0
        self.to_pickle = to_pickle
        self.shuffled_responses = np.array([]) # set as default
        self.sig_responses = None
        if to_pickle:
            if load_num_cells > 0: # have to specify this initially so the proper number is loaded (messy)
                self.num_cells = load_num_cells 
            self.save_pickle_path = os.path.join(sess.savepath, sess.session_type+'-S_shuffled-'+self.mouse+'-'+str(self.num_cells)+'cells')
            self.load_pickle_path = self.save_pickle_path+'.npz'
            if load_num_cells > 0: # then we want to try to load some from pickle, this specifies how many cells
                if os.path.exists(self.load_pickle_path):
                    load_data = np.load(self.load_pickle_path, allow_pickle=True)
                    self.cells = load_data['cells']
                    self.shuffled_responses = load_data['shuffled_responses']
                    self.num_cells = len(self.cells)
            
            # Save sig_responses separately (ugly; should refactor all this to use Saver system)
            self.save_pickle_path_sig_responses = os.path.join(self.sess.savepath, self.sess.session_type+'-sig_responses-'+self.mouse+'-'+str(self.num_cells)+'cells')
            self.load_pickle_path_sig_responses = self.save_pickle_path_sig_responses+'.npz'
            if os.path.exists(self.load_pickle_path_sig_responses):
                load_data = np.load(self.load_pickle_path_sig_responses, allow_pickle=True)
                self.sig_responses = load_data['sig_responses'][()]
        else:
            self.pickle_path = None
        self.print_pcell_maps = print_pcell_maps
        self.max_fields = max_fields

        # Key save structure; see class constructor for all data and meanings. Primarily used by FluorescenceMap.find_place_fields()
        self.pf = PlaceFields(to_pickle=to_pickle, sess=sess, mouse=mouse, num_cells=load_num_cells)

    def check_fluorescence_map(self):
        if self.fluorescence_map is None:
            self.fluorescence_map = self.generate_map()

    def check_fluorescence_map_occup(self):
        if self.fluorescence_map_occup is None:
            self.fluorescence_map_occup = self.generate_occupancy_map(want_occup_map=False)

    def generate_map(self, S_to_use=None):
        if S_to_use is not None:
            S = S_to_use
        else:
            S = self.S

        fluorescence_map = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1, self.num_cells)) # so if max is 20, it's the 21st entry into its dimension (0 being the 1st)
        for cell, i in zip(self.cells, range(self.num_cells)):
            S_cell = S[cell,:]
            for j in range(len(S_cell)):
                fluorescence_map[self.loc.binned_Y[j], self.loc.binned_X[j], i] += S_cell[j]
        return fluorescence_map

    def get_map(self):
        self.check_fluorescence_map()
        return self.fluorescence_map

    def generate_occupancy_map(self, want_occup_map=True, S_to_use=None):
        if S_to_use is not None:
            S = S_to_use
        else:
            S = self.S

        # so if max is 20, it's the 21st entry into its dimension (0 being the 1st)
        fluorescence_map_occup = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1, self.num_cells)).astype(np.float32, copy=False) 
        if want_occup_map:
            occup_map = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1))

        S_gauss = gaussian_filter(S.flatten(),sigma=SMOOTH_LOC_SIGMA).reshape(S.shape)
        for i in range(S.shape[1]):
            idx_y = sanitize_XY_bounds(self.loc.binned_Y[i], fluorescence_map_occup.shape[0] - 1)
            idx_x = sanitize_XY_bounds(self.loc.binned_X[i], fluorescence_map_occup.shape[1] - 1)
            fluorescence_map_occup[idx_y, idx_x, :] += S_gauss[self.cells, i]

            if want_occup_map:
                occup_map[idx_y, idx_x] += 1
            
        for y in range(fluorescence_map_occup.shape[0]): # rows are actually Y 
            for x in range(fluorescence_map_occup.shape[1]): # columns are X
                if self.loc.occupancy[y,x] == 0:
                    fluorescence_map_occup[y, x, :] = 0
                else:
                    fluorescence_map_occup[y, x, :] /= self.loc.occupancy[y, x]

        if want_occup_map:
            return [fluorescence_map_occup, occup_map]
        else:
            return fluorescence_map_occup

    def generate_occupancy_map_old(self, want_occup_map=True, S_to_use=None):
        if S_to_use is not None:
            S = S_to_use
        else:
            S = self.S

        fluorescence_map_occup = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1, self.num_cells)) # so if max is 20, it's the 21st entry into its dimension (0 being the 1st)
        if want_occup_map:
            occup_map = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1, self.num_cells))

        for cell, i in zip(self.cells, range(self.num_cells)):
            #fluorescence_map = np.zeros((num_bins_y+1, num_bins_x+1)) # so if max is 20, it's the 21st entry into its dimension (0 being the 1st)
            S_cell = gaussian_filter(S[cell,:], sigma=SMOOTH_LOC_SIGMA)
            # S_gauss = gaussian_filter(S.flatten(),sigma=SMOOTH_LOC_SIGMA).reshape(S.shape)
            for j in range(len(S_cell)):
                fluorescence_map_occup[self.loc.binned_Y[j], self.loc.binned_X[j], i] += S_cell[j]
                if want_occup_map:
                    occup_map[self.loc.binned_Y[j], self.loc.binned_X[j], i] += 1
            for y in range(fluorescence_map_occup.shape[0]): # rows are actually Y 
                for x in range(fluorescence_map_occup.shape[1]): # columns are X
                    if self.loc.occupancy[y,x] == 0:
                        fluorescence_map_occup[y, x, i] = 0
                    else:
                        fluorescence_map_occup[y, x, i] /= self.loc.occupancy[y, x]
        if want_occup_map:
            return [fluorescence_map_occup, occup_map]
        else:
            return fluorescence_map_occup

    def get_occupancy_map(self):
        self.check_fluorescence_map_occup()
        return self.fluorescence_map_occup

    def get_max_loc(self, fm=None):
        """
        Get the coordinate of the maximum bin in the occupancy-corrected fluorescence map for each cell. Allows passing of a 
        fluorescence map in case want to do things like zero out the max, and pass it back again to find the next max, etc.
        Otherwise, use the self.fluorescence_map_occup.

        Returns a tuple of coordinates using row,idx format corresponding to the max intensities for each cell (z dimension
        in fluorescence map).
        """
        if fm is None:
            fm = self.fluorescence_map_occup
        ind = []
        for i in range(fm.shape[2]): # num of cells (z dimension)
            ind.append(np.unravel_index(np.argmax(fm[:,:,i],axis=None),fm[:,:,i].shape))
        return ind

    def get_max(self, fm=None):
        """
        Similar to get_max_loc() but instead get max *value* per cell, not the location. This is most useful for comparing maximum
        response amplitudes to shuffles.
        """
        if fm is None:
            fm = self.fluorescence_map_occup
        max_values = []
        for i in range(fm.shape[2]): # num of cells (z dimension)
            ind = np.unravel_index(np.argmax(fm[:,:,i],axis=None),fm[:,:,i].shape)
            max_values.append(fm[ind[0],ind[1],i])
        return max_values

    def get_significant_response_profiles(self, percentile=99.0):
        """
        Calculate significant responses from temporal traces and occupancy-corrected fluorescence maps, as an
        extension of the methods of Fournier et al. (2020). 

        Returns sig_responses dict with mapping of cell -> list of x top fields with sorted maximum fluorescence
        that were significant (>99th percentile) after circularly shifting the temporal traces in time.
        """
        self.check_fluorescence_map_occup()
        if self.DEBUG:
            print('FluorescenceMap: we have {} max fields'.format(self.max_fields))

        if not self.sig_responses:

            # Get max_fields number of max values per cell.
            ind_fields = []  
            sig_responses = dict()
            fm = self.fluorescence_map_occup.copy()
            for field in range(self.max_fields):

                # Get maximum repsonses for all cells
                ind = self.get_max_loc(fm=fm)
                ind_fields.append(ind)

                # Each element in ind is the location of that cell's max value for this iteration.
                for i,max_field in zip(range(len(ind)), ind):
                    #if max_field in [(x,y) for x,y,_ in sig_responses[i]]:
                        # Then, we ran out of significant locations (fm is 0 everywhere) and if adding spuriously the same (0,0) location
                        # we should rather skip (one of many ways to solve this problem...)
                        #continue
                    #cell_max_response = self.fluorescence_map_occup[max_field[0],max_field[1],i]
                    cell_max_response = fm[max_field[0],max_field[1],i]
                    percentile_99th_response = np.percentile(self.shuffled_responses[:,i], percentile)
                    if cell_max_response > percentile_99th_response:
                        pfield_data = [max_field[0], max_field[1], cell_max_response]
                        if i not in sig_responses:
                            sig_responses[i] = []
                        sig_responses[i].append(pfield_data)

                # At the end, zero out the max responses, so we go through it again and find the next
                # largest responses and test if they are place fields or not.
                for i in range(len(ind)):
                    fm[ind[i][0], ind[i][1], i] = 0

            if self.save_path is not None and self.print_pcell_maps:
                for cellnum, responses in sig_responses.items():
                    self.save_map(cellnum, responses, percentile)
            self.sig_responses = sig_responses
            if self.to_pickle:
                np.savez(self.save_pickle_path_sig_responses, sig_responses=sig_responses)
        return self.sig_responses

    def get_shuffled_responses(self, num_shifts=500):
        self.check_fluorescence_map_occup()
        if not self.shuffled_responses.any():
            rng = default_rng()
            # Shifted range will be selected from a minimum of 5 seconds worth of frames up to 5 minutes worth of frames.

            # Shift entire spike array by random amount, up to num_shifts number of times.
            # Only need to shuffle once (num_shifts number of times, of course) to compare all place fields.
            shuffled_matrix = np.zeros((num_shifts, self.fluorescence_map_occup.shape[0], \
                self.fluorescence_map_occup.shape[1], self.num_cells))        
            for i in range(num_shifts):
                shift_range = np.arange(5*MINISCOPE_FPS, (5*60)*MINISCOPE_FPS)
                print(".", end='')
                shift = rng.choice(shift_range) 
                S_shifted = np.roll(self.S, shift, axis=0)
                fm_shifted = self.generate_occupancy_map(want_occup_map=False, S_to_use=S_shifted)
                shuffled_matrix[i,:,:,:] = fm_shifted
            shape = shuffled_matrix.shape
            self.shuffled_responses = shuffled_matrix.reshape(shape[0]*shape[1]*shape[2], shape[3])
            if self.to_pickle:
                np.savez_compressed(self.save_pickle_path, cells=self.cells, shuffled_responses=self.shuffled_responses)

    def get_shuffled_responses_old(self, num_shifts=500):
        rng = default_rng()
        # Shifted range will be selected from a minimum of 5 seconds worth of frames up to 5 minutes worth of frames.
        shift_range = np.arange(5*MINISCOPE_FPS, (5*60)*MINISCOPE_FPS)

        # Shift entire spike array by random amount, up to num_shifts number of times.
        # Only need to shuffle once (num_shifts number of times, of course) to compare all place fields.
        shuffled_responses = np.array([])
        for i in range(num_shifts):
            shift = rng.choice(shift_range) 
            S_shifted = np.roll(self.S, shift, axis=0)
            fm_shifted = self.generate_occupancy_map(want_occup_map=False, S_to_use=S_shifted)
            fm_flatten = fm_shifted.reshape(fm_shifted.shape[0]*fm_shifted.shape[1], fm_shifted.shape[2])
            if not shuffled_responses.any():
                shuffled_responses = fm_flatten
            else:
                shuffled_responses = np.append(shuffled_responses, fm_flatten, axis=0)
        self.shuffled_responses = shuffled_responses

    def save_map(self, cell_num, responses, percentile, num_fields=''):
        if self.save_counter <= self.max_save:
            plt.figure()
            plt.imshow(self.fluorescence_map_occup[:,:,cell_num])
            # black magic from https://stackoverflow.com/questions/12142133/how-to-get-first-element-in-a-list-of-tuples
            if responses is not None:
                plt.scatter(list(zip(*responses))[1], list(zip(*responses))[0], marker="x", color='w')
            plt.title('{} cell {} percentile {}'.format(self.mouse, self.cells[cell_num], percentile))
            #plt.savefig(os.path.join(self.save_path, 'pcells_{}_{}cell_{}_perc_{}.png'.format(self.mouse, num_fields, self.cells[cell_num], percentile)), format='png', dpi=300)
            plt.savefig(os.path.join(self.save_path, 'pcells_{}_max_fields_{}_num_fields_{}_cell_{}_perc_{}.png'.format(self.mouse, self.max_fields, len(responses), cell_num, percentile)), format='png', dpi=300)
            plt.close()

    def find_place_fields(self, sess, method='iterative_gauss', n_comp_start=10, merge_distance=4):
        '''
        Find place fields using sig_responses (old 'pcells') dict. Uses various methods:

        iterative_gauss - use variational Bayesian estimation of a Gaussian mixture model with post-hoc merging.
        kmeans - K-means clustering with post-hoc cluster merging. (<-NOT USED/NOT FULLY IMPLEMENTED)
        '''
        
        save_path = os.path.join(self.save_path, 'pfields')
        os.makedirs(save_path, exist_ok=True)
        self.pf.responses_pf = {} # the assigned place fields of all significant responses -> IMPORTANT OUTPUT

        if method == 'iterative_gauss' and not self.pf.loaded:
            m_ = self.mouse

            cells_done = 1
            cells_tot = len(self.sig_responses)
            for cell_, responses in self.sig_responses.items():
                print('cell {}, {} of {} ({} %)'.format(cell_, cells_done, cells_tot, ((cells_done/cells_tot)*100)))
                cells_done += 1
                converged_iterative_bgmm = False                                                                                                                                                                              
                n_comp = np.min((n_comp_start, len(responses)))

                fm = self.fluorescence_map_occup
                #save_path_cell = os.path.join(save_path, str(cell_))

                max_intensity = np.max(fm[:,:,cell_])
                #increments = 10
                #intensity_increments = max_intensity / increments
                intensity_increments = 0.05
                st = time.time()
                sample_weights = []
                data = []
                for entry in responses:
                    for i in range(math.ceil(max_intensity / intensity_increments)+1):
                        data.append([entry[0], entry[1]])
                        sample_weights.append(entry[2])
                data = np.array(data)
                sample_weights = np.array(sample_weights)
                et = time.time()
                print('took {}'.format(et-st))

                #data = fm[:,:,340]
                x = np.linspace(-0.5,fm.shape[0]-0.5)
                y = np.linspace(-0.5,fm.shape[1]-0.5)

                m,n = data.shape
                R,C = np.mgrid[:m,:n]
                out = np.column_stack((C.ravel(),R.ravel(), data.ravel()))
                #gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(data)
                #n_comp = 2
                #w_conc_prior = (1./n_comp)/100
                #w_conc_prior = (1./n_comp) * 1e2

                #w_conc_prior = 1e-3
                w_conc_prior = None # default of 1/n_comp seems to work well after all..

                iter = 1
                while not converged_iterative_bgmm:
                    gmm_obtained = False
                    while not gmm_obtained:
                        try:
                            gmm = mixture.BayesianGaussianMixture(n_components=n_comp, covariance_type='full', \
                                weight_concentration_prior=w_conc_prior, warm_start=True, init_params='k-means++').fit(data)
                            #kmeans = KMeans(n_clusters=n_comp, random_state=0).fit(data, sample_weight=sample_weights)
                            gmm_obtained = True
                        except ValueError:
                            n_comp = max(1, n_comp - 1)
                            print('**** ValueError with BayesianGaussianMixture, switching to n_comp = {}'.format(n_comp))

                    # Thresholded bgmm
                    X, Y = np.meshgrid(x, y)
                    XX = np.array([X.ravel(), Y.ravel()]).T
                    Z = -gmm.score_samples(XX)
                    Z = Z.reshape(X.shape)
                    plt.figure()
                    plt.imshow(fm[:,:,cell_], cmap='viridis')

                    responses = sess.sig_responses[cell_]
                    plt.scatter(list(zip(*responses))[1], list(zip(*responses))[0], marker="o", color='w', s=1)

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
                    plt.title('n_comp {}'.format(n_comp))
                    plt.savefig(os.path.join(save_path, '{}_cell_{}_bgmm_iter_{}_n_comp_{}.png'.format(m_, cell_, iter, n_comp)), format='png', dpi=300)

                    means_over_thres = [weight for weight in gmm.weights_ if weight > thres]
                    if len(means_over_thres) < n_comp:
                        n_comp = len(means_over_thres)
                        iter += 1
                    else:
                        converged_iterative_bgmm = True

                # *** For plotting purposes to show covariance shapes, not used for model fitting or pf classification -VS
                # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py    
                dpgmm = mixture.BayesianGaussianMixture(n_components=n_comp, covariance_type='full', \
                    weight_concentration_prior=w_conc_prior).fit(data)
                plot_bgmm_covariances(
                    data,
                    dpgmm.predict(data),
                    dpgmm.means_,
                    dpgmm.covariances_,
                    1,
                    "Bayesian Gaussian Mixture with a Dirichlet process prior",
                    save_path,
                    m_,
                    cell_
                )
                #_END_##############################################################################
                plt.close()

                model_ = gmm
                self.pf.model_[cell_] = model_

                # Merge Gaussians if means below distance threshold
                #merge_distance = 9 # with each bin being 2cm, this is then 18cm as per Dombeck et al 2010.
                #merge_distance = 4 # Now a function parameter set to 4 as the default (2026.02.02 VS)
                model_means = model_.means_
                merged_means = []
                for i in range(len(model_means)):
                    for j in range(i+1,len(model_means)):
                        mean_i = model_means[i]
                        mean_j = model_means[j]
                        if linalg.norm(mean_i - mean_j) < merge_distance:
                            print('cell', cell_, i, mean_i, j, mean_j, 'are under merge_distance of', merge_distance)
                            added = False
                            for m in merged_means:
                                if i in m:
                                    m.append(j)
                                    added = True
                                if j in m:
                                    m.append(i)
                                    added = True
                            if not added:
                                m = []
                                m.append(i)
                                m.append(j)
                                merged_means.append(m)
                merged_means_uniq = []
                for m in merged_means:
                    merged_means_uniq.append(np.unique(m).tolist())
                merged_means = merged_means_uniq

                # Add any non-merged means as their own single element set of the merged_means list so that
                # we only use that list from now on for data assignment.
                for i in range(len(model_means)):
                    if not [x for x in merged_means if i in x]:
                        s = [i]
                        merged_means.append(s)
                self.pf.merged_means[cell_] = merged_means

                # Compute merged Gaussian parameters per place field
                cell_merged_mu = []
                cell_merged_cov = []
                cell_merged_w = []
                for comp_idxs in merged_means:
                    w = model_.weights_[comp_idxs]
                    w_sum = w.sum()
                    w_norm = w / w_sum
                    mu = w_norm @ model_.means_[comp_idxs]
                    cov = np.zeros((2, 2))
                    for ci, wi in zip(comp_idxs, w_norm):
                        diff = model_.means_[ci] - mu
                        cov += wi * (model_.covariances_[ci]
                                     + np.outer(diff, diff))
                    cell_merged_mu.append(mu)
                    cell_merged_cov.append(cov)
                    cell_merged_w.append(float(w_sum))
                self.pf.merged_means_[cell_] = cell_merged_mu
                self.pf.merged_covariances_[cell_] = cell_merged_cov
                self.pf.merged_weights_[cell_] = cell_merged_w

                # Find place fields based on merged Gaussians and assign significant responses to them using
                # Gaussian model prediction of mean, taking into consideration merged ones.
                responses_coords = [[r[0],r[1]] for r in responses]
                predicted_means = model_.predict(responses_coords)
                assigned_means = []
                for mean in predicted_means:
                    assigned_mean = [i for i,x in zip(range(len(merged_means)),merged_means) if mean in x]
                    assigned_means.append(assigned_mean[0])
                self.pf.responses_means[cell_] = assigned_means

                # Determine place field "compactness" and calculate sums and means of in-field and out-of-field S activities
                self.pf.responses_pf[cell_] = []
                self.pf.pf_size[cell_] = []
                self.pf.compactness_pf[cell_] = []
                self.pf.infield_mean[cell_] = []
                self.pf.infield_sum_[cell_] = []
                for i in range(len(merged_means)):
                    responses_pf = [pf for mean,pf in zip(assigned_means,responses_coords) if mean==i]
                    if responses_pf: # possible that all responses assigned to one mean, leaving the other empty
                        self.pf.responses_pf[cell_].append(responses_pf)
                    else:
                        print('*** Oops: Gaussian mean {} to be merged into nothingness..'.format(i))
                pf_ranges_y = []
                pf_ranges_x = []
                pf_area_tot = 0
                for pf in self.pf.responses_pf[cell_]:
                    y_coords = [p[0] for p in pf]
                    x_coords = [p[1] for p in pf]
                    min_y = np.min(y_coords)
                    max_y = np.max(y_coords)
                    min_x = np.min(x_coords)
                    max_x = np.max(x_coords)
                    print(min_y, max_y, min_x, max_x)
                    pf_ranges_y.append(range(min_y,max_y+1))
                    pf_ranges_x.append(range(min_x,max_x+1))

                    pf_area = (max_y - min_y + 1) * (max_x - min_x + 1)
                    print(pf_area)
                    pf_area_tot += pf_area
                    compactness = len(pf) / pf_area
                    self.pf.compactness_pf[cell_].append(compactness)
                    self.pf.pf_size[cell_].append(pf_area)

                    fm_subset = fm[min_y:max_y+1, min_x:max_x+1, cell_]
                    self.pf.infield_mean[cell_].append(np.mean(fm_subset))
                    self.pf.infield_sum_[cell_].append(np.sum(fm_subset))

                # Calculate place field spatial selectivity (we can't do this in the above loop since need to have 
                # the mean in-field activities for all pf's already calculated beforehand)
                outfield_sum = np.sum(fm[:,:,cell_]) - np.sum(self.pf.infield_sum_[cell_])
                self.pf.outfield_mean[cell_] = outfield_sum / ((fm.shape[0] * fm.shape[1]) - pf_area_tot)
                #self.spatial_selectivity[cell_] = self.infield_mean[cell_] / self.outfield_mean[cell_]
                self.pf.spatial_selectivity[cell_] = self.pf.infield_sum_[cell_] / np.sum(fm[:,:,cell_])

                # Plot assigned place field(s)
                color_iter = itertools.cycle(["k", "w", "b", "g", "r"])
                marker_iter = itertools.cycle(["+", "x", "o", "v", "^", "<", ">"])
                means_to_colours = {}
                means_to_markers = {}
                for m,c in zip(range(len(merged_means)), color_iter):
                    means_to_colours[m] = c
                for mean,marker in zip(range(len(merged_means)), marker_iter):
                    means_to_markers[mean] = marker
                plt.figure()
                plt.imshow(fm[:,:,cell_], cmap='viridis')
                for coord, mean in zip(responses_coords, assigned_means):
                    plt.scatter(coord[1], coord[0], marker=means_to_markers[mean], color=means_to_colours[mean])
                plt.title('cell {} place fields'.format(cell_))

                for mean_coords, spatial_sel in zip(self.pf.model_[cell_].means_, self.pf.spatial_selectivity[cell_]):
                    plt.text(mean_coords[1], mean_coords[0], round(spatial_sel,2), c='w')
                plt.savefig(os.path.join(save_path, '{}_cell_{}_place_fields.png'.format(m_, cell_)), format='png', dpi=300)
                plt.close()

                # Once all is done, save place fields data to pickle. IMPORTANT!
                self.pf.save_data()


# ==============================================================================
# ====  2D Population Vector (PV) Correlation across sessions                 ==
# ==============================================================================


def _infer_dt_sec_spatial(sess):
    """Infer frame duration in seconds from session timestamps (ms)."""
    if hasattr(sess, "tstamp_miniscope") and sess.tstamp_miniscope is not None and len(sess.tstamp_miniscope) > 10:
        ts = np.asarray(sess.tstamp_miniscope, dtype=float)
        d = np.diff(ts)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            return float(np.median(d)) / 1000.0
    return 1.0 / MINISCOPE_FPS


def _build_pretone_mask(sess, first_n_sec=180.0):
    """
    Build a boolean mask for the pre-tone period (first *first_n_sec* seconds).
    For TFC_cond: uses tone_onsets[0] if available.
    For Test_A/Test_A_1wk: no tones, uses first_n_sec directly.
    For Test_B/Test_B_1wk: uses tone_onsets[0] if available.
    No speed filtering — occupancy itself handles spatial sampling.
    """
    T = sess.S.shape[1]
    mask = np.zeros(T, dtype=bool)
    if hasattr(sess, 'tone_onsets') and len(sess.tone_onsets) > 0:
        last_frame = min(int(sess.tone_onsets[0]), T)
    else:
        dt = _infer_dt_sec_spatial(sess)
        last_frame = min(int(first_n_sec / dt), T)
    mask[:last_frame] = True
    return mask


def _get_crossreg_cells(sess_a, sess_b, mapping):
    """
    Get cross-registered cell unit IDs for two sessions using the provided mapping.
    Searches crossreg and crossreg_full on both sessions.

    Returns
    -------
    (cells_a, cells_b) : lists of int unit IDs, paired element-wise.
    Returns ([], []) if no mapping found.
    """
    candidates = []
    seen = set()
    for attr in ('crossreg', 'crossreg_full'):
        for sess in (sess_a, sess_b):
            c = getattr(sess, attr, None)
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                candidates.append(c)

    crossreg = None
    df_mapping = None
    for candidate in candidates:
        try:
            df_mapping = candidate.get_mappings_cells(mapping_type=mapping)
            crossreg = candidate
            break
        except (KeyError, Exception):
            continue

    if crossreg is None:
        # Try pairwise fallback
        type_a = sess_a.session_type
        type_b = sess_b.session_type
        pair_mapping = "+".join(sorted([type_a, type_b]))
        for candidate in candidates:
            try:
                df_mapping = candidate.get_mappings_cells(mapping_type=pair_mapping)
                crossreg = candidate
                break
            except (KeyError, Exception):
                continue

    if crossreg is None:
        return [], []

    col_a = sess_a.get_df_col(with_crossreg=crossreg)
    col_b = sess_b.get_df_col(with_crossreg=crossreg)

    # Keep only rows where both sessions have a valid cell ID
    df_valid = df_mapping[[col_a, col_b]].dropna()
    cells_a = df_valid[col_a].astype(float).astype(int).tolist()
    cells_b = df_valid[col_b].astype(float).astype(int).tolist()
    return cells_a, cells_b


def _get_mapped_S(sess, unit_ids):
    """Convert unit_id list to S subset (n_cells, T).

    Silently drops IDs not present in sess.S_idx.
    """
    s_idx_set = set(np.asarray(sess.S_idx).ravel().tolist())
    unit_ids = [int(u) for u in unit_ids
                if u is not None and np.isfinite(u) and int(u) in s_idx_set]
    if len(unit_ids) == 0:
        return np.zeros((0, sess.S.shape[1]))
    inds = sess.get_S_indeces(unit_ids)
    return sess.S[inds, :]


_VALID_Z_SCORE_MODES = ("none", "per-session", "across-sessions")


def _validate_use_z_score(use_z_score: str) -> str:
    mode = str(use_z_score).strip().lower()
    if mode not in _VALID_Z_SCORE_MODES:
        raise ValueError(
            f"Invalid use_z_score='{use_z_score}'. "
            f"Expected one of {_VALID_Z_SCORE_MODES}."
        )
    return mode


def _zscore_rows_per_session(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    mu = np.nanmean(S, axis=1, keepdims=True)
    sd = np.nanstd(S, axis=1, keepdims=True)
    bad = (~np.isfinite(sd)) | (sd <= 0)
    sd_safe = sd.copy()
    sd_safe[bad] = 1.0
    Z = (S - mu) / sd_safe
    if np.any(bad):
        Z[bad[:, 0], :] = 0.0
    return Z


def _zscore_rows_across_sessions(S1: np.ndarray, S2: np.ndarray):
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    if S1.shape[0] != S2.shape[0]:
        n = min(S1.shape[0], S2.shape[0])
        S1 = S1[:n]
        S2 = S2[:n]
    S_all = np.concatenate([S1, S2], axis=1)
    mu = np.nanmean(S_all, axis=1, keepdims=True)
    sd = np.nanstd(S_all, axis=1, keepdims=True)
    bad = (~np.isfinite(sd)) | (sd <= 0)
    sd_safe = sd.copy()
    sd_safe[bad] = 1.0
    Z1 = (S1 - mu) / sd_safe
    Z2 = (S2 - mu) / sd_safe
    if np.any(bad):
        Z1[bad[:, 0], :] = 0.0
        Z2[bad[:, 0], :] = 0.0
    return Z1, Z2


def _apply_zscore_pair(S1: np.ndarray, S2: np.ndarray, use_z_score: str):
    mode = _validate_use_z_score(use_z_score)
    if mode == "none":
        return S1, S2
    if mode == "per-session":
        return _zscore_rows_per_session(S1), _zscore_rows_per_session(S2)
    return _zscore_rows_across_sessions(S1, S2)


def _z_score_dir_tag(use_z_score: str) -> str:
    """Return a directory-name token for the z-score mode."""
    mode = _validate_use_z_score(use_z_score)
    if mode == "per-session":
        return "z-score_per"
    if mode == "across-sessions":
        return "z-score_across"
    return "z-score_none"


def build_2D_rate_maps(S, x, y, mask, n_bins, smooth_sigma=1.0, min_occupancy_frames=4):
    """
    Build occupancy-corrected, smoothed 2D rate maps for all neurons.

    Parameters
    ----------
    S : (N, T) array — neural activity (deconvolved spikes).
    x, y : (T,) arrays — position coordinates, normalized to [0, 1].
    mask : (T,) bool — which frames to include.
    n_bins : int — grid resolution (n_bins × n_bins).
    smooth_sigma : float — Gaussian smoothing sigma in bin units.
    min_occupancy_frames : int — minimum frames per bin to be considered valid.

    Returns
    -------
    rate_maps : (N, n_bins, n_bins) — occupancy-corrected rate map per neuron.
    occupancy : (n_bins, n_bins) — frame count per bin (before smoothing).
    valid_mask : (n_bins, n_bins) bool — bins with sufficient occupancy.
    """
    N = S.shape[0]
    T_mask = np.sum(mask)
    if T_mask == 0:
        return (np.full((N, n_bins, n_bins), np.nan),
                np.zeros((n_bins, n_bins)),
                np.zeros((n_bins, n_bins), dtype=bool))

    # Apply mask
    x_m = x[mask]
    y_m = y[mask]
    S_m = S[:, mask]

    # Bin positions — clip to [0, 1-eps] so digitize stays in range
    x_m = np.clip(x_m, 0.0, 1.0 - 1e-9)
    y_m = np.clip(y_m, 0.0, 1.0 - 1e-9)
    bin_x = (x_m * n_bins).astype(int)
    bin_y = (y_m * n_bins).astype(int)

    # Accumulate activity and occupancy
    activity_maps = np.zeros((N, n_bins, n_bins))
    occupancy = np.zeros((n_bins, n_bins))

    for t in range(T_mask):
        bx, by = bin_x[t], bin_y[t]
        occupancy[by, bx] += 1
        activity_maps[:, by, bx] += S_m[:, t]

    # Smooth activity and occupancy before division
    occ_smooth = gaussian_filter(occupancy.astype(float), sigma=smooth_sigma)
    activity_smooth = np.zeros_like(activity_maps)
    for n in range(N):
        activity_smooth[n] = gaussian_filter(activity_maps[n].astype(float), sigma=smooth_sigma)

    # Valid bins based on raw occupancy
    valid_mask = occupancy >= min_occupancy_frames

    # Divide: rate = smoothed_activity / smoothed_occupancy
    rate_maps = np.full((N, n_bins, n_bins), np.nan)
    for n in range(N):
        with np.errstate(divide='ignore', invalid='ignore'):
            rm = activity_smooth[n] / occ_smooth
        rm[~valid_mask] = np.nan
        rate_maps[n] = rm

    return rate_maps, occupancy, valid_mask


def compute_2D_pv_correlation(
    sess_dict_1,
    sess_dict_2,
    mouse_groups,
    mapping,
    PLOTS_DIR,
    sess_1_label="TFC_cond",
    sess_2_label="Test_A",
    n_bins=10,
    smooth_sigma=1.0,
    min_occupancy_frames=4,
    first_n_sec=180.0,
    auto_close=True,
    use_z_score="none",
):
    """
    Compute 2D population vector correlation between two session types
    across all mice, using pre-tone data only.

    For each mouse:
      1. Normalize x, y coordinates to [0, 1] using combined bounds.
      2. Build 2D rate maps for all cross-registered neurons in each session.
      3. Compute same-bin PV correlation (raw + z-scored).
      4. Compute full bin×bin PV correlation matrix (raw + z-scored).
      5. Extract summary metrics.

    Parameters
    ----------
    sess_dict_1 : dict[str, Session] — {mouse: session} for session 1.
    sess_dict_2 : dict[str, Session] — {mouse: session} for session 2.
    mouse_groups : dict[str, str] — {mouse: group_label}.
    mapping : str — cross-registration mapping string.
    PLOTS_DIR : str — output directory for plots.
    sess_1_label, sess_2_label : str — labels for the two session types.
    n_bins : int — grid resolution.
    smooth_sigma : float — Gaussian smoothing in bin units.
    min_occupancy_frames : int — minimum frames in a bin to be valid.
    first_n_sec : float — pre-tone duration in seconds.
    auto_close : bool — close matplotlib figures after saving.

    Returns
    -------
    results : dict — keyed by mouse, containing all PV correlation metrics.
    """
    save_dir = os.path.join(PLOTS_DIR, f"PV_2D_{sess_1_label}_vs_{sess_2_label}_bins_{n_bins}_{_z_score_dir_tag(use_z_score)}")
    os.makedirs(save_dir, exist_ok=True)

    mice = sorted(set(sess_dict_1.keys()) & set(sess_dict_2.keys()))
    results = {}

    for mouse in mice:
        s1 = sess_dict_1[mouse]
        s2 = sess_dict_2[mouse]
        group = mouse_groups.get(mouse, "NA")

        # Get cross-registered cells
        cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping)
        if len(cells_1) == 0:
            print(f"[PV-2D] {mouse} ({group}): no cross-registered cells for "
                  f"{sess_1_label}→{sess_2_label}, skipping.")
            continue

        S1 = _get_mapped_S(s1, cells_1)
        S2 = _get_mapped_S(s2, cells_2)
        n_cells = min(S1.shape[0], S2.shape[0])
        S1 = S1[:n_cells]
        S2 = S2[:n_cells]

        S1, S2 = _apply_zscore_pair(S1, S2, use_z_score)

        if n_cells < 3:
            print(f"[PV-2D] {mouse} ({group}): only {n_cells} cells, skipping.")
            continue

        # Position data
        x1 = np.asarray(s1.loc_X_miniscope_smooth, dtype=float)[:S1.shape[1]]
        y1 = np.asarray(s1.loc_Y_miniscope_smooth, dtype=float)[:S1.shape[1]]
        x2 = np.asarray(s2.loc_X_miniscope_smooth, dtype=float)[:S2.shape[1]]
        y2 = np.asarray(s2.loc_Y_miniscope_smooth, dtype=float)[:S2.shape[1]]

        # Normalize to [0, 1] using combined bounds across both sessions
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
        y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        x1_n = (x1 - x_min) / x_range
        y1_n = (y1 - y_min) / y_range
        x2_n = (x2 - x_min) / x_range
        y2_n = (y2 - y_min) / y_range

        # Pre-tone masks
        mask1 = _build_pretone_mask(s1, first_n_sec=first_n_sec)[:S1.shape[1]]
        mask2 = _build_pretone_mask(s2, first_n_sec=first_n_sec)[:S2.shape[1]]

        if np.sum(mask1) < 20 or np.sum(mask2) < 20:
            print(f"[PV-2D] {mouse} ({group}): insufficient pre-tone frames, skipping.")
            continue

        # Build rate maps
        rm1, occ1, valid1 = build_2D_rate_maps(
            S1, x1_n, y1_n, mask1, n_bins,
            smooth_sigma=smooth_sigma,
            min_occupancy_frames=min_occupancy_frames,
        )
        rm2, occ2, valid2 = build_2D_rate_maps(
            S2, x2_n, y2_n, mask2, n_bins,
            smooth_sigma=smooth_sigma,
            min_occupancy_frames=min_occupancy_frames,
        )

        # Joint valid bins (both sessions must have sufficient occupancy)
        joint_valid = valid1 & valid2
        n_valid_bins = np.sum(joint_valid)

        if n_valid_bins < 3:
            print(f"[PV-2D] {mouse} ({group}): only {n_valid_bins} jointly valid bins, skipping.")
            continue

        # ---- Build PV arrays: (n_valid_bins, n_cells) ----
        valid_ij = np.argwhere(joint_valid)  # (n_valid_bins, 2) — [row(y), col(x)]
        pv1_raw = np.zeros((n_valid_bins, n_cells))
        pv2_raw = np.zeros((n_valid_bins, n_cells))
        for idx, (bi, bj) in enumerate(valid_ij):
            pv1_raw[idx, :] = rm1[:, bi, bj]
            pv2_raw[idx, :] = rm2[:, bi, bj]

        # Z-scored version: z-score each neuron across valid bins
        pv1_z = np.copy(pv1_raw)
        pv2_z = np.copy(pv2_raw)
        for c in range(n_cells):
            mu1, sd1 = np.nanmean(pv1_raw[:, c]), np.nanstd(pv1_raw[:, c])
            mu2, sd2 = np.nanmean(pv2_raw[:, c]), np.nanstd(pv2_raw[:, c])
            pv1_z[:, c] = (pv1_raw[:, c] - mu1) / sd1 if sd1 > 0 else 0.0
            pv2_z[:, c] = (pv2_raw[:, c] - mu2) / sd2 if sd2 > 0 else 0.0

        # ---- Same-bin PV correlation ----
        def _same_bin_corr(pv_a, pv_b):
            """Pearson r for each bin's PV across neurons."""
            n_b = pv_a.shape[0]
            r_vals = np.full(n_b, np.nan)
            for b in range(n_b):
                v1, v2 = pv_a[b], pv_b[b]
                finite = np.isfinite(v1) & np.isfinite(v2)
                if np.sum(finite) >= 3:
                    std1, std2 = np.std(v1[finite]), np.std(v2[finite])
                    if std1 > 0 and std2 > 0:
                        r_vals[b], _ = pearsonr(v1[finite], v2[finite])
            return r_vals

        samebin_r_raw = _same_bin_corr(pv1_raw, pv2_raw)
        samebin_r_z = _same_bin_corr(pv1_z, pv2_z)

        # ---- Full bin×bin PV correlation matrix ----
        def _full_pv_corr_matrix(pv_a, pv_b):
            """M×M correlation matrix between all bin pairs."""
            M = pv_a.shape[0]
            corr_mat = np.full((M, M), np.nan)
            for a in range(M):
                for b in range(M):
                    v1, v2 = pv_a[a], pv_b[b]
                    finite = np.isfinite(v1) & np.isfinite(v2)
                    if np.sum(finite) >= 3:
                        std1, std2 = np.std(v1[finite]), np.std(v2[finite])
                        if std1 > 0 and std2 > 0:
                            corr_mat[a, b], _ = pearsonr(v1[finite], v2[finite])
            return corr_mat

        corr_mat_raw = _full_pv_corr_matrix(pv1_raw, pv2_raw)
        corr_mat_z = _full_pv_corr_matrix(pv1_z, pv2_z)

        # ---- Metrics ----
        def _compute_metrics(samebin_r, corr_mat):
            valid_same = samebin_r[np.isfinite(samebin_r)]
            diag = np.diag(corr_mat)
            valid_diag = diag[np.isfinite(diag)]

            # Off-diagonal
            off_diag_mask = ~np.eye(corr_mat.shape[0], dtype=bool)
            off_vals = corr_mat[off_diag_mask]
            valid_off = off_vals[np.isfinite(off_vals)]

            metrics = {}
            # Same-bin metrics
            metrics['mean_samebin_r'] = np.nanmean(valid_same) if len(valid_same) > 0 else np.nan
            metrics['median_samebin_r'] = np.nanmedian(valid_same) if len(valid_same) > 0 else np.nan
            metrics['frac_positive_samebin'] = np.mean(valid_same > 0) if len(valid_same) > 0 else np.nan

            # Full-matrix metrics
            metrics['mean_diag_r'] = np.nanmean(valid_diag) if len(valid_diag) > 0 else np.nan
            metrics['mean_offdiag_r'] = np.nanmean(valid_off) if len(valid_off) > 0 else np.nan
            metrics['diag_minus_offdiag'] = metrics['mean_diag_r'] - metrics['mean_offdiag_r']

            # Best-match analysis
            n_m = corr_mat.shape[0]
            best_match_is_same = 0
            displacement_bins = []
            for a in range(n_m):
                row = corr_mat[a, :]
                if np.all(np.isnan(row)):
                    continue
                best_b = np.nanargmax(row)
                if best_b == a:
                    best_match_is_same += 1
                # Displacement in 2D bin coordinates
                yi_a, xi_a = valid_ij[a]
                yi_b, xi_b = valid_ij[best_b]
                disp = np.sqrt((yi_a - yi_b)**2 + (xi_a - xi_b)**2)
                displacement_bins.append(disp)

            n_with_valid_row = sum(1 for a in range(n_m) if not np.all(np.isnan(corr_mat[a, :])))
            metrics['frac_best_match_same_bin'] = (
                best_match_is_same / n_with_valid_row if n_with_valid_row > 0 else np.nan
            )
            metrics['mean_best_match_displacement'] = (
                np.mean(displacement_bins) if len(displacement_bins) > 0 else np.nan
            )

            return metrics

        metrics_raw = _compute_metrics(samebin_r_raw, corr_mat_raw)
        metrics_z = _compute_metrics(samebin_r_z, corr_mat_z)

        results[mouse] = {
            'group': group,
            'n_cells': n_cells,
            'n_valid_bins': int(n_valid_bins),
            'valid_ij': valid_ij,
            'samebin_r_raw': samebin_r_raw,
            'samebin_r_z': samebin_r_z,
            'corr_mat_raw': corr_mat_raw,
            'corr_mat_z': corr_mat_z,
            'metrics_raw': metrics_raw,
            'metrics_z': metrics_z,
            'rate_maps_1': rm1,
            'rate_maps_2': rm2,
            'occupancy_1': occ1,
            'occupancy_2': occ2,
            'joint_valid': joint_valid,
        }

        print(f"[PV-2D] {mouse} ({group}): {n_cells} cells, {n_valid_bins} valid bins | "
              f"raw mean_same={metrics_raw['mean_samebin_r']:.3f}  "
              f"diag-off={metrics_raw['diag_minus_offdiag']:.3f} | "
              f"z   mean_same={metrics_z['mean_samebin_r']:.3f}  "
              f"diag-off={metrics_z['diag_minus_offdiag']:.3f}")

    return results


# ---------------------------------------------------------------------------
#  Paper-style constants (match SSTCa2_decoder.py paradigm plots)
# ---------------------------------------------------------------------------
_PV_GROUP_ORDER = ["mCherry", "hM3D", "hM4D"]
_PV_GROUP_LABELS = {"mCherry": "Ctl", "hM3D": "Exc", "hM4D": "Inh"}
_PV_BOX_COLORS = {
    "mCherry": "#4a4a4a",   # dark grey
    "hM3D":    "#c45a5a",   # muted red
    "hM4D":    "#5b8db8",   # muted blue
}
_PV_DOT_COLORS = {
    "mCherry": "#7a7a7a",
    "hM3D":    "#d98e8e",
    "hM4D":    "#8fb5d5",
}
_PV_BOX_ALPHA = 0.30
_PV_SCATTER_SIZE = 24


def _pv_p_to_star(p_holm, p_raw):
    """Convert p-value to significance star string."""
    if p_holm < 0.001:
        return "***", 9
    elif p_holm < 0.01:
        return "**", 9
    elif p_holm < 0.05:
        return "*", 9
    elif 0.05 <= p_raw <= 0.08:
        return f"p={p_raw:.2f}", 6.5
    return None, None


def _pv_draw_bracket(ax, x1, x2, by, bdy, y_rng, star, fs, color="#333333"):
    """Draw a significance bracket above boxplots."""
    ax.plot([x1, x1, x2, x2], [by, by + bdy, by + bdy, by],
            color=color, lw=0.7, clip_on=False)
    ax.text((x1 + x2) / 2, by + bdy + 0.005 * y_rng, star,
            ha="center", va="bottom", fontsize=fs, color=color,
            fontweight="bold")


def _pv_paper_boxplot(ax, groups_data, ylabel, title):
    """
    Draw a single paper-quality boxplot panel with ANOVA + Holm-corrected
    pairwise t-tests.

    Parameters
    ----------
    ax : matplotlib Axes
    groups_data : dict[str, array] — {group_name: values_array}
    ylabel, title : str
    """
    from scipy.stats import f_oneway, ttest_ind
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations

    grp_order = [g for g in _PV_GROUP_ORDER if g in groups_data and len(groups_data[g]) > 0]
    n_groups = len(grp_order)
    if n_groups == 0:
        ax.set_title(title, fontsize=9)
        return

    bw = 0.60
    bp_kw = dict(patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", lw=1.2),
                 whiskerprops=dict(color="black", lw=0.6),
                 capprops=dict(color="black", lw=0.6))

    for gi, grp in enumerate(grp_order):
        vals = groups_data[grp]
        if len(vals) == 0:
            continue
        bp = ax.boxplot([vals], positions=[gi], widths=bw, **bp_kw)
        for patch in bp["boxes"]:
            patch.set_facecolor(_PV_BOX_COLORS[grp])
            patch.set_alpha(_PV_BOX_ALPHA)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        # Scatter with jitter
        jit = np.random.default_rng(42 + gi * 100).uniform(
            -bw * 0.18, bw * 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), gi) + jit, vals,
                   color=_PV_DOT_COLORS.get(grp, "gray"),
                   s=_PV_SCATTER_SIZE, zorder=5,
                   edgecolors="white", linewidths=0.3)

    # Axis formatting
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([_PV_GROUP_LABELS.get(g, g) for g in grp_order], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.6)

    # ---- ANOVA + post-hoc ----
    arrays = [groups_data[g] for g in grp_order if len(groups_data[g]) >= 2]
    if len(arrays) < 2:
        return

    F_stat, p_anova = f_oneway(*arrays)

    # Pairwise t-tests with Holm correction
    pairs = list(combinations(range(n_groups), 2))
    raw_ps = []
    pair_info = []
    for g1i, g2i in pairs:
        v1 = groups_data[grp_order[g1i]]
        v2 = groups_data[grp_order[g2i]]
        if len(v1) >= 2 and len(v2) >= 2:
            _, p = ttest_ind(v1, v2)
        else:
            p = 1.0
        raw_ps.append(p)
        pair_info.append((g1i, g2i, p))

    if raw_ps:
        _, p_holm, _, _ = multipletests(raw_ps, method="holm")
    else:
        p_holm = np.array([])

    # Draw brackets for significant pairs
    all_vals = np.concatenate([groups_data[g] for g in grp_order if len(groups_data[g]) > 0])
    y_max = np.nanmax(all_vals)
    y_min = np.nanmin(all_vals)
    y_rng = max(y_max - y_min, 0.01)
    bdy = 0.03 * y_rng
    bgap = 2.6 * bdy
    by = y_max + 0.04 * y_rng

    for k, (g1i, g2i, p_raw) in enumerate(pair_info):
        ph = float(p_holm[k]) if k < len(p_holm) else 1.0
        star, fs = _pv_p_to_star(ph, p_raw)
        if star is None:
            continue
        _pv_draw_bracket(ax, g1i, g2i, by, bdy, y_rng, star, fs)
        by += bgap

    # Print ANOVA result as text annotation
    anova_str = f"F={F_stat:.2f}, p={p_anova:.3g}"
    ax.text(0.02, 0.98, anova_str, transform=ax.transAxes,
            fontsize=6, va="top", ha="left", color="#555555")


# ---------------------------------------------------------------------------
#  Shared PF turnover plotting helpers (used by both TFC and LT pipelines)
# ---------------------------------------------------------------------------

_TURNOVER_CAT_ORDER = ["stable-same", "stable-reduced", "stable-expanded",
                       "gained", "lost", "silent"]
# Nature-style minimal color palette with hatching support
_TURNOVER_CAT_COLORS = {
    "stable-same": (224/255.0, 224/255.0, 224/255.0),      # light gray
    "stable-reduced": (168/255.0, 168/255.0, 168/255.0),   # medium gray
    "stable-expanded": (90/255.0, 90/255.0, 90/255.0),     # dark gray
    "gained": (232/255.0, 152/255.0, 139/255.0),           # warm red/salmon
    "lost": (106/255.0, 185/255.0, 212/255.0),             # cool blue
    "silent": (245/255.0, 245/255.0, 245/255.0),           # very light gray
}

# Hatching patterns for stacked bar chart to minimize color use
_TURNOVER_CAT_HATCHES = {
    "stable-same": "",              # no hatch
    "stable-reduced": "//",         # diagonal lines
    "stable-expanded": "xxx",       # cross-hatch
    "gained": "",                   # no hatch
    "lost": "",                     # no hatch
    "silent": "..",                 # dots
}


def _count_is_stable(label):
    """Return True if *label* is any of the stable subtypes."""
    return label.startswith("stable")


def plot_turnover_metric_boxplots(results, save_dir, *, title_prefix="",
                                  auto_close=True):
    """Per-mouse boxplots for recurrence/turnover/gain/loss rates.

    Parameters
    ----------
    results : dict[mouse, dict]
        Each value must contain keys ``recurrence_prob``, ``turnover_rate``,
        ``gain_rate``, ``loss_rate``, and ``group``.
    """
    metric_defs = [
        ("recurrence_prob", "Recurrence probability",
         "P(PC in sess2 | PC in sess1)"),
        ("turnover_rate", "Turnover rate",
         "(Gained+Lost) / ever-PC"),
        ("gain_rate", "Gain rate", "Gained / N cross-reg"),
        ("loss_rate", "Loss rate", "Lost / N cross-reg"),
    ]

    for mkey, ylabel, title_suffix in metric_defs:
        groups_data = {}
        for mouse, md in results.items():
            val = md[mkey]
            if np.isfinite(val):
                groups_data.setdefault(md["group"], []).append(val)
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        if not any(len(v) >= 2 for v in groups_data.values()):
            continue

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel=ylabel,
                          title=f"{title_suffix}\n{title_prefix}")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(save_dir,
                                     f"turnover_{mkey}.{ext}"),
                        dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()


def plot_turnover_stacked_bar(group_counts, save_dir, *, title="",
                              fname_stem="turnover_stacked",
                              auto_close=True):
    """Stacked bar chart of PF turnover category fractions per group.

    Parameters
    ----------
    group_counts : dict[group_str, dict]
        Each inner dict has keys ``stable-same``, ``stable-reduced``,
        ``stable-expanded``, ``gained``, ``lost``, ``silent``, ``total``.
    """
    from scipy.stats import chi2_contingency

    groups_to_plot = [g for g in _PV_GROUP_ORDER
                      if g in group_counts]
    if len(groups_to_plot) < 2:
        return

    # Chi-squared test on contingency table
    contingency = []
    for g in groups_to_plot:
        row = [group_counts[g].get(cat, 0) for cat in _TURNOVER_CAT_ORDER]
        contingency.append(row)
    contingency = np.array(contingency)
    try:
        chi2_val, p_chi, dof, _ = chi2_contingency(contingency)
        chi_str = f"\u03c7\u00b2={chi2_val:.1f}, p={p_chi:.4g}, dof={dof}"
    except ValueError:
        chi_str = "\u03c7\u00b2: N/A"

    x = np.arange(len(groups_to_plot))
    fracs = {cat: [] for cat in _TURNOVER_CAT_ORDER}
    for g in groups_to_plot:
        tot = group_counts[g].get("total", 0)
        for cat in _TURNOVER_CAT_ORDER:
            fracs[cat].append(group_counts[g].get(cat, 0) / tot if tot > 0 else 0)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    bottom = np.zeros(len(groups_to_plot))
    for cat in _TURNOVER_CAT_ORDER:
        vals = np.array(fracs[cat])
        ax.bar(x, vals, bottom=bottom, width=0.6,
               color=_TURNOVER_CAT_COLORS[cat], edgecolor="black",
               linewidth=0.7, label=cat.capitalize(),
               hatch=_TURNOVER_CAT_HATCHES.get(cat, ""))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([_PV_GROUP_LABELS.get(g, g) for g in groups_to_plot],
                       fontsize=8)
    ax.set_ylabel("Fraction of neurons", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.05)
    ax.text(0.02, 0.98, chi_str, transform=ax.transAxes,
            fontsize=6, va="top", ha="left", color="black")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"{fname_stem}.{ext}"),
                    dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig)
    else:
        plt.show()


def _aggregate_turnover_counts(results):
    """Build group_counts dict from per-mouse turnover results.

    Parameters
    ----------
    results : dict[mouse, dict]
        Each value has keys ``n_stable_same``, ``n_stable_reduced``,
        ``n_stable_expanded``, ``n_gained``, ``n_lost``, ``n_silent``,
        ``n_crossreg``, ``group``.

    Returns
    -------
    dict[group, dict] with per-category counts and ``total``.
    """
    group_counts = {}
    for mouse, md in results.items():
        g = md["group"]
        gc = group_counts.setdefault(g, {
            "stable-same": 0, "stable-reduced": 0, "stable-expanded": 0,
            "gained": 0, "lost": 0, "silent": 0, "total": 0})
        gc["stable-same"] += md["n_stable_same"]
        gc["stable-reduced"] += md["n_stable_reduced"]
        gc["stable-expanded"] += md["n_stable_expanded"]
        gc["gained"] += md["n_gained"]
        gc["lost"] += md["n_lost"]
        gc["silent"] += md["n_silent"]
        gc["total"] += md["n_crossreg"]
    return group_counts


def plot_turnover_pooled_proportion_bars(group_labels, save_dir, *,
                                         title_prefix="",
                                         auto_close=True):
    """Per-category proportion bar charts with Wilson CI + pairwise chi-squared.

    Also plots Ziv-style pooled summary metrics (recurrence, turnover,
    gain-rate, loss-rate) as proportion bars.

    Parameters
    ----------
    group_labels : dict[group_str, list[str]]
        Per-neuron category labels (e.g. ``"stable-same"``, ``"gained"``, …)
        pooled across mice within each group.
    """
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations

    groups_to_plot = [g for g in _PV_GROUP_ORDER
                      if g in group_labels and len(group_labels[g]) > 0]
    if len(groups_to_plot) < 2:
        return

    # ---- Per-category proportion bar charts ----
    for cat, ylabel_cat in [
        ("stable-same",     "Fraction stable-same PCs"),
        ("stable-reduced",  "Fraction stable-reduced PCs"),
        ("stable-expanded", "Fraction stable-expanded PCs"),
        ("gained",          "Fraction gained PF"),
        ("lost",            "Fraction lost PF"),
        ("silent",          "Fraction silent neurons"),
    ]:
        props, cis, ns = [], [], []
        for g in groups_to_plot:
            total = len(group_labels[g])
            k = sum(1 for c in group_labels[g] if c == cat)
            p_hat = k / total if total > 0 else 0
            z = 1.96
            if total > 0:
                denom = 1 + z**2 / total
                centre = (p_hat + z**2 / (2 * total)) / denom
                half = z * np.sqrt(
                    (p_hat * (1 - p_hat) + z**2 / (4 * total)) / total
                ) / denom
            else:
                centre, half = 0, 0
            lo = max(0, centre - half)
            hi = min(1, centre + half)
            props.append(p_hat)
            cis.append((max(0, p_hat - lo), max(0, hi - p_hat)))
            ns.append(total)

        # Pairwise chi-squared (2×2) with Holm correction
        pair_pvals, pair_labels_idx = [], []
        for i, j in combinations(range(len(groups_to_plot)), 2):
            gi, gj = groups_to_plot[i], groups_to_plot[j]
            ki = sum(1 for c in group_labels[gi] if c == cat)
            kj = sum(1 for c in group_labels[gj] if c == cat)
            ni, nj = ns[i], ns[j]
            table_2x2 = np.array([[ki, ni - ki], [kj, nj - kj]])
            if ni > 0 and nj > 0 and table_2x2.min() >= 0:
                try:
                    _, pv, _, _ = chi2_contingency(table_2x2, correction=True)
                    pair_pvals.append(pv)
                except ValueError:
                    pair_pvals.append(np.nan)
            else:
                pair_pvals.append(np.nan)
            pair_labels_idx.append((i, j))

        adj_pvals = np.full(len(pair_pvals), np.nan)
        valid_mask = np.isfinite(pair_pvals)
        if np.any(valid_mask):
            _, adj_p, _, _ = multipletests(
                np.array(pair_pvals)[valid_mask], method="holm")
            adj_pvals[valid_mask] = adj_p

        x = np.arange(len(groups_to_plot))
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        bar_colors = [_PV_BOX_COLORS.get(g, "gray") for g in groups_to_plot]
        err_lo = [ci[0] for ci in cis]
        err_hi = [ci[1] for ci in cis]
        ax.bar(x, props, width=0.6, color=bar_colors, alpha=0.55,
               edgecolor="black", linewidth=0.5)
        ax.errorbar(x, props, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{_PV_GROUP_LABELS.get(g, g)}\n(n={ns[i]})"
             for i, g in enumerate(groups_to_plot)], fontsize=7)
        ax.set_ylabel(ylabel_cat, fontsize=8)
        ax.set_title(f"{ylabel_cat}\n{title_prefix}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

        # Significance brackets
        y_max = max(props) + max(err_hi) if props else 0.5
        step = 0.06
        for idx, (i, j) in enumerate(pair_labels_idx):
            pv = adj_pvals[idx]
            if np.isnan(pv):
                continue
            if pv < 0.001:
                star = "***"
            elif pv < 0.01:
                star = "**"
            elif pv < 0.05:
                star = "*"
            else:
                star = f"p={pv:.2f}"
            y_bar = y_max + step * (idx + 1)
            ax.plot([x[i], x[i], x[j], x[j]],
                    [y_bar - 0.01, y_bar, y_bar, y_bar - 0.01],
                    lw=0.8, color="black")
            ax.text((x[i] + x[j]) / 2, y_bar + 0.005, star,
                    ha="center", va="bottom", fontsize=7)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(save_dir,
                                     f"turnover_{cat}_pooled.{ext}"),
                        dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # ---- Ziv-style pooled metrics as proportion bars ----
    group_cts = {}
    for g in groups_to_plot:
        labs = group_labels[g]
        n_tot = len(labs)
        n_s = sum(1 for c in labs if _count_is_stable(c))
        n_g = sum(1 for c in labs if c == "gained")
        n_l = sum(1 for c in labs if c == "lost")
        group_cts[g] = {"total": n_tot, "stable": n_s,
                        "gained": n_g, "lost": n_l,
                        "ever_pc": n_s + n_g + n_l,
                        "rec_denom": n_s + n_l}

    ziv_defs = [
        ("recurrence",
         "Recurrence probability",
         lambda ct: ct["stable"] / ct["rec_denom"] if ct["rec_denom"] > 0 else 0,
         lambda ct: (ct["stable"], ct["rec_denom"])),
        ("turnover_rate",
         "Turnover rate",
         lambda ct: (ct["gained"] + ct["lost"]) / ct["ever_pc"] if ct["ever_pc"] > 0 else 0,
         lambda ct: (ct["gained"] + ct["lost"], ct["ever_pc"])),
        ("gain_rate",
         "Gain rate",
         lambda ct: ct["gained"] / ct["total"] if ct["total"] > 0 else 0,
         lambda ct: (ct["gained"], ct["total"])),
        ("loss_rate",
         "Loss rate",
         lambda ct: ct["lost"] / ct["total"] if ct["total"] > 0 else 0,
         lambda ct: (ct["lost"], ct["total"])),
    ]

    for mkey, ylabel, prop_fn, count_fn in ziv_defs:
        props, cis, ks_and_ns = [], [], []
        for g in groups_to_plot:
            ct = group_cts[g]
            p_hat = prop_fn(ct)
            k, n = count_fn(ct)
            z = 1.96
            if n > 0:
                denom_w = 1 + z**2 / n
                centre = (p_hat + z**2 / (2 * n)) / denom_w
                half = z * np.sqrt(
                    (p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom_w
            else:
                centre, half = 0, 0
            lo = max(0, centre - half)
            hi = min(1, centre + half)
            props.append(p_hat)
            cis.append((max(0, p_hat - lo), max(0, hi - p_hat)))
            ks_and_ns.append((k, n))

        # Pairwise chi-squared (2×2) with Holm
        pair_pvals, pair_idx = [], []
        for i, j in combinations(range(len(groups_to_plot)), 2):
            ki, ni = ks_and_ns[i]
            kj, nj = ks_and_ns[j]
            tbl = np.array([[ki, ni - ki], [kj, nj - kj]])
            if ni > 0 and nj > 0 and tbl.min() >= 0:
                try:
                    _, pv, _, _ = chi2_contingency(tbl, correction=True)
                    pair_pvals.append(pv)
                except ValueError:
                    pair_pvals.append(np.nan)
            else:
                pair_pvals.append(np.nan)
            pair_idx.append((i, j))

        adj_pv = np.full(len(pair_pvals), np.nan)
        vm = np.isfinite(pair_pvals)
        if np.any(vm):
            _, adj_p, _, _ = multipletests(
                np.array(pair_pvals)[vm], method="holm")
            adj_pv[vm] = adj_p

        x = np.arange(len(groups_to_plot))
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        bar_colors = [_PV_BOX_COLORS.get(g, "gray") for g in groups_to_plot]
        ax.bar(x, props, width=0.6, color=bar_colors, alpha=0.55,
               edgecolor="black", linewidth=0.5)
        ax.errorbar(x, props,
                    yerr=[[ci[0] for ci in cis], [ci[1] for ci in cis]],
                    fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{_PV_GROUP_LABELS.get(g, g)}\n(n={ks_and_ns[i][1]})"
             for i, g in enumerate(groups_to_plot)], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"{ylabel}\n{title_prefix}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

        # Significance brackets
        y_top = max(props) + max(ci[1] for ci in cis) if props else 0.5
        st = 0.06
        for idx2, (i, j) in enumerate(pair_idx):
            pv = adj_pv[idx2]
            if np.isnan(pv):
                continue
            if pv < 0.001:
                star = "***"
            elif pv < 0.01:
                star = "**"
            elif pv < 0.05:
                star = "*"
            else:
                star = f"p={pv:.2f}"
            yb = y_top + st * (idx2 + 1)
            ax.plot([x[i], x[i], x[j], x[j]],
                    [yb - 0.01, yb, yb, yb - 0.01],
                    lw=0.8, color="black")
            ax.text((x[i] + x[j]) / 2, yb + 0.005, star,
                    ha="center", va="bottom", fontsize=7)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(save_dir,
                                     f"turnover_{mkey}_pooled.{ext}"),
                        dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()


def plot_2D_pv_correlation_results(
    results,
    mouse_groups,
    PLOTS_DIR,
    sess_1_label="TFC_cond",
    sess_2_label="Test_A",
    n_bins=10,
    auto_close=True,
    dir_suffix="",
):
    """
    Plot summary figures for 2D PV correlation results.

    Nature-style figures with ANOVA + Holm-corrected pairwise t-tests.
    Group order: Ctl (mCherry) — Exc (hM3D) — Inh (hM4D).
    Colors: dark grey — muted red — muted blue.

    Produces:
      1. Same-bin PV correlation boxplots by group (raw and z-scored).
      2. Full-matrix metrics boxplots by group (raw and z-scored).
      3. Per-mouse correlation matrix heatmaps.
      4. Per-mouse same-bin correlation histograms.
    """
    if not results:
        print("[PV-2D plot] No results to plot.")
        return

    save_dir = os.path.join(PLOTS_DIR, f"PV_2D_{sess_1_label}_vs_{sess_2_label}_bins_{n_bins}{dir_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Collect per-mouse metrics into grouped arrays ----
    def _gather(metric_name, norm_type):
        """Return {group: np.array of values} for a given metric."""
        out = {}
        for mouse, res in results.items():
            g = res['group']
            v = res[f'metrics_{norm_type}'][metric_name]
            if g not in out:
                out[g] = []
            if np.isfinite(v):
                out[g].append(v)
        return {g: np.array(vs) for g, vs in out.items()}

    # ---- 1. Same-bin PV correlation boxplots by group ----
    samebin_metrics = [
        ('mean_samebin_r', 'Mean same-bin r'),
        ('median_samebin_r', 'Median same-bin r'),
        ('frac_positive_samebin', 'Frac. positive'),
    ]
    for norm_type, norm_label in [('raw', 'Raw'), ('z', 'Z-scored')]:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), dpi=300)
        for ax, (mname, ylabel) in zip(axes, samebin_metrics):
            gdata = _gather(mname, norm_type)
            _pv_paper_boxplot(ax, gdata, ylabel=ylabel,
                              title=f"{ylabel}\n({norm_label})")
        fig.suptitle(f'{sess_1_label} vs {sess_2_label} — Same-bin PV corr ({norm_label})',
                     fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f'samebin_PV_boxplots_{norm_type}_{sess_1_label}_vs_{sess_2_label}.png'),
                    dpi=300, bbox_inches='tight')
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # ---- 2. Full-matrix metrics boxplots by group ----
    fullmat_metrics = [
        ('diag_minus_offdiag', 'Diag − Off-diag'),
        ('frac_best_match_same_bin', 'Frac. best = same'),
        ('mean_best_match_displacement', 'Mean disp. (bins)'),
        ('mean_diag_r', 'Mean diagonal r'),
    ]
    for norm_type, norm_label in [('raw', 'Raw'), ('z', 'Z-scored')]:
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.2), dpi=300)
        for ax, (mname, ylabel) in zip(axes, fullmat_metrics):
            gdata = _gather(mname, norm_type)
            _pv_paper_boxplot(ax, gdata, ylabel=ylabel,
                              title=f"{ylabel}\n({norm_label})")
        fig.suptitle(f'{sess_1_label} vs {sess_2_label} — Full-matrix PV ({norm_label})',
                     fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f'fullmatrix_PV_boxplots_{norm_type}_{sess_1_label}_vs_{sess_2_label}.png'),
                    dpi=300, bbox_inches='tight')
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # ---- 3. Per-mouse correlation matrix heatmaps ----
    mouse_heatmap_dir = os.path.join(save_dir, 'per_mouse_heatmaps')
    os.makedirs(mouse_heatmap_dir, exist_ok=True)

    for mouse, res in results.items():
        group = res['group']
        for norm_type, norm_label in [('raw', 'Raw'), ('z', 'Z-scored')]:
            corr_mat = res[f'corr_mat_{norm_type}']
            fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
            im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1,
                           aspect='equal', interpolation='nearest')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Pearson r', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            ax.set_xlabel(f'{sess_2_label} bin', fontsize=7)
            ax.set_ylabel(f'{sess_1_label} bin', fontsize=7)
            ax.set_title(f'{mouse} ({_PV_GROUP_LABELS.get(group, group)}) '
                         f'— {norm_label}\n'
                         f'{res["n_cells"]} cells, {res["n_valid_bins"]} bins',
                         fontsize=8)
            ax.tick_params(labelsize=6)
            fig.tight_layout()
            fig.savefig(os.path.join(mouse_heatmap_dir,
                        f'corrmat_{norm_type}_{mouse}_{group}.png'),
                        dpi=300, bbox_inches='tight')
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

    # ---- 4. Per-mouse same-bin correlation distribution histograms ----
    mouse_hist_dir = os.path.join(save_dir, 'per_mouse_samebin_hist')
    os.makedirs(mouse_hist_dir, exist_ok=True)

    for mouse, res in results.items():
        group = res['group']
        grp_color = _PV_BOX_COLORS.get(group, "#4a4a4a")
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5), dpi=300)
        for ax, norm_type, norm_label in zip(axes, ['raw', 'z'], ['Raw', 'Z-scored']):
            r_vals = res[f'samebin_r_{norm_type}']
            valid = r_vals[np.isfinite(r_vals)]
            ax.hist(valid, bins=20, color=grp_color, alpha=0.6, edgecolor='k',
                    linewidth=0.4)
            if len(valid) > 0:
                ax.axvline(np.nanmean(valid), color='red', ls='--', lw=0.8,
                           label=f'mean={np.nanmean(valid):.3f}')
                ax.axvline(np.nanmedian(valid), color='orange', ls='--', lw=0.8,
                           label=f'med={np.nanmedian(valid):.3f}')
            ax.set_xlabel('Pearson r', fontsize=7)
            ax.set_ylabel('Count', fontsize=7)
            ax.set_title(f'Same-bin ({norm_label})', fontsize=8)
            ax.legend(fontsize=5.5, frameon=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(labelsize=6)

        fig.suptitle(f'{mouse} ({_PV_GROUP_LABELS.get(group, group)}) — '
                     f'{sess_1_label} vs {sess_2_label}  '
                     f'({res["n_cells"]} cells, {res["n_valid_bins"]} bins)',
                     fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(os.path.join(mouse_hist_dir,
                    f'samebin_hist_{mouse}_{group}.png'),
                    dpi=300, bbox_inches='tight')
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # ---- 5. Print stats summary table ----
    print(f"\n{'='*70}")
    print(f"  PV-2D Stats Summary: {sess_1_label} vs {sess_2_label}")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f_oneway
    for norm_type, norm_label in [('raw', 'Raw'), ('z', 'Z-scored')]:
        print(f"\n  --- {norm_label} ---")
        for mname, mlabel in samebin_metrics + fullmat_metrics:
            gdata = _gather(mname, norm_type)
            arrays = [gdata[g] for g in _PV_GROUP_ORDER
                      if g in gdata and len(gdata[g]) >= 2]
            if len(arrays) >= 2:
                F, p = _f_oneway(*arrays)
                grp_strs = []
                for g in _PV_GROUP_ORDER:
                    if g in gdata and len(gdata[g]) > 0:
                        grp_strs.append(
                            f"{_PV_GROUP_LABELS[g]}={np.mean(gdata[g]):.3f}±{np.std(gdata[g]):.3f}"
                        )
                print(f"  {mlabel:30s}  F={F:6.2f}  p={p:.4g}  "
                      f"{'  '.join(grp_strs)}")
    print(f"{'='*70}\n")

    print(f"[PV-2D plot] Saved figures to {save_dir}")


def run_2D_pv_correlation_pipeline(
    TFC_cond_dict,
    test_session_dicts,
    mouse_groups,
    mappings,
    PLOTS_DIR,
    n_bins=10,
    smooth_sigma=1.0,
    min_occupancy_frames=4,
    first_n_sec=180.0,
    auto_close=True,
    use_z_score="none",
):
    """
    Run the full 2D PV correlation pipeline for TFC_cond vs each test session.

    Parameters
    ----------
    TFC_cond_dict : dict[str, Session] — {mouse: TFC_cond session}.
    test_session_dicts : dict[str, dict[str, Session]]
        {"Test_A": {mouse: session}, "Test_B": {mouse: session}, ...}
    mouse_groups : dict[str, str].
    mappings : dict[str, str]
        Cross-registration mapping string per test session label.
        e.g. {"Test_A": "TFC_cond+Test_A+Test_A_1wk",
               "Test_B": "TFC_cond+Test_B+Test_B_1wk", ...}
    PLOTS_DIR : str.
    n_bins : int — spatial grid resolution.
    smooth_sigma : float.
    min_occupancy_frames : int.
    first_n_sec : float — pre-tone cutoff in seconds.
    auto_close : bool.

    Returns
    -------
    all_results : dict[str, dict] — keyed by test session label.
    """
    all_results = {}

    for test_label, test_dict in test_session_dicts.items():
        mapping = mappings.get(test_label)
        if mapping is None:
            print(f"[PV-2D] No mapping provided for {test_label}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  2D PV Correlation: TFC_cond vs {test_label}")
        print(f"  mapping: {mapping}")
        print(f"  {len(set(TFC_cond_dict.keys()) & set(test_dict.keys()))} shared mice")
        print(f"  Grid: {n_bins}x{n_bins}, smooth_sigma={smooth_sigma}, "
              f"min_occ={min_occupancy_frames}, first_n_sec={first_n_sec}")
        print(f"  use_z_score: {use_z_score}")
        print(f"{'='*60}")

        pv_results = compute_2D_pv_correlation(
            TFC_cond_dict,
            test_dict,
            mouse_groups,
            mapping=mapping,
            PLOTS_DIR=PLOTS_DIR,
            sess_1_label="TFC_cond",
            sess_2_label=test_label,
            n_bins=n_bins,
            smooth_sigma=smooth_sigma,
            min_occupancy_frames=min_occupancy_frames,
            first_n_sec=first_n_sec,
            auto_close=auto_close,
            use_z_score=use_z_score,
        )

        plot_2D_pv_correlation_results(
            pv_results,
            mouse_groups,
            PLOTS_DIR=PLOTS_DIR,
            sess_1_label="TFC_cond",
            sess_2_label=test_label,
            n_bins=n_bins,
            auto_close=auto_close,
            dir_suffix=f"_{_z_score_dir_tag(use_z_score)}",
        )

        all_results[test_label] = pv_results

    return all_results


# =========================================================================
#  Temporal Δ analysis  &  mixed-model analysis
# =========================================================================

_FAMILY_PAIRS = {
    "A": ("Test_A", "Test_A_1wk"),
    "B": ("Test_B", "Test_B_1wk"),
}


def compute_pv_delta_scores(all_results, mouse_groups, metric_name="frac_best_match_same_bin",
                            norm_type="raw"):
    """
    Within-mouse temporal change score: metric(1wk) − metric(48h).

    Parameters
    ----------
    all_results : dict returned by run_2D_pv_correlation_pipeline
        Keys are "Test_A", "Test_A_1wk", "Test_B", "Test_B_1wk", etc.
    mouse_groups : dict[str, str]
    metric_name : str — metric key inside metrics_raw / metrics_z.
    norm_type : str — "raw" or "z".

    Returns
    -------
    deltas : dict[str, dict[str, dict]]
        {family: {mouse: {"group": ..., "val_48h": ..., "val_1wk": ..., "delta": ...}}}
    """
    mkey = f"metrics_{norm_type}"
    deltas = {}
    for fam, (label_48h, label_1wk) in _FAMILY_PAIRS.items():
        res_48h = all_results.get(label_48h, {})
        res_1wk = all_results.get(label_1wk, {})
        shared = sorted(set(res_48h.keys()) & set(res_1wk.keys()))
        fam_deltas = {}
        for mouse in shared:
            v48 = res_48h[mouse][mkey][metric_name]
            v1w = res_1wk[mouse][mkey][metric_name]
            if np.isfinite(v48) and np.isfinite(v1w):
                fam_deltas[mouse] = {
                    "group": mouse_groups.get(mouse, "NA"),
                    "val_48h": float(v48),
                    "val_1wk": float(v1w),
                    "delta": float(v1w - v48),
                }
        deltas[fam] = fam_deltas
    return deltas


def plot_pv_delta_scores(deltas, PLOTS_DIR, n_bins=10, metric_label="Frac best=same",
                         norm_label="raw", auto_close=True):
    """
    Plot Δ(1wk − 48h) boxplots per family (A, B) with group comparison,
    using the same Nature-style panel as the other PV figures.
    """
    for fam, fam_data in deltas.items():
        if not fam_data:
            continue
        groups_data = {}
        for mouse, md in fam_data.items():
            g = md["group"]
            groups_data.setdefault(g, []).append(md["delta"])
        groups_data = {g: np.array(vs) for g, vs in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel=f"Δ {metric_label}",
                          title=f"TFC→Test_{fam}: 1wk − 48h\n({norm_label})")
        ax.axhline(0, color="gray", ls=":", lw=0.6, zorder=0)
        fig.tight_layout()

        save_dir = os.path.join(PLOTS_DIR, f"PV_2D_delta_bins_{n_bins}")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir,
                    f"delta_{norm_label}_family{fam}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"  PV-2D Delta Summary (1wk − 48h): {metric_label} ({norm_label})")
    print(f"{'='*60}")
    from scipy.stats import f_oneway as _f
    for fam, fam_data in deltas.items():
        if not fam_data:
            continue
        gd = {}
        for md in fam_data.values():
            gd.setdefault(md["group"], []).append(md["delta"])
        grp_strs = []
        arrays = []
        for g in _PV_GROUP_ORDER:
            if g in gd and len(gd[g]) > 0:
                arr = np.array(gd[g])
                arrays.append(arr)
                grp_strs.append(f"{_PV_GROUP_LABELS[g]}={np.mean(arr):.3f}±{np.std(arr):.3f}")
        if len(arrays) >= 2:
            F, p = _f(*arrays)
            print(f"  Family {fam}: F={F:.2f}  p={p:.4g}  {'  '.join(grp_strs)}")
        else:
            print(f"  Family {fam}: insufficient groups ({', '.join(grp_strs)})")
    print(f"{'='*60}\n")


def run_pv_mixed_model(all_results, mouse_groups, PLOTS_DIR, n_bins=10,
                       metric_name="frac_best_match_same_bin", norm_type="raw",
                       auto_close=True, dir_suffix=""):
    """
    Fit a linear mixed model:  metric ~ group * target * delay + (1|mouse)

    target = A vs B,  delay = 48h vs 1wk.

    Uses statsmodels MixedLM.  Saves coefficient table + type-III–like
    ANOVA table to text and also produces a grouped bar/box figure.

    Parameters
    ----------
    all_results : dict from run_2D_pv_correlation_pipeline.
    mouse_groups : dict[str, str].
    PLOTS_DIR : str.
    n_bins : int.
    metric_name : str.
    norm_type : str — "raw" or "z".
    auto_close : bool.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    mkey = f"metrics_{norm_type}"
    rows = []
    for label, target, delay in [
        ("Test_A",     "A", "48h"),
        ("Test_A_1wk", "A", "1wk"),
        ("Test_B",     "B", "48h"),
        ("Test_B_1wk", "B", "1wk"),
    ]:
        res = all_results.get(label, {})
        for mouse, rd in res.items():
            v = rd[mkey][metric_name]
            if np.isfinite(v):
                rows.append({
                    "mouse": mouse,
                    "group": mouse_groups.get(mouse, "NA"),
                    "target": target,
                    "delay": delay,
                    "value": float(v),
                })

    if len(rows) < 10:
        print("[PV-2D mixed model] Too few observations, skipping.")
        return None

    df = pd.DataFrame(rows)
    # Ensure categorical treatment with explicit reference levels
    df["group"] = pd.Categorical(df["group"], categories=_PV_GROUP_ORDER)
    df["target"] = pd.Categorical(df["target"], categories=["A", "B"])
    df["delay"] = pd.Categorical(df["delay"], categories=["48h", "1wk"])

    save_dir = os.path.join(PLOTS_DIR, f"PV_2D_mixed_model_bins_{n_bins}{dir_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Fit mixed model ----
    formula = "value ~ C(group) * C(target) * C(delay)"
    try:
        md = smf.mixedlm(formula, df, groups=df["mouse"])
        mdf = md.fit(reml=True)
    except Exception as e:
        print(f"[PV-2D mixed model]  mixedlm failed: {e}")
        print("  Falling back to OLS (no random intercept).")
        md = smf.ols(formula, df)
        mdf = md.fit()

    summary_str = str(mdf.summary())

    # ---- Type-III–like ANOVA via statsmodels ----
    anova_str = ""
    try:
        import statsmodels.api as sm
        ols_fit = smf.ols(formula, df).fit()
        anova_table = sm.stats.anova_lm(ols_fit, typ=3)
        anova_str = str(anova_table)
    except Exception as e:
        anova_str = f"(Type-III ANOVA failed: {e})"

    # Save text
    txt_path = os.path.join(save_dir,
                f"mixed_model_{norm_type}_{metric_name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Metric: {metric_name}  ({norm_type})\n")
        f.write(f"Formula: {formula} + (1|mouse)\n\n")
        f.write("="*70 + "\n  Mixed-model summary\n" + "="*70 + "\n")
        f.write(summary_str + "\n\n")
        f.write("="*70 + "\n  Type-III ANOVA (OLS, for reference)\n" + "="*70 + "\n")
        f.write(anova_str + "\n")

    print(f"\n{'='*60}")
    print(f"  Mixed-model: {metric_name} ({norm_type})")
    print(f"{'='*60}")
    print(summary_str)
    if anova_str:
        print(f"\n  Type-III ANOVA (OLS):\n{anova_str}")
    print(f"\n  Saved to {txt_path}")
    print(f"{'='*60}\n")

    # ---- Cross-group comparisons per delay (and per delay×target) ----
    from scipy.stats import f_oneway, ttest_ind
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations

    crossgroup_lines = []
    crossgroup_lines.append("\n" + "="*70)
    crossgroup_lines.append("  Cross-group comparisons (ANOVA + Holm pairwise t-tests)")
    crossgroup_lines.append("="*70)

    # Helper to run one comparison slice
    def _run_crossgroup(sub_df, slice_label):
        lines = []
        lines.append(f"\n  --- {slice_label} ---")
        grps = [g for g in _PV_GROUP_ORDER if g in sub_df["group"].values]
        if len(grps) < 2:
            lines.append("    Fewer than 2 groups present, skipping.")
            return lines
        arrays = [sub_df[sub_df["group"] == g]["value"].values for g in grps]
        arrays = [a for a in arrays if len(a) >= 2]
        if len(arrays) < 2:
            lines.append("    Fewer than 2 groups with n>=2, skipping.")
            return lines

        # Group means
        for g in grps:
            v = sub_df[sub_df["group"] == g]["value"].values
            lines.append(f"    {_PV_GROUP_LABELS.get(g, g):4s}: "
                         f"n={len(v)}, mean={np.mean(v):.4f}, "
                         f"sd={np.std(v):.4f}, "
                         f"median={np.median(v):.4f}")

        # One-way ANOVA
        grp_arrays = {g: sub_df[sub_df["group"] == g]["value"].values
                       for g in grps if len(sub_df[sub_df["group"] == g]) >= 2}
        F, p = f_oneway(*[grp_arrays[g] for g in grps if g in grp_arrays])
        lines.append(f"    ANOVA: F={F:.3f}, p={p:.4g}")

        # Pairwise t-tests with Holm correction
        pairs = list(combinations([g for g in grps if g in grp_arrays], 2))
        raw_ps = []
        pair_labels = []
        for g1, g2 in pairs:
            _, p_tt = ttest_ind(grp_arrays[g1], grp_arrays[g2])
            raw_ps.append(p_tt)
            pair_labels.append((g1, g2))

        if raw_ps:
            _, p_holm, _, _ = multipletests(raw_ps, method="holm")
            for k, (g1, g2) in enumerate(pair_labels):
                star, _ = _pv_p_to_star(p_holm[k], raw_ps[k])
                sig_str = f"  {star}" if star else ""
                lines.append(f"    {_PV_GROUP_LABELS.get(g1, g1)} vs "
                             f"{_PV_GROUP_LABELS.get(g2, g2)}: "
                             f"p_raw={raw_ps[k]:.4g}, "
                             f"p_holm={p_holm[k]:.4g}{sig_str}")
        return lines

    # Per-delay (pooled across targets)
    for delay in ["48h", "1wk"]:
        sub = df[df["delay"] == delay]
        crossgroup_lines.extend(_run_crossgroup(sub, f"Delay={delay} (pooled targets)"))

    # Per-delay × per-target
    for tgt in ["A", "B"]:
        for delay in ["48h", "1wk"]:
            sub = df[(df["target"] == tgt) & (df["delay"] == delay)]
            crossgroup_lines.extend(
                _run_crossgroup(sub, f"Target={tgt}, Delay={delay}"))

    crossgroup_lines.append("="*70 + "\n")
    crossgroup_text = "\n".join(crossgroup_lines)

    # Append to text file
    with open(txt_path, "a") as f:
        f.write(crossgroup_text)

    # Print
    print(crossgroup_text)

    # ---- Grouped box-plot: group × delay, faceted by target ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=300, sharey=True)
    for ax, tgt in zip(axes, ["A", "B"]):
        sub = df[df["target"] == tgt]
        grp_order = [g for g in _PV_GROUP_ORDER if g in sub["group"].values]
        n_grp = len(grp_order)
        delay_labels = ["48h", "1wk"]
        n_delay = len(delay_labels)
        bar_w = 0.35
        offsets = [-(bar_w / 2 + 0.02), (bar_w / 2 + 0.02)]

        for di, (dl, off) in enumerate(zip(delay_labels, offsets)):
            for gi, grp in enumerate(grp_order):
                vals = sub[(sub["group"] == grp) & (sub["delay"] == dl)]["value"].values
                if len(vals) == 0:
                    continue
                pos = gi + off
                bp = ax.boxplot([vals], positions=[pos], widths=bar_w,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=1.0),
                                whiskerprops=dict(color="black", lw=0.5),
                                capprops=dict(color="black", lw=0.5))
                alpha = _PV_BOX_ALPHA if dl == "48h" else _PV_BOX_ALPHA + 0.20
                for patch in bp["boxes"]:
                    patch.set_facecolor(_PV_BOX_COLORS.get(grp, "gray"))
                    patch.set_alpha(alpha)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(0.5)
                jit = np.random.default_rng(42 + gi * 7 + di * 31).uniform(
                    -bar_w * 0.15, bar_w * 0.15, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jit, vals,
                           color=_PV_DOT_COLORS.get(grp, "gray"),
                           s=18, zorder=5, edgecolors="white", linewidths=0.3)

        ax.set_xticks(range(n_grp))
        ax.set_xticklabels([_PV_GROUP_LABELS.get(g, g) for g in grp_order], fontsize=8)
        ax.set_title(f"Context {tgt}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_linewidth(0.6)
        ax.tick_params(labelsize=7, length=3, width=0.6)
        if tgt == "A":
            ax.set_ylabel(metric_name.replace("_", " "), fontsize=8)
        # Legend for delay
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="gray", alpha=_PV_BOX_ALPHA, label="48 h"),
                           Patch(facecolor="gray", alpha=_PV_BOX_ALPHA + 0.20, label="1 wk")],
                  fontsize=6, frameon=False, loc="upper right")

    fig.suptitle(f"Group × Delay: {metric_name} ({norm_type})", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,
                f"mixed_model_boxplot_{norm_type}_{metric_name}.png"),
                dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig)
    else:
        plt.show()

    # ---- Cross-group boxplots: per delay (pooled targets) ----
    fig_cg, axes_cg = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=300, sharey=True)
    for ax, delay in zip(axes_cg, ["48h", "1wk"]):
        sub = df[df["delay"] == delay]
        gdata = {}
        for g in _PV_GROUP_ORDER:
            v = sub[sub["group"] == g]["value"].values
            if len(v) > 0:
                gdata[g] = v
        _pv_paper_boxplot(ax, gdata,
                          ylabel=metric_name.replace("_", " "),
                          title=f"{delay} — across groups")
    fig_cg.suptitle(f"Cross-group: {metric_name} ({norm_type})", fontsize=10, y=1.02)
    fig_cg.tight_layout()
    fig_cg.savefig(os.path.join(save_dir,
                   f"crossgroup_delay_{norm_type}_{metric_name}.png"),
                   dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig_cg)
    else:
        plt.show()

    # ---- Cross-group boxplots: per delay × target (2×2) ----
    fig_dt, axes_dt = plt.subplots(2, 2, figsize=(7.2, 6.5), dpi=300, sharey=True)
    for ri, tgt in enumerate(["A", "B"]):
        for ci, delay in enumerate(["48h", "1wk"]):
            ax = axes_dt[ri, ci]
            sub = df[(df["target"] == tgt) & (df["delay"] == delay)]
            gdata = {}
            for g in _PV_GROUP_ORDER:
                v = sub[sub["group"] == g]["value"].values
                if len(v) > 0:
                    gdata[g] = v
            _pv_paper_boxplot(ax, gdata,
                              ylabel=metric_name.replace("_", " ") if ci == 0 else "",
                              title=f"Context {tgt} — {delay}")
    fig_dt.suptitle(f"Cross-group by Target×Delay: {metric_name} ({norm_type})",
                    fontsize=10, y=1.02)
    fig_dt.tight_layout()
    fig_dt.savefig(os.path.join(save_dir,
                   f"crossgroup_target_delay_{norm_type}_{metric_name}.png"),
                   dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig_dt)
    else:
        plt.show()

    return mdf


# =========================================================================
#  1. Skaggs Spatial Information (bits/spike)
# =========================================================================

def _skaggs_spatial_info(rate_map, occupancy, valid_mask):
    """
    Compute Skaggs spatial information (bits/spike) for a single neuron.

    SI = sum_i [ p_i * (r_i / r_mean) * log2(r_i / r_mean) ]

    where p_i = occupancy_i / total_occupancy (probability of being in bin i),
    r_i = firing rate in bin i, r_mean = overall mean rate.
    Only valid bins (sufficient occupancy) are used.

    Returns SI in bits/spike, or np.nan if undefined.
    """
    rm = rate_map[valid_mask]
    occ = occupancy[valid_mask]
    if len(rm) == 0 or np.all(np.isnan(rm)):
        return np.nan
    rm = np.nan_to_num(rm, nan=0.0)
    occ = np.nan_to_num(occ, nan=0.0)
    total_occ = occ.sum()
    if total_occ == 0:
        return np.nan
    p = occ / total_occ
    r_mean = np.sum(p * rm)
    if r_mean <= 0:
        return np.nan
    ratio = rm / r_mean
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(ratio > 0, np.log2(ratio), 0.0)
    si = np.sum(p * ratio * log_ratio)
    return float(si)


# ---------------------------------------------------------------------------
#  Pooled-neuron spatial information helpers
# ---------------------------------------------------------------------------
_SPATIAL_METHODS_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "analysis_methods_templates")


def _copy_si_methods_template(template_filename, dest_dir):
    """Copy a METHODS template file into *dest_dir*.  Hard-fails if missing."""
    dest_path = os.path.join(dest_dir, template_filename)
    if os.path.isfile(dest_path):
        return
    src_path = os.path.join(_SPATIAL_METHODS_TEMPLATES_DIR, template_filename)
    assert os.path.isfile(src_path), (
        f"[SI-METHODS] Template not found: {src_path}. "
        f"Create the file before running the pooled analysis.")
    shutil.copy2(src_path, dest_path)
    print(f"[SI-METHODS] Copied {template_filename} → {dest_path}")


def _pooled_lmm_fit_and_plot(
    df, save_dir, panel_label, ylabel, value_col="value", auto_close=True,
):
    """Fit per-neuron LMM ``value ~ group + (1|mouse)`` and draw a forest plot.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per neuron, with columns ``group`` (str), ``mouse`` (str),
        and ``value_col`` (float, finite).
    save_dir : str
        Directory in which to write the figure and CSV.
    panel_label : str
        Filename stem used for ``{panel_label}_lmm_forest.{png,pdf}`` and
        ``{panel_label}_lmm_coefs.csv``.
    ylabel : str
        Metric name used in the forest plot x-label
        (``Δ {ylabel} vs Ctl``).
    value_col : str
        Column name in *df* containing the per-neuron metric.
    auto_close : bool

    Returns
    -------
    mdf : statsmodels MixedLMResultsWrapper
        The fitted model. The caller can use the textual summary if desired.
    coef_df : pandas.DataFrame
        Tidy coefficient table written to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    assert value_col in df.columns, (
        f"[LMM-pooled] {panel_label}: value column '{value_col}' missing.")
    work = df[["group", "mouse", value_col]].copy()
    work = work.rename(columns={value_col: "value"})
    work = work[np.isfinite(work["value"])].copy()
    assert not work.empty, (
        f"[LMM-pooled] {panel_label}: no finite '{value_col}' rows.")
    work["group_cat"] = pd.Categorical(
        work["group"], categories=_PV_GROUP_ORDER, ordered=False)

    md = smf.mixedlm(
        "value ~ C(group_cat, Treatment('mCherry'))",
        work, groups=work["mouse"])
    # Try a cascade of optimizers — MixedLM's default lbfgs sometimes returns
    # finite fixed-effect estimates but non-finite SEs when the Hessian is
    # ill-conditioned at the solution. Accept the first fit whose group-term
    # standard errors are all finite.
    fit_attempts = [
        {"reml": True,  "method": "lbfgs"},
        {"reml": True,  "method": "bfgs"},
        {"reml": True,  "method": "powell"},
        {"reml": False, "method": "lbfgs"},
        {"reml": False, "method": "bfgs"},
    ]
    mdf = None
    fit_log = []
    for attempt in fit_attempts:
        candidate = md.fit(**attempt)
        group_terms_local = [t for t in candidate.params.index
                             if "group_cat" in str(t)]
        ses = np.array([float(candidate.bse[t]) for t in group_terms_local])
        params = np.array([float(candidate.params[t]) for t in group_terms_local])
        all_finite = bool(np.all(np.isfinite(ses)) and np.all(np.isfinite(params)))
        fit_log.append((attempt, all_finite, ses.tolist(), params.tolist()))
        if all_finite:
            mdf = candidate
            break
    assert mdf is not None, (
        f"[LMM-pooled] {panel_label}: all optimiser attempts produced "
        f"non-finite β/SE. Attempts: {fit_log}")

    # Build coefficient table for the two non-reference group fixed effects.
    coef_rows = []
    for term in mdf.params.index:
        term_str = str(term)
        if "group_cat" not in term_str:
            continue
        beta = float(mdf.params[term])
        se = float(mdf.bse[term])
        p = float(mdf.pvalues[term])
        z = beta / se if se > 0 else np.nan
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se
        # Decode reference-coded group name from term, e.g.
        # "C(group_cat, Treatment('mCherry'))[T.hM3D]" → "hM3D"
        m = re.search(r"\[T\.([^\]]+)\]", term_str)
        assert m is not None, (
            f"[LMM-pooled] {panel_label}: unable to parse group from term "
            f"'{term_str}'.")
        grp_key = m.group(1)
        coef_rows.append({
            "term": term_str,
            "group": grp_key,
            "group_label": _PV_GROUP_LABELS.get(grp_key, grp_key),
            "beta": beta,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "z": z,
            "p": p,
        })
    assert coef_rows, (
        f"[LMM-pooled] {panel_label}: no group fixed-effect terms found.")
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(
        os.path.join(save_dir, f"{panel_label}_lmm_coefs.csv"), index=False)

    # Hard-fail on non-finite β / SE / CI: indicates a near-singular fit
    # (e.g. one group has too few neurons / near-zero residual variance).
    bad = coef_df[~np.isfinite(coef_df[["beta", "se", "ci_low", "ci_high"]]).all(axis=1)]
    assert bad.empty, (
        f"[LMM-pooled] {panel_label}: non-finite LMM coefficients/SEs for "
        f"groups {list(bad['group'])}. n_neurons={len(work)}, "
        f"per-group counts={work.groupby('group').size().to_dict()}, "
        f"per-mouse counts={work.groupby('mouse').size().to_dict()}. "
        f"Coef table written to {panel_label}_lmm_coefs.csv. Investigate "
        f"upstream data (likely degenerate group/mouse coverage).")

    # Variance components: random intercept (mouse) and residual.
    cov_re = float(np.asarray(mdf.cov_re).reshape(-1)[0])
    sigma2_resid = float(mdf.scale)

    # ----- Pairwise LMM contrasts (Exc-Ctl, Inh-Ctl, Exc-Inh) -----
    # Exc-Ctl and Inh-Ctl come straight from the fixed-effect coefficients
    # (Ctl = mCherry is the reference). Exc-Inh is computed from the model
    # covariance: β(hM3D) - β(hM4D) with SE from the joint covariance matrix.
    cov_params = mdf.cov_params()
    coef_by_group = {r["group"]: r for r in coef_rows}
    pair_specs = [
        ("hM3D", "mCherry"),  # Exc vs Ctl  (direct β, SE)
        ("hM4D", "mCherry"),  # Inh vs Ctl  (direct β, SE)
        ("hM3D", "hM4D"),     # Exc vs Inh  (contrast)
    ]
    pair_rows = []
    for g_a, g_b in pair_specs:
        if g_b == "mCherry":
            assert g_a in coef_by_group, (
                f"[LMM-pooled] {panel_label}: missing fixed-effect coef for "
                f"{g_a} (vs Ctl).")
            r = coef_by_group[g_a]
            beta_d = float(r["beta"])
            se_d = float(r["se"])
            p_d = float(r["p"])
        else:
            assert g_a in coef_by_group and g_b in coef_by_group, (
                f"[LMM-pooled] {panel_label}: missing fixed-effect coef for "
                f"{g_a} or {g_b} contrast.")
            ta = coef_by_group[g_a]["term"]
            tb = coef_by_group[g_b]["term"]
            beta_d = float(mdf.params[ta] - mdf.params[tb])
            var_d = (
                float(cov_params.loc[ta, ta])
                + float(cov_params.loc[tb, tb])
                - 2.0 * float(cov_params.loc[ta, tb])
            )
            se_d = float(np.sqrt(max(var_d, 0.0)))
            z_d = beta_d / se_d if se_d > 0 else np.nan
            # two-sided normal-approx p-value
            p_d = float(2.0 * _stats_norm.sf(abs(z_d))) if np.isfinite(z_d) else np.nan
        pair_rows.append({
            "group_a": g_a,
            "group_b": g_b,
            "label_a": _PV_GROUP_LABELS.get(g_a, g_a),
            "label_b": _PV_GROUP_LABELS.get(g_b, g_b),
            "beta_diff": beta_d,
            "se_diff": se_d,
            "ci_low": beta_d - 1.96 * se_d,
            "ci_high": beta_d + 1.96 * se_d,
            "p": p_d,
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(
        os.path.join(save_dir, f"{panel_label}_lmm_pairwise.csv"), index=False)

    # ----- Violin plot with LMM-derived significance brackets -----
    grp_order_present = [g for g in _PV_GROUP_ORDER
                         if (work["group"] == g).any()]
    grp_vals = {g: work.loc[work["group"] == g, "value"].values
                for g in grp_order_present}

    fig_v, ax_v = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    positions = list(range(len(grp_order_present)))
    parts = ax_v.violinplot(
        [grp_vals[g] for g in grp_order_present],
        positions=positions, showmedians=True, showextrema=True, widths=0.60,
    )
    for pc, g in zip(parts["bodies"], grp_order_present):
        pc.set_facecolor(_PV_BOX_COLORS[g])
        pc.set_alpha(_PV_BOX_ALPHA)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2 if key == "cmedians" else 0.6)
    ax_v.set_xticks(positions)
    ax_v.set_xticklabels(
        [f"{_PV_GROUP_LABELS.get(g, g)}\nn={len(grp_vals[g])}"
         for g in grp_order_present], fontsize=8)
    ax_v.set_ylabel(ylabel, fontsize=8)
    ax_v.set_title(f"{panel_label} — LMM contrasts", fontsize=9)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    ax_v.spines["bottom"].set_linewidth(0.6)
    ax_v.spines["left"].set_linewidth(0.6)
    ax_v.tick_params(axis="both", labelsize=7, length=3, width=0.6)

    # Significance brackets, ordered by group separation (closest pair first
    # so brackets stack neatly upward).
    all_finite_v = np.concatenate(list(grp_vals.values()))
    all_finite_v = all_finite_v[np.isfinite(all_finite_v)]
    y_rng = float(np.nanmax(all_finite_v) - np.nanmin(all_finite_v))
    if y_rng == 0.0:
        y_rng = 0.01
    by = float(np.nanmax(all_finite_v)) + 0.03 * y_rng
    bdy = 0.03 * y_rng
    bgap = 2.8 * bdy

    # Pairs sorted by x-distance for clean stacking
    sortable_pairs = []
    for pr in pair_rows:
        ga, gb = pr["group_a"], pr["group_b"]
        if ga not in grp_order_present or gb not in grp_order_present:
            continue
        xa = grp_order_present.index(ga)
        xb = grp_order_present.index(gb)
        sortable_pairs.append((abs(xb - xa), pr, xa, xb))
    sortable_pairs.sort(key=lambda t: t[0])

    for _, pr, xa, xb in sortable_pairs:
        p_val = pr["p"]
        if not np.isfinite(p_val):
            continue
        star, fs = _pv_p_to_star(p_val, p_val)
        if star is None:
            continue
        xi, xj = sorted([xa, xb])
        _pv_draw_bracket(ax_v, xi, xj, by, bdy, y_rng, star, fs)
        by += bgap

    fig_v.tight_layout()
    stem_v = os.path.join(save_dir, f"{panel_label}_lmm_violin")
    fig_v.savefig(f"{stem_v}.png", dpi=300, bbox_inches="tight")
    fig_v.savefig(f"{stem_v}.pdf", dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig_v)
    else:
        plt.show()

    # ----- Forest plot -----
    nrow = len(coef_rows)
    fig, ax = plt.subplots(figsize=(3.5, 0.55 * nrow + 1.4), dpi=300)
    y_positions = np.arange(nrow)[::-1]  # top row = first non-ref group

    for yp, row in zip(y_positions, coef_rows):
        col = _PV_BOX_COLORS.get(row["group"], "#444444")
        ax.errorbar(
            row["beta"], yp,
            xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
            fmt="o", color=col, ecolor=col, elinewidth=1.0, capsize=2.5,
            markersize=5, markeredgecolor="black", markeredgewidth=0.5,
        )

    ax.axvline(0.0, color="#888888", lw=0.6, ls="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{r['group_label']}\nvs Ctl" for r in coef_rows], fontsize=7)
    ax.set_xlabel(f"Δ {ylabel} vs Ctl (LMM β)", fontsize=8)
    ax.set_title(f"{panel_label}\nLMM: value ~ group + (1|mouse)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=6, length=3, width=0.6)

    # Symmetric x-range with small pad around the CIs.
    xmin = min(r["ci_low"] for r in coef_rows)
    xmax = max(r["ci_high"] for r in coef_rows)
    span = max(xmax - xmin, 1e-9)
    ax.set_xlim(xmin - 0.10 * span, xmax + 0.10 * span)
    ax.set_ylim(-0.7, nrow - 0.3)

    fig.tight_layout()
    stem = os.path.join(save_dir, f"{panel_label}_lmm_forest")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig)
    else:
        plt.show()

    # ----- Sidecar text file with all LMM stats (kept off the figures) -----
    n_mice = int(work["mouse"].nunique())
    txt_lines = [
        f"LMM pooled analysis — {panel_label}",
        "=" * 60,
        f"Model: value ~ C(group, Treatment('mCherry')) + (1|mouse)",
        f"n_neurons = {len(work)}    n_mice = {n_mice}",
        f"per-group n: " + ", ".join(
            f"{_PV_GROUP_LABELS.get(g, g)}={int((work['group']==g).sum())}"
            for g in _PV_GROUP_ORDER if (work['group']==g).any()),
        f"σ²_mouse = {cov_re:.6g}    σ²_resid = {sigma2_resid:.6g}",
        "",
        "Fixed-effect coefficients (vs Ctl=mCherry):",
    ]
    for row in coef_rows:
        txt_lines.append(
            f"  {row['group_label']:<6s} vs Ctl: β={row['beta']:+.4f}  "
            f"SE={row['se']:.4f}  95% CI=[{row['ci_low']:+.4f}, "
            f"{row['ci_high']:+.4f}]  z={row['z']:+.3f}  p={row['p']:.4g}"
        )
    txt_lines.append("")
    txt_lines.append("Pairwise LMM contrasts (no multiple-comparison correction):")
    for pr in pair_rows:
        txt_lines.append(
            f"  {pr['label_a']:<3s} vs {pr['label_b']:<3s}: "
            f"β_diff={pr['beta_diff']:+.4f}  SE={pr['se_diff']:.4f}  "
            f"95% CI=[{pr['ci_low']:+.4f}, {pr['ci_high']:+.4f}]  "
            f"p={pr['p']:.4g}"
        )
    with open(os.path.join(save_dir, f"{panel_label}_lmm_stats.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt_lines) + "\n")

    _copy_si_methods_template("pooled_lmm_methods.txt", save_dir)
    print(f"[LMM-pooled] {panel_label}: n={len(work)} neurons, "
          f"{n_mice} mice  → {save_dir}")
    return mdf, coef_df


def _run_si_pooled_analysis(records, save_dir, panel_label, auto_close=True):
    """Pooled-neuron SI analysis: violin plot + MixedLM + KW/MW sensitivity.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain "group" (str), "mouse" (str), "si" (float).
        Non-finite SI values are dropped internally.
    save_dir : str
        Output directory; created automatically if absent.
    panel_label : str
        Short label used as filename stem and figure title
        (e.g. "SI_all_TFC_cond").
    auto_close : bool
    """
    os.makedirs(save_dir, exist_ok=True)
    assert records, f"[SI-pooled] {panel_label}: records list is empty."

    df = pd.DataFrame(records)
    df = df[np.isfinite(df["si"])].copy()
    assert not df.empty, (
        f"[SI-pooled] {panel_label}: all SI values are non-finite.")

    # Neuron-level data CSV
    df.to_csv(os.path.join(save_dir, f"{panel_label}_neurons.csv"), index=False)

    grp_order = [g for g in _PV_GROUP_ORDER if g in df["group"].values]
    assert grp_order, f"[SI-pooled] {panel_label}: no recognised groups found."
    grp_si = {g: df.loc[df["group"] == g, "si"].values for g in grp_order}

    # ---- Figure (matches _pv_paper_boxplot style) ----
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    positions = list(range(len(grp_order)))

    parts = ax.violinplot(
        [grp_si[g] for g in grp_order],
        positions=positions,
        showmedians=True,
        showextrema=True,
        widths=0.60,
    )
    for pc, g in zip(parts["bodies"], grp_order):
        pc.set_facecolor(_PV_BOX_COLORS[g])
        pc.set_alpha(_PV_BOX_ALPHA)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2 if key == "cmedians" else 0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{_PV_GROUP_LABELS.get(g, g)}\nn={len(grp_si[g])}"
         for g in grp_order],
        fontsize=8,
    )
    ax.set_ylabel("SI (bits/spike)", fontsize=8)
    ax.set_title(panel_label.replace("_", " "), fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.6)

    # ---- Statistics ----
    out_lines = []

    # Primary: MixedLM SI ~ group + (1|mouse) — fit + forest plot via shared helper.
    lmm_p_group = np.nan
    mdf, _coef_df = _pooled_lmm_fit_and_plot(
        df, save_dir, panel_label,
        ylabel="SI (bits/spike)", value_col="si",
        auto_close=auto_close,
    )
    lmm_summary = mdf.summary().as_text()
    group_terms = [t for t in mdf.pvalues.index
                   if "group_cat" in str(t)]
    if group_terms:
        lmm_p_group = float(min(mdf.pvalues[group_terms]))
    out_lines.append("=== MixedLM: SI ~ group + (1|mouse) ===")
    out_lines.append(lmm_summary)

    # Sensitivity: Kruskal-Wallis + pairwise Mann-Whitney (Holm)
    kw_H, kw_p = np.nan, np.nan
    kw_arrays = [grp_si[g] for g in grp_order if len(grp_si[g]) >= 3]
    if len(kw_arrays) >= 2:
        kw_H, kw_p = kruskal(*kw_arrays)
    out_lines.append(f"\n=== Kruskal-Wallis (pooled neurons) ===")
    out_lines.append(
        f"H={kw_H:.3f}, p={kw_p:.4g} (groups: {', '.join(grp_order)})")

    mw_pairs = []
    pair_ps = []
    for g1, g2 in itertools.combinations(grp_order, 2):
        v1, v2 = grp_si[g1], grp_si[g2]
        if len(v1) >= 3 and len(v2) >= 3:
            U, p_mw = mannwhitneyu(v1, v2, alternative="two-sided")
            mw_pairs.append((g1, g2, U, p_mw))
            pair_ps.append(p_mw)

    stat_rows = []
    if pair_ps:
        _, p_holm, _, _ = multipletests(pair_ps, method="holm")
        out_lines.append("\nPairwise Mann-Whitney U (Holm-corrected):")
        for (g1, g2, U, p_raw), ph in zip(mw_pairs, p_holm):
            out_lines.append(
                f"  {_PV_GROUP_LABELS.get(g1, g1)} vs "
                f"{_PV_GROUP_LABELS.get(g2, g2)}: "
                f"U={U:.1f}, p_raw={p_raw:.4g}, p_holm={ph:.4g}")
            stat_rows.append({
                "comparison": f"{g1} vs {g2}",
                "U": float(U),
                "p_raw": float(p_raw),
                "p_holm": float(ph),
            })

        # Significance brackets on figure
        all_finite = np.concatenate(
            [grp_si[g] for g in grp_order if len(grp_si[g]) > 0])
        all_finite = all_finite[np.isfinite(all_finite)]
        y_rng = float(np.nanmax(all_finite) - np.nanmin(all_finite))
        if y_rng == 0.0:
            y_rng = 0.01
        by = float(np.nanmax(all_finite)) + 0.03 * y_rng
        bdy = 0.03 * y_rng
        bgap = 2.8 * bdy
        for k, (g1, g2, _U, p_raw) in enumerate(mw_pairs):
            ph = float(p_holm[k])
            star, fs = _pv_p_to_star(ph, p_raw)
            if star is None:
                continue
            xi = grp_order.index(g1)
            xj = grp_order.index(g2)
            _pv_draw_bracket(ax, xi, xj, by, bdy, y_rng, star, fs)
            by += bgap

        pd.DataFrame(stat_rows).to_csv(
            os.path.join(save_dir, f"{panel_label}_stats_mw.csv"),
            index=False)

    # Annotate figure with omnibus p-values
    annot_parts = []
    if np.isfinite(lmm_p_group):
        annot_parts.append(f"LMM p={lmm_p_group:.3g}")
    if np.isfinite(kw_p):
        annot_parts.append(f"KW p={kw_p:.3g}")
    if annot_parts:
        ax.text(0.02, 0.98, "  ".join(annot_parts),
                transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#555555")

    fig.tight_layout()
    stem = os.path.join(save_dir, panel_label)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    if auto_close:
        plt.close(fig)
    else:
        plt.show()

    # Full stats text
    with open(os.path.join(save_dir, f"{panel_label}_stats.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines))

    _copy_si_methods_template("spatial_info_pooled_methods.txt", save_dir)
    print(f"[SI-pooled] {panel_label}: n={len(df)} neurons  "
          f"KW p={kw_p:.4g}  → {save_dir}")


def compute_spatial_information(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    n_bins=10, smooth_sigma=1.0, min_occupancy_frames=4,
    first_n_sec=180.0, auto_close=True,
):
    """
    Compute Skaggs spatial information (bits/spike) for every neuron in every
    session, both for **all neurons** and for **cross-registered neurons only**.

    Cross-registration groups:
      - A-family: TFC_cond + Test_A + Test_A_1wk
      - B-family: TFC_cond + Test_B + Test_B_1wk

    Parameters
    ----------
    session_dicts : dict[str, dict[str, Session]]
        {"TFC_cond": {...}, "Test_A": {...}, ...}
    mouse_groups : dict[str, str]
    mappings : dict[str, str]  — mapping string per test session label.
    PLOTS_DIR : str
    n_bins, smooth_sigma, min_occupancy_frames, first_n_sec : spatial params.
    auto_close : bool

    Returns
    -------
    si_results : dict  —  nested: {sess_label: {mouse: {"all": [...], "crossreg": [...],
                                                         "group": str}}}
    """
    save_dir = os.path.join(PLOTS_DIR, f"spatial_info_bins_{n_bins}")
    os.makedirs(save_dir, exist_ok=True)

    all_sess_labels = sorted(session_dicts.keys())
    si_results = {}

    # --- Which crossreg family does each session belong to? ---
    crossreg_families = {
        "TFC_cond":    None,   # present in both; handled per-family below
        "Test_A":      "A",
        "Test_A_1wk":  "A",
        "Test_B":      "B",
        "Test_B_1wk":  "B",
    }

    # For TFC_cond we compute crossreg for each family separately.
    # Build crossreg cell sets per mouse per family.
    tfc_dict = session_dicts.get("TFC_cond", {})

    # Pre-compute crossreg cells per family
    crossreg_cells = {}  # (family, mouse) -> list[int] cell IDs in TFC_cond or test session
    for fam, (test48, test1w) in _FAMILY_PAIRS.items():
        test48_dict = session_dicts.get(test48, {})
        mapping_str = mappings.get(test48)
        if not mapping_str:
            continue
        for mouse in sorted(set(tfc_dict.keys()) & set(test48_dict.keys())):
            cells_tfc, cells_test = _get_crossreg_cells(
                tfc_dict[mouse], test48_dict[mouse], mapping_str)
            crossreg_cells[(fam, mouse, "TFC_cond")] = cells_tfc
            crossreg_cells[(fam, mouse, test48)] = cells_test
        # For the 1wk session, crossreg uses same mapping
        test1w_dict = session_dicts.get(test1w, {})
        for mouse in sorted(set(tfc_dict.keys()) & set(test1w_dict.keys())):
            cells_tfc, cells_1w = _get_crossreg_cells(
                tfc_dict[mouse], test1w_dict[mouse], mapping_str)
            crossreg_cells[(fam, mouse, test1w)] = cells_1w
            # TFC crossreg for this family already stored (or overwrite — same mapping)
            crossreg_cells[(fam, mouse, "TFC_cond")] = cells_tfc

    for sess_label in all_sess_labels:
        sess_dict = session_dicts.get(sess_label, {})
        si_results[sess_label] = {}

        for mouse, sess in sorted(sess_dict.items()):
            group = mouse_groups.get(mouse, "NA")
            S = sess.S
            N, T = S.shape
            x = np.asarray(sess.loc_X_miniscope_smooth, dtype=float)[:T]
            y = np.asarray(sess.loc_Y_miniscope_smooth, dtype=float)[:T]

            # Normalize coords to [0,1]
            x_min, x_max = np.nanmin(x), np.nanmax(x)
            y_min, y_max = np.nanmin(y), np.nanmax(y)
            x_rng = x_max - x_min if x_max > x_min else 1.0
            y_rng = y_max - y_min if y_max > y_min else 1.0
            x_n = (x - x_min) / x_rng
            y_n = (y - y_min) / y_rng

            mask = _build_pretone_mask(sess, first_n_sec=first_n_sec)[:T]

            # All neurons
            rm_all, occ_all, valid_all = build_2D_rate_maps(
                S, x_n, y_n, mask, n_bins, smooth_sigma, min_occupancy_frames)
            si_all = np.array([_skaggs_spatial_info(rm_all[n], occ_all, valid_all)
                               for n in range(N)])

            # Cross-registered neurons
            fam = crossreg_families.get(sess_label)
            families_to_use = [fam] if fam else ["A", "B"]
            si_crossreg_by_fam = {}
            for f in families_to_use:
                cr_key = (f, mouse, sess_label)
                cr_ids = crossreg_cells.get(cr_key, [])
                if len(cr_ids) == 0:
                    si_crossreg_by_fam[f] = np.array([])
                    continue
                S_cr = _get_mapped_S(sess, cr_ids)
                rm_cr, occ_cr, valid_cr = build_2D_rate_maps(
                    S_cr, x_n, y_n, mask, n_bins, smooth_sigma, min_occupancy_frames)
                si_cr = np.array([_skaggs_spatial_info(rm_cr[n], occ_cr, valid_cr)
                                  for n in range(S_cr.shape[0])])
                si_crossreg_by_fam[f] = si_cr

            si_results[sess_label][mouse] = {
                "group": group,
                "si_all": si_all,
                "si_crossreg": si_crossreg_by_fam,
                "n_all": N,
            }

    # ---- Plot: median SI per mouse, by group, for each session ----
    from scipy.stats import kruskal

    for neuron_set, set_label in [("all", "All neurons"), ("crossreg", "Cross-reg neurons")]:
        for sess_label in all_sess_labels:
            if sess_label not in si_results:
                continue

            # Determine which families apply
            fam = crossreg_families.get(sess_label)
            families = [fam] if fam else ["A", "B"]

            for f in families:
                groups_data = {}
                for mouse, md in si_results[sess_label].items():
                    g = md["group"]
                    if neuron_set == "all":
                        vals = md["si_all"]
                    else:
                        vals = md["si_crossreg"].get(f, np.array([]))
                    valid = vals[np.isfinite(vals)] if len(vals) > 0 else np.array([])
                    if len(valid) > 0:
                        groups_data.setdefault(g, []).append(float(np.nanmedian(valid)))
                groups_data = {g: np.array(v) for g, v in groups_data.items()}

                fam_tag = f"_fam{f}" if neuron_set == "crossreg" else ""
                fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
                _pv_paper_boxplot(ax, groups_data,
                                  ylabel="Median SI (bits/spike)",
                                  title=f"{sess_label} — {set_label}{fam_tag}")
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir,
                            f"SI_{neuron_set}{fam_tag}_{sess_label}.png"),
                            dpi=300, bbox_inches="tight")
                if auto_close:
                    plt.close(fig)
                else:
                    plt.show()

    # ---- Pooled neuron analysis (neuron-level, no per-mouse averaging) ----
    pooled_save_dir = save_dir + "_pooled"
    for neuron_set, _set_label in [("all", "all"), ("crossreg", "crossreg")]:
        for sess_label in all_sess_labels:
            if sess_label not in si_results:
                continue
            fam = crossreg_families.get(sess_label)
            families = [fam] if fam else ["A", "B"]
            for f in families:
                records = []
                for mouse, md in si_results[sess_label].items():
                    g = md["group"]
                    if neuron_set == "all":
                        vals = md["si_all"]
                    else:
                        vals = md["si_crossreg"].get(f, np.array([]))
                    for v in vals:
                        if np.isfinite(v):
                            records.append(
                                {"group": g, "mouse": mouse, "si": float(v)})
                if not records:
                    continue
                fam_tag = f"_fam{f}" if neuron_set == "crossreg" else ""
                panel_label = f"SI_{neuron_set}{fam_tag}_{sess_label}"
                _run_si_pooled_analysis(
                    records, pooled_save_dir, panel_label,
                    auto_close=auto_close)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Spatial Information Summary")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for sess_label in all_sess_labels:
        if sess_label not in si_results:
            continue
        gd = {}
        for mouse, md in si_results[sess_label].items():
            g = md["group"]
            vals = md["si_all"]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                gd.setdefault(g, []).append(float(np.nanmedian(valid)))
        arrays = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrays) >= 2:
            F, p = _f(*arrays)
            grp_str = "  ".join(
                f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.3f}±{np.std(gd[g]):.3f}"
                for g in _PV_GROUP_ORDER if g in gd)
            print(f"  {sess_label:15s}  F={F:.2f}  p={p:.4g}  {grp_str}")
    print(f"{'='*70}\n")
    print(f"[SI] Saved figures to {save_dir}")

    return si_results


# =========================================================================
#  2. Place Field Stability (per-neuron rate-map correlation)
# =========================================================================

def compute_place_field_stability(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    n_bins=10, smooth_sigma=1.0, min_occupancy_frames=4,
    first_n_sec=180.0, auto_close=True,
):
    """
    For each cross-registered neuron, correlate its 2D rate map in TFC_cond
    with its rate map in each Test session (Pearson r over jointly valid bins).
    Compare the distribution of per-neuron r values across groups.

    Parameters
    ----------
    session_dicts : dict[str, dict[str, Session]]
    mouse_groups : dict[str, str]
    mappings : dict[str, str]
    PLOTS_DIR : str
    n_bins, smooth_sigma, min_occupancy_frames, first_n_sec : spatial params.
    auto_close : bool

    Returns
    -------
    stab_results : dict[test_label][mouse] -> {
        "group", "per_neuron_r": array, "median_r": float}
    """
    save_dir = os.path.join(PLOTS_DIR, f"field_stability_bins_{n_bins}")
    os.makedirs(save_dir, exist_ok=True)

    tfc_dict = session_dicts.get("TFC_cond", {})
    test_labels = [l for l in session_dicts if l != "TFC_cond"]

    stab_results = {}

    for test_label in sorted(test_labels):
        test_dict = session_dicts.get(test_label, {})
        mapping_str = mappings.get(test_label)
        if not mapping_str:
            continue
        stab_results[test_label] = {}

        mice = sorted(set(tfc_dict.keys()) & set(test_dict.keys()))
        for mouse in mice:
            s1 = tfc_dict[mouse]
            s2 = test_dict[mouse]
            group = mouse_groups.get(mouse, "NA")

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            if len(cells_1) < 3:
                continue

            S1 = _get_mapped_S(s1, cells_1)
            S2 = _get_mapped_S(s2, cells_2)
            nc = min(S1.shape[0], S2.shape[0])
            S1, S2 = S1[:nc], S2[:nc]

            # Positions — normalize jointly
            T1, T2 = S1.shape[1], S2.shape[1]
            x1 = np.asarray(s1.loc_X_miniscope_smooth, dtype=float)[:T1]
            y1 = np.asarray(s1.loc_Y_miniscope_smooth, dtype=float)[:T1]
            x2 = np.asarray(s2.loc_X_miniscope_smooth, dtype=float)[:T2]
            y2 = np.asarray(s2.loc_Y_miniscope_smooth, dtype=float)[:T2]
            ax_all = np.concatenate([x1, x2])
            ay_all = np.concatenate([y1, y2])
            xmn, xmx = np.nanmin(ax_all), np.nanmax(ax_all)
            ymn, ymx = np.nanmin(ay_all), np.nanmax(ay_all)
            xr = xmx - xmn if xmx > xmn else 1.0
            yr = ymx - ymn if ymx > ymn else 1.0
            x1n, y1n = (x1 - xmn) / xr, (y1 - ymn) / yr
            x2n, y2n = (x2 - xmn) / xr, (y2 - ymn) / yr

            mask1 = _build_pretone_mask(s1, first_n_sec=first_n_sec)[:T1]
            mask2 = _build_pretone_mask(s2, first_n_sec=first_n_sec)[:T2]

            rm1, occ1, v1 = build_2D_rate_maps(
                S1, x1n, y1n, mask1, n_bins, smooth_sigma, min_occupancy_frames)
            rm2, occ2, v2 = build_2D_rate_maps(
                S2, x2n, y2n, mask2, n_bins, smooth_sigma, min_occupancy_frames)

            joint_valid = v1 & v2

            per_neuron_r = np.full(nc, np.nan)
            for c in range(nc):
                a = rm1[c][joint_valid]
                b = rm2[c][joint_valid]
                fin = np.isfinite(a) & np.isfinite(b)
                if np.sum(fin) >= 3 and np.std(a[fin]) > 0 and np.std(b[fin]) > 0:
                    per_neuron_r[c], _ = pearsonr(a[fin], b[fin])

            valid_r = per_neuron_r[np.isfinite(per_neuron_r)]
            stab_results[test_label][mouse] = {
                "group": group,
                "per_neuron_r": per_neuron_r,
                "median_r": float(np.nanmedian(valid_r)) if len(valid_r) > 0 else np.nan,
                "mean_r": float(np.nanmean(valid_r)) if len(valid_r) > 0 else np.nan,
                "n_cells": nc,
            }

    # ---- Plot: median per-neuron r by group for each test session ----
    for test_label, tdata in stab_results.items():
        # 1. Boxplot of median per-neuron r (one value per mouse)
        groups_data = {}
        for mouse, md in tdata.items():
            g = md["group"]
            if np.isfinite(md["median_r"]):
                groups_data.setdefault(g, []).append(md["median_r"])
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="Median per-neuron r",
                          title=f"Field stability\nTFC→{test_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f"field_stability_median_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # 2. Histogram overlay: all per-neuron r pooled by group
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        for g in _PV_GROUP_ORDER:
            all_r = []
            for mouse, md in tdata.items():
                if md["group"] == g:
                    v = md["per_neuron_r"]
                    all_r.extend(v[np.isfinite(v)].tolist())
            if len(all_r) > 0:
                ax.hist(all_r, bins=30, alpha=0.45,
                        color=_PV_BOX_COLORS.get(g, "gray"),
                        edgecolor="black", linewidth=0.3,
                        label=f"{_PV_GROUP_LABELS[g]} (n={len(all_r)})",
                        density=True)
        ax.set_xlabel("Per-neuron r (TFC vs Test)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"Field stability dist. — TFC→{test_label}", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f"field_stability_hist_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Place Field Stability Summary")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for test_label, tdata in stab_results.items():
        gd = {}
        for mouse, md in tdata.items():
            if np.isfinite(md["median_r"]):
                gd.setdefault(md["group"], []).append(md["median_r"])
        arrays = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrays) >= 2:
            F, p = _f(*arrays)
            gs = "  ".join(f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.3f}±{np.std(gd[g]):.3f}"
                           for g in _PV_GROUP_ORDER if g in gd)
            print(f"  TFC→{test_label:12s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    print(f"[Field stability] Saved to {save_dir}")

    return stab_results


# =========================================================================
#  3. Population Dimensionality (PCA Participation Ratio)
# =========================================================================

def _participation_ratio(S_masked):
    """
    Participation ratio from the eigenvalue spectrum of the neuron×neuron
    covariance matrix.

    PR = (Σ λ_i)^2 / Σ λ_i^2

    Parameters
    ----------
    S_masked : (N, T) array — neural activity for selected frames.

    Returns
    -------
    pr : float — participation ratio (effective dimensionality).
    """
    N, T = S_masked.shape
    if N < 2 or T < 2:
        return np.nan
    # Centre
    S_c = S_masked - S_masked.mean(axis=1, keepdims=True)
    # Covariance (N×N)
    cov = S_c @ S_c.T / (T - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return np.nan
    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return float(pr)


def compute_population_dimensionality(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    first_n_sec=180.0, auto_close=True,
):
    """
    Compute PCA participation ratio for each mouse/session, using both
    all neurons and cross-registered neurons.

    Parameters
    ----------
    session_dicts : dict[str, dict[str, Session]]
    mouse_groups : dict[str, str]
    mappings : dict[str, str]
    PLOTS_DIR : str
    first_n_sec : float
    auto_close : bool

    Returns
    -------
    dim_results : dict[sess_label][mouse] -> {"group", "pr_all", "pr_crossreg": {fam: pr}}
    """
    save_dir = os.path.join(PLOTS_DIR, "pop_dimensionality")
    os.makedirs(save_dir, exist_ok=True)

    tfc_dict = session_dicts.get("TFC_cond", {})
    crossreg_families = {
        "TFC_cond": None,
        "Test_A": "A", "Test_A_1wk": "A",
        "Test_B": "B", "Test_B_1wk": "B",
    }

    # Pre-compute crossreg cell IDs per (family, mouse, session)
    cr_ids = {}
    for fam, (test48, test1w) in _FAMILY_PAIRS.items():
        test48_dict = session_dicts.get(test48, {})
        test1w_dict = session_dicts.get(test1w, {})
        mapping_str = mappings.get(test48)
        if not mapping_str:
            continue
        for mouse in sorted(set(tfc_dict.keys()) & set(test48_dict.keys())):
            c_tfc, c_test = _get_crossreg_cells(tfc_dict[mouse], test48_dict[mouse], mapping_str)
            cr_ids[(fam, mouse, "TFC_cond")] = c_tfc
            cr_ids[(fam, mouse, test48)] = c_test
        for mouse in sorted(set(tfc_dict.keys()) & set(test1w_dict.keys())):
            c_tfc, c_1w = _get_crossreg_cells(tfc_dict[mouse], test1w_dict[mouse], mapping_str)
            cr_ids[(fam, mouse, test1w)] = c_1w
            cr_ids[(fam, mouse, "TFC_cond")] = c_tfc

    dim_results = {}
    all_sess_labels = sorted(session_dicts.keys())

    for sess_label in all_sess_labels:
        sess_dict = session_dicts.get(sess_label, {})
        dim_results[sess_label] = {}

        for mouse, sess in sorted(sess_dict.items()):
            group = mouse_groups.get(mouse, "NA")
            S = sess.S
            T = S.shape[1]
            mask = _build_pretone_mask(sess, first_n_sec=first_n_sec)[:T]
            S_m = S[:, mask]

            pr_all = _participation_ratio(S_m)

            # Crossreg
            fam = crossreg_families.get(sess_label)
            families_to_use = [fam] if fam else ["A", "B"]
            pr_cr = {}
            for f in families_to_use:
                ids = cr_ids.get((f, mouse, sess_label), [])
                if len(ids) < 3:
                    pr_cr[f] = np.nan
                    continue
                S_cr = _get_mapped_S(sess, ids)[:, mask]
                pr_cr[f] = _participation_ratio(S_cr)

            dim_results[sess_label][mouse] = {
                "group": group,
                "pr_all": pr_all,
                "pr_crossreg": pr_cr,
                "n_all": S.shape[0],
            }

    # ---- Plot: PR by group for each session ----
    for neuron_set, set_label in [("all", "All neurons"), ("crossreg", "Cross-reg")]:
        for sess_label in all_sess_labels:
            if sess_label not in dim_results:
                continue
            fam_key = crossreg_families.get(sess_label)
            families = [fam_key] if fam_key else ["A", "B"]

            for f in families:
                groups_data = {}
                for mouse, md in dim_results[sess_label].items():
                    g = md["group"]
                    if neuron_set == "all":
                        v = md["pr_all"]
                    else:
                        v = md["pr_crossreg"].get(f, np.nan)
                    if np.isfinite(v):
                        groups_data.setdefault(g, []).append(v)
                groups_data = {g: np.array(v) for g, v in groups_data.items()}

                fam_tag = f"_fam{f}" if neuron_set == "crossreg" else ""
                fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
                _pv_paper_boxplot(ax, groups_data,
                                  ylabel="Participation ratio",
                                  title=f"{sess_label} — {set_label}{fam_tag}")
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir,
                            f"PR_{neuron_set}{fam_tag}_{sess_label}.png"),
                            dpi=300, bbox_inches="tight")
                if auto_close:
                    plt.close(fig)
                else:
                    plt.show()

    # ---- Also plot PR normalized by N (PR/N) ----
    for sess_label in all_sess_labels:
        if sess_label not in dim_results:
            continue
        groups_data = {}
        for mouse, md in dim_results[sess_label].items():
            g = md["group"]
            if np.isfinite(md["pr_all"]) and md["n_all"] > 0:
                groups_data.setdefault(g, []).append(md["pr_all"] / md["n_all"])
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="PR / N",
                          title=f"{sess_label} — Norm. dimensionality")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f"PR_norm_{sess_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Population Dimensionality (Participation Ratio)")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for sess_label in all_sess_labels:
        if sess_label not in dim_results:
            continue
        gd = {}
        for mouse, md in dim_results[sess_label].items():
            g = md["group"]
            if np.isfinite(md["pr_all"]):
                gd.setdefault(g, []).append(md["pr_all"])
        arrays = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrays) >= 2:
            F, p = _f(*arrays)
            gs = "  ".join(f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.1f}±{np.std(gd[g]):.1f}"
                           for g in _PV_GROUP_ORDER if g in gd)
            print(f"  {sess_label:15s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    print(f"[Dimensionality] Saved to {save_dir}")

    return dim_results


# =========================================================================
#  Place-Field–based analyses
#  (require sess.fm to be populated from plot_fluorescence_map)
# =========================================================================

def _pos_to_uid_map(fm):
    """Build {position_index: unit_id} mapping from fm.sess.S_idx.

    Returns an empty dict when session metadata isn't available (in which
    case position == unit_id is assumed).
    """
    sess = getattr(fm, 'sess', None)
    if sess is None:
        return {}
    s_idx = getattr(sess, 'S_idx', None)
    if s_idx is None:
        return {}
    return {int(pos): int(uid) for pos, uid in enumerate(np.asarray(s_idx))}


def _pf_pos(fm, uid):
    """Convert a minian unit_id to the FM/PF position index.

    Internally the PF dicts (model_, merged_means, pf_size …) are keyed
    by the position index used when the FluorescenceMap was created (i.e.
    the index into S_mov).  Cross-registration returns minian unit_ids.
    This helper bridges the two.  Falls back to uid unchanged when session
    metadata is absent (backward-compatible with the old assumption that
    position == unit_id).
    """
    sess = getattr(fm, 'sess', None)
    if sess is not None:
        s_idx = getattr(sess, 'S_idx', None)
        if s_idx is not None:
            match = np.where(np.asarray(s_idx) == uid)[0]
            if len(match):
                return int(match[0])
    return uid


def _get_pf_cells(fm, max_pf_count=None):
    """
    Return list of cell **unit IDs** that have detected place fields.

    Parameters
    ----------
    fm : FluorescenceMap with fm.pf populated.
    max_pf_count : int or None — if set, only return cells with exactly that
        many place fields (after merging).  None = all place cells.

    Returns
    -------
    list[int] — minian unit IDs (converted from FM position indices via
                fm.sess.S_idx when available).
    """
    p2u = _pos_to_uid_map(fm)
    out = []
    for pos_idx in fm.pf.pf_size:
        n_pf = len(fm.pf.pf_size[pos_idx])
        if n_pf == 0:
            continue
        if max_pf_count is not None and n_pf != max_pf_count:
            continue
        uid = p2u.get(pos_idx, pos_idx)
        out.append(uid)
    return out


def _pf_count_for_uid(fm, uid):
    """Return the number of detected place fields for a cell (by unit ID)."""
    pos = _pf_pos(fm, uid)
    return len(getattr(fm.pf, 'pf_size', {}).get(pos, []))


def _stable_subtype(fm1, c1, fm2, c2):
    """Return 'stable-same', 'stable-reduced', or 'stable-expanded'."""
    n1 = _pf_count_for_uid(fm1, c1)
    n2 = _pf_count_for_uid(fm2, c2)
    if n2 < n1:
        return "stable-reduced"
    elif n2 > n1:
        return "stable-expanded"
    return "stable-same"


def _get_fm_ratemap(fm, cell_id):
    """Get occupancy-normalised rate map for *cell_id* (unit ID).

    Converts the minian unit_id to an FM position index via
    fm.sess.S_idx before indexing into the rate-map array.
    """
    idx = _pf_pos(fm, cell_id)
    return fm.fluorescence_map_occup[:, :, idx]


def _render_gaussian_pf(fm, cell_id, oversample=4, min_sigma=1.0):
    """Render smooth 2D Gaussian place-field map from merged GMM params.

    Returns an array at *oversample*× the native bin resolution,
    evaluated from the merged means / covariances / weights.
    Cells with no detected PF return an all-zero array.

    Parameters
    ----------
    fm : FluorescenceMap
    cell_id : int  (minian unit ID)
    oversample : int
        Factor by which to upsample the native grid for smoothness.
    min_sigma : float
        Minimum standard deviation (in bin units) along each principal
        axis.  Prevents single-bin or very sparse fields from producing
        an invisible, delta-like Gaussian.  Default 1.0 bin ≈ 2 cm.

    Returns
    -------
    gauss_map : ndarray, shape (H*oversample, W*oversample)
    """
    _pos = _pf_pos(fm, cell_id)
    raw = fm.fluorescence_map_occup[:, :, _pos]
    h, w = raw.shape

    m_means = getattr(fm.pf, 'merged_means_', {}).get(_pos)
    m_covs  = getattr(fm.pf, 'merged_covariances_', {}).get(_pos)
    m_wts   = getattr(fm.pf, 'merged_weights_', {}).get(_pos)

    H, W = h * oversample, w * oversample
    result = np.zeros((H, W), dtype=np.float64)
    if not m_means or not m_covs:
        return result

    min_var = min_sigma ** 2  # floor for eigenvalues

    # Coordinate grid in native-bin units (centre of each oversampled pixel)
    yy = (np.arange(H) + 0.5) / oversample
    xx = (np.arange(W) + 0.5) / oversample
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')
    coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)  # (N,2)

    for pf_idx in range(len(m_means)):
        mu  = m_means[pf_idx]         # (2,) [row, col]
        cov = np.array(m_covs[pf_idx], dtype=np.float64)  # (2,2)
        wt  = m_wts[pf_idx] if m_wts else 1.0

        # Enforce minimum spread so tiny / single-bin fields stay visible
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_var)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        inv_cov = np.linalg.inv(cov)
        diff = coords - mu            # (N,2)
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        result += wt * np.exp(exponent).reshape(H, W)

    # Normalise to [0, 1] for clean display
    rmax = result.max()
    if rmax > 0:
        result /= rmax
    return result


def _draw_sig_responses(ax, fm, cell_id):
    """Overlay sig_responses markers on an axes that already has a ratemap."""
    _pos = _pf_pos(fm, cell_id)
    sig = getattr(fm, 'sig_responses', {})
    if _pos not in sig:
        ax.text(0.5, 0.02, "no sig resp", ha="center", va="bottom",
                transform=ax.transAxes, fontsize=6, color="white",
                alpha=0.7)
        return
    fields = sig[_pos]  # list of [row, col, fluorescence]
    for row, col, _ in fields:
        ax.plot(col, row, "x", color="white", markersize=8, mew=2)


def _pf_centroid(fm, cell_id, pf_idx=0):
    """
    Get the centroid of a place field in bin coordinates [row, col].

    *cell_id* is a minian unit_id; internally converted to the FM
    position index used by the PF dicts.

    Uses precomputed merged_means_ when available, otherwise falls back
    to averaging component means from the raw GMM.
    """
    pos = _pf_pos(fm, cell_id)
    merged = fm.pf.merged_means[pos]
    if pf_idx >= len(merged):
        return np.array([np.nan, np.nan])
    m_means = getattr(fm.pf, 'merged_means_', {}).get(pos)
    if m_means:
        return np.asarray(m_means[pf_idx])
    component_idxs = merged[pf_idx]
    means = fm.pf.model_[pos].means_  # (n_components, 2)
    centroid = np.mean(means[component_idxs], axis=0)  # [row, col]
    return centroid


def _match_pf_shifts(fm1, c1, fm2, c2):
    """
    Compute PF centroid shifts between two sessions for a single neuron
    using nearest-centroid (Hungarian) matching.

    Returns
    -------
    dict with keys:
        shifts : list of float — Euclidean distance for each matched PF pair
        n_dropped : int — PFs in session 1 with no match in session 2
        n_acquired : int — PFs in session 2 with no match in session 1
    """
    from scipy.optimize import linear_sum_assignment

    n1 = len(fm1.pf.merged_means.get(_pf_pos(fm1, c1), []))
    n2 = len(fm2.pf.merged_means.get(_pf_pos(fm2, c2), []))
    if n1 == 0 and n2 == 0:
        return {"shifts": [], "n_dropped": 0, "n_acquired": 0}
    if n1 == 0:
        return {"shifts": [], "n_dropped": 0, "n_acquired": n2}
    if n2 == 0:
        return {"shifts": [], "n_dropped": n1, "n_acquired": 0}

    cens1 = [_pf_centroid(fm1, c1, i) for i in range(n1)]
    cens2 = [_pf_centroid(fm2, c2, i) for i in range(n2)]

    # Drop any NaN centroids
    valid1 = [(i, c) for i, c in enumerate(cens1) if not np.any(np.isnan(c))]
    valid2 = [(i, c) for i, c in enumerate(cens2) if not np.any(np.isnan(c))]
    nv1, nv2 = len(valid1), len(valid2)
    if nv1 == 0 and nv2 == 0:
        return {"shifts": [], "n_dropped": 0, "n_acquired": 0}
    if nv1 == 0:
        return {"shifts": [], "n_dropped": 0, "n_acquired": nv2}
    if nv2 == 0:
        return {"shifts": [], "n_dropped": nv1, "n_acquired": 0}

    arr1 = np.array([c for _, c in valid1])  # (nv1, 2)
    arr2 = np.array([c for _, c in valid2])  # (nv2, 2)

    # Build cost matrix: Euclidean distance between every pair
    cost = np.linalg.norm(arr1[:, None, :] - arr2[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    n_matched = len(row_ind)
    return {
        "shifts": [cost[r, c] for r, c in zip(row_ind, col_ind)],
        "n_dropped": nv1 - n_matched,
        "n_acquired": nv2 - n_matched,
    }


# ---- helpers for spatial-info from FM occupancy maps -----------------------

def _skaggs_from_fm(fm, cell_id):
    """
    Skaggs spatial information (bits/spike) computed from the FluorescenceMap
    occupancy-normalised rate map and the Location_XY occupancy.
    """
    rm = _get_fm_ratemap(fm, cell_id)
    occ = fm.loc.occupancy  # (rows, cols), counts in ms
    # Flatten and align shapes
    h, w = rm.shape
    occ_h, occ_w = occ.shape
    # Use overlap region
    h_use, w_use = min(h, occ_h), min(w, occ_w)
    rm_flat = rm[:h_use, :w_use].ravel().astype(float)
    occ_flat = occ[:h_use, :w_use].ravel().astype(float)
    # Valid bins: non-zero occupancy and finite rate
    valid = (occ_flat > 0) & np.isfinite(rm_flat)
    if np.sum(valid) < 3:
        return np.nan
    rm_v = rm_flat[valid]
    occ_v = occ_flat[valid]
    total_occ = occ_v.sum()
    p = occ_v / total_occ
    r_mean = np.sum(p * rm_v)
    if r_mean <= 0:
        return np.nan
    ratio = rm_v / r_mean
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.where(ratio > 0, np.log2(ratio), 0.0)
    return float(np.sum(p * ratio * lr))


# ===========================================================================
#  PF-based Spatial Information
# ===========================================================================

def compute_spatial_information_PF(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    max_pf_count=None, auto_close=True,
):
    """
    Compute Skaggs SI from the FluorescenceMap rate maps, restricted to
    neurons with detected place fields.

    Two runs expected from the caller:
      • max_pf_count=None  → all place cells
      • max_pf_count=1     → only single-PF neurons

    Saves figures into PLOTS_DIR/spatial_info_PF[_npf1]/.

    Parameters
    ----------
    session_dicts : dict[str, dict[str, Session]]
    mouse_groups, mappings : dicts
    PLOTS_DIR : str
    max_pf_count : int or None
    auto_close : bool

    Returns
    -------
    si_results : dict[sess_label][mouse] -> {group, si_values, n_pf_cells}
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(PLOTS_DIR, f"spatial_info_PF{pf_tag}")
    os.makedirs(save_dir, exist_ok=True)

    si_results = {}
    for sess_label, sess_dict in sorted(session_dicts.items()):
        si_results[sess_label] = {}
        for mouse, sess in sorted(sess_dict.items()):
            group = mouse_groups.get(mouse, "NA")
            fm = getattr(sess, 'fm', None)
            if fm is None or fm.pf is None or fm.fluorescence_map_occup is None:
                continue
            pf_cells = _get_pf_cells(fm, max_pf_count=max_pf_count)
            if len(pf_cells) == 0:
                continue
            si_vals = np.array([_skaggs_from_fm(fm, c) for c in pf_cells])
            si_results[sess_label][mouse] = {
                "group": group,
                "si_values": si_vals,
                "n_pf_cells": len(pf_cells),
            }

    # ---- Plot: median SI per mouse, by group, for each session ----
    for sess_label in sorted(si_results.keys()):
        groups_data = {}
        for mouse, md in si_results[sess_label].items():
            v = md["si_values"]
            valid = v[np.isfinite(v)]
            if len(valid) > 0:
                groups_data.setdefault(md["group"], []).append(float(np.nanmedian(valid)))
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="Median SI (bits/spike)",
                          title=f"{sess_label} — PF neurons{pf_tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"SI_PF_{sess_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # ---- Pooled neuron analysis (neuron-level, no per-mouse averaging) ----
    pooled_save_dir = save_dir + "_pooled"
    for sess_label in sorted(si_results.keys()):
        records = []
        for mouse, md in si_results[sess_label].items():
            g = md["group"]
            for val in md["si_values"]:
                if np.isfinite(val):
                    records.append(
                        {"group": g, "mouse": mouse, "si": float(val)})
        if not records:
            continue
        panel_label = f"SI_PF{pf_tag}_{sess_label}"
        _run_si_pooled_analysis(
            records, pooled_save_dir, panel_label,
            auto_close=auto_close)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Spatial Information (PF{pf_tag})")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for sess_label in sorted(si_results.keys()):
        gd = {}
        for mouse, md in si_results[sess_label].items():
            v = md["si_values"][np.isfinite(md["si_values"])]
            if len(v) > 0:
                gd.setdefault(md["group"], []).append(float(np.nanmedian(v)))
        arrs = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrs) >= 2:
            F, p = _f(*arrs)
            gs = "  ".join(f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.3f}±{np.std(gd[g]):.3f}"
                           for g in _PV_GROUP_ORDER if g in gd)
            print(f"  {sess_label:15s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    return si_results


# ===========================================================================
#  PF-based Field Stability (per-neuron rate-map correlation)
# ===========================================================================

def compute_place_field_stability_PF(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    max_pf_count=None, auto_close=True,
):
    """
    For each cross-registered neuron that has a place field in BOTH TFC and
    the Test session, correlate its FluorescenceMap rate map (occupancy-norm.).

    Two variants:
      • max_pf_count=None → all place cells
      • max_pf_count=1   → only single-PF neurons

    Returns
    -------
    stab_results : dict[test_label][mouse] -> {group, per_neuron_r, ...}
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(PLOTS_DIR, f"field_stability_PF{pf_tag}")
    os.makedirs(save_dir, exist_ok=True)

    tfc_dict = session_dicts.get("TFC_cond", {})
    test_labels = [l for l in session_dicts if l != "TFC_cond"]
    stab_results = {}

    for test_label in sorted(test_labels):
        test_dict = session_dicts.get(test_label, {})
        mapping_str = mappings.get(test_label)
        if not mapping_str:
            continue
        stab_results[test_label] = {}
        mice = sorted(set(tfc_dict.keys()) & set(test_dict.keys()))

        for mouse in mice:
            s1, s2 = tfc_dict[mouse], test_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue
            if fm1.fluorescence_map_occup is None or fm2.fluorescence_map_occup is None:
                continue

            # Cross-registered cell pairs
            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            if len(cells_1) < 3:
                continue

            # Filter to cells that have PFs in BOTH sessions
            pf1_set = set(_get_pf_cells(fm1, max_pf_count=max_pf_count))
            pf2_set = set(_get_pf_cells(fm2, max_pf_count=max_pf_count))

            paired_1, paired_2 = [], []
            for c1, c2 in zip(cells_1, cells_2):
                if c1 in pf1_set and c2 in pf2_set:
                    paired_1.append(c1)
                    paired_2.append(c2)

            if len(paired_1) < 2:
                continue

            per_neuron_r = np.full(len(paired_1), np.nan)
            for ci, (c1, c2) in enumerate(zip(paired_1, paired_2)):
                rm_a = _get_fm_ratemap(fm1, c1)
                rm_b = _get_fm_ratemap(fm2, c2)
                # Align shapes (may differ slightly between sessions)
                h = min(rm_a.shape[0], rm_b.shape[0])
                w = min(rm_a.shape[1], rm_b.shape[1])
                a_flat = rm_a[:h, :w].ravel().astype(float)
                b_flat = rm_b[:h, :w].ravel().astype(float)
                fin = np.isfinite(a_flat) & np.isfinite(b_flat) & (a_flat != 0) & (b_flat != 0)
                if np.sum(fin) >= 3 and np.std(a_flat[fin]) > 0 and np.std(b_flat[fin]) > 0:
                    per_neuron_r[ci], _ = pearsonr(a_flat[fin], b_flat[fin])

            valid_r = per_neuron_r[np.isfinite(per_neuron_r)]
            stab_results[test_label][mouse] = {
                "group": group,
                "per_neuron_r": per_neuron_r,
                "median_r": float(np.nanmedian(valid_r)) if len(valid_r) > 0 else np.nan,
                "mean_r": float(np.nanmean(valid_r)) if len(valid_r) > 0 else np.nan,
                "n_paired": len(paired_1),
            }

    # ---- Plot ----
    for test_label, tdata in stab_results.items():
        # Boxplot of median per-neuron r
        groups_data = {}
        for mouse, md in tdata.items():
            if np.isfinite(md["median_r"]):
                groups_data.setdefault(md["group"], []).append(md["median_r"])
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="Median per-neuron r",
                          title=f"PF stability{pf_tag}\nTFC→{test_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_stability_median_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # Histogram overlay
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        for g in _PV_GROUP_ORDER:
            all_r = []
            for mouse, md in tdata.items():
                if md["group"] == g:
                    v = md["per_neuron_r"]
                    all_r.extend(v[np.isfinite(v)].tolist())
            if len(all_r) > 0:
                ax.hist(all_r, bins=30, alpha=0.45,
                        color=_PV_BOX_COLORS.get(g, "gray"),
                        edgecolor="black", linewidth=0.3,
                        label=f"{_PV_GROUP_LABELS[g]} (n={len(all_r)})",
                        density=True)
        ax.set_xlabel("Per-neuron r", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"PF stability dist.{pf_tag} — TFC→{test_label}", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_stability_hist_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Place Field Stability (PF{pf_tag})")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for test_label, tdata in stab_results.items():
        gd = {}
        for mouse, md in tdata.items():
            if np.isfinite(md["median_r"]):
                gd.setdefault(md["group"], []).append(md["median_r"])
        arrs = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrs) >= 2:
            F, p = _f(*arrs)
            gs = "  ".join(f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.3f}±{np.std(gd[g]):.3f}"
                           for g in _PV_GROUP_ORDER if g in gd)
            print(f"  TFC→{test_label:12s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    return stab_results


# ===========================================================================
#  PF Centroid Shift (single-PF neurons by default)
# ===========================================================================

def compute_pf_centroid_shift(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    max_pf_count=1, auto_close=True,
):
    """
    For cross-registered neurons with exactly *max_pf_count* place fields in
    both TFC and each Test session, compute the Euclidean distance (in bins)
    between the PF centroid in TFC and the PF centroid in the Test session.

    Compares the distribution of shifts across groups.

    Parameters
    ----------
    max_pf_count : int or None — only include neurons with exactly this many
        PFs.  None = all PF neurons.
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(PLOTS_DIR, f"pf_centroid_shift{pf_tag}")
    os.makedirs(save_dir, exist_ok=True)

    tfc_dict = session_dicts.get("TFC_cond", {})
    test_labels = [l for l in session_dicts if l != "TFC_cond"]
    shift_results = {}

    for test_label in sorted(test_labels):
        test_dict = session_dicts.get(test_label, {})
        mapping_str = mappings.get(test_label)
        if not mapping_str:
            continue
        shift_results[test_label] = {}
        mice = sorted(set(tfc_dict.keys()) & set(test_dict.keys()))

        for mouse in mice:
            s1, s2 = tfc_dict[mouse], test_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            pf1_set = set(_get_pf_cells(fm1, max_pf_count=max_pf_count))
            pf2_set = set(_get_pf_cells(fm2, max_pf_count=max_pf_count))

            shifts = []
            per_neuron_acquired = []
            per_neuron_dropped = []
            for c1, c2 in zip(cells_1, cells_2):
                if c1 not in pf1_set or c2 not in pf2_set:
                    continue
                res = _match_pf_shifts(fm1, c1, fm2, c2)
                shifts.extend(res["shifts"])
                # Only count neurons with unequal PF numbers across sessions
                if res["n_acquired"] + res["n_dropped"] > 0:
                    per_neuron_acquired.append(res["n_acquired"])
                    per_neuron_dropped.append(res["n_dropped"])

            if len(shifts) < 2:
                continue
            shifts = np.array(shifts)

            # Get bin width for converting to physical units
            bw = getattr(fm1.loc, 'bin_width', 1.0)

            shift_results[test_label][mouse] = {
                "group": group,
                "shifts_bins": shifts,
                "shifts_px": shifts * bw,
                "mean_shift_bins": float(np.mean(shifts)),
                "median_shift_bins": float(np.median(shifts)),
                "n_cells": len(shifts),
                "bin_width": bw,
                "per_neuron_acquired": np.array(per_neuron_acquired),
                "per_neuron_dropped": np.array(per_neuron_dropped),
            }

    # ---- Plot: median shift per mouse ----
    for test_label, tdata in shift_results.items():
        # Boxplot: median shift
        groups_data = {}
        for mouse, md in tdata.items():
            groups_data.setdefault(md["group"], []).append(md["median_shift_bins"])
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="Median PF shift (bins)",
                          title=f"PF centroid shift{pf_tag}\nTFC→{test_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_shift_median_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # Histogram overlay: all per-neuron shifts pooled by group
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        for g in _PV_GROUP_ORDER:
            all_s = []
            for mouse, md in tdata.items():
                if md["group"] == g:
                    all_s.extend(md["shifts_bins"].tolist())
            if len(all_s) > 0:
                ax.hist(all_s, bins=25, alpha=0.45,
                        color=_PV_BOX_COLORS.get(g, "gray"),
                        edgecolor="black", linewidth=0.3,
                        label=f"{_PV_GROUP_LABELS[g]} (n={len(all_s)})",
                        density=True)
        ax.set_xlabel("PF centroid shift (bins)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"PF shift dist.{pf_tag} — TFC→{test_label}", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_shift_hist_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # ---- Acquired PFs per neuron (mean per mouse) ----
        acq_data = {}
        for mouse, md in tdata.items():
            arr = md["per_neuron_acquired"]
            if len(arr) > 0:
                acq_data.setdefault(md["group"], []).append(float(np.mean(arr)))
        acq_data = {g: np.array(v) for g, v in acq_data.items()}

        if any(len(v) >= 2 for v in acq_data.values()):
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
            _pv_paper_boxplot(ax, acq_data,
                              ylabel="Mean # acquired PFs / neuron",
                              title=f"Acquired PFs{pf_tag}\nTFC\u2192{test_label}")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"PF_acquired_mean_{test_label}.png"),
                        dpi=300, bbox_inches="tight")
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

        # ---- Dropped PFs per neuron (mean per mouse) ----
        drop_data = {}
        for mouse, md in tdata.items():
            arr = md["per_neuron_dropped"]
            if len(arr) > 0:
                drop_data.setdefault(md["group"], []).append(float(np.mean(arr)))
        drop_data = {g: np.array(v) for g, v in drop_data.items()}

        if any(len(v) >= 2 for v in drop_data.values()):
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
            _pv_paper_boxplot(ax, drop_data,
                              ylabel="Mean # dropped PFs / neuron",
                              title=f"Dropped PFs{pf_tag}\nTFC\u2192{test_label}")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"PF_dropped_mean_{test_label}.png"),
                        dpi=300, bbox_inches="tight")
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

        # ---- Discrimination index: (NumAcq - NumLost) / (NumAcq + NumLost) ----
        di_data = {}
        for mouse, md in tdata.items():
            acq_arr = md["per_neuron_acquired"]
            drop_arr = md["per_neuron_dropped"]
            if len(acq_arr) == 0 and len(drop_arr) == 0:
                continue
            n_acq = int(np.sum(acq_arr > 0)) if len(acq_arr) > 0 else 0
            n_lost = int(np.sum(drop_arr > 0)) if len(drop_arr) > 0 else 0
            denom = n_acq + n_lost
            if denom == 0:
                continue
            di = (n_acq - n_lost) / denom
            di_data.setdefault(md["group"], []).append(di)
        di_data = {g: np.array(v) for g, v in di_data.items()}

        if any(len(v) >= 2 for v in di_data.values()):
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
            _pv_paper_boxplot(ax, di_data,
                              ylabel="DI  (Acq\u2212Lost) / (Acq+Lost)",
                              title=f"PF gain/loss DI{pf_tag}\nTFC\u2192{test_label}")
            ax.axhline(0, color="gray", ls="--", lw=0.6)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"PF_gainloss_DI_{test_label}.png"),
                        dpi=300, bbox_inches="tight")
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

    # Print summary
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for test_label, tdata in shift_results.items():
        gd = {}
        for mouse, md in tdata.items():
            gd.setdefault(md["group"], []).append(md["median_shift_bins"])
        arrs = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrs) >= 2:
            F, p = _f(*arrs)
            gs = "  ".join(
                f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.2f}±{np.std(gd[g]):.2f}"
                for g in _PV_GROUP_ORDER if g in gd)
            print(f"  TFC→{test_label:12s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    return shift_results


# ===========================================================================
#  PF-filtered 2D PV Correlation (full pipeline)
# ===========================================================================

def compute_2D_pv_correlation_PF(
    sess_dict_1,
    sess_dict_2,
    mouse_groups,
    mapping,
    PLOTS_DIR,
    sess_1_label="TFC_cond",
    sess_2_label="Test_A",
    n_bins=10,
    smooth_sigma=1.0,
    min_occupancy_frames=4,
    first_n_sec=180.0,
    max_pf_count=None,
    auto_close=True,
    use_z_score="none",
):
    """
    Same as compute_2D_pv_correlation, but restricted to cross-registered
    neurons that have place fields in BOTH sessions.

    Parameters
    ----------
    max_pf_count : int or None
        None = all place cells; 1 = only single-PF neurons; etc.
    (all other parameters identical to compute_2D_pv_correlation)

    Returns
    -------
    results : dict — keyed by mouse, same structure as compute_2D_pv_correlation.
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(
        PLOTS_DIR,
        f"PV_2D_{sess_1_label}_vs_{sess_2_label}_bins_{n_bins}_PF{pf_tag}_{_z_score_dir_tag(use_z_score)}",
    )
    os.makedirs(save_dir, exist_ok=True)

    mice = sorted(set(sess_dict_1.keys()) & set(sess_dict_2.keys()))
    results = {}

    for mouse in mice:
        s1 = sess_dict_1[mouse]
        s2 = sess_dict_2[mouse]
        group = mouse_groups.get(mouse, "NA")

        # --- Cross-registered cells ---
        cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping)
        if len(cells_1) == 0:
            print(f"[PV-2D-PF] {mouse} ({group}): no crossreg cells, skipping.")
            continue

        # --- PF filter: require PF in BOTH sessions ---
        fm1 = getattr(s1, 'fm', None)
        fm2 = getattr(s2, 'fm', None)
        if fm1 is None or fm2 is None or fm1.pf is None or fm2.pf is None:
            print(f"[PV-2D-PF] {mouse} ({group}): fm/pf not available, skipping.")
            continue

        pf1_set = set(_get_pf_cells(fm1, max_pf_count=max_pf_count))
        pf2_set = set(_get_pf_cells(fm2, max_pf_count=max_pf_count))

        filt_1, filt_2 = [], []
        for c1, c2 in zip(cells_1, cells_2):
            if c1 in pf1_set and c2 in pf2_set:
                filt_1.append(c1)
                filt_2.append(c2)

        if len(filt_1) < 3:
            print(f"[PV-2D-PF] {mouse} ({group}): only {len(filt_1)} PF cells "
                  f"(of {len(cells_1)} crossreg), skipping.")
            continue

        S1 = _get_mapped_S(s1, filt_1)
        S2 = _get_mapped_S(s2, filt_2)
        n_cells = min(S1.shape[0], S2.shape[0])
        S1 = S1[:n_cells]
        S2 = S2[:n_cells]

        S1, S2 = _apply_zscore_pair(S1, S2, use_z_score)

        if n_cells < 3:
            print(f"[PV-2D-PF] {mouse} ({group}): only {n_cells} cells, skipping.")
            continue

        # Position data
        x1 = np.asarray(s1.loc_X_miniscope_smooth, dtype=float)[:S1.shape[1]]
        y1 = np.asarray(s1.loc_Y_miniscope_smooth, dtype=float)[:S1.shape[1]]
        x2 = np.asarray(s2.loc_X_miniscope_smooth, dtype=float)[:S2.shape[1]]
        y2 = np.asarray(s2.loc_Y_miniscope_smooth, dtype=float)[:S2.shape[1]]

        # Normalize to [0, 1] using combined bounds
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
        y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        x1_n = (x1 - x_min) / x_range
        y1_n = (y1 - y_min) / y_range
        x2_n = (x2 - x_min) / x_range
        y2_n = (y2 - y_min) / y_range

        # Pre-tone masks
        mask1 = _build_pretone_mask(s1, first_n_sec=first_n_sec)[:S1.shape[1]]
        mask2 = _build_pretone_mask(s2, first_n_sec=first_n_sec)[:S2.shape[1]]

        if np.sum(mask1) < 20 or np.sum(mask2) < 20:
            print(f"[PV-2D-PF] {mouse} ({group}): insufficient pre-tone frames, skipping.")
            continue

        # Build rate maps
        rm1, occ1, valid1 = build_2D_rate_maps(
            S1, x1_n, y1_n, mask1, n_bins,
            smooth_sigma=smooth_sigma,
            min_occupancy_frames=min_occupancy_frames,
        )
        rm2, occ2, valid2 = build_2D_rate_maps(
            S2, x2_n, y2_n, mask2, n_bins,
            smooth_sigma=smooth_sigma,
            min_occupancy_frames=min_occupancy_frames,
        )

        # Joint valid bins
        joint_valid = valid1 & valid2
        n_valid_bins = np.sum(joint_valid)

        if n_valid_bins < 3:
            print(f"[PV-2D-PF] {mouse} ({group}): only {n_valid_bins} valid bins, skipping.")
            continue

        # ---- Build PV arrays: (n_valid_bins, n_cells) ----
        valid_ij = np.argwhere(joint_valid)
        pv1_raw = np.zeros((n_valid_bins, n_cells))
        pv2_raw = np.zeros((n_valid_bins, n_cells))
        for idx, (bi, bj) in enumerate(valid_ij):
            pv1_raw[idx, :] = rm1[:, bi, bj]
            pv2_raw[idx, :] = rm2[:, bi, bj]

        # Z-scored version
        pv1_z = np.copy(pv1_raw)
        pv2_z = np.copy(pv2_raw)
        for c in range(n_cells):
            mu1, sd1 = np.nanmean(pv1_raw[:, c]), np.nanstd(pv1_raw[:, c])
            mu2, sd2 = np.nanmean(pv2_raw[:, c]), np.nanstd(pv2_raw[:, c])
            pv1_z[:, c] = (pv1_raw[:, c] - mu1) / sd1 if sd1 > 0 else 0.0
            pv2_z[:, c] = (pv2_raw[:, c] - mu2) / sd2 if sd2 > 0 else 0.0

        # ---- Same-bin PV correlation ----
        def _same_bin_corr(pv_a, pv_b):
            n_b = pv_a.shape[0]
            r_vals = np.full(n_b, np.nan)
            for b in range(n_b):
                v1, v2 = pv_a[b], pv_b[b]
                finite = np.isfinite(v1) & np.isfinite(v2)
                if np.sum(finite) >= 3:
                    std1, std2 = np.std(v1[finite]), np.std(v2[finite])
                    if std1 > 0 and std2 > 0:
                        r_vals[b], _ = pearsonr(v1[finite], v2[finite])
            return r_vals

        samebin_r_raw = _same_bin_corr(pv1_raw, pv2_raw)
        samebin_r_z = _same_bin_corr(pv1_z, pv2_z)

        # ---- Full bin×bin PV correlation matrix ----
        def _full_pv_corr_matrix(pv_a, pv_b):
            M = pv_a.shape[0]
            corr_mat = np.full((M, M), np.nan)
            for a in range(M):
                for b in range(M):
                    v1, v2 = pv_a[a], pv_b[b]
                    finite = np.isfinite(v1) & np.isfinite(v2)
                    if np.sum(finite) >= 3:
                        std1, std2 = np.std(v1[finite]), np.std(v2[finite])
                        if std1 > 0 and std2 > 0:
                            corr_mat[a, b], _ = pearsonr(v1[finite], v2[finite])
            return corr_mat

        corr_mat_raw = _full_pv_corr_matrix(pv1_raw, pv2_raw)
        corr_mat_z = _full_pv_corr_matrix(pv1_z, pv2_z)

        # ---- Metrics ----
        def _compute_metrics(samebin_r, corr_mat):
            valid_same = samebin_r[np.isfinite(samebin_r)]
            diag = np.diag(corr_mat)
            valid_diag = diag[np.isfinite(diag)]
            off_diag_mask = ~np.eye(corr_mat.shape[0], dtype=bool)
            off_vals = corr_mat[off_diag_mask]
            valid_off = off_vals[np.isfinite(off_vals)]

            metrics = {}
            metrics['mean_samebin_r'] = np.nanmean(valid_same) if len(valid_same) > 0 else np.nan
            metrics['median_samebin_r'] = np.nanmedian(valid_same) if len(valid_same) > 0 else np.nan
            metrics['frac_positive_samebin'] = np.mean(valid_same > 0) if len(valid_same) > 0 else np.nan
            metrics['mean_diag_r'] = np.nanmean(valid_diag) if len(valid_diag) > 0 else np.nan
            metrics['mean_offdiag_r'] = np.nanmean(valid_off) if len(valid_off) > 0 else np.nan
            metrics['diag_minus_offdiag'] = metrics['mean_diag_r'] - metrics['mean_offdiag_r']

            n_m = corr_mat.shape[0]
            best_match_is_same = 0
            displacement_bins = []
            for a in range(n_m):
                row = corr_mat[a, :]
                if np.all(np.isnan(row)):
                    continue
                best_b = np.nanargmax(row)
                if best_b == a:
                    best_match_is_same += 1
                yi_a, xi_a = valid_ij[a]
                yi_b, xi_b = valid_ij[best_b]
                disp = np.sqrt((yi_a - yi_b)**2 + (xi_a - xi_b)**2)
                displacement_bins.append(disp)

            n_with_valid_row = sum(1 for a in range(n_m) if not np.all(np.isnan(corr_mat[a, :])))
            metrics['frac_best_match_same_bin'] = (
                best_match_is_same / n_with_valid_row if n_with_valid_row > 0 else np.nan
            )
            metrics['mean_best_match_displacement'] = (
                np.mean(displacement_bins) if len(displacement_bins) > 0 else np.nan
            )
            return metrics

        metrics_raw = _compute_metrics(samebin_r_raw, corr_mat_raw)
        metrics_z = _compute_metrics(samebin_r_z, corr_mat_z)

        results[mouse] = {
            'group': group,
            'n_cells': n_cells,
            'n_cells_crossreg_total': len(cells_1),
            'n_valid_bins': int(n_valid_bins),
            'valid_ij': valid_ij,
            'samebin_r_raw': samebin_r_raw,
            'samebin_r_z': samebin_r_z,
            'corr_mat_raw': corr_mat_raw,
            'corr_mat_z': corr_mat_z,
            'metrics_raw': metrics_raw,
            'metrics_z': metrics_z,
            'rate_maps_1': rm1,
            'rate_maps_2': rm2,
            'occupancy_1': occ1,
            'occupancy_2': occ2,
            'joint_valid': joint_valid,
        }

        print(f"[PV-2D-PF{pf_tag}] {mouse} ({group}): {n_cells}/{len(cells_1)} PF cells, "
              f"{n_valid_bins} bins | "
              f"raw mean_same={metrics_raw['mean_samebin_r']:.3f}  "
              f"diag-off={metrics_raw['diag_minus_offdiag']:.3f} | "
              f"z   mean_same={metrics_z['mean_samebin_r']:.3f}  "
              f"diag-off={metrics_z['diag_minus_offdiag']:.3f}")

    return results


def run_2D_pv_correlation_pipeline_PF(
    TFC_cond_dict,
    test_session_dicts,
    mouse_groups,
    mappings,
    PLOTS_DIR,
    n_bins=10,
    smooth_sigma=1.0,
    min_occupancy_frames=4,
    first_n_sec=180.0,
    max_pf_count=None,
    auto_close=True,
    use_z_score="none",
):
    """
    PF-filtered version of run_2D_pv_correlation_pipeline.

    Runs compute_2D_pv_correlation_PF for each test session, then plots
    results via plot_2D_pv_correlation_results (reusing the same plotter).

    Parameters
    ----------
    max_pf_count : int or None
        None = all PF neurons; 1 = single-PF only.
    (rest same as run_2D_pv_correlation_pipeline)
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    all_results = {}

    for test_label, test_dict in test_session_dicts.items():
        mapping = mappings.get(test_label)
        if mapping is None:
            print(f"[PV-2D-PF] No mapping for {test_label}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  2D PV Correlation (PF{pf_tag}): TFC_cond vs {test_label}")
        print(f"  mapping: {mapping}")
        print(f"  Grid: {n_bins}x{n_bins}, smooth_sigma={smooth_sigma}")
        print(f"  use_z_score: {use_z_score}")
        print(f"{'='*60}")

        pv_results = compute_2D_pv_correlation_PF(
            TFC_cond_dict,
            test_dict,
            mouse_groups,
            mapping=mapping,
            PLOTS_DIR=PLOTS_DIR,
            sess_1_label="TFC_cond",
            sess_2_label=test_label,
            n_bins=n_bins,
            smooth_sigma=smooth_sigma,
            min_occupancy_frames=min_occupancy_frames,
            first_n_sec=first_n_sec,
            max_pf_count=max_pf_count,
            auto_close=auto_close,
            use_z_score=use_z_score,
        )

        # Reuse the existing plotter — it works on the same result dict structure
        # but save into the PF-tagged directory
        plot_2D_pv_correlation_results(
            pv_results,
            mouse_groups,
            PLOTS_DIR=PLOTS_DIR,
            sess_1_label="TFC_cond",
            sess_2_label=test_label,
            n_bins=n_bins,
            auto_close=auto_close,
            dir_suffix=f"_PF{pf_tag}_{_z_score_dir_tag(use_z_score)}",
        )

        all_results[test_label] = pv_results

    return all_results


# ===========================================================================
#  Pooled (neuron-level) versions — stability, shift, dimensionality-PF
# ===========================================================================

def compute_place_field_stability_PF_pooled(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    max_pf_count=None, auto_close=True,
):
    """
    Same as compute_place_field_stability_PF but pools all per-neuron r
    values across mice within each group, rather than taking the per-mouse
    median first.
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(PLOTS_DIR, f"field_stability_PF{pf_tag}_pooled")
    os.makedirs(save_dir, exist_ok=True)
    _copy_si_methods_template("pf_stability_pooled_methods.txt", save_dir)

    tfc_dict = session_dicts.get("TFC_cond", {})
    test_labels = [l for l in session_dicts if l != "TFC_cond"]
    stab_results = {}

    for test_label in sorted(test_labels):
        test_dict = session_dicts.get(test_label, {})
        mapping_str = mappings.get(test_label)
        if not mapping_str:
            continue
        stab_results[test_label] = {}
        mice = sorted(set(tfc_dict.keys()) & set(test_dict.keys()))

        for mouse in mice:
            s1, s2 = tfc_dict[mouse], test_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue
            if fm1.fluorescence_map_occup is None or fm2.fluorescence_map_occup is None:
                continue

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            if len(cells_1) < 3:
                continue

            pf1_set = set(_get_pf_cells(fm1, max_pf_count=max_pf_count))
            pf2_set = set(_get_pf_cells(fm2, max_pf_count=max_pf_count))

            paired_1, paired_2 = [], []
            for c1, c2 in zip(cells_1, cells_2):
                if c1 in pf1_set and c2 in pf2_set:
                    paired_1.append(c1)
                    paired_2.append(c2)

            if len(paired_1) < 2:
                continue

            per_neuron_r = np.full(len(paired_1), np.nan)
            for ci, (c1, c2) in enumerate(zip(paired_1, paired_2)):
                rm_a = _get_fm_ratemap(fm1, c1)
                rm_b = _get_fm_ratemap(fm2, c2)
                h = min(rm_a.shape[0], rm_b.shape[0])
                w = min(rm_a.shape[1], rm_b.shape[1])
                a_flat = rm_a[:h, :w].ravel().astype(float)
                b_flat = rm_b[:h, :w].ravel().astype(float)
                fin = np.isfinite(a_flat) & np.isfinite(b_flat) & (a_flat != 0) & (b_flat != 0)
                if np.sum(fin) >= 3 and np.std(a_flat[fin]) > 0 and np.std(b_flat[fin]) > 0:
                    per_neuron_r[ci], _ = pearsonr(a_flat[fin], b_flat[fin])

            stab_results[test_label][mouse] = {
                "group": group,
                "per_neuron_r": per_neuron_r,
                "n_paired": len(paired_1),
            }

    # ---- Plot: pooled per-neuron r by group ----
    from scipy.stats import f_oneway as _f
    for test_label, tdata in stab_results.items():
        groups_data = {}
        for mouse, md in tdata.items():
            v = md["per_neuron_r"]
            valid = v[np.isfinite(v)]
            groups_data.setdefault(md["group"], []).extend(valid.tolist())
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="Per-neuron r (pooled)",
                          title=f"PF stability{pf_tag} pooled\nTFC→{test_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_stability_pooled_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # Histogram overlay
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        for g in _PV_GROUP_ORDER:
            vals = groups_data.get(g, np.array([]))
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.45,
                        color=_PV_BOX_COLORS.get(g, "gray"),
                        edgecolor="black", linewidth=0.3,
                        label=f"{_PV_GROUP_LABELS[g]} (n={len(vals)})",
                        density=True)
        ax.set_xlabel("Per-neuron r", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"PF stability dist.{pf_tag} pooled — TFC→{test_label}", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_stability_pooled_hist_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # ---- Per-neuron LMM forest plot (r ~ group + (1|mouse)) ----
        records = []
        for mouse, md in tdata.items():
            g = md["group"]
            for v in md["per_neuron_r"]:
                if np.isfinite(v):
                    records.append(
                        {"group": g, "mouse": mouse, "r": float(v)})
        if records:
            df_r = pd.DataFrame(records)
            df_r.to_csv(
                os.path.join(save_dir,
                             f"PF_stability_pooled{pf_tag}_{test_label}_neurons.csv"),
                index=False)
            _pooled_lmm_fit_and_plot(
                df_r, save_dir,
                panel_label=f"PF_stability_pooled{pf_tag}_{test_label}",
                ylabel="Per-neuron r",
                value_col="r",
                auto_close=auto_close,
            )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  PF Stability POOLED (PF{pf_tag})")
    print(f"{'='*70}")
    for test_label, tdata in stab_results.items():
        gd = {}
        for mouse, md in tdata.items():
            v = md["per_neuron_r"][np.isfinite(md["per_neuron_r"])]
            gd.setdefault(md["group"], []).extend(v.tolist())
        arrs = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrs) >= 2:
            F, p = _f(*arrs)
            gs = "  ".join(f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.3f}±{np.std(gd[g]):.3f}(n={len(gd[g])})"
                           for g in _PV_GROUP_ORDER if g in gd)
            print(f"  TFC→{test_label:12s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    return stab_results


def compute_pf_centroid_shift_pooled(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    max_pf_count=1, auto_close=True,
):
    """
    Same as compute_pf_centroid_shift but pools all per-neuron shift values
    across mice within each group.
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(PLOTS_DIR, f"pf_centroid_shift{pf_tag}_pooled")
    os.makedirs(save_dir, exist_ok=True)
    _copy_si_methods_template("pf_centroid_shift_pooled_methods.txt", save_dir)

    tfc_dict = session_dicts.get("TFC_cond", {})
    test_labels = [l for l in session_dicts if l != "TFC_cond"]
    shift_results = {}

    for test_label in sorted(test_labels):
        test_dict = session_dicts.get(test_label, {})
        mapping_str = mappings.get(test_label)
        if not mapping_str:
            continue
        shift_results[test_label] = {}
        mice = sorted(set(tfc_dict.keys()) & set(test_dict.keys()))

        for mouse in mice:
            s1, s2 = tfc_dict[mouse], test_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            pf1_set = set(_get_pf_cells(fm1, max_pf_count=max_pf_count))
            pf2_set = set(_get_pf_cells(fm2, max_pf_count=max_pf_count))

            shifts = []
            per_neuron_acquired = []
            per_neuron_dropped = []
            for c1, c2 in zip(cells_1, cells_2):
                if c1 not in pf1_set or c2 not in pf2_set:
                    continue
                res = _match_pf_shifts(fm1, c1, fm2, c2)
                shifts.extend(res["shifts"])
                # Only count neurons with unequal PF numbers across sessions
                if res["n_acquired"] + res["n_dropped"] > 0:
                    per_neuron_acquired.append(res["n_acquired"])
                    per_neuron_dropped.append(res["n_dropped"])

            if len(shifts) < 2:
                continue

            bw = getattr(fm1.loc, 'bin_width', 1.0)
            shift_results[test_label][mouse] = {
                "group": group,
                "shifts_bins": np.array(shifts),
                "shifts_px": np.array(shifts) * bw,
                "n_cells": len(shifts),
                "bin_width": bw,
                "per_neuron_acquired": np.array(per_neuron_acquired),
                "per_neuron_dropped": np.array(per_neuron_dropped),
            }

    # ---- Plot: pooled per-neuron shifts by group ----
    from scipy.stats import f_oneway as _f
    for test_label, tdata in shift_results.items():
        groups_data = {}
        for mouse, md in tdata.items():
            groups_data.setdefault(md["group"], []).extend(md["shifts_bins"].tolist())
        groups_data = {g: np.array(v) for g, v in groups_data.items()}

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        _pv_paper_boxplot(ax, groups_data,
                          ylabel="PF shift (bins, pooled)",
                          title=f"PF centroid shift{pf_tag} pooled\nTFC→{test_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_shift_pooled_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # Histogram overlay
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        for g in _PV_GROUP_ORDER:
            vals = groups_data.get(g, np.array([]))
            if len(vals) > 0:
                ax.hist(vals, bins=25, alpha=0.45,
                        color=_PV_BOX_COLORS.get(g, "gray"),
                        edgecolor="black", linewidth=0.3,
                        label=f"{_PV_GROUP_LABELS[g]} (n={len(vals)})",
                        density=True)
        ax.set_xlabel("PF centroid shift (bins)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"PF shift dist.{pf_tag} pooled — TFC→{test_label}", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"PF_shift_pooled_hist_{test_label}.png"),
                    dpi=300, bbox_inches="tight")
        if auto_close:
            plt.close(fig)
        else:
            plt.show()

        # ---- Per-neuron LMM forest plot (shift ~ group + (1|mouse)) ----
        records = []
        for mouse, md in tdata.items():
            g = md["group"]
            for v in md["shifts_bins"]:
                if np.isfinite(v):
                    records.append(
                        {"group": g, "mouse": mouse, "shift": float(v)})
        if records:
            df_s = pd.DataFrame(records)
            df_s.to_csv(
                os.path.join(save_dir,
                             f"PF_shift_pooled{pf_tag}_{test_label}_neurons.csv"),
                index=False)
            _pooled_lmm_fit_and_plot(
                df_s, save_dir,
                panel_label=f"PF_shift_pooled{pf_tag}_{test_label}",
                ylabel="PF shift (bins)",
                value_col="shift",
                auto_close=auto_close,
            )

        # ---- Acquired PFs per neuron (pooled across mice) ----
        acq_data = {}
        for mouse, md in tdata.items():
            arr = md["per_neuron_acquired"]
            if len(arr) > 0:
                acq_data.setdefault(md["group"], []).extend(arr.tolist())
        acq_data = {g: np.array(v) for g, v in acq_data.items()}

        if any(len(v) >= 2 for v in acq_data.values()):
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
            _pv_paper_boxplot(ax, acq_data,
                              ylabel="# acquired PFs / neuron (pooled)",
                              title=f"Acquired PFs{pf_tag} pooled\nTFC\u2192{test_label}")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"PF_acquired_pooled_{test_label}.png"),
                        dpi=300, bbox_inches="tight")
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

        # ---- Dropped PFs per neuron (pooled across mice) ----
        drop_data = {}
        for mouse, md in tdata.items():
            arr = md["per_neuron_dropped"]
            if len(arr) > 0:
                drop_data.setdefault(md["group"], []).extend(arr.tolist())
        drop_data = {g: np.array(v) for g, v in drop_data.items()}

        if any(len(v) >= 2 for v in drop_data.values()):
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
            _pv_paper_boxplot(ax, drop_data,
                              ylabel="# dropped PFs / neuron (pooled)",
                              title=f"Dropped PFs{pf_tag} pooled\nTFC\u2192{test_label}")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"PF_dropped_pooled_{test_label}.png"),
                        dpi=300, bbox_inches="tight")
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

        # ---- Discrimination index: (NumAcq - NumLost) / (NumAcq + NumLost) ----
        # Truly pooled: pool all neurons across mice within each group,
        # count gainers (acquired>0) and losers (dropped>0), compute group DI.
        # Bootstrap 95% CI on DI; pairwise permutation tests with Holm correction.
        group_acq_drop = {}   # {group: (acq_array, drop_array)}
        for mouse, md in tdata.items():
            g = md["group"]
            acq_arr = md["per_neuron_acquired"]
            drop_arr = md["per_neuron_dropped"]
            if len(acq_arr) == 0 and len(drop_arr) == 0:
                continue
            prev_a, prev_d = group_acq_drop.get(g, ([], []))
            prev_a.extend(acq_arr.tolist())
            prev_d.extend(drop_arr.tolist())
            group_acq_drop[g] = (prev_a, prev_d)

        groups_to_plot = [g for g in _PV_GROUP_ORDER if g in group_acq_drop]

        if len(groups_to_plot) >= 2:
            # Compute DI per group
            def _compute_di(acq_list, drop_list):
                n_acq = sum(1 for v in acq_list if v > 0)
                n_lost = sum(1 for v in drop_list if v > 0)
                denom = n_acq + n_lost
                return (n_acq - n_lost) / denom if denom > 0 else 0.0

            di_vals = []
            di_cis = []
            n_neurons = []
            rng = np.random.RandomState(42)
            for g in groups_to_plot:
                acq_list, drop_list = group_acq_drop[g]
                acq_arr_g = np.array(acq_list)
                drop_arr_g = np.array(drop_list)
                di_obs = _compute_di(acq_list, drop_list)
                di_vals.append(di_obs)
                n_neurons.append(len(acq_list))
                # Bootstrap 95% CI
                n = len(acq_list)
                boot_dis = np.empty(2000)
                for bi in range(2000):
                    idx = rng.randint(0, n, size=n)
                    boot_dis[bi] = _compute_di(acq_arr_g[idx], drop_arr_g[idx])
                lo = np.percentile(boot_dis, 2.5)
                hi = np.percentile(boot_dis, 97.5)
                di_cis.append((di_obs - lo, hi - di_obs))

            # Pairwise permutation test with Holm correction
            from itertools import combinations
            pair_pvals = []
            pair_labels = []
            for i, j in combinations(range(len(groups_to_plot)), 2):
                gi, gj = groups_to_plot[i], groups_to_plot[j]
                acq_i, drop_i = np.array(group_acq_drop[gi][0]), np.array(group_acq_drop[gi][1])
                acq_j, drop_j = np.array(group_acq_drop[gj][0]), np.array(group_acq_drop[gj][1])
                obs_diff = abs(di_vals[i] - di_vals[j])
                pooled_acq = np.concatenate([acq_i, acq_j])
                pooled_drop = np.concatenate([drop_i, drop_j])
                ni = len(acq_i)
                n_perm = 5000
                count = 0
                for _ in range(n_perm):
                    perm = rng.permutation(len(pooled_acq))
                    a1, a2 = pooled_acq[perm[:ni]], pooled_acq[perm[ni:]]
                    d1, d2 = pooled_drop[perm[:ni]], pooled_drop[perm[ni:]]
                    di1 = _compute_di(a1, d1)
                    di2 = _compute_di(a2, d2)
                    if abs(di1 - di2) >= obs_diff:
                        count += 1
                pair_pvals.append((count + 1) / (n_perm + 1))
                pair_labels.append((i, j))

            # Holm correction
            valid_mask = np.isfinite(pair_pvals)
            adj_pvals = np.full(len(pair_pvals), np.nan)
            if np.any(valid_mask):
                from statsmodels.stats.multitest import multipletests
                _, adj_p, _, _ = multipletests(
                    np.array(pair_pvals)[valid_mask], method="holm")
                adj_pvals[valid_mask] = adj_p

            x = np.arange(len(groups_to_plot))
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
            bar_colors = [_PV_BOX_COLORS.get(g, "gray") for g in groups_to_plot]
            err_lo = [ci[0] for ci in di_cis]
            err_hi = [ci[1] for ci in di_cis]
            ax.bar(x, di_vals, color=bar_colors, edgecolor="black",
                   linewidth=0.6, width=0.55, zorder=2)
            ax.errorbar(x, di_vals, yerr=[err_lo, err_hi],
                        fmt="none", ecolor="black", capsize=4, lw=1.0, zorder=3)
            ax.axhline(0, color="gray", ls="--", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([_PV_GROUP_LABELS.get(g, g) for g in groups_to_plot])
            ax.set_ylabel("DI  (Acq\u2212Lost) / (Acq+Lost)")
            ax.set_title(f"PF gain/loss DI{pf_tag} pooled\nTFC\u2192{test_label}")
            # Annotate n
            for xi, n in zip(x, n_neurons):
                ax.text(xi, ax.get_ylim()[0], f"n={n}", ha="center",
                        va="top", fontsize=7, color="gray")
            # Annotate pairwise significance
            y_max = max(d + ci[1] for d, ci in zip(di_vals, di_cis))
            y_min = min(d - ci[0] for d, ci in zip(di_vals, di_cis))
            y_range = y_max - y_min if y_max != y_min else 0.1
            y_step = y_range * 0.08
            y_bar = y_max + y_step
            for (i_idx, j_idx), pv in zip(pair_labels, adj_pvals):
                if np.isfinite(pv) and pv < 0.05:
                    stars = "***" if pv < 0.001 else ("**" if pv < 0.01 else "*")
                    ax.plot([i_idx, j_idx], [y_bar, y_bar], color="black", lw=0.8)
                    ax.text((i_idx + j_idx) / 2, y_bar, stars,
                            ha="center", va="bottom", fontsize=9)
                    y_bar += y_step * 1.5
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"PF_gainloss_DI_{test_label}.png"),
                        dpi=300, bbox_inches="tight")
            if auto_close:
                plt.close(fig)
            else:
                plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"  PF Centroid Shift POOLED (PF{pf_tag})")
    print(f"{'='*70}")
    for test_label, tdata in shift_results.items():
        gd = {}
        for mouse, md in tdata.items():
            gd.setdefault(md["group"], []).extend(md["shifts_bins"].tolist())
        arrs = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrs) >= 2:
            F, p = _f(*arrs)
            gs = "  ".join(
                f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.2f}±{np.std(gd[g]):.2f}(n={len(gd[g])})"
                for g in _PV_GROUP_ORDER if g in gd)
            print(f"  TFC→{test_label:12s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    return shift_results


def compute_population_dimensionality_PF(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    first_n_sec=180.0, max_pf_count=None, auto_close=True,
):
    """
    Compute PCA participation ratio using only neurons with place fields.

    One PR per mouse per session (PF neurons only).
    Cross-registered variant: PF neurons that are cross-registered AND have
    PFs in both sessions.

    Note: PR is inherently a per-mouse metric (one value per recording),
    so "pooled" here means PF-filtered, not neuron-pooled.
    """
    pf_tag = "" if max_pf_count is None else f"_npf{max_pf_count}"
    save_dir = os.path.join(PLOTS_DIR, f"pop_dimensionality_PF{pf_tag}")
    os.makedirs(save_dir, exist_ok=True)

    tfc_dict = session_dicts.get("TFC_cond", {})
    crossreg_families = {
        "TFC_cond": None,
        "Test_A": "A", "Test_A_1wk": "A",
        "Test_B": "B", "Test_B_1wk": "B",
    }

    dim_results = {}
    all_sess_labels = sorted(session_dicts.keys())

    for sess_label in all_sess_labels:
        sess_dict = session_dicts.get(sess_label, {})
        dim_results[sess_label] = {}

        for mouse, sess in sorted(sess_dict.items()):
            group = mouse_groups.get(mouse, "NA")
            fm = getattr(sess, 'fm', None)
            if fm is None or fm.pf is None:
                continue

            T = sess.S.shape[1]
            mask = _build_pretone_mask(sess, first_n_sec=first_n_sec)[:T]

            # All-PF PR: PF neurons in this session
            pf_cells = _get_pf_cells(fm, max_pf_count=max_pf_count)
            if len(pf_cells) >= 3:
                S_pf = _get_mapped_S(sess, pf_cells)[:, mask]
                pr_all_pf = _participation_ratio(S_pf)
            else:
                pr_all_pf = np.nan

            # Crossreg-PF PR: PF neurons that are cross-registered
            fam = crossreg_families.get(sess_label)
            families_to_use = [fam] if fam else ["A", "B"]
            pr_cr = {}
            for f in families_to_use:
                # Find partner session for crossreg
                if sess_label == "TFC_cond":
                    # For TFC, use 48h partner of family f
                    partner_label = _FAMILY_PAIRS[f][0]
                else:
                    partner_label = "TFC_cond"
                partner_dict = session_dicts.get(partner_label, {})
                partner_sess = partner_dict.get(mouse)
                if partner_sess is None:
                    pr_cr[f] = np.nan
                    continue
                mapping_str = mappings.get(_FAMILY_PAIRS[f][0])
                if not mapping_str:
                    pr_cr[f] = np.nan
                    continue

                cells_this, cells_partner = _get_crossreg_cells(sess, partner_sess, mapping_str)
                if len(cells_this) < 3:
                    pr_cr[f] = np.nan
                    continue

                # Filter to PF neurons in both sessions
                fm_partner = getattr(partner_sess, 'fm', None)
                if fm_partner is None or fm_partner.pf is None:
                    pr_cr[f] = np.nan
                    continue
                pf_this = set(_get_pf_cells(fm, max_pf_count=max_pf_count))
                pf_partner = set(_get_pf_cells(fm_partner, max_pf_count=max_pf_count))

                filt_cells = [c for c, cp in zip(cells_this, cells_partner)
                              if c in pf_this and cp in pf_partner]
                if len(filt_cells) < 3:
                    pr_cr[f] = np.nan
                    continue

                S_cr = _get_mapped_S(sess, filt_cells)[:, mask]
                pr_cr[f] = _participation_ratio(S_cr)

            dim_results[sess_label][mouse] = {
                "group": group,
                "pr_all_pf": pr_all_pf,
                "pr_crossreg_pf": pr_cr,
                "n_pf": len(pf_cells),
            }

    # ---- Plot ----
    for neuron_set, set_label in [("all_pf", "PF neurons"), ("crossreg_pf", "PF cross-reg")]:
        for sess_label in all_sess_labels:
            if sess_label not in dim_results:
                continue
            fam_key = crossreg_families.get(sess_label)
            families = [fam_key] if fam_key else ["A", "B"]

            for f in families:
                groups_data = {}
                for mouse, md in dim_results[sess_label].items():
                    g = md["group"]
                    if neuron_set == "all_pf":
                        v = md["pr_all_pf"]
                    else:
                        v = md["pr_crossreg_pf"].get(f, np.nan)
                    if np.isfinite(v):
                        groups_data.setdefault(g, []).append(v)
                groups_data = {g: np.array(v) for g, v in groups_data.items()}

                fam_tag = f"_fam{f}" if neuron_set == "crossreg_pf" else ""
                fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
                _pv_paper_boxplot(ax, groups_data,
                                  ylabel="Participation ratio",
                                  title=f"{sess_label} — {set_label}{pf_tag}{fam_tag}")
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir,
                            f"PR_{neuron_set}{fam_tag}_{sess_label}.png"),
                            dpi=300, bbox_inches="tight")
                if auto_close:
                    plt.close(fig)
                else:
                    plt.show()

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Dimensionality PF{pf_tag}")
    print(f"{'='*70}")
    from scipy.stats import f_oneway as _f
    for sess_label in all_sess_labels:
        gd = {}
        for mouse, md in dim_results.get(sess_label, {}).items():
            v = md["pr_all_pf"]
            if np.isfinite(v):
                gd.setdefault(md["group"], []).append(v)
        arrays = [np.array(gd[g]) for g in _PV_GROUP_ORDER if g in gd and len(gd[g]) >= 2]
        if len(arrays) >= 2:
            F, p = _f(*arrays)
            gs = "  ".join(f"{_PV_GROUP_LABELS[g]}={np.mean(gd[g]):.1f}±{np.std(gd[g]):.1f}"
                           for g in _PV_GROUP_ORDER if g in gd)
            print(f"  {sess_label:15s}  F={F:.2f}  p={p:.4g}  {gs}")
    print(f"{'='*70}\n")
    return dim_results


# ===========================================================================
#  Place Field Turnover (Ziv et al., 2013 style)
# ===========================================================================

def _classify_pf_turnover(fm1, cells_1, fm2, cells_2):
    """Classify cross-registered neuron pairs into PF turnover categories.

    Categories:
        Stable  — place cell in both sessions
          (subtypes: stable-same, stable-reduced, stable-expanded)
        Gained  — non-place-cell in session 1, place cell in session 2
        Lost    — place cell in session 1, non-place-cell in session 2
        Silent  — non-place-cell in either session

    Returns dict with counts (n_stable, n_stable_same, n_stable_reduced,
    n_stable_expanded, n_gained, n_lost, n_silent, n_crossreg) and derived
    rates (recurrence_prob, turnover_rate, gain_rate, loss_rate).
    """
    pf1_set = set(_get_pf_cells(fm1, max_pf_count=None))
    pf2_set = set(_get_pf_cells(fm2, max_pf_count=None))

    n_stable_same = 0
    n_stable_reduced = 0
    n_stable_expanded = 0
    n_gained = 0
    n_lost = 0
    n_silent = 0
    for c1, c2 in zip(cells_1, cells_2):
        is_pf1 = c1 in pf1_set
        is_pf2 = c2 in pf2_set
        if is_pf1 and is_pf2:
            st = _stable_subtype(fm1, c1, fm2, c2)
            if st == "stable-same":
                n_stable_same += 1
            elif st == "stable-reduced":
                n_stable_reduced += 1
            else:
                n_stable_expanded += 1
        elif (not is_pf1) and is_pf2:
            n_gained += 1
        elif is_pf1 and (not is_pf2):
            n_lost += 1
        else:
            n_silent += 1

    n_stable = n_stable_same + n_stable_reduced + n_stable_expanded
    n_crossreg = len(cells_1)
    n_ever_pc = n_stable + n_gained + n_lost
    rec_denom = n_stable + n_lost

    return {
        "n_crossreg": n_crossreg,
        "n_stable": n_stable,
        "n_stable_same": n_stable_same,
        "n_stable_reduced": n_stable_reduced,
        "n_stable_expanded": n_stable_expanded,
        "n_gained": n_gained,
        "n_lost": n_lost,
        "n_silent": n_silent,
        "recurrence_prob": n_stable / rec_denom if rec_denom > 0 else np.nan,
        "turnover_rate": (n_gained + n_lost) / n_ever_pc if n_ever_pc > 0 else np.nan,
        "gain_rate": n_gained / n_crossreg if n_crossreg > 0 else np.nan,
        "loss_rate": n_lost / n_crossreg if n_crossreg > 0 else np.nan,
    }


def _collect_pf_turnover_labels(fm1, cells_1, fm2, cells_2):
    """Return a list of per-neuron category labels for a cross-registered pair.

    Labels are one of: ``'stable-same'``, ``'stable-reduced'``,
    ``'stable-expanded'``, ``'gained'``, ``'lost'``, ``'silent'``.

    This is the neuron-level analogue of ``_classify_pf_turnover``, used for
    pooled analyses that aggregate across mice.
    """
    pf1_set = set(_get_pf_cells(fm1, max_pf_count=None))
    pf2_set = set(_get_pf_cells(fm2, max_pf_count=None))
    labels = []
    for c1, c2 in zip(cells_1, cells_2):
        is_pf1 = c1 in pf1_set
        is_pf2 = c2 in pf2_set
        if is_pf1 and is_pf2:
            labels.append(_stable_subtype(fm1, c1, fm2, c2))
        elif (not is_pf1) and is_pf2:
            labels.append("gained")
        elif is_pf1 and (not is_pf2):
            labels.append("lost")
        else:
            labels.append("silent")
    return labels


def _normalize_pf_turnover_comparison_specs(session_dicts, mappings,
                                            comparison_specs=None):
    """Return normalized PF turnover comparison specs.

    Each returned spec is a dict with keys:
        ``sess1_label``
        ``sess2_label``
        ``mapping``
        ``output_key``
        ``display_label``

    When ``comparison_specs`` is ``None``, the historical default is used:
    ``TFC_cond`` is session 1 and every other available session is session 2.
    """
    if comparison_specs is None:
        tfc_label = "TFC_cond"
        if tfc_label not in session_dicts:
            raise KeyError("session_dicts must contain 'TFC_cond' for default PF turnover comparisons.")
        specs = []
        for sess2_label in sorted(k for k in session_dicts if k != tfc_label):
            mapping_str = mappings.get(sess2_label)
            if not mapping_str:
                raise KeyError(f"Missing PF turnover mapping for session '{sess2_label}'.")
            specs.append({
                "sess1_label": tfc_label,
                "sess2_label": sess2_label,
                "mapping": mapping_str,
                "output_key": sess2_label,
                "display_label": f"TFC→{sess2_label}",
            })
        return specs

    specs = []
    for spec in comparison_specs:
        sess1_label = spec["sess1_label"]
        sess2_label = spec["sess2_label"]
        if sess1_label not in session_dicts:
            raise KeyError(f"session_dicts is missing requested sess1_label '{sess1_label}'.")
        if sess2_label not in session_dicts:
            raise KeyError(f"session_dicts is missing requested sess2_label '{sess2_label}'.")
        mapping_str = spec.get("mapping")
        if not mapping_str:
            mapping_key = spec.get("mapping_key")
            if mapping_key is None:
                raise KeyError(
                    f"Comparison {sess1_label}->{sess2_label} must define 'mapping' or 'mapping_key'.")
            mapping_str = mappings.get(mapping_key)
            if not mapping_str:
                raise KeyError(
                    f"mappings is missing requested mapping_key '{mapping_key}' for {sess1_label}->{sess2_label}.")
        specs.append({
            "sess1_label": sess1_label,
            "sess2_label": sess2_label,
            "mapping": mapping_str,
            "output_key": spec.get("output_key", f"{sess1_label}_to_{sess2_label}"),
            "display_label": spec.get("display_label", f"{sess1_label}→{sess2_label}"),
        })
    return specs


def compute_pf_turnover(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    auto_close=True,
    comparison_specs=None,
):
    """Place field turnover analysis across arbitrary session pairs.

    By default, reproduces the historical behaviour:
    ``TFC_cond`` is used as session 1 and every other available session is
    used as session 2. When ``comparison_specs`` is provided, each comparison
    is run explicitly using the supplied session labels and mapping.
    """
    save_dir = os.path.join(PLOTS_DIR, "pf_turnover")
    os.makedirs(save_dir, exist_ok=True)

    specs = _normalize_pf_turnover_comparison_specs(
        session_dicts, mappings, comparison_specs=comparison_specs)
    turnover_results = {}

    for spec in specs:
        sess1_label = spec["sess1_label"]
        sess2_label = spec["sess2_label"]
        output_key = spec["output_key"]
        sess1_dict = session_dicts[sess1_label]
        sess2_dict = session_dicts[sess2_label]
        mapping_str = spec["mapping"]
        tdata = {}
        mice = sorted(set(sess1_dict.keys()) & set(sess2_dict.keys()))

        for mouse in mice:
            s1, s2 = sess1_dict[mouse], sess2_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            md = _classify_pf_turnover(fm1, cells_1, fm2, cells_2)
            md["group"] = group
            tdata[mouse] = md

        turnover_results[output_key] = {
            "display_label": spec["display_label"],
            "results": tdata,
        }

    print(f"\n{'='*70}")
    print(f"  Place Field Turnover")
    print(f"{'='*70}")
    for output_key, entry in turnover_results.items():
        display_label = entry["display_label"]
        tdata = entry["results"]
        test_dir = os.path.join(save_dir, output_key)
        os.makedirs(test_dir, exist_ok=True)
        plot_turnover_metric_boxplots(
            tdata, test_dir,
            title_prefix=display_label,
            auto_close=auto_close)
        group_counts = _aggregate_turnover_counts(tdata)
        plot_turnover_stacked_bar(
            group_counts, test_dir,
            title=f"PF turnover — {display_label}",
            fname_stem=f"turnover_stacked_{output_key}",
            auto_close=auto_close)

        gd = {}
        for mouse, md in tdata.items():
            gd.setdefault(md["group"], []).append(md)
        parts = []
        for g in _PV_GROUP_ORDER:
            if g not in gd:
                continue
            recs = [d["recurrence_prob"] for d in gd[g] if np.isfinite(d["recurrence_prob"])]
            turns = [d["turnover_rate"] for d in gd[g] if np.isfinite(d["turnover_rate"])]
            if recs:
                parts.append(f"{_PV_GROUP_LABELS[g]}: rec={np.mean(recs):.2f}"
                             f" turn={np.mean(turns):.2f} (n={len(recs)})")
        print(f"  {display_label:20s}  {'  '.join(parts)}")
    print(f"{'='*70}\n")
    return {k: v["results"] for k, v in turnover_results.items()}


# ===========================================================================
#  Place Field Turnover — Pooled (neuron-level)
# ===========================================================================

def compute_pf_turnover_pooled(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    auto_close=True,
    comparison_specs=None,
):
    """
    Pooled (neuron-level) place field turnover analysis.

    Same classification as compute_pf_turnover (Stable/Gained/Lost/Silent),
    but instead of one summary rate per mouse, every cross-registered neuron
    is labelled and pooled across mice within each group.

    Plots:
        - Per-neuron category proportions (stacked bar per group)
        - Boxplots of per-neuron binary indicators pooled by group
          (fraction of neurons that are Stable, Gained, Lost — tested via
          chi-squared across groups)
    """
    save_dir = os.path.join(PLOTS_DIR, "pf_turnover_pooled")
    os.makedirs(save_dir, exist_ok=True)

    specs = _normalize_pf_turnover_comparison_specs(
        session_dicts, mappings, comparison_specs=comparison_specs)
    turnover_results = {}

    for spec in specs:
        sess1_label = spec["sess1_label"]
        sess2_label = spec["sess2_label"]
        output_key = spec["output_key"]
        sess1_dict = session_dicts[sess1_label]
        sess2_dict = session_dicts[sess2_label]
        mapping_str = spec["mapping"]
        group_labels = {g: [] for g in _PV_GROUP_ORDER}
        mice = sorted(set(sess1_dict.keys()) & set(sess2_dict.keys()))

        for mouse in mice:
            s1, s2 = sess1_dict[mouse], sess2_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            labels = _collect_pf_turnover_labels(fm1, cells_1, fm2, cells_2)
            if group in group_labels:
                group_labels[group].extend(labels)

        turnover_results[output_key] = {
            "display_label": spec["display_label"],
            "group_labels": group_labels,
        }

    print(f"\n{'='*70}")
    print(f"  Place Field Turnover POOLED")
    print(f"{'='*70}")
    for output_key, entry in turnover_results.items():
        display_label = entry["display_label"]
        gdata = entry["group_labels"]
        test_dir = os.path.join(save_dir, output_key)
        os.makedirs(test_dir, exist_ok=True)

        gc = {}
        for g in _PV_GROUP_ORDER:
            if not gdata[g]:
                continue
            labs = gdata[g]
            gc[g] = {cat: sum(1 for c in labs if c == cat)
                     for cat in _TURNOVER_CAT_ORDER}
            gc[g]["total"] = len(labs)

        plot_turnover_stacked_bar(
            gc, test_dir,
            title=f"PF turnover — {display_label}",
            fname_stem=f"turnover_stacked_pooled_{output_key}",
            auto_close=auto_close)
        plot_turnover_pooled_proportion_bars(
            gdata, test_dir,
            title_prefix=display_label,
            auto_close=auto_close)

        parts = []
        for g in _PV_GROUP_ORDER:
            if not gdata[g]:
                continue
            n = len(gdata[g])
            n_s = sum(1 for c in gdata[g] if _count_is_stable(c))
            n_g = sum(1 for c in gdata[g] if c == "gained")
            n_l = sum(1 for c in gdata[g] if c == "lost")
            parts.append(f"{_PV_GROUP_LABELS[g]}: S={n_s} G={n_g} L={n_l} "
                         f"(n={n}, rec={n_s/(n_s+n_l):.2f})" if (n_s+n_l) > 0
                         else f"{_PV_GROUP_LABELS[g]}: S={n_s} G={n_g} L={n_l} (n={n})")
        print(f"  {display_label:20s}  {'  '.join(parts)}")
    print(f"{'='*70}\n")
    return {k: v["group_labels"] for k, v in turnover_results.items()}


# ===========================================================================
#  Place Field Turnover Example Gallery
# ===========================================================================

def plot_pf_turnover_examples(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    n_examples=3, auto_close=True, comparison_specs=None,
):
    """
    Plot example neurons for each turnover category across groups.

    By default, uses TFC_cond as session 1 and each available non-TFC session
    as session 2. If ``comparison_specs`` is provided, examples are generated
    for those explicit session pairs.
    """
    from matplotlib.patches import Ellipse
    from numpy import linalg

    save_dir = os.path.join(PLOTS_DIR, "pf_turnover_examples")
    os.makedirs(save_dir, exist_ok=True)

    specs = _normalize_pf_turnover_comparison_specs(
        session_dicts, mappings, comparison_specs=comparison_specs)

    for spec in specs:
        sess1_label = spec["sess1_label"]
        sess2_label = spec["sess2_label"]
        output_key = spec["output_key"]
        display_label = spec["display_label"]
        mapping_str = spec["mapping"]
        sess1_dict = session_dicts[sess1_label]
        sess2_dict = session_dicts[sess2_label]
        mice = sorted(set(sess1_dict.keys()) & set(sess2_dict.keys()))

        # Collect neuron info per group × category
        # {group: {cat: [(mouse, c1, c2, fm1, fm2), ...]}}
        _STABLE_SUBS = ["stable-same", "stable-reduced", "stable-expanded"]
        _GALLERY_CATS = _STABLE_SUBS + ["gained", "lost"]
        cat_neurons = {g: {cat: [] for cat in _GALLERY_CATS}
                       for g in _PV_GROUP_ORDER}

        for mouse in mice:
            s1, s2 = sess1_dict[mouse], sess2_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            if group not in cat_neurons:
                continue
            fm1 = getattr(s1, 'fm', None)
            fm2 = getattr(s2, 'fm', None)
            if fm1 is None or fm2 is None:
                continue

            cells_1, cells_2 = _get_crossreg_cells(s1, s2, mapping_str)
            pf1_set = set(_get_pf_cells(fm1, max_pf_count=None))
            pf2_set = set(_get_pf_cells(fm2, max_pf_count=None))

            for c1, c2 in zip(cells_1, cells_2):
                is_pf1 = c1 in pf1_set
                is_pf2 = c2 in pf2_set
                if is_pf1 and is_pf2:
                    cat = _stable_subtype(fm1, c1, fm2, c2)
                elif (not is_pf1) and is_pf2:
                    cat = "gained"
                elif is_pf1 and (not is_pf2):
                    cat = "lost"
                else:
                    continue  # skip silent for gallery
                cat_neurons[group][cat].append(
                    (mouse, c1, c2, fm1, fm2, s1, s2))

        # For each group × category, pick representative examples
        rng = np.random.RandomState(0)
        for group in _PV_GROUP_ORDER:
            if group not in cat_neurons:
                continue
            for cat in _GALLERY_CATS:
                pool = cat_neurons[group][cat]
                if len(pool) == 0:
                    continue
                # Pick up to n_examples (prefer neurons with more PFs for variety)
                idxs = rng.choice(len(pool),
                                  size=min(n_examples, len(pool)),
                                  replace=False)
                examples = [pool[i] for i in idxs]

                for ex_i, (mouse, c1, c2, fm1, fm2,
                           s1_ex, s2_ex) in enumerate(examples):
                    # Ensure A matrices are loaded
                    for sess in (s1_ex, s2_ex):
                        if not hasattr(sess, 'A') or sess.A is None:
                            sess.get_A_matrix()

                    fig, axes = plt.subplots(3, 2, figsize=(6, 8), dpi=300)

                    rmap1 = _get_fm_ratemap(fm1, c1)
                    rmap2 = _get_fm_ratemap(fm2, c2)

                    for ax, fm, cell_id, sess_label, rmap in [
                        (axes[0, 0], fm1, c1, sess1_label, rmap1),
                        (axes[0, 1], fm2, c2, sess2_label, rmap2),
                    ]:
                        # Rate map
                        ax.imshow(rmap, cmap="hot", interpolation="bilinear",
                                  origin="upper", aspect="equal")

                        # GMM ellipses for each merged place field
                        _pos = _pf_pos(fm, cell_id)
                        has_pf = (_pos in fm.pf.model_ and
                                  _pos in fm.pf.merged_means)
                        if has_pf:
                            merged = fm.pf.merged_means[_pos]
                            pf_colors = plt.cm.Set2(
                                np.linspace(0, 1, max(len(merged), 1)))
                            m_means = getattr(fm.pf, 'merged_means_', {}).get(_pos)
                            m_covs  = getattr(fm.pf, 'merged_covariances_', {}).get(_pos)
                            for pf_idx in range(len(merged)):
                                col = pf_colors[pf_idx % len(pf_colors)]
                                if m_means and m_covs:
                                    mu = m_means[pf_idx]
                                    cov_merged = m_covs[pf_idx]
                                else:
                                    # Fallback for old data without merged attrs
                                    model = fm.pf.model_[_pos]
                                    comp_idxs = merged[pf_idx]
                                    weights = model.weights_[comp_idxs]
                                    weights = weights / weights.sum()
                                    mu = weights @ model.means_[comp_idxs]
                                    cov_merged = np.zeros((2, 2))
                                    for ci, wi in zip(comp_idxs, weights):
                                        diff = model.means_[ci] - mu
                                        cov_merged += wi * (
                                            model.covariances_[ci]
                                            + np.outer(diff, diff))
                                eigvals, eigvecs = linalg.eigh(cov_merged)
                                widths = 2.0 * 2.0 * np.sqrt(eigvals)
                                angle = np.degrees(
                                    np.arctan2(eigvecs[1, 0],
                                               eigvecs[0, 0]))
                                ell = Ellipse(
                                    xy=(mu[1], mu[0]),
                                    width=widths[1], height=widths[0],
                                    angle=angle,
                                    edgecolor=col, facecolor="none",
                                    linewidth=1.5, linestyle="-")
                                ax.add_patch(ell)
                                ax.plot(mu[1], mu[0], "+",
                                        color=col, markersize=8, mew=1.5)

                        ax.set_title(sess_label, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Middle row: sig_responses overlay on ratemap
                    for ax, fm, cid, rmap, sess_label in [
                        (axes[1, 0], fm1, c1, rmap1,
                        f"sig resp {sess1_label} (c{c1})"),
                        (axes[1, 1], fm2, c2, rmap2,
                        f"sig resp {sess2_label} (c{c2})"),
                    ]:
                        ax.imshow(rmap, cmap="hot",
                                  interpolation="bilinear",
                                  origin="upper", aspect="equal")
                        _draw_sig_responses(ax, fm, cid)
                        ax.set_title(sess_label, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Bottom row: binarized A matrix spatial footprints
                    for ax, sess, cid, sess_label in [
                        (axes[2, 0], s1_ex, c1, f"FOV {sess1_label} (c{c1})"),
                        (axes[2, 1], s2_ex, c2,
                        f"FOV {sess2_label} (c{c2})"),
                    ]:
                        a_match = np.where(sess.A_idx == cid)[0]
                        if len(a_match):
                            footprint = sess.A[a_match[0]]
                            ax.imshow(footprint, cmap="gray",
                                      interpolation="nearest",
                                      origin="upper", aspect="equal",
                                      vmin=0, vmax=1)
                        else:
                            ax.text(0.5, 0.5, "No A", ha="center",
                                    va="center", transform=ax.transAxes,
                                    fontsize=8, color="red")
                        ax.set_title(sess_label, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    glabel = _PV_GROUP_LABELS.get(group, group)
                    fig.suptitle(
                        f"{glabel} — {cat.capitalize()} — {display_label} — "
                        f"{mouse} (c{c1}→c{c2})",
                        fontsize=9, y=0.99)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    fname = (f"example_{glabel}_{cat}_{ex_i}_"
                             f"{mouse}_{output_key}.png")
                    fig.savefig(os.path.join(save_dir, fname),
                                dpi=300, bbox_inches="tight")
                    if auto_close:
                        plt.close(fig)
                    else:
                        plt.show()

                    # ---- Gaussian-rendered duplicate figure ----
                    fig_g, axes_g = plt.subplots(3, 2,
                                                 figsize=(6, 8), dpi=300)

                    for ax, fm, cid, sess_label in [
                        (axes_g[0, 0], fm1, c1, f"{sess1_label} (Gauss)"),
                        (axes_g[0, 1], fm2, c2,
                        f"{sess2_label} (Gauss)"),
                    ]:
                        gmap = _render_gaussian_pf(fm, cid)
                        ax.imshow(gmap, cmap="viridis",
                                  interpolation="bilinear",
                                  origin="upper", aspect="equal",
                                  vmin=0, vmax=1)
                        ax.set_title(sess_label, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Middle row: sig_responses
                    for ax, fm, cid, rmap, sess_label in [
                        (axes_g[1, 0], fm1, c1, rmap1,
                        f"sig resp {sess1_label} (c{c1})"),
                        (axes_g[1, 1], fm2, c2, rmap2,
                        f"sig resp {sess2_label} (c{c2})"),
                    ]:
                        ax.imshow(rmap, cmap="hot",
                                  interpolation="bilinear",
                                  origin="upper", aspect="equal")
                        _draw_sig_responses(ax, fm, cid)
                        ax.set_title(sess_label, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Bottom row: A matrix footprints
                    for ax, sess, cid, sess_label in [
                        (axes_g[2, 0], s1_ex, c1,
                        f"FOV {sess1_label} (c{c1})"),
                        (axes_g[2, 1], s2_ex, c2,
                        f"FOV {sess2_label} (c{c2})"),
                    ]:
                        a_match = np.where(sess.A_idx == cid)[0]
                        if len(a_match):
                            footprint = sess.A[a_match[0]]
                            ax.imshow(footprint, cmap="gray",
                                      interpolation="nearest",
                                      origin="upper", aspect="equal",
                                      vmin=0, vmax=1)
                        else:
                            ax.text(0.5, 0.5, "No A", ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                    fontsize=8, color="red")
                        ax.set_title(sess_label, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    fig_g.suptitle(
                        f"{glabel} — {cat.capitalize()} (Gaussian) — "
                        f"{display_label} — {mouse} (c{c1}→c{c2})",
                        fontsize=9, y=0.99)
                    fig_g.tight_layout(rect=[0, 0, 1, 0.95])
                    fname_g = (f"gauss_{glabel}_{cat}_{ex_i}_"
                               f"{mouse}_{output_key}.png")
                    fig_g.savefig(os.path.join(save_dir, fname_g),
                                 dpi=300, bbox_inches="tight")
                    if auto_close:
                        plt.close(fig_g)
                    else:
                        plt.show()

    print(f"  [Gallery] Saved example PF turnover plots → {save_dir}")


# ===========================================================================
#  Place Field Turnover Example Gallery — VS style (3-session rows)
# ===========================================================================

def plot_pf_turnover_examples_VS(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    n_examples=3, auto_close=True,
):
    """
    Show example place-cell rate-map triplets (TFC → Test_48h → Test_1wk)
    for each group × turnover category (Stable / Gained / Lost).

    Style mirrors the original debug-snippets visualisations:
    viridis colormap, session × cell titles, GMM PF ellipses overlaid.

    One PNG per example neuron: 1×3 subplot row (TFC, 48h, 1wk).
    """
    from matplotlib.patches import Ellipse
    from numpy import linalg

    save_dir = os.path.join(PLOTS_DIR, "pf_turnover_examples_VS")
    os.makedirs(save_dir, exist_ok=True)

    tfc_label = "TFC_cond"
    tfc_dict = session_dicts.get(tfc_label, {})
    if not tfc_dict:
        return

    # Group test sessions into families: A (Test_A + Test_A_1wk), B (Test_B + Test_B_1wk)
    families = {}
    for lab48, lab1w in [("Test_A", "Test_A_1wk"), ("Test_B", "Test_B_1wk")]:
        if lab48 in session_dicts and lab1w in session_dicts:
            families[lab48] = (lab48, lab1w)

    def _draw_pf_ellipses(ax, fm, cell_id):
        """Overlay 2-sigma merged-covariance ellipse for each PF."""
        _pos = _pf_pos(fm, cell_id)
        has_pf = (_pos in getattr(fm.pf, 'model_', {}) and
                  _pos in getattr(fm.pf, 'merged_means', {}))
        if not has_pf:
            return
        merged = fm.pf.merged_means[_pos]
        pf_colors = plt.cm.Set2(np.linspace(0, 1, max(len(merged), 1)))
        m_means = getattr(fm.pf, 'merged_means_', {}).get(_pos)
        m_covs  = getattr(fm.pf, 'merged_covariances_', {}).get(_pos)
        for pf_idx in range(len(merged)):
            col = pf_colors[pf_idx % len(pf_colors)]
            if m_means and m_covs:
                mu = m_means[pf_idx]
                cov_merged = m_covs[pf_idx]
            else:
                model = fm.pf.model_[_pos]
                comp_idxs = merged[pf_idx]
                weights = model.weights_[comp_idxs]
                weights = weights / weights.sum()
                mu = weights @ model.means_[comp_idxs]
                cov_merged = np.zeros((2, 2))
                for ci, wi in zip(comp_idxs, weights):
                    diff = model.means_[ci] - mu
                    cov_merged += wi * (
                        model.covariances_[ci] + np.outer(diff, diff))
            eigvals, eigvecs = linalg.eigh(cov_merged)
            widths = 2.0 * 2.0 * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ell = Ellipse(xy=(mu[1], mu[0]),
                          width=widths[1], height=widths[0],
                          angle=angle,
                          edgecolor=col, facecolor="none",
                          linewidth=1.5, linestyle="-")
            ax.add_patch(ell)
            ax.plot(mu[1], mu[0], "+",
                    color=col, markersize=8, mew=1.5)

    for fam_key, (lab48, lab1w) in families.items():
        dict48 = session_dicts[lab48]
        dict1w = session_dicts[lab1w]
        mapping_str = mappings.get(lab48, "crossreg_unit_IDs")

        # Mice that have all three sessions
        mice = sorted(set(tfc_dict.keys()) & set(dict48.keys()) & set(dict1w.keys()))

        # Collect neurons classified by TFC→48h turnover category
        # Store (mouse, c_tfc, c_48, c_1w, fm_tfc, fm_48, fm_1w)
        _STABLE_SUBS = ["stable-same", "stable-reduced", "stable-expanded"]
        _GALLERY_CATS = _STABLE_SUBS + ["gained", "lost"]
        cat_neurons = {g: {cat: [] for cat in _GALLERY_CATS}
                       for g in _PV_GROUP_ORDER}

        for mouse in mice:
            s_tfc = tfc_dict[mouse]
            s_48  = dict48[mouse]
            s_1w  = dict1w[mouse]
            group = mouse_groups.get(mouse, "NA")
            if group not in cat_neurons:
                continue

            fm_tfc = getattr(s_tfc, 'fm', None)
            fm_48  = getattr(s_48,  'fm', None)
            fm_1w  = getattr(s_1w,  'fm', None)
            if fm_tfc is None or fm_48 is None or fm_1w is None:
                continue

            # Pairwise crossreg (TFC↔48h, TFC↔1wk) using same 3-session mapping
            cells_tfc_48, cells_48 = _get_crossreg_cells(s_tfc, s_48, mapping_str)
            cells_tfc_1w, cells_1w = _get_crossreg_cells(s_tfc, s_1w, mapping_str)

            # Build map: tfc_cell → (cell_48, cell_1w)  (only neurons in all 3)
            map_48 = dict(zip(cells_tfc_48, cells_48))
            map_1w = dict(zip(cells_tfc_1w, cells_1w))
            common_tfc = set(map_48.keys()) & set(map_1w.keys())

            pf_tfc_set = set(_get_pf_cells(fm_tfc, max_pf_count=None))
            pf_48_set  = set(_get_pf_cells(fm_48,  max_pf_count=None))

            for c_tfc in common_tfc:
                c_48 = map_48[c_tfc]
                c_1w = map_1w[c_tfc]
                is_pf_tfc = c_tfc in pf_tfc_set
                is_pf_48  = c_48  in pf_48_set
                if is_pf_tfc and is_pf_48:
                    cat = _stable_subtype(fm_tfc, c_tfc, fm_48, c_48)
                elif (not is_pf_tfc) and is_pf_48:
                    cat = "gained"
                elif is_pf_tfc and (not is_pf_48):
                    cat = "lost"
                else:
                    continue  # silent
                cat_neurons[group][cat].append(
                    (mouse, c_tfc, c_48, c_1w,
                     fm_tfc, fm_48, fm_1w, s_tfc, s_48, s_1w))

        # Pretty session labels for column titles
        fam_short = lab48.replace("Test_", "")          # "A" or "B"
        col_titles = [
            f"TFC_cond (CNO)",
            f"Test_{fam_short} +48h",
            f"Test_{fam_short} +1wk",
        ]

        # Pick examples and plot
        rng = np.random.RandomState(0)
        for group in _PV_GROUP_ORDER:
            if group not in cat_neurons:
                continue
            for cat in _GALLERY_CATS:
                pool = cat_neurons[group][cat]
                if not pool:
                    continue
                idxs = rng.choice(len(pool),
                                  size=min(n_examples, len(pool)),
                                  replace=False)
                examples = [pool[i] for i in idxs]

                for ex_i, (mouse, c_tfc, c_48, c_1w,
                           fm_tfc, fm_48, fm_1w,
                           s_tfc_ex, s_48_ex, s_1w_ex) in enumerate(examples):

                    # Ensure A matrices are loaded
                    for sess in (s_tfc_ex, s_48_ex, s_1w_ex):
                        if not hasattr(sess, 'A') or sess.A is None:
                            sess.get_A_matrix()

                    fig, axes = plt.subplots(3, 3, figsize=(9, 8), dpi=300)

                    panels = [
                        (axes[0, 0], fm_tfc, c_tfc, col_titles[0]),
                        (axes[0, 1], fm_48,  c_48,  col_titles[1]),
                        (axes[0, 2], fm_1w,  c_1w,  col_titles[2]),
                    ]

                    # Compute shared colour range across the 3 panels
                    rmaps = []
                    for _, fm, cid, _ in panels:
                        rmaps.append(_get_fm_ratemap(fm, cid))
                    vmin = min(r.min() for r in rmaps)
                    vmax = max(r.max() for r in rmaps)

                    for (ax, fm, cid, col_title), rmap in zip(panels, rmaps):
                        ax.imshow(rmap, cmap="viridis",
                                  interpolation="nearest",
                                  origin="upper", aspect="equal",
                                  vmin=vmin, vmax=vmax)
                        _draw_pf_ellipses(ax, fm, cid)
                        ax.set_title(
                            f"{col_title}\n{mouse} cell {cid}",
                            fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Middle row: sig_responses overlay on ratemap
                    sr_panels = [
                        (axes[1, 0], fm_tfc, c_tfc,
                         f"sig resp {col_titles[0]}\n{mouse} cell {c_tfc}"),
                        (axes[1, 1], fm_48,  c_48,
                         f"sig resp {col_titles[1]}\n{mouse} cell {c_48}"),
                        (axes[1, 2], fm_1w,  c_1w,
                         f"sig resp {col_titles[2]}\n{mouse} cell {c_1w}"),
                    ]
                    for (ax, fm, cid, title), rmap in zip(sr_panels, rmaps):
                        ax.imshow(rmap, cmap="viridis",
                                  interpolation="nearest",
                                  origin="upper", aspect="equal",
                                  vmin=vmin, vmax=vmax)
                        _draw_sig_responses(ax, fm, cid)
                        ax.set_title(title, fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Bottom row: binarized A matrix spatial footprints
                    a_panels = [
                        (axes[2, 0], s_tfc_ex, c_tfc,
                         f"FOV {col_titles[0]}\n{mouse} cell {c_tfc}"),
                        (axes[2, 1], s_48_ex,  c_48,
                         f"FOV {col_titles[1]}\n{mouse} cell {c_48}"),
                        (axes[2, 2], s_1w_ex,  c_1w,
                         f"FOV {col_titles[2]}\n{mouse} cell {c_1w}"),
                    ]
                    for ax, sess, cid, title in a_panels:
                        a_match = np.where(sess.A_idx == cid)[0]
                        if len(a_match):
                            footprint = sess.A[a_match[0]]
                            ax.imshow(footprint, cmap="gray",
                                      interpolation="nearest",
                                      origin="upper", aspect="equal",
                                      vmin=0, vmax=1)
                        else:
                            ax.text(0.5, 0.5, "No A", ha="center",
                                    va="center", transform=ax.transAxes,
                                    fontsize=8, color="red")
                        ax.set_title(title, fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    glabel = _PV_GROUP_LABELS.get(group, group)
                    fig.suptitle(
                        f"{glabel} — {cat.capitalize()} — "
                        f"TFC→{lab48}→{lab1w}",
                        fontsize=9, fontweight="bold", y=0.98)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    fname = (f"VS_{glabel}_{cat}_{ex_i}_{mouse}_"
                             f"{fam_short}.png")
                    fig.savefig(os.path.join(save_dir, fname),
                                dpi=300, bbox_inches="tight")
                    if auto_close:
                        plt.close(fig)
                    else:
                        plt.show()

                    # ---- Gaussian-rendered duplicate figure ----
                    fig_g, axes_g = plt.subplots(3, 3,
                                                 figsize=(9, 8), dpi=300)

                    g_panels = [
                        (axes_g[0, 0], fm_tfc, c_tfc, col_titles[0]),
                        (axes_g[0, 1], fm_48,  c_48,  col_titles[1]),
                        (axes_g[0, 2], fm_1w,  c_1w,  col_titles[2]),
                    ]
                    gmaps = []
                    for _, fm, cid, _ in g_panels:
                        gmaps.append(_render_gaussian_pf(fm, cid))

                    for (ax, fm, cid, col_title), gmap in zip(g_panels,
                                                               gmaps):
                        ax.imshow(gmap, cmap="viridis",
                                  interpolation="bilinear",
                                  origin="upper", aspect="equal",
                                  vmin=0, vmax=1)
                        ax.set_title(
                            f"{col_title}\n{mouse} cell {cid}",
                            fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Middle row: sig_responses overlay on ratemap
                    gsr_panels = [
                        (axes_g[1, 0], fm_tfc, c_tfc,
                         f"sig resp {col_titles[0]}\n{mouse} cell {c_tfc}"),
                        (axes_g[1, 1], fm_48,  c_48,
                         f"sig resp {col_titles[1]}\n{mouse} cell {c_48}"),
                        (axes_g[1, 2], fm_1w,  c_1w,
                         f"sig resp {col_titles[2]}\n{mouse} cell {c_1w}"),
                    ]
                    for (ax, fm, cid, title), rmap in zip(gsr_panels, rmaps):
                        ax.imshow(rmap, cmap="viridis",
                                  interpolation="nearest",
                                  origin="upper", aspect="equal",
                                  vmin=vmin, vmax=vmax)
                        _draw_sig_responses(ax, fm, cid)
                        ax.set_title(title, fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Bottom row: A matrix footprints (same as original)
                    ga_panels = [
                        (axes_g[2, 0], s_tfc_ex, c_tfc,
                         f"FOV {col_titles[0]}\n{mouse} cell {c_tfc}"),
                        (axes_g[2, 1], s_48_ex,  c_48,
                         f"FOV {col_titles[1]}\n{mouse} cell {c_48}"),
                        (axes_g[2, 2], s_1w_ex,  c_1w,
                         f"FOV {col_titles[2]}\n{mouse} cell {c_1w}"),
                    ]
                    for ax, sess, cid, title in ga_panels:
                        a_match = np.where(sess.A_idx == cid)[0]
                        if len(a_match):
                            footprint = sess.A[a_match[0]]
                            ax.imshow(footprint, cmap="gray",
                                      interpolation="nearest",
                                      origin="upper", aspect="equal",
                                      vmin=0, vmax=1)
                        else:
                            ax.text(0.5, 0.5, "No A", ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                    fontsize=8, color="red")
                        ax.set_title(title, fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    fig_g.suptitle(
                        f"{glabel} — {cat.capitalize()} (Gaussian) — "
                        f"TFC→{lab48}→{lab1w}",
                        fontsize=9, fontweight="bold", y=0.98)
                    fig_g.tight_layout(rect=[0, 0, 1, 0.95])
                    fname_g = (f"VS_gauss_{glabel}_{cat}_{ex_i}_"
                               f"{mouse}_{fam_short}.png")
                    fig_g.savefig(os.path.join(save_dir, fname_g),
                                 dpi=300, bbox_inches="tight")
                    if auto_close:
                        plt.close(fig_g)
                    else:
                        plt.show()

    print(f"  [Gallery VS] Saved 3-session example plots → {save_dir}")


# ===========================================================================
#  Cross-registration sanity check — random neurons, 3-panel plots + log
# ===========================================================================

def crossreg_sanity_check(
    session_dicts, mouse_groups, mappings, PLOTS_DIR,
    n_per_group=10, auto_close=True,
):
    """
    For each group (Ctl / Exc / Inh), randomly sample *n_per_group*
    cross-registered neurons from the TFC_cond ↔ Test_B ↔ Test_B_1wk
    mapping and:
      1. Write detailed crossreg verification text to a log file.
      2. Save a 3-panel rate-map plot (TFC → Test_B → Test_B_1wk) with
         GMM ellipses for each sampled neuron.

    All outputs go to  <PLOTS_DIR>/pf_crossreg_sanity_checks/
    """
    from matplotlib.patches import Ellipse
    from numpy import linalg

    save_dir = os.path.join(PLOTS_DIR, "pf_crossreg_sanity_checks")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "crossreg_sanity_log.txt")

    tfc_label = "TFC_cond"
    tfc_dict = session_dicts.get(tfc_label, {})
    if not tfc_dict:
        print("[sanity-check] No TFC_cond sessions found.")
        return

    # We need Test_B and Test_B_1wk
    tb_dict  = session_dicts.get("Test_B", {})
    tb1_dict = session_dicts.get("Test_B_1wk", {})
    if not tb_dict or not tb1_dict:
        print("[sanity-check] Need both Test_B and Test_B_1wk in session_dicts.")
        return

    mapping_str = mappings.get("Test_B", "TFC_cond+Test_B+Test_B_1wk")

    # ---- helpers (local) ----
    def _draw_ellipses(ax, fm, cell_id):
        _pos = _pf_pos(fm, cell_id)
        has_pf = (_pos in getattr(fm.pf, 'model_', {}) and
                  _pos in getattr(fm.pf, 'merged_means', {}))
        if not has_pf:
            ax.text(0.5, 0.5, "No PF", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="white")
            return
        merged = fm.pf.merged_means[_pos]
        pf_colors = plt.cm.Set2(np.linspace(0, 1, max(len(merged), 1)))
        m_means = getattr(fm.pf, 'merged_means_', {}).get(_pos)
        m_covs  = getattr(fm.pf, 'merged_covariances_', {}).get(_pos)
        for pf_idx in range(len(merged)):
            col = pf_colors[pf_idx % len(pf_colors)]
            if m_means and m_covs:
                mu = m_means[pf_idx]
                cov_merged = m_covs[pf_idx]
            else:
                model = fm.pf.model_[_pos]
                comp_idxs = merged[pf_idx]
                weights = model.weights_[comp_idxs]
                weights = weights / weights.sum()
                mu = weights @ model.means_[comp_idxs]
                cov_merged = np.zeros((2, 2))
                for ci, wi in zip(comp_idxs, weights):
                    diff = model.means_[ci] - mu
                    cov_merged += wi * (
                        model.covariances_[ci] + np.outer(diff, diff))
            eigvals, eigvecs = linalg.eigh(cov_merged)
            widths = 2.0 * 2.0 * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ell = Ellipse(xy=(mu[1], mu[0]),
                          width=widths[1], height=widths[0],
                          angle=angle, edgecolor=col, facecolor="none",
                          linewidth=1.5, linestyle="-")
            ax.add_patch(ell)
            ax.plot(mu[1], mu[0], "+",
                    color=col, markersize=8, mew=1.5)

    # ---- collect all cross-registered triplets per group ----
    mice = sorted(set(tfc_dict.keys()) & set(tb_dict.keys()) & set(tb1_dict.keys()))

    # { group: [ (mouse, c_tfc, c_tb, c_tb1, sess_tfc, sess_tb, sess_tb1), ... ] }
    group_pool = {g: [] for g in _PV_GROUP_ORDER}

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Cross-registration sanity check\n")
        log.write(f"{'='*60}\n")
        log.write(f"Mapping string: {mapping_str}\n")
        log.write(f"Mice checked: {mice}\n\n")

        for mouse in mice:
            s_tfc = tfc_dict[mouse]
            s_tb  = tb_dict[mouse]
            s_tb1 = tb1_dict[mouse]
            group = mouse_groups.get(mouse, "NA")
            if group not in group_pool:
                continue

            # Resolve crossreg
            candidates = []
            seen = set()
            for attr in ('crossreg', 'crossreg_full'):
                for sess in (s_tfc, s_tb, s_tb1):
                    c = getattr(sess, attr, None)
                    if c is not None and id(c) not in seen:
                        seen.add(id(c))
                        candidates.append((attr, sess.session_type, c))

            crossreg = None
            source_attr = source_sess = None
            df = None
            for attr, stype, cand in candidates:
                try:
                    df = cand.get_mappings_cells(mapping_type=mapping_str)
                    crossreg = cand
                    source_attr, source_sess = attr, stype
                    break
                except Exception:
                    continue

            if crossreg is None:
                log.write(f"[{mouse}] ({group}) — NO crossreg found, skipping\n\n")
                continue

            col_tfc = s_tfc.get_df_col(with_crossreg=crossreg)
            col_tb  = s_tb.get_df_col(with_crossreg=crossreg)
            col_tb1 = s_tb1.get_df_col(with_crossreg=crossreg)

            df_valid = df[[col_tfc, col_tb, col_tb1]].dropna()
            for c in [col_tfc, col_tb, col_tb1]:
                df_valid[c] = df_valid[c].astype(float).astype(int)

            log.write(f"[{mouse}] group={group}\n")
            log.write(f"  crossreg source: {source_sess}.{source_attr} "
                      f"(type={crossreg.crossreg_type})\n")
            log.write(f"  groups (timestamps): {crossreg.groups}\n")
            log.write(f"  labels: {crossreg.mappings_labels}\n")
            log.write(f"  columns: TFC='{col_tfc}', "
                      f"Test_B='{col_tb}', Test_B_1wk='{col_tb1}'\n")
            log.write(f"  3-way registered cells: {len(df_valid)}\n\n")

            for _, row in df_valid.iterrows():
                group_pool[group].append((
                    mouse,
                    int(row[col_tfc]), int(row[col_tb]), int(row[col_tb1]),
                    s_tfc, s_tb, s_tb1,
                ))

        # ---- sample and plot ----
        rng = np.random.RandomState(42)
        total_plotted = 0

        for group in _PV_GROUP_ORDER:
            pool = group_pool[group]
            glabel = _PV_GROUP_LABELS.get(group, group)
            if not pool:
                log.write(f"\n--- {glabel} ({group}): no neurons available ---\n")
                continue

            n_pick = min(n_per_group, len(pool))
            idxs = rng.choice(len(pool), size=n_pick, replace=False)
            samples = [pool[i] for i in sorted(idxs)]

            log.write(f"\n{'='*60}\n")
            log.write(f"{glabel} ({group}): sampled {n_pick}/{len(pool)} neurons\n")
            log.write(f"{'='*60}\n")

            for si, (mouse, c_tfc, c_tb, c_tb1,
                     s_tfc, s_tb, s_tb1) in enumerate(samples):

                log.write(f"  [{si+1}] {mouse}: "
                          f"TFC c{c_tfc} → Test_B c{c_tb} → "
                          f"Test_B_1wk c{c_tb1}\n")

                fm_tfc = s_tfc.fm
                fm_tb  = s_tb.fm
                fm_tb1 = s_tb1.fm

                # Ensure A matrices are loaded
                for sess in (s_tfc, s_tb, s_tb1):
                    if not hasattr(sess, 'A') or sess.A is None:
                        sess.get_A_matrix()

                fig, axes = plt.subplots(3, 3, figsize=(9, 8), dpi=300)

                panels = [
                    (axes[0, 0], fm_tfc, c_tfc, f"TFC (c{c_tfc})"),
                    (axes[0, 1], fm_tb,  c_tb,  f"Test_B (c{c_tb})"),
                    (axes[0, 2], fm_tb1, c_tb1, f"Test_B_1wk (c{c_tb1})"),
                ]

                # Shared colour range
                rmaps = [_get_fm_ratemap(fm, cid) for _, fm, cid, _ in panels]
                vmin = min(r.min() for r in rmaps)
                vmax = max(r.max() for r in rmaps)

                for (ax, fm, cid, title), rmap in zip(panels, rmaps):
                    ax.imshow(rmap, cmap="viridis", interpolation="nearest",
                              origin="upper", aspect="equal",
                              vmin=vmin, vmax=vmax)
                    _draw_ellipses(ax, fm, cid)
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Middle row: sig_responses overlay on ratemap
                sr_panels = [
                    (axes[1, 0], fm_tfc, c_tfc,
                     f"sig resp TFC (c{c_tfc})"),
                    (axes[1, 1], fm_tb,  c_tb,
                     f"sig resp Test_B (c{c_tb})"),
                    (axes[1, 2], fm_tb1, c_tb1,
                     f"sig resp Test_B_1wk (c{c_tb1})"),
                ]
                for (ax, fm, cid, title), rmap in zip(sr_panels, rmaps):
                    ax.imshow(rmap, cmap="viridis",
                              interpolation="nearest",
                              origin="upper", aspect="equal",
                              vmin=vmin, vmax=vmax)
                    _draw_sig_responses(ax, fm, cid)
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Bottom row: binarized A matrix spatial footprints
                a_panels = [
                    (axes[2, 0], s_tfc, c_tfc, f"FOV TFC (c{c_tfc})"),
                    (axes[2, 1], s_tb,  c_tb,  f"FOV Test_B (c{c_tb})"),
                    (axes[2, 2], s_tb1, c_tb1, f"FOV Test_B_1wk (c{c_tb1})"),
                ]
                for ax, sess, cid, title in a_panels:
                    a_match = np.where(sess.A_idx == cid)[0]
                    if len(a_match):
                        footprint = sess.A[a_match[0]]
                        ax.imshow(footprint, cmap="gray",
                                  interpolation="nearest",
                                  origin="upper", aspect="equal",
                                  vmin=0, vmax=1)
                    else:
                        ax.text(0.5, 0.5, "No A", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8,
                                color="red")
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                fig.suptitle(
                    f"{glabel} — {mouse} — "
                    f"TFC c{c_tfc} → Test_B c{c_tb} → Test_B_1wk c{c_tb1}",
                    fontsize=9, fontweight="bold", y=0.98)
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                fname = f"sanity_{glabel}_{si:02d}_{mouse}_c{c_tfc}.png"
                fig.savefig(os.path.join(save_dir, fname),
                            dpi=300, bbox_inches="tight")
                if auto_close:
                    plt.close(fig)
                else:
                    plt.show()

                # ---- Gaussian-rendered duplicate figure ----
                fig_g, axes_g = plt.subplots(3, 3,
                                             figsize=(9, 8), dpi=300)

                g_panels = [
                    (axes_g[0, 0], fm_tfc, c_tfc, f"TFC Gauss (c{c_tfc})"),
                    (axes_g[0, 1], fm_tb,  c_tb,  f"Test_B Gauss (c{c_tb})"),
                    (axes_g[0, 2], fm_tb1, c_tb1,
                     f"Test_B_1wk Gauss (c{c_tb1})"),
                ]
                for ax, fm, cid, title in g_panels:
                    gmap = _render_gaussian_pf(fm, cid)
                    ax.imshow(gmap, cmap="viridis",
                              interpolation="bilinear",
                              origin="upper", aspect="equal",
                              vmin=0, vmax=1)
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Middle row: sig_responses
                gsr_panels = [
                    (axes_g[1, 0], fm_tfc, c_tfc,
                     f"sig resp TFC (c{c_tfc})"),
                    (axes_g[1, 1], fm_tb,  c_tb,
                     f"sig resp Test_B (c{c_tb})"),
                    (axes_g[1, 2], fm_tb1, c_tb1,
                     f"sig resp Test_B_1wk (c{c_tb1})"),
                ]
                for (ax, fm, cid, title), rmap in zip(gsr_panels, rmaps):
                    ax.imshow(rmap, cmap="viridis",
                              interpolation="nearest",
                              origin="upper", aspect="equal",
                              vmin=vmin, vmax=vmax)
                    _draw_sig_responses(ax, fm, cid)
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Bottom row: A matrix footprints
                ga_panels = [
                    (axes_g[2, 0], s_tfc, c_tfc,
                     f"FOV TFC (c{c_tfc})"),
                    (axes_g[2, 1], s_tb,  c_tb,
                     f"FOV Test_B (c{c_tb})"),
                    (axes_g[2, 2], s_tb1, c_tb1,
                     f"FOV Test_B_1wk (c{c_tb1})"),
                ]
                for ax, sess, cid, title in ga_panels:
                    a_match = np.where(sess.A_idx == cid)[0]
                    if len(a_match):
                        footprint = sess.A[a_match[0]]
                        ax.imshow(footprint, cmap="gray",
                                  interpolation="nearest",
                                  origin="upper", aspect="equal",
                                  vmin=0, vmax=1)
                    else:
                        ax.text(0.5, 0.5, "No A", ha="center",
                                va="center", transform=ax.transAxes,
                                fontsize=8, color="red")
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                fig_g.suptitle(
                    f"{glabel} (Gaussian) — {mouse} — "
                    f"TFC c{c_tfc} → Test_B c{c_tb} → "
                    f"Test_B_1wk c{c_tb1}",
                    fontsize=9, fontweight="bold", y=0.98)
                fig_g.tight_layout(rect=[0, 0, 1, 0.95])

                fname_g = (f"sanity_gauss_{glabel}_{si:02d}_"
                           f"{mouse}_c{c_tfc}.png")
                fig_g.savefig(os.path.join(save_dir, fname_g),
                              dpi=300, bbox_inches="tight")
                if auto_close:
                    plt.close(fig_g)
                else:
                    plt.show()

                total_plotted += 1

        log.write(f"\n{'='*60}\n")
        log.write(f"Total neurons plotted: {total_plotted}\n")

    print(f"  [Sanity Check] {total_plotted} plots + log → {save_dir}")


print("loaded")
from random import shuffle
import numpy as np
from caban.utilities import *
from scipy.ndimage import gaussian_filter
from numpy.random import default_rng
import time
from sklearn import mixture
from sklearn.cluster import KMeans
from numpy import linalg
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

'''
Process mouse location and establish bins.
'''
class Location_XY:
    def __init__(self, mouse, group, sess, bin_width, DEBUG=True):
        self.MAX_X = 613.9776984686001
        self.MAX_Y = 467.7186760984302
        self.MIN_X = 27.142475257551364
        self.MIN_Y = 1.2439073151409752
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

        fluorescence_map_occup = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1, self.num_cells)) # so if max is 20, it's the 21st entry into its dimension (0 being the 1st)
        if want_occup_map:
            occup_map = np.zeros((self.loc.num_bins_y+1, self.loc.num_bins_x+1))

        S_gauss = gaussian_filter(S.flatten(),sigma=SMOOTH_LOC_SIGMA).reshape(S.shape)
        for i in range(S.shape[1]):
            fluorescence_map_occup[self.loc.binned_Y[i], self.loc.binned_X[i], :] += S_gauss[self.cells,i]

            if want_occup_map:
                occup_map[self.loc.binned_Y[i], self.loc.binned_X[i]] += 1
            
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
                np.savez(self.save_pickle_path, cells=self.cells, shuffled_responses=self.shuffled_responses)

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
    def find_place_fields(self, sess, method='iterative_gauss', n_comp_start=10):
    
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
                merge_distance = 4
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

print("loaded")
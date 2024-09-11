from tkinter.filedialog import SaveFileDialog
import numpy as np
from caban.utilities import *
import ast
import glob
from natsort import natsorted
import pickle
from scipy.ndimage import gaussian_filter

####################

class CrossRegMapping:
    def __init__(self, mouse, dpath_mappings, crossreg_type=1, groups_mappings=None, savepath=''):
        self.mouse = mouse
        self.groups_mappings = groups_mappings
        self.session_list = ['session', 'session.1', 'session.2']
        self.crossreg_type=crossreg_type
        self.session_suffixes = {
            0 : '',
            1 : '.1',
            2 : '.2'
        }

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

        # Create dict of different mapping types to rows of the df
        self.mappings = dict()
        groups_set = set()
        for i in range(1, len(df['group'])):
            current_mapping = df['group'][i]

            # Get all unique group names
            group_list = ast.literal_eval(current_mapping)
            for g in group_list:
                groups_set.add(g)

            if not current_mapping in self.mappings:
                self.mappings[current_mapping] = []
            self.mappings[current_mapping].append(i)
        
        # Convert set of groups to list & sort, which will result in chronological ordering
        self.groups = list(groups_set)
        self.groups.sort()

        if crossreg_type == 1:
            # LT1, LT2, TFC_cond in chronological order.
            # This dict provides a convenient way to convert desired mappings provided as labels
            # to group strings to index into the main mappings DataFrame.
            assert len(self.groups) == 3 # should have 3 unique strings for each of LT1, LT2, TFC_cond
            self.mappings_labels = dict()
            self.mappings_labels['LT1'] = self.groups[0]
            self.mappings_labels['LT2'] = self.groups[1]
            self.mappings_labels['TFC_cond'] = self.groups[2]

        # TFC_cond/TFC_cond; Test_B/Test_B; Test_B_1wk/Test_B_1wk
        if crossreg_type == 4: 
            self.mappings_labels = dict()
            if mouse == 'G15':
                labels = ['TFC_cond', 'Test_B']
            elif mouse == 'G07':
                labels = ['TFC_cond', 'Test_B_1wk']
            else:
                labels = ['TFC_cond', 'Test_B', 'Test_B_1wk']

            for label in labels:
                self.mappings_labels[label] = self.groups_mappings[label]

        return

    def get_mappings_for_groups(self, mapping_type):
        groups = mapping_type.split('+')
        groups.sort() # Then can just append to groups_list
        groups_list = []

        for label in groups:
            group_str = self.mappings_labels[label]
            group_idx = self.groups.index(group_str)
            groups_list.append(self.groups[group_idx])

        groups_list.sort() # Need this in the end, for non-TFC_cond where sessions are across days so not chronological order

        # Finally, convert the desired list of groups into a tuple (for round brackets), then a string, to act as a key
        # into self.mappings (caller can do the latter).
        return str(tuple(groups_list))        

    def get_mappings_cells(self, mapping_type='LT1+LT2+TFC_cond', mappings_cell=-1, mappings_row=-1, return_df_rows=False):
        """
        Return a subset of the class instance's mappings DataFrame object corresponding to a desired row in that mapping.
        The mapping_type must be a string of the format 'Session1+Session2+...', and the desired combination
        must already exist in the class instance's dpath_mappings csv file (output of minian cross registration 
        module).

        mappings_row is relative to all the mappings of the desired type, in the order extracted into 
        self.mappings by the class constructor.

        The default mapping type is arbitarily set to the LT1+LT2+TFC_cond which is the one most useful for
        the intended dataset, but can be set otherwise ('LT1+LT2', 'LT1', etc).

        If all cells under a desired mapping are wanted, mappings_cell and mappings_row should remain unassigned.
        In this case, the subset of the instance's mappings_df DataFrame will be returned corresponding to the mapping_type.
        """
        df = self.mappings_df
        cells = dict()
        #if mapping_type == 'full':
        #    return self.mappings_df.iloc[:]

        mapping_str = self.get_mappings_for_groups(mapping_type)
        if mappings_cell > -1:
            pass # WIP
        elif mappings_row > -1:

            mapping = self.mappings[mapping_str]
            df_row = mapping[mappings_row]
            return self.mappings_df.iloc[df_row]
        else:

            # Return all cells under the mapping
            mapping = self.mappings[mapping_str]
            return self.mappings_df.iloc[mapping]
        return cells

class BehaviourSession:
    def __init__(self, mouse, dpath, session_bounds=[], plot_sample_cell=False, data_dir='', session_group='session', crossreg='', savepath='', session_type='Behaviour',
        behaviour_type=None, saver_prefix=''):

        self.mouse = mouse
        self.dpath = dpath
        self.start_idx = 0
        self.stop_idx = 1
        self.session_bounds = session_bounds # frame numbers [start, stop] in behav cam
        self.plot_sample_cell = plot_sample_cell
        self.data_dir = data_dir
        self.session_group = session_group
        self.thres = 2
        self.crossreg = crossreg
        self.session_type = session_type
        self.session_str = session_type.replace('_',' ')
        self.savepath = savepath
        self.behaviour_type = behaviour_type
        self.minian_output_dir = ''

        self.fnum_miniscope = []
        self.tstamp_miniscope = []
        self.fnum_behavcam = []
        self.tstamp_behavcam = []

        self.behavcam_exp_ts = []
        self.miniscope_exp_fnum = []
        self.miniscope_exp_ts = []

        self.tstamp_tol = 70 # ms

        self.saver_timestamps = Saver(parent_path=savepath, subdirs=['timestamps'], prefix='{}_{}_timestamps'.format(saver_prefix, mouse))
        self.saver_CS_matrices = Saver(parent_path=savepath, subdirs=['CS_matrices'], prefix='{}_{}_CS_matrices'.format(saver_prefix, mouse))
        self.saver_loc_data = Saver(parent_path=savepath, subdirs=['loc_data'], prefix='{}_{}_loc_data'.format(saver_prefix, mouse))
        self.saver_Y = Saver(parent_path=savepath, subdirs=['Y'], prefix='{}_{}_Y'.format(saver_prefix, mouse))

        self.find_exp_boundaries()
        self.get_location_data() # Need to do this first as this can be used for masking spike data in get_CS_matrices()
        self.get_CS_matrices()

    def set_minian_output_dir(self):
        if not self.minian_output_dir:
            if self.data_dir:
                self.minian_output_dir = os.path.join(self.dpath, 'Miniscope', self.data_dir)
            else:
                glob_results = glob.glob(os.path.join(self.dpath, 'Miniscope', 'minian_crossreg*'))
                #assert len(glob_results) == 1 # Should be only 1 directory!
                if glob_results:
                    self.minian_output_dir = glob_results[0]
                else:
                    self.minian_output_dir = ''

    def get_timestamps(self):
        frame_col = 0
        tstamp_col = 1
        dpath = self.dpath 

        # Get timestamps for Miniscope cam
        if self.saver_timestamps.check_exists('Miniscope_df'):
            df = self.saver_timestamps.load('Miniscope_df')
        else:
            ts_file_miniscope = os.path.join(dpath, 'Miniscope', 'timeStamps.csv')
            with open(ts_file_miniscope) as ts_file:
                df = pd.read_csv(ts_file)
            self.saver_timestamps.save(df, 'Miniscope_df')

        fnum_miniscope = df.iloc[:,frame_col]
        tstamp_miniscope = df.iloc[:,tstamp_col]

        # Get timestampes for BehavCam
        if self.saver_timestamps.check_exists('BehavCam_df'):
            df = self.saver_timestamps.load('BehavCam_df')
        else:
            ts_file_behavcam = os.path.join(dpath, 'BehavCam', 'timeStamps.csv')
            with open(ts_file_behavcam) as ts_file:
                df = pd.read_csv(ts_file)
            self.saver_timestamps.save(df, 'BehavCam_df')

        fnum_behavcam = df.iloc[:,frame_col]
        tstamp_behavcam = df.iloc[:,tstamp_col]

        return [fnum_miniscope, tstamp_miniscope, fnum_behavcam, tstamp_behavcam]

    def find_exp_boundaries(self):

        [self.fnum_miniscope, self.tstamp_miniscope, self.fnum_behavcam, self.tstamp_behavcam] \
            = self.get_timestamps()

        # Logic: 
        # Step 1. Find frames in actual videofiles where the session starts and ends to mark the boundaries
        #     of the experiment. These go in light_frames (required; provided in constructor).

        # Step 2. Find the corresponding timestamps of these frames in the **BehavCam** timestamps file.
        self.behavcam_exp_ts = [self.tstamp_behavcam.iloc[self.session_bounds[self.start_idx]], \
            self.tstamp_behavcam.iloc[self.session_bounds[self.stop_idx]]]

        # Step 3. Find the closest corresponding timestamps of these frames in the **Miniscope** timestamps file.
        # These then become directly the frame numbers.
        beg = np.where(abs(self.tstamp_miniscope - self.behavcam_exp_ts[self.start_idx]) < self.tstamp_tol)
        end = np.where(abs(self.tstamp_miniscope - self.behavcam_exp_ts[self.stop_idx]) < self.tstamp_tol)
        # there may be more than one in beg, end, so we just take the first, which is also the closest
        self.miniscope_exp_fnum = [beg[0][0], end[0][0]] 
        self.miniscope_exp_ts = [self.tstamp_miniscope[self.miniscope_exp_fnum[self.start_idx]], \
            self.tstamp_miniscope[self.miniscope_exp_fnum[self.stop_idx]]]

        # Assign the experiment-subsetted miniscope and behavcam tstamps so that we never have to worry
        # about this further on. The originals get clobbered. NB: https://en.wikipedia.org/wiki/Clobbering
        # 
        # Honestly can probably be much more simply handled in get_timestamps()...
        self.tstamp_miniscope = \
            self.tstamp_miniscope[self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]] \
                - self.tstamp_miniscope[self.miniscope_exp_fnum[self.start_idx]]
        self.tstamp_miniscope = self.tstamp_miniscope.to_numpy() # Series is too annoying to deal with, sorry

        self.fnum_miniscope = \
            self.fnum_miniscope[self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]] \
                - self.fnum_miniscope[self.miniscope_exp_fnum[self.start_idx]]
        self.fnum_miniscope = self.fnum_miniscope.to_numpy()

        self.tstamp_behavcam = \
            self.tstamp_behavcam[self.session_bounds[self.start_idx]:self.session_bounds[self.stop_idx]] \
                - self.tstamp_behavcam[self.session_bounds[self.start_idx]]
        self.tstamp_behavcam = self.tstamp_behavcam.to_numpy()

        self.fnum_behavcam = \
            self.fnum_behavcam[self.session_bounds[self.start_idx]:self.session_bounds[self.stop_idx]] \
                - self.fnum_behavcam[self.session_bounds[self.start_idx]]
        self.fnum_behavcam = self.fnum_behavcam.to_numpy()

    def get_CS_matrices(self):

        # zarr.load returns zarr LazyLoader dict, so get the numpy values for the calcium and spike arrays
        if self.saver_CS_matrices.check_exists('C_full') and self.saver_CS_matrices.check_exists('C_idx'):
            self.C_full = self.saver_CS_matrices.load('C_full')
            self.C_idx = self.saver_CS_matrices.load('C_idx')
        else:
            self.set_minian_output_dir()
            self.C_zarr = zarr.load(os.path.join(self.minian_output_dir, "C.zarr"))
            self.C_full = self.C_zarr['C']
            self.C_idx = self.C_zarr['unit_id']
            self.saver_CS_matrices.save(self.C_full, 'C_full')
            self.saver_CS_matrices.save(self.C_idx, 'C_idx')

        if self.saver_CS_matrices.check_exists('S_full') and self.saver_CS_matrices.check_exists('S_idx'):
            self.S_full = self.saver_CS_matrices.load('S_full')
            self.S_idx = self.saver_CS_matrices.load('S_idx')
        else:
            self.set_minian_output_dir()
            self.S_zarr = zarr.load(os.path.join(self.minian_output_dir, "S.zarr"))
            self.S_full = self.S_zarr['S']
            self.S_idx = self.S_zarr['unit_id']
            self.saver_CS_matrices.save(self.S_full, 'S_full')
            self.saver_CS_matrices.save(self.S_idx, 'S_idx')

        # Get subset of S, C corresponding to the experiment bounds of the session.
        self.S = self.S_full[:,self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]]
        self.C = self.C_full[:,self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]]
        #self.S = self.S_full
        #self.C = self.C_full

        # Get original motion-corrected fluorescence data, if available.
        if self.saver_Y.check_exists('Y'):
            self.Y = self.saver_Y.load('Y')
        else:
            self.set_minian_output_dir()
            Y_path = os.path.join(self.minian_output_dir, "YrA.zarr")
            if os.path.exists(Y_path):
                self.Y_zarr = zarr.load(Y_path)
                self.Y = self.Y_zarr['YrA']
                self.saver_Y.save(self.Y, 'Y')

        # Get movement-filtered spiking activity, if desired.
        self.savefile_S_mov = os.path.join(self.savepath, self.session_type+'-S_mov-'+self.mouse+'.npy')
        self.savefile_S_imm = os.path.join(self.savepath, self.session_type+'-S_imm-'+self.mouse+'.npy')
        self.savefile_C_mov = os.path.join(self.savepath, self.session_type+'-C_mov-'+self.mouse+'.npy')
        self.savefile_C_imm = os.path.join(self.savepath, self.session_type+'-C_imm-'+self.mouse+'.npy')

        if self.behaviour_type == 'movement':
            #self.vel_mask = (self.velocities_miniscope >= 2.0).astype(int)
            self.vel_mask = (self.velocities_miniscope_smooth >= 2.0).astype(int)
            if os.path.exists(self.savefile_S_mov) and os.path.exists(self.savefile_S_imm):
                ###self.S_mov = pickle.load(open(self.savefile_S_mov, 'rb')) # old pickle way
                ###self.S_imm = pickle.load(open(self.savefile_S_imm, 'rb')) # old pickle way 

                # don't save anymore, too big and not enough speed gains
                #self.S_mov = np.load(self.savefile_S_mov, allow_pickle=True).item() 
                #self.S_imm = np.load(self.savefile_S_imm, allow_pickle=True).item()
                pass
            else:
                # In case there was mismatch in video processing due to errors in last .avi, just use subset that exists
                # in velocities_miniscope
                self.S_mov = np.multiply(self.S, (self.velocities_miniscope_smooth[:self.S.shape[1]] >= 2.0).astype(int))
                self.S_imm = np.multiply(self.S, (self.velocities_miniscope_smooth[:self.S.shape[1]] < 2.0).astype(int))

                # don't save anymore, too big and not enough speed gains
                ###with open(self.savefile_S_mov, 'wb') as f: pickle.dump(self.savefile_S_mov, f) # old pickle way
                ###with open(self.savefile_S_imm, 'wb') as f: pickle.dump(self.savefile_S_imm, f) # old pickle way
                #np.save(self.savefile_S_mov, self.S_mov)
                #np.save(self.savefile_S_imm, self.S_imm)

            # For symmetry with S
            if os.path.exists(self.savefile_C_mov) and os.path.exists(self.savefile_C_imm):
                pass
            else:
                self.C_mov = np.multiply(self.C, (self.velocities_miniscope_smooth[:self.C.shape[1]] >= 2.0).astype(int))
                self.C_imm = np.multiply(self.C, (self.velocities_miniscope_smooth[:self.C.shape[1]] < 2.0).astype(int))

            # CRITICAL: design decision is to just make S, C be the 'movement-filtered' spike ndarrays. Then, no other
            # code has to change. However, for immobility-specific code, it will have to know to check self.S_imm, etc. 
            #
            # 2023.11.7. No, we are changing this. S, C will be the "orig" and S_mov, C_mov will be movement-only.
            # Only spatial analysis needs to deal with movement (place fields) so it's too awkward to have everything else 
            # always use *_orig.
            #self.S_orig = self.S
            #self.S = np.copy(self.S_mov)
            #self.C_orig = self.C 
            #self.C = np.copy(self.C_mov)

        # Find spike times of all cells in this session, to reduce duplicated work later by analyses.
        self.savefile_S_spikes = os.path.join(self.savepath, self.session_type+'-S_spikes-'+self.mouse+'.npy')
        self.savefile_S_peakval = os.path.join(self.savepath, self.session_type+'-S_peakval-'+self.mouse+'.npy')
        if os.path.exists(self.savefile_S_spikes) and os.path.exists(self.savefile_S_peakval):
            #self.S_spikes = pickle.load(open(self.savefile_S_spikes, 'rb'))
            #self.S_peakval = pickle.load(open(self.savefile_S_peakval, 'rb'))
            self.S_spikes = np.load(self.savefile_S_spikes, allow_pickle=True).item()
            self.S_peakval = np.load(self.savefile_S_peakval, allow_pickle=True).item()
            #print('*** {} {} loaded S_spikes from {}'.format(self.mouse, self.session_type, self.savefile))

        else:
            [self.S_spikes, self.S_peakval] = find_spikes_ca_S(self.S, self.thres, want_peakval=True)
            if self.behaviour_type == 'movement':
                [self.S_spikes_mov, self.S_peakval_mov] = find_spikes_ca_S(self.S_mov, self.thres, want_peakval=True)
                [self.S_spikes_imm, self.S_peakval_imm] = find_spikes_ca_S(self.S_imm, self.thres, want_peakval=True)
            print('*** processed spikes {} {}'.format(self.mouse, self.session_type))
            ##with open(self.savefile_S_spikes, 'wb') as f: pickle.dump(self.savefile_S_spikes, f)
            ##with open(self.savefile_S_peakval, 'wb') as f: pickle.dump(self.savefile_S_peakval, f)
            #np.save(self.savefile_S_spikes, self.S_spikes)
            #np.save(self.savefile_S_peakval, self.S_peakval)
            ##print('*** {} {} saved S_spikes into {}'.format(self.mouse, self.session_type, self.savefile))

    def get_A_matrix(self):
        self.A_zarr = zarr.load(os.path.join(self.minian_output_dir, "A.zarr"))
        self.A = self.A_zarr['A']
        self.A_idx = self.A_zarr['unit_id']

    def get_S_indeces(self, unit_id):
        '''
        Since cell unit_id's determined by minian are not exactly 1-to-1 in terms of indeces into the S_zarr,
        but rather require to use the mapping between the 'unit_id' column in S_zarr, we perform that function
        here so the caller doesn't have to worry about this complication.

        Returns a list of indices that can be directly subscripted into either self.S or self.S_spikes.

        *** IMPORTANT*** the list is sorted according to the unit_id list, so you can do cross-registration 
        and get the values of each corresponding S from each cross-registered session so long as you pass the 
        same unit_id array (e.g., S_idx returned from get_actual_cells_from_df_session()).
        '''
        # Old way, returns sorted which is BAD for crossreg mapping purposes
        #indeces=np.where(np.isin(self.S_idx, unit_id))[0]
        #return indeces
        indeces=[]
        for i in unit_id: # really ugly
            indeces.append(np.where(np.isin(self.S_idx, i))[0][0]) 
        return indeces

        '''# debug code
for l in unit_id:
    x=np.where(self.S_zarr['unit_id'] == l)[0]
    if not x:
        print('not found: '+str(l))
    else:
        print(str(l)+' found in S_zarr[unit_id] at '+str(x))
        '''

    def get_df_col(self, with_crossreg=None):
        '''
        Returns as string of the form, ``session'', ``session.1'', ``session.2''.
        '''
        if with_crossreg:
            crossreg = with_crossreg
        else:
            crossreg = self.crossreg

        idx = crossreg.mappings_labels[self.session_type]
        session_suffix = crossreg.session_suffixes[crossreg.groups.index(idx)]
        return 'session'+session_suffix 
    
        ''' # Old
        if self.session_type == 'Test_B' or self.session_type == 'Test_B_1wk':
            df_col = self.crossreg.session_list[self.crossreg.groups.index(self.crossreg.groups_mappings[self.session_type])]
        else:
            df_col = self.session_group
        return df_col
        '''

    def get_S_mapping(self, mapping, with_crossreg=None, with_peakval=False):
        '''
        Return S as an ndarray corresponding to the mapping provided by self's crossreg mapping, along with a
        desired session, i.e., mapping must be a string of the type 'LT1+LT2+TFC_cond', etc.

        Returns:
          S - ndarray of shape (cell indeces, frames of experiment)
          S_spikes - dict of cell indeces : spike frames
          S_idx - list of cell ID's found in mapping -> indeces into S. 
                  N.B. can't use this directly with S, A etc; must pass S_idx into self.get_S_indeces().
        '''
        if with_crossreg:
            crossreg = with_crossreg
        else:
            crossreg = self.crossreg
        df_mapping = crossreg.get_mappings_cells(mapping_type=mapping)
        df_col = self.get_df_col(with_crossreg=with_crossreg)
        S_idx = get_actual_cells_from_df_session(df_mapping[df_col])
        indeces_into_S = self.get_S_indeces(S_idx)

        S = self.S[indeces_into_S]

        # Must do dictionary comprehension to get subset of S_spikes dictionary with keys (S indeces) that match
        # those of indeces_into_S.
        S_spikes = {k : v for k,v in self.S_spikes.items() if k in indeces_into_S}
        if with_peakval:
            S_peakval = {k : v for k,v in self.S_peakval.items() if k in indeces_into_S}
        else:
            S_peakval = None

        return [S, S_spikes, S_peakval, S_idx]

        '''
        S_exp = S_exp[:,self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]]
        self.S_exp = S_exp
        return [S_exp, cell_indices]
        '''

    def process_binned_sp_rates_mapping(self, mapping, bin_width, plot_it=False, with_peakval=False, with_engram=False, crossreg=None):
        '''
        Calculate binned average spike rates for cells in a given session for the given mapping.

        bin_width should be in number of frames.
        '''

        # Get actual cells of self's session that are mapped to the desired mapping. (We don't need the resulting S_idx here.)
        if mapping == 'full':
            S_spikes = self.S_spikes
            S_peakval = self.S_peakval
            S = self.S
        else:
            [S, S_spikes, S_peakval, _] = self.get_S_mapping(mapping, with_peakval=with_peakval, with_crossreg=crossreg)

        #f_spikes_mapping = self.S_spikes[S_idx]

        if with_engram:
            S_mask, S_indices, S, S_spikes, S_peakval = get_engram_cells(S, S_spikes, S_peakval)
            if mapping == 'full': # save for further reuse
                self.S_mask_eng = S_mask
                self.S_indices_eng = S_indices
                self.S_eng = S
                self.S_spikes_eng = S_spikes
                self.S_peakval_eng = S_peakval

        binned_rates = []
        t_length = S.shape[1]-1
        beg_period = range(0, t_length - bin_width, bin_width)
        end_period = range(bin_width, t_length, bin_width)
        if with_peakval:
            binned_rates = get_avg_activity_in_period(S_spikes, S_peakval, beg_period, end_period)
        else:
            binned_rates = get_avg_sp_rate_in_period(S_spikes, beg_period, end_period)
        return binned_rates

    def get_ROI_mapping(self, mapping, with_peakval=False):
        [S, S_spikes, _, S_idx] = self.get_S_mapping(mapping, with_peakval=with_peakval)
        indeces_into_A = self.get_S_indeces(S_idx)
        A_mapping = self.A[indeces_into_A,:,:]
        #A_mapping = self.A[S_idx,:,:]
        if with_peakval:
            A_peakval = np.zeros((self.A.shape[1], self.A.shape[2]))
            for i in indeces_into_A:
                if self.S_peakval[i].size == 0:
                    peakval_mean = 0
                else:
                    peakval_mean = np.mean(self.S_peakval[i])
                A_peakval += (self.A[i,:,:]>0).astype(int) * peakval_mean
                if np.isnan(A_peakval[0,0]):
                    print(self.mouse, mapping, i, ' caused nan')
            return A_peakval
        else:
            A_flattened = np.sum(A_mapping,0)
            return A_flattened

    def get_location_data(self):
        self.behavcam_output_dir = os.path.join(self.dpath, 'BehavCam')

        if self.behaviour_type == 'movement':

            if self.saver_loc_data.check_exists('loc_df'):
                loc_df = self.saver_loc_data.load('loc_df')
            else:
                self.set_minian_output_dir()
                num_files = len(glob.glob(os.path.join(self.behavcam_output_dir, "*LocationOutput.csv")))
                loc_files = []
                for i in range(num_files):
                    loc_files.append(os.path.join(self.behavcam_output_dir, '{}_LocationOutput.csv'.format(i)))

                # should already be sorted but this is just in case. (Confirmed to work if swap elements in original list)
                self.loc_files = natsorted(loc_files) 

                loc_dfs = []
                for loc_file in self.loc_files:
                    df = pd.read_csv(loc_file)
                    loc_dfs.append(df)

                if not loc_dfs:
                    return # don't have .csv files processed yet..

                loc_df = pd.concat(loc_dfs)
                self.saver_loc_data.save(loc_df, 'loc_df')

            # Trim loc_df to fit into experiment bounds as in BehaviourSession.find_exp_boundaries().
            # VERY IMPORTANT. Otherwise locations/velocities won't match behaviour/miniscope data.
            loc_df = loc_df[self.session_bounds[self.start_idx]:self.session_bounds[self.stop_idx]]

            # NB: session_bounds are the time stamps for the behavcam beginning and end, so use those to find appropriate frame
            # numbers for location data. Trim loc_df accordingly.
            #loc_df = loc_df.iloc[self.session_bounds[0]:self.session_bounds[1]]
            #^^^ don't do this now, self.tstamp_behavcam etc are not bounded as such.

            # If we want movement, we need to "assign" velocities to each miniscope frame. Because the framerate of the
            # miniscope cam is higher than the behavcam, we need to go through all miniscope frames and assign the velocity 
            # for the closest behavcam timestamp frame to it. Thus, first we need to build a "velocity vector" for the behavcam
            # frame list.
            #velocities_behavcam = np.zeros(self.fnum_behavcam.shape[0])
            velocities_behavcam = np.zeros(loc_df['Distance_cm'].shape[0])
            loc_X_behavcam = np.zeros(loc_df['X'].shape[0])
            loc_Y_behavcam = np.zeros(loc_df['Y'].shape[0])
            max_behavcam_fnum = 0
            for i in range(velocities_behavcam.shape[0]):

                # It's possible for there to be a slight difference in the total number of frames due to video processing error
                # (e.g., G09 TFC_cond/TFC_cond) In that case just pad the end of the velocities_behavcam with the last calculated
                # distance.
                if i >= loc_df['Distance_cm'].shape[0]:
                    if max_behavcam_fnum == 0:
                        max_behavcam_fnum = i-1
                    idx_use = max_behavcam_fnum
                else:
                    idx_use = i

                dist = loc_df['Distance_cm'].iloc[idx_use]
                time_msec = self.tstamp_behavcam[i] - self.tstamp_behavcam[i-1]
                velocities_behavcam[i] = dist / (time_msec * 10**-3) # so values in this vector are cm/s !!
                loc_X_behavcam[i] = loc_df['X'].iloc[idx_use]
                loc_Y_behavcam[i] = loc_df['Y'].iloc[idx_use]

            # Logic: keep j as a moving index into tstamp_behavcam, and iterate through all tstamp_miniscope times;
            # assign the j'th behavcam *velocity* to the i'th miniscope *velocity* continually until the divergence between
            # the two timestamps is larger than between the current i'th miniscope tstamp and the j+1'st behavcam one, at which
            # point increment j before making the velocity assignment. Only increment if not at the end of tstamp_behavcam; 
            # if we are, just assign the last miniscope velocity to that behavcam one (when j is at the end of tstamp_behavcam 
            # typically i should also be at the end of the tstamp_miniscope). 
            velocities_miniscope = np.zeros(self.tstamp_miniscope.shape[0])
            loc_X_miniscope = np.zeros(self.tstamp_miniscope.shape[0])
            loc_Y_miniscope = np.zeros(self.tstamp_miniscope.shape[0])            
            j = 1 # behavcam velocity index
            for i in range(1,self.tstamp_miniscope.shape[0]):
                if (j + 1) != velocities_behavcam.shape[0]:
                    if abs(self.tstamp_miniscope[i] - self.tstamp_behavcam[j]) > \
                        abs(self.tstamp_miniscope[i] - self.tstamp_behavcam[j+1]):
                        j += 1
                velocities_miniscope[i] = velocities_behavcam[j]
                loc_X_miniscope[i] = loc_X_behavcam[j]
                loc_Y_miniscope[i] = loc_Y_behavcam[j]
            
            self.velocities_behavcam = velocities_behavcam
            self.velocities_miniscope = velocities_miniscope
            self.loc_X_behavcam = loc_X_behavcam
            self.loc_Y_behavcam = loc_Y_behavcam
            self.loc_X_miniscope = loc_X_miniscope
            self.loc_Y_miniscope = loc_Y_miniscope

            # Not needed anymore..
            '''
            self.velocities_behavcam_full = velocities_behavcam
            self.velocities_miniscope_full = velocities_miniscope
            self.loc_X_behavcam_full = loc_X_behavcam
            self.loc_Y_behavcam_full = loc_Y_behavcam
            self.loc_X_miniscope_full = loc_X_miniscope
            self.loc_Y_miniscope_full = loc_Y_miniscope

            self.velocities_behavcam = \
                self.velocities_behavcam_full[self.behavcam_exp_ts[self.start_idx]:self.behavcam_exp_ts[self.stop_idx]]
            self.loc_X_behavcam = \
                self.loc_X_behavcam_full[self.behavcam_exp_ts[self.start_idx]:self.behavcam_exp_ts[self.stop_idx]]
            self.loc_Y_behavcam = \
                self.loc_Y_behavcam_full[self.behavcam_exp_ts[self.start_idx]:self.behavcam_exp_ts[self.stop_idx]]

            self.velocities_miniscope = \
                self.velocities_miniscope_full[self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]]
            self.loc_X_miniscope = \
                self.loc_X_miniscope_full[self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]]
            self.loc_Y_miniscope = \
                self.loc_Y_miniscope_full[self.miniscope_exp_fnum[self.start_idx]:self.miniscope_exp_fnum[self.stop_idx]]
            '''
            
            # Smooth the locations and velocities as well
            self.velocities_behavcam_smooth = gaussian_filter(self.velocities_behavcam, sigma=SMOOTH_SIGMA)
            self.loc_X_behavcam_smooth = gaussian_filter(self.loc_X_behavcam, sigma=SMOOTH_SIGMA)
            self.loc_Y_behavcam_smooth = gaussian_filter(self.loc_Y_behavcam, sigma=SMOOTH_SIGMA)

            self.velocities_miniscope_smooth = gaussian_filter(self.velocities_miniscope, sigma=SMOOTH_SIGMA)
            self.loc_X_miniscope_smooth = gaussian_filter(self.loc_X_miniscope, sigma=SMOOTH_SIGMA)
            self.loc_Y_miniscope_smooth = gaussian_filter(self.loc_Y_miniscope, sigma=SMOOTH_SIGMA)

    def ts2frame(self, ts):
        '''
        Gets the frame closest to the given timestamp (in ms). Works with ints or lists.
        '''
        out_frames = []
        if type(ts) == int:
            ts = [ts]
        for t in ts:
            found_fnum = np.where(abs(self.tstamp_miniscope - t) < self.tstamp_tol)[0][0]
            out_frames.append(found_fnum)
        if len(out_frames)==1:
            return out_frames[0]
        else:
            return out_frames

    def t2frame(self, t):
        '''
        Gets the frame closest to the given timestamp (in s). Mostly for plotting conversions.
        '''
        return self.ts2frame(t*1000)


class TraceFearCondSession(BehaviourSession):
    def __init__(self, mouse, dpath, session_bounds=[], period_override=[], plot_sample_cell=False, data_dir='', crossreg='', savepath='', behaviour_type=None):
        self.light_onsets = np.array([0, 1299])
        self.light_duration = 1
        self.tone_onsets_def = np.array([185, 420, 660, 900, 1140]) # alas, 185s for first tone set by accident rather than 180 but kept consistent for all mice..
        self.tone_duration = 20
        self.shock_onsets_def = np.array([220, 460, 700, 940, 1180])
        self.shock_duration = 2

        if period_override:
            self.periods = period_override
        else:
            self.periods = range(len(self.tone_onsets_def))

        # These are frame numbers
        # First group - relative to start of *experiment* (light on).
        self.tone_onsets = []
        self.tone_offsets = []
        self.shock_onsets = []
        self.shock_offsets = []
        self.post_shock_onsets = []
        self.post_shock_offsets = []
        # Second group - 'adjusted', that is, absolute location in raw recording (not relative to start of experiment)
        self.tone_onsets_adj = []
        self.tone_offsets_adj = []
        self.shock_onsets_adj = []
        self.shock_offsets_adj = []
        self.post_shock_onsets_adj = []
        self.post_shock_offsets_adj = []
        self.period_bounds = []

        # Placeholders for calculations of spike rates during tone/shock periods for 
        # specified mappings
        self.tone_sp_rates_mapping = dict()
        self.shock_sp_rates_mapping = dict()
        self.post_shock_sp_rates_mapping = dict()
        self.tone_activity_mapping = dict()
        self.shock_activity_mapping = dict()
        self.post_shock_activity_mapping = dict()

        session_type = 'TFC_cond'
        super().__init__(mouse, dpath, session_bounds=session_bounds, plot_sample_cell=plot_sample_cell, data_dir=data_dir, \
            session_group='session.2', crossreg=crossreg, savepath=savepath, session_type=session_type, behaviour_type=behaviour_type, 
            saver_prefix='TFC_cond')

    def find_exp_boundaries(self):
        super().find_exp_boundaries()

        # Step 4. From there, interpolate where the various tone/shock periods are in the **Miniscope** timestamps
        # and then frame numbers. Expand all times to milliseconds (hence * 1000 in the below).

        # Get interpolated times.
        tone_onsets = self.tone_onsets_def[self.periods] * 1000
        tone_offsets = tone_onsets[self.periods] + self.tone_duration*1000
        shock_onsets = self.shock_onsets_def[self.periods] * 1000
        shock_offsets = shock_onsets[self.periods] + self.shock_duration*1000

        # Find closest corresponding Miniscope timestamps and get the frame numbers.
        for ts in tone_onsets:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_onsets.append(found_ts[0][0])
        for ts in tone_offsets:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_offsets.append(found_ts[0][0])
        for ts in shock_onsets:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.shock_onsets.append(found_ts[0][0])
        for ts in shock_offsets:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.shock_offsets.append(found_ts[0][0])

        # Now find post-shock, based on the already found tone/shock boundaries
        for i in range(len(self.tone_onsets)):
            if i == len(self.tone_onsets)-1:
                self.post_shock_onsets.append(self.shock_offsets[i])
                self.post_shock_offsets.append(self.miniscope_exp_fnum[self.stop_idx])
            else:
                self.post_shock_onsets.append(self.shock_offsets[i])
                self.post_shock_offsets.append(self.tone_onsets[i+1])

        for i in range(len(self.tone_onsets_adj)):
            if i == len(self.tone_onsets_adj)-1:
                self.post_shock_onsets_adj.append(self.shock_offsets_adj[i])
                self.post_shock_offsets_adj.append(self.miniscope_exp_fnum[self.stop_idx])
            else:
                self.post_shock_onsets_adj.append(self.shock_offsets_adj[i])
                self.post_shock_offsets_adj.append(self.tone_onsets_adj[i+1])

    def process_avg_sp_rates_mapping(self, mapping, with_peakval=False):
        '''
        Calculate average spike rates for cells in TFC_cond session for the given mapping.

        If mapping is set to 'full', then all cells for this session will be processed and 
        placed in a separate internal variable.
        '''

        # Get actual cells of self's session that are mapped to the desired mapping. (We don't need the resulting S_idx here.)
        if mapping == 'full':
            S_spikes = self.S_spikes
            S_peakval = self.S_peakval
            S_idx = self.S_idx
        else:
            [S, S_spikes, S_peakval, S_idx] = self.get_S_mapping(mapping, with_peakval=with_peakval) 

        # Get actual cells of TFC_cond ('session.2') that are mapped to the desired mapping
        #[S, S_idx] = self.get_S_mapping(df_mapping[self.session_group])

        #s_idx = S_idx.index(test_unit_id[mouse])
        if self.plot_sample_cell:
            s_idx = S_idx[0]
            find_spikes_ca(S[s_idx,:], self.thres, plotit=True)
            plt.title(self.mouse+' mapped TFC_cond cell '+str(S_idx[s_idx])+' in '+mapping)
            [plt.axvline(x, c='b', ls='--') for x in self.tone_onsets]
            [plt.axvline(x, c='b', ls='--') for x in self.tone_offsets]
            [plt.axvline(x, c='r', ls='--') for x in self.shock_onsets]
            [plt.axvline(x, c='r', ls='--') for x in self.shock_offsets]

        # Get all frame numbers for spikes for this session, bounded by the experiment (in the TraceFearCondSession constructor)
        #self.f_spikes_mapping = find_spikes_ca_S(S, self.thres)
        #f_spikes_mapping = self.get_S_spikes_mapping(mapping)

        # Get spikes within tone and shock periods and calculate average firing rate for all cells across all respective periods
        if with_peakval:
            self.tone_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, self.tone_onsets, self.tone_offsets)
            self.shock_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, self.shock_onsets, self.shock_offsets)
            self.post_shock_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, self.post_shock_onsets, self.post_shock_offsets)
        else:
            self.tone_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, self.tone_onsets, self.tone_offsets)
            self.shock_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, self.shock_onsets, self.shock_offsets)
            self.post_shock_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, self.post_shock_onsets, self.post_shock_offsets)

    def find_period_bounds(self):
        '''
        Generate a dictionary of tuples of onset,offset frames for behaviourally relevant periods, as well as the labels.
        '''
        period_bounds = []
        for i in self.periods:
            if i == 0:
                period_bounds.append(('initial', (0, self.tone_onsets[0]-1)))
            else:
                period_bounds.append(('iti', (self.shock_onsets[i-1], self.tone_onsets[i]-1)))
            
            period_bounds.append(('tone', (self.tone_onsets[i], self.tone_offsets[i]-1)))
            period_bounds.append(('post-tone', (self.tone_offsets[i], self.shock_onsets[i]-1)))
            period_bounds.append(('shock', (self.shock_onsets[i], self.shock_offsets[i]-1)))        
        period_bounds.append(('iti', (self.shock_offsets[max(self.periods)], self.S.shape[1]-1)))
        self.period_bounds = period_bounds

    def get_period_bounds(self, wanted_type='all'):
        if not self.period_bounds:
            self.find_period_bounds()
        if wanted_type == 'all':
            return self.period_bounds
        wanted_pb = []
        for (type, bounds) in self.period_bounds:
            if type == wanted_type:
                wanted_pb.append((type, bounds))
        return wanted_pb


class TestBSession(BehaviourSession):
    def __init__(self, mouse, dpath, session_bounds=[], period_override=[], plot_sample_cell=False, data_dir='', crossreg='', savepath='', is_1wk=False, \
        session_group='', behaviour_type=None):
        self.light_onsets = np.array([0, 899])
        self.light_duration = 1
        self.tone_onsets_def = np.array([180, 420, 660])
        self.tone_duration = 20
        self.is_1wk = is_1wk
        self.session_group = session_group

        if period_override:
            self.periods = period_override
        else:
            self.periods = range(len(self.tone_onsets_def))

        # These are frame numbers
        # First group - relative to start of *experiment* (light on).            
        self.tone_onsets = []
        self.tone_offsets = []
        self.post_tone_onsets = []
        self.post_tone_offsets = []
        self.tone_post_tone_onsets = []
        self.tone_post_tone_offsets = []
        # Second group - 'adjusted', that is, absolute location in raw recording (not relative to start of experiment)
        self.tone_onsets_adj = []
        self.tone_offsets_adj = []
        self.post_tone_onsets_adj = []
        self.post_tone_offsets_adj = []
        self.tone_post_tone_onsets_adj = []
        self.tone_post_tone_offsets_adj = []
        self.period_bounds = []

        # Placeholders for calculations of spike rates during tone/shock periods for 
        # specified mappings
        self.tone_sp_rates_mapping = dict()
        self.post_tone_sp_rates_mapping = dict()
        self.tone_post_tone_sp_rates_mapping = dict()
        self.tone_activity_mapping = dict()
        self.post_tone_activity_mapping = dict()
        self.tone_post_tone_activity_mapping = dict()

        if self.is_1wk:
            session_type = 'Test_B_1wk'
        else:
            session_type = 'Test_B'
        super().__init__(mouse, dpath, session_bounds=session_bounds, plot_sample_cell=plot_sample_cell, data_dir=data_dir, \
            session_group=session_group, crossreg=crossreg, savepath=savepath, session_type=session_type, behaviour_type=behaviour_type, \
            saver_prefix='Test_B')

    def find_exp_boundaries(self):
        super().find_exp_boundaries()

        # Step 4. From there, interpolate where the various tone/shock periods are in the **Miniscope** timestamps
        # and then frame numbers. Expand all times to milliseconds (hence * 1000 in the below).

        # Get interpolated times.
        tone_onsets = self.tone_onsets_def[self.periods] * 1000
        tone_offsets = tone_onsets[self.periods] + self.tone_duration*1000

        # Find closest corresponding Miniscope timestamps and get the frame numbers.
        for ts in tone_onsets:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_onsets.append(found_ts[0][0])
        for ts in tone_offsets:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_offsets.append(found_ts[0][0])
    
        # Now find post-tone, based on the already found tone/shock boundaries
        for i in range(len(self.tone_onsets)):
            self.post_tone_onsets.append(self.tone_offsets[i])
            self.tone_post_tone_onsets.append(self.tone_onsets[i])

            if i == len(self.tone_onsets)-1:
                self.post_tone_offsets.append(self.miniscope_exp_fnum[self.stop_idx])
                self.tone_post_tone_offsets.append(self.miniscope_exp_fnum[self.stop_idx])
            else:
                self.post_tone_offsets.append(self.tone_onsets[i+1])
                self.tone_post_tone_offsets.append(self.tone_onsets[i+1])

    def process_avg_sp_rates_mapping(self, mapping, with_peakval=False):
        '''
        Calculate average spike rates for cells in TFC_cond session for the given mapping.

        If mapping is set to 'full', then all cells for this session will be processed and 
        placed in a separate internal variable.
        '''

        # Get actual cells of self's session that are mapped to the desired mapping. (We don't need the resulting S_idx here.)
        if mapping == 'full':
            S_spikes = self.S_spikes
            S_peakval = self.S_peakval
            S_idx = self.S_idx
        else:
            [S, S_spikes, S_peakval, S_idx] = self.get_S_mapping(mapping, with_peakval=with_peakval) 

        # Get actual cells of TFC_cond ('session.2') that are mapped to the desired mapping
        #[S, S_idx] = self.get_S_mapping(df_mapping[self.session_group])

        #s_idx = S_idx.index(test_unit_id[mouse])
        if self.plot_sample_cell:
            s_idx = S_idx[0]
            find_spikes_ca(S[s_idx,:], self.thres, plotit=True)
            if self.is_1wk:
                suffix_str = ' 1wk'
            else:
                suffix_str = ''
            plt.title(self.mouse+' mapped Test B{} cell '.format(suffix_str)+str(S_idx[s_idx])+' in '+mapping)
            [plt.axvline(x, c='b', ls='--') for x in self.tone_onsets]
            [plt.axvline(x, c='b', ls='--') for x in self.tone_offsets]

        # Get all frame numbers for spikes for this session, bounded by the experiment (in the TraceFearCondSession constructor)
        #self.f_spikes_mapping = find_spikes_ca_S(S, self.thres)
        #f_spikes_mapping = self.get_S_spikes_mapping(mapping)

        # Get spikes within tone and shock periods and calculate average firing rate for all cells across all respective periods
        if with_peakval:
            self.tone_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, self.tone_onsets, self.tone_offsets)
            self.post_tone_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, self.post_tone_onsets, self.post_tone_offsets)
            self.tone_post_tone_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, self.tone_post_tone_onsets, self.tone_post_tone_offsets)
        else:
            self.tone_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, self.tone_onsets, self.tone_offsets)
            self.post_tone_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, self.post_tone_onsets, self.post_tone_offsets)
            self.tone_post_tone_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, self.tone_post_tone_onsets, self.tone_post_tone_offsets)

    def find_period_bounds(self):
        '''
        Generate a dictionary of tuples of onset,offset frames for behaviourally relevant periods, as well as the labels.
        '''
        period_bounds = []
        for i in self.periods:
            if i == 0:
                period_bounds.append(('initial', (0, self.tone_onsets[0]-1)))
            else:
                period_bounds.append(('post-tone', (self.tone_onsets[i-1], self.tone_onsets[i]-1)))
            period_bounds.append(('tone', (self.tone_onsets[i], self.tone_offsets[i]-1)))
        period_bounds.append(('post-tone', (self.tone_offsets[max(self.periods)], self.S.shape[1]-1)))
        self.period_bounds = period_bounds

    def get_period_bounds(self, wanted_type='all'):
        if not self.period_bounds:
            self.find_period_bounds()
        if wanted_type == 'all':
            return self.period_bounds
        wanted_pb = []
        for (type, bounds) in self.period_bounds:
            if type == wanted_type:
                wanted_pb.append((type, bounds))
        return wanted_pb

class LinearTrackSession(BehaviourSession):
    def __init__(self, mouse, dpath, session_bounds=[], plot_sample_cell=False, LT_type='', data_dir='', crossreg='', savepath='', behaviour_type=None):
        if LT_type == 'LT1':
            self.LT_group = 'session'
        else:
            self.LT_group = 'session.1'
        self.LT_type = LT_type
        self.exp_sp_rates_mapping = dict()
        self.exp_activity_mapping = dict()

        session_type = LT_type
        super().__init__(mouse, dpath, session_bounds=session_bounds, plot_sample_cell=plot_sample_cell, data_dir=data_dir, \
            session_group=self.LT_group, crossreg=crossreg, savepath=savepath, session_type=session_type, behaviour_type=behaviour_type, \
            saver_prefix=LT_type)
    
    def process_avg_sp_rates_mapping(self, mapping, with_peakval=False):
        '''
        Calculate average spike rates for cells in a LinearTrack session for the given mapping.
        '''

        # Get actual cells of self's session that are mapped to the desired mapping. (We don't need the resulting S_idx here.)
        if mapping == 'full':
            S_spikes = self.S_spikes
            S_peakval = self.S_peakval
            S_idx = self.S_idx
        else:
            [S, S_spikes, S_peakval, S_idx] = self.get_S_mapping(mapping, with_peakval=with_peakval)

        # Get actual cells of LT session ('session.2') that are mapped to the desired mapping
        #[S, S_idx] = self.get_S_exp(df_mapping[self.LT_group])

        #s_idx = S_idx.index(test_unit_id[mouse])
        if self.plot_sample_cell:
            s_idx = S_idx[0]
            find_spikes_ca(S[s_idx,:], self.thres, plotit=True)
            plt.title(self.mouse+' mapped '+self.LT_type+' cell '+str(S_idx[s_idx]))

        # Get all frame numbers for spikes for this session, bounded by the experiment (in the TraceFearCondSession constructor)
        #self.f_spikes_mapping = find_spikes_ca_S(S, self.thres)

        # Get spikes within tone and shock periods and calculate average firing rate for all cells across all respective periods
        # Put beg, end periods in a list so that the get_avg_sp_rate_in_period() can iterate over them; it's also written to handle
        # tone periods from the equivalent TraceFearCondSession function.
        if with_peakval:
            self.exp_activity_mapping[mapping] = get_avg_activity_in_period(S_spikes, S_peakval, \
                [self.miniscope_exp_fnum[self.start_idx]], [self.miniscope_exp_fnum[self.stop_idx]])
        else:
            self.exp_sp_rates_mapping[mapping] = get_avg_sp_rate_in_period(S_spikes, \
                [self.miniscope_exp_fnum[self.start_idx]], [self.miniscope_exp_fnum[self.stop_idx]])

def further_process_TFC_cond():
        
    # Then ultimately use the Miniscope frame numbers to delinate S and C arrays for Minian analysis.
    # We assume the subdirectories are already renamed to correspond to the cross-registration labels.
    # For TFC-cond, this is 'crossreg1'.
    minian_output_dir = os.path.join(dpath, 'Miniscope', 'minian_crossreg1')
    [C, S] = get_CS_matrices(minian_output_dir)

    # Sanity check that the lengths of C,S are the same as the number of frames we got earlier
    # Recall dims of C,S are (time, frame).
    assert C.shape[1] == S.shape[1] == len(fnum_miniscope)
    
    get_mappings_crossreg1(path_prefix+dpath_TFC_cond_parent)
    return

###############
### LT1 session
###############

#def process_LT1(dpath):
#    [fnum_miniscope, tstamp_miniscope, fnum_behavcam, tstamp_behavcam] = get_timeStamps(dpath)

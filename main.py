import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from caban.utilities import *
from caban.sessions import *
from caban.analysis import *
from datetime import datetime

behavcam_fps = 15
LOCAL_DATA = True
data_dir = 'minian_crossreg1'
crossreg_file_TFC_cond = 'mappings_crossreg1.csv' # TFC_cond/{LT1,LT2,TFC_Cond}
crossreg_file_4 = 'mappings_crossreg_4.csv' # TFC_cond/TFC_cond; Test_B/Test_B; Test_B_1wk/Test_B_1wk
DEBUG = False
DEVEL_SWITCH = False # Turn on when don't want to run all analyses while adding new code, but turn off when running all analyses.
BIN_WIDTH = MINISCOPE_FPS*10 # so, 10 seconds. Units in *frames*

now = datetime.now()
PLOTS_DIR = MAIN_DRIVE+'\\data\\vsekulic\\OF_test\\plots'
PLOTS_DIR = os.path.join(PLOTS_DIR, now.strftime('%Y-%m-%d %H_%M_%S'))
PAPER_DIR = os.path.join('C:\\','Users','vlads','Dropbox','1-McHugh postdoc','3-PAPER','paper_plots')

### Analysis switches
BEHAVIOUR_TYPE = 'movement' # other options: 'immobility', None

### Plot switches
plot_sample_cell = False
plot_sp_rates = True
plot_binned_sp_rates = True
plot_ROIs = False
plot_proportional_activities = True
plot_LT_firing_rate_changes = True
want_sample_traces_paper = True
plot_PSTH = True
plot_population_vectors = True
perform_agglomerative_clustering = True
process_for_R = False
plot_pf_and_loc = True
plot_binned_activities = True

### General plotting
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

###############################
##### ADD MOUSE DATA BELOW ####
###############################

mouse_groups = {
    'G05' : 'hM3D',
    'G06' : 'hM4D',
    'G07' : 'hM4D',
    'G08' : 'mCherry',
    'G09' : 'mCherry',
    'G10' : 'hM3D',
    'G11' : 'hM3D',
    'G12' : 'mCherry',
    'G13' : 'mCherry',
    'G14' : 'hM4D',
    'G15' : 'hM4D',
    'G16' : 'mCherry',
    'G17' : 'mCherry',
    'G18' : 'hM3D',
    'G19' : 'hM3D',
    'G20' : 'hM4D',
    'G21' : 'hM4D'
}
if DEBUG:
    #mouse_groups = { 'G06' : 'hM4D', 'G08' : 'mCherry', 'G10' : 'hM3D' }
    mouse_groups = { 'G05' : 'hM3D' }
mouse_list = mouse_groups.keys()

mice_per_group = dict()
mice_per_group_Test_B_B_1wk = dict()
for mouse, group in mouse_groups.items():
    if group not in mice_per_group:
        mice_per_group[group] = []
    if group not in mice_per_group_Test_B_B_1wk:
        mice_per_group_Test_B_B_1wk[group] = []
    mice_per_group[group].append(mouse)
    if mouse not in ['G07', 'G10', 'G15', 'G21']:
        mice_per_group_Test_B_B_1wk[group].append(mouse)

mice_skip_LT = ['G05']

# Miniscope data drives mount points
DRIVE_1a = 'E' #'D'
DRIVE_1b = 'F' #'E'

mouse_drive_prefix = {
    'G05' : DRIVE_1a, 'G06' : DRIVE_1a, 'G07' : DRIVE_1a, 'G08' : DRIVE_1a, 'G09' : DRIVE_1a, 'G10' : DRIVE_1a, 'G11' : DRIVE_1a,
    'G12' : DRIVE_1b, 'G13' : DRIVE_1b, 'G14' : DRIVE_1b, 'G14' : DRIVE_1b, 'G15' : DRIVE_1b, 'G16' : DRIVE_1b, 'G17' : DRIVE_1b,
    'G18' : DRIVE_1b, 'G19' : DRIVE_1b, 'G20' : DRIVE_1b, 'G21' : DRIVE_1b
}

mouse_path_prefix = dict()
for mouse in mouse_drive_prefix:
    if LOCAL_DATA:
        path_prefix = ":\\\\data\\vsekulic\\OF_test\\"
        mouse_path_prefix[mouse] = mouse_drive_prefix[mouse] + path_prefix
    else:
        mouse_path_prefix[mouse] = "\\\\cbp-db.bnf.brain.riken.jp\\vsekulic\data\\vsekulic\\OF_test\\"

dpath_mouse = {
    'G05' : 'G05-ST637_hM3D',
    'G06' : 'G06-ST688_hM4D',
    'G07' : 'G07-ST689_hM4D',
    'G08' : 'G08-ST701_mCherry',
    'G09' : 'G09-ST702_mCherry',
    'G10' : 'G10-ST703_hM3D',
    'G11' : 'G11-ST705_hM3D',
    'G12' : 'G12-ST709-mCherry',
    'G13' : 'G13-ST710-mCherry',
    'G14' : 'G14-ST719-hM4D',
    'G15' : 'G15-ST721-hM4D',
    'G16' : 'G16-ST731-mCherry',
    'G17' : 'G17-ST741-mCherry',
    'G18' : 'G18-ST734-hM3D',
    'G19' : 'G19-ST735-hM3D',
    'G20' : 'G20-ST760-hM4D',
    'G21' : 'G21-ST762-hM4D'
}

###########
# TFC_Cond
###########

dpath_TFC_cond_day = {
    'G05' : '2021_08_30-TFC_cond',
    'G06' : '2021_10_18-TFC_cond',
    'G07' : '2021_10_18-TFC_cond',
    'G08' : '2021_11_08-TFC_cond',
    'G09' : '2021_11_08-TFC_cond',
    'G10' : '2021_11_23-TFC_cond',
    'G11' : '2021_11_23-TFC_cond',
    'G12' : '2022_01_03-TFC_cond',
    'G13' : '2022_01_03-TFC_cond',
    'G14' : '2022_01_11-TFC_cond',
    'G15' : '2022_01_11-TFC_cond',
    'G16' : '2022_01_24-TFC_cond',
    'G17' : '2022_01_24-TFC_cond',
    'G18' : '2022_02_07-TFC_cond',
    'G19' : '2022_02_07-TFC_cond',
    'G20' : '2022_03_22-TFC_cond',
    'G21' : '2022_03_22-TFC_cond'
}
dpath_TFC_cond = {
    'G05' : '18_22_57-TFC_cond',
    'G06' : '11_42_00-TFC_cond',
    'G07' : '15_14_32-TFC_cond',
    'G08' : '15_13_23-TFC_cond',
    'G09' : '18_54_05-TFC_cond',
    'G10' : '16_32_14-TFC_cond',
    'G11' : '19_24_52-TFC_cond',
    'G12' : '15_12_38-TFC_cond',
    'G13' : '17_13_06-TFC_cond',
    'G14' : '16_00_53-TFC_cond',
    'G15' : '18_10_56-TFC_cond',
    'G16' : '16_06_44-TFC_cond',
    'G17' : '918_26_30-TFC_cond',
    'G18' : '15_29_25-TFC_cond',
    'G19' : '17_35_25-TFC_cond',
    'G20' : '14_37_19-TFC_cond',
    'G21' : '17_39_30-TFC_cond'
}
TFC_cond_exp_frames = { # frame numbers are from BehavCam, not Miniscope cam!
    'G05' : [75, 19550],
    'G06' : [135, 19617],
    'G07' : [83, 19564],
    'G08' : [67, 19549],
    'G09' : [51, 14440], # because Miniscope cam 19.avi and some behav cam were corrupted
    'G10' : [72, 19553],
    'G11' : [48, 19530],
    'G12' : [43, 19524],
    'G13' : [44, 19527],
    'G14' : [676, 20159],
    'G15' : [37, 19519],
    'G16' : [43, 19525],
    'G17' : [38, 19519],
    'G18' : [65, 19548],
    'G19' : [80, 19562],
    'G20' : [37, 19519],
    'G21' : [34, 19516]
}

dpath_TFC_cond_LT1 = {
    'G05' : '16_47_02-LT1',
    'G06' : '10_14_41-LT1',
    #'G07' : '13_13_24-LT1'
    'G07' : '13_25_42-LT1b',
    'G08' : '13_22_42-LT1',
    'G09' : '16_53_07-LT1',
    'G10' : '14_51_37-LT1',
    'G11' : '17_51_25-LT1',
    'G12' : '14_11_51-LT1',
    'G13' : '16_10_53-LT1',
    'G14' : '14_53_32-LT1',
    'G15' : '17_05_25-LT1',
    'G16' : '14_59_15-LT1',
    'G17' : '17_38_16-LT1',
    'G18' : '14_19_24-LT1',
    'G19' : '16_25_38-LT1',
    'G20' : '13_31_20-LT1',
    'G21' : '16_32_01-LT1'
}
dpath_TFC_cond_LT2 = {
    'G05' : '17_44_51-LT2',
    'G06' : '11_05_51-LT2',
    'G07' : '14_35_40-LT2',
    'G08' : '14_43_03-LT2',
    'G09' : '18_08_37-LT2',
    'G10' : '15_53_29-LT2',
    'G11' : '18_55_26-LT2',
    'G12' : '14_52_29-LT2',
    'G13' : '16_53_52-LT2',
    'G14' : '15_39_23-LT2',
    'G15' : '17_51_17-LT2',
    'G16' : '15_47_20-LT2',
    'G17' : '19_02_10-LT2',
    'G18' : '15_10_08-LT2',
    'G19' : '17_14_57-LT2',
    'G20' : '14_16_55-LT2',
    'G21' : '17_19_40-LT2'
}
LT1_exp_frames = {
    'G05' : [0,-1],
    'G06' : [0,-1],
    'G07' : [0,-1],
    'G08' : [442,-1],
    'G09' : [314,-1],
    'G10' : [249,-1],
    'G11' : [240,-1],
    'G12' : [370,-1],
    'G13' : [304,-1],
    'G14' : [275,-1],
    'G15' : [220,-1],
    'G16' : [582,-1],
    'G17' : [200,-1],
    'G18' : [0,-1],
    'G19' : [288,-1],
    'G20' : [246,-1],
    'G21' : [188,-1]
}
LT2_exp_frames = {
    'G05' : [0,-1],
    'G06' : [0,-1],
    'G07' : [0,-1],
    'G08' : [291,-1],
    'G09' : [252,-1],
    'G10' : [227,-1],
    'G11' : [248,-1],
    'G12' : [430,-1],
    'G13' : [229,-1],
    'G14' : [352,-1],
    'G15' : [261,-1],
    'G16' : [200,-1],
    'G17' : [326,-1],
    'G18' : [264,-1],
    'G19' : [244,-1],
    'G20' : [226,-1],
    'G21' : [201,-1]
}


#########
# Test B
#########

dpath_Test_B_day = {
    'G05' : '2021_09_01-TFC_test_B',
    'G06' : '2021_10_20-TFC_test_B',
    'G07' : '2021_10_20-TFC_test_B',
    'G08' : '2021_11_10-TFC_test_B',
    'G09' : '2021_11_10-TFC_test_B',
    'G10' : '2021_11_25-TFC_test_B',
    'G11' : '2021_11_25-TFC_test_B',
    'G12' : '2022_01_05-TFC_test_B',
    'G13' : '2022_01_05-TFC_test_B',
    'G14' : '2022_01_13-TFC_test_B',
    'G15' : '2022_01_13-TFC_test_B',
    'G16' : '2022_01_26-TFC_test_B',
    'G17' : '2022_01_26-TFC_test_B',
    'G18' : '2022_02_09-TFC_test_B',
    'G19' : '2022_02_09-TFC_test_B',
    'G20' : '2022_03_24-TFC_test_B',
    'G21' : '2022_03_24-TFC_test_B'
}
dpath_Test_B = {
    'G05' : '16_20_07-TFC_test_B',
    'G06' : '13_32_28-TFC_test_B',
    'G07' : '15_49_59-TFC_test_B',
    'G08' : '14_45_42-TFC_test_B',
    'G09' : '17_12_18-TFC_test_B',
    'G10' : '15_28_34-TFC_test_B',
    'G11' : '17_17_33-TFC_test_B',
    'G12' : '14_28_46-TFC_test_B',
    'G13' : '15_53_23-TFC_test_B',
    'G14' : '15_05_26-TFC_test_B',
    'G15' : '16_19_13-TFC_test_B',
    'G16' : '14_47_09-TFC_test_B',
    'G17' : '16_14_47-TFC_test_B',
    'G18' : '15_25_42-TFC_test_B',
    'G19' : '16_44_35-TFC_test_B',
    'G20' : '13_21_06-TFC_test_B',
    'G21' : '14_35_02-TFC_test_B'
}
Test_B_exp_frames = {
    'G05' : [56, 13537],
    'G06' : [751, 14231],
    'G07' : [],
    'G08' : [44, 13524],
    'G09' : [86, 13565],
    'G10' : [129, 13610],
    'G11' : [58, 13539],
    'G12' : [35, 13516],
    'G13' : [43, 13523],
    'G14' : [39, 13520],
    'G15' : [37, 13518],
    'G16' : [37, 13515],
    'G17' : [70, 13551],
    'G18' : [32, 13512],
    'G19' : [42, 13523],
    'G20' : [36, 13520],
    'G21' : [43, 13524]
}

dpath_Test_B_LT1 = {
    'G05' : '12_45_49-LT1',
}
Test_B_LT1_exp_frames = {
    'G05' : [0,-1],
}


#############
# Test B-1wk
#############

dpath_Test_B_1wk_day = {
    'G05' : '2021_09_06-TFC_test_B_1wk',
    'G06' : '2021_10_25-TFC_test_B_1wk',
    'G07' : '2021_10_25-TFC_test_B_1wk',
    'G08' : '2021_11_15-TFC_test_B_1wk',
    'G09' : '2021_11_15-TFC_test_B_1wk',
    'G10' : '2021_11_30-TFC_test_B_1wk',
    'G11' : '2021_11_30-TFC_test_B_1wk',
    'G12' : '2022_01_10-TFC_test_B_1wk',
    'G13' : '2022_01_10-TFC_test_B_1wk',
    'G14' : '2022_01_18-TFC_test_B_1wk',
    'G15' : '',
    'G16' : '2022_01_31-TFC_test_B_1wk',
    'G17' : '2022_01_31-TFC_test_B_1wk',
    'G18' : '2022_02_15-TFC_test_B_1wk',
    'G19' : '2022_02_15-TFC_test_B_1wk',
    'G20' : '2022_03_29-TFC_test_B_1wk',
    'G21' : '2022_03_29-TFC_test_B_1wk'
}
dpath_Test_B_1wk = {
    'G05' : '16_51_41-TFC_test_B_1wk', 
    'G06' : '12_23_30-TFC_test_B_1wk',
    'G07' : '15_19_51-TFC_test_B_1wk',
    'G08' : '14_58_20-TFC_test_B_1wk',
    'G09' : '17_15_18-TFC_test_B_1wk',
    'G10' : '16_07_16-TFC_test_B_1wk',
    'G11' : '17_57_20-TFC_test_B_1wk',
    'G12' : '15_31_10-TFC_test_B_1wk',
    'G13' : '16_50_10-TFC_test_B_1wk',
    'G14' : '15_10_50-TFC_test_B_1wk',
    'G15' : '',
    'G16' : '14_08_02-TFC_test_B_1wk',
    'G17' : '16_06_22-TFC_test_B_1wk',
    'G18' : '13_49_23-TFC_test_B_1wk',
    'G19' : '15_18_44-TFC_test_B_1wk',
    'G20' : '15_06_19-TFC_test_B_1wk',
    'G21' : '16_30_54-TFC_test_B_1wk'
}
Test_B_1wk_exp_frames = {
    'G05' : [56, 13537],
    'G06' : [50, 13531],
    'G07' : [49, 13529],
    'G08' : [467, 13947],
    'G09' : [52, 13533],
    'G10' : [53, 13533],
    'G11' : [49, 13529],
    'G12' : [40, 13520],
    'G13' : [43, 13524],
    'G14' : [40, 13520],
    'G15' : [],
    'G16' : [42, 13523],
    'G17' : [39, 13520],
    'G18' : [35, 13516],
    'G19' : [36, 13517],
    'G20' : [52, 13532],
    'G21' : [45, 13525]
}

dpath_Test_B_1wk_LT1 = {
    'G05' : '15_58_07-LT1',
}
Test_B_1wk_LT1_exp_frames = {
    'G05' : [0,-1],
}

test_unit_id = {
    'G05' : 22,
    'G06' : 413
}

'''    'G06' : 413,
    'G07' : 130,
    'G08' : 474,
    'G09' : 220,
    'G10' : 394,
    'G11' : 216,
    'G12' : 24,
    'G13' : 86
'''

period_overrides = {
    'G09' : [0,1,2,3]
}

###################
# Crossreg mappings
###################
# For cross-day mappings, can't count on chronological order, so have to explicitly state
# which session timestamp corresponds to which session name.

TFC_B_B_1wk_crossreg_groups = {
    'G05' : {'TFC_cond': '18_22_57', 'Test_B': '16_20_07', 'Test_B_1wk': '16_51_41'},
    'G06' : {'TFC_cond': '11_42_00', 'Test_B': '13_32_28', 'Test_B_1wk': '12_23_30'},
    #'G07' : {'TFC_cond': '15_14_32', 'Test_B': '15_49_59', 'Test_B_1wk' : '15_19_51'},
    'G07' : {'TFC_cond': '15_14_32', 'Test_B_1wk' : '15_19_51'},
    'G08' : {'TFC_cond': '15_13_23', 'Test_B': '14_45_42', 'Test_B_1wk' : '14_58_20'},
    'G09' : {'TFC_cond': '18_54_05', 'Test_B': '17_12_18', 'Test_B_1wk' : '17_15_18'},    
    'G10' : {'TFC_cond': '16_32_14', 'Test_B': '15_28_34', 'Test_B_1wk' : '16_07_16'},    
    'G11' : {'TFC_cond': '19_24_52', 'Test_B': '17_17_33', 'Test_B_1wk' : '17_57_20'},    
    'G12' : {'TFC_cond': '15_12_38', 'Test_B': '14_28_46', 'Test_B_1wk' : '15_31_10'},    
    'G13' : {'TFC_cond': '17_13_06', 'Test_B': '15_53_23', 'Test_B_1wk' : '16_50_10'},    
    'G14' : {'TFC_cond': '16_00_53', 'Test_B': '15_05_26', 'Test_B_1wk' : '15_10_50'},    
    'G15' : {'TFC_cond': '18_10_56', 'Test_B': '16_19_13'},    
    'G16' : {'TFC_cond': '16_06_44', 'Test_B': '14_47_09', 'Test_B_1wk' : '14_08_02'},    
    'G17' : {'TFC_cond': '18_26_30', 'Test_B': '16_14_47', 'Test_B_1wk' : '16_06_22'},    
    'G18' : {'TFC_cond': '15_29_25', 'Test_B': '15_25_42', 'Test_B_1wk' : '13_49_23'},    
    'G19' : {'TFC_cond': '17_35_25', 'Test_B': '16_44_35', 'Test_B_1wk' : '15_18_44'},    
    'G20' : {'TFC_cond': '14_37_19', 'Test_B': '13_21_06', 'Test_B_1wk' : '15_06_19'},    
    'G21' : {'TFC_cond': '17_39_30', 'Test_B': '14_35_02', 'Test_B_1wk' : '16_30_54'}   
}

#########################
##### END MOUSE DATA ####
#########################

# Save paths for S_spikes to speed up subsequent runs
TFC_cond_savepath = os.path.join(NPY_SAVE_PATH, 'TFC_cond')
Test_B_savepath = os.path.join(NPY_SAVE_PATH, 'Test_B')
Test_B_1wk_savepath = os.path.join(NPY_SAVE_PATH, 'Test_B_1wk')
os.makedirs(TFC_cond_savepath, exist_ok=True)
os.makedirs(Test_B_savepath, exist_ok=True)
os.makedirs(Test_B_1wk_savepath, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Build up paths
dpath_TFC_cond_crossreg_file = dict()
dpath_TFC_B_B_1wk_crossreg_file = dict()
for mouse in mouse_list:
    dpath_TFC_cond_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_TFC_cond_day[mouse])
    dpath_TFC_cond[mouse] = os.path.join(dpath_TFC_cond_day[mouse], dpath_TFC_cond[mouse])
    dpath_TFC_cond_LT1[mouse] = os.path.join(dpath_TFC_cond_day[mouse], dpath_TFC_cond_LT1[mouse])
    dpath_TFC_cond_LT2[mouse] = os.path.join(dpath_TFC_cond_day[mouse], dpath_TFC_cond_LT2[mouse])
    dpath_TFC_cond_crossreg_file[mouse] = os.path.join(dpath_TFC_cond_day[mouse], crossreg_file_TFC_cond)

    dpath_Test_B_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_Test_B_day[mouse])
    dpath_Test_B[mouse] = os.path.join(dpath_Test_B_day[mouse], dpath_Test_B[mouse])
    dpath_Test_B_1wk_day[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], dpath_Test_B_1wk_day[mouse])
    dpath_Test_B_1wk[mouse] = os.path.join(dpath_Test_B_1wk_day[mouse], dpath_Test_B_1wk[mouse])
    dpath_TFC_B_B_1wk_crossreg_file[mouse] = os.path.join(mouse_path_prefix[mouse], dpath_mouse[mouse], crossreg_file_4)

TFC_cond = dict()
TFC_cond_LT1 = dict()
TFC_cond_LT2 = dict()
TFC_cond_crossreg = dict()

Test_B = dict()
Test_B_1wk = dict()
TFC_B_B_1wk_crossreg = dict()

# Build up mappings we want to analyze & plot for TFC_cond
mapping_FULL = 'full'
mapping_LT1_LT2_TFC_cond = 'LT1+LT2+TFC_cond'
mapping_TFC_cond = 'TFC_cond'
mapping_LT2_TFC_cond = 'LT2+TFC_cond'
mapping_LT1_TFC_cond = 'LT1+TFC_cond'
mappings_all_TFC_cond = [ 
    mapping_LT1_LT2_TFC_cond, mapping_TFC_cond, mapping_LT2_TFC_cond, mapping_LT1_TFC_cond, mapping_FULL
]

# ... for Test B [1wk]
mapping_TFC_cond_Test_B_Test_B_1wk = 'TFC_cond+Test_B+Test_B_1wk'
mapping_TFC_cond_Test_B = 'TFC_cond+Test_B'
mapping_TFC_cond_Test_B_1wk = 'TFC_cond+Test_B_1wk'
mapping_Test_B_Test_B_1wk = 'Test_B+Test_B_1wk'
mappings_all_Test_B = [
    mapping_TFC_cond_Test_B_Test_B_1wk, mapping_TFC_cond_Test_B, mapping_Test_B_Test_B_1wk, mapping_FULL
]
mappings_all_Test_B_G15 = [
    mapping_TFC_cond_Test_B, mapping_FULL
]
mappings_all_Test_B_1wk = [
    mapping_TFC_cond_Test_B_Test_B_1wk, mapping_TFC_cond_Test_B_1wk, mapping_Test_B_Test_B_1wk, mapping_FULL
]
mappings_all_Test_B_1wk_G07 = [
    mapping_TFC_cond_Test_B_1wk, mapping_FULL
]

# TFC cond
tone_sp_rates_mapping = dict()
shock_sp_rates_mapping = dict()
post_shock_sp_rates_mapping = dict()
tone_activity_mapping = dict()
shock_activity_mapping = dict()
post_shock_activity_mapping = dict()

TFC_cond_binned_sp_rates_mapping = dict()
TFC_cond_binned_activity_mapping = dict()
TFC_cond_binned_activity_mapping_engram = dict()
TFC_cond_ROI_mappings = dict()
TFC_cond_ROI_mappings_peakval = dict()

for mapping in mappings_all_TFC_cond:
    tone_sp_rates_mapping[mapping] = dict()
    shock_sp_rates_mapping[mapping] = dict()
    post_shock_sp_rates_mapping[mapping] = dict()
    tone_activity_mapping[mapping] = dict()
    shock_activity_mapping[mapping] = dict()
    post_shock_activity_mapping[mapping] = dict()

    TFC_cond_binned_sp_rates_mapping[mapping] = dict()
    TFC_cond_binned_activity_mapping[mapping] = dict()
    TFC_cond_binned_activity_mapping_engram[mapping] = dict()    
    TFC_cond_ROI_mappings[mapping] = dict()
    TFC_cond_ROI_mappings_peakval[mapping] = dict()

# Test B [1wk]
Test_B_tone_sp_rates_mapping = dict()
Test_B_post_tone_sp_rates_mapping = dict()
Test_B_tone_post_tone_sp_rates_mapping = dict()
Test_B_tone_activity_mapping = dict()
Test_B_post_tone_activity_mapping = dict()
Test_B_tone_post_tone_activity_mapping = dict()

Test_B_1wk_tone_sp_rates_mapping = dict()
Test_B_1wk_post_tone_sp_rates_mapping = dict()
Test_B_1wk_tone_post_tone_sp_rates_mapping = dict()
Test_B_1wk_tone_activity_mapping = dict()
Test_B_1wk_post_tone_activity_mapping = dict()
Test_B_1wk_tone_post_tone_activity_mapping = dict()

Test_B_binned_sp_rates_mapping = dict()
Test_B_binned_activity_mapping = dict()
Test_B_binned_activity_mapping_engram = dict()
Test_B_ROI_mappings = dict()
Test_B_ROI_mappings_peakval = dict()

Test_B_1wk_binned_sp_rates_mapping = dict()
Test_B_1wk_binned_activity_mapping = dict()
Test_B_1wk_binned_activity_mapping_engram = dict()
Test_B_1wk_ROI_mappings = dict()
Test_B_1wk_ROI_mappings_peakval = dict()

for mapping in mappings_all_Test_B:
    Test_B_tone_sp_rates_mapping[mapping] = dict()
    Test_B_post_tone_sp_rates_mapping[mapping] = dict()
    Test_B_tone_post_tone_sp_rates_mapping[mapping] = dict()
    Test_B_tone_activity_mapping[mapping] = dict()
    Test_B_post_tone_activity_mapping[mapping] = dict()
    Test_B_tone_post_tone_activity_mapping[mapping] = dict()

    Test_B_binned_sp_rates_mapping[mapping] = dict()
    Test_B_binned_activity_mapping[mapping] = dict()
    Test_B_binned_activity_mapping_engram[mapping] = dict()
    Test_B_ROI_mappings[mapping] = dict()
    Test_B_ROI_mappings_peakval[mapping] = dict()

for mapping in mappings_all_Test_B_1wk:
    Test_B_1wk_tone_sp_rates_mapping[mapping] = dict()
    Test_B_1wk_post_tone_sp_rates_mapping[mapping] = dict()
    Test_B_1wk_tone_post_tone_sp_rates_mapping[mapping] = dict()
    Test_B_1wk_tone_activity_mapping[mapping] = dict()
    Test_B_1wk_post_tone_activity_mapping[mapping] = dict()
    Test_B_1wk_tone_post_tone_activity_mapping[mapping] = dict()

    Test_B_1wk_binned_sp_rates_mapping[mapping] = dict()
    Test_B_1wk_binned_activity_mapping[mapping] = dict()
    Test_B_1wk_binned_activity_mapping_engram[mapping] = dict()
    Test_B_1wk_ROI_mappings[mapping] = dict()
    Test_B_1wk_ROI_mappings_peakval[mapping] = dict()

# Build up mappings for LT1, LT2 within TFC_cond day
mapping_LT1 = 'LT1'
mapping_LT2 = 'LT2'
mapping_LT1_LT2 = 'LT1+LT2'
mappings_all_LT1 = [
    mapping_LT1, mapping_LT1_TFC_cond, mapping_LT1_LT2_TFC_cond, mapping_LT1_LT2, mapping_FULL
]
mappings_all_LT2 = [
    mapping_LT2, mapping_LT2_TFC_cond, mapping_LT1_LT2_TFC_cond, mapping_LT1_LT2, mapping_FULL
]
LT1_exp_sp_rates_mapping = dict()
LT2_exp_sp_rates_mapping = dict()
LT1_exp_activity_mapping = dict()
LT2_exp_activity_mapping = dict()
LT1_ROI_mappings = dict()
LT2_ROI_mappings = dict()
LT1_ROI_mappings_peakval = dict()
LT2_ROI_mappings_peakval = dict()
for mapping in mappings_all_LT1:
    LT1_exp_sp_rates_mapping[mapping] = dict()
    LT1_exp_activity_mapping[mapping] = dict()    
    LT1_ROI_mappings[mapping] = dict()
    LT1_ROI_mappings_peakval[mapping] = dict()
for mapping in mappings_all_LT2:
    LT2_exp_sp_rates_mapping[mapping] = dict()
    LT2_exp_activity_mapping[mapping] = dict()
    LT2_ROI_mappings[mapping] = dict()
    LT2_ROI_mappings_peakval[mapping] = dict()

#
# Main analysis loop
#
for mouse in mouse_list:
    msg_start('*** Processing mouse '+mouse+'\n')

    #########################
    # Define various crossreg
    #########################

    # For LT1-LT2-TFC_cond during TFC_cond day ('default')
    TFC_cond_crossreg[mouse] = CrossRegMapping(mouse, dpath_TFC_cond_crossreg_file[mouse], crossreg_type=1, savepath=TFC_cond_savepath)

    #TFC_B_B_1wk_crossreg[mouse] = CrossRegMapping(mouse, dpath_TFC_B_B_1wk_crossreg_file[mouse], crossreg_type=4, \
    #    groups_mappings=TFC_B_B_1wk_crossreg_groups[mouse], savepath=Test_B_savepath)
    TFC_B_B_1wk_crossreg[mouse] = CrossRegMapping(mouse, dpath_TFC_B_B_1wk_crossreg_file[mouse], crossreg_type=4, \
        groups_mappings=TFC_B_B_1wk_crossreg_groups[mouse], savepath=TFC_cond_savepath)
    
    ###########
    # TFC cond
    ###########
    if mouse in period_overrides:
        period_override = period_overrides[mouse]
    else:
        period_override = []

    TFC_cond[mouse] = TraceFearCondSession(mouse, dpath_TFC_cond[mouse], session_bounds=TFC_cond_exp_frames[mouse], period_override=period_override, \
        plot_sample_cell=plot_sample_cell, data_dir=data_dir, crossreg=TFC_cond_crossreg[mouse], savepath=TFC_cond_savepath, behaviour_type=BEHAVIOUR_TYPE)

    if mouse in mice_skip_LT:
        wanted_behaviour = None
    else:
        wanted_behaviour = BEHAVIOUR_TYPE
    TFC_cond_LT1[mouse] = LinearTrackSession(mouse, dpath_TFC_cond_LT1[mouse], session_bounds=LT1_exp_frames[mouse], \
        LT_type='LT1', data_dir=data_dir, crossreg=TFC_cond_crossreg[mouse], savepath=TFC_cond_savepath, behaviour_type=wanted_behaviour)
    TFC_cond_LT2[mouse] = LinearTrackSession(mouse, dpath_TFC_cond_LT2[mouse], session_bounds=LT2_exp_frames[mouse], \
        LT_type='LT2', data_dir=data_dir, crossreg=TFC_cond_crossreg[mouse], savepath=TFC_cond_savepath, behaviour_type=wanted_behaviour)

    TFC = TFC_cond[mouse]
    LT1 = TFC_cond_LT1[mouse]
    LT2 = TFC_cond_LT2[mouse]

    for mapping in mappings_all_TFC_cond:
        TFC.process_avg_sp_rates_mapping(mapping)
        TFC.process_avg_sp_rates_mapping(mapping, with_peakval=True)
        tone_sp_rates_mapping[mapping][mouse] = TFC.tone_sp_rates_mapping[mapping]
        shock_sp_rates_mapping[mapping][mouse] = TFC.shock_sp_rates_mapping[mapping]
        post_shock_sp_rates_mapping[mapping][mouse] = TFC.post_shock_sp_rates_mapping[mapping]
        tone_activity_mapping[mapping][mouse] = TFC.tone_activity_mapping[mapping]
        shock_activity_mapping[mapping][mouse] = TFC.shock_activity_mapping[mapping]
        post_shock_activity_mapping[mapping][mouse] = TFC.post_shock_activity_mapping[mapping]
        TFC_cond_binned_sp_rates_mapping[mapping][mouse] = TFC.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
        TFC_cond_binned_activity_mapping[mapping][mouse] = TFC.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, with_peakval=True)
        TFC_cond_binned_activity_mapping_engram[mapping][mouse] = TFC.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, with_peakval=True, with_engram=True)

        if plot_ROIs:
            TFC.get_A_matrix()
            TFC_cond_ROI_mappings[mapping][mouse] = TFC.get_ROI_mapping(mapping, want_peakval=False)
            TFC_cond_ROI_mappings_peakval[mapping][mouse] = TFC.get_ROI_mapping(mapping, want_peakval=True)

    for mapping in mappings_all_LT1:
        LT1.process_avg_sp_rates_mapping(mapping)
        LT1.process_avg_sp_rates_mapping(mapping, with_peakval=True)
        LT1_exp_sp_rates_mapping[mapping][mouse] = LT1.exp_sp_rates_mapping[mapping]
        LT1_exp_activity_mapping[mapping][mouse] = LT1.exp_activity_mapping[mapping]

        if plot_ROIs:
            LT1.get_A_matrix()
            LT1_ROI_mappings[mapping][mouse] = LT1.get_ROI_mapping(mapping, want_peakval=False)
            LT1_ROI_mappings_peakval[mapping][mouse] = LT1.get_ROI_mapping(mapping, want_peakval=True)

    for mapping in mappings_all_LT2:
        LT2.process_avg_sp_rates_mapping(mapping)
        LT2.process_avg_sp_rates_mapping(mapping, with_peakval=True)
        LT2_exp_sp_rates_mapping[mapping][mouse] = LT2.exp_sp_rates_mapping[mapping]
        LT2_exp_activity_mapping[mapping][mouse] = LT2.exp_activity_mapping[mapping]

        if plot_ROIs:
            LT2.get_A_matrix()
            LT2_ROI_mappings[mapping][mouse] = LT2.get_ROI_mapping(mapping, want_peakval=False)
            LT2_ROI_mappings_peakval[mapping][mouse] = LT2.get_ROI_mapping(mapping, want_peakval=True)

    #########
    # Test B
    #########

    if mouse not in ['G07']: # Because Test B wasn't recorded for this mouse... refine to perhaps include for Test B 1wk analyses...

        Test_B[mouse] = TestBSession(mouse, dpath_Test_B[mouse], session_bounds=Test_B_exp_frames[mouse], \
            plot_sample_cell=plot_sample_cell, crossreg=TFC_B_B_1wk_crossreg[mouse], savepath=Test_B_savepath, \
                session_group=TFC_B_B_1wk_crossreg[mouse].mappings_labels['Test_B'], behaviour_type=BEHAVIOUR_TYPE)
        
        B = Test_B[mouse]

        if mouse == 'G15':
            mappings_all_list = mappings_all_Test_B_G15
        else:
            mappings_all_list = mappings_all_Test_B

        for mapping in mappings_all_list:
            B.process_avg_sp_rates_mapping(mapping)
            B.process_avg_sp_rates_mapping(mapping, with_peakval=True)
            Test_B_tone_sp_rates_mapping[mapping][mouse] = B.tone_sp_rates_mapping[mapping]
            Test_B_post_tone_sp_rates_mapping[mapping][mouse] = B.post_tone_sp_rates_mapping[mapping]
            Test_B_tone_post_tone_sp_rates_mapping[mapping][mouse] = B.tone_post_tone_sp_rates_mapping[mapping]
            Test_B_tone_activity_mapping[mapping][mouse] = B.tone_activity_mapping[mapping]
            Test_B_post_tone_activity_mapping[mapping][mouse] = B.post_tone_activity_mapping[mapping]
            Test_B_tone_post_tone_activity_mapping[mapping][mouse] = B.tone_post_tone_activity_mapping[mapping]
            Test_B_binned_sp_rates_mapping[mapping][mouse] = B.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
            Test_B_binned_activity_mapping[mapping][mouse] = B.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, with_peakval=True)
            Test_B_binned_activity_mapping_engram[mapping][mouse] = B.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, with_peakval=True, with_engram=True)

            if plot_ROIs:
                B.get_A_matrix()
                Test_B_ROI_mappings[mapping][mouse] = B.get_ROI_mapping(mapping, want_peakval=False)
                Test_B_ROI_mappings_peakval[mapping][mouse] = B.get_ROI_mapping(mapping, want_peakval=True)

    ############
    # Test B 1wk
    ############

    if mouse not in ['G15']:

        Test_B_1wk[mouse] = TestBSession(mouse, dpath_Test_B_1wk[mouse], session_bounds=Test_B_1wk_exp_frames[mouse], \
            plot_sample_cell=plot_sample_cell, crossreg=TFC_B_B_1wk_crossreg[mouse], savepath=Test_B_1wk_savepath, \
                session_group=TFC_B_B_1wk_crossreg[mouse].mappings_labels['Test_B_1wk'], is_1wk=True, behaviour_type=BEHAVIOUR_TYPE)
        
        B_1wk = Test_B_1wk[mouse]

        if mouse == 'G07':
            mappings_all_list = mappings_all_Test_B_1wk_G07
        else:
            mappings_all_list = mappings_all_Test_B_1wk

        for mapping in mappings_all_list:
            B_1wk.process_avg_sp_rates_mapping(mapping)
            B_1wk.process_avg_sp_rates_mapping(mapping, with_peakval=True)
            Test_B_1wk_tone_sp_rates_mapping[mapping][mouse] = B_1wk.tone_sp_rates_mapping[mapping]
            Test_B_1wk_post_tone_sp_rates_mapping[mapping][mouse] = B_1wk.post_tone_sp_rates_mapping[mapping]
            Test_B_1wk_tone_post_tone_sp_rates_mapping[mapping][mouse] = B_1wk.tone_post_tone_sp_rates_mapping[mapping]
            Test_B_1wk_tone_activity_mapping[mapping][mouse] = B_1wk.tone_activity_mapping[mapping]
            Test_B_1wk_post_tone_activity_mapping[mapping][mouse] = B_1wk.post_tone_activity_mapping[mapping]
            Test_B_1wk_tone_post_tone_activity_mapping[mapping][mouse] = B_1wk.tone_post_tone_activity_mapping[mapping]
            Test_B_1wk_binned_sp_rates_mapping[mapping][mouse] = B_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH)
            Test_B_1wk_binned_activity_mapping[mapping][mouse] = B_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, with_peakval=True)
            Test_B_1wk_binned_activity_mapping_engram[mapping][mouse] = B_1wk.process_binned_sp_rates_mapping(mapping, BIN_WIDTH, with_peakval=True, with_engram=True)

            if plot_ROIs:
                B_1wk.get_A_matrix()
                Test_B_1wk_ROI_mappings[mapping][mouse] = B_1wk.get_ROI_mapping(mapping, want_peakval=False)
                Test_B_1wk_ROI_mappings_peakval[mapping][mouse] = B_1wk.get_ROI_mapping(mapping, want_peakval=True)

    msg_end()

if plot_sp_rates and not DEVEL_SWITCH:
    msg_start('*** Generating TFC_cond plots')
    for mapping in mappings_all_TFC_cond:
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, tone_sp_rates_mapping[mapping], 'TFC_cond', 'Tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, shock_sp_rates_mapping[mapping], 'TFC_cond', 'Shock '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, post_shock_sp_rates_mapping[mapping], 'TFC_cond', 'Post-shock '+mapping)
    msg_end()

    msg_start('*** Generating Test_B plots')
    for mapping in mappings_all_Test_B:
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_sp_rates_mapping[mapping], 'Test_B', 'Tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_post_tone_sp_rates_mapping[mapping], 'Test_B', 'Post-tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_post_tone_sp_rates_mapping[mapping], 'Test_B', 'Tones+Post-tones '+mapping)

    msg_start('*** Generating Test_B_1wk plots')
    for mapping in mappings_all_Test_B_1wk:
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_sp_rates_mapping[mapping], 'Test_B_1wk', 'Tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_post_tone_sp_rates_mapping[mapping], 'Test_B_1wk', 'Post-tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_post_tone_sp_rates_mapping[mapping], 'Test_B_1wk', 'Tones+Post-tones '+mapping)
    msg_end()

    msg_start('*** Generating LT1 plots')
    for mapping in mappings_all_LT1:
        plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT1_exp_sp_rates_mapping[mapping], 'LT1 track '+mapping)
    msg_end()

    msg_start('*** Generating LT2 plots')
    for mapping in mappings_all_LT2:
        plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT2_exp_sp_rates_mapping[mapping], 'LT2 track '+mapping)
    msg_end()

    ### Now activities (with_peakval)

    msg_start('*** Generating TFC_cond plots (activities)')
    for mapping in mappings_all_TFC_cond:
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, tone_activity_mapping[mapping], 'TFC_cond-activity', 'Tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, shock_activity_mapping[mapping], 'TFC_cond-activity', 'Shock '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, post_shock_activity_mapping[mapping], 'TFC_cond-activity', 'Post-shock '+mapping)
    msg_end()

    msg_start('*** Generating Test_B plots (activities)')
    for mapping in mappings_all_Test_B:
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_activity_mapping[mapping], 'Test_B-activity', 'Tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_post_tone_activity_mapping[mapping], 'Test_B-activity', 'Post-tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_tone_post_tone_activity_mapping[mapping], 'Test_B-activity', 'Tones+Post-tones '+mapping)

    msg_start('*** Generating Test_B_1wk plots (activities)')
    for mapping in mappings_all_Test_B_1wk:
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_activity_mapping[mapping], 'Test_B_1wk-activity', 'Tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_post_tone_activity_mapping[mapping], 'Test_B_1wk-activity', 'Post-tones '+mapping)
        plot_session_sp_rates(PLOTS_DIR, mouse_groups, Test_B_1wk_tone_post_tone_activity_mapping[mapping], 'Test_B_1wk-activity', 'Tones+Post-tones '+mapping)
    msg_end()

    msg_start('*** Generating LT1 plots (activities)')
    for mapping in mappings_all_LT1:
        plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT1_exp_activity_mapping[mapping], 'LT1 track '+mapping, with_peakval=True)
    msg_end()

    msg_start('*** Generating LT2 plots (activities)')
    for mapping in mappings_all_LT2:
        plot_LT_sp_rates(PLOTS_DIR, mouse_groups, LT2_exp_activity_mapping[mapping], 'LT2 track '+mapping, with_peakval=True)
    msg_end()

#if plot_binned_sp_rates and not DEVEL_SWITCH:
if plot_binned_sp_rates:
    msg_start('*** Generating binned spiking TFC_cond plots')
    for mapping in mappings_all_TFC_cond:
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_sp_rates_mapping[mapping], mapping, TFC_cond, 'TFC_cond', BIN_WIDTH)
    msg_end()

    msg_start('*** Generating binned spiking Test_B plots')
    for mapping in mappings_all_Test_B:
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_sp_rates_mapping[mapping], mapping, Test_B, 'Test_B', BIN_WIDTH)
    msg_end()

    msg_start('*** Generating binned spiking Test_B_1wk plots')
    for mapping in mappings_all_Test_B_1wk:
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_sp_rates_mapping[mapping], mapping, Test_B_1wk, 'Test_B_1wk', BIN_WIDTH)
    msg_end()

    ### Now activities (with_peakval)

    msg_start('*** Generating binned activities TFC_cond plots')
    for mapping in mappings_all_TFC_cond:
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_activity_mapping[mapping], mapping, TFC_cond, 'TFC_cond-activity', BIN_WIDTH)
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_activity_mapping[mapping], mapping, TFC_cond, 'TFC_cond-activity', BIN_WIDTH,
                                     plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'))
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, TFC_cond_binned_activity_mapping_engram[mapping], mapping, TFC_cond, 'TFC_cond-activity', BIN_WIDTH,
                                     plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'), suffix='_engram')

    msg_end()

    msg_start('*** Generating binned activities Test_B plots')
    for mapping in mappings_all_Test_B:
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_activity_mapping[mapping], mapping, Test_B, 'Test_B-activity', BIN_WIDTH)
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_activity_mapping[mapping], mapping, Test_B, 'Test_B-activity', BIN_WIDTH,
                                     plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'))
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_binned_activity_mapping_engram[mapping], mapping, Test_B, 'Test_B-activity', BIN_WIDTH,
                                     plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'), suffix='_engram')

    msg_end()

    msg_start('*** Generating binned activities Test_B_1wk plots')
    for mapping in mappings_all_Test_B_1wk:
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_activity_mapping[mapping], mapping, Test_B_1wk, 'Test_B_1wk-activity', BIN_WIDTH)
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_activity_mapping[mapping], mapping, Test_B_1wk, 'Test_B_1wk-activity', BIN_WIDTH,
                                     plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'))        
        plot_binned_sp_rates_mapping(PLOTS_DIR, mouse_groups, Test_B_1wk_binned_activity_mapping_engram[mapping], mapping, Test_B_1wk, 'Test_B_1wk-activity', BIN_WIDTH,
                                     plot_bars=False, paper_dir=get_paper_dir(PAPER_DIR, 'fig2'), suffix='_engram')        

    msg_end()

if plot_ROIs and not DEVEL_SWITCH:
    msg_start('*** Generating ROI mapping plots for TFC_cond')
    plot_ROI_mappings(PLOTS_DIR, mouse_groups, TFC_cond_ROI_mappings, mappings_all_TFC_cond, LT1_ROI_mappings, mappings_all_LT1, LT2_ROI_mappings, mappings_all_LT2)
    plot_ROI_mappings(PLOTS_DIR, mouse_groups, TFC_cond_ROI_mappings_peakval, mappings_all_TFC_cond, \
        LT1_ROI_mappings_peakval, mappings_all_LT1, LT2_ROI_mappings_peakval, mappings_all_LT2, \
            want_peakval=True)
    msg_end()

if plot_proportional_activities and not DEVEL_SWITCH:
    msg_start('*** Generating proportional activities plots')
    proportional_activities(PLOTS_DIR, mice_per_group, TFC_cond, TFC_cond_LT1, TFC_cond_LT2)
    proportional_activities_TFC_B_B_1wk(PLOTS_DIR, mice_per_group, TFC_cond, Test_B, Test_B_1wk)

    proportional_activities_donut(PLOTS_DIR, mouse_groups, TFC_cond, TFC_cond_LT1, TFC_cond_LT2, ['TFC_cond','TFC_cond_LT1','TFC_cond_LT2'], crossreg_type='TFC_cond', \
        crossreg_to_use=TFC_cond_crossreg)
    proportional_activities_donut(PLOTS_DIR, mouse_groups, TFC_cond, Test_B, Test_B_1wk, ['TFC_cond', 'Test_B', 'Test_B_1wk'], crossreg_type='TFC_B_B_1wk', \
        crossreg_to_use=TFC_B_B_1wk_crossreg)
    msg_end()

if plot_LT_firing_rate_changes and not DEVEL_SWITCH:
    msg_start('*** Generating LT1->LT2 firing rate changes plots')
    LT_firing_rate_changes(PLOTS_DIR, mice_per_group, TFC_cond_LT1, TFC_cond_LT2)
    LT_firing_rate_changes(PLOTS_DIR, mice_per_group, TFC_cond_LT1, TFC_cond_LT2, use_peakval=True)
    msg_end()

    msg_start('*** Generating TFC_cond, Test_B, Test_B_1wk firing rate changes plots')
    plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B, mapping_TFC_cond_Test_B_Test_B_1wk)
    plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B, mapping_TFC_cond_Test_B_Test_B_1wk, use_peakval=True)
    plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk)
    plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, TFC_cond, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, use_peakval=True)
    plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk)
    plot_firing_rate_changes(PLOTS_DIR, mice_per_group_Test_B_B_1wk, TFC_B_B_1wk_crossreg, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, use_peakval=True)
    msg_end()

if want_sample_traces_paper and not DEVEL_SWITCH:
    msg_start('*** Generating sample traces for paper (TFC_cond) (activities)')
    selections = {
        'hM3D' : [(582, 5149), (130, 23781), (735, 6450)],
        'hM4D' : [(154, 23413), (208, 6672), (271, 14012)],
        'mCherry' : [(319, 18272), (230, 7538), (370, 4138)]
    }
    paper_dir = get_paper_dir(PAPER_DIR, 'fig2')
    plot_sample_traces(PLOTS_DIR, {'hM3D':'G10', 'hM4D':'G14', 'mCherry':'G17'}, TFC_cond, paper_dir=paper_dir, selections=selections, len_trace=1200)
    msg_end()

if plot_PSTH and not DEVEL_SWITCH:
    msg_start('*** Generating tone PSTH')
    for mapping in [mapping_FULL]: #mappings_all_TFC_cond: #
        nonzero_active_cells = dict()
        nonzero_suppr_cells = dict()
        frac_tots_active = dict()
        frac_tots_suppr = dict()
        trapz_cells_active = dict()
        trapz_cells_suppr = dict()
        PSTH_cells_active = dict()
        PSTH_cells_suppr = dict()
        max_per_cell_active = dict()
        max_per_cell_suppr = dict()

        tot_cells, frac_tots, sig_cells, group_PSTH, group_percentiles, group_PSTH_vel = process_PSTH_shuffle(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim='shock', num_shuffles=100)

        # Old, obsolete way vvv
        '''
        for fl in [20]:
            for stim in ['tone', 'shock']:
                for ba in [True]:
                    nonzero_active_cells[stim], frac_tots_active[stim], trapz_cells_active[stim], PSTH_cells_active[stim], max_per_cell_active[stim] = \
                        process_PSTH_simple(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim, frames_lookaround=fl, normalize=True, binary_activity=ba, binary_thresh=80)
                    nonzero_suppr_cells[stim], frac_tots_suppr[stim], trapz_cells_suppr[stim], PSTH_cells_suppr[stim], max_per_cell_suppr[stim] = \
                        process_PSTH_simple(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim, frames_lookaround=fl, normalize=True, binary_activity=ba, binary_thresh=-80, binary_flip=True)
                    nonzero_active_cells[stim], frac_tots_active[stim], trapz_cells_active[stim], PSTH_cells_active[stim], max_per_cell_active[stim] = \
                        process_PSTH_simple(PLOTS_DIR, mice_per_group, TFC_cond_crossreg, TFC_cond, mapping, stim, frames_lookaround=fl, normalize=False, binary_activity=ba, binary_thresh=80)
                    #plot_PSTH_overlay(PLOTS_DIR, TFC_cond, nonzero_active_cells, nonzero_suppr_cells, stim, mapping)
                    plot_PSTH_activities(PLOTS_DIR, frac_tots_active[stim], stim, 'Active', mapping)
                    plot_PSTH_activities(PLOTS_DIR, frac_tots_suppr[stim], stim, 'Suppressed', mapping)
                    plot_PSTH_intensities(PLOTS_DIR, trapz_cells_active[stim], stim, mapping)
                    plot_PSTH_peaks(PLOTS_DIR, PSTH_cells_active[stim], max_per_cell_active[stim], stim, mapping)
        '''
    msg_end()

if plot_pf_and_loc and not DEVEL_SWITCH:
    if BEHAVIOUR_TYPE == 'movement':
        pcells_mice = dict()
        plot_location_map(PLOTS_DIR, mice_per_group, TFC_cond, 'TFC_cond')
        plot_fluorescence_map(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond', bin_width=34, random_width=4, want_3D=True, pcells_mice=pcells_mice, \
            max_fields=45, only_fm_pcells=True, print_pcell_maps=True) 
        plot_pf_analyses(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond')
        plot_pf_analyses(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond', crossreg=TFC_B_B_1wk_crossreg, mapping=mapping)

        pcells_mice_B = dict()
        plot_location_map(PLOTS_DIR, mice_per_group, Test_B, 'Test_B')
        plot_fluorescence_map(PLOTS_DIR, Test_B, mouse_groups, 'Test_B', bin_width=34, random_width=4, want_3D=True, pcells_mice=pcells_mice_B, \
            max_fields=45, only_fm_pcells=True, print_pcell_maps=True)
        plot_pf_analyses(PLOTS_DIR, Test_B, mouse_groups, 'Test_B')
        plot_pf_analyses(PLOTS_DIR, Test_B, mouse_groups, 'Test_B', crossreg=TFC_B_B_1wk_crossreg, mapping=mapping)

        pcells_mice_B_1wk = dict()
        plot_location_map(PLOTS_DIR, mice_per_group, Test_B_1wk, 'Test_B_1wk')
        plot_fluorescence_map(PLOTS_DIR, Test_B_1wk, mouse_groups, 'Test_B_1wk', bin_width=34, random_width=4, want_3D=True, pcells_mice=pcells_mice_B_1wk, \
            max_fields=45, only_fm_pcells=True, print_pcell_maps=True)
        plot_pf_analyses(PLOTS_DIR, Test_B_1wk, mouse_groups, 'Test_B_1wk')
        plot_pf_analyses(PLOTS_DIR, Test_B_1wk, mouse_groups, 'Test_B_1wk', crossreg=TFC_B_B_1wk_crossreg, mapping=mapping)
        
        # Finally, across sessions (i.e., within groups). Sessions are handled in the function.
        plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_B, Test_B_1wk, mouse_groups, crossreg=None, mapping=None)
        plot_pf_analyses_within_group(PLOTS_DIR, TFC_cond, Test_B, Test_B_1wk, mouse_groups, crossreg=TFC_B_B_1wk_crossreg, mapping=mapping)

if plot_population_vectors: #and not DEVEL_SWITCH:
    #plot_pop_vectors(PLOTS_DIR, TFC_cond, mouse_groups, 'TFC_cond')
    if process_for_R:
        for mouse in [m for m in mouse_groups if m not in ['G07', 'G15']]:
            process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                        binarize=False, normalize=False, normalize_full=False, spk_cutoff=2)             
            process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                        binarize=True, normalize=False, spk_cutoff=2)
            process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                        binarize=False, normalize=True, spk_cutoff=2)        
            process_mice_for_R(mouse, TFC_cond, Test_B, Test_B_1wk, mapping_TFC_cond_Test_B_Test_B_1wk, \
                        binarize=False, normalize=False, normalize_full=True, spk_cutoff=2)    

    if perform_agglomerative_clustering:
        sess_all = [TFC_cond, Test_B, Test_B_1wk]
        sess_label_all = ['TFC_cond', 'Test_B', 'Test_B_1wk']
        PV_sess = dict()
        use_silhouette = True
        saver_PV_sess= Saver(parent_path=NPY_SAVE_PATH, subdirs=['PV_sess'])
        if saver_PV_sess.check_exists('PV_sess'):
            PV_sess = saver_PV_sess.load('PV_sess')
        else:
            for Ca_act_type in ['full', 'mov', 'imm']:
                PV_sess[Ca_act_type] = dict()
                for sess, sess_label in zip(sess_all, sess_label_all):
                    PV_sess[Ca_act_type][sess_label] = dict()
                    for only_crossreg in [True, False]:
                        only_crossreg_str = get_only_crossreg_str(only_crossreg)
                        for transpose_wanted in [True, False]:
                            transpose_str = get_transpose_str(transpose_wanted)
                            PV_sess[Ca_act_type][sess_label][only_crossreg_str] = dict()
                            print('\n*** WORKING cluster_pop_vectors() for {}, only_crossreg = {}, use_silhouette = {}, transpose_wanted = {}'.format(Ca_act_type, only_crossreg, use_silhouette, transpose_wanted))
                            PV_group = cluster_pop_vectors(PLOTS_DIR, sess, sess_label, mice_per_group, transpose_wanted=transpose_wanted, auto_close=True, \
                                crossreg=TFC_B_B_1wk_crossreg, Ca_act_type=Ca_act_type, only_crossreg=only_crossreg, use_silhouette=use_silhouette, bin_width=1, \
                                sess_all=sess_all)
                            PV_sess[Ca_act_type][sess_label][only_crossreg_str][transpose_str] = PV_group
                saver_PV_sess.save(PV_sess[Ca_act_type], 'PV_sess_{}'.format(Ca_act_type))

        for only_crossreg in [True, False]:
            process_pop_vectors(PLOTS_DIR, PV_sess, only_crossreg, crossreg_str='TFC_B_B_1wk', plot_type='boxplot', want_scatter=True, auto_close=True, \
                use_silhouette=use_silhouette, frac_type_l=[1/2], force_calc=True, cohens_thresh=0.1)

    '''
        [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
            cluster_pop_vectors(PLOTS_DIR, TFC_cond, 'TFC_cond', mice_per_group, transpose_wanted=False, auto_close=True, bin_width=2, \
                spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)
        [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
            cluster_pop_vectors(PLOTS_DIR, Test_B, 'Test_B', mice_per_group, transpose_wanted=False, auto_close=True, bin_width=2, \
                spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)
        [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
            cluster_pop_vectors(PLOTS_DIR, Test_B_1wk, 'Test_B_1wk', mice_per_group, transpose_wanted=False, auto_close=True, bin_width=2, \
                spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)
    '''
    PV_group = cluster_pop_vectors(PLOTS_DIR, TFC_cond, 'TFC_cond', mice_per_group, \
            transpose_wanted=True, auto_close=True, bin_width=1, spk_cutoff=2, crossreg=TFC_B_B_1wk_crossreg)        

if plot_binned_activities:
    print('here')
    pass   

msg_start('*** Plotting')
plt.show()
msg_end()
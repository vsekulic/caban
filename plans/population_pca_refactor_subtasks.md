# Population PCA Refactor - Subtask Breakdown for Separate Context Runs

## How to Execute Subtasks in Separate Contexts

Each subtask below is designed to be executed in a **separate context run**. To do so:

1. Copy the "Instructions for Code Mode" block from the subtask you want to execute
2. Paste it as your prompt when you start a new Code mode session
3. The subtask includes all necessary context (file references, source code locations, expected signatures)

**Execution Order**: Subtasks must be executed in numerical order (1 → 7) because later subtasks depend on earlier ones. Subtask 4 is split into 4a and 4b, and Subtask 6 is split into 6a and 6b for better manageability.

**Verification Step**: After each subtask completes, verify the code works before proceeding to the next subtask. Run a quick test or check that the file was saved correctly.

---

## Subtask 1: PopulationPCA Base Class

**Instructions for Code Mode**:
```
Execute Subtask 1 from plans/population_pca_refactor_subtasks.md:
Create the PopulationPCA base class in SSTCa2_population.py with:
- __init__, z_normalize, select_engram_cells, apply_smoothing, fit_pca, transform
Reference: SSTCa2_debug_snippets3.py lines 465-506 (Basic PCA section)
Dependencies: None (first subtask)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:465-506)
**Dependencies**: None

---

## Subtask 2: TrialAveragedPCA Class

**Instructions for Code Mode**:
```
Execute Subtask 2 from plans/population_pca_refactor_subtasks.md:
Create TrialAveragedPCA class (extends PopulationPCA) in SSTCa2_population.py with:
- __init__, average_trials, compute_best_view, plot_3d_trajectory
Reference: SSTCa2_debug_snippets3.py lines 508-642 (Trial-Averaged PCA section)
Dependencies: Subtask 1 must be complete (PopulationPCA class exists)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:508-642)
**Dependencies**: Subtask 1

---

## Subtask 3: CrossregPCA Class - Core Methods

**Instructions for Code Mode**:
```
Execute Subtask 3 from plans/population_pca_refactor_subtasks.md:
Create CrossregPCA class (extends PopulationPCA) in SSTCa2_population.py with:
- __init__, get_crossreg_data, fit_pca_method1, fit_pca_method2
Reference: SSTCa2_debug_snippets3.py lines 664-925 (Crossreg PCA AVG) and lines 927-1323 (Crossreg PCA FULL)
Dependencies: Subtask 1 must be complete (PopulationPCA class exists)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:664-925), [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:927-1323)
**Dependencies**: Subtask 1

---

## Subtask 4a: CrossregPCA Class - Helper Methods for Visualization

**Instructions for Code Mode**:
```
Execute Subtask 4a from plans/population_pca_refactor_subtasks.md:
Add helper methods to CrossregPCA class in SSTCa2_population.py:

1. calculate_overlap(self, ax, PCs): Calculate point overlap in 3D projection
   - Project 3D points to 2D using ax.get_proj()
   - Use histogram2d to count bins with >1 point
   Reference: SSTCa2_debug_snippets3.py lines 680-695, 978-993

2. find_best_view(self, PCs): Find optimal 3D viewing angles to minimize overlap
   - Grid search over elev (0-90, step 10) and azim (0-360, step 10)
   - Return best (elev, azim) pair with minimum overlap
   Reference: SSTCa2_debug_snippets3.py lines 697-718, 995-1016

3. _compute_pca_limits(self, PCs_TFC, PCs_TestB, PCs_TestB1wk, n_components=3):
   - Compute axis limits from concatenated PCs for consistent scaling
   - Optionally compute tone/shock-only limits
   Reference: SSTCa2_debug_snippets3.py lines 837-847, 1095-1112

Dependencies: Subtask 3 must be complete (CrossregPCA core exists)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:680-718), [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:978-1021)
**Dependencies**: Subtask 3

---

## Subtask 4b: CrossregPCA Class - Public Visualization Methods

**Instructions for Code Mode**:
```
Execute Subtask 4b from plans/population_pca_refactor_subtasks.md:
Add public visualization methods to CrossregPCA class in SSTCa2_population.py:

1. plot_crossreg_trajectories(self, save_dir=None, want_engram=True, n_components=3, method=1, plot_types=None):
   - Plot trial-averaged PCA trajectories for TFC_cond/Test_B/Test_B_1wk
   - Create 3D subplot for each session (1x3 layout)
   - Mark tone periods (blue triangles) and shock periods (red x)
   - Support 'full' and 'zoom' plot types
   - Save to save_dir with appropriate filename
   Reference: SSTCa2_debug_snippets3.py lines 725-925

2. plot_2d_location_3d_pca(self, sess, PCs, trajectory_type='Time', save_path=None, only_tone_shock=True):
   - Plot 2D location (left) + 3D PCA space (right) side by side
   - Color points by time or 2D location gradient
   - Apply mouse-specific elevation/azimuth presets
   Reference: SSTCa2_debug_snippets3.py lines 1115-1228

3. plot_crossreg_full_trajectories(self, save_dir=None, want_engram=True, only_tone_shock=True, method=2):
   - Plot full time course PCA trajectories (not trial-averaged)
   - Create 3D subplot for each session (1x3 layout)
   - Mark tone onsets/offsets (blue squares) and shock onsets/offsets (red circles)
   - Support both shock/tone and post-tone/post-shock periods
   - Apply mouse-specific angle presets from mouse_elev_azim dict
   Reference: SSTCa2_debug_snippets3.py lines 1023-1323, lines 1329-1484

Dependencies: Subtask 4a must be complete (helper methods exist)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:725-925), [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:1023-1323), [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:1329-1484)
**Dependencies**: Subtask 4a

---

## Subtask 5: UMAPAnalysis Class

**Instructions for Code Mode**:
```
Execute Subtask 5 from plans/population_pca_refactor_subtasks.md:
Create UMAPAnalysis class (extends PopulationPCA) in SSTCa2_population.py with:
- __init__, fit_umap_single, fit_umap_crossreg, fit_umap_concatenated, plot_umap_results
Reference: SSTCa2_debug_snippets3.py lines 1486-1753 (UMAP sections)
Dependencies: Subtask 1 must be complete (PopulationPCA class exists)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`SSTCa2_debug_snippets3.py`](SSTCa2_debug_snippets3.py:1486-1753)
**Dependencies**: Subtask 1

---

## Subtask 6a: Pipeline Function - TrialAveragedPCA and CrossregPCA AVG

**Instructions for Code Mode**:
```
Execute Subtask 6a from plans/population_pca_refactor_subtasks.md:
Create run_population_pca_pipeline function in SSTCa2_population.py that orchestrates:
- TrialAveragedPCA pipeline
- CrossregPCA (AVG mode) pipeline

The function should:
1. Iterate over mice (excluding G07, G15)
2. For each mouse:
   a. TrialAveragedPCA:
      - .z_normalize()
      - .average_trials()
      - .fit_pca()
      - .plot_3d_trajectory()
      -> save to PCA_avg/
   b. CrossregPCA (AVG mode):
      - .get_crossreg_data()
      - .z_normalize()
      - .select_engram_cells() [optional]
      - .average_trials()
      - .fit_pca_method1() / .fit_pca_method2()
      - .plot_crossreg_trajectories()
      -> save to PCA_crossreg_avg/

Reference: plans/population_pca_refactor_plan.md lines 355-376 (Execution Flow)
Dependencies: Subtasks 1-5 must be complete (all classes exist)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`population_pca_refactor_plan.md`](plans/population_pca_refactor_plan.md:355-376)
**Dependencies**: Subtasks 1-5

---

## Subtask 6b: Pipeline Function - CrossregPCA FULL and UMAPAnalysis

**Instructions for Code Mode**:
```
Execute Subtask 6b from plans/population_pca_refactor_subtasks.md:
Extend run_population_pca_pipeline function in SSTCa2_population.py to add:
- CrossregPCA (FULL mode) pipeline
- UMAPAnalysis pipeline

The function should additionally:
3. For each mouse:
   c. CrossregPCA (FULL mode):
      - .get_crossreg_data()
      - .z_normalize()
      - .apply_smoothing() [optional]
      - .select_engram_cells() [optional]
      - .fit_pca_method2()
      - .plot_2d_location_3d_pca()
      - .plot_crossreg_trajectories()
      -> save to PCA_crossreg_full_toneshock/
   d. UMAPAnalysis:
      - .fit_umap_single()
      - .fit_umap_crossreg()
      - .fit_umap_concatenated()
      - .plot_umap_results()
      -> save to UMAP/

Reference: plans/population_pca_refactor_plan.md lines 378-394 (Execution Flow)
Dependencies: Subtask 6a must be complete (base pipeline function exists)
```

**File**: `SSTCa2_population.py`
**Reference code**: [`population_pca_refactor_plan.md`](plans/population_pca_refactor_plan.md:378-394)
**Dependencies**: Subtask 6a

---

## Subtask 7: Integration with SSTCa2_main.py

**Instructions for Code Mode**:
```
Execute Subtask 7 from plans/population_pca_refactor_subtasks.md:
Add population PCA pipeline call to SSTCa2_main.py at line ~5611
Reference: plans/population_pca_refactor_plan.md lines 218-251 (Integration section)
Dependencies: Subtask 6b must be complete (pipeline function exists)
```

**File**: `SSTCa2_main.py`
**Reference code**: [`population_pca_refactor_plan.md`](population_pca_refactor_plan.md:218-251)
**Dependencies**: Subtask 6b

---

## Dependency Graph

```
Subtask 1 (PopulationPCA)
    |
    +---> Subtask 2 (TrialAveragedPCA)
    |
    +---> Subtask 3 (CrossregPCA Core)
    |         |
    |         +---> Subtask 4a (CrossregPCA Helper Methods)
    |                   |
    |                   +---> Subtask 4b (CrossregPCA Public Visualization)
    |
    +---> Subtask 5 (UMAPAnalysis)
              |
              v
Subtask 6a (Pipeline: TrialAveragedPCA + CrossregPCA AVG)
              |
              v
Subtask 6b (Pipeline: CrossregPCA FULL + UMAPAnalysis) <---+
              |
              v
Subtask 7 (Integration)
```

## Shared Configuration Constants

These constants should be defined at module level in `SSTCa2_population.py`:

```python
# Marker settings
MARKER_SIZE = 2
MARKER_ALPHA = 0.6
TONE_MARKER_SIZE = 5
SHOCK_MARKER_SIZE = 5

# Period definitions (frames)
PERIOD_FRAMES = {
    'pre_tone': 20,
    'tone': 20,
    'post_tone': 20,
    'shock': 2,
    'post_shock': 20,
}

# Mouse-specific angle presets (elevation, azimuth) for consistent visualization
# Extracted from SSTCa2_debug_snippets3.py lines 958-973
MOUSE_ELEV_AZIM = {
    'G05' : {'Time' : (18,4), 'Location': (19,16)},
    'G06' : {'Time' : (17,16), 'Location' : (14,10)},
    'G08' : {'Time' : (36,11), 'Location' : (30,10)},
    'G09' : {'Time' : (27,12), 'Location' : (30,10)},
    'G10' : {'Time' : (27,30), 'Location' : (24,20)},
    'G11' : {'Time' : (24,34), 'Location' : (25,22)},
    'G12' : {'Time' : (30,10), 'Location' : (30,10)},
    'G13' : {'Time' : (30,10), 'Location' : (28,12)},
    'G14' : {'Time' : (25,14), 'Location' : (24,24)},
    'G16' : {'Time' : (30,10), 'Location' : (30,10)},
    'G17' : {'Time' : (24,22), 'Location' : (21,18)},
    'G18' : {'Time' : (25,23), 'Location' : (30,10)},
    'G19' : {'Time' : (13,11), 'Location' : (13,21)},
    'G20' : {'Time' : (24,-33), 'Location' : (26,-23)},
    'G21' : {'Time' : (3,25), 'Location' : (3,25)}
}

# Azim rotation offset
AZIM_ROTATION = 70
```

## Output Directory Structure

Each subtask that saves plots should create these directories if they don't exist:
```
PLOTS_DIR/
    PCA_avg/
    PCA_crossreg_avg/
    PCA_crossreg_avg_engram/
    PCA_crossreg_full_toneshock/
    PCA_crossreg_full_toneshock_engram/
    PCA_crossreg_full_posttoneshock/
    PCA_crossreg_full_posttoneshock_engram/
    UMAP/
        crossreg/
```

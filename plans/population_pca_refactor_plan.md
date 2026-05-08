# Population PCA Refactoring Plan

## Overview

Refactor PCA analysis from [`SSTCa2_debug_snippets3.py`](../SSTCa2_debug_snippets3.py:1) into an OOP framework in [`SSTCa2_population.py`](../SSTCa2_population.py:1), then integrate into [`SSTCa2_main.py`](../SSTCa2_main.py:1) after current analyses.

## Current Debug PCA Sections (to be refactored)

| Section | Lines | Description |
|---------|-------|-------------|
| Basic PCA | 465-506 | Single-session PCA with 2D/3D trajectory visualization |
| Trial-Averaged PCA | 508-642 | Trial-averaged neural activity before PCA |
| Crossreg PCA (AVG) | 664-925 | Cross-session PCA on averaged time courses |
| Crossreg PCA (FULL - Shock&Tone) | 927-1323 | Full time-course PCA with tone/shock highlighting |
| Crossreg PCA (FULL - Post-Tone/Shock) | 1325-1484 | Full time-course PCA with post-event highlighting |
| UMAP (Single Session) | 1486-1565 | UMAP on single session neural activity |
| UMAP (Crossreg) | 1567-1672 | UMAP on cross-registered sessions |
| UMAP (Concatenated) | 1675-1753 | UMAP on concatenated sessions |
| Odds Ratio | 1756-2065 | Fisher exact test for spike-stimulus association |

## OOP Class Design

### Class Hierarchy

```
PopulationAnalysis (Base)
    |
    +-- PopulationPCA
    |       - Z-normalization
    |       - Engram cell selection
    |       - Gaussian smoothing
    |       - PCA fit/transform
    |
    +-- TrialAveragedPCA
    |       - Trial averaging before PCA
    |       - Period definition (pre-tone, tone, post-tone, shock, post-shock)
    |       - Inherits PopulationPCA
    |
    +-- CrossregPCA
    |       - Cross-session PCA (Method 1: fit on encoding, Method 2: concatenate)
    |       - Engram cell selection
    |       - Full time-course trajectory plotting
    |       - 2D location + 3D PCA joint plots
    |       - Angle presets per mouse
    |       - Inherits PopulationPCA
    |
    +-- UMAPAnalysis
            - Single-session UMAP
            - Cross-session UMAP
            - Concatenated UMAP
            - Inherits PopulationPCA
```

### Class Specifications

#### 1. `PopulationPCA` (Base Class)

```python
class PopulationPCA:
    """Base class for population dimensionality analysis."""
    
    def __init__(self, session, mouse, mouse_group, config=None):
        """
        Parameters
        ----------
        session : Session object
            Must have .S (neural activity), .tone_onsets, .shock_onsets,
            .loc_X_miniscope_smooth, .loc_Y_miniscope_smooth
        mouse : str
            Mouse identifier
        mouse_group : str
            Experimental group
        config : dict, optional
            Configuration for normalization, smoothing, engram selection
        """
    
    def z_normalize(self):
        """Z-score normalize neural activity per neuron."""
        # S_normalized = nan_to_num((S - mean(axis=1)) / std(axis=1))
    
    def select_engram_cells(self, threshold=0):
        """Select engram cells based on z-scored total activity."""
        # score = zscore(sum(S, axis=1))
        # engram_cells = where(score > threshold)
    
    def apply_smoothing(self, sigma=1.5):
        """Apply Gaussian smoothing along time axis."""
        # gaussian_filter1d(S, sigma=sigma, axis=1)
    
    def fit_pca(self, n_components=3):
        """Fit PCA on normalized data."""
        # pca = PCA(n_components)
        # pca.fit(S_normalized.T)
    
    def transform(self):
        """Transform data using fitted PCA."""
        # return pca.transform(S_normalized.T)
```

#### 2. `TrialAveragedPCA`

```python
class TrialAveragedPCA(PopulationPCA):
    """PCA on trial-averaged neural activity."""
    
    def __init__(self, session, mouse, mouse_group, config=None):
        super().__init__(session, mouse, mouse_group, config)
        # Period definitions
        self.periods = {
            'pre_tone': 20,
            'tone': 20,
            'post_tone': 20,
            'shock': 2,
            'post_shock': 20
        }
    
    def average_trials(self):
        """Average neural activity across trials for each period."""
        # For each cell, average across trials for each period
        # Build S_new with averaged periods
    
    def compute_best_view(self, PCs):
        """Find viewing angles that minimize point overlap."""
        # Grid search over elev/azim angles
        # Minimize 2D histogram overlap
    
    def plot_3d_trajectory(self, PCs, save_path, filename):
        """Plot 3D PCA trajectory with period coloring."""
        # Scatter by period color (pre-tone=black, tone=blue, etc.)
        # Save to save_path/filename
```

#### 3. `CrossregPCA`

```python
class CrossregPCA(PopulationPCA):
    """Cross-session PCA for engram/memory analysis."""
    
    def __init__(self, sessions, mouse, mouse_group, mappings, config=None):
        """
        Parameters
        ----------
        sessions : dict[str, Session]
            Must contain 'TFC_cond', 'Test_B', 'Test_B_1wk'
        mappings : dict
            Cross-registered neuron indices for each session
        """
        self.sessions = sessions
        self.mappings = mappings
        # Get cross-registered S matrices
    
    def fit_pca_method1(self, n_components=3):
        """Method 1: Fit PCA on encoding session, transform others."""
        # pca.fit(S_TFC_cond.T)
        # PCs_Test_B = pca.transform(S_Test_B.T)
        # PCs_Test_B_1wk = pca.transform(S_Test_B_1wk.T)
    
    def fit_pca_method2(self, n_components=3):
        """Method 2: Concatenate all sessions, fit joint PCA."""
        # S_tot = hstack([S_TFC_cond, S_Test_B, S_Test_B_1wk])
        # pca.fit(S_tot.T)
        # Transform each session separately
    
    def plot_crossreg_trajectories(self, PCs_dict, plot_type='full', 
                                    only_tone_shock=True, trajectory_type='Time'):
        """
        Plot cross-session PCA trajectories.
        
        Parameters
        ----------
        PCs_dict : dict[str, ndarray]
            PCA components for each session
        plot_type : str
            'full' or 'zoom' (for Test_B/Test_B_1wk only)
        only_tone_shock : bool
            If True, limit axis to tone/shock region only
        trajectory_type : str
            'Time' (color by frame) or 'Location' (color by position)
        """
    
    def plot_2d_location_3d_pca(self, PCs_dict, sess_names, 
                                   trajectory_type='Time'):
        """
        Plot 2D location trajectory alongside 3D PCA trajectory.
        
        Left panel: 2D location (x, y) colored by time/position
        Right panel: 3D PCA trajectory colored by time/position
        """
```

#### 4. `UMAPAnalysis`

```python
class UMAPAnalysis(PopulationPCA):
    """UMAP dimensionality reduction for neural population activity."""
    
    def __init__(self, sessions, mouse, mouse_group, config=None):
        super().__init__(sessions, mouse, mouse_group, config)
    
    def fit_umap_single(self, n_components=2, n_neighbors=15, random_state=None):
        """Fit UMAP on single session."""
        # umap = UMAP(n_components, n_neighbors, random_state)
        # embedding = umap.fit_transform(S_normalized.T)
    
    def fit_umap_crossreg(self, n_components=2, n_neighbors=15, random_state=None):
        """Fit separate UMAP on each cross-registered session."""
        # Separate UMAP for TFC_cond, Test_B, Test_B_1wk
    
    def fit_umap_concatenated(self, n_components=2, n_neighbors=15, random_state=None):
        """Fit single UMAP on concatenated sessions."""
        # combined_S = hstack([S_TFC_cond, S_Test_B, S_Test_B_1wk])
        # umap.fit_transform(combined_S_normalized.T)
    
    def plot_umap_results(self, embedding, session_name, ax):
        """Plot UMAP embedding with tone/shock markers."""
```

## Integration with SSTCa2_main.py

### Location in Pipeline

Add at line 5611 in [`SSTCa2_main.py`](SSTCa2_main.py:5611), between the commented-out `cluster_pop_vectors` block and the `if plot_population_vector_distances:` block:

```python
    '''
        [PV_mice, labels_mice, frac_labels_mice, labels_tot_mice] = \
            cluster_pop_vectors(PLOTS_DIR, TFC_cond, 'TFC_cond', mice_per_group, ...)
        ...
    '''

# ---- Population PCA Trajectory Analysis ----
msg_start('*** Population PCA Trajectory Analysis')
from SSTCa2_population import run_population_pca_pipeline

run_population_pca_pipeline(
    TFC_cond=TFC_cond,
    Test_B=Test_B,
    Test_B_1wk=Test_B_1wk,
    mouse_groups=mouse_groups,
    TFC_B_B_1wk_crossreg=TFC_B_B_1wk_crossreg,
    mapping_TFC_cond_Test_B_Test_B_1wk=mapping_TFC_cond_Test_B_Test_B_1wk,
    PLOTS_DIR=PV_2D_PLOTS_DIR,
    auto_close=True,
)
msg_end()

if plot_population_vector_distances:
    #
    # Population vector distance calculations
    #
```

### Required Variables Available at Line 5611

At this point in the code, the following variables are available:
- `TFC_cond` - dict of TFC_cond session objects
- `Test_B` - dict of Test_B session objects
- `Test_B_1wk` - dict of Test_B_1wk session objects
- `mouse_groups` - dict mapping mouse IDs to experimental groups
- `TFC_B_B_1wk_crossreg` - cross-registered neuron mappings
- `mapping_TFC_cond_Test_B_Test_B_1wk` - crossreg mapping string
- `PV_2D_PLOTS_DIR` - plots output directory

### Pipeline Function Signature

```python
def run_population_pca_pipeline(
    TFC_cond: dict,
    Test_B: dict,
    Test_B_1wk: dict,
    mouse_groups: dict,
    TFC_B_B_1wk_crossreg: dict,
    mapping_TFC_cond_Test_B_Test_B_1wk: str,
    PLOTS_DIR: str,
    auto_close: bool = True,
    want_svg: bool = False,
    engram_thresh: float = 0.0,
    smoothing_sigma: float = 1.5,
    use_gaussian_smoothing: bool = True,
    use_engram: bool = True,
    only_tone_shock: bool = True,
    preset_angles: bool = True,
    auto_angle_adjust: bool = False,
) -> dict:
    """
    Run complete population PCA analysis pipeline.
    
    Returns
    -------
    results : dict
        Analysis results per mouse
    """
```

## Configuration Constants

From debug snippets, these should be configurable:

```python
# Mouse-specific angle presets (from SSTCa2_debug_snippets3.py:958-973)
DEFAULT_ELEV_AZIM = {
    'G05': {'Time': (18, 4), 'Location': (19, 16)},
    'G06': {'Time': (17, 16), 'Location': (14, 10)},
    # ... etc
}

# Period definitions (from SSTCa2_debug_snippets3.py:566-570)
PERIOD_FRAMES = {
    'pre_tone': 20,
    'tone': 20,
    'post_tone': 20,
    'shock': 2,
    'post_shock': 20,
}

# Marker settings
MARKER_SIZE = 2
MARKER_ALPHA = 0.6
TONE_MARKER_SIZE = 5
SHOCK_MARKER_SIZE = 5
```

## Output Directory Structure

```
PLOTS_DIR/
    PCA_avg/
        PCA_avg-{group}-{mouse}.png
    PCA_crossreg_avg/
        PCA_crossreg_avg-want_engram_{True/False}-n_components3-method{1,2}-{plot_type}-{group}-{mouse}.png
    PCA_crossreg_full_toneshock/
        trajectory-2Dloc_and_PCA_crossreg_full-method{1,2}-type-{Time,Location}-{group}-{mouse}-{session}.png
        PCA_crossreg_full-want_engram-{True/False}-only_tone_shock_{True/False}-method{1,2}-{group}-{mouse}.png
    PCA_crossreg_full_posttoneshock/
        ...
    UMAP/
        UMAP-{group}-{mouse}_iter{iter}_seed-{seed}-n_neighbors_{n}.png
        crossreg/
            UMAP-crossreg-{group}-{mouse}_iter{iter}_seed-{seed}.png
```

## Dependencies

The new module will import:
- `numpy` for array operations
- `sklearn.decomposition.PCA` for PCA
- `umap.UMAP` for UMAP
- `scipy.ndimage.gaussian_filter1d` for smoothing
- `scipy.stats` for engram cell selection (zscore, fisher_exact)
- `matplotlib.pyplot` for plotting
- `matplotlib.figure` and `mpl_toolkits.mplot3d` for 3D plots
- `os` for directory creation
- `SSTCa2_utilities` for `get_S_indeces_crossreg`, `MINISCOPE_FPS`

## Execution Flow

```
run_population_pca_pipeline()
    |
    +-- For each mouse (excluding G07, G15):
    |       |
    |       +-- TrialAveragedPCA
    |               .z_normalize()
    |               .average_trials()
    |               .fit_pca()
    |               .plot_3d_trajectory()
    |               -> save to PCA_avg/
    |
    +-- CrossregPCA (AVG mode)
    |       .get_crossreg_data()
    |       .z_normalize()
    |       .select_engram_cells() [optional]
    |       .average_trials()
    |       .fit_pca_method1() / .fit_pca_method2()
    |       .plot_crossreg_trajectories()
    |       -> save to PCA_crossreg_avg/
    |
    +-- CrossregPCA (FULL mode)
    |       .get_crossreg_data()
    |       .z_normalize()
    |       .apply_smoothing() [optional]
    |       .select_engram_cells() [optional]
    |       .fit_pca_method2()
    |       .plot_2d_location_3d_pca()
    |       .plot_crossreg_trajectories()
    |       -> save to PCA_crossreg_full_toneshock/
    |
    +-- UMAPAnalysis
            .fit_umap_single()
            .fit_umap_crossreg()
            .fit_umap_concatenated()
            .plot_umap_results()
            -> save to UMAP/
```

## Notes

1. **Excluded mice**: G07 and G15 are excluded from crossreg analysis (per debug snippets)
2. **G09 and G21**: Special handling for loc_X/Y arrays (Miniscope 19.avi recording issue)
3. **Method 1 vs Method 2**: Method 1 (fit on encoding, apply to others) is retired for FULL mode due to variance differences across sessions
4. **Engram cells**: Selected based on z-scored total activity > threshold during encoding
5. **Angle presets**: Per-mouse elevation/azim angles for consistent visualization across mice

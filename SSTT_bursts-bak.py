#!/usr/bin/env python3
"""
SST+ Burst Analysis Script
Analyzes burst firing properties of SST interneurons (PYR/PYR_nWf cells)
comparing SST_hM3D (Exc) vs SST_hM4D (Inh) groups.

Reads data from data/VS_SST_IN_HPc_8tt_slp_v2.csv and produces outputs in SSTT_outputs/:
- SSTT_burst_analysis.png: Publication-ready figure (300 DPI)
- SSTT_burst_analysis.pdf: Vector format alternative
- SSTT_burst_statistics.csv: Statistics summary table
- SSTT_burst_analysis_spike_hist.png: Spike count histogram
- SSTT_burst_analysis_spike_hist.pdf: Spike count histogram (PDF)
- SSTT_filtered_data.csv: Filtered intermediate data
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import mixedlm

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==========================================================================
# Configuration Constants
# ==========================================================================

# File paths - all outputs go to SSTT_outputs/ directory
BASE_DIR = Path(__file__).parent.resolve()
DATA_FILE = BASE_DIR / 'data' / 'VS_SST_IN_HPc_8tt_slp_v2.csv'
OUTPUT_DIR = BASE_DIR / 'SSTT_outputs'
FILTERED_DATA_FILE = OUTPUT_DIR / 'SSTT_filtered_data.csv'
STATISTICS_FILE = OUTPUT_DIR / 'SSTT_burst_statistics.csv'
OUTPUT_PNG = OUTPUT_DIR / 'SSTT_burst_analysis.png'
OUTPUT_PDF = OUTPUT_DIR / 'SSTT_burst_analysis.pdf'
SPIKE_HIST_PNG = OUTPUT_DIR / 'SSTT_burst_analysis_spike_hist.png'
SPIKE_HIST_PDF = OUTPUT_DIR / 'SSTT_burst_analysis_spike_hist.pdf'

# Colors
COLOR_EXC = '#FF6B6B'   # Light red for hM3D (Exc)
COLOR_INH = '#4ECDC4'   # Light blue for hM4D (Inh)
COLOR_EXC_ALPHA = 0.6
COLOR_INH_ALPHA = 0.6

# Cell type filter
CELL_TYPES_TO_INCLUDE = ['PYR', 'PYR_nWf']

# Group filter
GROUPS_TO_INCLUDE = ['SST_hM3D', 'SST_hM4D']

# Group label mapping
GROUP_LABEL_MAP = {
    'SST_hM3D': 'Exc',
    'SST_hM4D': 'Inh'
}

# Burst analysis metrics (column name -> natural language title)
BURST_METRICS = {
    'B_num_total': 'Total Bursts',
    'B_num_per_min': 'Bursts per Minute',
    'B_ibi_sec': 'Inter-Burst Interval (s)',
    'B_dur_msec': 'Burst Duration (ms)',
    'Nspk_B_Nspk_train_ratio': 'Burst Spike Ratio',
    'Spk_per_burst': 'Spikes per Burst'
}

# Spike count columns (2-20)
SPIKE_COUNT_COLS = [str(i) for i in range(2, 21)]

# Nature journal styling constants (5-6pt fonts, 2X larger figure dimensions)
NATURE_FONT_SIZE = 5
NATURE_TITLE_SIZE = 6
NATURE_LINE_WIDTH = 0.8
NATURE_FIGURE_WIDTH_PT = 180  # Single column width in points (~63mm)
# Figure dimensions in inches (2X larger than before: was 2.5x1.67, now 5x3.33)
VIOLIN_FIG_WIDTH_IN = 5.0
VIOLIN_FIG_HEIGHT_IN = 3.5
SPIKE_FIG_WIDTH_IN = 2.5
SPIKE_FIG_HEIGHT_IN = 1.75


def load_and_filter_data():
    """
    Load and filter the raw CSV data for burst analysis.
    
    Filters:
    - Only PYR and PYR_nWf cell types
    - Only SST_hM3D and SST_hM4D groups
    
    Returns:
        pd.DataFrame: Filtered DataFrame with Group_Label column added
    """
    print("=" * 60)
    print("SST+ BURST ANALYSIS - Data Loading and Filtering")
    print("=" * 60)
    
    # Check input file exists
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input data file not found: {DATA_FILE}")
    
    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} rows")
    
    # Filter by cell type
    initial_count = len(df)
    df = df[df['Cell_TYPE'].isin(CELL_TYPES_TO_INCLUDE)].copy()
    filtered_cells = initial_count - len(df)
    print(f"  Cell type filter: removed {filtered_cells} rows (kept {len(df)})")
    print(f"    Kept cell types: {CELL_TYPES_TO_INCLUDE}")
    
    # Filter by group
    initial_count = len(df)
    df = df[df['Group'].isin(GROUPS_TO_INCLUDE)].copy()
    filtered_groups = initial_count - len(df)
    print(f"  Group filter: removed {filtered_groups} rows (kept {len(df)})")
    print(f"    Kept groups: {GROUPS_TO_INCLUDE}")
    
    # Create group label mapping
    df['Group_Label'] = df['Group'].map(GROUP_LABEL_MAP)
    
    # Verify no NaN labels
    if df['Group_Label'].isna().any():
        print(f"  WARNING: {df['Group_Label'].isna().sum()} rows have NaN Group_Label")
        df = df.dropna(subset=['Group_Label'])
    
    # Validate burst columns for missing values
    print("\n  Data quality check for burst metrics:")
    for col in BURST_METRICS.keys():
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"    {col}: {missing} missing values ({missing/len(df)*100:.1f}%)")
            else:
                print(f"    {col}: OK (no missing values)")
    
    # Summary statistics
    print("\n  Summary statistics:")
    print(f"    Total cells: {len(df)}")
    print(f"    Total mice: {df['Mouse'].nunique()}")
    for group in GROUPS_TO_INCLUDE:
        group_data = df[df['Group'] == group]
        label = GROUP_LABEL_MAP[group]
        print(f"    {label} ({group}): {len(group_data)} cells from {group_data['Mouse'].nunique()} mice")
    
    # Save filtered data for downstream use
    FILTERED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTERED_DATA_FILE, index=False)
    print(f"\n  Filtered data saved to: {FILTERED_DATA_FILE}")
    
    print("\n" + "=" * 60)
    print("Data loading and filtering complete.")
    print("=" * 60)
    
    return df


def run_statistical_analysis(filtered_df):
    """
    Perform statistical analysis on burst metrics.
    
    For each of the 6 burst metrics:
    - Fits LMM with Group_Label as fixed effect and Mouse as random effect
    - Extracts coefficient, p-value, confidence interval
    - Calculates Cohen's d effect size
    - Applies Bonferroni correction
    
    For spike count distribution:
    - Expands spike count columns into individual observations
    - Performs Kolmogorov-Smirnov test
    
    Args:
        filtered_df (pd.DataFrame): Filtered data from load_and_filter_data()
        
    Returns:
        pd.DataFrame: Statistics DataFrame with results for all metrics
    """
    print("\n" + "=" * 60)
    print("SST+ BURST ANALYSIS - Statistical Analysis")
    print("=" * 60)
    
    stats_results = []
    
    # ---- LMM for each burst metric ----
    print("\nFitting Linear Mixed-Effects Models...")
    print("-" * 50)
    
    for metric, title in BURST_METRICS.items():
        # Get non-null data for this metric
        data = filtered_df[[metric, 'Group_Label', 'Mouse']].dropna()
        
        if len(data) < 10:
            print(f"  {title}: Skipped (insufficient data: {len(data)} rows)")
            stats_results.append({
                'Metric': metric,
                'Natural_Title': title,
                'N_observations': len(data),
                'Coefficient': np.nan,
                'SE': np.nan,
                'Z_value': np.nan,
                'P_value': np.nan,
                'P_corrected': np.nan,
                'Effect_Size_d': np.nan,
                'KS_statistic': np.nan,
                'KS_p_value': np.nan,
                'Significant': 'ns'
            })
            continue
        
        # Fit LMM: metric ~ Group_Label + (1 | Mouse)
        data = data.copy()
        data['Group_Encoded'] = (data['Group_Label'] == 'Exc').astype(int)
        
        try:
            model = mixedlm(
                f"{metric} ~ Group_Encoded",
                data=data,
                groups=data['Mouse']
            )
            result = model.fit()
          
            # Extract statistics
            coef = result.params['Group_Encoded']
            se = result.bse['Group_Encoded']
            z_value = coef / se
            p_value = 2 * stats.norm.sf(abs(z_value))  # Two-tailed
           
            # Calculate Cohen's d effect size
            exc_data = data[data['Group_Label'] == 'Exc'][metric].dropna()
            inh_data = data[data['Group_Label'] == 'Inh'][metric].dropna()
           
            if len(exc_data) > 1 and len(inh_data) > 1:
                # Pooled standard deviation
                pooled_sd = np.sqrt(((len(exc_data) - 1) * exc_data.std()**2 + 
                                   (len(inh_data) - 1) * inh_data.std()**2) / 
                                   (len(exc_data) + len(inh_data) - 2))
               
                if pooled_sd > 0:
                    mean_diff = exc_data.mean() - inh_data.mean()
                    cohens_d = mean_diff / pooled_sd
                else:
                    cohens_d = 0.0
            else:
                cohens_d = np.nan
           
            # Store results
            stats_results.append({
                'Metric': metric,
                'Natural_Title': title,
                'N_observations': len(data),
                'Coefficient': round(coef, 4),
                'SE': round(se, 4),
                'Z_value': round(z_value, 4),
                'P_value': round(p_value, 6),
                'P_corrected': round(min(p_value * 6, 1.0), 6),  # Bonferroni
                'Effect_Size_d': round(cohens_d, 4),
                'KS_statistic': np.nan,
                'KS_p_value': np.nan,
                'Significant': ''  # Will be filled below
            })
           
            # Significance annotation
            p_corr = min(p_value * 6, 1.0)
            if p_corr < 0.001:
                sig_marker = '***'
            elif p_corr < 0.01:
                sig_marker = '**'
            elif p_corr < 0.05:
                sig_marker = '*'
            else:
                sig_marker = 'ns'
           
            stats_results[-1]['Significant'] = sig_marker
           
            print(f"  {title}:")
            print(f"    Coefficient: {coef:.4f} (SE={se:.4f}, z={z_value:.4f})")
            print(f"    P-value: {p_value:.6f}, P-corrected: {p_corr:.6f}")
            print(f"    Effect size (Cohen's d): {cohens_d:.4f}")
            print(f"    Significance: {sig_marker}")
           
        except Exception as e:
            print(f"  {title}: ERROR - {e}")
            stats_results.append({
                'Metric': metric,
                'Natural_Title': title,
                'N_observations': len(data),
                'Coefficient': np.nan,
                'SE': np.nan,
                'Z_value': np.nan,
                'P_value': np.nan,
                'P_corrected': np.nan,
                'Effect_Size_d': np.nan,
                'KS_statistic': np.nan,
                'KS_p_value': np.nan,
                'Significant': 'err'
            })
    
    # ---- Spike count distribution test ----
    print("\nAnalyzing spike count per burst distribution...")
    print("-" * 50)
    
    # Expand spike count columns into individual observations
    exc_spikes = []
    inh_spikes = []
    
    for _, row in filtered_df.iterrows():
        group = row['Group_Label']
        for col in SPIKE_COUNT_COLS:
            if col in filtered_df.columns and pd.notna(row[col]):
                count = int(row[col])
                spike_count = int(col)  # The column name IS the spike count
                if count > 0:
                    spikes = [spike_count] * count
                    if group == 'Exc':
                        exc_spikes.extend(spikes)
                    else:
                        inh_spikes.extend(spikes)
    
    exc_spikes = np.array(exc_spikes)
    inh_spikes = np.array(inh_spikes)
    
    print(f"  Exc observations: {len(exc_spikes)}")
    print(f"  Inh observations: {len(inh_spikes)}")
    
    # Kolmogorov-Smirnov test
    if len(exc_spikes) > 0 and len(inh_spikes) > 0:
        ks_stat, ks_p_value = stats.ks_2samp(exc_spikes, inh_spikes)
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  KS p-value: {ks_p_value:.6f}")
        
        # Significance annotation
        if ks_p_value < 0.001:
            ks_sig = '***'
        elif ks_p_value < 0.01:
            ks_sig = '**'
        elif ks_p_value < 0.05:
            ks_sig = '*'
        else:
            ks_sig = 'ns'
        print(f"  Significance: {ks_sig}")
    else:
        ks_stat = np.nan
        ks_p_value = np.nan
        ks_sig = 'ns'
        print("  Insufficient data for KS test")
    
    # Add KS results to the Spk_per_burst metric's stats
    for result in stats_results:
        if result['Metric'] == 'Spk_per_burst':
            result['KS_statistic'] = round(ks_stat, 4) if not np.isnan(ks_stat) else np.nan
            result['KS_p_value'] = round(ks_p_value, 6) if not np.isnan(ks_p_value) else np.nan
            break
    
    # ---- Save statistics to CSV ----
    stats_df = pd.DataFrame(stats_results)
    
    # Select and order columns for output
    output_cols = ['Metric', 'Natural_Title', 'N_observations', 'Coefficient', 'SE',
                   'Z_value', 'P_value', 'P_corrected', 'Effect_Size_d', 
                   'KS_statistic', 'KS_p_value', 'Significant']
    stats_df = stats_df[[c for c in output_cols if c in stats_df.columns]]
    
    STATISTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(STATISTICS_FILE, index=False)
    print(f"\n  Statistics saved to: {STATISTICS_FILE}")
    
    # ---- Print formatted summary table ----
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'P-corr':<10} {'d':<10} {'Sig':<6}")
    print("-" * 56)
    
    for _, row in stats_df.iterrows():
        metric_name = row['Natural_Title'][:28]
        p_corr = f"{row['P_corrected']:.4g}" if pd.notna(row['P_corrected']) else 'N/A'
        d = f"{row['Effect_Size_d']:.3f}" if pd.notna(row['Effect_Size_d']) else 'N/A'
        sig = str(row['Significant']) if pd.notna(row.get('Significant', np.nan)) else 'N/A'
        print(f"{metric_name:<30} {p_corr:<10} {d:<10} {sig:<6}")
    
    print("\n" + "=" * 60)
    print("Statistical analysis complete.")
    print("=" * 60)
    
    return stats_df


def plot_violins(filtered_df, stats_dict):
    """
    Create violin plots for all 6 burst metrics with significance bars.
    
    Args:
        filtered_df (pd.DataFrame): Filtered data from load_and_filter_data()
        stats_dict (pd.DataFrame): Statistics from run_statistical_analysis()
    """
    print("\n" + "=" * 60)
    print("SST+ BURST ANALYSIS - Violin Plot Generation")
    print("=" * 60)
    
    # Set Nature journal styling (5-6pt fonts, larger figure)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 5,
        'axes.linewidth': NATURE_LINE_WIDTH,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
    })
    
    # Prepare statistics lookup by metric
    stats_lookup = {}
    for _, row in stats_dict.iterrows():
        stats_lookup[row['Metric']] = row
    
    # Create figure with 6 subplots (2 rows x 3 columns) - 2X larger dimensions
    fig, axes = plt.subplots(2, 3, figsize=(VIOLIN_FIG_WIDTH_IN, VIOLIN_FIG_HEIGHT_IN),
                             dpi=300, sharex=False)
    axes = axes.flatten()
    
    # Letter labels for subplots
    letter_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for idx, (metric, title) in enumerate(BURST_METRICS.items()):
        ax = axes[idx]
        
        # Prepare data for violin plot
        exc_data = filtered_df[filtered_df['Group_Label'] == 'Exc'][metric].dropna()
        inh_data = filtered_df[filtered_df['Group_Label'] == 'Inh'][metric].dropna()
        
        # Prepare all_values for y-axis calculations
        all_values = np.concatenate([exc_data.values, inh_data.values]) if len(exc_data) > 0 or len(inh_data) > 0 else np.array([])
        
        # Create violin plots
        violin_parts = ax.violinplot(
            [exc_data.values, inh_data.values],
            positions=[0, 1],
            widths=0.35,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Color Exc (position 1) with light red
        violin_parts['bodies'][0].set_facecolor(COLOR_EXC)
        violin_parts['bodies'][0].set_alpha(COLOR_EXC_ALPHA)
        violin_parts['bodies'][0].set_edgecolor(COLOR_EXC)
        violin_parts['bodies'][0].set_linewidth(1.5)
        
        # Color Inh (position 2) with light blue
        violin_parts['bodies'][1].set_facecolor(COLOR_INH)
        violin_parts['bodies'][1].set_alpha(COLOR_INH_ALPHA)
        violin_parts['bodies'][1].set_edgecolor(COLOR_INH)
        violin_parts['bodies'][1].set_linewidth(1.5)
        
        # Add quartile lines inside violins
        for i, group_data in enumerate([exc_data.values, inh_data.values]):
            if len(group_data) > 0:
                q25, q50, q75 = np.percentile(group_data, [25, 50, 75])
                ax.plot([i, i], [q25, q75], color='black', linewidth=1.0)
                ax.plot(i, q50, 'o', color='black', markersize=3)
        
        # Add raw data points (jittered)
        if len(exc_data) > 0:
            jitter = np.random.normal(0, 0.05, len(exc_data))
            ax.scatter(jitter, exc_data.values, color=COLOR_EXC, alpha=0.3, s=10, edgecolors='none')
        if len(inh_data) > 0:
            jitter = np.random.normal(0, 0.05, len(inh_data))
            ax.scatter(jitter + 1, inh_data.values, color=COLOR_INH, alpha=0.3, s=10, edgecolors='none')
        
        # Significance bar
        if metric in stats_lookup:
            stat_row = stats_lookup[metric]
            p_corrected = stat_row.get('P_corrected', np.nan)
          
            if not np.isnan(p_corrected) and p_corrected < 0.05:
                # Determine significance marker
                if p_corrected < 0.001:
                    sig_marker = '***'
                elif p_corrected < 0.01:
                    sig_marker = '**'
                else:
                    sig_marker = '*'
              
                # Calculate y-axis limit for significance bar placement
                all_values = np.concatenate([exc_data.values, inh_data.values])
                y_max = np.nanmax(all_values)
                y_range = np.nanpercentile(all_values, 95) - np.nanpercentile(all_values, 5)
                sig_y = y_max + 0.05 * y_range if y_range > 0 else y_max * 1.05
               
                # Draw significance bar
                ax.plot([0, 1], [sig_y, sig_y], color='black', linewidth=1.0)
                ax.plot([0, 0], [sig_y - 0.02 * y_range, sig_y], color='black', linewidth=1.0)
                ax.plot([1, 1], [sig_y - 0.02 * y_range, sig_y], color='black', linewidth=1.0)
                ax.text(0.5, sig_y + 0.02 * y_range, sig_marker,
                       ha='center', va='bottom', fontsize=5, color='black')
        
        # Set subplot title (natural language)
        ax.set_title(title, fontsize=6, fontweight='bold')
        
        # Add letter label in top-left corner
        ax.text(0.02, 0.98, letter_labels[idx], transform=ax.transAxes,
               fontsize=6, fontweight='bold',
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
        
        # Set x-axis labels and ticks
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Exc', 'Inh'], fontsize=5)
        
        # Set y-axis label
        metric_unit = metric.replace('_', ' ').title()
        ax.set_ylabel(metric_unit, fontsize=5)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limits with some padding
        if len(all_values) > 0:
            y_min = np.nanmin(all_values)
            y_max = np.nanmax(all_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)
        else:
            ax.set_ylim(0, 1)
        
        # Light grid for readability
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Hide any unused subplots
    for idx in range(len(BURST_METRICS), len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
    
    # Save to PNG (300 DPI)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {OUTPUT_PNG}")
    
    # Save to PDF
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches='tight')
    print(f"  Saved PDF: {OUTPUT_PDF}")
    
    # Close figure explicitly
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("Violin plot generation complete.")
    print("=" * 60)


def plot_spike_histogram(filtered_df):
    """
    Create overlaid histogram of spike counts per burst with KS test significance annotation.
    
    Args:
        filtered_df (pd.DataFrame): Filtered data from load_and_filter_data()
    """
    print("\n" + "=" * 60)
    print("SST+ BURST ANALYSIS - Spike Count Histogram Generation")
    print("=" * 60)
    
    # Set Nature journal styling (5-6pt fonts, larger figure)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 5,
        'axes.linewidth': NATURE_LINE_WIDTH,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
    })
    
    # Expand spike count columns into individual observations
    exc_spikes = []
    inh_spikes = []
    
    for _, row in filtered_df.iterrows():
        group = row['Group_Label']
        for col in SPIKE_COUNT_COLS:
            if col in filtered_df.columns and pd.notna(row[col]):
                count = int(row[col])
                spike_count = int(col)  # The column name IS the spike count
                if count > 0:
                    spikes = [spike_count] * count
                    if group == 'Exc':
                        exc_spikes.extend(spikes)
                    else:
                        inh_spikes.extend(spikes)
    
    exc_spikes = np.array(exc_spikes, dtype=int)
    inh_spikes = np.array(inh_spikes, dtype=int)
    
    print(f"  Exc observations: {len(exc_spikes)}")
    print(f"  Inh observations: {len(inh_spikes)}")
    
    # Kolmogorov-Smirnov test for distribution difference
    if len(exc_spikes) > 0 and len(inh_spikes) > 0:
        ks_stat, ks_p_value = stats.ks_2samp(exc_spikes, inh_spikes)
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  KS p-value: {ks_p_value:.6f}")
        
        # Determine significance marker
        if ks_p_value < 0.001:
            sig_marker = '***'
        elif ks_p_value < 0.01:
            sig_marker = '**'
        elif ks_p_value < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'
        print(f"  Significance: {sig_marker}")
    else:
        ks_stat = np.nan
        ks_p_value = np.nan
        sig_marker = 'ns'
        print("  Insufficient data for KS test")
    
    # Create figure for histogram - 2X larger dimensions
    fig, ax = plt.subplots(figsize=(SPIKE_FIG_WIDTH_IN, SPIKE_FIG_HEIGHT_IN),
                           dpi=300)
    
    # Define bins for histogram (integer spike counts 2-20)
    bins = np.arange(1.5, 21.5)  # Half-integer bins for proper integer alignment
    
    # Create overlaid histogram
    n_exc, bins_exc, patches_exc = ax.hist(exc_spikes, bins=bins, color=COLOR_EXC, 
                                            alpha=0.5, edgecolor='none', label='Exc',
                                            density=False)
    n_inh, bins_inh, patches_inh = ax.hist(inh_spikes, bins=bins, color=COLOR_INH,
                                            alpha=0.5, edgecolor='none', label='Inh',
                                            density=False)
    
    # Set x-axis to show integer spike counts
    x_ticks = range(2, 21)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=5)
    
    # Set labels and title
    ax.set_xlabel('Spikes per Burst', fontsize=5)
    ax.set_ylabel('Count of Bursts', fontsize=5)
    ax.set_title('Spike Count Distribution per Burst', fontsize=6, fontweight='bold')
    
    # Add legend
    ax.legend(fontsize=5, loc='upper right')
    
    # Add letter label in top-left corner
    ax.text(0.02, 0.98, 'G', transform=ax.transAxes,
           fontsize=6, fontweight='bold',
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
    
    # Add significance annotation above histogram
    if sig_marker != 'ns':
        # Get the maximum y value for positioning
        max_count = max(np.max(n_exc) if len(n_exc) > 0 else 0, 
                       np.max(n_inh) if len(n_inh) > 0 else 0)
        sig_y = max_count * 1.1 if max_count > 0 else 10
        ax.text(20.5, sig_y, sig_marker,
               ha='left', va='bottom', fontsize=6, fontweight='bold',
               color='black')
        # Draw a horizontal line at sig_y
        ax.plot([2, 20], [sig_y * 0.95, sig_y * 0.95], 
               color='black', linewidth=0.5, alpha=0.5)
    else:
        # If not significant, place 'ns' in upper right
        max_count = max(np.max(n_exc) if len(n_exc) > 0 else 0, 
                       np.max(n_inh) if len(n_inh) > 0 else 0)
        sig_y = max_count * 1.1 if max_count > 0 else 10
        ax.text(20.5, sig_y, 'ns',
               ha='left', va='bottom', fontsize=5, fontweight='bold',
               color='gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limits with padding
    max_count = max(np.max(n_exc) if len(n_exc) > 0 else 0, 
                   np.max(n_inh) if len(n_inh) > 0 else 0)
    if max_count > 0:
        ax.set_ylim(0, max_count * 1.2)
    else:
        ax.set_ylim(0, 10)
    
    # Light grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save to PNG (separate figure)
    SPIKE_HIST_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SPIKE_HIST_PNG, dpi=300, bbox_inches='tight')
    print(f"  Saved spike histogram PNG: {SPIKE_HIST_PNG}")
    
    # Save to PDF
    SPIKE_HIST_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SPIKE_HIST_PDF, bbox_inches='tight')
    print(f"  Saved spike histogram PDF: {SPIKE_HIST_PDF}")
    
    # Close figure explicitly
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("Spike count histogram generation complete.")
    print("=" * 60)


def main():
    """
    Main execution pipeline for SST+ burst analysis.
    
    Steps:
    1. Load and filter data
    2. Run statistical analysis
    3. Generate violin plots
    4. Generate spike count histogram
    5. Print completion message with output file paths
    """
    print("=" * 60)
    print("SST+ BURST ANALYSIS - Full Pipeline")
    print("=" * 60)
    
    # Step 1: Load and filter data
    df = load_and_filter_data()
    
    # Step 2: Run statistical analysis
    stats_df = run_statistical_analysis(df)
    
    # Step 3: Generate violin plots
    plot_violins(df, stats_df)
    
    # Step 4: Generate spike count histogram
    plot_spike_histogram(df)
    
    # Step 5: Print completion message
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE")
    print("=" * 60)
    print(f"All output files saved to: {OUTPUT_DIR}")
    print(f"  Violin plots PNG: {OUTPUT_PNG}")
    print(f"  Violin plots PDF: {OUTPUT_PDF}")
    print(f"  Spike histogram PNG: {SPIKE_HIST_PNG}")
    print(f"  Spike histogram PDF: {SPIKE_HIST_PDF}")
    print(f"  Statistics CSV: {STATISTICS_FILE}")
    print(f"  Filtered data CSV: {FILTERED_DATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

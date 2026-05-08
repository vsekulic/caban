# SST+ Burst Analysis Plan

## Overview
Create a standalone Python script `SSTT_bursts.py` that reads `data/VS_SST_IN_HPc_8tt_slp_v2.csv`, filters for PYR/PYR_nWf cells only, and performs burst analysis comparisons between SST_hM3D (Exc) and SST_hM4D (Inh) groups. The script runs once and performs all calculations, plotting, and statistical analysis.

## Data Filtering
- **Cell type filter:** Only include rows where `Cell_TYPE` starts with `PYR` (i.e., `PYR` and `PYR_nWf`)
- **Group filter:** Only include `SST_hM3D` and `SST_hM4D` groups
- **Group labeling:** `SST_hM3D` -> "Exc", `SST_hM4D` -> "Inh"
- **Color scheme:** hM3D = light red, hM4D = light blue

## Burst Analysis Metrics (6 violin plots)
| Column | Natural Language Title |
|--------|------------------------|
| `B_num_total` | "Total Bursts" |
| `B_num_per_min` | "Bursts per Minute" |
| `B_ibi_sec` | "Inter-Burst Interval (s)" |
| `B_dur_msec` | "Burst Duration (ms)" |
| `Nspk_B_Nspk_train_ratio` | "Burst Spike Ratio" |
| `Spk_per_burst` | "Spikes per Burst" |

## Spike Count per Burst Histogram
- Columns `2` through `20` represent spike counts per burst (e.g., column "2" = number of bursts with exactly 2 spikes)
- Create overlaid histograms (one per group) showing distribution of spikes per burst
- Statistical test: Kolmogorov-Smirnov test comparing distributions between groups
- Note: Each row has counts, so need to expand counts into individual burst observations for histogram

## Statistical Analysis
1. **Linear Mixed-Effects Models (LMM)** for each of the 6 burst metrics:
   - Fixed effect: `Group` (Exc vs Inh)
   - Random effect: `(1 | Mouse)` - Mouse identity as clustering factor
   - Use `statsmodels.formula.api.mixedlm`
2. **Post-hoc tests:**
   - Wald test for the Group coefficient from LMM
   - Report effect size (Cohen's d or similar)
3. **Spike count distribution test:**
   - Kolmogorov-Smirnov test comparing spike-per-burst distributions between groups

## Plot Specifications
- **Format:** Nature journal style
  - Single column width (~180pt or ~7.5cm)
  - Arial fonts throughout
  - Resolution: 300+ DPI for publication
  - Line widths: 0.8-1.0pt for axes, 1.0-1.5pt for data elements
- **Layout:**
  - 6 subplots for burst metrics (2 rows x 3 columns)
  - Each subplot labeled with capital letter (A, B, C, D, E, F)
  - Subplot titles: natural language versions from table above
  - X-axis labels: "Exc" and "Inh"
  - Y-axis labels: metric name with units
- **Violin plot style:**
  - Semi-transparent fills (light red for Exc, light blue for Inh)
  - Show median as dot, quartiles as line inside violin
  - Or use standard violin with inner="quartile"
  - **Significance bars:** When LMM finds significant difference (p < 0.05), add significance bar above violins:
    - `*` for p < 0.05
    - `**` for p < 0.01
    - `***` for p < 0.001
    - Bar spans across both groups with horizontal line and vertical ticks
- **Histogram style:**
  - Overlaid histograms for both groups in same plot
  - hM3D (Exc): light red with transparency (alpha=0.5)
  - hM4D (Inh): light blue with transparency (alpha=0.5)
  - If KS test finds significant difference, add significance annotation (*, **, ***) above histogram

## Memory-Conscious Design (GPU Server with 11GB VRAM)
- Process data in pandas (CPU memory, not GPU)
- No large arrays loaded into GPU VRAM
- Use efficient pandas operations
- Close figures explicitly after saving

## Output Files
- `SSTT_burst_analysis.png` - Publication-ready figure (300 DPI)
- `SSTT_burst_analysis.pdf` - Vector format alternative
- `SSTT_burst_statistics.csv` - Statistics summary table

## Dependencies
- `pandas` for data loading and manipulation
- `seaborn` for violin plots
- `statsmodels` for LMM (`statsmodels.formula.api.mixedlm`)
- `scipy` for post-hoc tests (KS test, Mann-Whitney U)
- `matplotlib` for figure customization
- `numpy` for numerical operations

---

## Implementation Subtasks for Local LLM

Each subtask below can be copied and pasted into a new task instance. Each subtask builds on the previous one by appending to the same `SSTT_bursts.py` file.

### Subtask 1: File Header, Imports, and Data Loading Function

**Context:** This is the first subtask. Create a new file `SSTT_bursts.py` in the project root.

**Instructions:**
1. Create `SSTT_bursts.py` with:
   - Shebang line and module docstring
   - All required imports (pandas, numpy, seaborn, matplotlib, statsmodels, scipy)
   - Configuration constants (file paths, colors, metric definitions)
   - `load_and_filter_data()` function that:
     - Reads `data/VS_SST_IN_HPc_8tt_slp_v2.csv` using pandas
     - Filters for rows where `Cell_TYPE` is `PYR` or `PYR_nWf`
     - Filters for rows where `Group` is `SST_hM3D` or `SST_hM4D`
     - Creates `Group_Label` column mapping hM3D->"Exc", hM4D->"Inh"
     - Validates data quality (missing values in burst columns)
     - Prints summary statistics (cells per group, mice per group)
     - Returns filtered DataFrame

**Copy-paste this context for the next subtask:**
> Subtask 1 is complete. `SSTT_bursts.py` exists with imports, configuration, and `load_and_filter_data()` function. The function reads the CSV, filters for PYR/PYR_nWf cells and SST_hM3D/SST_hM4D groups, creates Group_Label mapping, validates data, and returns the filtered DataFrame.

---

### Subtask 2: Statistical Analysis Functions

**Context:** Subtask 1 is complete. `SSTT_bursts.py` exists with imports, configuration, and `load_and_filter_data()` function.

**Instructions:**
1. Append to `SSTT_bursts.py`:
   - `run_statistical_analysis(filtered_df)` function that:
     - Takes the filtered DataFrame from Subtask 1
     - For each of the 6 burst metrics, fits an LMM using `statsmodels.formula.api.mixedlm`:
       - Formula: `metric ~ Group_Label`
       - Re: `(1 | Mouse)` as random effect
       - Prints model summary (coefficient, p-value, CI)
     - Post-hoc analysis:
       - Extract Group_Label coefficient p-value from each LMM
       - Calculate Cohen's d effect size for Exc vs Inh
       - Apply Bonferroni correction for 6 comparisons
     - Spike count distribution:
       - Expand spike count columns (2-20) into individual observations
       - For each row, if column "3" = 5, add five 3s to the group list
       - Perform KS test comparing Exc vs Inh distributions
     - Save results to `SSTT_burst_statistics.csv`
     - Print formatted summary table
     - Return statistics dictionary

**Copy-paste this context for the next subtask:**
> Subtasks 1 and 2 are complete. `SSTT_bursts.py` exists with data loading, filtering, and statistical analysis functions. The LMM models have been fitted, post-hoc tests performed, and statistics saved to `SSTT_burst_statistics.csv`.

---

### Subtask 3: Violin Plot Generation Function

**Context:** Subtasks 1 and 2 are complete. `SSTT_bursts.py` exists with data loading, filtering, and statistical analysis functions. `SSTT_burst_statistics.csv` exists with results.

**Instructions:**
1. Append to `SSTT_bursts.py`:
   - `plot_violins(filtered_df, stats_dict)` function that:
     - Takes filtered DataFrame and statistics dictionary
     - Creates figure with 6 subplots (2 rows x 3 columns)
     - For each subplot:
       - X-axis: Group_Label ("Exc", "Inh")
       - Y-axis: metric value
       - Violin plots with light red (Exc) and light blue (Inh), alpha=0.7
       - inner="quartile" for quartile display
       - **Significance bars:** When corrected p-value < 0.05, add significance bar above violins with `*` (p<0.05), `**` (p<0.01), or `***` (p<0.001)
       - Subplot title: natural language title
       - Letter labels (A-F) in top-left corner
     - Nature journal styling:
       - Arial font, ~180pt width
       - Font sizes: 7-8pt labels, 9-10pt titles
       - Remove top/right spines
     - Save to `SSTT_burst_analysis.png` (300 DPI)
     - Also save to `SSTT_burst_analysis.pdf`
     - Close figure explicitly

**Copy-paste this context for the next subtask:**
> Subtasks 1-3 are complete. `SSTT_bursts.py` exists with data loading, statistical analysis, and violin plot functions. The violin plot figure has been saved.

---

### Subtask 4: Spike Count Histogram Function and Main Execution

**Context:** Subtasks 1-3 are complete. `SSTT_bursts.py` exists with data loading, statistical analysis, and violin plot functions.

**Instructions:**
1. Append to `SSTT_bursts.py`:
   - `plot_spike_histogram(filtered_df)` function that:
     - Takes filtered DataFrame
     - Expands spike count columns (2-20) into individual observations
     - Creates overlaid histogram (Exc: light red alpha=0.5, Inh: light blue alpha=0.5)
     - X-axis: spikes per burst (integers 2-20)
     - Y-axis: count of bursts
     - **Significance annotation:** If KS test finds significant difference, add `*` (p<0.05), `**` (p<0.01), or `***` (p<0.001) above histogram
     - Nature journal styling (same as violin plots)
     - Save to `SSTT_burst_analysis.png` (append as additional panel or separate figure)
   - `main()` function that:
     - Calls `load_and_filter_data()`
     - Calls `run_statistical_analysis()`
     - Calls `plot_violins()`
     - Calls `plot_spike_histogram()`
     - Prints completion message with output file paths
   - `if __name__ == "__main__":` block calling `main()`

**Final state:** `SSTT_bursts.py` is complete and runnable. Running `python SSTT_bursts.py` will:
1. Load and filter data
2. Run statistical analysis
3. Generate violin plots
4. Generate spike count histogram
5. Save all outputs

---

## Running the Script

```bash
python SSTT_bursts.py
```

The script will execute all steps sequentially and produce:
- `SSTT_burst_analysis.png` (300 DPI publication figure)
- `SSTT_burst_analysis.pdf` (vector format)
- `SSTT_burst_statistics.csv` (statistics summary)

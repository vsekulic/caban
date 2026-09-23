# docs/ index

Reference documents, runbooks and manuscript text for caban. Grouped by role; within each group the document a newcomer needs first comes first. See also: [plans index](../plans/0_INDEX_plans.md).

## Setup and runbooks

| # | Document | Date | Description |
|---|---|---|---|
| 1 | [JUPYTER_SETUP.md](JUPYTER_SETUP.md) | 2026-05-11 | End-to-end setup for running the pipeline in remote JupyterLab on a Linux server, driven from VS Code. |
| 2 | [local_mac_migration.md](local_mac_migration.md) | 2026-08-27 | Runbook for moving `run_pipeline.ipynb` work off the RIKEN cbp-db server onto the local Mac; written against `feat/sp-rates-axis-labels`. |

## Analysis references

| # | Document | Date | Description |
|---|---|---|---|
| 3 | [sp_rates_lmm.md](sp_rates_lmm.md) | 2026-08-21 → 08-25 | Pyramidal event-amplitude and event-rate analysis of DREADD effects during TFC. Part A holds the two mouse-level mixed models that the paper reports. Part B holds the sensitivity, Bayesian, permutation and historical analyses, and is not what the paper reports. Also defines the epoch table, including `post_shock_late`. |
| 4 | [epoch_modulation.md](epoch_modulation.md) | 2026-08-26 → 09-21 | Per-cell epoch modulation, the single-cell block of Figure 2. **Headline is a null**: no group × epoch interaction and no within-epoch contrast survives correction. Epoch modulation is itself marginal in every group. Carries the YrA shear background (§A.4, open item O.5). |
| 5 | [navigation_aware_single_cell_analyses.md](navigation_aware_single_cell_analyses.md) | 2026-08-15 | Map of the `navigation_aware_single_cell` suite (`NAV_AWARE_DIR`). It splits whole-session rates by place vs non-place cell and by movement vs immobility frame, and relates them to the sp_rates and place-field panels. |
| 6 | [conditioning_recall_link.md](conditioning_recall_link.md) | 2026-09-18 | **Exploratory only.** Per-animal screen of whether conditioning-session coding predicts recall freezing. It ran 16 predictor × outcome combinations with uncorrected P values, and exists so that a later pre-specified test can justify its choice of predictor. |

## Manuscript text — `paper/`

| # | Document | Date | Description |
|---|---|---|---|
| 7 | [paper/paper_figure2.md](paper/paper_figure2.md) | 2026-08-25 | Results, Methods and legend for Figure 2, the claim that SST-IN modulation dissociates the size and the frequency of pyramidal events during conditioning. Drawn from sp_rates_lmm Part A; it contains no recall numbers. |
| 8 | [paper/paper_figure2_supplement.md](paper/paper_figure2_supplement.md) | 2026-08-25 | Supplementary Figure 2: the epoch-resolved decomposition of conditioning-day activity, plus the drug-free recall analyses (Test B, Test B 1 wk). |

# Figure panels → plot files → notebook sections

Where every panel of Figure 2 and Supplementary Figure 2 comes from, and the per-cell analyses whose
panels are still to be chosen. Reconstructed 2026-10-02 from [paper_figure2.md](paper_figure2.md) §1,
[paper_figure2_supplement.md](paper_figure2_supplement.md) §1 and
[../epoch_modulation.md](../epoch_modulation.md), and checked against the files on disk. Paths are relative
to `PLOTS_DIR` = `~/data/vsekulic/OF_test/plots/CURRENT`; the sections are those of `run_pipeline.ipynb`.

**Two runs on disk.** `CURRENT` — 2026-10-01, recomputed YrA and the timing fix (this map's "new").
`CURRENT-20260929-exported-YrA` — the reference: `sp_rates_lmm` from 2026-08-24, `epoch_modulation` from
2026-09-18, `epoch_sequence` from 2026-08-28.

## Figure 2 (main) — [paper_figure2.md](paper_figure2.md)

| panel | content | file | notebook section | in new run |
|---|---|---|---|---|
| a | viral strategy, schematic | — (drawn) | — | — |
| b | FOV and Minian footprints | — (Minian output) | — | — |
| c | example traces ΔF/F, C, S (+ YrA) | `plot_sample_traces/plot_sample_traces.png` | Initial checks → **Sample traces** | **not run yet** (plots YrA, so it changes) |
| d | manipulation check LT1 → LT2 | `sp_rates_lmm/TFC_cond/manipulation_check.png` | Initial checks → **sp_rates_lmm** | yes |
| e | trace-interval decomposition | `sp_rates_lmm/TFC_cond/decomposition.png` | sp_rates_lmm | yes |
| f | height-matched example events | `sp_rates_lmm/TFC_cond/width_height_matched_examples_vertical.png` | sp_rates_lmm | yes |
| g | per-animal amplitude across 5 epochs | `sp_rates_lmm/TFC_cond/epoch_profile.png` | sp_rates_lmm | yes |
| h | trace-interval amplitude ECDF | `sp_rates_lmm/TFC_cond/amplitude_ecdf.png` | sp_rates_lmm | yes |
| i | **primary analysis**: amplitude and rate by epoch | `sp_rates_lmm/paper/tfc_amplitude_rate/tfc_amplitude_rate_by_epoch.png` | sp_rates_lmm | yes |

Numbers for d–i: [paper_figure2.md](paper_figure2.md) §5 (every value → its stats file).

## Supplementary Figure 2 — [paper_figure2_supplement.md](paper_figure2_supplement.md)

| panel | content | file | notebook section |
|---|---|---|---|
| a | decomposition, pre-tone baseline | `sp_rates_lmm/TFC_cond/decomposition_pre_tone_matched.png` | sp_rates_lmm |
| b | decomposition, post-shock | `sp_rates_lmm/TFC_cond/decomposition_post_shock.png` | sp_rates_lmm |
| c | all components × epochs, estimates and CIs | `sp_rates_lmm/TFC_cond/decomposition_grid.png` | sp_rates_lmm |
| d | recall amplitude and rate, 48 h / 1 wk | `sp_rates_lmm/paper/recall/Test_B/testb_amplitude_rate_by_epoch.png`, `…/Test_B_1wk/testb_1wk_amplitude_rate_by_epoch.png` | sp_rates_lmm |
| e | recall pre → post-tone modulation | `sp_rates_lmm/paper/recall/Test_B/testb_amplitude_rate_modulation.png`, `…/Test_B_1wk/testb_1wk_amplitude_rate_modulation.png` | sp_rates_lmm |

## The per-cell block — panels to be chosen from the results (VS, 2026-09-29)

All computed on the TFC_cond sessions, for both signals: `<signal>` = `YrA` (primary) or `C`
(confirmatory). Design and earlier results: [../epoch_modulation.md](../epoch_modulation.md).

| label | content | file | notebook section |
|---|---|---|---|
| K | tone-aligned per-cell heatmaps, by group | `epoch_modulation/<signal>/panel_K_tone_aligned_heatmaps.png` | Single-unit responses → **Epoch modulation** |
| L | modulation index by epoch × group (mouse-level model) | `epoch_modulation/<signal>/panel_L_epoch_modulation.png` | Epoch modulation |
| — | same, hierarchical cell-level lane | `epoch_modulation/<signal>/hierarchical_cells/hierarchical_epoch_modulation.png` | Epoch modulation |
| M | event-aligned group-mean traces | `epoch_modulation/<signal>/event_proximal/panel_M_event_aligned_traces.png` | Epoch modulation |
| N | event-proximal (short-window) index | `epoch_modulation/<signal>/event_proximal/panel_N_event_proximal_index.png` | Epoch modulation |
| O | held-out heatmaps by preferred event | `epoch_modulation/<signal>/cell_selectivity/panel_O_heldout_heatmaps.png` | Epoch modulation |
| P | cross-validated selectivity | `epoch_modulation/<signal>/cell_selectivity/panel_P_selectivity.png` | Epoch modulation |
| — | YrA vs C agreement | `epoch_modulation/signal_comparison.{txt,csv}` | Epoch modulation |
| — | cross-validated sequence test (the honest version of K's sort) | `epoch_sequence/cross_validated_sequence.png`, `…/split_half_correlations.png` | **Cross-validated sequence test** |
| — | event-locked responsiveness: fractions, magnitudes, tone/shock heatmaps — encoding, recall Test_B, Test_A | `event_locked_responsiveness/{encoding,recall_testB,recall_testA}/*.png` | **Event-locked responsiveness** |
| — | per-cell amplitude and event-rate distributions, by session and by crossreg mapping | `cell_activity_distributions/cell-dist-*.png` | **Per-cell activity distributions** |

## Earlier Figure 2 exports — not the current figure

`0-PAPER_PLOTS/fig2/` (present in the reference only): `amplitudes-*`, `event-rate-*`, `frac-active-*` (.svg)
and `plots/*-activity_binned_sp_rates_mapping-*.png`, written by the **Spike-rate panels** and **Binned
spike-rate panels** sections (`caban/analysis.py`, `caban/analyses.py`, `get_paper_dir(PAPER_DIR, 'fig2')`).
That is the earlier, pre-`sp_rates_lmm` version of Figure 2 (fractions active and binned rates across
cross-registration mappings); the current text does not use them.

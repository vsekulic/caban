"""caban — hippocampal Ca2+ imaging analysis pipeline.

Submodules:
    config, loader, pipeline       — refactored entry points (notebook-driven)
    main                            — full analysis script (data load +
                                      all analyses in one file). The
                                      notebook drives this section by
                                      section via
                                      ``caban.pipeline.run_main_section``.
    utilities, sessions, analysis,
    decoder, spatial, plotting,
    engram, engram_sanity,
    epoch_analysis, isomap,
    population, pca_state_metrics,
    TFC, CFC                        — domain modules
"""

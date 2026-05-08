# Copilot Instructions — sstca2

## Naming

- **NEVER name analyses, pipeline stages, functions, files, or directories with bare numeric/sequential labels** like `phase1`, `phase2`, `phase3`, `analysis_1`, `step2`, `Phase 1`, etc.
- Always use **meaningful, descriptive names** that convey what the code/analysis actually does — e.g. `independent` (per-session independent manifolds), `crossreg` (cross-registered cells projected into a reference frame), `anchor` (anchored to a reference session), `similarity_session_level`, `similarity_per_trial`, `similarity_vs_position`, etc.
- This applies equally to:
  - function names (`_fit_independent_manifolds`, NOT `_phase1_run`)
  - local variables and parameters (`crossreg_result`, NOT `phase2`)
  - log message prefixes (`[crossreg]`, NOT `[Phase 2]`)
  - comments and docstrings
  - output directories and filenames
- The only acceptable use of an integer in a name is when the integer is a *meaningful domain quantity* (e.g. `dim_2d`, `n_neighbors_15`, `lag_1`), not a sequence index.

## Error Handling

- **NEVER add silent skips or fallbacks.** If something fails, raise/assert with a clear error message.
- No `try/except` that swallows errors and `continue`s.
- No `if condition_bad: print("skip"); continue` patterns.
- Code must hard-fail so the root cause is immediately visible.

## Imports

- **All imports at the top of the file.** NEVER use lazy/inline imports inside function bodies.
- This applies to both standard-library and project-internal imports.

## Code Deduplication

- **Aggressively refactor.** NEVER blindly copy-paste or duplicate existing functionality. Call the shared function instead.
- If a helper/pathway already exists, wire up to it — do NOT create a parallel implementation with subtly different parameters.
- All decoder analyses (Phase 1–6, main PVT pipeline, true-2D pipeline) must call the same underlying decode functions (`decode_within_LT_position_vs_time`, `_cross_session_decode_PF2D_raw`) with the full parameter set threaded through from the caller.
- When adding a new analysis that re-uses an existing computation, refactor the existing code into a callable function if needed — don't copy the block.

## METHODS Files

- Every major analysis must have a METHODS text file in `analysis_methods_templates/` describing the statistical approach.
- The analysis code must copy its METHODS template into the plots subdirectory at runtime using `_copy_analysis_methods_template(filename, save_dir)`.
- This ensures each output directory is self-documenting with a co-located METHODS description.

## Reload Snippet After Code Changes

- Whenever finishing a round of code changes, also display how to regenerate the new result from the user's currently running debug session, using `importlib.reload()` on the affected modules and a re-call of the relevant pipeline entry point with the same locals already in scope (e.g. `pop_pca_results`, `mouse_groups`, `POP_PCA_PLOTS_DIR`).
- The snippet must be copy-pasteable into the live session — reload modules bottom-up (dependencies first), then re-import any `from X import Y` names that were re-bound by the reload, then invoke the entry point.


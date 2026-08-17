# CLAUDE.md — caban

Project instructions for Claude Code. These mirror `.github/copilot-instructions.md` and apply to all Claude Code assistance in this repo.

## Naming

- **NEVER name analyses, pipeline stages, functions, files, or directories with bare numeric/sequential labels** like `phase1`, `phase2`, `analysis_1`, `step2`, etc.
- Always use **meaningful, descriptive names** that convey what the code/analysis actually does — e.g. `independent`, `crossreg`, `anchor`, `similarity_session_level`, `similarity_per_trial`, etc.
- This applies equally to function names, local variables, log message prefixes, comments/docstrings, and output directories/filenames.
- The only acceptable use of an integer in a name is when the integer is a *meaningful domain quantity* (e.g. `dim_2d`, `n_neighbors_15`, `lag_1`), not a sequence index.

## Error Handling

- **NEVER add silent skips or fallbacks.** If something fails, raise/assert with a clear error message.
- No `try/except` that swallows errors and `continue`s.
- No `if condition_bad: print("skip"); continue` patterns.
- Code must hard-fail so the root cause is immediately visible.

## Imports

- **All imports at the top of the file.** NEVER use lazy/inline imports inside function bodies.
- This applies to both standard-library and project-internal imports.
- **Resolve circular imports by ordering top-level definitions, never with lazy/in-function imports.** If module A needs a name from module B and B imports A back, define/expose that name in B *before* B's import of A (see the deferred cross-module import block in `caban/decoder.py`, which is placed after the constants/helpers that `caban/analysis.py` imports at its top). Do not paper over a cycle with a function-body import.

## Code Deduplication

- **Aggressively refactor.** NEVER blindly copy-paste or duplicate existing functionality. Call the shared function instead.
- If a helper/pathway already exists, wire up to it — do NOT create a parallel implementation with subtly different parameters.
- All decoder analyses must call the same underlying decode functions with the full parameter set threaded through from the caller.
- When adding a new analysis that re-uses an existing computation, refactor the existing code into a callable function if needed — don't copy the block.

## METHODS Files

- Every major analysis must have a METHODS text file in `analysis_methods_templates/` describing the statistical approach.
- The analysis code must copy its METHODS template into the plots subdirectory at runtime using `_copy_analysis_methods_template(filename, save_dir)`.

## DREADD Comparison Plots

- Use the established pastel box-and-strip style for categorical comparisons: pastel box fill, darker matching points, `alpha=0.85`, black point edges, `linewidth=0.3`, and black median lines.
- **Group DISPLAY order is always `mCherry`, `hM3D`, `hM4D`** (control first as the reference, then excitatory, then inhibitory) — on every panel of every figure, with no per-analysis exceptions. This matches `decoder._PAPER_GROUP_ORDER`, `spatial._PV_GROUP_ORDER`, and the `_paper_group_order` locals throughout `analysis.py`. Use `caban.single_unit_common.DREADD_DISPLAY_ORDER` rather than defining another local copy.
  - This is DISPLAY order only. `single_unit_common.GROUP_ORDER` (`hM3D`, `hM4D`, `mCherry`) is a separate thing — it drives model dummy-coding and iteration order, and must not be changed to match: existing published analyses depend on it. Plotting helpers index their inputs by group name, so display order never changes which values are compared.
  - Older code contains two stale orders (`hM3D, mCherry, hM4D` and `hM3D, hM4D, mCherry`) in display positions. Those are wrong; fix them when touching that code.
- Save separate per-panel files for multi-panel DREADD figures when practical.
- Reuse existing group color conventions — do not introduce a new palette.

## Reload Snippet After Code Changes

- After finishing a round of code changes, display a copy-pasteable `importlib.reload()` snippet for the user's live debug session.
- Reload modules bottom-up (dependencies first), re-import any `from X import Y` names rebound by the reload, then invoke the entry point with the same locals already in scope.

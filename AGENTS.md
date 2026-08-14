# AGENTS.md — caban

Agent instructions for this repo. These apply in addition to `CLAUDE.md`.

## Never re-run pipeline sections in the background — tell the user which cell to run

- The user works from `run_pipeline.ipynb`, with a live kernel holding `ds` and `cfg`. Any
  `caban.sections.run_*` call (or anything else that regenerates plots into `cfg.PLOTS_DIR`) is
  something the user can run themselves from that notebook.
- **Do not launch these as a background script or process**, even to verify a change, even if it
  would be faster or more convenient than waiting for the user. Background runs duplicate a
  kernel the user already has warm, race with whatever they're doing in the notebook, and write
  into the same live `PLOTS_DIR` the user is watching.
- Instead: identify the exact cell(s) by position (count top-to-bottom in the saved notebook,
  1 = the first cell — cell IDs in the `.ipynb` JSON are not useful to the user) and tell the user
  which to run. Example format:

  | Cell # | Call |
  |---|---|
  | 33 | `run_place_cell_properties(ds, cfg)` |

- If a change only affects a subset of sections, say so explicitly and scope the cell list to
  just those — don't tell the user to re-run everything when only one section changed.
- This applies to regenerating plots after a code change, not to small read-only verification
  snippets (e.g. unit-checking a helper function, inspecting a CSV) run to sanity-check an edit
  before handing it off — those don't touch `PLOTS_DIR` and aren't a substitute for a notebook
  cell the user already has.

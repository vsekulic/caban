# METHODS — Shock-locked pyramidal events (Figure 2 v2, panel i)

`caban/shock_events.py`. Descriptive; adds no statistical test.

Events were detected on each conditioning session's deconvolved trace S exactly as for the primary
analysis (contiguous supra-threshold runs, one event per run, amplitude the run integral, threshold
2) and placed in time at their peak frame. For every trial, events were binned in 1 s bins from 5 s
before to 10 s after shock onset (all five trials, all cells). Per animal and bin: the event rate is
the number of events divided by cells × trials × 1 s; the amplitude is the mean, over that bin's
events, of the log per-event amplitude (a bin with no event leaves that animal's amplitude undefined
there; it is not filled). The panel shows the mean ± s.e.m. across animals per group. Shading marks
the 2 s shock (darker) and the following seconds (lighter); the dashed line is shock offset.

The group comparison at the shock is the declared secondary comparison of the primary model —
per-event amplitude in the 20 s after shock offset, hM3D versus control — reported with that model;
this panel shows its time course. During the 2 s shock, brief motion-correction failures occurred in
four animals (Supplementary Fig. 2h); events in that window are retained and the window is
interpreted with that caveat.

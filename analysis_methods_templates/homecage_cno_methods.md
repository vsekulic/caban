# METHODS — Pyramidal events in the homecage before vs after CNO (conditioning day)

`caban/homecage_cno.py`; pre-specified in `plans/figure2_redesign_plan.md` §4.1.1 (revised after a
statistics review and committed before any result was computed).

## Recordings

On the conditioning day each animal was recorded in its homecage for about 5 min at several points:
before the CNO injection (HC1; G12–G21 also HC2) and after it (CNO1, CNO2 in some animals, and a
later homecage recording before conditioning in most). Each recording was processed by Minian with
the cohort's single frozen parameter set; its units are its own (no cross-registration).

The injection time was taken as the start of CNO1 in animals that also have a CNO2 recording, and
as 30 min before the start of CNO1 where CNO1 is the only CNO recording (the experimenter's
labelling; the label CNO1 changed meaning during data collection).

- **pre**: HC1.
- **post**: the single homecage recording that starts 15–35 min after the injection and before
  conditioning — the window in which CNO acts — chosen by time, not by recording name.

## Events and measures

Identical to the primary conditioning analysis. Events are contiguous runs of the deconvolved trace
S above 2 (`find_event_runs_ca_S`), one event per run, amplitude the run's integral. Exposure is the
recording's imaged duration (frames / 19.76 fps; each recording's frame rate checked against it).
Per animal and window: the mean over active units of each unit's log mean per-event amplitude, and
the population event rate (all events over all units' exposure), as in
`sp_rates_lmm.build_mouse_epoch_unified_table`. These are statistics of the detected population, not
within-cell changes.

## Statistics

The animal is the unit. Per animal and measure, Δ = log(post) − log(pre). Exc − Ctl and Inh − Ctl
were compared by Welch two-sample t-tests, Holm-corrected across the two within each measure
(two-sided; the predicted directions, amplitude Exc > Ctl and rate Inh < Ctl, were stated before the
analysis); the exact pairwise mouse-label randomization of the same mean difference is reported
beside each. Exc − Inh is reported uncorrected as exploratory. A mixed model
`log(metric) ~ group × window + (1|animal)` is a consistency check only.

## Detection diagnostics

Per recording: number of units, fraction of units with at least one event, events per unit per
minute. Because pre and post are different recordings with different detected units, a group ×
window change in these would mean the measured change partly reflects which cells were detected;
mCherry animals also received CNO and so control for injection and time, but not for a
DREADD-specific effect on detection.

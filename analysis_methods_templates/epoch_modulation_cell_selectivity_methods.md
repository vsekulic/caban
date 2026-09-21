# Cross-validated per-cell event selectivity during trace fear conditioning

Companion to the epoch-modulation analysis (`epoch_modulation_methods.md`) and to its
event-proximal lane (`epoch_modulation_event_proximal_methods.md`). Those analyses compare the
**average** cell's modulation profile between groups. This one asks whether **individual cells**
hold a reproducible preference for one conditioning event, and whether chemogenetic manipulation
of SST interneurons changes it.

## Measurement

Per cell, activity was standardized once over the whole session (see the parent METHODS; YrA
primary, C confirmatory), and for every retained conditioning trial an index was formed for each
of the four events — tone onset, trace onset (tone offset), shock onset and post-shock onset — as
the mean of the standardized trace over the **3 s from that onset** minus the mean over the same
trial's 20 s pre-tone window. This is the event-proximal index, unchanged.

Each (mouse, trial, event) mean **across that mouse's cells** was then subtracted, so every index
expresses a cell's departure from its own animal's average cell on that trial. Without this
centring a response shared by all cells — such as the cohort-wide suppression after tone offset —
would be read as per-cell selectivity, because most cells would select the same event and the
held-out trial would confirm it.

## Cross-validation

Preference was established and tested on **disjoint trials**, by leave-one-trial-out. For each
held-out trial *k*:

1. **Choice (training trials only).** Each cell's four centred indices were averaged over the
   other retained trials. The preferred event *e\** was the one with the largest absolute average,
   and *s* was its sign, so both increases and decreases could be preferences.
2. **Measurement (held-out trial only).** Selectivity was

   *s* × ( index at *e\** − mean of the indices at the other three events ),

   evaluated on trial *k* alone, in units of the cell's own session-wide SD.

Each cell contributed one value per retained trial (4 trials for 16 of 17 mice, 3 for one). When
cells hold no reproducible preference beyond their animal's average cell, the held-out value has
expectation zero, because the held-out trial took no part in the choice. **No cell was ever scored
on the data that selected it**, and no responsive/non-responsive threshold was applied at any
point.

## Endpoints and inference

The animal is the unit of inference (*n* = 17 mice); cells and held-out trials are descriptive.
Each mouse contributed its mean selectivity, pooled over all cells and held-out trials, and
separately within each preferred event.

- **Is there reproducible selectivity at all?** An exact sign-flip test of the per-mouse pooled
  values against zero, enumerating all 2¹⁷ = 131,072 sign vectors.
- **Do the groups differ?** For each endpoint, the difference in group means of the per-mouse
  values (each animal weighted equally), with an exact mouse-label randomization *P* obtained by
  enumerating every relabelling of the two compared groups' animals (462 or 924), holding each
  animal's own cells fixed. Within each endpoint the two treatment-versus-control comparisons were
  Holm-corrected together; hM3D versus hM4D was computed but belongs to no correction family and
  carries no bracket. Welch 95% confidence intervals on the same per-mouse values are reported as
  descriptive intervals.

The composition of preferences — the share of choices falling on each event — is reported as a
**descriptive** quantity only. Selecting the largest absolute index favours events whose index is
noisier, so composition partly reflects per-event noise rather than per-event preference. The
held-out selectivity is unaffected by this, since a choice made on noise contributes zero in
expectation on the held-out trial.

## Display

The held-out heatmap shows one row per cell per held-out trial: that trial's tone-aligned activity
only, grouped by the preference chosen **without** it and ordered within each block by the
training-trial strength of that preference. Neither the grouping nor the ordering uses the
displayed data, which is what distinguishes it from a heatmap sorted on the statistic it shows —
such a sort yields a clean structure even for pure noise.

## Validation

Before use, the procedure was checked on simulated data with known ground truth: it holds its
false-positive rate under pure noise and under a population-wide response (which the same data,
analysed without within-mouse centring, misreports as strong selectivity); it recovers a planted
selective subpopulation and the planted event and sign; it detects a planted difference in the
selective fraction in the manipulated group only; and a large effect confined to a single animal
does not produce a significant group difference.

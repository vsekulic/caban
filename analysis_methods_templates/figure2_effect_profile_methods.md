# METHODS — Effect profile and cell-weighted fraction active (Figure 2 checks)

`caban/figure2_effect_profile.py`; pre-specified in `plans/figure2_redesign_plan.md` §12 before
either test was computed. Both use the primary conditioning model's per-animal trace-interval values
(`unified_lmm_mouse_epoch_values.csv`). The animal is the unit.

## Effect profile

Per measure (mean log per-event amplitude; log population event rate), each animal's value minus the
mCherry mean, divided by the pooled within-group standard deviation of that measure. A is the
standardized amplitude effect (positive = increase); R is minus the standardized rate effect
(positive = decrease); the profile index is P = A − R. An animal whose change lies mainly in amplitude
has P > 0, one whose change lies mainly in event frequency has P < 0. hM3D and hM4D were compared on P
by a two-sided Welch t-test, with the exact pairwise mouse-label randomization (all 462 relabellings)
beside it. The control group defines the origin and is not tested. This index summarizes which
measure carries each animal's change; it does not test that either measure is unchanged.

## Fraction of cells active, weighted

The number of cells with at least one trace-interval event out of all detected cells, per animal,
modelled by a binomial generalized linear model on group (reference mCherry) with the dispersion
estimated from the Pearson χ² statistic (quasi-binomial) and t-based inference, so that each animal
contributes in proportion to its cell count and between-animal variation widens the intervals.
Treatment-versus-control odds ratios with 95% CI, Holm-corrected across the two; the exact pairwise
mouse-label randomization of the cell-weighted difference in fraction active beside each. Reported
alongside the unweighted Welch test on per-animal fractions.

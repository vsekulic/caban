# Split-half stability of per-cell response profiles during trace fear conditioning

Supplementary companion to the epoch-modulation analysis (`epoch_modulation_methods.md`). That
analysis compares the **average** cell's modulation between groups. This one asks whether a cell's
response, **relative to the other cells of the same animal**, is reproducible from one set of
conditioning trials to another, and whether chemogenetic manipulation of SST interneurons changes
that reproducibility. It measures the stability of relative response profiles; it is not a test of
a stable cell-identity code.

## Measurement

The index is the parent analysis's, unchanged: per cell, activity standardized once over the whole
session (YrA primary, C confirmatory), and for every retained conditioning trial the mean over each
response epoch (tone, trace, shock, post-shock) minus the mean over the same trial's pre-tone
window.

No further centring is needed. A response shared by all cells of an animal shifts every cell's
value by the same amount, which a correlation across cells ignores, so only differences between
cells can contribute.

## Split-half correlation

For each animal, the retained trials were divided into two disjoint halves. Each cell's index was
averaged within each half, and the Spearman correlation across the animal's cells
between the two halves' averages was computed for each epoch. Because the halves share no trials,
the correlation has expectation zero when cells hold no reproducible differences relative to the
other cells of their animal.

**Every balanced split was used rather than one.** With five trials there are ten ways to divide
them into two and three, and with four trials (one animal) three ways to divide them into two
pairs; the animal's value for an epoch is the mean of its split correlations on
the Fisher-z scale. This all-splits form of split-half reliability removes the dependence on one
arbitrary split. The correlations describe half-trial averages as they are; no Spearman–Brown
step-up to the full trial count was applied.

Two pooled endpoints were defined in advance, each the mean of four per-epoch Fisher-z values, so
that no single epoch's spread dominates:

- **All epochs** — any reproducible relative response. A cell that responds more than its
  neighbours in every epoch (a general gain or responsiveness trait) is stable in each epoch, so
  this endpoint largely reflects such traits.
- **Epoch profile** — reproducible epoch-specific tuning. Before correlating, each cell's mean over
  the four epochs was removed within each half, so a pure gain trait contributes nothing.

The four per-epoch correlations are reported as epoch-specific results.

## Inference

The animal is the unit of inference throughout.

- **Is there reproducible cell-specific structure?** Exact sign-flip test of the per-animal mean
  Fisher z against zero, enumerating all 2^n sign vectors (n = number of animals). The two pooled
  endpoints are reported alone; the four epochs are Holm-corrected among themselves. The test
  pools all three groups, so it describes stability in this cohort rather than under baseline
  conditions; the control group's value is reported separately. For a single split the null
  distribution is exactly symmetric about zero; for the mean over splits, which reuse the same
  trials, its validity was confirmed by simulation (below).
- **Do the groups differ?** Difference in group means of the per-animal Fisher z, with an exact
  randomization of animal labels between the two compared groups. Holm correction over the two
  treatment-versus-control comparisons within each endpoint; the comparison between the two
  treatment groups is reported but is in no family. Welch 95% intervals are descriptive.

Values are displayed as correlations (the back-transform of the mean Fisher z).

## Validation

A hand-run check on synthetic data with a known answer (`verify_split_half_stability_synthetic`)
confirms that the cohort tests hold their false-positive rate when no stable cell differences exist
(including under a response shared by all cells, and with right-skewed indices), that both pooled
endpoints detect planted stable per-cell preferences, that a pure gain trait is detected by the
all-epochs endpoint but not by the epoch-profile endpoint, that a planted group difference is
detected only in the group that carries it, and that a group difference driven by a single animal
is not reported.

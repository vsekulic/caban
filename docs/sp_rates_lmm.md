# sp_rates_lmm — analysis reference and decision record

Pyramidal event-**amplitude** and event-**rate** analysis of DREADD effects during trace fear
conditioning.

**This document has two parts, and the distinction is the most important thing in it.**

- **[Part A](#part-a--the-current-paper-facing-analysis)** is the analysis the manuscript reports:
  two mouse-level mixed models, one per outcome, given identical treatment. Every number in a
  paper figure or a Results sentence comes from here.
- **[Part B](#part-b--sensitivity-supplementary-and-historical-analyses)** is everything else this
  module computes — the cell-level amplitude model and its three-test Holm family, the Bayesian
  negative-binomial rate model, the mouse-label permutation tests, the BH-FDR secondary family,
  threshold sensitivity, run structure, the exploratory DREADD-vs-DREADD contrasts, and the
  decision record behind all of it. All of it still runs, all of it is worth having, and **none of
  it is a paper-facing result.**

Exactly one analysis is primary. If you find a sentence anywhere that reads as though a Part B
p-value were the headline, it is stale — fix it.

If you are picking this up cold: read §A (the models), then §2.3 and §8 before changing anything,
and §3.4 plus §4.5 before touching an epoch definition.

| | |
|---|---|
| Module | [caban/sp_rates_lmm.py](../caban/sp_rates_lmm.py) |
| Entry point | `caban.sections.run_sp_rates_lmm(ds, cfg)`, gated on `cfg.plot_sp_rates_lmm` |
| Paper analysis | conditioning: `caban.sp_rates_lmm.render_paper_tfc_amplitude_rate()`; recall: `render_paper_recall_amplitude_rate()` (§A.7) |
| METHODS templates | [sp_rates_lmm_paper_methods.md](../analysis_methods_templates/sp_rates_lmm_paper_methods.md) (manuscript, conditioning), [sp_rates_lmm_recall_paper_methods.md](../analysis_methods_templates/sp_rates_lmm_recall_paper_methods.md) (manuscript, recall), [sp_rates_lmm_methods.md](../analysis_methods_templates/sp_rates_lmm_methods.md) (internal) |
| Notebook | `run_pipeline.ipynb`, cells 24–29 (immediately after `run_sp_rates`) |
| Output | paper: `PLOTS_DIR/sp_rates_lmm/paper/tfc_amplitude_rate/` and `.../paper/recall/{Test_B,Test_B_1wk}/`; internal: `PLOTS_DIR/sp_rates_lmm/{TFC_cond,Test_B,Test_B_1wk}/` + `stats/` |
| Design | 17 mice — hM3D n=5, hM4D n=6, mCherry n=6 |

---

# Part A — the current paper-facing analysis

## A. Two outcomes, one statistical framework

Per-event amplitude and population event rate are presented as parallel rows of one figure, so
they get one statistical treatment rather than two. Amplitude remains the **primary biological
endpoint** (the claim under test is that hM3D increases per-event Ca²⁺ influx, not merely event
frequency) and rate the secondary one — but that is a scientific ordering, not a statistical one.
Both are analysed identically.

**This analysis was finalised after substantial inspection of this dataset.** It is the *main
statistical analysis* / *primary paper-facing analysis*. It is not prospectively preregistered, is
not described as prospectively confirmatory, and does not introduce an evidential-tier taxonomy.

### A.1 One mouse-level value per epoch

`build_mouse_epoch_unified_table()` produces the inferential dataset: **17 mice × 3 epochs = 51
rows**, both outcomes on one frame, from one call to `restrict_to_exposure_matched_trials()` over
`pre_tone_matched` / `trace` / `post_shock` (§3.4a). Because both outcomes are derived from that
single frame, the two figure rows provably describe the same animals, epochs and trials — there is
no code path in which the rate row comes from a different trial set than the amplitude row above
it.

| | per-event amplitude | population event rate |
|---|---|---|
| per cell | mean event-run integral over the retained trials | — |
| then | natural log of that | — |
| per mouse | arithmetic mean of those cell-level logs | total events ÷ total cell-seconds, then logged |
| zero-event cells | excluded (amplitude undefined) | **retained in the denominator** |
| column | `mouse_mean_log_amplitude` | `log_population_rate` |

The amplitude weighting is deliberate and unchanged from this module's original design: each
active cell contributes equally within its animal, each animal contributes exactly one value to
the model. Cells are never pooled across animals before the group mean.

**`exp(a group contrast)` on amplitude is a ratio of GEOMETRIC means** of the cell-level mean
event-run integrals — the mean of logs is the log of the geometric mean. Describing it as a ratio
of pooled arithmetic mean event amplitudes would be wrong by Jensen's inequality, by an amount
that depends on each group's cell-level spread. Say "geometric" wherever this ratio is reported.

A zero population rate would make `log()` undefined. It is checked for and **raises**; no
pseudocount is applied, since the constant chosen would determine the answer.

### A.2 The models

For each outcome, identically:

```
log(metric) ~ group * epoch + (1 | mouse)
```

- reference group `mCherry`, reference epoch `pre_tone_matched`, both set as ordered
  `pd.Categorical` so nothing depends on alphabetical order;
- fit by `caban.single_unit_common.fit_mixed_model` (statsmodels `MixedLM`, REML, L-BFGS);
- **if either outcome fails to fit as a mixed model the paper analysis hard-fails**
  (`require_common_unified_method`). Letting one outcome take the documented clustered-OLS
  fallback while the other stays on MixedLM would falsify the claim that the two received the
  same treatment. Adopting clustered OLS as a *common* fallback for both is a deliberate decision
  to be made after inspecting the failure, never applied automatically.

### A.3 Epoch dependence: one joint Wald test per outcome

`joint_wald_test` on the four `group × epoch` coefficients, `df2 = n_mice − 1 = 16` (§4's
denominator-df convention, applied identically to both outcomes and to the contrasts below, so
the omnibus and its simple effects are one procedure).

**Wording.** A null interaction is *no evidence that the treatment effect differed across the
pre-tone, trace and post-shock epochs*. Internally the shorthand "the treatment effect was not
detectably epoch-specific" is fine; the manuscript uses the literal form. Nothing anywhere may
turn a null interaction into positive evidence that the effect is identical, global, tonic, or
equivalent across epochs.

And in the other direction: a significant treatment-control comparison within one epoch does not
imply that the treatment effect differs from another epoch. Epoch dependence is tested by the
interaction, once per outcome, and by nothing else.

### A.4 Contrasts and multiplicity: within-epoch treatment-vs-control families

Twelve treatment-vs-control simple effects — 3 epochs × {hM3D, hM4D} vs mCherry, per outcome —
each a linear contrast of the fitted model (`linear_contrast_test`, same `df = 16`):

- at `pre_tone_matched`: the group coefficient alone;
- at `trace` / `post_shock`: the group coefficient **plus** that group's interaction coefficient,
  with a variance that includes their covariance.

A simple effect is a contrast, not a coefficient; reading the group coefficient off the model
summary and calling it "the effect at trace" is wrong at every non-reference epoch.

**The multiplicity family is within an epoch, and is the same in every epoch.** For each
`(outcome, epoch)` the two treatment-versus-control comparisons — hM3D vs mCherry and hM4D vs
mCherry — are Holm-corrected together (`p_holm_epoch`, finite on every row). That is six
two-member families: 3 epochs × 2 outcomes. The procedure is identical in all six, so every
visually equivalent panel of the paper figure is treated equivalently and no epoch receives a
different kind of inferential treatment.

Amplitude and rate remain separate outcomes and therefore separate families. hM3D-vs-hM4D is in
no family and is not computed here — it remains exploratory internal output (§5.1).

So the whole paper-facing hierarchy is six steps:

1. one mouse-level value per animal per epoch (§A.1);
2. the same LMM for amplitude and for rate (§A.2);
3. two model-derived treatment-versus-control post-hoc contrasts within each epoch;
4. Holm correction across exactly those two comparisons within that epoch → `p_holm_epoch`;
5. the group × epoch joint Wald test for epoch dependence (§A.3);
6. the six-way across-epoch Holm correction retained as sensitivity only.

**A simple effect and an interaction answer different questions.** A significant comparison in one
epoch and not in another is *not* evidence that the treatment effect differs between those epochs;
that is what step 5 tests, and it is tested by nothing else. Symmetrically, a significant simple
effect and a null interaction are not in conflict — *does treatment differ from control in this
epoch?* is a different question from *does that difference itself vary across epochs?*

**The conservative six-comparison family is retained as sensitivity output**, not deleted:
`p_holm_six` Holm-corrects all six treatment-versus-control simple effects across the three epochs
within each outcome. Both families are computed from the same twelve raw contrasts; only the
grouping differs, and the wider one is kept so the choice stays auditable. `p_holm_six` determines
no figure annotation and no Results significance statement, and it appears only under Part 2 of
`paper_results_summary.md`.

Reported per contrast: log estimate, SE, log 95% CI, exponentiated ratio and ratio CI, raw P,
`p_holm_epoch` (all rows, paper-facing) and `p_holm_six` (all rows, sensitivity), each with its
rejection flag. For rate, additionally the **observed absolute difference in events/s/cell**
(equal-mouse-weighted group means) — descriptive, no test, reported because a rate ratio off a
small base overstates the practical size of a change.

*(Earlier passes used a single six-member family, and then a privileged trace-only primary family
`p_holm_trace`. Both are superseded; see §7 for the correction record.)*

### A.5 The figures come from these two tables and nothing else

| file | what it is |
|---|---|
| `stats/unified_lmm_mouse_epoch_values.csv` | the 51-row inferential dataset |
| `stats/unified_lmm_posthoc_contrasts.csv` | **authoritative** — every estimate, interval, P and asterisk |
| `stats/unified_lmm_interactions.csv` | the two joint Wald interaction tests |

- **Large mouse markers** on `tfc_amplitude_rate_by_epoch.png` are the exact model values, on the
  natural scale: `exp(mouse_mean_log_amplitude)` (that animal's geometric mean event amplitude)
  for the top row, `population_rate` for the bottom. `_assert_markers_match_model` hard-fails if a
  drawn marker is not the corresponding row of the values CSV, so figure/model identity is
  mechanical rather than visual. Per-cell clouds are display only and are plotted on the same
  natural scale as their marker.
- **Every panel is annotated by the same procedure.** Each of the six panels takes its own
  `(outcome, epoch)` pair of `p_holm_epoch` values from the contrasts CSV, handed to the panel by
  `_precomputed_stat_fn` via `_paper_panel_star_pvalues`, which ignores the data arrays entirely
  and hard-fails on a non-finite adjusted P. There is no per-epoch branch: a bracket is drawn
  wherever the adjusted P clears α, so a panel without one is a panel where neither comparison
  reached significance, not one that was exempted from testing. A paper panel cannot compute a
  statistic of its own, and no `p_holm_six` value can reach a figure.
- **`tfc_decomposition_forest.png`** draws the same twelve rows as ratios with CIs — all three
  epochs, because the estimates are scientifically informative — each row labelled with that
  outcome's joint Wald test. No significance stars are placed on the forest. It is a second view
  of one table, not a second estimate.
- No Bayesian HDI, no permutation p, no per-panel Welch test and no BH q-value appears on a paper
  figure. Those all still exist under Part B.

### A.6 Diagnostics and verification

Descriptive, gating nothing, adding no inference:

- `unified_lmm_diagnostics.png` — residual-vs-fitted and normal Q-Q per outcome;
  `unified_lmm_residuals.csv` the underlying rows. **No residual-normality test is used as an
  acceptance criterion** — a mixed model's residuals are not 51 independent observations, and a
  normality test crossing 0.05 is not a principled pass/fail rule at this n. Read the plots.
- `unified_lmm_influence.csv` — leave-one-mouse-out refits of both models, reporting each animal's
  effect on the twelve contrast **estimates**. Significance-decision flips are deliberately not
  computed: at n = 17 an adjusted p crossing 0.05 when one animal is dropped is expected and says
  nothing about robustness. Only a genuinely large estimate change is flagged (more than half the
  full-data effect *and* more than 0.10 log units — the absolute floor matters, since on a null
  contrast a relative rule alone fires constantly).
- `unified_lmm_synthetic_verification.txt` — two planted designs run through the real fitting and
  contrast code. An equal shift in every epoch must be detected by the simple effects while the
  interaction stays null; a trace-only shift must make the interaction significant. Margins are
  wide by construction, so this tests the code rather than one lucky realization.

## A.7 The recall lane — Test_B and Test_B_1wk, each on its own

`render_paper_recall_amplitude_rate()`, one call per recall session, writing
`PLOTS_DIR/sp_rates_lmm/paper/recall/<session>/`. It exists so the manuscript can describe
conditioning and retrieval with **one** description of statistical method.

**It is the same analysis as §A, with a different epoch set and a different cohort.** Same event
definition (one contiguous supra-threshold run of `S`, amplitude = its integral — unchanged and
not re-tuned), same animal-level summarization (§A.1), same
`log(metric) ~ group * epoch + (1|mouse)` for both outcomes, same `require_common_unified_method`
hard-fail, same within-epoch two-member Holm family, same joint Wald interaction test, same figure
grammar and the same `_assert_markers_match_model` marker check. The unified helpers took a
`reference_epoch` parameter for this; nothing about the conditioning models changed.

| | conditioning (§A) | recall (§A.7) |
|---|---|---|
| epochs | `pre_tone_matched`, `trace`, `post_shock` | `pre_tone`, `post_tone` — 20 s each |
| reference epoch | `pre_tone_matched` | `pre_tone` |
| interaction | joint Wald, 4 coefficients, df1 = 4 | joint Wald, 2 coefficients, df1 = 2 |
| denominator df | 16 (`n_animals − 1`) | that session's own `n_animals_present − 1` |
| across-epoch Holm sensitivity | `p_holm_six`, retained | **not computed** — no such family was ever declared for recall, and there would be four, not six |

**The two 20 s windows.** `pre_tone` ends at tone onset; `post_tone` begins at tone offset and is
the retrieval analogue of the conditioning trace interval. The **tone epoch itself is excluded**,
as are the legacy 35 s post-tone window and `TestBSession.post_tone_offsets`' run-to-the-next-tone
interval. `get_recall_epoch_frames()` is a thin guard over `get_testb_epoch_frames` that adds the
completeness check the latter does not make: a post-tone window that would run past the end of the
recording (or into the next tone) returns `None`, and a pre-tone window that would overlap the
previous trial's post-tone window **raises**. Missing and ambiguous are different problems and are
handled differently. Each animal is then restricted to the trials carrying both complete windows
(`restrict_to_exposure_matched_trials`, measured exposure, never a trial index), and the per-animal
coverage is written to `stats/unified_recall_trial_coverage.csv`. Both outcomes come off that one
frame, so the rate row cannot describe different trials from the amplitude row.

**Test_B and Test_B_1wk are never compared with each other here, and the code cannot be made to.**
The hM4D animal missing at 48 h is not the one missing at 1 week, so *"significant at 48 h but not
at 1 week"* is not evidence that anything declined — it is two independent analyses of two
different animal sets. A formal timepoint comparison needs a fixed-cohort model restricted to
animals present at both sessions; that is not fit here and is deliberately not approximated by
putting both sessions on one figure. Each session's `paper_results_summary.md` says so at the top
and contains no cross-session sentence.

**Retrieval preferentiality is the interaction and nothing else.** A significant post-tone
comparison with a null interaction does *not* establish that the difference is tone-evoked. The
readings of each (simple effect, interaction) pattern are fixed in advance in
`_RECALL_SIMPLE_EFFECT_READING` / `_RECALL_INTERACTION_READING` so they cannot drift toward
whichever sentence the observed numbers would flatter — and the interaction's verdict is written
per **outcome**, never per group, because a joint 2-df test does not localize to one group.

**No CNO was present at recall.** Every generated file says so. A recall difference is a
persistent consequence of the conditioning-day manipulation, not evidence of ongoing receptor
activation; "hM3D activation increased activity at recall" is not a supportable sentence.

### A.7.1 The pre→post modulation decomposition (Test_B)

The interaction test above answers *does the pre→post change differ among the groups?* but shows
nobody the change. This block adds the per-animal change scores and the pairwise decomposition of
that same interaction. **It fits no model.** Every number is a linear contrast of the two fits the
lane already made, on the same `df = n_animals_present − 1`.

- `build_recall_modulation_by_mouse` pivots the inferential table to one row per (animal, outcome)
  carrying `delta_log = post − pre` and `fold_change_post_vs_pre = exp(delta_log)`. It is derived
  from `<prefix>_mouse_epoch_values.csv` itself, so a plotted change score is a difference of the
  numbers the model saw — there is no second aggregation path.
- `recall_modulation_contrasts` returns two blocks. **Within-group** pre→post changes (the control
  group's is the epoch main coefficient; a treatment group's is that coefficient plus its own
  interaction term) are **descriptive** — they characterize the trajectory and explain the shape
  of the interaction, and are not a between-group test. **Pairwise** comparisons of that change
  are the inferential question: against mCherry each is a single group × epoch coefficient, and
  hM3D-vs-hM4D is the *difference* of the two, which needs their covariance and is therefore a
  contrast, not a subtraction of two published SEs. All three identities are asserted in code.
- **The reported values are the unadjusted model-derived contrasts** — `p_raw`, a linear contrast
  of the same fitted model, which is what the modulation panel's brackets and the drafted
  paragraph read. A Holm adjustment across the three pairwise comparisons WITHIN one outcome (two
  independent three-member families, amplitude and rate never pooled, the same shape as the
  within-epoch families) is still computed and still written to the CSV and the companion markdown
  as a **multiplicity reference** for auditability. It is not the governing decision rule for this
  lane. Neither column is relabelled as the other, and neither is a prospectively preregistered
  decision. This is a change in what is *reported*, not in what is computed: the estimates,
  intervals, *t*, df and both P columns are exactly the values the model has always produced.
  The within-epoch `p_holm_epoch` families are unaffected and remain Holm-governed.
- **Symmetric across groups and outcomes.** All three comparisons are computed and tabulated for
  both outcomes; no group is a headline, and a decomposition that singled one group out would be a
  different and unstated inferential structure. Which comparisons the *figure* brackets
  (`RECALL_MODULATION_FIGURE_PAIRS`) is a display choice on top of that complete table — hM4D vs
  mCherry is tabulated but not drawn — and changes no computed value.

**Three questions the outputs keep apart**, because they can disagree in both directions:

| | question | where |
|---|---|---|
| A | groups differ *within* an epoch (absolute level) | `unified_recall_posthoc_contrasts.csv` |
| B | a group changes at all pre→post | the within-group block — descriptive |
| C | the pre→post change *differs between* groups | the pairwise block, and its omnibus |

Equal post-tone levels reached from unequal baselines is a large C with a null A;  parallel decline
from unequal baselines is a large A with a null C. `verify_recall_modulation_synthetic` plants
exactly that case (along with a shared decline, a decline abolished in each DREADD group in turn,
and an hM3D/hM4D separation around a midway control) and hard-fails if the code does not recover
it — for both outcomes.

Outputs, under `paper/recall/Test_B/`: `stats/unified_recall_modulation_by_mouse.csv`,
`stats/unified_recall_modulation_contrasts.csv` (both blocks, distinguished by `block`),
`stats/unified_recall_modulation_contrasts.md`,
`stats/unified_recall_modulation_synthetic_verification.txt`,
`testb_amplitude_rate_modulation.{png,svg}` (one panel per outcome, one point per animal, a zero
reference line, brackets read from the unadjusted model-derived contrasts via
`_precomputed_stat_fn` — a bracket in `RECALL_MODULATION_NS_LABEL_PAIRS` carries its *P*-value even
when *P* ≥ 0.05 — and each panel's own omnibus in its title) and `testb_amplitude_rate_prepost_trajectories.{png,svg}`
(descriptive; one line per animal, natural scale, 2 outcomes × 3 group facets).

**Scoped to Test_B by `RECALL_MODULATION_SESSIONS`.** The machinery is session-agnostic; extending
it to `Test_B_1wk` is adding the key and nothing else. That does not license comparing the two.

**Not in this lane:** any cross-session comparison, cross-registered cell-identity persistence,
responder classification, tone-epoch statistics, and the negative-binomial count model as a
distribution-aware rate sensitivity analysis (permitted by the plan, not the paper-facing
statistic, and not fit in this pass).

The older `PLOTS_DIR/sp_rates_lmm/{Test_B,Test_B_1wk}/post_tone_amplitude.*` output is untouched
and remains a Part B secondary (a single-epoch cell-level amplitude omnibus in the BH-FDR family).
It is not this lane and supplies no paper number.

### A.7.2 The hierarchical cell-level companion analysis (Test_B)

> **Explicitly invoked — not part of a routine pass.** `run_hierarchical_cell_analysis` defaults to
> **`False`**, so a normal `run_sp_rates_lmm()` never runs any of this. The suite costs ~40 min
> (exact mouse-label MixedLM enumerations over the paired-cell table, the hierarchical NB count
> model, and the prior/posterior-predictive machinery) against a few minutes for everything else,
> and its evaluation is complete. Rerun it deliberately with
> `run_sp_rates_lmm(..., run_hierarchical_cell_analysis=True)` or
> `render_paper_recall_amplitude_rate(..., run_hierarchical_cell_analysis=True)`. Nothing was
> removed: every function, every component and every output file it has already written is intact,
> and passing `True` reproduces the same suite. The mouse-level recall lane is the paper-facing
> analysis and is complete without it.
>
> **What the completed run is good for.** Its **paired-cell amplitude** result stands as
> *supplementary sensitivity evidence* that the hM3D effect magnitude survives explicit pre→post
> pairing of the same neurons: 1.465× against the mouse-level 1.501×, exact mouse-label
> *P* ≈ 0.052. It is not the source of any figure annotation. Its **hierarchical NB rate model** is
> *not* usable as sensitivity evidence at all — it failed its own posterior-predictive adequacy
> check — and is deliberately **not** redesigned or replaced. The mouse-level Test_B
> population-rate analysis is unaffected and unchanged.

**Why this exists.** A.7.1's decomposition, like everything above it, is computed from one scalar
per animal per epoch. That is a valid hierarchical analysis and it stays the primary one — but it
is **not the only valid one**, and it throws away two things:

- **within-cell pairing** — an animal's pre-tone amplitude averages the cells active at pre-tone
  and its post-tone amplitude averages the cells active at post-tone, so cell-identity variance
  never cancels out of the change score;
- **within-mouse cellular heterogeneity** — whether an animal's shift is coherent across its
  population or carried by a handful of cells.

This block asks the complementary question: *does the pre→post modulation pattern occur coherently
across the cellular population within animals, with the cell hierarchy modelled rather than
collapsed before fitting?* It is **strictly additive** and writes into its own subdirectory,
`paper/recall/Test_B/hierarchical_cells/`, so its numbers cannot be mistaken for the primary lane's.
Nothing above it changed.

**Mouse-level assignment is still the basis of inference.** Treatment was assigned to 16 animals,
not to cells. Every paper-facing *P* in this block comes from permuting the 16 **mouse** labels,
with every cell fixed to its own animal. `n = 16 mice`; cell counts are descriptive and never a
sample size.

#### How the cell information is retained, and what makes it a hierarchical *test*

| | | |
|---|---|---|
| **estimator** | `delta_log_amplitude_cell ~ group + (1\|mouse)` on the paired cells | cells enter through the model's own within/between-mouse variance partition |
| **inference** | exact mouse-label randomisation **of that model's coefficient** | the model is refit under every relabelling, so the hierarchy is *inside* the statistic being tested |
| **sensitivity** | randomisation of the 16 mouse-mean deltas | model-free check, but collapses each animal to one number — carries no cellular hierarchy, and is in no Holm family |

The distinction matters: permuting a per-mouse average would be a better-computed mouse mean, not a
hierarchical test. Both are reported, neither is described as doing the other's job, and no
numerical equivalence between them is claimed (the model's GLS weighting and equal-mouse weighting
coincide only at equal cluster sizes; cells per mouse vary here).

**The restriction is on the randomisation, not the data.** For comparison (a, b) the third group's
mice keep their true labels and stay in every fit; only a's and b's labels are exchanged, at their
observed sizes. The full **three-group** model is refit every time — never a two-group subset — so
the reported estimate, its Wald CI and the permuted statistic all describe one specification. The
hM3D-vs-hM4D statistic is `β_Exc − β_Inh` as a linear contrast, using the coefficients' covariance.

**Exact, because the restriction makes it affordable.** With 6/5/5 mice the restricted spaces hold
C(11,6)=462, C(11,6)=462 and C(10,5)=252 relabellings — computed from the observed sizes at run
time and checked against the enumerator — so every one is enumerated and the three *P*-values are
**exact**, with no Monte Carlo error and no `+1/+1` correction (the observed labelling is one of the
enumerated draws). **Holm across exactly those three** is the paper-facing family and the source of
the figure's brackets. The omnibus cannot work this way — the global space is 2,018,016 and each
draw refits a mixed model — so it is Monte Carlo at a frozen seed over a **2-df model-based**
statistic (the joint Wald test of both group coefficients). Its H0 is that those coefficients are
jointly zero; it is not a test of the cellular distribution's shape, spread or tails.

#### How pseudoreplication is prevented

Treating thousands of cells as independently randomised treatment replicates is the failure mode
this block is built against. Three things prevent it, and the synthetic suite checks all of them
(**as a development tool, run by hand — it is not part of a real-data run**; see below):

1. **No asymptotic cell-level *P* is paper-facing.** Every model here writes its `P>|z|` column out
   with a header saying so; none enters a family or a figure.
2. **A huge effect confined to ONE animal must not reach significance** — a cell-level test would
   call it overwhelming; the mouse-label randomisation cannot, because that animal's *label* is what
   moves.
3. **Adding cells at a fixed number of mice must sharpen the per-animal estimate and buy no
   treatment-level precision.** Measured, not asserted: over a 100× increase in cells/mouse under a
   true null the mean within-mouse SEM fell ~10× (0.0704 → 0.0071) while the **group-contrast SE
   did not move** (0.098 → 0.105) and the randomisation *P* never approached significance. The
   contrast SE is a *between-mouse* quantity, floored by between-mouse variance, and no number of
   cells reduces it — that gap is the guard, and the run fails if the contrast SE ever starts
   tracking cell count downward.
   Worth recording, because an earlier draft of this section got it wrong: the asymptotic *P*
   printed beside it does **not** shrink either, because `linear_contrast_test` already uses
   `df = n_mice − 1` (§4.1's animal-level convention). The naive cell-level *P* that *would* shrink
   is the MixedLM summary's own `P>|z|` — which is exactly why this suite never reports it.

#### The estimand is conditional — and the rate model is why that is not the whole picture

The paired amplitude estimate is **conditional on cells with measurable amplitude in both epochs**.
That is unavoidable for a paired quantity but is a real restriction: eligibility is selected on
activity in both windows and could itself differ by group. `stats/hierarchical_cell_eligibility.csv`
is therefore a **reported result**, and `fraction_eligible_both` travels with the estimate
everywhere. The path that includes **zero-event cells** is the hierarchical NB count model, which is
one of the reasons it is a required component rather than an optional extra.

#### Both long-format models carry a mouse epoch random slope

The NB rate model and the unpaired amplitude sensitivity model are `group * epoch` models, so
between-mouse variation in the pre→post **change** — the effect being compared across groups — has
nowhere to go without `(0 + post_indicator|mouse)`, and the group × epoch terms would get intervals
that are too narrow with cells acting as replicates for the epoch contrast. That is the same
pseudoreplication in count-model form, so the slope is part of the frozen structure rather than
something added later. Independent (uncorrelated) slope; **no cell-level epoch slope**. The primary
Δ model needs no such term — fit on the within-cell difference, `(1|mouse)` already *is* the
mouse-specific modulation effect.

#### One predefined flow, no fallbacks, no result-dependent branching

`run_hierarchical_cell_analysis` is the **only** gate and it is all-or-nothing: whenever the suite
is invoked, all eleven components in `_HIERARCHICAL_CELL_COMPONENTS` run — there is no
per-component switch, and nothing is activated or omitted because of what another result showed.
(The gate itself defaults to `False`; see the note opening A.7.2.) Those eleven are **analyses and
their diagnostics only** — paired-cell construction, the hierarchical amplitude fit and its
mouse-label randomisation inference, the declared amplitude sensitivity model, the hierarchical
rate fit with its prior/posterior-predictive and convergence diagnostics, and the figures and
reports.

**Implementation validation is not part of a real-data run.** `verify_hierarchical_cell_synthetic`
and `verify_hierarchical_cell_rate_synthetic` plant a known truth in *simulated* data and check
that this machinery recovers it. They validate the **code**, not the experiment, and are
**development tools called by hand** after changing the module — never by
`run_hierarchical_cell_suite`. A run that produces the reported numbers should compute those
numbers and nothing else; the amplitude suite alone also cost ~19 min, most of a real Test_B pass.
The design-3 and design-4 results quoted above come from such a hand run and are recorded here so
the guarantee is on record without re-deriving it every time. Nothing is switched on because
another result was or was not significant. If a specified model fails to fit or fails its
predefined adequacy gate the suite **raises** — no pooled-cell test, no Gaussian approximation, no
quasi-Poisson, no reduced random-effect structure, no clustered-OLS. `fit_mixed_model`'s own
documented clustered-OLS fallback is correct for the other lanes and is explicitly barred here
(clustered OLS over cell rows with 16 clusters is exactly the anti-conservative inference this
block exists to avoid). A **boundary** variance estimate is *not* a failure — it is a statement
about the data, reported explicitly, and distinct from the term having been dropped, which is
checked separately.

NB priors are **frozen before fitting** (Bambi would otherwise pick data-dependent defaults), the
dispersion parameterisation is confirmed empirically against the installed backend, a
prior-predictive check runs **before** the inferential fit, and posterior-predictive checks of the
count distribution, zero fraction and overdispersion **by group × epoch** are part of adequacy —
convergence diagnostics alone are not. The convergence gate's scope is fixed in advance: fixed
effects, dispersion, RE scale hyperparameters, mouse intercepts and slopes and the derived
contrasts are gated; the thousands of nuisance per-cell intercepts are reported in full but do not
define failure. Divergences are gated **globally** at zero — a divergence belongs to the sampler's
trajectory, not to a parameter, so it is never scoped away.

Output is **staged and promoted**: everything is written to `hierarchical_cells__staging/` and the
previous complete directory is replaced wholesale only after every component succeeds. A failure
leaves the previous complete output untouched, a `FAILED.txt` naming the failing component in
staging, and re-raises. Stale outputs cannot survive a partial run.

**This does not replace the mouse-level analysis and the Results section is not rewritten around
it.** `stats/hierarchical_cell_vs_mouse_level_summary.md` places the two side by side, reading every
existing number off the frames the lane already produced, and states explicitly that the smaller
*P* is not automatically preferred. Where the hierarchical analysis is stronger, the supportable
wording is that *a hierarchical cell-level analysis, retaining within-mouse cellular variation while
preserving mouse-level treatment assignment, provided additional evidence that pre-to-post amplitude
modulation differed among groups* — never `n = thousands of cells`.

**Scoped to Test_B by `RECALL_HIERARCHICAL_CELL_SESSIONS`.** No CNO was present at recall.

---

## M. Manuscript draft (Nature style)

**This is the single canonical draft, and it is written from Part A only.** No Part B number
appears in it. The figure guide's Part 2 used to hold a second copy; it now points here, because
the two had already diverged.

Register is Nature's — past tense, effect size with its interval before any *P*, the mouse as the
stated unit of inference.

**Values written `{{ABnn}}` must be read off this run's own output before submission**; §M.3 maps
every code to its file, row and column. Sections that are not this analysis' to write — animals,
surgery, viral constructs, CNO dosing, behavioural apparatus, miniscope hardware, source
extraction parameters — are marked ⟨…⟩ rather than invented here.

### M.1 Methods

**Calcium imaging and source extraction.** ⟨Miniscope model, lens, FOV, illumination, and CNMF-E
parameters.⟩ Imaging was performed at 20 Hz. Regions of interest and their deconvolved activity
traces were extracted with CNMF-E, and cells were registered across sessions with ⟨CellReg
version and parameters⟩. Cross-session correspondence was taken from the registration table
directly; no analysis matched cells by within-session identifier across sessions.

**Calcium event detection.** Events were detected on the deconvolved spike-inference trace (*S*)
of each dorsal CA1 pyramidal cell. A single event was defined as one contiguous run of *S* above a
threshold of 2 s.d., and its amplitude as the integral of *S* over that run. This definition
departs from the more common one — one event per local maximum, quantified by the value at its
peak frame — for a specific reason: summed over a window, the peak-based quantity is exactly the
product of event rate and mean peak height, and therefore cannot distinguish larger events from
more frequent ones. Under the run definition, temporally clustered peaks merge into a single wider
event of larger integral (median run length 4 frames; 11.7% of runs were confined to a single
frame), so event counts are not interchangeable between the two definitions and were recomputed
under this one throughout.

Cells with no detected events were retained in the population event-rate denominator, so that a
manipulation silencing cells appears as a reduced population rate. Per-event amplitude is
undefined for such cells, which are therefore necessarily absent from amplitude analyses; this is
a definitional exclusion rather than missing data.

**Behavioural epochs.** Trial timing was measured per animal from each recording rather than taken
from nominal protocol constants, as recordings were ragged and trial counts varied. Analyses
focused on three duration-matched 20 s windows during trace fear conditioning: a pre-tone baseline
ending at tone onset, the trace interval (tone offset to shock onset), and a post-shock window
beginning at shock offset. The 2 s shock was excluded from all models: at the observed event
rates, expected counts in a 2 s window are dominated by counting noise (P(zero events per cell)
67–91% at 0.05–0.2 Hz).

Because the trace interval on the first trial was 15 s rather than 20 s, analyses were restricted
for each animal to trials in which all three windows were present at the full 20 s duration,
determined from each window's measured exposure rather than from a trial index. Both outcomes were
computed from that same set of animal × trial windows.

**Statistical analysis.** The animal was the unit of inference (*n* = 5 hM3D, 6 hM4D, 6 mCherry).
Cells and trials contributed precision, not replication, and no analysis treated cells as
independent experimental units. Figures showing per-cell distributions display them for
description only; all statistics were computed from per-animal values (SuperPlot convention, Lord
et al., *J. Cell Biol.* 219, e202001064, 2020). The inferential dataset contained one value per
animal per epoch ({{AB00}} rows).

For each animal and epoch, per-event amplitude was summarized by first calculating each active
cell's mean event-run integral across the retained trials, log-transforming this value, and then
averaging across cells within the animal; an exponentiated group contrast on this response is
therefore a ratio of geometric means of the cell-level mean event amplitudes. Population event
rate was calculated for the same trials as the total number of detected events divided by the
corresponding total cell-seconds across all detected pyramidal cells, including cells with zero
events, and was log-transformed before statistical analysis. No animal × epoch had a zero
population rate and no pseudocount was applied.

Per-event amplitude and population event rate were analysed separately using the same linear
mixed-effects model, with DREADD group (mCherry, hM3D or hM4D), epoch (pre-tone, trace or
post-shock), and their interaction as fixed effects and animal as a random intercept
(`log(metric) ~ group * epoch + (1|animal)`; reference levels mCherry and pre-tone, set
explicitly). Within each epoch, hM3D and hM4D were compared with mCherry using linear contrasts of
the fitted model — at the reference epoch the group coefficient alone, and at the trace and
post-shock epochs that coefficient plus the corresponding group × epoch interaction coefficient,
with their covariance — with Holm correction across the two treatment-versus-control comparisons
in that epoch. Amplitude and rate formed separate correction families, and the direct
hM3D-versus-hM4D comparison was in neither. Group × epoch interactions were assessed by joint Wald
tests of the four interaction coefficients. All Wald and contrast inference used
an animal-level denominator degrees of freedom of *n*<sub>animals</sub> − 1 = 16, applied
consistently to both outcomes; no Satterthwaite or Kenward–Roger approximation was applied.

Effects are reported as exponentiated model contrasts (fold change or rate ratio) with 95%
confidence intervals; for amplitude, exponentiation yields the ratio of animal-level geometric
mean event amplitudes, and for population rate a rate ratio. Population-rate effects are
additionally reported as observed absolute differences in events s⁻¹ per cell, which are
descriptive summaries of the animal-level means and carry no separate test. *P* < 0.05 after Holm
correction was considered statistically significant.

As a conservative sensitivity analysis, the six treatment-versus-control simple effects spanning
all three epochs were additionally corrected together by the Holm procedure within each outcome.
This sensitivity correction did not determine the figure annotations or any manuscript
significance statement.

Locomotion and freezing were not covaried: both are post-treatment variables, and conditioning on
them would remove part of the effect being estimated. One cross-registration cell set was primary;
the others were sensitivity analyses, not replications.

Analyses used Python 3.11.15 with statsmodels 0.14.6 (`MixedLM`, REML, L-BFGS), scipy 1.17.1,
numpy 2.4.5 and pandas 3.0.2.

*Supplementary Methods.* A negative-binomial mixed-effects count model with a log(total
cell-seconds) exposure offset, random intercepts for animal and for animal × trial, and jointly
estimated dispersion (Bambi/PyMC; ⟨draws, tuning, chains, convergence diagnostics⟩) was used as a
sensitivity analysis of the population-rate result. Further sensitivity analyses — a cell-level
amplitude model, mouse-label permutation tests of both the group contrasts and the group × epoch
interaction, event-detection threshold sensitivity, event run-structure measures, a group × trial
photobleaching control, and cross-registration subset variants — are described in the analysis
methods documentation.

The analysis above was finalised after substantial prior inspection of this dataset; it is the
main statistical analysis and is not prospectively preregistered. The trace interval's privileged
role in the comparison structure derives from the biological question and the design and predates
this statistical analysis, but the final multiplicity structure was settled after that inspection,
which is why the more conservative six-comparison correction is reported alongside it. With
*n* = 5/6/6 animals the
design has 80% power only for very large standardized effects (Cohen's *d* ≈ 1.8–2.0 pairwise), so
every null is reported with its interval and with what that interval still admits, and none is
presented as evidence of absence.

**Code availability.** Analysis code is available at ⟨repository/DOI⟩.

### M.2 Results

**Chemogenetic modulation of SST interneurons dissociates the size and the frequency of CA1
pyramidal calcium events.**

To ask how SST-interneuron modulation reshapes dorsal CA1 pyramidal output during trace fear
conditioning, we detected calcium events as contiguous supra-threshold runs of the deconvolved
signal and quantified each by its integral rather than its peak, so that a wider event is
distinguishable from a taller one. Because summed event amplitude per second is the exact product
of event rate and per-event amplitude, we analysed the two factors separately, using the same
mouse-level mixed-effects framework for both, and report them together (Fig. 1). All statistics
treat the mouse as the unit of inference (*n* = 5 hM3D, 6 hM4D, 6 mCherry).

The manipulation was effective. Within cells tracked across two same-day linear-track sessions
recorded before and after CNO, hM3D increased per-event amplitude relative to control
(difference-in-differences +0.47 log units); control cells declined by 0.27 log units across the
session pair, consistent with photobleaching, whereas hM3D cells rose by 0.20. The fraction of
cells failing to re-register between the two sessions did not differ between hM4D and control
(77.1% versus 76.8%; hM3D 81.7%), arguing against activity-dependent loss of silenced cells as an
explanation for the hM4D results below.

Chemogenetic modulation of SST interneurons differentially altered the magnitude and frequency of
dorsal CA1 pyramidal-cell calcium events during trace fear conditioning. Within each epoch, hM3D
and hM4D were compared with mCherry controls using model-derived post-hoc contrasts, Holm-
corrected across those two treatment-versus-control comparisons. hM3D mice showed increased
per-event amplitudes relative to controls during the trace interval ({{AB07}}-fold, 95% CI
{{AB08}}–{{AB09}}; Holm-adjusted *P* = {{AB10}}) and the post-shock period ({{AB06}}-fold, 95% CI
{{AB31}}–{{AB32}}; Holm-adjusted *P* = {{AB33}}), whereas the pre-tone comparison did not reach
significance ({{AB05}}-fold; *P* = {{AB30}}). hM4D did not significantly alter per-event amplitude
in any of the three epochs (trace {{AB11}}-fold, 95% CI {{AB12}}–{{AB13}}; *P* = {{AB14}})
(Fig. 1a).

Population event rate showed a complementary pattern. hM4D mice exhibited a reduced event rate
during the trace interval (rate ratio {{AB19}}, 95% CI {{AB20}}–{{AB21}}; absolute difference
{{AB22}} events s⁻¹ per cell; Holm-adjusted *P* = {{AB23}}), whereas the pre-tone
({{AB24}}-fold; *P* = {{AB34}}) and post-shock ({{AB25}}-fold; *P* = {{AB35}}) comparisons did not
reach significance after correction. hM3D did not significantly alter population event rate in any
epoch (trace rate ratio {{AB15}}, 95% CI {{AB16}}–{{AB17}}; *P* = {{AB18}}) (Fig. 1b).

Despite these epoch-wise simple effects, there was no evidence that the magnitude of the treatment
effect differed across epochs, for either per-event amplitude (*F*({{AB02}},{{AB03}}) = {{AB01}},
*P* = {{AB04}}) or population event rate (*F*({{AB27}},{{AB28}}) = {{AB26}}, *P* = {{AB29}}). Thus
the data support differential effects of SST-interneuron excitation and inhibition on CA1 event
magnitude and frequency — hM3D predominantly affecting event magnitude and hM4D predominantly
affecting event frequency — but do not demonstrate that either treatment effect was specific to a
particular conditioning epoch. A negative-binomial count model of the same event counts, fit as a
sensitivity analysis, yielded the same direction of effect for the population-rate result.

The hM3D amplitude signature persisted at recall: it was still present in the post-tone window
48 h after conditioning (1.41-fold, ⟨95% CI⟩) but was no longer detectable at one week (1.25-fold,
⟨95% CI⟩). The one-week interval is wide and this is not evidence that the effect had resolved.
⟨The recall sessions are not yet part of the unified framework — see §9.⟩

A burst-like origin for the larger events is suggested but not established. Supra-threshold runs
were wider in hM3D than in control (5.74 versus 4.90 frames, with per-animal means almost fully
separated), although this difference was not resolved at the present sample size, and the fraction
of runs containing more than one local maximum did not differ between groups (0.112 versus 0.131).
Because temporally clustered events merge into a single run under this event definition, run width
is the measurement that would distinguish genuine bursting from that merging. The amplitude effect
was stable across event-detection thresholds spanning 1.5–3.0 s.d. ⟨coefficient range⟩, excluding
a threshold artifact as its sole source. Per-event amplitude declined monotonically across the
five conditioning trials in every group (~31% by trial 5), consistent with photobleaching; this is
a main effect of trial and cancels in a between-group contrast.

Finally, cellular analyses of this kind are conditional on the neurons that source extraction
detects. The absence of an hM4D amplitude effect cannot exclude changes in neurons that became
undetectable, although the matched cross-session dropout reported above makes such loss unlikely
to account for it.

### M.3 Placeholder lookup

All paths are relative to `PLOTS_DIR/sp_rates_lmm/paper/tfc_amplitude_rate/`. In
`unified_lmm_posthoc_contrasts.csv` a row is keyed by `(outcome, epoch, comparison)`; in
`unified_lmm_interactions.csv` by `outcome`.

| code | value | source |
|---|---|---|
| AB00 | number of rows in the inferential dataset (expect 51) | `stats/unified_lmm_mouse_epoch_values.csv` (row count) |
| AB01 | amplitude interaction *F* | `stats/unified_lmm_interactions.csv` — `amplitude`, `F` |
| AB02 | amplitude interaction df1 (expect 4) | same row, `df1` |
| AB03 | amplitude interaction df2 (expect 16) | same row, `df2` |
| AB04 | amplitude interaction *P* | same row, `p` |
| AB05 | hM3D amplitude ratio, pre-tone | contrasts — `amplitude / pre_tone_matched / hM3D_vs_mCherry`, `ratio` |
| AB06 | hM3D amplitude ratio, post-shock | contrasts — `amplitude / post_shock / hM3D_vs_mCherry`, `ratio` |
| AB07 | hM3D amplitude ratio, trace | contrasts — `amplitude / trace / hM3D_vs_mCherry`, `ratio` |
| AB08 | …its CI low | same row, `ratio_ci_low` |
| AB09 | …its CI high | same row, `ratio_ci_high` |
| AB10 | …its within-epoch Holm-adjusted *P* | same row, `p_holm_epoch` |
| AB11 | hM4D amplitude ratio, trace | contrasts — `amplitude / trace / hM4D_vs_mCherry`, `ratio` |
| AB12 | …its CI low | same row, `ratio_ci_low` |
| AB13 | …its CI high | same row, `ratio_ci_high` |
| AB14 | …its within-epoch Holm-adjusted *P* | same row, `p_holm_epoch` |
| AB15 | hM3D rate ratio, trace | contrasts — `population_rate / trace / hM3D_vs_mCherry`, `ratio` |
| AB16 | …its CI low | same row, `ratio_ci_low` |
| AB17 | …its CI high | same row, `ratio_ci_high` |
| AB18 | …its within-epoch Holm-adjusted *P* | same row, `p_holm_epoch` |
| AB19 | hM4D rate ratio, trace | contrasts — `population_rate / trace / hM4D_vs_mCherry`, `ratio` |
| AB20 | …its CI low | same row, `ratio_ci_low` |
| AB21 | …its CI high | same row, `ratio_ci_high` |
| AB22 | …its absolute difference (events/s/cell) | same row, `absolute_difference_events_per_s_per_cell` |
| AB23 | …its within-epoch Holm-adjusted *P* | same row, `p_holm_epoch` |
| AB24 | hM4D rate ratio, pre-tone | contrasts — `population_rate / pre_tone_matched / hM4D_vs_mCherry`, `ratio` |
| AB25 | hM4D rate ratio, post-shock | contrasts — `population_rate / post_shock / hM4D_vs_mCherry`, `ratio` |
| AB26 | rate interaction *F* | `stats/unified_lmm_interactions.csv` — `population_rate`, `F` |
| AB27 | rate interaction df1 (expect 4) | same row, `df1` |
| AB28 | rate interaction df2 (expect 16) | same row, `df2` |
| AB29 | rate interaction *P* | same row, `p` |
| AB30 | hM3D amplitude, pre-tone — within-epoch Holm-adjusted *P* | contrasts — `amplitude / pre_tone_matched / hM3D_vs_mCherry`, `p_holm_epoch` |
| AB31 | hM3D amplitude ratio CI low, post-shock | contrasts — `amplitude / post_shock / hM3D_vs_mCherry`, `ratio_ci_low` |
| AB32 | …its CI high | same row, `ratio_ci_high` |
| AB33 | …its within-epoch Holm-adjusted *P* | same row, `p_holm_epoch` |
| AB34 | hM4D rate, pre-tone — within-epoch Holm-adjusted *P* | contrasts — `population_rate / pre_tone_matched / hM4D_vs_mCherry`, `p_holm_epoch` |
| AB35 | hM4D rate, post-shock — within-epoch Holm-adjusted *P* | contrasts — `population_rate / post_shock / hM4D_vs_mCherry`, `p_holm_epoch` |

(AB05, AB06, AB24 and AB25 are the pre-tone/post-shock *ratios* already listed above; AB30 and
AB33–AB35 are the adjusted *P* values that now accompany them in the Results.)

**Quote AB33 to enough digits to be unambiguous.** The post-shock hM3D amplitude adjusted *P*
sits just under α; write it as `0.0497`, never rounded to `0.050`, which reads as failing to
reject. The same applies to any other `p_holm_epoch` landing within a rounding step of 0.05.

**The `ABnn` namespace is the paper-facing analysis only.** Every `ABnn` *P* above is either an
interaction *P* or a `p_holm_epoch`. The conservative six-comparison across-epoch sensitivity
values live in a deliberately separate `SENSnn` namespace so that one placeholder can never stand
for two multiplicity families. Quote a `SENSnn` value only in a passage explicitly labelled as the
sensitivity correction (Part B / supplementary), never in the Results sentences above.

| code | value | source |
|---|---|---|
| SENS01 | hM3D amplitude, trace — six-comparison Holm *P* | contrasts — `amplitude / trace / hM3D_vs_mCherry`, `p_holm_six` |
| SENS02 | hM4D amplitude, trace — six-comparison Holm *P* | contrasts — `amplitude / trace / hM4D_vs_mCherry`, `p_holm_six` |
| SENS03 | hM3D rate, trace — six-comparison Holm *P* | contrasts — `population_rate / trace / hM3D_vs_mCherry`, `p_holm_six` |
| SENS04 | hM4D rate, trace — six-comparison Holm *P* | contrasts — `population_rate / trace / hM4D_vs_mCherry`, `p_holm_six` |
| SENS05–SENS12 | the same for the pre-tone and post-shock rows of both outcomes | contrasts — corresponding row, `p_holm_six` |

`stats/paper_results_summary.md` prints every one of these already formatted — the `ABnn` values
in Part 1, the `SENSnn` values only under Part 2's *Sensitivity — conservative six-comparison Holm
correction*; it is the fastest place to read them.

Still ⟨angle-bracketed⟩ and not from this analysis:

| Placeholder | Where to read it |
|---|---|
| Recall 95% CIs | `Test_B*/stats/post_tone_amplitude.txt` |
| Threshold-sensitivity coefficient range | `TFC_cond/stats/threshold_sensitivity.csv` |
| Sampler settings and convergence for the NB sensitivity model | `TFC_cond/stats/secondary_rate.txt` |
| Imaging hardware, CNMF-E and CellReg parameters | not in this analysis |

Claims that must not drift back into the draft:

- **Any Part B p-value, HDI or q-value as a headline result.** The cell-level Holm *p*, the
  Bayesian HDIs and the permutation *q*s are sensitivity output. They may be mentioned as such;
  they may not be reported as the main test.
- **"Tonic", "global shift", "the effect is equivalent across epochs", or "73% of the trace effect
  is already present at baseline" as an inferential statement.** A null interaction is not
  evidence of equivalence, and a ratio of two uncertain effect estimates is not a decomposition of
  one effect into components.
- **A trace-specific amplitude effect,** unless the interaction actually supports it.

---

# Part B — sensitivity, supplementary and historical analyses

**Nothing below is a paper-facing result.** Every model, family and p-value in Part B is retained
because it tests whether Part A's conclusions survive a different modelling choice, weighting or
distributional assumption — and because the decision record is worth keeping. None of it supplies
an asterisk, an interval or a P-value to any manuscript sentence or paper figure.

The evidential-tier vocabulary used below applies **within Part B's internal output only**. It is
not the manuscript's framework.

---

## 1. What this replaced, and why

The original `sp_rates` section ([sections.py](../caban/sections.py), `run_sp_rates`) produced
**~205 uncorrected three-group one-way ANOVAs** — every combination of session × behaviour period ×
cross-registration subset × metric, each with ANOVA-gated Tukey. Counting: TFC_cond alone was
3 periods × 5 mappings × 2 metrics × 2 panel variants = 60, plus 10 whole-session; Test_B and
Test_B_1wk 56 each; LT1/LT2 10 each; Test_A/Test_A_1wk 2 each.

Three structural problems, not just multiplicity:

1. **No correction across the ~205 omnibus tests.** Tukey only limits multiplicity *within* one
   three-group ANOVA.
2. **The unit of analysis changed between panels.** The `-avg` variant plotted one point per
   *behaviour period* (n=3–5) — periods were never independently assigned to a DREADD group, so
   that is pseudoreplication, not a treatment test.
3. **Overlapping cross-registration subsets were presented as if they were replications.** They are
   nested views of the same 17 animals.

`sp_rates_lmm` replaces the inferential backbone with **three Holm-corrected confirmatory tests**
and re-files everything else as declared secondary or sensitivity analysis. The old panels are not
deleted — their status changes to a sensitivity/multiverse archive.

> The family was **two** tests until the epoch split (§3.4) gave the post-shock window a real
> 20 s definition and made `post_shock_vs_baseline` a co-equal member. Anything in this document
> quoting a two-test family or a `p_holm` computed against one is stale — the primary trace
> amplitude p_holm moved 0.018 → **0.027** for this reason alone, with an unchanged p_raw.

---

## 2. The external methodological review

A Deep Research report was commissioned mid-design (PDF supplied by the user; not in the repo). It
is a good report and most of it was adopted. It is **not** authoritative where it conflicts with
this document — several of its assumptions about the data turned out to be wrong, and one of its
central recommendations was deliberately inverted.

### 2.1 What it recommended

| Issue | Review's recommendation |
|---|---|
| Experimental unit | Mouse is the independent treatment unit; cells/trials add precision, not replication |
| Primary TFC outcome | **Trace-interval event RATE** as the confirmatory centrepiece |
| Model | Negative-binomial mixed model on counts with `log(exposure)` offset, mouse × trial level |
| Epoch analysis | One integrated `group × epoch` model, not per-epoch ANOVAs |
| Multiplicity | 1–2 primary hypotheses under Holm; remainder exploratory under BH-FDR |
| Locomotion | Primary estimand **unadjusted** for freezing/speed (they are post-treatment); movement-conditioned analyses reported separately |
| Cell subsets | One primary definition; the others are sensitivity analyses, never replications |
| Amplitude | Mean amplitude **conditional on a detected event** as a *secondary* property; summed-amplitude-per-second confounds frequency with amplitude |
| Detection bias | Real; reference-seeded / joint cross-session extraction is preferable to intersecting independently detected ROI lists |
| Nulls | Report rate ratios **and** absolute differences with 95% CIs; a non-significant result at n=5/6/6 is weak evidence of absence |
| Software | `glmmTMB` first choice, `brms` second; explicitly **not** `statsmodels.MixedLM` for counts |

Useful quantitative content worth keeping (all from the review, none of it implemented as code):

- **Power.** At α=0.05 / 80%: detectable Cohen's *d* ≈ **1.91** (5 vs 6), **1.80** (6 vs 6),
  **2.02** (5 vs 5); at α≈0.025, 2.18 and 2.04. Three-group omnibus at N=17: Cohen's *f* ≈ **0.84**.
  These are very large standardized effects.
- **Rate-ratio conversion.** `RR_detectable ≈ exp(d · σ_log r)`. At *d*≈1.9: mouse-level CV 30% →
  RR 1.75; 50% → 2.45; 70% → 3.32.
- **Repeated trials.** For ICC ρ over *m* trials, variance of the mean scales as `ρ + (1−ρ)/m`.
  Five trials give an effective standardized-effect gain of ×1.67 (ρ=0.2), ×1.29 (ρ=0.5),
  ×1.09 (ρ=0.8). **5 trials × 5 epochs ≠ 25 replicates** — epochs are conditions, not repeats.
- **2 s shock window.** At 0.05–0.2 Hz, P(zero events per cell) is 90.5%–67.0% for 2 s, vs
  36.8%–1.8% for 20 s. Poisson counting CV `1/√(λT)` ≈ 158–316% at 2 s. This is why shock is
  excluded from the confirmatory factorial model.
- **Bibliographic correction the review makes:** Yu et al. 2022 is the mixed-effects primer;
  the hierarchical-bootstrap paper is **Saravanan, Berman & Sober 2020**. Aarts et al. is the
  nested-data / pseudoreplication reference.

### 2.2 Where we followed it

- Mouse as the unit of inference, everywhere, without exception.
- Small confirmatory family under Holm (now three tests — §4.1); everything else declared
  secondary/sensitivity, with the BH-FDR family the review asked for now implemented (§4.2).
- Integrated epoch model rather than per-epoch tests.
- Locomotion **not** covaried in the primary estimand.
- One primary cell set (`full`); cross-registration subsets demoted to sensitivity.
- Zero-event cells retained in every rate/fraction-active denominator.
- Shock (2 s) excluded from the confirmatory factorial model.
- Trace interval privileged as the primary epoch, on the review's own *a priori* grounds
  (hippocampal dependence of trace conditioning) rather than because it gave a good result.
- "Locked confirmatory reanalysis", **not** prospective preregistration — the endpoint was chosen
  after prior inspection of this dataset. The METHODS file says so explicitly.
- Detection-bias limitation stated in the review's own terms (activity-dependent ROI inclusion).

### 2.3 Where we deliberately departed — **do not reopen these**

**(a) Amplitude is the PRIMARY BIOLOGICAL endpoint; rate is secondary. The review recommended the
opposite.**

This is the single most-relitigated point. The user's scientific claim is specifically about
**burst magnitude** — that hM3D produces larger per-event Ca²⁺ influx — not about firing more
often. The review recommended rate primary partly on the grounds that deconvolved `S` amplitude is
not calibrated, which is true and is handled by relabelling the measurement (§3.1), not by changing
the endpoint. The user's position, verbatim: *"It should be precisely the other way around
(amplitude is primary; frequency secondary)."*

**This ordering is scientific, not statistical.** Since the unified models (§A) the two outcomes
receive *identical* statistical treatment: the same model, the same interaction test, the same
contrasts, and the same within-epoch Holm families (§A.4). "Secondary" means amplitude
is the endpoint the hypothesis is about; it does not mean rate is analysed differently or shown
less prominently. Rate is reported with a full interval in every panel alongside amplitude.

> ⚠️ **Superseded argument, kept for the record.** This section used to add: *"the rate model is
> Bayesian (LOO/HDI), and you cannot Holm-correct across a mixed Bayesian/frequentist family."*
> That is still true **of the negative-binomial count model**, which remains Bayesian and remains
> outside every frequentist correction family. It is no longer true of the **paper-facing rate
> endpoint**, which is the Gaussian LMM on log(population rate) described in §A: it has a
> frequentist P-value and carries a Holm-corrected contrast family of its own (§A.4). Do not
> cite this paragraph as a reason the rate endpoint cannot carry a corrected p-value.

**(b) Within-cell epoch differencing instead of `(1|mouse) + (1|mouse:cell)`.**

The review is right that a mouse random intercept alone is insufficient for a cell × trial × epoch
table. But the nested version is ~8,000 cell levels over ~100k rows in `statsmodels` — slow and
likely singular. Instead, for each cell we compute `Δ = mean log amplitude[epoch X] − mean log
amplitude[pre_tone]` and fit `Δ ~ group + (1|mouse)`. **The cell random intercept cancels by
construction**, which is both honest and tractable, and reuses the paired logic already validated
in `compute_lt1_lt2_amplitude_delta`. Cost: a cell must have ≥1 event in *both* epochs to
contribute; report how many qualify.

**(c) Bambi/PyMC instead of `glmmTMB`/`brms`.**

Python-only constraint. `statsmodels.mixedlm` is Gaussian-only with no NB-GLMM path — the review is
correct to reject it for counts. Bambi (on PyMC) gives a genuine NB mixed model with dispersion
estimated jointly, which is what `glmmTMB` would do. No R runtime exists on this machine.

**(d) Per-event INTEGRAL, not peak amplitude.** See §3.1. The review's critique of
"summed-amplitude-per-second" is accepted and that metric is retired as an *amplitude* measure —
but the replacement is a per-event integral, not the peak.

### 2.4 Where the review was factually wrong about this dataset

- **Trace duration is not ~35 s.** From [sessions.py:1261-1264](../caban/sessions.py#L1261):
  `tone_onsets_def = [185, 420, 660, 900, 1140]`, `tone_duration = 20`,
  `shock_onsets_def = [220, 460, 700, 940, 1180]` → trace = **[15, 20, 20, 20, 20] s**. Trial 1 is
  short because `tone_onsets_def[0] = 185` rather than an intended 180 (flagged as an accident in
  the source comment). Handled automatically: exposure is computed from each trial's actual frames.
- **`S` is not impulse-like.** The review implicitly treats events as point processes. Measured on
  real data: only **11.7%** of supra-threshold runs are single-frame; median 4 frames, mean 5.44,
  max 118. This is why the per-event integral is meaningful at all (§3.1).
- **hM4D detection dropout is not elevated.** The review flagged activity-dependent ROI loss as
  likely to bias hM4D upward. Measured LT1→LT2 dropout: hM3D 81.7%, hM4D 77.1%, mCherry 76.8% —
  hM4D is indistinguishable from control. This is a reportable **negative** result that makes the
  hM4D null interpretable. The larger finding is that dropout is ~77–82% in *all* groups.

---

## 3. Measurement definitions

### 3.1 Event definition: per-event integral of a contiguous run

`find_spikes_ca` / `find_spikes_ca_S` ([utilities.py](../caban/utilities.py)) detect one event per
**local maximum** and report the `S` value at that single peak frame. Summed over a window that is
exactly `rate × mean peak height` — mathematically incapable of separating "bigger events" from
"more events".

This analysis uses `find_event_runs_ca` / `find_event_runs_ca_S` instead:

- **one event = one contiguous supra-threshold run** of `S`
- **amplitude = `sum(S[run])`**, the per-event integral, not the peak
- `n_local_maxima` is returned as a diagnostic (how many events the legacy detector would have
  reported for that run), reusing `find_spikes_ca` itself so the two can never disagree

**Consequences that must be stated, not glossed:**

- A run with two local maxima is **one** event here, not two. Event *counts* therefore differ from
  the legacy convention (~50,318 → 45,857 on one mouse; 4.5% of runs are multi-peak). Rate
  endpoints are recomputed under this same definition — never mix with legacy counts, and never
  compare new rate numbers directly against old `sp_rates` panels.
- Because temporally clustered peaks **merge**, part of the observed "high amplitude + low rate
  among active cells" pattern in hM3D is a mechanical consequence of the event definition. The
  run-structure evidence (§4.4) is what distinguishes genuine bursting from a merging artifact.
- **"Bursting" is the interpretation; "larger deconvolved event run integral" is the
  measurement.** Keep that distinction in all text and figure captions.

### 3.2 Zero-event cells

Kept in every rate / fraction-active denominator (the codebase convention — a manipulation that
silences cells must show up as reduced population rate, which only happens if silenced cells stay in
the denominator). Amplitude is **undefined** for a cell with zero events, so those cells are
excluded from amplitude computations — definitional, not missing data. Say so in captions, or the
amplitude panel looks like it silently dropped cells.

### 3.3 The decomposition identity

At the cell-window level, exactly:

```
summed S per second  =  event rate  ×  mean per-event amplitude
log(total S/sec)     =  log(rate)   +  log(mean amplitude)
```

`report_decomposition_additivity()` performs the arithmetic check. It is **not** a fitted model or a
significance test — a proper SE for the sum needs the joint covariance of two separately-fit models.

**Weighting rule (state it):** aggregated across cells the identity holds with **event-weighted**
mean amplitude, not cell-weighted. The per-cell models use **cell-weighted** means (one row per
cell). Only the additivity check is event-weighted.

### 3.4 Epochs

**`post_shock` used to mean the entire 198 s inter-trial interval.** It now means a 20 s response
window from shock offset. Any result predating that split averaged the post-shock response over
roughly ten times its own duration and **must not be compared with results computed after it**.
The full ITI is still available under the name it should always have had, `iti`.

| epoch | window | role |
|---|---|---|
| `pre_tone` | 35 s ending at tone onset | locked confirmatory reference |
| `pre_tone_matched` | 20 s ending at tone onset | exposure-matched baseline; **overlaps `pre_tone`** |
| `tone` | 20 s | descriptive |
| `trace` | tone offset → shock onset (15 s trial 1, 20 s trials 2–5, per §2.4) | primary |
| `shock` | 2 s | **excluded** from everything confirmatory (review's Poisson numbers, §2.1) |
| `post_shock` | 20 s from shock offset | co-primary |
| `post_shock_late` | 20 s, beginning 90 s after shock offset | descriptive; early-vs-late control |
| `iti` | shock offset → next tone onset (198 s) | defined, not used by the pipeline |

**Four epoch sets exist and the distinctions are load-bearing.** Getting these confused is how
the sampler once ran for 113 minutes instead of one.

| constant | members | why this set |
|---|---|---|
| `TFC_EPOCHS` | everything the event table computes | carries **both** baselines |
| `TFC_DISJOINT_EPOCHS` | `pre_tone`, `tone`, `trace`, `post_shock`, `post_shock_late` | safe to model **jointly**; excludes `pre_tone_matched`, which overlaps `pre_tone` in time |
| `TFC_RATE_MODEL_EPOCHS` | `pre_tone`, `tone`, `trace`, `post_shock` | what the **rate model** may use — see §4.5 |
| `TFC_MATCHED_PROFILE_EPOCHS` | `pre_tone_matched`, `trace`, `post_shock` | the three **complete** 20 s windows — what the joint epoch test profiles over (§4.6). Excludes `post_shock_late`: it is ragged, and the joint test needs every mouse to contribute a complete profile |

#### 3.4a Exposure-matched trials

`TFC_MATCHED_EPOCHS` are called exposure-matched because each is *defined* as 20 s — but **trace
on trial 1 is only 15 s** (§2.4). Since `aggregate_over_trials` sums exposure across trials, the
pooled comparison is ~95 s of trace against ~100 s of `pre_tone_matched`.

That 5% mismatch does not touch mean per-event **amplitude**, which is a per-event quantity —
which is why the confirmatory contrasts can and do keep the unmatched 35 s `pre_tone`. It does
matter for the **duration-sensitive** components: `P(active) = 1 − e^(−λT)` rises with `T` at a
fixed underlying rate, and conditioning on `N > 0` makes rate-among-active duration-dependent too.

`restrict_to_exposure_matched_trials(df_fine, epochs)` keeps the `(mouse, trial)` pairs where every
requested epoch is present **at the measured 20 s**. Two rules it follows deliberately:

- **Derived from measured `exposure_seconds`, never from a trial index.** Hard-coding "trials 2–5"
  would encode the nominal protocol timing, which §8 forbids and correction #14 records the cost
  of. Testing the window instead drops trial 1 *because its trace is short* and drops a truncated
  final trial *because its window is missing*, under one rule that survives either assumption
  changing.
- **Matched per `(mouse, trial)` pair, not per trial index** — recordings are ragged, so a global
  index filter would cut good trials from mice whose recordings ran long. Same reasoning as
  `restrict_to_shared_trials`.

Coverage goes to `stats/matched_decomposition_trial_coverage.csv`; **report it with the estimate.**

Putting both time-overlapping baselines on one epoch factor is a duplicated-data bug, not an
inefficiency: 20 s of every baseline enters the likelihood twice, inflating the baseline's
precision, corrupting the epochs-within-trial variance component, and leaving NUTS on a
near-collinear ridge. `build_mouse_trial_epoch_rate_table` filters at its own source rather than
trusting callers. **Preserve that.**

#### The early-vs-late post-shock contrast

Puhger et al. 2024 (*iScience* 27:109035) report no bulk CA1 calcium response during the trace
interval at all, a large sustained response to the footshock, memory impairment when CA1 is
silenced 0–40 s after the shock, and **no** impairment when the same silencing arrives 140 s
after it. That result is a *contrast* between early and delayed, so reproducing it needs a late
window — which is what `post_shock_late` is for.

`post_shock` vs `pre_tone` is **not** that contrast. Each epoch is pooled across trials *before*
differencing, so the reference is a mixture: trial 1's baseline is shock-naive, trials 2–5's sit
163 s after the preceding shock, the indices don't align, and trial 5's shock has no `pre_tone`
counterpart at all. The descriptive contrast `log_amp(post_shock) − log_amp(post_shock_late)` is
within-trial at aligned indices with no naive window in the reference.

Two things about it that are easy to get wrong:

- **It is DESCRIPTIVE.** It is in neither multiplicity family, spends no alpha, and is reported as
  an estimate with a 95% interval, never a significance verdict. Its output is named
  `descriptive_*` rather than `coprimary_*` so the status is legible from the filename. The
  confirmatory family stays at exactly three.
- **The late window does not exist on every trial, and both sides are trial-matched.** The final
  trial's ITI ends with the *recording*, and real sessions stop as little as **20.5 s** after the
  last shock offset — barely `post_shock` itself. On such a trial no late window fits at *any*
  onset value. `get_epoch_frames` returns `None` there (a definitional absence, as for `pre_tone`
  on trial 1) while still **raising** for `post_shock`, which every trial must supply.
  `restrict_to_shared_trials` then cuts both sides to the trials where both exist, per mouse, and
  writes the surviving counts to `stats/descriptive_early_vs_late_trial_coverage.csv`. **Report
  that coverage with the estimate** — observed range is 3–5 of 5 trials per mouse.

**Why 90 s and not Puhger's 140 s.** It maximizes trial coverage; it does not rescue the truncated
final trial, which nothing can. What the constant controls is the *mid*-session trials, where a
shorter onset can only ever admit more trials than a longer one. Raising it toward 140 s buys
protocol fidelity at the cost of trials — check the coverage file first.

Test_B / Test_B_1wk post-tone is pinned to **20 s** explicitly via
`get_testb_epoch_frames(..., post_tone_duration_s=20.0)`. Neither existing default is the recall
analog of trace: the function's own default is 35 s, and `TestBSession.post_tone_onsets/offsets`
runs to the *next tone onset* (~200+ s ITI).

### 3.5 Cohort gaps

`Test_B_exp_frames['G07'] = []` and `dpath_Test_B_1wk_day['G15'] = ''` — both hM4D. So **each
recall session runs at hM4D n=5, N=16, and it is a *different* mouse missing from each**. The
48 h → 1 wk comparison is therefore not on a fixed cohort. This is session *existence*, distinct
from the mapping-availability lists (`mappings_all_Test_B_G15` etc.), which describe which crossreg
subsets exist for a mouse that **is** present.

---

## 4. Statistical architecture (internal / sensitivity)

*The paper-facing architecture is §A: two models, two joint Wald interaction tests, twelve
model-derived contrasts, and a primary Holm family of the two trace comparisons per outcome with
the six-comparison correction kept as sensitivity output. Everything in §4 is the internal
machinery that now serves as sensitivity evidence and as the decision record.*

### 4.1 Internal confirmatory family — Holm across exactly three tests

| key | Endpoint | Model |
|---|---|---|
| `trace_amplitude` | Trace-period per-event amplitude (primary) | `log_amplitude ~ C(group)`, cell-level, trials pooled, `(1\|mouse)` |
| `trace_vs_baseline` | Trace epoch specificity (co-primary) | Within-cell `Δ = trace − pre_tone`, then `Δ ~ C(group) + (1\|mouse)` |
| `post_shock_vs_baseline` | Post-shock epoch specificity (co-primary) | Same, `Δ = post_shock − pre_tone` |

The two co-primary members go through **one** code path,
`fit_and_report_epoch_delta(..., is_confirmatory=True)`, so they cannot diverge in cell selection,
model specification, or what gets reported — a difference between them has to be a difference in
the data. That function also serves the descriptive early-vs-late contrast with
`is_confirmatory=False`, which is not cosmetic: it decides whether the stats file *claims* its p
enters the Holm family, and writing "CONFIRMATORY" over a descriptive estimate would be a false
inferential claim in a file someone will quote.

Its output paths key on **both** epochs whenever the reference is not `pre_tone`, so two contrasts
sharing a response epoch cannot silently overwrite each other. The two confirmatory calls keep
their historical single-epoch filenames byte-identical.

All use `fit_mixed_model` ([single_unit_common.py](../caban/single_unit_common.py)), which falls
back to mouse-clustered OLS on convergence failure or boundary degeneracy — a documented fallback,
not a silent error swallow. Omnibus tests use `joint_wald_test`.

**Adding a fourth member is a design change, not a code change.** The family is locked; new
epochs go in as descriptive (that is what `post_shock_late` did).

### 4.2 Secondary family

Event rate (Bambi NB-GLMM, mouse × trial × epoch, `log(total cell-seconds)` offset, LOO vs a
no-interaction model); fraction active; Test_B / Test_B_1wk post-tone amplitude; `group × trial`
(photobleaching control); run structure; threshold sensitivity.

**BH-FDR is implemented** (`build_secondary_fdr_table`, α=0.05, written to
`stats/secondary_fdr_family.csv`/`.txt`). Members are accumulated into `secondary_pvalues` as each
is fit and corrected in one pass at the end: `group_x_trial_interaction`, `run_structure_*`, and
`{Test_B,Test_B_1wk}_post_tone_amplitude`.

Members are `group_x_trial_interaction`, `run_structure_*`, `{Test_B,Test_B_1wk}_post_tone_amplitude`,
and **`epoch_specificity_{component}`** — one per decomposition component (§4.6).

**The family is frequentist secondary tests only.** Explicitly outside it, by design:

- the three confirmatory omnibus tests (own Holm family, different error rate — a test in two
  families is corrected twice);
- the Bambi NB rate model — posterior contrasts, HDIs and ELPD-LOO, no p-value to correct.
  Manufacturing one so a Bayesian result can be folded into a frequentist FDR family is a category
  error, not a conservative choice;
- all descriptive and sensitivity output, **including the early-vs-late post-shock delta**.

### 4.3 Small-n inference — mouse-label permutation

`mouse_label_permutation_test` shuffles **group labels across the 17 mice** (never cells) and
recomputes an arbitrary statistic. ~5.7 M distinct 5/6/6 relabellings exist, so this is Monte Carlo
(default 20,000 draws), not exhaustive enumeration.

`make_contrast_stat(..., weight='cell'|'mouse')`. **The equal-mouse-weighted contrast is the primary
reported point estimate** — cell-weighted estimates let mice with more active cells dominate, and
cell inclusion is itself post-treatment and activity-dependent (hM3D contributes ~561
amplitude-active cells/mouse vs ~379–389 for the others). Inference is valid either way; only the
point estimate's weighting changes.

**Prefer the permutation p-value to the clustered-OLS p-value.** Cluster-robust SEs are
anti-conservative at 17 clusters. Triangulate three ways every time: model estimate, 17-point
mouse-level plot, permutation CI.

#### Two nulls: `restrict_to_groups`

`mouse_label_permutation_test` shuffles labels across **every** mouse in `mice_per_group` by
default, which tests the **global null** — "no group differs from any other". Passing
`restrict_to_groups=(a, b)` restricts exchangeability to those two groups, testing the
**pairwise null** — "a does not differ from b", assuming nothing about the third group.

**These can differ materially, and the difference is not a rounding detail.** Under the global
null a permuted `hM3D` bucket can contain mCherry mice, so mCherry's between-mouse spread enters
the null distribution even though mCherry appears nowhere in the statistic. When the *control* is
the most variable group — which in this dataset it is, trace fraction-active SD 0.131 for mCherry
against 0.032 for hM3D — that widens the null and makes the observed contrast look less extreme.

Measured on the trace fraction-active hM3D-vs-hM4D contrast:

| null | p |
|---|---|
| global (all 17 exchangeable) | 0.099 |
| pairwise (11 DREADD mice) | 0.027 |
| exhaustive enumeration, 462 splits | 0.026 |

The Monte Carlo pairwise value converges to the exact answer, which is the correctness check on
the option.

**The default stays global**, because every reported number in this module was computed under it
and changing the default would silently move published results. The run-structure stats file now
prints **both** — the global-null p is the BH family member, the pairwise-null p is a labelled
sensitivity value in no family. Entering both would correct one question twice and halve the
family's power. Pass `restrict_to_groups` explicitly when the claim is specifically "group A
differs from group B".

### 4.4 Run structure — the bursting evidence

Direct, height-independent burst signature: run **width**, **local maxima per run**, **fraction of
multi-peak runs**, each with mouse-level points and permutation tests, plus threshold sensitivity at
`thres ∈ {1.5, 2.0, 3.0}`. Measured on real data (trace epoch):

| group | impulse_frac | mean width | p99 | max |
|---|---|---|---|---|
| hM3D | 0.112 | 5.74 | 26 | 118 |
| hM4D | 0.138 | 4.85 | 22 | 119 |
| mCherry | 0.131 | 4.90 | 22 | 110 |

hM3D runs are **wider with the same median (4.0)** — a fattened right tail. Per-mouse means separate
almost completely (hM3D min 5.12 > hM4D max 5.09; only mCherry's G16 at 5.25 overlaps).

### 4.5 The rate model takes four epochs, not five — `TFC_RATE_MODEL_EPOCHS`

Being disjoint in time is **necessary but not sufficient** to enter the NB-GLMM.
`post_shock_late` is disjoint (verified pairwise against all other windows, across trial indices
too) and is in `TFC_DISJOINT_EPOCHS`, but `build_mouse_trial_epoch_rate_table` filters to the
narrower `TFC_RATE_MODEL_EPOCHS`, holding it out for three reasons:

1. **It has no role here.** It is a descriptive window carrying a within-cell *amplitude*
   contrast. The rate model is the prespecified secondary *rate* endpoint; adding a descriptive
   epoch enlarges a model that makes claims without contributing to any of them.
2. **It is ragged in a way that matters specifically here.** The window is absent on final trials
   whose recording stops early, so its rows skew toward low trial indices — and `trial` is a
   continuous covariate in this very model, so its epoch dummy correlates with `trial`. The
   amplitude contrasts, differenced within cell and trial-matched, never face this.
3. **Empirically it makes NUTS pathological.** Measured on this model: 4 epochs / 340 rows samples
   in **~27 s**; 5 epochs runs **>280 s without finishing** — and does so even when the fifth
   level is made artificially balanced across all trials, so the cost is the extra level in an
   already weakly-identified `group × epoch` interaction, *not* the raggedness.

**This is not the duplicated-data ridge of §3.4** — that one is about overlapping windows, and
these windows are verifiably disjoint. Different failure, same symptom, and conflating them will
send you looking for an overlap that isn't there. Holding this set at four also means the reported
rate results are the same model that produced them before `post_shock_late` existed, so they need
no reinterpretation.

### 4.6 The joint `group × epoch` specificity test — one p, not three compared by eye

**The problem it fixes.** The confirmatory family gives three numbers: trace amplitude
(p_holm = 0.027, rejected), `trace_vs_baseline` (0.998), `post_shock_vs_baseline` (0.935). Reading
those against each other — "significant in trace, not significant in the epoch contrasts,
therefore the effect is/isn't epoch-specific" — is the **difference-of-significance fallacy**. A
difference between two p-values is not a test of anything. "Does the group effect *change* across
epochs" is a distinct question and needs its own single test.

`fit_and_report_epoch_interaction` is that test:

| | |
|---|---|
| Profile | `build_mouse_epoch_profile` — per mouse, mean elevation of one component over `pre_tone_matched` in each of `trace`, `post_shock`. A 17 × 2 matrix. |
| Components | **One test per decomposition component**, all four of `_DECOMPOSITION_COMPONENTS` |
| Epochs | `TFC_MATCHED_PROFILE_EPOCHS` — the three complete 20 s windows. `post_shock_late` is excluded (ragged, same reason as §4.5). |
| Trials | Exposure-matched only (§3.4a). |
| Statistic | `make_group_epoch_interaction_stat` — standardized sum of squared difference-of-differences, `Σ_g Σ_e ((mean_g[Δ_e] − mean_ctl[Δ_e]) / sd_pooled(Δ_e))²` |
| Inference | `mouse_label_permutation_test` — the 17 group labels shuffle, each mouse keeps its whole profile |
| Status | **Secondary BH-FDR family** (§4.2), not confirmatory |

**One test per component, not one overall.** A single pooled test across all four would answer
"did *any* component's profile change", which nobody asks, and would be driven by whichever
component has the largest between-mouse spread. Per component, each test annotates its own row of
the decomposition grid.

**The four p-values are strongly dependent** — the components are one exact identity computed over
overlapping cells. BH is valid under positive dependence so the q-values stand, but they describe
one decomposition and must not be counted as four independent findings.

**`fraction_active` has no paired variant.** It is a proportion computed *over* a mouse's cells,
so there is no per-cell value to pair; `build_mouse_epoch_profile` forces `paired=False` for it.
Its paired and unpaired tests are the same test, not two.

`_DECOMPOSITION_COMPONENTS` is the single definition of what a component *is*, shared by the grid
and these tests, so a row and the test annotating it cannot drift apart.

Four things about it that are easy to get wrong:

- **Differencing against the reference epoch removes the epoch main effect by construction**, so
  what is left is only the interaction. A mouse with uniformly high amplitude contributes a flat
  profile of zeros no matter how high — which is exactly why an equal shift across epochs gives a
  null here and an epoch-specific effect would not. Verified on synthetic data: a planted
  trace-only elevation gives p = 0.0005, a planted all-epoch elevation of the same size gives
  p = 0.61.
- **The statistic must be SCALAR.** `mouse_label_permutation_test` accumulates into
  `np.empty(n_perm, dtype=float)`; a vector-valued `stat_fn` raises on assignment. Squaring and
  summing also makes the test properly omnibus and directionless.
- **Standardizing by the per-epoch pooled SD is not cosmetic.** Without it whichever Δ has the
  larger scale dominates the sum and the test silently becomes a test about that one epoch. The SD
  is recomputed under each permutation, so it is part of what is being permuted.
- **The statistic is non-negative**, so `mouse_label_permutation_test`'s two-sided
  `|null| ≥ |observed|` rule is an upper-tail omnibus test. That is correct here, not a bug.
- **An epoch with zero between-mouse variance contributes 0, never `NaN`** — see correction #18.
  A `NaN` returned into the permutation null is not a neutral value there; it is a false
  negative in the "as extreme" count and therefore a false *positive* in the p-value.

**It does not join, replace, or shrink the confirmatory family.** That family stays at exactly
three. Re-forming it now that its two epoch members are known to be null would move the primary
`p_holm` from 0.027 back toward 0.009 and would rightly read as outcome-driven. The review that
prompted this test says so itself.

A **paired** (within-cell triple intersection) and an **unpaired** (per-epoch active cells)
variant are both run and both reported. Permutation validity never depended on the cell pairing —
only the 17 labels move — so disagreement between them is evidence the triple intersection selects
a special subpopulation, not that either is invalid.

---

## 5. Sensitivity results as of the last full run

**None of the numbers in this section is a paper-facing result.** They are the internal/sensitivity
analyses' own output, retained as the decision record and as robustness evidence. The manuscript's
numbers come from Part A's two models and live in
`paper/tfc_amplitude_rate/stats/unified_lmm_posthoc_contrasts.csv` and
`unified_lmm_interactions.csv`. Nothing here should be quoted alongside those as if it were an
alternative primary result, and the unified models may reach different decisions — especially
under the six-contrast Holm family — which is expected and is not a reason to prefer these.

**Internal confirmatory family (Holm across 3) — historical primary:**

| test | p_raw | p_holm | |
|---|---|---|---|
| `trace_amplitude` | 0.0089 | **0.027** | reject |
| `trace_vs_baseline` | 0.998 | 1.0 | — |
| `post_shock_vs_baseline` | 0.935 | 1.0 | — |

- **Historical primary — trace amplitude: hM3D +0.435 log units (~1.5× control), p_holm = 0.027.**
  hM4D null (+0.092, p=0.60). The permutation p is also 0.027. ⚠️ **This is not the manuscript's
  amplitude result and must not be quoted as one.** It is a cell-level model on a different epoch
  definition and a different unit of aggregation, retained as a sensitivity check. The reported
  amplitude effect is the unified mouse-level LMM contrast and its `p_holm_epoch` (§A.4).
- **Both co-primary epoch-specificity tests are NULL.** Neither the trace window nor the
  post-shock window shows a group-specific within-cell elevation over baseline.

> `p_holm` for the primary was **0.018** under the old two-test family. `p_raw` did not move;
> only the family size did (§1). Do not mix the two numbers across document versions.

**The co-primary nulls matter and are easy to get wrong.** An earlier version reported
p = 1.2 × 10⁻¹¹ from a `group × epoch` model fit on 100,264 cell × trial × epoch rows with only a
mouse random intercept — that was pure pseudoreplication. Corrected, these tests give **no
evidence that the amplitude effect differed between epochs**; the estimates are elevated at every
epoch. That is not evidence that the effect is identical, global or tonic across them, and no
figure, caption or text may imply a trace-specific effect either. The trace interval's privileged
status rests on prior anatomy and
behaviour, not on this result.

**The joint `group × epoch` test (§4.6) has now run, and every component is null:**

| component | p_raw | q (BH) |
|---|---|---|
| `epoch_specificity_amplitude` | 0.266 | 0.717 |
| `epoch_specificity_fraction_active` | 0.388 | 0.717 |
| `epoch_specificity_rate_active` | 0.511 | 0.738 |
| `epoch_specificity_population_rate` | 0.637 | 0.753 |

Paired (within-cell) values, 5,291 cells active in all three matched epochs, 3–4 of 5 trials per
mouse. The unpaired sensitivity variant agrees for amplitude (p = 0.568).

**This is a direct test of epoch dependence rather than an inference from three nulls.** For every
component, there is **no evidence that the treatment effect differed across epochs** — which is
not evidence that it was identical, global, tonic, or present throughout conditioning. Any
apparent left-to-right gradient on the decomposition grid is unsupported by these tests, and if
the same drift appears in *both* DREADD groups it is an epoch main effect (a property of the trial
structure), not a treatment effect.

The same holds for the post-shock window: with a properly-windowed 20 s definition (§3.4) its
interaction test is *also* null. `plot_epoch_profile` and `epoch_delta_forest` exist so the
per-epoch estimates can be read side by side rather than taken on trust — a figure showing only
the trace contrast cannot distinguish "no evidence of epoch dependence" from "no effect". Note
that a null interaction and a resolved trace-period simple effect (§A.4) answer different
questions and are not in conflict.

**Dissociation (the main scientific story):** hM3D → amplitude; hM4D → rate. hM4D's
rate-among-active-cells is reduced vs control while its amplitude is flat; the Bambi model agrees
independently (`hM4D:trace` rate coefficient −0.278, HDI [−0.527, −0.047]).

**Manipulation check (LT1 drug-free → LT2 CNO, within-cell):** hM3D +0.474 vs control, p = 5×10⁻⁷.
Control cells *drop* 0.27 log units LT1→LT2 (time/bleaching); hM3D reverses that to +0.20.

**Recall:** Test_B (48 h) hM3D +0.343, p<0.001 — persists. Test_B_1wk +0.220, p=0.26 with a wide CI
— underpowered, **not** demonstrated absence. Note this diverges from the behavioural phenotype
(hM3D keeps elevated post-tone freezing at 1 wk), so it cannot be reported as supporting that link.

**Trial effect:** amplitude declines monotonically across trials (−0.098, −0.187, −0.287, −0.372;
~31% by trial 5), consistent with photobleaching. Does not threaten a between-group contrast unless
the *rate of decline* differs by group — hence the `group × trial` check.

**Within-trial post-shock delta (descriptive, §7 #17):** null, F(2,16) = 0.698, p = 0.512, on
5,026 of 7,410 trace-active cells. Note this fit fell back to **mouse-clustered OLS** — the mixed
model's fixed-effect SEs were not all finite. Documented fallback, not a silent failure (§4.1),
but quote the permutation or the interval, not the clustered p (§4.3).

**Per-epoch contrasts across the three matched windows** (from the paper lane's own output,
`paper/tfc_amplitude_rate/tfc_amplitude_rate_by_epoch_contrasts.md`; these reproduce the
`TFC_cond` numbers exactly — the re-cut moved nothing):

| | pre_tone_matched | trace | post_shock |
|---|---|---|---|
| Amplitude, Exc/Ctl | 1.38× [1.02, 1.86] | 1.55× [1.14, 2.12] | 1.47× [1.00, 2.17] |
| Amplitude, Inh/Ctl | 1.14× [0.77, 1.69] | 1.18× [0.86, 1.63] | 1.18× [0.81, 1.73] |
| Population rate, Exc/Ctl | 0.89× [0.51, 1.56] | 0.78× [0.53, 1.15] | 1.10× [0.68, 1.79] |
| Population rate, Inh/Ctl | 0.63× [0.36, 1.12] | 0.51× [0.33, 0.76] | 0.65× [0.40, 1.06] |
| Fraction active, Exc/Ctl | +0.013 [−0.117, +0.142] | +0.047 [−0.094, +0.188] | +0.104 [−0.011, +0.220] |
| Fraction active, Inh/Ctl | −0.054 [−0.166, +0.058] | −0.095 [−0.236, +0.046] | −0.024 [−0.142, +0.093] |

Two readings this table settles.

**The dissociation is clean.** hM3D moves amplitude and not rate; hM4D moves rate and not
amplitude. Neither group's *off-target* row has an interval excluding the null in any window.
That is the shape of the result — though the manuscript's version of it is Part A's, and prefers
"dissociation" to "double dissociation".

**Descriptive note on the epoch profile.** In log units the hM3D amplitude contrast is
**+0.323 at pre-tone, +0.440 at trace, +0.389 at post-shock**. The apparent trace-specific
increment is +0.117 log units against a CI half-width of ~±0.31 — indistinguishable from zero,
consistent with §4.6's joint test being null.

> ⚠️ **Do not turn the 0.323/0.440 ≈ 73% ratio into an inferential statement.** It is a ratio of
> two uncertain effect estimates, not a decomposition of one effect into a baseline component and
> a conditioning component, and it does not license "the effect is tonic". A null interaction is
> *no evidence that the effect differed across epochs* — nothing more. Whether a genuinely
> conditioning-naive, drug-on baseline shows the elevation is a different measurement, and it is
> not the one made here. If the manuscript ever wants a mechanistic tonic-CNO claim, it needs that
> measurement first.

> The defensible framing is that the estimated effect was positive in all three windows and the
> interaction gave no evidence that it differed between them. The trace interval keeps its
> privileged status from prior anatomy and from the behavioural phenotype, not from this
> measurement.

**A star-versus-interval trap, recorded because it is live in the current figures.** On
`tfc_amplitude_rate_by_epoch`, amplitude is starred at trace and nowhere else — yet the pre-tone
amplitude interval, 1.38× [1.02, 1.86], **excludes 1**. It carries no star only because the stars
are Holm-corrected across two contrasts and the intervals are uncorrected (the companion
`_contrasts.md` says so). "Starred here, not there" is not a difference between epochs. This is
the exact reading §4.6 exists to prevent, and this figure is where someone will attempt it.

**hM4D's rate reduction is not mainly a recruitment effect.** Fraction active at trace is only
−0.095 [−0.236, +0.046] while population rate is 0.51× [0.33, 0.76], so most of the halving must
sit in `rate_active`. Read that row off `decomposition_grid_contrasts.md` before writing the
mechanism up — "hM4D reduces how often active cells fire" and "hM4D recruits fewer cells" are
different claims and the decomposition can distinguish them.

**Report the absolute differences with the ratios.** hM4D's rate effect is −0.030 to −0.039
events/s against a control base of roughly 0.07/s. "Halved" is accurate and, alone, oversells it:
that is about one event per 25–30 s per cell.

### 5.1 The direct hM3D-vs-hM4D contrasts

New on the decomposition grid (§6 rule 5). These exclude the null in several panels where
**neither** vs-control interval does:

| | Exc/Ctl | Inh/Ctl | **Exc/Inh** |
|---|---|---|---|
| Fraction active — trace | +0.047 [−0.094, +0.188] | −0.095 [−0.236, +0.046] | **+0.142 [+0.080, +0.204]** |
| Fraction active — post_shock | +0.104 [−0.011, +0.220] | −0.024 [−0.142, +0.093] | **+0.129 [+0.070, +0.188]** |
| Population rate — trace | 0.78× [0.53, 1.15] | 0.51× [0.33, 0.76] | **1.54× [1.15, 2.06]** |
| Population rate — post_shock | 1.10× [0.68, 1.79] | 0.65× [0.40, 1.06] | **1.68× [1.28, 2.22]** |
| Rate \| active — post_shock | 0.96× [0.66, 1.39] | 0.67× [0.46, 0.97] | **1.43× [1.12, 1.82]** |

The mechanism is the variance decomposition in §6 rule 5: mCherry is this dataset's most variable
group, both vs-control contrasts pay for that noise, and the direct contrast does not.

**Read §5.2 before reporting any of these.**

### 5.2 What may be claimed from these internal results

*This section governs Part B's output only. Part A's twelve model contrasts are reported with
their estimates and intervals — and, for the two primary trace comparisons per outcome, their
Holm-adjusted P values — and need none of this machinery; that is the point of the
simplification.*

This section exists because the question *"can I report this as statistically significant?"* has
come up once already and will again. Every internal output sits in exactly one of three tiers, and
the tier — not the p-value — determines how it may be written up. **None of these tiers reaches
the manuscript: a Part B result is at most a sensitivity statement, whatever its tier.**

| tier | members | how to report |
|---|---|---|
| **Confirmatory** | the three in §4.1 | "significant", with `p_holm` |
| **Secondary** | the BH-FDR family (§4.2), incl. the four epoch-specificity tests | "significant" with `q`, described as secondary |
| **Exploratory** | everything else — all 16 direct Exc/Inh contrasts, the descriptive epoch deltas, the early-vs-late contrast, threshold sensitivity, cross-registration subsets | **effect estimate + interval only.** Never "significant" |

**The direct DREADD-vs-DREADD contrasts are tier 3.** Five reasons, and a better p-value fixes
none of them:

1. **Post-hoc.** The contrast was added *after* looking at the grid and asking whether hM3D
   exceeded hM4D. The whole three-tier architecture exists to prevent a contrast invented that way
   from re-entering as confirmatory. This analysis is a *locked confirmatory reanalysis* (§2.2),
   which makes the distinction load-bearing rather than pedantic.
2. **Uncorrected, and there are 16.** Deliberately in no family (§4.2).
3. **Neither vs-control contrast is significant.** The licensed claim is "hM3D and hM4D differ
   from each other", **not** "hM3D increases fraction active". The difference cannot be attributed
   to either group.
4. **It is not this design's question.** `PANEL_HOLM_FAMILY = 'vs_control'` because the experiment
   asks whether each DREADD differs from *its own control*. Neither group here is a control.
5. **The estimate moves with the trial subset** — +0.142 on exposure-matched trials, +0.090 on the
   full trace epoch. Two defensible subsets, two numbers: a researcher-degrees-of-freedom flag.

A worked example, since the numbers are tempting. Trace fraction-active, hM3D vs hM4D:

| | value |
|---|---|
| Welch CI (what the grid draws) | +0.142 [+0.080, +0.204] |
| Global-null permutation | p = 0.099 |
| Pairwise-null permutation | p = 0.027 |
| Exhaustive enumeration (462 splits) | p = 0.026 |

The test holds up under the appropriate null (§4.3). It is *still* tier 3, because reasons 1–5 are
untouched by it.

**Wording that is defensible:**

> In an exploratory decomposition, hM3D and hM4D differed in the fraction of active cells during
> the trace interval (+0.14, 95% CI [0.08, 0.20], uncorrected); neither group differed
> significantly from mCherry.

**Wording that is not:**

> ~~hM3D significantly increased the fraction of active cells during trace and post-shock.~~

The cheaper route is usually to fold such a result into the dissociation that *is* supported —
hM3D → amplitude and hM4D → rate, both from the unified LMMs and their `p_holm_epoch` values
(§A.4) — and cite the fraction-active gap as *consistent with* it, which costs no alpha. If it
needs to be a finding, declare it prospectively for the next cohort.

---

## 6. Plotting architecture

Three layers. Per CLAUDE.md's dedup rule, **extend the shared primitive rather than adding a
parallel one.**

**Shared primitives — [caban/single_unit_common.py](../caban/single_unit_common.py):**

- `draw_superplot_triplet(ax, cell_values_per_group, mouse_means_per_group, group_order,
  group_colours, stat_fn, ylabel=, cell_size=1.5, cell_alpha=0.18, mean_size=90,
  jitter_width=0.34, yscale='auto')` — SuperPlot (Lord et al. 2020, *J Cell Biol* 219:e202001064):
  every cell semi-transparent and coloured by mouse (`_mouse_colour_shades`), per-mouse means as
  large black-edged markers. Cells share one per-mouse x-offset so a mouse's cloud and its mean
  group visually.
- `_resolve_superplot_yscale` — `'auto'` → `'log'` (all positive), `'symlog'` (non-negative with
  exact zeros; `linthresh` = smallest positive value), `'linear'` (any negative).
- `ecdf_panel`, `grow_ylim_for_bracket_headroom`, `bracket_ylim`, `save_fig` (PNG+SVG pair).

**[caban/analysis.py](../caban/analysis.py):** `_draw_violin_triplet`, `do_pairwise_holm_plot` /
`do_anova1_plot` (the `stat_fn` interface), and `barplot_annotate_brackets` (in
[utilities.py](../caban/utilities.py)).

**[caban/sp_rates_lmm.py](../caban/sp_rates_lmm.py) panel wrappers:**
`_mouse_values_per_group`, `_draw_mouse_violin_panel`, `_draw_cell_superplot_panel` (takes the
*same* tidy per-cell frame, so the two are interchangeable per panel), `_save_panel`.

`_draw_superplot_panel_grid(panels, row_ylabels, col_titles, ...)` draws a rows × columns grid of
SuperPlot panels sharing one y-scale per row, saves it, and writes the companion contrasts file.
Cells are `_GridPanel` tuples carrying their own frame, column, axis scale and contrast scale, so
the driver knows nothing about what is being plotted. **Two figures already use it over different
pairs of factors** — `plot_paper_epoch_distributions` (component × epoch) and
`plot_conditioning_phase_amplitude` (epoch × conditioning phase) — which is why it exists rather
than the second one copying the first. A third grid over some other pair of factors should go
through it too.

Entry points: `plot_primary_trace_amplitude`, `plot_epoch_profile`, `plot_amplitude_ecdf`,
`plot_amplitude_p90`, `plot_decomposition`, `plot_manipulation_check`, `plot_run_structure`,
`plot_threshold_sensitivity`, `plot_effect_forest`, `plot_decomposition_grid`, `plot_example_traces`,
`plot_width_vs_height_matched_examples`, `plot_conditioning_phase_amplitude` (§6.0a), the
paper lane's `plot_paper_epoch_distributions` / `render_paper_tfc_amplitude_rate` (§6.1), and the
recall lane's `plot_recall_modulation` / `plot_recall_prepost_trajectories` (§A.7.1). The first of
those reuses `_draw_mouse_violin_panel` with `_precomputed_stat_fn`, so its brackets come from the
fitted models rather than from a second test on the plotted values.

`write_decomposition_contrasts_markdown` takes `figure_has_stars`, `no_star_note` and `title`.
The `figure_has_stars=False` default note is written **about the four-component decomposition**
and says so in as many words; a star-free figure whose panels are one component across epochs or
trial phases must pass its own `no_star_note` rather than inherit a description that is false of
it. Same principle as `fit_and_report_epoch_delta`'s `is_confirmatory` (correction #12): a
companion file that mis-describes its own figure is a false claim in a document people quote.

### Four plotting rules that are not negotiable

1. **Unit of inference is the mouse; unit of display may be the cell.** `draw_superplot_triplet`
   passes **only** the per-mouse means to `stat_fn`; the cell cloud never reaches it. A regression
   test asserts a SuperPlot's p-value equals a standalone mouse-level `do_pairwise_holm_plot` on the
   same data — keep it passing. This is the one way these figures could silently reintroduce
   pseudoreplication.
2. **Axis scale is load-bearing.** A linear axis on these heavy-tailed quantities crushes all 17
   mouse means into the bottom 6–13% of the panel — that is why a first SuperPlot attempt was
   reverted. **But pass `yscale='linear'` for an already-log-transformed column** (e.g.
   `log_amplitude`): `'auto'` sees all-positive values and applies a *second* log, plotting
   log10(ln(x)).
3. **Brackets only work on linear axes.** `barplot_annotate_brackets` scales offsets as a fraction
   of `ylim[1]−ylim[0]` in *data* coordinates — meaningless in log space. On non-linear axes,
   `stat_fn` is called with `annotate=False` and the p-values are rendered as a text block. Same
   numbers either way.
4. **Display order is `DREADD_DISPLAY_ORDER = ['mCherry','hM3D','hM4D']`** (Ctl, Exc, Inh) on every
   panel — CLAUDE.md. Do **not** confuse with `GROUP_ORDER = ['hM3D','hM4D','mCherry']`, which is
   model dummy-coding and `stat_fn`'s positional argument order and must not change.
5. **hM3D-vs-hM4D is not readable off two vs-control intervals.** "Red is above the line, blue is
   below it, therefore they differ" compares two contrasts that *share* mCherry as their
   reference, by eye, with no interval for the comparison being made.
   `_panel_contrasts(..., include_exc_vs_inh=True)` computes it (via
   `mouse_contrast_ci(reference='hM4D')`), and the grid draws it as a third grey point per panel.
   It stays **off** on the standalone panels: `PANEL_HOLM_FAMILY` is `'vs_control'` because that
   is where this design's question sits, and the contrast spends no alpha there.

   **It is not systematically wider than the vs-control contrasts** — its variance is
   `Va/na + Vb/nb` against their `Va/na + Vc/nc`, so it drops the control's contribution and keeps
   both DREADDs'. Which way it goes depends on whether the *control* is the noisy group, and
   **in this dataset it is**: trace fraction-active between-mouse SD is 0.131 for mCherry vs 0.032
   for hM3D. So the direct contrast is frequently the tightest of the three and excludes the null
   in several panels where neither vs-control interval does. Real, and worth noting — but these
   16 intervals are **uncorrected and in no declared family**, so treat them as effect estimates
   to follow up, not as findings.

---

### 6.0a Early vs late conditioning — `conditioning_phase_amplitude`

Every epoch contrast in this module compares windows **within** a trial. That leaves an
orthogonal question unanswered: the epoch tests give no evidence that the amplitude effect
differed between windows *within* a trial (§4.6), but is the effect present from the first trial —
a property of the drug being on board — or does it **develop** as conditioning proceeds? `plot_conditioning_phase_amplitude` splits the five trials
into early (1–2) and late (3–5) and draws the amplitude contrast in each, for `pre_tone` and
`trace`.

- **Descriptive. Spends no alpha, in neither family.** The formal version of this question is
  already asked and corrected: `group_x_trial_interaction` is a secondary BH-FDR member. This
  figure makes that result visible; it is not a second test of it, and it draws **no stars**
  (`annotate='none'`) because comparing stars across the two columns is the
  difference-of-significance fallacy.
- **The forest is where it is read**, not the distribution panels. Four intervals per epoch —
  Exc/Inh × early/late — and the question is whether a group's two points sit in the same place.
- **Amplitude only, deliberately.** The rate analog cannot be drawn honestly here:
  rate and fraction active are duration-sensitive, so they need exposure-matched trials, and
  `restrict_to_exposure_matched_trials` drops trial 1 (15 s trace) — which would reduce the
  *early* column to trial 2 alone, gutting the half of the figure the question rests on.
  Amplitude is a per-event quantity and keeps every trial.
- **`pre_tone` (35 s), not `pre_tone_matched`.** Amplitude is duration-insensitive, the longer
  window gives more events per cell, and it is the reference the confirmatory contrasts already
  use — so these numbers sit on the same scale as §5's.
- **Read the contrast, never the level.** Amplitude declines ~31% by trial 5 (photobleaching).
  That is a main effect of trial: it moves every group down together and cancels in a contrast.
  A drop between the two columns is expected and says nothing about the manipulation. The
  companion `_contrasts.md` states this where someone quoting a number will see it.

Validated on synthetic data with planted ground truth: a trial-invariant group effect gives early 1.65 /
late 1.60 (overlapping), a ramping one of the same final size gives 1.12 / 1.36 (separated). The
figure distinguishes the two hypotheses it claims to — worth keeping, since a figure that would
show flat points either way would be worse than none.

The distribution grid is drawn by the shared `_draw_superplot_panel_grid` (§6), the same
primitive the paper lane's figure uses; only the `_GridPanel` contents differ.

**The observed result belongs here once the run reports it.** §5 records that the amplitude
contrast is comparable in the pre-tone, trace and post-shock windows, which is a statement about
*epochs within a trial*. Whether the pre-tone elevation is there from trial 1 is the separate
question this figure answers, and it is not yet filled in.

`CONDITIONING_PHASES` uses **1-based trial numbers**, since that is how the protocol is
discussed; the event table's `trial` column is the session's 0-based `trial_idx`.
`split_by_conditioning_phase` converts once, at its single point of use, and raises if any mouse
has no trials in a phase — otherwise the two columns would rest on different cohorts while
looking like a within-cohort comparison.

### 6.1 The paper lane — `sp_rates_lmm/paper/tfc_amplitude_rate/`

Everything above describes the **internal record**: three evidential tiers, two multiplicity
families, ~15 figures. That is the right output for a lab that has to answer "did you account for
the 15 s first trace?" — and it is **not** a paper. A methodological review of this document made
the point directly: once the original structural problems (pseudoreplication, ~205 uncorrected
ANOVAs, shifting units of analysis) were fixed, further inferential machinery stopped buying
anything. *There is no prize for constructing the most elaborate possible inferential framework.*

The paper lane is the answer to that. **It is not a re-cut of the internal output any more — it is
where the paper-facing analysis is fit** (Part A, §A). Everything it draws comes from the two
unified mouse-level models and their contrast table.

| | |
|---|---|
| Entry point | `render_paper_tfc_amplitude_rate(...)`, last call in `run_sp_rates_lmm` |
| Models | `build_mouse_epoch_unified_table` → `fit_unified_group_epoch_model` ×2 → `unified_posthoc_contrasts` |
| Figures | `tfc_amplitude_rate_by_epoch` (distributions + stars), `tfc_decomposition_forest` (the same contrasts as estimates) |
| Numbers | `stats/unified_lmm_*.csv/.txt`, and `stats/paper_results_summary.md` (Part 1 = paper, Part 2 = sensitivity) |
| METHODS | `sp_rates_lmm_paper_methods.md`, alongside the full internal template |

It also reformats two things it did not fit: `summarize_rate_group_epoch_contrasts` turns the
already-fitted NB model's posterior draws into per-epoch rate ratios, and the manipulation-check
contrasts are recomputed for the summary. Both appear only under Part 2 of the summary.

Decisions it embodies, so they are not re-argued:

- **Two rows, not three or five.** `population_rate` and `amplitude` (`PAPER_COMPONENT_KEYS`).
  `fraction_active` was dropped when the unified models arrived: it has no unified mouse-level
  model, so including it would put a second, differently derived inferential source on a paper
  figure — the exact heterogeneity this design removed. `rate_active` and the total amplitude-rate
  are diagnostics of the identity, not separate biological claims. All three stay in `TFC_cond/`.
- **No Exc/Inh contrasts.** `include_exc_vs_inh=False`, and `_precomputed_stat_fn` returns NaN for
  that pair so no bracket can be drawn. §5.2 explains why those intervals are exploratory; the
  defensible sentence is that the two DREADDs show divergent recruitment profiles while neither is
  claimed to differ from the other.
- **Three columns, sharing one y-scale per row.** `pre_tone_matched | trace | post_shock`,
  exposure-matched trials, both rows from the same frame. Independently autoscaled columns would
  let a reader take an epoch difference straight off the axis limits; `sharey='row'` makes that
  comparison honest. Whether the effect actually differs between epochs is not read off the
  columns at all — it is the interaction, on the forest's row labels.
- **Stars on the distributions, none on the forest.** Conventional, and what a normal paper does.
  Both come from the same twelve rows of `unified_lmm_posthoc_contrasts.csv`: the distribution
  figure shows the decision, the forest shows the effect size and what its interval still admits.
- **Natural-scale display on both rows.** The models are fit on logs; the figure plots raw per-cell
  values on a log axis with the marker at `exp(fitted response)`. Cells and markers must never sit
  on mismatched scales — see `_PAPER_ROW_DISPLAY`.

`plot_decomposition_grid` gained `contrast_payloads=`, `interaction_note=`, `subtitle=`,
`contrasts_note=` and `row_height=` alongside the earlier `components=`/`include_exc_vs_inh=`/
`epoch_labels=`, all defaulting to the internal behaviour — the paper forest is that same function
given a different set of contrasts, not a second implementation of it. Likewise
`_draw_cell_superplot_panel` gained `stat_fn=`/`mouse_means_override=`, and `_GridPanel` three
optional fields.

**Nothing on a paper figure computes its own statistic.** `_precomputed_stat_fn` ignores the data
arrays it is handed and returns the two looked-up adjusted p-values;
`_assert_markers_match_model` hard-fails if a drawn marker is not the corresponding row of
`unified_lmm_mouse_epoch_values.csv`.

`docs/sp_rates_lmm.md`'s Part B stays the internal record. **Do not try to make the manuscript
resemble it.** Its value is that a reviewer's question already has an answer.

---

## 7. Corrections history

Everything below was found and fixed. Listed so nobody re-fixes, re-introduces, or re-argues them.

| # | Problem | Resolution |
|---|---|---|
| 1 | `joint_wald_test` used observation-level denominator df for cluster-robust fits | `n_groups` is now a **required** argument; `df2 = n_groups − 1`. Primary trace p went 0.0016 → **0.0089**, matching statsmodels' own `Prob(F-statistic)`. Also fixed the same pre-existing bug in `pca_state_metrics._lmm_holm_pairs` — its omnibus p-values changed. |
| 2 | Rate-model exposure offset used `'first'` (one cell's window) while summing counts across cells | `'sum'` → total cell-seconds. Mattered because cell count is group-correlated, so the omitted `log(n_cells)` leaked into the group fixed effect rather than being absorbed by the mouse intercept. |
| 3 | Recall figures carried a hardcoded "Trace-period amplitude" title | `title` parameter threaded through. |
| 4 | Co-primary understated its uncertainty (p = 1.2e-11) | Replaced with within-cell differencing (§2.3b). Now p = 0.998. |
| 5 | Primary point estimate was implicitly cell-weighted | `weight='mouse'|'cell'`; equal-mouse-weighted is now primary. |
| 6 | SuperPlot on a linear axis was unreadable | `yscale='auto'`; smaller/fainter markers. |
| 7 | Amplitude panel was double-logged | Explicit `yscale='linear'` for pre-logged columns. |
| 8 | `compute_lt1_lt2_amplitude_delta` asserted equal `unit_id` across LT1/LT2 | Wrong invariant — see §8. Now asserts equal *length* and pairs by row position. |
| 9 | Section took ~50–60 min | `find_spikes_ca`'s per-frame Python loop vectorised (**22×**, output bit-identical, verified on 3,008 adversarial cases); `make_contrast_stat` precomputes per-mouse arrays outside the permutation closure (**85–205×**). Now well under 10 min. |
| 10 | `post_shock` meant the whole 198 s ITI, diluting a ~20 s response by ~10× | Split into `post_shock` (20 s response) and `iti` (the old meaning). `post_shock_vs_baseline` became the third confirmatory member, moving the primary `p_holm` 0.018 → 0.027. |
| 11 | Fitting both time-overlapping baselines on one epoch factor | Sampler ran **113 min instead of ~1**. `TFC_DISJOINT_EPOCHS` introduced and enforced inside `build_mouse_trial_epoch_rate_table`. |
| 12 | `fit_and_report_epoch_delta` hardcoded "CONFIRMATORY" and keyed output filenames on `response_epoch` alone | Gained `is_confirmatory`; paths key on both epochs when the reference isn't `pre_tone`. Without this the first descriptive reuse would have both overwritten `coprimary_epoch_delta_post_shock.txt` and stamped a false inferential claim on it. |
| 13 | `post_shock_late` in the rate model | 4→5 epoch levels took NUTS from ~27 s to >280 s without finishing, *not* from any overlap. `TFC_RATE_MODEL_EPOCHS` added (§4.5). |
| 14 | The last ITI was assumed ~117 s from the nominal protocol timing | It is data-dependent and can be **20.5 s**. Assumption was written into code comments and METHODS before being measured; corrected, and the early-vs-late contrast is now trial-matched rather than relying on a fixed trial count. |
| 15 | Epoch specificity was being read off three separate p-values compared against each other — the difference-of-significance fallacy | One joint `group × epoch` mouse-label permutation test (§4.6), in the secondary BH-FDR family. The confirmatory family is unchanged at three members. |
| 16 | `TFC_MATCHED_EPOCHS` were exposure-matched by definition but not in the pooled data: trial 1's trace is 15 s, so trace pooled to ~95 s against `pre_tone_matched`'s ~100 s | `restrict_to_exposure_matched_trials` (§3.4a), applied to the joint test and the decomposition grid. Derived from measured exposure, not a trial index. |
| 17 | `post_shock_vs_baseline` pools each epoch across trials before differencing, so its reference mixes trial 1's shock-naive baseline with four post-shock ones at unaligned indices | `compute_epoch_delta_table(..., pair_within_trial=True)` differences within `(cell, trial)` then averages to one row per cell. **Descriptive** — a second, independent route to the problem `post_shock_late` also addresses, not a replacement for the locked confirmatory member. |
| 18 | **`mouse_label_permutation_test` counted non-finite null draws as "not extreme"** — `abs(nan) >= abs(observed)` is `False`, so every `NaN` a degenerate `stat_fn` returned pushed the p-value *down*, toward `1/(n_perm+1)` | Measured on a deliberately degenerate fixture: **p ≤ 0.05 on 87% of true nulls.** Fixed in two places — the statistic now contributes 0 for a zero-variance epoch instead of `NaN`, and the permutation test computes p over the finite draws only, raising if the observed statistic is non-finite or if >50% of draws are. Anti-conservative and indistinguishable from a real result, so worth knowing it was ever possible. Affects any `stat_fn` that can return `NaN`, including the run-structure ones. |
| 19 | The grid's per-row epoch-specificity number was drawn before the BH pass, so it could only show a raw p while its own stats file said to report the corrected value | `plot_decomposition_grid` moved to after `build_secondary_fdr_table` and takes `interaction_q`. |
| 20 | Every permutation test shuffled all 17 labels, so a two-group contrast was tested against a null containing the *third* group's variance — and mCherry is this dataset's most variable group | `restrict_to_groups` added (§4.3). Not a bug: the global null is a legitimate hypothesis and stays the default so no reported number moves. But it is not the pairwise null, and on the trace fraction-active hM3D-vs-hM4D contrast the two give p = 0.099 vs 0.027. Run-structure output now prints both. |
| 21 | **Per-panel bracket headroom COMPOUNDS under `sharey='row'`.** `annotate_pairwise_brackets` reserves the top 20% of its own axes for the bracket stack; on a shared-y row every panel reserves again on the limits the previous one already grew, so three panels leave `0.8³ ≈ 51%` of the row to the data and the clouds sit squashed in the bottom half against a band of empty axis | `_draw_superplot_panel_grid` resets each row to its own pooled data range and calls `reserve_top_fraction` **once**, after the whole row is drawn. Brackets are positioned in axes fractions, so they follow the new limits rather than being orphaned by them. Headroom is reserved only when `annotate='stats'` — a star-free grid was otherwise leaving a fifth of every panel empty for annotations that do not exist |
| 22 | Column titles on a bracket-annotated grid were drawn **through** the topmost asterisks — both the title and the bracket stack live above the axes, and `set_title`'s default pad is zero | Top-row titles are re-set with `pad=26` after the panel is drawn. Cosmetic, but it made the first real render of the paper figure unreadable at exactly the point a reader looks first |
| 23 | The paper figure's two rows were annotated from three unrelated frameworks | Amplitude had a cell-level frequentist LMM, rate a Bayesian NB model with HDIs, the asterisks came from a per-panel Welch/Holm test, the intervals from equal-mouse-weighted Welch CIs, and epoch specificity from a permutation statistic. No number on the figure traced to a model. Replaced by the unified analysis (Part A): one mouse-level model per outcome, one interaction test each, twelve planned contrasts, and `unified_lmm_posthoc_contrasts.csv` as the only source of a paper estimate, interval, P or asterisk. All the old machinery is retained as sensitivity output. |
| 24 | Post-hoc multiplicity on the paper figure was per-panel and of a different kind from the models | Made model-derived: Holm across model contrasts rather than per-panel Welch tests, with the conservative six-comparison across-epoch correction retained as `p_holm_six` sensitivity output. Decisions may differ from the old per-panel stars; that is the correction working, not a regression. *(This entry is HISTORICAL in one respect: the intermediate step it introduced — a privileged trace-only primary family `p_holm_trace` / `is_primary_trace_contrast` — was itself superseded by row 26. Those columns no longer exist.)* |
| 25 | The amplitude row plotted `log_amplitude` cells against a `mouse_mean_log_amplitude` marker on a linear axis | Both rows now display natural-scale values on a log axis, with the marker at `exp(fitted response)` — the geometric mean amplitude, and the population rate. `_assert_markers_match_model` hard-fails if a drawn marker is not the model's own value. |
| 26 | The trace-only primary family made six visually equivalent panels statistically asymmetric | Only the trace column carried asterisks, pre-tone and post-shock were described as "not tests", and the paper Methods needed a long multiplicity-philosophy passage to justify it. Replaced by the symmetric rule: within EACH epoch the two treatment-vs-control comparisons are Holm-corrected together (`p_holm_epoch`), so all six panels run the same procedure and the Methods reads like an ordinary analysis. `p_holm_trace` and `is_primary_trace_contrast` were removed outright. **No model, contrast, raw P-value or interaction test changed** — only the grouping of the corrections and what the figure is allowed to annotate. The trace interval's biological importance is now argued in the Introduction/Results rather than encoded in the multiplicity structure. |
| — | *Not a correction, recorded so it is not re-litigated:* an earlier draft of §6 rule 5 and the METHODS claimed the direct DREADD-vs-DREADD contrast is "generally wider" than a vs-control one. **That is wrong.** Its variance is `Va/na + Vb/nb` against `Va/na + Vc/nc` — it drops the control's contribution, so it is wider or narrower depending on which group is noisiest. Here the control is noisiest and the direct contrast is often the tightest of the three. |

---

## 8. Gotchas that have already bitten

- **Cross-registration correspondence is ROW ORDER, never `unit_id` equality.** The same physical
  cell has a *different* `unit_id` in each session — each session reads its own column of the
  crossreg table. This is why `get_actual_cells_from_df_session` must never sort (see its
  `# OMG NO!!!` comment). Pair across sessions by zipping the two `S_idx` lists positionally.
- **LT1, LT2 and TFC_cond share one crossreg object** (`TFC_cond_crossreg`), which is what makes
  positional pairing between them valid. **Test_B / Test_B_1wk use a different one**
  (`TFC_B_B_1wk_crossreg`) — never pair those positionally against TFC_cond.
- **`build_epoch_event_table`'s `'cell'` column is session-local.** Never join it across sessions.
- **Import order:** `caban.decoder` must be imported before `caban.analysis` (deferred cross-module
  block at the bottom of `decoder.py`). Any standalone script needs this.
- **Environment:** `/Users/vsekulic/miniforge3/envs/caban/bin/python`. The repo `.venv` has no
  numpy. `bambi`/`pymc`/`arviz` live in the conda env.
- **Never run `caban.sections.run_*` from an agent session** (AGENTS.md) — the user drives it from a
  live notebook kernel. Tell them which cell to run. Synthetic-data renders and read-only snippets
  are fine and are how bug #7 was caught.
- **Synthetic test fixtures must give LT1/LT2 different `unit_id`s per session**, or they encode the
  same false assumption as bug #8 and cannot catch it. That is exactly how it slipped through.
- **Never derive session timing from the nominal protocol constants.** `tone_onsets_def` /
  `shock_onsets_def` / `light_onsets = [0, 1299]` describe the *intended* session. The last ITI in
  particular ends at `post_shock_offsets[-1] = miniscope_exp_fnum[stop_idx]` — the real recording
  stop — and has been observed at **20.5 s** where the nominal figure suggests ~117 s. Trial count
  varies too. Measure per mouse; do not reason from the constants (correction #14).
- **A slow sampler has two distinct causes.** Overlapping windows on one epoch factor (§3.4) and
  simply too many epoch levels (§4.5) present identically. Check disjointness *first* — it is
  cheap and settles which one you have — before assuming an overlap.
- **`get_epoch_frames` returning `None` is not the same as an error.** It means the window does
  not exist for that trial (`pre_tone` on trial 1, `post_shock_late` on a truncated final trial),
  and `_iter_event_windows` skips it. That is definitional absence, not a swallowed failure — but
  it does mean any epoch that can be absent needs trial-matching before it is differenced against
  one that cannot (`restrict_to_shared_trials`).
- **Trial indices are 0-BASED in the data and 1-BASED in every conversation about the protocol.**
  `session.periods` is `range(len(tone_onsets_def))`, so the event table's `trial` column runs
  0–4 while "trials 1–2" means `trial_idx` 0 and 1. `CONDITIONING_PHASES` is declared in 1-based
  trial *numbers* because that is how the protocol is described, and
  `split_by_conditioning_phase` converts exactly once, at its single point of use. Do not leave
  both conventions loose in new code — an off-by-one here silently moves a trial between the two
  columns of a figure and nothing raises.
- **`pre_tone` DOES exist on the first trial.** `get_epoch_frames` returns `None` for it only if
  the window would start before frame 0, and the first tone is at 185 s, so a 35 s (or 20 s)
  baseline fits comfortably. §8's `None`-is-not-an-error note lists `pre_tone` as an example of a
  definitionally-absent window; that is the general mechanism, not a statement that trial 1 lacks
  a baseline in this protocol. It does not.
- **The event table is built in ONE pass.** `build_epoch_and_run_tables` does a single
  `find_event_runs_ca_S` per mouse and slices every epoch out of it, so adding an epoch costs one
  extra window slice, **not** a re-detection. Never build a second table.

---

## 9. Open items

**Done since this list was written** — left visible so they are not re-proposed: `(1|mouse:trial)`
in the Bambi rate model (now `(1|mouse) + (1|mouse_trial)`); the BH-FDR family (§4.2); descriptive
`tone`/`post_shock` within-cell deltas alongside the confirmatory trace one
(`compute_all_epoch_deltas`, plotted by `plot_epoch_delta_forest`); the joint `group × epoch`
specificity test (§4.6); the exposure-matched trial restriction (§3.4a); the within-trial
`post_shock − pre_tone_matched` delta; and the **decomposition grid**
(`plot_decomposition_grid`) — effect estimates + CIs across components × epochs with no stars,
which is what the "mixed annotation style" and "move to effect estimates + CIs" items below asked
for. The per-epoch `decomposition_*.png` panels deliberately keep their brackets: they are the
distribution figures and the sensitivity archive, and the grid is what carries the claim.
Also done: the direct hM3D-vs-hM4D contrast (§6 rule 5), the `restrict_to_groups` pairwise null
(§4.3), and the evidential-tier reporting standard (§5.2).

Done in the most recent round: the **unified paper-facing analysis** (Part A) — two mouse-level
models, two joint Wald interaction tests, twelve model-derived contrasts, model diagnostics,
synthetic verification, and the figure/forest rewiring that makes
`unified_lmm_posthoc_contrasts.csv` the only source of a paper number. The internal families are
untouched (still exactly three and 13 members) and now serve as sensitivity evidence.

Done in the most recent round on top of that: the **within-epoch post-hoc families** (§A.4). The
models, matched trials, estimates, raw P-values and interaction tests are unchanged; within each
epoch the two treatment-vs-control comparisons are Holm-corrected together (`p_holm_epoch`), the
six-comparison across-epoch correction is retained as `p_holm_six` sensitivity output reported
only under Part 2 of `paper_results_summary.md`, and all six panels of the distribution figure are
annotated by that one procedure. This replaced an intermediate trace-only primary family
(`p_holm_trace`), which is gone (§7 row 26).

Done in an earlier round, all of it **additive**: the **paper lane** (§6.1) with its two figures,
`paper_results_summary.md` and its own short METHODS template;
`summarize_rate_group_epoch_contrasts`, which turns the NB model's raw interaction coefficients
into per-epoch rate ratios with HDIs (a reformatting of an existing fit, not a new test);
`plot_decomposition_grid`'s `components` / `include_exc_vs_inh` / `epoch_labels` arguments, all
defaulting to the previous behaviour; the **early-vs-late conditioning figure** (§6.0a) and
`split_by_conditioning_phase`; and the shared `_draw_superplot_panel_grid` primitive (§6).

**Statistical:**

- **The develops-with-conditioning question is half-answered and should be closed out.** §5
  records comparable amplitude contrasts in all three within-trial windows; §6.0a's figure asks
  whether the pre-tone elevation is present from trial 1. Fill the result into §6.0a when the run
  reports it. Note what it would and would not license: a flat early-vs-late profile is evidence
  that the effect does not develop over conditioning, **not** proof of a tonic drug effect — that
  claim needs a conditioning-naive, drug-on baseline, which this design does not provide. Do not
  write "tonic" into the manuscript on the strength of these two analyses.

- **Decide whether the four epoch-specificity tests should all stay in the BH family.** They are
  strongly dependent (one decomposition identity over overlapping cells), and adding them took the
  family from 9 to 13 members, nudging every existing `q` upward. The alternative is to keep only
  `epoch_specificity_amplitude` (matching the primary endpoint) and demote the other three to
  descriptive. One-line change to the orchestrator loop; currently all four are members.
- **Decide whether the run-structure family members should use the pairwise null** (§4.3) rather
  than the global one. Currently global, with pairwise printed beside it as a sensitivity value.
  Switching would move published numbers (run width p = 0.096 in §4.4), so it is a deliberate
  call, not a cleanup.
- The direct Exc/Inh contrasts have Welch intervals but no permutation p (§4.3 argues permutation
  should be preferred at n = 17). Adding 16 permutation tests is cheap but would want a declared
  home; as tier-3 output (§5.2) they arguably do not need one.

- Confirmatory panels currently Holm-correct all 3 pairwise contrasts; only hM3D-vs-control and
  hM4D-vs-control need alpha (Dunnett-style). Current state is *conservative* — costs power, does
  not inflate error. Add an **option** to `do_pairwise_holm_plot`; do not change its default, it is
  shared with `freezing_tuned_cells` and others.
- Report absolute differences alongside ratios (matters most for the hM4D null).
- Power/sensitivity statement: every null must report its interval and what it excludes
  (hM4D amplitude ~1.10 [0.77, 1.55] ⇒ effects up to +55% not excluded). This now applies to
  **both** co-primary nulls, and to the early-vs-late delta, which is reported as an interval by
  construction.
- Decide whether `POST_SHOCK_LATE_ONSET_S` should move toward Puhger's 140 s now that the
  coverage numbers are known (3–5 of 5 trials at 90 s). Purely a fidelity-vs-coverage trade;
  one-line change plus a re-run.

**Plotting:**

- Rate panels show horizontal **striping** — per-cell rate is `n_events / exposure`, so small
  integer counts give discrete values. Real quantization, not a bug. Options: leave it, small
  vertical jitter, or revert those panels to mouse-level violins.
- ~~**Mixed annotation style** in the decomposition row~~ — addressed by `plot_decomposition_grid`
  rather than by restyling those panels; see the "done" note above.
- Overall event rate visually dominant in the decomposition; per-mouse ECDF curves more prominent
  than the pooled curve.

**Deferred (documented as future work, not silently missing):** LT1→LT2 dropout decomposition;
cross-registration sensitivity forest plot; movement-conditioned / speed-binned / movement-
standardized analyses; ROI-per-FOV QC panel; reference-seeded (CaliAli) re-extraction; count-model
posterior-predictive and zero-inflation diagnostics; peri-shock analysis.

---

## 10. File map

| File | Role |
|---|---|
| [caban/sp_rates_lmm.py](../caban/sp_rates_lmm.py) | Everything: tables, models, permutations, all figures, orchestrator |
| [caban/single_unit_common.py](../caban/single_unit_common.py) | `fit_mixed_model`, `joint_wald_test`, `linear_contrast_test`, `holm_correct`, `fdr_correct`, `draw_superplot_triplet`, `ecdf_panel` |
| [caban/utilities.py](../caban/utilities.py) | `find_event_runs_ca{,_S}`, `find_spikes_ca{,_S}`, `barplot_annotate_brackets` |
| [caban/epoch_analysis.py](../caban/epoch_analysis.py) | `get_epoch_frames`, `get_testb_epoch_frames`, `TRACE_MATCHED_WINDOW_S`, `POST_SHOCK_LATE_ONSET_S`. `EPOCH_NAMES` here is the **PV/RDM pipeline's** epoch set, not a list of every window `get_epoch_frames` supports — `post_shock_late` is deliberately absent from it, since adding it would put a new row and column on every existing RDM figure |
| [caban/analysis.py](../caban/analysis.py) | `_draw_violin_triplet`, `do_pairwise_holm_plot` |
| [caban/pca_state_metrics.py](../caban/pca_state_metrics.py) | Also consumes `joint_wald_test` — inherits the df fix |
| [caban/sections.py](../caban/sections.py) | `run_sp_rates_lmm(ds, cfg)` |
| [caban/config.py](../caban/config.py) | `plot_sp_rates_lmm`, `sp_rates_lmm_n_perm`, `sp_rates_lmm_seed` |
| [analysis_methods_templates/sp_rates_lmm_methods.md](../analysis_methods_templates/sp_rates_lmm_methods.md) | METHODS text copied into every output dir at runtime |
| [analysis_methods_templates/sp_rates_lmm_paper_methods.md](../analysis_methods_templates/sp_rates_lmm_paper_methods.md) | Short paper-facing METHODS text, copied into the paper output dir (§6.1) |
| [analysis_methods_templates/sp_rates_lmm_figure_guide.md](../analysis_methods_templates/sp_rates_lmm_figure_guide.md) | Per-figure reading guide |

Paper-lane outputs (`PLOTS_DIR/sp_rates_lmm/paper/tfc_amplitude_rate/`):

| File | Role |
|---|---|
| `stats/unified_lmm_mouse_epoch_values.csv` | the 51-row inferential dataset; what the figures' mouse markers plot |
| `stats/unified_lmm_posthoc_contrasts.csv` | **authoritative** — every paper estimate, interval, P and asterisk |
| `stats/unified_lmm_interactions.csv` | the two joint group × epoch Wald tests |
| `stats/unified_lmm_{amplitude,rate}_summary.txt` | full model summaries |
| `stats/unified_lmm_diagnostics.{png,txt}`, `unified_lmm_residuals.csv`, `unified_lmm_influence.csv` | descriptive diagnostics (§A.6) |
| `stats/unified_lmm_synthetic_verification.txt` | the two planted-effect regression designs |
| `stats/paper_results_summary.md` | Part 1 = paper numbers, Part 2 = sensitivity |
| `stats/rate_group_epoch_contrasts.csv` | the NB sensitivity model's rate ratios + HDIs |
| `tfc_amplitude_rate_by_epoch.{png,svg}` + `_contrasts.md` | the two-row distribution figure |
| `tfc_decomposition_forest.{png,svg}` + `_contrasts.md` | the same contrasts as estimates |

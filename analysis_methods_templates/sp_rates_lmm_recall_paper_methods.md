# METHODS — cellular calcium event amplitude and rate during drug-free recall

*Paper-facing methods text for the RECALL lane (`Test_B` at 48 h and `Test_B_1wk` at 1 week).
The conditioning analysis this mirrors is described in `sp_rates_lmm_paper_methods.md`, copied
into the conditioning output folder; the full statistical architecture and decision record are in
`sp_rates_lmm_methods.md` and `docs/sp_rates_lmm.md`. Nothing here contradicts those; it applies
the same framework to retrieval.*

*Values written `{{...}}` are placeholders. Each recall session's own numbers are written by the
analysis into `stats/paper_results_summary.md` in that session's output folder, with a drafted
Results paragraph already filled in.*

## The question this lane answers

Does the cellular activity phenotype present during conditioning persist during later memory
retrieval **in the absence of CNO**, and is any group difference specifically evoked by the
tone/retrieval period rather than already present during the pre-tone baseline?

The second half of that question is not answered by whether the post-tone comparison reaches
significance. It is answered by the group × epoch interaction and by nothing else.

## Event detection and measurement — unchanged from conditioning

Event detection is **identical** to the conditioning analysis and was not re-tuned for recall.
Calcium events were detected on the deconvolved spike-inference trace (`S`) of each pyramidal
cell; **one event is one contiguous supra-threshold run of `S`**, and its amplitude is the
integral of `S` over that run. Local-maxima/peak-height amplitude is not used anywhere.

Cells with no detected events were retained in the population event-rate denominator. Per-event
amplitude is undefined for such cells, which are therefore necessarily absent from amplitude
averaging; this is a definitional exclusion, not missing data.

## Recall windows

Trial timing was measured per animal from each recording rather than taken from nominal protocol
constants. Two duration-matched 20 s windows were used around each tone presentation:

- **pre-tone**: the 20 s immediately preceding tone onset;
- **post-tone**: the 20 s immediately following tone offset.

The post-tone window is the retrieval analogue of the conditioning trace interval — the tone has
ended, and CA1 activity is measured over the following 20 s with no shock. **The tone epoch itself
is not analysed in this pass.** Neither the legacy 35 s post-tone window nor the full interval
until the next tone is used: the two epochs must be duration-matched, because population event
rate is duration-sensitive.

Analyses were restricted, within each animal, to tone trials on which **both** the 20 s pre-tone
and the 20 s post-tone window were completely observed, determined from each window's measured
extent rather than from a trial index. A recording that stops shortly after a tone offset
therefore loses that trial's post-tone window, and that trial contributes to neither outcome.
Per-animal coverage is reported in `stats/unified_recall_trial_coverage.csv`. Amplitude and rate
were computed from that same set of animal × trial windows.

## Animal-level summarization — unchanged from conditioning

The animal is the unit of inference. The inferential dataset contains one value per animal per
epoch ({{N_ROWS}} rows = {{N_ANIMALS}} animals × 2 epochs).

For each animal and epoch, **per-event amplitude** was summarised by calculating each active
cell's mean event-run integral across the retained trials, log-transforming it, and averaging
those cell-level logs within the animal. An exponentiated group contrast on this response is
therefore a **ratio of geometric means** of the cell-level mean event amplitudes.

**Population event rate** was calculated for the same trials as the total number of detected
events divided by the total cell-seconds across all detected pyramidal cells, including cells with
zero events, and was log-transformed. No pseudocount was applied; a zero population rate raises
rather than being smoothed over.

## Statistical analysis

Recall activity was analysed **separately** at 48 h (Test B) and 1 week (Test B 1wk). For each
recall session, per-event amplitude and population event rate were analysed separately using the
same linear mixed-effects model, with DREADD group (mCherry, hM3D or hM4D), epoch (pre-tone or
post-tone), and their interaction as fixed effects and animal as a random intercept
(`log(metric) ~ group * epoch + (1|animal)`; reference levels mCherry and pre-tone, set
explicitly).

Within each epoch, hM3D and hM4D were compared with mCherry using linear contrasts of the fitted
model — at the pre-tone reference epoch the group coefficient alone, and at post-tone that
coefficient plus the corresponding group × epoch interaction coefficient, with their covariance —
with **Holm correction across the two treatment-versus-control comparisons in that epoch**.
Amplitude and rate formed separate correction families; the direct hM3D-versus-hM4D comparison was
in neither.

**Group × epoch interactions were assessed by joint Wald tests of the two group × post-tone
interaction coefficients** and tested whether treatment effects differed between the pre-tone and
post-tone periods. This is the formal test of retrieval preferentiality.

All Wald and contrast inference used an animal-level denominator degrees of freedom of
*n*<sub>animals present in that session</sub> − 1, derived from the session and applied
consistently to both outcomes; no Satterthwaite or Kenward–Roger approximation was applied.
Effects are reported as exponentiated model contrasts (fold change or rate ratio) with 95%
confidence intervals; population-rate effects are additionally reported as observed absolute
differences in events s⁻¹ per cell, which are descriptive and carry no separate test. For the
within-epoch between-group comparisons, *P* < 0.05 after Holm correction was considered
statistically significant. The pre-tone → post-tone modulation contrasts are reported unadjusted;
see that paragraph below.

**Test B and Test B 1wk were analysed independently because the animals available at the two
recall sessions were not identical** ({{N_TESTB}} and {{N_TESTB_1WK}} animals respectively; the
animal absent at 48 h is not the animal absent at 1 week). No direct 48 h versus 1 week comparison
is made. A formal longitudinal comparison would require a separate fixed-cohort model restricted
to animals present at both sessions.

**Pre-tone → post-tone modulation.** To represent the interaction directly, each animal's change
in the log outcome (`post-tone − pre-tone`) was computed for both outcomes, and the three pairwise
between-group comparisons of that change — hM3D versus mCherry, hM4D versus mCherry and hM3D
versus hM4D — were obtained as linear contrasts of the same fitted mixed models. Against mCherry
each comparison is that group's group × epoch coefficient; between the two DREADD groups it is the
difference of their two group × epoch coefficients, computed with their covariance. **These
comparisons are reported as unadjusted model-derived contrast *P*-values** — they are not adjusted
for multiplicity and were not prospectively preregistered, and any value quoted from them must be
described that way. A Holm adjustment across the three comparisons within each outcome (two
three-member families) is additionally tabulated as a multiplicity reference for transparency. Each
group's own model-implied pre-to-post change is reported descriptively to characterize the
trajectory; it is not a between-group test. **The group × epoch joint Wald test remains the
omnibus test of whether pre-to-post modulation differed among the groups**, and is reported
alongside the pairwise contrasts, which decompose it rather than replace it. No group and no
outcome was treated as privileged, and no test was computed on the change scores outside the
fitted models.

Analyses used Python 3.11.15 with statsmodels 0.14.6 (`MixedLM`, REML, L-BFGS), scipy 1.17.1,
numpy 2.4.5 and pandas 3.0.2.

## Results template

*Filled with this run's own values in each session's `stats/paper_results_summary.md`. Write one
paragraph per session; do not write a sentence spanning both.*

### 48-h recall

> At 48-h recall, in the absence of CNO, hM3D mice [showed / did not show] altered post-tone
> per-event amplitude relative to mCherry controls (ratio `{{R48_A_EXC}}`, 95% CI
> `{{CI48_A_EXC}}`; Holm-adjusted *P* = `{{P48_A_EXC}}`), whereas hM4D mice `{{...}}`. Population
> event rate `{{...}}`. The group × epoch interaction was *F*(2, `{{DF48}}`) = `{{F48_A}}`,
> *P* = `{{P48_INT_A}}` for per-event amplitude and *F*(2, `{{DF48}}`) = `{{F48_R}}`,
> *P* = `{{P48_INT_R}}` for population event rate.

The per-animal pre-to-post modulation, its three pairwise between-group comparisons and a drafted
paragraph using them are in `stats/unified_recall_modulation_contrasts.md`; quote the omnibus
alongside any pairwise value taken from it.

### 1-week recall

> Same structure, with that session's own values and its own *n*.

## Interpretive constraints

**No CNO was present during recall.** A group difference at recall is a persistent consequence of
the conditioning-day manipulation. The supportable wording is:

> "Transient SST-interneuron manipulation during conditioning was associated with a persistent
> alteration in later CA1 activity during drug-free recall."

Do **not** write "hM3D activation increased activity at recall": no DREADD ligand was present
then. A recall difference may reflect altered memory formation, subsequent network plasticity, a
different behavioural state or freezing level, or another downstream consequence of the
conditioning-day manipulation, and does not imply ongoing receptor activation.

Do **not** write "the effect disappeared by 1 week", "the effect persisted significantly longer",
or "48 h differed from 1 week". None of those is tested here, and the two recall cohorts are not
the same animals. **"Significant at 48 h but not significant at 1 week" is not evidence that the
effect declined with time.**

Reading of the two results per outcome, stated in advance:

- **Significant post-tone difference with a null interaction** — the group differs during
  post-tone, but there is no evidence the difference is specifically retrieval-evoked; it may
  reflect a persistent group difference already present around recall.
- **Similar groups pre-tone, significant post-tone difference, significant interaction** —
  stronger evidence that the phenotype is preferentially recruited by the tone/retrieval period.
- **Significant pre-tone AND post-tone differences with a null interaction** — evidence for a
  persistent network-level phenotype at recall rather than a specifically tone-evoked effect.

None of these may be inferred from the presence or absence of an asterisk alone. Use the
interaction.

A non-significant interaction is **no evidence that the treatment effect was the same in both
windows**; it is an absence of evidence that it differed. Every null is reported with its
interval, and at these group sizes a non-significant result is weak evidence of absence.

## Deliberately out of scope in this pass

- Any direct Test_B versus Test_B_1wk comparison, and any common-cohort restriction of the two.
- Cross-registered cell-identity persistence analysis.
- Classification of neurons as positive/negative responders.
- Tone-epoch statistics.
- The negative-binomial count model as a distribution-aware sensitivity analysis of the recall
  rate result. It is permitted by the analysis plan but is not the paper-facing statistic,
  generates no figure annotation, and is not fit in this pass.

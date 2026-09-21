# Does conditioning activity predict recall memory? — exploratory screen

**Status: EXPLORATORY. Nothing here is a confirmatory result.** It records a screen run on
2026-09-18 so that a later pre-specified analysis can say *why* it tests what it tests, rather than
appearing to have picked its predictor out of the air. Every P below is uncorrected and 16
predictor x outcome combinations were run.

**Why it exists.** The four epoch-modulation lanes all return group nulls, while the behaviour
shows clear CS-evoked freezing at recall. If conditioning-session coding supports that memory, a
per-animal association should exist even where a three-group comparison is null — it uses
within-group variation, which the group contrasts discard.

## Measures

| | |
|---|---|
| Predictor | per-mouse conditioning index from `epoch_modulation/<signal>/…/per_mouse_modulation.csv` — `shock` and `post_shock`, each on the full-epoch lane (2 s US / 20 s) and the event-proximal lane (3 s) |
| Outcome | CS-evoked freezing at recall = mean(tone, post_tone) − pre_tone, per mouse. Run on BOTH freezing measures — the FreezeFrame scores (`freeze_data/TFC_miniscope.json`, the authoritative one) and the miniscope-velocity proxy (< 2 cm/s, `epoch_analysis._freezing_fraction_in_window`) |
| Unit | mouse (n = 16 per timepoint: Test_B has no G07, Test_B_1wk has no G15) |
| Test | Pearson partial correlation controlling for group (each group's mean removed from both variables), with the predictor permuted WITHIN group, 20,000 draws |

**The outcome is differenced against each mouse's own pre-tone baseline** so that G21 — immobile in
89% of Test_B frames, so its raw freezing saturates at 1.00 — cannot enter as a ceiling value. Its
CS-evoked value is exactly 0.

## The screen

**Read §"Which freezing measure" below first.** The table immediately below uses the
VELOCITY-DERIVED proxy. On the authoritative FreezeFrame scores its headline association does not
replicate, so the table is kept as the record of what was screened, not as a finding.

Partial r (controlling for group) and its within-group permutation P:

| signal | session | predictor | n | partial r | P (uncorrected) |
|---|---|---|---|---|---|
| YrA | Test_B | **shock, full (2 s US)** | 16 | **+0.626** | **0.026** |
| YrA | Test_B | shock, 3 s | 16 | +0.571 | 0.054 |
| YrA | Test_B | post_shock, 3 s | 16 | −0.101 | 0.718 |
| YrA | Test_B | post_shock, full (20 s) | 16 | −0.067 | 0.820 |
| YrA | Test_B_1wk | shock, 3 s | 16 | +0.290 | 0.473 |
| YrA | Test_B_1wk | shock, full | 16 | +0.249 | 0.533 |
| YrA | Test_B_1wk | post_shock, 3 s | 16 | +0.200 | 0.405 |
| YrA | Test_B_1wk | post_shock, full | 16 | +0.147 | 0.613 |
| C | Test_B | shock, 3 s | 16 | +0.559 | 0.059 |
| C | Test_B | shock, full | 16 | +0.570 | 0.062 |
| C | Test_B | post_shock, 3 s | 16 | +0.300 | 0.272 |
| C | Test_B | post_shock, full | 16 | +0.174 | 0.512 |
| C | Test_B_1wk | shock, 3 s | 16 | +0.178 | 0.574 |
| C | Test_B_1wk | shock, full | 16 | +0.100 | 0.753 |
| C | Test_B_1wk | post_shock, 3 s | 16 | +0.370 | 0.104 |
| C | Test_B_1wk | post_shock, full | 16 | +0.294 | 0.149 |

**What the velocity-proxy screen said.** The four largest associations are the four **shock** predictors at
**Test_B** (+0.56 to +0.63), on both signals and both windows; every **post-shock** predictor at
Test_B is flat (−0.10 to +0.30). At 1 wk nothing reaches +0.37. So a confirmatory analysis should
test the US response against 48 h CS-evoked freezing — chosen by this screen, on the signal-
and window-agreement of the top rows, not on the single smallest P.

**This is the opposite of the lane that shows a group difference.** The post-shock window is where
hM3D differs from control (+0.096 SD, exact P = 0.076, `event_proximal/`), and it is exactly the
predictor with no behavioural association on either measure.

## The same screen on the FreezeFrame scores — the association does NOT replicate

Identical predictors, identical model, outcome taken from the FreezeFrame bins on each session's
own clock (n = 16 per timepoint):

| signal | session | predictor | partial r | P | Spearman partial |
|---|---|---|---|---|---|
| YrA | Test_B | shock, full | **+0.060** | 0.877 | −0.109 |
| YrA | Test_B | shock, 3 s | −0.079 | 0.820 | −0.027 |
| C | Test_B | shock, full / 3 s | −0.035 / −0.075 | 0.91 / 0.82 | +0.079 / +0.133 |
| YrA | Test_B | post_shock, 3 s | −0.267 | 0.288 | −0.039 |
| **YrA** | **Test_B_1wk** | **shock, full** | **+0.432** | **0.148** | **+0.386** |
| YrA | Test_B_1wk | shock, 3 s | +0.310 | 0.288 | +0.285 |
| C | Test_B_1wk | shock, full / 3 s | +0.260 / +0.230 | 0.37 / 0.40 | +0.224 / +0.171 |
| YrA | Test_B_1wk | post_shock, 3 s | −0.322 | 0.270 | −0.184 |

**The 48 h association is measure-dependent and therefore not credible**: +0.63 on the velocity
proxy, +0.06 on the scored data. The largest FreezeFrame association is instead **1 wk**, YrA shock
(full) at +0.43 (Spearman +0.39), which is not significant at n = 16 but is at least consistent in
sign across signals and windows, and is the timepoint the behavioural memory effect is expected at.
Nothing here is a result; it is a hypothesis for a pre-specified test on better behavioural data.

## Robustness of the top row (YrA shock, full, Test_B)

| check | result |
|---|---|
| partial r | +0.626 |
| Spearman partial r | **+0.344** — much weaker; the linear fit is helped by the spread of values |
| per-group Pearson r | mCherry +0.78, hM3D +0.64, hM4D +0.23 — same sign in all three |
| leave one mouse out | +0.42 to +0.73 over all 16 — no single animal carries it |

The rank-based weakening is the main reason this needs confirmation rather than reporting.

## The scored FreezeFrame source, and a mapping bug it exposed

**Source of truth: `/Volumes/VLAD/FreezeFrame-MyRoom/<date> <mouse> <group>/<date> TFC test B[ 1wk]/*.csv`.**
Each mouse has its own folder, so the scored bins carry an unambiguous mouse id; the bins are 20 s
(onsets 0…880 s), threshold 31.35, bout 1.20 s. `freeze_data/TFC_miniscope.json` is a curated
extract of these files with the ids dropped.

Matching every JSON row back to its source CSV by value (2026-09-18) shows the positional mapping in
`sections.run_freeze_mobility_verification` — row *i* ↔ the *i*-th mouse of the group in
`ds.mouse_groups` order — is **correct for mCherry and hM3D at both timepoints and for hM4D at 1 wk,
and WRONG for hM4D at Test_B**:

| JSON `Inh` row | actually | assumed by the code |
|---|---|---|
| 0 | G06 | G06 ✓ |
| 1 | G07 | G14 ✗ |
| 2 | G14 | G15 ✗ |
| 3 | G20 | G20 ✓ |
| 4 | G21 | G21 ✓ |

The count check passes because five rows meet five hM4D mice with a Test_B session, but **G07 has no
Test_B imaging session and G15 was never scored**, so the rows shift. Any per-animal use of that
mapping mislabels two hM4D mice at 48 h.

**Not scored at all:** G15 (both recall timepoints) and G14 (Test_B). Those folders hold raw
`.ffii`/`.ffdd` and `Freeze_Log.xls` but no scored CSV.

**Where a session has several CSVs** (G05 Test_B ×3, G10 Test_B, G11 `freeze_A.csv`, G12 1 wk rows
`A`/`A1`, G14 1 wk, G21 1 wk), the file and row chosen are the ones the curated JSON already used,
recorded per mouse in the scratch table rather than guessed. G12's 1 wk row `A` reads 100% freezing
throughout and is clearly bad; the JSON uses `A1`.

## The screen, on per-mouse scored data

Outcome: CS-evoked freezing = mean(tone bin, two following bins) − two preceding bins, per mouse,
on the 20 s FreezeFrame bins with tones at 180/420/660 s. n = 15 at Test_B (G14, G15 unscored),
n = 16 at 1 wk (G15 unscored).

| signal | session | predictor | n | partial r | Spearman | P |
|---|---|---|---|---|---|---|
| **YrA** | **Test_B_1wk** | **shock, full (2 s US)** | 16 | **+0.629** | **+0.664** | **0.047** |
| YrA | Test_B_1wk | shock, 3 s | 16 | +0.535 | +0.599 | 0.115 |
| C | Test_B_1wk | shock, full | 16 | +0.427 | +0.398 | 0.174 |
| C | Test_B_1wk | shock, 3 s | 16 | +0.382 | +0.318 | 0.218 |
| YrA | Test_B_1wk | post_shock, 3 s | 16 | −0.318 | −0.198 | 0.208 |
| YrA/C | Test_B | any predictor | 15 | −0.30 … −0.06 | −0.36 … +0.04 | 0.28–0.88 |

**The conditioning US response predicts 1 wk memory, not 48 h.** All four shock predictors agree in
sign and order at 1 wk on both signals, and Spearman is as strong as Pearson (+0.66 vs +0.63), so it
is not a leverage artefact. Every 48 h association is null or slightly negative. Post-shock — the
window carrying the hM3D group difference — predicts nothing at either timepoint.

**Still not a confirmatory result.** It is one row of a 16-row screen at P = 0.047 uncorrected, with
n = 16. What it does establish is which single test a confirmatory analysis should run.

**Group-level CS-evoked freezing** (percentage points, scored data):

| | mCherry | hM3D | hM4D |
|---|---|---|---|
| Test_B (48 h) | 11.4 ± 12.1 (6) | 12.4 ± 18.1 (5) | −1.2 ± 1.8 (4) |
| Test_B 1 wk | 1.2 ± 7.9 (6) | −0.7 ± 11.6 (5) | 0.9 ± 1.9 (5) |

hM4D shows no CS-evoked freezing at 48 h, which is the direction a memory impairment would take.
Cohort-mean CS-evoked freezing at 1 wk is near zero in every group, so the 1 wk association above
lives in the between-animal spread rather than in a group-mean effect.

## Superseded: the velocity-proxy screen — read before using any number below

## Which freezing measure — read before using any number above

Two independent freezing measures exist and **they do not agree**:

1. **Velocity-derived** (used above): from the miniscope-tracked speed, mouse-matched by
   construction. It detects the memory: CS-evoked freezing +0.177 at Test_B (t(15) = 3.46,
   P = 0.0035) and +0.121 at 1 wk (P = 0.012). Session-mean freezing 9–89%.
2. **FreezeFrame scores** (`freeze_data/TFC_miniscope.json`): 45 bins per mouse per timepoint.
   Session-mean 0–30%, mostly under 10%.

Checked on 2026-09-18, assumed mapping, n = 16: session-mean freezing on the two measures
correlates **r = +0.05 (P = 0.87)** at 48 h and **+0.29 (P = 0.27)** at 1 wk. They disagree in
level *and* in ranking of animals.

**The JSON carries no mouse ids.** `sections.run_freeze_mobility_verification` pairs row *i* with
the *i*-th mouse of that group in `ds.mouse_groups` order and checks only that the COUNTS match.
That mapping has never been verified, and it does not survive a check: matching rows to mice by
correlating binned freezing time courses picks a different assignment than the assumed one
(hM3D 48 h: assumed r = −0.36 … +0.28). Either the rows are not in that order, or they are not
these mice, or the JSON holds something other than % freeze per bin — its sparse small values
(`[8.11, 0, 0, 0, 8, 0, …]`) do not look like a percentage per ~20 s bin for a conditioned animal.

**The FreezeFrame bins ARE correctly aligned in time** (checked 2026-09-18). With 45 bins over an
899 s session the bin width is exactly 20 s, and the cohort-mean profile rises in the bins
following each tone onset (180, 420, 660 s): at Test_B, 11.2% in the tone bins against 5.9% in the
preceding bins. So the time axis and the bin width are right; what remains unverified is only
WHICH ROW IS WHICH MOUSE within a group.

**Consequence.** The scored data is usable for cohort-level and group-level statements, and is what
the association screen above should be read from. It is NOT yet safe for per-animal claims: an
incorrect within-group row order would attenuate any true per-animal association toward zero, which
is one candidate explanation for the FreezeFrame nulls at 48 h. Resolving it needs the original
FreezeFrame export with animal identifiers.

**Group-level memory in this 16-mouse imaging subset**, from the scored data — CS-evoked freezing
(percentage points):

| | mCherry | hM3D | hM4D |
|---|---|---|---|
| Test_B (48 h) | 12.2 ± 9.5 | 9.8 ± 14.9 | 0.7 ± 3.6 |
| Test_B 1 wk | 3.8 ± 7.6 | 1.7 ± 5.7 | 1.4 ± 2.6 |

Tone-evoked freezing is **weaker at 1 wk than at 48 h** in this subset on the scored data (cohort
profile: 3.2% in tone bins against 3.6% before, versus 11.2% against 5.9% at 48 h). If the
behavioural memory effect is strongest at 1 wk, it is not visible in these 16 animals with this
measure — worth reconciling against the full behavioural cohort (`FSTE`/`FSTI`/`FSTC`, analysed in
`caban/TFC.py`), which has far more animals than the imaging subset.

## Files

| Path | Contents |
|---|---|
| `caban/epoch_analysis.py::_freezing_fraction_in_window` | the freezing definition reused here |
| `caban/epoch_analysis.py::get_testb_epoch_frames` | recall epoch windows (pre_tone / tone / post_tone) |
| `freeze_data/TFC_miniscope.json` | FreezeFrame bins, no mouse ids — see the caveat above |
| `caban/sections.py::run_freeze_mobility_verification` | the QC overlay whose row-to-mouse mapping is assumed, not verified |

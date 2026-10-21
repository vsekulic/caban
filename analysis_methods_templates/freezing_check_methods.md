# METHODS — Freezing check for the conditioning effects

`caban/freezing_check.py`; pre-specified in `plans/figure2_redesign_plan.md` §14 (revised after a
statistics review, before it was built). The animal is the unit.

**Freezing.** Behaviour-camera speed assigned to each miniscope frame (nearest timestamp, smoothed);
freezing = speed below 2 cm/s sustained for at least 1 s. The frame-wise definition without the
duration criterion is reported beside it. Epochs and trials are those of the four-epoch conditioning
model: pre-tone (20 s), tone (20 s), trace, post-shock (20 s), on the exposure-matched trials.

**(a) Freezing by group.** Per animal and epoch, the fraction of frames freezing. Per epoch, hM3D and
hM4D were compared with mCherry on its empirical logit, log((k + 0.5)/(n − k + 0.5)), by exact
pairwise mouse-label randomization (Holm across the two comparisons), with the Welch difference and
its 95% CI beside. With 5–6 animals per group, an absence of a difference is not evidence that
freezing was equal.

**(b) Effects at matched behaviour.** "Moving" frames are all frames outside a freezing bout, so they
include slow pauses shorter than 1 s (no frame lacked a speed value). The four-epoch model was refitted
using only moving frames:
events whose peak fell in a moving frame, per-event amplitude as before, and event rate per moving
cell-second. Its treatment-versus-control ratios and 95% CIs were compared with those of the
all-frames model by effect size. An animal × epoch cell with no moving frames or no event is listed
and left out of that refit. Because freezing may lie on the path from the manipulation to the neural
effect, this asks whether the effect persists at matched behaviour, not whether it is an artefact. A
secondary model added the freezing fraction (logit) to the all-frames model, split into each animal's
mean and the epoch deviation from it; freezing-only frames are reported descriptively.

**Caveats.** The bout and frame-wise definitions answer the same question and are not separate
tests; Holm is applied across the two treatment-versus-control comparisons within each epoch and
definition only, so the 16 freezing comparisons are reported as a description of behaviour, not as a
family of findings. In the state-restricted refits each event counts in the state of its peak frame,
even if it spans a state change. The freezing-only refit rests on few events (one animal × epoch cell
has none) and is descriptive only. In the covariate model the animal-mean freezing term varies between
animals, as group does, and can absorb part of a group difference; it is therefore secondary to the
moving-frames refit.

# METHODS — Average single-cell calcium event by epoch (Figure 2 v2, panel h)

`caban/event_waveform.py`. Descriptive; the statistics shown are those of the primary conditioning
model.

Events were the primary analysis's: contiguous runs of the deconvolved trace S above threshold 2, in
all cells, in each of the primary model's three epochs of every conditioning trial — the 20 s before
the tone, the trace interval and the 20 s after shock offset — assigned to an epoch by their peak. For each event, the same
cell's traces were taken around the frame of the event's S peak: S from 1 s before to 1 s after;
the denoised calcium trace C and the Minian residual YrA (the cell's signal plus noise before the
denoising fit) from 1 s before to 3 s after, each minus its own mean over −1 to −0.5 s. Cells were
matched to their C and YrA traces by unit identity. Each animal's traces were averaged over all of its
events, and the panel shows the mean ± s.e.m. of those per-animal averages across animals, so that
every animal counts once whatever its number of events. Samples of a window that fall beyond the end
of the recording are left undefined, and each time point averages the events that have it. Below each epoch's average, the
per-animal log per-event amplitude of that epoch, with the primary model's Holm-adjusted contrasts
against mCherry for that epoch.

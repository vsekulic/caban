# METHODS — Average trace-interval event (Figure 2 v2, panel h)

`caban/event_waveform.py`. Descriptive; the statistics shown are those of the primary conditioning
model.

Events were the primary analysis's: contiguous runs of the deconvolved trace S above threshold 2
during the trace interval of all five conditioning trials, in all cells. For each event, the same
cell's traces were taken around the frame of the event's S peak: S from 1 s before to 1 s after;
the denoised calcium trace C and the Minian residual YrA (the cell's signal plus noise before the
denoising fit) from 1 s before to 3 s after, each minus its own mean over −1 to −0.5 s. Cells were
matched to their C and YrA traces by unit identity. Each animal's traces were averaged over all of its
events, and the panel shows the mean ± s.e.m. of those per-animal averages across animals, so that
every animal counts once whatever its number of events. The ratio and Holm-adjusted P above the panel
are the trace-interval per-event amplitude contrasts against mCherry from the primary model.

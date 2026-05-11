import numpy as np
import caban.utilities as U
import caban.population as P

rng = np.random.default_rng(0)

S = rng.normal(0.5, 0.3, size=(4, 100))
S_spikes = {0: np.array([1,2,3]), 1: np.array([5,6]), 2: np.array([10]), 3: np.array([20,30,40])}
S_peakval = {0: np.array([0.1, 0.2, 0.1]), 1: np.array([0.5, 0.6]), 2: np.array([0.05]), 3: np.array([1.0, 1.1, 1.2])}

mask, idx, Sf, _, _ = U.get_engram_cells(S, S_spikes, S_peakval, zscore_thresh=0)
print('default:', mask, 'kept', Sf.shape[0])

mask, idx, Sf, _, _ = U.get_engram_cells(S, S_spikes, S_peakval, zscore_thresh=0, ext_norm=('zscore', 2.0, 1.0))
print('ext zscore mu=2 sigma=1:', mask, 'kept', Sf.shape[0])

mask, idx, Sf, _, _ = U.get_engram_cells(S, S_spikes, S_peakval, ext_norm=('absolute', 1.0))
print('ext absolute cutoff=1.0:', mask, 'kept', Sf.shape[0])

score_arrays = [np.array([0.4, 1.3, 0.05, 3.3]), np.array([0.1, 0.5, 2.0])]
print('ext-norm zscore   :', U.compute_engram_external_norm(score_arrays, mode='zscore'))
print('ext-norm percentile=50:', U.compute_engram_external_norm(score_arrays, mode='percentile', percentile=50))

S_raw_by_mouse = {'m1': rng.normal(1, 0.5, size=(50, 200)),
                  'm2': rng.normal(0.9, 0.4, size=(40, 200))}
print('pop ext-norm zscore   :', P.compute_control_engram_norm_from_raw(S_raw_by_mouse, ['m1','m2'], mode='zscore'))
print('pop ext-norm percentile:', P.compute_control_engram_norm_from_raw(S_raw_by_mouse, ['m1','m2'], mode='percentile', percentile=50))

S_test = rng.normal(1, 0.5, size=(20, 200))
m_default = P._engram_mask(S_test, threshold=0)
m_ext = P._engram_mask(S_test, threshold=0, ext_norm=('zscore', np.sum(S_test, axis=1).mean(), np.sum(S_test, axis=1).std()))
m_abs = P._engram_mask(S_test, threshold=0, ext_norm=('absolute', float(np.median(np.sum(S_test, axis=1)))))
print('pop default mask size:', m_default.size, 'ext z:', m_ext.size, 'absolute median:', m_abs.size)
print('OK')

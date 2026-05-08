import importlib, numpy as np
import SSTCa2_pca_state_metrics as m
importlib.reload(m)
rng = np.random.default_rng(0)
gd = {'mCherry': rng.normal(0,1,6), 'hM3D': rng.normal(0.5,1,5), 'hM4D': rng.normal(-1,1,4)}
om, cs = m._bootstrap_perm_pairs(gd, n_boot=2000, n_perm=2000)
print('OMN F=%.3f p=%.4f' % (om['F'], om['p']))
for c in cs:
    print('  %s vs %s diff=%+.3f CI=[%+.3f, %+.3f] p_raw=%.4f p_holm=%.4f' % (c['a'], c['b'], c['mean_diff'], c['ci_lo'], c['ci_hi'], c['p_raw'], c['p_holm']))
txt = m._format_stats_text('metric_x', 'ylabel', om, cs, gd, 'Permutation + BCa bootstrap')
print('---')
print(txt)
assert all(np.isfinite([om['F'], om['p']])), 'omnibus not finite'
assert all(np.isfinite([c['mean_diff'], c['ci_lo'], c['ci_hi'], c['p_raw'], c['p_holm']]) for c in cs)
print('OK')

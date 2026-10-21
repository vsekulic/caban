"""Two checks on the conditioning result, pre-specified in plans/figure2_redesign_plan.md §12 before
either was computed (METHODS: analysis_methods_templates/figure2_effect_profile_methods.md).

- Effect profile (the dissociation, version (b)): per animal, A = standardized amplitude effect (up
  positive), R = minus the standardized population-rate effect (down positive), P = A - R; hM3D vs
  hM4D on P.
- Fraction of cells active, weighted by each animal's cell count: quasi-binomial GLM on group.

Both read the primary model's per-animal trace-interval table, so they use exactly the values the
paper's conditioning model was fitted on.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

from caban.single_unit_common import (GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER,
                                      ensure_dirs, save_fig, write_text, holm_correct)
from caban.decoder import _copy_analysis_methods_template
from caban.sp_rates_lmm import mouse_label_permutation_test, TFC_TRACE_EPOCH

UNIFIED_TABLE = os.path.join('sp_rates_lmm', 'paper', 'tfc_amplitude_rate', 'stats',
                             'unified_lmm_mouse_epoch_values.csv')
OUT_DIRNAME = 'figure2_effect_profile'
METHODS_FILENAME = 'figure2_effect_profile_methods.md'
REFERENCE = 'mCherry'


def trace_table(PLOTS_DIR, mice_per_group):
    path = os.path.join(PLOTS_DIR, UNIFIED_TABLE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found: run run_sp_rates_lmm first.')
    df = pd.read_csv(path)
    df = df[df['epoch'] == TFC_TRACE_EPOCH].set_index('mouse')
    expected = {m for ms in mice_per_group.values() for m in ms}
    if set(df.index) != expected:
        raise RuntimeError(f'figure2_effect_profile: trace table has mice {sorted(df.index)}, the cohort '
                           f'is {sorted(expected)}.')
    return df


def _pooled_sd(values, groups):
    ss, dof = 0.0, 0
    for g in pd.unique(groups):
        v = values[groups == g]
        ss += ((v - v.mean()) ** 2).sum()
        dof += len(v) - 1
    return float(np.sqrt(ss / dof))


def effect_profile(df):
    """Per animal: A, R and P = A - R (see module docstring)."""
    out = df[['group']].copy()
    for col, name, sign in (('mouse_mean_log_amplitude', 'A', 1.0), ('log_population_rate', 'R', -1.0)):
        ref = df.loc[df['group'] == REFERENCE, col].mean()
        out[name] = sign * (df[col] - ref) / _pooled_sd(df[col], df['group'])
    out['P'] = out['A'] - out['R']
    return out


def _exact_pair_p(mice_per_group, a, b, stat):
    def stat_fn(mouse_to_group):
        return stat([m for m, g in mouse_to_group.items() if g == a],
                    [m for m, g in mouse_to_group.items() if g == b])
    return mouse_label_permutation_test(stat_fn, mice_per_group, restrict_to_groups=(a, b),
                                        exact=True)['p_two_sided']


def test_profile(profile, mice_per_group):
    a, b = 'hM3D', 'hM4D'
    pa = profile.loc[profile['group'] == a, 'P']
    pb = profile.loc[profile['group'] == b, 'P']
    welch = stats.ttest_ind(pa, pb, equal_var=False)
    P = profile['P'].to_dict()
    p_perm = _exact_pair_p(mice_per_group, a, b,
                           lambda ma, mb: np.mean([P[m] for m in ma]) - np.mean([P[m] for m in mb]))
    return {'hM3D_mean_P': float(pa.mean()), 'hM4D_mean_P': float(pb.mean()),
            'difference': float(pa.mean() - pb.mean()), 'welch_t': float(welch.statistic),
            'welch_df': float(welch.df), 'p_welch': float(welch.pvalue), 'p_permutation': float(p_perm)}


def fraction_active_weighted(df, mice_per_group):
    """Quasi-binomial GLM of active/total cells per animal on group; contrasts vs mCherry, Holm."""
    groups = [g for g in DREADD_DISPLAY_ORDER if g != REFERENCE]
    X = pd.DataFrame({f'is_{g}': (df['group'] == g).astype(float) for g in groups}, index=df.index)
    X = sm.add_constant(X)
    y = np.column_stack([df['n_active_cells'], df['n_cells'] - df['n_active_cells']])
    # Dispersion set explicitly as Pearson chi2 / residual df: statsmodels' scale='X2' with a
    # two-column (successes, failures) response returned 0.05 where chi2/df is ~11 (2026-10-21),
    # which would shrink every interval ~15-fold.
    binomial = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    dispersion = float(binomial.pearson_chi2 / binomial.df_resid)
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(scale=dispersion, use_t=True)
    if not np.isclose(fit.scale, dispersion):
        raise RuntimeError(f'fraction_active_weighted: fitted scale {fit.scale} is not the Pearson '
                           f'dispersion {dispersion}.')
    ci = fit.conf_int()
    n_act, n_all = df['n_active_cells'].to_dict(), df['n_cells'].to_dict()

    def weighted_diff(ma, mb):
        return (sum(n_act[m] for m in ma) / sum(n_all[m] for m in ma)
                - sum(n_act[m] for m in mb) / sum(n_all[m] for m in mb))

    rows = []
    for g in groups:
        term = f'is_{g}'
        rows.append({'contrast': f'{GROUP_LABELS[g]} - {GROUP_LABELS[REFERENCE]}',
                     'odds_ratio': float(np.exp(fit.params[term])),
                     'ci_low': float(np.exp(ci.loc[term, 0])), 'ci_high': float(np.exp(ci.loc[term, 1])),
                     'p_glm': float(fit.pvalues[term]),
                     'weighted_fraction_diff': weighted_diff(mice_per_group[g], mice_per_group[REFERENCE]),
                     'p_permutation': float(_exact_pair_p(mice_per_group, g, REFERENCE, weighted_diff))})
    out = pd.DataFrame(rows)
    out['p_holm'] = holm_correct(out['p_glm'].to_numpy())[1]
    return out, fit


def profile_sensitivity(df, mice_per_group):
    """Robustness of 12.1 (reviews of 2026-10-21; checks, not further tests): the hM3D - hM4D
    difference in P under other scalings, each with its exact pairwise permutation p; the scaling is
    recomputed inside every relabelling in 'pooled SD, recomputed per relabelling'."""
    amp, rate = df['mouse_mean_log_amplitude'], df['log_population_rate']
    ctl = df['group'] == REFERENCE
    a, b = 'hM3D', 'hM4D'

    def p_from(A, R):
        return (A - R).to_dict()

    variants = {
        'pooled SD (as reported)': lambda g: (_pooled_sd(amp, g), _pooled_sd(rate, g)),
        'control SD only': lambda g: (float(amp[ctl].std(ddof=1)), float(rate[ctl].std(ddof=1))),
        'raw log units (no scaling)': lambda g: (1.0, 1.0),
    }
    rows = []
    for name, scale_fn in variants.items():
        s_amp, s_rate = scale_fn(df['group'])
        P = p_from((amp - amp[ctl].mean()) / s_amp, -(rate - rate[ctl].mean()) / s_rate)
        diff = lambda ma, mb, P=P: np.mean([P[m] for m in ma]) - np.mean([P[m] for m in mb])
        rows.append({'variant': name, 'difference': diff(mice_per_group[a], mice_per_group[b]),
                     'p_permutation': _exact_pair_p(mice_per_group, a, b, diff),
                     'hM3D_all_positive': all(P[m] > 0 for m in mice_per_group[a]),
                     'hM4D_all_negative': all(P[m] < 0 for m in mice_per_group[b])})

    observed_groups = df['group'].copy()

    def recomputed(ma, mb):
        g = observed_groups.copy()
        g.loc[ma], g.loc[mb] = a, b
        P = p_from((amp - amp[ctl].mean()) / _pooled_sd(amp, g),
                   -(rate - rate[ctl].mean()) / _pooled_sd(rate, g))
        return np.mean([P[m] for m in ma]) - np.mean([P[m] for m in mb])

    rows.append({'variant': 'pooled SD, recomputed per relabelling',
                 'difference': recomputed(mice_per_group[a], mice_per_group[b]),
                 'p_permutation': _exact_pair_p(mice_per_group, a, b, recomputed),
                 'hM3D_all_positive': np.nan, 'hM4D_all_negative': np.nan})
    out = pd.DataFrame(rows)
    A = (amp - amp[ctl].mean()) / _pooled_sd(amp, df['group'])
    R = -(rate - rate[ctl].mean()) / _pooled_sd(rate, df['group'])
    corr = {g: stats.spearmanr(A[df['group'] == g], R[df['group'] == g])[0] for g in DREADD_DISPLAY_ORDER}
    corr['all animals'] = stats.spearmanr(A, R)[0]
    return out, corr


def plot_profile_plane(profile, out_path):
    """One point per animal in the (A, R) plane; the control MEAN is at the origin."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.axhline(0, color='0.7', lw=0.6)
    ax.axvline(0, color='0.7', lw=0.6)
    lim = float(np.abs(profile[['A', 'R']].to_numpy()).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color='0.7', lw=0.6, ls='--')
    for g in DREADD_DISPLAY_ORDER:
        sub = profile[profile['group'] == g]
        ax.scatter(sub['A'], sub['R'], s=34, color=GROUP_COLOURS[g], alpha=0.85, edgecolors='black',
                   linewidths=0.3, label=GROUP_LABELS[g], zorder=3)
        for m, r in sub.iterrows():
            ax.annotate(m, (r['A'], r['R']), fontsize=5, xytext=(2, 2), textcoords='offset points')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('A: amplitude effect (s.d., up +)')
    ax.set_ylabel('R: rate effect (s.d., down +)')
    ax.legend(fontsize='x-small', frameon=False)
    ax.set_title('Effect profile per animal (trace interval)\nbelow the diagonal: amplitude-led; above: rate-led',
                 fontsize='small')
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def run_figure2_effect_profile(PLOTS_DIR, mice_per_group):
    out_dir = os.path.join(PLOTS_DIR, OUT_DIRNAME)
    ensure_dirs(out_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)
    df = trace_table(PLOTS_DIR, mice_per_group)
    profile = effect_profile(df)
    prof_test = test_profile(profile, mice_per_group)
    frac, fit = fraction_active_weighted(df, mice_per_group)
    sens, corr = profile_sensitivity(df, mice_per_group)
    sens.to_csv(os.path.join(out_dir, 'effect_profile_sensitivity.csv'), index=False)
    profile.to_csv(os.path.join(out_dir, 'effect_profile_per_animal.csv'))
    frac.to_csv(os.path.join(out_dir, 'fraction_active_weighted.csv'), index=False)
    plot_profile_plane(profile, os.path.join(out_dir, 'effect_profile_plane.png'))
    report = ['Figure 2 checks (plans/figure2_redesign_plan.md §12; pre-specified)', '=' * 66, '',
              '12.1 Effect profile, trace interval: P = A - R per animal; hM3D vs hM4D',
              pd.Series(prof_test).round(4).to_string(), '',
              'Per animal:', profile.round(3).to_string(), '',
              'Robustness of 12.1 (checks, not further tests):', sens.round(4).to_string(index=False), '',
              'Spearman correlation of A and R across animals: '
              + ', '.join(f'{k} {v:+.2f}' for k, v in corr.items()), '',
              '12.2 Fraction active, quasi-binomial GLM weighted by cell count (vs mCherry, Holm over two)',
              frac.round(4).to_string(index=False), '',
              f'Dispersion (Pearson chi2 / df): {fit.scale:.2f}', '', fit.summary().as_text()]
    text = '\n'.join(report) + '\n'
    write_text(os.path.join(out_dir, 'figure2_effect_profile_report.txt'), text)
    print(text)
    return prof_test, frac

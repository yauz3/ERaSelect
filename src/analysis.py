"""Aggregation, figures and tables. Reads only what experiments.py wrote; fits nothing.

Every number that enters the manuscript is produced here from fold-level results, reported as
mean +/- SD across outer folds with a fold-level bootstrap confidence interval. The resampling
unit is the outer fold, never the molecule, because molecules within a fold are not independent
of one another under a scaffold split.
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import chemspace as CS
import metrics as MT
import selective as SEL
import viz as V
from calibration import ProbabilityCalibrator  # noqa: F401  (kept for provenance in lockfile)
from data import ROOT

warnings.filterwarnings('ignore', category=FutureWarning)

ABLATION = [('hierarchy_raw', 'Hierarchy, uncalibrated'),
            ('hierarchy_calibrated', '+ probability calibration'),
            ('hierarchy_gate_recall', '+ recall-targeted gate'),
            ('hierarchy_gray', '+ gray-zone abstention'),
            ('hierarchy_conformal_feasible', '+ feasible conformal'),
            ('hierarchy_knn', '+ support verifier'),
            ('full_knn', 'Full pipeline')]

BASELINES = ['flat_raw', 'flat_platt', 'joint4_raw', 'endpoint_pair']
# Metrics where a smaller number is a better result. Used so that a 'significant'
# difference is never reported as a win when it is in fact a loss.
LOWER_IS_BETTER = {'selective_risk', 'aurc', 'mc_brier', 'mc_ece',
                   'activity_brier', 'activity_ece'}
CAL_METHODS = ['flat_raw', 'flat_platt', 'flat_isotonic', 'flat_temperature']


# ------------------------------------------------------------------ io helpers

def make_table(out):
    (out / 'tables').mkdir(parents=True, exist_ok=True)

    def table(df, name, index=False, float_fmt='%.4f'):
        df = pd.DataFrame(df)
        df.to_csv(out / 'tables' / f'{name}.csv', index=index)
        try:
            df.to_markdown(out / 'tables' / f'{name}.md', index=index, floatfmt='.4f')
        except Exception:
            (out / 'tables' / f'{name}.md').write_text(df.to_string(index=index))
        try:
            with open(out / 'tables' / f'{name}.tex', 'w') as fh:
                fh.write(df.to_latex(index=index, float_format=float_fmt, longtable=False,
                                     escape=True))
        except Exception:
            pass
        return df
    return table


def summarise(metrics, keys, cols, reps=2000):
    """mean / SD / bootstrap CI over folds for each group."""
    rows = []
    for key, grp in metrics.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys if isinstance(keys, list) else [keys], key))
        rec['n_folds'] = int(len(grp))
        for c in cols:
            if c not in grp:
                continue
            v = pd.to_numeric(grp[c], errors='coerce').dropna().to_numpy()
            if not len(v):
                continue
            mean, lo, hi = MT.bootstrap_ci(v, reps=reps, seed=11)
            rec[f'{c}_mean'] = mean
            rec[f'{c}_sd'] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            rec[f'{c}_ci_low'] = lo
            rec[f'{c}_ci_high'] = hi
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ headline figure

def calibration_budget_figure(out, table):
    path = out / 'calibration_budget_feasibility.csv'
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[d.status == 'allocated']
    if not len(d):
        return None
    table(d, 'calibration_budget_feasibility')

    comps = [('conformal_mondrian', 0), ('conformal_mondrian', 1), ('conformal_mondrian', 2),
             ('support_verifier', 1), ('support_verifier', 2)]
    comps = [c for c in comps if ((d.component == c[0]) & (d.klass == c[1])).any()]
    labels = [f'{"Conformal" if a.startswith("conformal") else "Verifier"}\n{V.CLASS_LABEL[b]}'
              for a, b in comps]
    groups, reqs, feas = [], [], []
    for a, b in comps:
        sub = d[(d.component == a) & (d.klass == b)]
        groups.append(sub.n.to_numpy())
        reqs.append(float(sub.required_n.iloc[0]))
        feas.append(float(sub.feasible.mean()))

    short = [f'{"Conf." if a.startswith("conformal") else "Verif."}\n{V.CLASS_LABEL[b]}'
             for a, b in comps]
    colors = [V.GOOD if f == 1 else (V.WARNING if f > 0 else V.CRITICAL) for f in feas]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8),
                             gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    V.boxplot(ax, groups, short, colors=colors)
    for i, r in enumerate(reqs):
        ax.plot([i - .36, i + .36], [r, r], color=V.INK, linewidth=2.4, zorder=6,
                solid_capstyle='butt')
    ax.set_yscale('log')
    ax.set_ylabel('Calibration molecules available (log scale)')
    ax.set_ylim(min(1.6, min(min(g) for g in groups) * 0.55),
                max(max(max(g) for g in groups), max(reqs)) * 2.6)
    ax.plot([], [], color=V.INK, linewidth=2.4, label='Required by the criterion')
    ax.scatter([], [], s=11, facecolor=V.INK, edgecolor=V.SURFACE, label='One outer fold')
    ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.tick_params(axis='x', labelsize=plt.rcParams['font.size'] - 1.5)
    V.title(ax, 'Calibration budget against requirement',
            'Rule marks n = ceil(1/alpha) − 1 (conformal) and ceil(1/(1−q)) (verifier)')

    ax = axes[1]
    pos = np.arange(len(comps))
    V.rounded_bars(ax, pos, [f * 100 for f in feas], None, width=0.55)
    for patch, c in zip([p for p in ax.patches], colors):
        patch.set_facecolor(c)
    for i, f in enumerate(feas):
        ax.text(i, f * 100 + 2.5, f'{f*100:.0f}%', ha='center', va='bottom',
                fontsize=plt.rcParams['font.size'] - 1,
                color=V.GOOD if f == 1 else V.CRITICAL)
    ax.set_xticks(pos)
    ax.set_xticklabels(short, fontsize=plt.rcParams['font.size'] - 1.5,
                       rotation=30, ha='right')
    ax.set_ylabel('Folds meeting the criterion (%)')
    ax.set_ylim(0, 115)
    V.title(ax, 'Feasible folds', 'Computed before any model is fitted')

    summary = pd.DataFrame({'component': [f'{a}_class{b}' for a, b in comps],
                            'required_n': reqs,
                            'median_available_n': [float(np.median(g)) for g in groups],
                            'min_available_n': [float(np.min(g)) for g in groups],
                            'folds_feasible_fraction': feas,
                            'attainable_alpha_at_min': [SEL.attainable_alpha(int(np.min(g)))
                                                        for g in groups]})
    table(summary, 'calibration_budget_summary')
    return V.save(fig, '03_calibration_budget', out,
                  'Per-class calibration budget against the requirement implied by the nominal '
                  'miscoverage level, computed from training-partition labels before any model '
                  'is fitted. Classes falling below the rule cannot certify the nominal level; '
                  'their conformal threshold is infinite and admits every molecule.')


def conformal_coverage_figure(out, metrics, table):
    variants = [v for v in ('hierarchy_conformal_nominal', 'hierarchy_conformal_feasible',
                            'hierarchy_conformal_pooled') if (metrics.variant == v).any()]
    if not variants:
        return None
    cols = ['coverage', 'coverage_class0', 'coverage_class1', 'coverage_class2',
            'selective_risk', 'degenerate_threshold_fraction', 'mean_set_size']
    s = summarise(metrics[metrics.variant.isin(variants) & (metrics.split == 'scaffold')],
                  ['variant'], cols)
    if not len(s):
        return None
    table(s, 'conformal_coverage_by_variant')

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    classes = [0, 1, 2]
    width = 0.78 / max(1, len(variants))
    for i, var in enumerate(variants):
        row = s[s.variant == var]
        if not len(row):
            continue
        vals = [row[f'coverage_class{c}_mean'].iloc[0] for c in classes]
        errs = [row[f'coverage_class{c}_sd'].iloc[0] for c in classes]
        pos = np.arange(len(classes)) + (i - (len(variants) - 1) / 2) * (width + 0.012)
        ax.bar(pos, vals, width=width, color=V.SERIES[i], linewidth=0,
               label=var.replace('hierarchy_conformal_', ''))
        ax.errorbar(pos, vals, yerr=errs, fmt='none', ecolor=V.INK_3, elinewidth=1.0)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([V.CLASS_LABEL[c] for c in classes])
    ax.set_ylabel('Empirical acceptance (coverage)')
    ax.set_ylim(0, 1.08)
    ax.legend(title='Conformal mode')
    V.title(ax, 'Acceptance by class and conformal mode',
            'A degenerate threshold shows up as coverage pinned at 1.0')

    ax = axes[1]
    if 'degenerate_threshold_fraction_mean' in s:
        vals = s.set_index('variant').reindex(variants)['degenerate_threshold_fraction_mean']
        V.rounded_bars(ax, np.arange(len(variants)), vals.fillna(0).to_numpy() * 100,
                       V.CRITICAL, width=0.5)
        for i, v in enumerate(vals.fillna(0)):
            ax.text(i, v * 100 + 2, f'{v*100:.0f}%', ha='center', va='bottom',
                    fontsize=plt.rcParams['font.size'] - 1, color=V.INK_2)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([v.replace('hierarchy_conformal_', '') for v in variants],
                           rotation=15, ha='right')
        ax.set_ylabel('Degenerate (infinite) thresholds (%)')
        ax.set_ylim(0, 112)
        V.title(ax, 'Thresholds that admit everything')
    else:
        V.empty(ax, 'degeneracy not recorded')
    return V.save(fig, '20_conformal_coverage', out,
                  'Empirical class-conditional acceptance under pooled, nominal-alpha and '
                  'feasible-alpha conformal calibration, with the fraction of infinite '
                  'thresholds. Pooled calibration can look valid while the minority class is '
                  'uncalibrated.')


# ------------------------------------------------------------------ benchmark figures

def benchmark_boxplots(out, metrics, table, metric, variant, fname, ylabel, caption):
    d = metrics[(metrics.variant == variant) & (metrics.split == 'scaffold')]
    if metric not in d or not len(d):
        return None
    d = d[pd.to_numeric(d[metric], errors='coerce').notna()]
    if not len(d):
        return None
    reps = sorted(d.representation.unique())
    models = sorted(d.model.unique())
    nice = {'lr': 'LR', 'rf': 'RF', 'extra': 'ExtraTrees', 'xgb': 'XGBoost',
            'lgbm': 'LightGBM', 'catboost': 'CatBoost', 'mlp': 'MLP'}
    labels = [nice.get(m, m) for m in models]
    rot = 35 if (len(models) > 4 or max(len(x) for x in labels) > 6) else 0
    width = min(0.58, 0.16 + 0.1 * len(models))
    fig, axes = plt.subplots(1, len(reps), figsize=(max(2.3 * len(reps) + 1.2 * len(models), 6.4),
                                                    4.3),
                             sharey=True, squeeze=False)
    for ax, rep in zip(axes[0], reps):
        sub = d[d.representation == rep]
        groups = [pd.to_numeric(sub.loc[sub.model == m, metric], errors='coerce').dropna()
                  .to_numpy() for m in models]
        V.boxplot(ax, groups, labels, width=width,
                  colors=[V.SERIES[i % len(V.SERIES)] for i in range(len(models))])
        if rot:
            ax.tick_params(axis='x', labelrotation=rot)
            for t in ax.get_xticklabels():
                t.set_horizontalalignment('right')
        V.title(ax, rep)
        ax.grid(axis='x', visible=False)
        ax.set_xlim(-0.6, len(models) - 0.4)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(f'{ylabel} across outer scaffold folds', x=0.02, ha='left', y=1.05)
    s = summarise(d, ['representation', 'model'], [metric])
    table(s.sort_values(f'{metric}_mean', ascending=False), f'benchmark_{metric}')
    return V.save(fig, fname, out, caption)


def benchmark_heatmap(out, metrics, table, metric='activity_pr_auc',
                      variant='hierarchy_gate_recall'):
    d = metrics[(metrics.variant == variant) & (metrics.split == 'scaffold')]
    if metric not in d or not len(d):
        return None
    piv = d.pivot_table(index='model', columns='representation', values=metric, aggfunc='mean')
    sd = d.pivot_table(index='model', columns='representation', values=metric, aggfunc='std')
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(1.35 * piv.shape[1] + 3.0, 0.52 * piv.shape[0] + 2.2))
    im = ax.imshow(piv.to_numpy(), cmap=V.SEQUENTIAL, aspect='auto')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                norm = (v - np.nanmin(piv.to_numpy())) / max(
                    1e-9, np.nanmax(piv.to_numpy()) - np.nanmin(piv.to_numpy()))
                ax.text(j, i, f'{v:.3f}\n±{sd.iloc[i, j]:.3f}' if np.isfinite(sd.iloc[i, j])
                        else f'{v:.3f}', ha='center', va='center',
                        fontsize=plt.rcParams['font.size'] - 2.5,
                        color=V.SURFACE if norm > 0.55 else V.INK)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, ha='right')
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(metric.replace('_', ' '), color=V.INK_2)
    cb.outline.set_visible(False)
    V.title(ax, 'Model family by representation', 'mean ± SD over outer scaffold folds')
    table(piv.round(4).reset_index(), 'benchmark_model_x_representation')
    return V.save(fig, '14_benchmark_heatmap', out,
                  'Activity-gate PR-AUC for every model family and representation, averaged '
                  'over outer scaffold folds with the fold standard deviation.')


def calibration_benchmark(out, metrics, table):
    d = metrics[metrics.variant.isin(CAL_METHODS) & (metrics.split == 'scaffold')]
    if not len(d):
        return None
    s = summarise(d, ['variant'], ['mc_ece', 'mc_brier', 'mc_balanced_accuracy'])
    if not len(s):
        return None
    order = [v for v in CAL_METHODS if v in set(s.variant)]
    s = s.set_index('variant').reindex(order).reset_index()
    table(s, 'calibration_method_benchmark')
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for ax, col, lab in ((axes[0], 'mc_ece', 'Expected calibration error'),
                         (axes[1], 'mc_brier', 'Multiclass Brier score')):
        if f'{col}_mean' not in s:
            V.empty(ax, f'{col} unavailable')
            continue
        vals = s[f'{col}_mean'].to_numpy()
        errs = s[f'{col}_sd'].to_numpy()
        V.rounded_bars(ax, np.arange(len(s)), vals, V.SERIES[0], width=0.5)
        ax.errorbar(np.arange(len(s)), vals, yerr=errs, fmt='none', ecolor=V.INK_3,
                    elinewidth=1.0)
        best = int(np.nanargmin(vals))
        ax.text(best, vals[best] * 1.04 + 1e-4, 'best', ha='center', va='bottom',
                fontsize=plt.rcParams['font.size'] - 1.5, color=V.GOOD)
        ax.set_xticks(np.arange(len(s)))
        ax.set_xticklabels([v.replace('flat_', '') for v in s.variant], rotation=15, ha='right')
        ax.set_ylabel(lab)
        ax.set_ylim(0, np.nanmax(vals + np.nan_to_num(errs)) * 1.25)
        V.title(ax, lab)
    fig.suptitle('Probability calibration methods on matched partitions', x=0.02, ha='left',
                 y=1.04)
    return V.save(fig, '16_calibration_benchmark', out,
                  'Calibration quality of Platt scaling, isotonic regression and temperature '
                  'scaling, each fitted on the same disjoint calibration partition.')


def reliability_curves(out, table, bins=10):
    pred = CS._primary_predictions(out)
    if pred is None or 'p_active' not in pred:
        return None
    pred = pred[pred.split == 'scaffold']
    if not len(pred):
        return None
    y = (pred.label > 0).astype(int).to_numpy()
    p = pred.p_active.to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() >= 5:
            rows.append({'bin_low': lo, 'bin_high': hi, 'n': int(m.sum()),
                         'mean_predicted': float(p[m].mean()),
                         'observed_frequency': float(y[m].mean())})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), gridspec_kw={'width_ratios': [1, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=V.INK_3, linewidth=1.0, linestyle=(0, (4, 3)),
            label='Perfect calibration')
    ax.plot(r.mean_predicted, r.observed_frequency, marker='o', color=V.SERIES[0],
            markeredgecolor=V.SURFACE, markeredgewidth=1.6, label='Calibrated gate')
    ax.set_xlabel('Mean predicted P(active)')
    ax.set_ylabel('Observed active frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    V.title(ax, 'Reliability of the activity gate',
            f'ECE = {MT.expected_calibration_error(np.column_stack([1-p, p]), y, bins):.4f}')
    ax = axes[1]
    ax.hist(p, bins=40, range=(0, 1), color=V.SERIES[0], linewidth=0, alpha=0.9)
    ax.set_yscale('log')
    ax.set_xlabel('Predicted P(active)')
    ax.set_ylabel('Molecules (log scale)')
    V.title(ax, 'Score distribution', 'Most of the library sits near zero, as screening implies')
    table(r, 'reliability_activity_gate')
    return V.save(fig, '15_reliability', out,
                  'Reliability diagram and score histogram for the calibrated activity gate on '
                  'held-out scaffold folds.')


def risk_coverage(out, metrics, table):
    pred = CS._primary_predictions(out)
    if pred is None:
        return None
    pred = pred[(pred.split == 'scaffold') & (pred.label < 3)]
    if not len(pred):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4),
                             gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.42})
    ax = axes[0]
    grid = np.linspace(0.05, 1.0, 40)
    curves = {}
    for i, (col, lab) in enumerate((('confidence', 'Confidence ranking'),)):
        cov, risk = SEL.risk_coverage_curve(pred.label.to_numpy(), pred.prediction.to_numpy(),
                                            pred[col].to_numpy(), grid)
        ax.plot(cov, risk, color=V.SERIES[0], label=lab)
        curves[lab] = (cov, risk)
    marks = [('accept_gray', 'Gray-zone', 1), ('accept_conformal_feasible', 'Conformal', 2),
             ('accept_full_knn', 'Full pipeline', 3)]
    rows = []
    for col, lab, ci in marks:
        if col not in pred:
            continue
        acc = pred[col].astype(bool).to_numpy()
        m = SEL.selective_metrics(pred.label.to_numpy(), pred.prediction.to_numpy(), acc)
        ax.scatter([m['coverage']], [m['selective_risk']], s=72, marker=V.CLASS_MARKER[ci],
                   color=V.SERIES[ci], edgecolor=V.SURFACE, linewidth=1.8, zorder=5, label=lab)
        rows.append({'operating_point': lab, **m})
    base = SEL.selective_metrics(pred.label.to_numpy(), pred.prediction.to_numpy(),
                                 np.ones(len(pred), bool))
    rows.insert(0, {'operating_point': 'No abstention', **base})
    ax.set_xlabel('Coverage (fraction of molecules answered)')
    ax.set_ylabel('Selective risk (error on answered)')
    ax.legend(loc='lower right')
    V.title(ax, 'Risk–coverage trade-off',
            'Operating points must be read against the curve, not in isolation')

    ax = axes[1]
    d = metrics[(metrics.split == 'scaffold') & metrics.variant.isin(
        [v for v, _ in ABLATION])]
    if 'aurc' in d and len(d):
        s = summarise(d, ['variant'], ['aurc'])
        s = s.set_index('variant').reindex([v for v, _ in ABLATION]).dropna(
            subset=['aurc_mean']).reset_index()
        # A dot with its bootstrap interval, not a bar from zero: AURC differences between
        # stages are small relative to the value, and bars from zero hide them.
        y = np.arange(len(s))
        for i, row in s.iterrows():
            lo = row.get('aurc_ci_low', np.nan)
            hi = row.get('aurc_ci_high', np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=V.SERIES[0], linewidth=2.0,
                        solid_capstyle='round', alpha=0.85)
        ax.scatter(s.aurc_mean, y, s=46, color=V.SERIES[0], edgecolor=V.SURFACE,
                   linewidth=1.6, zorder=5)
        best = int(np.nanargmin(s.aurc_mean.to_numpy()))
        ax.scatter([s.aurc_mean.iloc[best]], [best], s=46, color=V.GOOD,
                   edgecolor=V.SURFACE, linewidth=1.6, zorder=6)
        span = np.nanmax(s.aurc_ci_high.to_numpy()) - np.nanmin(s.aurc_ci_low.to_numpy())
        ax.text(s.aurc_mean.iloc[best] + max(span, 1e-3) * 0.03, best + 0.34, 'lowest',
                ha='left', va='center', fontsize=plt.rcParams['font.size'] - 1.5, color=V.GOOD)
        ax.set_yticks(y)
        ax.set_yticklabels([dict(ABLATION)[v] for v in s.variant],
                           fontsize=plt.rcParams['font.size'] - 1.5)
        ax.invert_yaxis()
        ax.set_xlabel('AURC (lower is better)')
        ax.grid(axis='x')
        ax.grid(axis='y', visible=False)
        V.title(ax, 'Area under the risk–coverage curve',
                'Dot is the fold mean; bar is the bootstrap 95% CI')
        table(s, 'aurc_by_variant')
    else:
        V.empty(ax, 'AURC unavailable')
    table(pd.DataFrame(rows), 'risk_coverage_operating_points')
    return V.save(fig, '17_risk_coverage', out,
                  'Risk–coverage curve for the primary configuration with each abstention '
                  'mechanism marked, beside AURC across the ablation. Reporting a lower error '
                  'on an accepted subset is meaningless without its coverage.')


def ablation_cascade(out, metrics, table):
    d = metrics[(metrics.split == 'scaffold')
                & metrics.variant.isin([v for v, _ in ABLATION])]
    if not len(d):
        return None
    cols = ['coverage', 'selective_risk', 'retained_active_fraction', 'mc_balanced_accuracy',
            'moa_balanced_accuracy', 'dual_rejection_rate', 'aurc']
    s = summarise(d, ['variant'], cols)
    order = [v for v, _ in ABLATION if v in set(s.variant)]
    s = s.set_index('variant').reindex(order).reset_index()
    s['stage'] = [dict(ABLATION)[v] for v in s.variant]
    table(s, 'ablation_cascade')
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(len(s))
    ax.plot(x, s.coverage_mean, marker='o', color=V.SERIES[0], label='Coverage',
            markeredgecolor=V.SURFACE, markeredgewidth=1.6)
    ax.fill_between(x, s.coverage_mean - s.coverage_sd.fillna(0),
                    s.coverage_mean + s.coverage_sd.fillna(0), color=V.SERIES[0], alpha=0.10,
                    linewidth=0)
    ax.plot(x, s.selective_risk_mean, marker='s', color=V.SERIES[1], label='Selective risk',
            markeredgecolor=V.SURFACE, markeredgewidth=1.6)
    ax.fill_between(x, s.selective_risk_mean - s.selective_risk_sd.fillna(0),
                    s.selective_risk_mean + s.selective_risk_sd.fillna(0), color=V.SERIES[1],
                    alpha=0.10, linewidth=0)
    if 'retained_active_fraction_mean' in s:
        ax.plot(x, s.retained_active_fraction_mean, marker='^', color=V.SERIES[2],
                label='Retained actives', markeredgecolor=V.SURFACE, markeredgewidth=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(s.stage, rotation=28, ha='right',
                       fontsize=plt.rcParams['font.size'] - 1.5)
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center left')
    V.title(ax, 'What each selective mechanism costs and buys',
            'Shaded bands are ±1 SD across outer folds')
    return V.save(fig, '19_ablation_cascade', out,
                  'Coverage, selective risk and retained-active fraction as each selective '
                  'mechanism is added. Risk falling while retained actives fall is a trade, '
                  'not an improvement.')


def primary_comparison(out, metrics, cfg, table):
    """Pre-declared comparison family, paired by fold, Holm-corrected."""
    primary = cfg.get('primary', {})
    pm, pr = primary.get('model'), primary.get('representation')
    pv = primary.get('variant', 'full_knn')
    d = metrics[(metrics.split == 'scaffold') & (metrics.model == pm)
                & (metrics.representation == pr)]
    if not len(d):
        return None
    key = ['seed', 'fold']
    rows = []
    family = [v for v in BASELINES + ['hierarchy_gate_recall', 'hierarchy_conformal_nominal']
              if (d.variant == v).any()]
    for metric in ('mc_balanced_accuracy', 'moa_balanced_accuracy', 'selective_risk', 'aurc'):
        if metric not in d:
            continue
        a = d[d.variant == pv].set_index(key)[metric].astype(float)
        if not len(a):
            continue
        for base in family:
            b = d[d.variant == base].set_index(key)[metric].astype(float)
            j = a.align(b, join='inner')
            x, y = j[0].dropna(), j[1].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 3:
                continue
            x, y = x.loc[idx].to_numpy(), y.loc[idx].to_numpy()
            diff = x - y
            mean, lo, hi = MT.bootstrap_ci(diff, reps=int(cfg.get('bootstrap_repetitions', 2000)),
                                           seed=23)
            try:
                p = float(stats.wilcoxon(x, y, zero_method='zsplit').pvalue) if np.any(diff) else 1.0
            except Exception:
                p = np.nan
            higher_is_better = metric not in LOWER_IS_BETTER
            if abs(mean) < 1e-12:
                favours = 'tie'
            elif (mean > 0) == higher_is_better:
                favours = 'proposed'
            else:
                favours = 'baseline'
            rows.append({'metric': metric,
                         'direction': 'higher is better' if higher_is_better
                                      else 'LOWER is better',
                         'proposed': pv, 'baseline': base,
                         'n_folds': len(idx), 'proposed_mean': float(x.mean()),
                         'baseline_mean': float(y.mean()), 'delta_mean': mean,
                         'delta_ci_low': lo, 'delta_ci_high': hi, 'wilcoxon_p': p,
                         'favours': favours})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    ok = r.wilcoxon_p.notna()
    r.loc[ok, 'holm_p'] = MT.holm(r.loc[ok, 'wilcoxon_p'].to_numpy())
    r['significant_holm_0.05'] = r.holm_p < 0.05
    r['ci_excludes_zero'] = (r.delta_ci_low > 0) | (r.delta_ci_high < 0)
    # A significant difference is not a win. Separate the two explicitly, because for
    # selective risk and AURC a positive delta means the proposed system is WORSE.
    r['verdict'] = np.where(~r['significant_holm_0.05'], 'no difference after Holm',
                            np.where(r.favours == 'proposed', 'PROPOSED BETTER',
                                     'BASELINE BETTER'))
    r = r[['metric', 'direction', 'proposed', 'baseline', 'n_folds', 'proposed_mean',
           'baseline_mean', 'delta_mean', 'delta_ci_low', 'delta_ci_high', 'wilcoxon_p',
           'holm_p', 'significant_holm_0.05', 'ci_excludes_zero', 'favours', 'verdict']]
    table(r, 'primary_paired_comparison')
    wins = r[r.verdict == 'PROPOSED BETTER']
    losses = r[r.verdict == 'BASELINE BETTER']
    print(f'[analysis] paired comparisons: {len(wins)} favour the proposed system, '
          f'{len(losses)} favour a baseline, {len(r) - len(wins) - len(losses)} undecided',
          flush=True)

    sub = r[r.metric == 'mc_balanced_accuracy']
    if not len(sub):
        sub = r
    fig, ax = plt.subplots(figsize=(8.0, 0.52 * len(sub) + 2.3))
    ypos = np.arange(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        sig = bool(row.get('ci_excludes_zero'))
        c = V.SERIES[0] if row.delta_mean >= 0 else V.SERIES[1]
        ax.plot([row.delta_ci_low, row.delta_ci_high], [i, i], color=c,
                linewidth=2.0, solid_capstyle='round', alpha=1.0 if sig else 0.45)
        ax.scatter([row.delta_mean], [i], s=46, color=c, edgecolor=V.SURFACE, linewidth=1.6,
                   zorder=5, alpha=1.0 if sig else 0.45)
    V.reference_line(ax, 0, 'x', 'no difference')
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'vs {b}' for b in sub.baseline])
    ax.invert_yaxis()
    ax.set_xlabel(f'Paired difference in {sub.metric.iloc[0].replace("_", " ")}')
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    V.title(ax, f'Pre-declared primary ({pv}) against each baseline',
            'Fold-paired bootstrap 95% CI; faded where the interval spans zero')
    return V.save(fig, '18_primary_comparison', out,
                  'Fold-paired differences between the pre-declared primary configuration and '
                  'each baseline, with bootstrap confidence intervals and Holm-corrected '
                  'Wilcoxon p-values in tables/primary_paired_comparison.csv.')


def confusion_figure(out, table):
    pred = CS._primary_predictions(out)
    if pred is None:
        return None
    pred = pred[pred.split == 'scaffold']
    panels = [('All predictions', np.ones(len(pred), bool)),
              ('Accepted: gray zone', pred.accept_gray.astype(bool).to_numpy()
               if 'accept_gray' in pred else None),
              ('Accepted: full pipeline', pred.accept_full_knn.astype(bool).to_numpy()
               if 'accept_full_knn' in pred else None)]
    panels = [(t, m) for t, m in panels if m is not None]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels) + 0.6, 3.7))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, (t, mask) in zip(axes, panels):
        sub = pred[mask & (pred.label < 3)]
        cm = MT.confusion(sub.label.to_numpy(), sub.prediction.to_numpy(), 3)
        norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        im = ax.imshow(norm, cmap=V.SEQUENTIAL, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i, j]:,}\n{norm[i, j]*100:.0f}%', ha='center', va='center',
                        fontsize=plt.rcParams['font.size'] - 2.5,
                        color=V.SURFACE if norm[i, j] > 0.55 else V.INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Inact.', 'Agon.', 'Antag.'])
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Inact.', 'Agon.', 'Antag.'] if ax is axes[0] else ['', '', ''])
        ax.set_xlabel('Predicted')
        if ax is axes[0]:
            ax.set_ylabel('True')
        ax.grid(False)
        V.title(ax, t, f'n = {len(sub):,}')
        for i in range(3):
            for j in range(3):
                rows.append({'panel': t, 'true': V.CLASS_LABEL[i], 'predicted': V.CLASS_LABEL[j],
                             'n': int(cm[i, j]), 'row_fraction': float(norm[i, j])})
    table(pd.DataFrame(rows), 'confusion_matrices')
    return V.save(fig, '22_confusion_matrices', out,
                  'Row-normalised confusion matrices before abstention and on each accepted '
                  'subset, with absolute counts. Panel sample sizes differ by construction.')


def threshold_stability(out, table):
    files = sorted((out / 'folds').glob('*/thresholds.json')) if (out / 'folds').exists() else []
    if not files:
        return None
    rows = []
    for f in files:
        try:
            t = json.loads(f.read_text())
        except Exception:
            continue
        parts = f.parent.name.split('_')
        rows.append({'cell': f.parent.name, 'split': parts[0],
                     'gate_threshold': t.get('gate_threshold'),
                     'gray_threshold': t.get('gray_threshold'),
                     'allocation_mode': t.get('allocation_mode'),
                     'budget_satisfied': t.get('calibration_budget_satisfied'),
                     'alpha_class2': (t.get('alpha_effective') or {}).get('2')})
    d = pd.DataFrame(rows)
    d = d[d.split == 'scaffold']
    if not len(d):
        return None
    table(d, 'threshold_stability')
    cols = [c for c in ('gate_threshold', 'gray_threshold') if d[c].notna().any()]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(5.4 + 1.4 * len(cols), 3.8))
    V.boxplot(ax, [pd.to_numeric(d[c], errors='coerce').dropna().to_numpy() for c in cols],
              [c.replace('_', ' ') for c in cols],
              colors=[V.SERIES[0], V.SERIES[1]][:len(cols)])
    ax.set_ylabel('Threshold value')
    V.title(ax, 'Operating thresholds across folds',
            'Each threshold is re-selected inside its own fold; spread is the honest uncertainty')
    return V.save(fig, '24_threshold_stability', out,
                  'Gate and gray-zone thresholds selected independently in each outer fold.')


def dual_and_control(out, metrics, table):
    d = metrics[metrics.split == 'scaffold']
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    dv = d[d.variant.isin([v for v, _ in ABLATION])]
    if 'dual_rejection_rate' in dv and dv.dual_rejection_rate.notna().any():
        s = summarise(dv, ['variant'], ['dual_rejection_rate'])
        order = [v for v, _ in ABLATION if v in set(s.variant)]
        s = s.set_index('variant').reindex(order).dropna(
            subset=['dual_rejection_rate_mean']).reset_index()
        V.rounded_bars(ax, np.arange(len(s)), s.dual_rejection_rate_mean.to_numpy(),
                       V.CLASS_COLOR[3], width=0.52)
        ax.errorbar(np.arange(len(s)), s.dual_rejection_rate_mean, yerr=s.dual_rejection_rate_sd,
                    fmt='none', ecolor=V.INK_3, elinewidth=1.0)
        ax.set_xticks(np.arange(len(s)))
        ax.set_xticklabels([dict(ABLATION)[v] for v in s.variant], rotation=30, ha='right',
                           fontsize=plt.rcParams['font.size'] - 2)
        ax.set_ylabel('Dual-active rejection rate')
        ax.set_ylim(0, 1.05)
        V.title(ax, 'Abstention on molecules with both activities',
                'Never trained on; a trustworthy system should decline them')
        table(s, 'dual_active_rejection')
    else:
        V.empty(ax, 'no dual-active molecules in the held-out folds')

    ax = axes[1]
    ctrl = d[d.variant == 'permuted_labels_control']
    real = d[d.variant == 'hierarchy_calibrated']
    if len(ctrl) and 'activity_pr_auc' in ctrl:
        groups, labels, colors = [], [], []
        for frame, lab, c in ((real, 'Real labels', V.SERIES[0]),
                              (ctrl, 'Permuted labels', V.SERIES[7])):
            v = pd.to_numeric(frame.get('activity_pr_auc'), errors='coerce').dropna()
            if len(v):
                groups.append(v.to_numpy())
                labels.append(lab)
                colors.append(c)
        if groups:
            V.boxplot(ax, groups, labels, colors=colors)
            prev = pd.to_numeric(d.get('activity_prevalence'), errors='coerce').dropna()
            if len(prev):
                V.reference_line(ax, float(prev.mean()), 'y', 'prevalence (chance PR-AUC)')
            ax.set_ylabel('Activity-gate PR-AUC')
            V.title(ax, 'Label-permutation negative control',
                    'Signal must collapse to prevalence when labels are scrambled')
            table(summarise(d[d.variant.isin(['hierarchy_calibrated',
                                              'permuted_labels_control'])],
                            ['variant'], ['activity_pr_auc', 'activity_roc_auc',
                                          'mc_balanced_accuracy']),
                  'permutation_control')
    else:
        V.empty(ax, 'permutation control not run')
    return V.save(fig, '25_dual_and_control', out,
                  'Dual-active abstention across the ablation, and the label-permutation '
                  'negative control against the prevalence floor.')


def external_figure(out, table):
    path = out / 'external/external_metrics.csv'
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if not len(d):
        return None
    cols = [c for c in ('external_specificity', 'external_false_positive_rate',
                        'external_recall', 'external_moa_balanced_accuracy',
                        'external_moa_accepted_accuracy', 'coverage',
                        'median_nn_tanimoto_fit') if c in d]
    s = summarise(d, ['cohort'], cols)
    table(s, 'external_cohort_summary')
    panels = [c for c in ('external_specificity', 'external_recall',
                          'external_moa_balanced_accuracy') if f'{c}_mean' in s]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 0.8, 3.9),
                             squeeze=False)
    for ax, col in zip(axes[0], panels):
        sub = s[s[f'{col}_mean'].notna()]
        if not len(sub):
            V.empty(ax, f'{col}\nunavailable')
            continue
        groups = [pd.to_numeric(d.loc[d.cohort == c, col], errors='coerce').dropna().to_numpy()
                  for c in sub.cohort]
        V.boxplot(ax, groups, [c.replace('chembl_', '').replace('_', '\n') for c in sub.cohort],
                  colors=[V.SERIES[(i + 2) % len(V.SERIES)] for i in range(len(sub))])
        ax.set_ylabel(col.replace('external_', '').replace('_', ' '))
        ax.set_ylim(0, 1.05)
        V.title(ax, col.replace('external_', '').replace('_', ' '))
    fig.suptitle('Independent external cohorts (never trained, tuned or calibrated on)',
                 x=0.02, ha='left', y=1.05)
    return V.save(fig, '27_external_cohorts', out,
                  'Behaviour on ChEMBL and cross-platform BLA cohorts. Each point is one '
                  'fold model applied to the whole cohort; cohort labels come from different '
                  'assay systems, so these measure behaviour under distribution shift.')


def importance_stability(out, table, top=18):
    files = sorted((out / 'folds').glob('*/models.joblib')) if (out / 'folds').exists() else []
    if not files:
        return None
    import joblib
    rows = []
    for f in files[:12]:
        try:
            obj = joblib.load(f)
            mdl = obj['gate'].named_steps['model']
            imp = getattr(mdl, 'feature_importances_', None)
            if imp is None:
                coef = getattr(mdl, 'coef_', None)
                if coef is None:
                    continue
                imp = np.abs(coef).mean(0)
            rows.append(pd.Series(imp, name=f.parent.name))
        except Exception:
            continue
    if len(rows) < 2:
        return None
    M = pd.concat(rows, axis=1)
    mean = M.mean(1).sort_values(ascending=False)
    idx = mean.index[:top]
    d = pd.DataFrame({'feature_index': idx, 'mean_importance': mean[idx].to_numpy(),
                      'sd_importance': M.loc[idx].std(1).to_numpy(),
                      'folds': M.shape[1]})
    table(d, 'feature_importance_stability')
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(idx) + 1.8))
    V.rounded_bars(ax, np.arange(len(idx)), d.mean_importance.to_numpy(), V.SERIES[0],
                   width=0.6, horizontal=True)
    ax.errorbar(d.mean_importance, np.arange(len(idx)), xerr=d.sd_importance, fmt='none',
                ecolor=V.INK_3, elinewidth=1.0)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([f'feature {i}' for i in idx], fontsize=plt.rcParams['font.size'] - 2)
    ax.invert_yaxis()
    ax.set_xlabel('Gate feature importance')
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    V.title(ax, 'Feature-importance stability across folds',
            f'mean ± SD over {M.shape[1]} fold models')
    return V.save(fig, '30_importance_stability', out,
                  'Most important activity-gate features with their across-fold variability. '
                  'Large error bars indicate the ranking is fold-specific.')


def tier_sensitivity(out, table, cfg):
    """Compare this run against a sibling results directory built from the other label tier."""
    this_tier = cfg.get('label_tier')
    other = 'core' if this_tier == 'extended' else 'extended'
    sib = out.parent / f'{out.name}_{other}'
    rows = []
    for tier, base in ((this_tier, out), (other, sib)):
        f = base / 'fold_metrics.csv'
        s = base / 'calibration_budget_summary.json'
        if not f.exists():
            continue
        m = pd.read_csv(f, low_memory=False)
        pr = cfg.get('primary', {})
        # Restrict to the pre-declared primary cell. Averaging the variant over every model and
        # representation answers a different question and produces numbers that do not match
        # anything else in the manuscript.
        m = m[(m.split == 'scaffold') & (m.variant == pr.get('variant', 'full_knn'))
              & (m.model == pr.get('model')) & (m.representation == pr.get('representation'))]
        rec = {'tier': tier, 'model': pr.get('model'),
               'representation': pr.get('representation'), 'n_folds': int(len(m))}
        for c in ('mc_balanced_accuracy', 'moa_balanced_accuracy', 'coverage',
                  'selective_risk', 'n_conformal_class2'):
            if c in m:
                v = pd.to_numeric(m[c], errors='coerce').dropna()
                if len(v):
                    rec[f'{c}_mean'] = float(v.mean())
                    rec[f'{c}_sd'] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        if s.exists():
            rec['all_components_feasible'] = json.loads(s.read_text()).get(
                'all_components_feasible')
        rows.append(rec)
    if len(rows) < 2:
        (out / 'tables').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows or [{'tier': this_tier,
                               'note': f'run results/{out.name}_{other} with label_tier={other} '
                                       f'to populate the tier sensitivity comparison'}]
                     ).to_csv(out / 'tables/tier_sensitivity.csv', index=False)
        return None
    return table(pd.DataFrame(rows), 'tier_sensitivity')


def headline_tables(out, metrics, cfg, table):
    cols = ['coverage', 'selective_risk', 'retained_active_fraction', 'mc_balanced_accuracy',
            'mc_mcc', 'mc_brier', 'mc_ece', 'moa_balanced_accuracy', 'moa_mcc',
            'moa_f1_antagonist', 'activity_pr_auc', 'activity_roc_auc', 'activity_bedroc20',
            'activity_bedroc20_random', 'activity_ef1', 'activity_ef5', 'gate_recall',
            'gate_specificity', 'aurc', 'dual_rejection_rate']
    sc = metrics[metrics.split == 'scaffold']
    table(summarise(sc, ['variant'], cols, reps=int(cfg.get('bootstrap_repetitions', 2000))),
          'mean_sd_all_metrics_scaffold')
    if (metrics.split == 'random').any():
        table(summarise(metrics[metrics.split == 'random'], ['variant'], cols),
              'mean_sd_all_metrics_random')
    table(metrics, 'fold_metrics_raw')
    counts = pd.read_csv(out / 'data/master.csv', low_memory=False, usecols=['label_name',
                                                                            'moa_confidence'])
    table(counts.value_counts().reset_index(name='n'), 'dataset_counts')
    if (out / 'calibration_budget_all.csv').exists():
        table(pd.read_csv(out / 'calibration_budget_all.csv'), 'calibration_budget_per_cell')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/full.json')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / 'results' / cfg['profile']
    fm = out / 'fold_metrics.csv'
    if not fm.exists():
        raise SystemExit(f'{fm} missing. Run 01_run_experiments.sh first.')
    V.apply_theme()
    cap = out / 'figures/captions.md'
    if cap.exists():
        cap.unlink()
    table = make_table(out)
    metrics = pd.read_csv(fm, low_memory=False)
    master = pd.read_csv(out / 'data/master.csv', low_memory=False)
    print(f'[analysis] {len(metrics):,} metric rows, {metrics.variant.nunique()} variants, '
          f'{len(master):,} molecules', flush=True)

    headline_tables(out, metrics, cfg, table)
    made = []
    steps = [
        ('calibration_budget', lambda: calibration_budget_figure(out, table)),
        ('conformal_coverage', lambda: conformal_coverage_figure(out, metrics, table)),
        ('box_pr_auc', lambda: benchmark_boxplots(
            out, metrics, table, 'activity_pr_auc', 'hierarchy_gate_recall',
            '12_benchmark_box_activity_pr_auc', 'Activity-gate PR-AUC',
            'Activity-gate PR-AUC for every model family, one box per representation and one '
            'point per outer scaffold fold.')),
        ('box_moa', lambda: benchmark_boxplots(
            out, metrics, table, 'moa_balanced_accuracy', 'hierarchy_gate_recall',
            '13_benchmark_box_moa', 'Mode-of-action balanced accuracy',
            'Mode-of-action balanced accuracy across outer scaffold folds. Spread reflects the '
            'small antagonist class.')),
        ('heatmap', lambda: benchmark_heatmap(out, metrics, table)),
        ('reliability', lambda: reliability_curves(out, table)),
        ('cal_benchmark', lambda: calibration_benchmark(out, metrics, table)),
        ('risk_coverage', lambda: risk_coverage(out, metrics, table)),
        ('primary_comparison', lambda: primary_comparison(out, metrics, cfg, table)),
        ('ablation', lambda: ablation_cascade(out, metrics, table)),
        ('confusion', lambda: confusion_figure(out, table)),
        ('thresholds', lambda: threshold_stability(out, table)),
        ('dual_control', lambda: dual_and_control(out, metrics, table)),
        ('external', lambda: external_figure(out, table)),
        ('importance', lambda: importance_stability(out, table)),
        ('tier_sensitivity', lambda: tier_sensitivity(out, table, cfg)),
    ]
    for name, fn in steps:
        try:
            r = fn()
            if r:
                made.extend(r if isinstance(r, list) else [r])
        except Exception as e:
            print(f'[analysis] {name} skipped: {e!r}', flush=True)
    made += CS.run_all(out, master, metrics, table)

    from paper import build_paper
    build_paper(out, cfg, metrics, table)

    figs = sorted(p.name for p in (out / 'figures').glob('*.png'))
    tabs = sorted(p.name for p in (out / 'tables').glob('*.csv'))
    (out / 'analysis_manifest.json').write_text(json.dumps(
        {'figures_png': figs, 'tables_csv': tabs, 'figure_count': len(figs),
         'table_count': len(tabs), 'variants': sorted(metrics.variant.unique()),
         'profile': cfg['profile'], 'label_tier': cfg.get('label_tier')}, indent=2))
    print(f'[analysis] {len(figs)} figures, {len(tabs)} tables written to results/'
          f'{cfg["profile"]}', flush=True)


if __name__ == '__main__':
    main()

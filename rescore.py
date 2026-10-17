"""Threshold-policy and external-discrimination diagnostics. No model is refitted.

Two questions the main run left open, both answerable from the serialized primary fold models:

1. **Was the operating point the problem?** The gate threshold was selected to hit a 95% recall
   target. At a ~18% active prevalence that drives tau very low, so the gate flags most of the
   library. This re-selects tau on the SAME disjoint policy partition under three declared
   policies -- the original recall target, Youden's J, and maximum MCC -- and reports what each
   costs and buys on the untouched test fold. Selection never sees the test fold.

2. **Does the model discriminate out of domain, or is only the threshold wrong?** The ChEMBL
   specificity and sensitivity cohorts are scored and POOLED into one labelled set, which gives
   a threshold-free external ROC-AUC and PR-AUC. A high external AUC with a bad external
   specificity means the ranking transfers and the threshold does not; an AUC near 0.5 means the
   model does not transfer at all. These are different papers, and this distinguishes them.

    python rescore.py --profile full          # run before 02_finish_analysis.sh
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

import joblib                                                      # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from sklearn.metrics import (average_precision_score, roc_auc_score,  # noqa: E402
                             roc_curve, matthews_corrcoef, balanced_accuracy_score)

import viz as V                                                    # noqa: E402
from data import ROOT, features                                    # noqa: E402
from metrics import bootstrap_ci                                   # noqa: E402

POLICIES = ('recall_target', 'youden_j', 'max_mcc')
POLICY_LABEL = {'recall_target': 'Recall target (as run)',
                'youden_j': "Youden's J",
                'max_mcc': 'Maximum MCC'}


def _operating(y, p, tau):
    flag = np.asarray(p) >= tau
    y = np.asarray(y).astype(int)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    mcc = (matthews_corrcoef(y, flag.astype(int))
           if flag.any() and (~flag).any() else np.nan)
    bal = (balanced_accuracy_score(y, flag.astype(int))
           if len(np.unique(y)) > 1 else np.nan)
    return {'tau': float(tau), 'recall': rec, 'specificity': spec, 'precision': prec,
            'mcc': mcc, 'balanced_accuracy': bal, 'flagged_fraction': float(flag.mean()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def _select(y_pol, p_pol, target_recall):
    """Choose tau under each policy using ONLY the policy partition."""
    y_pol = np.asarray(y_pol).astype(int)
    p_pol = np.asarray(p_pol, dtype=float)
    out = {}
    pos = np.sort(p_pol[y_pol == 1])
    idx = int(np.clip(np.floor((1 - target_recall) * len(pos)), 0, max(len(pos) - 1, 0)))
    out['recall_target'] = float(pos[idx]) if len(pos) else 0.5
    grid = np.unique(np.quantile(p_pol, np.linspace(0.005, 0.995, 400)))
    stats = [_operating(y_pol, p_pol, t) for t in grid]
    j = [(s['recall'] + s['specificity'] - 1) if np.isfinite(s['recall']) and
         np.isfinite(s['specificity']) else -np.inf for s in stats]
    m = [s['mcc'] if np.isfinite(s['mcc']) else -np.inf for s in stats]
    out['youden_j'] = float(grid[int(np.argmax(j))])
    out['max_mcc'] = float(grid[int(np.argmax(m))])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='full')
    args = ap.parse_args()

    cfg = json.loads((ROOT_DIR / f'config/{args.profile}.json').read_text())
    cfg['pretrained'] = False
    out = ROOT / 'results' / args.profile
    primary = cfg.get('primary', {})
    rep, model = primary.get('representation'), primary.get('model')
    folds = [out / 'folds' / f'scaffold_s{s}_f{f}_{rep}_{model}'
             for s in cfg['seeds'] for f in range(cfg['outer_folds'])]
    folds = [p for p in folds if (p / 'models.joblib').exists()]
    if not folds:
        raise SystemExit(f'no serialized primary models under results/{args.profile}/folds')
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(out / 'data/master.csv', low_memory=False)
    print(f'[rescore] {len(folds)} primary fold models, {len(master):,} molecules', flush=True)
    X, _ = features(master, cfg, out / 'data', write=False)
    x_all = X[rep]
    y_active_all = (master.label.to_numpy() > 0).astype(int)
    rid_to_pos = {int(r): i for i, r in enumerate(master.row_id.to_numpy())}

    # ---------------------------------------------------------------- 1. thresholds
    rows, curves = [], []
    for fd in folds:
        obj = joblib.load(fd / 'models.joblib')
        member = pd.read_csv(fd / 'split_membership.csv')
        pol = np.array([rid_to_pos[int(r)] for r in
                        member.loc[member.role == 'policy', 'row_id']])
        tst = np.array([rid_to_pos[int(r)] for r in
                        member.loc[member.role == 'test', 'row_id']])
        score = lambda idx: obj['gate_calibrator'].transform(
            obj['gate'].predict_proba(x_all[idx]))[:, 1]
        p_pol, p_tst = score(pol), score(tst)
        y_pol, y_tst = y_active_all[pol], y_active_all[tst]
        taus = _select(y_pol, p_pol, cfg['target_active_recall'])
        for pol_name, tau in taus.items():
            rec = _operating(y_tst, p_tst, tau)
            rec.update({'policy': pol_name, 'fold': fd.name,
                        'n_policy': int(len(pol)), 'n_test': int(len(tst)),
                        'test_roc_auc': float(roc_auc_score(y_tst, p_tst))
                        if len(np.unique(y_tst)) > 1 else np.nan,
                        'test_pr_auc': float(average_precision_score(y_tst, p_tst))
                        if len(np.unique(y_tst)) > 1 else np.nan,
                        'test_prevalence': float(y_tst.mean())})
            rows.append(rec)
        if len(np.unique(y_tst)) > 1:
            fpr, tpr, _ = roc_curve(y_tst, p_tst)
            curves.append((fpr, tpr))
    thr = pd.DataFrame(rows)
    thr.to_csv(out / 'tables/threshold_policy_per_fold.csv', index=False)

    summary = []
    for pol_name in POLICIES:
        s = thr[thr.policy == pol_name]
        rec = {'policy': pol_name, 'label': POLICY_LABEL[pol_name], 'n_folds': int(len(s))}
        for c in ('tau', 'recall', 'specificity', 'precision', 'mcc', 'balanced_accuracy',
                  'flagged_fraction'):
            v = pd.to_numeric(s[c], errors='coerce').dropna().to_numpy()
            mean, lo, hi = bootstrap_ci(v, reps=2000, seed=5)
            rec[f'{c}_mean'] = mean
            rec[f'{c}_sd'] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            rec[f'{c}_ci_low'], rec[f'{c}_ci_high'] = lo, hi
        summary.append(rec)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'tables/threshold_policy_comparison.csv', index=False)
    try:
        summary.to_markdown(out / 'tables/threshold_policy_comparison.md', index=False,
                            floatfmt='.4f')
    except Exception:
        pass
    print('[rescore] threshold policies (test fold, selection on policy partition):', flush=True)
    for r in summary.to_dict('records'):
        print(f"   {r['label']:24s} tau={r['tau_mean']:.3f}  recall={r['recall_mean']:.3f}  "
              f"spec={r['specificity_mean']:.3f}  MCC={r['mcc_mean']:.3f}", flush=True)

    # ---------------------------------------------------------------- 2. external
    ext_dir = out / 'external'
    spec_p = ext_dir / 'chembl_specificity.csv'
    sens_p = ext_dir / 'chembl_sensitivity.csv'
    ext_rows, ext_curves = [], []
    if spec_p.exists() and sens_p.exists():
        neg = pd.read_csv(spec_p).assign(y=0)
        pos = pd.read_csv(sens_p).assign(y=1)
        pool = pd.concat([neg, pos], ignore_index=True)
        pool = pool.drop_duplicates('inchikey').reset_index(drop=True)
        pool['row_id'] = np.arange(len(pool))
        print(f'[rescore] external pool: {int((pool.y==0).sum())} asserted non-binders + '
              f'{int((pool.y==1).sum())} asserted binders', flush=True)
        Xe, _ = features(pool, cfg, ext_dir / 'pool_features', write=False)
        xe = Xe[rep]
        ye = pool.y.to_numpy()
        for fd in folds:
            obj = joblib.load(fd / 'models.joblib')
            pe = obj['gate_calibrator'].transform(obj['gate'].predict_proba(xe))[:, 1]
            rec = {'fold': fd.name, 'n': int(len(pool)),
                   'n_negative': int((ye == 0).sum()), 'n_positive': int((ye == 1).sum()),
                   'external_roc_auc': float(roc_auc_score(ye, pe)),
                   'external_pr_auc': float(average_precision_score(ye, pe)),
                   'external_prevalence': float(ye.mean())}
            tau_by_policy = thr[thr.fold == fd.name].set_index('policy').tau.to_dict()
            for pol_name, tau in tau_by_policy.items():
                op = _operating(ye, pe, tau)
                rec[f'{pol_name}_tau'] = op['tau']
                rec[f'{pol_name}_external_recall'] = op['recall']
                rec[f'{pol_name}_external_specificity'] = op['specificity']
                rec[f'{pol_name}_external_mcc'] = op['mcc']
            ext_rows.append(rec)
            fpr, tpr, _ = roc_curve(ye, pe)
            ext_curves.append((fpr, tpr))
        ext = pd.DataFrame(ext_rows)
        ext.to_csv(out / 'tables/external_discrimination_per_fold.csv', index=False)
        es = {'cohort': 'chembl_pooled', 'n': int(len(pool)),
              'n_negative': int((ye == 0).sum()), 'n_positive': int((ye == 1).sum())}
        for c in [c for c in ext.columns if c not in ('fold', 'n', 'n_negative', 'n_positive')]:
            v = pd.to_numeric(ext[c], errors='coerce').dropna().to_numpy()
            if not len(v):
                continue
            mean, lo, hi = bootstrap_ci(v, reps=2000, seed=9)
            es[f'{c}_mean'] = mean
            es[f'{c}_sd'] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            es[f'{c}_ci_low'], es[f'{c}_ci_high'] = lo, hi
        pd.DataFrame([es]).to_csv(out / 'tables/external_discrimination.csv', index=False)
        auc = es.get('external_roc_auc_mean', float('nan'))
        print(f'[rescore] EXTERNAL ROC-AUC (threshold-free) = {auc:.3f} '
              f'± {es.get("external_roc_auc_sd", float("nan")):.3f}', flush=True)
        verdict = ('ranking transfers; the operating point is what fails'
                   if auc >= 0.65 else
                   'the model does not discriminate out of domain either' if auc < 0.58 else
                   'partial transfer; interpret with care')
        print(f'[rescore] verdict: {verdict}', flush=True)
        (out / 'tables/external_discrimination_verdict.json').write_text(json.dumps(
            {'external_roc_auc': auc, 'verdict': verdict,
             'interpretation': 'A high threshold-free AUC alongside a low specificity at the '
                               'deployed threshold means the ranking transfers across the '
                               'distribution shift while the threshold does not. A low AUC '
                               'means the model itself does not transfer.'}, indent=2))
    else:
        print('[rescore] ChEMBL cohorts not found; external discrimination skipped', flush=True)

    # ---------------------------------------------------------------- figures
    V.apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3),
                             gridspec_kw={'width_ratios': [1, 1.1], 'wspace': 0.32})
    ax = axes[0]
    for fpr, tpr in curves:
        ax.plot(fpr, tpr, color=V.SERIES[0], alpha=0.22, linewidth=1.2)
    ax.plot([0, 1], [0, 1], color=V.INK_3, linewidth=1.0, linestyle=(0, (4, 3)))
    for i, pol_name in enumerate(POLICIES):
        s = summary[summary.policy == pol_name].iloc[0]
        ax.scatter([1 - s.specificity_mean], [s.recall_mean], s=74,
                   marker=V.CLASS_MARKER[i], color=V.SERIES[i + 1], edgecolor=V.SURFACE,
                   linewidth=1.8, zorder=5, label=POLICY_LABEL[pol_name])
    ax.set_xlabel('False-positive rate (1 − specificity)')
    ax.set_ylabel('Recall')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')
    V.title(ax, 'Where each threshold policy lands',
            'Faint lines: per-fold ROC. Thresholds chosen on the policy partition only')
    ax = axes[1]
    pos = np.arange(len(POLICIES))
    width = 0.26
    for k, (col, lab) in enumerate((('recall', 'Recall'), ('specificity', 'Specificity'),
                                    ('mcc', 'MCC'))):
        vals = [summary[summary.policy == p][f'{col}_mean'].iloc[0] for p in POLICIES]
        errs = [summary[summary.policy == p][f'{col}_sd'].iloc[0] for p in POLICIES]
        ax.bar(pos + (k - 1) * (width + 0.015), vals, width=width, color=V.SERIES[k],
               linewidth=0, label=lab)
        ax.errorbar(pos + (k - 1) * (width + 0.015), vals, yerr=errs, fmt='none',
                    ecolor=V.INK_3, elinewidth=1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels([POLICY_LABEL[p] for p in POLICIES], rotation=18, ha='right',
                       fontsize=plt.rcParams['font.size'] - 1.5)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc='upper center')
    V.title(ax, 'What each policy costs and buys', 'Held-out folds, mean ± SD')
    V.save(fig, '31_threshold_policy', out,
           'Activity-gate operating points under three declared threshold policies, each '
           'selected on the disjoint policy partition and evaluated on the untouched test fold. '
           'The recall target used in the main run buys sensitivity at a specificity most '
           'screening workflows would reject.')

    if ext_curves:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                 gridspec_kw={'wspace': 0.3})
        ax = axes[0]
        for fpr, tpr in ext_curves:
            ax.plot(fpr, tpr, color=V.SERIES[2], alpha=0.3, linewidth=1.3)
        ax.plot([0, 1], [0, 1], color=V.INK_3, linewidth=1.0, linestyle=(0, (4, 3)),
                label='Chance')
        ax.set_xlabel('False-positive rate')
        ax.set_ylabel('Recall')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc='lower right')
        V.title(ax, 'External discrimination, threshold-free',
                f'Pooled ChEMBL cohorts, AUC = {es.get("external_roc_auc_mean", float("nan")):.3f}'
                f' ± {es.get("external_roc_auc_sd", float("nan")):.3f}')
        ax = axes[1]
        for k, pol_name in enumerate(POLICIES):
            vals = [es.get(f'{pol_name}_external_recall_mean', np.nan),
                    es.get(f'{pol_name}_external_specificity_mean', np.nan)]
            ax.bar(np.arange(2) + (k - 1) * 0.27, vals, width=0.25, color=V.SERIES[k + 1],
                   linewidth=0, label=POLICY_LABEL[pol_name])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['External recall', 'External specificity'])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=plt.rcParams['font.size'] - 1.5)
        V.title(ax, 'Same model, different thresholds',
                'On the independent ChEMBL pool')
        V.save(fig, '32_external_discrimination', out,
               'Threshold-free external discrimination on the pooled ChEMBL cohorts, and the '
               'recall/specificity each threshold policy produces there. A high AUC beside a '
               'low specificity at the deployed threshold locates the failure in the operating '
               'point rather than in the model.')

    print(f'\n[rescore] wrote tables/threshold_policy_comparison.csv, '
          f'tables/external_discrimination.csv and two figures.\n'
          f'Next:  PROFILE={args.profile} bash 02_finish_analysis.sh', flush=True)


if __name__ == '__main__':
    main()

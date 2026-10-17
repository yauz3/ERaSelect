"""External evaluation on cohorts the pipeline never trained, tuned or calibrated on.

Cohorts
-------
chembl_specificity  ChEMBL ERa measurements asserting weak/absent potency  -> external FPR
chembl_sensitivity  ChEMBL ERa measurements asserting potency              -> external recall
chembl_moa          ChEMBL curated AGONIST/ANTAGONIST mechanism labels     -> external MoA
bla_cross_platform  Tox21 ER-BLA (GAL4/LBD fusion) exclusive actives       -> platform transfer

Every cohort is InChIKey-disjoint from the entire Tox21 VM7 curation, and every cohort's
Tanimoto distance to the training partition is measured and reported, because an "external"
set that is chemically identical to the training set validates nothing.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import metrics as MT
import selective as SEL
from data import ROOT, standardize, features
from experiments import nearest_similarity

BLA_REFERENCE = ROOT / 'data/reference/Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv'


def _load_bla(out):
    """Tox21 ER-BLA exclusive actives, disjoint from the whole VM7 curation."""
    if not BLA_REFERENCE.exists():
        return pd.DataFrame(), {'available': False, 'reason': f'{BLA_REFERENCE.name} absent'}
    src = pd.read_csv(BLA_REFERENCE, sep=';', encoding='utf-8-sig')
    smi_col = next((c for c in src if 'smiles' in c.lower()), None)
    lab_col = next((c for c in src if 'binary_label' in c.lower()), None)
    if smi_col is None or lab_col is None:
        return pd.DataFrame(), {'available': False,
                                'reason': f'unexpected BLA schema: {list(src)[:8]}'}
    rows, bad = [], 0
    for _, r in src.iterrows():
        if not isinstance(r[smi_col], str) or not r[smi_col].strip():
            bad += 1
            continue
        try:
            s, key, group = standardize(r[smi_col])
        except Exception:
            bad += 1
            continue
        # The reference encodes agonist=1 / antagonist=0.
        rows.append({'smiles': s, 'inchikey': key, 'group': group,
                     'label': 1 if int(r[lab_col]) == 1 else 2, 'expected_active': 1})
    d = pd.DataFrame(rows)
    if d.empty:
        return d, {'available': False, 'reason': 'no parsable BLA rows'}
    conflict = d.groupby('inchikey').label.nunique()
    d = d[~d.inchikey.isin(conflict[conflict > 1].index)].drop_duplicates('inchikey')
    curated = set(pd.read_csv(out / 'data/curation_all.csv', usecols=['inchikey']).inchikey.dropna())
    n0 = len(d)
    d = d[~d.inchikey.isin(curated)].reset_index(drop=True)
    d['row_id'] = np.arange(len(d))
    return d, {'available': True, 'candidates': int(n0), 'disjoint': int(len(d)),
               'removed_overlapping_identities': int(n0 - len(d)), 'unparsable': int(bad),
               'population': 'BLA exclusive actives only; contains no screening negatives, so '
                             'no specificity or enrichment metric is computable from it'}


def _fold_dirs(out, rep, model):
    return sorted((out / 'folds').glob(f'scaffold_*_{rep}_{model}'))


def evaluate_external(cfg, out, master):
    dest = out / 'external'
    dest.mkdir(parents=True, exist_ok=True)
    primary = cfg.get('primary', {})
    rep, model = primary.get('representation'), primary.get('model')
    report = {'primary': {'model': model, 'representation': rep}}

    cohorts = {}
    try:
        from chembl import build as build_chembl
        ch, ch_report = build_chembl(cfg, out)
        report['chembl'] = ch_report
        for k, v in ch.items():
            if len(v):
                cohorts[f'chembl_{k}'] = v
    except Exception as e:
        report['chembl'] = {'enabled': cfg.get('chembl', {}).get('enabled', False),
                            'error': repr(e),
                            'note': 'External ChEMBL cohorts unavailable; no substitute was '
                                    'generated. Claims requiring external validation must not '
                                    'be made from this run.'}
        print(f'[external] ChEMBL unavailable: {e!r}', flush=True)

    bla, bla_report = _load_bla(out)
    report['bla_cross_platform'] = bla_report
    if len(bla):
        cohorts['bla_cross_platform'] = bla

    folds = _fold_dirs(out, rep, model)
    if not folds:
        report['status'] = (f'no serialized primary models at scaffold_*_{rep}_{model}; '
                            f'external evaluation skipped')
        (dest / 'external_report.json').write_text(json.dumps(report, indent=2, default=str))
        return report
    if not cohorts:
        report['status'] = 'no external cohort available'
        (dest / 'external_report.json').write_text(json.dumps(report, indent=2, default=str))
        return report

    rows, sims = [], []
    for cname, cohort in cohorts.items():
        cohort.to_csv(dest / f'cohort_{cname}.csv', index=False)
        Xc, fpc = features(cohort, cfg, dest / f'features_{cname}', write=False)
        x = Xc[rep]
        for fd in folds:
            obj = joblib.load(fd / 'models.joblib')
            th = obj['thresholds']
            pa = obj['gate_calibrator'].transform(obj['gate'].predict_proba(x))[:, 1]
            pm = obj['moa_calibrator'].transform(obj['moa'].predict_proba(x))[:, 1]
            p = np.column_stack([1 - pa, pa * pm, pa * (1 - pm)])
            tau = float(th['gate_threshold'])
            pred = np.where(pa < tau, 0, np.where(pm >= .5, 1, 2))
            conf = np.where(pred == 0, 1 - pa, np.maximum(pm, 1 - pm))

            # Similarity to this fold's own fit partition (the only legitimate reference set).
            member = pd.read_csv(fd / 'split_membership.csv')
            fit_ids = member.loc[member.role == 'fit', 'row_id'].to_numpy()
            fit_rows = master.index[master.row_id.isin(fit_ids)].to_numpy()
            _, fp_master = features(master.iloc[fit_rows], cfg, dest / 'tmp_fit', write=False)
            sim = nearest_similarity(fpc, fp_master)
            sims.append(pd.DataFrame({'cohort': cname, 'fold_model': fd.name,
                                      'inchikey': cohort.inchikey, 'nn_tanimoto_fit': sim}))

            z = obj['verifiers']['preprocess'].transform(x)
            support = np.ones(len(cohort), bool)
            for c in (1, 2):
                m = pred == c
                if m.any():
                    knn = obj['verifiers'][f'knn_{c}']
                    s = -knn.kneighbors(z[m], return_distance=True)[0].mean(1)
                    support[m] = s >= th['verifier_thresholds']['knn'][c - 1]
            accepted = support & (conf >= float(th['gray_threshold']))

            rec = {'cohort': cname, 'fold_model': fd.name, 'n': int(len(cohort)),
                   'median_nn_tanimoto_fit': float(np.nanmedian(sim)),
                   'fraction_nn_tanimoto_below_0.3': float(np.mean(sim < .3)),
                   'coverage': float(accepted.mean()),
                   'flagged_active_fraction': float((pred > 0).mean())}
            exp = cohort.expected_active.to_numpy()
            if (exp == 0).all():
                rec['external_false_positive_rate'] = float((pred > 0).mean())
                rec['external_specificity'] = float((pred == 0).mean())
            elif (exp == 1).all():
                rec['external_recall'] = float((pred > 0).mean())
            if 'label' in cohort and cohort.label.notna().any():
                yl = cohort.label.astype(int).to_numpy()
                rec.update(MT.moa_metrics(yl, pred, pm, 'external_moa'))
                acc_mask = accepted & (pred > 0)
                if acc_mask.sum():
                    rec['external_moa_accepted_accuracy'] = float(
                        (pred[acc_mask] == yl[acc_mask]).mean())
                    rec['external_moa_accepted_n'] = int(acc_mask.sum())
                rec.update({f'selective_{k}': v for k, v in
                            SEL.selective_metrics(yl, pred, accepted).items()})
            rows.append(rec)

    frame = pd.DataFrame(rows)
    frame.to_csv(dest / 'external_metrics.csv', index=False)
    if sims:
        pd.concat(sims, ignore_index=True).to_csv(dest / 'external_similarity.csv', index=False)

    summary = {}
    for cname, grp in frame.groupby('cohort'):
        s = {'n': int(grp.n.iloc[0]), 'fold_models': int(len(grp))}
        for col in grp.columns:
            if col in ('cohort', 'fold_model', 'n'):
                continue
            v = pd.to_numeric(grp[col], errors='coerce').dropna()
            if len(v):
                mean, lo, hi = MT.bootstrap_ci(v.to_numpy(),
                                               reps=int(cfg.get('bootstrap_repetitions', 2000)),
                                               seed=7)
                s[col] = {'mean': mean, 'sd': float(v.std(ddof=1)) if len(v) > 1 else None,
                          'ci_low': lo, 'ci_high': hi}
        summary[cname] = s
    report['cohort_summary'] = summary
    report['status'] = 'evaluated'
    report['caveat'] = ('Cohort labels originate from different assay systems and, for ChEMBL, '
                        'from heterogeneous literature measurements curated by third parties. '
                        'These results measure behaviour under distribution shift, not '
                        'replication of the Tox21 VM7Luc4E2 endpoints.')
    (dest / 'external_report.json').write_text(json.dumps(report, indent=2, default=str))
    print('[external] ' + json.dumps({k: v.get('n') for k, v in summary.items()}), flush=True)
    return report

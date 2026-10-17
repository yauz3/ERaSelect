"""Apply a trained fold model to new molecules.

    python src/predict.py --config config/full.json --smiles input.csv --out scored.csv

`--smiles` takes a CSV with a `smiles` column (any other columns are carried through).
Predictions are the ensemble over the serialized scaffold-fold models for the pre-declared
primary configuration; each fold model carries its own frozen thresholds, so the reported
acceptance is the fraction of fold models that accept the molecule.

Molecules that fall outside the training chemical space are flagged. A prediction on a molecule
with low nearest-training similarity is an extrapolation, and the accompanying analysis shows
accuracy degrading with that distance; `nn_tanimoto_fit` is returned so this is visible rather
than implicit.
"""
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import ROOT, standardize, features
from experiments import nearest_similarity

LABELS = {0: 'inactive', 1: 'agonist', 2: 'antagonist'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/full.json')
    ap.add_argument('--smiles', required=True, help='CSV with a "smiles" column')
    ap.add_argument('--out', default='predictions_new.csv')
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / 'results' / cfg['profile']
    primary = cfg.get('primary', {})
    rep, model = primary.get('representation'), primary.get('model')
    folds = sorted((out / 'folds').glob(f'scaffold_*_{rep}_{model}'))
    if not folds:
        raise SystemExit(f'No serialized models at results/{cfg["profile"]}/folds/'
                         f'scaffold_*_{rep}_{model}. Run 01_run_experiments.sh first.')

    src = pd.read_csv(args.smiles)
    if 'smiles' not in src:
        raise SystemExit(f'{args.smiles} must contain a "smiles" column; found {list(src)[:8]}')
    rows, bad = [], []
    for i, r in src.iterrows():
        try:
            s, key, group = standardize(r['smiles'])
            rows.append({**r.to_dict(), 'input_smiles': r['smiles'], 'smiles': s,
                         'inchikey': key, 'group': group})
        except Exception as e:
            bad.append({'row': int(i), 'smiles': r['smiles'],
                        'reason': str(e).splitlines()[0][:100]})
    if not rows:
        raise SystemExit(f'No parsable molecules in {args.smiles}')
    d = pd.DataFrame(rows).reset_index(drop=True)
    d['row_id'] = np.arange(len(d))
    X, fp = features(d, cfg, out / 'inference', write=False)
    x = X[rep]

    master = pd.read_csv(out / 'data/master.csv', low_memory=False)
    _, fp_train = features(master, cfg, out / 'inference', write=False)
    d['nn_tanimoto_fit'] = nearest_similarity(fp, fp_train)

    P, acc, preds = [], [], []
    for fd in folds:
        obj = joblib.load(fd / 'models.joblib')
        th = obj['thresholds']
        pa = obj['gate_calibrator'].transform(obj['gate'].predict_proba(x))[:, 1]
        pm = obj['moa_calibrator'].transform(obj['moa'].predict_proba(x))[:, 1]
        p = np.column_stack([1 - pa, pa * pm, pa * (1 - pm)])
        pred = np.where(pa < float(th['gate_threshold']), 0, np.where(pm >= .5, 1, 2))
        conf = np.where(pred == 0, 1 - pa, np.maximum(pm, 1 - pm))
        z = obj['verifiers']['preprocess'].transform(x)
        support = np.ones(len(d), bool)
        for c in (1, 2):
            m = pred == c
            if m.any():
                knn = obj['verifiers'][f'knn_{c}']
                support[m] = (-knn.kneighbors(z[m], return_distance=True)[0].mean(1)
                              >= th['verifier_thresholds']['knn'][c - 1])
        P.append(p)
        preds.append(pred)
        acc.append(support & (conf >= float(th['gray_threshold'])))

    P = np.mean(P, axis=0)
    preds = np.stack(preds)
    acc = np.stack(acc)
    d['p_inactive'], d['p_agonist'], d['p_antagonist'] = P[:, 0], P[:, 1], P[:, 2]
    d['consensus_prediction'] = [LABELS[int(np.bincount(c, minlength=3).argmax())]
                                 for c in preds.T]
    d['fold_agreement'] = [float(np.bincount(c, minlength=3).max() / len(c)) for c in preds.T]
    d['accept_fraction'] = acc.mean(0)
    d['n_fold_models'] = len(folds)
    d['outside_training_space'] = d.nn_tanimoto_fit < 0.3
    d.drop(columns=['row_id']).to_csv(args.out, index=False)
    if bad:
        bad_path = Path(args.out).with_suffix('.rejected.csv')
        pd.DataFrame(bad).to_csv(bad_path, index=False)
        print(f'{len(bad)} molecule(s) could not be standardized -> {bad_path}')
    print(f'Scored {len(d)} molecules with {len(folds)} fold models -> {args.out}')
    print(f'  {int(d.outside_training_space.sum())} fall below Tanimoto 0.3 to training '
          f'(extrapolation; read accept_fraction and fold_agreement before trusting these)')


if __name__ == '__main__':
    main()

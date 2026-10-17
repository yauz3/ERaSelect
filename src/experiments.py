"""Nested scaffold benchmark with a pre-flight calibration-budget feasibility audit.

Partition roles, all scaffold-disjoint, all drawn from the outer TRAINING split only:

    fit          model fitting (and inner-CV hyperparameter selection within it)
    probability  probability recalibration
    policy       operating-threshold selection (gate recall target, gray-zone quantile)
    conformal    conformal nonconformity calibration
    verifier     one-class support acceptance calibration
    ------------------------------------------------------------------
    test         outer held-out scaffolds; never touched until everything above is frozen

Allocation is budget-driven: the sizes the conformal and verifier calibrators need in order to
be non-degenerate at the configured alpha are computed from the criterion in selective.py and
satisfied BEFORE any model is fitted. When they cannot be satisfied the fold is marked
calibration_budget_satisfied=False, per-class alphas are raised to the attainable floor, and the
nominal-alpha variant is still evaluated so the degeneracy is measurable rather than hidden.
"""
import argparse, hashlib, json, platform, sys, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import metrics as MT
import selective as SEL
from calibration import ProbabilityCalibrator
from data import ROOT, prepare, features

CLASSES = (0, 1, 2)
CALIBRATION_METHODS = ('platt', 'isotonic', 'temperature')

# sklearn warns whenever a fold's predictions include a class absent from that fold's truth.
# With a minority class this is expected and is already visible in the per-class counts.
warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class Infeasible(RuntimeError):
    """Raised when even the relaxed allocation cannot be built from the available scaffolds."""


# ------------------------------------------------------------------ estimators

def _grid(name, cfg, seed):
    t, j = cfg['trees'], cfg['jobs']
    g = {
        'lr': [dict(C=c, max_iter=3000, class_weight='balanced') for c in (0.03, 0.3, 3.0, 30.0)],
        'rf': [dict(n_estimators=t, max_depth=d, min_samples_leaf=l, class_weight='balanced',
                    n_jobs=j, random_state=seed)
               for d, l in ((None, 2), (12, 5), (None, 1), (24, 3))],
        'extra': [dict(n_estimators=t, min_samples_leaf=l, max_features=f,
                       class_weight='balanced', n_jobs=j, random_state=seed)
                  for l, f in ((2, 'sqrt'), (5, 'sqrt'), (1, 0.3), (3, 'log2'))],
        'xgb': [dict(n_estimators=t, max_depth=d, learning_rate=lr, subsample=.8,
                     colsample_bytree=.8, reg_lambda=1.0, n_jobs=j, random_state=seed,
                     eval_metric='logloss', tree_method='hist')
                for d, lr in ((3, .05), (6, .05), (4, .1), (8, .03))],
        'lgbm': [dict(n_estimators=t, num_leaves=nl, learning_rate=lr, min_child_samples=5,
                      class_weight='balanced', n_jobs=j, random_state=seed, verbosity=-1)
                 for nl, lr in ((15, .05), (31, .05), (63, .03), (15, .1))],
        'catboost': [dict(iterations=t, depth=d, learning_rate=lr, auto_class_weights='Balanced',
                          verbose=False, thread_count=j, random_seed=seed,
                          allow_writing_files=False)
                     for d, lr in ((4, .05), (6, .05), (6, .1), (8, .03))],
        'mlp': [dict(hidden_layer_sizes=h, alpha=a, max_iter=400, random_state=seed,
                     early_stopping=False)
                for h, a in (((128, 64), 1e-2), ((256, 128), 1e-2), ((64,), 1e-3), ((256, 64), 1e-1))],
    }
    if name not in g:
        raise ValueError(f'unknown model {name!r}')
    return g[name][:max(1, int(cfg.get('grid_size', 3)))]


def estimator(name, cfg, seed, choice=0):
    params = _grid(name, cfg, seed)[choice]
    if name == 'lr':
        model = LogisticRegression(**params)
    elif name == 'rf':
        model = RandomForestClassifier(**params)
    elif name == 'extra':
        model = ExtraTreesClassifier(**params)
    elif name == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
    elif name == 'lgbm':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
    elif name == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**params)
    else:
        model = MLPClassifier(**params)
    # Imputer/scaler are refitted inside every training partition; empty columns are retained so
    # feature dimensionality is identical across folds.
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()),
                     ('model', model)])


def split_cv(y, g, k, seed, kind='scaffold'):
    if kind == 'scaffold':
        n = min(k, len(np.unique(g)))
        if n < 2:
            raise Infeasible('fewer than two scaffold groups')
        return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y)), y, g))
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y))


def tune_fit(name, X, y, g, cfg, seed, task, records):
    """Inner scaffold CV over the configured grid. Never sees the outer test fold."""
    classes = np.unique(y)
    if len(classes) < 2:
        raise Infeasible(f'{task}: training partition has a single class')
    grid = _grid(name, cfg, seed)
    try:
        splits = split_cv(y, g, cfg['inner_folds'], seed)
    except Infeasible:
        splits = split_cv(y, g, cfg['inner_folds'], seed, 'random')
    scores = []
    for choice in range(len(grid)):
        fold_scores = []
        for a, b in splits:
            if len(np.unique(y[a])) < len(classes):
                fold_scores.append(np.nan)
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = estimator(name, cfg, seed, choice).fit(X[a], y[a])
                p = m.predict_proba(X[b])
            if len(classes) == 2:
                fold_scores.append(average_precision_score(y[b], p[:, 1])
                                   if len(np.unique(y[b])) == 2 else np.nan)
            else:
                fold_scores.append(-log_loss(y[b], p, labels=classes))
        scores.append(float(np.nanmean(fold_scores)) if np.isfinite(fold_scores).any() else np.nan)
    if not np.isfinite(scores).any():
        raise Infeasible(f'{task}: no valid inner-validation score')
    best = int(np.nanargmax(scores))
    records.append({'task': task, 'model': name, 'seed': seed, 'chosen': best,
                    'inner_scores': json.dumps([None if not np.isfinite(s) else round(s, 6)
                                                for s in scores]),
                    'params': json.dumps({k: str(v) for k, v in grid[best].items()})})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return estimator(name, cfg, seed, best).fit(X, y)


# ------------------------------------------------------------------ allocation

def role_budgets(cfg, strict=True):
    if strict:
        c = SEL.required_calibration_n(cfg['alpha'])
        v = SEL.required_verifier_n(cfg['verifier_acceptance'])
        return ({'conformal': {0: c, 1: c, 2: c},
                 'verifier': {1: v, 2: v},
                 'probability': {0: 5, 1: 5, 2: 5},
                 'policy': {0: 5, 1: 5, 2: 3}},
                {0: 20, 1: 10, 2: 10})
    return ({'conformal': {0: 2, 1: 2, 2: 2},
             'verifier': {1: 3, 2: 3},
             'probability': {0: 2, 1: 2, 2: 2},
             'policy': {0: 2, 1: 2, 2: 2}},
            {0: 4, 1: 4, 2: 4})


ROLE_PRIORITY = ('conformal', 'verifier', 'probability', 'policy')


def budget_shortfall(train, d, cfg, strict=True):
    """Classes whose outer-train count cannot cover the summed role budgets, with the gap.

    A cheap necessary condition checked before any split is attempted: when the minority class
    holds fewer molecules than every role needs put together, no allocation can exist and
    searching for one is wasted work.
    """
    y = d.label.to_numpy()
    budgets, fit_min = role_budgets(cfg, strict)
    need = {c: sum(budgets[r].get(c, 0) for r in ROLE_PRIORITY) + fit_min.get(c, 0)
            for c in CLASSES}
    have = {c: int((y[train] == c).sum()) for c in CLASSES}
    return {c: need[c] - have[c] for c in CLASSES if have[c] < need[c]}, need, have


def allocate(train, d, seed, kind, cfg, strict=True, n_parts=10, attempts=12):
    """Greedy budget-driven assignment of scaffold-disjoint parts to calibration roles.

    Uses OUTER-TRAIN LABELS ONLY and runs before any model is fitted, so the feasibility
    verdict cannot be influenced by held-out performance.
    """
    y, g = d.label.to_numpy(), d.group.to_numpy()
    budgets, fit_min = role_budgets(cfg, strict)
    short, _, _ = budget_shortfall(train, d, cfg, strict)
    if short:
        # Arithmetically impossible at these budgets; do not search.
        if strict:
            return allocate(train, d, seed, kind, cfg, strict=False, n_parts=n_parts,
                            attempts=attempts)
        raise Infeasible('outer-train class counts cannot cover even the relaxed role budgets '
                         f'(shortfall by class: {short})')
    for attempt in range(attempts):
        try:
            splits = split_cv(y[train], g[train], n_parts, seed + attempt * 101, kind)
        except Infeasible:
            break
        parts = [train[b] for _, b in splits]
        if len(parts) < 5:
            continue
        avail = list(range(len(parts)))
        counts = [np.bincount(y[p], minlength=4) for p in parts]
        assign, ok = {}, True
        for role in ROLE_PRIORITY:
            need = dict(budgets[role])
            chosen = []
            while any(v > 0 for v in need.values()):
                if len(avail) <= 2:
                    ok = False
                    break
                best = None
                for i in avail:
                    gain = sum(min(need[k], int(counts[i][k])) for k in need)
                    if gain <= 0:
                        continue
                    key = (-gain, int(counts[i].sum()), i)
                    if best is None or key < best[0]:
                        best = (key, i)
                if best is None:
                    ok = False
                    break
                i = best[1]
                avail.remove(i)
                chosen.append(i)
                for k in need:
                    need[k] = max(0, need[k] - int(counts[i][k]))
            if not ok or any(v > 0 for v in need.values()) or not chosen:
                ok = False
                break
            assign[role] = np.concatenate([parts[i] for i in chosen])
        if not ok or not avail:
            continue
        fit = np.concatenate([parts[i] for i in avail])
        fy, fg = y[fit], g[fit]
        if any(int((fy == c).sum()) < fit_min[c] or pd.Series(fg[fy == c]).nunique() < 2
               for c in CLASSES):
            continue
        assign['fit'] = fit
        return assign, ('strict' if strict else 'relaxed')
    if strict:
        return allocate(train, d, seed, kind, cfg, strict=False, n_parts=n_parts, attempts=attempts)
    raise Infeasible('no viable scaffold allocation even under relaxed budgets')


def role_counts(assign, y):
    return {role: {int(c): int((y[idx] == c).sum()) for c in range(4)}
            for role, idx in assign.items()}


# ------------------------------------------------------------------ similarity

def nearest_similarity(test_fp, train_fp):
    """Max Tanimoto of each test molecule to the training set (binary fingerprints)."""
    a = np.asarray(test_fp, dtype=np.float32)
    b = np.asarray(train_fp, dtype=np.float32)
    if len(b) == 0 or len(a) == 0:
        return np.full(len(a), np.nan)
    bs = b.sum(1)
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), 256):
        x = a[i:i + 256]
        inter = x @ b.T
        union = x.sum(1)[:, None] + bs[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        out[i:i + 256] = sim.max(1)
    return out


# ------------------------------------------------------------------ verifier

def class_support(Xfit, yfit, Xver, yver, Xtest, cfg, seed):
    k = min(32, Xfit.shape[1], max(2, len(Xfit) - 1))
    prep = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()),
                     ('pca', PCA(n_components=k, random_state=seed))])
    z = prep.fit_transform(Xfit)
    v, t = prep.transform(Xver), prep.transform(Xtest)
    out, objects = {}, {'preprocess': prep}
    for kind in ('knn', 'ocsvm', 'iforest'):
        scores = np.zeros((len(Xtest), 2))
        accepted = np.zeros((len(Xtest), 2), dtype=bool)
        thresholds, budget = [], []
        for c in (1, 2):
            tr, cal = z[yfit == c], v[yver == c]
            if len(tr) < 3 or len(cal) < 1:
                raise Infeasible(f'verifier class {c}: {len(tr)} fit / {len(cal)} calibration')
            if kind == 'knn':
                mdl = NearestNeighbors(n_neighbors=min(5, len(tr))).fit(tr)
                def sc(a, _m=mdl): return -_m.kneighbors(a, return_distance=True)[0].mean(1)
            elif kind == 'ocsvm':
                mdl = OneClassSVM(nu=.05, gamma='scale').fit(tr)
                def sc(a, _m=mdl): return _m.decision_function(a).ravel()
            else:
                mdl = IsolationForest(n_estimators=200, contamination='auto',
                                      n_jobs=cfg['jobs'], random_state=seed).fit(tr)
                def sc(a, _m=mdl): return _m.score_samples(a)
            # Prespecified empirical in-class acceptance quantile. Non-degenerate only when
            # len(cal) >= required_verifier_n(acceptance); recorded either way.
            thr = float(np.quantile(sc(cal), 1 - cfg['verifier_acceptance'], method='lower'))
            scores[:, c - 1] = sc(t)
            accepted[:, c - 1] = scores[:, c - 1] >= thr
            thresholds.append(thr)
            budget.append({'klass': c, 'n_fit': int(len(tr)), 'n_calibration': int(len(cal)),
                           'required_n': SEL.required_verifier_n(cfg['verifier_acceptance']),
                           'feasible': bool(len(cal) >= SEL.required_verifier_n(
                               cfg['verifier_acceptance']))})
            objects[f'{kind}_{c}'] = mdl
        out[kind] = (accepted, scores, thresholds, budget)
    return out, objects


def activity_threshold(p, y, target):
    """Lowest cutoff retaining `target` recall of actives, chosen on the POLICY partition only."""
    pos = np.sort(np.asarray(p, dtype=float)[np.asarray(y) == 1])
    if len(pos) == 0:
        raise Infeasible('activity threshold: no actives in policy partition')
    idx = int(np.clip(np.floor((1 - target) * len(pos)), 0, len(pos) - 1))
    return float(pos[idx])


# ------------------------------------------------------------------ one fold

def _emit(meta, variant, y, p, pred, accepted, score, extra=None):
    row = dict(meta)
    row['variant'] = variant
    row.update(SEL.selective_metrics(y, pred, accepted))
    row.update(MT.multiclass_metrics(y, p, pred, 3, 'mc'))
    row.update(MT.moa_metrics(y, pred, p[:, 1] / np.clip(p[:, 1] + p[:, 2], 1e-9, None), 'moa'))
    row.update(MT.dual_active_behaviour(y, accepted))
    row['aurc'] = SEL.aurc(y[y < 3], pred[y < 3], score[y < 3]) if (y < 3).any() else np.nan
    if extra:
        row.update(extra)
    return row


def run_fold(d, X, fp, train, test, rep, name, cfg, seed, fold, kind, out):
    y, g = d.label.to_numpy(), d.group.to_numpy()
    assign, mode = allocate(train, d, seed + fold, kind, cfg)
    counts = role_counts(assign, y)
    budget_rows = SEL.budget_report(counts, cfg['alpha'], cfg['verifier_acceptance'])
    eff_alpha, relaxed = SEL.effective_alphas(counts, cfg['alpha'])
    satisfied = all(r['feasible'] for r in budget_rows)

    stem = f'{kind}_s{seed}_f{fold}_{rep}_{name}'
    sd = out / 'folds' / stem
    sd.mkdir(parents=True, exist_ok=True)

    # Scaffold disjointness is asserted, not assumed.
    if kind == 'scaffold':
        allidx = dict(assign, test=test)
        for r1, a in allidx.items():
            for r2, b in allidx.items():
                if r1 < r2 and set(g[a]) & set(g[b]):
                    raise AssertionError(f'scaffold leakage between {r1} and {r2} in {stem}')
    pd.DataFrame([{'row_id': int(i), 'inchikey': d.inchikey.iloc[i], 'group': g[i], 'role': role}
                  for role, idx in dict(assign, test=test).items() for i in idx]
                 ).to_csv(sd / 'split_membership.csv', index=False)

    fi, pi, po, co, ve = (assign['fit'], assign['probability'], assign['policy'],
                          assign['conformal'], assign['verifier'])
    clean = lambda ix: ix[y[ix] != 3]
    excl = lambda ix: ix[np.isin(y[ix], [1, 2])]
    records, rows = [], []

    sim_train = nearest_similarity(fp[test], fp[fi])
    sim_active = nearest_similarity(fp[test], fp[fi[y[fi] > 0]])
    meta = {'split': kind, 'seed': seed, 'fold': fold, 'representation': rep, 'model': name,
            'tier': cfg.get('label_tier'), 'allocation_mode': mode,
            'calibration_budget_satisfied': bool(satisfied),
            'alpha_nominal': float(cfg['alpha']),
            'alpha_effective_class2': float(eff_alpha[2]),
            'alpha_relaxed_any': bool(any(relaxed.values())),
            'n_fit': int(len(fi)), 'n_conformal_class2': counts['conformal'][2],
            'n_verifier_class2': counts['verifier'][2],
            'median_nn_tanimoto': float(np.nanmedian(sim_train))}
    yt = y[test]

    # ---------------- flat 3-class baseline
    tri = clean(fi)
    flat = tune_fit(name, X[tri], y[tri], g[tri], cfg, seed, 'flat3', records)
    raw_flat = flat.predict_proba(X[test])
    rows.append(_emit(meta, 'flat_raw', yt, raw_flat, raw_flat.argmax(1),
                      np.ones(len(test), bool), raw_flat.max(1)))
    flat_cals = {}
    for method in CALIBRATION_METHODS:
        cal = ProbabilityCalibrator(method).fit(flat.predict_proba(X[clean(pi)]), y[clean(pi)])
        flat_cals[method] = cal
        p = cal.transform(raw_flat)
        rows.append(_emit(meta, f'flat_{method}', yt, p, p.argmax(1),
                          np.ones(len(test), bool), p.max(1)))
    cal_flat = flat_cals['platt']
    p_flat = cal_flat.transform(raw_flat)
    pc_flat = cal_flat.transform(flat.predict_proba(X[clean(co)]))
    for tag, a, mo in (('conformal_pooled', cfg['alpha'], 'pooled'),
                       ('conformal_nominal', cfg['alpha'], 'mondrian'),
                       ('conformal_feasible', eff_alpha, 'mondrian')):
        q = SEL.conformal_thresholds(pc_flat, y[clean(co)], a, mo)
        sets = SEL.prediction_sets(p_flat, q)
        acc = SEL.singleton_accept(sets, p_flat.argmax(1))
        rows.append(_emit(meta, f'flat_{tag}', yt, p_flat, p_flat.argmax(1), acc, p_flat.max(1),
                          {'degenerate_threshold_fraction': SEL.degenerate_fraction(q),
                           'mean_set_size': float(sets.sum(1).mean())}))

    # ---------------- supervised four-class control (uses dual labels)
    if int((y[fi] == 3).sum()) >= 4 and int((y[pi] == 3).sum()) >= 1:
        joint = tune_fit(name, X[fi], y[fi], g[fi], cfg, seed, 'joint4', records)
        pj = joint.predict_proba(X[test])
        k = pj.shape[1]
        p3 = pj[:, :3] / np.clip(pj[:, :3].sum(1, keepdims=True), 1e-9, None)
        rows.append(_emit(meta, 'joint4_raw', yt, p3, p3.argmax(1), np.ones(len(test), bool),
                          p3.max(1), {'p_dual_mean': float(pj[:, 3].mean()) if k > 3 else np.nan}))
    else:
        joint = None

    # ---------------- independent endpoint-pair baseline
    end_models, end_cals, end_p = [], [], []
    for col in ('agonist', 'antagonist'):
        ey = d[col].fillna(0).astype(int).to_numpy()
        mdl = tune_fit(name, X[fi], ey[fi], g[fi], cfg, seed, f'{col}_endpoint', records)
        cc = ProbabilityCalibrator('platt').fit(mdl.predict_proba(X[pi]), ey[pi])
        end_models.append(mdl)
        end_cals.append(cc)
        end_p.append(cc.transform(mdl.predict_proba(X[test]))[:, 1])
    pa_e, pb_e = end_p
    pe = np.column_stack([(1 - pa_e) * (1 - pb_e), pa_e * (1 - pb_e),
                          (1 - pa_e) * pb_e, pa_e * pb_e])
    pe3 = pe[:, :3] / np.clip(pe[:, :3].sum(1, keepdims=True), 1e-9, None)
    rows.append(_emit(meta, 'endpoint_pair', yt, pe3, pe3.argmax(1), np.ones(len(test), bool),
                      pe3.max(1)))

    # ---------------- proposed hierarchy
    ya = (y > 0).astype(int)
    gate = tune_fit(name, X[fi], ya[fi], g[fi], cfg, seed, 'gate', records)
    mo_fit = excl(fi)
    moa = tune_fit(name, X[mo_fit], (y[mo_fit] == 1).astype(int), g[mo_fit], cfg, seed, 'moa', records)
    cg = ProbabilityCalibrator('platt').fit(gate.predict_proba(X[pi]), ya[pi])
    cm = ProbabilityCalibrator('platt').fit(moa.predict_proba(X[excl(pi)]),
                                            (y[excl(pi)] == 1).astype(int))

    def hier(ix, cal=True):
        a = gate.predict_proba(X[ix])
        b = moa.predict_proba(X[ix])
        if cal:
            a, b = cg.transform(a), cm.transform(b)
        pa, pm = a[:, 1], b[:, 1]
        return np.column_stack([1 - pa, pa * pm, pa * (1 - pm)]), pa, pm

    p_h, pa, pm = hier(test)
    raw_h, ra, rm = hier(test, False)
    p_po, pa_po, pm_po = hier(po)
    tau = activity_threshold(pa_po, ya[po], cfg['target_active_recall'])
    pred_h = np.where(pa < tau, 0, np.where(pm >= .5, 1, 2))
    conf_h = np.where(pred_h == 0, 1 - pa, np.maximum(pm, 1 - pm))
    gate_m = MT.gate_metrics(ya[test], pa) | MT.gate_operating_point(ya[test], pa, tau)

    rows.append(_emit(meta, 'hierarchy_raw', yt, raw_h,
                      np.where(ra < .5, 0, np.where(rm >= .5, 1, 2)),
                      np.ones(len(test), bool), raw_h.max(1), MT.gate_metrics(ya[test], ra)))
    rows.append(_emit(meta, 'hierarchy_calibrated', yt, p_h,
                      np.where(pa < .5, 0, np.where(pm >= .5, 1, 2)),
                      np.ones(len(test), bool), p_h.max(1), gate_m))
    rows.append(_emit(meta, 'hierarchy_gate_recall', yt, p_h, pred_h,
                      np.ones(len(test), bool), conf_h, gate_m))

    policy_conf = np.where(y[po] == 0, 1 - pa_po, np.maximum(pm_po, 1 - pm_po))
    gray = float(np.quantile(policy_conf, cfg.get('gray_quantile', .15)))
    gray_acc = conf_h >= gray
    rows.append(_emit(meta, 'hierarchy_gray', yt, p_h, pred_h, gray_acc, conf_h,
                      gate_m | {'gray_threshold': gray}))

    p_co, _, _ = hier(clean(co))
    conformal_acc = {}
    for tag, a, mo in (('conformal_pooled', cfg['alpha'], 'pooled'),
                       ('conformal_nominal', cfg['alpha'], 'mondrian'),
                       ('conformal_feasible', eff_alpha, 'mondrian')):
        q = SEL.conformal_thresholds(p_co, y[clean(co)], a, mo)
        sets = SEL.prediction_sets(p_h, q)
        acc = SEL.singleton_accept(sets, pred_h)
        conformal_acc[tag] = acc
        rows.append(_emit(meta, f'hierarchy_{tag}', yt, p_h, pred_h, acc, conf_h,
                          gate_m | {'degenerate_threshold_fraction': SEL.degenerate_fraction(q),
                                    'mean_set_size': float(sets.sum(1).mean()),
                                    'conformal_threshold_class2': float(q[2])}))

    supports, verifiers = class_support(X[mo_fit], y[mo_fit], X[excl(ve)], y[excl(ve)],
                                        X[test], cfg, seed)
    for vk, (acc_mat, sc, thr, budget) in supports.items():
        sel = np.ones(len(test), bool)
        act = pred_h > 0
        sel[act] = acc_mat[np.arange(len(test))[act], pred_h[act] - 1]
        vfeas = all(b['feasible'] for b in budget)
        rows.append(_emit(meta, f'hierarchy_{vk}', yt, p_h, pred_h, sel, conf_h,
                          gate_m | {'verifier_budget_satisfied': vfeas}))
        full = sel & gray_acc & conformal_acc['conformal_feasible']
        rows.append(_emit(meta, f'full_{vk}', yt, p_h, pred_h, full, conf_h,
                          gate_m | {'verifier_budget_satisfied': vfeas}))
        full_nom = sel & gray_acc & conformal_acc['conformal_nominal']
        rows.append(_emit(meta, f'full_{vk}_nominal_alpha', yt, p_h, pred_h, full_nom, conf_h,
                          gate_m | {'verifier_budget_satisfied': vfeas}))

    # ---------------- label-permutation negative control
    if cfg.get('permutation_control', True):
        rng = np.random.default_rng(seed * 1000 + fold)
        yp = y.copy()
        yp[fi] = rng.permutation(y[fi])
        yap = (yp > 0).astype(int)
        try:
            pg = tune_fit(name, X[fi], yap[fi], g[fi], cfg, seed, 'gate_permuted', records)
            pmo = excl_p = fi[np.isin(yp[fi], [1, 2])]
            pmoa = tune_fit(name, X[excl_p], (yp[excl_p] == 1).astype(int), g[excl_p], cfg,
                            seed, 'moa_permuted', records)
            paa = pg.predict_proba(X[test])[:, 1]
            pmm = pmoa.predict_proba(X[test])[:, 1]
            pp = np.column_stack([1 - paa, paa * pmm, paa * (1 - pmm)])
            predp = np.where(paa < .5, 0, np.where(pmm >= .5, 1, 2))
            rows.append(_emit(meta, 'permuted_labels_control', yt, pp, predp,
                              np.ones(len(test), bool), pp.max(1),
                              MT.gate_metrics(ya[test], paa)))
        except Infeasible as e:
            records.append({'task': 'permutation_control', 'model': name, 'seed': seed,
                            'chosen': -1, 'inner_scores': 'null', 'params': f'skipped: {e}'})

    # ---------------- persist
    pd.DataFrame(records).to_csv(sd / 'inner_selection.csv', index=False)
    pd.DataFrame(budget_rows).assign(**{k: meta[k] for k in
                                        ('split', 'seed', 'fold', 'representation', 'model')}
                                     ).to_csv(sd / 'calibration_budget.csv', index=False)
    thresholds = {'gate_threshold': tau, 'gate_recall_target': cfg['target_active_recall'],
                  'gray_threshold': gray, 'alpha_nominal': cfg['alpha'],
                  'alpha_effective': {str(k): v for k, v in eff_alpha.items()},
                  'alpha_relaxed': {str(k): v for k, v in relaxed.items()},
                  'conformal_mode': cfg.get('conformal_mode', 'mondrian'),
                  'verifier_thresholds': {k: v[2] for k, v in supports.items()},
                  'verifier_budget': {k: v[3] for k, v in supports.items()},
                  'role_counts': counts, 'allocation_mode': mode,
                  'calibration_budget_satisfied': bool(satisfied)}
    (sd / 'thresholds.json').write_text(json.dumps(thresholds, indent=2, default=str))

    frame = pd.DataFrame(rows)
    frame.to_csv(sd / 'metrics.csv', index=False)

    primary = cfg.get('primary', {})
    is_primary = (name == primary.get('model') and rep == primary.get('representation'))
    if is_primary:
        # Molecule-level predictions are retained for the primary configuration only; keeping
        # them for all 500+ cells would produce tens of millions of rows.
        pd.DataFrame({'row_id': d.row_id.to_numpy()[test], 'inchikey': d.inchikey.iloc[test],
                      'smiles': d.smiles.iloc[test], 'group': g[test], 'label': yt,
                      'label_name': d.label_name.iloc[test],
                      'moa_confidence': d.moa_confidence.iloc[test],
                      'split': kind, 'seed': seed, 'fold': fold,
                      'p_inactive': p_h[:, 0], 'p_agonist': p_h[:, 1], 'p_antagonist': p_h[:, 2],
                      'p_active': pa, 'p_agonist_given_active': pm,
                      'prediction': pred_h, 'confidence': conf_h,
                      'accept_gray': gray_acc,
                      'accept_conformal_feasible': conformal_acc['conformal_feasible'],
                      'accept_conformal_nominal': conformal_acc['conformal_nominal'],
                      'accept_full_knn': (supports['knn'][0][np.arange(len(test)),
                                                             np.maximum(pred_h - 1, 0)]
                                          | (pred_h == 0)) & gray_acc
                                         & conformal_acc['conformal_feasible'],
                      'support_agonist': supports['knn'][1][:, 0],
                      'support_antagonist': supports['knn'][1][:, 1],
                      'nn_tanimoto_train': sim_train,
                      'nn_tanimoto_active': sim_active,
                      'flat_prediction': p_flat.argmax(1)}).to_csv(sd / 'predictions.csv',
                                                                   index=False)
        joblib.dump({'flat': flat, 'flat_calibrators': flat_cals, 'joint4': joint,
                     'gate': gate, 'moa': moa, 'gate_calibrator': cg, 'moa_calibrator': cm,
                     'endpoint_models': end_models, 'endpoint_calibrators': end_cals,
                     'verifiers': verifiers, 'thresholds': thresholds, 'config': cfg,
                     'representation': rep}, sd / 'models.joblib', compress=3)
    return frame, budget_rows


# ------------------------------------------------------------------ driver

def preflight(d, cfg, out):
    """Audit the calibration budget for every (seed, fold) BEFORE fitting anything."""
    y, g = d.label.to_numpy(), d.group.to_numpy()
    rows = []
    for kind in ['scaffold'] + (['random'] if cfg.get('random_split') else []):
        seeds = cfg['seeds'] if kind == 'scaffold' else cfg['seeds'][:1]
        for seed in seeds:
            for fold, (tr, te) in enumerate(split_cv(y, g, cfg['outer_folds'], seed, kind)):
                try:
                    assign, mode = allocate(tr, d, seed + fold, kind, cfg)
                except Infeasible as e:
                    rows.append({'split': kind, 'seed': seed, 'fold': fold, 'status': str(e),
                                 'allocation_mode': 'none', 'feasible': False})
                    continue
                counts = role_counts(assign, y)
                for r in SEL.budget_report(counts, cfg['alpha'], cfg['verifier_acceptance']):
                    rows.append({'split': kind, 'seed': seed, 'fold': fold, 'status': 'allocated',
                                 'allocation_mode': mode,
                                 'n_test': int(len(te)),
                                 'n_test_class2': int((y[te] == 2).sum()), **r})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'calibration_budget_feasibility.csv', index=False)
    sat = frame[frame.status == 'allocated']
    summary = {'rows': int(len(frame)),
               'folds_with_allocation': int(sat.groupby(['split', 'seed', 'fold']).ngroups)
               if len(sat) else 0,
               'all_components_feasible': bool(sat.feasible.all()) if len(sat) else False,
               'infeasible_components': (sat[~sat.feasible].groupby(['component', 'klass'])
                                         .agg(n_folds_infeasible=('feasible', 'size'),
                                              min_calibration_n=('n', 'min'),
                                              required_n=('required_n', 'first'))
                                         .reset_index().to_dict('records')
                                         if len(sat) else []),
               'relaxed_allocations': int((sat.allocation_mode == 'relaxed')
                                          .groupby([sat.split, sat.seed, sat.fold]).any().sum())
               if len(sat) else 0}
    (out / 'calibration_budget_summary.json').write_text(json.dumps(summary, indent=2))
    print('[preflight] ' + json.dumps(summary)[:600], flush=True)
    return frame, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/full.json')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / 'results' / cfg['profile']
    (out / 'data').mkdir(parents=True, exist_ok=True)

    code = ''.join(p.read_text() for p in sorted((ROOT / 'src').glob('*.py'))
                   if p.name != 'analysis.py')
    digest = hashlib.sha256((json.dumps(cfg, sort_keys=True) + code).encode()).hexdigest()
    idf = out / 'run_identity.json'
    if idf.exists():
        if json.loads(idf.read_text())['digest'] != digest:
            raise SystemExit(f'Config or source changed since this results directory was started. '
                             f'Rename results/{cfg["profile"]} before rerunning; refusing to mix runs.')
    else:
        idf.write_text(json.dumps({'digest': digest, 'config': cfg, 'python': sys.version,
                                   'platform': platform.platform(),
                                   'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                                time.gmtime())}, indent=2))

    d = prepare(cfg, out / 'data')
    src = json.loads((out / 'data/data_manifest.json').read_text())
    sdig = hashlib.sha256(json.dumps(src, sort_keys=True).encode()).hexdigest()
    sidf = out / 'source_identity.txt'
    if sidf.exists() and sidf.read_text() != sdig:
        raise SystemExit('Raw source data changed; refusing to resume onto mixed-data results.')
    sidf.write_text(sdig)

    preflight(d, cfg, out)
    X, fp = features(d, cfg, out / 'data')

    # A representation that could not be built (typically the optional pretrained encoder)
    # drops out of the run with a recorded reason rather than aborting work that is otherwise
    # complete. The primary configuration is the one exception: losing it invalidates the run.
    requested = list(cfg['representations'])
    available = [r for r in requested if r in X]
    missing = [r for r in requested if r not in X]
    if missing:
        primary_rep = cfg.get('primary', {}).get('representation')
        if primary_rep in missing:
            raise SystemExit(
                f'The pre-declared primary representation {primary_rep!r} could not be built, '
                f'so this run cannot support its primary claim. Fix the cause or change '
                f'"primary" in the config.')
        print(f'[experiments] representations unavailable and skipped: {missing}. '
              f'Continuing with {available}.', flush=True)
        (out / 'skipped_representations.json').write_text(json.dumps(
            {'requested': requested, 'available': available, 'skipped': missing,
             'see': 'data/embedding_manifest.json for the reason'}, indent=2))
    if not available:
        raise SystemExit('No requested representation could be built.')

    frames, budgets, failures = [], [], []
    total = (len(cfg['models']) * len(available) * cfg['outer_folds']
             * (len(cfg['seeds']) + (1 if cfg.get('random_split') else 0)))
    done = 0
    t0 = time.time()
    for kind in ['scaffold'] + (['random'] if cfg.get('random_split') else []):
        for seed in (cfg['seeds'] if kind == 'scaffold' else cfg['seeds'][:1]):
            splits = split_cv(d.label.to_numpy(), d.group.to_numpy(), cfg['outer_folds'],
                              seed, kind)
            for fold, (tr, te) in enumerate(splits):
                for rep in available:
                    for name in cfg['models']:
                        stem = f'{kind}_s{seed}_f{fold}_{rep}_{name}'
                        mp = out / 'folds' / stem / 'metrics.csv'
                        done += 1
                        if mp.exists():
                            frames.append(pd.read_csv(mp))
                            bp = mp.parent / 'calibration_budget.csv'
                            if bp.exists():
                                budgets.append(pd.read_csv(bp))
                            continue
                        el = time.time() - t0
                        eta = (el / max(1, done - 1)) * (total - done) / 60 if done > 1 else 0
                        print(f'[{done}/{total}] {stem}  eta~{eta:.0f}m', flush=True)
                        try:
                            f, b = run_fold(d, X[rep], fp, tr, te, rep, name, cfg, seed,
                                            fold, kind, out)
                            frames.append(f)
                            budgets.append(pd.DataFrame(b))
                        except Infeasible as e:
                            failures.append({'run': stem, 'error': f'Infeasible: {e}'})
                            (out / 'failures.json').write_text(json.dumps(failures, indent=2))
                            print(f'  !! infeasible, recorded and skipped: {e}', flush=True)
                        except Exception as e:
                            failures.append({'run': stem, 'error': repr(e)})
                            (out / 'failures.json').write_text(json.dumps(failures, indent=2))
                            raise

    if not frames:
        raise SystemExit('No fold produced metrics; inspect failures.json')
    allm = pd.concat(frames, ignore_index=True)
    allm.to_csv(out / 'fold_metrics.csv', index=False)
    if budgets:
        pd.concat(budgets, ignore_index=True).to_csv(out / 'calibration_budget_all.csv',
                                                     index=False)

    from external import evaluate_external
    evaluate_external(cfg, out, d)

    (out / 'COMPLETE.json').write_text(json.dumps({
        'profile': cfg['profile'], 'digest': digest, 'cells_attempted': total,
        'cells_with_metrics': int(allm.groupby(['split', 'seed', 'fold', 'representation',
                                                'model']).ngroups),
        'variants_per_cell': int(allm.variant.nunique()), 'failures': len(failures),
        'minutes': round((time.time() - t0) / 60, 1),
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'experiments_complete'}, indent=2))
    print(f'Experiments complete in {(time.time()-t0)/60:.1f} min. '
          f'Next: bash 02_finish_analysis.sh', flush=True)


if __name__ == '__main__':
    main()

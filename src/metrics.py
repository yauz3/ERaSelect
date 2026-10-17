"""Screening and classification metrics. Every function tolerates degenerate folds (returns NaN)."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, matthews_corrcoef,
                             balanced_accuracy_score, f1_score, confusion_matrix)

from calibration import expected_calibration_error, multiclass_brier

NAN = float('nan')


def _binary_ok(y):
    y = np.asarray(y)
    return len(np.unique(y)) == 2


def enrichment_factor(y, score, fraction):
    """EF@fraction: actives found in the top fraction, relative to the overall active rate."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    rate = y.mean() if n else 0.0
    k = int(np.ceil(fraction * n))
    if n == 0 or k == 0 or rate == 0:
        return NAN
    top = y[np.argsort(-score)[:k]]
    return float(top.mean() / rate)


def bedroc(y, score, alpha=20.0):
    """BEDROC (Truchon & Bayly, J Chem Inf Model 2007), early-recognition weighted.

        RIE     = mean_i exp(-alpha * r_i / n) / [ (1/n) (1 - e^-alpha) / (e^(alpha/n) - 1) ]
        BEDROC  = (RIE - RIE_min) / (RIE_max - RIE_min)

    A perfect ranking gives 1 and the worst possible ranking gives 0. The value expected under a
    RANDOM ranking is NOT 0.5: it equals 1/RIE_max, which depends on alpha and on the active
    ratio Ra. For alpha=20 and Ra=0.05 it is approximately 0.079. BEDROC values are therefore
    comparable only at matched prevalence, and `bedroc_random_baseline` must be reported beside
    them. Use it; do not describe a BEDROC of 0.3 as "better than random" without it.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    m = int(y.sum())
    if n == 0 or m == 0 or m == n:
        return NAN
    # Average competing ranks so that tied scores cannot be ordered favourably.
    order = np.argsort(-score, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    srt = score[order]
    start = 0
    for i in range(1, n + 1):
        if i == n or srt[i] != srt[start]:
            if i - start > 1:
                ranks[order[start:i]] = np.mean(np.arange(start + 1, i + 1))
            start = i
    ra = m / n
    expected = (1.0 / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    if not np.isfinite(expected) or expected == 0:
        return NAN
    rie = float(np.mean(np.exp(-alpha * ranks[y == 1] / n)) / expected)
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    if not np.isfinite(rie_max - rie_min) or rie_max == rie_min:
        return NAN
    return float((rie - rie_min) / (rie_max - rie_min))


def bedroc_random_baseline(y, alpha=20.0):
    """BEDROC expected from a random ranking at this prevalence (1 / RIE_max)."""
    y = np.asarray(y).astype(int)
    n, m = len(y), int(y.sum())
    if n == 0 or m == 0 or m == n:
        return NAN
    ra = m / n
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    if not np.isfinite(rie_max - rie_min) or rie_max == rie_min:
        return NAN
    return float((1.0 - rie_min) / (rie_max - rie_min))


def gate_metrics(y_active, p_active, prefix='activity'):
    """Binary screening metrics for the Stage 1 activity gate."""
    y = np.asarray(y_active).astype(int)
    p = np.asarray(p_active, dtype=float)
    out = {}
    if _binary_ok(y):
        out[f'{prefix}_pr_auc'] = float(average_precision_score(y, p))
        out[f'{prefix}_roc_auc'] = float(roc_auc_score(y, p))
        out[f'{prefix}_bedroc20'] = bedroc(y, p, 20.0)
        out[f'{prefix}_bedroc20_random'] = bedroc_random_baseline(y, 20.0)
        for f in (0.01, 0.05, 0.10):
            out[f'{prefix}_ef{int(f * 100)}'] = enrichment_factor(y, p, f)
        out[f'{prefix}_brier'] = float(np.mean((p - y) ** 2))
        two = np.column_stack([1 - p, p])
        out[f'{prefix}_ece'] = expected_calibration_error(two, y, bins=10)
        out[f'{prefix}_prevalence'] = float(y.mean())
    else:
        for k in (f'{prefix}_pr_auc', f'{prefix}_roc_auc', f'{prefix}_bedroc20',
                  f'{prefix}_bedroc20_random', f'{prefix}_ef1', f'{prefix}_ef5',
                  f'{prefix}_ef10', f'{prefix}_brier', f'{prefix}_ece', f'{prefix}_prevalence'):
            out[k] = NAN
    return out


def gate_operating_point(y_active, p_active, tau, prefix='gate'):
    y = np.asarray(y_active).astype(int)
    flag = np.asarray(p_active, dtype=float) >= tau
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else NAN
    spec = tn / (tn + fp) if tn + fp else NAN
    prec = tp / (tp + fp) if tp + fp else NAN
    return {f'{prefix}_threshold': float(tau), f'{prefix}_recall': rec,
            f'{prefix}_specificity': spec, f'{prefix}_precision': prec,
            f'{prefix}_tp': tp, f'{prefix}_fp': fp, f'{prefix}_fn': fn, f'{prefix}_tn': tn}


def moa_metrics(y_true, y_pred, p_agonist=None, prefix='moa'):
    """Mode-of-action metrics restricted to exclusive actives (labels 1 and 2)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = np.isin(y_true, [1, 2])
    out = {f'{prefix}_n': int(m.sum())}
    if m.sum() == 0:
        return {**out, f'{prefix}_balanced_accuracy': NAN, f'{prefix}_mcc': NAN,
                f'{prefix}_f1_antagonist': NAN, f'{prefix}_roc_auc': NAN}
    yt, yp = y_true[m], y_pred[m]
    out[f'{prefix}_balanced_accuracy'] = (float(balanced_accuracy_score(yt, yp))
                                         if len(np.unique(yt)) > 1 else NAN)
    out[f'{prefix}_mcc'] = (float(matthews_corrcoef(yt, yp))
                            if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1 else NAN)
    out[f'{prefix}_f1_antagonist'] = float(f1_score(yt == 2, yp == 2, zero_division=0))
    if p_agonist is not None and len(np.unique(yt)) > 1:
        out[f'{prefix}_roc_auc'] = float(roc_auc_score(yt == 1, np.asarray(p_agonist)[m]))
    else:
        out[f'{prefix}_roc_auc'] = NAN
    return out


def multiclass_metrics(y_true, p, y_pred, n_classes=3, prefix='mc'):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p = np.asarray(p, dtype=float)
    keep = y_true < n_classes
    out = {}
    if keep.sum() == 0:
        return {f'{prefix}_balanced_accuracy': NAN, f'{prefix}_mcc': NAN,
                f'{prefix}_brier': NAN, f'{prefix}_ece': NAN}
    yt, yp, pp = y_true[keep], y_pred[keep], p[keep][:, :n_classes]
    out[f'{prefix}_balanced_accuracy'] = (float(balanced_accuracy_score(yt, yp))
                                          if len(np.unique(yt)) > 1 else NAN)
    out[f'{prefix}_mcc'] = (float(matthews_corrcoef(yt, yp))
                            if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1 else NAN)
    out[f'{prefix}_brier'] = multiclass_brier(pp, yt, n_classes)
    out[f'{prefix}_ece'] = expected_calibration_error(pp, yt, bins=10)
    return out


def dual_active_behaviour(y_true, accepted, prefix='dual'):
    """Dual-active molecules are never trained on; a trustworthy system should abstain on them."""
    y_true = np.asarray(y_true).astype(int)
    accepted = np.asarray(accepted, dtype=bool)
    m = y_true == 3
    n = int(m.sum())
    return {f'{prefix}_n': n,
            f'{prefix}_rejection_rate': float((~accepted[m]).mean()) if n else NAN}


def confusion(y_true, y_pred, n_classes=3):
    y_true = np.asarray(y_true).astype(int)
    keep = y_true < n_classes
    if keep.sum() == 0:
        return np.zeros((n_classes, n_classes), dtype=int)
    return confusion_matrix(y_true[keep], np.asarray(y_pred).astype(int)[keep],
                            labels=list(range(n_classes)))


def bootstrap_ci(values, reps=2000, seed=0, agg=np.nanmean):
    """Percentile bootstrap CI over fold-level values (the resampling unit is the fold)."""
    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)])
    if len(v) == 0:
        return NAN, NAN, NAN
    if len(v) == 1:
        return float(v[0]), NAN, NAN
    rng = np.random.default_rng(seed)
    draws = agg(v[rng.integers(0, len(v), size=(reps, len(v)))], axis=1)
    return float(agg(v)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def holm(pvalues, labels=None):
    """Holm-Bonferroni step-down adjustment over a declared comparison family."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj if labels is None else dict(zip(labels, adj))

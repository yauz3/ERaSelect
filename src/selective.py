"""Selective-prediction machinery and, centrally, its calibration-budget feasibility criterion.

The criterion
-------------
Split conformal prediction at nominal miscoverage alpha takes the order statistic at index
k = ceil((n+1)(1-alpha)) of n calibration nonconformity scores. That index exists only when

    k <= n   <=>   alpha*(n+1) >= 1   <=>   n >= 1/alpha - 1.

When it does not exist the conformal threshold is +infinity: the prediction set admits every
class, every molecule is "accepted", and any coverage or selective-risk number computed from it
is an artifact of the missing calibration budget rather than a property of the model. The
smallest miscoverage level a calibration sample of size n can certify is

    alpha_attainable(n) = 1/(n+1).

For Mondrian (class-conditional) conformal the criterion binds PER CLASS, so the minority class
governs the whole system. A one-class support verifier accepting an in-class fraction q is
subject to the analogous bound: its empirical (1-q) quantile is only non-degenerate once
n >= 1/(1-q).

Everything here is computed from training-partition labels BEFORE any held-out molecule is
scored, so the feasibility verdict can never be chosen to flatter a result.
"""
import numpy as np

# --------------------------------------------------------------------------- feasibility


def required_calibration_n(alpha):
    """Smallest calibration sample size for which nominal `alpha` is attainable."""
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError(f'alpha must lie in (0,1), got {alpha}')
    # Round before ceil: 1/(1-0.9) is 10.000000000000002 in IEEE double and would inflate the
    # requirement by one molecule, which matters when the minority class has tens of members.
    return int(np.ceil(round(1.0 / alpha - 1.0, 9)))


def attainable_alpha(n):
    """Smallest miscoverage level certifiable from `n` calibration scores."""
    n = int(n)
    return 1.0 if n <= 0 else 1.0 / (n + 1.0)


def required_verifier_n(acceptance):
    """Smallest calibration sample for a non-degenerate (1-acceptance) empirical quantile."""
    acceptance = float(acceptance)
    if not 0 < acceptance < 1:
        raise ValueError(f'verifier acceptance must lie in (0,1), got {acceptance}')
    return int(np.ceil(round(1.0 / (1.0 - acceptance), 9)))


def budget_report(counts, alpha, acceptance, classes=(0, 1, 2), moa_classes=(1, 2)):
    """Per-class feasibility verdict for one fold's calibration partitions.

    `counts` maps role name -> {class: n}. Returns a flat dict ready for a results table.
    """
    need_conf = required_calibration_n(alpha)
    need_ver = required_verifier_n(acceptance)
    rows = []
    for c in classes:
        n = int(counts.get('conformal', {}).get(c, 0))
        rows.append({'component': 'conformal_mondrian', 'klass': int(c), 'n': n,
                     'required_n': need_conf, 'nominal_alpha': float(alpha),
                     'attainable_alpha': attainable_alpha(n),
                     'feasible': bool(n >= need_conf)})
    pooled = int(sum(counts.get('conformal', {}).get(c, 0) for c in classes))
    rows.append({'component': 'conformal_pooled', 'klass': -1, 'n': pooled,
                 'required_n': need_conf, 'nominal_alpha': float(alpha),
                 'attainable_alpha': attainable_alpha(pooled),
                 'feasible': bool(pooled >= need_conf)})
    for c in moa_classes:
        n = int(counts.get('verifier', {}).get(c, 0))
        rows.append({'component': 'support_verifier', 'klass': int(c), 'n': n,
                     'required_n': need_ver, 'nominal_alpha': float(1 - acceptance),
                     'attainable_alpha': attainable_alpha(n),
                     'feasible': bool(n >= need_ver)})
    return rows


def effective_alphas(counts, alpha, classes=(0, 1, 2)):
    """Per-class alpha raised to the attainable floor where the nominal level is unreachable."""
    out, relaxed = {}, {}
    for c in classes:
        n = int(counts.get('conformal', {}).get(c, 0))
        floor = attainable_alpha(n)
        out[c] = float(max(alpha, floor))
        relaxed[c] = bool(floor > alpha)
    return out, relaxed


# --------------------------------------------------------------------------- conformal


def conformal_quantile(scores, alpha):
    """Exact finite-sample split-conformal order statistic; +inf when the index does not exist."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float('inf')
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else float('inf')


def conformal_thresholds(p, y, alpha, mode='mondrian', classes=(0, 1, 2)):
    """Return per-class thresholds on the nonconformity score 1 - p[true class].

    mode='pooled'   : one threshold from all calibration points (valid marginally only).
    mode='mondrian' : class-conditional thresholds (valid per class; minority class binds).
    `alpha` may be a scalar or a {class: alpha} mapping.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    alphas = {c: (alpha if np.isscalar(alpha) else alpha[c]) for c in classes}
    if mode == 'pooled':
        a = float(np.mean(list(alphas.values())))
        q = conformal_quantile(1 - p[np.arange(len(y)), y], a)
        return np.array([q] * len(classes), dtype=float)
    if mode != 'mondrian':
        raise ValueError(f'unknown conformal mode {mode!r}')
    return np.array([conformal_quantile(1 - p[y == c, c], alphas[c]) for c in classes], dtype=float)


def prediction_sets(p, thresholds):
    """Boolean set membership: class c is admitted when 1 - p_c <= threshold_c."""
    return (1 - np.asarray(p, dtype=float)) <= np.asarray(thresholds, dtype=float)[None, :]


def singleton_accept(sets, prediction):
    """Accept only when the conformal set is the single predicted class."""
    sets = np.asarray(sets, dtype=bool)
    idx = np.arange(len(prediction))
    return (sets.sum(1) == 1) & sets[idx, np.asarray(prediction, dtype=int)]


def degenerate_fraction(thresholds):
    t = np.asarray(thresholds, dtype=float)
    return float(np.mean(~np.isfinite(t))) if t.size else 1.0


# --------------------------------------------------------------------------- metrics


def selective_metrics(y_true, y_pred, accepted, n_classes=3):
    """Coverage, selective risk and retained-active fraction on one fold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    accepted = np.asarray(accepted, dtype=bool)
    n = len(y_true)
    cov = float(accepted.mean()) if n else float('nan')
    risk = float((y_pred[accepted] != y_true[accepted]).mean()) if accepted.any() else float('nan')
    active = y_true > 0
    retained = float(accepted[active].mean()) if active.any() else float('nan')
    out = {'coverage': cov, 'selective_risk': risk, 'retained_active_fraction': retained,
           'n_test': int(n), 'n_accepted': int(accepted.sum())}
    for c in range(n_classes):
        m = y_true == c
        out[f'coverage_class{c}'] = float(accepted[m].mean()) if m.any() else float('nan')
        am = m & accepted
        out[f'risk_class{c}'] = float((y_pred[am] != c).mean()) if am.any() else float('nan')
    return out


def risk_coverage_curve(y_true, y_pred, score, grid=None):
    """Selective risk as a function of coverage, sweeping a confidence score."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score)
    wrong = (y_pred[order] != y_true[order]).astype(float)
    n = len(y_true)
    if n == 0:
        return np.array([]), np.array([])
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(wrong) / np.arange(1, n + 1)
    if grid is None:
        return cov, risk
    grid = np.asarray(grid, dtype=float)
    return grid, np.interp(grid, cov, risk)


def aurc(y_true, y_pred, score):
    """Area under the risk-coverage curve; lower is better."""
    cov, risk = risk_coverage_curve(y_true, y_pred, score)
    if len(cov) <= 1:
        return float('nan')
    # np.trapezoid is NumPy >= 2.0; np.trapz is the <2.0 spelling. Chemistry stacks
    # frequently pin NumPy 1.x, so resolve at call time rather than assuming.
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return float(integrate(risk, cov))

"""Probability recalibration fitted on a disjoint partition. Several methods for benchmarking."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar


def require_classes(y, classes, where):
    missing = set(classes) - set(np.asarray(y).tolist())
    if missing:
        raise ValueError(f'{where}: missing classes {sorted(missing)}')


def _log(p):
    return np.log(np.clip(p, 1e-9, 1.0))


class ProbabilityCalibrator:
    """method: platt (multinomial LR on log-probs) | isotonic (per class, renormalised)
    | temperature (single scalar on logits) | none (identity)."""

    def __init__(self, method='platt'):
        if method not in ('platt', 'isotonic', 'temperature', 'none'):
            raise ValueError(f'unknown calibration method {method!r}')
        self.method = method

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.arange(p.shape[1])
        if self.method == 'none':
            return self
        require_classes(y, self.classes_, f'{self.method} calibration')
        if self.method == 'platt':
            self.model_ = LogisticRegression(C=1.0, max_iter=2000).fit(_log(p), y)
        elif self.method == 'isotonic':
            self.model_ = [IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                           .fit(p[:, c], (y == c).astype(float)) for c in self.classes_]
        else:
            z = _log(p)

            def nll(log_t):
                q = np.exp(z / np.exp(log_t))
                q /= q.sum(1, keepdims=True)
                return -np.mean(_log(q)[np.arange(len(y)), y])

            r = minimize_scalar(nll, bounds=(-3.0, 3.0), method='bounded')
            self.temperature_ = float(np.exp(r.x))
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == 'none':
            return p / p.sum(1, keepdims=True)
        if self.method == 'platt':
            out = self.model_.predict_proba(_log(p))
            # LogisticRegression drops absent classes; restore the full simplex.
            if out.shape[1] != len(self.classes_):
                full = np.zeros((len(p), len(self.classes_)))
                full[:, self.model_.classes_.astype(int)] = out
                out = full
        elif self.method == 'isotonic':
            out = np.column_stack([m.predict(p[:, c]) for c, m in enumerate(self.model_)])
        else:
            z = _log(p) / self.temperature_
            out = np.exp(z - z.max(1, keepdims=True))
        s = out.sum(1, keepdims=True)
        return np.divide(out, s, out=np.full_like(out, 1.0 / out.shape[1]), where=s > 0)


def expected_calibration_error(p, y, bins=10):
    """Multiclass ECE on the top-1 confidence."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def multiclass_brier(p, y, n_classes=None):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    k = n_classes or p.shape[1]
    onehot = np.zeros((len(y), k))
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(((p[:, :k] - onehot) ** 2).sum(1)))

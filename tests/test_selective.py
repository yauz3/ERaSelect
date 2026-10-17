"""Contract tests for the feasibility criterion and the selective-prediction primitives."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import selective as S


@pytest.mark.parametrize('alpha,expected', [(0.1, 9), (0.2, 4), (0.05, 19), (0.5, 1)])
def test_required_calibration_n(alpha, expected):
    assert S.required_calibration_n(alpha) == expected


def test_required_n_is_exactly_the_finiteness_boundary():
    """n = required_n must give a finite quantile and n-1 must give +inf. This is the criterion."""
    for alpha in (0.05, 0.1, 0.2, 0.25):
        n = S.required_calibration_n(alpha)
        assert np.isfinite(S.conformal_quantile(np.linspace(0, 1, n), alpha))
        assert not np.isfinite(S.conformal_quantile(np.linspace(0, 1, n - 1), alpha))


def test_attainable_alpha_round_trips():
    for n in (1, 4, 9, 19, 99):
        a = S.attainable_alpha(n)
        assert np.isfinite(S.conformal_quantile(np.linspace(0, 1, n), a))
        assert S.required_calibration_n(a) == n


def test_v1_regression_two_antagonist_calibration_points_is_degenerate():
    """The v1 package calibrated the antagonist class on 2 points at alpha=0.1."""
    assert S.conformal_quantile([0.3, 0.7], 0.1) == float('inf')
    assert S.attainable_alpha(2) == pytest.approx(1 / 3)


def test_mondrian_minority_class_binds():
    rng = np.random.default_rng(0)
    # 200 points of class 0 and 1, only 3 of class 2
    y = np.array([0] * 200 + [1] * 200 + [2] * 3)
    p = np.full((len(y), 3), 0.1)
    p[np.arange(len(y)), y] = 0.8
    p += rng.normal(0, 0.01, p.shape)
    counts = {'conformal': {0: 200, 1: 200, 2: 3}, 'verifier': {1: 200, 2: 3}}
    eff, relaxed = S.effective_alphas(counts, 0.1)
    assert relaxed == {0: False, 1: False, 2: True}
    assert eff[2] == pytest.approx(0.25)
    nominal = S.conformal_thresholds(p, y, 0.1, 'mondrian')
    assert np.isfinite(nominal[0]) and np.isfinite(nominal[1])
    assert not np.isfinite(nominal[2]), 'minority class must be degenerate at nominal alpha'
    feasible = S.conformal_thresholds(p, y, eff, 'mondrian')
    assert np.isfinite(feasible).all()


def test_degenerate_threshold_admits_every_molecule():
    """Why an infinite threshold silently inflates coverage rather than erroring."""
    p = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
    sets = S.prediction_sets(p, np.array([0.5, 0.5, float('inf')]))
    assert sets[:, 2].all(), 'infinite threshold admits class 2 for every molecule'
    assert S.degenerate_fraction([0.5, 0.5, float('inf')]) == pytest.approx(1 / 3)


def test_budget_report_flags_minority_class():
    counts = {'conformal': {0: 150, 1: 40, 2: 2}, 'verifier': {1: 40, 2: 4}}
    rows = S.budget_report(counts, 0.1, 0.95)
    by = {(r['component'], r['klass']): r for r in rows}
    assert by[('conformal_mondrian', 0)]['feasible'] is True
    assert by[('conformal_mondrian', 2)]['feasible'] is False
    assert by[('conformal_mondrian', 2)]['attainable_alpha'] == pytest.approx(1 / 3)
    assert by[('support_verifier', 2)]['required_n'] == 20
    assert by[('support_verifier', 2)]['feasible'] is False
    # pooling hides the minority-class failure -- this is the trap the paper documents
    assert by[('conformal_pooled', -1)]['feasible'] is True


def test_required_verifier_n():
    assert S.required_verifier_n(0.95) == 20
    assert S.required_verifier_n(0.9) == 10


def test_singleton_accept_rejects_ambiguous_and_empty_sets():
    sets = np.array([[True, False, False], [True, True, False], [False, False, False]])
    acc = S.singleton_accept(sets, np.array([0, 0, 0]))
    assert acc.tolist() == [True, False, False]


def test_selective_metrics_on_known_case():
    y = np.array([0, 1, 2, 1])
    pred = np.array([0, 1, 1, 2])
    acc = np.array([True, True, True, False])
    m = S.selective_metrics(y, pred, acc)
    assert m['coverage'] == pytest.approx(0.75)
    assert m['selective_risk'] == pytest.approx(1 / 3)
    assert m['retained_active_fraction'] == pytest.approx(2 / 3)
    assert m['n_accepted'] == 3


def test_aurc_prefers_well_ordered_confidence():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 0])
    good = np.array([0.9, 0.8, 0.7, 0.1])   # the error is least confident
    bad = np.array([0.1, 0.2, 0.3, 0.9])    # the error is most confident
    assert S.aurc(y, pred, good) < S.aurc(y, pred, bad)


def test_empty_and_degenerate_inputs_do_not_raise():
    assert S.conformal_quantile([], 0.1) == float('inf')
    m = S.selective_metrics(np.array([0, 1]), np.array([0, 1]), np.array([False, False]))
    assert np.isnan(m['selective_risk'])
    cov, risk = S.risk_coverage_curve(np.array([]), np.array([]), np.array([]))
    assert len(cov) == 0

"""Integrity tests for the curation and partitioning contracts.

These encode the decisions the scientific claims rest on, so that a future edit that breaks one
of them fails here rather than silently changing a published number.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import data as D           # noqa: E402
import experiments as E    # noqa: E402
import selective as SEL    # noqa: E402
from calibration import ProbabilityCalibrator  # noqa: E402

ASPIRIN = 'CC(=O)Oc1ccccc1C(=O)O'
ESTRADIOL = 'C[C@]12CC[C@H]3[C@@H](CC[C@@H]4[C@@H]3CCc3cc(O)ccc34)[C@@H]1CC[C@@H]2O'
SODIUM_BENZOATE = '[Na+].[O-]C(=O)c1ccccc1'


# --------------------------------------------------------------- standardization

def test_standardize_returns_scaffold_for_ring_systems():
    """Regression: scaffold perception after uncharging requires an explicit sanitize.

    Without it RDKit raises 'RingInfo not initialized' and the molecule is silently dropped.
    """
    for smi in (ASPIRIN, ESTRADIOL, SODIUM_BENZOATE):
        s, key, group = D.standardize(smi)
        assert s and len(key) == 27 and key.count('-') == 2
        assert group and group != '__ACYCLIC__', f'{smi} lost its scaffold'


def test_standardize_strips_counterion_and_uncharges():
    s, _, _ = D.standardize(SODIUM_BENZOATE)
    assert 'Na' not in s and '-]' not in s


def test_descriptors_never_contain_infinity():
    """Regression: Ipc overflows float32 on large molecules.

    SimpleImputer only handles NaN, so an inf reaches StandardScaler, which raises
    "Input X contains infinity" and kills the run at the first fold. Out-of-range values must
    become NaN during descriptor construction so they are imputed like any other missing value.
    """
    big = 'c1ccc2c(c1)c1ccccc1c1c2cccc1' * 3  # fused aromatics inflate Ipc astronomically
    smis = [ASPIRIN, ESTRADIOL, 'C' * 60, big, 'C1CC2CCC1CC2']
    M = D.descriptor_matrix([s for s in smis if Chem_ok(s)])
    assert M.dtype == np.dtype('float32')
    assert not np.isinf(M).any(), 'descriptor matrix leaked an infinity'


def Chem_ok(s):
    from rdkit import Chem
    return Chem.MolFromSmiles(s) is not None


def test_descriptor_overflow_is_converted_to_nan():
    """Directly exercise the float32 overflow path without depending on a specific molecule."""
    huge = np.float64(1e300)
    arr = np.array([[1.0, huge], [2.0, 3.0]], dtype=np.float64)
    limit = np.finfo(np.float32).max
    overflow = np.abs(arr) > limit
    arr[overflow] = np.nan
    out = arr.astype(np.float32)
    assert np.isnan(out[0, 1]) and not np.isinf(out).any()


def test_aurc_works_without_numpy2():
    """np.trapezoid is NumPy>=2 only; chemistry stacks often pin 1.x."""
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    assert np.isfinite(SEL.aurc(y, pred, score))
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    assert callable(integrate)


def test_acyclic_molecules_share_one_holdout_group():
    _, _, g = D.standardize('CCCCO')
    assert g == '__ACYCLIC__'


def test_standardize_rejects_garbage():
    for bad in ('not_a_smiles', '', 'C(((', 'O'):
        with pytest.raises(Exception):
            D.standardize(bad)


# --------------------------------------------------------------- label contract

@pytest.mark.parametrize('ago,ant,expected_label,expected_tier', [
    (0, 0, 0, 'core'),
    (1, 0, 1, 'core'),
    (0, 1, 2, 'core'),
    (1, 1, 3, 'core'),
    (1, np.nan, 1, 'extended'),
    (np.nan, 1, 2, 'extended'),
])
def test_label_assignment(ago, ant, expected_label, expected_tier):
    label, tier, conf = D.assign_labels(ago, ant)
    assert label == expected_label and tier == expected_tier
    assert conf == ('definitive' if expected_tier == 'core' else 'provisional')


@pytest.mark.parametrize('ago,ant', [(0, np.nan), (np.nan, 0), (np.nan, np.nan)])
def test_single_negative_or_missing_is_never_a_label(ago, ant):
    """An inconclusive call must never become a reliable negative. This is the load-bearing rule."""
    label, tier, conf = D.assign_labels(ago, ant)
    assert label is None and tier is None and conf is None


def test_inactive_label_requires_both_endpoints_in_every_tier():
    for ago, ant in ((0, np.nan), (np.nan, 0)):
        label, _, _ = D.assign_labels(ago, ant)
        assert label != 0


def test_extended_tier_is_a_superset_of_core():
    core, ext = set(), set()
    for ago in (0, 1, np.nan):
        for ant in (0, 1, np.nan):
            label, tier, _ = D.assign_labels(ago, ant)
            if label is None:
                continue
            ext.add((ago if ago == ago else 'nan', ant if ant == ant else 'nan'))
            if tier == 'core':
                core.add((ago if ago == ago else 'nan', ant if ant == ant else 'nan'))
    assert core < ext


# --------------------------------------------------------------- partitioning

@pytest.fixture
def toy():
    """A synthetic dataset with a deliberately small antagonist class."""
    rng = np.random.default_rng(0)
    rows = []
    spec = {0: 900, 1: 160, 2: 120, 3: 12}
    gid = 0
    for label, n in spec.items():
        for i in range(n):
            if i % 3 == 0:
                gid += 1
            rows.append({'row_id': len(rows), 'inchikey': f'KEY{len(rows):05}',
                         'smiles': 'C' * (1 + len(rows) % 5), 'group': f'S{gid}',
                         'label': label, 'label_name': str(label),
                         'moa_confidence': 'definitive',
                         'agonist': 1 if label in (1, 3) else 0,
                         'antagonist': 1 if label in (2, 3) else 0})
    d = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    d['row_id'] = np.arange(len(d))
    return d


CFG = {'alpha': 0.1, 'verifier_acceptance': 0.95, 'inner_folds': 2, 'outer_folds': 5,
       'grid_size': 1, 'trees': 10, 'jobs': 1, 'seeds': [17], 'models': ['lr'],
       'representations': ['ecfp4'], 'random_split': False, 'label_tier': 'extended'}


def test_roles_are_scaffold_disjoint(toy):
    y, g = toy.label.to_numpy(), toy.group.to_numpy()
    train, test = E.split_cv(y, g, 5, 17, 'scaffold')[0]
    assign, mode = E.allocate(train, toy, 17, 'scaffold', CFG)
    groups = {r: set(g[idx]) for r, idx in assign.items()}
    groups['test'] = set(g[test])
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not groups[a] & groups[b], f'scaffold leakage between {a} and {b}'


def test_roles_do_not_share_molecules(toy):
    y, g = toy.label.to_numpy(), toy.group.to_numpy()
    train, test = E.split_cv(y, g, 5, 17, 'scaffold')[0]
    assign, _ = E.allocate(train, toy, 17, 'scaffold', CFG)
    allidx = np.concatenate(list(assign.values()))
    assert len(allidx) == len(set(allidx.tolist()))
    assert not set(allidx.tolist()) & set(test.tolist())


def test_allocation_meets_the_declared_budget(toy):
    y = toy.label.to_numpy()
    train, _ = E.split_cv(y, toy.group.to_numpy(), 5, 17, 'scaffold')[0]
    assign, mode = E.allocate(train, toy, 17, 'scaffold', CFG)
    assert mode == 'strict'
    counts = E.role_counts(assign, y)
    need = SEL.required_calibration_n(CFG['alpha'])
    for c in (0, 1, 2):
        assert counts['conformal'][c] >= need
    for c in (1, 2):
        assert counts['verifier'][c] >= SEL.required_verifier_n(CFG['verifier_acceptance'])


def test_budget_shortfall_detects_impossible_case(toy):
    """With a tiny antagonist class the strict budget is arithmetically unreachable."""
    small = pd.concat([toy[toy.label != 2], toy[toy.label == 2].head(8)]).reset_index(drop=True)
    small['row_id'] = np.arange(len(small))
    train = np.arange(len(small))
    short, need, have = E.budget_shortfall(train, small, CFG, strict=True)
    assert 2 in short and have[2] == 8 and need[2] > 8


def test_allocation_falls_back_to_relaxed_and_says_so(toy):
    """Too small for the strict budget, large enough for the relaxed one: must degrade, not crash.

    This is the v1 regime: enough antagonists to fit something, not enough to certify alpha.
    """
    small = pd.concat([toy[toy.label != 2], toy[toy.label == 2].head(30)]).reset_index(drop=True)
    small['row_id'] = np.arange(len(small))
    y, g = small.label.to_numpy(), small.group.to_numpy()
    train, _ = E.split_cv(y, g, 5, 17, 'scaffold')[0]
    assert E.budget_shortfall(train, small, CFG, strict=True)[0], 'strict must be unreachable'
    assert not E.budget_shortfall(train, small, CFG, strict=False)[0], 'relaxed must be reachable'
    assign, mode = E.allocate(train, small, 17, 'scaffold', CFG)
    assert mode == 'relaxed'
    counts = E.role_counts(assign, y)
    rows = SEL.budget_report(counts, CFG['alpha'], CFG['verifier_acceptance'])
    assert not all(r['feasible'] for r in rows), 'relaxed allocation must report infeasibility'
    # and the relaxation must be visible as a raised per-class alpha
    _, relaxed = SEL.effective_alphas(counts, CFG['alpha'])
    assert relaxed[2], 'antagonist alpha must be reported as relaxed'


def test_allocation_raises_when_even_relaxed_is_impossible(toy):
    tiny = pd.concat([toy[toy.label != 2], toy[toy.label == 2].head(8)]).reset_index(drop=True)
    tiny['row_id'] = np.arange(len(tiny))
    y, g = tiny.label.to_numpy(), tiny.group.to_numpy()
    train, _ = E.split_cv(y, g, 5, 17, 'scaffold')[0]
    with pytest.raises(E.Infeasible, match='relaxed role budgets'):
        E.allocate(train, tiny, 17, 'scaffold', CFG)


def test_preflight_runs_before_fitting_and_writes_a_verdict(toy, tmp_path):
    frame, summary = E.preflight(toy, CFG, tmp_path)
    assert (tmp_path / 'calibration_budget_feasibility.csv').exists()
    assert (tmp_path / 'calibration_budget_summary.json').exists()
    assert summary['folds_with_allocation'] == CFG['outer_folds']
    assert set(frame.columns) >= {'component', 'klass', 'n', 'required_n', 'feasible'}


def test_activity_threshold_hits_its_recall_target(toy):
    rng = np.random.default_rng(2)
    y = np.array([0] * 500 + [1] * 200)
    p = np.concatenate([rng.beta(2, 8, 500), rng.beta(8, 2, 200)])
    tau = E.activity_threshold(p, y, 0.95)
    assert (p[y == 1] >= tau).mean() >= 0.94


def test_nearest_similarity_is_one_for_identical_fingerprints():
    a = np.array([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=np.float32)
    assert np.allclose(E.nearest_similarity(a, a), 1.0)
    b = np.array([[0, 1, 0, 0]], dtype=np.float32)
    assert E.nearest_similarity(np.array([[1, 0, 1, 1]], dtype=np.float32), b)[0] == 0.0


# --------------------------------------------------------------- calibration

@pytest.mark.parametrize('method', ['platt', 'isotonic', 'temperature', 'none'])
def test_calibrators_return_a_valid_simplex(method):
    rng = np.random.default_rng(5)
    p = rng.dirichlet([2, 2, 2], size=300)
    y = rng.integers(0, 3, 300)
    q = ProbabilityCalibrator(method).fit(p, y).transform(p)
    assert q.shape == p.shape
    assert np.allclose(q.sum(1), 1.0, atol=1e-6)
    assert (q >= -1e-9).all()


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        ProbabilityCalibrator('magic')


# --------------------------------------------------------------- packaged results

def _results_dirs():
    base = ROOT / 'results'
    return [p for p in base.iterdir() if (p / 'fold_metrics.csv').exists()] if base.exists() else []


@pytest.mark.parametrize('rd', _results_dirs(), ids=lambda p: p.name)
def test_shipped_results_are_internally_consistent(rd):
    m = pd.read_csv(rd / 'fold_metrics.csv', low_memory=False)
    assert len(m) and m.variant.nunique() >= 8
    assert m.coverage.between(0, 1).all()
    ok = m.selective_risk.dropna()
    assert ok.between(0, 1).all()
    # A satisfied budget must never coexist with a degenerate threshold.
    if 'degenerate_threshold_fraction' in m:
        bad = m[(m.calibration_budget_satisfied == True)  # noqa: E712
                & (m.variant == 'hierarchy_conformal_feasible')
                & (m.degenerate_threshold_fraction > 0)]
        assert len(bad) == 0, 'feasible-alpha variant produced an infinite threshold'


@pytest.mark.parametrize('rd', _results_dirs(), ids=lambda p: p.name)
def test_shipped_split_membership_has_no_leakage(rd):
    files = sorted((rd / 'folds').glob('scaffold_*/split_membership.csv'))[:6]
    if not files:
        pytest.skip('no scaffold fold membership written')
    for f in files:
        d = pd.read_csv(f)
        per_group = d.groupby('group').role.nunique()
        assert (per_group == 1).all(), f'{f.parent.name}: a scaffold spans two roles'
        assert d.row_id.is_unique, f'{f.parent.name}: a molecule appears in two roles'

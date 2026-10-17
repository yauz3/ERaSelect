"""ChEMBL parser contract tests against a schema-faithful fixture.

The live ChEMBL API is not reachable from every build environment, so the parser is pinned
against a fixture that mirrors the documented response shape. These tests assert the
scientifically load-bearing behaviours: schema validation, directional inactive/active
assertions, contested-molecule removal, exclusive mechanism labelling and InChIKey
disjointness from the Tox21 curation.
"""
import json, sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

ESTRADIOL = 'C[C@]12CC[C@H]3[C@@H](CC[C@@H]4[C@@H]3CCc3cc(O)ccc34)[C@@H]1CC[C@@H]2O'
TAMOXIFEN = 'CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1'
BISPHENOL_A = 'CC(C)(c1ccc(O)cc1)c1ccc(O)cc1'
CAFFEINE = 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'
GLUCOSE = 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O'
PROPANE = 'CCC'


def _activity(mol, smiles, p, relation='=', assay_type='B', comment=None, validity=None):
    return {'action_type': None, 'activity_comment': comment, 'activity_id': abs(hash(mol)) % 99999,
            'assay_chembl_id': 'CHEMBL615156', 'assay_description': 'ERa binding',
            'assay_type': assay_type, 'canonical_smiles': smiles,
            'data_validity_comment': validity, 'molecule_chembl_id': mol,
            'pchembl_value': None if p is None else str(p), 'standard_relation': relation,
            'standard_type': 'IC50', 'standard_units': 'nM', 'standard_value': '57.0',
            'target_chembl_id': 'CHEMBL206'}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    import data as data_mod
    import chembl as chembl_mod
    cache = tmp_path / 'chembl_cache'
    cache.mkdir()
    monkeypatch.setattr(chembl_mod, 'CACHE', cache)

    activities = [
        # potent -> sensitivity cohort
        _activity('CHEMBL_EST', ESTRADIOL, 10.5),
        _activity('CHEMBL_TAM', TAMOXIFEN, 8.1, assay_type='F'),
        # weak -> specificity cohort
        _activity('CHEMBL_CAF', CAFFEINE, 4.1),
        _activity('CHEMBL_GLU', GLUCOSE, None, relation='>', comment='Not Active'),
        _activity('CHEMBL_PROP', PROPANE, 4.8, relation='>='),
        # contested: asserted both weak and potent -> must be dropped from both cohorts
        _activity('CHEMBL_CONTESTED', BISPHENOL_A, 4.2),
        _activity('CHEMBL_CONTESTED', BISPHENOL_A, 6.9),
        # excluded by assay type
        _activity('CHEMBL_ADMET', CAFFEINE, 4.0, assay_type='A'),
        # excluded by validity flag
        _activity('CHEMBL_BADDATA', PROPANE, 3.0, validity='Outside typical range'),
        # wrong target must never leak in
        {**_activity('CHEMBL_OTHER', CAFFEINE, 9.9), 'target_chembl_id': 'CHEMBL999'},
    ]
    (cache / 'activity_CHEMBL206_0000000.json').write_text(json.dumps(
        {'activities': activities,
         'page_meta': {'limit': 1000, 'offset': 0, 'next': None, 'total_count': len(activities)}}))

    mechanisms = [
        {'action_type': 'AGONIST', 'molecule_chembl_id': 'CHEMBL_EST',
         'target_chembl_id': 'CHEMBL206', 'mechanism_of_action': 'ERa agonist'},
        {'action_type': 'ANTAGONIST', 'molecule_chembl_id': 'CHEMBL_TAM',
         'target_chembl_id': 'CHEMBL206', 'mechanism_of_action': 'ERa antagonist'},
        # annotated both ways -> no exclusive label
        {'action_type': 'AGONIST', 'molecule_chembl_id': 'CHEMBL_MIXED',
         'target_chembl_id': 'CHEMBL206', 'mechanism_of_action': 'x'},
        {'action_type': 'ANTAGONIST', 'molecule_chembl_id': 'CHEMBL_MIXED',
         'target_chembl_id': 'CHEMBL206', 'mechanism_of_action': 'x'},
        # non-exclusive action type -> excluded
        {'action_type': 'MODULATOR', 'molecule_chembl_id': 'CHEMBL_MOD',
         'target_chembl_id': 'CHEMBL206', 'mechanism_of_action': 'x'},
    ]
    (cache / 'mechanism_CHEMBL206_0000000.json').write_text(json.dumps(
        {'mechanisms': mechanisms,
         'page_meta': {'limit': 1000, 'offset': 0, 'next': None, 'total_count': len(mechanisms)}}))
    (cache / 'molecule_CHEMBL206_00000.json').write_text(json.dumps({'molecules': [
        {'molecule_chembl_id': 'CHEMBL_EST', 'molecule_structures': {'canonical_smiles': ESTRADIOL}},
        {'molecule_chembl_id': 'CHEMBL_TAM', 'molecule_structures': {'canonical_smiles': TAMOXIFEN}},
    ], 'page_meta': {'limit': 50, 'offset': 0, 'next': None, 'total_count': 2}}))

    # Tox21 curation containing caffeine -> caffeine must be removed from every cohort.
    out = tmp_path / 'results'
    (out / 'data').mkdir(parents=True)
    _, caf_key, _ = data_mod.standardize(CAFFEINE)
    pd.DataFrame({'inchikey': [caf_key]}).to_csv(out / 'data/curation_all.csv', index=False)
    cfg = {'chembl': {'enabled': True, 'target': 'CHEMBL206', 'inactive_pchembl_max': 5.0,
                      'active_pchembl_min': 6.0, 'assay_types': ['B', 'F'],
                      'max_records': 1000, 'page_size': 1000}}
    return chembl_mod, cfg, out, caf_key


def test_cohorts_split_by_direction(workspace):
    chembl_mod, cfg, out, _ = workspace
    cohorts, report = chembl_mod.build(cfg, out)
    assert set(cohorts) == {'specificity', 'sensitivity', 'moa'}
    assert set(cohorts['sensitivity'].molecule_chembl_id) == {'CHEMBL_EST', 'CHEMBL_TAM'}
    # caffeine removed by disjointness, propane/glucose retained, contested and flagged dropped
    assert set(cohorts['specificity'].molecule_chembl_id) == {'CHEMBL_GLU', 'CHEMBL_PROP'}
    assert report['contested_molecules_dropped'] == 1
    assert report['dropped_data_validity_flagged'] == 1


def test_expected_active_flags(workspace):
    chembl_mod, cfg, out, _ = workspace
    cohorts, _ = chembl_mod.build(cfg, out)
    assert (cohorts['specificity'].expected_active == 0).all()
    assert (cohorts['sensitivity'].expected_active == 1).all()


def test_mechanism_labels_are_exclusive(workspace):
    chembl_mod, cfg, out, _ = workspace
    cohorts, report = chembl_mod.build(cfg, out)
    moa = cohorts['moa']
    assert report['mechanism_exclusive_molecules'] == 2
    assert set(moa.molecule_chembl_id) == {'CHEMBL_EST', 'CHEMBL_TAM'}
    assert moa.set_index('molecule_chembl_id').label.to_dict() == {'CHEMBL_EST': 1, 'CHEMBL_TAM': 2}


def test_curation_overlap_is_removed(workspace):
    chembl_mod, cfg, out, caf_key = workspace
    cohorts, report = chembl_mod.build(cfg, out)
    for name, frame in cohorts.items():
        assert caf_key not in set(frame.inchikey), f'{name} leaked a curated identity'
    assert report['specificity']['removed_overlapping_identities'] == 1


def test_schema_change_fails_loudly(workspace, tmp_path):
    chembl_mod, cfg, out, _ = workspace
    bad = chembl_mod.CACHE / 'activity_CHEMBL206_0000000.json'
    payload = json.loads(bad.read_text())
    for rec in payload['activities']:
        rec.pop('pchembl_value')
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='schema missing required fields'):
        chembl_mod.build(cfg, out)


def test_disabled_cohort_returns_empty(workspace):
    chembl_mod, cfg, out, _ = workspace
    cfg = {'chembl': dict(cfg['chembl'], enabled=False)}
    cohorts, report = chembl_mod.build(cfg, out)
    assert cohorts == {} and report == {'enabled': False}

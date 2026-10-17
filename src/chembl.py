"""Independent ChEMBL cohorts for external evaluation of the ERa selective pipeline.

Three cohorts are built, each InChIKey-disjoint from the ENTIRE Tox21 curation
(including molecules the curation could not label), and each answering a different question:

  specificity : ERa measurements that assert weak/absent potency (pChEMBL < inactive_pchembl_max,
                or an explicit inactive annotation). Question: what fraction does the activity
                gate wrongly promote?  -> external false-positive rate.
  sensitivity : ERa measurements asserting potency (pChEMBL >= active_pchembl_min).
                Question: what fraction does the gate retain? -> external recall.
  moa         : ChEMBL curated mechanism annotations with action_type AGONIST or ANTAGONIST.
                Question: does Stage 2 recover an independently assigned mode of action?

INTERPRETATION LIMIT, stated here because it governs every claim downstream: ChEMBL aggregates
heterogeneous binding and functional assays from the primary literature. A ChEMBL "inactive" is
not a Tox21 VM7Luc4E2 reporter negative, and a ChEMBL mechanism label is a curator's reading of a
publication, not a replicate of these assays. These cohorts therefore estimate external
behaviour under distribution shift. They are NOT a replication of the training endpoints and the
pipeline never trains on them.
"""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from data import ROOT, standardize

CACHE = ROOT / 'data/chembl'
BASE = 'https://www.ebi.ac.uk/chembl/api/data'
# Fields the parser depends on. A ChEMBL schema change must fail loudly, not silently drop data.
ACTIVITY_FIELDS = {'canonical_smiles', 'pchembl_value', 'standard_type', 'standard_relation',
                   'standard_value', 'standard_units', 'assay_type', 'molecule_chembl_id',
                   'assay_chembl_id', 'target_chembl_id'}
MECHANISM_FIELDS = {'molecule_chembl_id', 'action_type', 'target_chembl_id'}


def _page(url, path, timeout=120):
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=timeout, headers={'Accept': 'application/json'})
            r.raise_for_status()
            payload = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            time.sleep(.2)
            return payload
        except (requests.RequestException, ValueError) as e:
            if attempt == 4:
                raise RuntimeError(
                    f'ChEMBL acquisition failed for {url} after 5 attempts ({e!r}) and no cache '
                    f'exists at {path}. Either enable network access or set chembl.enabled=false '
                    f'in the config; the pipeline will NOT fabricate an external cohort.') from e
            time.sleep(min(2 ** attempt, 15))


def _collect(resource, params, key, fields, max_records, page_size, tag):
    rows, offset = [], 0
    while len(rows) < max_records:
        q = dict(params, limit=page_size, offset=offset, format='json')
        url = f'{BASE}/{resource}.json?' + '&'.join(f'{k}={v}' for k, v in q.items())
        payload = _page(url, CACHE / f'{tag}_{offset:07}.json')
        batch = payload.get(key)
        if batch is None:
            raise ValueError(f'ChEMBL response for {resource} lacks "{key}": {list(payload)[:8]}')
        if not batch:
            break
        missing = fields - set(batch[0])
        if missing:
            raise ValueError(f'ChEMBL {resource} schema missing required fields {sorted(missing)}. '
                             f'Parser must be updated before results can be trusted.')
        rows.extend(batch)
        total = (payload.get('page_meta') or {}).get('total_count')
        offset += page_size
        if len(batch) < page_size or (total is not None and offset >= total):
            break
    return pd.DataFrame(rows[:max_records])


def _standardize_frame(df, smiles_col, label_col=None):
    out, bad = [], []
    for _, r in df.iterrows():
        s0 = r.get(smiles_col)
        if not isinstance(s0, str) or not s0.strip():
            bad.append({'molecule_chembl_id': r.get('molecule_chembl_id'), 'reason': 'no_smiles'})
            continue
        try:
            s, key, group = standardize(s0)
        except Exception as e:
            bad.append({'molecule_chembl_id': r.get('molecule_chembl_id'),
                        'reason': str(e).splitlines()[0][:100]})
            continue
        rec = {'molecule_chembl_id': r.get('molecule_chembl_id'), 'smiles': s,
               'inchikey': key, 'group': group}
        if label_col:
            rec['label'] = r[label_col]
        out.append(rec)
    return pd.DataFrame(out), pd.DataFrame(bad, columns=['molecule_chembl_id', 'reason'])


def build(cfg, out):
    """Return {cohort_name: DataFrame} plus a scope report. Never raises on an empty cohort."""
    spec = cfg.get('chembl', {})
    dest = out / 'external'
    dest.mkdir(parents=True, exist_ok=True)

    # Reuse cohorts this results directory already built. Cohort construction is deterministic
    # given the curation, so re-deriving it would only risk drift; and it lets the external step
    # be re-run offline after a failure further downstream, without refetching ChEMBL.
    existing = {n: dest / f'chembl_{n}.csv' for n in ('specificity', 'sensitivity', 'moa')}
    scope = dest / 'chembl_scope.json'
    if spec.get('enabled', False) and scope.exists() and any(p.exists() for p in existing.values()):
        cohorts = {}
        for name, path in existing.items():
            if path.exists():
                frame = pd.read_csv(path)
                if len(frame):
                    cohorts[name] = frame
        report = json.loads(scope.read_text())
        report['reused_from_disk'] = True
        print('[chembl] reusing cohorts already on disk: '
              + json.dumps({k: len(v) for k, v in cohorts.items()}), flush=True)
        return cohorts, report

    if not spec.get('enabled', False):
        (dest / 'chembl_scope.json').write_text(json.dumps(
            {'enabled': False, 'reason': 'chembl.enabled is false in the active config'}, indent=2))
        return {}, {'enabled': False}

    target = spec['target']
    page_size = int(spec.get('page_size', 1000))
    max_records = int(spec.get('max_records', 60000))
    assay_types = set(spec.get('assay_types', ['B', 'F']))

    act = _collect('activity', {'target_chembl_id': target}, 'activities', ACTIVITY_FIELDS,
                   max_records, page_size, f'activity_{target}')
    mech = _collect('mechanism', {'target_chembl_id': target}, 'mechanisms', MECHANISM_FIELDS,
                    5000, page_size, f'mechanism_{target}')

    report = {'enabled': True, 'target': target,
              'activity_records': int(len(act)), 'mechanism_records': int(len(mech))}

    # ---- activity-derived cohorts -------------------------------------------------
    act = act[act.target_chembl_id == target].copy()
    if 'data_validity_comment' in act:
        flagged = act.data_validity_comment.notna()
        report['dropped_data_validity_flagged'] = int(flagged.sum())
        act = act[~flagged]
    act = act[act.assay_type.isin(assay_types)]
    act['p'] = pd.to_numeric(act.pchembl_value, errors='coerce')
    comment = (act.get('activity_comment', pd.Series(index=act.index, dtype=object))
               .astype(str).str.lower())

    # An inactive assertion is either a measured weak potency, or a ">" relation at a weak
    # concentration, or an explicit curator annotation. All three are directional statements.
    weak = act.p.notna() & (act.p < float(spec.get('inactive_pchembl_max', 5.0)))
    right_censored_weak = (act.standard_relation.astype(str).isin(['>', '>=']) & act.p.notna()
                           & (act.p <= float(spec.get('inactive_pchembl_max', 5.0))))
    annotated = comment.str.contains('not active|inactive|no activity', na=False)
    potent = act.p.notna() & (act.p >= float(spec.get('active_pchembl_min', 6.0)))

    neg_src = act[(weak | right_censored_weak | annotated) & ~potent]
    pos_src = act[potent]

    # A molecule asserted both weak and potent across the literature is ambiguous: drop it.
    contested = set(neg_src.molecule_chembl_id) & set(pos_src.molecule_chembl_id)
    report['contested_molecules_dropped'] = len(contested)
    neg_src = neg_src[~neg_src.molecule_chembl_id.isin(contested)]
    pos_src = pos_src[~pos_src.molecule_chembl_id.isin(contested)]

    cohorts, bad_frames = {}, {}
    for name, src in (('specificity', neg_src), ('sensitivity', pos_src)):
        d = src.drop_duplicates('molecule_chembl_id')
        frame, bad = _standardize_frame(d, 'canonical_smiles')
        frame['expected_active'] = 0 if name == 'specificity' else 1
        cohorts[name] = frame.drop_duplicates('inchikey').reset_index(drop=True)
        bad_frames[name] = bad

    # ---- mechanism cohort ---------------------------------------------------------
    mech = mech[mech.target_chembl_id == target].copy()
    mech['action'] = mech.action_type.astype(str).str.upper().str.strip()
    mech = mech[mech.action.isin(['AGONIST', 'ANTAGONIST'])]
    # One molecule annotated both ways on the same target carries no usable exclusive label.
    nun = mech.groupby('molecule_chembl_id').action.nunique()
    mech = mech[mech.molecule_chembl_id.isin(nun[nun == 1].index)].drop_duplicates('molecule_chembl_id')
    report['mechanism_exclusive_molecules'] = int(len(mech))
    if len(mech):
        ids = sorted(mech.molecule_chembl_id.unique())
        structures = []
        for j in range(0, len(ids), 50):
            batch = ids[j:j + 50]
            url = (f'{BASE}/molecule.json?molecule_chembl_id__in=' + ','.join(batch)
                   + f'&limit=50&format=json')
            payload = _page(url, CACHE / f'molecule_{target}_{j:05}.json')
            for m in payload.get('molecules', []):
                st = m.get('molecule_structures') or {}
                structures.append({'molecule_chembl_id': m.get('molecule_chembl_id'),
                                   'canonical_smiles': st.get('canonical_smiles')})
        s = pd.DataFrame(structures)
        mech = mech.merge(s, on='molecule_chembl_id', how='left')
        frame, bad = _standardize_frame(mech, 'canonical_smiles', 'action')
        if len(frame):
            frame['label'] = frame.label.map({'AGONIST': 1, 'ANTAGONIST': 2})
            frame['expected_active'] = 1
        cohorts['moa'] = frame.drop_duplicates('inchikey').reset_index(drop=True)
        bad_frames['moa'] = bad
    else:
        cohorts['moa'] = pd.DataFrame(columns=['molecule_chembl_id', 'smiles', 'inchikey',
                                               'group', 'label', 'expected_active'])
        bad_frames['moa'] = pd.DataFrame(columns=['molecule_chembl_id', 'reason'])

    # ---- disjointness against the whole Tox21 curation ---------------------------
    curated = pd.read_csv(out / 'data/curation_all.csv', usecols=['inchikey'])
    seen = set(curated.inchikey.dropna())
    for name, frame in list(cohorts.items()):
        n0 = len(frame)
        frame = frame[~frame.inchikey.isin(seen)].reset_index(drop=True)
        frame['row_id'] = np.arange(len(frame))
        cohorts[name] = frame
        report[name] = {'candidates': int(n0), 'disjoint': int(len(frame)),
                        'removed_overlapping_identities': int(n0 - len(frame)),
                        'unparsable': int(len(bad_frames[name]))}
        frame.to_csv(dest / f'chembl_{name}.csv', index=False)
        bad_frames[name].to_csv(dest / f'chembl_{name}_rejected.csv', index=False)
        if name == 'moa' and len(frame):
            report[name]['label_counts'] = frame.label.value_counts().to_dict()

    report['interpretation'] = (
        'ChEMBL aggregates heterogeneous literature assays. These cohorts estimate external '
        'behaviour under distribution shift, not replication of the Tox21 VM7Luc4E2 endpoints. '
        'No cohort is used for training, hyperparameter selection, calibration or thresholding.')
    (dest / 'chembl_scope.json').write_text(json.dumps(report, indent=2))
    print('[chembl] ' + json.dumps({k: v.get('disjoint') for k, v in report.items()
                                    if isinstance(v, dict) and 'disjoint' in v}), flush=True)
    return cohorts, report

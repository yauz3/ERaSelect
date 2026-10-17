"""Tox21 ERa paired-endpoint acquisition, two-tier label curation and provenance.

Curation contract
-----------------
label 0 inactive        agonist==0 AND antagonist==0     both endpoints definitive, NEVER relaxed
label 1 agonist_only    agonist==1 AND antagonist==0     definitive
                        agonist==1 AND antagonist NaN    extended tier only, moa_confidence=provisional
label 2 antagonist_only antagonist==1 AND agonist==0     definitive
                        antagonist==1 AND agonist NaN    extended tier only, moa_confidence=provisional
label 3 dual_active     agonist==1 AND antagonist==1     definitive

Rationale for the extended tier: a definitive POSITIVE call on one endpoint paired with an
inconclusive call on the other is unambiguous evidence of ERa activity (Stage 1 gains a true
active with zero label noise) but only provisional evidence of exclusivity (Stage 2 may absorb
an unrecognised dual-active). The inactive class is never relaxed, because an inconclusive call
can never become a reliable negative. Both tiers are carried through every experiment so the
sensitivity of every claim to this decision is measurable rather than assumed.
"""
import argparse, hashlib, io, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.warning')
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
AGONIST_AID, ANTAGONIST_AID = 743079, 743091


def get(url, path, timeout=120):
    """Cached GET. Cache hit never touches the network, so a curated run is reproducible offline."""
    if path.exists():
        return path.read_bytes()
    last = None
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return r.content
        except requests.RequestException as e:
            last = e
            if attempt == 4:
                raise RuntimeError(
                    f'Network acquisition failed for {url} after 5 attempts ({e!r}). '
                    f'Expected cache file {path} is absent. Restore data/raw or enable network.'
                ) from e
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(str(last))


def acquire(download=True):
    RAW.mkdir(exist_ok=True, parents=True)
    for aid in (AGONIST_AID, ANTAGONIST_AID):
        descpath = RAW / f'{aid}_description.json'
        if not descpath.exists() and not download:
            raise FileNotFoundError(descpath)
        get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/JSON', descpath)
        desc = json.loads(descpath.read_text())['PC_AssayContainer'][0]['assay']['descr']
        name = desc['name'].lower()
        required = 'agonists' if aid == AGONIST_AID else 'antagonists'
        if required not in name or 'bg1' not in name:
            raise ValueError(f'Unexpected assay metadata for {aid}: {name}')
        final = RAW / f'{aid}.csv'
        if not final.exists():
            if not download:
                raise FileNotFoundError(final)
            # PubChem full-record API caps retrieval; enumerate SIDs from the concise view and chunk.
            concise = RAW / f'{aid}_concise.csv'
            cd = pd.read_csv(io.BytesIO(get(
                f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV', concise)))
            sids = sorted(cd.SID.dropna().astype(int).unique())
            chunks = [sids[j:j + 400] for j in range(0, len(sids), 400)]

            def fetch(item):
                j, ids = item
                p = RAW / f'{aid}_chunk_{j:03}.csv'
                content = get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV?sid='
                              + ','.join(map(str, ids)), p)
                q = pd.read_csv(io.BytesIO(content), low_memory=False)
                q = q[pd.to_numeric(q.PUBCHEM_SID, errors='coerce').notna()]
                time.sleep(.25)
                return q

            with ThreadPoolExecutor(max_workers=3) as pool:
                pieces = list(pool.map(fetch, enumerate(chunks)))
            pd.concat(pieces, ignore_index=True).to_csv(final, index=False)

    tables = []
    for aid in (AGONIST_AID, ANTAGONIST_AID):
        d = pd.read_csv(RAW / f'{aid}.csv', low_memory=False)
        needed = {'PUBCHEM_SID', 'PUBCHEM_ACTIVITY_OUTCOME', 'PUBCHEM_CID'}
        if not needed <= set(d):
            raise ValueError(f'Unknown PubChem schema for {aid}: {list(d)[:15]}')
        d = d[pd.to_numeric(d.PUBCHEM_SID, errors='coerce').notna()].copy()
        d.PUBCHEM_SID = pd.to_numeric(d.PUBCHEM_SID).astype(int)
        d.PUBCHEM_CID = pd.to_numeric(d.PUBCHEM_CID, errors='coerce')
        tables.append(d)

    cids = sorted(set(pd.concat(tables).PUBCHEM_CID.dropna().astype(int)))
    props = []
    for j in range(0, len(cids), 200):
        ids = cids[j:j + 200]
        p = RAW / f'structures_{j // 200:03}.json'
        if not p.exists() and not download:
            raise FileNotFoundError(p)
        payload = json.loads(get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'
                                 + ','.join(map(str, ids)) + '/property/IsomericSMILES/JSON', p))
        props.extend(payload['PropertyTable']['Properties'])
    p = pd.DataFrame(props)
    smiles_col = next((c for c in ['SMILES', 'IsomericSMILES', 'ConnectivitySMILES', 'CanonicalSMILES']
                       if c in p), None)
    if smiles_col is None:
        raise ValueError('PubChem SMILES property missing')
    p = p.rename(columns={smiles_col: 'smiles', 'CID': 'cid'})[['cid', 'smiles']]
    manifest = {f.name: {'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'bytes': f.stat().st_size}
                for f in sorted(RAW.iterdir()) if f.is_file()}
    return tables, p, manifest


def endpoint(d, prefix, quarantine_cytotoxic=True):
    """SID-level consensus. Conflicting replicate calls are quarantined, never majority-voted."""
    d = d.copy()
    d['outcome'] = d.PUBCHEM_ACTIVITY_OUTCOME.astype(str).str.strip().str.lower().map(
        {'active': 1, 'inactive': 0})
    audit = {'rows': int(len(d))}

    def consensus(s):
        return s.iloc[0] if s.notna().all() and s.nunique() == 1 else np.nan

    if prefix == 'antagonist':
        col = next((c for c in d if c.strip().lower() == 'activity summary'), None)
        if col is None:
            raise ValueError('Antagonist table lacks "Activity Summary": cannot audit viability confounding')
        d['summary'] = d[col].astype(str).str.lower()
        # A cytotoxicity-flagged well cannot serve as a reliable antagonist call in either direction.
        bad = d['summary'].str.contains('cytotox', na=False)
        audit['cytotoxicity_flagged_rows'] = int(bad.sum())
        d.loc[bad, 'outcome'] = np.nan
    q = d.groupby('PUBCHEM_SID').agg(cid=('PUBCHEM_CID', consensus),
                                     **{prefix: ('outcome', consensus)})
    audit['sids'] = int(len(q))
    audit['definitive_sids'] = int(q[prefix].notna().sum())
    return q, audit


def standardize(smiles):
    """Conservative standardization. Sanitize before scaffold perception (ring info must exist)."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        raise ValueError('invalid_smiles')
    m = rdMolStandardize.Cleanup(m)
    m = rdMolStandardize.FragmentParent(m)
    m = rdMolStandardize.Uncharger().uncharge(m)
    # Uncharger returns a molecule whose ring information is not initialised. MurckoScaffold
    # perception requires it; omitting this sanitize silently discarded ~22% of the library.
    Chem.SanitizeMol(m)
    if m.GetNumHeavyAtoms() < 2:
        raise ValueError('fewer_than_two_heavy_atoms')
    # No tautomer canonicalization: preserve the assayed chemical identity and stereochemistry.
    s = Chem.MolToSmiles(m, isomericSmiles=True)
    key = Chem.MolToInchiKey(m)
    if not key:
        raise ValueError('no_inchikey')
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    # Acyclic molecules intentionally share a single holdout group (conservative).
    return s, key, (scaffold or '__ACYCLIC__')


def assign_labels(agonist, antagonist):
    """Return (label, tier, moa_confidence) or (None, None, None) when unusable."""
    a_def, b_def = pd.notna(agonist), pd.notna(antagonist)
    a = int(agonist) if a_def else None
    b = int(antagonist) if b_def else None
    if a_def and b_def:
        return {(0, 0): (0, 'core', 'definitive'), (1, 0): (1, 'core', 'definitive'),
                (0, 1): (2, 'core', 'definitive'), (1, 1): (3, 'core', 'definitive')}[(a, b)]
    # Exactly one definitive POSITIVE call: activity is certain, exclusivity is not.
    if a_def and a == 1 and not b_def:
        return 1, 'extended', 'provisional'
    if b_def and b == 1 and not a_def:
        return 2, 'extended', 'provisional'
    # A single definitive NEGATIVE call proves nothing about the other endpoint.
    return None, None, None


LABEL_NAMES = {0: 'inactive', 1: 'agonist_only', 2: 'antagonist_only', 3: 'dual_active'}


def prepare(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    tables, structures, manifest = acquire(cfg.get('download', True))
    a, audit_a = endpoint(tables[0], 'agonist')
    b, audit_b = endpoint(tables[1], 'antagonist', cfg.get('quarantine_cytotoxic', True))
    joined = a.join(b, how='outer', lsuffix='_ago', rsuffix='_ant')
    joined['cid'] = joined.cid_ago.combine_first(joined.cid_ant)
    cid_conflict = (joined.cid_ago.notna() & joined.cid_ant.notna()
                    & joined.cid_ago.ne(joined.cid_ant))
    joined.loc[cid_conflict, ['agonist', 'antagonist']] = np.nan
    joined = joined.reset_index().merge(structures, on='cid', how='left')

    rows, audit = [], []
    for _, r in joined.iterrows():
        if pd.isna(r.smiles):
            audit.append({'sid': r.PUBCHEM_SID, 'stage': 'standardization',
                          'reason': 'no_structure_retrieved_for_cid'})
            continue
        try:
            s, key, group = standardize(r.smiles)
        except Exception as e:
            audit.append({'sid': r.PUBCHEM_SID, 'stage': 'standardization',
                          'reason': str(e).splitlines()[0][:120]})
            continue
        rows.append({'sid': int(r.PUBCHEM_SID), 'cid': r.cid, 'smiles': s, 'inchikey': key,
                     'group': group, 'agonist': r.agonist, 'antagonist': r.antagonist})
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError('Standardization produced no molecules')

    # Collapse to unique standardized chemical identities; replicates must agree.
    clean = []
    for key, d in raw.groupby('inchikey', sort=True):
        r = d.iloc[0].to_dict()
        r['sids'] = '|'.join(map(str, d.sid))
        r['replicates'] = int(len(d))
        for col in ('agonist', 'antagonist'):
            v = d[col].dropna().unique()
            r[col] = v[0] if len(v) == 1 and d[col].notna().all() else np.nan
            if d[col].dropna().nunique() > 1:
                audit.append({'sid': r['sids'], 'stage': 'identity_merge',
                              'reason': f'{col}_replicate_conflict'})
        clean.append(r)
    d = pd.DataFrame(clean)

    assigned = d.apply(lambda r: assign_labels(r.agonist, r.antagonist), axis=1, result_type='expand')
    assigned.columns = ['label', 'tier', 'moa_confidence']
    d = pd.concat([d.drop(columns=[c for c in assigned.columns if c in d]), assigned], axis=1)
    d['label_name'] = d.label.map(LABEL_NAMES).fillna('unusable')
    d.to_csv(out / 'curation_all.csv', index=False)
    pd.DataFrame(audit, columns=['sid', 'stage', 'reason']).to_csv(
        out / 'curation_exclusions.csv', index=False)

    tier = cfg.get('label_tier', 'extended')
    if tier not in ('core', 'extended'):
        raise ValueError(f'label_tier must be core or extended, got {tier!r}')
    eligible = d[d.label.notna()].copy()
    if tier == 'core':
        eligible = eligible[eligible.tier == 'core'].copy()
    eligible['label'] = eligible.label.astype(int)

    if cfg.get('max_rows') and len(eligible) > cfg['max_rows']:
        # Smoke sampling preserves real labels and is never used for manuscript claims.
        per = max(1, cfg['max_rows'] // max(1, eligible.label.nunique()))
        eligible = pd.concat(
            [q.sample(min(len(q), per), random_state=17) for _, q in eligible.groupby('label')],
            ignore_index=True)

    eligible = eligible.sort_values('inchikey').reset_index(drop=True)
    eligible['row_id'] = np.arange(len(eligible))
    if not {0, 1, 2} <= set(eligible.label):
        raise ValueError(f'Tier {tier} lacks one of the inactive/agonist/antagonist classes')

    eligible.to_csv(out / 'master.csv', index=False)

    # Curation waterfall: every molecule-count transition is auditable and plotted in analysis.
    waterfall = [
        {'stage': 'assay_rows_agonist', 'n': audit_a['rows']},
        {'stage': 'assay_rows_antagonist', 'n': audit_b['rows']},
        {'stage': 'antagonist_cytotoxicity_flagged_rows',
         'n': audit_b.get('cytotoxicity_flagged_rows', 0)},
        {'stage': 'paired_substances', 'n': int(len(joined))},
        {'stage': 'standardization_failed',
         'n': int(sum(1 for x in audit if x['stage'] == 'standardization'))},
        {'stage': 'unique_chemical_identities', 'n': int(len(d))},
        {'stage': 'replicate_label_conflicts',
         'n': int(sum(1 for x in audit if x['stage'] == 'identity_merge'))},
        {'stage': 'labelled_core', 'n': int((d.tier == 'core').sum())},
        {'stage': 'labelled_extended_provisional', 'n': int((d.tier == 'extended').sum())},
        {'stage': 'unusable_single_negative_or_inconclusive', 'n': int(d.label.isna().sum())},
        {'stage': f'eligible_tier_{tier}', 'n': int(len(eligible))},
    ]
    pd.DataFrame(waterfall).to_csv(out / 'curation_waterfall.csv', index=False)

    counts = {t: d[d.tier == t].label_name.value_counts().to_dict() for t in ('core', 'extended')}
    both = d[d.label.notna()]
    (out / 'data_manifest.json').write_text(json.dumps({
        'files': manifest,
        'endpoint_aids': {'agonist': AGONIST_AID, 'antagonist': ANTAGONIST_AID},
        'endpoint_audit': {'agonist': audit_a, 'antagonist': audit_b},
        'identity': 'standardized isomeric SMILES InChIKey',
        'curation_rules': 'replicates must agree; a single definitive negative is never a label; '
                          'inactive requires both endpoints definitive in every tier',
        'tier_selected': tier,
        'counts_by_tier': counts,
        'counts_union': both.label_name.value_counts().to_dict(),
        'eligible_counts': eligible.label_name.value_counts().to_dict(),
        'eligible_moa_confidence': eligible.moa_confidence.value_counts().to_dict(),
        'unusable': int(d.label.isna().sum()),
    }, indent=2))
    print(f'[data] tier={tier} counts={eligible.label_name.value_counts().to_dict()}', flush=True)
    print(f'[data] moa_confidence={eligible.moa_confidence.value_counts().to_dict()}', flush=True)
    return eligible


# ----------------------------------------------------------------------------- features

DESCRIPTOR_NAMES = [x[0] for x in Descriptors._descList]
_DESCRIPTOR_FNS = [x[1] for x in Descriptors._descList]


def morgan_matrix(smiles_iter, bits, radius=2, features=False):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if features else None)
    out = []
    for s in smiles_iter:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f'standardized SMILES failed to parse: {s}')
        arr = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
        out.append(arr)
    return np.asarray(out)


def descriptor_matrix(smiles_iter):
    rows = []
    for s in smiles_iter:
        m = Chem.MolFromSmiles(s)
        vals = []
        for fn in _DESCRIPTOR_FNS:
            try:
                vals.append(fn(m))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    arr = np.asarray(rows, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    # Some RDKit descriptors (Ipc above all) grow super-exponentially with molecule size and
    # exceed float32 range for a handful of molecules. Casting them to float32 yields +/-inf,
    # which SimpleImputer passes through (it only handles NaN) and StandardScaler then rejects
    # with "Input X contains infinity", killing the run at the first fold. Values outside
    # float32 range therefore become NaN here and are imputed like any other missing value.
    limit = np.finfo(np.float32).max
    overflow = np.abs(arr) > limit
    if overflow.any():
        cols = np.unique(np.where(overflow)[1])
        print(f'[data] {int(overflow.sum())} descriptor value(s) exceed float32 range in '
              f'{", ".join(DESCRIPTOR_NAMES[c] for c in cols)}; set to NaN for imputation',
              flush=True)
        arr[overflow] = np.nan
    return arr.astype(np.float32)


def features(d, cfg, out, write=True):
    out.mkdir(parents=True, exist_ok=True)
    bits = cfg['fingerprint_bits']
    ecfp4 = morgan_matrix(d.smiles, bits, 2, False)
    ecfp6 = morgan_matrix(d.smiles, bits, 3, False)
    fcfp4 = morgan_matrix(d.smiles, bits, 2, True)
    desc = descriptor_matrix(d.smiles)
    if write:
        pd.DataFrame(desc, columns=DESCRIPTOR_NAMES).to_csv(out / 'descriptors.csv', index=False)
        np.savez_compressed(out / 'features.npz', ecfp4=ecfp4, ecfp6=ecfp6, fcfp4=fcfp4,
                            descriptors=desc)
    X = {
        'ecfp4': ecfp4.astype(np.float32),
        'fcfp4': fcfp4.astype(np.float32),
        'descriptors': desc,
        'fusion': np.column_stack([ecfp4, desc]).astype(np.float32),
    }
    if cfg.get('pretrained'):
        try:
            emb = pretrained_embeddings(d.smiles.tolist(), cfg, out, write=write)
            X['pretrained_fusion'] = np.column_stack([X['fusion'], emb]).astype(np.float32)
        except Exception as e:
            # The pretrained encoder is an ablation arm, never the primary configuration, so a
            # failure to load it must not destroy a run that is otherwise complete. The arm is
            # dropped loudly and the reason is recorded.
            print('\n[data] WARNING: the pretrained encoder could not be loaded, so the '
                  "'pretrained_fusion' representation is UNAVAILABLE for this run.\n"
                  f'        reason: {type(e).__name__}: {str(e).strip().splitlines()[0][:200]}\n'
                  '        Every other representation is unaffected and the run continues.\n'
                  '        To silence this, set "pretrained": false in the config.\n',
                  flush=True)
            if write:
                (out / 'embedding_manifest.json').write_text(json.dumps({
                    'model': cfg.get('embedding_model'),
                    'revision_requested': cfg.get('embedding_revision', 'main'),
                    'status': 'unavailable',
                    'error': f'{type(e).__name__}: {str(e).strip()[:500]}',
                    'consequence': 'pretrained_fusion was not evaluated in this run',
                }, indent=2))
    return X, ecfp4


def pretrained_embeddings(smiles, cfg, out, write=True):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import model_info
    model = cfg['embedding_model']
    # Resolve whatever the config asks for to a concrete commit, and RECORD it. Pinning a
    # hand-written SHA is worse than useless: an early commit of a model repo may predate its
    # tokenizer files, in which case transformers falls back to converting a slow tokenizer,
    # finds no vocab file, and reports a confusing missing-protobuf error instead.
    revision = cfg.get('embedding_revision') or 'main'
    resolved = model_info(model, revision=revision).sha
    try:
        tok = AutoTokenizer.from_pretrained(model, revision=resolved, use_fast=True)
    except Exception as fast_error:
        try:
            tok = AutoTokenizer.from_pretrained(model, revision=resolved, use_fast=False)
        except Exception:
            raise RuntimeError(
                f'Neither the fast nor the slow tokenizer for {model} could be loaded at '
                f'revision {resolved}. If the config pins "embedding_revision" to a specific '
                f'commit, that commit may predate the tokenizer files; set it to "main". '
                f'Otherwise install the optional tokenizer dependencies '
                f'(pip install tokenizers protobuf sentencepiece). '
                f'Underlying fast-tokenizer error: {fast_error}') from fast_error
    enc = AutoModel.from_pretrained(model, revision=resolved).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc.to(device)
    chunks, truncated = [], 0
    bs = cfg['embedding_batch_size']
    for i in range(0, len(smiles), bs):
        ss = smiles[i:i + bs]
        truncated += sum(len(tok.encode(s)) > 512 for s in ss)
        batch = tok(ss, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            z = enc(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            chunks.append(((z * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    e = np.concatenate(chunks)
    if write:
        np.save(out / 'pretrained_embeddings.npy', e)
        (out / 'embedding_manifest.json').write_text(json.dumps({
            'model': model, 'revision': resolved, 'frozen': True,
            'pooling': 'attention-mask weighted mean over final hidden states',
            'truncated_smiles': int(truncated),
            'pretraining_overlap': 'UNKNOWN. The checkpoint was pretrained on public chemical '
                                   'libraries that may contain these molecules. Any representation '
                                   'using it cannot be described as externally validated.',
        }, indent=2))
    return e


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Curate only (no modelling).')
    ap.add_argument('--config', default='config/full.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dest = Path(args.out) if args.out else ROOT / 'results' / cfg['profile'] / 'data'
    prepare(cfg, dest)

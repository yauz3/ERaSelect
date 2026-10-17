"""Chemical-space description and the train/test dissimilarity evidence.

The Tanimoto figures are not decoration. A scaffold split is only worth its cost if it actually
places chemically dissimilar molecules in the held-out fold, and an external cohort only
validates something if it is chemically distant from the training set. Both are measured here
and reported as distributions, not single numbers.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import viz as V

BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0001)]
BAND_LABELS = ['< 0.3', '0.3 – 0.5', '0.5 – 0.7', '> 0.7']


def band_of(sim):
    s = np.asarray(sim, dtype=float)
    out = np.full(len(s), -1, dtype=int)
    for i, (lo, hi) in enumerate(BANDS):
        out[(s >= lo) & (s < hi)] = i
    return out


def tanimoto_matrix(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nearest(a, b, exclude_self=False):
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.nan)
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), 256):
        m = tanimoto_matrix(a[i:i + 256], b)
        if exclude_self:
            for r in range(m.shape[0]):
                m[r, i + r] = -1
        out[i:i + 256] = m.max(1)
    return out


# ------------------------------------------------------------------ descriptive figures

def curation_waterfall(out, table):
    w = pd.read_csv(out / 'data/curation_waterfall.csv')
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(w) + 1.1))
    pretty = [s.replace('_', ' ') for s in w.stage]
    V.rounded_bars(ax, np.arange(len(w)), w.n.to_numpy(), V.SERIES[0], width=0.62,
                   horizontal=True)
    for i, n in enumerate(w.n):
        ax.text(n + max(w.n) * 0.012, i, f'{int(n):,}', va='center',
                fontsize=plt.rcParams['font.size'] - 1, color=V.INK_2)
    ax.set_yticks(np.arange(len(w)))
    ax.set_yticklabels(pretty)
    ax.invert_yaxis()
    ax.set_xlim(0, max(w.n) * 1.16)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Molecules')
    V.title(ax, 'Curation waterfall',
            'Every count transition from raw assay rows to the modelling set')
    table(w, 'curation_waterfall')
    return V.save(fig, '01_curation_waterfall', out,
                  'Molecule flow through curation. Counts are auditable in '
                  'tables/curation_waterfall.csv.')


def class_distribution(out, master, table):
    allc = pd.read_csv(out / 'data/curation_all.csv', low_memory=False)
    allc = allc[allc.label.notna()]
    rows = []
    for tier in ('core', 'extended'):
        sub = allc if tier == 'extended' else allc[allc.tier == 'core']
        for c in (0, 1, 2, 3):
            rows.append({'tier': tier, 'klass': c,
                         'label_key': ['inactive', 'agonist_only', 'antagonist_only',
                                       'dual_active'][c],
                         'label_name': V.CLASS_LABEL[c],
                         'n': int((sub.label == c).sum())})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    w = 0.34
    for k, (tier, hatch) in enumerate((('core', None), ('extended', '///'))):
        sub = d[d.tier == tier].sort_values('klass')
        pos = sub.klass.to_numpy() + (k - 0.5) * (w + 0.03)
        for p, n, c in zip(pos, sub.n, sub.klass):
            ax.bar(p, n, width=w, color=V.CLASS_COLOR[c], linewidth=0,
                   hatch=hatch, edgecolor=V.SURFACE,
                   label=tier if c == 0 else None)
            ax.text(p, n * 1.06, f'{int(n):,}', ha='center', va='bottom',
                    fontsize=plt.rcParams['font.size'] - 2, color=V.INK_2)
    ax.set_yscale('log')
    ax.set_xticks(range(4))
    ax.set_xticklabels([V.CLASS_LABEL[c] for c in range(4)])
    ax.set_ylabel('Molecules (log scale)')
    ax.legend(title='Label tier', loc='upper right')
    V.title(ax, 'Class sizes by label-confidence tier',
            'Hatched bars add provisional-exclusivity molecules; the inactive class is never relaxed')
    table(d, 'class_distribution_by_tier')
    return V.save(fig, '02_class_distribution', out,
                  'Class sizes under the core and extended label tiers. The extended tier '
                  'recovers the antagonist-only minority class; inactive counts are identical '
                  'because that label is never relaxed.')


def descriptor_heatmap(out, table, max_desc=40):
    path = out / 'data/descriptors.csv'
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d.loc[:, d.notna().mean() > 0.9]
    d = d.loc[:, d.std(numeric_only=True) > 0]
    if d.shape[1] < 3:
        return None
    var = d.var().sort_values(ascending=False)
    cols = list(var.index[:max_desc])
    corr = d[cols].corr(method='spearman')
    order = PCA(n_components=1, random_state=0).fit_transform(
        np.nan_to_num(corr.to_numpy())).ravel().argsort()
    corr = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(corr.to_numpy(), cmap=V.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=5.6)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label('Spearman correlation', color=V.INK_2)
    cb.outline.set_visible(False)
    V.title(ax, 'Descriptor redundancy',
            f'{len(cols)} highest-variance RDKit descriptors, PCA-ordered')
    table(corr.reset_index().rename(columns={'index': 'descriptor'}), 'descriptor_spearman')
    return V.save(fig, '04_descriptor_heatmap', out,
                  'Spearman correlation among the highest-variance descriptors. Dense blocks '
                  'motivate the fused representation rather than descriptors alone.')


def _embed_features(out, master):
    desc = pd.read_csv(out / 'data/descriptors.csv')
    npz = np.load(out / 'data/features.npz')
    fp = npz['ecfp4'].astype(np.float32)
    keep = desc.columns[(desc.notna().mean() > 0.9) & (desc.std(numeric_only=True) > 0)]
    dd = SimpleImputer(strategy='median').fit_transform(desc[keep])
    dd = StandardScaler().fit_transform(dd)
    return fp, dd


def pca_figures(out, master, table):
    fp, dd = _embed_features(out, master)
    y = master.label.to_numpy()
    p = PCA(n_components=4, random_state=0)
    z = p.fit_transform(np.column_stack([fp, dd]))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    V.scatter_classes(ax, z, y)
    ax.set_xlabel(f'PC1 ({p.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({p.explained_variance_ratio_[1]*100:.1f}% variance)')
    V.title(ax, 'Chemical space by functional class',
            'PCA of fused ECFP4 and standardized descriptors')
    table(pd.DataFrame({'row_id': master.row_id, 'label': y, 'label_name': master.label_name,
                        'pc1': z[:, 0], 'pc2': z[:, 1], 'pc3': z[:, 2], 'pc4': z[:, 3]}),
          'pca_coordinates')
    name = V.save(fig, '05_pca_classes', out,
                  'PCA of the fused representation coloured and shaped by functional class. '
                  'Overlap between agonist-only and antagonist-only molecules is the reason a '
                  'mode-of-action stage is hard and a support verifier is informative.')
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ev = p.explained_variance_ratio_[:4] * 100
    V.rounded_bars(ax, np.arange(4), ev, V.SERIES[0], width=0.5)
    for i, v in enumerate(ev):
        ax.text(i, v * 1.03, f'{v:.1f}%', ha='center', va='bottom',
                fontsize=plt.rcParams['font.size'] - 1, color=V.INK_2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'PC{i+1}' for i in range(4)])
    ax.set_ylabel('Variance explained (%)')
    ax.set_ylim(0, max(ev) * 1.2)
    V.title(ax, 'Variance captured by the leading components')
    V.save(fig, '05b_pca_scree', out, 'Scree plot for the fused representation.')
    return name


def tsne_figures(out, master, table, perplexities=(15, 30, 50)):
    fp, dd = _embed_features(out, master)
    y = master.label.to_numpy()
    base = PCA(n_components=min(50, fp.shape[1]), random_state=0).fit_transform(
        np.column_stack([fp, dd]))
    names = []
    fig, axes = plt.subplots(1, len(perplexities), figsize=(4.6 * len(perplexities), 4.6))
    axes = np.atleast_1d(axes)
    for ax, per in zip(axes, perplexities):
        if len(base) <= per * 3:
            V.empty(ax, f'n={len(base)} too small\nfor perplexity {per}')
            continue
        z = TSNE(n_components=2, perplexity=per, init='pca', random_state=0,
                 max_iter=1000).fit_transform(base)
        V.scatter_classes(ax, z, y, size=13, alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        V.title(ax, f'perplexity {per}')
        table(pd.DataFrame({'row_id': master.row_id, 'label': y,
                            'tsne1': z[:, 0], 'tsne2': z[:, 1]}), f'tsne_p{per}')
        if ax is not axes[0]:
            ax.get_legend().remove()
    fig.suptitle('t-SNE of the fused representation at three perplexities', x=0.02, ha='left',
                 y=1.03)
    names.append(V.save(fig, '06_tsne_perplexities', out,
                        't-SNE embeddings at three perplexities. Structure that survives all '
                        'three is not a perplexity artefact.'))
    return names


def radviz_figure(out, master, table, n_desc=8):
    desc = pd.read_csv(out / 'data/descriptors.csv')
    keep = [c for c in ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                        'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'HeavyAtomCount',
                        'NumAromaticRings'] if c in desc]
    keep = keep[:n_desc]
    if len(keep) < 3:
        return None
    d = desc[keep].copy()
    d = d.fillna(d.median())
    rng = d.max() - d.min()
    norm = ((d - d.min()) / rng.replace(0, 1)).to_numpy()
    k = norm.shape[1]
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    anchors = np.column_stack([np.cos(angles), np.sin(angles)])
    s = norm.sum(1, keepdims=True)
    xy = (norm @ anchors) / np.clip(s, 1e-9, None)
    y = master.label.to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    circle = plt.Circle((0, 0), 1, fill=False, color=V.GRID, linewidth=0.9)
    ax.add_patch(circle)
    V.scatter_classes(ax, xy, y, size=14, alpha=0.7)
    for (ax_x, ax_y), lab in zip(anchors, keep):
        ax.plot([0, ax_x], [0, ax_y], color=V.GRID, linewidth=0.8, zorder=1)
        ax.text(ax_x * 1.08, ax_y * 1.08, lab, ha='center', va='center',
                fontsize=plt.rcParams['font.size'] - 2, color=V.INK_3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    V.title(ax, 'RadViz projection onto interpretable descriptors')
    table(pd.DataFrame({'row_id': master.row_id, 'label': y, 'x': xy[:, 0], 'y': xy[:, 1]}),
          'radviz_coordinates')
    return V.save(fig, '08_radviz', out,
                  'RadViz projection onto eight interpretable physicochemical descriptors, '
                  'with each class additionally distinguished by marker shape.')


def tanimoto_heatmap(out, master, table, cap=420):
    npz = np.load(out / 'data/features.npz')
    fp = npz['ecfp4'].astype(np.float32)
    y = master.label.to_numpy()
    idx = []
    rng = np.random.default_rng(3)
    per = max(20, cap // 4)
    for c in (0, 1, 2, 3):
        pool = np.where(y == c)[0]
        if len(pool):
            idx.extend(rng.choice(pool, min(len(pool), per), replace=False))
    idx = np.array(sorted(idx, key=lambda i: (y[i], i)))
    if len(idx) < 10:
        return None
    m = tanimoto_matrix(fp[idx], fp[idx])
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(m, cmap=V.SEQUENTIAL, vmin=0, vmax=1, interpolation='nearest')
    bounds, start = [], 0
    for c in (0, 1, 2, 3):
        n = int((y[idx] == c).sum())
        if n:
            bounds.append((start, start + n, c))
            start += n
    for lo, hi, c in bounds:
        for pos in (lo, hi):
            ax.axhline(pos - .5, color=V.SURFACE, linewidth=1.4)
            ax.axvline(pos - .5, color=V.SURFACE, linewidth=1.4)
        ax.text(-len(idx) * 0.035, (lo + hi) / 2, V.CLASS_LABEL[c], rotation=90,
                va='center', ha='center', fontsize=plt.rcParams['font.size'] - 2,
                color=V.CLASS_COLOR[c])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label('Tanimoto similarity (ECFP4)', color=V.INK_2)
    cb.outline.set_visible(False)
    V.title(ax, 'Pairwise similarity within and between classes',
            f'Stratified sample of {len(idx)} molecules, blocked by class')
    stats = []
    for lo, hi, c in bounds:
        for lo2, hi2, c2 in bounds:
            block = m[lo:hi, lo2:hi2]
            if c == c2:
                tri = block[np.triu_indices_from(block, k=1)]
            else:
                tri = block.ravel()
            if len(tri):
                stats.append({'class_a': V.CLASS_LABEL[c], 'class_b': V.CLASS_LABEL[c2],
                              'mean': float(tri.mean()), 'median': float(np.median(tri)),
                              'p95': float(np.percentile(tri, 95))})
    table(pd.DataFrame(stats), 'tanimoto_block_statistics')
    return V.save(fig, '09_tanimoto_heatmap', out,
                  'Within- and between-class ECFP4 Tanimoto similarity. Agonist and antagonist '
                  'blocks are not separable by similarity alone.')


def split_dissimilarity(out, metrics, table):
    """Does the scaffold split actually produce dissimilar held-out molecules? Measured, per fold."""
    if 'median_nn_tanimoto' not in metrics:
        return None
    d = (metrics.drop_duplicates(['split', 'seed', 'fold', 'representation', 'model'])
         [['split', 'seed', 'fold', 'median_nn_tanimoto']]
         .drop_duplicates(['split', 'seed', 'fold']))
    groups = [d.loc[d.split == s, 'median_nn_tanimoto'].to_numpy() for s in ('scaffold', 'random')
              if (d.split == s).any()]
    labels = [s for s in ('scaffold', 'random') if (d.split == s).any()]
    if not groups:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={'width_ratios': [1, 1.45]})
    V.boxplot(axes[0], groups, [f'{l.capitalize()} split' for l in labels],
              colors=[V.SERIES[0], V.SERIES[1]][:len(groups)])
    axes[0].set_ylabel('Median nearest-train Tanimoto')
    V.title(axes[0], 'Held-out similarity by split design', 'One point per outer fold')

    pred = _primary_predictions(out)
    if pred is not None and 'nn_tanimoto_train' in pred:
        for i, s in enumerate(labels):
            sub = pred[pred.split == s]
            if not len(sub):
                continue
            axes[1].hist(sub.nn_tanimoto_train.dropna(), bins=40, range=(0, 1), density=True,
                         histtype='stepfilled', color=V.SERIES[i], alpha=0.16, linewidth=0)
            axes[1].hist(sub.nn_tanimoto_train.dropna(), bins=40, range=(0, 1), density=True,
                         histtype='step', color=V.SERIES[i], linewidth=2.0,
                         label=f'{s.capitalize()} (median {sub.nn_tanimoto_train.median():.2f})')
        axes[1].set_xlabel('Nearest-train Tanimoto of each held-out molecule')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        V.reference_line(axes[1], 0.3, 'x', 'applicability concern')
        V.title(axes[1], 'Per-molecule distribution', 'Primary configuration')
    else:
        V.empty(axes[1], 'primary per-molecule predictions unavailable')
    table(d, 'split_dissimilarity_by_fold')
    return V.save(fig, '10_split_dissimilarity', out,
                  'Nearest-neighbour Tanimoto similarity of held-out molecules to their own '
                  'training partition. A scaffold split shifts the distribution downward '
                  'relative to a random split; this is the evidence that the primary split '
                  'design is doing work rather than being asserted.')


def external_dissimilarity(out, table):
    path = out / 'external/external_similarity.csv'
    if not path.exists():
        return None
    s = pd.read_csv(path)
    if not len(s):
        return None
    cohorts = sorted(s.cohort.unique())
    groups = [s.loc[s.cohort == c, 'nn_tanimoto_fit'].dropna().to_numpy() for c in cohorts]
    fig, ax = plt.subplots(figsize=(max(6.2, 1.9 * len(cohorts) + 2.6), 3.9))
    V.boxplot(ax, groups, [c.replace('_', '\n') for c in cohorts],
              colors=[V.SERIES[(i + 2) % len(V.SERIES)] for i in range(len(cohorts))],
              showpoints=False)
    V.reference_line(ax, 0.3, 'y', 'applicability concern')
    ax.set_ylabel('Nearest-train Tanimoto')
    ax.set_ylim(0, 1)
    V.title(ax, 'Chemical distance of each external cohort from training',
            'An external set that is chemically identical to training validates nothing')
    summary = (s.groupby('cohort').nn_tanimoto_fit
               .agg(n='size', mean='mean', median='median',
                    q05=lambda v: v.quantile(.05), q95=lambda v: v.quantile(.95),
                    frac_below_0_3=lambda v: float((v < .3).mean())).reset_index())
    table(summary, 'external_cohort_similarity')
    return V.save(fig, '11_external_dissimilarity', out,
                  'Nearest-training-neighbour similarity for each external cohort.')


def similarity_band_performance(out, table):
    pred = _primary_predictions(out)
    if pred is None or 'nn_tanimoto_train' not in pred:
        return None
    pred = pred[pred.split == 'scaffold'].copy()
    if not len(pred):
        return None
    pred['band'] = band_of(pred.nn_tanimoto_train)
    rows = []
    for b in range(len(BANDS)):
        sub = pred[(pred.band == b) & (pred.label < 3)]
        if not len(sub):
            continue
        rows.append({'band': BAND_LABELS[b], 'n': int(len(sub)),
                     'accuracy': float((sub.prediction == sub.label).mean()),
                     'coverage_full_knn': float(sub.accept_full_knn.mean()),
                     'accepted_accuracy': float(
                         (sub.loc[sub.accept_full_knn, 'prediction']
                          == sub.loc[sub.accept_full_knn, 'label']).mean())
                     if sub.accept_full_knn.any() else np.nan,
                     'n_antagonist': int((sub.label == 2).sum())})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(d))
    ax.plot(x, d.accuracy, marker='o', color=V.SERIES[0], label='All predictions',
            markeredgecolor=V.SURFACE, markeredgewidth=1.6)
    ax.plot(x, d.accepted_accuracy, marker='s', color=V.SERIES[1],
            label='Accepted subset (full pipeline)', markeredgecolor=V.SURFACE,
            markeredgewidth=1.6)
    for i, (a, n) in enumerate(zip(d.accuracy, d.n)):
        ax.text(i, a + 0.025, f'n={n:,}', ha='center', fontsize=plt.rcParams['font.size'] - 2,
                color=V.INK_3)
    ax.set_xticks(x)
    ax.set_xticklabels(d.band)
    ax.set_xlabel('Nearest-train Tanimoto band')
    ax.set_ylabel('Three-class accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    V.title(ax, 'Performance decays with chemical distance',
            'Scaffold folds, primary configuration')
    table(d, 'similarity_band_performance')
    return V.save(fig, '23_similarity_band_performance', out,
                  'Accuracy stratified by nearest-training-neighbour similarity, with and '
                  'without selective acceptance.')


def hard_negatives(out, table, threshold=0.6):
    pred = _primary_predictions(out)
    if pred is None or 'nn_tanimoto_active' not in pred:
        return None
    pred = pred[pred.split == 'scaffold']
    hard = pred[(pred.label == 0) & (pred.nn_tanimoto_active >= threshold)]
    easy = pred[(pred.label == 0) & (pred.nn_tanimoto_active < threshold)]
    if not len(hard):
        return None
    rows = [{'subset': f'Hard negatives (Tanimoto to actives >= {threshold})', 'n': len(hard),
             'false_positive_rate': float((hard.prediction > 0).mean()),
             'coverage': float(hard.accept_full_knn.mean())},
            {'subset': f'Other inactives (< {threshold})', 'n': len(easy),
             'false_positive_rate': float((easy.prediction > 0).mean()),
             'coverage': float(easy.accept_full_knn.mean())}]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    V.rounded_bars(ax, np.arange(len(d)), d.false_positive_rate.to_numpy(), V.SERIES[1],
                   width=0.5)
    for i, (v, n) in enumerate(zip(d.false_positive_rate, d.n)):
        ax.text(i, v + 0.012, f'{v:.3f}  (n={n:,})', ha='center', va='bottom',
                fontsize=plt.rcParams['font.size'] - 1, color=V.INK_2)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(['Hard negatives', 'Other inactives'])
    ax.set_ylabel('False-positive rate at the gate')
    ax.set_ylim(0, max(d.false_positive_rate) * 1.35 + 0.02)
    V.title(ax, 'Inactives that look like actives',
            'Structurally similar negatives are the failure mode screening cares about')
    table(d, 'hard_negative_challenge')
    return V.save(fig, '28_hard_negatives', out,
                  'Gate false-positive rate on inactive molecules with high Tanimoto '
                  'similarity to training actives, versus the remaining inactives.')


def scaffold_sizes(out, master, table):
    sizes = master.group.value_counts()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.hist(sizes.to_numpy(), bins=np.arange(1, min(40, sizes.max() + 2)), color=V.SERIES[0],
            linewidth=0)
    ax.set_yscale('log')
    ax.set_xlabel('Molecules sharing a Bemis–Murcko scaffold')
    ax.set_ylabel('Scaffold groups (log scale)')
    V.title(ax, 'Scaffold group sizes',
            f'{len(sizes):,} groups; largest holds {int(sizes.max()):,} molecules')
    table(sizes.rename('n_molecules').reset_index().rename(columns={'index': 'scaffold'}),
          'scaffold_sizes')
    return V.save(fig, '29_scaffold_sizes', out,
                  'Distribution of scaffold group sizes. Group sizes bound how finely the '
                  'outer folds and calibration partitions can be split.')


# ------------------------------------------------------------------ helper

_PRIMARY_CACHE = {}


def _primary_predictions(out):
    if out in _PRIMARY_CACHE:
        return _PRIMARY_CACHE[out]
    files = sorted((out / 'folds').glob('*/predictions.csv')) if (out / 'folds').exists() else []
    d = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None)
    _PRIMARY_CACHE[out] = d
    return d


def run_all(out, master, metrics, table):
    made = []
    for fn, args in ((curation_waterfall, (out, table)),
                     (class_distribution, (out, master, table)),
                     (descriptor_heatmap, (out, table)),
                     (pca_figures, (out, master, table)),
                     (tsne_figures, (out, master, table)),
                     (radviz_figure, (out, master, table)),
                     (tanimoto_heatmap, (out, master, table)),
                     (split_dissimilarity, (out, metrics, table)),
                     (external_dissimilarity, (out, table)),
                     (similarity_band_performance, (out, table)),
                     (hard_negatives, (out, table)),
                     (scaffold_sizes, (out, master, table))):
        try:
            r = fn(*args)
            if r:
                made.extend(r if isinstance(r, list) else [r])
        except Exception as e:
            print(f'[chemspace] {fn.__name__} skipped: {e!r}', flush=True)
    return made

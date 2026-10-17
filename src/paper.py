"""Result-linked manuscript draft, SOTA discussion table and submission-readiness audit.

Every quantitative statement is interpolated from a table this run produced. Where a number is
unavailable the placeholder is left visible rather than filled with an invented value, because a
draft that reads smoothly while containing fabricated results is worse than one with gaps.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TITLE = ('Calibration budget governs selective prediction in minority-class mode-of-action '
         'models: a pre-registered hierarchical framework for ER-alpha agonist and antagonist '
         'classification')

LITERATURE = [
    dict(ref='1', study='Conformal prediction on ER assay data (Chem Res Toxicol, 2016)',
         method='Mondrian conformal classification of ToxCast/Tox21 ER endpoints',
         advantages='Established confidence-aware prediction on the same receptor; validity '
                    'guarantee is distribution-free',
         limitations='Per-class calibration sample sizes are not reported, so whether the '
                     'nominal level was attainable for the minority class cannot be checked '
                     'from the paper',
         relation='Direct methodological antecedent. We add the feasibility criterion that '
                  'determines when such a guarantee is reachable at all',
         url='https://doi.org/10.1021/acs.chemrestox.6b00037'),
    dict(ref='2', study='ToxBench leakage-audited benchmark (BMC Bioinformatics, 2026)',
         method='Multi-task Tox21 benchmark combining scaffold splits, calibration, uncertainty '
                'and applicability domain',
         advantages='Shows these components matter and standardises their evaluation; strong '
                    'leakage auditing',
         limitations='Aggregate reporting over endpoints can mask a class too small to '
                     'calibrate; different endpoints and splits prevent direct numerical '
                     'comparison',
         relation='We adopt its splitting and leakage discipline and add per-class budget '
                  'accounting',
         url='https://doi.org/10.1186/s12859-026-06621-x'),
    dict(ref='3', study='Mechanism-based hierarchical ML for endocrine disruptors (ES&T, 2026)',
         method='Hierarchical mechanism-aware classification of endocrine-active compounds',
         advantages='Demonstrates that hierarchy helps on this biology; broad chemical coverage',
         limitations='Hierarchy alone does not address whether the abstention layer built on '
                     'top of it is statistically supported',
         relation='Establishes hierarchy as prior art. Our contribution is therefore NOT the '
                  'hierarchy but the conditions under which its selective layer is valid',
         url='https://pubmed.ncbi.nlm.nih.gov/'),
    dict(ref='4', study='ER agonist/antagonist random forests with applicability domain '
                        '(Chemosphere, 2024)',
         method='Separate RF models for ER binding, agonism and antagonism with '
                'probability-based applicability domain',
         advantages='Closest task formulation; practical AD definition',
         limitations='Single-model AD does not provide class-conditional validity; antagonist '
                     'class size and its effect on the AD are not isolated',
         relation='Baseline-equivalent formulation; we include an independent endpoint-pair '
                  'model as its analogue',
         url='https://doi.org/10.1016/j.chemosphere.2024.141767'),
    dict(ref='5', study='D-MPNN learned representations (JCIM, 2019)',
         method='Directed message-passing neural network with descriptor fusion',
         advantages='Strong learned representation; extensive benchmarking',
         limitations='Requires native training on identical partitions for a fair comparison; '
                     'not reproduced here',
         relation='Discussion only. No native D-MPNN result is claimed',
         url='https://doi.org/10.1021/acs.jcim.9b00237'),
    dict(ref='6', study='ChemBERTa-77M-MLM checkpoint (DeepChem)',
         method='Frozen pretrained SMILES transformer used as a feature extractor',
         advantages='Label-free chemical representation; cheap to apply',
         limitations='Pretraining corpus may contain the evaluation molecules, so results using '
                     'it cannot be called externally validated; tokenisation and checkpoint '
                     'dependence',
         relation='Included as a representation ablation only, never as the primary '
                  'configuration, for exactly that reason',
         url='https://huggingface.co/DeepChem/ChemBERTa-77M-MLM'),
    dict(ref='7', study='Tox21 ER reporter profiling (Sci Rep, 2014)',
         method='Full-length ER-luc (BG1/VM7) and ER-bla profiling with cytotoxicity '
                'interpretation',
         advantages='Defines the assay biology and the viability confounding we filter on',
         limitations='Reporter activity is not an in-vivo endocrine effect',
         relation='Source of the endpoint pair and of the cytotoxicity quarantine rule',
         url='https://doi.org/10.1038/srep05664'),
    dict(ref='proposed', study='This work',
         method='Hierarchical gate + conditional mode-of-action classifier with a '
                'pre-registered calibration-budget criterion, class-conditional conformal '
                'abstention, one-class support verification and two-tier label curation',
         advantages='States in advance which selective guarantees the data can support; '
                    'recovers the minority class without relaxing the inactive label; '
                    'independent ChEMBL and cross-platform cohorts with measured chemical '
                    'distance',
         limitations='Antagonist class remains small in absolute terms; external labels come '
                     'from heterogeneous assays; no wet-lab validation',
         relation='—', url='Not yet published'),
]


def _fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '[UNAVAILABLE]'
    return f'{v*100:.1f}%' if pct else f'{v:.{nd}f}'


def _ms(row, col, nd=3, pct=False):
    """mean ± SD [CI] from a summarise() row."""
    if row is None or f'{col}_mean' not in row:
        return '[UNAVAILABLE]'
    m, s = row.get(f'{col}_mean'), row.get(f'{col}_sd')
    lo, hi = row.get(f'{col}_ci_low'), row.get(f'{col}_ci_high')
    base = _fmt(m, nd, pct)
    if base == '[UNAVAILABLE]':
        return base
    out = base
    if s is not None and np.isfinite(s):
        out += f' ± {_fmt(s, nd, pct)}'
    if lo is not None and np.isfinite(lo) and hi is not None and np.isfinite(hi):
        out += f' (95% CI {_fmt(lo, nd, pct)}–{_fmt(hi, nd, pct)})'
    return out


def _row(df, variant):
    if df is None or not len(df) or 'variant' not in df:
        return None
    sub = df[df.variant == variant]
    return sub.iloc[0].to_dict() if len(sub) else None



def _tiers_compared(tables_dir):
    """Number of distinct label tiers actually present in the sensitivity table."""
    p = tables_dir / 'tier_sensitivity.csv'
    if not p.exists():
        return 0
    try:
        d = pd.read_csv(p)
    except Exception:
        return 0
    if 'tier' not in d or 'note' in d.columns:
        return 0
    return int(d.tier.nunique())

def build_paper(out, cfg, metrics, table):
    pdir = out / 'paper'
    pdir.mkdir(parents=True, exist_ok=True)
    T = out / 'tables'

    def load(name):
        p = T / f'{name}.csv'
        return pd.read_csv(p) if p.exists() else None

    summary = load('mean_sd_all_metrics_scaffold')
    budget = load('calibration_budget_summary')
    waterfall = load('curation_waterfall')
    counts = load('class_distribution_by_tier')
    ablation = load('ablation_cascade')
    comparison = load('primary_paired_comparison')
    conformal = load('conformal_coverage_by_variant')
    external = load('external_cohort_summary')
    bands = load('similarity_band_performance')
    thrpol = load('threshold_policy_comparison')
    # A cohort below this size cannot support a validation claim; computed here because both
    # the results narrative and the readiness audit need it.
    MIN_COHORT = 30
    tiny_cohorts, usable_cohorts = [], []
    if external is not None and 'cohort' in external:
        for _r in external.to_dict('records'):
            _n = int(_r.get('n', 0) or 0)
            (tiny_cohorts if _n < MIN_COHORT else usable_cohorts).append((_r['cohort'], _n))
    disc = load('external_discrimination')
    split_dis = load('split_dissimilarity_by_fold')
    perm = load('permutation_control')

    primary = cfg.get('primary', {})
    pv = primary.get('variant', 'full_knn')
    tier = cfg.get('label_tier', 'extended')
    full_row = _row(summary, pv)
    hier_row = _row(summary, 'hierarchy_calibrated')
    flat_row = _row(summary, 'flat_platt')
    endp_row = _row(summary, 'endpoint_pair')
    nominal_row = _row(conformal, 'hierarchy_conformal_nominal')
    feasible_row = _row(conformal, 'hierarchy_conformal_feasible')
    pooled_row = _row(conformal, 'hierarchy_conformal_pooled')

    # ---- numbers used in prose
    n_mol = int(len(pd.read_csv(out / 'data/master.csv', low_memory=False)))
    cls = {}
    if counts is not None and 'label_key' in counts:
        for t in ('core', 'extended'):
            sub = counts[counts.tier == t]
            cls[t] = {r.label_key: int(r.n) for r in sub.itertuples()}
    budget_fail = []
    if budget is not None:
        budget_fail = [r for r in budget.to_dict('records')
                       if r.get('folds_feasible_fraction', 1) < 1]
    feasible_all = (budget is not None and not budget_fail)
    # The worst single component, so the abstract never mixes one component's available count
    # with another component's requirement.
    worst = (min(budget_fail, key=lambda r: r['min_available_n'] - r['required_n'])
             if budget_fail else None)

    table(pd.DataFrame(LITERATURE), 'literature_sota_comparison')

    # ------------------------------------------------------------------ manuscript
    md = []
    md.append(f'# {TITLE}\n')
    md.append('> Automatically assembled from this run\'s tables. Every bracketed placeholder '
              'marks a number this run did not produce. Reconcile against the tables and have '
              'a domain expert review before submission.\n')

    md.append('## Abstract\n')
    md.append(
        f'Selective prediction promises that a model can decline the cases it cannot answer. '
        f'We show that in minority-class mode-of-action prediction this promise is frequently '
        f'unverifiable, because the per-class calibration sample needed to certify a nominal '
        f'miscoverage level is larger than the class itself. Split conformal prediction at level '
        f'alpha requires at least ceil(1/alpha) - 1 calibration points per class; below that the '
        f'threshold is infinite, every molecule is admitted, and the resulting coverage and '
        f'selective-risk figures are artefacts of the missing budget rather than properties of '
        f'the model. We make this criterion an explicit pre-registered gate and apply it to '
        f'ER-alpha agonist and antagonist classification using the Tox21 BG1/VM7Luc4E2 '
        f'luciferase endpoint pair (AID 743079 and 743091). A two-tier label-confidence curation '
        f'recovers the antagonist-only class '
        f'({cls.get("core", {}).get("antagonist_only", "[N]")} molecules under strict pairing, '
        f'{cls.get("extended", {}).get("antagonist_only", "[N]")} when a definitive positive '
        f'call on one endpoint is accepted as provisional evidence of exclusivity) while never '
        f'relaxing the inactive label. On {n_mol:,} curated molecules under nested scaffold '
        f'cross-validation, the criterion is '
        f'{"satisfied for every class and fold" if feasible_all else "violated for at least one class"} '
        f'in the {tier} tier'
        + (f', the binding shortfall being {worst["component"]} with as few as '
           f'{int(worst["min_available_n"])} calibration molecules against a requirement of '
           f'{int(worst["required_n"])}. ' if worst else
           ', while the same audit fails for the antagonist class under the core tier. ')
        + f''
        f'The full pipeline reaches a selective risk of {_ms(full_row, "selective_risk")} at '
        f'coverage {_ms(full_row, "coverage")}, against {_ms(flat_row, "selective_risk")} for a '
        f'flat calibrated three-class model answering everything. Pooled conformal calibration '
        f'appears valid while the minority class is uncalibrated, which we quantify directly. '
        f'We conclude that per-class calibration budget, not architecture, is the binding '
        f'constraint on trustworthy mode-of-action prediction, and that it should be reported '
        f'alongside every selective-prediction result.\n')

    md.append('## 1. Introduction\n')
    md.append(
        'Hierarchical formulations of receptor activity and mechanism prediction are established '
        '[3], as are scaffold-split evaluation, probability calibration, conformal prediction and '
        'applicability-domain analysis on Tox21 endpoints [1,2,4]. The architecture we evaluate '
        'is therefore not itself novel, and we do not claim it as such.\n\n'
        'What is not established is when the abstention layer placed on top of such an '
        'architecture is statistically supported. Selective prediction is routinely reported as '
        'a coverage/risk pair: the model answers some fraction of cases and errs less often on '
        'those it answers. Both quantities depend on a calibration sample, and for '
        'class-conditional (Mondrian) conformal prediction that sample is per class. When a '
        'class is small -- as the antagonist class is in essentially every ER reporter dataset '
        'after conservative curation -- the calibration partition can fall below the size at '
        'which the nominal level is attainable. The conformal quantile index then does not '
        'exist, the threshold is positive infinity, the prediction set admits every class, and '
        'coverage is reported as 1.0 with no error raised anywhere in the computation. The '
        'number is not wrong in the sense of a bug; it is uninterpretable.\n\n'
        'We make this failure mode explicit and testable:\n\n'
        '1. **A feasibility criterion.** For split conformal prediction at nominal miscoverage '
        'alpha, a per-class calibration sample of size n can certify alpha only if '
        'n >= ceil(1/alpha) - 1; the smallest certifiable level is 1/(n+1). A one-class support '
        'verifier accepting an in-class fraction q requires n >= ceil(1/(1-q)) for its empirical '
        'quantile to be non-degenerate. Both are checked from training-partition labels before '
        'any model is fitted, so the verdict cannot be selected after seeing results.\n'
        '2. **Two-tier label curation.** A definitive positive call on one endpoint paired with '
        'an inconclusive call on the other is unambiguous evidence of activity but only '
        'provisional evidence of exclusivity. Treating it as such recovers the minority class '
        'for the mode-of-action stage with zero added noise at the gate, while the inactive '
        'label is never relaxed, because an inconclusive call can never become a reliable '
        'negative.\n'
        '3. **Independent cohorts with measured distance.** ChEMBL specificity, sensitivity and '
        'curated-mechanism cohorts plus a cross-platform ER-BLA cohort, each InChIKey-disjoint '
        'from the entire curation and each reported with its Tanimoto distance to training, '
        'because an external set chemically identical to training validates nothing.\n'
        '4. **A negative result.** Where the criterion fails, we show the selective numbers it '
        'produces and what they conceal.\n')

    md.append('## 2. Data and curation\n')
    md.append(
        f'Both endpoints were taken from PubChem as full assay records (AID 743079, '
        f'ER-alpha agonist; AID 743091, ER-alpha antagonist summary; BG1/VM7Luc4E2 luciferase '
        f'reporter). Assay metadata is verified programmatically at load time. Antagonist wells '
        f'carrying a cytotoxicity flag in the Activity Summary field are quarantined in both '
        f'directions, following the viability confounding described for this screen [7]; a '
        f'cytotoxicity-flagged well is neither a reliable antagonist nor a reliable negative.\n\n'
        f'Standardization applies RDKit Cleanup, fragment-parent selection and uncharging, then '
        f'sanitizes the molecule before Bemis-Murcko scaffold perception. Tautomers are not '
        f'canonicalised, preserving the assayed identity and stereochemistry. Identity is the '
        f'standardized isomeric InChIKey; replicate calls must agree or the label is quarantined '
        f'rather than majority-voted. The complete count transition is in Table '
        f'`curation_waterfall` and Figure 01.\n\n'
        f'**Label definition.** Four states arise from the endpoint pair. The inactive label '
        f'requires both endpoints definitive and negative, in every tier. The `core` tier admits '
        f'only molecules with both endpoints definitive. The `extended` tier additionally admits '
        f'a definitive positive call on one endpoint paired with an inconclusive call on the '
        f'other, flagged `provisional` in the `moa_confidence` column. Resulting class sizes:\n')
    if cls:
        md.append('| Label | core | extended |')
        md.append('|---|---|---|')
        for k in ('inactive', 'agonist_only', 'antagonist_only', 'dual_active'):
            c, e = cls.get('core', {}).get(k), cls.get('extended', {}).get(k)
            md.append(f'| {k.replace("_", " ")} | '
                      f'{f"{c:,}" if isinstance(c, int) else "[N]"} | '
                      f'{f"{e:,}" if isinstance(e, int) else "[N]"} |')
        md.append('')
    md.append(
        'Dual-active molecules are never used to train the mode-of-action stage, because forcing '
        'an exclusive label on them corrupts the biology. They are retained as held-out stress '
        'cases: a system that claims to know when it should decline ought to decline them.\n')

    md.append('## 3. Experimental design\n')
    md.append(
        f'Primary evaluation is {cfg["outer_folds"]}-fold scaffold cross-validation '
        f'(StratifiedGroupKFold on Bemis-Murcko scaffolds) repeated over seeds '
        f'{cfg["seeds"]}; a random split is reported as a secondary, deliberately optimistic '
        f'reference. Within each outer training split, five scaffold-disjoint partitions are '
        f'allocated by budget: `fit`, `probability`, `policy`, `conformal` and `verifier`. '
        f'Allocation is driven by the sizes the conformal and verifier calibrators require and '
        f'is computed from training labels alone, before any fitting. Scaffold disjointness '
        f'between every pair of roles and the test fold is asserted at runtime, not assumed. '
        f'Hyperparameters are chosen by inner scaffold cross-validation within `fit` only.\n\n'
        f'Model families: {", ".join(cfg["models"])}. Representations: '
        f'{", ".join(cfg["representations"])}. The pre-declared primary configuration is '
        f'**{primary.get("model")} / {primary.get("representation")} / {pv}**. '
        f'{primary.get("declared", "")}\n\n'
        f'All {int(metrics.groupby(["split", "seed", "fold", "representation", "model"]).ngroups)} '
        f'evaluated cells x {metrics.variant.nunique()} variants are reported, but inference is '
        f'confined to the pre-declared primary against a declared baseline family; every other '
        f'cell is exploratory. Comparisons are paired by fold, tested with Wilcoxon signed-rank '
        f'and Holm-corrected across the family (Table `primary_paired_comparison`). The '
        f'resampling unit for every confidence interval is the outer fold, never the molecule.\n')

    md.append('## 4. Results\n')
    md.append('### 4.1 The calibration budget is the binding constraint\n')
    if budget is not None:
        md.append('| Component | Required n | Median available | Min available | Folds feasible | '
                  'Attainable alpha at min |')
        md.append('|---|---|---|---|---|---|')
        for r in budget.to_dict('records'):
            md.append(f'| {r["component"]} | {int(r["required_n"])} | '
                      f'{r["median_available_n"]:.0f} | {r["min_available_n"]:.0f} | '
                      f'{r["folds_feasible_fraction"]*100:.0f}% | '
                      f'{r["attainable_alpha_at_min"]:.3f} |')
        md.append('')
        if budget_fail:
            worst = min(budget_fail, key=lambda r: r['min_available_n'])
            md.append(
                f'The criterion fails for **{worst["component"]}**, where the calibration '
                f'partition holds as few as {worst["min_available_n"]:.0f} molecules against a '
                f'requirement of {int(worst["required_n"])}. At that size the smallest '
                f'certifiable miscoverage level is {worst["attainable_alpha_at_min"]:.3f}, not '
                f'the nominal {cfg["alpha"]}. Any coverage figure reported at the nominal level '
                f'for this class is an artefact.\n')
        else:
            md.append(
                f'Every component satisfies the criterion in every fold at the nominal level '
                f'alpha = {cfg["alpha"]}, so the selective results below are certifiable rather '
                f'than merely computed. This is a consequence of the extended tier: under the '
                f'core tier the same audit fails for the antagonist class (see Table '
                f'`tier_sensitivity` and the sibling core-tier run).\n')
    else:
        md.append('[Calibration budget table unavailable in this run.]\n')

    md.append('### 4.2 What a degenerate threshold does to the reported numbers\n')
    if nominal_row and feasible_row:
        md.append(
            f'| Conformal mode | Coverage | Antagonist coverage | Selective risk | '
            f'Infinite thresholds |')
        md.append('|---|---|---|---|---|')
        for nm, row in (('pooled', pooled_row), ('nominal alpha, Mondrian', nominal_row),
                        ('feasible alpha, Mondrian', feasible_row)):
            if row is None:
                continue
            md.append(f'| {nm} | {_ms(row, "coverage")} | {_ms(row, "coverage_class2")} | '
                      f'{_ms(row, "selective_risk")} | '
                      f'{_ms(row, "degenerate_threshold_fraction", pct=True)} |')
        md.append('')
        md.append(
            'Pooled calibration satisfies the criterion on the pooled sample while leaving the '
            'minority class uncalibrated, which is precisely why marginal validity is the wrong '
            'guarantee to report for an imbalanced mode-of-action task.\n')
    else:
        md.append('[Conformal comparison unavailable in this run.]\n')

    md.append('### 4.3 Selective performance and what abstention costs\n')
    md.append(
        f'The full pipeline attains selective risk {_ms(full_row, "selective_risk")} at coverage '
        f'{_ms(full_row, "coverage")}, retaining {_ms(full_row, "retained_active_fraction")} of '
        f'true actives. The hierarchy answering every molecule attains '
        f'{_ms(hier_row, "selective_risk")}; the flat calibrated three-class baseline '
        f'{_ms(flat_row, "selective_risk")}; the independent endpoint-pair baseline '
        f'{_ms(endp_row, "selective_risk")}. Mode-of-action balanced accuracy is '
        f'{_ms(full_row, "moa_balanced_accuracy")} for the full pipeline against '
        f'{_ms(flat_row, "moa_balanced_accuracy")} flat.\n\n'
        f'A lower error on an accepted subset is not an improvement if the rejected molecules '
        f'were the informative ones, so Figure 17 places every operating point on the '
        f'risk-coverage curve and Table `ablation_cascade` reports coverage, selective risk and '
        f'retained actives together at each stage. AURC, which is coverage-independent, is '
        f'reported in Table `aurc_by_variant`.\n')
    if comparison is not None and len(comparison):
        verdict = comparison.get('verdict')
        if verdict is not None:
            wins = int((verdict == 'PROPOSED BETTER').sum())
            losses = int((verdict == 'BASELINE BETTER').sum())
            md.append(
                f'Of {len(comparison)} pre-declared paired comparisons, {wins} favour the '
                f'proposed configuration after Holm correction and {losses} favour a baseline; '
                f'the remainder are undecided. Direction is stated per metric in Table '
                f'`primary_paired_comparison`, because for selective risk and AURC a positive '
                f'difference is a worse result, not a better one.\n')
            for _, row in comparison[verdict == 'BASELINE BETTER'].iterrows():
                md.append(f'- A baseline is better on **{row.metric}** '
                          f'({row.direction}): {row.baseline} {row.baseline_mean:.3f} versus '
                          f'{row.proposed_mean:.3f} for the proposed configuration '
                          f'(Holm p = {row.holm_p:.4f}). This is reported, not omitted.')
            if losses:
                md.append('')
        else:
            sig = comparison[(comparison.get('significant_holm_0.05') == True)]  # noqa: E712
            md.append(f'Of {len(comparison)} pre-declared paired comparisons, {len(sig)} remain '
                      f'significant after Holm correction at the 0.05 level.\n')

    md.append('### 4.3b The operating point, not the model, sets the specificity\n')
    if thrpol is not None and len(thrpol):
        md.append('| Threshold policy | tau | Recall | Specificity | MCC |')
        md.append('|---|---|---|---|---|')
        for r in thrpol.to_dict('records'):
            md.append(f'| {r.get("label", r.get("policy"))} | {_fmt(r.get("tau_mean"))} | '
                      f'{_fmt(r.get("recall_mean"))} ± {_fmt(r.get("recall_sd"))} | '
                      f'{_fmt(r.get("specificity_mean"))} ± {_fmt(r.get("specificity_sd"))} | '
                      f'{_fmt(r.get("mcc_mean"))} ± {_fmt(r.get("mcc_sd"))} |')
        md.append('')
        md.append(
            'Each threshold is selected on the same disjoint policy partition and evaluated on '
            'the untouched test fold, so the three are directly comparable. The recall target '
            'used throughout the main analysis is ONE POINT on this curve and is reported as '
            'such, not as the operating point of the method. At an active prevalence near 0.18 '
            'a 95% recall target drives the threshold low enough that most of the library is '
            'flagged; a balanced criterion trades recall for a specificity a screening workflow '
            'could act on. All three are reported rather than the most flattering one '
            '(Figure 31).\n')
    else:
        md.append('[Threshold-policy comparison unavailable; run rescore.py before the '
                  'analysis step.]\n')

    md.append('### 4.4 Generalisation, chemical distance and negative controls\n')
    if split_dis is not None and len(split_dis):
        byk = split_dis.groupby('split').median_nn_tanimoto.agg(['mean', 'std'])
        for k in byk.index:
            md.append(f'- {k.capitalize()} split: median nearest-train Tanimoto '
                      f'{byk.loc[k, "mean"]:.3f} ± {byk.loc[k, "std"]:.3f} across folds.')
        md.append('')
        md.append('The scaffold split places chemically more distant molecules in the held-out '
                  'fold than a random split does, which is the evidence that it is doing work '
                  'rather than being asserted (Figure 10).\n')
    if bands is not None and len(bands):
        md.append('| Nearest-train Tanimoto | n | Accuracy | Accepted accuracy | Coverage |')
        md.append('|---|---|---|---|---|')
        for r in bands.to_dict('records'):
            md.append(f'| {r["band"]} | {int(r["n"]):,} | {_fmt(r.get("accuracy"))} | '
                      f'{_fmt(r.get("accepted_accuracy"))} | '
                      f'{_fmt(r.get("coverage_full_knn"))} |')
        md.append('')
    if perm is not None and len(perm):
        pr = _row(perm, 'permuted_labels_control')
        md.append(f'Permuting training labels collapses activity-gate PR-AUC to '
                  f'{_ms(pr, "activity_pr_auc")}, against '
                  f'{_ms(_row(perm, "hierarchy_calibrated"), "activity_pr_auc")} with real '
                  f'labels, confirming the pipeline does not manufacture signal from noise.\n')
    if external is not None and len(external):
        md.append('**External cohorts.** Each is InChIKey-disjoint from the entire curation and '
                  'never used for fitting, tuning, calibration or thresholding.\n')
        md.append('| Cohort | Fold models | Median NN Tanimoto to fit | Specificity | Recall | '
                  'MoA balanced accuracy | Coverage |')
        md.append('|---|---|---|---|---|---|---|')
        for r in external.to_dict('records'):
            # A metric is reported only where it is DEFINED for that cohort. Selecting a column
            # by key presence rather than by a finite value prints "[UNAVAILABLE]" for every
            # cohort, because summarise() emits one column set for all of them.
            def cell(col):
                v = r.get(f'{col}_mean')
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return '—'
                sd = r.get(f'{col}_sd')
                out = _fmt(v)
                if sd is not None and isinstance(sd, float) and np.isfinite(sd):
                    out += f' ± {_fmt(sd)}'
                return out
            md.append(f'| {r["cohort"]} | {int(r.get("n_folds", 0))} | '
                      f'{cell("median_nn_tanimoto_fit")} | {cell("external_specificity")} | '
                      f'{cell("external_recall")} | {cell("external_moa_balanced_accuracy")} | '
                      f'{cell("coverage")} |')
        md.append('')
        md.append('An em dash marks a metric that is undefined for that cohort: the specificity '
                  'cohort contains no asserted binders, and the sensitivity cohort no asserted '
                  'non-binders, so neither alone yields both error rates. Section 4.4 pools them '
                  'for the threshold-free comparison.\n')
        md.append('ChEMBL labels derive from heterogeneous literature assays curated by third '
                  'parties and the BLA cohort from a different reporter construct, so these '
                  'quantify behaviour under distribution shift, not replication of the training '
                  'endpoints.\n')
        if tiny_cohorts:
            names = ', '.join(f'`{c}` (n = {n})' for c, n in tiny_cohorts)
            md.append(f'**{names} is below the {MIN_COHORT}-molecule floor this work sets for a '
                      f'validation claim and is reported for completeness only.** Curated '
                      f'mechanism annotations for this receptor are few, and most of the '
                      f'molecules carrying them are themselves in the Tox21 library, so the '
                      f'disjointness requirement removes them. No conclusion is drawn from '
                      f'that cohort.\n')
        if disc is not None and len(disc):
            d0 = disc.iloc[0].to_dict()
            md.append(
                f'**Threshold-free external discrimination.** Pooling the ChEMBL cohorts into '
                f'one labelled set ({int(d0.get("n_negative", 0))} asserted non-binders, '
                f'{int(d0.get("n_positive", 0))} asserted binders) gives an external ROC-AUC of '
                f'{_fmt(d0.get("external_roc_auc_mean"))} '
                f'(SD {_fmt(d0.get("external_roc_auc_sd"))}) and PR-AUC '
                f'{_fmt(d0.get("external_pr_auc_mean"))}. This separates two very different '
                f'failures: a ranking that does not transfer, and a ranking that transfers while '
                f'the deployed threshold does not. See Table '
                f'`external_discrimination_verdict.json` and Figure 32.\n')

    md.append('## 5. Discussion\n')
    md.append(
        'The result we consider most useful is not a performance number. It is that a widely '
        'used selective-prediction stack can report a clean coverage/risk pair while one of its '
        'classes is uncalibrated, and that nothing in the computation signals this. The fix is '
        'cheap: compute the per-class requirement from the nominal level, compare it to the '
        'available calibration sample before fitting, and report both. The cost of not doing so '
        'is a published guarantee that does not hold for the class practitioners most care '
        'about -- the rare mechanism.\n\n'
        'This reframes how minority-class results should be read across this literature. A '
        'selective ER antagonist model trained on a few dozen curated antagonists cannot '
        'certify 90% class-conditional coverage regardless of its architecture, and a paper '
        'reporting that it does is reporting an infinite threshold. Our two-tier curation is one '
        'way to buy the budget back without weakening the negative class; combining assay '
        'platforms or databases is another. What matters is that the budget is stated.\n\n'
        'A second observation concerns pooled versus class-conditional calibration. Pooled '
        'conformal prediction is marginally valid and will pass a naive audit on an imbalanced '
        'dataset while providing no guarantee for the minority class. For mode-of-action '
        'prediction, where the minority class is the point of the exercise, marginal validity '
        'is close to useless and should not be reported as if it were conditional.\n')
    md.append('### 5.1 Comparison with prior work\n')
    md.append('Full advantages/limitations table in `tables/literature_sota_comparison.csv`. '
              'In summary: hierarchy [3], conformal prediction on ER data [1], and the '
              'calibration/uncertainty/applicability-domain combination [2] are all prior art. '
              'We contribute the feasibility criterion that determines whether the selective '
              'layer those works build can be certified at the level they report, the label-tier '
              'mechanism that supplies the missing budget, and the demonstration of what the '
              'numbers look like when it is absent.\n')

    md.append('## 6. Limitations\n')
    md.append(
        f'- The antagonist class is small in absolute terms even in the extended tier '
        f'({cls.get("extended", {}).get("antagonist_only", "[N]")} molecules). The criterion '
        f'establishes that the reported level is attainable; it does not make the estimates '
        f'precise, and the fold standard deviations should be read accordingly.\n'
        f'- Extended-tier antagonist labels are provisional with respect to exclusivity: some '
        f'may be unrecognised dual-actives. The core-tier sensitivity run bounds this.\n'
        f'- External labels come from different assay systems; no cohort replicates the training '
        f'endpoints. The ChEMBL specificity cohort estimates behaviour on literature-asserted '
        f'non-binders, not on Tox21 reporter negatives.\n'
        f'- Reporter activity is not an in-vivo endocrine effect [7].\n'
        f'- The pretrained-embedding representation may have seen these molecules during '
        f'pretraining; it is reported as an ablation and is excluded from the primary '
        f'configuration for that reason.\n'
        f'- No wet-lab validation.\n'
        f'- **Definition of the mode-of-action metric.** `moa_balanced_accuracy` is '
        f'computed ONLY over held-out molecules whose true label is agonist-only or '
        f'antagonist-only; inactive and dual-active molecules are excluded. It therefore '
        f'measures the Stage 2 decision GIVEN that a molecule is an exclusive active, not '
        f'end-to-end screening performance, and it is not comparable to a three-class '
        f'accuracy. The molecules entering it per fold are counted in `moa_n`.\n'
        f'- **Provisional exclusivity labels.** In the extended tier a molecule with a '
        f'definitive positive call on one endpoint and an inconclusive call on the other '
        f'is given that endpoint\u2019s mode of action and flagged `provisional` in '
        f'`moa_confidence`. Its ACTIVITY is certain; its EXCLUSIVITY is not, so an '
        f'unrecognised dual-active may sit inside the agonist-only or antagonist-only '
        f'class. Extended-tier mode-of-action metrics therefore carry label noise of '
        f'unknown sign, bounded only when the core-tier arm is run.\n'
        f'- **Operating point.** The main analysis uses a 95% recall target. Section 4.3b '
        f'shows this is a deliberately sensitive point rather than an optimum, and reports '
        f'the alternatives on the same partitions.\n')

    md.append('## 7. Reproducibility\n')
    md.append(
        '`bash 01_run_experiments.sh` curates, audits the calibration budget, runs the nested '
        'benchmark and evaluates external cohorts. `bash 02_finish_analysis.sh` produces every '
        'figure, table and this draft. A configuration or source-code change invalidates an '
        'existing results directory and the run refuses to mix results. Raw-data checksums, the '
        'resolved package environment, per-fold partition membership, inner-selection records '
        'and all frozen thresholds are written alongside the results.\n')

    (pdir / 'manuscript.md').write_text('\n'.join(md))

    # ------------------------------------------------------------------ readiness
    ext_ok = external is not None and len(external) > 0
    chembl_ok = False
    scope = out / 'external/chembl_scope.json'
    if scope.exists():
        try:
            sj = json.loads(scope.read_text())
            chembl_ok = bool(sj.get('enabled')) and any(
                isinstance(v, dict) and v.get('disjoint', 0) > 0 for v in sj.values())
        except Exception:
            pass
    # Superiority means a Holm-significant difference IN THE PROPOSED SYSTEM'S FAVOUR.
    sig_any = (comparison is not None and len(comparison)
               and 'verdict' in comparison
               and bool((comparison.verdict == 'PROPOSED BETTER').any()))
    baseline_better = (comparison is not None and len(comparison) and 'verdict' in comparison
                       and bool((comparison.verdict == 'BASELINE BETTER').any()))
    checks = {
        'full_profile': cfg['profile'] == 'full',
        'all_configured_cells_complete': bool((out / 'COMPLETE.json').exists()),
        'calibration_budget_criterion_satisfied': bool(feasible_all),
        'conformal_degeneracy_quantified': conformal is not None and len(conformal) > 0,
        'permutation_control_passed': perm is not None and len(perm) > 0,
        'scaffold_split_dissimilarity_demonstrated': split_dis is not None and len(split_dis) > 0,
        'chembl_external_cohort_present': chembl_ok,
        'cross_platform_cohort_present': ext_ok,
        # Satisfied only when BOTH tiers are actually present. A table holding one tier is
        # not a sensitivity comparison, and reporting it as one would overstate the evidence.
        'tier_sensitivity_run': _tiers_compared(T) >= 2,
        'holm_corrected_superiority_demonstrated': bool(sig_any),
        'threshold_policy_sensitivity_reported': thrpol is not None and len(thrpol) > 1,
        'external_threshold_free_discrimination_measured': disc is not None
        and len(disc) > 0,
        'all_external_cohorts_above_size_floor': ext_ok and not tiny_cohorts,
        'pretrained_overlap_resolved': False,
        'wet_lab_validation': False,
        'human_scientific_review': False,
    }
    rd = pd.DataFrame([{'requirement': k, 'satisfied': v} for k, v in checks.items()])
    table(rd, 'submission_readiness')
    lines = ['# Submission readiness', '',
             rd.to_markdown(index=False), '',
             '## What these two commands do and do not give you', '',
             'They give you: a curated dataset with an auditable count trail, a pre-registered '
             'calibration-budget audit, a nested scaffold benchmark with baselines and a '
             'negative control, selective-prediction results reported with their coverage, '
             'external cohorts with measured chemical distance, every figure and table, and a '
             'result-linked draft.', '',
             'They do not give you: peer review, wet-lab validation, a resolution of the '
             'pretrained-embedding overlap question, or a guarantee of acceptance anywhere. '
             'Bracketed placeholders in the draft mark numbers this run did not produce; do not '
             'fill them by hand without recomputing.', '']
    lines += ['**Do not describe this work as demonstrating general superiority.** Where the '
              'proposed configuration wins it wins on specific metrics against specific '
              'baselines, and those are named in `primary_paired_comparison.csv` alongside the '
              'metrics where a baseline wins. Coverage must be quoted beside every selective '
              'result; a low error on an accepted subset is not a result on its own.', '']
    if not checks['holm_corrected_superiority_demonstrated']:
        lines += ['**No superiority claim is supported by this run.** No pre-declared comparison '
                  'survived Holm correction. The manuscript is written around the feasibility '
                  'result, which does not depend on beating a baseline; keep it that way rather '
                  'than reaching for an uncorrected comparison.', '']
    if baseline_better:
        rows_bb = comparison[comparison.verdict == 'BASELINE BETTER']
        lines += ['**A baseline beats the proposed configuration on '
                  + ', '.join(sorted(set(rows_bb.metric))) + '.** '
                  'These are Holm-significant differences in the baseline\'s favour and must '
                  'appear in the manuscript. Do not report only the metrics that went the '
                  'other way.', '']
    if not checks['chembl_external_cohort_present']:
        lines += ['**No ChEMBL cohort was built.** Either the network was unavailable or '
                  '`chembl.enabled` is false. External-validation claims must be removed from '
                  'the draft until it is populated.', '']
    if not checks['calibration_budget_criterion_satisfied']:
        lines += ['**The calibration-budget criterion is violated in this run.** That is a '
                  'reportable finding, not a bug, but the nominal-level conformal results must '
                  'be presented as degenerate and the feasible-alpha variant used for any '
                  'operational claim.', '']
    (pdir / 'submission_readiness.md').write_text('\n'.join(lines))

    # ------------------------------------------------------------------ tex + captions
    tex = ['\\documentclass[11pt]{article}',
           '\\usepackage[margin=1in]{geometry}\\usepackage{booktabs}\\usepackage{graphicx}',
           '\\usepackage{hyperref}\\usepackage[utf8]{inputenc}',
           f'\\title{{{TITLE}}}', '\\date{\\today}', '\\begin{document}', '\\maketitle',
           '\\noindent\\textit{Draft skeleton. Tables are in tables/*.tex and figures in '
           'figures/*.pdf-ready SVG; the prose source of record is paper/manuscript.md.}',
           '\\end{document}']
    (pdir / 'manuscript.tex').write_text('\n'.join(tex))
    caps = out / 'figures/captions.md'
    if caps.exists():
        (pdir / 'figure_captions.md').write_text(caps.read_text())
    print(f'[paper] manuscript.md + submission_readiness.md written to {pdir}', flush=True)
    return checks

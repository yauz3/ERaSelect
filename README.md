# ERaSelect

Calibration-budget-aware selective modelling of estrogen receptor alpha
activity and mode of action.

## Dataset

The paired Tox21 endpoints are PubChem AIDs 743079 and 743091.
The curated extended dataset contains 5,569 unique molecules.

- `data/raw_data.csv`: combined source assay records, distinguished by
  `source_aid`. This is a merged export, not a byte-identical source download.
- `data/train.csv`: curated molecules and original training-side allocations.
- `data/test.csv`: original held-out outer-fold allocations.
- `data/raw/`: cached molecular structures and assay descriptions.
- `data/manifest.json`: provenance, class counts and SHA-256 checksums.

There are 15 scaffold partitions (five folds for each of three seeds)
and five random partitions. Molecules recur across partitions.

**Select one `split_type`, `seed` and `fold` before using these files.
Never fit a model on the concatenated training file and evaluate it on the
concatenated test file.**

Within each training partition, `role` distinguishes `fit`, `probability`,
`policy`, `conformal`, and `verifier`. Only `fit` is the model-fitting subset;
the remaining roles preserve the original calibration and selection protocol.

The paired endpoint labels, label-confidence information, SMILES and molecular
identifiers are retained. Model/representation-specific copies of identical
partition assignments have been checked and collapsed.

## Code

The original Python sources, shell entry points, requirements and
configuration files are included. Generated results and fitted models are
excluded.

The original experiment scripts use their original data-loading paths;
the release CSVs document the completed experiment allocations and are not
automatically substituted into that loader.

External ChEMBL and cross-platform evaluation datasets are not included in
this initial data release. This release alone does not reproduce those
external evaluations.

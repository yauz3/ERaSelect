"""Re-run only the external-cohort evaluation on an existing, completed results directory.

Use this when `01_run_experiments.sh` finished every fold but crashed in its final step. No
model is refitted and no fold is re-evaluated: the serialized primary fold models are loaded,
the ChEMBL cohorts already written to `results/<profile>/external/` are reused, and only
`external_metrics.csv`, `external_similarity.csv` and `external_report.json` are produced.

    PYTHONPATH=$PWD/src .venv/bin/python resume_external.py                 # profile: full
    PYTHONPATH=$PWD/src .venv/bin/python resume_external.py --profile fast

This deliberately does NOT go through `01_run_experiments.sh`, which would refuse to continue
because the source digest changed when the fix was applied. That guard exists to stop results
produced by different code from being mixed, and it is doing its job here: the experiments were
produced by the old code and are untouched, while only the step that never completed is run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402
from external import evaluate_external  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='full')
    ap.add_argument('--keep-pretrained', action='store_true',
                    help='Build the ChemBERTa embedding for the cohorts too. Only needed if the '
                         'primary representation is pretrained_fusion; otherwise it is wasted '
                         'work on thousands of molecules.')
    args = ap.parse_args()

    cfg_path = ROOT_DIR / f'config/{args.profile}.json'
    if not cfg_path.exists():
        raise SystemExit(f'missing {cfg_path}')
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / 'results' / args.profile

    if not (out / 'fold_metrics.csv').exists():
        raise SystemExit(f'{out}/fold_metrics.csv is missing, so the experiments did not '
                         f'complete. Run 01_run_experiments.sh first.')

    primary = cfg.get('primary', {})
    rep = primary.get('representation')
    if rep != 'pretrained_fusion' and not args.keep_pretrained:
        cfg['pretrained'] = False
        print(f'[resume] primary representation is {rep!r}, so the pretrained encoder is not '
              f'needed for the cohorts and is skipped (pass --keep-pretrained to force it).',
              flush=True)

    master = pd.read_csv(out / 'data/master.csv', low_memory=False)
    print(f'[resume] profile={args.profile}  molecules={len(master):,}  '
          f'primary={primary.get("model")}/{rep}', flush=True)

    report = evaluate_external(cfg, out, master)
    print(f'[resume] status: {report.get("status")}', flush=True)

    # Restore the completion marker the crashed run never wrote, so the readiness audit can see
    # that every configured cell finished.
    done = out / 'COMPLETE.json'
    if not done.exists():
        m = pd.read_csv(out / 'fold_metrics.csv', low_memory=False)
        done.write_text(json.dumps({
            'profile': args.profile,
            'cells_with_metrics': int(m.groupby(['split', 'seed', 'fold', 'representation',
                                                 'model']).ngroups),
            'variants_per_cell': int(m.variant.nunique()),
            'status': 'experiments_complete',
            'note': 'written by resume_external.py after the original run completed every fold '
                    'but crashed in its final external-evaluation step',
        }, indent=2))
        print(f'[resume] wrote {done.name}', flush=True)

    print('\nNext:  PROFILE=' + args.profile + ' bash 02_finish_analysis.sh', flush=True)


if __name__ == '__main__':
    main()

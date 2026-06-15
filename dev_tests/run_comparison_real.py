"""Compare BayesOpt vs LegacyGridSearch with real DYNAMITE orblib.

Runs BayesOpt first (if not already done), then LegacyGridSearch, and
prints a side-by-side per-iteration chi2 comparison.

Run from dev_tests/:
    cd /Users/pesmith/research/dynamite/dev_tests
    /opt/miniconda3/envs/main/bin/python3 run_comparison_real.py
"""
import sys
import time
import numpy as np
sys.path.insert(0, '..')
import dynamite as dyn
import warnings
warnings.filterwarnings('ignore')


def run_and_collect(config_file, label):
    print(f'\n--- Running {label} ---', flush=True)
    c = dyn.config_reader.Configuration(config_file)
    t0 = time.time()
    dyn.model_iterator.ModelIterator(c, plots=False)
    wall = time.time() - t0
    t = c.all_models.table
    chi2_col = np.asarray(t['kinchi2'], dtype=float)
    best_chi2 = float(np.nanmin(chi2_col))
    best_idx = int(np.nanargmin(chi2_col))
    ml_best = float(t['ml'][best_idx])

    running_best = {}
    cur = np.inf
    for it in sorted(set(int(i) for i in t['which_iter'])):
        mask = np.asarray(t['which_iter'], dtype=int) == it
        vals = chi2_col[mask]
        finite = vals[np.isfinite(vals)]
        if len(finite):
            cur = min(cur, float(np.min(finite)))
        running_best[int(it)] = cur

    print(f'  n_models={len(t)}  best_kinchi2={best_chi2:.4f}  '
          f'ml_best={ml_best:.3f}  wall={wall:.0f}s')
    return {
        'label': label,
        'n_models': len(t),
        'best_chi2': best_chi2,
        'ml_best': ml_best,
        'wall_s': wall,
        'running_best': running_best,
    }


results = []
results.append(run_and_collect('bayesopt_ml_modelinner.yaml',
                               'BayesOpt (ModelInner)'))
results.append(run_and_collect('legacygrid_ml_modelinner.yaml',
                               'LegacyGridSearch     '))

print(f'\n{"="*65}')
print('REAL ORBLIB COMPARISON — ml free, nE=2, nI2=4, nI3=3')
print(f'{"="*65}')
hdr = (f'{"Generator":<25} {"N_models":>9} {"best_kinchi2":>13} '
       f'{"ml_best":>8} {"wall_s":>8}')
print(hdr)
print('-' * 65)
for r in results:
    print(f'{r["label"]:<25} {r["n_models"]:>9} {r["best_chi2"]:>13.4f} '
          f'{r["ml_best"]:>8.3f} {r["wall_s"]:>8.0f}')
print(f'{"="*65}')

all_iters = sorted(set(it for r in results for it in r['running_best']))
print('\nRunning-best kinchi2 per iteration:')
hdr2 = f'{"iter":>5}' + ''.join(f'{r["label"][:14]:>16}' for r in results)
print(hdr2)
for it in all_iters:
    row_str = f'{it:>5}'
    for r in results:
        v = r['running_best'].get(it)
        row_str += f'{v:>16.4f}' if v is not None else f'{"—":>16}'
    print(row_str)

for r in results:
    assert np.isfinite(r['best_chi2']), f'{r["label"]}: best chi2 is not finite'
    assert r['best_chi2'] > 0, f'{r["label"]}: chi2 must be positive'

print('\nREAL COMPARISON COMPLETE')

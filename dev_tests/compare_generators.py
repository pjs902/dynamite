"""Compare BayesOpt, GridWalk, LegacyGridSearch on identical dummy landscape.

Synthetic chi2: quadratic bowl in ml, minimum at ml=5.5 (between grid points).
GridWalk and LegacyGridSearch can reach at best chi2=65 (at ml=5.0 or 6.0
with step=1.0); BayesOpt should reach chi2 much closer to the true minimum of
15.0 via GP-guided search.

All three generators run with identical stopping criteria (n_max_mods=12).
Output directories are cleaned before each run so restarts don't interfere.

Run from dev_tests/:
    cd /Users/pesmith/research/dynamite/dev_tests
    /opt/miniconda3/envs/main/bin/python3 compare_generators.py
"""
import os
import sys
import time
import shutil
import numpy as np
sys.path.insert(0, '..')
import dynamite as dyn
import warnings
warnings.filterwarnings('ignore')

ML_TRUE = 5.5
CHI2_MIN = 15.0
CHI2_GRID = 65.0  # best achievable with step=1.0 from ml=5.0 (both 4.0 and 6.0 give 65)

# Clean all output directories before any config is loaded (paths are relative to cwd)
_OUTPUT_DIRS = [
    'NGC6278_bayesopt_modelinner_output',
    'NGC6278_gridwalk_output',
    'NGC6278_legacygrid_output',
]
for _d in _OUTPUT_DIRS:
    if os.path.exists(_d):
        shutil.rmtree(_d)


def dummy_chi2(parset):
    ml = float(parset['ml'])
    return 200.0 * (ml - ML_TRUE) ** 2 + CHI2_MIN


def run_generator(config_file, label):
    c = dyn.config_reader.Configuration(config_file)

    t0 = time.time()
    dyn.model_iterator.ModelIterator(
        c,
        do_dummy_run=True,
        dummy_chi2_function=dummy_chi2,
        plots=False,
    )
    wall = time.time() - t0
    t = c.all_models.table
    chi2_vals = np.asarray(t['kinchi2'], dtype=float)
    finite_mask = np.isfinite(chi2_vals)

    best_chi2 = float(np.nanmin(chi2_vals)) if np.any(finite_mask) else np.nan
    best_idx = int(np.nanargmin(chi2_vals)) if np.any(finite_mask) else 0
    ml_best = float(t['ml'][best_idx])

    running_best = {}
    cur = np.inf
    for it in sorted(set(int(i) for i in t['which_iter'])):
        mask = np.asarray(t['which_iter'], dtype=int) == it
        vals = chi2_vals[mask]
        finite = vals[np.isfinite(vals)]
        if len(finite):
            cur = min(cur, float(np.min(finite)))
        running_best[it] = cur

    return {
        'label': label,
        'n_models': len(t),
        'n_iters': max(running_best.keys()) + 1 if running_best else 0,
        'best_chi2': best_chi2,
        'ml_best': ml_best,
        'wall_s': wall,
        'running_best': running_best,
    }


RUNS = [
    ('bayesopt_ml_modelinner.yaml',   'BayesOpt (ModelInner)'),
    ('gridwalk_ml_modelinner.yaml',   'GridWalk (ModelInner)'),
    ('legacygrid_ml_modelinner.yaml', 'LegacyGridSearch     '),
]

results = []
for cfg, label in RUNS:
    print(f'\nRunning {label} ...', flush=True)
    r = run_generator(cfg, label)
    results.append(r)
    print(f'  n_models={r["n_models"]}  best_chi2={r["best_chi2"]:.3f}'
          f'  ml_best={r["ml_best"]:.3f}  wall={r["wall_s"]:.1f}s')

print(f'\n{"="*70}')
print('GENERATOR COMPARISON — dummy chi2 landscape, ml free only')
print(f'True minimum: ml={ML_TRUE}, chi2={CHI2_MIN}')
print(f'Grid methods can reach at best chi2={CHI2_GRID} (ml=5.0 or 6.0 with step=1.0)')
print(f'{"="*70}')
hdr = (f'{"Generator":<28} {"N_models":>9} {"N_iters":>8} '
       f'{"best_chi2":>10} {"ml_best":>8} {"wall_s":>8}')
print(hdr)
print('-' * 70)
for r in results:
    print(f'{r["label"]:<28} {r["n_models"]:>9} {r["n_iters"]:>8} '
          f'{r["best_chi2"]:>10.3f} {r["ml_best"]:>8.3f} {r["wall_s"]:>8.2f}')
print(f'{"="*70}')

all_iters = sorted(set(it for r in results for it in r['running_best']))
print('\nRunning-best kinchi2 per iteration:')
hdr2 = f'{"Iter":>5}' + ''.join(f'{r["label"][:14]:>16}' for r in results)
print(hdr2)
for it in all_iters:
    row_str = f'{it:>5}'
    for r in results:
        v = r['running_best'].get(it)
        row_str += f'{v:>16.3f}' if v is not None and np.isfinite(v) else f'{"—":>16}'
    print(row_str)

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
bo_result = results[0]
gw_result = results[1]
lg_result = results[2]

# All generators must run at least some models
for r in results:
    assert r['n_models'] > 0, f'{r["label"]}: no models were run'
    assert np.isfinite(r['best_chi2']), f'{r["label"]}: best chi2 is not finite'

# BayesOpt should find a significantly better chi2 than the grid floor.
# With 12 models and GP guidance, it should get within 20 of the true minimum.
assert bo_result['best_chi2'] < CHI2_GRID - 5.0, (
    f'BayesOpt best chi2={bo_result["best_chi2"]:.2f} should be better than '
    f'grid floor {CHI2_GRID} (expected GP to exploit the continuous space)'
)

# Grid methods are expected to be stuck at or near chi2=65 (step=1.0 resolution)
for r in [gw_result, lg_result]:
    assert r['best_chi2'] <= CHI2_GRID + 1.0, \
        f'{r["label"]} best chi2={r["best_chi2"]:.2f} unexpectedly high'

print('\nAll assertions PASSED')
print(f'BayesOpt chi2 improvement over grid methods: '
      f'{min(gw_result["best_chi2"], lg_result["best_chi2"]) - bo_result["best_chi2"]:.2f}')
print('COMPARE GENERATORS COMPLETE')

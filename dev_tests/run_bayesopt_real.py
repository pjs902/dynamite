"""Real end-to-end run of BayesOptGenerator with actual DYNAMITE orblib.

Runs bayesopt_ml_modelinner.yaml using the true orbit integration and
weight solver. Expected to take several minutes per model.

Run from dev_tests/:
    cd /Users/pesmith/research/dynamite/dev_tests
    /opt/miniconda3/envs/main/bin/python3 run_bayesopt_real.py
"""
import sys
import time
import numpy as np
sys.path.insert(0, '..')
import dynamite as dyn
import warnings
warnings.filterwarnings('ignore')

print('Loading bayesopt_ml_modelinner.yaml ...', flush=True)
c = dyn.config_reader.Configuration('bayesopt_ml_modelinner.yaml')
free = [p.name for p in c.parspace if not p.fixed]
n_max = c.settings.parameter_space_settings['stopping_criteria']['n_max_mods']
print(f'Free parameters: {free}')
print(f'n_max_mods: {n_max}')

t0 = time.time()
print('Starting ModelIterator (real orblib) ...', flush=True)
dyn.model_iterator.ModelIterator(c, plots=False)
wall = time.time() - t0

t = c.all_models.table
n_models = len(t)
chi2_col = np.asarray(t['kinchi2'], dtype=float)
best_chi2 = float(np.nanmin(chi2_col)) if n_models > 0 else np.nan
best_idx = int(np.nanargmin(chi2_col))
ml_best = float(t['ml'][best_idx])

print(f'\n--- Real run complete ---')
print(f'  Total models run:  {n_models}')
print(f'  Best kinchi2:      {best_chi2:.4f}')
print(f'  Best ml:           {ml_best:.3f}')
print(f'  Wall time:         {wall:.1f}s ({wall/60:.1f} min)')

print(f'\nPer-iteration summary:')
for it in sorted(set(int(i) for i in t['which_iter'])):
    mask = np.asarray(t['which_iter'], dtype=int) == it
    mls = [f'{v:.2f}' for v in t['ml'][mask]]
    chi2s = [f'{float(v):.2f}' for v in t['kinchi2'][mask]]
    print(f'  iter {it}: ml={mls} kinchi2={chi2s}')

# Basic sanity checks
assert n_models > 0, 'No models were run'
n_done = int(np.sum(t['all_done']))
assert n_done > 0, f'No models completed: {n_done}/{n_models}'
assert np.isfinite(best_chi2), 'Best chi2 is not finite'
assert best_chi2 > 0, 'chi2 must be positive for real data'

print('\nREAL RUN ASSERTIONS PASSED')

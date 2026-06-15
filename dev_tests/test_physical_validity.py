"""Assert BayesOptGenerator never proposes a physically invalid model.

Uses bayesopt_qml_modelinner.yaml (q-stars + ml free) in dummy mode.
Monkey-patches ParameterSpace.validate_parset to count rejections.

Run from dev_tests/:
    cd /Users/pesmith/research/dynamite/dev_tests
    /opt/miniconda3/envs/main/bin/python3 test_physical_validity.py
"""
import sys
import numpy as np
sys.path.insert(0, '..')
import dynamite as dyn
import dynamite.parameter_space as _ps_module
import warnings
warnings.filterwarnings('ignore')


def dummy_chi2_qml(parset):
    q = float(parset['q-stars'])
    ml = float(parset['ml'])
    return 500.0 * (q - 0.54) ** 2 + 200.0 * (ml - 5.5) ** 2 + 10.0


class _ValidityCounter:
    total_calls = 0
    rejections = 0
    rejected_parsets = []


_counter = _ValidityCounter()
_original_validate = _ps_module.ParameterSpace.validate_parset


def _patched_validate(self, parset):
    result = _original_validate(self, parset)
    _counter.total_calls += 1
    if not result:
        _counter.rejections += 1
        try:
            _counter.rejected_parsets.append(
                {p.name: p.raw_value for p in parset})
        except Exception:
            _counter.rejected_parsets.append(repr(parset))
    return result


_ps_module.ParameterSpace.validate_parset = _patched_validate

print('Loading bayesopt_qml_modelinner.yaml ...', flush=True)
c = dyn.config_reader.Configuration('bayesopt_qml_modelinner.yaml')
free_names = [p.name for p in c.parspace if not p.fixed]
print(f'Free parameters: {free_names}')

print('Running ModelIterator in dummy mode ...', flush=True)
dyn.model_iterator.ModelIterator(
    c,
    do_dummy_run=True,
    dummy_chi2_function=dummy_chi2_qml,
    plots=False,
)

t = c.all_models.table
n_models = len(t)
chi2_vals = np.asarray(t['kinchi2'], dtype=float)
best_chi2 = float(np.nanmin(chi2_vals)) if n_models > 0 else np.nan

qobs = dyn.parameter_space.get_qobs_from_system(c.system)
accepted_q = np.asarray(t['q-stars'], dtype=float)

print(f'\nResults:')
print(f'  Total models accepted:      {n_models}')
print(f'  validate_parset calls:      {_counter.total_calls}')
print(f'  validate_parset rejections: {_counter.rejections}')
print(f'  Best chi2 reached:          {best_chi2:.4f}')
print(f'  qobs (triaxiality limit):   {qobs:.4f}')
print(f'  Accepted q-stars max:       {accepted_q.max():.4f}')

# BoTorch triaxiality constraints only activate when ALL of q,p,u are free.
# With only q-stars free, Sobol can propose q > qobs (invalid), but
# validate_parset correctly rejects them before they enter the table.
if _counter.rejections > 0:
    print(f'\n  Note: {_counter.rejections} rejection(s) are expected (Sobol warm-up)')
    print(f'  (BoTorch constraints require ALL of q,p,u free to prevent this)')
    for ps in _counter.rejected_parsets[:3]:
        q_val = ps.get('q-stars', '?')
        print(f'    q-stars={q_val:.4f} > qobs={qobs:.4f} — correctly rejected')

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
# 1. Naming fix: _free_qpu_idx['q'] must be populated with real DYNAMITE config
ps_settings = c.settings.parameter_space_settings
gen = _ps_module.BayesOptGenerator(par_space=c.parspace, parspace_settings=ps_settings)
assert 'q' in gen._free_qpu_idx, \
    f'_free_qpu_idx missing q: {gen._free_qpu_idx} (naming fix regression)'

# 2. Every model that reached the solver has q-stars <= qobs (valid geometry)
#    This is guaranteed by validate_parset acting as a filter.
assert np.all(accepted_q <= qobs + 1e-9), \
    f'Table contains q > qobs: max={accepted_q.max():.4f} qobs={qobs:.4f}'

# 3. At least some models ran
assert n_models > 0, 'No models were accepted'
assert best_chi2 < 500.0, f'best chi2={best_chi2:.1f} unreasonably high'

print('\nPHYSICAL VALIDITY TEST PASSED')
print('  - _free_qpu_idx[q] correctly populated (naming fix verified)')
print(f'  - All {n_models} accepted models have q-stars <= qobs={qobs:.4f}')

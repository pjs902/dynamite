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

print(f'\nResults:')
print(f'  Total models accepted:      {n_models}')
print(f'  validate_parset calls:      {_counter.total_calls}')
print(f'  validate_parset rejections: {_counter.rejections}')
print(f'  Best chi2 reached:          {best_chi2:.4f}')

if _counter.rejections > 0:
    print(f'\nREJECTED PARSETS (first 5):')
    for ps in _counter.rejected_parsets[:5]:
        print(f'  {ps}')

assert _counter.rejections == 0, (
    f'BayesOptGenerator proposed {_counter.rejections} physically invalid '
    f'model(s) out of {_counter.total_calls} validate_parset calls.\n'
    f'First rejected: {_counter.rejected_parsets[:3]}'
)
assert n_models > 0, 'No models were accepted'
assert best_chi2 < 500.0, f'best chi2={best_chi2:.1f} unreasonably high'

print('\nPHYSICAL VALIDITY TEST PASSED — zero invalid proposals')

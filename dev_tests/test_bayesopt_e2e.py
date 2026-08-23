"""End-to-end integration test for BayesOptGenerator."""
import sys
import types
import numpy as np  # type: ignore[import-untyped]

# --- stub-import trick (same as test_bayesopt_generator.py) ---
sys.path.insert(0, '/Users/pesmith/research/dynamite')
_stub_dyn = types.ModuleType('dynamite')
sys.modules.setdefault('dynamite', _stub_dyn)
import importlib.util as _ilu
_ps_spec = _ilu.spec_from_file_location(
    'dynamite.parameter_space',
    '/Users/pesmith/research/dynamite/dynamite/parameter_space.py',
)
_ps_mod = _ilu.module_from_spec(_ps_spec)
sys.modules['dynamite.parameter_space'] = _ps_mod
setattr(sys.modules['dynamite'], 'parameter_space', _ps_mod)
_ps_spec.loader.exec_module(_ps_mod)
import dynamite.parameter_space as ps

from astropy.table import Table, Column  # type: ignore[import-untyped]


# --------------------------------------------------------------------------
# Mock infrastructure (copied from test_bayesopt_generator.py)
# --------------------------------------------------------------------------
class MockComponent:
    def __init__(self, name='cmp'):
        self.name = name
        self.parameters = []

    def get_parname(self, name):
        return name

    def validate_parset(self, par):
        return True


class MockSystem:
    def __init__(self, params, components=None):
        self.parameters = []
        if components is None:
            cmp = MockComponent('test_cmp')
            cmp.parameters = list(params)
            self.cmp_list = [cmp]
        else:
            self.cmp_list = components

    def validate_parset(self, par):
        return True


def make_parspace(params, system=None):
    """Build a real ParameterSpace around a MockSystem holding `params`."""
    if system is None:
        system = MockSystem(params)
    return ps.ParameterSpace(system)


class MockAllModels:
    """AllModels stand-in with the columns BayesOptGenerator reads/writes."""
    def __init__(self, par_names, chi2_col='kinchi2'):
        self.table = Table()
        for name in par_names:
            self.table[name] = Column([], dtype=float)
        self.table['chi2'] = Column([], dtype=float)
        self.table['kinchi2'] = Column([], dtype=float)
        self.table['kinmapchi2'] = Column([], dtype=float)
        self.table['time_modified'] = Column([], dtype='U256')
        self.table['orblib_done'] = Column([], dtype=bool)
        self.table['weights_done'] = Column([], dtype=bool)
        self.table['all_done'] = Column([], dtype=bool)
        self.table['which_iter'] = Column([], dtype=int)
        self.table['directory'] = Column([], dtype='U256')


def _mk_param(name, lo, hi, value, logarithmic=False, fixed=False):
    return ps.Parameter(name=name, fixed=fixed, logarithmic=logarithmic,
                        value=value,
                        par_generator_settings={'lo': lo, 'hi': hi})


# --------------------------------------------------------------------------
# Landscape and evaluation helper
# --------------------------------------------------------------------------
def chi2_landscape(q, p, ml):
    """3D Gaussian: minimum at (q=0.4, p=0.6, ml=3.0), chi2=0."""
    return (q - 0.4)**2 / 0.01 + (p - 0.6)**2 / 0.01 + (ml - 3.0)**2 / 0.25


def evaluate_pending_models(am):
    """Evaluate any rows where all_done=False using the chi2 landscape."""
    t = am.table
    for i in range(len(t)):
        if not t['all_done'][i]:
            q_val = t['q'][i]
            p_val = t['p'][i]
            ml_val = t['ml'][i]
            chi2 = chi2_landscape(q_val, p_val, ml_val)
            t['kinchi2'][i] = chi2
            t['chi2'][i] = chi2
            t['kinmapchi2'][i] = chi2
            t['all_done'][i] = True
            t['orblib_done'][i] = True
            t['weights_done'][i] = True


# --------------------------------------------------------------------------
# E2E test
# --------------------------------------------------------------------------
def test_e2e_bayesopt_improves():
    """Run 6 generate->evaluate cycles and verify chi2 improves."""
    # Build parameter space: 3 free + 1 fixed
    q_par = _mk_param('q', 0.1, 0.9, 0.5)
    p_par = _mk_param('p', 0.1, 0.9, 0.5)
    ml_par = _mk_param('ml', 1.0, 5.0, 3.0)
    bh_par = _mk_param('bh', 1e7, 1e9, 1e8)
    bh_par.fixed = True

    par_space = make_parspace([q_par, p_par, ml_par, bh_par])

    settings = {
        'which_chi2': 'kinchi2',
        'generator_type': 'BayesOptGenerator',
        'generator_settings': {
            # n_initial_random < 2*batch_size so the GP phase starts by cycle 2
            'n_initial_random': 6,
            'batch_size': 4,
            'n_orblib_configs': 4,
            'n_ml_per_config': 1,
            'acquisition_type': 'qLogEI',
            'max_gp_variance_threshold': 1.0,
            'min_ei_threshold': -1.5,
        },
        'stopping_criteria': {
            'n_max_mods': 200,
            'n_max_iter': 30,
            # Set very permissive threshold so random-to-random comparisons
            # never trigger early stopping; we assert improvement ourselves.
            'min_delta_chi2_abs': -1000.0,
        },
    }

    gen = ps.BayesOptGenerator(par_space=par_space, parspace_settings=settings)
    am = MockAllModels(par_space.par_names)

    best_chi2_history = []

    for cycle in range(6):
        status = gen.generate(current_models=am)
        n_new = status.get('n_new_models', 0)
        if n_new == 0:
            print(f'  Cycle {cycle}: generator added 0 new models, stopping')
            break
        evaluate_pending_models(am)
        # find best so far
        t = am.table
        done_mask = np.asarray(t['all_done'], dtype=bool)
        chi2_vals = np.asarray(t['kinchi2'], dtype=float)
        valid = chi2_vals[done_mask & np.isfinite(chi2_vals)]
        if len(valid) > 0:
            best = float(np.min(valid))
            best_chi2_history.append(best)
            n_evals = int(np.sum(done_mask))
            print(f'  Cycle {cycle + 1}: best chi2={best:.4f}, n_evals={n_evals}')

    assert len(best_chi2_history) >= 2, \
        f'Need at least 2 cycles to test improvement, got {len(best_chi2_history)}'
    assert best_chi2_history[-1] < best_chi2_history[0], \
        f'chi2 did not improve: {best_chi2_history[0]:.4f} -> {best_chi2_history[-1]:.4f}'
    print(f'  chi2 improved from {best_chi2_history[0]:.4f} to {best_chi2_history[-1]:.4f}')
    print('test_e2e_bayesopt_improves PASSED')


if __name__ == '__main__':
    print('Task 8: end-to-end integration test')
    test_e2e_bayesopt_improves()
    print('ALL E2E TESTS PASSED')

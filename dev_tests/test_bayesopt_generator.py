"""Unit + integration tests for BayesOptGenerator and its helpers.

Run with:  /opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py
"""
import sys
import types
import copy  # noqa: F401 — used by BayesOptGenerator methods added in later tasks
import importlib.util as ilu
import numpy as np  # type: ignore[import-untyped]
from astropy.table import Table, Column  # type: ignore[import-untyped]

# --- Load dynamite.parameter_space without triggering dynamite/__init__ ----
DYN_ROOT = '/Users/pesmith/research/dynamite'
sys.path.insert(0, DYN_ROOT)
_stub = types.ModuleType('dynamite')
sys.modules.setdefault('dynamite', _stub)
_spec = ilu.spec_from_file_location(
    'dynamite.parameter_space',
    f'{DYN_ROOT}/dynamite/parameter_space.py')
if _spec is None:
    raise ImportError('Could not find parameter_space.py')
if _spec.loader is None:
    raise ImportError('spec.loader is None')
ps = ilu.module_from_spec(_spec)
sys.modules['dynamite.parameter_space'] = ps
setattr(sys.modules['dynamite'], 'parameter_space', ps)
_spec.loader.exec_module(ps)

Parameter = ps.Parameter
ParameterSpace = ps.ParameterSpace
extract_gp_training_data = ps.extract_gp_training_data
denormalize_to_raw = ps.denormalize_to_raw
raw_to_par_values = ps.raw_to_par_values


# --------------------------------------------------------------------------
# Mock infrastructure
# --------------------------------------------------------------------------
class MockComponent:
    def __init__(self, name='cmp'):
        self.name = name
        self.parameters = []

    def get_parname(self, name):
        return name

    def validate_parset(self, par):
        return True


class MockTriaxialComponent(MockComponent):
    """Stands in for TriaxialVisibleComponent; carries qobs."""
    def __init__(self, name='stars', qobs=0.65):
        super().__init__(name=name)
        self.qobs = qobs

# Make the class name match TriaxialVisibleComponent for the get_qobs_from_system check
MockTriaxialComponent.__name__ = 'TriaxialVisibleComponent'
MockTriaxialComponent.__qualname__ = 'TriaxialVisibleComponent'


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
    return ParameterSpace(system)


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
    return Parameter(name=name, fixed=fixed, logarithmic=logarithmic,
                     value=value,
                     par_generator_settings={'lo': lo, 'hi': hi})


# --------------------------------------------------------------------------
# Task 2 tests: pipeline round-trips and filtering
# --------------------------------------------------------------------------
def test_roundtrip_linear():
    ml = _mk_param('ml', 4.0, 6.0, 5.0, logarithmic=False)
    ps_ = make_parspace([ml])
    t = Table(names=['ml', 'kinchi2', 'all_done'],
              dtype=[float, float, bool])
    for v, c in [(4.5, 2.0), (5.0, 1.5), (5.5, 3.0)]:
        t.add_row([v, c, True])
    X, y, names, lo, hi = extract_gp_training_data(t, ps_, 'kinchi2')
    assert X.shape == (3, 1)
    assert names == ['ml']
    raw_back = denormalize_to_raw(X, lo, hi)
    np.testing.assert_allclose(raw_back.ravel(), [4.5, 5.0, 5.5], rtol=1e-12)
    print('  test_roundtrip_linear PASSED')


def test_roundtrip_log():
    f = _mk_param('f', -1.0, 3.0, 1.0, logarithmic=True)
    ps_ = make_parspace([f])
    t = Table(names=['f', 'kinchi2', 'all_done'], dtype=[float, float, bool])
    for v, c in [(1.0, 5.0), (10.0, 3.0), (100.0, 4.0)]:
        t.add_row([v, c, True])
    X, y, names, lo, hi = extract_gp_training_data(t, ps_, 'kinchi2')
    raw = denormalize_to_raw(X, lo, hi)
    np.testing.assert_allclose(raw.ravel(), [0.0, 1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(X.ravel(), [0.25, 0.5, 0.75], atol=1e-12)
    free = [p for p in ps_ if not p.fixed]
    par_back = raw_to_par_values(raw[1], free)
    np.testing.assert_allclose(par_back, [10.0], rtol=1e-12)
    print('  test_roundtrip_log PASSED')


def test_filtering():
    ml = _mk_param('ml', 1.0, 10.0, 5.0)
    ps_ = make_parspace([ml])
    t = Table(names=['ml', 'kinchi2', 'all_done'], dtype=[float, float, bool])
    rows = [(2.0, 1.0, True), (3.0, 2.0, False),
            (4.0, float('nan'), True), (5.0, 3.0, False), (6.0, 4.0, True)]
    for v, c, d in rows:
        t.add_row([v, c, d])
    X, y, names, lo, hi = extract_gp_training_data(t, ps_, 'kinchi2')
    assert X.shape[0] == 2, f'expected 2 valid rows, got {X.shape[0]}'
    np.testing.assert_allclose(y, [1.0, 4.0])
    print('  test_filtering PASSED')


# --------------------------------------------------------------------------
# Task 3 tests: BayesOptGenerator __init__
# --------------------------------------------------------------------------
def _bo_settings():
    return {
        'which_chi2': 'kinchi2',
        'generator_type': 'BayesOptGenerator',
        'generator_settings': {
            'batch_size': 8, 'n_orblib_configs': 4, 'n_ml_per_config': 2,
            'n_initial_random': 6, 'acquisition_type': 'qLogEI',
            'max_gp_variance_threshold': 1.0, 'min_ei_threshold': -1.5,
        },
        'stopping_criteria': {
            'n_max_mods': 200, 'n_max_iter': 30,
            'min_delta_chi2_abs': 0.001,
        },
    }


def test_init():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    f = _mk_param('f', -1.0, 3.0, 1.0, logarithmic=True)
    q = _mk_param('q', 0.05, 0.99, 0.6)
    ps_ = make_parspace([ml, f, q])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings())
    assert gen.batch_size == 8
    assert gen.n_initial_random == 6
    assert gen.free_param_names == ['ml', 'f', 'q']
    assert gen._gp_model is None
    assert gen.qobs is None  # MockComponent carries no qobs
    print('  test_init PASSED')


def test_init_rejects_double_delta():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['stopping_criteria']['min_delta_chi2_rel'] = 0.01  # now BOTH present
    try:
        ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    except ValueError:
        print('  test_init_rejects_double_delta PASSED')
        return
    raise AssertionError('expected ValueError for two chi2-delta options')


def test_init_rejects_missing_delta():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    del s['stopping_criteria']['min_delta_chi2_abs']  # now NEITHER present
    try:
        ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    except ValueError:
        print('  test_init_rejects_missing_delta PASSED')
        return
    raise AssertionError('expected ValueError for missing chi2-delta option')


# --------------------------------------------------------------------------
# Task 4 tests: random warm-up phase
# --------------------------------------------------------------------------
def test_random_phase_count_and_bounds():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    f = _mk_param('f', -1.0, 3.0, 1.0, logarithmic=True)
    q = _mk_param('q', 0.05, 0.99, 0.6)
    ps_ = make_parspace([ml, f, q])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings())
    am = MockAllModels(ps_.par_names)
    gen.current_models = am
    gen.specific_generate_method()
    assert len(gen.model_list) == gen.batch_size, \
        f'expected {gen.batch_size} models, got {len(gen.model_list)}'
    for model in gen.model_list:
        assert len(model) == ps_.n_par, \
            f'model has {len(model)} params, expected {ps_.n_par}'
        for p in model:
            if not p.fixed:
                j = gen.free_param_names.index(p.name)
                lo = gen.lo_free[j]
                hi = gen.hi_free[j]
                assert lo - 1e-9 <= p.raw_value <= hi + 1e-9, \
                    f'{p.name} raw_value {p.raw_value} out of [{lo},{hi}]'
    assert gen._gp_model is None, 'GP should not be fitted in random phase'
    print('  test_random_phase_count_and_bounds PASSED')


def test_random_phase_guard_empty_table():
    """Second call on iter=0 sees empty-ish table — must still return random."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['generator_settings']['batch_size'] = 4
    s['generator_settings']['n_ml_per_config'] = 1
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    am = MockAllModels(ps_.par_names)
    gen.current_models = am
    # Call twice (simulates iter-0 double-call)
    gen.specific_generate_method()
    first_batch = list(gen.model_list)
    gen.specific_generate_method()
    second_batch = list(gen.model_list)
    assert len(second_batch) == 4
    assert gen._gp_model is None
    print('  test_random_phase_guard_empty_table PASSED')


# --------------------------------------------------------------------------
# Task 5 tests: GP acquisition phase
# --------------------------------------------------------------------------
def _fill_table(gen, am, landscape, n, seed=0):
    """Add n completed valid models to am.table by evaluating landscape."""
    rng = np.random.default_rng(seed)
    lo = np.array(gen.lo_free, dtype=float)
    hi = np.array(gen.hi_free, dtype=float)
    for _ in range(n):
        raw = lo + (hi - lo) * rng.random(len(lo))
        model = gen._raw_free_to_model(raw)
        chi2 = landscape(raw)
        par_vals = [p.par_value for p in model]
        row = par_vals + [chi2, chi2, chi2, 'now', True, True, True, 0, '']
        am.table.add_row(row)


def test_gp_phase_count_and_bounds():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    f = _mk_param('f', -1.0, 3.0, 1.0, logarithmic=True)
    ps_ = make_parspace([ml, f])
    s = _bo_settings()
    s['generator_settings']['n_initial_random'] = 4
    s['generator_settings']['batch_size'] = 4
    s['generator_settings']['n_ml_per_config'] = 1
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    am = MockAllModels(ps_.par_names)
    gen.current_models = am

    def landscape(raw):
        c = np.array([5.0, 1.0])
        return float(np.sum((raw - c) ** 2))

    _fill_table(gen, am, landscape, n=6, seed=1)
    gen.specific_generate_method()
    assert gen._gp_model is not None, 'GP should be fitted in GP phase'
    assert len(gen.model_list) == gen.batch_size, \
        f'expected {gen.batch_size} candidates, got {len(gen.model_list)}'
    for model in gen.model_list:
        for p in model:
            if not p.fixed:
                j = gen.free_param_names.index(p.name)
                lo, hi = gen.lo_free[j], gen.hi_free[j]
                assert lo - 1e-5 <= p.raw_value <= hi + 1e-5, \
                    f'{p.name} raw={p.raw_value} out of [{lo},{hi}]'
    assert gen._last_acq_value is not None
    print('  test_gp_phase_count_and_bounds PASSED')


def test_gp_phase_triaxiality():
    """GP candidates must satisfy triaxiality when qobs is set."""
    q_p = _mk_param('q', 0.05, 0.99, 0.6)
    p_p = _mk_param('p', 0.05, 0.999, 0.8)
    u_p = _mk_param('u', 0.05, 1.0, 0.9)
    tri = MockTriaxialComponent('stars', qobs=0.65)
    tri.parameters = [q_p, p_p, u_p]
    system = MockSystem([q_p, p_p, u_p], components=[tri])
    ps_ = ParameterSpace(system)
    s = _bo_settings()
    s['generator_settings']['n_initial_random'] = 4
    s['generator_settings']['batch_size'] = 4
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    assert gen.qobs == 0.65
    am = MockAllModels(ps_.par_names)
    gen.current_models = am

    qobs = 0.65
    rng = np.random.default_rng(3)
    added = 0
    while added < 6:
        qv = rng.uniform(0.1, 0.8)
        pv = rng.uniform(max(qv, 0.1), 0.95)
        u_lo = max(qv / qobs, pv)
        u_hi = min(pv / qobs, 1.0)
        if u_hi <= u_lo:
            continue
        uv = rng.uniform(u_lo, u_hi)
        raw = np.array([qv, pv, uv])
        model = gen._raw_free_to_model(raw)
        chi2 = float(np.sum((raw - np.array([0.5, 0.7, 0.75])) ** 2))
        par_vals = [p.par_value for p in model]
        am.table.add_row(par_vals + [chi2, chi2, chi2, 'now', True, True, True, 0, ''])
        added += 1

    gen.specific_generate_method()
    assert len(gen.model_list) == gen.batch_size
    for model in gen.model_list:
        d = {p.name: p.raw_value for p in model}
        qv, pv, uv = d['q'], d['p'], d['u']
        assert pv >= qv - 1e-4, f'p={pv} < q={qv}'
        u_lo = max(qv / qobs, pv)
        u_hi = min(pv / qobs, 1.0)
        if u_hi > u_lo:
            assert u_lo - 1e-4 <= uv <= u_hi + 1e-4, \
                f'u={uv} not in [{u_lo},{u_hi}]'
    print('  test_gp_phase_triaxiality PASSED')


if __name__ == '__main__':
    print('Task 2: pipeline tests')
    test_roundtrip_linear()
    test_roundtrip_log()
    test_filtering()
    print('TASK 2 TESTS PASSED')
    print('Task 3: __init__ tests')
    test_init()
    test_init_rejects_double_delta()
    test_init_rejects_missing_delta()
    print('TASK 3 TESTS PASSED')
    print('Task 4: random-phase tests')
    test_random_phase_count_and_bounds()
    test_random_phase_guard_empty_table()
    print('TASK 4 TESTS PASSED')
    print('Task 5: GP-phase tests')
    test_gp_phase_count_and_bounds()
    test_gp_phase_triaxiality()
    print('TASK 5 TESTS PASSED')

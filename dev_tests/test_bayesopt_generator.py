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
# Task 1 tests: warmup_mode parsing
# --------------------------------------------------------------------------
def _bo_settings_axial(guess=None, step=0.1):
    """Settings dict for initial_guess warmup mode."""
    s = _bo_settings()
    s['generator_settings']['warmup_mode'] = 'initial_guess'
    s['generator_settings']['initial_step_size'] = step
    if guess is not None:
        s['generator_settings']['initial_guess'] = guess
    return s


def test_warmup_mode_default_is_sobol():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    assert gen.warmup_mode == 'sobol'
    assert gen.initial_step_size == 0.1
    assert gen._axial_queue == []
    print('  test_warmup_mode_default_is_sobol PASSED')


def test_warmup_mode_initial_guess_parsed():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 5.5}, step=0.15))
    assert gen.warmup_mode == 'initial_guess'
    assert gen.initial_step_size == 0.15
    # _axial_queue has 3 items (1 center + 2 axial steps for 1 free param)
    assert isinstance(gen._axial_queue, list)
    print('  test_warmup_mode_initial_guess_parsed PASSED')


def test_warmup_mode_invalid_raises():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['generator_settings']['warmup_mode'] = 'bad_mode'
    try:
        ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    except ValueError:
        print('  test_warmup_mode_invalid_raises PASSED')
        return
    raise AssertionError('expected ValueError for invalid warmup_mode')


# --------------------------------------------------------------------------
# Task 2 tests: _initial_guess_to_unit
# --------------------------------------------------------------------------
def test_initial_guess_to_unit_midpoint_default():
    """Parameters absent from initial_guess → 0.5."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    q = _mk_param('q-stars', 0.1, 0.9, 0.5)
    ps_ = make_parspace([ml, q])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(guess={}))
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [0.5, 0.5], atol=1e-12)
    print('  test_initial_guess_to_unit_midpoint_default PASSED')


def test_initial_guess_to_unit_linear():
    """ml=4.5 on [4,6] → normalized 0.25."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 4.5}))
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [0.25], atol=1e-12)
    print('  test_initial_guess_to_unit_linear PASSED')


def test_initial_guess_to_unit_log():
    """Logarithmic param f: raw bounds [0,2] (physical 1..100).
    Physical 10 → raw 1.0 → normalized (1-0)/(2-0) = 0.5."""
    f = _mk_param('f', 0.0, 2.0, 1.0, logarithmic=True)
    ps_ = make_parspace([f])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'f': 10.0}))
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [0.5], atol=1e-12)
    print('  test_initial_guess_to_unit_log PASSED')


def test_initial_guess_to_unit_clips():
    """Values outside bounds are clipped to [0,1]."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 99.0}))
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [1.0], atol=1e-12)
    print('  test_initial_guess_to_unit_clips PASSED')


# --------------------------------------------------------------------------
# Task 3 tests: _build_axial_queue and _propose_axial_batch
# --------------------------------------------------------------------------
def test_build_axial_queue_size():
    """1 + 2*n_free points in the queue."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    q = _mk_param('q-stars', 0.1, 0.9, 0.5)
    ps_ = make_parspace([ml, q])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 5.0, 'q-stars': 0.5}))
    assert len(gen._axial_queue) == 5  # 1 + 2*2
    print('  test_build_axial_queue_size PASSED')


def test_build_axial_queue_center_is_first():
    """First point is the normalized center."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)  # 5.0 on [4,6] → 0.5
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 5.0}, step=0.1))
    center = gen._axial_queue[0]
    np.testing.assert_allclose(center, [0.5], atol=1e-12)
    print('  test_build_axial_queue_center_is_first PASSED')


def test_build_axial_queue_axial_steps():
    """Points 1 and 2 are center ± step."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)  # center → 0.5
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 5.0}, step=0.1))
    np.testing.assert_allclose(gen._axial_queue[1], [0.6], atol=1e-12)
    np.testing.assert_allclose(gen._axial_queue[2], [0.4], atol=1e-12)
    print('  test_build_axial_queue_axial_steps PASSED')


def test_build_axial_queue_clips_at_boundary():
    """Step that exceeds [0,1] is clipped."""
    ml = _mk_param('ml', 4.0, 6.0, 6.0)  # center at raw 6.0 → normalized 1.0
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 6.0}, step=0.2))
    np.testing.assert_allclose(gen._axial_queue[1], [1.0], atol=1e-12)
    np.testing.assert_allclose(gen._axial_queue[2], [0.8], atol=1e-12)
    print('  test_build_axial_queue_clips_at_boundary PASSED')


def test_propose_axial_batch_pops_queue():
    """_propose_axial_batch pops batch_size items from the front."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings_axial(guess={'ml': 5.0}, step=0.1)
    s['generator_settings']['batch_size'] = 2
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    assert len(gen._axial_queue) == 3
    models = gen._propose_axial_batch()
    assert len(models) == 2
    assert len(gen._axial_queue) == 1
    print('  test_propose_axial_batch_pops_queue PASSED')


def test_propose_axial_batch_partial():
    """When fewer than batch_size points remain, proposes all remaining."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings_axial(guess={'ml': 5.0}, step=0.1)
    s['generator_settings']['batch_size'] = 4  # bigger than queue (3)
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    models = gen._propose_axial_batch()
    assert len(models) == 3
    assert len(gen._axial_queue) == 0
    print('  test_propose_axial_batch_partial PASSED')


def test_propose_axial_batch_raw_values():
    """First batch (center) gives correct raw ml value."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)  # center 0.5 → raw 5.0
    ps_ = make_parspace([ml])
    s = _bo_settings_axial(guess={'ml': 5.0}, step=0.1)
    s['generator_settings']['batch_size'] = 1
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    models = gen._propose_axial_batch()
    assert len(models) == 1
    ml_param = [p for p in models[0] if p.name == 'ml'][0]
    np.testing.assert_allclose(ml_param.raw_value, 5.0, atol=1e-10)
    print('  test_propose_axial_batch_raw_values PASSED')


# --------------------------------------------------------------------------
# Task 4 tests: specific_generate_method dispatch
# --------------------------------------------------------------------------
def _make_gen_axial(n_free=1, guess=None, step=0.1, batch_size=1):
    """Build a BayesOptGenerator in initial_guess mode with empty mock table."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    params = [ml]
    if n_free == 2:
        q = _mk_param('q-stars', 0.1, 0.9, 0.5)
        params.append(q)
    ps_ = make_parspace(params)
    names = [p.name for p in params]
    s = _bo_settings_axial(guess=guess or {'ml': 5.0}, step=step)
    s['generator_settings']['batch_size'] = batch_size
    s['generator_settings']['n_initial_random'] = 0
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    gen.current_models = MockAllModels(names)
    gen.chi2 = 'kinchi2'
    return gen


def test_generate_axial_uses_queue_not_sobol():
    """specific_generate_method pops from queue while it's non-empty."""
    gen = _make_gen_axial(batch_size=1)
    initial_queue_len = len(gen._axial_queue)  # 3 for 1 free param
    gen.specific_generate_method()
    assert len(gen._axial_queue) == initial_queue_len - 1
    assert len(gen.model_list) == 1
    print('  test_generate_axial_uses_queue_not_sobol PASSED')


def test_generate_axial_exhausts_queue_in_order():
    """All 3 axial points proposed before GP is attempted (1 free param)."""
    gen = _make_gen_axial(batch_size=1)
    queue_before = [q.copy() for q in gen._axial_queue]  # 3 points
    lo_raw, hi_raw = gen._norm_bounds_arrays()
    span = hi_raw[0] - lo_raw[0]
    expected_raws = [q[0] * span + lo_raw[0] for q in queue_before]
    proposed_raws = []
    for _ in range(3):
        gen.specific_generate_method()
        ml_val = [p for p in gen.model_list[0] if p.name == 'ml'][0].raw_value
        proposed_raws.append(ml_val)
    assert len(gen._axial_queue) == 0
    np.testing.assert_allclose(proposed_raws, expected_raws, atol=1e-10)
    print('  test_generate_axial_exhausts_queue_in_order PASSED')


def test_generate_sobol_mode_unchanged():
    """In sobol mode with n_valid=0, uses Sobol (not axial) and _gp_model stays None."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()   # default sobol mode, n_initial_random=6
    s['generator_settings']['batch_size'] = 2
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    gen.current_models = MockAllModels(['ml'])
    gen.chi2 = 'kinchi2'
    gen.specific_generate_method()
    assert len(gen.model_list) == 2
    assert gen._gp_model is None
    assert gen._axial_queue == []
    print('  test_generate_sobol_mode_unchanged PASSED')


# --------------------------------------------------------------------------
# Task 5 tests: random warm-up phase (previously mislabeled as Task 4)
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


# --------------------------------------------------------------------------
# Task 6 tests: stopping criteria
# --------------------------------------------------------------------------
def _make_fitted_gp(gen, n_points, cluster, seed):
    """Create a fitted SingleTaskGP for testing stopping criteria."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    rng = np.random.default_rng(seed)
    d = len(gen.free_par_idx)
    if cluster:
        X = np.clip(0.5 + 0.04 * rng.standard_normal((n_points, d)), 0, 1)
    else:
        X = rng.random((n_points, d))
    y = np.sum((X - 0.5) ** 2, axis=1)
    X_t = torch.tensor(X, dtype=torch.double)
    Y_t = -torch.tensor(y, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(X_t, Y_t).to(torch.double)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _bo_gen_2d():
    """Helper: 2-param BayesOptGenerator with one valid row in the table."""
    a = _mk_param('a', 0.0, 1.0, 0.5)
    b = _mk_param('b', 0.0, 1.0, 0.5)
    ps_ = make_parspace([a, b])
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    am = MockAllModels(ps_.par_names)
    am.table.add_row([0.5, 0.5, 1.0, 1.0, 1.0, 'now', True, True, True, 0, ''])
    gen.current_models = am
    return gen


def test_stopping_no_gp_is_safe():
    gen = _bo_gen_2d()
    gen._gp_model = None
    gen.status = {'stop': False}
    gen.check_specific_stopping_criteria()
    assert 'gp_max_variance_low' not in gen.status
    assert 'gp_min_ei_low' not in gen.status
    print('  test_stopping_no_gp_is_safe PASSED')


def test_stopping_not_converged():
    gen = _bo_gen_2d()
    gen._gp_model = _make_fitted_gp(gen, n_points=20, cluster=False, seed=5)
    gen._last_acq_value = 0.4
    # Sparse GP on 2D unit cube: max posterior variance ~0.003 (after Standardize).
    # Set threshold well below that so gp_max_variance_low is False.
    gen.max_gp_variance_threshold = 1e-5
    gen.status = {'stop': False}
    gen.check_specific_stopping_criteria()
    assert gen.status.get('gp_max_variance_low') is False, gen.status
    assert gen.status.get('gp_min_ei_low') is False, gen.status
    print('  test_stopping_not_converged PASSED')


def test_stopping_converged():
    gen = _bo_gen_2d()
    gen._gp_model = _make_fitted_gp(gen, n_points=100, cluster=True, seed=7)
    gen._last_acq_value = -3.0
    # Clustered GP: max posterior variance ~1e-5 (concentrated training data).
    # Set threshold above that so gp_max_variance_low is True.
    gen.max_gp_variance_threshold = 1e-4
    gen.status = {'stop': False}
    gen.check_specific_stopping_criteria()
    assert gen.status.get('gp_max_variance_low') is True, \
        f'expected variance low=True, got status={gen.status}'
    assert gen.status.get('gp_min_ei_low') is True, gen.status
    print('  test_stopping_converged PASSED')


def test_free_qpu_idx_with_suffixed_names():
    """_free_qpu_idx must be populated when param names carry a component suffix
    (e.g. 'q-stars'), as happens in all real DYNAMITE configs."""
    q = _mk_param('q-stars', 0.05, 0.99, 0.6)
    p_ = _mk_param('p-stars', 0.05, 0.999, 0.8)
    u = _mk_param('u-stars', 0.05, 1.0, 0.9)
    ml = _mk_param('ml', 1.0, 9.0, 5.0)
    tri = MockTriaxialComponent('stars', qobs=0.55)
    tri.parameters = [q, p_, u]
    system = MockSystem([ml], components=[tri])
    ps_ = ParameterSpace(system)
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    assert gen._free_qpu_idx == {'q': 0, 'p': 1, 'u': 2}, \
        f'Expected qpu idx {{q:0,p:1,u:2}}, got {gen._free_qpu_idx}'
    assert abs(gen.qobs - 0.55) < 1e-9, f'Expected qobs=0.55, got {gen.qobs}'
    print('  test_free_qpu_idx_with_suffixed_names PASSED')


# --------------------------------------------------------------------------
# Task 5 tests: dummy-mode integration for initial_guess warm-up
# --------------------------------------------------------------------------
def test_axial_warmup_dummy_run():
    """Full dummy loop: 2 free params → 5 axial proposals before GP."""
    import yaml
    import tempfile
    import os
    import importlib

    # The module-level setup stubs 'dynamite' to avoid loading the full package.
    # For this integration test we need the real package. Remove the stub
    # temporarily, import the real dynamite, then reload the submodules so
    # the stub in sys.modules['dynamite'] is replaced by the real package.
    import sys as _sys

    # The test file stubs sys.modules['dynamite'] to avoid loading the full
    # package. ModelIterator uses multiprocess, whose workers deserialize
    # pickled objects referencing dynamite submodules (e.g. dynamite.data).
    # The real package must stay in sys.modules for the entire ModelIterator
    # call — restore the stub only after the run completes.
    _stub = _sys.modules.pop('dynamite', None)
    import dynamite as _dyn
    _sys.modules['dynamite'] = _dyn
    config_reader = importlib.import_module('dynamite.config_reader')
    model_iterator = importlib.import_module('dynamite.model_iterator')

    try:
        base_yaml = '/Users/pesmith/research/dynamite/dev_tests/bayesopt_qml_modelinner.yaml'
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f)

        gs = cfg['parameter_space_settings']['generator_settings']
        gs['warmup_mode'] = 'initial_guess'
        gs['initial_guess'] = {'ml': 5.0, 'q-stars': 0.5}
        gs['initial_step_size'] = 0.1
        gs['batch_size'] = 2
        gs['n_orblib_configs'] = 2
        gs['n_ml_per_config'] = 1
        cfg['parameter_space_settings']['stopping_criteria']['n_max_mods'] = 12
        cfg['parameter_space_settings']['stopping_criteria']['n_max_iter'] = 10
        # min_delta_chi2 fires when an axial probe explores a bad direction; set to
        # a large negative so it never triggers (exactly one must be present)
        cfg['parameter_space_settings']['stopping_criteria']['min_delta_chi2_abs'] = -1e6
        cfg['parameter_space_settings']['stopping_criteria'].pop('min_delta_chi2_rel', None)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg['io_settings']['output_directory'] = tmpdir + '/'
            cfg['io_settings']['input_directory'] = (
                '/Users/pesmith/research/dynamite/dev_tests/NGC6278_input/')
            cfg_path = os.path.join(tmpdir, 'test_axial.yaml')
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg, f)

            def _chi2(parset):
                q  = float(parset['q-stars'])
                ml = float(parset['ml'])
                return 80.0 * (q - 0.4)**2 + 200.0 * (ml - 5.5)**2 + 15.0

            print('  [axial test] Config loaded, starting dummy ModelIterator...')
            c = config_reader.Configuration(cfg_path, reset_logging=False)
            model_iterator.ModelIterator(c, do_dummy_run=True,
                                         dummy_chi2_function=_chi2,
                                         plots=False)

            table = c.all_models.table
            done = [row for row in table if row['all_done']]
            print(f'  [axial test] Completed models: {len(done)}')
            for i, row in enumerate(done):
                q_val = float(row['q-stars']) if 'q-stars' in table.colnames else float('nan')
                print(f'    [{i}] q={q_val:.3f}  ml={float(row["ml"]):.3f}  '
                      f'kinchi2={float(row["kinchi2"]):.2f}  '
                      f'phase={"axial" if i < 5 else "GP"}')
            assert len(done) >= 5, (
                f'Expected ≥5 completed models (1+2*2=5 axial points), '
                f'got {len(done)}')
            finite_chi2 = [r for r in done if np.isfinite(float(r['kinchi2']))]
            assert len(finite_chi2) >= 1, 'Expected at least 1 finite kinchi2'
    finally:
        # Restore stub so remaining unit tests are unaffected
        if _stub is not None:
            _sys.modules['dynamite'] = _stub

    print('  test_axial_warmup_dummy_run PASSED')


if __name__ == '__main__':
    print('Task 2: pipeline tests')
    test_roundtrip_linear()
    test_roundtrip_log()
    test_filtering()
    print('TASK 2 TESTS PASSED')
    print('Task 3a: __init__ tests')
    test_init()
    test_init_rejects_double_delta()
    test_init_rejects_missing_delta()
    print('TASK 3a TESTS PASSED')
    print('Task 1: warmup_mode tests')
    test_warmup_mode_default_is_sobol()
    test_warmup_mode_initial_guess_parsed()
    test_warmup_mode_invalid_raises()
    print('TASK 1 TESTS PASSED')
    print('Task 2b: initial_guess_to_unit tests')
    test_initial_guess_to_unit_midpoint_default()
    test_initial_guess_to_unit_linear()
    test_initial_guess_to_unit_log()
    test_initial_guess_to_unit_clips()
    print('TASK 2b TESTS PASSED')
    print('Task 3: _build_axial_queue and _propose_axial_batch tests')
    test_build_axial_queue_size()
    test_build_axial_queue_center_is_first()
    test_build_axial_queue_axial_steps()
    test_build_axial_queue_clips_at_boundary()
    test_propose_axial_batch_pops_queue()
    test_propose_axial_batch_partial()
    test_propose_axial_batch_raw_values()
    print('TASK 3 TESTS PASSED')
    print('Task 4: random-phase tests')
    test_random_phase_count_and_bounds()
    test_random_phase_guard_empty_table()
    print('TASK 4 TESTS PASSED')
    print('Task 5: GP-phase tests')
    test_gp_phase_count_and_bounds()
    test_gp_phase_triaxiality()
    print('TASK 5 TESTS PASSED')
    print('Task 6: stopping-criteria tests')
    test_stopping_no_gp_is_safe()
    test_stopping_not_converged()
    test_stopping_converged()
    print('TASK 6 TESTS PASSED')
    print('Name fix: qpu suffix regression test')
    test_free_qpu_idx_with_suffixed_names()
    print('NAME FIX TESTS PASSED')
    print('Task 5 integration: axial dummy-mode test')
    test_axial_warmup_dummy_run()
    print('TASK 5 INTEGRATION TESTS PASSED')
    print('ALL BAYESOPT TESTS PASSED')

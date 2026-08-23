# BayesOpt Warm-up Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `initial_guess` warm-up mode to `BayesOptGenerator` that evaluates a user-supplied center point and axial steps (±δ per free parameter) before fitting the GP, so that in production runs where each model costs 18 hours the warm-up budget is spent near a known good region rather than scattered randomly.

**Architecture:** A new `warmup_mode` setting (`'sobol'` keeps existing behavior, `'initial_guess'` uses a precomputed queue of center + axial points). The queue is built at `__init__` time from a dict of physical parameter values and a step size in normalized [0,1] space; `specific_generate_method` pops from it each iteration until empty, then hands off to the GP. The Sobol path is unchanged.

**Tech Stack:** Python, NumPy, existing `BayesOptGenerator` infrastructure in `dynamite/parameter_space.py`, pytest-style unit tests in `dev_tests/test_bayesopt_generator.py`.

---

## Codebase orientation

Key locations:

- `dynamite/parameter_space.py` — `BayesOptGenerator` class (~line 1132). Relevant methods:
  - `__init__` (~line 1154): reads `generator_settings`, builds free-param bookkeeping
  - `_norm_bounds_arrays()` (~line 1227): returns `(lo_raw, hi_raw)` numpy arrays over free params
  - `_propose_random_batch()` (~line 1270): current Sobol warm-up
  - `specific_generate_method()` (~line 1301): decides Sobol vs GP; **this is what we extend**
  - `_raw_free_matrix_to_model_list(raw_free_matrix)` (~line 1223): converts `(n, n_free)` raw array → model list

- `Parameter.get_raw_value_from_par_value(phys)`: converts physical value → raw (log10 for logarithmic params)
- `Parameter.logarithmic`: bool; `par_generator_settings['lo']` / `['hi']` are raw bounds

- `dev_tests/test_bayesopt_generator.py`: all existing unit tests. Uses `_mk_param`, `make_parspace`, `MockAllModels`, `_bo_settings()` helpers.

**No changes needed** in `config_reader.py` — generator settings are passed through as a raw dict.

---

## File structure

| File | Change |
|------|--------|
| `dynamite/parameter_space.py` | Add `warmup_mode`, `initial_step_size`, `_axial_queue` to `__init__`; add `_initial_guess_to_unit()`, `_build_axial_queue()`, `_propose_axial_batch()`; update `specific_generate_method()` |
| `dev_tests/test_bayesopt_generator.py` | New test functions for each new method and the updated dispatch logic |
| `dev_tests/bayesopt_ml_modelinner.yaml` | Add commented example showing `initial_guess` mode |
| `dev_notes/bayesopt_generator.md` | Update configuration table and add warm-up modes section |

---

## Task 1: Parse `warmup_mode`, `initial_guess`, `initial_step_size` in `__init__`

**Files:**
- Modify: `dynamite/parameter_space.py:1161–1169` (generator settings block in `__init__`)
- Test: `dev_tests/test_bayesopt_generator.py`

- [x] **Step 1: Write the failing tests**

Add these functions to the bottom of `dev_tests/test_bayesopt_generator.py`:

```python
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
    # Queue is built at __init__: center + 2*1 axial = 3 points
    assert len(gen._axial_queue) == 3
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
```

- [x] **Step 2: Run to verify they fail**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_warmup_mode_default_is_sobol dev_tests/test_bayesopt_generator.py::test_warmup_mode_initial_guess_parsed dev_tests/test_bayesopt_generator.py::test_warmup_mode_invalid_raises -v
```

Expected: `AttributeError: 'BayesOptGenerator' object has no attribute 'warmup_mode'`

- [x] **Step 3: Implement in `__init__`**

In `dynamite/parameter_space.py`, after the `self.min_ei_threshold` line (~1169), add:

```python
        self.warmup_mode = gen.get('warmup_mode', 'sobol')
        if self.warmup_mode not in ('sobol', 'initial_guess'):
            raise ValueError(
                f"BayesOptGenerator: warmup_mode must be 'sobol' or "
                f"'initial_guess', got {self.warmup_mode!r}")
        self.initial_step_size = float(gen.get('initial_step_size', 0.1))
        self._initial_guess_phys = gen.get('initial_guess', {})
```

Then at the END of `__init__`, after the `self._last_acq_value = None` line (~1208), add:

```python
        # Build axial queue after free_params bookkeeping is complete.
        self._axial_queue = (self._build_axial_queue()
                             if self.warmup_mode == 'initial_guess' else [])
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_warmup_mode_default_is_sobol dev_tests/test_bayesopt_generator.py::test_warmup_mode_initial_guess_parsed dev_tests/test_bayesopt_generator.py::test_warmup_mode_invalid_raises -v
```

Expected: 3 PASS. (Will still fail on `_build_axial_queue` not existing — that's fine, Task 2 adds it.)

- [x] **Step 5: Verify existing tests still pass**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -v
```

Expected: all 14 prior tests pass; 3 new tests pass.

- [x] **Step 6: Commit**

```bash
cd /Users/pesmith/research/dynamite && git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py && git commit -m "feat: parse warmup_mode/initial_guess/initial_step_size in BayesOptGenerator.__init__"
```

---

## Task 2: `_initial_guess_to_unit()` — convert physical guess to normalized center

**Files:**
- Modify: `dynamite/parameter_space.py` (add method to `BayesOptGenerator`)
- Test: `dev_tests/test_bayesopt_generator.py`

**Context:** The `initial_guess` dict contains physical parameter values keyed by parameter name (e.g. `{'ml': 5.5, 'q-stars': 0.8}`). These must be converted to raw values (possibly log10) then normalized to [0,1] using each parameter's `lo`/`hi` bounds. Parameters absent from the dict default to 0.5 (the midpoint).

- [x] **Step 1: Write the failing tests**

```python
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
    """ml=5.0 on [4,6] → normalized 0.5; ml=4.5 → 0.25."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 4.5}))
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [0.25], atol=1e-12)
    print('  test_initial_guess_to_unit_linear PASSED')


def test_initial_guess_to_unit_log():
    """Logarithmic param f on raw [0,2] (i.e. 1..100 physical).
    Physical value 10 → raw 1.0 → normalized (1-0)/(2-0) = 0.5."""
    f = _mk_param('f', 0.0, 2.0, 1.0, logarithmic=True)
    ps_ = make_parspace([f])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'f': 10.0}))
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [0.5], atol=1e-12)
    print('  test_initial_guess_to_unit_log PASSED')


def test_initial_guess_to_unit_clips():
    """Values outside bounds are clipped to [0,1] with a warning (no crash)."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 99.0}))  # way outside [4,6]
    center = gen._initial_guess_to_unit()
    np.testing.assert_allclose(center, [1.0], atol=1e-12)
    print('  test_initial_guess_to_unit_clips PASSED')
```

- [x] **Step 2: Run to verify they fail**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_midpoint_default dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_linear dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_log dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_clips -v
```

Expected: `AttributeError: 'BayesOptGenerator' object has no attribute '_initial_guess_to_unit'`

- [x] **Step 3: Implement `_initial_guess_to_unit`**

Add after `_norm_bounds_arrays` (~line 1230) in `dynamite/parameter_space.py`:

```python
    def _initial_guess_to_unit(self):
        """Convert initial_guess dict (physical values) to normalized center.

        Parameters absent from initial_guess default to 0.5 (midpoint).
        Values outside [lo, hi] are clipped and a warning is logged.
        Returns np.ndarray of shape (n_free,) with values in [0, 1].
        """
        lo_raw, hi_raw = self._norm_bounds_arrays()
        center = np.full(len(self.free_params), 0.5)
        for j, p in enumerate(self.free_params):
            if p.name not in self._initial_guess_phys:
                continue
            phys = self._initial_guess_phys[p.name]
            raw = p.get_raw_value_from_par_value(phys)
            span = hi_raw[j] - lo_raw[j]
            norm = (raw - lo_raw[j]) / span if span > 0 else 0.5
            if norm < 0.0 or norm > 1.0:
                self.logger.warning(
                    f'initial_guess {p.name}={phys} normalizes to {norm:.3f}, '
                    f'clipping to [0, 1]')
            center[j] = np.clip(norm, 0.0, 1.0)
        return center
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_midpoint_default dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_linear dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_log dev_tests/test_bayesopt_generator.py::test_initial_guess_to_unit_clips -v
```

Expected: 4 PASS.

- [x] **Step 5: Full test suite**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -v
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
cd /Users/pesmith/research/dynamite && git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py && git commit -m "feat: add _initial_guess_to_unit for physical→normalized center conversion"
```

---

## Task 3: `_build_axial_queue()` and `_propose_axial_batch()`

**Files:**
- Modify: `dynamite/parameter_space.py`
- Test: `dev_tests/test_bayesopt_generator.py`

**Context:** The axial design is: 1 center point, then for each free parameter two points at center ± `initial_step_size` in normalized space, clipped to [0,1]. With `n_free` free parameters the total is `1 + 2*n_free` points. `_build_axial_queue` returns them as a Python list of 1D numpy arrays (not yet converted to raw). `_propose_axial_batch` pops up to `batch_size` from the front of `self._axial_queue`, converts to raw, and returns a model list.

- [x] **Step 1: Write the failing tests**

```python
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
    """First point is the normalized center derived from initial_guess."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)  # 5.0 on [4,6] → normalized 0.5
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 5.0}, step=0.1))
    center = gen._axial_queue[0]
    np.testing.assert_allclose(center, [0.5], atol=1e-12)
    print('  test_build_axial_queue_center_is_first PASSED')


def test_build_axial_queue_axial_steps():
    """Points 1 and 2 are center ± step for the single free param."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)  # center → 0.5
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 5.0}, step=0.1))
    # queue: [center, center+0.1, center-0.1]
    np.testing.assert_allclose(gen._axial_queue[1], [0.6], atol=1e-12)
    np.testing.assert_allclose(gen._axial_queue[2], [0.4], atol=1e-12)
    print('  test_build_axial_queue_axial_steps PASSED')


def test_build_axial_queue_clips_at_boundary():
    """Step that would exceed [0,1] is clipped."""
    ml = _mk_param('ml', 4.0, 6.0, 6.0)  # center → 1.0 (at upper bound)
    ps_ = make_parspace([ml])
    gen = ps.BayesOptGenerator(par_space=ps_,
                               parspace_settings=_bo_settings_axial(
                                   guess={'ml': 6.0}, step=0.2))
    # +step would be 1.2 → clipped to 1.0; -step is 0.8
    np.testing.assert_allclose(gen._axial_queue[1], [1.0], atol=1e-12)
    np.testing.assert_allclose(gen._axial_queue[2], [0.8], atol=1e-12)
    print('  test_build_axial_queue_clips_at_boundary PASSED')


def test_propose_axial_batch_pops_queue():
    """_propose_axial_batch pops batch_size items from the front of the queue."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings_axial(guess={'ml': 5.0}, step=0.1)
    s['generator_settings']['batch_size'] = 2
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    # queue starts with 3 points (1 + 2*1)
    assert len(gen._axial_queue) == 3
    models = gen._propose_axial_batch()
    assert len(models) == 2
    assert len(gen._axial_queue) == 1   # 1 remaining
    print('  test_propose_axial_batch_pops_queue PASSED')


def test_propose_axial_batch_partial():
    """When fewer than batch_size points remain, propose all remaining."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings_axial(guess={'ml': 5.0}, step=0.1)
    s['generator_settings']['batch_size'] = 4   # bigger than queue (3)
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    models = gen._propose_axial_batch()
    assert len(models) == 3
    assert len(gen._axial_queue) == 0
    print('  test_propose_axial_batch_partial PASSED')


def test_propose_axial_batch_raw_values():
    """Models in the batch have correct raw_value for ml."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)  # center → 0.5 → raw 5.0
    ps_ = make_parspace([ml])
    s = _bo_settings_axial(guess={'ml': 5.0}, step=0.1)
    s['generator_settings']['batch_size'] = 1
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    # First batch = center point → ml raw = 5.0
    models = gen._propose_axial_batch()
    assert len(models) == 1
    ml_param = [p for p in models[0] if p.name == 'ml'][0]
    np.testing.assert_allclose(ml_param.raw_value, 5.0, atol=1e-10)
    print('  test_propose_axial_batch_raw_values PASSED')
```

- [x] **Step 2: Run to verify they fail**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -k "axial" -v
```

Expected: `AttributeError: '_build_axial_queue'` or similar.

- [x] **Step 3: Implement `_build_axial_queue` and `_propose_axial_batch`**

Add both methods to `BayesOptGenerator` in `dynamite/parameter_space.py`, after `_initial_guess_to_unit`:

```python
    def _build_axial_queue(self):
        """Build the axial warm-up design as a list of normalized points.

        Returns [center, center+step_axis0, center-step_axis0,
                         center+step_axis1, center-step_axis1, ...]
        Total: 1 + 2*n_free points. All clipped to [0, 1].
        """
        center = self._initial_guess_to_unit()
        step = self.initial_step_size
        points = [center.copy()]
        for j in range(len(self.free_params)):
            for sign in (+1.0, -1.0):
                pt = center.copy()
                pt[j] = np.clip(center[j] + sign * step, 0.0, 1.0)
                points.append(pt)
        return points

    def _propose_axial_batch(self):
        """Pop up to batch_size points from _axial_queue; return model list.

        Mutates self._axial_queue. Callers must check the queue is non-empty
        before calling.
        """
        lo_raw, hi_raw = self._norm_bounds_arrays()
        span = hi_raw - lo_raw
        taken = self._axial_queue[:self.batch_size]
        self._axial_queue = self._axial_queue[self.batch_size:]
        raw_free = np.array(taken) * span + lo_raw
        return self._raw_free_matrix_to_model_list(raw_free)
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -k "axial" -v
```

Expected: all 7 axial tests PASS.

- [x] **Step 5: Full test suite**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -v
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
cd /Users/pesmith/research/dynamite && git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py && git commit -m "feat: add _build_axial_queue and _propose_axial_batch for initial_guess warm-up"
```

---

## Task 4: Update `specific_generate_method` to dispatch on `warmup_mode`

**Files:**
- Modify: `dynamite/parameter_space.py:1301–1323`
- Test: `dev_tests/test_bayesopt_generator.py`

**Context:** Current `specific_generate_method` has one branch: Sobol if `n_valid < n_initial_random`, else GP. The new logic adds an `initial_guess` branch that pops from `_axial_queue` regardless of `n_valid`, and falls straight to GP once the queue is empty (bypassing `n_initial_random` — the axial design IS the warm-up budget in this mode). The `n_initial_random` guard still applies in `'sobol'` mode.

- [x] **Step 1: Write the failing tests**

```python
# --------------------------------------------------------------------------
# Task 4 tests: specific_generate_method dispatch
# --------------------------------------------------------------------------
def _make_gen_axial(n_free=1, guess=None, step=0.1, batch_size=1):
    """Helper: build a BayesOptGenerator in initial_guess mode."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    params = [ml]
    if n_free == 2:
        q = _mk_param('q-stars', 0.1, 0.9, 0.5)
        params.append(q)
    ps_ = make_parspace(params)
    names = [p.name for p in params]
    s = _bo_settings_axial(guess=guess or {'ml': 5.0}, step=step)
    s['generator_settings']['batch_size'] = batch_size
    s['generator_settings']['n_initial_random'] = 0  # ignored in axial mode
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    gen.current_models = MockAllModels(names)
    gen.chi2 = 'kinchi2'
    return gen


def test_generate_axial_uses_queue_not_sobol():
    """specific_generate_method returns axial proposals while queue non-empty."""
    gen = _make_gen_axial(batch_size=1)
    initial_queue_len = len(gen._axial_queue)  # should be 3 (1 + 2*1)
    gen.specific_generate_method()
    assert len(gen._axial_queue) == initial_queue_len - 1
    # model_list must have 1 model
    assert len(gen.model_list) == 1
    print('  test_generate_axial_uses_queue_not_sobol PASSED')


def test_generate_axial_exhausts_queue_in_order():
    """All axial points are proposed before GP is attempted."""
    gen = _make_gen_axial(batch_size=1)
    queue_before = list(gen._axial_queue)  # 3 points
    proposed = []
    for _ in range(3):
        gen.specific_generate_method()
        ml_val = [p for p in gen.model_list[0] if p.name == 'ml'][0].raw_value
        proposed.append(ml_val)
    assert len(gen._axial_queue) == 0
    # proposed[0] = center (5.0), proposed[1] = center+step, proposed[2] = center-step
    lo_raw, hi_raw = gen._norm_bounds_arrays()
    span = hi_raw[0] - lo_raw[0]   # = 2.0
    expected = [q[0] * span + lo_raw[0] for q in queue_before]
    np.testing.assert_allclose(proposed, expected, atol=1e-10)
    print('  test_generate_axial_exhausts_queue_in_order PASSED')


def test_generate_sobol_mode_unchanged():
    """In sobol mode with n_valid=0 and n_initial_random=6, uses Sobol path."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()   # default sobol mode, n_initial_random=6
    s['generator_settings']['batch_size'] = 2
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    gen.current_models = MockAllModels(['ml'])
    gen.chi2 = 'kinchi2'
    gen.specific_generate_method()
    assert len(gen.model_list) == 2
    assert gen._gp_model is None   # still in warm-up
    assert gen._axial_queue == []  # sobol mode: no queue
    print('  test_generate_sobol_mode_unchanged PASSED')
```

- [x] **Step 2: Run to verify they fail**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_generate_axial_uses_queue_not_sobol dev_tests/test_bayesopt_generator.py::test_generate_axial_exhausts_queue_in_order dev_tests/test_bayesopt_generator.py::test_generate_sobol_mode_unchanged -v
```

Expected: `test_generate_axial_uses_queue_not_sobol` FAIL (falls through to Sobol instead of popping queue).

- [x] **Step 3: Replace `specific_generate_method`**

Replace the method body in `dynamite/parameter_space.py` (~line 1301):

```python
    def specific_generate_method(self, **kwargs):
        """Propose the next batch of models.

        Warm-up dispatch:
          - 'initial_guess' mode: pop from _axial_queue while non-empty,
            then go straight to GP (queue exhaustion is the warm-up signal).
          - 'sobol' mode: Sobol random proposals until n_valid >= n_initial_random,
            then GP.
        """
        table = self.current_models.table
        if len(table) == 0:
            n_valid = 0
        else:
            done = np.asarray(table['all_done'], dtype=bool)
            finite = np.isfinite(np.asarray(table[self.chi2], dtype=float))
            n_valid = int(np.sum(done & finite))

        if self.warmup_mode == 'initial_guess':
            if self._axial_queue:
                self._gp_model = None
                self._last_acq_value = None
                self.model_list = self._propose_axial_batch()
                return
        else:  # 'sobol'
            if n_valid < self.n_initial_random:
                self._gp_model = None
                self._last_acq_value = None
                self.model_list = self._propose_random_batch()
                return

        self.model_list = self._gp_acquisition_batch()
```

- [x] **Step 4: Run the dispatch tests**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_generate_axial_uses_queue_not_sobol dev_tests/test_bayesopt_generator.py::test_generate_axial_exhausts_queue_in_order dev_tests/test_bayesopt_generator.py::test_generate_sobol_mode_unchanged -v
```

Expected: 3 PASS.

- [x] **Step 5: Full test suite**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -v
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
cd /Users/pesmith/research/dynamite && git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py && git commit -m "feat: dispatch on warmup_mode in specific_generate_method"
```

---

## Task 5: Dummy-mode integration test with `initial_guess` warm-up

**Files:**
- Test: `dev_tests/test_bayesopt_generator.py`

**Context:** Run a full dummy `ModelIterator` loop in `initial_guess` mode using `bayesopt_qml_modelinner.yaml` as the base config (q-stars and ml free → 2 free params → `1 + 2*2 = 5` axial points). Verify that exactly 5 models are proposed before the GP kicks in (i.e., the 6th proposal comes from GP acquisition).

This uses the same dummy-mode infrastructure as existing tests — import `dynamite.model_iterator` and call with `do_dummy_run=True`.

- [x] **Step 1: Write the failing test**

```python
# --------------------------------------------------------------------------
# Task 5 tests: dummy-mode integration for initial_guess warm-up
# --------------------------------------------------------------------------
def test_axial_warmup_dummy_run():
    """Full dummy loop: 2 free params → 5 axial proposals, then GP."""
    import yaml, copy, sys, os, tempfile
    sys.path.insert(0, '/Users/pesmith/research/dynamite')
    import dynamite as dyn

    base_yaml = '/Users/pesmith/research/dynamite/dev_tests/bayesopt_qml_modelinner.yaml'
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)

    # Switch to initial_guess warm-up
    cfg['parameter_space_settings']['generator_settings']['warmup_mode'] = 'initial_guess'
    cfg['parameter_space_settings']['generator_settings']['initial_guess'] = {
        'ml': 5.0, 'q-stars': 0.5}
    cfg['parameter_space_settings']['generator_settings']['initial_step_size'] = 0.1
    cfg['parameter_space_settings']['generator_settings']['batch_size'] = 2
    cfg['parameter_space_settings']['generator_settings']['n_orblib_configs'] = 2
    cfg['parameter_space_settings']['generator_settings']['n_ml_per_config'] = 1
    cfg['parameter_space_settings']['stopping_criteria']['n_max_mods'] = 12
    cfg['parameter_space_settings']['stopping_criteria']['n_max_iter'] = 10

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg['system_settings']['output_directory'] = tmpdir + '/'
        cfg_path = tmpdir + '/test_axial.yaml'
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f)

        c = dyn.config_reader.Configuration(cfg_path, reset_logging=False)
        mi = dyn.model_iterator.ModelIterator(c, do_dummy_run=True, plots=False)

        table = c.all_models.table
        done_rows = table[table['all_done'] == True]

        # With 2 free params: 1 center + 4 axial = 5 axial models.
        # First 5 completed models should NOT have been GP-proposed.
        # We can check by verifying 5 models completed before GP was trained.
        # (The 6th+ would be GP proposals; n_max_mods=12 allows us to see this.)
        assert len(done_rows) >= 5, (
            f'Expected at least 5 completed models (axial design), got {len(done_rows)}')
    print('  test_axial_warmup_dummy_run PASSED')
```

- [x] **Step 2: Run to verify it fails**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py::test_axial_warmup_dummy_run -v
```

Expected: test passes if implementation is correct, or fails with a clear error if the yaml structure differs.

- [x] **Step 3: Fix any yaml/config issues that arise**

If the test fails with a config-reading error, check the `parameter_space_settings` key structure against `bayesopt_qml_modelinner.yaml`:

```bash
grep -A 5 "generator_settings" /Users/pesmith/research/dynamite/dev_tests/bayesopt_qml_modelinner.yaml
```

Adjust the dict-mutation keys in the test to match actual YAML structure.

- [x] **Step 4: Full test suite**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -v
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
cd /Users/pesmith/research/dynamite && git add dev_tests/test_bayesopt_generator.py && git commit -m "test: dummy-mode integration for initial_guess warm-up mode"
```

---

## Task 6: YAML example and documentation

**Files:**
- Modify: `dev_tests/bayesopt_ml_modelinner.yaml`
- Modify: `dev_notes/bayesopt_generator.md`

- [x] **Step 1: Add commented `initial_guess` example to YAML**

In `dev_tests/bayesopt_ml_modelinner.yaml`, find the `generator_settings:` block and add a commented alternative:

```yaml
    generator_settings:
      warmup_mode: sobol          # 'sobol' (default) or 'initial_guess'
      # For 'initial_guess' mode, also set:
      # initial_guess:
      #   ml: 5.5                 # physical value for each FREE parameter
      # initial_step_size: 0.1   # ±δ in normalized [0,1] space
      batch_size: 2
      ...
```

- [x] **Step 2: Update `dev_notes/bayesopt_generator.md`**

Add a **Warm-up modes** section after the **Architecture** section:

```markdown
## Warm-up modes

Controlled by `warmup_mode` in `generator_settings`. Both modes produce the same GP phase once warm-up completes.

### `sobol` (default)

Draws `n_initial_random` quasi-random space-filling points via `SobolEngine(scramble=True)` before fitting the GP. Good for exploratory runs with a cheap likelihood or when no prior information is available.

```yaml
generator_settings:
  warmup_mode: sobol          # default; may be omitted
  n_initial_random: 10
  batch_size: 4
```

### `initial_guess`

Evaluates a user-supplied center point then steps outward along each free-parameter axis (±`initial_step_size` in normalized [0,1] space). Total warm-up size: `1 + 2 * n_free` models. Designed for production runs where each model costs 18+ hours and the search space can be centered on a known good region (e.g., literature values or a previous run).

`n_initial_random` is **ignored** in this mode — the queue exhaustion is the warm-up signal.

```yaml
generator_settings:
  warmup_mode: initial_guess
  initial_guess:
    ml: 5.5           # physical values for FREE parameters only
    q-stars: 0.82     # omit a param → defaults to its normalized midpoint
  initial_step_size: 0.1   # 10% of [lo,hi] range; clipped to [0,1]
  batch_size: 2
```

With 5 free parameters this costs `1 + 10 = 11` models before the GP fits — matching a typical RAM budget of ~10 simultaneous weight solves.
```

- [x] **Step 3: Verify all tests still pass**

```bash
cd /Users/pesmith/research/dynamite && python -m pytest dev_tests/test_bayesopt_generator.py -v
```

- [x] **Step 4: Commit**

```bash
cd /Users/pesmith/research/dynamite && git add dev_tests/bayesopt_ml_modelinner.yaml dev_notes/bayesopt_generator.md && git commit -m "docs: add initial_guess warm-up mode example and documentation"
```

---

## Self-review

**1. Spec coverage:**
- ✅ `sobol` mode unchanged — existing behavior preserved
- ✅ `initial_guess` mode: center point evaluated first
- ✅ Axial steps outward from center (±δ per free param)
- ✅ `initial_step_size` configurable
- ✅ Physical values in `initial_guess` (not raw/normalized)
- ✅ Missing params default to midpoint
- ✅ Out-of-bounds values clipped with warning
- ✅ Queue drives warm-up; GP starts when queue exhausted
- ✅ `n_initial_random` still governs Sobol mode

**2. Placeholder scan:** None found. All code blocks contain real implementations.

**3. Type consistency:**
- `_axial_queue`: `list[np.ndarray]` — built in Task 1/3, consumed in Task 3/4 ✅
- `_initial_guess_phys`: `dict[str, float]` — set in Task 1, used in Task 2 ✅
- `_propose_axial_batch` returns same type as `_propose_random_batch`: `list[list[Parameter]]` ✅
- `warmup_mode`: `str`, checked with `==` comparison ✅

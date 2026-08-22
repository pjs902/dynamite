# BayesOptGenerator Modernization v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BayesOptGenerator production-ready for NGC5139 server-scale runs: partial-free triaxial feasibility, in-place GP warm-start, batch integrity, and literature-grounded acquisition upgrades (R1–R4), validated on NGC6278.

**Architecture:** All generator work lives in `dynamite/parameter_space.py` (BayesOptGenerator + module-level helpers). Tests extend the standalone-script suite `dev_tests/test_bayesopt_generator.py` (plain `test_*` functions, `MockSystem`/`MockTriaxialComponent`/`_mk_param`/`_bo_settings` fixtures, `gen.current_models = MockAllModels(names)` pattern). Spec: `docs/superpowers/specs/2026-08-22-bayesopt-modernization-design.md`.

**Tech Stack:** Python 3, numpy, torch/BoTorch (>=0.18.1), gpytorch; astropy Table for mock tables.

## Global Constraints

- Test runner: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py` (from the file's own header). Full suite must print every `PASSED` line and no tracebacks.
- botorch pin: `botorch>=0.18.1,<0.21` (0.18.1 is the validated version).
- Follow `parameter_space.py` local style: single quotes, heavy docstrings, `self.logger` for messages.
- Feasibility margin constant: `1e-6` relative, everywhere (spec H1).
- R3 defaults: `eps_rel=0.01`, `eps_abs=0.0`, `m = max(4, ceil(d/2))` consecutive hits.
- NGC6278 configs and the existing 38 tests must keep passing; behavior changes are additive (new generator_settings keys default to old behavior unless the spec says otherwise).
- Work on branch `bayesopt`. Commit after every task.

---

### Task 1: Dependency pin + environment smoke test (H4)

**Files:**
- Modify: `requirements.txt` (line 22)
- Create: `dev_tests/test_bayesopt_smoke.py`

**Interfaces:**
- Produces: `dev_tests/test_bayesopt_smoke.py` — standalone env check, no DYNAMITE imports; later tasks and cluster nodes use it to validate the botorch stack before anything else.

- [ ] **Step 1: Pin botorch in requirements.txt**

Replace line 22:

```text
botorch>=0.18.1
```

with:

```text
botorch>=0.18.1,<0.21
```

- [ ] **Step 2: Write the smoke test**

Create `dev_tests/test_bayesopt_smoke.py`:

```python
"""Environment smoke test for the BayesOpt stack: torch + botorch + gpytorch.

No DYNAMITE imports. Run on any node before production:
    python dev_tests/test_bayesopt_smoke.py
"""
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


def main():
    torch.manual_seed(0)
    d = 3
    X = torch.rand(20, d, dtype=torch.double)
    y = -((X - 0.5) ** 2).sum(dim=-1, keepdim=True) * 10.0
    gp = SingleTaskGP(X, y).to(torch.double)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acqf = qLogExpectedImprovement(gp, best_f=y.max())
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(torch.double)
    cands, acq_val = optimize_acqf(
        acq_function=acqf, bounds=bounds, q=2, num_restarts=2, raw_samples=16)
    assert cands.shape == (2, d), cands.shape
    assert torch.isfinite(cands).all() and torch.isfinite(acq_val)
    print(f'SMOKE OK: acq_value={acq_val.item():.4f} '
          f'candidates={cands.squeeze().tolist()}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run it**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_smoke.py`
Expected: `SMOKE OK: ...`. If botorch is missing in that env: `pip install 'botorch>=0.18.1,<0.21'` into it first and note the env used.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt dev_tests/test_bayesopt_smoke.py
git commit -m "feat(bayesopt): pin botorch range and add standalone env smoke test"
```

---

### Task 2: Partial-free triaxial projection (H1a)

**Files:**
- Modify: `dynamite/parameter_space.py` — add `_fixed_qpu_values()` helper and rewrite `_project_unit_to_feasible_qpu()` (currently at ~line 1342, gated on all three free)
- Modify: `dev_tests/test_bayesopt_generator.py` — add tests + runner entries

**Interfaces:**
- Produces:
  - `BayesOptGenerator._fixed_qpu_values(self) -> dict[str, float | None]` — keys `'q'/'p'/'u'`; raw value (shape params are never logarithmic) for FIXED axes among q/p/u, `None` for free or absent axes.
  - `_project_unit_to_feasible_qpu(self, X_unit) -> np.ndarray` — same signature as today, now valid for ANY free subset. Task 3 (constraints), Task 5 (dedup fill), Task 7 (annealed sampling) all consume it.

- [ ] **Step 1: Write the failing tests**

Append to `dev_tests/test_bayesopt_generator.py` (before the `if __name__` block):

```python
# --------------------------------------------------------------------------
# v2 Task 2: partial-free triaxial projection
# --------------------------------------------------------------------------
def _qpu_gen(free=('q', 'p', 'u'), qobs=0.65):
    """Generator with q/p/u; axes in `free` are free, others fixed."""
    params = []
    for axis, (lo, hi, val) in [('q', (0.05, 0.99, 0.5)),
                                ('p', (0.90, 0.999, 0.99)),
                                ('u', (0.95, 1.0, 0.9999))]:
        params.append(_mk_param(f'{axis}', lo, hi, val,
                                fixed=axis not in free))
    tri = MockTriaxialComponent('stars', qobs=qobs)
    sysm = MockSystem([], components=[tri])
    sysm.cmp_list[0].parameters = params
    ps_ = make_parspace(params, system=sysm)
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    return gen


def _qpu_valid(gen, qv, pv, uv, qobs=0.65):
    """Algebraic triaxiality check: mirrors triax_pqu2tpp feasibility."""
    return (qv <= pv + 1e-12
            and max(qv / qobs, pv) <= uv + 1e-12
            and uv <= min(pv / qobs, 1.0) + 1e-12
            and qv > 0 and pv > 0 and uv > 0)


def test_fixed_qpu_values_subsets():
    gen = _qpu_gen(free=('q', 'p'))
    fx = gen._fixed_qpu_values()
    assert fx['q'] is None and fx['p'] is None
    assert abs(fx['u'] - 0.9999) < 1e-12
    gen = _qpu_gen(free=())
    fx = gen._fixed_qpu_values()
    assert all(fx[a] is not None for a in 'qpu')
    gen = _qpu_gen(free=('u',))
    fx = gen._fixed_qpu_values()
    assert fx['u'] is None and fx['q'] is not None and fx['p'] is not None
    print('  test_fixed_qpu_values_subsets PASSED')


def test_projection_partial_free_qp():
    """q,p free, u fixed: p>=q and q<=u*qobs and p>=u*qobs and p<=u."""
    gen = _qpu_gen(free=('q', 'p'))
    rng = np.random.default_rng(7)
    X = rng.random((500, 2))
    out = gen._project_unit_to_feasible_qpu(X)
    lo, hi = gen._norm_bounds_arrays()
    raw = out * (hi - lo) + lo
    u_f, qobs = 0.9999, 0.65
    for qv, pv in raw:
        assert _qpu_valid(gen, qv, pv, u_f, qobs), (qv, pv)
    print('  test_projection_partial_free_qp PASSED')


def test_projection_single_free_axes():
    for free in [('q',), ('p',), ('u',), ()]:
        gen = _qpu_gen(free=free)
        rng = np.random.default_rng(11)
        n = 300
        X = rng.random((n, len(free) if free else 1))
        out = gen._project_unit_to_feasible_qpu(X)
        lo, hi = gen._norm_bounds_arrays()
        raw = out * (hi - lo) + lo
        fixed = gen._fixed_qpu_values()
        qv = raw[:, 0] if 'q' in free else np.full(n, fixed['q'])
        pv = raw[:, 1] if 'p' in free else np.full(n, fixed['p'])
        uv = raw[:, -1] if 'u' in free else np.full(n, fixed['u'])
        for j in range(n):
            assert _qpu_valid(gen, qv[j], pv[j], uv[j]), (free, raw[j])
    print('  test_projection_single_free_axes PASSED')


def test_projection_all_free_unchanged():
    """Regression: all-three-free path still satisfies validity."""
    gen = _qpu_gen(free=('q', 'p', 'u'))
    rng = np.random.default_rng(3)
    X = rng.random((500, 3))
    out = gen._project_unit_to_feasible_qpu(X)
    lo, hi = gen._norm_bounds_arrays()
    raw = out * (hi - lo) + lo
    for qv, pv, uv in raw:
        assert _qpu_valid(gen, qv, pv, uv)
    print('  test_projection_all_free_unchanged PASSED')
```

In the `__main__` runner, add before the final print:

```python
    print('v2 Task 2: partial-free projection tests')
    test_fixed_qpu_values_subsets()
    test_projection_partial_free_qp()
    test_projection_single_free_axes()
    test_projection_all_free_unchanged()
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: FAIL — `AttributeError: ... _fixed_qpu_values` and/or projection tests failing on invalid points.

- [ ] **Step 3: Implement**

In `BayesOptGenerator`, add helper and replace `_project_unit_to_feasible_qpu`:

```python
    def _fixed_qpu_values(self):
        """Raw values of FIXED q/p/u axes (None for free or absent).

        Shape parameters are never logarithmic, so par_value == raw value.
        """
        out = {'q': None, 'p': None, 'u': None}
        for p in self.par_space:
            base = p.name.split('-')[0]
            if base in out and getattr(p, 'fixed', False):
                out[base] = float(p.par_value())
        return out

    def _project_unit_to_feasible_qpu(self, X_unit):
        """Project unit-cube samples so the free (q,p,u) subset satisfies
        the triaxiality conditions p >= q, u >= max(q/qobs, p),
        u <= min(p/qobs, 1), using FIXED axis values for the rest.

        Operates in normalized [0,1] space. No-op if qobs is None or no
        qpu axis is free. Feasibility margin 1e-6 relative (spec H1).
        """
        if self.qobs is None:
            return X_unit
        free_axes = [a for a in ('q', 'p', 'u') if a in self._free_qpu_idx]
        if not free_axes:
            return X_unit
        lo_raw, hi_raw = self._norm_bounds_arrays()
        span = hi_raw - lo_raw
        raw = X_unit * span + lo_raw
        fixed = self._fixed_qpu_values()
        m = 1.0 - 1.0e-6
        jq = self._free_qpu_idx.get('q')
        jp = self._free_qpu_idx.get('p')
        ju = self._free_qpu_idx.get('u')
        qobs_m = float(self.qobs) * m
        qv = raw[:, jq] if jq is not None else fixed['q']
        pv = raw[:, jp] if jp is not None else fixed['p']
        uv = raw[:, ju] if ju is not None else fixed['u']
        # p >= q always
        if jp is not None:
            pv = np.maximum(pv, qv)
        if ju is None:
            # u fixed: window constraints become bounds on q and p
            uf = fixed['u']
            if jq is not None:
                qv = np.minimum(qv, uf * qobs_m)
            if jp is not None:
                pv = np.clip(pv, uf * qobs_m, uf * m)
                pv = np.maximum(pv, qv)
        elif jp is not None and jq is not None:
            # all three free: legacy path
            pv = np.maximum(pv, qv)
            u_lo = np.maximum(qv / self.qobs, pv)
            u_hi = np.minimum(pv / self.qobs, 1.0)
            mid = 0.5 * (u_lo + u_hi)
            good = u_hi > u_lo
            uv = uv.copy()
            uv[good] = np.clip(uv[good], u_lo[good], u_hi[good])
            uv[~good] = mid[~good]
        else:
            # u free with q or p fixed: clip u into its window
            u_lo = np.maximum(qv / self.qobs, pv)
            u_hi = np.minimum(pv / self.qobs, 1.0)
            mid = 0.5 * (u_lo + u_hi)
            good = u_hi > u_lo
            uv = uv.copy()
            uv[good] = np.clip(uv[good], u_lo[good], u_hi[good])
            uv[~good] = mid[~good]
        if jq is not None:
            raw[:, jq] = qv
        if jp is not None:
            raw[:, jp] = pv
        if ju is not None:
            raw[:, ju] = uv
        X_proj = (raw - lo_raw) / span
        return np.clip(X_proj, 0.0, 1.0)
```

- [ ] **Step 4: Run tests**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: all PASS including the four new tests and the pre-existing `test_gp_phase_triaxiality`.

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): partial-free triaxial projection for any q/p/u free subset"
```

---

### Task 3: Partial-free triaxiality constraints (H1b)

**Files:**
- Modify: `dynamite/parameter_space.py` — `_make_triaxiality_constraints()` (currently gated on all three free, ~line 1438)
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Consumes: `_fixed_qpu_values()` (Task 2).
- Produces: same signature `(nonlinear, linear)`; returns constraint list covering only the free subset. Task 9 (trust region) reuses it via `_gp_acquisition_batch`.

- [ ] **Step 1: Write the failing test**

```python
def test_constraints_partial_free():
    gen = _qpu_gen(free=('q', 'p'))
    import torch
    nonlinear, linear = gen._make_triaxiality_constraints()
    assert nonlinear is not None and linear is None
    assert len(nonlinear) == 1, 'u fixed -> only p>=q constraint'
    fn, intra = nonlinear[0]
    assert intra is True
    lo, hi = gen._norm_bounds_arrays()
    # feasible point (q=0.3, p=0.95) and infeasible (q=0.95 > p=0.9)
    def unit(vals):
        x = torch.zeros(len(gen.free_params), dtype=torch.double)
        for j, p in enumerate(gen.free_params):
            base = p.name.split('-')[0]
            k = {'q': 0, 'p': 1}[base]
            lo_r, hi_r = lo[k], hi[k]
            x[j] = (vals[base] - lo_r) / (hi_r - lo_r)
        return x
    assert fn(unit({'q': 0.3, 'p': 0.95})).item() >= 0.0
    assert fn(unit({'q': 0.95, 'p': 0.9})).item() < 0.0
    gen = _qpu_gen(free=('u',))
    nonlinear, _ = gen._make_triaxiality_constraints()
    assert nonlinear is None, 'single free axis -> bounds suffice'
    print('  test_constraints_partial_free PASSED')
```

Add `test_constraints_partial_free()` to the runner under the Task 2 block.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: FAIL — `assert nonlinear is not None` (current code returns `(None, None)` when u is fixed).

- [ ] **Step 3: Implement**

Replace `_make_triaxiality_constraints` with:

```python
    def _make_triaxiality_constraints(self):
        """Return (nonlinear_constraints, linear_constraints) over NORMALIZED
        free-parameter coordinates, or (None, None) if not applicable.

        Handles any free subset of (q, p, u): fixed axes enter the
        constraint callables as constants from _fixed_qpu_values().
        With u fixed, the u-window reduces to bounds on q/p (enforced by
        _project_unit_to_feasible_qpu), leaving only p - q >= 0.
        """
        if self.qobs is None:
            return None, None
        free_axes = [a for a in ('q', 'p', 'u') if a in self._free_qpu_idx]
        if not free_axes:
            return None, None
        import torch
        lo_raw, hi_raw = self._norm_bounds_arrays()
        lo_t = torch.tensor(lo_raw, dtype=torch.double)
        span_t = torch.tensor(hi_raw - lo_raw, dtype=torch.double)
        fixed = self._fixed_qpu_values()
        qobs = float(self.qobs)

        def _val(x, axis):
            j = self._free_qpu_idx.get(axis)
            if j is None:
                return torch.tensor(fixed[axis], dtype=torch.double)
            return lo_t[j] + x[j] * span_t[j]

        if 'q' in free_axes and 'p' in free_axes:
            def c_p_ge_q(x):
                return _val(x, 'p') - _val(x, 'q')  # p - q >= 0
            nonlinear = [(c_p_ge_q, True)]
        else:
            nonlinear = None
        if 'u' in free_axes and 'q' in free_axes and 'p' in free_axes:
            def c_u_lower(x):
                return _val(x, 'u') - torch.maximum(
                    _val(x, 'q') / qobs, _val(x, 'p'))
            def c_u_upper(x):
                return torch.clamp(_val(x, 'p') / qobs, max=1.0) \
                    - _val(x, 'u')
            nonlinear = (nonlinear or []) + [(c_u_lower, True),
                                             (c_u_upper, True)]
        return nonlinear, None
```

- [ ] **Step 4: Run tests**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): triaxiality GP constraints for any free q/p/u subset"
```

---

### Task 4: Warm-start guardrails (H2)

**Files:**
- Modify: `dynamite/parameter_space.py` — `specific_generate_method`, `_build_axial_queue`, `_initial_guess_to_unit`, `_gp_acquisition_batch`; add `_clip_training_to_bounds`, `_best_known_unit`
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Produces:
  - `_clip_training_to_bounds(self, X_norm) -> np.ndarray` — warns per-axis with out-of-range counts, returns clipped copy.
  - `_best_known_unit(self, table) -> np.ndarray | None` — normalized coords of min-chi2 valid row, or None.
  - `self._initial_guess_explicit` (bool) and lazy axial-center rebuild at first `initial_guess`-mode generate. Existing tests that inspect `_axial_queue` at init keep passing: init-time build stays when `initial_guess` given; when absent, queue starts as midpoint-centered (today's behavior) and is rebuilt once at first generate IF history exists.

- [ ] **Step 1: Write the failing tests**

```python
def test_warmstart_clip_and_log():
    import logging
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    X = np.array([[0.5], [1.5], [-0.2]])  # two out of bounds
    with caplog_warnings(gen) as records:
        out = gen._clip_training_to_bounds(X)
    np.testing.assert_allclose(out, [[0.5], [1.0], [0.0]])
    assert any('outside' in r for r in records)
    print('  test_warmstart_clip_and_log PASSED')


def test_best_known_unit():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    am = MockAllModels(['ml'])
    for v, c in [(4.2, 9.0), (5.0, 3.5), (5.8, 7.0)]:
        am.table.add_row([v, c, c, float('nan'), '', True, True, True, 0, 'd'])
    gen.current_models = am
    center = gen._best_known_unit(am.table)
    np.testing.assert_allclose(center, [0.5], atol=1e-12)  # ml=5.0 -> 0.5
    print('  test_best_known_unit PASSED')


def test_axial_center_defaults_to_best():
    """No initial_guess + history -> axial queue rebuilt around best row."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['generator_settings']['warmup_mode'] = 'initial_guess'
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=s)
    am = MockAllModels(['ml'])
    for v, c in [(4.2, 9.0), (5.6, 2.0)]:
        am.table.add_row([v, c, c, float('nan'), '', True, True, True, 0, 'd'])
    gen.current_models = am
    gen.specific_generate_method()
    # model_list[0] is a list of Parameter objects; center point has
    # ml ~ 5.6 (best row), not the midpoint 5.0
    ml_par = [p for p in gen.model_list[0] if p.name == 'ml'][0]
    np.testing.assert_allclose(ml_par.raw_value, 5.6, atol=0.15)
    print('  test_axial_center_defaults_to_best PASSED')


class caplog_warnings:
    """Collect this generator's logger warning messages."""
    def __init__(self, gen):
        self.gen = gen
        self.records = []
        self._h = None
    def __enter__(self):
        import logging
        self._h = _ListHandler(self.records)
        self.gen.logger.addHandler(self._h)
        self.gen.logger.setLevel(logging.DEBUG)
        return self.records
    def __exit__(self, *a):
        self.gen.logger.removeHandler(self._h)


class _ListHandler(logging.Handler):
    def __init__(self, out):
        super().__init__()
        self.out = out
    def emit(self, record):
        self.out.append(record.getMessage())
```

Add the three tests to the runner.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: FAIL — `_clip_training_to_bounds` / `_best_known_unit` missing.

- [ ] **Step 3: Implement**

(a) In `__init__`, after the `_axial_queue` assignment, add:

```python
        self._initial_guess_explicit = bool(self._initial_guess_phys)
        self._axial_rebuilt = False
```

(b) Add methods:

```python
    def _clip_training_to_bounds(self, X_norm):
        """Clip normalized training rows into [0,1]; warn per axis.

        Historical rows from a warm-start may lie outside the current
        lo/hi; clipped rows keep the GP anchored at the boundary instead
        of extrapolating.
        """
        X_norm = np.asarray(X_norm, dtype=float)
        n_lo = np.sum(X_norm < 0.0, axis=0)
        n_hi = np.sum(X_norm > 1.0, axis=0)
        for j, name in enumerate(self.free_param_names):
            if n_lo[j] or n_hi[j]:
                self.logger.warning(
                    f'{int(n_lo[j]) + int(n_hi[j])} warm-start training rows '
                    f'outside [{self.lo_free[j]}, {self.hi_free[j]}] for '
                    f'{name}; clipping to the bounds')
        return np.clip(X_norm, 0.0, 1.0)

    def _best_known_unit(self, table):
        """Normalized coords of the valid row with the lowest chi2."""
        done = np.asarray(table['all_done'], dtype=bool)
        chi2 = np.asarray(table[self.chi2], dtype=float)
        ok = done & np.isfinite(chi2)
        if not np.any(ok):
            return None
        idx = np.nanargmin(np.where(ok, chi2, np.inf))
        X_norm, _, _, _, _ = extract_gp_training_data(
            table[np.where(ok)[0][np.searchsorted(np.where(ok)[0], idx)]:
                  np.where(ok)[0][np.searchsorted(np.where(ok)[0], idx)] + 1],
            self.par_space, which_chi2=self.chi2)
        return X_norm[0]
```

Simpler equivalent (preferred):

```python
    def _best_known_unit(self, table):
        """Normalized coords of the valid row with the lowest chi2."""
        done = np.asarray(table['all_done'], dtype=bool)
        chi2 = np.asarray(table[self.chi2], dtype=float)
        ok = done & np.isfinite(chi2)
        if not np.any(ok):
            return None
        rows = table[np.where(ok)[0]]
        X_norm, _, _, _, _ = extract_gp_training_data(
            rows, self.par_space, which_chi2=self.chi2)
        chi2v = np.asarray(rows[self.chi2], dtype=float)
        return X_norm[int(np.argmin(chi2v))]
```

(c) In `_gp_acquisition_batch`, immediately after `X_norm, y, ... = extract_gp_training_data(...)`, insert:

```python
        X_norm = self._clip_training_to_bounds(X_norm)
```

(d) In `specific_generate_method`, initial_guess branch becomes:

```python
        if self.warmup_mode == 'initial_guess':
            if not self._initial_guess_explicit and not self._axial_rebuilt:
                self._axial_rebuilt = True
                if n_valid > 0:
                    center = self._best_known_unit(table)
                    if center is not None:
                        self.logger.info(
                            'warm-start: axial warm-up centered on the '
                            'best historical model')
                        self._axial_queue = \
                            self._build_axial_queue(center=center)
            if self._axial_queue:
                self._gp_model = None
                self._last_acq_value = None
                self.model_list = self._propose_axial_batch()
                return
```

(e) `_build_axial_queue` gains an optional center:

```python
    def _build_axial_queue(self, center=None):
        """... (existing docstring)"""
        if center is None:
            center = self._initial_guess_to_unit()
        step = self.initial_step_size
        points = [center.copy()]
        for j in range(len(self.free_params)):
            for sign in (+1.0, -1.0):
                pt = center.copy()
                pt[j] = np.clip(center[j] + sign * step, 0.0, 1.0)
                points.append(pt)
        return points
```

- [ ] **Step 4: Run tests**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: all PASS (init-time `_axial_queue` assertions unchanged: explicit-guess and empty-history paths behave as before).

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): warm-start guardrails — bounds clipping, best-known axial center, explicit logging"
```

---

### Task 5: Post-snap collision handling (B1)

**Files:**
- Modify: `dynamite/parameter_space.py` — add `_dedup_and_fill`; call it in `_gp_acquisition_batch` right after `_snap_to_grid`
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Consumes: `_sobol_unit`, `_project_unit_to_feasible_qpu` (Task 2), `_norm_steps`.
- Produces: `_dedup_and_fill(self, X_unit) -> np.ndarray` — unique snapped non-ml cells, full `batch_size` rows. Task 7 slots annealed members in before the Sobol filler by extending this function's fill order.

- [ ] **Step 1: Write the failing test**

```python
def test_dedup_and_fill():
    q = _mk_param('q', 0.3, 0.9, 0.6)
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([q, ml])  # q is the snappable non-ml column
    s = _bo_settings()
    s['generator_settings']['discretize_non_ml_params'] = True
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=s)
    assert gen.discretize_non_ml_params
    dup = np.array([[0.501, 5.0], [0.509, 4.6], [0.30, 5.2]])  # row 0/1 same cell
    out = gen._dedup_and_fill(dup)
    assert out.shape == (3, 2)
    step = gen._norm_steps[0]
    cells = [round(v[0] / step) for v in out if step > 0]
    assert len(set(cells)) == len(cells), 'cells must be unique'
    np.testing.assert_allclose(out[0, 0], [0.5], atol=1e-9)  # best kept
    print('  test_dedup_and_fill PASSED')
```

Add to runner.

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — `_dedup_and_fill` missing.

- [ ] **Step 3: Implement**

Add method:

```python
    def _cell_keys(self, X_unit):
        """Integer cell ids of snapped non-ml columns; continuous columns
        (step<=0 or ml) enter the key unrounded."""
        steps = np.asarray(self._norm_steps, dtype=float)
        keys = np.empty_like(X_unit)
        for j in range(X_unit.shape[1]):
            if steps[j] > 0:
                keys[:, j] = np.round(X_unit[:, j] / steps[j])
            else:
                keys[:, j] = X_unit[:, j]
        return np.round(keys, 9)

    def _dedup_and_fill(self, X_unit):
        """Keep the first candidate per snapped non-ml cell; refill freed
        slots with feasible Sobol draws so the batch stays full.

        Duplicate snapped cells would integrate identical orbit libraries,
        wasting orblib slots (spec B1). With discretize disabled this is a
        no-op passthrough of the input.
        """
        if not self.discretize_non_ml_params or self._norm_steps is None:
            return X_unit
        X_unit = np.asarray(X_unit, dtype=float)
        keys = self._cell_keys(X_unit)
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        keep = X_unit[np.sort(first_idx)]
        guard = 0
        while keep.shape[0] < self.batch_size and guard < 100:
            guard += 1
            filler = self._project_unit_to_feasible_qpu(
                self._sobol_unit(self.batch_size))
            fkeys = self._cell_keys(filler)
            existing = self._cell_keys(keep)
            for row, k in zip(filler, fkeys):
                if keep.shape[0] >= self.batch_size:
                    break
                if not np.any(np.all(existing == k, axis=1)):
                    keep = np.vstack([keep, row[None, :]])
                    existing = np.vstack([existing, k[None, :]])
        return keep[:self.batch_size]
```

In `_gp_acquisition_batch`, change:

```python
        cand_np = self._snap_to_grid(candidates.detach().numpy())
```

to:

```python
        cand_np = self._dedup_and_fill(
            self._snap_to_grid(candidates.detach().numpy()))
```

- [ ] **Step 4: Run tests**

Expected: all PASS. (If ml-only spaces make `_norm_steps` all-zero, the dedup passthrough guard covers it.)

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): deduplicate snapped batch cells, refill with feasible Sobol"
```

---

### Task 6: Dimensional exploration schedule (R1)

**Files:**
- Modify: `dynamite/parameter_space.py` — `__init__` settings, add `_exploration_beta`, use in `_gp_acquisition_batch`
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Produces: `_exploration_beta(self, n_gp_batches_done) -> float`. Settings: `exploration_schedule` in `{'constant','annealed'}` (default `'constant'`), `beta_start=8.0`, `beta_end=0.2`, `anneal_batches=10`. qLogEI call gains `beta=self._exploration_beta(...)`.

- [ ] **Step 1: Write the failing test**

```python
def test_exploration_schedule():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['generator_settings']['exploration_schedule'] = 'annealed'
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=s)
    b0 = gen._exploration_beta(0)
    b5 = gen._exploration_beta(5)
    b20 = gen._exploration_beta(20)
    assert b0 == 8.0 and b20 == 0.2 and 0.2 < b5 < 8.0
    s['generator_settings']['exploration_schedule'] = 'constant'
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=s)
    assert gen._exploration_beta(0) is None
    print('  test_exploration_schedule PASSED')
```

Add to runner.

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — `_exploration_beta` missing.

- [ ] **Step 3: Implement**

In `__init__` after `self.min_ei_threshold = ...`:

```python
        self.exploration_schedule = gen.get('exploration_schedule',
                                            'constant')
        if self.exploration_schedule not in ('constant', 'annealed'):
            raise ValueError(
                "exploration_schedule must be 'constant' or 'annealed'")
        self.beta_start = float(gen.get('beta_start', 8.0))
        self.beta_end = float(gen.get('beta_end', 0.2))
        self.anneal_batches = int(gen.get('anneal_batches', 10))
        self._gp_batches_done = 0
```

Add method:

```python
    def _exploration_beta(self, n_gp_batches_done):
        """qLogEI exploration weight; None -> BoTorch heuristic (constant
        mode). Annealed mode linearly decays beta_start -> beta_end over
        `anneal_batches` GP batches (GPry-style dimension-scaled
        exploration, spec R1)."""
        if self.exploration_schedule == 'constant':
            return None
        frac = min(1.0, n_gp_batches_done / max(1, self.anneal_batches))
        return self.beta_start + frac * (self.beta_end - self.beta_start)
```

In `_gp_acquisition_batch`, replace the acqf construction:

```python
        acqf = qLogExpectedImprovement(model=model, best_f=Y_t.max())
```

with:

```python
        beta = self._exploration_beta(self._gp_batches_done)
        acqf = (qLogExpectedImprovement(model=model, best_f=Y_t.max())
                if beta is None else
                qLogExpectedImprovement(model=model, best_f=Y_t.max(),
                                        beta=beta))
```

and after `self._last_acq_value = float(acq_value.item())` add:

```python
        self._gp_batches_done += 1
```

- [ ] **Step 4: Run tests**

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): annealed exploration schedule on qLogEI beta (R1)"
```

---

### Task 7: Tempered-posterior batch members (R2)

**Files:**
- Modify: `dynamite/parameter_space.py` — add `_gp_posterior_mean`, `_sample_annealed_members`; integrate in `_gp_acquisition_batch` between dedup and final assembly
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Consumes: `_project_unit_to_feasible_qpu`, `_dedup_and_fill` (Task 5), `self._gp_model`.
- Produces: `_sample_annealed_members(self, n, tau) -> np.ndarray` (unit space, feasible). Settings: `n_annealed_members` (default `max(1, batch_size // 4)`; 0 disables), `tau_start=1.0`, `tau_decay=0.7`, `tau_min=0.05`, `annealed_max_draws=200000`.

- [ ] **Step 1: Write the failing test**

```python
def test_annealed_members_concentrate():
    """With a linear mean in x, small tau draws must sit near x=1."""
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['generator_settings']['n_annealed_members'] = 4
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=s)

    class FakeGP:
        class posterior:
            @staticmethod
            def mean(X):
                return X.sum(dim=-1)
    gen._gp_model = FakeGP()
    pts = gen._sample_annealed_members(n=8, tau=0.01)
    assert pts.shape == (8, 1)
    assert np.all(pts > 0.9), pts
    print('  test_annealed_members_concentrate PASSED')


def test_annealed_disabled_by_default():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    assert gen.n_annealed_members == 2  # ceil(8/4)
    print('  test_annealed_disabled_by_default PASSED')
```

Add both to runner.

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — `_sample_annealed_members` missing.

- [ ] **Step 3: Implement**

In `__init__` after the R1 block:

```python
        self.n_annealed_members = int(gen.get('n_annealed_members',
                                              max(1, self.batch_size // 4)))
        self.tau_start = float(gen.get('tau_start', 1.0))
        self.tau_decay = float(gen.get('tau_decay', 0.7))
        self.tau_min = float(gen.get('tau_min', 0.05))
        self.annealed_max_draws = int(gen.get('annealed_max_draws', 200000))
```

Add methods:

```python
    def _gp_posterior_mean(self, X_unit_t):
        """Posterior mean of the fitted GP at unit-space points (torch)."""
        import torch
        with torch.no_grad():
            return self._gp_model.posterior(X_unit_t).mean

    def _sample_annealed_members(self, n, tau):
        """Draw n feasible unit-space points ~ exp(mu(x)/tau) by rejection
        (SALE's annealed objective, mean-only; spec R2).

        Falls back to projected Sobol draws if the feasible acceptance
        volume is too small within `annealed_max_draws` candidates.
        """
        import torch
        lo_raw, hi_raw = self._norm_bounds_arrays()
        span = hi_raw - lo_raw
        chunk = max(256, 16 * n)
        out = []
        total = 0
        while len(out) < n and total < self.annealed_max_draws:
            cand = self._project_unit_to_feasible_qpu(
                self._sobol_unit(min(chunk, self.annealed_max_draws - total)))
            total += cand.shape[0]
            mu = self._gp_posterior_mean(
                torch.tensor(cand, dtype=torch.double)).numpy()
            w = np.exp((mu - mu.max()) / tau)
            acc = np.random.random(cand.shape[0]) < w
            out.extend(cand[acc].tolist())
        if len(out) < n:
            self.logger.warning(
                f'annealed sampling accepted {len(out)}/{n}; filling '
                'remainder with projected Sobol draws')
            fill = self._project_unit_to_feasible_qpu(
                self._sobol_unit(n - len(out)))
            out.extend(fill.tolist())
        return np.array(out[:n])
```

In `_gp_acquisition_batch`, after the `cand_np = self._dedup_and_fill(...)` line, insert:

```python
        if self.n_annealed_members > 0:
            tau = max(self.tau_min,
                      self.tau_start * (self.tau_decay ** self._gp_batches_done))
            n_annealed = min(self.n_annealed_members, self.batch_size - 1)
            annealed = self._sample_annealed_members(n_annealed, tau)
            cand_np = np.vstack([cand_np[:self.batch_size - n_annealed],
                                 annealed])
            cand_np = self._dedup_and_fill(cand_np)
```

- [ ] **Step 4: Run tests**

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): tempered-posterior annealed batch members (R2, SALE-style)"
```

---

### Task 8: Prediction-accuracy stopping (R3)

**Files:**
- Modify: `dynamite/parameter_space.py` — add `_record_predictions`, `_score_new_predictions`; call the former in `_gp_acquisition_batch`, the latter at the top of `specific_generate_method`; add status flag in `check_specific_stopping_criteria`
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Produces:
  - `self._pending_predictions: dict[key(float-tuple), float]` — snapped-unit coords → GP posterior mean of NEGATIVE chi2 (i.e. `-mu`).
  - `self._pred_streak: int`, `self._pred_hits: int`; status key `gp_predictions_accurate` (bool), raised after `self.pred_hits_needed = max(4, ceil(d/2))` consecutive hits.

- [ ] **Step 1: Write the failing test**

```python
def test_prediction_accuracy_counter():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    gen._pred_hits_needed = 2
    am = MockAllModels(['ml'])
    # row ml=5.0 -> unit 0.5; GP predicted -3.0 (chi2 3.0)
    gen._pending_predictions = {(0.5,): -3.0}
    am.table.add_row([5.0, 3.0, 3.0, float('nan'), '',
                      True, True, True, 0, 'd'])
    gen._score_new_predictions(am.table)
    assert gen._pred_streak == 1 and gen._pending_predictions == {}
    am.table.add_row([5.0, 3.05, 3.05, float('nan'), '',
                      True, True, True, 0, 'd'])
    gen._score_new_predictions(am.table)
    assert gen._pred_streak == 2
    assert gen.status.get('gp_predictions_accurate') is True
    am.table.add_row([5.0, 50.0, 50.0, float('nan'), '',
                      True, True, True, 0, 'd'])
    gen._pending_predictions = {(0.5,): -3.0}
    gen._score_new_predictions(am.table)
    assert gen._pred_streak == 0
    assert gen.status.get('gp_predictions_accurate') is False
    print('  test_prediction_accuracy_counter PASSED')
```

Add to runner.

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — `_score_new_predictions` missing.

- [ ] **Step 3: Implement**

In `__init__` (R3 block):

```python
        self.pred_eps_rel = float(gen.get('pred_eps_rel', 0.01))
        self.pred_eps_abs = float(gen.get('pred_eps_abs', 0.0))
        d_free = len(self.free_par_idx)
        self.pred_hits_needed = int(gen.get(
            'pred_hits_needed', max(4, -(-d_free // 2))))
        self._pending_predictions = {}
        self._pred_streak = 0
```

Add methods:

```python
    def _record_predictions(self, X_unit, mu_neg_chi2):
        """Store GP predictions (as negative chi2) keyed by snapped unit
        coordinates, for scoring when the models finish."""
        for x, m in zip(X_unit, mu_neg_chi2):
            self._pending_predictions[tuple(np.round(x, 9))] = float(m)

    def _score_new_predictions(self, table):
        """Compare finished models' kinchi2 against stored predictions.

        Hit: |mu - y| < eps_abs + eps_rel * |y_best - mu| (GPry
        CorrectCounter). `pred_hits_needed` consecutive hits raise
        status['gp_predictions_accurate'].
        """
        done = np.asarray(table['all_done'], dtype=bool)
        chi2 = np.asarray(table[self.chi2], dtype=float)
        ok = done & np.isfinite(chi2)
        if not np.any(ok):
            return
        X_norm, y, _, _, _ = extract_gp_training_data(
            table, self.par_space, which_chi2=self.chi2)
        valid_rows = table[ok]
        best = float(np.min(chi2[ok]))
        for row, x in zip(valid_rows, X_norm):
            key = tuple(np.round(x, 9))
            if key not in self._pending_predictions:
                continue
            mu_neg = self._pending_predictions.pop(key)
            mu = -mu_neg
            yv = float(row[self.chi2])
            if abs(mu - yv) < (self.pred_eps_abs
                               + self.pred_eps_rel * abs(best - mu)):
                self._pred_streak += 1
                self.logger.info(
                    f'GP prediction accurate ({self._pred_streak}/'
                    f'{self.pred_hits_needed} consecutive)')
            else:
                if self._pred_streak:
                    self.logger.debug(
                        f'GP prediction missed: mu={mu:.2f} y={yv:.2f}')
                self._pred_streak = 0
        self.status['gp_predictions_accurate'] = (
            self._pred_streak >= self.pred_hits_needed)
```

Wire-up — in `specific_generate_method`, right after `table = self.current_models.table`:

```python
        self._score_new_predictions(table)
```

In `_gp_acquisition_batch`, after the final `cand_np` is assembled (after dedup/annealed block) and before `raw_free = denormalize_to_raw(...)`:

```python
        with torch.no_grad():
            pred_mu = self._gp_model.posterior(
                torch.tensor(cand_np, dtype=torch.double)).mean.numpy()
        self._record_predictions(cand_np, pred_mu)
```

In `check_specific_stopping_criteria`, alongside the other status flags:

```python
        self.status.setdefault('gp_predictions_accurate', False)
```

- [ ] **Step 4: Run tests**

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): prediction-accuracy convergence counter (R3, GPry CorrectCounter)"
```

---

### Task 9: Trust-region refinement (R4)

**Files:**
- Modify: `dynamite/parameter_space.py` — add `_knn_radius`, `_maybe_update_tr`, `_tr_bounds`; use in `_gp_acquisition_batch`
- Modify: `dev_tests/test_bayesopt_generator.py`

**Interfaces:**
- Consumes: `extract_gp_training_data`, `_clip_training_to_bounds`.
- Produces:
  - Settings: `trust_region=False`, `tr_trigger_frac=0.1`, `tr_side_init=0.3`, `tr_grow=1.3`, `tr_shrink=0.7`, `tr_min_side=0.05`, `tr_max_side=0.6`, `tr_patience=2`.
  - `_tr_bounds(self) -> np.ndarray shape (2, d) | None` — unit-space box used by `_gp_acquisition_batch` in place of the full box when active.

- [ ] **Step 1: Write the failing test**

```python
def test_trust_region_lifecycle():
    ml = _mk_param('ml', 4.0, 6.0, 5.0)
    ps_ = make_parspace([ml])
    s = _bo_settings()
    s['generator_settings'].update({
        'trust_region': True, 'tr_trigger_frac': 0.1,
        'tr_side_init': 0.3, 'tr_min_side': 0.05, 'tr_max_side': 0.6})
    gen = BayesOptGenerator(par_space=ps_, parspace_settings=s)
    am = MockAllModels(['ml'])
    for v, c in [(5.00, 3.0), (5.02, 3.1), (4.98, 3.2),
                 (5.01, 3.3), (5.03, 3.4), (4.99, 3.5)]:
        am.table.add_row([v, c, c, float('nan'), '',
                          True, True, True, 0, 'd'])
    gen.current_models = am
    bounds = gen._tr_bounds()
    assert bounds is not None, 'clustered points -> TR active'
    assert bounds[0, 0] >= 0.0 and bounds[1, 0] <= 1.0
    side0 = bounds[1, 0] - bounds[0, 0]
    gen._tr_stale_batches = gen.tr_patience  # simulate stale batches
    gen._maybe_update_tr(am.table)
    assert gen._tr_side < side0, 'stale -> shrink'
    print('  test_trust_region_lifecycle PASSED')


def test_trust_region_off_by_default():
    gen = BayesOptGenerator(par_space=make_parspace([_mk_param('ml', 4., 6., 5.)]),
                            parspace_settings=_bo_settings())
    assert gen.trust_region is False
    assert gen._tr_bounds() is None
    print('  test_trust_region_off_by_default PASSED')
```

Add both to runner.

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — `_tr_bounds` missing.

- [ ] **Step 3: Implement**

In `__init__` (R4 block):

```python
        self.trust_region = bool(gen.get('trust_region', False))
        self.tr_trigger_frac = float(gen.get('tr_trigger_frac', 0.1))
        self.tr_side_init = float(gen.get('tr_side_init', 0.3))
        self.tr_grow = float(gen.get('tr_grow', 1.3))
        self.tr_shrink = float(gen.get('tr_shrink', 0.7))
        self.tr_min_side = float(gen.get('tr_min_side', 0.05))
        self.tr_max_side = float(gen.get('tr_max_side', 0.6))
        self.tr_patience = int(gen.get('tr_patience', 2))
        self._tr_side = self.tr_side_init
        self._tr_center = None
        self._tr_stale_batches = 0
        self._tr_best_seen = None
```

Add methods:

```python
    def _knn_radius(self, X_norm):
        """Mean distance from the incumbent to its 5 nearest evaluated
        neighbours, as a fraction of the box diagonal (SALE's local
        resolution proxy, curvature dropped for v1)."""
        chi2 = np.asarray(self.current_models.table[self.chi2], dtype=float)
        i0 = int(np.nanargmin(np.where(np.isfinite(chi2), chi2, np.inf)))
        diffs = X_norm - X_norm[i0]
        dist = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dist)
        knn = [dist[j] for j in order[1:6] if dist[j] > 0][:5]
        if not knn:
            return np.inf
        return float(np.mean(knn)) / np.sqrt(len(self.free_par_idx))

    def _maybe_update_tr(self, table):
        """Enter/grow/shrink the trust region (TuRBO-lite, spec R4)."""
        if not self.trust_region:
            return
        done = np.asarray(table['all_done'], dtype=bool)
        chi2 = np.asarray(table[self.chi2], dtype=float)
        best = float(np.nanmin(np.where(done & np.isfinite(chi2), chi2,
                                        np.inf)))
        if self._tr_center is not None:
            if best < self._tr_best_seen - 1e-12:
                self._tr_side = min(self._tr_side * self.tr_grow,
                                    self.tr_max_side)
                self._tr_stale_batches = 0
            else:
                self._tr_stale_batches += 1
                if self._tr_stale_batches >= self.tr_patience:
                    self._tr_side = max(self._tr_side * self.tr_shrink,
                                        self.tr_min_side)
                    self._tr_stale_batches = 0
        self._tr_best_seen = best

    def _tr_bounds(self):
        """Unit-space acquisition box: trust region if active+triggered,
        else None (full box)."""
        if not self.trust_region:
            return None
        table = self.current_models.table
        X_norm, _, _, _, _ = extract_gp_training_data(
            table, self.par_space, which_chi2=self.chi2)
        if X_norm.shape[0] < 10:
            return None
        if self._knn_radius(X_norm) > self.tr_trigger_frac:
            return None
        self._maybe_update_tr(table)
        center = self._best_known_unit(table)
        if center is None:
            return None
        self._tr_center = center
        half = self._tr_side / 2.0
        lo = np.clip(center - half, 0.0, 1.0)
        hi = np.clip(center + half, 0.0, 1.0)
        return np.stack([lo, hi])
```

In `_gp_acquisition_batch`, replace the bounds construction:

```python
        d = len(self.free_par_idx)
        bounds = torch.stack([
            torch.zeros(d, dtype=torch.double),
            torch.ones(d, dtype=torch.double)])
```

with:

```python
        d = len(self.free_par_idx)
        tr = self._tr_bounds()
        if tr is not None:
            self.logger.info(
                f'trust region active: side={self._tr_side:.3f} around '
                f'{np.round(self._tr_center, 3).tolist()}')
            bounds = torch.tensor(tr, dtype=torch.double)
        else:
            bounds = torch.stack([
                torch.zeros(d, dtype=torch.double),
                torch.ones(d, dtype=torch.double)])
```

- [ ] **Step 4: Run tests**

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dynamite/parameter_space.py dev_tests/test_bayesopt_generator.py
git commit -m "feat(bayesopt): single trust-region refinement phase (R4, TuRBO-lite)"
```

---

### Task 10: SplitModelIterator alignment test + YAML sweep (H3)

**Files:**
- Modify: `dev_tests/test_bayesopt_generator.py` — add e2e-style dummy test
- Modify: `dev_tests/bayesopt_qml_modelinner.yaml` — add `modeliterator: 'SplitModelIterator'` variant only if a split twin is missing (bayesopt_ml_split.yaml already sets it at line 175)

**Interfaces:**
- Consumes: the existing real-config e2e `test_axial_warmup_dummy_run` (test file, ~line 820) as the template — including its real-`dynamite` import dance (pop stub → import → restore in `finally`), `tempfile.TemporaryDirectory`, and NGC6278_input path.

- [ ] **Step 1: Write the split-iterator e2e test**

Copy `test_axial_warmup_dummy_run` as `test_e2e_split_iterator` with exactly these changes:

```python
def test_e2e_split_iterator():
    """Dummy e2e under SplitModelIterator: batch completes and snapped
    duplicates reuse orbit libraries (fewer orblib dirs than models)."""
    import sys as _sys
    _stub = _sys.modules.pop('dynamite', None)
    import dynamite as _dyn
    _sys.modules['dynamite'] = _dyn
    config_reader = importlib.import_module('dynamite.config_reader')
    model_iterator = importlib.import_module('dynamite.model_iterator')
    try:
        base_yaml = ('/Users/pesmith/research/dynamite/dev_tests/'
                     'bayesopt_ml_split.yaml')  # <- split template
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f)
        assert cfg['multiprocessing_settings']['modeliterator'] == \
            'SplitModelIterator'
        gs = cfg['parameter_space_settings']['generator_settings']
        gs.update({'discretize_non_ml_params': True,
                   'batch_size': 4, 'n_orblib_configs': 2,
                   'n_ml_per_config': 2})
        cfg['parameter_space_settings']['stopping_criteria']['n_max_mods'] = 12

        def _chi2(parset):
            q = float(parset['q-stars'])
            ml = float(parset['ml'])
            return 80.0 * (q - 0.4) ** 2 + 200.0 * (ml - 5.5) ** 2 + 15.0

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg['io_settings']['output_directory'] = tmpdir + '/'
            cfg['io_settings']['input_directory'] = (
                '/Users/pesmith/research/dynamite/dev_tests/NGC6278_input/')
            cfg_path = os.path.join(tmpdir, 'test_split.yaml')
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg, f)
            c = config_reader.Configuration(cfg_path, reset_logging=False)
            model_iterator.ModelIterator(c, do_dummy_run=True,
                                         dummy_chi2_function=_chi2,
                                         plots=False)
            table = c.all_models.table
            dirs = [d for d in table['directory'] if d]
            assert len(dirs) >= 8, f'expected >=8 models, got {len(dirs)}'
            n_orblibs = len({d.split('/ml')[0] for d in dirs})
            assert n_orblibs < len(dirs), (
                f'snapped duplicates must reuse orblibs: '
                f'{n_orblibs} libs for {len(dirs)} models')
    finally:
        if _stub is not None:
            _sys.modules['dynamite'] = _stub
    print('  test_e2e_split_iterator PASSED')
```

Keep every other detail of the template (logging setup, guard comments). Add `test_e2e_split_iterator()` to the runner after the axial test.

- [ ] **Step 2: Run it**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: new test PASSES alongside `test_axial_warmup_dummy_run`. If bayesopt_ml_split.yaml's free parameters differ from (q-stars, ml), adjust `_chi2`/gs keys to that YAML's actual free-parameter names (read the file first).

- [ ] **Step 3: YAML sweep**

Run: `grep -L "modeliterator" dev_tests/bayesopt_*.yaml`
For each file listed (expected: `bayesopt_qml_modelinner.yaml`), add under `multiprocessing_settings:`:

```yaml
    modeliterator: 'SplitModelIterator'
```

- [ ] **Step 4: Commit**

```bash
git add dev_tests/test_bayesopt_generator.py dev_tests/bayesopt_qml_modelinner.yaml
git commit -m "test(bayesopt): SplitModelIterator e2e dummy test; split iterator in all shipped YAMLs"
```

---

### Task 11: Production config patch (V5 — keep all four kinematic sets)

**Files:**
- Create: `dev_tests/NGC5139_config_production.yaml` (the audited config, patched)

**Interfaces:**
- Consumes: the audited YAML from the 2026-08-22 feasibility review (user-confirmed: keep `lvm` and `gaia_pm` sets).
- Produces: the reference production config for the xeast run.

- [ ] **Step 1: Create the patched config**

Copy the user's production YAML verbatim, then apply exactly these diffs:

(a) `multiprocessing_settings` — add as first key:

```yaml
multiprocessing_settings:
    # REQUIRED: weight solves must run in the split flow, or
    # ncpus_weights is ignored and up to 45 concurrent solves x 190 GiB
    # can coexist (8.5 TB on a 1416 GB node).
    modeliterator: "SplitModelIterator"
    # The whole budget. ...
    total_cores: 90
```

(b) `q` parameter — tighten the upper bound (validity q <= u*qobs; u=0.9999, qobs≈0.90 for omega Cen):

```yaml
            q: # intrinsic flattening (C/A)
                par_generator_settings:
                    lo: 0.05
                    # 0.9999 * qobs(0.90) = 0.89991; keep clear of the
                    # validity wall. RECOMPUTE if your qobs differs.
                    hi: 0.89
                    step: 0.04
                    minstep: 0.02
                fixed: False
                value: 0.5
                LaTeX: "$q$"
```

(c) Header comment — replace the copy instructions with:

```yaml
# Copy into NGC5139_production_input_xeast/ (ALL four kinematic sets):
#   mge.ecsv
#   veldist_kinematics.ecsv veldist_aperture.dat veldist_bins.dat
#   hst_veldist_pm_2dhist.npz hst_veldist_aperture.dat hst_veldist_bins.dat
#   lvm_kins.ecsv lvm_aperture.dat lvm_bins.dat
#   gaia_veldist_pm_2dhist.npz gaia_veldist_aperture.dat gaia_veldist_bins.dat
```

(d) `parameter_space_settings` — add the warm-start note:

```yaml
parameter_space_settings:
    generator_type: "GridWalk"
    # To continue with BayesOpt after this GridWalk run: switch
    # generator_type to "BayesOptGenerator" IN THIS SAME output_directory;
    # the GP warm-starts from every row of all_models.ecsv (spec H2).
    which_chi2: "kinchi2"
```

- [ ] **Step 2: Validate it parses**

```bash
cd dev_tests && /opt/miniconda3/envs/main/bin/python3 -c "
import yaml
cfg = yaml.safe_load(open('NGC5139_config_production.yaml'))
assert cfg['multiprocessing_settings']['modeliterator'] == 'SplitModelIterator'
assert cfg['system_components']['stars']['parameters']['q']\
       ['par_generator_settings']['hi'] == 0.89
assert set(cfg['system_components']['stars']['kinematics']) == \
       {'veldist', 'hst_pm', 'lvm', 'gaia_pm'}
print('production YAML OK')"
```

Expected: `production YAML OK`.

- [ ] **Step 3: Commit**

```bash
git add dev_tests/NGC5139_config_production.yaml
git commit -m "feat(production): NGC5139 config — split iterator, feasible q range, full input list"
```

---

### Task 12: Ablation runner + docs (V2/V3 prep)

**Files:**
- Create: `dev_tests/run_ablation.py`
- Modify: `dev_notes/bayesopt_generator.md` — append "v2 (2026-08-22)" section
- Modify: `CLAUDE.md` — update BayesOptGenerator section

**Interfaces:**
- Consumes: `run_comparison_real.py` conventions (config injection, output dirs).
- Produces: ablation matrix runner used for V2; documentation for all new generator_settings keys.

- [ ] **Step 1: Write the ablation runner**

Create `dev_tests/run_ablation.py`:

```python
#!/usr/bin/env python3
"""Dummy-mode ablation matrix for BayesOptGenerator v2 (spec V2).

Runs {sobol, initial_guess, warm_start} x {baseline, r1r2, full} on the
synthetic landscape and reports models-to-threshold and best chi2.

Usage:
    python run_ablation.py [--quick]
"""
import argparse
import copy
import itertools
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the test file's standalone loader for parameter_space (it stubs
# sys.modules['dynamite'], so do NOT `import dynamite` here).
import test_bayesopt_generator as T  # noqa: E402

BayesOptGenerator = T.ps.BayesOptGenerator


def landscape(ml, q):
    """Anisotropic multi-modal chi2 over (ml, q) — stands in for NGC6278."""
    return (30.0 * (ml - 5.12) ** 2 / 0.8
            + 400.0 * (q - 0.62) ** 2 / 0.2
            + 8.0 * np.sin(12.0 * ml) * np.cos(9.0 * q) + 5800.0)


def _model_raws(model_list_entry, names):
    """model_list entries are lists of Parameter objects -> {name: raw}."""
    return {[p.name for p in model_list_entry][i]:
            [p.raw_value for p in model_list_entry][i]
            for i, name in enumerate(names)}


VARIANTS = {
    'baseline': {},
    'r1r2': {'exploration_schedule': 'annealed',
             'n_annealed_members': 2},
    'full': {'exploration_schedule': 'annealed',
             'n_annealed_members': 2,
             'trust_region': True,
             'discretize_non_ml_params': True},
}
WARMUPS = ['sobol', 'initial_guess', 'warm_start']
N_MODELS = 120


def run_one(warmup, variant_name, quick=False):
    s = copy.deepcopy(T._bo_settings())
    gs = s['generator_settings']
    gs['batch_size'] = 8
    if warmup == 'warm_start':
        gs['n_initial_random'] = 0
    gs['warmup_mode'] = 'initial_guess' if warmup == 'initial_guess' \
        else 'sobol'
    gs.update(copy.deepcopy(VARIANTS[variant_name]))
    n_max = 40 if quick else N_MODELS
    ml = T._mk_param('ml', 4.0, 6.0, 5.0)
    q = T._mk_param('q-stars', 0.3, 0.9, 0.6)
    pspace = T.make_parspace([ml, q])
    gen = BayesOptGenerator(par_space=pspace, parspace_settings=s)
    am = T.MockAllModels(['ml', 'q-stars'])
    if warmup == 'warm_start':
        rng = np.random.default_rng(0)
        for _ in range(30):
            mlv, qv = 4.0 + 2.0 * rng.random(), 0.3 + 0.6 * rng.random()
            c = landscape(mlv, qv)
            am.table.add_row([mlv, qv, c, c,
                              float('nan'), '', True, True, True, 0, 'd'])
    gen.current_models = am
    while len(am.table) < n_max and not gen.status.get('stop'):
        gen.generate(current_models=am)
        if gen.status.get('stop'):
            break
        for entry in gen.model_list:
            if len(am.table) >= n_max:
                break
            raws = _model_raws(entry, ['ml', 'q-stars'])
            c = landscape(raws['ml'], raws['q-stars'])
            am.table.add_row([raws['ml'], raws['q-stars'], c, c,
                              float('nan'), '', True, True, True, 0, 'd'])
    chi2s = np.asarray(am.table['kinchi2'], dtype=float)
    finite = chi2s[np.isfinite(chi2s)]
    return {'variant': variant_name, 'warmup': warmup,
            'n_models': len(am.table),
            'best_chi2': float(np.min(finite))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    rows = [run_one(w, v, args.quick)
            for w, v in itertools.product(WARMUPS, VARIANTS)]
    print(f"{'warmup':<12} {'variant':<10} {'n_models':>8} {'best_chi2':>10}")
    for r in rows:
        print(f"{r['warmup']:<12} {r['variant']:<10} "
              f"{r['n_models']:>8} {r['best_chi2']:>10.2f}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the quick matrix**

Run: `cd dev_tests && /opt/miniconda3/envs/main/bin/python3 run_ablation.py --quick`
Expected: a 9-row table, no exceptions. (Slow full run is V2 on the cluster, not here.)

- [ ] **Step 3: Update docs**

(a) In `dynamite/parameter_space.py`, extend the `BayesOptGenerator` class
docstring's generator_settings list with (spec B2):

```
        n_orblib_configs x n_ml_per_config control orblib reuse: each
        batch pairs n_ml_per_config distinct ml values per potential
        config. Higher n_ml_per_config buys cheaper batches (one orbit
        library serves several weight solves) at the cost of ml freedom;
        with discretize_non_ml_params=True, reuse also happens organically
        when proposals snap to the same grid cell. GPry guideline: keep
        batch_size ~ min(n_free_params, n_workers).
```

(b) Append to `dev_notes/bayesopt_generator.md`:

```markdown
## v2 modernization (2026-08-22)

Production-driven upgrade, spec: docs/superpowers/specs/2026-08-22-bayesopt-modernization-design.md.

New generator_settings keys (defaults preserve v1 behavior):
- exploration_schedule: 'constant'|'annealed' (R1, GPry zeta-dim scaling), beta_start/beta_end/anneal_batches
- n_annealed_members: tempered-posterior batch members, tau_start/tau_decay/tau_min (R2, SALE annealed objective)
- trust_region: TuRBO-lite refinement, tr_* keys (R4)
- pred_eps_rel/pred_eps_abs/pred_hits_needed: prediction-accuracy counter, status key gp_predictions_accurate (R3, GPry CorrectCounter)

Behavior changes (not behind flags):
- Triaxial feasibility now works for ANY free subset of (q,p,u) — fixed axes
  use their parset values (production: q,p free, u=0.9999 fixed).
- Warm-start: restarting BayesOptGenerator on a populated output_directory
  trains on all historical rows; out-of-bounds rows are clipped with a
  warning; axial warm-up centers on the best historical model when
  initial_guess is absent.
- Snapped batch cells are deduplicated; freed slots refill with annealed
  then Sobol draws.

Literature: SALE (arXiv:2608.00841), GPry (2023 JCAP 10 021),
TuRBO (arXiv:1910.01739); full list in the spec.
```

In `CLAUDE.md`, under the BayesOptGenerator section, add:

```markdown
**v2 (2026-08-22):** partial-free triaxial feasibility (q/p free with u fixed works), in-place GP warm-start from existing all_models.ecsv (switch generator_type in the same output_directory), batch dedup after grid snapping, and opt-in acquisition upgrades: `exploration_schedule: annealed`, `n_annealed_members`, `trust_region: true`, prediction-accuracy stopping (`gp_predictions_accurate`). Env smoke test: `python dev_tests/test_bayesopt_smoke.py`. Production reference config: `dev_tests/NGC5139_config_production.yaml` (REQUIRES `modeliterator: SplitModelIterator` — memory cap depends on it).
```

- [ ] **Step 4: Full suite + commit**

Run: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py`
Expected: all PASS.

```bash
git add dev_tests/run_ablation.py dev_notes/bayesopt_generator.md CLAUDE.md
git commit -m "feat(bayesopt): ablation runner and v2 documentation"
```

---

## Post-plan validation (manual, after Task 12)

1. Full suite: `/opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_generator.py` — all PASS.
2. Smoke: `python dev_tests/test_bayesopt_smoke.py` — SMOKE OK.
3. V3 (cluster): rerun `dev_tests/run_comparison_real.py --skip-runs` plots first, then a fresh 100-model comparison; beat 5814 chi2 at equal budget.
4. V4 (cluster): NGC5139 production config at reduced orblib, then warm-start continuation from the GridWalk run.

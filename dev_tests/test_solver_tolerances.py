"""Solver tolerances must be derived from nnls_dtype, not inherited blindly.

A tolerance at or below the dtype's epsilon makes the convergence test it
controls unsatisfiable: the solve can then only stop by exhausting its
iteration budget, and it does so silently.

This is what took down a production grid. The config set ``nnls_dtype:
float32`` (eps 1.19e-07) and left ``adelie_tol`` unset, so it inherited the
float64-era default of 1e-10. Every BVLS call ran to ``max_iters=2e5``; at
omega Cen's matrix size that is ~29 h per ALM iterate, times 200 iterates.
Five workers sat in ALM iterate 0 for 31.5 h and completed 0 of 90 models,
with flat RSS and nothing in the logs to say why.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_solver_tolerances.py
"""

import logging

import numpy as np

from dynamite import weight_solvers
from dynamite.weight_solvers import NNLS, WeightSolver


def _make(settings):
    """Run the real NNLS.__init__ with the heavy base constructor neutralised.

    Only WeightSolver.__init__ is stubbed (it needs a Configuration, a System
    and a model directory); every line of NNLS.__init__ still executes.
    """
    original_base = WeightSolver.__init__
    original_mass = NNLS.get_observed_mass_constraints
    # Both need a Configuration/System; neither is involved in deriving the
    # tolerances, which is all this file is about.
    WeightSolver.__init__ = lambda self, **kw: None
    NNLS.get_observed_mass_constraints = lambda self: None
    try:
        solver = NNLS.__new__(NNLS)
        solver.settings = settings
        NNLS.__init__(solver, nnls_solver=settings["nnls_solver"])
        return solver
    finally:
        WeightSolver.__init__ = original_base
        NNLS.get_observed_mass_constraints = original_mass


def test_float32_defaults_stay_above_epsilon():
    s = _make({"nnls_solver": "adelie", "nnls_dtype": "float32"})
    eps = np.finfo(np.float32).eps
    assert s.adelie_tol > eps, (s.adelie_tol, eps)
    assert s.adelie_gap_tol > eps, (s.adelie_gap_tol, eps)


def test_float64_defaults_are_unchanged():
    """float64 behaviour must not shift - existing results depend on it."""
    s = _make({"nnls_solver": "adelie", "nnls_dtype": "float64"})
    assert s.adelie_tol == 1.0e-10, s.adelie_tol
    assert s.adelie_gap_tol == 1.0e-10, s.adelie_gap_tol
    assert s.adelie_tol > np.finfo(np.float64).eps


def test_the_default_differs_by_dtype():
    """The whole point: the same config text must not yield the same tolerance
    at both dtypes, because eps differs by nine orders of magnitude."""
    f32 = _make({"nnls_solver": "adelie", "nnls_dtype": "float32"})
    f64 = _make({"nnls_solver": "adelie", "nnls_dtype": "float64"})
    assert f32.adelie_tol != f64.adelie_tol


def test_explicit_setting_still_wins():
    s = _make({"nnls_solver": "adelie", "nnls_dtype": "float64",
               "adelie_tol": 3e-9, "adelie_gap_tol": 4e-7})
    assert s.adelie_tol == 3e-9
    assert s.adelie_gap_tol == 4e-7


def test_an_unreachable_explicit_tolerance_warns(caplog=None):
    """Overriding to something below eps is allowed, but must not be silent."""
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    logger = logging.getLogger(weight_solvers.__name__ + ".NNLS")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        _make({"nnls_solver": "adelie", "nnls_dtype": "float32",
               "adelie_tol": 1e-10})
    finally:
        logger.removeHandler(handler)
    msgs = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert any("adelie_tol" in m and "epsilon" in m for m in msgs), msgs


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("dtype-derived solver tolerances, OK")

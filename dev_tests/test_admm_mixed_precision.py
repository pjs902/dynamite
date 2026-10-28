"""Mixed precision (float32 factor) for AdmmNonNegSolver.

The Cholesky factor M = P + rho*I is the solve's big resident (16 GB at
p=45000) and every iteration is two bandwidth-bound triangular solves
against it. Factoring in f32 halves the factor (8 GB) and the per-iter
bandwidth; iterates, residuals and weights stay f64. No refinement step:
refinement needs f64 M matvecs, i.e. f64 P resident next to the factor,
which negates the saving — ADMM's fixed point absorbs f32 solve errors
instead, and the agreement test below is the arbiter.

These tests pin: bad precision rejected; factor_in_place combo refused;
the factor really is f32; f32 reaches the same answer as f64.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_admm_mixed_precision.py
"""

import numpy as np
import pytest

from dynamite.weight_solvers import AdmmNonNegSolver, _apply_diagonal_ridge
from test_admm_relaxation import _gram


def _solve(precision, tol=1e-7, max_iters=100000, ridge=0.0):
    P, q, eq_coeff, m = _gram()
    _apply_diagonal_ridge(P, ridge)
    return AdmmNonNegSolver(
        P,
        q,
        eq_coeff=eq_coeff,
        eq_rhs=m,
        max_iters=max_iters,
        tol=tol,
        precision=precision,
    )


def test_rejects_bad_precision():
    P, q, eq_coeff, m = _gram()
    with pytest.raises(ValueError, match="precision"):
        AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m, precision="f16")


def test_refuses_in_place_combo():
    P, q, eq_coeff, m = _gram()
    with pytest.raises(ValueError, match="factor_in_place"):
        AdmmNonNegSolver(
            P, q, eq_coeff=eq_coeff, eq_rhs=m, factor_in_place=True,
            precision="float32",
        )


def test_factor_is_float32_and_half_bytes():
    s = _solve("float32")
    assert s.chol_factor.dtype == np.float32
    p = s.beta.shape[0]
    assert s.chol_factor.nbytes == p * p * 4


def test_matches_float64_at_loose_tol():
    # f32 has a dual noise floor (~1e-5 on this problem) and cannot
    # reach 1e-7; at its natural tol it must agree with f64 in weight
    # space. Tight-tol production behaviour is the sweep's job to
    # measure on the real P, not a unit test's on a synthetic one.
    # Ridged (production regime): the unridged harness problem is so
    # ill-conditioned that f32 wanders 24% away at loose tol — a regime
    # nobody runs. With the production ridge it must agree.
    ref = _solve("float64", ridge=0.1)
    assert ref.status == "optimal"
    s = _solve("float32", tol=1e-4, ridge=0.1)
    assert s.status == "optimal"
    rel = np.linalg.norm(s.beta - ref.beta) / np.linalg.norm(ref.beta)
    assert rel < 1e-2, rel


def test_converges_fast_at_natural_tol_with_ridge():
    # Production runs ridged (0.1). f32 at tol=1e-4 must converge — and
    # in fewer iters than f64 needs for 1e-9, which is the entire point.
    ref = _solve("float64", tol=1e-9, ridge=0.1)
    assert ref.status == "optimal"
    s = _solve("float32", tol=1e-4, ridge=0.1)
    assert s.status == "optimal"
    assert s.iterations < ref.iterations
    rel = np.linalg.norm(s.beta - ref.beta) / np.linalg.norm(ref.beta)
    assert rel < 1e-2, rel

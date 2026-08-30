"""Surrogate chi2-vector and KKT identities vs direct computation.

The surrogates are algebraically identical to the A-based forms but re-order
the floating-point operations, so they are tested to rtol 1e-10, NOT bitwise.
The scaled KKT value must stay inside [0, 1].

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_surrogate_chi2_kkt.py
"""

import numpy as np
from scipy.optimize import nnls

from dynamite.weight_solvers import NNLS, chi2_vector_from_residuals


def _problem(rng, n_rows, n_orbs, dtype=np.float64, row0_scale=1e8, noise=0.0):
    """A test problem. NOTE: with ``noise=0`` the right-hand side is
    ``A @ w_true`` for a NONNEGATIVE ``w_true``, so it lies in the feasible
    cone and nnls recovers it EXACTLY - such a problem exercises the
    exact-fit guard, not the general ratio. Pass ``noise > 0`` for a
    genuinely inexact fit."""
    A = rng.standard_normal((n_rows, n_orbs)).astype(dtype)
    A[0, :] = row0_scale  # mimic the total-mass row scale
    w_true = rng.random(n_orbs) ** 3  # mostly-near-zero weights
    w_true[:3] = 0.0  # force some exact zeros
    b = A @ w_true.astype(dtype)
    if noise:
        b = b + (rng.standard_normal(n_rows) * noise).astype(dtype)
        b[0] = A[0] @ np.full(n_orbs, 0.3, dtype=dtype)
    return A, b


def _kkt_pair(A, b, mu=1e7):
    """(augmented, stock) KKT values for the same problem and solution."""
    X, col_norm, _y = NNLS._build_augmented_X(A[1:], b[1:], np.sqrt(mu), np.float64)
    w, _ = nnls(A, b)
    resid_full = A @ w - b  # plain residual aligned to A rows
    got = NNLS.kkt_violation_augmented(A[0], b[0], X, col_norm, resid_full, w, mu)
    ref = NNLS.kkt_violation(A, b, w)
    return got, ref


def test_kkt_matches_stock():
    """The SCALED value agrees between the two forms on the production
    scaling: a 1e8 total-mass row and a genuinely inexact fit.

    Only the scaled value is asserted. `raw` is a max over
    `grad = A^T (Aw - b)`, and with A[0, :] = 1e8 that gradient is dominated
    by `A[0, j] * r0` where `r0 = A[0] @ w - b[0]` cancels ~8 significant
    digits; the two forms sum those terms in different orders and disagree
    by tens of percent while BOTH sit far from the true value. Asserting
    agreement on it would pin round-off, not algebra - see
    test_kkt_raw_matches_stock_when_well_conditioned for the raw claim.
    """
    rng = np.random.default_rng(7)
    A, b = _problem(rng, 300, 40, noise=0.5)
    got, ref = _kkt_pair(A, b)
    assert 0.0 <= got[0] <= 1.0, got
    assert ref[0] > 0.0, ref  # a real residual: not the exact-fit branch
    assert np.isclose(got[0], ref[0], rtol=1e-10), (got, ref)


def test_exact_fit_with_the_production_mass_row_is_zero():
    """The exact-fit guard must survive A[0, :] = 1e8.

    This is the case that rules out the obvious guard
    `||resid|| <= rtol * ||b||`: here ||b|| ~ 1e8, so that form would also
    swallow genuine residuals up to ~1e-4 (see
    test_a_small_but_genuine_residual_is_not_called_an_exact_fit). The floor
    has to be per row.
    """
    rng = np.random.default_rng(7)
    A, b = _problem(rng, 300, 40)  # attainable rhs -> nnls fits exactly
    got, ref = _kkt_pair(A, b)
    assert got[0] == 0.0, got
    assert ref[0] == 0.0, ref


def test_a_small_but_genuine_residual_is_not_called_an_exact_fit():
    """A real 1e-3 residual against ||b|| ~ 1e8 must NOT read as exact."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((30, 12))
    A[0, :] = 1e8
    w = np.zeros(12)
    w[:5] = rng.random(5) + 0.1
    b = A @ w
    b[7] += 1.0e-3  # genuine, and 11 orders below ||b||
    mu = 1e7
    X, col_norm, _y = NNLS._build_augmented_X(A[1:], b[1:], np.sqrt(mu), np.float64)
    resid_full = A @ w - b
    got = NNLS.kkt_violation_augmented(A[0], b[0], X, col_norm, resid_full, w, mu)
    ref = NNLS.kkt_violation(A, b, w)
    assert got[0] > 0.0 and ref[0] > 0.0, (got, ref)
    assert np.isclose(got[0], ref[0], rtol=1e-10), (got, ref)


def test_kkt_raw_matches_stock_when_well_conditioned():
    """Same identity, without the cancellation: here `raw` must agree.

    Dropping the mass row to O(1) removes the catastrophic subtraction that
    makes `raw` meaningless above, leaving the pure algebraic claim that the
    augmented gradient equals A^T (Aw - b). It agrees exactly at this seed.
    """
    rng = np.random.default_rng(7)
    A, b = _problem(rng, 300, 40, row0_scale=1.0)
    got, ref = _kkt_pair(A, b)
    assert 0.0 <= got[0] <= 1.0, got
    assert np.isclose(got[0], ref[0], rtol=1e-10), (got, ref)
    assert np.isclose(got[1], ref[1], rtol=1e-10), (got, ref)


def test_exact_fit_returns_zero_scaled():
    """An exactly-fitting solution reports scaled = 0, not a ratio of noise.

    Regression test: the guard used to be `if not np.any(scale > 0)`, which
    fires only on an identically-zero residual. Here the fit IS exact but
    r0 = A[0] @ w - b[0] rounds to -1.1e-16, so the guard missed and the
    ratio of two round-off quantities was reported as scaled = 0.285 - above
    solve_adelie_alm's 0.1 warning threshold, i.e. a perfect fit raising a
    spurious non-convergence warning. See _residual_is_all_noise.
    """
    rng = np.random.default_rng(11)
    n_orbs, n_mass = 12, 30
    A = rng.standard_normal((n_mass, n_orbs))
    w = np.zeros(n_orbs)
    w[:5] = rng.random(5) + 0.1
    b = A @ w  # exactly attainable
    mu = 1e7
    X, col_norm, _y = NNLS._build_augmented_X(A[1:], b[1:], np.sqrt(mu), np.float64)
    resid_full = A @ w - b  # ~0 everywhere -> ||r|| ~ 0 guard
    scaled, raw = NNLS.kkt_violation_augmented(
        A[0], b[0], X, col_norm, resid_full, w, mu
    )
    assert raw <= 1e-8 and scaled <= 1e-8, (scaled, raw)


def test_chi2_vector_identity():
    rng = np.random.default_rng(13)
    A, b = _problem(rng, 120, 25)
    w, _ = nnls(A, b)
    resid_full = A @ w - b
    got = chi2_vector_from_residuals(resid_full, float(resid_full[0]) ** 2)
    ref = (A @ w - b) ** 2
    assert np.array_equal(got, ref)  # pure reshape/square: bitwise
    assert got.shape == ref.shape


if __name__ == "__main__":
    test_kkt_matches_stock()
    test_exact_fit_returns_zero_scaled()
    test_chi2_vector_identity()
    print("test_surrogate_chi2_kkt OK")

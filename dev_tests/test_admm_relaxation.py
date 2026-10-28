"""Over-relaxation (alpha) for AdmmNonNegSolver.

Relaxed ADMM extrapolates past the w-update before projecting:
``w_hat = alpha*w + (1-alpha)*z_old; z = max(w_hat + u, 0)`` with
alpha in (0, 2); alpha=1 is exactly today's unrelaxed method. The
zigzag between w (overshooting the feasible set) and z (clipping back)
is what relaxation damps, typically worth 1.2-2x fewer iterations for
two changed lines, no refactorization, no new memory.

Theory: J. Eckstein & D. P. Bertsekas, Math. Programming 55:293-318,
1992 (generalized ADMM with relaxation factors); practical alpha range
1.5-1.8: Boyd et al., Found. Trends Mach. Learn. 3(1):1-122, 2011.

These tests pin: alpha=1.0 reproduces today's iterates bitwise (the
relaxation is cleanly isolated, not a rewrite); other alphas reach the
SAME answer (weights agreement + chi2 vs cvxopt, mirroring the rho
invariance test) while genuinely engaging different trajectories; and
out-of-range alpha is rejected rather than silently diverging.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_admm_relaxation.py
"""

import numpy as np
import pytest

from dynamite.weight_solvers import AdmmNonNegSolver, CvxoptNonNegSolver

ALPHAS = (1.5, 1.6, 1.8)


def _problem(seed=4, p=300):
    """Same row structure as test_kktsolver_and_admm._problem (mass rows
    with a wide dynamic range + kinematics), i.e. the conditioning regime
    the relaxation must survive."""
    rng = np.random.default_rng(seed)
    n_int, n_prj, n_kin = 60, 50, 250
    n = n_int + n_prj + n_kin
    mass_int = 10.0 ** rng.uniform(-17, -4, n_int)
    mass_prj = 10.0 ** rng.uniform(-8, -2, n_prj)
    w_true = np.abs(rng.random(p)) ** 3
    w_true /= w_true.sum()
    share = rng.dirichlet(np.ones(p) * 2.0, size=n_int + n_prj)
    orbmat = np.empty((n, p))
    econ = np.empty(n)
    orbmat[:n_int] = mass_int[:, None] * share[:n_int] * p
    orbmat[n_int : n_int + n_prj] = mass_prj[:, None] * share[n_int:] * p
    econ[:n_int] = mass_int * 0.01
    econ[n_int : n_int + n_prj] = mass_prj * 0.02
    orbmat[n_int + n_prj :] = rng.standard_normal((n_kin, p)) * 0.5
    econ[n_int + n_prj :] = np.abs(rng.random(n_kin)) * 0.3 + 0.05
    con = orbmat @ w_true + 1e-3 * np.abs(rng.random(n)) * econ
    A = orbmat / econ[:, None]
    b = con / econ
    return A, b


def _gram(total_mass=1.0):
    A, b = _problem()
    col_norm = np.linalg.norm(A, axis=0)
    col_norm[col_norm == 0] = 1.0
    b_max = np.max(np.abs(b)) or 1.0
    An, bn = A / col_norm, b / b_max
    P = An.T @ An
    q = -(An.T @ bn)
    eq_coeff = b_max / col_norm
    return P, q, eq_coeff, total_mass


def _solve(alpha, tol=1e-7, max_iters=100000):
    P, q, eq_coeff, m = _gram()
    return AdmmNonNegSolver(
        P,
        q,
        eq_coeff=eq_coeff,
        eq_rhs=m,
        max_iters=max_iters,
        tol=tol,
        alpha=alpha,
    )


def test_alpha_one_is_identical_to_unrelaxed():
    """alpha=1.0 must reduce to today's method EXACTLY -- same iteration
    count, bitwise-identical weights and residuals. If the relaxation is
    anything but a strict generalization, this fails. Capped budget: this
    problem stalls its dual residual above 1e-9, so both runs stop at the
    cap -- with identical trajectories, which is exactly what is pinned."""
    P, q, eq_coeff, m = _gram()
    kw = dict(eq_coeff=eq_coeff, eq_rhs=m, max_iters=2000, tol=1e-9)
    s_plain = AdmmNonNegSolver(P, q, **kw)
    s_one = AdmmNonNegSolver(P, q, alpha=1.0, **kw)
    assert s_one.iterations == s_plain.iterations == 2000
    assert np.array_equal(s_one.beta, s_plain.beta)
    assert s_one.r_pri == s_plain.r_pri and s_one.r_dual == s_plain.r_dual
    assert s_one.alpha == 1.0


def test_alpha_default_is_one():
    """Production behavior is unchanged until alpha is explicitly set."""
    P, q, eq_coeff, m = _gram()
    s = AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m, max_iters=100, tol=1e-6)
    assert s.alpha == 1.0


def test_alpha_changes_trajectory_not_the_answer():
    """Relaxed runs must converge to the same weights (rel < 2e-2, the rho
    invariance test's standard, at tol=1e-7 -- tight enough to converge
    past this problem's dual-residual stall, loose enough to stay fast)
    while taking genuinely different trajectories -- otherwise alpha would
    be dead code that only renames the same iterates."""
    ref = _solve(1.0)
    assert ref.success
    iters = {}
    for alpha in ALPHAS:
        s = _solve(alpha)
        assert s.success, (alpha, s.status, s.iterations)
        rel = np.linalg.norm(s.beta - ref.beta) / np.linalg.norm(ref.beta)
        assert rel < 2e-2, (alpha, rel)
        iters[alpha] = s.iterations
    assert any(n != ref.iterations for n in iters.values()), (
        f"alpha never engaged: all iters == {ref.iterations}"
    )


def test_alpha_agrees_with_cvxopt_on_chi2():
    """Independent-algorithm anchor per alpha: chi2 must match cvxopt
    within the same 1e-6 relative standard as the unrelaxed solver."""
    P, q, eq_coeff, m = _gram()
    s_cvx = CvxoptNonNegSolver(
        P, q, eq_coeff=eq_coeff, eq_rhs=m, tol=1e-12, maxiters=200
    )
    assert s_cvx.success

    def chi2(w):
        return float(0.5 * w @ (P @ w) + q @ w)

    c_cvx = chi2(s_cvx.beta)
    for alpha in (1.0,) + ALPHAS:
        s = AdmmNonNegSolver(
            P, q, eq_coeff=eq_coeff, eq_rhs=m, max_iters=20000, tol=5e-6, alpha=alpha
        )
        assert s.success, (alpha, s.status)
        c = chi2(s.beta)
        assert abs(c - c_cvx) <= 1e-6 * max(abs(c_cvx), 1.0), (alpha, c)


@pytest.mark.parametrize("bad", [0.0, -0.5, 2.0, 2.5])
def test_alpha_out_of_range_raises(bad):
    """alpha outside (0, 2) has no convergence guarantee -- reject it
    loudly rather than silently diverging or (at 0) freezing."""
    P, q, eq_coeff, m = _gram()
    with pytest.raises(ValueError):
        AdmmNonNegSolver(
            P, q, eq_coeff=eq_coeff, eq_rhs=m, max_iters=100, tol=1e-6, alpha=bad
        )


if __name__ == "__main__":
    test_alpha_one_is_identical_to_unrelaxed()
    print("alpha=1 identical OK")
    test_alpha_default_is_one()
    print("alpha default OK")
    test_alpha_changes_trajectory_not_the_answer()
    print("alpha trajectory/answer OK")
    test_alpha_agrees_with_cvxopt_on_chi2()
    print("alpha chi2 OK")
    for _bad in (0.0, -0.5, 2.0, 2.5):
        try:
            test_alpha_out_of_range_raises(_bad)
        except TypeError:
            raise AssertionError(f"alpha={_bad} must raise ValueError")
    print("alpha range OK")
    print("test_admm_relaxation OK")

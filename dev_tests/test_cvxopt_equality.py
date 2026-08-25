"""cvxopt path: total mass as a hard equality, not a 1e8-weighted matrix row.

``orbmat[0,:]`` is all ones and ``econ[0] = max(|1-total_mass|, 1e-8)``, which
for a normalised MGE lands on the 1e-8 floor. That makes ``A[0,:] = 1e8`` in
every column - a single constant row six orders above the rest of the matrix.
Carrying it as a least-squares row costs ~5 orders of magnitude of conditioning,
and ``P = An^T An`` squares that, which is past float64's limit. cvxopt takes
equality constraints natively, so the row is dropped and ``sum(w) = total_mass``
is stated exactly.

These tests pin the three properties that buys us: the constraint holds to
machine precision, the conditioning is orders better, and a non-converged solve
is reported rather than returned silently.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_cvxopt_equality.py
"""

import numpy as np

from dynamite.weight_solvers import CvxoptNonNegSolver


def _problem(seed=2, p=200):
    """A matrix with the real row structure: the row scale cancels between
    orbmat and econ, and only the total-mass row is left at 1e8."""
    rng = np.random.default_rng(seed)
    n_int, n_prj, n_kin = 120, 100, 300
    n = 1 + n_int + n_prj + n_kin
    mass_int = 10.0 ** rng.uniform(-17, -4, n_int)
    mass_prj = 10.0 ** rng.uniform(-8, -2, n_prj)
    w_true = np.abs(rng.random(p)) ** 3
    w_true /= w_true.sum()
    share = rng.dirichlet(np.ones(p) * 2.0, size=n_int + n_prj)
    orbmat = np.empty((n, p))
    econ = np.empty(n)
    con = np.empty(n)
    orbmat[0, :], econ[0], con[0] = 1.0, 1e-8, 1.0
    orbmat[1 : 1 + n_int] = mass_int[:, None] * share[:n_int] * p
    orbmat[1 + n_int : 1 + n_int + n_prj] = mass_prj[:, None] * share[n_int:] * p
    econ[1 : 1 + n_int] = mass_int * 0.01
    econ[1 + n_int : 1 + n_int + n_prj] = mass_prj * 0.02
    orbmat[1 + n_int + n_prj :] = rng.standard_normal((n_kin, p)) * 0.5
    econ[1 + n_int + n_prj :] = np.abs(rng.random(n_kin)) * 0.3 + 0.05
    con[1:] = orbmat[1:] @ w_true + 1e-3 * np.abs(rng.random(n - 1)) * econ[1:]
    return orbmat / econ[:, None], con / econ, w_true


def _solve_equality(A, b, total_mass=1.0, tol=1e-11, maxiters=200):
    """The scaling and call that WeightSolver.solve uses for nnls_solver=cvxopt."""
    A_rest, b_rest = A[1:], b[1:]
    col_norm = np.linalg.norm(A_rest, axis=0)
    col_norm[col_norm == 0] = 1.0
    b_max = np.max(np.abs(b_rest)) or 1.0
    An, bn = A_rest / col_norm, b_rest / b_max
    P = An.T @ An
    q = -(An.T @ bn)
    solver = CvxoptNonNegSolver(
        P, q, eq_coeff=b_max / col_norm, eq_rhs=total_mass,
        tol=tol, maxiters=maxiters,
    )
    return solver, solver.beta * b_max / col_norm, P


def test_mass_constraint_is_exact_and_weights_nonnegative():
    A, b, _ = _problem()
    solver, w, _ = _solve_equality(A, b)
    assert solver.success, solver.status
    # a hard equality holds to machine precision; the 1e8-row form gave ~1e-9
    assert abs(w.sum() - 1.0) < 1e-12, w.sum() - 1.0
    assert w.min() > -1e-10, w.min()


def test_dropping_the_mass_row_conditions_P():
    """P must stay solvable in float64. With the 1e8 row carried in the matrix
    kappa(P) runs to ~1e14+, at which cvxopt reports 'optimal' on a bad answer."""
    A, b, _ = _problem()
    _, _, P_eq = _solve_equality(A, b)
    A_max = np.max(np.abs(A), axis=0)
    An_row = A / A_max
    P_row = An_row.T @ An_row
    k_eq, k_row = np.linalg.cond(P_eq), np.linalg.cond(P_row)
    assert k_eq < 1e9, k_eq
    assert k_eq < k_row / 1e4, (k_eq, k_row)


def test_agrees_with_an_independent_constrained_solve():
    """Cross-check against a different algorithm (scipy's trust-region BVLS) on
    the same scaled, equality-augmented problem.

    This is also what pins ``cvxopt_tol``: at 1e-9 cvxopt still reports
    'optimal' but sits 1.25-3.3x above this reference chi2, so the status flag
    alone does not establish convergence.
    """
    from scipy.optimize import lsq_linear

    A, b, _ = _problem()
    _, w, _ = _solve_equality(A, b)
    A_rest, b_rest = A[1:], b[1:]
    col_norm = np.linalg.norm(A_rest, axis=0)
    col_norm[col_norm == 0] = 1.0
    b_max = np.max(np.abs(b_rest))
    An, bn = A_rest / col_norm, b_rest / b_max
    lam = 1e3 * np.linalg.norm(An)  # heavy row => equality in all but name
    ref = lsq_linear(
        np.vstack([An, lam * (b_max / col_norm)[None, :]]),
        np.concatenate([bn, [lam * 1.0]]),
        bounds=(0, np.inf), tol=1e-14, max_iter=500,
    )
    chi2 = lambda x: float(np.sum((A @ x - b) ** 2))
    assert chi2(w) <= 1.10 * chi2(ref.x * b_max / col_norm), (
        chi2(w), chi2(ref.x * b_max / col_norm)
    )


def test_non_convergence_is_reported():
    """cvxopt returns iterates when it gives up; the caller must be able to see
    that rather than turn them into a chi2 like any other."""
    A, b, _ = _problem()
    solver, _, _ = _solve_equality(A, b, tol=1e-14, maxiters=3)
    assert not solver.success
    assert solver.status != "optimal"
    assert solver.iterations == 3


def test_solver_options_are_left_as_found():
    """cvxopt.solvers.options is module-global; the solver must not leak into it."""
    import cvxopt

    before = dict(cvxopt.solvers.options)
    A, b, _ = _problem()
    _solve_equality(A, b)
    assert dict(cvxopt.solvers.options) == before


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("cvxopt equality-constraint path, OK")

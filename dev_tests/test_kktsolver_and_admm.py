"""cvxopt custom kktsolver, its observability, and the ADMM Gram-form solver.

cvxopt's vendored OpenBLAS is built SINGLE-THREADED: measured
cvxopt.lapack.potrf at 0.06 Tflop/s at any thread count, vs scipy's dpotrf at
0.93-1.33 Tflop/s. At omega Cen's p=45000 a production run burned 4262s of
wall time in ONE Cholesky factorization (py-spy: stuck in
cvxopt/misc.py:factor while 46 other threads sat idle), and two production
runs died or were killed without ever producing weights.

``CvxoptNonNegSolver(..., kktsolver="custom")`` (the new default) replaces
cvxopt's own KKT reduction with one routed through scipy's LAPACK instead -
same linear algebra, faster BLAS. These tests pin: it agrees with cvxopt's
own ("default") KKT path to ~1e-8 on a problem with the real row structure;
it falls back to "default" (and says so) when the equality is missing or
has more than one row; the solve is no longer silent - status, iterations,
gap and elapsed time are always logged at INFO, and a non-optimal status is
logged at WARNING rather than returned as if converged (the same failure
class as the incident where a silent 31.5h solve finished 0/90 models with
nothing in the logs).

``AdmmNonNegSolver`` solves the identical Gram-form QP with a fixed-rho ADMM
splitting instead of interior point, factoring M = P + rho*I ONCE (rho does
not change, unlike interior point's P + W^-2) and reusing that factorization
for every iteration. These tests pin: it agrees with cvxopt on chi2; the
equality constraint holds to machine precision; it produces EXACT zeros
(cvxopt's interior point iterate does not); and a non-converged ADMM solve
is reported, not silently returned.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_kktsolver_and_admm.py
"""

import logging

import numpy as np

from dynamite.weight_solvers import AdmmNonNegSolver, CvxoptNonNegSolver


def _problem(seed=4, p=300):
    """Same row structure as test_cvxopt_equality.py's _problem: an
    intrinsic/projected-mass/kinematics matrix with the 1e8 total-mass row
    already dropped, i.e. what CvxoptNonNegSolver/AdmmNonNegSolver actually
    see once WeightSolver.solve has done its equality-drop scaling."""
    rng = np.random.default_rng(seed)
    # n > p so P = A^T A is full rank (strictly convex objective) - with a
    # rank-deficient P the QP has a flat direction and different rho values
    # can converge to different points along it, which is a degeneracy of
    # the test problem, not a property of ADMM/rho.
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
    return P, q, eq_coeff, total_mass, col_norm, b_max


def test_custom_and_default_kktsolver_agree():
    """Same problem, both KKT paths: cvxopt's iterates should match to
    ~1e-8, matching what was measured at small p in the reference diagnostic
    (identical to ~1e-15 for a hand-checked synthetic P; the wider tolerance
    here allows for this problem's worse conditioning)."""
    P, q, eq_coeff, m, _, _ = _gram()
    s_custom = CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                                   tol=1e-12, maxiters=200, kktsolver="custom")
    s_default = CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                                    tol=1e-12, maxiters=200, kktsolver="default")
    assert s_custom.kktsolver_used == "custom"
    assert s_default.kktsolver_used == "default"
    assert s_custom.success and s_default.success
    rel = np.linalg.norm(s_custom.beta - s_default.beta) / np.linalg.norm(s_default.beta)
    assert rel < 1e-8, rel


def test_custom_kktsolver_falls_back_without_a_single_equality_row():
    """The custom KKT reduction is only valid for exactly one equality row.
    With none, or more than one, it must fall back to cvxopt's default and
    say so via kktsolver_used - never silently misapply the reduction."""
    p = 60
    rng = np.random.default_rng(1)
    P = np.eye(p) + 0.01 * rng.standard_normal((p, p))
    P = P @ P.T
    q = -rng.random(p)

    s_no_eq = CvxoptNonNegSolver(P, q, eq_coeff=None, eq_rhs=None,
                                  tol=1e-10, maxiters=100, kktsolver="custom")
    assert s_no_eq.kktsolver_used == "default"

    A2 = np.vstack([np.ones(p), rng.random(p)])
    s_multi_eq = CvxoptNonNegSolver(P, q, eq_coeff=A2, eq_rhs=[1.0, 0.5],
                                     tol=1e-10, maxiters=100, kktsolver="custom")
    assert s_multi_eq.kktsolver_used == "default"


def test_solve_is_logged_even_when_show_progress_is_false(caplog):
    """A multi-hour cvxopt solve with show_progress=False (the default,
    preserved for stdout) must still be observable: status, iterations, gap
    and elapsed time are always logged at INFO. This is the fix for the
    silent-solve failure class in the module docstring."""
    P, q, eq_coeff, m, _, _ = _gram()
    logger = logging.getLogger("test_kktsolver_and_admm.cvxopt_silent")
    with caplog.at_level(logging.INFO, logger=logger.name):
        solver = CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                                     tol=1e-11, maxiters=200,
                                     show_progress=False, logger=logger)
    assert solver.success
    msgs = [r.message for r in caplog.records if r.name == logger.name]
    joined = " ".join(msgs)
    assert "status=" in joined and "iterations=" in joined
    assert "elapsed=" in joined and "gap=" in joined


def test_nonoptimal_cvxopt_status_warns(caplog):
    """A status other than 'optimal' must be logged at WARNING, not folded
    silently into the returned iterate."""
    P, q, eq_coeff, m, _, _ = _gram()
    logger = logging.getLogger("test_kktsolver_and_admm.cvxopt_nonopt")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        solver = CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                                     tol=1e-14, maxiters=2, logger=logger)
    assert not solver.success
    warnings = [r for r in caplog.records
                if r.name == logger.name and r.levelno == logging.WARNING]
    assert len(warnings) >= 1
    assert "NOT a verified optimum" in warnings[0].message


def test_kktsolver_heartbeat_fires_every_log_every_factorizations(caplog):
    """The heartbeat is the ONLY way to tell a slow solve from a hung one at
    production scale (~20s/factorization with the custom solver). Pin that
    it actually logs, at the requested cadence."""
    P, q, eq_coeff, m, _, _ = _gram()
    logger = logging.getLogger("test_kktsolver_and_admm.heartbeat")
    with caplog.at_level(logging.INFO, logger=logger.name):
        CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                            tol=1e-12, maxiters=200, kktsolver="custom",
                            log_every=1, logger=logger)
    hb = [r.message for r in caplog.records
          if r.name == logger.name and "factorization" in r.message]
    assert len(hb) >= 1, "expected at least one heartbeat log line"


def test_admm_agrees_with_cvxopt_on_chi2():
    """Different algorithm, same convex QP: the objective value must agree
    even though ADMM's iterate is exactly sparse and cvxopt's is not."""
    P, q, eq_coeff, m, _, _ = _gram()
    s_cvx = CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                                tol=1e-12, maxiters=200)
    s_admm = AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                               max_iters=20000, tol=5e-6)
    assert s_cvx.success and s_admm.success, (s_cvx.status, s_admm.status)

    def chi2(w):
        return float(0.5 * w @ (P @ w) + q @ w)

    c_cvx, c_admm = chi2(s_cvx.beta), chi2(s_admm.beta)
    assert abs(c_admm - c_cvx) <= 1e-6 * max(abs(c_cvx), 1.0), (c_cvx, c_admm)


def test_admm_equality_constraint_holds_to_machine_precision():
    P, q, eq_coeff, m, _, _ = _gram()
    solver = AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                               max_iters=20000, tol=5e-6)
    assert solver.success
    resid = abs(float(eq_coeff @ solver.beta) - m)
    assert resid < 1e-8, resid


def test_admm_yields_exact_zeros_cvxopt_does_not():
    """ADMM's z-update clips to 0.0; cvxopt's interior-point iterate sits
    strictly interior and (as measured in the reference diagnostics) is
    essentially fully dense. Neither is a defect - they are different
    stopping rules on the same convex problem."""
    P, q, eq_coeff, m, _, _ = _gram()
    s_cvx = CvxoptNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                                tol=1e-11, maxiters=200)
    s_admm = AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                               max_iters=20000, tol=5e-6)
    n_zero_admm = int(np.sum(s_admm.beta == 0.0))
    n_zero_cvx = int(np.sum(s_cvx.beta == 0.0))
    assert n_zero_admm > 0
    assert n_zero_admm > n_zero_cvx


def test_admm_rho_changes_iterations_not_the_answer():
    """rho is a pure cost knob: changing it must not move the converged
    solution, even though the iteration count varies a lot (measured
    elsewhere: up to 3 orders of magnitude). Larger rho needs more
    iterations to bring the DUAL residual down (r_dual = rho*||Z-Zo||), so
    this stays within a factor of 2 of the scale-matched default to keep
    the test fast while still genuinely converging every run, then
    compares the resulting weights directly."""
    P, q, eq_coeff, m, _, _ = _gram()
    results = {}
    base_rho = float(np.trace(P)) / P.shape[0]
    for factor in (0.5, 1.0, 2.0):
        s = AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                              rho=base_rho * factor, max_iters=100000, tol=1e-9)
        results[factor] = s
    ref = results[1.0].beta
    for factor, s in results.items():
        rel = np.linalg.norm(s.beta - ref) / np.linalg.norm(ref)
        assert rel < 2e-2, (factor, rel, s.status, s.iterations, s.r_pri, s.r_dual)


def test_admm_non_convergence_is_reported():
    """A tiny iteration budget must be reported as not converged, exactly
    like the analogous cvxopt test - ADMM must not silently hand back an
    unconverged iterate as if it were optimal."""
    P, q, eq_coeff, m, _, _ = _gram()
    solver = AdmmNonNegSolver(P, q, eq_coeff=eq_coeff, eq_rhs=m,
                               max_iters=1, tol=1e-14)
    assert not solver.success
    assert solver.status != "optimal"
    assert solver.iterations == 1


if __name__ == "__main__":
    import inspect

    class _FakeCaplog:
        """Minimal stand-in for pytest's caplog fixture so this file can
        also run as a plain script (matching this repo's convention of
        standalone dev_tests, not a pytest suite)."""

        def __init__(self):
            self.records = []
            self._handler = None

        def at_level(self, level, logger=None):
            import contextlib

            class _Handler(logging.Handler):
                def __init__(self2, outer):
                    super().__init__()
                    self2.outer = outer

                def emit(self2, record):
                    self2.outer.records.append(record)

            handler = _Handler(self)
            target_logger = logging.getLogger(logger)

            @contextlib.contextmanager
            def _ctx():
                target_logger.addHandler(handler)
                target_logger.setLevel(level)
                self.records = []
                try:
                    yield
                finally:
                    target_logger.removeHandler(handler)

            return _ctx()

    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        sig = inspect.signature(fn)
        if "caplog" in sig.parameters:
            fn(_FakeCaplog())
        else:
            fn()
        print(f"{name}: OK")
    print("kktsolver custom/default + ADMM, OK")

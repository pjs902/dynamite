"""ADMM frees P after factorization; chi2 recovered from C alone.

Two changes are pinned here (design: PM_grid/_diag_21_admm_free/NOTES.md):

1. ``AdmmNonNegSolver(factor_in_place=True)`` factors M = P + rho*I IN PLACE,
   destroying P and keeping only the Cholesky factor C. The iteration never
   touches P, so the resident set during the solve drops from P + C to C
   alone (~2.2x -> ~1.1x one p x p matrix at production scale). The iterates
   MUST be bitwise identical to the copy path - same matrix, same factor.

2. The Gram quadratic form survives without either big matrix:

       w'P_solver w            = ||L'w||^2 - rho          * ||w||^2
       w'G_norm w (unridged)   = ||L'w||^2 - (rho + lam*s)* ||w||^2

   via a TRIANGULAR matvec. Two silent-failure modes are pinned:
     - dpotrf(clean=0) leaves the unused triangle as garbage; a dense C.T@w
       reads it and corrupts the result at O(1) (measured). dtrmv must be
       used.
     - shift bookkeeping: subtracting anything other than exactly what was
       added to the factored matrix relative to the reference form produces
       errors of exactly lam*s*||w||^2 / w'Pw.

And the chi2 identity that makes raw G disposable:

    with w_raw = x * b_max / col,  P = G/outer(col,col),  q = -v/(col*b_max):

        w'Gw - 2 w'v = b_max^2 * ( x'Px + 2 x'q )

so chi2_rest can be read off normalized quantities alone and raw G freed at
finalize(). Verified against the direct residual sum on the same problem.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_admm_free_p.py
"""

import numpy as np
from scipy.linalg.lapack import dpotrf

from dynamite.weight_solvers import AdmmNonNegSolver


def _make_problem(p=400, seed=7, kappa=1e5, noise=1e-3):
    """SPD Gram-like P at kappa, positive sparse ground truth, consistent q."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    eig = np.logspace(0.0, -np.log10(kappa), p)
    P = (Q * eig) @ Q.T
    P = np.asfortranarray(0.5 * (P + P.T))
    w_true = np.zeros(p)
    idx = rng.choice(p, size=max(2, p // 6), replace=False)
    w_true[idx] = rng.random(idx.size) + 0.1
    b = P @ w_true
    sigma = noise * float(np.sqrt(np.mean(b**2)))
    q = -(b + rng.normal(0.0, sigma, p))
    a = np.abs(rng.standard_normal(p)) + 0.5
    m = float(np.sum(w_true))
    return P, q, a, m


def test_in_place_iterates_bitwise_identical():
    P, q, a, m = _make_problem()
    ref = AdmmNonNegSolver(P=P, q=q, eq_coeff=a, eq_rhs=m, max_iters=20000)
    Pc = P.copy()
    ip = AdmmNonNegSolver(
        P=Pc, q=q, eq_coeff=a, eq_rhs=m, max_iters=20000, factor_in_place=True
    )
    assert ip.chol_factor is not None
    assert not np.array_equal(Pc, P), "in-place run left the input untouched"
    assert np.array_equal(ip.beta, ref.beta), "iterates differ between paths"
    assert ip.iterations == ref.iterations
    assert ip.r_pri == ref.r_pri and ip.r_dual == ref.r_dual
    assert ip.success and ref.success


def test_quadratic_form_identity_accuracy():
    P, q, a, m = _make_problem(p=300)
    lam, s = 10.0, float(np.trace(P)) / P.shape[0]
    # real copy - np.asfortranarray would ALIAS an already-F-order P
    Pr = P.copy(order="F")
    Pr.flat[:: Pr.shape[0] + 1] += lam * s
    Pr_ref = Pr.copy()  # factor_in_place CONSUMES Pr; keep a pristine copy
    sol = AdmmNonNegSolver(
        P=Pr, q=q, eq_coeff=a, eq_rhs=m, max_iters=2000, factor_in_place=True
    )
    assert sol.success, (sol.iterations, sol.r_pri, sol.r_dual)
    w = sol.beta
    # ridged form: reference from the untouched copy
    g_ridged_direct = float(w @ (Pr_ref @ w))
    g_ridged_C = sol.gram_quadratic_form(w, extra_shift=0.0)
    rel_r = abs(g_ridged_C - g_ridged_direct) / abs(g_ridged_direct)
    assert rel_r < 1e-12, rel_r
    # unridged form (production chi2 convention)
    g_unridged_direct = float(w @ (P @ w))
    g_unridged_C = sol.gram_quadratic_form(w, extra_shift=lam * s)
    rel_u = abs(g_unridged_C - g_unridged_direct) / max(abs(g_unridged_direct), 1e-300)
    assert rel_u < 1e-11, rel_u


def test_chi2_from_normalized_arrays_matches_residual():
    """b_max^2 (x'Px + 2x'q) + ||bn||^2 == ||An w - bn||^2, where everything
    lives in the ECON-DIVIDED space the accumulator is fed (production adds
    the row-0 term and the mass-block split on top; both are O(p))."""
    from dynamite.weight_solvers import NormalEquationAccumulator

    rng = np.random.default_rng(3)
    n_rows, p = 900, 250
    An = rng.random((n_rows, p))  # non-negative, like real blocks
    bn = np.abs(rng.standard_normal(n_rows)) * 10.0

    acc = NormalEquationAccumulator(p)
    acc.add(An[:500], bn[:500])
    acc.add(An[500:], bn[500:])
    P, q, col, b_max = acc.finalize()

    w_true = np.zeros(p)
    idx = rng.choice(p, size=40, replace=False)
    w_true[idx] = rng.random(40) + 0.05
    x = w_true * col / b_max  # normalized-space solution
    # un-normalization exactly as production does it (ridge_sweep.py: w = x*b_max/col)
    w = x * b_max / col

    resid = An @ w - bn
    chi2_ref = float(resid @ resid)
    chi2_gram = b_max**2 * (float(x @ (P @ x)) + 2.0 * float(x @ q)) + float(bn @ bn)
    rel = abs(chi2_gram - chi2_ref) / chi2_ref
    assert rel < 1e-10, rel


def test_dense_ctw_reads_garbage_triangle():
    """The trap itself: with clean=0 the unused triangle is garbage, so a
    DENSE C.T@w disagrees with the triangular matvec at O(1). Guards against
    'simplifying' dtrmv back into a dense product."""
    rng = np.random.default_rng(11)
    p = 200
    B = rng.standard_normal((p, p))
    M = np.asfortranarray(B @ B.T + p * np.eye(p))
    C, info = dpotrf(M, lower=1, clean=0, overwrite_a=1)
    assert info == 0
    from scipy.linalg.blas import dtrmv

    w = rng.standard_normal(p)
    tri = dtrmv(C, w, lower=1, trans=1)
    dense = C.T @ w
    assert not np.allclose(tri, dense, rtol=1e-8), (
        "clean=0 upper triangle looks clean here; this guard is stale"
    )


def test_apply_diagonal_ridge_and_production_bookkeeping():
    """_apply_diagonal_ridge + factor_in_place + extra_shift, wired exactly
    as the NNLS.solve admm/free_p path wires them: ridge goes in before the
    solve, its ABSOLUTE shift comes back out as extra_shift, and the
    recovered quadratic form is the UNREGULARISED one."""
    from dynamite.weight_solvers import _apply_diagonal_ridge

    P, q, a, m = _make_problem(p=300)
    lam = 10.0
    # no-op at lam=0: identity, zero shift
    P0 = P.copy()
    assert _apply_diagonal_ridge(P0, 0.0) == 0.0
    assert np.array_equal(P0, P)
    # exact arithmetic at lam>0
    Pr = P.copy(order="F")
    shift = _apply_diagonal_ridge(Pr, lam)
    expected_scale = lam * (float(np.trace(P)) / P.shape[0])  # same order as impl
    assert shift == expected_scale
    assert np.allclose(
        np.diag(Pr), np.diag(P) + expected_scale, rtol=0, atol=1e-12 * expected_scale
    )
    # production sequence: ridged P -> in-place solve -> unridged form back
    sol = AdmmNonNegSolver(
        P=Pr, q=q, eq_coeff=a, eq_rhs=m, max_iters=20000, factor_in_place=True
    )
    assert sol.success
    w = sol.beta
    g_direct = float(w @ (P @ w))
    g_C = sol.gram_quadratic_form(w, extra_shift=shift)
    rel = abs(g_C - g_direct) / max(abs(g_direct), 1e-300)
    assert rel < 1e-11, rel


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")

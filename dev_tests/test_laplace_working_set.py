"""Option D1 of the Laplace speedup: the last re-expansion pass on a working set of orbits.

The pass is built and solved on an orbit subset S (``pm_laplace_working_set``: fraction of
the orbits; S = the previous pass's support plus the zero orbits nearest activation). The
final chi2 read also returns the gradient of THAT pass's objective for every orbit (cell
space), so the KKT conditions certify ``w_S`` padded with zeros as the full solution, with
the gap bound ``2 M max_j (-gamma_j)_+``; violators are added and the pass repeated.

Run from the repo root:
    PYTHONPATH=.:dev_tests pytest dev_tests/test_laplace_working_set.py
"""
import numpy as np
import pytest

from dynamite.weight_solvers import NormalEquationAccumulator

from test_laplace_pm import _toy_lap


def test_accumulator_on_a_column_subset_is_the_full_problem_restricted():
    """G and P are the full ones restricted to S; q, col over all orbits from the full diagonal
    (with a carried diagonal and a null orbit); p_diag_mean is the full P's mean diagonal."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(40, 9))
    A[:, 4] = 0.0  # null orbit
    b = rng.normal(size=40)
    carry = rng.random(9)
    carry[4] = 0.0
    S = np.array([0, 2, 3, 7])
    full = NormalEquationAccumulator(9)
    sub = NormalEquationAccumulator(9, cols=S)
    for acc in (full, sub):
        acc.add(np.asfortranarray(A[:25]), b[:25])
        acc.add(np.asfortranarray(A[25:]), b[25:])
        acc.add_diag_carry(carry)
    Pf, qf, cf, bf = full.finalize()
    Ps, qs, cs, bs = sub.finalize()
    np.testing.assert_allclose(Ps, Pf[np.ix_(S, S)], rtol=1e-13)
    np.testing.assert_allclose(qs, qf, rtol=1e-13)
    np.testing.assert_allclose(cs, cf, rtol=1e-13)
    assert bs == bf
    assert sub.p_diag_mean == pytest.approx(np.trace(Pf) / 9, rel=1e-13)


@pytest.mark.parametrize("kappa", [None, 10.0])
def test_final_read_returns_the_last_pass_gradient(tmp_path, kappa):
    """With ``_lap_grad`` the chi2 read's g is A'(A w - b) + c of the problem about the pass's h0
    (here: the same read with rows about that h0), measured against the Laplace part's size."""
    fake, lib, pm, w_true = _toy_lap(tmp_path, 1)
    if kappa is not None:
        fake.settings["pm_laplace_rowspace_kappa"] = kappa
    lib.vel_histograms = list(lib.full)
    w1 = w_true * np.random.default_rng(3).uniform(0.5, 1.5, w_true.size)
    fake._lap_reexpand = dict(w=w1, floor=1e-3, h0={})
    gp2 = fake.construct_gram_and_rhs_blockwise(lib)  # the pass's Gram (row-space rows if kappa)
    h0 = fake._lap_reexpand["h0"]
    fake._lap_reexpand = None
    w = w_true * 1.05
    want = gp2.G @ w - gp2.v

    def read(lap_grad):
        fake._lap_grad, fake._vp_grad_w = lap_grad, w
        fake.construct_gram_and_rhs_blockwise(lib, rhs_only=True, chi2_w=w)
        g = fake._vp_acc.g
        fake._lap_grad = fake._vp_grad_w = fake._vp_acc = None
        return g
    sep = np.abs(read(None) - want).max()  # the data-h0 gradient: what a wrong h0 gives
    assert sep > 0
    np.testing.assert_allclose(read(dict(h0=h0)), want, rtol=0, atol=1e-6 * sep)


def _solve(tmp_path, frac, lam=0.03, kappa=None):
    fake, lib, pm, _ = _toy_lap(tmp_path, 1)
    fake.ridge_lambda = lam
    if frac is not None:
        fake.settings["pm_laplace_working_set"] = frac
    if kappa is not None:
        fake.settings["pm_laplace_rowspace_kappa"] = kappa
    lib.vel_histograms = list(lib.full)
    w, _, _, chi2 = fake._solve_laplace(lib)
    return w, {c[0]: c[1] for c in chi2}, fake.laplace_info


@pytest.mark.parametrize("kappa", [None, 10.0])
def test_working_set_reproduces_the_full_solve(tmp_path, kappa):
    w_f, c_f, _ = _solve(tmp_path, None, kappa=kappa)
    w_s, c_s, info = _solve(tmp_path, 0.99, kappa=kappa)
    ws = info["working_set"]
    assert ws[-1]["gap"] <= 0.01
    np.testing.assert_allclose(w_s, w_f, rtol=1e-5, atol=1e-6 * w_f.max())
    for k in c_f:
        assert c_s[k] == pytest.approx(c_f[k], rel=1e-6, abs=1e-6)


def test_too_small_working_set_finds_violators_and_converges(tmp_path):
    """Start from a 2-orbit support (the toy's own pass 1 uses every orbit): the KKT check must
    find the missing orbits, add them and land on the full pass-2 solution."""
    w_f, c_f, info_f = _solve(tmp_path, None)
    w1 = info_f["w_pass"][0]
    fake, lib, pm, _ = _toy_lap(tmp_path, 1)
    fake.ridge_lambda = 0.03
    lib.vel_histograms = list(lib.full)
    fake._lap_reexpand = dict(w=w1, floor=1e-3, h0={})  # the same expansion point as the full run
    w_prev = np.where(w1 >= np.sort(w1)[-2], w1, 0.0)
    w, chi2, h0, rounds = fake._lap_working_set(lib, w_prev, None, 0.01, True)
    assert rounds[0]["m"] == 2 and rounds[0]["n_viol"] > 0 and len(rounds) > 1
    assert rounds[-1]["gap"] <= 0.01
    np.testing.assert_allclose(w, w_f, rtol=1e-5, atol=1e-6 * w_f.max())
    assert {c[0]: c[1] for c in chi2}["hst"] == pytest.approx(c_f["hst"], rel=1e-6, abs=1e-6)

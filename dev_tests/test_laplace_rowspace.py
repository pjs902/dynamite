"""Row-space Laplace rows (``pm_laplace_rowspace_kappa``), the dsyrk-row reduction.

F_a = M' diag(1/p0^2) M lives in the star row space V_a (any expansion point), so
H_a (F_a + delta I) H_a' = H_a V S V' H_a' + delta H_a (I - V V') H_a',  S = V'F V + delta I.
With the setting on, a Laplace ProperMotions set feeds only the rows of
R_a = sqrt(mu_k) U_k' V'  (eigenpairs of S with mu >= kappa delta), the dropped
diagonal is carried into the column scaling, the chi2 read reports the FULL statistic
and the bound E(w) = sum_a |Lt d|^2 - |R d|^2 >= 0 (d = H_a'w - h0) in ``laplace_bound``.

Run from the repo root:
    PYTHONPATH=. pytest dev_tests/test_laplace_rowspace.py
"""
import numpy as np
import pytest

from dynamite import fitstat_pm as fs
from dynamite.weight_solvers import NNLS, NormalEquationAccumulator

from test_laplace_pm import MEANS, _lap_at_data_h0, _toy_lap, make_pm_lap
from test_relative_pm import N_ORBS, _fake_nnls, _orblib


def _build(tmp_path, kappa, **kw):
    pm = make_pm_lap(tmp_path, MEANS, n_stars=(30, 80, 20), statistic="laplace")
    fake = _fake_nnls([__import__("test_relative_pm").FakeLOS(), pm], 2 + pm.n_spatial_bins)
    fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
    if kappa is not None:
        fake.settings = {"pm_laplace_rowspace_kappa": kappa}
    lib = _orblib(pm, rebin=True)
    lib.vel_histograms = list(lib.full)
    return fake, lib, pm, fake.construct_gram_and_rhs_blockwise(lib, **kw)


def _parts(pm, lib, prj, a, kappa):
    """(H_a, Lt, g, h0, R, delta, V) by hand for aperture a."""
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(N_ORBS, 3, -1)[:, a]
    Lt, g, h0 = pm.laplace_terms(a, prj[a])
    M = pm.laplace_design(a)
    _, sv, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt[sv > 1e-8 * sv[0]].T
    F, _ = fs.laplace_terms(M, h0, len(M), prj[a])
    delta = fs.default_delta(len(M), prj[a])
    mu, U = np.linalg.eigh(V.T @ F @ V + delta * np.eye(V.shape[1]))
    k = mu >= kappa * delta
    R = (np.sqrt(mu[k])[:, None] * U[:, k].T) @ V.T
    return H, Lt, g, h0, R, delta, V


def test_rowspace_is_rank_deficient_in_the_toy():
    # the identities below are only informative if some aperture has fewer stars than cells
    from test_relative_pm import NB
    assert 20 < NB * NB


@pytest.mark.parametrize("kappa", [0.0, 10.0])
def test_block_rows_carry_and_linear_term(tmp_path, kappa):
    """Per aperture: rows' Gram H R'R H', data cross term H R'R h0 (sign-free checks), zero
    padding after r_a rows, carried diagonal diag(H (Q - R'R) H'), linear term unchanged."""
    pm = make_pm_lap(tmp_path, MEANS, n_stars=(30, 80, 20), statistic="laplace")
    lib = _orblib(pm, rebin=True)
    fake = _fake_nnls([__import__("test_relative_pm").FakeLOS(), pm], 5)
    fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
    fake.settings = {"pm_laplace_rowspace_kappa": kappa}
    rec = {"lin": [], "carry": []}
    fake._lap_acc = type("A", (), {"chi2_w": None,
                                   "add_linear": lambda self, c: rec["lin"].append(np.array(c)),
                                   "add_diag_carry": lambda self, c: rec["carry"].append(np.array(c))})()
    obs, err = pm.get_observed_values_and_uncertainties({})
    prj = fake.projected_masses[2:]
    o, e, orb = fake._prepare_kinematic_block(pm, lib.full[1], (obs, err), prj)
    o, orb = o.reshape(3, -1), orb.reshape(N_ORBS, 3, -1)
    carry, lin = np.zeros(N_ORBS), np.zeros(N_ORBS)
    for a in range(3):
        H, Lt, g, h0, R, delta, V = _parts(pm, lib, prj, a, kappa)
        r = R.shape[0]
        assert r < H.shape[1]
        rows, dat = orb[:, a, :r], o[a, :r]
        assert not orb[:, a, r:].any() and not o[a, r:].any()
        np.testing.assert_allclose(rows @ rows.T, H @ R.T @ R @ H.T, rtol=1e-9,
                                   atol=1e-9 * np.abs(H @ R.T @ R @ H.T).max())
        np.testing.assert_allclose(rows @ dat, H @ R.T @ R @ h0, rtol=1e-9)
        carry += np.einsum("ij,jk,ik->i", H, Lt.T @ Lt - R.T @ R, H)
        lin += H @ g
    np.testing.assert_allclose(np.sum(rec["carry"], axis=0), carry, rtol=1e-6, atol=1e-12 * np.abs(carry).max())
    np.testing.assert_allclose(np.sum(rec["lin"], axis=0), lin, rtol=1e-10)


def test_finalize_column_scaling_includes_the_carry():
    rng = np.random.default_rng(1)
    A = rng.random((9, 4))
    acc = NormalEquationAccumulator(4)
    acc.add(A, rng.random(9))
    c = np.array([0.5, 0.0, 2.0, 1e-3])
    acc.add_diag_carry(c)
    _, _, col, _ = acc.finalize()
    np.testing.assert_allclose(col, np.sqrt(np.sum(A * A, axis=0) + c), rtol=1e-14)


def test_chi2_read_reports_the_full_statistic_and_the_bound(tmp_path):
    kappa = 10.0
    w = np.random.default_rng(4).random(N_ORBS) * 1e4
    fake, lib, pm, gp = _build(tmp_path, kappa, rhs_only=True, chi2_w=w)
    prj = fake.projected_masses[2:]
    ref = _lap_at_data_h0(pm, lib, prj, w)
    mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
    set_chi2 = gp.chi2_by_tag["set1/"] - mass_r2[fake.n_intrinsic + 2:].sum()
    assert set_chi2 == pytest.approx(ref, rel=1e-8, abs=1e-6)
    E = 0.0
    for a in range(3):
        H, Lt, g, h0, R, delta, V = _parts(pm, lib, prj, a, kappa)
        d = H.T @ w - h0
        E += np.sum((Lt @ d) ** 2) - np.sum((R @ d) ** 2)
    assert E > 0
    assert fake.laplace_bound["hst"] == pytest.approx(E, rel=1e-8)


def test_accumulator_drops_all_zero_rows(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.random((7, 5))
    A[[1, 4]] = 0.0
    b = rng.random(7)
    b[[1, 4]] = 0.0
    ref = NormalEquationAccumulator(5)
    ref.add(A, b)
    acc = NormalEquationAccumulator(5)
    acc.drop_zero_rows = True
    acc.add(np.asfortranarray(A), b)
    np.testing.assert_allclose(np.triu(acc.G), np.triu(ref.G), rtol=1e-14)
    np.testing.assert_allclose(acc.v, ref.v, rtol=1e-14)
    assert acc.b_sq_sum == pytest.approx(ref.b_sq_sum) and acc.rows_used == 5


def test_driver_solution_is_within_the_reported_bound(tmp_path):
    """Q(w_rs) - Q(w_full) <= E(w_rs) for the statistic about the data h0 (pass 1 only)."""
    out = {}
    for kappa in (None, 10.0):
        fake, lib, pm, _ = _toy_lap(tmp_path, reexpand=0)
        if kappa is not None:
            fake.settings["pm_laplace_rowspace_kappa"] = kappa
        lib.vel_histograms = list(lib.full)
        w, *_ , chi2 = fake._solve_laplace(lib)
        out[kappa] = (w, {c[0]: c[1] for c in chi2}, getattr(fake, "laplace_bound", {}))
    q_full, q_rs = out[None][1]["hst"], out[10.0][1]["hst"]
    E = out[10.0][2]["hst"]
    assert E >= 0
    assert q_rs - q_full <= E + 1e-6 * abs(q_full) + 1e-6


def test_dropped_rows_keep_their_data_term():
    """A row with zero orbit part but b != 0 adds nothing to G or v but still to |b|^2, b_max and the
    chi2 residual (cells no orbit reaches); dropping it must keep those."""
    rng = np.random.default_rng(2)
    A = rng.random((8, 5))
    A[[0, 3, 6]] = 0.0
    b = rng.random(8) * 3
    w = rng.random(5)
    ref = NormalEquationAccumulator(5, chi2_w=w)
    ref.add(A, b)
    acc = NormalEquationAccumulator(5, chi2_w=w)
    acc.drop_zero_rows = True
    acc.add(np.asfortranarray(A), b)
    np.testing.assert_allclose(np.triu(acc.G), np.triu(ref.G), rtol=1e-14)
    np.testing.assert_allclose(acc.v, ref.v, rtol=1e-14)
    assert acc.b_sq_sum == pytest.approx(ref.b_sq_sum, rel=1e-14)
    assert acc.b_max == ref.b_max
    assert acc.chi2[""] == pytest.approx(ref.chi2[""], rel=1e-14)
    assert acc.rows_used == 5
    allzero = NormalEquationAccumulator(5, chi2_w=w)
    allzero.drop_zero_rows = True
    allzero.add(np.zeros((3, 5)), np.array([1.0, 2.0, 0.0]))
    assert allzero.b_sq_sum == pytest.approx(5.0) and allzero.chi2[""] == pytest.approx(5.0)


@pytest.mark.parametrize("order", ["F", "C"])
def test_dropped_rows_with_a_working_set_and_vp_terms(order):
    """Zero rows dropped and a column subset together (one selection copy for the dsyrk), with the
    chi2, gradient and line terms on: everything equals the plain accumulator restricted to S."""
    rng = np.random.default_rng(4)
    A = rng.random((9, 6))
    A[[1, 2, 7]] = 0.0
    b = rng.random(9)
    w, dw = rng.random(6), rng.random(6)
    S = np.array([0, 2, 5])
    out = []
    for drop, cols in ((False, None), (True, S)):
        acc = NormalEquationAccumulator(6, chi2_w=w, cols=cols)
        acc.drop_zero_rows = drop
        acc.grad_w, acc.line = w, (w, dw)
        acc.add(np.asarray(A, order=order), b)
        out.append(acc)
    ref, acc = out
    np.testing.assert_allclose(np.triu(acc.G), np.triu(ref.G[np.ix_(S, S)]), rtol=1e-14)
    np.testing.assert_allclose(acc.diag_full, np.diag(ref.G), rtol=1e-14)
    for k in ("v", "g", "line_q"):
        np.testing.assert_allclose(getattr(acc, k), getattr(ref, k), rtol=1e-14)
    assert acc.chi2[""] == pytest.approx(ref.chi2[""], rel=1e-14)
    assert acc.b_sq_sum == pytest.approx(ref.b_sq_sum, rel=1e-14) and acc.b_max == ref.b_max
    assert acc.rows_used == 6

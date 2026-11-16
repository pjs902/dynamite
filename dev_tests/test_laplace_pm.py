"""Laplace (per-star likelihood) statistic for veldist ProperMotions sets.

A ``ProperMotions`` set with ``statistic: laplace`` is scored with the
quadratic expansion of the exact per-star likelihood about ``h0`` instead
of the diagonal chi2 (design: veldist ``docs/fit-statistic-design.md``).
The quadratic part enters as rows ``Lt H`` against ``Lt h0`` (unit errors),
the linear part ``2 g'h`` straight into the accumulator's cross term, so
no ~1e8-per-aperture constant ever reaches the float32 Gram path.

Run from the repo root:
    PYTHONPATH=. pytest dev_tests/test_laplace_pm.py
"""

import numpy as np
import pytest
from scipy.special import ndtr

from dynamite import fitstat_pm as fs
from dynamite import kinematics as dyn_kin
from dynamite.weight_solvers import NNLS

from test_relative_pm import (CEN, EDG, N_ORBS, NB, VR, FakeLOS, _fake_nnls,
                              _orblib, gauss_cells)


def star_rows(n, seed, err=0.6, s=(2.0, 1.6), m=(0.0, 0.0)):
    """Closed-grid design rows (n, NB*NB) of stars drawn from a Gaussian."""
    rng = np.random.default_rng(seed)
    v = rng.normal(m, s, (n, 2)) + rng.normal(0, err, (n, 2))
    px = np.diff(ndtr((EDG[None] - v[:, :1]) / err), axis=1)
    py = np.diff(ndtr((EDG[None] - v[:, 1:]) / err), axis=1)
    M = (px[:, :, None] * py[:, None, :]).reshape(n, -1)
    return M[M.sum(1) >= 1e-3]


def make_pm_lap(tmp_path, means, n_stars=(60, 80, 40), name="hst", seed=0,
                drop=(), **kw):
    hist = np.stack([gauss_cells(*mn) for mn in means])
    sigma = 0.01 + 0.05 * hist
    ptr, cell, mass, ap, h0 = [0], [], [], [], []
    for a, (mn, n) in enumerate(zip(means, n_stars)):
        M = star_rows(n, seed + a, m=mn)
        for row in M:
            nz = np.nonzero(row >= 1e-6 * row.max())[0]
            cell.append(nz.astype(np.int32))
            mass.append(row[nz].astype(np.float32))
            ptr.append(ptr[-1] + len(nz))
        ap.append(np.full(len(M), a, np.int32))
        mix = 0.99 * gauss_cells(*mn, sx=2.1, sy=1.7) + 0.01 / NB**2
        h0.append(mix / mix.sum())
    arrays = dict(PM_2dhist=hist, PM_2dhist_sigma=sigma, vxrange=VR, vyrange=VR,
                  PM_star_indptr=np.asarray(ptr, np.int64),
                  PM_star_cell=np.concatenate(cell),
                  PM_star_mass=np.concatenate(mass),
                  PM_star_aperture=np.concatenate(ap),
                  PM_lap_h0=np.asarray(h0), PM_star_sharpened=np.asarray(True))
    for k in drop:
        arrays.pop(k)
    np.savez(tmp_path / f"{name}.npz", **arrays)
    return dyn_kin.ProperMotions(name=name, datafile=f"{name}.npz",
                                 input_directory=str(tmp_path) + "/",
                                 type="ProperMotions", **kw)


MEANS = [(0, 0), (0.5, -0.5), (-1, 1)]


# ------------------------------------------------------------ fitstat_pm
def test_offset_free_stat_equals_veldist_rows_minus_their_value_at_h0():
    M = star_rows(300, 1)
    n, mass = len(M), 2.5e5
    h0 = mass * (0.99 * gauss_cells(0, 0, 2.1, 1.7).ravel() + 0.01 / NB**2)
    F, g = fs.laplace_terms(M, h0, n, mass)
    delta = fs.default_delta(n, mass)
    Lt = fs.laplace_lt(F, delta)
    # veldist.fitstat.laplace_rows: one row block with the linear term folded in
    tg = Lt @ (h0 - np.linalg.solve(F + delta * np.eye(len(g)), g))
    h = mass * gauss_cells(0.3, -0.2, 2.0, 1.6).ravel()
    ref = np.sum((Lt @ h - tg) ** 2) - np.sum((Lt @ h0 - tg) ** 2)
    assert np.sum((Lt @ h0 - tg) ** 2) > 1e4 * abs(ref)  # the constant DYNAMITE must not see
    assert fs.stat_offset_free(h, Lt, g, h0) == pytest.approx(ref, rel=1e-6)


def test_linear_term_vanishes_at_a_mass_normalised_h0():
    M = star_rows(200, 2)
    mass = 3e4
    h0 = 0.99 * gauss_cells(0, 0).ravel() + 0.01 / NB**2
    h0 *= mass / h0.sum()
    _, g = fs.laplace_terms(M, h0, len(M), mass)
    assert abs(g @ h0) < 1e-9 * len(M)


def test_quadratic_is_the_second_order_expansion_of_exact():
    # error vs exact shrinks ~8x per halving of the step: third order
    M = star_rows(2000, 3)
    n, mass = len(M), 1.0
    h0 = 0.99 * gauss_cells(0, 0).ravel() + 0.01 / NB**2
    d = gauss_cells(0.2, 0.0, 2.2, 1.6).ravel() - gauss_cells(0, 0).ravel()
    F, g = fs.laplace_terms(M, h0, n, mass)
    Lt = fs.laplace_lt(F, 1e-9 * n)  # no ridge: test the expansion itself
    err = []
    for t in (0.2, 0.1):
        h = h0 + t * d
        ex = fs.stat_exact(h, M, n, mass) - fs.stat_exact(h0, M, n, mass)
        err.append(abs(fs.stat_offset_free(h, Lt, g, h0) - ex))
    assert 6 < err[0] / err[1] < 10, err


# ------------------------------------------------------------ ProperMotions
def test_statistic_defaults_to_diag(tmp_path):
    pm = make_pm_lap(tmp_path, MEANS)
    assert pm.statistic == "diag"


@pytest.mark.parametrize("kw,drop,match", [
    (dict(statistic="chi"), (), "statistic"),
    (dict(statistic="laplace"), ("PM_star_cell",), "PM_star_cell"),
    (dict(statistic="laplace", relative=True), (), "relative"),
])
def test_statistic_validation(tmp_path, kw, drop, match):
    with pytest.raises(ValueError, match=match):
        make_pm_lap(tmp_path, MEANS, drop=drop, **kw)


def test_laplace_terms_per_aperture(tmp_path):
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace")
    a, mass = 1, 2e5
    Lt, g, h0 = pm.laplace_terms(a, mass)
    np.testing.assert_allclose(h0.sum(), mass)
    M = star_rows(80, 1, m=MEANS[1])
    F, gr = fs.laplace_terms(M, h0, len(M), mass)
    np.testing.assert_allclose(g, gr, rtol=1e-4, atol=1e-4 * np.abs(gr).max())  # float32 + sparse-cut rows
    np.testing.assert_allclose(Lt.T @ Lt, F + fs.default_delta(len(M), mass) * np.eye(NB**2),
                               rtol=1e-4, atol=1e-6 * np.abs(F).max())
    h1 = 1.1 * h0  # an override h0 (re-expansion) is used as given
    assert pm.laplace_terms(a, mass, h0=h1)[2] is h1


# ------------------------------------------------------------ the block
def test_laplace_block_rows_unit_errors_and_linear_term(tmp_path):
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace")
    lib = _orblib(pm, rebin=True)
    fake = _fake_nnls([FakeLOS(), pm], 5)
    fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
    lin = []
    fake._lap_acc = type("A", (), {"add_linear": lambda self, c: lin.append(np.array(c))})()
    obs, err = pm.get_observed_values_and_uncertainties({})
    prj = fake.projected_masses[2:]
    o, e, orb = fake._prepare_kinematic_block(pm, lib.full[1], (obs, err), prj)
    assert np.array_equal(e, np.ones(obs.size))
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(N_ORBS, 3, -1)
    c_ref = np.zeros(N_ORBS)
    for a in range(3):
        Lt, g, h0 = pm.laplace_terms(a, prj[a])
        np.testing.assert_allclose(o.reshape(3, -1)[a], Lt @ h0, rtol=1e-10)
        np.testing.assert_allclose(orb.reshape(N_ORBS, 3, -1)[:, a], H[:, a] @ Lt.T, rtol=1e-10)
        c_ref += H[:, a] @ g
    np.testing.assert_allclose(np.sum(lin, axis=0), c_ref, rtol=1e-10)


def test_laplace_needs_the_blockwise_gram_path(tmp_path):
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace")
    fake = _fake_nnls([FakeLOS(), pm], 5)
    fake.lib = _orblib(pm, rebin=True)
    fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
    with pytest.raises(ValueError, match="gram_blockwise"):
        fake.construct_nnls_matrix_and_rhs(fake.lib)


@pytest.mark.parametrize("stream,gap", [(False, 0), (False, 2), (True, 2)])
def test_gram_quadratic_form_is_the_offset_free_statistic(tmp_path, stream, gap):
    """w'Gw - 2w'v + |b|^2 over the Laplace set == sum_a stat_offset_free(H_a w)
    (the constant gone), on every blockwise read path; and the rhs-only
    chi2_w read gives the same per set."""
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace")
    n_ap_total = 2 + pm.n_spatial_bins
    w = np.random.default_rng(4).random(N_ORBS) * 1e4

    def build(**kw):
        fake = _fake_nnls([FakeLOS(), pm], n_ap_total, stream, gap)
        fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
        lib = _orblib(pm, rebin=True)
        lib.vel_histograms = list(lib.full)
        return fake, lib, fake.construct_gram_and_rhs_blockwise(lib, **kw)

    fake, lib, gp = build()
    # the same build with the PM set's rows only: drop the LOS set and mass rows
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(N_ORBS, 3, -1)
    prj = fake.projected_masses[2:]
    ref = sum(fs.stat_offset_free(H[:, a].T @ w, *pm.laplace_terms(a, prj[a]))
              for a in range(3))
    _, _, gp_c = build(rhs_only=True, chi2_w=w)
    mass_r2 = (gp_c.A_mass @ w - gp_c.b_mass) ** 2
    set_chi2 = gp_c.chi2_by_tag["set1/"] - mass_r2[fake.n_intrinsic + 2:].sum()
    assert set_chi2 == pytest.approx(ref, rel=1e-8, abs=1e-6)
    # total quadratic form = set0 (diag) + set1 (laplace) + mass rows
    total = w @ gp.G @ w - 2 * w @ gp.v + gp.b_sq_rest
    other = sum(v for k, v in gp_c.chi2_by_tag.items() if k != "set1/") \
        + mass_r2[fake.n_intrinsic + 2:].sum()
    assert total == pytest.approx(other + ref, rel=1e-8, abs=1e-6)
    assert fake._lap_acc is None  # no stale accumulator after the build

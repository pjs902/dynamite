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
                drop=(), s=(2.0, 1.6), h0_s=(2.1, 1.7), **kw):
    hist = np.stack([gauss_cells(*mn, *s) for mn in means])
    sigma = 0.01 + 0.05 * hist
    ptr, cell, mass, ap, h0 = [0], [], [], [], []
    for a, (mn, n) in enumerate(zip(means, n_stars)):
        M = star_rows(n, seed + a, s=s, m=mn)
        for row in M:
            nz = np.nonzero(row >= 1e-6 * row.max())[0]
            cell.append(nz.astype(np.int32))
            mass.append(row[nz].astype(np.float32))
            ptr.append(ptr[-1] + len(nz))
        ap.append(np.full(len(M), a, np.int32))
        mix = 0.99 * gauss_cells(*mn, sx=h0_s[0], sy=h0_s[1]) + 0.01 / NB**2
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


# ------------------------------------------------------------ the driver
def _toy_lap(tmp_path, reexpand=1, relative=False):
    """Laplace PM set whose stars are drawn from a known w_true on the toy library."""
    from test_relative_pm import DRIVER, ConsistentLOS
    import types
    w_true = np.array([1.3, 0.4, 0.9, 1.6, 0.2, 1.1])
    probe = make_pm_lap(tmp_path, MEANS, name="probe")
    lib = _orblib(probe, rebin=False)
    y = lib.full[1].y                                    # (o, vx, vy, a)
    M_pm = w_true @ y.sum(axis=(1, 2))
    H = np.einsum("o,oxya->axy", w_true, y) / M_pm[:, None, None]
    if relative:  # the data are the model histograms recentred to mean 0
        m_true = np.stack([H.sum(2) @ CEN, H.sum(1) @ CEN], -1)
        H = dyn_kin.ProperMotions.shift_histograms(H, (EDG, EDG), m_true).clip(1e-12)
    pm = make_pm_lap(tmp_path, [(0, 0)] * 3, statistic="laplace")
    z = dict(np.load(tmp_path / "hst.npz"))
    z["PM_2dhist"] = H
    rng = np.random.default_rng(7)
    ptr, cell, mass, ap = [0], [], [], []
    for a in range(3):  # stars from the true model histogram, errors 0.6
        idx = rng.choice(NB * NB, size=150, p=(H[a] / H[a].sum()).ravel())
        v = np.c_[CEN[idx // NB], CEN[idx % NB]] + rng.uniform(-0.5, 0.5, (150, 2))
        v += rng.normal(0, 0.6, v.shape)
        px = np.diff(ndtr((EDG[None] - v[:, :1]) / 0.6), axis=1)
        py = np.diff(ndtr((EDG[None] - v[:, 1:]) / 0.6), axis=1)
        for row in (px[:, :, None] * py[:, None, :]).reshape(150, -1):
            nz = np.nonzero(row >= 1e-6 * row.max())[0]
            cell.append(nz.astype(np.int32))
            mass.append(row[nz].astype(np.float32))
            ptr.append(ptr[-1] + len(nz))
        ap.append(np.full(150, a, np.int32))
    mix = 0.99 * H + 0.01 / NB**2
    z.update(PM_star_indptr=np.asarray(ptr, np.int64), PM_star_cell=np.concatenate(cell),
             PM_star_mass=np.concatenate(mass), PM_star_aperture=np.concatenate(ap),
             PM_lap_h0=mix / mix.sum(axis=(1, 2), keepdims=True))
    np.savez(tmp_path / "hst.npz", **z)
    pm = dyn_kin.ProperMotions(name="hst", datafile="hst.npz", input_directory=str(tmp_path) + "/",
                               type="ProperMotions", statistic="laplace", relative=relative)
    M_los = w_true @ lib.projected_masses[0]
    los_obs = np.einsum("o,oja->aj", w_true, lib.full[0].y[:, :4, :]) / M_los[:, None]
    los = ConsistentLOS(los_obs, 0.01 * np.abs(los_obs) + 1e-3)
    fake = _fake_nnls([los, pm], 5, dtype=np.float64)
    fake.lib = lib
    fake.total_mass = float(w_true.sum())
    fake.intrinsic_masses = w_true @ lib.intrinsic_masses
    fake.projected_masses = np.concatenate([M_los, M_pm])
    fake.nnls_solver, fake.gram_blockwise, fake.ridge_lambda = "admm", True, 0.0
    fake.admm_rho, fake.admm_alpha, fake.admm_precision = None, 1.0, "float64"
    fake.admm_max_iters, fake.admm_tol, fake.admm_free_p = 20000, 1e-12, False
    fake.settings = {"pm_laplace_reexpand": reexpand}
    if relative:
        fake.settings.update(pm_recentre_mode="vp", pm_recentre_max_rounds=8)
        fake.direc_with_ml = str(tmp_path) + "/"
        fake._pm_dsq_over_sigsq = NNLS._pm_dsq_over_sigsq
        fake._vp_fit_centres = NNLS._vp_fit_centres
    fake._relative_chi2_total = NNLS._relative_chi2_total
    for name in DRIVER + ("_solve_laplace", "_laplace_pm_block", "_solve_relative_vp_laplace"):
        setattr(fake, name, types.MethodType(getattr(NNLS, name), fake))
    return fake, lib, pm, w_true


def _lap_at_data_h0(pm, lib, prj, w):
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(N_ORBS, 3, -1)
    return sum(fs.stat_offset_free(H[:, a].T @ w, *pm.laplace_terms(a, prj[a]))
               for a in range(3))


@pytest.mark.parametrize("reexpand", [0, 1])
def test_driver_reports_laplace_at_the_data_h0(tmp_path, reexpand):
    fake, lib, pm, _ = _toy_lap(tmp_path, reexpand)
    lib.vel_histograms = list(lib.full)
    w, chi2_tot, chi2_kin, chi2 = fake._solve_laplace(lib)
    assert np.all(w >= -1e-9) and w.sum() == pytest.approx(fake.total_mass, rel=1e-3)
    by = {c[0]: c[1] for c in chi2}
    ref = _lap_at_data_h0(pm, lib, fake.projected_masses[2:], w)
    assert by["hst"] == pytest.approx(ref, rel=1e-8, abs=1e-6)
    assert chi2_kin == pytest.approx(chi2_tot - by["mass"])
    assert fake._lap_h0 is None and fake._lap_capture is None  # driver state cleared
    assert fake.laplace_info["reexpand"] == reexpand


def test_reexpansion_point_is_the_mixed_pass1_model(tmp_path):
    fake, lib, pm, _ = _toy_lap(tmp_path, 1)
    lib.vel_histograms = list(lib.full)
    seen = []
    orig = NNLS._relative_build

    def spy(self, orblib, gram, rhs_from=None):
        seen.append(None if self._lap_h0 is None else {k: v.copy() for k, v in self._lap_h0.items()})
        return orig(self, orblib, gram, rhs_from)
    fake._relative_build = spy.__get__(fake)
    fake._solve_laplace(lib)
    assert seen[0] is None and len(seen) == 2  # pass 1 at the data h0, then one rebuild
    w1 = fake.laplace_info["w_pass"][0]
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(N_ORBS, 3, -1)
    prj = fake.projected_masses[2:]
    for a in range(3):
        h = H[:, a].T @ w1
        want = (0.999 * h / h.sum() + 0.001 / NB**2) * prj[a]
        np.testing.assert_allclose(seen[1]["hst"][a], want, rtol=1e-10)


def test_mirrored_h0_is_refused(tmp_path):
    """PM_lap_h0 and PM_2dhist come from the same veldist fit; a frame flip
    between them (pm2 -> -pm2, as the x-East convention can introduce) must
    not pass silently."""
    pm = make_pm_lap(tmp_path, [(1.5, 1.0), (1.0, -1.5), (-1.5, 1.2)])
    z = dict(np.load(tmp_path / "hst.npz"))
    z["PM_lap_h0"] = z["PM_lap_h0"][:, :, ::-1]
    np.savez(tmp_path / "hst.npz", **z)
    with pytest.raises(ValueError, match="PM_lap_h0"):
        dyn_kin.ProperMotions(name="hst", datafile="hst.npz", input_directory=str(tmp_path) + "/",
                              type="ProperMotions", statistic="laplace")


# ------------------------------------------------------------ 1D (BayesLOSVD)
LOS_EDG = np.linspace(-3, 3, 5)  # data grid: 4 bins of 1.5 (orblib: 6 of 1.0)
LOS_CEN = 0.5 * (LOS_EDG[:-1] + LOS_EDG[1:])


def make_los_lap(tmp_path, statistic="laplace", flip=False):
    from astropy.table import Table
    n_ap, nv = 2, len(LOS_CEN)
    hist = np.stack([np.diff(ndtr((LOS_EDG - m) / 1.2)) for m in (-0.3, 0.4)])
    hist /= hist.sum(1, keepdims=True)
    t = Table({"binID_BayesLOSVD": np.arange(n_ap), "binID_dynamite": np.arange(1, n_ap + 1),
               "xbin": [0.0, 1.0], "ybin": [0.0, 0.0], "bin_flux": [1.0, 1.0]})
    for j in range(nv):
        t[f"losvd_{j}"] = hist[:, j]
        t[f"dlosvd_{j}"] = 0.01 + 0.05 * hist[:, j]
    t.meta.update(nbins=n_ap, nvbins=nv, vcent=list(LOS_CEN), dv=1.5,
                  PSF={"sigma": [0.0], "weight": [1.0]})
    t.write(tmp_path / "los.ecsv", format="ascii.ecsv", overwrite=True)
    rng = np.random.default_rng(3)
    ptr, cell, mass, ap = [0], [], [], []
    for a, m in enumerate((-0.3, 0.4)):
        v = rng.normal(m, 1.2, 120)
        rows = np.diff(ndtr((LOS_EDG[None] - v[:, None]) / 0.5), axis=1)
        rows = rows[rows.sum(1) >= 1e-3]
        for row in rows:
            nz = np.nonzero(row >= 1e-6 * row.max())[0]
            cell.append(nz.astype(np.int32))
            mass.append(row[nz].astype(np.float32))
            ptr.append(ptr[-1] + len(nz))
        ap.append(np.full(len(rows), a, np.int32))
    h0 = 0.99 * hist + 0.01 / nv
    h0 = h0 / h0.sum(1, keepdims=True)
    np.savez(tmp_path / "los_stars.npz", LOS_star_indptr=np.asarray(ptr, np.int64),
             LOS_star_cell=np.concatenate(cell), LOS_star_mass=np.concatenate(mass),
             LOS_star_aperture=np.concatenate(ap), LOS_lap_h0=h0[:, ::-1] if flip else h0,
             LOS_star_sharpened=np.asarray(True))
    return dyn_kin.BayesLOSVD(name="los", datafile="los.ecsv", input_directory=str(tmp_path) + "/",
                              type="BayesLOSVD", statistic=statistic)


def test_bayeslosvd_laplace_loads_and_guards(tmp_path):
    los = make_los_lap(tmp_path)
    assert los.statistic == "laplace" and len(los._lap_rows) == 2
    with pytest.raises(ValueError, match="LOS_lap_h0"):
        make_los_lap(tmp_path, flip=True)


@pytest.mark.parametrize("stream,gap", [(False, 0), (True, 2)])
def test_gram_quadratic_form_1d(tmp_path, stream, gap):
    """A 1D Laplace set (set 0, whole-set read) next to a diag PM set."""
    los = make_los_lap(tmp_path)
    pm = make_pm_lap(tmp_path, MEANS)  # diag
    n_ap_total = 2 + pm.n_spatial_bins
    w = np.random.default_rng(5).random(N_ORBS) * 1e4

    def build(**kw):
        fake = _fake_nnls([los, pm], n_ap_total, stream, gap)
        fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
        lib = _orblib(pm, rebin=True)
        lib.vel_histograms = list(lib.full)
        return fake, lib, fake.construct_gram_and_rhs_blockwise(lib, **kw)

    fake, lib, gp = build()
    h1 = dyn_kin.Histogram(xedg=lib.full[0].xedg, y=lib.full[0].y.copy())
    h1.y[:, 0, :] = h1.y[:, -1, :] = 0.0  # 1D: DYNAMITE drops the orblib's overflow bins
    H = los.transform_orblib_to_observables(h1, {}).reshape(N_ORBS, 2, -1)
    prj = fake.projected_masses[:2]
    ref = sum(fs.stat_offset_free(H[:, a].T @ w, *los.laplace_terms(a, prj[a])) for a in range(2))
    _, _, gp_c = build(rhs_only=True, chi2_w=w)
    mass_r2 = (gp_c.A_mass @ w - gp_c.b_mass) ** 2
    lap = gp_c.chi2_by_tag["set0/"] - mass_r2[fake.n_intrinsic:fake.n_intrinsic + 2].sum()
    assert lap == pytest.approx(ref, rel=1e-8, abs=1e-6)
    total = w @ gp.G @ w - 2 * w @ gp.v + gp.b_sq_rest
    assert total == pytest.approx(sum(gp_c.chi2_by_tag.values()), rel=1e-8, abs=1e-6)


def test_reexpand_mix_floor_default():
    """Re-expansion reference mixes the model histogram with a uniform floor of 1e-3 (C1d) by default."""
    from dynamite.weight_solvers import LAPLACE_REEXPAND_FLOOR, laplace_reexpand_mix
    assert LAPLACE_REEXPAND_FLOOR == 1e-3
    h = np.array([[0.0, 2.0, 2.0, 0.0]])
    mix = laplace_reexpand_mix(h)
    np.testing.assert_allclose(mix.sum(axis=1), 1.0)
    np.testing.assert_allclose(mix[0, 0], 1e-3 / 4)
    np.testing.assert_allclose(laplace_reexpand_mix(h, floor=0.01)[0, 0], 0.01 / 4)


# ------------------------------------------------- relative mode (vp), Laplace
from dynamite import fitstat_pm  # noqa: E402,F401  (project_shift lives there)

LIB_S = (1.2, 1.1)  # narrow library: a one-cell shift loses nothing off the grid


def _lap_library(n_ap=3):
    """(n_orbs, n_ap, NB, NB) orbit histograms, interior Gaussians."""
    y = np.empty((N_ORBS, n_ap, NB, NB))
    for o in range(N_ORBS):
        for a in range(n_ap):
            y[o, a] = (1 + o + a) * 1e3 * gauss_cells(0.4 * o - 1.0, 0.3 * a - 0.5, *LIB_S)
    return y


def _rel_block(pm, orb, centre, w_proj, prj=(1e5, 1.5e5, 2e5), relative_block=True):
    """Run the vp-mode relative block on a fake NNLS; returns (obs, rows, lin)."""
    fake = _fake_nnls([FakeLOS(), pm], 5)
    lin = []
    fake._lap_acc = type("A", (), {"add_linear": lambda self, c: lin.append(np.array(c))})()
    fake._laplace_pm_block = NNLS._laplace_pm_block.__get__(fake)
    fake._pm_vp = dict(centre={pm.name: np.asarray(centre, float)}, capture={},
                       H={}, D={}, E={}, w_proj=w_proj)
    obs, err = pm.get_observed_values_and_uncertainties({})
    prj = np.asarray(prj)
    if relative_block:
        o, _, rows = fake._relative_pm_block(pm, obs, err, orb, 0, prj)
    else:
        o, _, rows = fake._laplace_pm_block(pm, np.reshape(obs, (3, NB, NB)), orb, prj, 0)
    return (np.reshape(o, (3, -1)), np.reshape(rows, (N_ORBS, 3, -1)),
            np.sum(lin, axis=0))


def _stat(obs, rows, lin, w):
    r = np.einsum("o,oak->ak", w, rows) - obs
    return float((r * r).sum() + 2 * lin @ w)


def test_relative_laplace_set_is_accepted(tmp_path):
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace", relative=True)
    assert pm.relative and pm.statistic == "laplace"


def test_relative_laplace_statistic_invariant_to_a_whole_cell_data_shift(tmp_path):
    """Shift every star, h0 and the model-shift centre by one cell: same statistic."""
    kw = dict(statistic="laplace", relative=True, s=(1.0, 0.9), h0_s=(1.1, 1.0))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pm_a = make_pm_lap(tmp_path / "a", [(0, 0), (0.5, -0.5), (-1, 1)], **kw)
    pm_b = make_pm_lap(tmp_path / "b", [(1, 0), (1.5, -0.5), (0, 1)], **kw)  # +1 cell in x
    orb = _lap_library()
    w = np.random.default_rng(2).random(N_ORBS) * 3
    out = [_rel_block(pm, orb, np.tile(c, (3, 1)), w)
           for pm, c in ((pm_a, (0.0, 0.0)), (pm_b, (-1.0, 0.0)))]  # S shifts the model by -c
    sa, sb = (_stat(*o, w) for o in out)
    assert sa == pytest.approx(sb, rel=2e-3, abs=1e-6 * abs(sa))
    assert abs(sa) > 1.0  # not vacuous


def test_zero_shift_basis_reduces_to_the_addback_laplace_path(tmp_path):
    pm_r = make_pm_lap(tmp_path, MEANS, statistic="laplace", relative=True)
    pm_d = make_pm_lap(tmp_path, MEANS, statistic="laplace", name="hst_d")
    orb = _lap_library()
    zero = np.zeros((3, 2))
    ref = _rel_block(pm_d, orb, zero, None, relative_block=False)
    got = _rel_block(pm_r, orb, zero, np.zeros(N_ORBS))  # model derivative = 0: D = 0
    for g, r in zip(got, ref):
        np.testing.assert_allclose(g, r, rtol=1e-9, atol=1e-9 * np.abs(r).max())


def test_projected_linear_term_is_orthogonal_to_the_shift_directions(tmp_path):
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace", relative=True)
    Lt, g, h0 = pm.laplace_terms(1, 1.5e5)
    h = (1.5e5 * gauss_cells(0.2, -0.1, *LIB_S)).ravel()
    D = np.stack([np.gradient(h.reshape(NB, NB), axis=0).ravel(),
                  np.gradient(h.reshape(NB, NB), axis=1).ravel()], -1)
    G, Gp, gt = fitstat_pm.project_shift(Lt, g, D)
    assert np.abs(D.T @ g).max() > 1e-3 * np.abs(D.T @ gt).max()  # g itself is not orthogonal
    np.testing.assert_allclose(D.T @ gt, 0, atol=1e-9 * np.abs(D.T @ g).max())
    # and the whitened rows' projector kills G
    X = np.random.default_rng(0).normal(size=(4, G.shape[0]))
    np.testing.assert_allclose(fitstat_pm.project_rows(X, G, Gp) @ G, 0, atol=1e-8 * np.abs(X @ G).max())


def test_projected_statistic_is_the_profile_over_the_shift(tmp_path):
    """|P Lt(h-h0)|^2 + 2 gt'(h-h0) == min_c Q(h + D c) + g'D (G'G)^-1 D'g."""
    pm = make_pm_lap(tmp_path, MEANS, statistic="laplace", relative=True)
    Lt, g, h0 = pm.laplace_terms(1, 1.5e5)
    h = (1.5e5 * gauss_cells(0.4, -0.3, *LIB_S)).ravel()
    D = np.stack([np.gradient(h.reshape(NB, NB), axis=0).ravel(),
                  np.gradient(h.reshape(NB, NB), axis=1).ravel()], -1)
    G, Gp, gt = fitstat_pm.project_shift(Lt, g, D)
    d = h - h0
    r = fitstat_pm.project_rows((Lt @ d)[None], G, Gp)[0]
    proj = r @ r + 2 * gt @ d
    Q = lambda c: fs.stat_offset_free(h + D @ c, Lt, g, h0)
    GG = G.T @ G
    c = -np.linalg.solve(GG, G.T @ (Lt @ d) + D.T @ g)  # exact minimiser of the quadratic
    const = (D.T @ g) @ np.linalg.solve(GG, D.T @ g)  # the shift-only constant, dropped
    assert Q(c) == pytest.approx(proj - const, rel=1e-8, abs=1e-6 * abs(proj))
    assert Q(c) < Q(np.zeros(2)) and Q(c) < Q(c + 0.01)


def test_relative_laplace_solve_recovers_the_weights_as_well_as_addback(tmp_path):
    errs = {}
    for relative in (False, True):
        d = tmp_path / str(relative)
        d.mkdir()
        fake, lib, pm, w_true = _toy_lap(d, reexpand=0, relative=relative)
        lib.vel_histograms = list(lib.full)
        w = (fake._solve_relative if relative else fake._solve_laplace)(lib)[0]
        errs[relative] = np.linalg.norm(w - w_true) / np.linalg.norm(w_true)
        if relative:
            assert fake.pm_recentre_info["recentre_mode"] == "vp"
    assert errs[False] < 0.2  # the toy is informative
    assert errs[True] < 1.5 * errs[False] + 0.02, errs

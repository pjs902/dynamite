"""Relative proper motions (C1/C2 of the pm-recentre design).

A ``ProperMotions`` set with ``relative: True`` has an unknown mean per
aperture. ``shift_data`` moves the data by a reference shift, and
``NNLS._prepare_kinematic_block`` projects a free per-aperture shift out of
the set's whitened rows and records the orbit moments n_oa, mu_oa that the
pass-0/pass-1 driver needs for m_a(w) = w @ mu_a / w @ n_a.

Run from the repo root:
    PYTHONPATH=. pytest dev_tests/test_relative_pm.py
"""

import logging
import types

import numpy as np
import pytest
from scipy.special import ndtr

from astropy import table

from dynamite import kinematics as dyn_kin
from dynamite.weight_solvers import NNLS

VR, NB = 7.5, 15  # data grid: 15 bins of 1.0 over [-7.5, 7.5]
EDG = np.linspace(-VR, VR, NB + 1)
CEN = 0.5 * (EDG[:-1] + EDG[1:])
N_ORBS = 6


def gauss_cells(mx, my, sx=2.0, sy=1.6, edg=EDG):
    """Exact cell masses of a separable Gaussian (unit total)."""
    px = np.diff(ndtr((edg - mx) / sx))
    py = np.diff(ndtr((edg - my) / sy))
    return np.outer(px, py)


def mean_of(h):
    return np.array([h.sum(1) @ CEN, h.sum(0) @ CEN]) / h.sum()


def make_pm(tmp_path, means, relative=True, name="hst", seed=0):
    rng = np.random.default_rng(seed)
    hist = np.stack([gauss_cells(*m) for m in means])
    hist = hist + 1e-4 * rng.random(hist.shape)  # not exactly Gaussian
    sigma = 0.01 + 0.05 * hist
    np.savez(tmp_path / f"{name}.npz", PM_2dhist=hist, PM_2dhist_sigma=sigma,
             vxrange=VR, vyrange=VR)
    kw = dict(relative=True) if relative else {}
    return dyn_kin.ProperMotions(name=name, datafile=f"{name}.npz",
                                 input_directory=str(tmp_path) + "/",
                                 type="ProperMotions", **kw)


def projector(G):
    """Explicit P = I - G (G'G)^-1 G' for one aperture (G: k x 2)."""
    return np.eye(G.shape[0]) - G @ np.linalg.pinv(G)


# --------------------------------------------------------------- C2 basis/P
def _basis_vs_fd(nbins):
    edg = np.linspace(-9, 9, nbins + 1)
    c = 0.5 * (edg[:-1] + edg[1:])
    vx, vy = np.meshgrid(c, c, indexing="ij")
    cov = np.array([[4.0, 1.0], [1.0, 2.5]])
    d = np.stack([vx - 0.7, vy + 0.4], -1)
    h = np.exp(-0.5 * np.einsum("xyk,kl,xyl->xy", d, np.linalg.inv(cov), d))
    h = 3.0 * h / h.sum()  # noiseless, correlated
    G = dyn_kin.pm_shift_basis(h[None], c, c)[0]
    fd = np.stack(np.gradient(h, c, c), -1)
    return np.linalg.norm(G - fd) / np.linalg.norm(fd)


def test_shift_basis_matches_central_differences_on_a_gaussian():
    # agree up to discretisation error, which is O(dv^2): the production
    # grid (dv ~ sigma/2) is the coarse case here
    coarse, fine = _basis_vs_fd(19), _basis_vs_fd(73)
    assert coarse < 0.15 and fine < 0.01, (coarse, fine)
    assert 8 < coarse / fine < 20, (coarse, fine)  # ~ (73/19)^2 = 14.8


def test_projector_is_idempotent_symmetric_and_kills_a_small_shift():
    h0 = gauss_cells(0.3, -0.2)
    err = 0.01 + 0.05 * h0
    G = (dyn_kin.pm_shift_basis(h0[None], CEN, CEN)[0] / err[..., None]
         ).reshape(-1, 2)
    P = projector(G)
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.allclose(P, P.T, atol=1e-12)
    res = []
    for delta in (0.1, 0.05, 0.025):
        dd = ((gauss_cells(0.3 + delta, -0.2 - delta) - h0) / err).ravel()
        res.append(np.linalg.norm(P @ dd) / np.linalg.norm(dd))
    # unprojected change is O(delta); what survives P is O(delta^2), so the
    # surviving fraction halves with delta (plus a small discretisation floor)
    assert res[0] < 0.05
    assert 1.6 < res[0] / res[1] < 2.4 and 1.6 < res[1] / res[2] < 2.4, res


# ------------------------------------------------------------- C1 shift_data
def test_shift_data_conserves_mass_moves_mean_and_never_compounds(tmp_path):
    means = [(0.0, 0.0), (1.0, -0.5), (-1.0, 0.5)]
    pm = make_pm(tmp_path, means)
    h_orig = pm.data["PM_2dhist"].copy()
    s_orig = pm.data["PM_2dhist_sigma"].copy()
    m = np.array([[0.4, -0.3], [-0.8, 0.6], [1.0, 0.8]])
    pm.shift_data(m)
    h1 = pm.data["PM_2dhist"].copy()
    assert np.allclose(h1.sum(axis=(1, 2)), h_orig.sum(axis=(1, 2)),
                       rtol=1e-12)
    for a in range(3):
        assert np.allclose(mean_of(h1[a]) - mean_of(h_orig[a]), -m[a],
                           atol=0.02)
    assert np.all(pm.data["PM_2dhist_sigma"] > 0)
    # second call shifts from the original, not from h1
    pm.shift_data(m)
    assert np.array_equal(pm.data["PM_2dhist"], h1)
    assert np.array_equal(pm.data_unshifted["PM_2dhist"], h_orig)
    assert np.array_equal(pm.data_unshifted["PM_2dhist_sigma"], s_orig)
    # zero shift restores the original to regrid round-off
    pm.shift_data(np.zeros((3, 2)))
    assert np.allclose(pm.data["PM_2dhist"], h_orig, atol=1e-12)


def test_shift_plus_then_minus_round_trips():
    h = np.stack([gauss_cells(0.5, -0.5), gauss_cells(-1.0, 0.8)])
    m = np.array([[0.37, -0.61], [1.2, 0.45]])
    xedg = (EDG, EDG)
    back = dyn_kin.ProperMotions.shift_histograms(
        dyn_kin.ProperMotions.shift_histograms(h, xedg, m), xedg, -m)
    # two spline regrids at dv = sigma/2 (as in production) smooth by ~1%
    assert np.max(np.abs(back - h)) < 2e-2 * h.max()


def test_shift_data_rejects_bad_shape(tmp_path):
    pm = make_pm(tmp_path, [(0, 0), (1, 1)])
    with pytest.raises(ValueError):
        pm.shift_data(np.zeros(2))


# --------------------------------------------------------- C2 in the solver
def _fake_nnls(kin_sets, n_ap_total, stream=False, gap=0, dtype=np.float64):
    fake = types.SimpleNamespace()
    fake.nnls_dtype = dtype
    fake.stream_reads = stream
    fake.gram_aperture_group = gap
    fake.gram_chunk_rows = 5
    fake.CRcut = False
    fake.settings = {}
    fake.pm_moments = {}
    fake.adelie_mu = 1e7
    fake.n_intrinsic = 2
    fake.n_apertures = n_ap_total
    fake.n_mass_constraints = 1 + fake.n_intrinsic + n_ap_total
    fake.intrinsic_masses = np.array([3e5, 2e5])
    fake.intrinsic_mass_error = 0.05
    fake.projected_masses = np.linspace(1e5, 3e5, n_ap_total)
    fake.projected_mass_error = 0.1
    fake.total_mass = 1e6
    fake.total_mass_error = 1.0
    fake.direc_with_ml = "fake"
    fake.logger = logging.getLogger("test_relative_pm")
    fake.system = types.SimpleNamespace(
        get_unique_triaxial_visible_component=lambda: types.SimpleNamespace(
            kinematic_data=kin_sets))
    fake.calls = []
    for name in ("construct_nnls_matrix_and_rhs",
                 "construct_adelie_matrix_and_rhs",
                 "construct_gram_and_rhs_blockwise", "_econ_divide_block",
                 "apply_CR_cut", "_relative_pm_block", "_rotation_blocks",
                 "_rotation_mass"):
        setattr(fake, name, types.MethodType(getattr(NNLS, name), fake))

    def recording_prepare(self, kins, hist, tmp, prj, ap0=0):
        out = NNLS._prepare_kinematic_block(self, kins, hist, tmp, prj, ap0)
        self.calls.append((kins.name, ap0,
                           *(np.array(o, dtype=np.float64) for o in out)))
        return out
    fake._prepare_kinematic_block = types.MethodType(recording_prepare, fake)
    return fake


class FakeLOS:
    """1D GH-like set 0 (2 apertures) so the PM set is set 1 (streamable)."""
    name = "los"
    n_spatial_bins = 2
    hist_bins = 6

    def get_observed_values_and_uncertainties(self, settings):
        obs = np.arange(8, dtype=float).reshape(2, 4)
        return obs, 0.1 + 0.01 * obs

    def transform_orblib_to_observables(self, hist, settings):
        return np.moveaxis(hist.y[:, :4, :], 1, 2)


def _orblib(pm, rebin):
    rng = np.random.default_rng(11)
    n_ap = pm.n_spatial_bins
    hist1 = dyn_kin.Histogram(xedg=np.linspace(-3, 3, 7),
                              y=rng.random((N_ORBS, 6, 2)))
    # orbit PM histograms: Gaussians with orbit/aperture-dependent means,
    # on a stretched grid when rebin=True (as after ml scaling)
    edg = EDG * (1.07 if rebin else 1.0)
    y = np.empty((N_ORBS, NB, NB, n_ap))
    for o in range(N_ORBS):
        for a in range(n_ap):
            y[o, :, :, a] = (1 + o + a) * gauss_cells(
                0.5 * o - 1.0, 0.3 * a - 0.5, 1.5, 1.8, edg)
    hist2 = dyn_kin.Histogram2D(xedg=(edg, edg), y=y)
    lib = types.SimpleNamespace(n_orbs=N_ORBS)
    lib.full = [hist1, hist2]
    lib.vel_histograms = [hist1, hist2]
    lib.intrinsic_masses = rng.random((N_ORBS, 2))
    lib.projected_masses = [rng.random((N_ORBS, 2)) * 1e5,
                            y.sum(axis=(1, 2))]

    def read_vel_histograms(kin_sets, skip_density=True, dtype=None,
                            ap_group=None):
        for si in kin_sets:
            h = lib.full[si]
            if ap_group is not None:
                h = type(h)(xedg=h.xedg, y=h.y[..., ap_group[0]:ap_group[1]])
            lib.vel_histograms[si] = h
    lib.read_vel_histograms = read_vel_histograms
    return lib


@pytest.mark.parametrize("rebin", [False, True])
def test_moments_match_the_rebinned_histograms(tmp_path, rebin):
    pm = make_pm(tmp_path, [(0, 0), (0.5, -0.5), (-1, 1)])
    lib = _orblib(pm, rebin)
    y_before = lib.full[1].y.copy()
    fake = _fake_nnls([FakeLOS(), pm], 5)
    obs, err = pm.get_observed_values_and_uncertainties({})
    fake._prepare_kinematic_block(pm, lib.full[1], (obs, err),
                                  fake.projected_masses[2:])
    assert np.array_equal(lib.full[1].y, y_before)  # library not mutated
    H = pm.transform_orblib_to_observables(lib.full[1])  # (o, a, vx, vy)
    n_ref = H.sum(axis=(2, 3))
    mu_ref = np.stack([np.einsum("oaxy,x->oa", H, CEN),
                       np.einsum("oaxy,y->oa", H, CEN)], -1)
    mom = fake.pm_moments["hst"]
    assert np.allclose(mom["n"], n_ref, rtol=1e-12)
    assert np.allclose(mom["mu"], mu_ref, rtol=1e-12, atol=1e-12)


def test_relative_block_is_whitened_projected_with_unit_errors(tmp_path):
    pm = make_pm(tmp_path, [(0, 0), (0.5, -0.5), (-1, 1)])
    lib = _orblib(pm, rebin=True)
    fake = _fake_nnls([FakeLOS(), pm], 5)
    obs, err = pm.get_observed_values_and_uncertainties({})
    prj = fake.projected_masses[2:]
    o, e, orb = fake._prepare_kinematic_block(pm, lib.full[1], (obs, err), prj)
    assert np.array_equal(e, np.ones(obs.size))
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(N_ORBS, 3, -1)
    for a in range(3):
        G = (dyn_kin.pm_shift_basis(obs[a].reshape(1, NB, NB), CEN, CEN)[0]
             / err[a].reshape(NB, NB)[..., None]).reshape(-1, 2)
        P = projector(G)
        assert np.allclose(o.reshape(3, -1)[a], P @ (obs[a] / err[a]))
        assert np.allclose(orb.reshape(N_ORBS, 3, -1)[:, a],
                           (H[:, a] / (prj[a] * err[a])) @ P.T)


def test_non_relative_set_is_unchanged(tmp_path):
    """No ``relative`` key: the pre-existing formula, no moments recorded,
    and the same data as relative=False."""
    means = [(0, 0), (0.5, -0.5), (-1, 1)]
    pm = make_pm(tmp_path, means, relative=False)
    assert pm.relative is False
    lib = _orblib(pm, rebin=True)
    fake = _fake_nnls([FakeLOS(), pm], 5)
    obs, err = pm.get_observed_values_and_uncertainties({})
    prj = fake.projected_masses[2:]
    o, e, orb = NNLS._prepare_kinematic_block(fake, pm, lib.full[1],
                                              (obs, err), prj)
    assert fake.pm_moments == {}
    assert np.array_equal(o, np.ravel((obs.T * prj).T))
    assert np.array_equal(e, np.ravel((err.T * prj).T))
    assert np.array_equal(orb, pm.transform_orblib_to_observables(lib.full[1]))


def _collect(fake, path):
    """Run one read path; return per-set (obs, err, orb) reassembled over
    apertures, and the moments."""
    lib = fake.lib
    lib.vel_histograms = list(lib.full)
    fake.calls = []
    fake.pm_moments = {}
    getattr(fake, path)(lib)
    calls = sorted((c for c in fake.calls if c[0] == "hst"),
                   key=lambda c: c[1])
    n_orbs = N_ORBS
    obs = np.concatenate([c[2] for c in calls])
    err = np.concatenate([c[3] for c in calls])
    orb = np.concatenate([c[4].reshape(n_orbs, -1) for c in calls], axis=1)
    return obs, err, orb, {k: v.copy() for k, v in fake.pm_moments["hst"].items()}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_five_read_paths_agree(tmp_path, dtype):
    pm = make_pm(tmp_path, [(0, 0), (0.5, -0.5), (-1, 1), (1, 0.2), (0, -1)])
    n_ap_total = 2 + pm.n_spatial_bins
    paths = [  # (constructor, stream_reads, gram_aperture_group)
        ("construct_nnls_matrix_and_rhs", False, 0),
        ("construct_adelie_matrix_and_rhs", False, 0),
        ("construct_gram_and_rhs_blockwise", False, 0),  # whole set
        ("construct_gram_and_rhs_blockwise", False, 2),  # in-memory groups
        ("construct_gram_and_rhs_blockwise", True, 2),   # streamed groups
    ]
    results = []
    for path, stream, gap in paths:
        fake = _fake_nnls([FakeLOS(), pm], n_ap_total, stream, gap, dtype)
        fake.lib = _orblib(pm, rebin=True)
        if dtype == np.float32:
            for h in fake.lib.full:
                h.y = h.y.astype(np.float32)
        results.append(_collect(fake, path))
    # Neither dtype is compared bitwise: BLAS picks its kernel by matrix
    # shape, so one-aperture groups differ in the last bit (float32 sgemm on
    # master already; float64 dgemm by ~1.8e-16 relative under MKL, bitwise
    # on OpenBLAS). Compare to round-off instead.
    rtol, atol = (1e-12, 0) if dtype == np.float64 else (1e-5, 1e-6)
    same = lambda a, b: np.allclose(a, b, rtol=rtol, atol=atol * np.abs(a).max())
    ref = results[0]
    for (path, stream, gap), res in zip(paths[1:], results[1:]):
        for name, a, b in zip(("obs", "err", "orb"), ref[:3], res[:3]):
            assert same(a, b), (path, stream, gap, name)
        for k in ("n", "mu"):
            assert same(ref[3][k], res[3][k]), (path, stream, gap, k)


# ================================================================ C3 / C4
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from astropy.table import Table  # noqa: E402

from dynamite import weight_solvers as ws_new  # noqa: E402

DRIVER = ("_rotation_blocks", "_relative_build", "_relative_qp",
          "_relative_chi2", "_pm_model_mean", "_solve_relative", "solve",
          "_pm_shift_factor", "_relative_rank_update")


def write_rotation_files(tmp_path, superbins, components, mu, dmu, members):
    Table({"superbin": superbins, "component": components, "mu": mu,
           "dmu": dmu}).write(tmp_path / "rot.ecsv", format="ascii.ecsv",
                              overwrite=True)
    sb, ap, c = zip(*members)
    Table({"superbin": sb, "aperture": ap, "c": c}).write(
        tmp_path / "rotw.ecsv", format="ascii.ecsv", overwrite=True)
    return dict(datafile="rot.ecsv", weightfile="rotw.ecsv")


def test_rotation_rows_recover_a_known_linear_combination(tmp_path):
    rng = np.random.default_rng(3)
    n_orbs, n_ap = 7, 4
    members = [(10, 0, 0.25), (10, 1, 0.75), (20, 3, 1.0), (20, 1, 0.5)]
    cfg = write_rotation_files(tmp_path, [10, 20, 20], ["pm1", "pm2", "pm1"],
                               [0.1, -0.2, 0.3], [0.5, 0.25, 2.0], members)
    rot = dyn_kin.RotationRows(input_directory=str(tmp_path) + "/", **cfg)
    mu = rng.normal(size=(n_orbs, n_ap, 2))
    M = rng.uniform(1, 3, n_ap)
    w = rng.uniform(0, 1, n_orbs)
    A, b = rot.rows(mu, M)
    m = (w @ mu.reshape(n_orbs, -1)).reshape(n_ap, 2)  # sum_o w_o mu_oa
    expect = np.array([0.25 * m[0, 0] / M[0] + 0.75 * m[1, 0] / M[1],
                       m[3, 1] / M[3] + 0.5 * m[1, 1] / M[1],
                       m[3, 0] / M[3] + 0.5 * m[1, 0] / M[1]])
    assert np.allclose(A @ w * rot.dmu, expect, rtol=1e-13)
    assert np.allclose(b, np.array([0.1, -0.2, 0.3]) / rot.dmu)
    # only allowed on relative sets, and validated
    with pytest.raises(ValueError):
        dyn_kin.RotationRows(input_directory=str(tmp_path) + "/",
                             **write_rotation_files(tmp_path, [1], ["pmT"],
                                                    [0.], [1.], [(1, 0, 1.)]))


def test_rotation_rows_tangential_mix_both_axes(tmp_path):
    """c_pm1/c_pm2 columns: one row mixes both axes per aperture (1D pmT rings)."""
    rng = np.random.default_rng(4)
    n_orbs, n_ap = 6, 5
    Table({"superbin": [0, 1], "component": ["pmT", "pmT"], "mu": [-1., 2.],
           "dmu": [0.5, 2.0]}).write(tmp_path / "rot.ecsv", format="ascii.ecsv")
    Table({"superbin": [0, 0, 1, 1], "aperture": [0, 2, 1, 4],
           "c": [0.4, 0.6, 0.5, 0.5], "c_pm1": [0.3, -0.1, 0.2, 0.5],
           "c_pm2": [0.1, 0.6, -0.4, 0.0]}).write(tmp_path / "rotw.ecsv",
                                                 format="ascii.ecsv")
    rot = dyn_kin.RotationRows(input_directory=str(tmp_path) + "/",
                               datafile="rot.ecsv", weightfile="rotw.ecsv")
    mu = rng.normal(size=(n_orbs, n_ap, 2))
    M = rng.uniform(1, 3, n_ap)
    w = rng.uniform(0, 1, n_orbs)
    A, b = rot.rows(mu, M)
    m = np.einsum("o,oak->ak", w, mu) / M[:, None]
    expect = np.array([0.3 * m[0, 0] + 0.1 * m[0, 1] - 0.1 * m[2, 0] + 0.6 * m[2, 1],
                       0.2 * m[1, 0] - 0.4 * m[1, 1] + 0.5 * m[4, 0]])
    assert np.allclose(A @ w * rot.dmu, expect, rtol=1e-13)
    assert np.allclose(b, np.array([-1., 2.]) / rot.dmu)


class ConsistentLOS(FakeLOS):
    """FakeLOS whose data are those of given weights on the toy library."""

    def __init__(self, obs, err):
        self.obs, self.err = obs, err

    def get_observed_values_and_uncertainties(self, settings):
        return self.obs, self.err


def _toy(tmp_path, relative=True, rotation=True, solver="scipy", gram=False,
         d_tol=0.01, f_tol=0.05, norm=None, mge_scale=None, mode=None,
         start=None, step_tol=None, max_rounds=None):
    """Relative PM toy with a known truth.

    Library: _orblib(rebin=False) (orbit means depend on o in x, on a in y).
    Truth w_true; masses, LOS data and rotation rows consistent with it. The
    PM data are the true model histograms recentred to mean 0 (relative).
    ``norm`` sets the rotation rows' normalisation; ``mge_scale`` (n_ap,)
    multiplies the PM set's MGE projected masses (a library that cannot
    match the MGE). ``mode``, ``step_tol``, ``max_rounds``: the
    pm_recentre_* settings; ``start``: a callable m_true -> start_mean.
    Returns (fake, lib, pm, w_true, m_true).
    """
    w_true = np.array([1.3, 0.4, 0.9, 1.6, 0.2, 1.1])
    probe = make_pm(tmp_path, [(0, 0)] * 3, name="probe")
    lib = _orblib(probe, rebin=False)
    y = lib.full[1].y                                    # (o, vx, vy, a)
    M_pm = w_true @ y.sum(axis=(1, 2))                   # aperture masses
    H = np.einsum("o,oxya->axy", w_true, y) / M_pm[:, None, None]
    m_true = np.stack([H.sum(2) @ CEN, H.sum(1) @ CEN], -1)
    rel = dyn_kin.ProperMotions.shift_histograms(H, (EDG, EDG), m_true)
    hist = rel  # the data are relative either way
    sigma = 0.002 + 0.02 * hist.clip(0)
    np.savez(tmp_path / "hst.npz", PM_2dhist=hist, PM_2dhist_sigma=sigma,
             vxrange=VR, vyrange=VR)
    kw = {}
    if relative:
        kw["relative"] = True
        if rotation:
            kw["rotation_rows"] = write_rotation_files(
                tmp_path, [0, 0, 1, 1], ["pm1", "pm2", "pm1", "pm2"],
                [0.5 * (m_true[0, 0] + m_true[1, 0]),
                 0.5 * (m_true[0, 1] + m_true[1, 1]),
                 m_true[2, 0], m_true[2, 1]], [0.01] * 4,
                [(0, 0, 0.5), (0, 1, 0.5), (1, 2, 1.0)])
            if norm is not None:
                kw["rotation_rows"]["normalisation"] = norm
        if start is not None:
            c0 = start(m_true)
            table.Table({"pm1": c0[:, 0], "pm2": c0[:, 1]}).write(
                tmp_path / "start.ecsv", format="ascii.ecsv", overwrite=True)
            kw["start_mean"] = "start.ecsv"
    pm = dyn_kin.ProperMotions(name="hst", datafile="hst.npz",
                               input_directory=str(tmp_path) + "/",
                               type="ProperMotions", **kw)
    M_los = w_true @ lib.projected_masses[0]
    los_obs = (np.einsum("o,oja->aj", w_true, lib.full[0].y[:, :4, :])
               / M_los[:, None])
    los = ConsistentLOS(los_obs, 0.01 * np.abs(los_obs) + 1e-3)
    fake = _fake_nnls([los, pm], 5, dtype=np.float64)
    fake.lib = lib
    fake.total_mass = float(w_true.sum())
    fake.intrinsic_masses = w_true @ lib.intrinsic_masses
    fake.projected_masses = np.concatenate(
        [M_los, M_pm * (1 if mge_scale is None else np.asarray(mge_scale))])
    fake.nnls_solver = solver
    fake.gram_blockwise = gram
    fake.ridge_lambda = 0.0
    fake.admm_rho, fake.admm_alpha, fake.admm_precision = None, 1.0, "float64"
    fake.admm_max_iters, fake.admm_tol, fake.admm_free_p = 20000, 1e-12, False
    fake.settings = {"pm_recentre_delta_tol": d_tol,
                     "pm_recentre_frac_tol": f_tol}
    for key, val in (("mode", mode), ("step_tol", step_tol),
                     ("max_rounds", max_rounds)):
        if val is not None:
            fake.settings[f"pm_recentre_{key}"] = val
    fake.direc_with_ml = str(tmp_path) + "/"
    fake._pm_dsq_over_sigsq = NNLS._pm_dsq_over_sigsq
    fake._pm_fit_shift = NNLS._pm_fit_shift
    for name in DRIVER:
        setattr(fake, name, types.MethodType(getattr(NNLS, name), fake))
    return fake, lib, pm, w_true, m_true


def _model_mean(lib, w):
    y = lib.full[1].y
    H = np.einsum("o,oxya->axy", w, y)
    return np.stack([H.sum(2) @ CEN, H.sum(1) @ CEN], -1) / H.sum(
        axis=(1, 2))[:, None]


def test_recentring_recovers_the_true_moments_better_than_no_shift(tmp_path):
    fake, lib, pm, w_true, m_true = _toy(tmp_path)
    h_rel = pm.data["PM_2dhist"].copy()
    w, chi2_tot, chi2_kin, chi2 = fake._solve_relative(lib)
    err_driver = np.abs(_model_mean(lib, w) - m_true).max()
    # data restored to relative on exit
    assert np.array_equal(pm.data["PM_2dhist"], h_rel)
    # no shift: the same relative data treated as absolute
    (tmp_path / "plain").mkdir()
    fake0, lib0, *_ = _toy(tmp_path / "plain", relative=False)
    A, b = fake0.construct_nnls_matrix_and_rhs(lib0)
    A_max = np.abs(A).max(axis=0)
    x, _ = ws_new.optimize.nnls(A / A_max, b / np.abs(b).max())
    w_plain = x * np.abs(b).max() / A_max
    err_plain = np.abs(_model_mean(lib0, w_plain) - m_true).max()
    assert err_driver < 0.1 * err_plain, (err_driver, err_plain)
    assert err_driver < 0.02, err_driver
    names = [c[0] for c in chi2]
    assert names == ["mass", "los", "hst", "rot_hst"]
    assert dict((c[0], c[2]) for c in chi2)["hst"] == 3 * NB * NB - 6
    assert np.isclose(chi2_tot, sum(c[1] for c in chi2))


@pytest.mark.parametrize("solver,gram,stream,gap", [
    ("scipy", False, False, 0), ("admm", False, False, 0),
    ("admm", True, False, 0), ("admm", True, False, 2),
    ("admm", True, True, 2)])
def test_extra_pass_changes_only_the_right_hand_side(tmp_path, solver, gram,
                                                     stream, gap):
    # d_tol < 0 forces the extra pass
    fake, lib, pm, w_true, m_true = _toy(tmp_path, solver=solver, gram=gram,
                                         d_tol=-1.0, f_tol=0.0)
    fake.stream_reads, fake.gram_aperture_group = stream, gap
    builds = []

    def spy(self, orblib, gram_, rhs_from=None):
        prob = NNLS._relative_build(self, orblib, gram_, rhs_from)
        if rhs_from is not None:
            # a full rebuild under the frozen basis gives the same matrix
            full = NNLS._relative_build(self, orblib, gram_)
            if "P" in prob:
                assert prob["P"] is rhs_from["P"]
                assert np.array_equal(full["P"], rhs_from["P"])
                assert np.allclose(full["q"], prob["q"], rtol=1e-10,
                                   atol=1e-12 * np.abs(prob["q"]).max())
            if "A" in prob:
                assert np.array_equal(prob["A"], rhs_from["A"])
                assert not np.array_equal(prob["b"], rhs_from["b"])
        builds.append(rhs_from is not None)
        return prob
    fake._relative_build = types.MethodType(spy, fake)
    w, *_ = fake._solve_relative(lib)
    assert builds == [False, False, True]
    z = np.load(tmp_path / "recentre_shifts.npz")
    assert bool(z["extra_pass"])
    assert not np.array_equal(z["hst_m_ref"], z["hst_m_ref_pass1"])
    assert np.abs(_model_mean(lib, w) - m_true).max() < 0.05


def test_recentre_shifts_npz_keys_and_shapes(tmp_path):
    fake, lib, pm, w_true, m_true = _toy(tmp_path)
    w, *_ = fake._solve_relative(lib)
    z = np.load(tmp_path / "recentre_shifts.npz")
    n_ap = pm.n_spatial_bins
    shapes = {f"hst_{k}": (n_ap, 2) for k in
              ("m_ref_pass1", "m_ref", "m_model", "mhat")}
    shapes.update({f"hst_{k}": (n_ap,) for k in
                   ("dsq_over_sigsq_pass1", "dsq_over_sigsq", "mass_ratio",
                    "mass_mge", "mass_pass0", "rot_mass")})
    shapes.update(extra_pass=(), chi2_names=(4,), chi2=(4,), chi2_n=(4,),
                  hst_rot_normalisation=(), recentre_mode=(), n_rounds=(),
                  converged=(), step_rms=(int(z["n_rounds"]) + 1,))
    assert set(z.files) == set(shapes)
    for k, s in shapes.items():
        assert z[k].shape == s, k
    assert np.allclose(z["hst_m_model"], _model_mean(lib, w))
    assert np.allclose(z["hst_mass_ratio"], 1.0, atol=1e-3)
    # converged: mhat ~ m_ref - m_model ~ 0 and the delta check is quiet
    assert np.abs(z["hst_mhat"]).max() < 0.05
    assert z["hst_dsq_over_sigsq"].max() < 0.01


def test_pass0_normalisation_beats_an_mge_the_library_cannot_match(
        tmp_path):
    """MGE aperture masses off by -40..+50%: 'mge' rows are inconsistent at
    any weights that fit the data; 'pass0' rows (M_a = w0 @ n_a, frozen)
    share m_ref's denominator. The PM data are also scaled by the MGE mass,
    so pass 1 still drifts from w0's masses and the gain is partial."""
    scale = [1.5, 0.6, 1.3]
    err, rot, z = {}, {}, {}
    for norm in ("mge", "pass0"):
        d = tmp_path / norm
        d.mkdir()
        fake, lib, pm, w_true, m_true = _toy(d, norm=norm, mge_scale=scale)
        w, *_, chi2 = fake._solve_relative(lib)
        err[norm] = np.abs(_model_mean(lib, w) - m_true).max()
        rot[norm] = {c[0]: c[1] / c[2] for c in chi2}["rot_hst"]
        z[norm] = np.load(d / "recentre_shifts.npz")
    zp, zm = z["pass0"], z["mge"]
    assert str(zp["hst_rot_normalisation"]) == "pass0"
    assert np.array_equal(zp["hst_rot_mass"], zp["hst_mass_pass0"])
    assert not np.allclose(zp["hst_mass_pass0"], zp["hst_mass_mge"], rtol=0.1)
    assert np.array_equal(zm["hst_rot_mass"], zm["hst_mass_mge"])
    assert rot["pass0"] < 0.5 * rot["mge"], rot
    assert err["pass0"] < err["mge"], err


def test_rotation_rows_reject_bad_normalisation_and_zero_mass(tmp_path):
    cfg = write_rotation_files(tmp_path, [0], ["pm1"], [0.], [1.],
                               [(0, 1, 1.0)])
    with pytest.raises(ValueError):
        dyn_kin.RotationRows(input_directory=str(tmp_path) + "/",
                             normalisation="pass1", **cfg)
    rot = dyn_kin.RotationRows(input_directory=str(tmp_path) + "/", **cfg)
    assert rot.normalisation == "pass0"
    mu = np.ones((3, 2, 2))
    rot.rows(mu, np.array([0.0, 1.0]))  # aperture 0 is not a member
    with pytest.raises(ValueError):
        rot.rows(mu, np.array([1.0, 0.0]))


# -------------------------------------------- non-relative: bit-identical
def _load_parent_weight_solvers():
    """weight_solvers.py as of the commit before C3/C4, as a module."""
    root = Path(__file__).resolve().parents[1]
    try:
        src = subprocess.run(
            ["git", "-C", str(root), "show", "420afb8:dynamite/weight_solvers.py"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with 420afb8 not available")
    mod = types.ModuleType("weight_solvers_parent")
    mod.__file__ = "weight_solvers_parent.py"
    sys.modules["weight_solvers_parent"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("solver,gram", [("scipy", False), ("admm", True),
                                         ("admm", False)])
def test_non_relative_solve_is_bit_identical_to_before(tmp_path, solver, gram):
    parent = _load_parent_weight_solvers()
    out = {}
    for tag, cls in (("new", NNLS), ("old", parent.NNLS)):
        d = tmp_path / tag
        d.mkdir()
        fake, lib, *_ = _toy(d, relative=False, solver=solver, gram=gram)
        for name in ("construct_nnls_matrix_and_rhs",
                     "construct_gram_and_rhs_blockwise",
                     "_prepare_kinematic_block", "_econ_divide_block",
                     "apply_CR_cut", "solve") + (
                         ("_relative_pm_block", "_rotation_blocks")
                         if cls is NNLS else ()):
            setattr(fake, name, types.MethodType(getattr(cls, name), fake))
        fake.stream_reads = False
        fake.adelie_mu = 1e7
        fake.weight_file = str(d / "w.ecsv")
        fake.weight_file_exists = lambda: False
        fake.config = types.SimpleNamespace(
            copy_config_file=lambda _: None, config_file_name=str(d / "c.yaml"),
            settings=types.SimpleNamespace(io_settings={"output_directory": str(d)}))
        fake.chi2_kinmap = lambda w, orblib=None: float("nan")
        fake.cvxopt_tol = 1e-11
        fake._ridge_shift = 0.0
        lib.read_vel_histograms = lambda *a, **k: None
        out[tag] = fake.solve(lib, ignore_existing_weights=True)
    for a, b in zip(out["new"], out["old"]):
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def test_mhat_measures_the_residual_shift(tmp_path):
    """Data put at m_ref = m_true + s: m-hat(w_true) ~ m_ref - m_model = s."""
    fake, lib, pm, w_true, m_true = _toy(tmp_path)
    s = np.array([0.3, -0.2])
    pm.shift_data(-(m_true + s))  # shift_data(m) moves the mean by -m
    fake.construct_nnls_matrix_and_rhs(lib)
    mom = fake.pm_moments["hst"]
    mhat = np.einsum("o,oak->ak", w_true, mom["ghat"]) - mom["dhat"]
    assert np.allclose(mhat, s, atol=0.03), mhat


def test_node_settings_ridge_applied_once_and_free_p_ignored(tmp_path,
                                                              caplog):
    # node config: ridge_lambda 0.03, admm_free_p True, alpha 1.9; d_tol < 0
    # forces the extra pass, which reuses P: the ridge must be on P once
    fake, lib, pm, w_true, m_true = _toy(tmp_path, solver="admm", gram=True,
                                         d_tol=-1.0, f_tol=0.0)
    fake.stream_reads, fake.gram_aperture_group = False, 0
    fake.ridge_lambda, fake.admm_free_p, fake.admm_alpha = 0.03, True, 1.9
    seen = []

    def spy(self, orblib, gram_, rhs_from=None):
        prob = NNLS._relative_build(self, orblib, gram_, rhs_from)
        seen.append((rhs_from is not None, prob["P"], prob["P"].diagonal().copy(),
                     self._ridge_shift))
        return prob
    fake._relative_build = types.MethodType(spy, fake)
    with caplog.at_level(logging.INFO, logger="test_relative_pm"):
        w, *_ = fake._solve_relative(lib)
    assert "admm_free_p ignored" in caplog.text
    assert [s[0] for s in seen] == [False, False, True]
    (_, P1, d1, shift1), (_, P2, d2, _) = seen[1], seen[2]
    assert P2 is P1 and np.array_equal(d1, d2)  # extra pass: same P, no re-ridge
    raw = NNLS._relative_build(fake, lib, True)  # fresh build, ridged once
    p = raw["P"].shape[0]
    unridged = raw["P"].diagonal() - fake._ridge_shift
    assert np.isclose(fake._ridge_shift, 0.03 * unridged.sum() / p)
    assert np.allclose(d1, unridged + shift1, rtol=1e-12)
    assert np.abs(_model_mean(lib, w) - m_true).max() < 0.05


def test_float32_projection_matches_float64(tmp_path):
    # nnls_dtype float32: projected orbit rows and ghat agree with float64 to
    # single precision, and the projected rows stay orthogonal to G
    pm = make_pm(tmp_path, [(0.3, -0.2), (1.0, 0.5), (-0.7, 0.1)])
    lib = _orblib(pm, rebin=False)
    out = {}
    for dt in (np.float64, np.float32):
        fake = _fake_nnls([pm], 3, dtype=dt)
        obs, err = pm.get_observed_values_and_uncertainties(None)
        orb = pm.transform_orblib_to_observables(lib.full[1]).astype(dt)
        o, e, h = NNLS._relative_pm_block(fake, pm, obs.copy(), err.copy(),
                                          orb, 0)
        out[dt] = (h.astype(np.float64), fake.pm_moments["hst"])
    h64, m64 = out[np.float64]
    h32, m32 = out[np.float32]
    scale = np.abs(h64).max()
    assert np.abs(h32 - h64).max() < 1e-5 * scale
    assert np.allclose(m32["ghat"], m64["ghat"], rtol=1e-5,
                       atol=1e-6 * np.abs(m64["ghat"]).max())
    for a in range(3):
        G = m64["G"][a]
        resid = h32[:, a].reshape(h32.shape[0], -1) @ G
        assert np.abs(resid).max() < 1e-5 * scale * np.linalg.norm(G, axis=0).max()


def _builds_spy(fake):
    """Record, per _relative_build call, whether it was rhs-only."""
    builds = []

    def spy(self, orblib, gram_, rhs_from=None):
        builds.append(rhs_from is not None)
        return NNLS._relative_build(self, orblib, gram_, rhs_from)
    fake._relative_build = types.MethodType(spy, fake)
    return builds


@pytest.mark.parametrize("solver,gram", [("scipy", False), ("admm", True)])
def test_rebuild_mode_rebuilds_every_round_and_converges(tmp_path, solver,
                                                         gram):
    # start 2 units off the truth, so at least one rebuilt round is needed
    fake, lib, pm, w_true, m_true = _toy(
        tmp_path, solver=solver, gram=gram, mode="rebuild",
        start=lambda m: m + 2.0, step_tol=1e-3)
    builds = _builds_spy(fake)
    w, *_ = fake._solve_relative(lib)
    z = np.load(tmp_path / "recentre_shifts.npz")
    assert str(z["recentre_mode"]) == "rebuild" and bool(z["converged"])
    n = int(z["n_rounds"])
    assert n >= 1 and z["step_rms"].shape == (n + 1,)
    assert z["step_rms"][-1] < 1e-3 <= z["step_rms"][0]
    # pass 0 + pass 1 + n rebuilt rounds, none rhs-only; on the Gram path
    # the rounds are rank updates (_relative_rank_update), not builds
    assert builds == [False] * (2 if gram else n + 2)
    assert np.allclose(z["hst_m_ref_pass1"], m_true + 2.0)
    assert np.abs(_model_mean(lib, w) - m_true).max() < 0.02
    assert fake.pm_recentre_info == dict(recentre_mode="rebuild",
                                         n_rounds=n, converged=True)


def test_rebuild_mode_reaches_one_answer_from_different_starts(tmp_path):
    ends = []
    for i, off in enumerate((-2.0, 0.0, 2.5)):
        d = tmp_path / f"s{i}"
        d.mkdir()
        fake, lib, pm, w_true, m_true = _toy(
            d, mode="rebuild", start=lambda m, o=off: m + o, step_tol=1e-4,
            max_rounds=30)
        w, *_ = fake._solve_relative(lib)
        assert fake.pm_recentre_info["converged"]
        ends.append(_model_mean(lib, w))
    assert np.abs(ends[0] - ends[1]).max() < 1e-3
    assert np.abs(ends[2] - ends[1]).max() < 1e-3


@pytest.mark.parametrize("solver,gram", [("scipy", False), ("admm", True)])
def test_profile_mode_reaches_the_truth_from_different_starts(tmp_path,
                                                              solver, gram):
    # exact per-aperture centre fit, P never rebuilt: one answer, the truth.
    # The fitted shift levels off at ~2e-4 (model-side fit, data-side move)
    ends = []
    for i, off in enumerate((-2.0, 2.5)):
        d = tmp_path / f"s{i}"
        d.mkdir()
        fake, lib, pm, w_true, m_true = _toy(
            d, solver=solver, gram=gram, mode="profile",
            start=lambda m, o=off: m + o, step_tol=1e-3, max_rounds=30)
        builds = _builds_spy(fake)
        w, *_ = fake._solve_relative(lib)
        info = fake.pm_recentre_info
        assert info["recentre_mode"] == "profile" and info["converged"]
        # pass 0 and pass 1 build P; every later read is rhs-only
        assert builds[:2] == [False, False] and all(builds[2:])
        z = np.load(d / "recentre_shifts.npz")
        assert np.abs(z["hst_s_final"]).max() < 1e-3
        ends.append(_model_mean(lib, w))
        assert np.abs(ends[-1] - m_true).max() < 0.02, i
    assert np.abs(ends[0] - ends[1]).max() < 1e-3


def test_profile_mode_round_cap_flags_not_converged(tmp_path):
    fake, lib, pm, *_ = _toy(tmp_path, mode="profile",
                             start=lambda m: m + 2.0, step_tol=-1.0,
                             max_rounds=2)
    fake._solve_relative(lib)
    assert fake.pm_recentre_info == dict(recentre_mode="profile",
                                         n_rounds=2, converged=False)


def test_rebuild_mode_round_cap_flags_not_converged(tmp_path):
    fake, lib, pm, *_ = _toy(tmp_path, mode="rebuild",
                             start=lambda m: m + 2.0, step_tol=-1.0,
                             max_rounds=2)
    builds = _builds_spy(fake)
    fake._solve_relative(lib)
    assert builds == [False] * 4
    assert fake.pm_recentre_info == dict(recentre_mode="rebuild",
                                         n_rounds=2, converged=False)


def test_bad_recentre_mode_is_rejected(tmp_path):
    fake, lib, *_ = _toy(tmp_path, mode="sometimes")
    with pytest.raises(ValueError, match="pm_recentre_mode"):
        fake._solve_relative(lib)


def test_start_mean_needs_a_relative_set_and_one_row_per_aperture(tmp_path):
    _toy(tmp_path)  # writes hst.npz
    with pytest.raises(ValueError, match="start_mean needs a relative"):
        dyn_kin.ProperMotions(name="x", datafile="hst.npz",
                              input_directory=str(tmp_path) + "/",
                              type="ProperMotions", start_mean="start.ecsv")
    table.Table({"pm1": [0.0], "pm2": [0.0]}).write(
        tmp_path / "start.ecsv", format="ascii.ecsv", overwrite=True)
    with pytest.raises(ValueError, match="start_mean must be finite"):
        dyn_kin.ProperMotions(name="x", datafile="hst.npz",
                              input_directory=str(tmp_path) + "/",
                              type="ProperMotions", relative=True,
                              start_mean="start.ecsv")


@pytest.mark.parametrize("stream,gap", [(False, 0), (True, 2)])
def test_rank_update_round_equals_a_full_rebuild(tmp_path, stream, gap):
    # same rounds, rank-update Gram vs full rebuild: same P, q and weights
    out = {}
    for rank in (True, False):
        d = tmp_path / f"r{rank}"
        d.mkdir()
        fake, lib, pm, w_true, m_true = _toy(
            d, solver="admm", gram=True, mode="rebuild",
            start=lambda m: m + 2.0, step_tol=-1.0, max_rounds=2)
        fake.settings["pm_recentre_rank_update"] = rank
        fake.stream_reads, fake.gram_aperture_group = stream, gap
        seen = []
        orig = fake._relative_qp

        def spy(prob, warm=None, orig=orig, seen=seen):
            seen.append((prob["P"].copy(), prob["q"].copy()))
            return orig(prob, warm)
        fake._relative_qp = spy
        w, *_ = fake._solve_relative(lib)
        out[rank] = (seen, w, np.load(d / "recentre_shifts.npz")["chi2"])
    (s1, w1, c1), (s0, w0, c0) = out[True], out[False]
    assert len(s1) == len(s0) == 4  # pass 0, pass 1, 2 rounds
    for (P1, q1), (P0, q0) in zip(s1, s0):
        assert np.allclose(P1, P0, rtol=0, atol=1e-10 * np.abs(P0).max())
        assert np.allclose(q1, q0, rtol=0, atol=1e-10 * np.abs(q0).max())
    assert np.allclose(w1, w0, rtol=1e-6, atol=1e-9 * np.abs(w0).max())
    assert np.allclose(c1, c0, rtol=1e-6)


def test_rebuild_rounds_keep_the_pass1_errors(tmp_path):
    fake, lib, pm, *_ = _toy(tmp_path, solver="admm", gram=True,
                             mode="rebuild", start=lambda m: m + 2.0,
                             step_tol=-1.0, max_rounds=1)
    seen = []
    orig = NNLS._relative_pm_block

    def spy(self, kins, obs, err, orb, ap0):
        r = orig(self, kins, obs, err, orb, ap0)
        if not self._pm_moments_only:
            seen.append(self.pm_moments[kins.name]["e"].copy())
        return r
    fake._relative_pm_block = types.MethodType(spy, fake)
    fake._solve_relative(lib)
    assert len(seen) >= 3  # pass 1, the round's rhs read, the chi2 read
    for e in seen[1:]:
        assert np.array_equal(e, seen[0])


def test_rotation_rows_weight_scales_rows_by_sqrt_weight(tmp_path):
    kw = write_rotation_files(tmp_path, [0], ["pm1"], [0.5], [0.1],
                              [(0, 0, 1.0)])
    kw["input_directory"] = str(tmp_path) + "/"
    mu = np.random.default_rng(1).normal(size=(4, 2, 2))
    A1, b1 = dyn_kin.RotationRows(**kw).rows(mu, np.ones(2))
    A4, b4 = dyn_kin.RotationRows(weight=4.0, **kw).rows(mu, np.ones(2))
    assert np.allclose(A4, 2 * A1) and np.allclose(b4, 2 * b1)
    with pytest.raises(ValueError, match="weight must be > 0"):
        dyn_kin.RotationRows(weight=0.0, **kw)

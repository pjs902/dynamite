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
                 "apply_CR_cut", "_relative_pm_block"):
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
def test_five_read_paths_agree_bitwise(tmp_path, dtype):
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
    # float32: bitwise is impossible upstream of this change. Master's
    # ProperMotions.rebin_orblib_to_observations already differs in the last
    # float32 bit for a one-aperture group (sgemm takes a different kernel),
    # so float32 is compared to float32 round-off instead.
    same = (np.array_equal if dtype == np.float64 else
            lambda a, b: np.allclose(a, b, rtol=1e-5, atol=1e-6 * np.abs(a).max()))
    ref = results[0]
    for (path, stream, gap), res in zip(paths[1:], results[1:]):
        for name, a, b in zip(("obs", "err", "orb"), ref[:3], res[:3]):
            assert same(a, b), (path, stream, gap, name)
        for k in ("n", "mu"):
            assert same(ref[3][k], res[3][k]), (path, stream, gap, k)

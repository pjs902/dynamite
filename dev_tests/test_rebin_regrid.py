"""ml-rescaled orbit libraries are regridded to the data velocity bins by
kinematics.cdf_regrid_matrix. The former area-overlap rebin (kind='linear')
assumes uniform density inside each stretched bin and smears every histogram;
the cubic-spline regrid does not. Run: python -m pytest dev_tests/test_rebin_regrid.py"""
import types
import numpy as np
from scipy.special import erf

from dynamite import kinematics as kin


def _mass(edges, sigma):
    return np.diff(0.5 * (1 + erf(edges / (np.sqrt(2) * sigma))))


def _edges(n=23, dv=0.5):
    return (np.arange(n + 1) - n / 2) * dv


def _old_area_overlap(src, dst):
    """the matrix the old rebin built: fraction of source bin j in dest bin i"""
    f = np.zeros((len(dst) - 1, len(src) - 1))
    for i in range(len(dst) - 1):
        for j in range(len(src) - 1):
            lo, hi = max(dst[i], src[j]), min(dst[i + 1], src[j + 1])
            f[i, j] = max(hi - lo, 0) / (src[j + 1] - src[j])
    return f


def test_same_grid_is_identity():
    e = _edges()
    for kind in ('spline', 'linear'):
        assert np.allclose(kin.cdf_regrid_matrix(e, e, kind), np.eye(len(e) - 1))


def test_linear_is_the_old_area_overlap():
    src, dst = _edges() * 1.0377, _edges()
    assert np.allclose(kin.cdf_regrid_matrix(src, dst, 'linear'),
                       _old_area_overlap(src, dst))


def test_mass_inside_common_window_is_conserved():
    rng = np.random.default_rng(1)
    src = _edges() * 0.97                       # narrower window than dst
    dst = _edges()
    m = rng.random(len(src) - 1)
    for kind in ('spline', 'linear'):
        out = kin.cdf_regrid_matrix(src, dst, kind) @ m
        assert np.isclose(out.sum(), m.sum())   # nothing lost: dst covers src
    out = kin.cdf_regrid_matrix(dst, src * 0.9, 'spline') @ m  # window shrinks
    assert out.sum() < m.sum() and np.isclose(
        out.sum(), (kin.cdf_regrid_matrix(dst, src * 0.9, 'linear') @ m).sum(),
        rtol=0.05)


def test_spline_keeps_the_variance_the_linear_regrid_inflates():
    sig, s = 1.0, 1.0377                        # the stretch behind the old artefact
    dst = _edges(dv=0.5 * sig)                  # dv/sigma = 0.5, as for HST PMs
    xc = 0.5 * (dst[1:] + dst[:-1])
    truth = _mass(dst, sig * s)
    lib = _mass(dst, sig)                       # library histogram on its own grid
    bias = {}
    for kind in ('spline', 'linear'):
        out = kin.cdf_regrid_matrix(dst * s, dst, kind) @ lib
        bias[kind] = (np.sum(out * xc**2) / out.sum()
                      - np.sum(truth * xc**2) / truth.sum()) / sig**2
    assert abs(bias['spline']) < 1e-3
    assert bias['linear'] > 2e-2                # the old +3% variance blur


def test_proper_motion_rebin_is_separable_and_keeps_dtype():
    rng = np.random.default_rng(2)
    xe, ye = _edges(7, 1.0), _edges(9, 0.8)
    de = (_edges(7, 1.0 / 1.04), _edges(9, 0.8 / 1.04))     # data grid
    y = rng.random((5, 7, 9, 3)).astype(np.float32)
    vel_hist = types.SimpleNamespace(y=y, xedg=(xe, ye))
    h2d = types.SimpleNamespace(xedg=de)
    stub = types.SimpleNamespace(as_histogram2d=lambda: h2d)
    out = kin.ProperMotions.rebin_orblib_to_observations(stub, vel_hist)
    rx, ry = (kin.cdf_regrid_matrix(a, b) for a, b in zip((xe, ye), de))
    assert out.dtype == np.float32 and out.shape == (5, 7, 9, 3)
    assert np.allclose(out, np.einsum('ik,jl,oklp->oijp', rx, ry, y),
                       rtol=1e-4, atol=1e-5)


def test_los_rebin_uses_the_same_matrix():
    rng = np.random.default_rng(3)
    xedg = _edges(15, 1.0) * 1.03
    dvd = 1.0
    vc = (np.arange(15) - 7) * dvd
    y = rng.random((4, 15, 2))
    hist = types.SimpleNamespace(y=y, xedg=xedg, dx=np.diff(xedg))
    data = types.SimpleNamespace(meta={'vcent': vc, 'dv': dvd})
    stub = types.SimpleNamespace(data=data)
    out = kin.BayesLOSVD.rebin_orblib_to_observations(stub, hist)
    f = kin.cdf_regrid_matrix(xedg, np.concatenate((vc - dvd / 2, [vc[-1] + dvd / 2])))
    assert np.allclose(out, np.einsum('ij,ojp->oip', f, y))

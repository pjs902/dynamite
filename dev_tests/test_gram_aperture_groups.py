"""Aperture-group streaming (S1) for the blockwise Gram build: RED tests.

S1 feeds the dsyrk chunk loop aperture GROUPS instead of whole sets, so a
124.8 GiB set never materializes at once (target: G=128 of 1555 apertures
-> ~55 GiB peak). PM sets only; GH/1D sets stay full-set. Setting
`gram_aperture_group` (0 = off = today's behavior) mirrors the
`gram_chunk_rows` pattern: real NNLS reads it from settings, fakes set the
attribute directly.

Test roles mirror test_fused_assembly's chunking pair: the agreement test
would pass vacuously if grouping silently did nothing (one whole-set group
trivially agrees with itself), so the BOUNDS test is the true RED -- it
spies what each prepare call is actually fed.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_gram_aperture_groups.py
"""

import logging
import os
import tempfile
import types as _t

import numpy as np
import pytest

from dynamite import kinematics as dyn_kin
from dynamite.weight_solvers import NNLS
import test_hist_bulk_read as hbr

N_ORBS = 7


class FakeKins2D:
    """PM-like: 2 spatial bins, 16 constraints per aperture (32 kin rows)."""

    kind = "pm"
    n_spatial_bins = 2
    constraint_shape = (16,)
    hist_bins = [16, 16]  # 2D PM: list-valued (int => 1D GH/LOSVD)
    name = "fake_pm"

    def get_observed_values_and_uncertainties(self, settings):
        obs = np.arange(self.n_spatial_bins * 16, dtype=float).reshape(2, 16)
        err = np.abs(obs) * 0.02 + 0.02
        return obs, err

    def transform_orblib_to_observables(self, hist, settings):
        return np.moveaxis(hist.y[:, :4, :4, :], 3, 1)


def _make_fake_nnls(dtype, stream_reads=True, aperture_group=0):
    fake = _t.SimpleNamespace()
    fake.nnls_dtype = dtype
    fake.stream_reads = stream_reads
    fake.gram_chunk_rows = 3
    fake.gram_aperture_group = aperture_group
    fake.CRcut = False
    fake.settings = {}
    fake.n_intrinsic = 2
    fake.n_apertures = 2
    fake.n_mass_constraints = 1 + fake.n_intrinsic + fake.n_apertures
    fake.intrinsic_masses = np.array([3e5, 2e5])
    fake.intrinsic_mass_error = 0.05
    fake.projected_masses = np.array([4e5, 3e5])
    fake.projected_mass_error = 0.1
    fake.total_mass = 1e6
    fake.total_mass_error = max(abs(1.0 - 1e6), 1e-8)
    fake.logger = logging.getLogger("test_gram_aperture_groups")
    fake.system = _t.SimpleNamespace(
        get_unique_triaxial_visible_component=lambda: _t.SimpleNamespace(
            kinematic_data=[FakeKins2D()]
        )
    )
    for name in (
        "construct_gram_and_rhs_blockwise",
        "_prepare_kinematic_block",
        "_econ_divide_block",
        "apply_CR_cut",
    ):
        setattr(fake, name, _t.MethodType(getattr(NNLS, name), fake))
    return fake


def _make_fake_orblib():
    rng = np.random.default_rng(11)
    y2 = rng.random((N_ORBS, 4, 5, 2))
    hist2 = dyn_kin.Histogram2D(
        xedg=(np.linspace(-2.0, 2.0, 5), np.linspace(-2.5, 2.5, 6)), y=y2
    )
    orblib = _t.SimpleNamespace()
    orblib.n_orbs = N_ORBS
    orblib.vel_histograms = [hist2]
    orblib.intrinsic_masses = rng.random((N_ORBS, 2))
    orblib.projected_masses = [rng.random((N_ORBS, 2)) * 1e5]

    def read_vel_histograms(kin_sets=None, skip_density=False, dtype=None):
        pass  # preloaded; streaming re-read is a no-op for the fake

    orblib.read_vel_histograms = read_vel_histograms
    return orblib


def test_aperture_groups_bound_the_prepare(monkeypatch):
    """S1 must actually feed GROUPS to each prepare call, not the whole
    set. Records the aperture count each _prepare call receives and asserts
    it is bounded by gram_aperture_group -- the property the whole change
    exists to deliver (a whole-set feed peaks at the full 124.8 GiB)."""
    seen = []
    real_prepare = NNLS._prepare_kinematic_block

    def spy(self, kins, orb_veldist, tmp, prj_mass_i, ap0=0):
        seen.append(orb_veldist.y.shape[-1])
        return real_prepare(self, kins, orb_veldist, tmp, prj_mass_i, ap0)

    fake = _make_fake_nnls(np.float64, stream_reads=True, aperture_group=1)
    fake._prepare_kinematic_block = _t.MethodType(spy, fake)
    NNLS.construct_gram_and_rhs_blockwise(fake, _make_fake_orblib())
    assert seen, "prepare spy never ran"
    assert max(seen) <= 1, f"groups not bounded: prepares fed {seen}"
    assert len(seen) == 2, f"expected one prepare per aperture group: {seen}"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_grouped_gram_agrees_with_full_set(dtype):
    """Grouping must not change the arithmetic: group sizes {off, 1, 2}
    agree to round-off. (Passes vacuously pre-S1 -- the bounds test above
    is the one that fails.)"""
    tol = 1e-12 if dtype == np.float64 else 1e-6
    ref = None
    for groups in (0, 1, 2):
        fake = _make_fake_nnls(dtype, stream_reads=True, aperture_group=groups)
        gp = NNLS.construct_gram_and_rhs_blockwise(fake, _make_fake_orblib())
        if ref is None:
            ref = gp
            continue
        for attr in ("P", "q", "v"):
            a, b = getattr(gp, attr), getattr(ref, attr)
            assert np.abs(a - b).max() / np.abs(b).max() < tol, (groups, attr)
        assert abs(gp.b_sq_rest - ref.b_sq_rest) <= (tol * abs(ref.b_sq_rest)), groups


def _read_groups(ap_2d, norb, n_ap_file, gap, pm_path, kin_idx_per_ap,
                 idx_ap_reset, hist_bins, rng):
    """Read the pm file in aperture groups, mirroring read_orbit_base's
    ap_group path; returns the concatenated group hists (norb,nv0,nv1,n_ap).
    Pins that the S1 read-side change scatters each group into group-sized
    velhist0, bitwise-equal to a full read."""
    lib = hbr._lib()
    nv0, nv1 = hist_bins[0]
    out = np.zeros((norb, nv0, nv1, n_ap_file))
    for g0 in range(0, n_ap_file, gap):
        g1 = min(g0 + gap, n_ap_file)
        local = ap_2d - idx_ap_reset[kin_idx_per_ap[ap_2d]]
        keep = (local >= g0) & (local < g1)
        grp = np.zeros((norb, nv0, nv1, g1 - g0))
        lib._read_pm_hist_vectorised(
            pm_path, norb, ap_2d, kin_idx_per_ap, idx_ap_reset, hist_bins,
            [grp], chunk=1000, keep=keep, ap_off=g0)
        out[..., g0:g1] = grp
    return out


def test_read_side_grouped_parser_equals_full():
    """S1 read-side RED: reading a real pm file in aperture groups (ap_off +
    keep) must be bitwise equal to a full read, wherever the group falls. A
    mis-mapped ap_off (wrong group-local column) shows up as the group
    scattered into the wrong apertures. LAYOUT with 2d only, 6 apertures,
    gap 2 -> 3 groups."""
    rng = np.random.default_rng(42)
    norb = 5
    kin_specs = [(2, 6, (3, 5))]  # one 2d set, 6 apertures
    with tempfile.TemporaryDirectory() as td:
        pm = os.path.join(td, "pm.dat")
        losvd = os.path.join(td, "losvd.dat")
        truth, kipa, reset, bins, dims = hbr.build(kin_specs, norb, rng,
                                                   losvd, pm)
        # full read (no ap_off) as the reference
        full = np.zeros((norb, bins[0][0], bins[0][1], 6))
        hbr.bulk(losvd, pm, norb, kipa, reset, bins, dims,
                 [full], chunk=1000)
        ap_all = np.arange(len(kipa))
        ap_2d = ap_all[np.asarray(dims)[kipa] == 2]
        for gap in (1, 2, 4):
            grouped = _read_groups(ap_2d, norb, 6, gap, pm, kipa, reset,
                                   bins, rng)
            np.testing.assert_array_equal(grouped, full)


def test_projected_mass_reduction_is_aperture_separable():
    """S1's premise, pinned: projected masses are a per-aperture reduction
    over the velocity axes, so group partials concatenated in aperture
    order equal the full-set reduction EXACTLY (same elements, same order
    -- no summation reorder). Group accumulation is therefore bookkeeping,
    not numerics. Passes pre- and post-S1; it locks the property the
    group path will rely on."""
    orblib = _make_fake_orblib()
    y = orblib.vel_histograms[0].y
    full = np.sum(y, axis=(1, 2))
    assert full.shape == (N_ORBS, 2)
    parts = [np.sum(y[..., a : a + 1], axis=(1, 2)) for a in range(y.shape[-1])]
    # allclose, not exact: numpy's pairwise summation trees differ between
    # a (..., n_ap) reduction and n_ap scalar reductions (measured 1.9e-16
    # relative). Irrelevant downstream (masses feed econ at ~1e-2), but it
    # caps S1's bitwise aspiration for the MASS rows -- the kinematic rows
    # are unaffected since rebin is exactly per-aperture.
    np.testing.assert_allclose(
        np.concatenate(parts, axis=1), full, rtol=1e-12, atol=0.0
    )


if __name__ == "__main__":
    test_aperture_groups_bound_the_prepare()
    print("aperture groups bound the prepare OK")
    test_grouped_gram_agrees_with_full_set(np.float64)
    test_grouped_gram_agrees_with_full_set(np.float32)
    print("grouped gram agrees OK")
    test_projected_mass_reduction_is_aperture_separable()
    print("projected masses separable OK")
    print("test_gram_aperture_groups OK")

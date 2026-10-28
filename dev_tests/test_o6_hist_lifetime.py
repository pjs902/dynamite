"""O6a/O6b: the orbit histogram must not survive into the dsyrk chunk loop,
and the reshape feeding that loop must not copy the whole set.

Background (production, hst_pm, 744303 rows): the Gram peak sat at 394 GiB =
hist 124.8 (rebin input) + rebinned 124.8 + a 124.8 full-array copy made by
``src = orb_kins.reshape(n_orbs, -1)`` -- ``transform_orblib_to_observables``
returns a non-C-contiguous ``moveaxis`` view, so the reshape cannot be a view.
The +124.8 jump lands between ``prepare_kinematic_block`` (ends 268.0) and the
first ``kin_block_astype`` (starts 392.8), before any chunk allocation -- the
O5 comment's "climbed +127.3 GiB across the loop" misattributes it.

O6a: release ``orblib.vel_histograms[si]`` right after the set's kinematic
block is prepared (it is dead input from then on), not after its chunk loop.
O6b: make ``orb_kins`` C-contiguous once per set (peak transient only) so the
per-chunk feed is a view, never a second full copy.

Run from the repo root:
    PYTHONPATH=. python -m pytest dev_tests/test_o6_hist_lifetime.py -v
"""

import logging
import types as _t

import numpy as np
import pytest

from dynamite import kinematics as dyn_kin
from dynamite.weight_solvers import NNLS, NormalEquationAccumulator

N_ORBS = 7


class FakeKins2D:
    """PM-like: 2 spatial bins, 16 constraints per aperture. Transform is a
    non-contiguous moveaxis view of the histogram buffer (shares memory --
    the no-rebin production path)."""

    kind = "pm"
    n_spatial_bins = 2
    constraint_shape = (16,)
    name = "fake_pm"

    def get_observed_values_and_uncertainties(self, settings):
        obs = np.arange(self.n_spatial_bins * 16, dtype=float).reshape(2, 16)
        err = np.abs(obs) * 0.02 + 0.02
        return obs, err

    def transform_orblib_to_observables(self, hist, settings):
        return np.moveaxis(hist.y[:, :4, :4, :], 3, 1)


def _make_fake_nnls(dtype, stream_reads=True, kins_cls=None):
    fake = _t.SimpleNamespace()
    fake.nnls_dtype = dtype
    fake.stream_reads = stream_reads
    fake.gram_chunk_rows = 3
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
    fake.logger = logging.getLogger("test_o6_hist_lifetime")
    # NOTE: the lambda builds the namespace fresh on EVERY call, so the
    # kins class must be chosen here -- replacing .kinematic_data on one
    # returned namespace is silently discarded by the next call.
    fake.system = _t.SimpleNamespace(
        get_unique_triaxial_visible_component=(
            lambda kins_cls=kins_cls or FakeKins2D: _t.SimpleNamespace(
                kinematic_data=[kins_cls()]
            )
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


def test_hist_released_before_kin_chunks():
    """O6a: every accumulator feed that happens after a set's kinematic
    block is prepared must see that set's histogram already released.

    Fails on the pre-O6 code: the histogram is only cleared after the whole
    chunk loop, so all 91 production chunks accumulate with both the 124.8
    GiB histogram AND the 124.8 GiB transformed block resident.
    """
    add_hist_state = []
    n_adds_at_prepare = []

    real_add = NormalEquationAccumulator.add
    real_prepare = NNLS._prepare_kinematic_block

    def spy_add(self, A_block, b_block):
        add_hist_state.append(orblib.vel_histograms[0] is None)
        return real_add(self, A_block, b_block)

    def spy_prepare(self, kins, orb_veldist, tmp, prj_mass_i):
        out = real_prepare(self, kins, orb_veldist, tmp, prj_mass_i)
        n_adds_at_prepare.append(len(add_hist_state))
        return out

    fake = _make_fake_nnls(np.float64, stream_reads=True)
    orblib = _make_fake_orblib()
    fake._prepare_kinematic_block = _t.MethodType(spy_prepare, fake)
    NormalEquationAccumulator.add = spy_add
    try:
        NNLS.construct_gram_and_rhs_blockwise(fake, orblib)
    finally:
        NormalEquationAccumulator.add = real_add
    assert n_adds_at_prepare, "prepare spy never ran"
    kin_adds = add_hist_state[n_adds_at_prepare[0] :]
    assert kin_adds, "no accumulator feeds after prepare"
    assert all(kin_adds), (
        f"O6a violated: histogram still resident for "
        f"{kin_adds.count(False)}/{len(kin_adds)} post-prepare feeds"
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_streamed_matches_nonstreamed(dtype):
    """Releasing the histogram early must not change the arithmetic:
    streamed (hist freed per set) and non-streamed (hist retained) builds
    must agree, in both dtypes. Tolerance is dtype-dependent: the streamed
    float32 path downcasts the histograms in place (pre-existing behavior),
    so f32 agreement is at f32-rounding level, not 1e-12."""
    tol = 1e-12 if dtype == np.float64 else 1e-6
    gp_stream, gp_keep = None, None
    for stream in (True, False):
        fake = _make_fake_nnls(dtype, stream_reads=stream)
        gp = NNLS.construct_gram_and_rhs_blockwise(fake, _make_fake_orblib())
        if stream:
            gp_stream = gp
        else:
            gp_keep = gp
    for attr in ("P", "q", "v"):
        a, b = getattr(gp_stream, attr), getattr(gp_keep, attr)
        assert np.abs(a - b).max() / np.abs(b).max() < tol, attr
    assert abs(gp_stream.b_sq_rest - gp_keep.b_sq_rest) <= (
        tol * abs(gp_keep.b_sq_rest)
    )


class FakeKinsRebinned(FakeKins2D):
    """PM-like, but the transform returns a FRESH non-contiguous array that
    shares nothing with the histogram -- the rebinned production path
    (rebin allocates zeros_like, then moveaxis views the result)."""

    def transform_orblib_to_observables(self, hist, settings):
        fresh = np.array(hist.y[:, :4, :4, :])
        assert not np.shares_memory(fresh, hist.y)
        return np.moveaxis(fresh, 3, 1)


def test_hist_buffer_dead_during_contiguity_copy(monkeypatch):
    """O6b ordering (regression test for the 392.7 GiB production miss):
    in the rebinned streamed case the histogram BUFFER must already be dead
    when ascontiguousarray runs. Clearing the orblib list entry is not
    enough -- the constructor's own orb_veldist local keeps the 124.8 GiB
    buffer alive through the copy unless it is dropped FIRST, making the
    copy a third full-set resident. Spies the copy and checks a weakref to
    hist.y at call time."""
    import weakref

    import dynamite.weight_solvers as ws

    real_contig = np.ascontiguousarray
    states = []

    def spy(a, *args, **kwargs):
        states.append(hist_wr() is None)
        return real_contig(a, *args, **kwargs)

    fake = _make_fake_nnls(np.float64, stream_reads=True, kins_cls=FakeKinsRebinned)
    orblib = _make_fake_orblib()
    hist_wr = weakref.ref(orblib.vel_histograms[0].y)
    monkeypatch.setattr(np, "ascontiguousarray", spy)
    NNLS.construct_gram_and_rhs_blockwise(fake, orblib)
    assert ws._contiguate_orb_kins_and_release_hist is not None
    assert states == [True], (
        f"histogram buffer alive during {len(states)} contiguity copies: "
        f"{states} -- the copy runs with 3 full-set residents"
    )


def test_contiguate_helper_rebinned_branch():
    """O6b unit: rebinned input (shares nothing with the histogram) --
    the histogram is released BEFORE the contiguity copy, and the result
    is C-contiguous and value-identical."""
    from dynamite.weight_solvers import _contiguate_orb_kins_and_release_hist

    rng = np.random.default_rng(21)
    hist_y = rng.random((6, 3, 4, 2)).astype(np.float32)
    rebinned = np.moveaxis(rng.random((6, 3, 4, 2)).astype(np.float32), 3, 1)
    assert not rebinned.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(rebinned, hist_y)
    orblib = _t.SimpleNamespace(vel_histograms=[_t.SimpleNamespace(y=hist_y)])
    out = _contiguate_orb_kins_and_release_hist(orblib, 0, rebinned, stream_reads=True)
    assert orblib.vel_histograms[0] is None
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, rebinned)


def test_contiguate_helper_shared_branch():
    """O6b unit: no-rebin input (views the histogram) -- the copy comes
    first (it needs the buffer), then the histogram is released."""
    from dynamite.weight_solvers import _contiguate_orb_kins_and_release_hist

    rng = np.random.default_rng(22)
    hist_y = rng.random((6, 3, 4, 2)).astype(np.float32)
    view = np.moveaxis(hist_y, 3, 1)
    assert np.shares_memory(view, hist_y)
    orblib = _t.SimpleNamespace(vel_histograms=[_t.SimpleNamespace(y=hist_y)])
    out = _contiguate_orb_kins_and_release_hist(orblib, 0, view, stream_reads=True)
    assert orblib.vel_histograms[0] is None
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, view)


def test_contiguate_helper_nonstreamed_keeps_hist():
    """Non-streamed mode leaves the orbit library intact for later reuse."""
    from dynamite.weight_solvers import _contiguate_orb_kins_and_release_hist

    rng = np.random.default_rng(23)
    hist_y = rng.random((6, 3, 4, 2)).astype(np.float32)
    view = np.moveaxis(hist_y, 3, 1)
    orblib = _t.SimpleNamespace(vel_histograms=[_t.SimpleNamespace(y=hist_y)])
    out = _contiguate_orb_kins_and_release_hist(orblib, 0, view, stream_reads=False)
    assert orblib.vel_histograms[0] is not None
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, view)


if __name__ == "__main__":
    test_hist_released_before_kin_chunks()
    print("hist released before kin chunks OK")
    test_streamed_matches_nonstreamed(np.float64)
    test_streamed_matches_nonstreamed(np.float32)
    print("streamed matches non-streamed OK")
    test_contiguate_helper_rebinned_branch()
    test_contiguate_helper_shared_branch()
    test_contiguate_helper_nonstreamed_keeps_hist()
    print("contiguate helper OK")
    print("test_o6_hist_lifetime OK")

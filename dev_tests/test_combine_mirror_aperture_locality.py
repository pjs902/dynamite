"""``combine_and_mirror_orblibs`` is per-aperture local, and its dtype/free
options do not change the values.

The streaming Gram build wants to materialise ONE APERTURE GROUP at a time
instead of the whole dense per-set array (124.8 GiB for omega Cen's hst_pm).
That is only valid if combining+mirroring a slice of apertures in isolation
gives exactly what slicing the fully combined array would give - i.e. the
operation never mixes data across the aperture axis. The docstring asserts
this implicitly by mirroring only the velocity axes, but nothing pinned it:
the old ``duplicate_flip_and_interlace_orblib`` + ``combine_orblibs`` pair
that it was "verified bit-exact" against has since been deleted, so that
claim is unverifiable and this file replaces it.

Aperture is the LAST axis in the histogram layout used here:
``(n_orb, nv, n_ap)`` for 1d and ``(n_orb, nvx, nvy, n_ap)`` for 2d, and the
mirror reverses only the velocity axes. If someone later reverses or reorders
the aperture axis, or makes the combine depend on neighbouring apertures,
test_slice_matches_full_* fails and the streaming design is invalidated -
which is the point.

Run from the repo root:
    PYTHONPATH=. python -m pytest dev_tests/test_combine_mirror_aperture_locality.py
"""

import logging
import types

import numpy as np
import pytest

from dynamite import kinematics as dyn_kin
from dynamite.orblib import LegacyOrbitLibrary


def _bare_lib():
    """A LegacyOrbitLibrary carrying only what combine_and_mirror_orblibs uses."""
    lib = LegacyOrbitLibrary.__new__(LegacyOrbitLibrary)
    lib.mod_dir = "fake/"
    lib.logger = logging.getLogger("test_combine_mirror_aperture_locality")
    lib.logger.addHandler(logging.NullHandler())
    return lib


def _pair(ndim, n_ap, norb_t=4, norb_b=3, seed=20260907):
    """A (tube, box) histogram pair with ``n_ap`` apertures.

    Velocity edges are symmetric because mirroring asserts that.
    """
    rng = np.random.default_rng(seed)
    if ndim == 3:
        edg = np.linspace(-3.0, 3.0, 7)  # 6 velocity bins
        cls = dyn_kin.Histogram
        shape_t, shape_b = (norb_t, 6, n_ap), (norb_b, 6, n_ap)
        xedg = edg
    else:
        edg0 = np.linspace(-2.0, 2.0, 6)  # 5 bins
        edg1 = np.linspace(-2.5, 2.5, 6)  # 5 bins
        cls = dyn_kin.Histogram2D
        shape_t, shape_b = (norb_t, 5, 5, n_ap), (norb_b, 5, 5, n_ap)
        xedg = (edg0, edg1)
    return (cls(xedg=xedg, y=rng.random(shape_t)),
            cls(xedg=xedg, y=rng.random(shape_b)))


def _slice_apertures(hist, sl):
    """A copy of ``hist`` holding only the apertures in ``sl`` (last axis)."""
    cls = type(hist)
    return cls(xedg=hist.xedg, y=np.ascontiguousarray(hist.y[..., sl]))


@pytest.mark.parametrize("ndim", [3, 4], ids=["losvd_1d", "pm_2d"])
@pytest.mark.parametrize("mirror", [True, False], ids=["mirror", "no_mirror"])
def test_slice_matches_full(ndim, mirror):
    """Combining an aperture GROUP alone == combining everything, then slicing.

    This is the whole correctness argument for building the Gram matrix one
    aperture group at a time. Exact equality, not approximate: the operation
    is pure data movement, so any difference means apertures are being mixed.
    """
    n_ap = 7
    lib = _bare_lib()

    tube, box = _pair(ndim, n_ap)
    full = lib.combine_and_mirror_orblibs(tube, box, mirror=mirror)

    # every contiguous group, including single apertures and the whole range,
    # plus a group that starts and ends off the boundaries
    for start, stop in [(0, 1), (0, 3), (2, 5), (3, 7), (6, 7), (0, n_ap)]:
        sl = slice(start, stop)
        tube_g, box_g = _pair(ndim, n_ap)  # same seed -> same data
        part = lib.combine_and_mirror_orblibs(
            _slice_apertures(tube_g, sl), _slice_apertures(box_g, sl),
            mirror=mirror)
        assert np.array_equal(part.y, full.y[..., sl]), (
            f"aperture group [{start}:{stop}] differs from the full "
            f"combine sliced the same way (ndim={ndim}, mirror={mirror}) - "
            "combine_and_mirror_orblibs is NOT per-aperture local, so the "
            "streaming Gram build is invalid"
        )


@pytest.mark.parametrize("ndim", [3, 4], ids=["losvd_1d", "pm_2d"])
def test_non_contiguous_aperture_group(ndim):
    """Locality holds for an arbitrary aperture subset, not just a slice.

    An aperture group need not be contiguous once ``keep`` masks are in play,
    so pin the general case too.
    """
    n_ap = 7
    pick = [5, 0, 3]  # deliberately unordered and non-contiguous
    lib = _bare_lib()

    tube, box = _pair(ndim, n_ap)
    full = lib.combine_and_mirror_orblibs(tube, box, mirror=True)

    tube_g, box_g = _pair(ndim, n_ap)
    part = lib.combine_and_mirror_orblibs(
        _slice_apertures(tube_g, pick), _slice_apertures(box_g, pick),
        mirror=True)
    assert np.array_equal(part.y, full.y[..., pick])


@pytest.mark.parametrize("ndim", [3, 4], ids=["losvd_1d", "pm_2d"])
def test_dtype_matches_combine_then_downcast(ndim):
    """``dtype=float32`` == combining in float64 and downcasting afterwards.

    This is what makes the float32 allocation a memory change and not a
    numerical one. It holds ONLY because nothing arithmetic happens between
    the two: a float64 destined for float32 rounds the same either way. The
    moment a caller sums these values before downcasting (as the fused adelie
    constructor does when it rebins), the equality no longer applies - which
    is why ``dtype=`` is threaded through the Gram path only.
    """
    lib = _bare_lib()
    tube, box = _pair(ndim, 4)
    direct = lib.combine_and_mirror_orblibs(tube, box, mirror=True,
                                            dtype=np.float32)
    tube2, box2 = _pair(ndim, 4)
    after = lib.combine_and_mirror_orblibs(tube2, box2, mirror=True)

    assert direct.y.dtype == np.float32
    assert after.y.dtype == np.float64
    assert np.array_equal(direct.y, after.y.astype(np.float32))


@pytest.mark.parametrize("ndim", [3, 4], ids=["losvd_1d", "pm_2d"])
def test_free_inputs_releases_sources_without_changing_result(ndim):
    """``free_inputs=True`` drops tube.y/box.y and returns the same values.

    Releasing the sources is what stops tube, box and the combined array from
    all being resident at once. It also means anything read from ``tube.y``
    AFTER the copy is a use-after-free: that bug shipped once already (the
    class of the returned Histogram was chosen from ``tube.y.ndim`` below the
    release point) and surfaced only as a nan model, so pin it here.
    """
    lib = _bare_lib()
    tube, box = _pair(ndim, 4)
    kept = lib.combine_and_mirror_orblibs(tube, box, mirror=True)

    tube2, box2 = _pair(ndim, 4)
    freed = lib.combine_and_mirror_orblibs(tube2, box2, mirror=True,
                                           free_inputs=True)

    assert tube2.y is None, "free_inputs=True must release tube.y"
    assert box2.y is None, "free_inputs=True must release box.y"
    assert type(freed) is type(kept), (
        "returned histogram class must not depend on data released by "
        "free_inputs"
    )
    assert np.array_equal(freed.y, kept.y)
    assert freed.y.shape == kept.y.shape


@pytest.mark.parametrize("ndim", [3, 4], ids=["losvd_1d", "pm_2d"])
def test_mirror_reverses_velocity_axes_only(ndim):
    """The mirrored copy is the original with the VELOCITY axes reversed.

    Stated directly rather than inferred, so that reversing the aperture axis
    by mistake fails loudly here instead of quietly biasing a fit.
    """
    lib = _bare_lib()
    n_ap, norb_t = 4, 4
    tube, box = _pair(ndim, n_ap, norb_t=norb_t)
    out = lib.combine_and_mirror_orblibs(tube, box, mirror=True)

    unflipped = out.y[0 : 2 * norb_t : 2]
    flipped = out.y[1 : 2 * norb_t : 2]
    tube_ref, _ = _pair(ndim, n_ap, norb_t=norb_t)
    assert np.array_equal(unflipped, tube_ref.y)
    if ndim == 3:
        assert np.array_equal(flipped, tube_ref.y[:, ::-1, :])
    else:
        assert np.array_equal(flipped, tube_ref.y[:, ::-1, ::-1, :])
    # box orbits are appended unmirrored
    assert np.array_equal(out.y[2 * norb_t :], box.y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

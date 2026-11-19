"""Option B of the Laplace speedup: no separate capture read, cell-space final chi2.

The re-expansion point of aperture a, ``mix(w1 @ H_a) M_a``, depends only on that
aperture, so the pass-2 build computes it from the H_a it already holds
(``NNLS._lap_reexpand``) instead of a separate rhs-only capture read. The final chi2
read needs only ``Q(h_a) = |Lt (h_a - h0)|^2 + 2 g'(h_a - h0)``, not the rows ``H Lt'``.

Run from the repo root:
    PYTHONPATH=.:dev_tests pytest dev_tests/test_laplace_fused.py
"""
import numpy as np
import pytest

from dynamite import weight_solvers as ws
from dynamite.weight_solvers import NNLS, laplace_reexpand_mix

from test_laplace_pm import _lap_at_data_h0, _toy_lap


def _two_read_reference(fake, lib, w1, floor):
    """The pre-B path: rhs-only capture read at w1, then a full build about the mixed model."""
    fake._lap_capture = dict(w=w1, h={})
    fake.construct_gram_and_rhs_blockwise(lib, rhs_only=True)
    h = fake._lap_capture["h"]["hst"]
    fake._lap_capture = None
    fake._lap_h0 = {"hst": laplace_reexpand_mix(h, floor) * fake.projected_masses[2:, None]}
    gp = fake.construct_gram_and_rhs_blockwise(lib)
    h0 = fake._lap_h0["hst"]
    fake._lap_h0 = None
    return gp, h0


@pytest.mark.parametrize("stream,gap", [(False, 0), (True, 2)])
def test_fused_build_equals_capture_read_then_build(tmp_path, stream, gap):
    fake, lib, pm, w_true = _toy_lap(tmp_path, 1)
    fake.stream_reads, fake.gram_aperture_group = stream, gap  # 3 apertures in groups of 2
    lib.vel_histograms = list(lib.full)
    w1 = w_true * np.random.default_rng(3).uniform(0.5, 1.5, w_true.size)
    ref, h0_ref = _two_read_reference(fake, lib, w1, 1e-3)
    fake._lap_reexpand = dict(w=w1, floor=1e-3, h0={})
    gp = fake.construct_gram_and_rhs_blockwise(lib)
    h0 = fake._lap_reexpand["h0"]["hst"]
    fake._lap_reexpand = None
    np.testing.assert_array_equal(h0, h0_ref)
    np.testing.assert_allclose(gp.G, ref.G, rtol=1e-12, atol=1e-12 * np.abs(ref.G).max())
    np.testing.assert_allclose(gp.v, ref.v, rtol=1e-12, atol=1e-12 * np.abs(ref.v).max())


@pytest.mark.parametrize("reexpand", [1, 2])
def test_driver_reads_the_library_once_per_pass_plus_the_chi2_read(tmp_path, monkeypatch, reexpand):
    fake, lib, pm, _ = _toy_lap(tmp_path, reexpand)
    lib.vel_histograms = list(lib.full)
    calls = []
    orig = NNLS.construct_gram_and_rhs_blockwise

    def spy(self, orblib, **kw):
        calls.append(kw)
        return orig(self, orblib, **kw)
    fake.construct_gram_and_rhs_blockwise = spy.__get__(fake)
    w, *_ = fake._solve_laplace(lib)
    assert len(calls) == reexpand + 2  # one build per pass, one chi2 read
    assert fake._lap_reexpand is None  # driver state cleared
    # pass k's expansion point is the mixed model of pass k-1
    H = pm.transform_orblib_to_observables(lib.full[1]).reshape(w.size, 3, -1)
    prj = fake.projected_masses[2:]
    for k in range(reexpand):
        wp, h0 = fake.laplace_info["w_pass"][k], fake.laplace_info["h0_pass"][k]["hst"]
        for a in range(3):
            h = H[:, a].T @ wp
            np.testing.assert_allclose(h0[a], (0.999 * h / h.sum() + 0.001 / h.size) * prj[a], rtol=1e-10)


def test_chi2_read_forms_no_laplace_rows(tmp_path, monkeypatch):
    """The final read's Laplace-set blocks reach the accumulator as zero rows (residual in b)."""
    fake, lib, pm, _ = _toy_lap(tmp_path, 0)
    lib.vel_histograms = list(lib.full)
    seen = []
    orig = ws.NormalEquationAccumulator.add

    def spy(self, A, b):
        if self.chi2_w is not None and self.tag == "set1/":
            seen.append((np.shape(A)[0], bool(np.asarray(A).any())))
        return orig(self, A, b)
    monkeypatch.setattr(ws.NormalEquationAccumulator, "add", spy)
    w, _, _, chi2 = fake._solve_laplace(lib)
    n_ap = pm.n_spatial_bins  # the set's first block is its projected-mass rows
    assert seen[0] == (n_ap, True) and len(seen) > 1 and not any(z for _, z in seen[1:])
    ref = _lap_at_data_h0(pm, lib, fake.projected_masses[2:], w)
    assert {c[0]: c[1] for c in chi2}["hst"] == pytest.approx(ref, rel=1e-10, abs=1e-8)


def test_chi2_read_with_a_gradient_keeps_the_laplace_gradient(tmp_path):
    """A chi2 read that also asks for g = A'(A w - b) + c (vp reads) must not lose the Laplace rows' part."""
    fake, lib, pm, w_true = _toy_lap(tmp_path, 0)
    lib.vel_histograms = list(lib.full)
    gp = fake.construct_gram_and_rhs_blockwise(lib)
    w = w_true * 1.1
    fake._vp_grad_w = w
    fake.construct_gram_and_rhs_blockwise(lib, rhs_only=True, chi2_w=w)
    g, fake._vp_grad_w, fake._vp_acc = fake._vp_acc.g, None, None
    want = gp.G @ w - gp.v
    np.testing.assert_allclose(g, want, rtol=1e-10, atol=1e-10 * np.abs(want).max())

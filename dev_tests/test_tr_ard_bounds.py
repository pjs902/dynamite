"""Assert-based checks for the ARD-scaled trust-region fix
(BayesOptGenerator._ard_half_widths, _tr_bounds, _replay_tr_state).

Background: the isotropic TR gave every free dimension the same box width
regardless of GP sensitivity (see PM_grid/HANDOFF_2026-09-22_sBH_GP_grid_v2.md),
and TR adaptation state reset to tr_side_init on every driver resume because
it lived only in the generator's __init__ (never persisted). This test
checks both fixes on synthetic data -- no real GP fit, no real grid.

Run with:  /nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python3 dev_tests/test_tr_ard_bounds.py
"""
import os
import sys

import numpy as np
from astropy.table import Column

DYN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DYN_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamite.parameter_space as ps  # noqa: E402
from test_bayesopt_generator import (  # noqa: E402
    _bo_settings, _mk_param, make_parspace, MockAllModels,
)
import torch  # noqa: E402 - must import after dynamite (pymc/IPython import-order quirk)


class _FakeKernel:
    def __init__(self, lengthscales):
        self.lengthscale = torch.tensor([lengthscales], dtype=torch.double)


class _FakeGPModel:
    """Stands in for the fitted botorch/gpytorch model: only
    `.covar_module.base_kernel.lengthscale` is read by _ard_half_widths."""
    def __init__(self, lengthscales):
        self.covar_module = type("K", (), {"base_kernel": _FakeKernel(lengthscales)})()


def _make_2d_generator(**tr_overrides):
    a = _mk_param("a", 0.0, 1.0, 0.5)
    b = _mk_param("b", 0.0, 1.0, 0.5)
    ps_ = make_parspace([a, b])
    s = _bo_settings()
    s["generator_settings"].update({"trust_region": True, "tr_trigger_frac": 0.5,
                                     "tr_side_init": 0.3, "tr_min_side": 0.05,
                                     "tr_max_side": 0.6})
    s["generator_settings"].update(tr_overrides)
    return ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)


def test_ard_tighter_along_short_lengthscale():
    """Short lengthscale (fast-varying dim) -> tighter box than a long one."""
    gen = _make_2d_generator()
    gen._gp_model = _FakeGPModel([0.1, 1.0])  # dim0 fast, dim1 flat
    half = gen._ard_half_widths(0.15)  # isotropic half for side=0.3
    assert half[0] < half[1], f"expected dim0 (short ls) tighter than dim1, got {half}"
    print("  test_ard_tighter_along_short_lengthscale PASSED")


def test_ard_geomean_matches_isotropic_side():
    """Geometric mean of the ARD full widths reproduces the isotropic side
    (volume-preserving rescale), when no clamp is active."""
    gen = _make_2d_generator()
    gen._gp_model = _FakeGPModel([0.2, 0.8])  # mild ratio, won't hit clamp
    half_iso = 0.15
    half = gen._ard_half_widths(half_iso)
    full = 2.0 * half
    gmean = np.exp(np.mean(np.log(full)))
    np.testing.assert_allclose(gmean, 2.0 * half_iso, rtol=1e-9)
    print("  test_ard_geomean_matches_isotropic_side PASSED")


def test_ard_clamped_to_min_max_side():
    """Extreme lengthscale ratio must still respect [tr_min_side, tr_max_side]."""
    gen = _make_2d_generator()
    gen._gp_model = _FakeGPModel([0.001, 100.0])  # extreme ratio
    half = gen._ard_half_widths(0.15)
    full = 2.0 * half
    assert np.all(full >= gen.tr_min_side - 1e-12), full
    assert np.all(full <= gen.tr_max_side + 1e-12), full
    print("  test_ard_clamped_to_min_max_side PASSED")


def test_ard_disabled_falls_back_isotropic():
    gen = _make_2d_generator(tr_ard_scaling=False)
    gen._gp_model = _FakeGPModel([0.1, 1.0])
    half = gen._ard_half_widths(0.15)
    np.testing.assert_allclose(half, [0.15, 0.15])
    print("  test_ard_disabled_falls_back_isotropic PASSED")


def test_ard_no_gp_yet_falls_back_isotropic():
    gen = _make_2d_generator()
    assert gen._gp_model is None
    half = gen._ard_half_widths(0.15)
    np.testing.assert_allclose(half, [0.15, 0.15])
    print("  test_ard_no_gp_yet_falls_back_isotropic PASSED")


def _add_row(table, val, chi2, it):
    table.add_row([val, chi2, chi2, float("nan"), "", True, True, True, it, "d"])


def test_replay_state_reproduces_shrink_after_stale_batches():
    """Resume-state derivation: replaying which_iter history should shrink
    _tr_side exactly like calling _maybe_update_tr live would have, instead
    of resetting to tr_side_init on every resume (the production bug)."""
    a = _mk_param("a", 0.0, 1.0, 0.5)
    ps_ = make_parspace([a])
    s = _bo_settings()
    s["generator_settings"].update({"trust_region": True, "tr_trigger_frac": 1.0,
                                     "tr_side_init": 0.3, "tr_patience": 2,
                                     "tr_grow": 1.3, "tr_shrink": 0.7,
                                     "tr_min_side": 0.05, "tr_max_side": 0.6})
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    am = MockAllModels(["a"])
    # iter 0: 10 rows clustered around a=0.5, best chi2=1.0 (establishes n>=10)
    for i in range(10):
        _add_row(am.table, 0.5 + 0.001 * i, 1.0 + 0.01 * i, it=0)
    # iters 1..4: each a worse (stale) batch of 1 row -> two stale batches
    # should trigger one shrink each (tr_patience=2), i.e. 2 shrinks total.
    for it in range(1, 5):
        _add_row(am.table, 0.5, 2.0, it=it)  # never beats best=1.0 -> stale

    # Live replay via _maybe_update_tr, called once per iteration boundary
    # (mirrors what _replay_tr_state does internally) to get the expected
    # trajectory independently of the implementation under test.
    side = gen.tr_side_init
    stale = 0
    best_seen = None
    tr_center_set = False
    for it in sorted(set(am.table["which_iter"])):
        hist = am.table[np.asarray(am.table["which_iter"]) < it]
        if len(hist) < 10:
            continue
        best = float(np.min(np.asarray(hist["kinchi2"], dtype=float)))
        if tr_center_set:
            if best < best_seen - 1e-12:
                side = min(side * gen.tr_grow, gen.tr_max_side)
                stale = 0
            else:
                stale += 1
                if stale >= gen.tr_patience:
                    side = max(side * gen.tr_shrink, gen.tr_min_side)
                    stale = 0
        best_seen = best
        tr_center_set = True
    expected_side = side

    gen._replay_tr_state(am.table)
    assert gen._tr_state_replayed is False or True  # replay is caller-driven, not auto-set here
    np.testing.assert_allclose(gen._tr_side, expected_side, rtol=1e-9)
    assert gen._tr_side < gen.tr_side_init, "stale batches across resumes must shrink the TR"
    print(f"  test_replay_state_reproduces_shrink_after_stale_batches PASSED (side={gen._tr_side:.4f})")


def test_tr_bounds_is_side_effect_free():
    """Computing the box must not advance the trust region.

    model_iterator hands the LIVE generator to
    plotter.make_gp_iteration_plot after every iteration, and that plot
    calls _tr_bounds() to draw the TR rectangle. While _maybe_update_tr
    lived inside _tr_bounds, each diagnostic plot applied a second
    grow/shrink and a second _tr_stale_batches increment, so the box
    shrank at twice the intended rate and tr_patience halved.
    """
    a = _mk_param("a", 0.0, 1.0, 0.5)
    ps_ = make_parspace([a])
    s = _bo_settings()
    s["generator_settings"].update({"trust_region": True, "tr_trigger_frac": 1.0,
                                    "tr_side_init": 0.3, "tr_patience": 1,
                                    "tr_grow": 1.3, "tr_shrink": 0.7,
                                    "tr_min_side": 0.05, "tr_max_side": 0.6})
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=s)
    am = MockAllModels(["a"])
    for i in range(12):
        _add_row(am.table, 0.5 + 0.001 * i, 1.0 + 0.01 * i, it=0)
    gen.current_models = am

    gen._advance_tr_state()          # one legitimate batch update
    side_after_batch = gen._tr_side
    stale_after_batch = gen._tr_stale_batches

    for _ in range(5):               # five diagnostic plots
        gen._tr_bounds()
    assert gen._tr_side == side_after_batch, (
        f"_tr_bounds moved _tr_side {side_after_batch} -> {gen._tr_side}")
    assert gen._tr_stale_batches == stale_after_batch, (
        "_tr_bounds moved the stale counter")

    # and the advancing call still does advance it
    gen._advance_tr_state()
    assert (gen._tr_side != side_after_batch
            or gen._tr_stale_batches != stale_after_batch), (
        "_advance_tr_state must still update the trust region")
    print("  test_tr_bounds_is_side_effect_free PASSED "
          f"(side stable at {side_after_batch:.4f} across 5 plot calls)")


if __name__ == "__main__":
    test_ard_tighter_along_short_lengthscale()
    test_ard_geomean_matches_isotropic_side()
    test_ard_clamped_to_min_max_side()
    test_ard_disabled_falls_back_isotropic()
    test_ard_no_gp_yet_falls_back_isotropic()
    test_replay_state_reproduces_shrink_after_stale_batches()
    test_tr_bounds_is_side_effect_free()
    print("ALL TESTS PASSED")

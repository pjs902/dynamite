"""Regression checks for the code-review fixes in BayesOptGenerator.

Kept separate from test_bayesopt_generator.py so these can run without
fitting a GP.

Run with:  /opt/miniconda3/envs/main/bin/python3 dev_tests/test_bayesopt_review_fixes.py
"""

import numpy as np

from test_bayesopt_generator import (  # noqa: E402
    ps,
    make_parspace,
    ParameterSpace,
    MockSystem,
    MockTriaxialComponent,
    _mk_param,
    _mk_param_with_step,
    _bo_settings,
    _bo_settings_discrete,
)

QOBS = 0.65


def _gen(step_q=0.25):
    ml = _mk_param_with_step("ml", 4.0, 6.0, step=0.5, value=5.0)
    q = _mk_param_with_step("q-stars", 0.0, 1.0, step=step_q, value=0.5)
    return ps.BayesOptGenerator(par_space=make_parspace([q, ml]), parspace_settings=_bo_settings_discrete())


def _triaxial_gen():
    """q, p, u all free, so the feasibility projection is actually live."""
    q_p = _mk_param("q", 0.05, 0.99, 0.6)
    p_p = _mk_param("p", 0.05, 0.999, 0.8)
    u_p = _mk_param("u", 0.05, 1.0, 0.9)
    tri = MockTriaxialComponent("stars", qobs=QOBS)
    tri.parameters = [q_p, p_p, u_p]
    ps_ = ParameterSpace(MockSystem([q_p, p_p, u_p], components=[tri]))
    gen = ps.BayesOptGenerator(par_space=ps_, parspace_settings=_bo_settings())
    assert gen.qobs == QOBS
    return gen


def _raw(gen, X_unit):
    lo_raw, hi_raw = gen._norm_bounds_arrays()
    return X_unit * (hi_raw - lo_raw) + lo_raw


def _assert_triaxial(gen, X_unit, what):
    """p >= q and max(q/qobs, p) <= u <= min(p/qobs, 1) -- the conditions
    TriaxialVisibleComponent.validate_parset enforces before a model runs."""
    raw = _raw(gen, X_unit)
    jq, jp, ju = (gen._free_qpu_idx[k] for k in ("q", "p", "u"))
    q, p, u = raw[:, jq], raw[:, jp], raw[:, ju]
    tol = 1e-6
    bad_p = np.sum(p < q - tol)
    bad_u = np.sum((u < np.maximum(q / QOBS, p) - tol) | (u > np.minimum(p / QOBS, 1.0) + tol))
    assert bad_p == 0, f"{what}: {bad_p}/{len(q)} draws violate p >= q"
    assert bad_u == 0, f"{what}: {bad_u}/{len(q)} draws fall outside the u window"


def test_knn_radius_uses_training_incumbent():
    """The incumbent is indexed in X_norm's own (filtered) row space."""
    gen = _gen()
    # X_norm/y as extract_gp_training_data returns them: only valid rows.
    X = np.array([[0.9, 0.9], [0.8, 0.8], [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    y = np.array([50.0, 40.0, 1.0, 10.0, 20.0, 30.0])  # incumbent at row 2
    r = gen._knn_radius(X, y)
    # distances from X[2] to the other five, scaled by sqrt(d) which cancels
    expect = np.mean([0.1, 0.2, 0.3, 0.8, 0.9])
    np.testing.assert_allclose(r, expect, rtol=1e-9)
    print("  test_knn_radius_uses_training_incumbent PASSED")


def test_feasible_ic_generator_no_feasible_point():
    """All-infeasible constraints yield valid-shaped ICs, not an IndexError."""
    import torch

    gen = _gen()
    gen._make_triaxiality_constraints = lambda: ([(lambda x: torch.tensor(-1.0), True)], None)
    bounds = torch.stack([torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)])
    ic = gen._feasible_ic_generator(None, bounds, q=2, num_restarts=3, raw_samples=8)
    assert ic.shape == (3, 2, 2), ic.shape
    assert torch.isfinite(ic).all()
    print("  test_feasible_ic_generator_no_feasible_point PASSED")


def test_sobol_in_box_stays_feasible():
    """Confining draws to a box must not undo the triaxiality projection.

    Clipping a projected point back into the box re-breaks p >= q, and
    validate_parset then drops the model silently.
    """
    gen = _triaxial_gen()
    box = np.stack([np.array([0.55, 0.30, 0.60]), np.array([0.85, 0.60, 0.90])])
    _assert_triaxial(gen, gen._sobol_in_box(64, box), "boxed draws")
    _assert_triaxial(gen, gen._sobol_in_box(64), "full-cube draws")
    print("  test_sobol_in_box_stays_feasible PASSED")


def test_dedup_and_fill_returns_full_batch():
    """A box narrower than one grid cell must not starve the batch.

    A short batch reads downstream as "no new models" and stops the run.
    """
    gen = _gen(step_q=0.25)
    box = np.stack([np.array([0.4, 0.4]), np.array([0.6, 0.6])])
    X = np.array([[0.5, 0.41], [0.5, 0.45], [0.5, 0.50], [0.5, 0.55]])  # one q cell
    out = gen._dedup_and_fill(X, box)
    assert out.shape[0] == X.shape[0], f"batch starved: {out.shape[0]} of {X.shape[0]}"
    print("  test_dedup_and_fill_returns_full_batch PASSED")


def test_snap_to_grid_reaches_box_edges():
    """The grid points ON the box edges must be reachable.

    0.6/0.1 == 5.999..., so a bare floor() puts the upper edge a full step
    out of reach and pulls edge candidates inward.
    """
    gen = _gen(step_q=0.1)
    box = np.stack([np.array([0.4, 0.4]), np.array([0.6, 0.6])])
    snapped = gen._snap_to_grid(np.array([[0.60, 0.5], [0.40, 0.5], [0.58, 0.5]]), box)
    np.testing.assert_allclose(snapped[0, 0], 0.6, atol=1e-9)
    np.testing.assert_allclose(snapped[1, 0], 0.4, atol=1e-9)
    np.testing.assert_allclose(snapped[2, 0], 0.6, atol=1e-9)
    print("  test_snap_to_grid_reaches_box_edges PASSED")


def test_snap_to_grid_substep_box_stays_on_grid():
    """A box narrower than one step must still yield an ON-GRID value.

    Falling back to the raw box edge would silently forfeit the orblib
    reuse that discretize_non_ml_params exists to provide.
    """
    gen = _gen(step_q=0.25)
    box = np.stack([np.array([0.55, 0.4]), np.array([0.70, 0.6])])  # holds no 0.25 multiple
    snapped = gen._snap_to_grid(np.array([[0.56, 0.5], [0.69, 0.5]]), box)
    for v in snapped[:, 0]:
        np.testing.assert_allclose(v / 0.25, round(v / 0.25), atol=1e-9, err_msg=f"{v} is off-grid")
    print("  test_snap_to_grid_substep_box_stays_on_grid PASSED")


def test_candidates_stay_in_box():
    """Snapping and Sobol refills respect the box they are given."""
    gen = _gen(step_q=0.1)
    lo, hi = np.array([0.4, 0.4]), np.array([0.6, 0.6])
    box = np.stack([lo, hi])

    # column 0 is q-stars (gridded); ml has no step and is left continuous by
    # _snap_to_grid -- optimize_acqf already returns it inside `bounds`.
    snapped = gen._snap_to_grid(np.array([[0.05, 0.05], [0.95, 0.95], [0.52, 0.52]]), box)
    assert np.all(snapped[:, 0] >= lo[0] - 1e-9) and np.all(snapped[:, 0] <= hi[0] + 1e-9), snapped

    draws = gen._sobol_in_box(32, box)
    assert np.all(draws >= lo - 1e-9) and np.all(draws <= hi + 1e-9), draws

    full = gen._sobol_in_box(16)  # no box -> full cube
    assert np.all(full >= 0.0) and np.all(full <= 1.0)
    print("  test_candidates_stay_in_box PASSED")


if __name__ == "__main__":
    test_knn_radius_uses_training_incumbent()
    test_feasible_ic_generator_no_feasible_point()
    test_sobol_in_box_stays_feasible()
    test_dedup_and_fill_returns_full_batch()
    test_snap_to_grid_reaches_box_edges()
    test_snap_to_grid_substep_box_stays_on_grid()
    test_candidates_stay_in_box()
    print("ALL REVIEW-FIX TESTS PASSED")

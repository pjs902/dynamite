"""Regression checks for the three code-review fixes in BayesOptGenerator.

Kept separate from test_bayesopt_generator.py because that script segfaults
on this machine inside the torch GP-fit tests (duplicate libomp); these
checks avoid fitting a GP.

Run with:  python dev_tests/test_bayesopt_review_fixes.py
"""

import numpy as np

from test_bayesopt_generator import (  # noqa: E402
    ps,
    make_parspace,
    _mk_param_with_step,
    _bo_settings_discrete,
)


def _gen(step_q=0.25):
    ml = _mk_param_with_step("ml", 4.0, 6.0, step=0.5, value=5.0)
    q = _mk_param_with_step("q-stars", 0.0, 1.0, step=step_q, value=0.5)
    return ps.BayesOptGenerator(par_space=make_parspace([q, ml]), parspace_settings=_bo_settings_discrete())


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


def test_candidates_stay_in_trust_region():
    """Snapping and Sobol refills respect the acquisition box they are given."""
    gen = _gen(step_q=0.1)
    lo = np.array([0.4, 0.4])
    hi = np.array([0.6, 0.6])
    box = np.stack([lo, hi])

    # column 0 is q-stars (gridded); ml has no step and is left continuous by
    # _snap_to_grid -- optimize_acqf already returns it inside `bounds`.
    snapped = gen._snap_to_grid(np.array([[0.05, 0.05], [0.95, 0.95], [0.52, 0.52]]), box)
    assert np.all(snapped[:, 0] >= lo[0] - 1e-9) and np.all(snapped[:, 0] <= hi[0] + 1e-9), snapped

    draws = gen._sobol_in_box(32, box)
    assert np.all(draws >= lo - 1e-9) and np.all(draws <= hi + 1e-9), draws

    full = gen._sobol_in_box(16)  # no box -> full cube
    assert np.all(full >= 0.0) and np.all(full <= 1.0)
    print("  test_candidates_stay_in_trust_region PASSED")


if __name__ == "__main__":
    test_knn_radius_uses_training_incumbent()
    test_feasible_ic_generator_no_feasible_point()
    test_candidates_stay_in_trust_region()
    print("ALL REVIEW-FIX TESTS PASSED")

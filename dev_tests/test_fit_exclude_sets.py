"""``fit_exclude_sets``: fit without some kinematic sets (e.g. LOS-only, then predict the proper motions).

An excluded set is not read and adds no kinematic and no projected-mass rows, so the Gram problem is the one
of a solver that never had the set. Run from the repo root:
    PYTHONPATH=.:dev_tests pytest dev_tests/test_fit_exclude_sets.py
"""
import types

import numpy as np
import pytest

from test_laplace_pm import _toy_lap


def test_excluded_set_gives_the_problem_without_it(tmp_path):
    fake, lib, pm, _ = _toy_lap(tmp_path, 0)
    lib.vel_histograms = list(lib.full)
    fake.settings["fit_exclude_sets"] = ["hst"]
    gp = fake.construct_gram_and_rhs_blockwise(lib)
    ref, lib2, _, _ = _toy_lap(tmp_path, 0)
    lib2.vel_histograms = list(lib2.full)
    los = ref.system.get_unique_triaxial_visible_component().kinematic_data[0]
    ref.system = types.SimpleNamespace(get_unique_triaxial_visible_component=lambda: types.SimpleNamespace(
        kinematic_data=[los]))
    n_los = los.n_spatial_bins
    ref.n_apertures, ref.n_mass_constraints = n_los, 1 + ref.n_intrinsic + n_los
    ref.projected_masses = ref.projected_masses[:n_los]
    gr = ref.construct_gram_and_rhs_blockwise(lib2)
    np.testing.assert_allclose(gp.G, gr.G, rtol=1e-13, atol=1e-13 * np.abs(gr.G).max())
    np.testing.assert_allclose(gp.v, gr.v, rtol=1e-13, atol=1e-13 * np.abs(gr.v).max())
    assert gp.b_sq_rest == pytest.approx(gr.b_sq_rest, rel=1e-13)


def test_laplace_driver_with_the_pm_set_excluded(tmp_path):
    fake, lib, pm, _ = _toy_lap(tmp_path, 1)
    lib.vel_histograms = list(lib.full)
    fake.settings["fit_exclude_sets"] = ["hst"]
    w, chi2_tot, chi2_kin, chi2 = fake._solve_laplace(lib)
    by = {c[0]: c[1] for c in chi2}
    assert by["hst"] == 0.0 and np.isfinite(by["mass"]) and by["mass"] >= 0
    assert w.sum() == pytest.approx(fake.total_mass, rel=1e-3)

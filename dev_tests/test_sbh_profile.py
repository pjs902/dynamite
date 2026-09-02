#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone checks for the sBH (Zhao alpha-beta-gamma) component.

Run directly:  /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py
Exits nonzero on the first failure.
"""
import sys
import numpy as np
from scipy.integrate import quad

import dynamite as dyn
from dynamite.physical_system import StellarBlackHoles as SBH

# (alpha, beta, gamma) spanning both external models, incl. gamma > 2
CASES = [
    (2.15, 12.0, 1.74),   # PhaseFlow fiducial fit
    (3.91, 4.50, 2.24),   # LIMEPY 0.05-10 pc fit  (gamma > 2)
    (2.00, 4.00, 2.18),   # gamma > 2
    (0.31, 12.0, 0.51),   # LIMEPY full-range fit
    (1.00, 3.50, 1.75),   # Bahcall-Wolf-like
    (4.00, 5.00, 2.50),   # gamma > 2, steep
]


def _rho_scalar(r, rho0, a, al, b, g):
    x = r / a
    return rho0 * x ** -g * (1.0 + x ** al) ** (-(b - g) / al)


def test_density_matches_formula():
    """density() must equal the Zhao formula on a spherical shell."""
    rho0, a = 1.0e5, 1.5
    for al, b, g in CASES:
        for r in np.geomspace(1e-4, 1e3, 12):
            got = SBH.density(r, 0.0, 0.0, (rho0, a, al, b, g))
            want = _rho_scalar(r, rho0, a, al, b, g)
            rel = abs(float(got) / want - 1.0)
            assert rel < 1e-12, \
                f'density (al,b,g)=({al},{b},{g}) r={r}: rel err {rel:.2e}'
    print('  density_matches_formula OK')


def test_mass_enclosed_matches_quadrature():
    """mass_enclosed() must equal a direct integral of 4 pi r^2 rho."""
    rho0, a = 1.0e5, 1.5
    for al, b, g in CASES:
        for r in np.geomspace(1e-3, 1e2, 8):
            got = float(SBH.mass_enclosed(r, 0.0, 0.0, (rho0, a, al, b, g)))
            # integrate in log r: the linear-space integrand is near-singular
            f = lambda u: (4 * np.pi * np.exp(u) ** 3
                           * _rho_scalar(np.exp(u), rho0, a, al, b, g))
            want, _ = quad(f, np.log(r) - 60, np.log(r), limit=800)
            rel = abs(got / want - 1.0)
            assert rel < 1e-8, \
                f'M(<r) (al,b,g)=({al},{b},{g}) r={r}: rel err {rel:.2e}'
    print('  mass_enclosed_matches_quadrature OK')


def test_rho0_from_mass_round_trips():
    """rho0_from_mass must make M(<inf) equal the requested total mass."""
    m_target, a = 1.79e5, 3.06
    for al, b, g in CASES:
        rho0 = SBH.rho0_from_mass(m_target, a, al, b, g)
        # M(<r) approaches M_tot as (1-t)**q with q = (beta-3)/alpha, which
        # is only 0.5 for e.g. (1.0, 3.5, 1.75) -- so even at r = a*1e12 the
        # truncated integral is ~1e-6 short. The point of this test is that
        # the normalisation constant is right, not that a truncated integral
        # converges; Task 7's Fortran agreement test is the tighter net.
        m_inf = float(SBH.mass_enclosed(a * 1e12, 0.0, 0.0,
                                        (rho0, a, al, b, g)))
        rel = abs(m_inf / m_target - 1.0)
        assert rel < 1e-3, \
            f'rho0 round trip (al,b,g)=({al},{b},{g}): rel err {rel:.2e}'
    print('  rho0_from_mass_round_trips OK')


def test_incomplete_beta_negative_q():
    """The q<=0 branch must match quadrature (this is where zh_betai fails)."""
    for p, q in [(0.64, -0.061), (1.0, -0.09), (0.31, -0.45), (0.75, -0.125)]:
        for x in [1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999]:
            got = SBH.incomplete_beta(x, p, q)
            want, _ = quad(lambda u: u ** (p - 1) * (1 - u) ** (q - 1),
                           0.0, x, limit=400)
            rel = abs(got / want - 1.0)
            assert rel < 1e-9, \
                f'incomplete_beta p={p} q={q} x={x}: rel err {rel:.2e}'
    print('  incomplete_beta_negative_q OK')


def test_incomplete_beta_integer_q_raises():
    """A non-positive integer q hits qq=0 in the recurrence; must raise."""
    for q in [0.0, -1.0, -2.0, -5.0]:
        try:
            SBH.incomplete_beta(0.5, 0.64, q)
        except ValueError:
            continue
        raise AssertionError(
            f'incomplete_beta(0.5, 0.64, {q}) should have raised ValueError')
    print('  incomplete_beta_integer_q_raises OK')


TESTS = [
    test_density_matches_formula,
    test_mass_enclosed_matches_quadrature,
    test_rho0_from_mass_round_trips,
    test_incomplete_beta_negative_q,
    test_incomplete_beta_integer_q_raises,
]

if __name__ == '__main__':
    print(f'dynamite {dyn.__version__ if hasattr(dyn, "__version__") else ""}')
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'FAIL {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)

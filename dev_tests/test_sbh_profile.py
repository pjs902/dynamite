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


def _beta_reference(x, p, q):
    """High-precision B(x;p,q), independent of the implementation.

    mpmath at 60 digits if importable, else a tight quad of the defining
    integral split at the endpoint singularities.
    """
    try:
        import mpmath
    except ImportError:
        mpmath = None
    if mpmath is not None:
        with mpmath.workdps(60):
            v = mpmath.betainc(mpmath.mpf(p), mpmath.mpf(q),
                               0, mpmath.mpf(x))
        return float(v)
    f = lambda u: u ** (p - 1) * (1 - u) ** (q - 1)
    val, _ = quad(f, 0.0, x, limit=800, epsabs=1e-15, epsrel=1e-15,
                  points=None)
    return val


def test_incomplete_beta_near_one():
    """B(x;p,q) with q<=0 must stay accurate as x -> 1 (1-x down to 1e-12).

    The fixed-seed downward recurrence saturates here: betainc(p,q+n,x)
    returns exactly 1.0 once (1-x)**(q+n) underflows the mantissa, which
    silently destroys the information the recurrence steps down from.
    """
    worst = 0.0
    for p, q in [(0.64, -0.061), (1.0, -0.09), (0.31, -0.45),
                 (0.75, -0.125), (2.5, -0.9), (0.639, -0.0614)]:
        for t in np.geomspace(1e-1, 1e-12, 12):
            x = 1.0 - t
            got = SBH.incomplete_beta(x, p, q)
            want = _beta_reference(x, p, q)
            rel = abs(got / want - 1.0)
            worst = max(worst, rel)
            assert rel < 1e-12, \
                f'incomplete_beta p={p} q={q} 1-x={t:.1e}: rel err {rel:.2e}'
    print(f'  incomplete_beta_near_one OK (worst rel err {worst:.2e})')


def test_outer_tail_integral_accuracy():
    """int_y^inf s^(q-1)(1+s)^-(p+q) ds to 1e-12 over the whole y range.

    This is where the old code actually lost its digits: the caller
    formed ``x = 1 - y/(1+y)``, so for small r the complement fell off
    the end of the mantissa and x became exactly 1.0 -- a point where the
    integral is not even finite for q <= 0. Parametrising by y instead
    keeps the small quantity small. y spans (r/a)^alpha over the whole
    physical range, ~1e-16 to 1e12.
    """
    mpmath = _require_mpmath()
    worst, worst_at = 0.0, None
    cases = list(CASES) + [
        (0.5, 4.0, 2.5),         # q = -1 exactly (integer): no recurrence
        (0.2, 12.0, 2.9),        # p+q = 45.5, gamma near the gamma<3 wall
        (1.0, 6.0, 2.0 - 2e-6),  # q -> 0+, the closest validate_parset allows
        (1.0, 6.0, 2.0 + 2e-6),  # q -> 0-, ditto from the other side
        (3.5, 10.0, 1.993),      # small positive q = 0.002
    ]
    for al, b, g in cases:
        p, q = (b - 2) / al, (2 - g) / al
        for y in np.geomspace(1e12, 1e-16, 60):
            with mpmath.workdps(60):
                want = mpmath.betainc(mpmath.mpf(p), mpmath.mpf(q), 0,
                                      mpmath.mpf(1) / (1 + mpmath.mpf(y)))
            if abs(want) < 1e-290 or abs(want) > 1e290:
                continue          # outside the double-precision range
            got = SBH._outer_tail_integral(y, p, q)
            rel = float(abs(mpmath.mpf(got) / want - 1))
            if rel > worst:
                worst, worst_at = rel, (al, b, g, y)
            assert rel < 1e-12, \
                f'tail (al,b,g)=({al},{b},{g}) y={y:.2e}: rel err {rel:.2e}'
    print(f'  outer_tail_integral_accuracy OK '
          f'(worst {worst:.2e} at {worst_at})')


def _require_mpmath():
    """mpmath, or skip-by-raising if it is not installed."""
    try:
        import mpmath
    except ImportError:                                   # pragma: no cover
        raise AssertionError('mpmath needed for the accuracy gate')
    return mpmath


def test_potential_outer_tail_is_closed_form():
    """The outer tail must be the closed form, with no quadrature fallback.

    Guards the regression the quad workaround was papering over: the
    fitted LIMEPY case has gamma > 2, so q_out = (2-gamma)/alpha <= 0 is
    the production path, and Task 6 reimplements it in Fortran where
    scipy.integrate does not exist.
    """
    import dynamite.physical_system as ps

    calls = []
    real_quad = ps.integrate.quad

    def _spy(*a, **kw):
        calls.append(a)
        return real_quad(*a, **kw)

    rho0, a = 1.0e5, 1.5
    al, b, g = 3.91, 4.50, 2.24          # q_out = (2-g)/al < 0
    pars = (rho0, a, al, b, g)
    ps.integrate.quad = _spy
    try:
        phi = [float(SBH.potential(r, 0.0, 0.0, pars))
               for r in np.geomspace(1e-3, 1e2, 6)]
    finally:
        ps.integrate.quad = real_quad
    assert not calls, \
        f'potential() still calls scipy.integrate.quad ({len(calls)} times)'
    assert all(np.isfinite(phi)) and all(v < 0.0 for v in phi), \
        f'potential() not finite and negative: {phi}'

    # and the closed form must agree with a direct high-precision tail
    for r in [1e-3, 1e-2, 0.3, 5.0]:
        got = SBH._outer_tail(r, rho0, a, al, b, g)
        want, _ = quad(lambda u: (4 * np.pi * np.exp(u) ** 2
                                  * _rho_scalar(np.exp(u), rho0, a, al, b, g)),
                       np.log(r), np.log(r) + 60, limit=800)
        rel = abs(got / want - 1.0)
        assert rel < 1e-9, f'outer tail r={r}: rel err {rel:.2e}'
    print('  potential_outer_tail_is_closed_form OK')


def test_acceleration_equals_minus_grad_potential():
    """a_r must equal -dPhi/dr, which ties potential() and acceleration()."""
    rho0, a = 1.0e5, 1.5
    G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
    for al, b, g in CASES:
        pars = (rho0, a, al, b, g)
        for r in np.geomspace(1e-2, 1e2, 8):
            h = r * 1e-6
            phi_p = float(SBH.potential(r + h, 0.0, 0.0, pars))
            phi_m = float(SBH.potential(r - h, 0.0, 0.0, pars))
            num = -(phi_p - phi_m) / (2 * h)
            ana = -G * float(SBH.mass_enclosed(r, 0.0, 0.0, pars)) / r ** 2
            rel = abs(ana / num - 1.0)
            assert rel < 1e-5, \
                f'accel vs -dPhi/dr (al,b,g)=({al},{b},{g}) r={r}: {rel:.2e}'
    print('  acceleration_equals_minus_grad_potential OK')


def test_acceleration_points_inward():
    """Gravity must attract: the radial acceleration is negative."""
    par = {'m': 1.79e5, 'a_pc': 3.06, 'alpha': 3.91,
           'beta': 4.50, 'gamma': 2.24}
    for r in np.geomspace(1e-2, 1e2, 10):
        ax, ay, az = SBH.acceleration(r, 0.0, 0.0, par)
        assert float(ax) < 0.0, f'acceleration not inward at r={r}: {ax}'
    print('  acceleration_points_inward OK')


TESTS = [
    test_density_matches_formula,
    test_mass_enclosed_matches_quadrature,
    test_rho0_from_mass_round_trips,
    test_incomplete_beta_negative_q,
    test_incomplete_beta_integer_q_raises,
    test_incomplete_beta_near_one,
    test_outer_tail_integral_accuracy,
    test_potential_outer_tail_is_closed_form,
    test_acceleration_equals_minus_grad_potential,
    test_acceleration_points_inward,
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

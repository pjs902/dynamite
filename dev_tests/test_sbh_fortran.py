#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fortran-vs-Python agreement for the sBH component.

The two implementations of the same physics must not drift. Requires
legacy_fortran/sbh_probe to be built:  cd legacy_fortran && make sbh_probe

Run:  /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_fortran.py

Tolerance: rtol = 1e-12 across the whole probed range r/a = 1e-6 .. 1e4,
for every parameter set tested (including gamma > 2, where the incomplete-
beta recurrence in Fortran, and the explicit-complement rewrite in Python,
are both exercised). For the production parset the diff is tight enough
to also pass 1e-14, so that is asserted separately as the stronger check.

The Fortran ``pot`` returned by ``dm_potent_sbh_only`` is ``+G(M/r + tail)``,
i.e. psi = -Phi (positive). The Python comparison must be built the same
way, in km units, from ``StellarBlackHoles.mass_enclosed`` and
``StellarBlackHoles._outer_tail`` directly -- NOT from
``StellarBlackHoles.potential``/``.acceleration``, which convert to pc and
would produce a large, confusing, spurious disagreement.

Each radius is probed twice: once on-axis (r,0,0) and once off-axis
(s,s,s), s = r/sqrt(3) -- same radius, all three coordinates nonzero -- so
the acceleration-direction projection in dm_accel_sbh_only is exercised,
not just the x-only path.

If the probe binary is missing, this MUST fail loudly (never silently
skip/pass): a missing binary must never read as "the sides agree".
"""
import os
import subprocess
import sys

import numpy as np

import dynamite as dyn
from dynamite.physical_system import StellarBlackHoles as SBH

PROBE = os.path.join(os.path.dirname(__file__), '..',
                      'legacy_fortran', 'sbh_probe')

G = dyn.constants.GRAV_CONST_KM

# (label, alpha, beta, gamma); production parset, a spread of gamma regimes,
# and the two hard parsets from Task 6's own review that were found wrong by
# measurement before being fixed:
#   - (0.2, 12.0, 2.9): p=0.5, q=45 in sbh_menc/mass_enclosed -- the fixed
#     x<=1/2 reflection was measured wrong by 3.15e-10 here before the
#     continued-fraction fix.
#   - (3.0, 4.0, 1.99999): gamma -> 2, q_out -> 0 in the outer-tail integral
#     -- the naive expm1 formula lost 5.76e-13 here before the Kahan-identity
#     fix.
CASES = [
    ('production', 3.91, 4.50, 2.24),
    ('gamma<2, q>1 (CF branch)', 2.15, 12.0, 1.74),
    ('gamma<2, small alpha, worst case', 0.31, 12.0, 0.51),
    ('gamma>2, near-integer q', 2.00, 4.00, 2.18),
    ('gamma>2, steep', 4.00, 5.00, 2.50),
    ('gamma negative', 4.00, 20.0, -2.00),
    ('gamma>2, cuspy', 0.30, 3.05, 2.99),
    ('reflection stress (Task 6 defect a)', 0.20, 12.0, 2.90),
    ('gamma->2, expm1 (Task 6 defect b)', 3.00, 4.00, 1.99999),
]
RHO0, A = 1.0e-40, 3.0857e13   # Msun/km^3, km -- production-fit scale

RTOL_ALL = 1e-12
RTOL_PRODUCTION = 1e-14


def _require_probe():
    if not os.path.exists(PROBE):
        raise AssertionError(
            f'{PROBE} not built -- run: cd legacy_fortran && make sbh_probe. '
            'A missing probe must fail loudly here, never read as a pass.')


def run_probe(rho0, a, al, b, g):
    proc = subprocess.run(
        [PROBE, repr(rho0), repr(a), repr(al), repr(b), repr(g)],
        capture_output=True, text=True)
    out = proc.stdout
    if not out.strip():
        # A Fortran `stop 'msg'` exits status 0 under gfortran, so
        # `returncode` alone is not a reliable signal here -- surface
        # stderr explicitly rather than letting this die as an opaque
        # "not enough values to unpack" a few lines down.
        raise RuntimeError(
            f'sbh_probe produced no output for '
            f'(rho0,a,al,b,g)=({rho0},{a},{al},{b},{g}); '
            f'returncode={proc.returncode}, stderr={proc.stderr.strip()!r}')
    rows = [list(map(float, ln.split())) for ln in out.strip().splitlines()]
    arr = np.array(rows)
    # r, pot_onaxis, vx_onaxis, vy_onaxis, vz_onaxis,
    #    pot_offaxis, vx_offaxis, vy_offaxis, vz_offaxis
    return tuple(arr[:, i] for i in range(9))


def python_reference(r, rho0, a, al, b, g):
    """Potential (psi, positive) and acceleration, on- and off-axis, km."""
    pars = (rho0, a, al, b, g)
    m_enc = SBH.mass_enclosed(r, 0.0, 0.0, pars)   # depends on |r| only
    # _outer_tail is scalar-only (see _outer_tail_integral); loop over r.
    tail = np.array([SBH._outer_tail(ri, rho0, a, al, b, g) for ri in r])
    pot = G * (m_enc / r + tail)                    # same on- and off-axis
    a_r = -G * m_enc / r ** 2                        # radial component
    zero = np.zeros_like(a_r)
    f = 1.0 / np.sqrt(3.0)
    return (pot, a_r, zero, zero,           # on-axis: (1,0,0) direction
            pot, a_r * f, a_r * f, a_r * f)  # off-axis: (1,1,1)/sqrt3


def _case_errors(al, b, g):
    """Relative-error dict for one parameter set, plus the radius array."""
    r, pot_f, vx_f, vy_f, vz_f, pot2_f, wx_f, wy_f, wz_f = run_probe(
        RHO0, A, al, b, g)
    (pot_p, ax_p, ay_p, az_p,
     pot2_p, wx_p, wy_p, wz_p) = python_reference(r, RHO0, A, al, b, g)

    errs = {
        'pot': np.max(np.abs(pot_f / pot_p - 1.0)),
        'ax': np.max(np.abs(vx_f / ax_p - 1.0)),
        # on-axis transverse components should be (numerically) exactly
        # zero on both sides -- x/d*a_r with x=0 -- so check absolute, not
        # relative (dividing by an expected-zero denominator is meaningless).
        'ay_ax_abs': np.max(np.abs(vy_f)),
        'az_ax_abs': np.max(np.abs(vz_f)),
        'pot_off': np.max(np.abs(pot2_f / pot2_p - 1.0)),
        'ax_off': np.max(np.abs(wx_f / wx_p - 1.0)),
        'ay_off': np.max(np.abs(wy_f / wy_p - 1.0)),
        'az_off': np.max(np.abs(wz_f / wz_p - 1.0)),
    }
    return errs, r


def _assert_agreement(label, al, b, g, rtol, r=None, errs=None):
    """The one real comparison path. Raises AssertionError on disagreement.

    Both the main test and the discrimination test call this exact
    function, so a pass in the main test is genuinely evidence the two
    sides agree (rather than an inline re-implementation that could pass
    vacuously).
    """
    if errs is None:
        errs, r = _case_errors(al, b, g)

    checks = [
        ('pot on-axis', errs['pot']),
        ('accel on-axis, radial component', errs['ax']),
        ('pot off-axis', errs['pot_off']),
        ('accel off-axis, x component', errs['ax_off']),
        ('accel off-axis, y component', errs['ay_off']),
        ('accel off-axis, z component', errs['az_off']),
    ]
    for name, val in checks:
        assert val < rtol, (
            f'{name} ({label}, al,b,g)=({al},{b},{g}): rel err {val:.3e} '
            f'exceeds rtol={rtol:.0e}')
    assert errs['ay_ax_abs'] < 1e-250 and errs['az_ax_abs'] < 1e-250, (
        f'on-axis transverse acceleration not zero ({label}): '
        f"vy={errs['ay_ax_abs']:.3e}, vz={errs['az_ax_abs']:.3e}")
    return errs


def test_fortran_matches_python():
    _require_probe()

    worst_pot, worst_ax, worst_label = 0.0, 0.0, None

    for label, al, b, g in CASES:
        rtol = RTOL_PRODUCTION if label == 'production' else RTOL_ALL
        errs = _assert_agreement(label, al, b, g, rtol)

        ep = max(errs['pot'], errs['pot_off'])
        ea = max(errs['ax'], errs['ax_off'], errs['ay_off'], errs['az_off'])
        if ep > worst_pot or ea > worst_ax:
            worst_pot, worst_ax = max(worst_pot, ep), max(worst_ax, ea)
            worst_label = label

        print(f'  {label:38s} (al,b,g)=({al},{b},{g}): '
              f'pot {ep:.2e}, accel {ea:.2e}')

    print(f'  worst overall: pot {worst_pot:.2e}, accel {worst_ax:.2e} '
          f'(last hit at {worst_label!r})')
    print('  fortran_matches_python OK')


def test_probe_actually_discriminates():
    """Prove the real assertion path -- not a re-implementation -- can fail.

    Three checks, all going through ``_assert_agreement`` itself:
      1. it passes on genuine probe output (sanity: not vacuously false);
      2. tightening rtol far below the achieved precision makes it raise,
         so it is not comparing a value to itself;
      3. corrupting each individually-checked field (potential, on-axis
         accel, and each off-axis accel component) independently makes it
         raise, so a sign/unit/projection bug in any one of them would be
         caught, not masked by the others.
    """
    _require_probe()
    label, al, b, g = CASES[0]
    errs, r = _case_errors(al, b, g)

    # 1. real data really does pass the real assertion
    _assert_agreement(label, al, b, g, RTOL_ALL, r=r, errs=errs)

    # 2. an unreasonably tight rtol must fail
    try:
        _assert_agreement(label, al, b, g, 1e-20, r=r, errs=errs)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            'harness failed to detect disagreement at rtol=1e-20; the '
            'comparison may be vacuous (e.g. comparing a value to itself)')

    # 3. each checked field independently gates the assertion
    for field in ('pot', 'ax', 'pot_off', 'ax_off', 'ay_off', 'az_off'):
        bad = dict(errs)
        bad[field] = 1e-3
        try:
            _assert_agreement(label, al, b, g, RTOL_ALL, r=r, errs=bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f'field {field!r} is not actually gating the assertion')

    print('  discrimination checks OK (real path fails on real corruption)')


TESTS = [test_fortran_matches_python, test_probe_actually_discriminates]

if __name__ == '__main__':
    failed = 0
    for t in TESTS:
        try:
            t()
        except (AssertionError, RuntimeError) as e:
            print(f'FAIL {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)

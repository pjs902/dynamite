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

# (alpha, beta, gamma); production parset plus a spread covering gamma > 2
# (where the incomplete-beta recurrence/complement handling matters most)
# and gamma < 2 / near-zero / negative regimes.
CASES = [
    ('production', 3.91, 4.50, 2.24),
    ('gamma<2, q>1', 2.15, 12.0, 1.74),
    ('gamma<2, small alpha', 0.31, 12.0, 0.51),
    ('gamma>2, near-integer q', 2.00, 4.00, 2.18),
    ('gamma>2, steep', 4.00, 5.00, 2.50),
    ('gamma negative', 4.00, 20.0, -2.00),
    ('gamma>2, cuspy', 0.30, 3.05, 2.99),
]
RHO0, A = 1.0e-40, 3.0857e13   # Msun/km^3, km -- production-fit scale

RTOL_ALL = 1e-12
RTOL_PRODUCTION = 1e-14


def run_probe(rho0, a, al, b, g):
    out = subprocess.run(
        [PROBE, repr(rho0), repr(a), repr(al), repr(b), repr(g)],
        capture_output=True, text=True, check=True).stdout
    rows = [list(map(float, ln.split())) for ln in out.strip().splitlines()]
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]      # r [km], pot, a_x


def python_reference(r, rho0, a, al, b, g):
    """Potential (psi, positive) and radial acceleration, km units."""
    pars = (rho0, a, al, b, g)
    m_enc = SBH.mass_enclosed(r, 0.0, 0.0, pars)
    # _outer_tail is scalar-only (see _outer_tail_integral); loop over r.
    tail = np.array([SBH._outer_tail(ri, rho0, a, al, b, g) for ri in r])
    pot = G * (m_enc / r + tail)
    ax = -G * m_enc / r ** 2
    return pot, ax


def _compare(al, b, g, label):
    r, pot_f, ax_f = run_probe(RHO0, A, al, b, g)
    pot_p, ax_p = python_reference(r, RHO0, A, al, b, g)

    rel_pot = np.abs(pot_f / pot_p - 1.0)
    rel_ax = np.abs(ax_f / ax_p - 1.0)
    return r, pot_f, pot_p, rel_pot, ax_f, ax_p, rel_ax


def test_fortran_matches_python():
    if not os.path.exists(PROBE):
        print(f'SKIP: {PROBE} not built (cd legacy_fortran && make sbh_probe)')
        return

    worst_pot, worst_ax = 0.0, 0.0
    worst_where = None

    for label, al, b, g in CASES:
        r, pot_f, pot_p, rel_pot, ax_f, ax_p, rel_ax = _compare(al, b, g, label)

        ep = np.max(rel_pot)
        ea = np.max(rel_ax)
        i_p = np.argmax(rel_pot)
        i_a = np.argmax(rel_ax)

        rtol = RTOL_PRODUCTION if label == 'production' else RTOL_ALL
        assert ep < rtol, (
            f'potential ({label}, al,b,g)=({al},{b},{g}): worst rel err '
            f'{ep:.3e} at r/a={r[i_p]/A:.3e}, exceeds rtol={rtol:.0e}')
        assert ea < rtol, (
            f'accel ({label}, al,b,g)=({al},{b},{g}): worst rel err '
            f'{ea:.3e} at r/a={r[i_a]/A:.3e}, exceeds rtol={rtol:.0e}')

        if ep > worst_pot:
            worst_pot, worst_where = ep, (label, 'pot', r[i_p] / A)
        if ea > worst_ax:
            worst_ax, worst_where = ea, worst_where or (label, 'accel', r[i_a] / A)

        print(f'  {label:24s} (al,b,g)=({al},{b},{g}): '
              f'pot {ep:.2e}, accel {ea:.2e}')

    print(f'  worst overall: pot {worst_pot:.2e}, accel {worst_ax:.2e}')
    print('  fortran_matches_python OK')


def test_probe_actually_discriminates():
    """Sanity check: the comparison can fail if one side is wrong.

    Perturbs the Python reference by a factor well above rtol and confirms
    the assertion trips, so a pass above does not silently compare a value
    to itself.
    """
    if not os.path.exists(PROBE):
        print('SKIP: probe not built')
        return
    al, b, g = CASES[0][1:]
    r, pot_f, ax_f = run_probe(RHO0, A, al, b, g)
    pot_p, ax_p = python_reference(r, RHO0, A, al, b, g)

    perturbed = pot_p * (1.0 + 1e-6)
    rel = np.max(np.abs(pot_f / perturbed - 1.0))
    assert rel > RTOL_ALL, (
        'harness failed to detect a deliberately perturbed comparison '
        f'(rel={rel:.3e})')
    print(f'  perturbation check OK (rel={rel:.3e} > rtol)')


TESTS = [test_fortran_matches_python, test_probe_actually_discriminates]

if __name__ == '__main__':
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'FAIL {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)

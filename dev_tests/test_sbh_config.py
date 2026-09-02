#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config-level checks for the two-dark-slot rule.

Run:  /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py
"""
import sys
import dynamite as dyn
from dynamite import physical_system as physys


def _mk(cls, name):
    c = cls.__new__(cls)
    c.name = name
    return c


def test_helpers_split_halo_and_sbh():
    """get_halo_component / get_sbh_component must partition the dark comps."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh'),
                  _mk(physys.NFW, 'dh'),
                  _mk(physys.StellarBlackHoles, 'sbh')]
    halo = s.get_halo_component()
    sbh = s.get_sbh_component()
    assert isinstance(halo, physys.NFW), f'halo was {halo}'
    assert isinstance(sbh, physys.StellarBlackHoles), f'sbh was {sbh}'
    print('  helpers_split_halo_and_sbh OK')


def test_helpers_return_none_when_absent():
    """Both helpers must return None rather than raising when unused."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh')]
    assert s.get_halo_component() is None
    assert s.get_sbh_component() is None
    print('  helpers_return_none_when_absent OK')


def test_two_halos_still_rejected():
    """Two *halos* remain illegal; only halo+sBH is the new legal pairing."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh'),
                  _mk(physys.NFW, 'dh1'),
                  _mk(physys.Hernquist, 'dh2')]
    try:
        s.get_halo_component()
    except ValueError:
        print('  two_halos_still_rejected OK')
        return
    raise AssertionError('two halos were accepted')


def test_sbh_legacy_strings():
    """The legacy strings must carry code 6, 5 params, and derived rhoc."""
    import numpy as np
    from dynamite import constants

    c = _mk(physys.StellarBlackHoles, 'sbh')
    c.legacy_code = 6
    c.par_names = ['rhoc', 'a', 'alpha', 'beta', 'gamma']
    import logging
    c.logger = logging.getLogger('test')

    class _Sys:
        distMPc = 0.00543          # omega Cen, 5.43 kpc

    parset = {'m-sbh': 1.79e5, 'a-sbh': 100.0, 'alpha-sbh': 3.91,
              'beta-sbh': 4.50, 'gamma-sbh': 2.24}
    specs, vals = c.get_dh_legacy_strings(parset, _Sys())
    assert specs == '6 5', f'specs was {specs!r}'
    parts = [float(v) for v in vals.split()]
    assert len(parts) == 5, f'expected 5 values, got {parts}'
    rhoc, a_km, al, b, g = parts
    assert (al, b, g) == (3.91, 4.50, 2.24), f'shape params wrong: {parts}'
    # a_km must be the arcsec value times the arcsec->km factor
    want_a = 100.0 * constants.ARC_KM(0.00543)
    assert abs(a_km / want_a - 1) < 1e-12, f'a_km {a_km} != {want_a}'
    # and rhoc must reproduce the requested total mass
    m_back = float(physys.StellarBlackHoles.mass_enclosed(
        a_km * 1e8, 0.0, 0.0, (rhoc, a_km, al, b, g)))
    assert abs(m_back / 1.79e5 - 1) < 1e-6, f'mass round trip: {m_back}'
    print('  sbh_legacy_strings OK')


TESTS = [test_helpers_split_halo_and_sbh,
         test_helpers_return_none_when_absent,
         test_two_halos_still_rejected,
         test_sbh_legacy_strings]

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

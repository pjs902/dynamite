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


TESTS = [test_helpers_split_halo_and_sbh,
         test_helpers_return_none_when_absent,
         test_two_halos_still_rejected]

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

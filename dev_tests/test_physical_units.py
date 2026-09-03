#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks for ``AllModels.get_physical_parameter_table``.

Dark-component masses are stored per orbit library: a model reusing an
orblib built at ``ml_orblib`` has every mass in its potential scaled by
``ml/ml_orblib``.  These tests pin that correction down, including the one
property the IMBH-vs-sBH science case depends on -- that both central masses
take the *same* factor, so their ratio needs no correction at all.

Run:  /opt/miniconda3/envs/main/bin/python dev_tests/test_physical_units.py
"""
import os
import sys

import numpy as np
from astropy import table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamite import physical_system as physys  # noqa: E402
from dynamite.model import AllModels  # noqa: E402


def _mk(cls, name):
    c = cls.__new__(cls)
    c.name = name
    return c


class _Parspace:
    par_names = ['ml', 'm-bh', 'm-sbh']


class _Config:
    parspace = _Parspace()


class _System:
    """Stand-in with a central BH and a fitted sBH, and no dark halo."""

    def __init__(self):
        self.bh = _mk(physys.Plummer, 'bh')
        self.sbh = _mk(physys.StellarBlackHoles, 'sbh')

    def get_halo_component(self):
        return None

    def get_sbh_component(self):
        return self.sbh

    def get_component_from_class(self, cmp_class):
        return self.bh


def _all_models():
    """Three models: rows 0 and 1 share an orblib built at ml=2, row 2 is new.

    Row 1 therefore carries a correction factor of ml/ml_orblib = 4/2 = 2.
    """
    am = AllModels.__new__(AllModels)
    am.config = _Config()
    am.system = _System()
    am.logger = __import__('logging').getLogger(__name__)
    am.table = table.Table(
        {'ml': [2.0, 4.0, 3.0],
         'm-bh': [1.0e4, 1.0e4, 2.0e4],
         'm-sbh': [1.0e5, 1.0e5, 1.0e5],
         'directory': ['m0', 'm1', 'm2']}
    )
    return am


def test_reused_orblib_scales_both_central_masses():
    am = _all_models()
    phys = am.get_physical_parameter_table()
    # row 0 built its own orblib: factor 1
    assert np.isclose(phys['m-bh'][0], 1.0e4)
    assert np.isclose(phys['m-sbh'][0], 1.0e5)
    # row 1 reuses row 0's orblib at ml=4 vs ml_orblib=2: factor 2
    assert np.isclose(phys['m-bh'][1], 2.0e4)
    assert np.isclose(phys['m-sbh'][1], 2.0e5)
    # row 2 has a different m-bh, so it builds its own orblib: factor 1
    assert np.isclose(phys['m-bh'][2], 2.0e4)
    assert np.isclose(phys['m-sbh'][2], 1.0e5)


def test_imbh_to_sbh_ratio_is_correction_free():
    """The property the IMBH/sBH split relies on: same factor, so same ratio."""
    am = _all_models()
    stored = am.table['m-bh'] / am.table['m-sbh']
    phys = am.get_physical_parameter_table()
    assert np.allclose(stored, phys['m-bh'] / phys['m-sbh'])


def test_source_table_is_not_mutated():
    am = _all_models()
    before = am.table['m-bh'].copy()
    am.get_physical_parameter_table()
    assert np.allclose(before, am.table['m-bh'])


def test_empty_table_is_handled():
    am = _all_models()
    am.table = am.table[:0]
    assert len(am.get_physical_parameter_table()) == 0


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS  {name}')
            except AssertionError as e:
                fails += 1
                print(f'FAIL  {name}: {e}')
    print(f'\n{"all passed" if not fails else f"{fails} failed"}')
    sys.exit(1 if fails else 0)

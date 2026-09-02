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


# ---------------------------------------------------------------------------
# Minimal object graph for exercising
# LegacyOrbitLibrary.create_fortran_input_orblib directly, so the byte-level
# format of parameters_pot.in (halo slot / sBH slot / no-sBH byte-identity)
# is under automated regression coverage instead of manual-only verification.
# Everything below is a hand-built stand-in, not real physical_system /
# config_reader objects: constructing a full Configuration is far heavier
# than the file-writing code path this covers actually needs.
# ---------------------------------------------------------------------------

class _PotStars:
    """Stand-in for the triaxial visible ('stars') component."""
    name = 'stars'

    class _MGE:
        import numpy as _np
        data = _np.array([[100.0, 1.0, 0.8, 0.0]])

    mge_pot = _MGE()
    mge_lum = _MGE()

    def triax_pqu2tpp(self, p, q, u):
        return (10.0, 20.0, 30.0)


class _PotBH:
    """Stand-in for the central (Plummer) black hole component."""
    name = 'bh'


class _PotSystem:
    """Stand-in for physical_system.System, exposing only what
    create_fortran_input_orblib reads from it."""

    def __init__(self, halo=None, sbh=None):
        self._halo = halo
        self._sbh = sbh
        self.distMPc = 10.0
        self.H = 67.0

    def get_component_from_class(self, cls):
        return _PotBH()

    def is_bar_disk_system(self):
        return False

    def get_halo_component(self):
        return self._halo

    def get_sbh_component(self):
        return self._sbh


_POT_SETTINGS = {'nE': 6, 'logrmin': -1.0, 'logrmax': 1.0, 'nI2': 5,
                  'nI3': 4, 'dithering': 1, 'quad_nr': 10, 'quad_nth': 6,
                  'quad_nph': 6}

_POT_PARSET = {'ml': 1.0, 'm-bh': 1e6, 'a-bh': 0.01,
               'q-stars': 0.7, 'p-stars': 0.9, 'u-stars': 0.99}


def _make_halo():
    """A minimal NFW stand-in good enough for get_dh_legacy_strings(parset)."""
    halo = _mk(physys.NFW, 'dh')
    halo.legacy_code = 1
    halo.par_names = ['c', 'f']
    halo.parameters = ['c', 'f']  # only len() of this is used
    import logging
    halo.logger = logging.getLogger('test')
    return halo


def _make_sbh():
    sbh = _mk(physys.StellarBlackHoles, 'sbh')
    sbh.legacy_code = 6
    sbh.par_names = ['rhoc', 'a', 'alpha', 'beta', 'gamma']
    import logging
    sbh.logger = logging.getLogger('test')
    return sbh


_SBH_PARS = {'m-sbh': 1.79e5, 'a-sbh': 100.0, 'alpha-sbh': 3.91,
             'beta-sbh': 4.50, 'gamma-sbh': 2.24}


def _write_pot_file(halo=None, sbh=None, parset_extra=None):
    """Build a minimal LegacyOrbitLibrary and return the raw bytes it
    writes to parameters_pot.in.

    create_fortran_input_orblib writes parameters_pot.in and
    parameters_lum.in before going on to orbstart.in etc, which need
    settings (e.g. random_seed) this stand-in deliberately omits since
    they are irrelevant to the file under test; the resulting KeyError
    is expected and ignored.
    """
    import logging
    import tempfile
    from dynamite import orblib as orblib_mod

    ol = orblib_mod.LegacyOrbitLibrary.__new__(orblib_mod.LegacyOrbitLibrary)
    ol.logger = logging.getLogger('test')
    ol.system = _PotSystem(halo=halo, sbh=sbh)
    ol.settings = _POT_SETTINGS
    ol.parset = dict(_POT_PARSET)
    if parset_extra:
        ol.parset.update(parset_extra)
    ol.stars = _PotStars()

    path = tempfile.mkdtemp() + '/'
    try:
        ol.create_fortran_input_orblib(path)
    except KeyError:
        pass
    with open(path + 'parameters_pot.in', 'rb') as f:
        return f.read()


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


def test_pot_in_no_sbh_is_byte_identical():
    """No sBH present: parameters_pot.in must match the pre-sBH format
    exactly, byte for byte (no stray newline/space from the append code)."""
    data = _write_pot_file(halo=None, sbh=None)
    expected = (
        b'1\n    100.00    1.00000    0.80000       0.00\n10.0\n'
        b'10.000000000 30.000000000 20.000000000\n1.0\n1000000.0\n0.01\n'
        b'6 -1.0 1.0\n5\n4\n1\n10\n6\n6\n0 0\n\n6.7e-05\n'
    )
    assert data == expected, f'pot.in bytes changed: {data!r}'
    print('  pot_in_no_sbh_is_byte_identical OK')


def test_pot_in_halo_and_sbh_trailing_block():
    """Halo + sBH: slot 1 carries the halo as before, and the trailing
    two-line sBH block is appended exactly at end of file."""
    halo = _make_halo()
    sbh = _make_sbh()
    data = _write_pot_file(halo=halo, sbh=sbh,
                            parset_extra={'c-dh': 12.0, 'f-dh': 0.3,
                                          **_SBH_PARS})
    expected = (
        b'1\n    100.00    1.00000    0.80000       0.00\n10.0\n'
        b'10.000000000 30.000000000 20.000000000\n1.0\n1000000.0\n0.01\n'
        b'6 -1.0 1.0\n5\n4\n1\n10\n6\n6\n1 2\n12.0 0.3\n6.7e-05\n'
        b'6 5\n2.3413449967065113e-48 1.495978706811721e+17 3.91 4.5 2.24\n'
    )
    assert data == expected, f'pot.in bytes wrong with halo+sBH: {data!r}'
    print('  pot_in_halo_and_sbh_trailing_block OK')


def test_pot_in_sbh_no_halo_slot1_is_00():
    """sBH present, no halo: slot 1 must be the literal '0 0' and the
    trailing sBH block must still be correct."""
    sbh = _make_sbh()
    data = _write_pot_file(halo=None, sbh=sbh, parset_extra=_SBH_PARS)
    expected = (
        b'1\n    100.00    1.00000    0.80000       0.00\n10.0\n'
        b'10.000000000 30.000000000 20.000000000\n1.0\n1000000.0\n0.01\n'
        b'6 -1.0 1.0\n5\n4\n1\n10\n6\n6\n0 0\n\n6.7e-05\n'
        b'6 5\n2.3413449967065113e-48 1.495978706811721e+17 3.91 4.5 2.24\n'
    )
    assert data == expected, f'pot.in bytes wrong with sBH, no halo: {data!r}'
    assert b'\n0 0\n' in data, 'slot 1 must be the literal 0 0'
    print('  pot_in_sbh_no_halo_slot1_is_00 OK')


TESTS = [test_helpers_split_halo_and_sbh,
         test_helpers_return_none_when_absent,
         test_two_halos_still_rejected,
         test_sbh_legacy_strings,
         test_pot_in_no_sbh_is_byte_identical,
         test_pot_in_halo_and_sbh_trailing_block,
         test_pot_in_sbh_no_halo_slot1_is_00]

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

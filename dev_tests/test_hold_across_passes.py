"""A relative-PM solve decompresses each pm/losvd archive once, not per pass.

Runs the real ``NNLS._solve_relative`` on the toy of test_relative_pm.py. The
toy library gets the real hold/decompress machinery of ``LegacyOrbitLibrary``
with a fake ``_decompress`` that counts its calls, one archive per kinematic
set. Run from dev_tests/ (or with dev_tests on sys.path)."""
import glob
import os
import types

import numpy as np
import pytest

from dynamite import orblib as dyn_orblib
from test_relative_pm import _toy

LOL = dyn_orblib.LegacyOrbitLibrary


def _archive_lib(lib, tmp_path, hold=True):
    """Make the toy lib 'read' one archive per set through _decompress_held."""
    (tmp_path / "datfil").mkdir(exist_ok=True)
    lib.count, lib._held, lib._decompressed = {}, None, []

    def _decompress(src, dest):
        lib.count[src] = lib.count.get(src, 0) + 1
        open(dest, "w").close()
        lib._decompressed.append(os.path.abspath(dest))
    lib._decompress = _decompress
    for name in ("_decompress_held", "_remove_unless_held"):
        setattr(lib, name, types.MethodType(getattr(LOL, name), lib))
    if hold:
        lib.hold_decompressed = types.MethodType(LOL.hold_decompressed, lib)
    read = lib.read_vel_histograms
    lib.n_reads = 0

    def read_vel_histograms(kin_sets, **kw):
        lib.n_reads += 1
        for si in kin_sets:
            src = f"{tmp_path}/datfil/set{si}_pm_hist.dat.bz2"
            dest = dyn_orblib._pid_tmp(f"{tmp_path}/datfil/set{si}_pm_hist_1.dat")
            lib._decompress_held(src, dest)
            read([si], **kw)
            lib._remove_unless_held(dest)
    lib.read_vel_histograms = read_vel_histograms


def _solve(tmp_path, hold=True):
    fake, lib, *_ = _toy(tmp_path, solver="admm", gram=True,
                         d_tol=-1.0, f_tol=0.0)  # extra pass forced
    fake.stream_reads, fake.gram_aperture_group = True, 2
    _archive_lib(lib, tmp_path, hold)
    return fake, lib


def _temps(tmp_path):
    return glob.glob(f"{tmp_path}/datfil/*.dat")


def test_each_archive_decompressed_once_per_solve(tmp_path):
    fake, lib = _solve(tmp_path)
    fake._solve_relative(lib)
    assert lib.n_reads > 3 and lib.count
    assert set(lib.count.values()) == {1}, lib.count
    assert not _temps(tmp_path) and lib._held is None


def test_without_hold_archives_are_decompressed_per_read(tmp_path):
    fake, lib = _solve(tmp_path, hold=False)
    fake._solve_relative(lib)
    assert max(lib.count.values()) > 1


def test_temp_files_removed_after_an_exception(tmp_path):
    fake, lib = _solve(tmp_path)
    orig = fake._relative_qp
    n = []

    def boom(prob, warm=None):
        n.append(1)
        if len(n) == 2:
            assert _temps(tmp_path), "held files should exist mid-solve"
            raise KeyError("boom")
        return orig(prob, warm=warm)
    fake._relative_qp = boom
    with pytest.raises(KeyError):
        fake._solve_relative(lib)
    assert not _temps(tmp_path) and lib._held is None
    assert np.all(np.isfinite(fake.pm_moments["hst"]["n"]))  # state unharmed

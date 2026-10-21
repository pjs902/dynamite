"""lbzip2 (de)compression, process-unique temp names, decompress-once.

Uses the small orbit library of comparison_20260617_124144 (1d LOSVD
histograms only). Run with and without lbzip2 on the PATH: the lbzip2-specific
cases skip when it is absent and the fallback cases run either way."""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'comparison_20260617_124144')
HAVE_LBZIP2 = shutil.which('lbzip2') is not None
pytestmark = pytest.mark.skipif(not os.path.isdir(RUN), reason='needs the comparison run')


@pytest.fixture
def lib(monkeypatch):
    import dynamite as dyn
    monkeypatch.chdir(RUN)
    # no density cache files: leave the shared test library as found
    monkeypatch.setattr(dyn.orblib.LegacyOrbitLibrary, '_qgrid_cache_save',
                        lambda *a, **k: None)
    monkeypatch.setattr(dyn.orblib.LegacyOrbitLibrary, '_qgrid_cache_load',
                        lambda *a, **k: None)
    c = dyn.config_reader.Configuration('config_gridwalk.yaml')
    model = c.all_models.get_model_from_row(0)
    return model.get_orblib()


def temps(lib):
    return glob.glob(lib.mod_dir + 'datfil/*_p[0-9]*.dat')


def test_bzip2_cmd(monkeypatch):
    from dynamite import orblib
    monkeypatch.setattr(orblib.shutil, 'which', lambda n: None)
    assert orblib._bzip2_cmd(True) == 'bunzip2 -c'
    assert orblib._bzip2_cmd() == 'bzip2 -c'
    monkeypatch.setattr(orblib.shutil, 'which', lambda n: '/x/lbzip2')
    monkeypatch.setenv('DYNAMITE_BZIP2_THREADS', '3')
    assert orblib._bzip2_cmd(True) == 'lbzip2 -d -c -n 3'
    assert orblib._bzip2_cmd() == 'lbzip2 -c -n 3'


@pytest.mark.parametrize('use_lbzip2', [True, False])
def test_truncated_archive_raises_and_good_one_roundtrips(
        lib, monkeypatch, tmp_path, use_lbzip2):
    from dynamite import orblib
    if use_lbzip2 and not HAVE_LBZIP2:
        pytest.skip('lbzip2 not on PATH')
    if not use_lbzip2:
        monkeypatch.setattr(orblib.shutil, 'which', lambda n: None)
    raw = np.random.default_rng(0).random(300000).tobytes()
    (tmp_path / 'a').write_bytes(raw)
    subprocess.run(f'bzip2 -c {tmp_path}/a > {tmp_path}/a.bz2', shell=True, check=True)
    lib._decompress(str(tmp_path / 'a.bz2'), str(tmp_path / 'out'))
    assert (tmp_path / 'out').read_bytes() == raw
    blob = (tmp_path / 'a.bz2').read_bytes()
    (tmp_path / 'bad.bz2').write_bytes(blob[: len(blob) // 2])
    with pytest.raises(RuntimeError, match='cannot decompress'):
        lib._decompress(str(tmp_path / 'bad.bz2'), str(tmp_path / 'out2'))


def test_pid_names_and_walk_index_name():
    from dynamite import orblib
    t = orblib._pid_tmp('datfil/orblib_pm_hist_2.75.dat')
    assert t == f'datfil/orblib_pm_hist_2.75_p{os.getpid()}.dat'
    assert orblib._walk_cache_path(t) == 'datfil/orblib_pm_hist_2.75.dat.walk.npz'


def test_two_processes_read_one_datfil(lib):
    code = (
        "import sys, numpy as np, dynamite as dyn\n"
        "dyn.orblib.LegacyOrbitLibrary._qgrid_cache_save = lambda *a, **k: None\n"
        "dyn.orblib.LegacyOrbitLibrary._qgrid_cache_load = lambda *a, **k: None\n"
        "c = dyn.config_reader.Configuration('config_gridwalk.yaml')\n"
        "o = c.all_models.get_model_from_row(0).get_orblib()\n"
        "o.read_vel_histograms()\n"
        "print(sum(float(h.y.sum()) for h in o.vel_histograms))\n")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE))
    ps = [subprocess.Popen([sys.executable, '-c', code], cwd=RUN, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          for _ in range(2)]
    outs = [p.communicate() for p in ps]
    assert all(p.returncode == 0 for p in ps), outs
    assert outs[0][0] == outs[1][0] and outs[0][0].strip()
    assert not temps(lib)


def test_hold_decompresses_once_and_cleans_up(lib, monkeypatch):
    import dynamite as dyn
    calls = []
    orig = dyn.orblib.LegacyOrbitLibrary._decompress
    monkeypatch.setattr(dyn.orblib.LegacyOrbitLibrary, '_decompress',
                        lambda self, s, d: (calls.append(s), orig(self, s, d))[1])
    lib.read_vel_histograms()
    ref = [h.y.copy() for h in lib.vel_histograms]
    assert len([s for s in calls if 'losvd_hist' in s]) == 2  # tube + box
    calls.clear()
    with lib.hold_decompressed():
        for _ in range(3):
            lib.read_vel_histograms()
            for r, h in zip(ref, lib.vel_histograms):
                np.testing.assert_array_equal(r, h.y)
        assert temps(lib), 'held files should still exist inside the context'
    assert len([s for s in calls if 'losvd_hist' in s]) == 2
    assert not temps(lib)
    # forced exception inside the context: still nothing left
    with pytest.raises(KeyError):
        with lib.hold_decompressed():
            lib.read_vel_histograms()
            raise KeyError('boom')
    assert not temps(lib)
    assert lib._held is None


def test_hold_keys_on_destination_too(lib, monkeypatch, tmp_path):
    import dynamite as dyn
    calls = []
    monkeypatch.setattr(dyn.orblib.LegacyOrbitLibrary, '_decompress',
                        lambda self, s, d: (calls.append((s, d)),
                                            self._decompressed.append(os.path.abspath(d)))[0])
    a, b = str(tmp_path / 'a.dat'), str(tmp_path / 'b.dat')
    with lib.hold_decompressed():
        lib._decompress_held('x.bz2', a)
        lib._decompress_held('x.bz2', a)
        lib._decompress_held('x.bz2', b)  # same src, another dest (e.g. ml)
        assert calls == [('x.bz2', a), ('x.bz2', b)]
        assert set(lib._held.values()) == {a, b}

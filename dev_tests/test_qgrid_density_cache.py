"""read_vel_histograms must give identical results with and without the
qgrid density cache (datfil/*_qgrid.dat.bz2.dens.npz), and the cached read
must not decompress the qgrid file. Uses a small existing test orbit library,
with the config's absolute output_directory and with a relative one (the
cache used to be silently never written for a relative one)."""
import glob
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'comparison_20260617_124144')
CFG = 'config_gridwalk.yaml'
CFG_REL = 'config_gridwalk_relout.yaml'


@pytest.mark.skipif(not os.path.isdir(RUN), reason='needs the comparison run')
@pytest.mark.parametrize('relative_out', [False, True])
def test_qgrid_density_cache(monkeypatch, relative_out):
    import dynamite as dyn
    monkeypatch.chdir(RUN)
    cfg = CFG
    if relative_out:
        with open(CFG) as f:
            txt = f.read()
        old = f'output_directory: {RUN}/gridwalk/'
        assert old in txt
        with open(CFG_REL, 'w') as f:
            f.write(txt.replace(old, 'output_directory: gridwalk/'))
        cfg = CFG_REL
    try:
        c = dyn.config_reader.Configuration(cfg)
        model = c.all_models.get_model_from_row(0)
        assert os.path.isabs(model.directory_noml) != relative_out
        caches = lambda: glob.glob(model.directory_noml + 'datfil/*_qgrid.dat.bz2.dens.npz')
        for f in caches():
            os.remove(f)
        ref = model.get_orblib()
        ref.read_vel_histograms()
        assert caches(), 'cache not written'
        calls = []
        orig = dyn.orblib.LegacyOrbitLibrary._decompress
        monkeypatch.setattr(dyn.orblib.LegacyOrbitLibrary, '_decompress',
                            lambda self, s, d: (calls.append(s), orig(self, s, d))[1])
        new = model.get_orblib()
        new.read_vel_histograms()
        assert not [s for s in calls if 'qgrid' in s], calls
        np.testing.assert_array_equal(ref.intrinsic_masses, new.intrinsic_masses)
        for a, b in zip(ref.vel_histograms, new.vel_histograms):
            np.testing.assert_array_equal(a.y, b.y)
    finally:
        for f in glob.glob('gridwalk/models/*/datfil/*_qgrid.dat.bz2.dens.npz'):
            os.remove(f)
        if relative_out and os.path.exists(CFG_REL):
            os.remove(CFG_REL)

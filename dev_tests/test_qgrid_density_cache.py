"""read_vel_histograms must give identical results with and without the
qgrid density cache (datfil/*_qgrid.dat.bz2.dens.npz), and the cached read
must not decompress the qgrid file. Uses a small existing test orbit library."""
import glob
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'comparison_20260617_124144')


@pytest.mark.skipif(not os.path.isdir(RUN), reason='needs the comparison run')
def test_qgrid_density_cache(monkeypatch):
    import dynamite as dyn
    monkeypatch.chdir(RUN)
    c = dyn.config_reader.Configuration('config_gridwalk.yaml')
    model = c.all_models.get_model_from_row(0)
    caches = lambda: glob.glob(model.directory_noml + 'datfil/*_qgrid.dat.bz2.dens.npz')
    for f in caches():
        os.remove(f)
    try:
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
        for f in caches():
            os.remove(f)

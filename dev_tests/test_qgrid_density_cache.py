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
RUN = HERE
CFG = 'user_test_config_ml.yaml'  # relative output_directory
CFG_ABS = 'user_test_config_ml_absout.yaml'


@pytest.mark.skipif(not os.path.isdir(os.path.join(HERE, 'NGC6278_output', 'models')),
                    reason='needs NGC6278_output (run test_nnls.py first)')
@pytest.mark.parametrize('relative_out', [False, True])
def test_qgrid_density_cache(monkeypatch, relative_out):
    import dynamite as dyn
    monkeypatch.chdir(RUN)
    cfg = CFG
    if not relative_out:
        with open(CFG) as f:
            txt = f.read()
        old = 'output_directory: "NGC6278_output/"'
        assert old in txt
        with open(CFG_ABS, 'w') as f:
            f.write(txt.replace(old, f'output_directory: "{RUN}/NGC6278_output/"'))
        cfg = CFG_ABS
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
        for f in glob.glob('NGC6278_output/models/*/datfil/*_qgrid.dat.bz2.dens.npz'):
            os.remove(f)
        if not relative_out and os.path.exists(CFG_ABS):
            os.remove(CFG_ABS)

"""orblib_settings: initial_conditions: actions (dynamite/start_grid.py) through DYNAMITE on the NGC6278 test galaxy: config validation,
the IC hook (no orbitstart), Fortran integration of the generated files, and ics_match_settings when the option or
the grid changes; finally the default path (orbitstart) in the same directory. Needs agama.
  cd dev_tests; python test_start_grid.py     (a few minutes, one core)
"""
import copy, os, shutil
import numpy as np
import dynamite as dyn
from dynamite import start_grid
D = os.path.dirname(os.path.abspath(__file__)) + "/start_grid_test"
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
with open(os.path.dirname(D) + "/user_test_config.yaml") as f:
    cfg = f.read()
for a, b in (("    nE: 2\n", "    nE: 6\n"), ("    nI3: 3\n", "    nI3: 4\n"),
             ("    dithering: 1\n", "    dithering: 3\n    initial_conditions: actions\n"),
             ('output_directory: "NGC6278_output/"', f'output_directory: "{D}/out/"')):
    assert cfg.count(a) == 1, a
    cfg = cfg.replace(a, b)
with open(D + "/config.yaml", "w") as f:
    f.write(cfg)
os.chdir(os.path.dirname(D))
c = dyn.config_reader.Configuration(D + "/config.yaml", reset_logging=True)
print("validated initial_conditions:", c.settings.orblib_settings["initial_conditions"])
for bad in (True, "start_grid", {"row_law": "jr"}):
    try:
        start_grid.validate({"initial_conditions": bad})
        raise AssertionError(f"{bad!r} accepted")
    except ValueError as e:
        print("rejected:", e)
start_grid.validate({})
parset = c.parspace.get_parset()
m = dyn.model.Model(config=c, parset=parset, directory="orblib_000_000/ml_test/")
m.setup_directories()
ob = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=m.directory_noml, parset=parset)
ob.get_orblib()
dat = m.directory_noml + "datfil/"
assert os.path.isfile(dat + "tube_box_done") and os.path.isfile(dat + start_grid.SIDECAR)
assert not os.path.isfile(dat + "orbstart.log"), "orbitstart ran"
assert ob.ics_match_settings()
rows = np.loadtxt(dat + "begin.dat", skiprows=1)
print("begin.dat", open(dat + "begin.dat").readline().strip(), "rows", len(rows))
log = open(dat + "orblib.log").read()
print("orblib.log mentions", log.count("Time spent one orbit"), "orbit timings")
s = ob.settings
for label, change in (("initial_conditions default", lambda t: t.update(initial_conditions="default")),
                      ("no initial_conditions", lambda t: t.pop("initial_conditions")),
                      ("nI3 5", lambda t: t.update(nI3=5))):
    keep = copy.deepcopy(s)
    change(s)
    print(f"ics_match_settings after {label}: {ob.ics_match_settings()}")
    assert not ob.ics_match_settings()
    s.clear(); s.update(keep)
assert ob.ics_match_settings()
# default path: without initial_conditions the same directory gets orbitstart's files and no sidecar is consulted
s.pop("initial_conditions")
for f in ("begin.dat", "beginbox.dat", start_grid.SIDECAR):
    os.remove(dat + f)
ob.get_orbit_ics()
assert ob.ics_match_settings() and os.path.isfile(dat + "orbstart.log")
print("orbitstart path: begin.dat", open(dat + "begin.dat").readline().strip(), "match", ob.ics_match_settings())
print("SMOKE PASSED")

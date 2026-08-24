"""Single-model weight-solve array task (spec section 5.3).

One Slurm array task runs one ``solve_one.py --model-dir ...`` invocation.
Mirrors ModelInnerIterator's weights pass: get_orblib() short-circuits on
the sentinel, then get_weights() writes orbit_weights.ecsv with chi2 meta.
"""

import sys

from .task_model import task_main


def _solve(mod):
    orblib = mod.get_orblib()  # sentinel short-circuit
    if getattr(orblib, "vel_histograms", None) is None:
        # resume path: library built by an earlier task/process;
        # chi2_kinmap needs in-memory histograms
        orblib.read_vel_histograms()
    mod.get_weights(orblib)
    # get_weights already set these from the solver's return value; the ecsv
    # is the same numbers via an extra NFS read, on the one path where a
    # partial flush is a known hazard
    return {
        "chi2_tot": float(mod.chi2),
        "chi2_kin": float(mod.kinchi2),
        "chi2_kinmap": float(mod.kinmapchi2),
    }


def main(argv=None):
    return task_main("one-model weight solve", _solve, argv)


if __name__ == "__main__":
    sys.exit(main())

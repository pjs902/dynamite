"""Single-model weight-solve array task (spec section 5.3).

One Slurm array task runs one ``solve_one.py --model-dir ...`` invocation.
Mirrors ModelInnerIterator's weights pass: get_orblib() short-circuits on
the sentinel, then get_weights() writes orbit_weights.ecsv with chi2 meta.
"""

import argparse
import json
import os
import sys

from .task_model import build_model


def main(argv=None):
    ap = argparse.ArgumentParser(description="one-model weight solve")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(json.dumps({"model_dir": args.model_dir, "dry_run": True}))
        return 0

    try:
        import dynamite as dyn

        c, mod = build_model(args.config, args.model_dir)
        cwd = os.getcwd()
        try:
            mod.setup_directories()
            orblib = mod.get_orblib()          # sentinel short-circuit
            if getattr(orblib, "vel_histograms", None) is None:
                # resume path: library built by an earlier task/process;
                # chi2_kinmap needs in-memory histograms
                orblib.read_vel_histograms()
            mod.get_weights(orblib)
        finally:
            os.chdir(cwd)
        from astropy.io import ascii

        meta = ascii.read(os.path.join(mod.directory, dyn.constants.weight_file)).meta
        print(
            json.dumps(
                {
                    "model_dir": args.model_dir,
                    "chi2_tot": float(meta["chi2_tot"]),
                    "chi2_kin": float(meta["chi2_kin"]),
                    "chi2_kinmap": float(meta["chi2_kinmap"]),
                }
            )
        )
        return 0
    except Exception as e:  # task boundary: report, don't die
        import traceback

        traceback.print_exc()
        print(
            json.dumps({"error": repr(e), "model_dir": args.model_dir}), file=sys.stderr
        )
        return 3


if __name__ == "__main__":
    sys.exit(main())

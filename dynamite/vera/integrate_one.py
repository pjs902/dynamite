"""Single-library integration array task (spec section 5.2).

One process per chunk-family pair runs here, packed 12-wide per node by
integrate_package.sh. All orbit-integration machinery lives in
LegacyOrbitLibrary.get_orbit_library(); this wrapper only locates the row
and enforces cwd discipline.
"""

import argparse
import json
import os
import sys

from .task_model import build_model


def main(argv=None):
    ap = argparse.ArgumentParser(description="one-library orbit integration")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args(argv)

    try:
        _, mod = build_model(args.config, args.model_dir)
        cwd = os.getcwd()
        try:
            mod.setup_directories()
            mod.get_orblib()  # sentinel short-circuit applies here
        finally:
            os.chdir(cwd)
        print(json.dumps({"model_dir": args.model_dir, "integrated": True}))
        return 0
    except Exception as e:  # task boundary
        import traceback
        traceback.print_exc()
        print(
            json.dumps({"error": repr(e), "model_dir": args.model_dir}), file=sys.stderr
        )
        return 3


if __name__ == "__main__":
    sys.exit(main())

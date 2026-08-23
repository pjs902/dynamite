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


def main(argv=None):
    ap = argparse.ArgumentParser(description="one-library orbit integration")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args(argv)

    try:
        import dynamite as dyn

        c = dyn.config_reader.Configuration(args.config, reset_logging=True)
        idx = _find_row(c, args.model_dir)
        if idx is None:
            print(f"model dir {args.model_dir!r} not in table", file=sys.stderr)
            return 1
        mod = c.all_models.get_model_from_row(idx)
        cwd = os.getcwd()
        try:
            mod.get_orblib()  # sentinel short-circuit applies here
        finally:
            os.chdir(cwd)
        print(json.dumps({"model_dir": args.model_dir, "integrated": True}))
        return 0
    except Exception as e:  # task boundary
        print(
            json.dumps({"error": repr(e), "model_dir": args.model_dir}), file=sys.stderr
        )
        return 3


def _find_row(config, model_dir):
    for i, row in enumerate(config.all_models.table):
        if str(row["directory"]).strip("/") == model_dir.strip("/"):
            return i
    return None


if __name__ == "__main__":
    sys.exit(main())

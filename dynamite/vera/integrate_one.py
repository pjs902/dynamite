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


def _build_model(config_path, model_dir):
    """Configuration with a PRIVATE all-table + Model at the given dir.

    The campaign yaml's output tree is shared, but this worker's
    all_models_file is unique per task: Configuration's janitor then
    operates on an empty private table and cannot touch sibling work.
    Parset comes from vera_parset.json written by the driver.
    """
    import uuid

    import yaml as _yaml

    import dynamite as dyn
    from astropy.table import Column

    with open(config_path) as f:
        cfgd = _yaml.safe_load(f)
    io = cfgd.setdefault("io_settings", {})
    outroot = os.path.abspath(io.get("output_directory", "."))
    io["all_models_file"] = f"all_models_task_{uuid.uuid4().hex[:8]}.ecsv"
    task_cfg_path = os.path.join(outroot,
                                 f"vera_task_{uuid.uuid4().hex[:8]}.yaml")
    os.makedirs(os.path.dirname(task_cfg_path), exist_ok=True)
    with open(task_cfg_path, "w") as f:
        _yaml.safe_dump(cfgd, f, sort_keys=False)

    cwd = os.getcwd()
    try:
        c = dyn.config_reader.Configuration(task_cfg_path,
                                            reset_logging=True)
    finally:
        os.chdir(cwd)

    pfile = os.path.join(outroot, "models", model_dir, "vera_parset.json")
    with open(pfile) as f:
        payload = json.load(f)
    names = payload["par_names"]
    tbl = c.all_models.table
    from astropy.table import Table
    row = Table()
    for n in names:
        row[n] = [payload["values"][n]]
    mod = dyn.model.Model(config=c, parset=row[0], directory=model_dir)

    # Register into the private table: post-integration machinery
    # (intrinsic masses) looks the model up via get_model_from_parset.
    import numpy as _np

    defaults = {"chi2": _np.nan, "kinchi2": _np.nan,
                "kinmapchi2": _np.nan, "time_modified": "",
                "orblib_done": False, "weights_done": False,
                "all_done": False, "which_iter": 0,
                "directory": model_dir}
    rowd = {}
    for cn in tbl.colnames:
        if cn in payload["values"]:
            rowd[cn] = payload["values"][cn]
        elif cn in defaults:
            rowd[cn] = defaults[cn]
        elif "done" in cn:
            rowd[cn] = False
        elif isinstance(tbl[cn].dtype, _np.dtype("U")) if hasattr(
                tbl[cn].dtype, "kind") and tbl[cn].dtype.kind == "U" else False:
            rowd[cn] = ""
        else:
            rowd[cn] = _np.nan
    if len(tbl) == 0:
        try:
            tbl.add_row(rowd)
        except Exception:
            pass
    return c, mod


def main(argv=None):
    ap = argparse.ArgumentParser(description="one-library orbit integration")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args(argv)

    try:
        import dynamite as dyn

        c, mod = _build_model(args.config, args.model_dir)
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

"""Shared worker-side model construction for the array tasks.

Both entry points (integrate_one, solve_one) need the same thing: a
Configuration whose all-models table is PRIVATE to this task, plus a Model
pointed at one directory. The private table matters -- every Configuration
init runs update_model_table(), whose janitor would otherwise delete
sibling rows that other tasks are still working on.
"""

import json
import os
import uuid


def build_model(config_path, model_dir):
    """(Configuration, Model) for `model_dir`, with a per-task table.

    The parset comes from vera_parset.json, written by the driver: workers
    are pure functions of (config, parset) and never read the shared table.
    """
    import numpy as np
    import yaml
    from astropy.table import Table

    import dynamite as dyn

    with open(config_path) as f:
        cfgd = yaml.safe_load(f)
    io = cfgd.setdefault("io_settings", {})
    outroot = os.path.abspath(io.get("output_directory", "."))
    io["all_models_file"] = f"all_models_task_{uuid.uuid4().hex[:8]}.ecsv"
    task_cfg_path = os.path.join(outroot, f"vera_task_{uuid.uuid4().hex[:8]}.yaml")
    os.makedirs(os.path.dirname(task_cfg_path), exist_ok=True)
    with open(task_cfg_path, "w") as f:
        yaml.safe_dump(cfgd, f, sort_keys=False)

    cwd = os.getcwd()
    try:
        c = dyn.config_reader.Configuration(task_cfg_path, reset_logging=True)
    finally:
        os.chdir(cwd)

    with open(os.path.join(outroot, "models", model_dir, "vera_parset.json")) as f:
        payload = json.load(f)
    values = payload["values"]

    row = Table()
    for n in payload["par_names"]:
        row[n] = [values[n]]
    mod = dyn.model.Model(config=c, parset=row[0], directory=model_dir)

    # Register into the private table: post-integration machinery (intrinsic
    # masses) looks the model up via get_model_from_parset.
    tbl = c.all_models.table
    # only what the dtype-based fallback below cannot get right: nan is not a
    # valid int, and the directory is this task's identity
    defaults = {"which_iter": 0, "directory": model_dir}
    rowd = {}
    for cn in tbl.colnames:
        if cn in values:
            rowd[cn] = values[cn]
        elif cn in defaults:
            rowd[cn] = defaults[cn]
        elif tbl[cn].dtype.kind in "US":
            rowd[cn] = ""
        elif tbl[cn].dtype.kind == "b":
            rowd[cn] = False
        else:
            rowd[cn] = np.nan
    if not any(str(r["directory"]) == model_dir for r in tbl):
        # keyed on the directory, not on emptiness: Configuration init can
        # leave rows behind, and the model must still be findable by
        # get_model_from_parset after integration
        tbl.add_row(rowd)
    return c, mod

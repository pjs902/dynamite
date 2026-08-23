#!/usr/bin/env python3
"""Minimal REAL end-to-end run: Fortran orbit integration -> adelie weight
solve -> GP training-data extraction, driven by BayesOptGenerator.

Tiny orblib (nE=7/nI2=5/nI3=5, sampling=2000), single GaussHermite set,
12-model budget. This exercises the full production code path that dummy
tests cannot reach (Fortran I/O contract included).

Run: python dev_tests/test_real_minirun.py
"""

import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/pesmith/research/dynamite")

_stub = sys.modules.pop("dynamite", None)
import dynamite as _dyn  # noqa: E402

sys.modules["dynamite"] = _dyn
import importlib  # noqa: E402

config_reader = importlib.import_module("dynamite.config_reader")
model_iterator = importlib.import_module("dynamite.model_iterator")

BASE = yaml.safe_load(open(os.path.join(HERE, "bayesopt_qml_modelinner.yaml")))

cfg = {
    "system_attributes": BASE["system_attributes"],
    "system_components": {},
    "system_parameters": BASE["system_parameters"],
    "orblib_settings": {
        "nE": 7,
        "logrmin": -2.0,
        "logrmax": 3.7,
        "nI2": 5,
        "nI3": 5,
        "dithering": 1,
        "quad_nr": 10,
        "quad_nth": 6,
        "quad_nph": 6,
        "orbital_periods": 200,
        "sampling": 2000,
        "starting_orbit": 1,
        "number_orbits": -1,
        "accuracy": "1.0d-3",
        "random_seed": 4242,
    },
    "weight_solver_settings": {
        "type": "NNLS",
        "nnls_solver": "adelie",
        "nnls_dtype": "float64",
        "number_GH": 2,
        "GH_sys_err": "0.0 0.0 0.0 0.0 0.3 0.3 0.6 0.6",
        "lum_intr_rel_err": 0.01,
        "sb_proj_rel_err": 0.02,
    },
    "parameter_space_settings": {
        "generator_type": "BayesOptGenerator",
        "which_chi2": "kinchi2",
        "generator_settings": {
            "batch_size": 4,
            "n_orblib_configs": 2,
            "n_ml_per_config": 2,
            "n_initial_random": 4,
        },
        "stopping_criteria": {"min_delta_chi2_abs": -1e6, "n_max_mods": 8, "n_max_iter": 10},
    },
    "io_settings": {
        "input_directory": os.path.join(HERE, "NGC6278_input") + "/",
        "output_directory": None,  # filled below
        "all_models_file": "all_models.ecsv",
    },
    "multiprocessing_settings": {
        "modeliterator": "ModelInnerIterator",
        "total_cores": min(8, os.cpu_count() or 4),
        "ncpus": min(8, os.cpu_count() or 4),
    },
    "legacy_settings": {"directory": "default"},
}

# stars component: same shape params as the qml reference config, ONE
# GaussHermite kinematic set from NGC6278_input
stars = BASE["system_components"]["stars"]
kin = stars["kinematics"]
gh_key = next(k for k in kin if kin[k]["type"] == "GaussHermite")
stars["kinematics"] = {gh_key: kin[gh_key]}
cfg["system_components"]["stars"] = stars
cfg["system_components"]["bh"] = BASE["system_components"]["bh"]


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["io_settings"]["output_directory"] = tmpdir + "/"
        cfg_path = os.path.join(tmpdir, "mini_real.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        c = config_reader.Configuration(cfg_path, reset_logging=False)
        model_iterator.ModelIterator(c, do_dummy_run=False, plots=False)

        table = c.all_models.table
        n_done = int(sum(table["all_done"]))
        kinchi2 = [float(v) for v, d in zip(table["kinchi2"], table["all_done"]) if d]
        print(
            f"\nreal mini-run: {len(table)} rows, {n_done} done, kinchi2 range [{min(kinchi2):.1f}, {max(kinchi2):.1f}]"
        )
        assert n_done >= 8, f"only {n_done} models completed"
        assert len(set(kinchi2)) > 3, "chi2 values suspiciously identical"
        print("REAL MINIRUN PASSED")


if __name__ == "__main__":
    main()

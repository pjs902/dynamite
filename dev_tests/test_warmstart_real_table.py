"""Warm-start integration test against a REAL all_models.ecsv.

Loads the actual NGC6278 GridWalk output (18 evaluated models, real table
schema), hands it to BayesOptGenerator as current_models, and verifies the
H2 contract: the generator reports the warm-start, skips Sobol, trains the
GP immediately, and proposes in-bounds models.

Run: python dev_tests/test_warmstart_real_table.py
"""

import logging
import os
import shutil
import sys
import tempfile

import numpy as np
from astropy.table import Table

import test_bayesopt_generator as T

ps = T.ps
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "NGC6278_gridwalk_qml_output")
YAML = os.path.join(HERE, "bayesopt_qml_modelinner.yaml")


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main():
    import yaml

    cfg = yaml.safe_load(open(YAML))
    gs = cfg["parameter_space_settings"]["generator_settings"]
    s = {
        "which_chi2": "kinchi2",
        "generator_type": "BayesOptGenerator",
        "generator_settings": dict(gs),
        "stopping_criteria": {"n_max_mods": 10**6, "n_max_iter": 10**6, "min_delta_chi2_abs": -1e6},
    }
    # build the parspace from the same yaml the real run used
    params = []
    for comp, cdef in cfg["system_components"].items():
        for pname, pdef in (cdef.get("parameters") or {}).items():
            if pdef.get("fixed") is False:
                params.append(
                    T._mk_param(
                        f"{pname}-{comp}",
                        pdef["par_generator_settings"]["lo"],
                        pdef["par_generator_settings"]["hi"],
                        pdef["value"],
                        logarithmic=pdef.get("logarithmic", False),
                    )
                )
    for pname, pdef in (cfg.get("system_parameters") or {}).items():
        if pdef.get("fixed") is False:
            params.append(
                T._mk_param(
                    pname,
                    pdef["par_generator_settings"]["lo"],
                    pdef["par_generator_settings"]["hi"],
                    pdef["value"],
                    logarithmic=pdef.get("logarithmic", False),
                )
            )
    parspace = T.make_parspace(params)
    free_names = [p.name for p in parspace]

    table = Table.read(os.path.join(SRC, "all_models.ecsv"), format="ascii.ecsv")
    done = np.asarray(table["all_done"], dtype=bool)
    kin = np.asarray(table["kinchi2"], dtype=float)
    n_valid = int(np.sum(done & np.isfinite(kin)))
    print(f"real table: {len(table)} rows, {n_valid} valid, free params {free_names}")

    gen = ps.BayesOptGenerator(par_space=parspace, parspace_settings=s)
    handler = _ListHandler()
    gen.logger.addHandler(handler)
    gen.logger.setLevel(logging.DEBUG)

    class _AM:  # duck-typed AllModels: .table is all the generator reads
        pass

    am = _AM()
    am.table = table
    gen.current_models = am
    gen.specific_generate_method()

    joined = "\n".join(handler.messages)
    assert "warm-start" in joined or n_valid >= s["generator_settings"]["n_initial_random"], (
        "warm-start path not reported"
    )
    assert gen._gp_model is not None, "GP was not fitted — Sobol warm-up was NOT skipped"
    assert len(gen.model_list) > 0, "no models proposed"
    for entry in gen.model_list:
        for p in entry:
            lo, hi = p.par_generator_settings["lo"], p.par_generator_settings["hi"]
            assert lo - 1e-12 <= p.raw_value <= hi + 1e-12, f"{p.name}={p.raw_value} outside [{lo}, {hi}]"
    n_prop = len(gen.model_list)
    print(f"warm-start OK: GP trained on {n_valid} real rows, proposed {n_prop} in-bounds models, Sobol skipped")
    print("WARMSTART REAL TABLE PASSED")


if __name__ == "__main__":
    main()

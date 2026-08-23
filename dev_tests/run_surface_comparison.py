#!/usr/bin/env python3
"""Run GridWalk vs BayesOptGenerator on realistic chi2 landscapes.

Saves one .npz per (landscape, arm) with the full evaluation trajectory
into dev_tests/surface_runs/. Consumed by plot_surface_comparison.py.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_bayesopt_generator as T  # noqa: E402
import chi2_landscapes as L  # noqa: E402

ps = T.ps
BUDGET = 150


def make_parspace(land):
    """Parameter space matching the landscape axes (production-style)."""
    spec = {
        "ml": dict(log=False),
        "q-stars": dict(log=False),
        "p-stars": dict(log=False),
        "m-bh": dict(log=True),
        "c-dh": dict(log=False),
        "f-dh": dict(log=True),
    }
    steps = {
        "ml": (0.2, 0.1),
        "q-stars": (0.04, 0.02),
        "p-stars": (0.01, 0.005),
        "m-bh": (0.1, 0.05),
        "c-dh": (1.0, 0.5),
        "f-dh": (0.1, 0.1),
    }
    params = []
    for name, (lo, hi) in zip(land.names, land.bounds):
        key = name if "-" in name or name == "ml" else name
        # map display names to dynamite parameter names
        dname = {"log10 MBH": "m-bh", "c-dh": "c-dh", "log f-dh": "f-dh", "ml": "ml", "q": "q-stars", "p": "p-stars"}[
            name
        ]
        st, mst = steps[dname]
        params.append(T._mk_param(dname, lo, hi, 0.5 * (lo + hi), logarithmic=spec[dname]["log"]))
        params[-1].par_generator_settings.update({"step": st, "minstep": mst})
    return T.make_parspace(params)


def row_to_raw(parspace, am, i):
    """Table rows store PAR values (10^raw for log params) -> raw."""
    vals = [float(am.table[par.name][i]) for par in parspace]
    return parspace.get_raw_value_from_param_value(vals)


def fill(am, land, parspace):
    kin = np.asarray(am.table["kinchi2"], dtype=float)
    for i in np.where(~np.isfinite(kin))[0]:
        x = row_to_raw(parspace, am, i)
        am.table["chi2"][i] = land(x)[0]
        am.table["kinchi2"][i] = am.table["chi2"][i]
        am.table["all_done"][i] = True


def run_arm(land, arm, seed=0):
    parspace = make_parspace(land)
    am = T.MockAllModels([p.name for p in parspace])
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    stalled_at = None
    if arm == "gridwalk":
        gen = ps.GridWalk(
            par_space=parspace,
            parspace_settings={
                "which_chi2": "kinchi2",
                "generator_type": "GridWalk",
                "generator_settings": {"threshold_del_chi2_as_frac_of_sqrt2nobs": 0.1},
                "stopping_criteria": {"n_max_mods": 10**6, "n_max_iter": 10**6, "min_delta_chi2_abs": -1e6},
            },
        )
    else:
        s = {
            "which_chi2": "kinchi2",
            "generator_type": "BayesOptGenerator",
            "generator_settings": {
                "batch_size": 8,
                "n_orblib_configs": 4,
                "n_ml_per_config": 2,
                "n_initial_random": 8,
                "exploration_schedule": "annealed",
                "n_annealed_members": 2,
                "trust_region": True,
                "discretize_non_ml_params": False,
            },
            "stopping_criteria": {"n_max_mods": 10**6, "n_max_iter": 10**6, "min_delta_chi2_abs": -1e6},
        }
        gen = ps.BayesOptGenerator(par_space=parspace, parspace_settings=s)

    while len(am.table) < BUDGET:
        status = gen.generate(current_models=am)
        fill(am, land, parspace)
        if status.get("n_new_models", 1) == 0:
            stalled_at = len(am.table)
            break
    seconds = time.perf_counter() - t0
    X = np.array([row_to_raw(parspace, am, i) for i in range(len(am.table))])
    y = np.asarray(am.table["kinchi2"], dtype=float)
    return {"X": X, "y": y, "seconds": seconds, "stalled_at": stalled_at}


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surface_runs")
    os.makedirs(outdir, exist_ok=True)
    for land in [L.QML_RIDGE, L.PRODUCTION, L.HALO_BANANA]:
        for arm in ["gridwalk", "bayesopt"]:
            r = run_arm(land, arm)
            fn = os.path.join(outdir, f"{land.name}_{arm}.npz")
            np.savez_compressed(
                fn,
                X=r["X"],
                y=r["y"],
                seconds=r["seconds"],
                stalled_at=r["stalled_at"] or -1,
                names=np.array(land.names),
                threshold=land.threshold,
            )
            k = r["y"]
            print(
                f"{land.name:<14} {arm:<9} n={len(k):>3} "
                f"best={np.nanmin(k):7.1f} thr@{land.threshold:.0f} "
                f"{r['seconds']:6.1f}s" + (" STALLED" if r["stalled_at"] else "")
            )


if __name__ == "__main__":
    main()

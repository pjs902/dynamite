#!/usr/bin/env python3
"""Dummy-mode comparison matrix: GridWalk vs BayesOptGenerator variants.

The branch is only useful if it beats the incumbent search; this matrix
reports models-to-threshold and best chi2 at fixed budget for each arm on
the same synthetic landscape (see test_vs_gridwalk.py for the hard gate).

Usage:
    python run_ablation.py [--quick]
"""

import argparse
import copy
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_bayesopt_generator as T  # noqa: E402

BayesOptGenerator = T.ps.BayesOptGenerator
GridWalk = T.ps.GridWalk

BUDGET = 120
THRESHOLD = 5796.0


def landscape(ml, q):
    return (
        30.0 * (ml - 5.12) ** 2 / 0.8
        + 400.0 * (q - 0.62) ** 2 / 0.2
        + 8.0 * np.sin(12.0 * ml) * np.cos(9.0 * q)
        + 5800.0
    )


def make_parspace_qml():
    ml = T._mk_param("ml", 4.0, 6.0, 5.0)
    ml.par_generator_settings.update({"step": 0.2, "minstep": 0.1})
    q = T._mk_param("q-stars", 0.3, 0.9, 0.6)
    q.par_generator_settings.update({"step": 0.04, "minstep": 0.02})
    return T.make_parspace([ml, q])


def fill_chi2(am):
    kin = np.asarray(am.table["kinchi2"], dtype=float)
    for i in np.where(~np.isfinite(kin))[0]:
        c = landscape(float(am.table["ml"][i]), float(am.table["q-stars"][i]))
        am.table["chi2"][i] = c
        am.table["kinchi2"][i] = c
        am.table["all_done"][i] = True


def best_and_count(am):
    kin = np.asarray(am.table["kinchi2"], dtype=float)
    ok = np.isfinite(kin)
    if not np.any(ok):
        return float("inf"), None
    hit = ok & (kin <= THRESHOLD)
    cnt = int(np.argmax(hit) + 1) if np.any(hit) else None
    return float(np.min(kin[ok])), cnt


def drive(gen, am, seed_history=None):
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    t0 = time.perf_counter()
    stalled = False
    if seed_history is not None:
        for row in seed_history:
            am.table.add_row(row)
        fill_chi2(am)
    while len(am.table) < BUDGET and not gen.status.get("stop", False):
        status = gen.generate(current_models=am)
        fill_chi2(am)
        if isinstance(gen, GridWalk) and status.get("n_new_models", 1) == 0:
            stalled = True
            break
    seconds = time.perf_counter() - t0
    best, cnt = best_and_count(am)
    n_fresh = len(am.table) - (len(seed_history) if seed_history else 0)
    return {"n_models": n_fresh, "best_chi2": best, "to_threshold": cnt, "seconds": seconds, "stalled": stalled}


def arm_gridwalk():
    gen = GridWalk(
        par_space=make_parspace_qml(),
        parspace_settings={
            "which_chi2": "kinchi2",
            "generator_type": "GridWalk",
            "generator_settings": {"threshold_del_chi2_as_frac_of_sqrt2nobs": 0.1},
            "stopping_criteria": {"n_max_mods": 10**6, "n_max_iter": 10**6, "min_delta_chi2_abs": -1e6},
        },
    )
    return drive(gen, T.MockAllModels(["ml", "q-stars"]))


def arm_bayesopt(name, extra, seed_history=None):
    s = copy.deepcopy(T._bo_settings())
    s["stopping_criteria"]["n_max_mods"] = 10**6
    s["stopping_criteria"]["min_delta_chi2_abs"] = -1e6
    s["generator_settings"].update(extra)
    if seed_history is not None:
        # warm start: skip the Sobol phase entirely
        s["generator_settings"]["n_initial_random"] = 0
    gen = BayesOptGenerator(par_space=make_parspace_qml(), parspace_settings=s)
    return drive(gen, T.MockAllModels(["ml", "q-stars"]), seed_history=seed_history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global BUDGET
    if args.quick:
        BUDGET = 40

    rng = np.random.default_rng(0)
    history = []
    for _ in range(30):
        mlv, qv = 4.0 + 2.0 * rng.random(), 0.3 + 0.6 * rng.random()
        c = landscape(mlv, qv)
        history.append([mlv, qv, c, c, float("nan"), "", True, True, True, 0, "d"])

    arms = [
        ("GridWalk", lambda: arm_gridwalk()),
        ("BO baseline (sobol)", lambda: arm_bayesopt("baseline", {})),
        (
            "BO R1+R2 (sobol)",
            lambda: arm_bayesopt("r1r2", {"exploration_schedule": "annealed", "n_annealed_members": 2}),
        ),
        (
            "BO R1-R4+snap (sobol)",
            lambda: arm_bayesopt(
                "full",
                {
                    "exploration_schedule": "annealed",
                    "n_annealed_members": 2,
                    "trust_region": True,
                    "discretize_non_ml_params": True,
                },
            ),
        ),
        (
            "BO R1-R4+snap (warm-start)",
            lambda: arm_bayesopt(
                "full-warm",
                {
                    "exploration_schedule": "annealed",
                    "n_annealed_members": 2,
                    "trust_region": True,
                    "discretize_non_ml_params": True,
                },
                seed_history=history,
            ),
        ),
    ]
    print(f"\n{'arm':<28} {'fresh*':>7} {'best_chi2':>10} {'to_thr':>7} {'gen_s':>7}   (*new models; warm-start arm excludes seeded history)")
    results = {}
    for name, fn in arms:
        r = fn()
        results[name] = r
        print(
            f"{name:<28} {r['n_models']:>7} {r['best_chi2']:>10.2f} "
            f"{str(r['to_threshold']):>7} {r['seconds']:>7.2f}" + ("  [stalled]" if r["stalled"] else "")
        )

    gw = results["GridWalk"]
    winners = {k: v for k, v in results.items() if k != "GridWalk"}
    assert any(v["to_threshold"] is not None for v in winners.values()), "no BayesOpt variant reached the threshold"
    assert (
        gw["to_threshold"] is None
        or min(v["to_threshold"] for v in winners.values() if v["to_threshold"] is not None) < gw["to_threshold"]
    ), "GridWalk matched or beat every BayesOpt variant — branch adds nothing here"
    print("\nABLATION GATES PASSED")


if __name__ == "__main__":
    main()

"""Head-to-head runtime comparison: BayesOptGenerator vs GridWalk.

The branch is only useful if it reaches a given chi2 threshold in FEWER
MODELS than the incumbent GridWalk search — at production scale each model
costs hours of orbit integration + weight solving, so models-to-threshold
is the honest runtime proxy. This script asserts that gate, and that the
GP's own overhead stays negligible next to a model's cost.

Run:  python dev_tests/test_vs_gridwalk.py
"""

import sys
import time

import numpy as np

import test_bayesopt_generator as T  # loads dynamite.parameter_space standalone

ps = T.ps
BayesOptGenerator = ps.BayesOptGenerator
GridWalk = ps.GridWalk

BUDGET = 120
# Landscape minimum is ~5792 (base 5800 minus ripple); the walk's STARTING
# point already sits near 5801, so the threshold must demand real progress
# into the basin, not just proximity to the initial guess.
THRESHOLD = 5796.0


def landscape(ml, q):
    """Anisotropic, rippled chi2 surface — stands in for NGC6278."""
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
    """Evaluate the landscape on rows the generator appended (dummy run)."""
    kin = np.asarray(am.table["kinchi2"], dtype=float)
    for i in np.where(~np.isfinite(kin))[0]:
        c = landscape(float(am.table["ml"][i]), float(am.table["q-stars"][i]))
        am.table["chi2"][i] = c
        am.table["kinchi2"][i] = c
        am.table["all_done"][i] = True


def best_and_count(am, threshold):
    """(best chi2 so far, model count when threshold first met)."""
    kin = np.asarray(am.table["kinchi2"], dtype=float)
    ok = np.isfinite(kin)
    if not np.any(ok):
        return float("inf"), None
    best = float(np.min(kin[ok]))
    hit = ok & (kin <= threshold)
    cnt = int(np.argmax(hit) + 1) if np.any(hit) else None
    return best, cnt


def run_gridwalk():
    np.random.seed(0)
    gen = GridWalk(
        par_space=make_parspace_qml(),
        parspace_settings={
            "which_chi2": "kinchi2",
            "generator_type": "GridWalk",
            "generator_settings": {"threshold_del_chi2_as_frac_of_sqrt2nobs": 0.1},
            "stopping_criteria": {"n_max_mods": 10**6, "n_max_iter": 10**6, "min_delta_chi2_abs": 0.001},
        },
    )
    am = T.MockAllModels(["ml", "q-stars"])
    t0 = time.perf_counter()
    stalled_at = None
    while len(am.table) < BUDGET:
        status = gen.generate(current_models=am)
        fill_chi2(am)
        if status.get("n_new_models", 1) == 0:
            stalled_at = len(am.table)  # walk cornered: no new proposals
            break
    seconds = time.perf_counter() - t0
    best, cnt = best_and_count(am, THRESHOLD)
    return {
        "seconds": seconds,
        "models_to_threshold": cnt,
        "best_chi2": best,
        "stalled_at": stalled_at,
        "n_models": len(am.table),
    }


def run_bayesopt():
    np.random.seed(0)
    import torch

    torch.manual_seed(0)
    s = T._bo_settings()
    s["stopping_criteria"]["n_max_mods"] = 10**6
    # Drive to the full budget: delta-stopping would halt early on unlucky
    # batches (same reason production configs use -1e6 during warm-up).
    s["stopping_criteria"]["min_delta_chi2_abs"] = -1e6
    gen = BayesOptGenerator(par_space=make_parspace_qml(), parspace_settings=s)
    am = T.MockAllModels(["ml", "q-stars"])
    t0 = time.perf_counter()
    stopped = False
    while len(am.table) < BUDGET and not stopped:
        status = gen.generate(current_models=am)
        fill_chi2(am)
        stopped = bool(status.get("stop"))
    seconds = time.perf_counter() - t0
    best, cnt = best_and_count(am, THRESHOLD)
    n = len(am.table)
    return {
        "seconds": seconds,
        "models_to_threshold": cnt,
        "best_chi2": best,
        "overhead_s_per_model": seconds / max(1, n),
        "n_models": n,
    }


def main():
    gw = run_gridwalk()
    bo = run_bayesopt()
    print(f"\n{'metric':<30} {'GridWalk':>12} {'BayesOpt':>12}")
    print(
        f"{'models to chi2<=%.0f' % THRESHOLD:<30} "
        f"{str(gw['models_to_threshold']):>12} "
        f"{str(bo['models_to_threshold']):>12}"
    )
    print(f"{'best chi2 @ budget':<30} {gw['best_chi2']:>12.2f} {bo['best_chi2']:>12.2f}")
    print(f"{'models run (of budget)':<30} {gw['n_models']:>12} {bo['n_models']:>12}")
    print(f"{'generator seconds':<30} {gw['seconds']:>12.2f} {bo['seconds']:>12.2f}")
    print(f"{'BO overhead s/model':<30} {'-':>12} {bo['overhead_s_per_model']:>12.3f}")
    if gw["stalled_at"] is not None:
        print(f"NOTE: GridWalk stalled (no new proposals) at {gw['stalled_at']} models")

    assert bo["models_to_threshold"] is not None, "BayesOpt never reached the threshold — branch is not useful"
    assert gw["models_to_threshold"] is None or bo["models_to_threshold"] < gw["models_to_threshold"], (
        f"BayesOpt reached threshold in {bo['models_to_threshold']} models "
        f"vs GridWalk {gw['models_to_threshold']} — no runtime win"
    )
    assert bo["overhead_s_per_model"] < 10.0, (
        "GP overhead per model must stay far below a real model's cost "
        f"(hours); got {bo['overhead_s_per_model']:.1f} s/model"
    )
    print(
        f"\nVS-GRIDWALK GATES PASSED: BayesOpt "
        f"{bo['models_to_threshold']} models vs GridWalk "
        f"{gw['models_to_threshold']} (threshold chi2 <= {THRESHOLD})"
    )


if __name__ == "__main__":
    main()

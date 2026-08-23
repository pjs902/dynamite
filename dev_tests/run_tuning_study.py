#!/usr/bin/env python3
"""Tuning + robustness study on the realistic surfaces (T1-T5).

T1: generator hyperparameter sweep on production-4d (batch, warm-up,
    annealed-eta, R2 dose, trust region), 5 seeds each.
T2: warm-start dose-response (seed history k in {0,10,30,60}).
T3: R3 prediction-accuracy flag calibration (rides on T1 runs).
T4: noise robustness (extra multiplicative evaluation noise).
T5: 6D scaling (production + halo axes free).

Run:  python dev_tests/run_tuning_study.py [stage]   stage in t1 t2 t4 t5
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_bayesopt_generator as T  # noqa: E402
import chi2_landscapes as L  # noqa: E402

ps = T.ps
BayesOptGenerator = ps.BayesOptGenerator
BUDGET = 120
SEEDS = 5


def make_parspace(names_bounds):
    steps = {
        "ml": (0.2, 0.1),
        "q-stars": (0.04, 0.02),
        "p-stars": (0.01, 0.005),
        "m-bh": (0.1, 0.05),
        "c-dh": (1.0, 0.5),
        "f-dh": (0.1, 0.1),
    }
    params = []
    for name, (lo, hi), log in names_bounds:
        dname = {"log10 MBH": "m-bh", "c-dh": "c-dh", "log f-dh": "f-dh", "ml": "ml", "q": "q-stars", "p": "p-stars"}[
            name
        ]
        p = T._mk_param(dname, lo, hi, 0.5 * (lo + hi), logarithmic=log)
        p.par_generator_settings.update({"step": steps[dname][0], "minstep": steps[dname][1]})
        params.append(p)
    return T.make_parspace(params)


def drive(land, gs_over, seeds=SEEDS, noise=0.0, seed_hist=None, budget=BUDGET, collect_r3=False, arm="bayesopt"):
    """Run one generator arm over seeds; return per-seed metrics."""
    import torch

    out = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        parspace = make_parspace([(n, b, n == "log10 MBH" or n == "log f-dh") for n, b in zip(land.names, land.bounds)])
        names = [p.name for p in parspace]
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
                "generator_settings": dict(gs_over),
                "stopping_criteria": {"n_max_mods": 10**6, "n_max_iter": 10**6, "min_delta_chi2_abs": -1e6},
            }
            gen = BayesOptGenerator(par_space=parspace, parspace_settings=s)
        am = T.MockAllModels(names)
        if seed_hist is not None:
            for row in seed_hist:
                am.table.add_row(row)
        r3_events = []

        def fill(table):
            kin = np.asarray(table["kinchi2"], dtype=float)
            for i in np.where(~np.isfinite(kin))[0]:
                vals = [float(table[p_.name][i]) for p_ in parspace]
                x = parspace.get_raw_value_from_param_value(vals)
                c = float(land(x)[0])
                if noise > 0:
                    c *= 1.0 + np.random.normal(0, noise)
                table["chi2"][i] = c
                table["kinchi2"][i] = c
                table["all_done"][i] = True
            if collect_r3 and gen.status.get("gp_predictions_accurate"):
                r3_events.append(len(table))

        while len(am.table) < budget and not gen.status.get("stop"):
            status = gen.generate(current_models=am)
            fill(am.table)
            if arm == "gridwalk" and status.get("n_new_models", 1) == 0:
                break  # walk cornered
        kin = np.asarray(am.table["kinchi2"], dtype=float)
        ok = np.isfinite(kin)
        hit = ok & (kin <= land.threshold)
        cnt = int(np.argmax(hit) + 1) if np.any(hit) else None
        n_fresh = len(am.table) - (len(seed_hist) if seed_hist is not None else 0)
        out.append({"n_fresh": n_fresh, "to_thr": cnt, "best": float(np.min(kin[ok])), "r3_events": r3_events})
    return out


def summarize(tag, results, n_seeds):
    m2t = sorted(r["to_thr"] for r in results if r["to_thr"] is not None)
    hits = len(m2t)
    med = m2t[len(m2t) // 2] if m2t else None
    best = min(r["best"] for r in results)
    fresh = sorted(r["n_fresh"] for r in results)
    print(
        f"{tag:<44} thr={hits}/{n_seeds} med={med} "
        f"fresh={fresh[len(fresh) // 2]}[{min(fresh)}..{max(fresh)}] "
        f"best={best:.0f}",
        flush=True,
    )
    return {"tag": tag, "hits": hits, "med": med, "best": best, "results": results}


# ---------------------------------------------------------------------------
def t1():
    """Hyperparameter sweep on production-4d + R3 calibration (T3)."""
    land = L.PRODUCTION
    base = {
        "batch_size": 8,
        "n_initial_random": 8,
        "exploration_schedule": "annealed",
        "n_annealed_members": 2,
        "trust_region": True,
        "discretize_non_ml_params": False,
    }
    grid = [
        ("base(b8,w8,eta,r2,TR)", {}),
        ("batch=4", {"batch_size": 4}),
        ("batch=16", {"batch_size": 16}),
        ("warmup=4", {"n_initial_random": 4}),
        ("warmup=12", {"n_initial_random": 12}),
        ("eta=constant", {"exploration_schedule": "constant"}),
        ("R2=0", {"n_annealed_members": 0}),
        ("R2=4", {"n_annealed_members": 4}),
        ("TR=off", {"trust_region": False}),
        ("lean(b4,w4,eta,off,off)", {"batch_size": 4, "n_initial_random": 4, "trust_region": False}),
    ]
    print(f"=== T1/T3: hyperparameters on {land.name} ({SEEDS} seeds, budget {BUDGET}) ===", flush=True)
    scores = []
    for tag, over in grid:
        gs = dict(base)
        gs.update(over)
        res = summarize(tag, drive(land, gs, collect_r3=True), SEEDS)
        res["r3_fires"] = sum(len(r["r3_events"]) for r in res["results"])
        res["r3_in_runs"] = sum(1 for r in res["results"] if r["r3_events"])
        scores.append(res)
    print("\n=== T1 ranking (median models-to-threshold, then best) ===")
    for r in sorted(scores, key=lambda r: (-(r["hits"]), r["med"] or 10**9, r["best"])):
        print(f"  {r['tag']:<40} med={r['med']} best={r['best']:.0f} hits={r['hits']}/{SEEDS}")
    print("\n=== T3: R3 prediction-accuracy flag (eps_rel=0.01) ===")
    for r in scores:
        print(f"  {r['tag']:<40} fired in {r['r3_in_runs']}/{SEEDS} runs, {r['r3_fires']} events total")


def t2():
    """Warm-start dose-response on production-4d."""
    land = L.PRODUCTION
    rng = np.random.default_rng(7)
    history_pool = []
    for _ in range(80):
        mbh = rng.uniform(3.9, 4.78)
        ml = rng.uniform(1.0, 5.0)
        q = rng.uniform(0.3, 0.89)
        p = rng.uniform(0.90, 0.999)
        c = land([[mbh, ml, q, p]])[0]
        history_pool.append([mbh, ml, q, p, c, c, float("nan"), "", True, True, True, 0, "d"])
    gs = {
        "batch_size": 8,
        "n_initial_random": 0,
        "exploration_schedule": "annealed",
        "n_annealed_members": 2,
        "trust_region": True,
    }
    print(f"\n=== T2: warm-start dose-response on {land.name} ===", flush=True)
    for k in [0, 10, 30, 60]:
        hist = history_pool[:k] if k else None
        res = summarize(f"history={k:>3}", drive(land, gs, seed_hist=hist), SEEDS)


def t4():
    """Noise robustness: extra multiplicative evaluation noise."""
    land = L.PRODUCTION
    gs = {
        "batch_size": 8,
        "n_initial_random": 8,
        "exploration_schedule": "annealed",
        "n_annealed_members": 2,
        "trust_region": True,
    }
    print(f"\n=== T4: noise robustness on {land.name} ===", flush=True)
    for sigma in [0.0, 0.003, 0.01]:
        summarize(f"sigma={sigma:.1%} BO", drive(land, gs, noise=sigma), SEEDS)
        summarize(f"sigma={sigma:.1%} GW", drive(land, None, noise=sigma, arm="gridwalk"), SEEDS)


def t5():
    """6D scaling: production + halo axes free."""
    land6 = L.Landscape(
        "prod6d",
        ["log10 MBH", "ml", "q", "p", "c-dh", "log f-dh"],
        L.PRODUCTION.bounds + L.HALO_BANANA.bounds[1:],
        lambda X: (
            L.PRODUCTION(X[:, :4]) * 0.5
            + L.HALO_BANANA(np.column_stack([X[:, 1], np.full(len(X), 6.0), X[:, 5]])) * 0.5
            + 2900.0
        ),
        threshold=6120.0,
        desc="production 4D x halo (c inert, f coupled)",
    )
    gs = {
        "batch_size": 8,
        "n_initial_random": 12,
        "exploration_schedule": "annealed",
        "n_annealed_members": 2,
        "trust_region": True,
    }
    print(f"\n=== T5: 6D scaling ({land6.name}) ===", flush=True)
    summarize("6D BO", drive(land6, gs), SEEDS)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("t1", "all"):
        t1()
    if stage in ("t2", "all"):
        t2()
    if stage in ("t4", "all"):
        t4()
    if stage in ("t5", "all"):
        t5()

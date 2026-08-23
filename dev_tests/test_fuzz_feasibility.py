"""Fuzz test: triaxial feasibility invariants over randomized parameter spaces.

Random free subsets (including logarithmic params alongside q/p/u — the
production configuration's shape), random qobs, random fixed/free mixtures.
Asserts, for every trial: projection output is finite and in-bounds, and
satisfies p>=q and the u-window algebraically; GP-phase constraints are
non-negative at projected points.

Run: python dev_tests/test_fuzz_feasibility.py [n_trials]
"""

import sys

import numpy as np

import test_bayesopt_generator as T

ps = T.ps


def make_random_space(rng):
    """Random parspace: ml always free; log-mbh / q / p mixed; u mixed."""
    axes = {
        "ml": dict(lo=1.0, hi=5.0, log=False),
        "m-bh": dict(lo=3.90, hi=4.78, log=True),
        "q-stars": dict(lo=0.05, hi=0.89, log=False),
        "p-stars": dict(lo=0.90, hi=0.999, log=False),
    }
    free = ["ml"] + [a for a in ("m-bh", "q-stars", "p-stars") if rng.random() < 0.7]
    params = []
    for name, ax in axes.items():
        p = T._mk_param(
            name, ax["lo"], ax["hi"], 0.5 * (ax["lo"] + ax["hi"]), logarithmic=ax["log"], fixed=name not in free
        )
        params.append(p)
    u_free = rng.random() < 0.4
    u = T._mk_param("u-stars", 0.95, 1.0, rng.uniform(0.98, 0.9999), fixed=not u_free)
    params.append(u)
    # attach a triaxial system with random qobs so projection/constraints
    # actually engage whenever q or p is free
    if "q-stars" in free or "p-stars" in free:
        qobs = float(rng.uniform(0.5, 0.95))
        tri = T.MockTriaxialComponent("stars", qobs=qobs)
        sysm = T.MockSystem([], components=[tri])
        sysm.cmp_list[0].parameters = params
        return T.make_parspace(params, system=sysm), free, u_free
    return T.make_parspace(params), free, u_free


def qpu_valid(qv, pv, uv, qobs, rtol=5e-6):
    """Algebraic triaxiality check with RELATIVE tolerance: the projector
    guarantees its 1e-6 margin multiplicatively, so the validator must too."""
    return qv <= pv * (1 + rtol) and max(qv / qobs, pv) <= uv * (1 + rtol) and uv <= min(pv / qobs, 1.0) * (1 + rtol)


def main(n_trials=300):
    rng = np.random.default_rng(42)
    n_proj_fail = n_constr_fail = n_skipped = 0
    for trial in range(n_trials):
        parspace, free, u_free = make_random_space(rng)
        gen = ps.BayesOptGenerator(par_space=parspace, parspace_settings=T._bo_settings())
        if gen.qobs is None:
            n_skipped += 1
            continue
        d = len(gen.free_params)
        X = gen._project_unit_to_feasible_qpu(rng.random((80, d)))
        lo_raw, hi_raw = gen._norm_bounds_arrays()
        raw = X * (hi_raw - lo_raw) + lo_raw
        if not (np.isfinite(raw).all() and (raw >= lo_raw - 1e-12).all() and (raw <= hi_raw + 1e-12).all()):
            n_proj_fail += 1
            print(f"trial {trial}: out-of-bounds after projection")
            continue
        # algebraic triaxiality on the RAW values (fixed axes included)
        fx = gen._fixed_qpu_values()
        names = [p.name.split("-")[0] for p in gen.free_params]
        col = {n: i for i, n in enumerate(names)}
        qv = raw[:, col["q"]] if "q" in col else np.full(len(raw), fx["q"])
        pv = raw[:, col["p"]] if "p" in col else np.full(len(raw), fx["p"])
        uv = raw[:, col["u"]] if "u" in col else np.full(len(raw), fx["u"])
        ok = [qpu_valid(a, b, c, gen.qobs) for a, b, c in zip(qv, pv, uv)]
        if not all(ok):
            n_proj_fail += 1
            bad = int(np.sum(~np.array(ok)))
            print(f"trial {trial}: {bad}/80 invalid after projection (free={free}, u_free={u_free})")
            continue
        # GP constraints must be satisfied at projected points
        nl, lin = gen._make_triaxiality_constraints()
        if nl:
            import torch

            Xt = torch.tensor(X, dtype=torch.double)
            for fn, intra in nl:
                vals = fn(Xt.T).numpy() if Xt.T.shape[0] == len(gen.free_params) else None
                if vals is None:
                    continue
                if np.min(vals) < -1e-9:
                    n_constr_fail += 1
                    print(f"trial {trial}: constraint violated min={np.min(vals):.2e}")
    print(
        f"\nfuzz: {n_trials} trials, {n_skipped} w/o qobs, "
        f"projection failures={n_proj_fail}, constraint failures="
        f"{n_constr_fail}"
    )
    assert n_proj_fail == 0 and n_constr_fail == 0, "FUZZ FAILURES"
    print("FUZZ FEASIBILITY PASSED")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
